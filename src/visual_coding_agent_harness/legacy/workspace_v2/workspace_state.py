"""Evidence workspace for visual tool results."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...core.contracts import CONTRACT_VERSION, BudgetReason, EvidenceStage, GroundingQuality, SamplingPolicy
from .output_quality import is_unsupported_claim
from .open_questions import extract_candidate_options
from .transcript_binder import TranscriptEvidenceBinder
from ..contracts_v2 import (
    ClaimModality,
    ClaimRelation,
    TargetSpec,
    build_ordered_transcript_sequence,
)
from ...core.schemas import EvidenceRowV2
from ..memory import MemoryEntry, SourceAnchor, excerpt_hash, normalized_text


_TARGET_REF_RE = re.compile(r"^T[1-9]\d*$")


@dataclass(frozen=True)
class Observation:
    observation_id: str
    tool: str
    claim: str
    confidence: float
    input_artifacts: Sequence[str] = field(default_factory=list)
    regions: Sequence[Mapping[str, Any]] = field(default_factory=list)
    limitations: str = ""
    confidence_signal: str = ""
    raw_output: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = ""
    frame_set_id: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "Observation":
        return cls(
            observation_id=str(payload.get("observation_id", "")),
            tool=str(payload.get("tool", "")),
            claim=str(payload.get("claim", "")),
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            input_artifacts=list(payload.get("input_artifacts", [])),
            regions=list(payload.get("regions", [])),
            limitations=str(payload.get("limitations", "")),
            confidence_signal=str(payload.get("confidence_signal", "")),
            raw_output=dict(payload.get("raw_output", {}) or {}),
            created_at=str(payload.get("created_at", "")),
            frame_set_id=(None if payload.get("frame_set_id") is None else str(payload.get("frame_set_id"))),
        )


@dataclass(frozen=True)
class FrameSetManifest:
    frame_set_id: str
    video_path: str
    segment_id: str | None
    start_sec: float
    end_sec: float
    nframes: int
    target_nframes: int
    sampling_policy: SamplingPolicy
    frame_times_sec: list[float]
    frame_times_approximate: bool
    created_by_tool: str
    observation_id: str
    budget_reason: BudgetReason
    contract_version: str
    materialized_paths: list[str]
    created_at: float

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FrameSetManifest":
        return cls(
            frame_set_id=str(payload.get("frame_set_id", "")),
            video_path=str(payload.get("video_path", "")),
            segment_id=(None if payload.get("segment_id") is None else str(payload.get("segment_id"))),
            start_sec=float(payload.get("start_sec", 0.0) or 0.0),
            end_sec=float(payload.get("end_sec", 0.0) or 0.0),
            nframes=int(payload.get("nframes", 0) or 0),
            target_nframes=int(payload.get("target_nframes", 0) or 0),
            sampling_policy=str(payload.get("sampling_policy", "uniform")),  # type: ignore[arg-type]
            frame_times_sec=[float(item) for item in payload.get("frame_times_sec", [])],
            frame_times_approximate=bool(payload.get("frame_times_approximate", False)),
            created_by_tool=str(payload.get("created_by_tool", "")),
            observation_id=str(payload.get("observation_id", "")),
            budget_reason=str(payload.get("budget_reason", "default_contract")),  # type: ignore[arg-type]
            contract_version=str(payload.get("contract_version", CONTRACT_VERSION)),
            materialized_paths=[str(item) for item in payload.get("materialized_paths", [])],
            created_at=float(payload.get("created_at", 0.0) or 0.0),
        )


@dataclass
class EvidenceRecord:
    evidence_id: str
    stage: EvidenceStage
    parent_id: str | None
    tool: str
    observation_id: str | None
    frame_set_id: str | None
    content: dict[str, Any]
    grounding_quality: GroundingQuality
    confidence: float
    created_at: float

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EvidenceRecord":
        return cls(
            evidence_id=str(payload.get("evidence_id", "")),
            stage=str(payload.get("stage", "raw")),  # type: ignore[arg-type]
            parent_id=(None if payload.get("parent_id") is None else str(payload.get("parent_id"))),
            tool=str(payload.get("tool", "")),
            observation_id=(None if payload.get("observation_id") is None else str(payload.get("observation_id"))),
            frame_set_id=(None if payload.get("frame_set_id") is None else str(payload.get("frame_set_id"))),
            content=dict(payload.get("content", {}) or {}),
            grounding_quality=str(payload.get("grounding_quality", "navigation_only")),  # type: ignore[arg-type]
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            created_at=float(payload.get("created_at", 0.0) or 0.0),
        )


@dataclass
class MapUpdateProposal:
    proposal_id: str
    target_segment_id: str
    update_type: str
    payload: dict[str, Any]
    source_evidence_id: str
    source_frame_set_id: str
    confidence: float
    proposed_at: float
    committed_at: float | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MapUpdateProposal":
        return cls(
            proposal_id=str(payload.get("proposal_id", "")),
            target_segment_id=str(payload.get("target_segment_id", "")),
            update_type=str(payload.get("update_type", "")),
            payload=dict(payload.get("payload", {}) or {}),
            source_evidence_id=str(payload.get("source_evidence_id", "")),
            source_frame_set_id=str(payload.get("source_frame_set_id", "")),
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            proposed_at=float(payload.get("proposed_at", 0.0) or 0.0),
            committed_at=(None if payload.get("committed_at") is None else float(payload.get("committed_at"))),
        )


class EvidenceWorkspace:
    """Persist artifacts, observations, trace events, and an answer-facing ledger."""

    VISUAL_EVIDENCE_TOOLS = {
        "global_gist",
        "query_context",
        "caption_segment",
        "qa_segment",
        "inspect_segment",
        "vision_read",
        "verify_segment_anchors",
        "read_clip",
        "verify",
        "verify_window",
        "synthesize_memory",
    }
    GROUNDED_MEMORY_KINDS = frozenset(
        {
            "visual_support",
            "answer_support",
            "synthesized_support",
            "answer_conflict_resolved",
            "caption_support",
            # Intentionally excludes local_negative, answer_conflict,
            # verification_uncertain, and retrieval_candidate: those are useful
            # search-state facts, not final grounding.
        }
    )
    POSITIVE_SUPPORT_KINDS = frozenset(
        {
            "visual_support",
            "answer_support",
            "synthesized_support",
            "answer_conflict_resolved",
            "caption_support",
        }
    )
    LOCAL_WORKER_EVIDENCE_TOOLS = {
        "caption_image",
        "caption_region",
        "caption_segment",
        "caption_segments",
        "inspect_region",
        "inspect_segment",
        "ocr_region",
        "qa_region",
        "qa_segment",
    }
    NAVIGATION_TOOLS = {
        "video_ls",
        "search_segments",
        "ground_question",
        "read_segment",
        "read_segment_detail",
        "locate_targets_in_segment",
        "target_coverage",
        "expand_window",
        "zoom",
        "commit_map_proposals",
        "append_to_timeline",
        "view_observation",
        "grep_evidence",
        "query_evidence_table",
        "read_timeline_sorted",
        "read_hypothesis",
        "update_hypothesis_slot",
    }
    CONTEXT_ONLY_TOOLS = {"global_gist", "query_context"}
    AUTO_ACKNOWLEDGED_TOOLS = {
        "list",
        "read_workspace",
        "view_observation",
        "read_observation_detail",
        "grep_evidence",
        "query_evidence_table",
        "read_timeline_sorted",
        "read_hypothesis",
        "search_segments",
    }
    ANSWER_EVIDENCE_TOOLS = (VISUAL_EVIDENCE_TOOLS - CONTEXT_ONLY_TOOLS) | {
        "caption_image",
        "caption_region",
        "ocr_region",
        "qa_region",
        "inspect_region",
        "asr_cue_detail",
        "ordered_list_evidence",
        "ordered_transcript_sequence",
        "timeline_asr_summary",
        "transcript_evidence_binder",
        "verify_local_claim",
    }

    def __init__(self, root: Path) -> None:
        self.root = root

    @classmethod
    def create(cls, base_dir: Path, run_id: str) -> "EvidenceWorkspace":
        root = base_dir / "runs" / run_id
        for child in [
            root,
            root / "input",
            root / "artifacts",
            root / "artifacts" / "frames",
            root / "artifacts" / "clips",
            root / "artifacts" / "crops",
            root / "artifacts" / "masks",
            root / "frame_sets",
            root / "index",
            root / "entities",
            root / "events",
            root / "relations",
            root / "attributes",
            root / "memory",
            root / "notes",
            root / "observations",
            root / "pinned",
        ]:
            child.mkdir(parents=True, exist_ok=True)

        for filename in [
            "observations.jsonl",
            "produced_anchors.jsonl",
            "memory.jsonl",
            "trace.jsonl",
            "evidence.jsonl",
            "evidence_table.jsonl",
            "map_proposals.jsonl",
            "index/coarse_segments.jsonl",
            "index/asr.jsonl",
            "index/shots.jsonl",
            "index/captions.jsonl",
            "entities/entities.jsonl",
            "events/events.jsonl",
            "relations/relations.jsonl",
            "attributes/attributes.jsonl",
            "memory/memory.jsonl",
            "observations/observations.jsonl",
            "observations/disposition.jsonl",
            "pinned/pinned_anchors.jsonl",
        ]:
            (root / filename).touch(exist_ok=True)
        (root / "frame_sets" / "manifests.jsonl").touch(exist_ok=True)
        (root / "reflection_memory.jsonl").touch(exist_ok=True)

        ledger = root / "ledger.md"
        if not ledger.exists():
            ledger.write_text("# Evidence Ledger\n\n", encoding="utf-8")

        timeline = root / "timeline.md"
        if not timeline.exists():
            timeline.write_text("# Timeline\n\n", encoding="utf-8")

        hypothesis = root / "hypothesis.md"
        if not hypothesis.exists():
            hypothesis.write_text("# Hypothesis\n\n", encoding="utf-8")

        plan = root / "notes" / "plan.md"
        if not plan.exists():
            plan.write_text("# Plan\n\n", encoding="utf-8")

        open_questions = root / "notes" / "open_questions.md"
        if not open_questions.exists():
            open_questions.write_text("# Open Questions\n\n", encoding="utf-8")

        return cls(root=root)

    def write_observation(
        self,
        *,
        tool_name: str,
        claim: str,
        confidence: float,
        input_artifacts: Sequence[str] = (),
        regions: Sequence[Mapping[str, Any]] = (),
        limitations: str = "",
        confidence_signal: str = "",
        raw_output: Mapping[str, Any] | None = None,
        frame_set_id: str | None = None,
    ) -> Observation:
        observation_id = self._next_observation_id()
        raw_payload = dict(raw_output or {})
        produced_anchors = self._produced_anchors_for_observation(
            raw_payload.get("produced_anchors", ()),
            observation_id=observation_id,
        )
        if produced_anchors:
            raw_payload["produced_anchors"] = [anchor.to_dict() for anchor in produced_anchors]
        observation = Observation(
            observation_id=observation_id,
            tool=tool_name,
            claim=claim,
            confidence=confidence,
            input_artifacts=list(input_artifacts),
            regions=list(regions),
            limitations=limitations,
            confidence_signal=confidence_signal or str(raw_payload.get("confidence_signal", "")),
            raw_output=raw_payload,
            created_at=_utc_now(),
            frame_set_id=frame_set_id,
        )
        self._append_jsonl("observations.jsonl", asdict(observation))
        self._append_jsonl("observations/observations.jsonl", asdict(observation))
        claim_scope = str(raw_payload.get("claim_scope") or "").strip()
        if tool_name == "explore" and claim_scope and claim_scope != "direct_answer":
            self.write_trace_event(
                "caption_scope_downgrade",
                {
                    "observation_id": observation_id,
                    "claim_scope": claim_scope,
                    "task_type": str(raw_payload.get("task_type") or ""),
                },
            )
        if produced_anchors:
            written_anchors = self.write_produced_anchors(produced_anchors)
            self.write_trace_event(
                "observation_anchors_registered",
                {
                    "observation_id": observation_id,
                    "tool": tool_name,
                    "anchor_ids": [anchor.anchor_id for anchor in written_anchors],
                    "anchor_count": len(written_anchors),
                },
            )
        return self._apply_post_observation_hooks(observation)

    def write_produced_anchors(self, anchors: Sequence[SourceAnchor | Mapping[str, Any]]) -> list[SourceAnchor]:
        resolved: list[SourceAnchor] = []
        for item in anchors:
            anchor = item if isinstance(item, SourceAnchor) else SourceAnchor.from_mapping(item)
            if not anchor.anchor_id:
                raise ValueError("anchor_validation_failed: anchor_id must not be empty")
            if not anchor.observation_id:
                raise ValueError("anchor_validation_failed: observation_id must not be empty")
            payload = anchor.to_dict()
            payload["excerpt_hash"] = payload.get("excerpt_hash") or excerpt_hash(str(payload.get("excerpt", "")))
            anchor = SourceAnchor.from_mapping(payload)
            self._append_jsonl("produced_anchors.jsonl", anchor.to_dict())
            resolved.append(anchor)
        return resolved

    def read_produced_anchors(self) -> list[SourceAnchor]:
        return [SourceAnchor.from_mapping(row) for row in self._read_jsonl_dicts("produced_anchors.jsonl")]

    def read_produced_anchors_by_id(self) -> dict[str, SourceAnchor]:
        anchors: dict[str, SourceAnchor] = {}
        for anchor in self.read_produced_anchors():
            anchors[anchor.anchor_id] = anchor
        return anchors

    def write_memory(
        self,
        *,
        kind: str,
        claim: str,
        anchors: Sequence[Mapping[str, Any]],
        supports_option: str = "",
        confidence: str = "medium",
        previous_memory_refs: Sequence[str] = (),
        tags: Sequence[str] = (),
        round_number: int | None = None,
        role: str | None = None,
        layer: str | None = None,
        embedding_refs: Sequence[str] = (),
        metadata: Mapping[str, object] | None = None,
    ) -> MemoryEntry:
        if not anchors:
            raise ValueError("memory_validation_failed: anchors must not be empty")

        available = self.read_produced_anchors_by_id()
        resolved_anchors: list[SourceAnchor] = []
        for payload in anchors:
            anchor_id = str(payload.get("anchor_id", "")).strip()
            if anchor_id not in available:
                raise ValueError(f"memory_validation_failed: unknown anchor_id={anchor_id}")
            resolved = available[anchor_id]
            model_excerpt = str(payload.get("excerpt", "") or "").strip()
            if model_excerpt and normalized_text(model_excerpt) not in normalized_text(resolved.excerpt):
                raise ValueError(f"memory_validation_failed: excerpt not found in anchor {anchor_id}")
            resolved_anchors.append(resolved)

        option_id = str(supports_option or "").strip()
        if option_id and self._known_option_ids() and option_id not in self._known_option_ids():
            raise ValueError(f"memory_validation_failed: unknown option {option_id}")

        for ref in previous_memory_refs:
            ref_text = str(ref).strip()
            if ref_text and self.get_memory(ref_text) is None:
                raise ValueError(f"memory_validation_failed: unknown memory ref={ref_text}")

        entry = MemoryEntry(
            entry_id=self._next_memory_id(),
            round_number=round_number if round_number is not None else self.current_round(),
            kind=kind,  # type: ignore[arg-type]
            claim=claim,
            anchors=tuple(resolved_anchors),
            supports_option=option_id or None,
            confidence=confidence,  # type: ignore[arg-type]
            previous_memory_refs=tuple(str(ref) for ref in previous_memory_refs if str(ref).strip()),
            tags=tuple(str(tag) for tag in tags),
            created_at_sec=time.time(),
            role=role,
            layer=layer,
            embedding_refs=tuple(str(ref) for ref in embedding_refs if str(ref).strip()),
            metadata=dict(metadata or {}),
        )
        self._append_jsonl("memory.jsonl", entry.to_dict())
        self._append_jsonl("memory/memory.jsonl", entry.to_dict())
        self.write_trace_event(
            "memory_entry_written",
            {
                "entry_id": entry.entry_id,
                "kind": entry.kind,
                "supports_option": entry.supports_option or "",
                "anchor_ids": [anchor.anchor_id for anchor in entry.anchors],
            },
        )
        return entry

    def write_memory_entry(self, **kwargs: Any) -> MemoryEntry:
        return self.write_memory(**kwargs)

    def committed_memory_anchor_ids(self) -> set[str]:
        return {
            anchor.anchor_id
            for entry in self.memory_entries()
            for anchor in entry.anchors
            if anchor.anchor_id
        }

    def memory_entries(self) -> list[MemoryEntry]:
        return [MemoryEntry.from_mapping(row) for row in self._read_jsonl_dicts("memory.jsonl")]

    def get_memory(self, entry_id: str) -> MemoryEntry | None:
        for entry in self.memory_entries():
            if entry.entry_id == str(entry_id):
                return entry
        return None

    def read_memory_by_id(self, entry_id: str) -> MemoryEntry | None:
        return self.get_memory(entry_id)

    def note_entity(
        self,
        *,
        kind: str,
        name: str,
        evidence_obs_ids: Sequence[str] = (),
        attributes: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "entity_id": self._next_structured_id("entities/entities.jsonl", "e"),
            "kind": str(kind or "entity"),
            "name": str(name or "").strip(),
            "evidence_obs_ids": self._validated_observation_ids(evidence_obs_ids),
            "attributes": dict(attributes or {}),
            "created_at": _utc_now(),
        }
        if not payload["name"]:
            raise ValueError("entity_validation_failed: name must not be empty")
        self._append_jsonl("entities/entities.jsonl", payload)
        self.write_trace_event("workspace_entity_written", payload)
        return payload

    def note_event(
        self,
        *,
        label: str,
        time_range: Sequence[float] | None = None,
        evidence_obs_ids: Sequence[str] = (),
        attributes: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "event_id": self._next_structured_id("events/events.jsonl", "ev"),
            "label": str(label or "").strip(),
            "time_range": _normalize_optional_time_range(time_range),
            "evidence_obs_ids": self._validated_observation_ids(evidence_obs_ids),
            "attributes": dict(attributes or {}),
            "created_at": _utc_now(),
        }
        if not payload["label"]:
            raise ValueError("event_validation_failed: label must not be empty")
        self._append_jsonl("events/events.jsonl", payload)
        self.write_trace_event("workspace_event_written", payload)
        return payload

    def note_relation(
        self,
        *,
        subject: str,
        predicate: str,
        objects: Sequence[str],
        time_range: Sequence[float] | None = None,
        evidence_obs_ids: Sequence[str] = (),
        attributes: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        subject_text = str(subject or "").strip()
        object_texts = [str(item).strip() for item in objects if str(item).strip()]
        self._validate_relation_entities(subject_text, object_texts)
        payload = {
            "relation_id": self._next_structured_id("relations/relations.jsonl", "rel"),
            "subject": subject_text,
            "predicate": str(predicate or "").strip(),
            "objects": object_texts,
            "time_range": _normalize_optional_time_range(time_range),
            "evidence_obs_ids": self._validated_observation_ids(evidence_obs_ids),
            "attributes": dict(attributes or {}),
            "created_at": _utc_now(),
        }
        if not payload["subject"] or not payload["predicate"] or not payload["objects"]:
            raise ValueError("relation_validation_failed: subject, predicate, and objects are required")
        self._append_jsonl("relations/relations.jsonl", payload)
        self.write_trace_event("workspace_relation_written", payload)
        return payload

    def note_attribute(
        self,
        *,
        target: str,
        name: str,
        value: Any,
        evidence_obs_ids: Sequence[str] = (),
    ) -> dict[str, Any]:
        payload = {
            "attribute_id": self._next_structured_id("attributes/attributes.jsonl", "attr"),
            "target": str(target or "").strip(),
            "name": str(name or "").strip(),
            "value": value,
            "evidence_obs_ids": self._validated_observation_ids(evidence_obs_ids),
            "created_at": _utc_now(),
        }
        if not payload["target"] or not payload["name"]:
            raise ValueError("attribute_validation_failed: target and name are required")
        self._append_jsonl("attributes/attributes.jsonl", payload)
        self.write_trace_event("workspace_attribute_written", payload)
        return payload

    def pin_anchor(self, observation_id: str, anchor: Mapping[str, Any]) -> dict[str, Any]:
        observation = self._require_observation(observation_id)
        payload = dict(anchor)
        anchor_id = str(
            payload.get("anchor_id")
            or payload.get("candidate_anchor_id")
            or f"anch_{len(self.read_pinned_anchors()) + 1:04d}"
        ).strip()
        if not anchor_id:
            raise ValueError("anchor_validation_failed: anchor_id must not be empty")
        excerpt = str(payload.get("excerpt", "") or "").strip()
        self._validate_excerpt_in_observation(observation, excerpt)
        time_range = _normalize_optional_time_range(payload.get("time_range"))
        start_sec = payload.get("start_sec")
        end_sec = payload.get("end_sec")
        if time_range is not None:
            start_sec, end_sec = time_range
        source_kind = str(payload.get("source_kind") or payload.get("kind") or "observation_fact")
        source_anchor = SourceAnchor.from_mapping(
            {
                **payload,
                "anchor_id": anchor_id,
                "observation_id": observation.observation_id,
                "source_kind": source_kind,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "excerpt": excerpt,
            }
        )
        produced = self.read_produced_anchors_by_id()
        if source_anchor.anchor_id not in produced:
            self.write_produced_anchors([source_anchor])
        pinned = {
            **source_anchor.to_dict(),
            "kind": str(payload.get("kind") or source_kind),
            "time_range": time_range,
            "created_at": _utc_now(),
        }
        if source_anchor.anchor_id not in {row.get("anchor_id") for row in self.read_pinned_anchors()}:
            self._append_jsonl("pinned/pinned_anchors.jsonl", pinned)
            self.write_trace_event(
                "workspace_anchor_pinned",
                {"observation_id": observation.observation_id, "anchor_id": source_anchor.anchor_id},
            )
        return pinned

    def read_pinned_anchors(self) -> list[dict[str, Any]]:
        return self._read_jsonl_dicts("pinned/pinned_anchors.jsonl")

    def commit_observation(self, observation_id: str, *, writes: Mapping[str, Any]) -> dict[str, Any]:
        observation = self._require_disposable_observation(observation_id)
        normalized_writes = dict(writes or {})
        self._validate_commit_writes(observation, normalized_writes)

        snapshot = self._snapshot_transaction_files()
        try:
            pinned_anchors: list[dict[str, Any]] = []
            for anchor_payload in _mapping_list(normalized_writes.get("pinned_anchors")):
                pinned_anchors.append(self.pin_anchor(observation.observation_id, anchor_payload))

            entities: list[dict[str, Any]] = []
            for entity_payload in _mapping_list(normalized_writes.get("entities")):
                entities.append(
                    self.note_entity(
                        kind=str(entity_payload.get("kind", "entity")),
                        name=str(entity_payload.get("name", "")),
                        evidence_obs_ids=_default_evidence_obs_ids(entity_payload, observation.observation_id),
                        attributes=dict(entity_payload.get("attributes", {}) or {}),
                    )
                )

            events: list[dict[str, Any]] = []
            for event_payload in _mapping_list(normalized_writes.get("events")):
                events.append(
                    self.note_event(
                        label=str(event_payload.get("label") or event_payload.get("name") or ""),
                        time_range=event_payload.get("time_range"),
                        evidence_obs_ids=_default_evidence_obs_ids(event_payload, observation.observation_id),
                        attributes=dict(event_payload.get("attributes", {}) or {}),
                    )
                )

            attributes: list[dict[str, Any]] = []
            for attribute_payload in _mapping_list(normalized_writes.get("attributes")):
                attributes.append(
                    self.note_attribute(
                        target=str(attribute_payload.get("target", "")),
                        name=str(attribute_payload.get("name", "")),
                        value=attribute_payload.get("value"),
                        evidence_obs_ids=_default_evidence_obs_ids(attribute_payload, observation.observation_id),
                    )
                )

            relations: list[dict[str, Any]] = []
            for relation_payload in _mapping_list(normalized_writes.get("relations")):
                relations.append(
                    self.note_relation(
                        subject=str(relation_payload.get("subject", "")),
                        predicate=str(relation_payload.get("predicate", "")),
                        objects=[str(item) for item in _sequence_items(relation_payload.get("objects"))],
                        time_range=relation_payload.get("time_range"),
                        evidence_obs_ids=_default_evidence_obs_ids(relation_payload, observation.observation_id),
                        attributes=dict(relation_payload.get("attributes", {}) or {}),
                    )
                )

            memory_entries: list[dict[str, Any]] = []
            available_anchor_ids = [str(anchor.get("anchor_id")) for anchor in pinned_anchors if anchor.get("anchor_id")]
            for memory_payload in _mapping_list(normalized_writes.get("memory")):
                memory_metadata = _memory_commit_metadata(memory_payload, observation=observation)
                anchor_ids = [
                    str(item).strip()
                    for item in _sequence_items(memory_payload.get("anchor_ids"))
                    if str(item).strip()
                ] or available_anchor_ids
                if not anchor_ids:
                    raise ValueError("memory_validation_failed: commit memory requires anchor_ids")
                entry = self.write_memory(
                    kind=str(memory_payload.get("kind") or "support"),
                    claim=str(memory_payload.get("claim", "")),
                    anchors=[{"anchor_id": anchor_id} for anchor_id in anchor_ids],
                    supports_option=str(memory_payload.get("supports_option", "")),
                    confidence=str(memory_payload.get("confidence", "medium")),
                    previous_memory_refs=[str(item) for item in _sequence_items(memory_payload.get("previous_memory_refs"))],
                    tags=[str(item) for item in _sequence_items(memory_payload.get("tags"))],
                    role=str(memory_payload.get("role", "") or ""),
                    layer=str(memory_payload.get("layer", "") or ""),
                    embedding_refs=[str(item) for item in _sequence_items(memory_payload.get("embedding_refs"))],
                    metadata={
                        **memory_metadata,
                        "evidence_obs_ids": _default_evidence_obs_ids(memory_payload, observation.observation_id),
                        "disposition_observation_id": observation.observation_id,
                    },
                )
                memory_entries.append(entry.to_dict())

            plan_update = str(normalized_writes.get("plan_update", "") or "").strip()
            if plan_update:
                self._append_note_line("notes/plan.md", plan_update)
            for question in _sequence_items(normalized_writes.get("open_questions_add")):
                question_text = str(question).strip()
                if question_text:
                    self._append_note_line("notes/open_questions.md", question_text)

            disposition = self._write_disposition(
                observation.observation_id,
                disposition="committed",
                writes={
                    "pinned_anchors": pinned_anchors,
                    "entities": entities,
                    "events": events,
                    "relations": relations,
                    "attributes": attributes,
                    "memory": memory_entries,
                    "plan_update": plan_update,
                    "open_questions_add": [
                        str(item).strip()
                        for item in _sequence_items(normalized_writes.get("open_questions_add"))
                        if str(item).strip()
                    ],
                    "open_questions_resolve": [
                        str(item).strip()
                        for item in _sequence_items(normalized_writes.get("open_questions_resolve"))
                        if str(item).strip()
                    ],
                },
            )
            return disposition
        except Exception:
            self._restore_transaction_files(snapshot)
            raise

    def reject_observation(self, observation_id: str, *, reason: str) -> dict[str, Any]:
        self._require_disposable_observation(observation_id)
        return self._write_disposition(
            observation_id,
            disposition="rejected",
            reason=str(reason or "").strip(),
        )

    def defer_observation(self, observation_id: str, *, until: str = "", reason: str = "") -> dict[str, Any]:
        self._require_disposable_observation(observation_id)
        defer_count = self._defer_count(observation_id) + 1
        if defer_count > 3:
            raise ValueError("disposition_validation_failed: defer limit exceeded")
        return self._write_disposition(
            observation_id,
            disposition="deferred",
            until=str(until or "").strip(),
            reason=str(reason or "").strip(),
            defer_count=defer_count,
        )

    def no_commit_needed(self, observation_id: str, *, reason: str) -> dict[str, Any]:
        self._require_disposable_observation(observation_id)
        return self._write_disposition(
            observation_id,
            disposition="acknowledged",
            reason=str(reason or "").strip(),
        )

    def observation_dispositions(self) -> list[dict[str, Any]]:
        return self._read_jsonl_dicts("observations/disposition.jsonl")

    def observation_status(self, observation_id: str) -> str:
        latest = self._latest_disposition(observation_id)
        if latest is not None:
            return str(latest.get("disposition", "uncommitted"))
        if str(observation_id) in {
            anchor.observation_id
            for entry in self.memory_entries()
            for anchor in entry.anchors
        }:
            return "committed"
        observation = self.get_observation(observation_id)
        if observation is not None and observation.tool in self.AUTO_ACKNOWLEDGED_TOOLS:
            return "auto_acknowledged"
        return "uncommitted"

    def read_workspace_section(
        self,
        section: str,
        filter: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        section_name = str(section or "").strip()
        expected = dict(filter or {})
        if section_name == "memory":
            rows = [entry.to_dict() for entry in self.memory_entries()]
        elif section_name == "entities":
            rows = self._read_jsonl_dicts("entities/entities.jsonl")
        elif section_name == "events":
            rows = self._read_jsonl_dicts("events/events.jsonl")
        elif section_name == "relations":
            rows = self._read_jsonl_dicts("relations/relations.jsonl")
        elif section_name == "attributes":
            rows = self._read_jsonl_dicts("attributes/attributes.jsonl")
        elif section_name == "pinned_anchors":
            rows = self.read_pinned_anchors()
        elif section_name == "observations_by_id":
            rows = [asdict(observation) for observation in self.read_observations()]
        elif section_name == "plan":
            rows = [{"text": self._read_note_text("notes/plan.md")}]
        elif section_name == "open_questions":
            rows = [{"text": self._read_note_text("notes/open_questions.md")}]
        else:
            raise ValueError(f"workspace_read_failed: unknown section={section_name}")
        if expected:
            rows = [row for row in rows if _row_matches_expected(row, expected)]
        return rows

    def render_plan_view(
        self,
        *,
        question: str,
        max_recent: int = 5,
        max_per_section: int = 10,
        video_map: Any | None = None,
    ) -> str:
        lines = [
            "# Workspace",
            f"Question: {question}",
        ]

        def render_section(title: str, items: Sequence[Any], formatter: Any, *, hint: str) -> None:
            lines.extend(["", f"## {title}"])
            total = len(items)
            if not items:
                lines.append("(none)")
                return
            shown = list(items)[: max(0, int(max_per_section))]
            for item in shown:
                lines.append(f"- {formatter(item)}")
            if total > len(shown):
                lines.append(f"... shown {len(shown)}/{total}; use {hint} to see more")

        active_video_map = getattr(video_map, "current", video_map)
        if active_video_map is not None:
            root_segments = [
                segment for segment in getattr(active_video_map, "segments", ()) if getattr(segment, "index_level", "root") == "root"
            ]
            lines.extend(_root_index_lines(root_segments))
            lines.extend(["", "## Index Coverage"])
            lines.extend(_index_coverage_lines(active_video_map))
            lines.append("Index coverage != evidence coverage")
            latest_patch = getattr(video_map, "latest_refinement_patch", None)
            if latest_patch is not None:
                lines.extend(["", "## Latest Index Patch"])
                lines.extend(_latest_index_patch_lines(latest_patch))

            lines.extend(["", "## Evidence Coverage"])
            lines.extend(_evidence_coverage_lines(self))
            segment_time_coverage = _segment_time_coverage_lines(self, root_segments)
            if segment_time_coverage:
                lines.extend(["", "## Segment Time Coverage"])
                lines.extend(segment_time_coverage)
        render_section(
            "Pending Candidate Windows",
            _candidate_window_rows(self),
            lambda candidate: (
                f"{candidate.get('candidate_key') or candidate.get('candidate_id')} "
                f"{_format_time_range(candidate.get('time_range'))} "
                f"segment={candidate.get('segment_id')} "
                f"status={candidate.get('status') or 'pending'} "
                f"targets={','.join(str(item) for item in _sequence_items(candidate.get('matched_targets'))) or '-'} "
                f"modalities={','.join(str(item) for item in _sequence_items(candidate.get('source_modalities'))) or '-'}"
            ).rstrip(),
            hint='verify_window(candidate_key="obs_*:cand_*")',
        )

        render_section(
            "Verification Results",
            _verification_result_rows(self),
            lambda result: (
                f"{result.get('target_id') or '-'} "
                f"verdict={result.get('verdict') or 'uncertain'} "
                f"time={_format_time_range(result.get('scope', {}).get('time_range') if isinstance(result.get('scope'), Mapping) else None) or '-'} "
                f"conf={result.get('confidence') or '-'} "
                f"mem={result.get('committed_memory_id') or '-'}"
            ).rstrip(),
            hint='read_workspace(section="observations_by_id")',
        )

        render_section(
            "Deferred Observations",
            self._deferred_observations(),
            lambda item: f"{item.get('observation_id')} until={item.get('until') or '-'} reason={item.get('reason') or '-'}",
            hint='read_workspace(section="observations_by_id")',
        )

        render_section(
            "Committed Memory",
            self.memory_entries(),
            lambda entry: (
                f"{entry.entry_id} [{entry.kind}"
                + (f" supports {entry.supports_option}" if entry.supports_option else "")
                + f"] {entry.claim}"
            ),
            hint='read_workspace(section="memory")',
        )

        option_evidence_map = _render_option_evidence_map(question=question, memory_entries=self.memory_entries())
        if option_evidence_map:
            lines.extend(["", option_evidence_map])

        render_section(
            "Pinned Anchors",
            self.read_pinned_anchors(),
            lambda anchor: (
                f"{anchor.get('anchor_id')} "
                f"{_format_time_range(anchor.get('time_range') or [anchor.get('start_sec'), anchor.get('end_sec')])} "
                f"{str(anchor.get('excerpt', '')).strip()}"
            ).rstrip(),
            hint='read_workspace(section="pinned_anchors")',
        )

        render_section(
            "Plan Notes",
            self._note_bullets("notes/plan.md"),
            lambda item: str(item),
            hint='read_workspace(section="plan")',
        )

        render_section(
            "Entities",
            self.read_workspace_section("entities"),
            lambda entity: f"{entity.get('entity_id')} {entity.get('kind')} {entity.get('name')}",
            hint='read_workspace(section="entities")',
        )

        render_section(
            "Open Questions",
            self._note_bullets("notes/open_questions.md"),
            lambda item: str(item),
            hint='read_workspace(section="open_questions")',
        )

        lines.extend(["", "## Recent Activity"])
        recent = self.observation_dispositions()[-max(0, int(max_recent)) :]
        if recent:
            for item in recent:
                obs_id = str(item.get("observation_id") or "")
                observation = self.get_observation(obs_id)
                tool = observation.tool if observation is not None else "?"
                claim_head = (observation.claim if observation is not None else "")[:80]
                lines.append(f"- {obs_id} ({tool} -> {item.get('disposition')}): {claim_head}".rstrip())
        else:
            lines.append("(none)")
        approx_tokens = max(1, sum(len(line) for line in lines) // 4)
        lines.extend(["", "## Budget", f"round {self.current_round()}/?; workspace tokens ~{approx_tokens}"])
        return "\n".join(lines)

    def render_commit_view(self, *, question: str, observation_id: str) -> str:
        observation = self._require_observation(observation_id)
        raw_output = observation.raw_output if isinstance(observation.raw_output, Mapping) else {}
        lines = [
            "# Pending Observation",
            f"obs_id: {observation.observation_id}",
            f"tool: {observation.tool}",
            f"claim: {observation.claim}",
            f"limitations: {observation.limitations or '-'}",
        ]

        regions = _mapping_list(raw_output.get("regions")) or _mapping_list(observation.regions)
        if regions:
            lines.extend(["", "## Scope"])
            for region in regions[:3]:
                time_range = region.get("time_range") or [region.get("start_sec"), region.get("end_sec")]
                segment_id = str(region.get("segment_id") or "-")
                lines.append(f"- segment={segment_id} time={_format_time_range(time_range) or '-'}")
            if len(regions) > 3:
                lines.append(f"... more regions hidden: {len(regions) - 3}")

        facts = _mapping_list(raw_output.get("facts"))
        if facts:
            lines.extend(["", "## Facts"])
            for fact in facts[:8]:
                text = str(fact.get("text") or fact.get("claim") or fact.get("excerpt") or "").strip()
                source_kind = str(fact.get("source_kind") or fact.get("kind") or "fact")
                time_range = _format_time_range(fact.get("time_range"))
                confidence = str(fact.get("confidence") or "").strip()
                suffix = f" (conf={confidence})" if confidence else ""
                time_part = f" @{time_range}" if time_range else ""
                lines.append(f"- [{source_kind}{time_part}] {text}{suffix}".rstrip())
            if len(facts) > 8:
                lines.append(f"... more facts hidden: {len(facts) - 8}")

        produced = _mapping_list(raw_output.get("produced_anchors"))
        if not produced:
            produced = [anchor.to_dict() for anchor in self.observation_anchors(observation.observation_id)]
        candidate_anchor_ids = [
            str(item).strip()
            for item in _sequence_items(raw_output.get("candidate_anchor_ids"))
            if str(item).strip()
        ]
        if not candidate_anchor_ids:
            candidate_anchor_ids = [
                str(anchor.get("anchor_id") or "").strip()
                for anchor in produced
                if str(anchor.get("anchor_id") or "").strip()
            ]
        if candidate_anchor_ids:
            lines.extend(["", "## Candidate Anchor IDs", "candidate_anchor_ids: " + ", ".join(candidate_anchor_ids)])
        if produced:
            lines.extend(["", "## Candidate Anchors (verbatim excerpts you may pin)"])
            for anchor in produced[:8]:
                anchor_id = str(anchor.get("anchor_id") or "").strip()
                modality = str(anchor.get("modality") or anchor.get("source_kind") or "anchor")
                excerpt = str(anchor.get("excerpt") or "").strip()
                lines.append(f"- {anchor_id} [{modality}]: {excerpt}")
            if len(produced) > 8:
                lines.append(f"... more candidate anchors hidden: {len(produced) - 8}")

        results = _mapping_list(raw_output.get("results"))
        if results:
            lines.extend(["", "## Search Hits"])
            for hit in results[:8]:
                hit_id = str(hit.get("hit_id") or "-")
                modality = str(hit.get("modality") or "-")
                excerpt = str(hit.get("excerpt") or "").strip()
                lines.append(f"- {hit_id} [{modality}] {excerpt}".rstrip())
            if len(results) > 8:
                lines.append(f"... more search hits hidden: {len(results) - 8}")

        lines.extend(["", "## Relevant Committed State"])
        memory_entries = self.memory_entries()[-5:]
        if memory_entries:
            for entry in memory_entries:
                lines.append(f"- {entry.entry_id}: {entry.claim}")
        else:
            lines.append("(none)")
        lines.extend(
            [
                "",
                "## Dispositions Available",
                "commit_observation | reject_observation | defer_observation | no_commit_needed",
            ]
        )
        del question
        return "\n".join(lines)

    def current_round(self) -> int:
        rounds = [
            int(event.get("payload", {}).get("round", 0) or 0)
            for event in self._read_jsonl_dicts("trace.jsonl")
            if str(event.get("type", "")) == "iterative_round_start" and isinstance(event.get("payload"), Mapping)
        ]
        return max(rounds) if rounds else 0

    def memory_snapshot_text(self, *, max_entries: int = 12) -> str:
        entries = self.memory_entries()
        if not entries:
            return ""
        lines: list[str] = []
        for entry in entries[-max_entries:]:
            support = f" {entry.supports_option}" if entry.supports_option else ""
            lines.append(f"- {entry.entry_id} [{entry.kind}{support}, {entry.confidence}]")
            lines.append(f"  claim: {entry.claim}")
            if entry.anchors:
                anchor_refs = ", ".join(f"{anchor.observation_id} / {anchor.anchor_id}" for anchor in entry.anchors)
                lines.append(f"  anchors: {anchor_refs}")
                excerpt = entry.anchors[0].excerpt
                if excerpt:
                    lines.append(f"  excerpt: \"{excerpt}\"")
        return "\n".join(lines)

    def uncommitted_observations_text(self, *, current_round: int | None = None, max_items: int = 8) -> str:
        committed_observation_ids = {
            anchor.observation_id
            for entry in self.memory_entries()
            for anchor in entry.anchors
        }
        lines: list[str] = []
        for observation in self.read_observations():
            if observation.observation_id in committed_observation_ids:
                continue
            if self.observation_status(observation.observation_id) in {"committed", "rejected", "acknowledged", "auto_acknowledged"}:
                continue
            anchors = self._anchors_for_observation(observation.observation_id)
            if not anchors:
                continue
            segment = _observation_segment_id(observation)
            label = f"{observation.tool}({segment})" if segment else observation.tool
            lines.append(f"- {observation.observation_id} {label}")
            claim = normalized_text(observation.claim)
            if claim:
                lines.append(f"  claim: {claim[:240]}")
            lines.append("  anchors: " + ", ".join(anchor.anchor_id for anchor in anchors[:5]))
            if len(lines) >= max_items * 3:
                break
        return "\n".join(lines)

    def uncommitted_observations(self) -> list[dict[str, Any]]:
        committed_observation_ids = {
            anchor.observation_id
            for entry in self.memory_entries()
            for anchor in entry.anchors
        }
        observations: list[dict[str, Any]] = []
        for observation in self.read_observations():
            if observation.observation_id in committed_observation_ids:
                continue
            if self.observation_status(observation.observation_id) in {"committed", "rejected", "acknowledged", "auto_acknowledged"}:
                continue
            anchors = self._anchors_for_observation(observation.observation_id)
            if not anchors:
                continue
            observations.append(
                {
                    "observation_id": observation.observation_id,
                    "tool": observation.tool,
                    "segment_id": _observation_segment_id(observation),
                    "claim": observation.claim,
                    "anchor_ids": [anchor.anchor_id for anchor in anchors],
                    "anchors": [anchor.to_dict() for anchor in anchors],
                }
            )
        return observations

    def observation_anchors(self, observation_id: str) -> list[SourceAnchor]:
        return self._anchors_for_observation(observation_id)

    def _anchors_for_observation(self, observation_id: str) -> list[SourceAnchor]:
        return [anchor for anchor in self.read_produced_anchors() if anchor.observation_id == observation_id]

    def _produced_anchors_for_observation(
        self,
        anchors: Any,
        *,
        observation_id: str,
    ) -> list[SourceAnchor]:
        if not isinstance(anchors, Sequence) or isinstance(anchors, (str, bytes)):
            return []
        resolved: list[SourceAnchor] = []
        for item in anchors:
            if not isinstance(item, Mapping):
                continue
            payload = dict(item)
            if str(payload.get("observation_id", "")) in {"", "__pending__"}:
                payload["observation_id"] = observation_id
            resolved.append(SourceAnchor.from_mapping(payload))
        return resolved

    def _next_memory_id(self) -> str:
        return f"mem_{len(self._read_jsonl_dicts('memory.jsonl')) + 1:04d}"

    def _known_option_ids(self) -> set[str]:
        registry = getattr(self, "target_registry", None)
        options_by_id = getattr(registry, "options_by_id", {})
        if isinstance(options_by_id, Mapping):
            return {str(key) for key in options_by_id.keys()}
        return set()

    def promote_textual_answer_evidence_payload(
        self,
        *,
        tool_name: str,
        raw_output: Mapping[str, Any],
        observation_id: str = "",
    ) -> dict[str, Any]:
        """Return raw_output plus deterministic registry-backed transcript bindings."""

        promoted, _generated_rows = self._promoted_textual_answer_evidence(
            tool_name=tool_name,
            raw_output=raw_output,
            observation_id=observation_id,
        )
        return promoted

    def _apply_post_observation_hooks(self, observation: Observation) -> Observation:
        if not _legacy_binder_enabled():
            return observation
        promoted_raw_output, generated_rows = self._promoted_textual_answer_evidence(
            tool_name=observation.tool,
            raw_output=observation.raw_output,
            observation_id=observation.observation_id,
        )
        if promoted_raw_output == observation.raw_output:
            return observation

        promoted = replace(observation, raw_output=promoted_raw_output)
        rows = self._read_jsonl_dicts("observations.jsonl")
        for row in rows:
            if str(row.get("observation_id", "")) != observation.observation_id:
                continue
            row["raw_output"] = dict(promoted_raw_output)
            row["confidence_signal"] = str(
                row.get("confidence_signal", "")
                or promoted_raw_output.get("confidence_signal", "")
            )
            break
        self._write_jsonl("observations.jsonl", rows)

        written = 0
        for index, row in enumerate(generated_rows, start=1):
            payload = dict(row)
            payload.setdefault("obs_id", promoted.observation_id)
            payload.setdefault("observation_id", promoted.observation_id)
            payload.setdefault("evidence_id", f"ev_answer_{promoted.observation_id}_{index:02d}")
            self.write_evidence_row(payload)
            written += 1
        if written:
            self.write_trace_event(
                "post_observation_textual_evidence_promoted",
                {
                    "tool": promoted.tool,
                    "observation_id": promoted.observation_id,
                    "row_count": written,
                },
            )
            ordered_written = sum(1 for row in generated_rows if str(row.get("evidence_type", "")) == "ordered_list")
            if ordered_written:
                self.write_trace_event(
                    "post_observation_ordered_list_promoted",
                    {
                        "tool": promoted.tool,
                        "observation_id": promoted.observation_id,
                        "row_count": ordered_written,
                    },
                )
        return promoted

    def _promoted_textual_answer_evidence(
        self,
        *,
        tool_name: str,
        raw_output: Mapping[str, Any],
        observation_id: str,
    ) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
        if not _tool_allows_textual_promotion(tool_name=tool_name, raw_output=raw_output):
            return dict(raw_output), []

        targets = _workspace_registry_targets(self)
        ordered_sets = _workspace_ordered_sets(self)
        self.write_trace_event(
            "post_observation_textual_evidence_considered",
            {
                "tool": tool_name,
                "observation_id": observation_id,
                "segment_id": str(raw_output.get("segment_id", "")),
            },
        )
        if not targets and not ordered_sets:
            self.write_trace_event(
                "post_observation_textual_evidence_rejected",
                {"tool": tool_name, "observation_id": observation_id, "reason": "no_target_registry"},
            )
            return dict(raw_output), []

        text_sources = _promotion_text_sources(raw_output)
        if not text_sources:
            self.write_trace_event(
                "post_observation_textual_evidence_rejected",
                {"tool": tool_name, "observation_id": observation_id, "reason": "no_textual_source"},
            )
            return dict(raw_output), []
        text_sources = [
            {
                **source,
                "source_tool": tool_name,
                "anchors_for_vlm": raw_output.get("anchors_for_vlm", ()),
            }
            for source in text_sources
        ]

        relations = _workspace_registry_relations(self, targets=targets)
        options = _workspace_registry_options(self)
        evidence_bindings: list[dict[str, Any]] = []
        relation_bindings: list[dict[str, Any]] = []
        answer_rows: list[Mapping[str, Any]] = []
        for source in text_sources:
            source_targets = _targets_for_text_source(targets=targets, source=source)
            if not source_targets:
                continue
            result = TranscriptEvidenceBinder().bind(
                text=str(source["text"]),
                targets=source_targets,
                relations=relations,
                obs_id=observation_id,
                segment_id=str(raw_output.get("segment_id", "")),
                start_sec=_optional_float(source.get("start_sec")) or _optional_float(raw_output.get("start_sec")),
                source=str(source["source"]),
            )
            binding_payloads = [_binding_payload(binding) for binding in result.evidence_bindings]
            relation_payloads = [
                *_ordered_sequence_relation_bindings(
                    text=str(source["text"]),
                    targets=source_targets,
                    relations=relations,
                    options=options,
                    raw_output=raw_output,
                    observation_id=observation_id,
                    source=source,
                ),
                *[dict(asdict(binding)) for binding in result.relation_bindings],
            ]
            evidence_bindings.extend(binding_payloads)
            relation_bindings.extend(relation_payloads)
            answer_rows.extend(
                _answer_rows_from_bindings(
                    raw_output=raw_output,
                    bindings=binding_payloads,
                    relations=relation_payloads,
                    all_relations=relations,
                    source=source,
                )
            )
            answer_rows.extend(
                _ordered_list_answer_rows_from_source(
                    raw_output=raw_output,
                    source=source,
                    ordered_sets=ordered_sets,
                    observation_id=observation_id,
                )
            )

        promoted = dict(raw_output)
        existing_evidence = _mapping_list(raw_output.get("evidence_bindings"))
        existing_relations = _mapping_list(raw_output.get("relation_bindings"))
        existing_rows = _mapping_list(raw_output.get("answer_evidence_rows"))
        merged_evidence = _dedupe_mapping_rows([*existing_evidence, *evidence_bindings], keys=("evidence_id", "target_id", "source"))
        merged_relations = _dedupe_mapping_rows([*relation_bindings, *existing_relations], keys=("relation_id", "binding_id", "source"))
        merged_rows = _dedupe_mapping_rows([*answer_rows, *existing_rows], keys=("evidence_id", "tool", "event_label"))
        promoted["evidence_bindings"] = merged_evidence
        promoted["relation_bindings"] = merged_relations
        promoted["answer_evidence_rows"] = merged_rows

        existing_row_keys = {_mapping_key(row, keys=("evidence_id", "tool", "event_label")) for row in existing_rows}
        generated_rows = [
            row
            for row in merged_rows
            if _mapping_key(row, keys=("evidence_id", "tool", "event_label")) not in existing_row_keys
            or (
                bool(row.get("_workspace_promoted"))
                and observation_id
                and str(row.get("source_observation_id", "")) == observation_id
            )
        ]
        if not generated_rows:
            self.write_trace_event(
                "post_observation_textual_evidence_rejected",
                {"tool": tool_name, "observation_id": observation_id, "reason": "no_match"},
            )
        return promoted, generated_rows

    def annotate_candidate_option_relations(
        self,
        *,
        observation_ids: Sequence[str],
        relations: Sequence[Mapping[str, Any]],
        assigned_by: str = "answer_agent",
    ) -> int:
        """Attach AnswerAgent option mappings to cited visual observations."""

        target_ids = [str(item) for item in observation_ids if str(item)]
        normalized_relations = _candidate_option_relations(relations)
        if not target_ids or not normalized_relations:
            return 0

        rows = self._read_observation_dicts()
        evidence_records = self._load_evidence_records()
        evidence_by_id = {record.evidence_id: record for record in evidence_records}
        ledger_by_observation = _ledger_records_by_observation(evidence_records)
        changed = 0
        mapped_count = 0
        orphan_count = 0
        rejected_count = 0
        for observation in rows:
            observation_id = str(observation.get("observation_id", ""))
            if observation_id not in target_ids:
                continue
            if str(observation.get("tool", "")) in self.CONTEXT_ONLY_TOOLS:
                continue
            raw_output = observation.get("raw_output", {})
            if not isinstance(raw_output, Mapping):
                raw_output = {}
            scoped_relations = _relations_for_observation(
                normalized_relations,
                observation_id=observation_id,
                default_observation_id=target_ids[0],
                assigned_by=assigned_by,
            )
            if not scoped_relations:
                continue
            if _reject_answer_agent_relations_for_observation(raw_output=raw_output, relations=scoped_relations):
                rejected_count += len(scoped_relations)
                continue
            scoped_with_parents: list[tuple[dict[str, Any], EvidenceRecord]] = []
            for relation in scoped_relations:
                parent_record = _mapped_relation_parent(
                    relation,
                    observation_id=observation_id,
                    evidence_by_id=evidence_by_id,
                    ledger_by_observation=ledger_by_observation,
                )
                if parent_record is None:
                    orphan_count += 1
                    continue
                relation_with_parent = dict(relation)
                relation_with_parent["parent_evidence_id"] = parent_record.evidence_id
                scoped_with_parents.append((relation_with_parent, parent_record))
            if not scoped_with_parents:
                continue
            existing_relations = _candidate_option_relations(raw_output.get("candidate_option_relations"))
            merged = _merge_candidate_option_relations(
                existing_relations,
                [relation for relation, _parent_record in scoped_with_parents],
            )
            if merged == existing_relations:
                continue
            updated_raw_output = dict(raw_output)
            updated_raw_output["candidate_option_relations"] = merged
            observation["raw_output"] = updated_raw_output
            changed += 1
            for relation, parent_record in scoped_with_parents:
                if _candidate_option_relation_present(existing_relations, relation):
                    continue
                self.write_evidence(
                    _mapped_evidence_record(
                        workspace=self,
                        observation=observation,
                        relation=relation,
                        parent_record=parent_record,
                    )
                )
                mapped_count += 1

        if changed:
            self._write_jsonl("observations.jsonl", rows)
            self._rebuild_evidence_table_from_observations(rows)
            self.write_trace_event(
                "answer_agent_relations_persisted",
                {
                    "observation_ids": target_ids,
                    "relation_count": sum(
                        _observation_relation_count(row)
                        for row in rows
                        if str(row.get("observation_id", "")) in target_ids
                    ),
                    "assigned_by": assigned_by,
                    "mapped_evidence_count": mapped_count,
                    "mapped_evidence_orphan_count": orphan_count,
                },
            )
        elif orphan_count:
            self.write_trace_event(
                "answer_agent_relations_persisted",
                {
                    "observation_ids": target_ids,
                    "relation_count": 0,
                    "assigned_by": assigned_by,
                    "mapped_evidence_count": 0,
                    "mapped_evidence_orphan_count": orphan_count,
                },
            )
        if rejected_count:
            self.write_trace_event(
                "answer_agent_relations_rejected",
                {
                    "observation_ids": target_ids,
                    "relation_count": rejected_count,
                    "assigned_by": assigned_by,
                    "reason": "unsafe_observation_integrity",
                },
            )
        return changed

    def write_trace_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        self._append_jsonl(
            "trace.jsonl",
            {
                "type": event_type,
                "created_at": _utc_now(),
                "payload": dict(payload),
            },
        )

    @property
    def run_id(self) -> str:
        return self.root.name

    def create_manifest(
        self,
        *,
        video_path: str,
        segment_id: str | None,
        start_sec: float,
        end_sec: float,
        target_nframes: int,
        nframes: int,
        sampling_policy: SamplingPolicy,
        frame_times_sec: Sequence[float],
        frame_times_approximate: bool,
        created_by_tool: str,
        observation_id: str,
        budget_reason: BudgetReason,
        materialized_paths: Sequence[str] | None = None,
    ) -> FrameSetManifest:
        manifest = FrameSetManifest(
            frame_set_id=f"fs_{self.run_id}_{self._next_manifest_seq():05d}",
            video_path=str(video_path),
            segment_id=segment_id,
            start_sec=float(start_sec),
            end_sec=float(end_sec),
            nframes=int(nframes),
            target_nframes=int(target_nframes),
            sampling_policy=sampling_policy,
            frame_times_sec=[float(item) for item in frame_times_sec],
            frame_times_approximate=bool(frame_times_approximate),
            created_by_tool=str(created_by_tool),
            observation_id=str(observation_id),
            budget_reason=budget_reason,
            contract_version=CONTRACT_VERSION,
            materialized_paths=[str(item) for item in materialized_paths or []],
            created_at=time.time(),
        )
        self._append_jsonl("frame_sets/manifests.jsonl", asdict(manifest))
        return manifest

    def get_manifest(self, frame_set_id: str) -> FrameSetManifest | None:
        for payload in self._read_jsonl_dicts("frame_sets/manifests.jsonl"):
            if str(payload.get("frame_set_id", "")) == str(frame_set_id):
                return FrameSetManifest.from_mapping(payload)
        return None

    def load_all_manifests(self) -> list[FrameSetManifest]:
        return [
            FrameSetManifest.from_mapping(payload)
            for payload in self._read_jsonl_dicts("frame_sets/manifests.jsonl")
        ]

    def link_manifest(self, observation_id: str, frame_set_id: str) -> None:
        payload = {
            "observation_id": str(observation_id),
            "frame_set_id": str(frame_set_id),
            "created_at": _utc_now(),
        }
        self._append_jsonl("frame_sets/observation_links.jsonl", payload)
        self.write_trace_event("observation_manifest_link", payload)

    def get_observation(self, obs_id: str) -> Observation | None:
        for payload in self._read_observation_dicts():
            if str(payload.get("observation_id", "")) == str(obs_id):
                return Observation.from_mapping(payload)
        return None

    def next_evidence_id(self, stage: str, sequence_offset: int = 0) -> str:
        return f"ev_{stage}_{self.run_id}_{self._next_evidence_seq() + sequence_offset:05d}"

    def write_evidence(self, record: EvidenceRecord) -> None:
        self._append_jsonl("evidence.jsonl", asdict(record))

    def append_to_timeline(
        self,
        *,
        obs_id: str,
        entity: str,
        observed_at_sec: float | None = None,
        window: Sequence[float] | None = None,
        confidence_signal: str = "",
        claim: str = "",
    ) -> dict[str, Any]:
        row = _normalize_timeline_row(
            {
                "obs_id": obs_id,
                "entity": entity,
                "observed_at_sec": observed_at_sec,
                "window": window,
                "confidence_signal": confidence_signal,
                "claim": claim,
            }
        )
        timeline_path = self.root / "timeline.md"
        if not timeline_path.exists():
            timeline_path.write_text("# Timeline\n\n", encoding="utf-8")
        with timeline_path.open("a", encoding="utf-8") as handle:
            handle.write(f"- {json.dumps(row, ensure_ascii=True, sort_keys=True)}\n")
        return row

    def append_to_timeline_candidate(
        self,
        *,
        obs_id: str,
        entity: str,
        observed_at_sec: float | None = None,
        window: Sequence[float] | None = None,
        confidence_signal: str = "",
        claim: str = "",
        grounding_quality: str = "",
        requires_visual_verification: bool = False,
        source: str = "",
        candidate_id: str = "",
    ) -> dict[str, Any]:
        row = _normalize_timeline_row(
            {
                "obs_id": obs_id,
                "entity": entity,
                "observed_at_sec": observed_at_sec,
                "window": window,
                "confidence_signal": confidence_signal,
                "claim": claim,
                "grounding_quality": grounding_quality,
                "requires_visual_verification": requires_visual_verification,
                "source": source,
                "candidate_id": candidate_id,
            }
        )
        candidate_path = self.root / "timeline_candidates.md"
        if not candidate_path.exists():
            candidate_path.write_text("# Timeline Candidates\n\n", encoding="utf-8")
        with candidate_path.open("a", encoding="utf-8") as handle:
            handle.write(f"- {json.dumps(row, ensure_ascii=True, sort_keys=True)}\n")
        return row

    def append_timeline_from_observation(self, observation: Observation) -> dict[str, Any] | None:
        if observation.tool == "locate_targets_in_segment":
            raw_output = observation.raw_output if isinstance(observation.raw_output, Mapping) else {}
            rows = raw_output.get("ordered_list_timeline_rows", [])
            appended = []
            if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
                for row in rows:
                    if not isinstance(row, Mapping):
                        continue
                    appended.append(
                        self.append_to_timeline_candidate(
                            obs_id=observation.observation_id,
                            entity=str(row.get("entity", "")),
                            observed_at_sec=(
                                None
                                if row.get("observed_at_sec") is None
                                else float(row.get("observed_at_sec", 0.0) or 0.0)
                            ),
                            window=row.get("window", []),
                            confidence_signal=str(row.get("confidence_signal", "") or observation.confidence_signal),
                            claim=str(row.get("claim", "") or observation.claim),
                            grounding_quality=str(row.get("grounding_quality", "")),
                            requires_visual_verification=bool(row.get("requires_visual_verification", False)),
                            source=str(row.get("source", "")),
                            candidate_id=str(row.get("candidate_id", "")),
                        )
                    )
            return appended[0] if appended else None
        if observation.tool == "verify_segment_anchors":
            raw_output = observation.raw_output if isinstance(observation.raw_output, Mapping) else {}
            rows = raw_output.get("timeline_rows", [])
            appended = []
            if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
                for row in rows:
                    if not isinstance(row, Mapping):
                        continue
                    appended.append(
                        self.append_to_timeline(
                            obs_id=observation.observation_id,
                            entity=str(row.get("entity", "")),
                            observed_at_sec=(
                                None
                                if row.get("observed_at_sec") is None
                                else float(row.get("observed_at_sec", 0.0) or 0.0)
                            ),
                            window=row.get("window", []),
                            confidence_signal=str(row.get("confidence_signal", "") or observation.confidence_signal),
                            claim=str(row.get("claim", "") or observation.claim),
                        )
                    )
            return appended[0] if appended else None
        if observation.tool not in {"vision_read", "inspect_segment", "caption_segment"}:
            return None
        if is_unsupported_claim(observation.claim):
            return None
        raw_output = observation.raw_output if isinstance(observation.raw_output, Mapping) else {}
        entity = _observation_event_label(raw_output=raw_output, claim=observation.claim)
        if not entity and observation.tool == "caption_segment":
            entity = _caption_timeline_entity(observation.claim)
        if not entity:
            return None
        observed_at_sec = _observed_at_sec(raw_output)
        window = _observation_time_range(
            {
                "raw_output": raw_output,
                "regions": observation.regions,
            }
        )
        confidence_signal = _timeline_confidence_signal(
            raw_output=raw_output,
            observation_signal=observation.confidence_signal,
            observed_at_sec=observed_at_sec,
        )
        return self.append_to_timeline(
            obs_id=observation.observation_id,
            entity=entity,
            observed_at_sec=observed_at_sec,
            window=window,
            confidence_signal=confidence_signal,
            claim=observation.claim,
        )

    def read_timeline_sorted(self) -> list[dict[str, Any]]:
        timeline_path = self.root / "timeline.md"
        if not timeline_path.exists():
            return []
        rows = []
        with timeline_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line.startswith("- {"):
                    continue
                try:
                    payload = json.loads(line[2:])
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, Mapping):
                    rows.append(_normalize_timeline_row(payload))
        return sorted(rows, key=_timeline_sort_key)

    def ensure_hypothesis(self, question: str = "") -> None:
        path = self.root / "hypothesis.md"
        if not path.exists():
            path.write_text("# Hypothesis\n\n", encoding="utf-8")

    def write_hypothesis(self, slots: Mapping[str, Any]) -> dict[str, dict[str, str]]:
        normalized = _normalize_hypothesis_slots(slots)
        lines = ["# Hypothesis", ""]
        for slot_name in sorted(normalized):
            slot = normalized[slot_name]
            evidence_obs_id = slot["evidence_obs_id"] or "-"
            lines.append(f"- {slot_name} | status: {slot['status']} | evidence_obs_id: {evidence_obs_id}")
        lines.append("")
        (self.root / "hypothesis.md").write_text("\n".join(lines), encoding="utf-8")
        return normalized

    def read_hypothesis(self) -> dict[str, dict[str, str]]:
        path = self.root / "hypothesis.md"
        if not path.exists():
            return {}
        slots: dict[str, dict[str, str]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped.startswith("- "):
                continue
            parts = [part.strip() for part in stripped[2:].split("|")]
            if not parts:
                continue
            slot_name = parts[0]
            payload: dict[str, str] = {"status": "empty", "evidence_obs_id": ""}
            for part in parts[1:]:
                if ":" not in part:
                    continue
                key, value = part.split(":", 1)
                payload[key.strip()] = "" if value.strip() == "-" else value.strip()
            if slot_name:
                slots[slot_name] = _normalize_hypothesis_slot(payload)
        return slots

    def read_hypothesis_text(self) -> str:
        path = self.root / "hypothesis.md"
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def update_hypothesis_slot(
        self,
        *,
        slot_name: str,
        status: str,
        evidence_obs_id: str = "",
    ) -> dict[str, str]:
        slots = self.read_hypothesis()
        slot_key = str(slot_name).strip()
        if not slot_key:
            raise ValueError("slot_name must be non-empty")
        slots[slot_key] = _normalize_hypothesis_slot(
            {
                "status": status,
                "evidence_obs_id": evidence_obs_id,
            }
        )
        self.write_hypothesis(slots)
        return slots[slot_key]

    def unsatisfied_hypothesis_slots(self) -> list[str]:
        slots = self.read_hypothesis()
        return [
            name
            for name, slot in sorted(slots.items())
            if str(slot.get("status", "")).strip().lower() != "satisfied"
        ]

    def write_evidence_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Append one answer-facing evidence-table row to evidence_table.jsonl."""

        payload = _normalize_evidence_row(
            row,
            evidence_id=str(row.get("evidence_id") or self._next_evidence_row_id()),
        )
        self._append_jsonl("evidence_table.jsonl", payload)
        return payload

    def evidence_table_row_count(self) -> int:
        """Return the persisted answer-facing evidence table row count."""

        return len(self._read_jsonl_dicts("evidence_table.jsonl"))

    def observation_count(self, *, tool_name: str | None = None) -> int:
        """Return persisted observation count, optionally filtered by tool."""

        rows = self._read_observation_dicts()
        if tool_name is None:
            return len(rows)
        return sum(1 for row in rows if str(row.get("tool", "")) == str(tool_name))

    def read_observations(self, *, tool_name: str | None = None) -> list[Observation]:
        """Return persisted observations, optionally filtered by tool name."""

        observations = [Observation.from_mapping(row) for row in self._read_observation_dicts()]
        if tool_name is None:
            return observations
        return [observation for observation in observations if observation.tool == str(tool_name)]

    def recent_tool_outputs(self, *, limit: int = 3) -> list[dict[str, Any]]:
        """Return recent observation payloads for verbatim-safe planner feedback."""

        selected = self.read_observations()[-max(0, int(limit or 0)) :]
        evidence_rows_by_observation: dict[str, Mapping[str, Any]] = {}
        for row in self._read_jsonl_dicts("evidence_table.jsonl"):
            obs_id = str(row.get("observation_id") or row.get("obs_id") or "").strip()
            if obs_id and obs_id not in evidence_rows_by_observation:
                evidence_rows_by_observation[obs_id] = row
        outputs: list[dict[str, Any]] = []
        for observation in selected:
            payload = {
                "observation_id": observation.observation_id,
                "tool": observation.tool,
                "claim": observation.claim,
                "confidence": observation.confidence,
                "raw_output": _compact_recent_tool_payload(observation.raw_output),
            }
            evidence_row = evidence_rows_by_observation.get(observation.observation_id)
            if evidence_row is not None:
                payload.update(
                    {
                        "in_evidence_table": True,
                        "evidence_id": str(evidence_row.get("evidence_id", "")),
                        "segment_id": str(evidence_row.get("segment_id") or evidence_row.get("segment") or ""),
                        "modality": str(
                            evidence_row.get("modality")
                            or evidence_row.get("claim_modality")
                            or evidence_row.get("grounding_quality")
                            or ""
                        ),
                        "verdict": str(evidence_row.get("verdict") or evidence_row.get("status") or "supported"),
                    }
                )
            outputs.append(payload)
        return outputs

    def read_evidence_table_v3(
        self,
        *,
        question: str,
        options: Sequence[str] = (),
        include_legacy_worker_votes: bool = False,
    ) -> dict[str, Any]:
        """Return the persisted evidence-table artifact grouped for answer arbitration."""

        option_map = _option_letter_map(options)
        groups: dict[str, list[dict[str, Any]]] = {letter: [] for letter in option_map}
        groups.setdefault("unassigned", [])
        rows: list[dict[str, Any]] = []
        for raw_row in self._read_jsonl_dicts("evidence_table.jsonl"):
            row = _normalize_evidence_row(raw_row)
            tool_name = str(row.get("tool", ""))
            if tool_name in self.NAVIGATION_TOOLS:
                continue
            if self.ANSWER_EVIDENCE_TOOLS and tool_name not in self.ANSWER_EVIDENCE_TOOLS:
                continue
            if _tool_emits_candidate_hints_only(tool_name):
                row["supported_option"] = None
            elif row.get("legacy_worker_vote") and not include_legacy_worker_votes:
                row["supported_option"] = None
            else:
                row = _project_target_row_to_option(row, registry_obj=getattr(self, "target_registry", None))
                option_source = (
                    row.get("supported_option")
                    or _supported_option_from_relations(row.get("candidate_option_relations"), option_map=option_map)
                    or _bare_option_from_claim(str(row.get("claim", "")), option_map=option_map)
                    or _supported_option_from_claim(str(row.get("claim", "")))
                )
                row["supported_option"] = _normalize_supported_option(option_source, option_map=option_map)
            group_key = str(row.get("supported_option") or "unassigned")
            groups.setdefault(group_key, [])
            rows.append(row)
            groups[group_key].append(row)

        sorted_groups = {key: _sort_evidence_rows(value) for key, value in groups.items()}
        sorted_rows = [
            row
            for key in sorted(sorted_groups, key=_option_sort_key)
            for row in sorted_groups[key]
        ]
        return {
            "schema_version": "EvidenceTableV3",
            "source_artifact": "evidence_table.jsonl",
            "question": question,
            "options": list(options),
            "groups": sorted_groups,
            "rows": sorted_rows,
            "timeline": self.read_timeline_sorted(),
        }

    def load_evidence(self, evidence_id: str) -> EvidenceRecord | None:
        for payload in self._read_jsonl_dicts("evidence.jsonl"):
            if str(payload.get("evidence_id", "")) == str(evidence_id):
                return EvidenceRecord.from_mapping(payload)
        return None

    def _load_evidence_records(self) -> list[EvidenceRecord]:
        return [EvidenceRecord.from_mapping(payload) for payload in self._read_jsonl_dicts("evidence.jsonl")]

    def mapped_evidence_records(
        self,
        *,
        observation_ids: Sequence[str] = (),
        selected_option: str | None = None,
    ) -> list[EvidenceRecord]:
        cited = {str(item) for item in observation_ids if str(item)}
        option = _relation_option_letter(selected_option)
        records = []
        for record in self._load_evidence_records():
            if record.stage != "mapped":
                continue
            if cited and str(record.observation_id or "") not in cited:
                continue
            relation = record.content.get("candidate_option_relation", {})
            if not isinstance(relation, Mapping):
                continue
            if option and _relation_option_letter(relation.get("option")) != option:
                continue
            if str(relation.get("relation", "")).strip().lower() not in {"support", "supports", "supported"}:
                continue
            records.append(record)
        return records

    def evidence_chain(self, leaf_id: str) -> list[EvidenceRecord]:
        by_id = {
            str(payload.get("evidence_id", "")): EvidenceRecord.from_mapping(payload)
            for payload in self._read_jsonl_dicts("evidence.jsonl")
            if payload.get("evidence_id")
        }
        chain = []
        current = by_id.get(str(leaf_id))
        seen = set()
        while current is not None and current.evidence_id not in seen:
            seen.add(current.evidence_id)
            chain.append(current)
            if current.parent_id is None:
                break
            current = by_id.get(current.parent_id)
        return list(reversed(chain))

    def next_proposal_id(self) -> str:
        return f"mp_{self.run_id}_{self._next_proposal_seq():05d}"

    def write_proposal(self, proposal: MapUpdateProposal) -> None:
        self._append_jsonl("map_proposals.jsonl", asdict(proposal))

    def load_pending_proposals(self) -> list[MapUpdateProposal]:
        return [
            MapUpdateProposal.from_mapping(payload)
            for payload in self._read_jsonl_dicts("map_proposals.jsonl")
            if payload.get("committed_at") is None
        ]

    def mark_proposal_committed(self, proposal_id: str, *, committed_at: float | None = None) -> MapUpdateProposal | None:
        proposals = [MapUpdateProposal.from_mapping(payload) for payload in self._read_jsonl_dicts("map_proposals.jsonl")]
        if not proposals:
            return None
        resolved_at = time.time() if committed_at is None else float(committed_at)
        committed: MapUpdateProposal | None = None
        rewritten = []
        for proposal in proposals:
            if proposal.proposal_id == proposal_id:
                proposal = MapUpdateProposal(
                    proposal_id=proposal.proposal_id,
                    target_segment_id=proposal.target_segment_id,
                    update_type=proposal.update_type,
                    payload=dict(proposal.payload),
                    source_evidence_id=proposal.source_evidence_id,
                    source_frame_set_id=proposal.source_frame_set_id,
                    confidence=proposal.confidence,
                    proposed_at=proposal.proposed_at,
                    committed_at=proposal.committed_at if proposal.committed_at is not None else resolved_at,
                )
                committed = proposal
            rewritten.append(asdict(proposal))
        self._write_jsonl("map_proposals.jsonl", rewritten)
        if committed is not None:
            self.write_trace_event(
                "map_proposal_committed",
                {
                    "proposal_id": committed.proposal_id,
                    "target_segment_id": committed.target_segment_id,
                    "update_type": committed.update_type,
                    "committed_at": committed.committed_at,
                },
            )
        return committed

    def write_reflection_memory(self, *, route: str, failure_tag: str, rule: str) -> None:
        """Persist one compact Reflexion-style policy rule after a failure."""

        clean_rule = " ".join(str(rule).split())
        if not clean_rule:
            return
        payload = {
            "created_at": _utc_now(),
            "route": str(route),
            "failure_tag": str(failure_tag),
            "rule": clean_rule,
        }
        self._append_jsonl("reflection_memory.jsonl", payload)
        self.write_trace_event("reflection_memory", payload)

    def reflection_memory(self, *, route: str | None = None, max_items: int = 5) -> list[str]:
        """Return newest compact policy rules, optionally scoped by route."""

        path = self.root / "reflection_memory.jsonl"
        if not path.exists() or max_items <= 0:
            return []
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, Mapping):
                    continue
                if route is not None and str(payload.get("route", "")) != str(route):
                    continue
                rows.append(payload)
        compact = []
        seen = set()
        for payload in reversed(rows):
            item = (
                f"{payload.get('route', 'unknown')}: {payload.get('failure_tag', 'failure')} -> "
                f"{payload.get('rule', '')}"
            )
            if item in seen:
                continue
            seen.add(item)
            compact.append(item)
            if len(compact) >= max_items:
                break
        return list(reversed(compact))

    def write_text_artifact(
        self,
        relative_path: str | Path,
        text: str,
        *,
        max_chars: int | None = None,
    ) -> Mapping[str, Any]:
        """Write verbose debug text as an artifact and return compact metadata."""

        path = Path(relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Artifact path must stay inside the workspace: {relative_path}")
        full_path = self.root / path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        original_chars = len(text)
        stored_text = text
        truncated = False
        if max_chars is not None and max_chars >= 0 and original_chars > max_chars:
            stored_text = text[:max_chars]
            stored_text += f"\n\n[truncated {original_chars - max_chars} chars]\n"
            truncated = True

        full_path.write_text(stored_text, encoding="utf-8")
        return {
            "path": path.as_posix(),
            "chars": original_chars,
            "stored_chars": len(stored_text),
            "truncated": truncated,
        }

    def write_ledger_entry(
        self,
        observation: Observation,
        *,
        parent_records: Sequence[EvidenceRecord] = (),
    ) -> list[EvidenceRecord]:
        artifacts = ", ".join(observation.input_artifacts) or "-"
        base_limitation = observation.limitations or "-"
        confidence_signal = observation.confidence_signal or str(observation.raw_output.get("confidence_signal", ""))
        has_worker_option_vote = _has_worker_option_vote(
            tool_name=observation.tool,
            raw_output=observation.raw_output,
            claim=observation.claim,
        )
        if has_worker_option_vote:
            sanitized_claim = _claim_without_legacy_worker_vote(observation.claim)
            note = f"legacy local-worker option vote quarantined; fact_text: {sanitized_claim}"
            base_limitation = f"{base_limitation}; {note}" if base_limitation != "-" else note

        ledger_records: list[EvidenceRecord] = []
        parents: list[EvidenceRecord | None] = list(parent_records) or [None]
        with (self.root / "ledger.md").open("a", encoding="utf-8") as handle:
            for parent_record in parents:
                claim = _ledger_claim(observation, parent_record=parent_record)
                if has_worker_option_vote:
                    claim = _claim_without_legacy_worker_vote(claim)
                grounding_quality = _ledger_grounding_quality(observation, parent_record=parent_record)
                frame_set_id = (
                    parent_record.frame_set_id if parent_record is not None and parent_record.frame_set_id else observation.frame_set_id
                )
                ledger_record = EvidenceRecord(
                    evidence_id=self.next_evidence_id("ledger"),
                    stage="ledger",
                    parent_id=parent_record.evidence_id if parent_record is not None else None,
                    tool=observation.tool,
                    observation_id=observation.observation_id,
                    frame_set_id=frame_set_id,
                    content={
                        "claim": claim,
                        "regions": _ledger_regions(observation, parent_record=parent_record),
                        "limitations": base_limitation,
                        "artifacts": list(observation.input_artifacts),
                        "confidence_signal": confidence_signal,
                    },
                    grounding_quality=grounding_quality,  # type: ignore[arg-type]
                    confidence=parent_record.confidence if parent_record is not None else observation.confidence,
                    created_at=time.time(),
                )
                self.write_evidence(ledger_record)
                if observation.tool in self.ANSWER_EVIDENCE_TOOLS:
                    self.write_evidence_row(
                        _evidence_row_from_observation(
                            observation=observation,
                            evidence_record=ledger_record,
                            claim=claim,
                            grounding_quality=grounding_quality,
                            confidence=ledger_record.confidence,
                        )
                    )
                ledger_records.append(ledger_record)
                ledger_claim = _ledger_markdown_field(claim)
                ledger_limitations = _ledger_markdown_field(base_limitation)
                ledger_artifacts = _ledger_markdown_field(artifacts)
                line = (
                    f"- `{observation.observation_id}` | ev: `{ledger_record.evidence_id}` | "
                    f"fs: `{frame_set_id or '-'}` | gq: `{grounding_quality}` | "
                    f"tool: `{observation.tool}` | confidence: {ledger_record.confidence:.2f} | "
                    f"artifacts: {ledger_artifacts} | claim: {ledger_claim} | limitations: {ledger_limitations}\n"
                )
                handle.write(line)
        raw_relations = _candidate_option_relations(observation.raw_output.get("candidate_option_relations"))
        if _tool_emits_candidate_hints_only(observation.tool):
            raw_relations = []
        if not raw_relations and not has_worker_option_vote:
            raw_relations = _candidate_option_relations_from_supported_option(observation)
        if raw_relations:
            self.annotate_candidate_option_relations(
                observation_ids=[observation.observation_id],
                relations=raw_relations,
                assigned_by=observation.tool,
            )
        return ledger_records

    def compact_ledger_text(
        self,
        *,
        max_working_observations: int = 4,
        max_visual_evidence: int = 8,
    ) -> str:
        """Return a bounded answer-facing context derived from the raw ledger trace."""

        ledger_path = self.root / "ledger.md"
        if not ledger_path.exists():
            return ""
        raw_ledger = ledger_path.read_text(encoding="utf-8")
        entries = _parse_ledger_entries(raw_ledger)
        if not entries:
            return raw_ledger
        entries = _attach_observation_payloads(
            entries,
            observations=self._read_observation_dicts(),
        )

        visual_entries = [
            entry for entry in entries if str(entry.get("tool", "")) in self.ANSWER_EVIDENCE_TOOLS
        ][-max_visual_evidence:]
        context_entries = [
            entry for entry in entries if str(entry.get("tool", "")) in self.CONTEXT_ONLY_TOOLS
        ][-max_working_observations:]
        navigation_entries = [
            entry for entry in entries if str(entry.get("tool", "")) in self.NAVIGATION_TOOLS
        ]
        working_entries = (
            [
                entry
                for entry in entries
                if str(entry.get("tool", "")) not in self.CONTEXT_ONLY_TOOLS
                and str(entry.get("tool", "")) not in self.NAVIGATION_TOOLS
            ][-max_working_observations:]
            if max_working_observations > 0
            else []
        )

        sections = ["# Compact Evidence Context", ""]
        sections.append("## Long-Term Visual Evidence")
        if visual_entries:
            sections.extend(_format_compact_entry(entry) for entry in visual_entries)
        else:
            sections.append("(none)")

        sections.extend(["", "## Context-Only Visual Hints (Not Answer Support)"])
        if context_entries:
            sections.extend(_format_context_only_entry(entry) for entry in context_entries)
        else:
            sections.append("(none)")

        sections.extend(["", "## Navigation Summary"])
        if navigation_entries:
            sections.extend(
                _format_navigation_entry(entry)
                for entry in navigation_entries
            )
        else:
            sections.append("(none)")

        sections.extend(["", "## Short-Term Working Buffer"])
        if working_entries:
            sections.extend(_format_rawish_entry(entry) for entry in working_entries)
        else:
            sections.append("(none)")
        sections.append("")
        return "\n".join(sections)

    def evidence_table(
        self,
        *,
        question: str,
        options: Sequence[str] = (),
        include_legacy_worker_votes: bool = False,
    ) -> dict[str, Any]:
        """Return an option-grouped answer evidence table for arbitration."""

        option_map = _option_letter_map(options)
        groups: dict[str, list[dict[str, Any]]] = {letter: [] for letter in option_map}
        groups.setdefault("unassigned", [])
        rows = []

        for observation in self._read_observation_dicts():
            tool_name = str(observation.get("tool", ""))
            if tool_name in self.NAVIGATION_TOOLS:
                continue
            if self.ANSWER_EVIDENCE_TOOLS and tool_name not in self.ANSWER_EVIDENCE_TOOLS:
                continue

            raw_output = observation.get("raw_output", {})
            if not isinstance(raw_output, Mapping):
                raw_output = {}
            legacy_worker_vote = _has_worker_option_vote(
                tool_name=tool_name,
                raw_output=raw_output,
                claim=str(observation.get("claim", "")),
                option_map=option_map,
            )
            display_claim = str(observation.get("claim", ""))
            if legacy_worker_vote:
                display_claim = _claim_without_legacy_worker_vote(display_claim)
            if _tool_emits_candidate_hints_only(tool_name):
                option_source = None
            else:
                option_source = (
                    raw_output.get("supported_option")
                    or raw_output.get("supported_option_letter")
                    or raw_output.get("answer_option")
                    or _first_item(raw_output.get("supported_options"))
                    or _supported_option_from_relations(raw_output.get("candidate_option_relations"), option_map=option_map)
                    or _bare_option_from_claim(str(observation.get("claim", "")), option_map=option_map)
                    or _supported_option_from_claim(str(observation.get("claim", "")))
                )
            if legacy_worker_vote and not include_legacy_worker_votes:
                supported_option = None
            else:
                projected_relation = _project_target_row_to_option(
                    {
                        "tool": tool_name,
                        "event_label": _observation_event_label(raw_output=raw_output, claim=display_claim),
                        "target_id": raw_output.get("target_id") or raw_output.get("target_ref"),
                        "evidence_binding": raw_output.get("evidence_binding"),
                        "candidate_option_relations": raw_output.get("candidate_option_relations"),
                        "grounding_quality": _grounding_quality(
                            raw_output=raw_output,
                            limitations=str(observation.get("limitations", "")),
                            confidence_signal=str(observation.get("confidence_signal", "")),
                        ),
                        "confidence": float(observation.get("confidence", 0.0) or 0.0),
                        "obs_id": str(observation.get("observation_id", "")),
                    },
                    registry_obj=getattr(self, "target_registry", None),
                )
                if projected_relation.get("candidate_option_relations"):
                    raw_output = {
                        **raw_output,
                        "candidate_option_relations": _merge_candidate_option_relations(
                            _candidate_option_relations(raw_output.get("candidate_option_relations")),
                            _candidate_option_relations(projected_relation.get("candidate_option_relations")),
                        ),
                    }
                    option_source = option_source or _supported_option_from_relations(
                        raw_output.get("candidate_option_relations"),
                        option_map=option_map,
                    )
                supported_option = _normalize_supported_option(option_source, option_map=option_map)
            group_key = supported_option or "unassigned"
            groups.setdefault(group_key, [])

            row = EvidenceRowV2(
                obs_id=str(observation.get("observation_id", "")),
                segment_id=_observation_segment_id(observation),
                time_range=_observation_time_range(observation),
                tool=tool_name,
                supported_option=supported_option,
                event_label=_observation_event_label(
                    raw_output=raw_output,
                    claim=display_claim,
                ),
                claim=display_claim,
                confidence=float(observation.get("confidence", 0.0) or 0.0),
                grounding_quality=_grounding_quality(
                    raw_output=raw_output,
                    limitations=str(observation.get("limitations", "")),
                    confidence_signal=str(observation.get("confidence_signal", "")),
                ),
                candidate_option_relations=_candidate_option_relations(raw_output.get("candidate_option_relations")),
                confidence_signal=str(observation.get("confidence_signal", "") or raw_output.get("confidence_signal", "")),
                mutex_group_id=str(raw_output.get("mutex_group_id", "")),
                legacy_worker_vote=legacy_worker_vote,
                limitations=str(observation.get("limitations", "")),
                artifact=_first_item(observation.get("input_artifacts")) or "",
                **_evidence_provenance_fields(raw_output),
            ).to_dict()
            rows.append(row)
            groups[group_key].append(row)

        sorted_groups = {key: _sort_evidence_rows(value) for key, value in groups.items()}
        sorted_rows = [
            row
            for key in sorted(sorted_groups, key=_option_sort_key)
            for row in sorted_groups[key]
        ]
        return {
            "question": question,
            "options": list(options),
            "groups": sorted_groups,
            "rows": sorted_rows,
            "timeline": self.read_timeline_sorted(),
        }

    def evidence_table_v2(
        self,
        *,
        question: str,
        options: Sequence[str] = (),
        include_legacy_worker_votes: bool = False,
    ) -> dict[str, Any]:
        """Return the v4 typed evidence table with explicit schema metadata."""

        if self._read_jsonl_dicts("evidence_table.jsonl"):
            table = self.read_evidence_table_v3(
                question=question,
                options=options,
                include_legacy_worker_votes=include_legacy_worker_votes,
            )
            rows = table.get("rows", [])
            legacy_count = 0
            if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
                legacy_count = sum(
                    1 for row in rows if isinstance(row, Mapping) and bool(row.get("legacy_worker_vote"))
                )
            return {
                **table,
                "schema_version": "EvidenceTableV2",
                "artifact_schema_version": table["schema_version"],
                "legacy_worker_vote_rows": legacy_count,
            }

        table = self.evidence_table(
            question=question,
            options=options,
            include_legacy_worker_votes=include_legacy_worker_votes,
        )
        rows = table.get("rows", [])
        legacy_count = 0
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            legacy_count = sum(
                1 for row in rows if isinstance(row, Mapping) and bool(row.get("legacy_worker_vote"))
            )
        return {
            "schema_version": "EvidenceTableV2",
            **table,
            "legacy_worker_vote_rows": legacy_count,
        }

    def evidence_status_summary(
        self,
        *,
        question: str,
        options: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Return compact known/missing evidence state for planner prompts."""

        table = self.evidence_table_v2(question=question, options=options)
        groups = table.get("groups", {})
        if not isinstance(groups, Mapping):
            groups = {}

        option_status: dict[str, dict[str, Any]] = {}
        for option, raw_rows in groups.items():
            option_key = str(option)
            if option_key == "unassigned":
                continue
            rows = list(raw_rows) if isinstance(raw_rows, Sequence) and not isinstance(raw_rows, (str, bytes)) else []
            strong = [
                row
                for row in rows
                if isinstance(row, Mapping)
                and (
                    str(row.get("grounding_quality", "")) in {"visually_confirmed", "global_sparse", "indexed_transcript"}
                    or str(row.get("confidence_signal", "")) == "asr_claim_binding_supported"
                )
            ]
            weak = [
                row
                for row in rows
                if isinstance(row, Mapping) and str(row.get("grounding_quality", "")) in {"inferred", "weak", "external_knowledge"}
            ]
            option_status[option_key] = {
                "strong_evidence_count": len(strong),
                "weak_evidence_count": len(weak),
                "has_visual_citation": any(
                    isinstance(row, Mapping)
                    and str(row.get("tool", "")) in self.ANSWER_EVIDENCE_TOOLS
                    and str(row.get("tool", "")) not in self.NAVIGATION_TOOLS
                    for row in rows
                ),
            }

        total_options = len(option_status)
        covered = sum(1 for status in option_status.values() if int(status["strong_evidence_count"]) > 0)
        rows = table.get("rows", [])
        evidence_rows = list(rows) if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) else []
        claims = [
            str(row.get("claim", "")).strip()
            for row in evidence_rows
            if isinstance(row, Mapping) and str(row.get("claim", "")).strip()
        ]
        duplicate_observations = len(claims) - len(set(claims))
        coverage_pct = round(covered / total_options, 3) if total_options else 0.0

        return {
            "option_coverage": f"{covered}/{total_options}",
            "coverage_pct": coverage_pct,
            "option_status": option_status,
            "duplicate_observations": duplicate_observations,
            "total_evidence_rows": len(evidence_rows),
            "hypothesis_gaps": self.unsatisfied_hypothesis_slots(),
        }

    def observed_segment_window(self, segment_id: str) -> Mapping[str, Any] | None:
        """Return the latest observed time window for a segment id from tool outputs."""

        found: Mapping[str, Any] | None = None
        for observation in self._read_observation_dicts():
            for candidate in _iter_segment_window_dicts(observation):
                if str(candidate.get("segment_id", "")) != segment_id:
                    continue
                if candidate.get("start_sec") is None or candidate.get("end_sec") is None:
                    continue
                found = candidate
        return found

    def grounding_candidates(self, observation_id: str, *, max_candidates: int = 5) -> list[dict[str, Any]]:
        """Return normalized grounding candidate windows from a grounding observation."""

        for observation in self._read_observation_dicts():
            if str(observation.get("observation_id", "")) != str(observation_id):
                continue
            raw_output = observation.get("raw_output", {})
            candidates = []
            if isinstance(raw_output, Mapping):
                raw_candidates = raw_output.get("candidates", [])
                if isinstance(raw_candidates, Sequence) and not isinstance(raw_candidates, (str, bytes)):
                    candidates.extend(item for item in raw_candidates if isinstance(item, Mapping))
            regions = observation.get("regions", [])
            if isinstance(regions, Sequence) and not isinstance(regions, (str, bytes)):
                candidates.extend(item for item in regions if isinstance(item, Mapping))
            normalized = []
            seen = set()
            for candidate in candidates:
                if candidate.get("segment_id") is None or candidate.get("start_sec") is None or candidate.get("end_sec") is None:
                    continue
                key = (str(candidate.get("segment_id")), float(candidate.get("start_sec")), float(candidate.get("end_sec")))
                if key in seen:
                    continue
                seen.add(key)
                normalized.append(
                    {
                        "segment_id": key[0],
                        "start_sec": key[1],
                        "end_sec": key[2],
                        "reason": str(candidate.get("reason") or candidate.get("relevance_reason") or ""),
                        "modality": str(candidate.get("modality") or candidate.get("channel") or ""),
                        "confidence": float(candidate.get("confidence", candidate.get("score", 0.0)) or 0.0),
                    }
                )
                if len(normalized) >= max_candidates:
                    break
            return normalized
        return []

    def has_non_navigation_visual_citation(self, citations: Sequence[str]) -> bool:
        """Check whether cited ids include answer-facing visual evidence."""

        cited = {str(item).strip() for item in citations if str(item).strip()}
        if not cited:
            return False
        for entry in self.memory_entries():
            if entry.entry_id in cited and entry.kind in self.GROUNDED_MEMORY_KINDS:
                return True
        for observation in self._read_observation_dicts():
            if str(observation.get("observation_id", "")) not in cited:
                continue
            tool_name = str(observation.get("tool", ""))
            if tool_name in self.ANSWER_EVIDENCE_TOOLS and tool_name not in self.NAVIGATION_TOOLS:
                return True
        for raw_row in self._read_jsonl_dicts("evidence_table.jsonl"):
            row = _normalize_evidence_row(raw_row)
            if str(row.get("obs_id", "")) not in cited and str(row.get("evidence_id", "")) not in cited:
                continue
            tool_name = str(row.get("tool", ""))
            if tool_name in self.ANSWER_EVIDENCE_TOOLS and tool_name not in self.NAVIGATION_TOOLS:
                return True
        return False

    def evidence_chain_summaries(
        self,
        *,
        observation_ids: Sequence[str] = (),
        selected_option: str | None = None,
        max_chains: int | None = 100,
    ) -> list[dict[str, Any]]:
        chains = []
        for mapped_record in self.mapped_evidence_records(
            observation_ids=observation_ids,
            selected_option=selected_option,
        ):
            chain = self.evidence_chain(mapped_record.evidence_id)
            stages = [str(record.stage) for record in chain]
            if not _complete_evidence_chain(stages):
                continue
            chains.append(
                {
                    "leaf_evidence_id": mapped_record.evidence_id,
                    "observation_id": mapped_record.observation_id,
                    "frame_set_id": mapped_record.frame_set_id,
                    "stages": stages,
                    "records": [_compact_evidence_record(record) for record in chain],
                }
            )
            if max_chains is not None and len(chains) >= max_chains:
                break
        return chains

    def export_evidence_chains(
        self,
        *,
        output_path: str | Path = "artifacts/evidence_chains/evidence_chains.json",
        max_chains: int = 100,
    ) -> Mapping[str, Any]:
        chains = self.evidence_chain_summaries(max_chains=max_chains)
        if len(chains) < max_chains:
            chains.extend(self.memory_evidence_chain_summaries(max_chains=max_chains - len(chains)))
        total_mapped = len(self.mapped_evidence_records())
        payload: dict[str, Any] = {
            "schema_version": "EvidenceChainsV1",
            "workspace_root": self.root.as_posix(),
            "chain_count": len(chains),
            "total_mapped_evidence": total_mapped,
            "total_memory_evidence": len(_answer_support_memory_entries(self.memory_entries())),
            "truncated": total_mapped > len(chains),
            "chains": chains,
        }
        self.write_text_artifact(
            output_path,
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
        )
        self.write_trace_event(
            "evidence_chains_export",
            {
                "schema_version": payload["schema_version"],
                "path": Path(output_path).as_posix(),
                "chain_count": len(chains),
                "truncated": bool(payload["truncated"]),
            },
        )
        return payload

    def memory_evidence_chain_summaries(self, *, max_chains: int = 100) -> list[dict[str, Any]]:
        chains: list[dict[str, Any]] = []
        observations_by_id = {observation.observation_id: observation for observation in self.read_observations()}
        for entry in _answer_support_memory_entries(self.memory_entries()):
            anchor = next((candidate for candidate in entry.anchors if candidate.observation_id), None)
            observation = observations_by_id.get(anchor.observation_id) if anchor is not None else None
            stages = ["memory"]
            records: list[dict[str, Any]] = []
            if observation is not None:
                stages.insert(0, "observation")
                records.append(
                    {
                        "stage": "observation",
                        "observation_id": observation.observation_id,
                        "tool": observation.tool,
                        "claim": _bounded_inline(observation.claim, 240),
                        "confidence": observation.confidence,
                    }
                )
            if anchor is not None:
                anchor_stage_index = 1 if observation is not None else 0
                stages.insert(anchor_stage_index, "anchor")
                records.append(
                    {
                        "stage": "anchor",
                        "anchor_id": anchor.anchor_id,
                        "observation_id": anchor.observation_id,
                        "source_kind": anchor.source_kind,
                        "segment_id": anchor.segment_id,
                        "time_range": [anchor.start_sec, anchor.end_sec],
                        "modality": anchor.modality,
                        "excerpt": _bounded_inline(anchor.excerpt, 240),
                    }
                )
            records.append(
                {
                    "stage": "memory",
                    "memory_id": entry.entry_id,
                    "kind": entry.kind,
                    "claim": _bounded_inline(entry.claim, 240),
                    "supports_option": entry.supports_option,
                    "confidence": entry.confidence,
                    "anchor_ids": [anchor.anchor_id for anchor in entry.anchors],
                }
            )
            chains.append(
                {
                    "leaf_evidence_id": entry.entry_id,
                    "memory_id": entry.entry_id,
                    "observation_id": observation.observation_id if observation is not None else "",
                    "frame_set_id": observation.frame_set_id if observation is not None else None,
                    "stages": stages,
                    "records": records,
                }
            )
            if len(chains) >= max_chains:
                break
        return chains

    def export_longvideoagent_trajectory(
        self,
        *,
        question: str,
        video_path: str,
        final: Mapping[str, Any] | None = None,
        verifier_result: Mapping[str, Any] | None = None,
        reward_tags: Sequence[str] = (),
        output_path: str | Path = "artifacts/trajectories/longvideoagent_trajectory.json",
    ) -> Mapping[str, Any]:
        """Export a compact state-action-observation trajectory for GRPO-style training."""

        trace_events = self._read_jsonl_dicts("trace.jsonl")
        observations = self._read_observation_dicts()
        observations_by_id = {
            str(observation.get("observation_id", "")): observation
            for observation in observations
            if observation.get("observation_id")
        }
        actions = _trajectory_actions(trace_events=trace_events, observations_by_id=observations_by_id)
        payload: dict[str, Any] = {
            "schema_version": "LongVideoAgentTrajectoryV1",
            "state": {
                "question": question,
                "video_path": video_path,
                "workspace_root": self.root.as_posix(),
                "observation_count": len(observations),
            },
            "actions": actions,
            "observations": [_trajectory_observation(observation) for observation in observations],
            "final": dict(final or {}),
            "verifier_result": dict(verifier_result or {}),
            "reward_tags": [str(tag) for tag in reward_tags],
        }
        self.write_text_artifact(
            output_path,
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
        )
        self.write_trace_event(
            "trajectory_export",
            {
                "schema_version": payload["schema_version"],
                "path": Path(output_path).as_posix(),
                "action_count": len(actions),
                "reward_tags": list(payload["reward_tags"]),
            },
        )
        return payload

    def _require_observation(self, observation_id: str) -> Observation:
        observation = self.get_observation(str(observation_id))
        if observation is None:
            raise ValueError(f"observation_validation_failed: unknown observation_id={observation_id}")
        return observation

    def _require_disposable_observation(self, observation_id: str) -> Observation:
        observation = self._require_observation(observation_id)
        status = self.observation_status(observation.observation_id)
        if status in {"committed", "rejected", "acknowledged"}:
            raise ValueError(f"disposition_validation_failed: observation already {status}")
        return observation

    def _validated_observation_ids(self, observation_ids: Sequence[str]) -> list[str]:
        resolved: list[str] = []
        for item in observation_ids:
            obs_id = str(item or "").strip()
            if not obs_id:
                continue
            self._require_observation(obs_id)
            resolved.append(obs_id)
        return resolved

    def _validate_commit_writes(self, observation: Observation, writes: Mapping[str, Any]) -> None:
        observation_anchor_ids = {
            anchor.anchor_id
            for anchor in self.observation_anchors(observation.observation_id)
            if anchor.anchor_id
        }
        pending_anchor_ids = {
            str(anchor.get("anchor_id") or anchor.get("candidate_anchor_id") or "").strip()
            for anchor in _mapping_list(writes.get("pinned_anchors"))
        }
        pending_anchor_ids.discard("")
        available_anchor_ids = set(self.read_produced_anchors_by_id()) | {
            str(anchor.get("anchor_id", "")) for anchor in self.read_pinned_anchors()
        } | pending_anchor_ids

        for bucket in ("entities", "events", "relations", "attributes", "memory"):
            for payload in _mapping_list(writes.get(bucket)):
                self._validated_observation_ids(_default_evidence_obs_ids(payload, observation.observation_id))

        for anchor_payload in _mapping_list(writes.get("pinned_anchors")):
            anchor_id = str(anchor_payload.get("anchor_id") or anchor_payload.get("candidate_anchor_id") or "").strip()
            if not anchor_id:
                raise ValueError("anchor_validation_failed: anchor_id required")
            if observation_anchor_ids and anchor_id not in observation_anchor_ids:
                raise ValueError(
                    f"anchor_validation_failed: anchor_id={anchor_id} not in observation produced_anchors "
                    f"(allowed: {sorted(observation_anchor_ids)})"
                )
            excerpt = str(anchor_payload.get("excerpt", "") or "").strip()
            self._validate_excerpt_in_observation(observation, excerpt)

        for entity_payload in _mapping_list(writes.get("entities")):
            if not str(entity_payload.get("name", "")).strip():
                raise ValueError("entity_validation_failed: name must not be empty")

        for event_payload in _mapping_list(writes.get("events")):
            if not str(event_payload.get("label") or event_payload.get("name") or "").strip():
                raise ValueError("event_validation_failed: label must not be empty")

        for attribute_payload in _mapping_list(writes.get("attributes")):
            if not str(attribute_payload.get("target", "")).strip() or not str(attribute_payload.get("name", "")).strip():
                raise ValueError("attribute_validation_failed: target and name are required")

        entity_names = self._entity_names() | {
            str(payload.get("name", "")).strip()
            for payload in _mapping_list(writes.get("entities"))
            if str(payload.get("name", "")).strip()
        }
        for relation_payload in _mapping_list(writes.get("relations")):
            subject = str(relation_payload.get("subject", "")).strip()
            predicate = str(relation_payload.get("predicate", "")).strip()
            objects = [str(item).strip() for item in _sequence_items(relation_payload.get("objects")) if str(item).strip()]
            if not subject or not predicate or not objects:
                raise ValueError("relation_validation_failed: subject, predicate, and objects are required")
            missing = [item for item in [subject, *objects] if item not in entity_names]
            if missing:
                raise ValueError("relation_validation_failed: unknown entity reference " + ", ".join(missing))

        for memory_payload in _mapping_list(writes.get("memory")):
            _validate_memory_commit_payload(memory_payload, observation=observation)
            _validate_memory_observation_provenance(observation, memory_payload)
            anchor_ids = [
                str(item).strip()
                for item in _sequence_items(memory_payload.get("anchor_ids"))
                if str(item).strip()
            ]
            for anchor_id in anchor_ids:
                if anchor_id not in available_anchor_ids:
                    raise ValueError(f"memory_validation_failed: unknown anchor_id={anchor_id}")
            option_id = str(memory_payload.get("supports_option", "") or "").strip()
            if option_id and self._known_option_ids() and option_id not in self._known_option_ids():
                raise ValueError(f"memory_validation_failed: unknown option {option_id}")
            for ref in _sequence_items(memory_payload.get("previous_memory_refs")):
                ref_text = str(ref).strip()
                if ref_text and self.get_memory(ref_text) is None:
                    raise ValueError(f"memory_validation_failed: unknown memory ref={ref_text}")

    def _validate_excerpt_in_observation(self, observation: Observation, excerpt: str) -> None:
        if not excerpt:
            return
        haystack = _observation_text_for_excerpt_validation(observation)
        if normalized_text(excerpt) not in normalized_text(haystack):
            raise ValueError(
                f"anchor_validation_failed: excerpt must appear in observation {observation.observation_id}"
            )

    def _validate_relation_entities(self, subject: str, objects: Sequence[str]) -> None:
        names = self._entity_names()
        missing = [item for item in [subject, *objects] if item and item not in names]
        if missing:
            raise ValueError("relation_validation_failed: unknown entity reference " + ", ".join(missing))

    def _entity_names(self) -> set[str]:
        return {
            str(row.get("name", "")).strip()
            for row in self._read_jsonl_dicts("entities/entities.jsonl")
            if str(row.get("name", "")).strip()
        }

    def _write_disposition(
        self,
        observation_id: str,
        *,
        disposition: str,
        reason: str = "",
        until: str = "",
        writes: Mapping[str, Any] | None = None,
        defer_count: int | None = None,
    ) -> dict[str, Any]:
        payload = {
            "disposition_id": self._next_structured_id("observations/disposition.jsonl", "disp"),
            "observation_id": str(observation_id),
            "disposition": str(disposition),
            "reason": str(reason or ""),
            "until": str(until or ""),
            "defer_count": defer_count if defer_count is not None else self._defer_count(observation_id),
            "writes": dict(writes or {}),
            "created_at": _utc_now(),
        }
        self._append_jsonl("observations/disposition.jsonl", payload)
        self.write_trace_event(
            "observation_disposition_recorded",
            {
                "observation_id": payload["observation_id"],
                "disposition": payload["disposition"],
                "disposition_id": payload["disposition_id"],
            },
        )
        return payload

    def _latest_disposition(self, observation_id: str) -> dict[str, Any] | None:
        latest = None
        for row in self.observation_dispositions():
            if str(row.get("observation_id", "")) == str(observation_id):
                latest = row
        return latest

    def _defer_count(self, observation_id: str) -> int:
        return sum(
            1
            for row in self.observation_dispositions()
            if str(row.get("observation_id", "")) == str(observation_id)
            and str(row.get("disposition", "")) == "deferred"
        )

    def _deferred_observations(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for observation in self.read_observations():
            latest = self._latest_disposition(observation.observation_id)
            if latest is None or str(latest.get("disposition", "")) != "deferred":
                continue
            rows.append(
                {
                    **dict(latest),
                    "tool": observation.tool,
                    "claim": observation.claim,
                }
            )
        return rows

    def _transaction_files(self) -> tuple[str, ...]:
        return (
            "produced_anchors.jsonl",
            "memory.jsonl",
            "trace.jsonl",
            "entities/entities.jsonl",
            "events/events.jsonl",
            "relations/relations.jsonl",
            "attributes/attributes.jsonl",
            "memory/memory.jsonl",
            "observations/disposition.jsonl",
            "pinned/pinned_anchors.jsonl",
            "notes/plan.md",
            "notes/open_questions.md",
        )

    def _snapshot_transaction_files(self) -> dict[str, bytes | None]:
        snapshot: dict[str, bytes | None] = {}
        for filename in self._transaction_files():
            path = self.root / filename
            snapshot[filename] = path.read_bytes() if path.exists() else None
        return snapshot

    def _restore_transaction_files(self, snapshot: Mapping[str, bytes | None]) -> None:
        for filename, content in snapshot.items():
            path = self.root / filename
            if content is None:
                if path.exists():
                    path.unlink()
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

    def _append_note_line(self, filename: str, text: str) -> None:
        path = self.root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("# Notes\n\n", encoding="utf-8")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"- {text}\n")

    def _read_note_text(self, filename: str) -> str:
        path = self.root / filename
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def _note_bullets(self, filename: str) -> list[str]:
        lines = []
        for line in self._read_note_text(filename).splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                lines.append(stripped[2:].strip())
        return lines

    def _next_structured_id(self, filename: str, prefix: str) -> str:
        return f"{prefix}_{len(self._read_jsonl_dicts(filename)) + 1:04d}"

    def _append_jsonl(self, filename: str, payload: Mapping[str, Any]) -> None:
        path = self.root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True))
            handle.write("\n")

    def _write_jsonl(self, filename: str, payloads: Sequence[Mapping[str, Any]]) -> None:
        path = self.root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for payload in payloads:
                handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True))
                handle.write("\n")

    def _read_jsonl_dicts(self, filename: str) -> list[dict[str, Any]]:
        path = self.root / filename
        if not path.exists():
            return []
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
        return rows

    def _read_observation_dicts(self) -> list[dict[str, Any]]:
        observations = self._read_jsonl_dicts("observations.jsonl")
        links = self._observation_manifest_links()
        if not links:
            return observations
        merged = []
        for observation in observations:
            observation_id = str(observation.get("observation_id", ""))
            if observation_id in links:
                observation = dict(observation)
                observation["frame_set_id"] = links[observation_id]
            merged.append(observation)
        return merged

    def _next_observation_id(self) -> str:
        existing = 0
        observations = self.root / "observations.jsonl"
        if observations.exists():
            with observations.open("r", encoding="utf-8") as handle:
                existing = sum(1 for line in handle if line.strip())
        return f"obs_{existing + 1:04d}"

    def _next_manifest_seq(self) -> int:
        return len(self._read_jsonl_dicts("frame_sets/manifests.jsonl")) + 1

    def _next_evidence_seq(self) -> int:
        return len(self._read_jsonl_dicts("evidence.jsonl")) + 1

    def _next_evidence_row_id(self) -> str:
        return f"ev_table_{self.run_id}_{len(self._read_jsonl_dicts('evidence_table.jsonl')) + 1:05d}"

    def _next_proposal_seq(self) -> int:
        return len(self._read_jsonl_dicts("map_proposals.jsonl")) + 1

    def _rebuild_evidence_table_from_observations(self, observations: Sequence[Mapping[str, Any]]) -> None:
        rows = []
        for observation in observations:
            tool_name = str(observation.get("tool", ""))
            if tool_name not in self.ANSWER_EVIDENCE_TOOLS:
                continue
            rows.append(
                _normalize_evidence_row(
                    _evidence_row_from_observation_mapping(
                        observation=observation,
                        evidence_id=f"ev_table_{self.run_id}_{len(rows) + 1:05d}",
                    )
                )
            )
        self._write_jsonl("evidence_table.jsonl", rows)

    def _observation_manifest_links(self) -> dict[str, str]:
        links: dict[str, str] = {}
        for payload in self._read_jsonl_dicts("frame_sets/observation_links.jsonl"):
            observation_id = str(payload.get("observation_id", ""))
            frame_set_id = str(payload.get("frame_set_id", ""))
            if observation_id and frame_set_id:
                links[observation_id] = frame_set_id
        return links


def _tool_allows_textual_promotion(*, tool_name: str, raw_output: Mapping[str, Any]) -> bool:
    if tool_name == "read_segment_detail":
        return bool(raw_output.get("promote_answer_evidence"))
    return tool_name in {"read_segment", "locate_targets_in_segment"}


def _legacy_binder_enabled() -> bool:
    if os.environ.get("HARNESS_FINAL_GATE_MODE", "").strip().lower() == "legacy":
        return True
    return os.environ.get("HARNESS_LEGACY_BINDER_TELEMETRY", "").strip().lower() in {"1", "true", "yes", "on"}


def _workspace_registry_targets(workspace: EvidenceWorkspace) -> list[TargetSpec]:
    registry = getattr(workspace, "target_registry", None)
    targets_by_id = getattr(registry, "targets_by_id", None)
    if not isinstance(targets_by_id, Mapping):
        return []
    return [target for target in targets_by_id.values() if isinstance(target, TargetSpec)]


def _workspace_registry_relations(workspace: EvidenceWorkspace, *, targets: Sequence[TargetSpec]) -> list[ClaimRelation]:
    target_ids = {target.target_id for target in targets}
    registry = getattr(workspace, "target_registry", None)
    relations_by_id = getattr(registry, "relations_by_id", {})
    values = relations_by_id.values() if hasattr(relations_by_id, "values") else ()
    return [
        relation
        for relation in values
        if isinstance(relation, ClaimRelation)
        and relation.source_target_id in target_ids
        and relation.destination_target_id in target_ids
    ]


def _workspace_registry_options(workspace: EvidenceWorkspace) -> list[Any]:
    registry = getattr(workspace, "target_registry", None)
    options_by_id = getattr(registry, "options_by_id", {})
    values = options_by_id.values() if hasattr(options_by_id, "values") else ()
    return list(values)


def _workspace_ordered_sets(workspace: EvidenceWorkspace) -> list[Any]:
    ordered_sets = getattr(workspace, "ordered_sets", ())
    if not isinstance(ordered_sets, Sequence) or isinstance(ordered_sets, (str, bytes)):
        return []
    return [ordered_set for ordered_set in ordered_sets if hasattr(ordered_set, "hypotheses")]


def _ordered_list_answer_rows_from_source(
    *,
    raw_output: Mapping[str, Any],
    source: Mapping[str, Any],
    ordered_sets: Sequence[Any],
    observation_id: str,
) -> list[Mapping[str, Any]]:
    del raw_output, source, ordered_sets, observation_id
    return []

def _promotion_text_sources(raw_output: Mapping[str, Any]) -> list[dict[str, Any]]:
    segment_id = str(raw_output.get("segment_id", ""))
    start_sec = _optional_float(raw_output.get("start_sec"))
    end_sec = _optional_float(raw_output.get("end_sec"))
    sources: list[dict[str, Any]] = []
    for key in ("raw_asr_excerpt", "asr_summary", "asr_text"):
        text = str(raw_output.get(key) or "").strip()
        if text:
            sources.append(
                {
                    "text": text,
                    "source": "indexed_transcript",
                    "modality": "asr",
                    "segment_id": segment_id,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                }
            )
            break
    ocr_text = str(raw_output.get("ocr_text") or "").strip()
    if ocr_text:
        sources.append(
            {
                "text": ocr_text,
                "source": "indexed_ocr",
                "modality": "ocr",
                "segment_id": segment_id,
                "start_sec": start_sec,
                "end_sec": end_sec,
            }
        )
    for region in _mapping_list(raw_output.get("regions")):
        region_segment_id = str(region.get("segment_id") or segment_id)
        region_start = _optional_float(region.get("start_sec")) or start_sec
        region_end = _optional_float(region.get("end_sec")) or end_sec
        for field, source_name, modality in (
            ("asr_text", "indexed_transcript", "asr"),
            ("ocr_text", "indexed_ocr", "ocr"),
        ):
            text = str(region.get(field) or "").strip()
            if not text:
                continue
            sources.append(
                {
                    "text": text,
                    "source": source_name,
                    "modality": modality,
                    "segment_id": region_segment_id,
                    "start_sec": region_start,
                    "end_sec": region_end,
                }
            )
    for candidate in _mapping_list(raw_output.get("candidates")):
        source_name = str(candidate.get("source") or "")
        if not source_name.startswith(("asr", "ocr")):
            continue
        snippet = str(candidate.get("snippet") or "").strip()
        if not snippet:
            continue
        modality = "ocr" if source_name.startswith("ocr") else "asr"
        sources.append(
            {
                "text": snippet,
                "source": "indexed_ocr" if modality == "ocr" else "indexed_transcript",
                "modality": modality,
                "segment_id": str(candidate.get("segment_id") or segment_id),
                "start_sec": _optional_float(candidate.get("start_sec")) or start_sec,
                "end_sec": _optional_float(candidate.get("end_sec")) or end_sec,
            }
        )
    return _dedupe_text_sources(sources)


def _targets_for_text_source(*, targets: Sequence[TargetSpec], source: Mapping[str, Any]) -> list[TargetSpec]:
    modality = str(source.get("modality") or "")
    allowed = {
        "asr": {ClaimModality.NARRATED_FACT, ClaimModality.MIXED, ClaimModality.UNKNOWN},
        "ocr": {ClaimModality.OCR_FACT, ClaimModality.MIXED, ClaimModality.UNKNOWN},
    }.get(modality, {ClaimModality.UNKNOWN, ClaimModality.MIXED})
    return [target for target in targets if target.modality_hint in allowed]


def _binding_payload(binding: Any) -> dict[str, Any]:
    payload = dict(asdict(binding))
    modality = payload.get("claim_modality")
    if isinstance(modality, ClaimModality):
        payload["claim_modality"] = modality.value
    return payload


def _ordered_sequence_relation_bindings(
    *,
    text: str,
    targets: Sequence[TargetSpec],
    relations: Sequence[ClaimRelation],
    options: Sequence[Any],
    raw_output: Mapping[str, Any],
    observation_id: str,
    source: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if str(source.get("modality") or "") != "asr":
        return []
    segment_id = str(source.get("segment_id") or raw_output.get("segment_id") or "")
    relation_by_id = {relation.relation_id: relation for relation in relations}
    target_by_id = {target.target_id: target for target in targets}
    rows: list[dict[str, Any]] = []
    for option in options:
        ordered_refs = tuple(str(ref) for ref in getattr(option, "target_sequence", ()) if str(ref))
        ordered_targets = [target_by_id[ref] for ref in ordered_refs if ref in target_by_id]
        if len(ordered_targets) != len(ordered_refs) or len(ordered_targets) < 2:
            continue
        sequence = build_ordered_transcript_sequence(
            text=text,
            targets=ordered_targets,
            segment_id=segment_id,
            obs_id=observation_id,
            start_sec=_optional_float(source.get("start_sec")),
            end_sec=_optional_float(source.get("end_sec")),
        )
        if sequence is None or sequence.status != "supported" or sequence.ordered_target_refs != ordered_refs:
            continue
        timestamp_order = [
            item.mention_start_sec
            for item in sequence.items
            if item.mention_start_sec is not None
        ]
        for relation_id in getattr(option, "required_relations", ()):
            relation = relation_by_id.get(str(relation_id))
            if relation is None:
                continue
            rows.append(
                {
                    "binding_id": f"rel_bind_{relation.relation_id}",
                    "obs_id": observation_id,
                    "relation_id": relation.relation_id,
                    "status": "supported",
                    "source": sequence.source,
                    "snippet": sequence.snippet,
                    "mention_timestamp_sec": _optional_float(source.get("start_sec")),
                    "ordered_target_refs": list(sequence.ordered_target_refs),
                    "evidence_ids": [sequence.evidence_id],
                    "timestamp_order": timestamp_order,
                    "modality": ClaimModality.NARRATED_FACT.value,
                }
            )
    return rows


def _answer_rows_from_bindings(
    *,
    raw_output: Mapping[str, Any],
    bindings: Sequence[Mapping[str, Any]],
    relations: Sequence[Mapping[str, Any]],
    all_relations: Sequence[ClaimRelation],
    source: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    supported_relations = [
        relation
        for relation in relations
        if str(relation.get("status", "")).strip().lower() == "supported"
    ]
    rows: list[Mapping[str, Any]] = []
    segment_id = str(source.get("segment_id") or raw_output.get("segment_id") or "")
    start_sec = _optional_float(source.get("start_sec")) or _optional_float(raw_output.get("start_sec"))
    end_sec = _optional_float(source.get("end_sec")) or _optional_float(raw_output.get("end_sec"))
    for binding in bindings:
        if str(binding.get("status", "")).strip().lower() != "supported":
            continue
        target_id = str(binding.get("target_id") or "")
        if not target_id:
            continue
        scoped_relations = [
            relation
            for relation in supported_relations
            if _relation_touches_target_id(
                relation_id=str(relation.get("relation_id", "")),
                target_id=target_id,
                relations=all_relations,
            )
        ]
        binding_payload = dict(binding)
        binding_payload["segment_id"] = segment_id
        binding_payload["relation_bindings"] = scoped_relations
        rows.append(
            {
                "evidence_id": str(binding.get("evidence_id") or ""),
                "tool": "transcript_evidence_binder",
                "segment_id": segment_id,
                "time_range": [start_sec, end_sec] if start_sec is not None and end_sec is not None else None,
                "event_label": target_id,
                "claim": (
                    f"TranscriptEvidenceBinder marked {target_id} as supported in "
                    f"{segment_id or 'segment'}: {binding.get('snippet', '')}"
                ),
                "confidence": 0.88,
                "grounding_quality": "indexed_transcript" if source.get("modality") == "asr" else "ocr_textual",
                "confidence_signal": "explicit transcript binding",
                "limitations": "Conservative registry-backed textual binding; no raw option text is promoted as a target.",
                "source": str(source.get("source") or ""),
                "snippet": str(binding.get("snippet") or ""),
                "evidence_binding": binding_payload,
            }
        )
    return rows


def _relation_touches_target_id(*, relation_id: str, target_id: str, relations: Sequence[ClaimRelation]) -> bool:
    for relation in relations:
        if relation.relation_id != relation_id:
            continue
        return target_id in {relation.source_target_id, relation.destination_target_id}
    return False


def _mapping_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _sequence_items(value: Any) -> tuple[Any, ...]:
    if value is None or isinstance(value, (str, bytes)):
        return () if value is None else (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _normalize_optional_time_range(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        try:
            return [float(value[0]), float(value[1])]
        except (TypeError, ValueError):
            return None
    return None


def _default_evidence_obs_ids(payload: Mapping[str, Any], default_observation_id: str) -> list[str]:
    raw_ids = payload.get("evidence_obs_ids")
    ids = [str(item).strip() for item in _sequence_items(raw_ids) if str(item).strip()]
    return ids or [str(default_observation_id)]


def _validate_memory_commit_payload(payload: Mapping[str, Any], *, observation: Observation | None = None) -> None:
    kind = str(payload.get("kind") or "support")
    if kind not in {
        "note",
        "support",
        "answer_support",
        "caption_support",
        "visual_support",
        "answer_conflict",
        "locator",
        "conflict",
        "contradicting",
        "negation",
        "reject",
        "hypothesis",
        "open_question",
        "synthesized",
        "synthesized_support",
        "answer_conflict_resolved",
        "unverified_capture",
        "retrieval_candidate",
        "local_negative",
        "navigation_note",
        "verification_uncertain",
        "contradiction",
    }:
        raise ValueError(f"memory_validation_failed: unknown kind={kind}")
    confidence = str(payload.get("confidence") or "medium")
    if confidence not in {"high", "medium", "low"}:
        raise ValueError(f"memory_validation_failed: unknown confidence={confidence}")
    if kind == "local_negative":
        metadata = _memory_commit_metadata(payload, observation=observation)
        scope = metadata.get("scope")
        if not _valid_local_negative_scope(scope):
            raise ValueError("memory_validation_failed: local_negative requires metadata.scope with segment_id and time_range")
        if _truthy(metadata.get("global_negation_allowed")):
            raise ValueError("memory_validation_failed: local_negative must set global_negation_allowed=false")
        if str(payload.get("supports_option", "") or "").strip():
            raise ValueError("memory_validation_failed: local_negative cannot support an answer option")


def _validate_memory_observation_provenance(observation: Observation, payload: Mapping[str, Any]) -> None:
    kind = str(payload.get("kind") or "support")
    if observation.tool == "explore":
        raw_output = observation.raw_output if isinstance(observation.raw_output, Mapping) else {}
        mode = str(raw_output.get("mode") or "").strip()
        support_status = str(raw_output.get("support_status") or "").strip()
        caption_supported = mode in {"caption_fact", "mixed"} and support_status in {
            "caption_supported",
            "partial_caption_supported",
        }
        if caption_supported:
            if kind not in {"caption_support", "answer_support", "retrieval_candidate", "navigation_note"}:
                raise ValueError(f"commit_validation_failed: explore {mode} cannot become {kind}")
        elif kind not in {"retrieval_candidate", "navigation_note"}:
            raise ValueError("commit_validation_failed: candidate-only explore observations cannot become answer support")
        return
    if observation.tool != "verify_window":
        return
    metadata = _memory_commit_metadata(payload, observation=observation)
    verdict = str(metadata.get("verdict") or "").strip()
    if kind == "answer_support" and verdict and verdict != "supported":
        raise ValueError(f"commit_validation_failed: {verdict} cannot become answer_support")
    if kind == "visual_support" and verdict and verdict not in {"supported", "not_found_in_window"}:
        raise ValueError(f"commit_validation_failed: {verdict} cannot become visual_support")
    if kind == "answer_conflict" and verdict and verdict != "contradicted":
        raise ValueError(f"commit_validation_failed: {verdict} cannot become answer_conflict")
    if kind == "verification_uncertain" and verdict and verdict != "uncertain":
        raise ValueError(f"commit_validation_failed: {verdict} cannot become verification_uncertain")
    if kind == "local_negative" and verdict and verdict != "not_found_in_window":
        raise ValueError(f"commit_validation_failed: {verdict} cannot become local_negative")


def _memory_commit_metadata(payload: Mapping[str, Any], *, observation: Observation | None = None) -> dict[str, object]:
    metadata = dict(payload.get("metadata", {}) or {})
    if "scope" in payload and "scope" not in metadata:
        metadata["scope"] = payload["scope"]
    if "global_negation_allowed" in payload and "global_negation_allowed" not in metadata:
        metadata["global_negation_allowed"] = payload["global_negation_allowed"]
    if observation is not None and observation.tool in {"explore", "verify_window"}:
        metadata.setdefault("source_tool", observation.tool)
        metadata.setdefault("source_observation_id", observation.observation_id)
    if observation is not None and observation.tool == "explore":
        raw_output = observation.raw_output if isinstance(observation.raw_output, Mapping) else {}
        metadata.setdefault("mode", str(raw_output.get("mode") or ""))
        metadata.setdefault("support_status", str(raw_output.get("support_status") or ""))
        metadata.setdefault("cannot_final_cite", bool(raw_output.get("cannot_final_cite", False)))
        metadata.setdefault("requires_visual_verify", bool(raw_output.get("needs_visual_verify", False)))
        metadata.setdefault("task_type", str(raw_output.get("task_type") or ""))
        condition_match = raw_output.get("condition_match")
        if isinstance(condition_match, Mapping):
            metadata.setdefault("condition_match", dict(condition_match))
            metadata.setdefault("question_condition_match", bool(condition_match.get("matches_original_question")))
            metadata.setdefault("condition_match_level", str(condition_match.get("match_level") or ""))
        query_analysis = raw_output.get("query_analysis")
        if isinstance(query_analysis, Mapping):
            metadata.setdefault("query_analysis", dict(query_analysis))
        question_condition = raw_output.get("question_condition")
        if isinstance(question_condition, Mapping):
            metadata.setdefault("question_condition", dict(question_condition))
        answer_mapping = raw_output.get("answer_mapping")
        if isinstance(answer_mapping, Mapping):
            metadata.setdefault("answer_mapping", dict(answer_mapping))
    if observation is not None and observation.tool == "verify_window":
        result = _matching_verification_result(observation, payload)
        if result is not None:
            metadata.setdefault("target_id", str(result.get("target_id") or ""))
            metadata.setdefault("verdict", str(result.get("verdict") or ""))
            if isinstance(result.get("scope"), Mapping):
                metadata.setdefault("scope", dict(result.get("scope", {}) or {}))
            result_anchor_ids = [str(item).strip() for item in _sequence_items(result.get("anchor_ids")) if str(item).strip()]
            payload_anchor_ids = [str(item).strip() for item in _sequence_items(payload.get("anchor_ids")) if str(item).strip()]
            metadata.setdefault("anchor_ids", result_anchor_ids or payload_anchor_ids)
            metadata.setdefault("local_only", True)
    if str(payload.get("kind") or "support") == "local_negative":
        metadata.setdefault("global_negation_allowed", False)
    return metadata


def _matching_verification_result(observation: Observation, payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    raw_output = observation.raw_output if isinstance(observation.raw_output, Mapping) else {}
    results = _mapping_list(raw_output.get("verification_results"))
    if not results:
        return None
    metadata = dict(payload.get("metadata", {}) or {})
    target_id = str(payload.get("target_id") or metadata.get("target_id") or "").strip()
    if target_id:
        for result in results:
            if str(result.get("target_id") or "").strip() == target_id:
                return result
    anchor_ids = {str(item).strip() for item in _sequence_items(payload.get("anchor_ids")) if str(item).strip()}
    if anchor_ids:
        for result in results:
            result_anchor_ids = {str(item).strip() for item in _sequence_items(result.get("anchor_ids")) if str(item).strip()}
            if result_anchor_ids and anchor_ids.intersection(result_anchor_ids):
                return result
    if len(results) == 1:
        return results[0]
    return None


def _valid_local_negative_scope(scope: object) -> bool:
    if not isinstance(scope, Mapping):
        return False
    segment_id = str(scope.get("segment_id") or "").strip()
    time_range = scope.get("time_range")
    if not segment_id or not isinstance(time_range, Sequence) or isinstance(time_range, (str, bytes)):
        return False
    if len(time_range) != 2:
        return False
    try:
        start_sec = float(time_range[0])  # type: ignore[index]
        end_sec = float(time_range[1])  # type: ignore[index]
    except (TypeError, ValueError):
        return False
    return end_sec > start_sec


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y"}


def _row_matches_expected(row: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    for key, value in expected.items():
        if row.get(str(key)) != value:
            return False
    return True


def _bounded_inline(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _single_line_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _root_index_lines(root_segments: Sequence[Any]) -> list[str]:
    lines: list[str] = [
        "",
        "## Segment Cards",
        "Video map is navigation-only. Use explore to retrieve candidate windows ranked by token overlap, then verify_window to inspect each. You may also call verify_window with {segment_id, time_range} to sweep unexplored regions of this segment.",
    ]
    if not root_segments:
        return [*lines, "(none)"]
    for segment in root_segments:
        lines.extend(_root_segment_index_lines(segment))
    return lines


def _root_segment_index_lines(segment: Any) -> list[str]:
    summary = _single_line_text(getattr(segment, "low_fps_caption", "") or _compact_segment_text(segment))
    entities = _segment_card_entities(segment)
    modalities = _segment_card_modalities(segment)
    index_status = "available" if list(getattr(segment, "timeline_beats", ()) or ()) else "summary_only"
    lines = [
        (
            f"- {getattr(segment, 'segment_id', '-')} "
            f"[{float(getattr(segment, 'start_sec', 0.0)):.1f}-{float(getattr(segment, 'end_sec', 0.0)):.1f}s] "
            f"navigation_only=true index_status={index_status}"
        )
    ]
    lines.append(f"  Summary: {summary or '(no segment summary)'}")
    lines.append(f"  Entities: {', '.join(entities) if entities else '-'}")
    lines.append(f"  Modalities: {', '.join(modalities) if modalities else '-'}")
    lines.append(
        "  Coverage: explore proposes up to 3 candidate windows per segment. If those are exhausted without visual_support, call verify_window with an explicit time_range to scan other parts of the segment."
    )
    return lines


def _root_beat_line(beat: Any) -> str:
    hints = []
    entity_hints = [str(item) for item in getattr(beat, "entity_hints", ()) or () if str(item)]
    modality_hints = [str(item) for item in getattr(beat, "modality_hints", ()) or () if str(item)]
    if entity_hints:
        hints.append("entities=" + ", ".join(entity_hints))
    if modality_hints:
        hints.append("modalities=" + ", ".join(modality_hints))
    suffix = f" ({'; '.join(hints)})" if hints else ""
    return (
        f"  - {getattr(beat, 'beat_id', '-') or '-'} "
        f"[{float(getattr(beat, 'start_sec', 0.0)):.1f}-{float(getattr(beat, 'end_sec', 0.0)):.1f}s] "
        f"{_single_line_text(getattr(beat, 'summary', ''))}{suffix}"
    )


def _compact_segment_text(segment: Any) -> str:
    compact_text = getattr(segment, "compact_text", None)
    if callable(compact_text):
        return str(compact_text())
    return ""


def _segment_card_entities(segment: Any, *, max_entities: int = 8) -> list[str]:
    values: list[Any] = [*list(getattr(segment, "entities", ()) or ())]
    for beat in getattr(segment, "timeline_beats", ()) or ():
        values.extend(list(getattr(beat, "entity_hints", ()) or ()))
    return _unique_nonempty_texts(values)[:max_entities]


def _segment_card_modalities(segment: Any) -> list[str]:
    values: list[Any] = []
    if getattr(segment, "keyframe_paths", None) or getattr(segment, "low_fps_caption", ""):
        values.append("visual")
    if getattr(segment, "asr_text", "") or getattr(segment, "asr_sentences", None):
        values.append("asr")
    if getattr(segment, "ocr_text", "") or getattr(segment, "ocr_frames", None):
        values.append("ocr")
    for beat in getattr(segment, "timeline_beats", ()) or ():
        values.extend(list(getattr(beat, "modality_hints", ()) or ()))
    return _unique_nonempty_texts(values)


def _unique_nonempty_texts(values: Sequence[Any]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _index_coverage_lines(video_map: Any) -> list[str]:
    segments = list(getattr(video_map, "segments", ()) or ())
    roots = [segment for segment in segments if getattr(segment, "index_level", "root") == "root"]
    refined = [segment for segment in segments if getattr(segment, "index_level", "root") == "refined"]
    lines = []
    if roots:
        lines.append(f"root indexed: {_segment_range_summary(roots)} ({len(roots)} roots)")
    else:
        lines.append("root indexed: none")
    if refined:
        refined_ranges = ", ".join(f"{segment.start_sec:.1f}-{segment.end_sec:.1f}s" for segment in refined[:6])
        if len(refined) > 6:
            refined_ranges += f", ... {len(refined) - 6} more"
        lines.append(f"refined: {refined_ranges}")
    else:
        lines.append("refined: none")
    lines.append(f"index cache: root={len(roots)} / refinement={len(refined)}")
    return lines


def _latest_index_patch_lines(patch: Any) -> list[str]:
    children = list(getattr(patch, "children", ()) or ())
    lines = [
        (
            f"{getattr(patch, 'parent_segment_id', '-')} refined "
            f"[{float(getattr(patch, 'requested_start_sec', 0.0)):.1f}-{float(getattr(patch, 'requested_end_sec', 0.0)):.1f}s] "
            f"{getattr(patch, 'resolution', '-')}:"
        )
    ]
    if not children:
        lines.append("- (no children)")
        return lines
    for child in children[:5]:
        lines.append(
            f"- {child.segment_id} [{child.start_sec:.1f}-{child.end_sec:.1f}s] "
            f"{_bounded_inline(getattr(child, 'low_fps_caption', '') or getattr(child, 'compact_text', lambda: '')(), 180)}"
        )
    if len(children) > 5:
        lines.append(f"... more refined children hidden: {len(children) - 5}")
    return lines


def _answer_support_memory_entries(entries: Sequence[MemoryEntry]) -> list[MemoryEntry]:
    return [
        entry
        for entry in entries
        if entry.kind in {"answer_support", "caption_support", "visual_support", "synthesized_support", "answer_conflict_resolved"}
    ]


def _evidence_coverage_lines(workspace: "EvidenceWorkspace") -> list[str]:
    candidate_count = sum(
        len(_mapping_row_items((observation.raw_output if isinstance(observation.raw_output, Mapping) else {}).get("candidate_windows")))
        for observation in workspace.read_observations()
    )
    verification_results = _verification_result_rows(workspace)
    counts = {"supported": 0, "contradicted": 0, "local_negative": 0, "uncertain": 0}
    for result in verification_results:
        verdict = str(result.get("verdict") or "uncertain")
        if verdict == "not_found_in_window":
            counts["local_negative"] += 1
        elif verdict in counts:
            counts[verdict] += 1
        else:
            counts["uncertain"] += 1
    answer_support_count = sum(
        1
        for entry in workspace.memory_entries()
        if entry.kind in {"answer_support", "caption_support", "visual_support", "synthesized_support", "answer_conflict_resolved"}
    )
    return [
        f"candidate_windows: {candidate_count}",
        f"verified_supported: {counts['supported']}",
        f"contradicted: {counts['contradicted']}",
        f"local_negative: {counts['local_negative']}",
        f"uncertain: {counts['uncertain']}",
        f"answer_support_memories: {answer_support_count}",
    ]


def _segment_time_coverage_lines(workspace: "EvidenceWorkspace", root_segments: Sequence[Any]) -> list[str]:
    lines: list[str] = []
    for segment in root_segments:
        segment_id = str(getattr(segment, "segment_id", "") or "").strip()
        if not segment_id:
            continue
        start_sec = float(getattr(segment, "start_sec", 0.0) or 0.0)
        end_sec = float(getattr(segment, "end_sec", start_sec) or start_sec)
        verified = _verified_intervals_for_segment(workspace, segment_id=segment_id, start_sec=start_sec, end_sec=end_sec)
        verified_union = _merge_intervals(verified)
        total_sec = max(0.001, end_sec - start_sec)
        covered_sec = sum(max(0.0, b - a) for a, b in verified_union)
        pct = covered_sec / total_sec * 100.0
        uncovered = _complement_intervals(verified_union, start_sec=start_sec, end_sec=end_sec)
        covered_text = ", ".join(f"[{a:.1f}-{b:.1f}]" for a, b in verified_union) or "(none)"
        uncovered_text = ", ".join(f"[{a:.1f}-{b:.1f}] ({b - a:.1f}s)" for a, b in uncovered) or "(none)"
        lines.append(f"- {segment_id} [{start_sec:.1f}-{end_sec:.1f}]: verified {pct:.1f}%")
        lines.append(f"    covered: {covered_text}")
        lines.append(f"    uncovered: {uncovered_text}")
    return lines


def _verified_intervals_for_segment(
    workspace: "EvidenceWorkspace",
    *,
    segment_id: str,
    start_sec: float,
    end_sec: float,
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for observation in workspace.read_observations():
        if observation.tool != "verify_window":
            continue
        raw_output = observation.raw_output if isinstance(observation.raw_output, Mapping) else {}
        raw_segment_id = str(raw_output.get("segment_id") or "").strip()
        raw_time_range = raw_output.get("time_range")
        if not raw_segment_id:
            for region in observation.regions:
                if not isinstance(region, Mapping):
                    continue
                if str(region.get("segment_id") or "").strip() == segment_id:
                    raw_segment_id = segment_id
                    raw_time_range = region.get("time_range") or [region.get("start_sec"), region.get("end_sec")]
                    break
        if raw_segment_id != segment_id:
            continue
        time_range = _normalize_optional_time_range(raw_time_range)
        if time_range is None:
            continue
        a = max(start_sec, float(time_range[0]))
        b = min(end_sec, float(time_range[1]))
        if b > a:
            intervals.append((a, b))
    return intervals


def _merge_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start_sec, end_sec in sorted((float(a), float(b)) for a, b in intervals if b > a):
        if merged and start_sec <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_sec))
        else:
            merged.append((start_sec, end_sec))
    return merged


def _complement_intervals(
    intervals: Sequence[tuple[float, float]],
    *,
    start_sec: float,
    end_sec: float,
) -> list[tuple[float, float]]:
    gaps: list[tuple[float, float]] = []
    cursor = float(start_sec)
    for gap_start, gap_end in sorted(intervals):
        if gap_start > cursor:
            gaps.append((cursor, gap_start))
        cursor = max(cursor, gap_end)
    if cursor < end_sec:
        gaps.append((cursor, float(end_sec)))
    return gaps


def _candidate_window_rows(workspace: "EvidenceWorkspace") -> list[dict[str, Any]]:
    verified_ids: set[str] = set()
    verified_keys: set[str] = set()
    rows: list[dict[str, Any]] = []
    for observation in workspace.read_observations():
        raw_output = observation.raw_output if isinstance(observation.raw_output, Mapping) else {}
        candidate_id = str(raw_output.get("candidate_id") or "").strip()
        candidate_key = str(raw_output.get("candidate_key") or "").strip()
        if candidate_id and str(raw_output.get("worker") or "") == "EvidenceVerifier":
            verified_ids.add(candidate_id)
        if candidate_key and str(raw_output.get("worker") or "") == "EvidenceVerifier":
            verified_keys.add(candidate_key)
        for candidate in _mapping_row_items(raw_output.get("candidate_windows")):
            row = dict(candidate)
            if not row.get("time_range"):
                row["time_range"] = [row.get("start_sec"), row.get("end_sec")]
            row["status"] = (
                "verified"
                if str(row.get("candidate_key") or "") in verified_keys or str(row.get("candidate_id") or "") in verified_ids
                else row.get("status") or "pending"
            )
            rows.append(row)
    return rows


def _render_option_evidence_map(*, question: str, memory_entries: Sequence[MemoryEntry]) -> str:
    options = _options_from_question(question)
    if not options:
        return ""
    positive_kinds = EvidenceWorkspace.POSITIVE_SUPPORT_KINDS
    by_option: dict[str, list[MemoryEntry]] = {option: [] for option in options}
    for entry in memory_entries:
        option = str(entry.supports_option or "").strip().upper()[:1]
        if option in by_option and entry.kind in positive_kinds:
            by_option[option].append(entry)

    lines = [
        "# Option Evidence Map",
        "Choose the option whose Supporting facts include at least one visual_support, answer_support, caption_support, synthesized_support, or answer_conflict_resolved memory. local_negative does NOT support absence at global scope.",
    ]
    if _question_is_comparison_type(question):
        lines.append(
            "Note: this is a comparison question. Multiple option_<letter>_check memories do not prove equality; prefer synthesized or direct comparison evidence such as greater than, most, earliest, or largest."
        )
    for option in sorted(options):
        lines.append(f"{option}. {options[option]}")
        entries = by_option[option]
        if not entries:
            lines.append("   - Supporting facts: (none)")
            continue
        for entry in entries[:6]:
            lines.append(f"   - {entry.entry_id} [{entry.kind}]: {_compact_line(entry.claim, limit=180)}")
        if len(entries) > 6:
            lines.append(f"   - ... shown 6/{len(entries)}")
    return "\n".join(lines)


def _question_is_comparison_type(question: str) -> bool:
    text = str(question or "").lower()
    return any(
        marker in text
        for marker in (
            "largest",
            "smallest",
            "most",
            "least",
            "fewest",
            "earliest",
            "latest",
            "greater",
            "more than",
            "less than",
            "same number",
        )
    )


def _options_from_question(question: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for raw_option in extract_candidate_options(str(question or "")):
        match = re.match(r"^\s*([A-H])[\).]\s*(.+?)\s*$", str(raw_option), flags=re.IGNORECASE)
        if match:
            options[match.group(1).upper()] = " ".join(match.group(2).split())
    return options


def _compact_line(text: str, *, limit: int) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def _verification_result_rows(workspace: "EvidenceWorkspace") -> list[dict[str, Any]]:
    committed_by_target: dict[tuple[str, str], str] = {}
    for entry in workspace.memory_entries():
        metadata = dict(entry.metadata or {})
        target_id = str(metadata.get("target_id") or "").strip()
        verdict = str(metadata.get("verdict") or "").strip()
        if target_id and verdict:
            committed_by_target[(target_id, verdict)] = entry.entry_id
    rows: list[dict[str, Any]] = []
    for observation in workspace.read_observations():
        raw_output = observation.raw_output if isinstance(observation.raw_output, Mapping) else {}
        for result in _mapping_row_items(raw_output.get("verification_results")):
            row = dict(result)
            target_id = str(row.get("target_id") or "").strip()
            verdict = str(row.get("verdict") or "").strip()
            row["source_observation_id"] = observation.observation_id
            row["committed_memory_id"] = committed_by_target.get((target_id, verdict), "")
            rows.append(row)
    return rows


def _mapping_row_items(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _segment_range_summary(segments: Sequence[Any]) -> str:
    starts = [float(segment.start_sec) for segment in segments]
    ends = [float(segment.end_sec) for segment in segments]
    return f"{min(starts):.1f}-{max(ends):.1f}s"


def _format_time_range(value: Any) -> str:
    time_range = _normalize_optional_time_range(value)
    if time_range is None:
        return ""
    return f"[{time_range[0]:.1f}-{time_range[1]:.1f}]"


def _observation_text_for_excerpt_validation(observation: Observation) -> str:
    parts = [
        observation.claim,
        observation.limitations,
        json.dumps(observation.raw_output, ensure_ascii=False, sort_keys=True),
        json.dumps(list(observation.regions), ensure_ascii=False, sort_keys=True),
    ]
    return "\n".join(str(part) for part in parts if str(part).strip())


def _dedupe_mapping_rows(rows: Sequence[Mapping[str, Any]], *, keys: Sequence[str]) -> list[dict[str, Any]]:
    seen = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        key = _mapping_key(row, keys=keys)
        if key in seen:
            for index, existing in enumerate(result):
                if _mapping_key(existing, keys=keys) != key:
                    continue
                if _prefer_mapping_row(row, existing):
                    result[index] = dict(row)
                break
            continue
        seen.add(key)
        result.append(dict(row))
    return result


def _prefer_mapping_row(candidate: Mapping[str, Any], existing: Mapping[str, Any]) -> bool:
    candidate_rank = _support_rank(candidate.get("status") or candidate.get("support_status"))
    existing_rank = _support_rank(existing.get("status") or existing.get("support_status"))
    if candidate_rank != existing_rank:
        return candidate_rank > existing_rank
    candidate_relation_count = len(_mapping_list(candidate.get("relation_bindings")))
    existing_relation_count = len(_mapping_list(existing.get("relation_bindings")))
    if candidate_relation_count != existing_relation_count:
        return candidate_relation_count > existing_relation_count
    candidate_ordered = len(candidate.get("ordered_target_refs") or candidate.get("ordered_targets") or [])
    existing_ordered = len(existing.get("ordered_target_refs") or existing.get("ordered_targets") or [])
    return candidate_ordered > existing_ordered


def _support_rank(value: Any) -> int:
    status = str(value or "").strip().lower()
    return {
        "supported": 4,
        "partial": 3,
        "ambiguous": 2,
        "rejected": 1,
        "unsupported": 1,
        "conflicting": 0,
    }.get(status, 0)


def _mapping_key(row: Mapping[str, Any], *, keys: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(row.get(key) or "").strip() for key in keys)


def _dedupe_text_sources(sources: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    result: list[dict[str, Any]] = []
    for source in sources:
        key = (
            str(source.get("source") or ""),
            str(source.get("text") or ""),
            str(source.get("segment_id") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(source))
    return result


def _iter_segment_window_dicts(value: Any):
    if isinstance(value, Mapping):
        if "segment_id" in value and "start_sec" in value and "end_sec" in value:
            yield value
        for child in value.values():
            yield from _iter_segment_window_dicts(child)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            yield from _iter_segment_window_dicts(item)


def _trajectory_actions(
    *,
    trace_events: Sequence[Mapping[str, Any]],
    observations_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    pending: dict[tuple[str, str], dict[str, Any]] = {}
    for event in trace_events:
        event_type = str(event.get("type", ""))
        payload = event.get("payload", {})
        if not isinstance(payload, Mapping):
            payload = {}
        if event_type == "tool_use":
            step = str(payload.get("step", len(actions) + 1))
            tool_name = str(payload.get("tool", ""))
            arguments = payload.get("arguments", {})
            action = {
                "index": len(actions) + 1,
                "type": "tool",
                "step": int(payload.get("step", len(actions) + 1) or len(actions) + 1),
                "tool": tool_name,
                "arguments": dict(arguments) if isinstance(arguments, Mapping) else {},
                "created_at": str(event.get("created_at", "")),
            }
            actions.append(action)
            pending[(step, tool_name)] = action
            continue
        if event_type != "tool_result":
            continue
        step = str(payload.get("step", ""))
        tool_name = str(payload.get("tool", ""))
        action = pending.get((step, tool_name)) or _latest_unmatched_action(actions, tool_name=tool_name)
        if action is None:
            continue
        observation_id = str(payload.get("observation_id", ""))
        action["observation_id"] = observation_id
        observation = observations_by_id.get(observation_id)
        if observation is not None:
            action["observation"] = _trajectory_observation(observation)
    return actions


def _latest_unmatched_action(actions: Sequence[dict[str, Any]], *, tool_name: str) -> dict[str, Any] | None:
    for action in reversed(actions):
        if action.get("observation_id"):
            continue
        if tool_name and action.get("tool") != tool_name:
            continue
        return action
    return None


def _trajectory_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "observation_id": str(observation.get("observation_id", "")),
        "tool": str(observation.get("tool", "")),
        "claim": str(observation.get("claim", "")),
        "confidence": float(observation.get("confidence", 0.0) or 0.0),
        "regions": list(observation.get("regions", []) or []),
        "limitations": str(observation.get("limitations", "")),
        "confidence_signal": str(observation.get("confidence_signal", "")),
        "raw_output": dict(observation.get("raw_output", {}) or {}),
    }


def _complete_evidence_chain(stages: Sequence[str]) -> bool:
    return "distilled" in stages and "ledger" in stages and "mapped" in stages


def _compact_evidence_record(record: EvidenceRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "evidence_id": record.evidence_id,
        "stage": record.stage,
        "parent_id": record.parent_id,
        "tool": record.tool,
        "observation_id": record.observation_id,
        "frame_set_id": record.frame_set_id,
        "grounding_quality": record.grounding_quality,
        "confidence": record.confidence,
    }
    claim = record.content.get("claim")
    if claim:
        payload["claim"] = _compact_text(str(claim), limit=240)
    relation = record.content.get("candidate_option_relation")
    if isinstance(relation, Mapping):
        payload["candidate_option_relation"] = _compact_relation(relation)
    return payload


def _compact_relation(relation: Mapping[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in ["option", "relation", "strength", "observation_id", "parent_evidence_id", "assigned_by"]:
        if relation.get(key) is not None:
            compact[key] = relation[key]
    if relation.get("rationale"):
        compact["rationale"] = _compact_text(str(relation["rationale"]), limit=160)
    return compact


def _compact_text(text: str, *, limit: int) -> str:
    compact = " ".join(str(text).split())
    return compact[:limit] + ("..." if len(compact) > limit else "")


def _ledger_markdown_field(value: Any) -> str:
    return " ".join(str(value or "-").split())


def _ledger_claim(observation: Observation, *, parent_record: EvidenceRecord | None) -> str:
    if parent_record is None:
        return observation.claim
    claim = parent_record.content.get("claim")
    return str(claim if claim is not None else observation.claim)


def _ledger_regions(observation: Observation, *, parent_record: EvidenceRecord | None) -> list[Any]:
    if parent_record is None:
        return list(observation.regions)
    regions = parent_record.content.get("regions")
    if isinstance(regions, Sequence) and not isinstance(regions, (str, bytes)):
        return list(regions)
    return list(observation.regions)


def _ledger_grounding_quality(observation: Observation, *, parent_record: EvidenceRecord | None) -> str:
    if parent_record is not None:
        return str(parent_record.grounding_quality)
    return _grounding_quality(
        raw_output=observation.raw_output,
        limitations=observation.limitations,
        confidence_signal=observation.confidence_signal,
    )


def _parse_ledger_entries(ledger_text: str) -> list[Mapping[str, Any]]:
    entries = []
    for line in ledger_text.splitlines():
        obs_match = re.search(r"`(obs_[0-9]{4})`", line)
        if not obs_match:
            continue
        tool_match = re.search(r"tool:\s*`?([A-Za-z0-9_]+)`?", line)
        grounding_match = re.search(r"gq:\s*`?([^`|]+)`?", line)
        confidence_match = re.search(r"confidence:\s*([0-9.]+)", line)
        artifacts_match = re.search(r"artifacts:\s*(.*?)\s*\|\s*claim:", line)
        claim_match = re.search(r"claim:\s*(.*?)\s*\|\s*limitations:", line)
        limitation_match = re.search(r"limitations:\s*(.*)$", line)
        entries.append(
            {
                "observation_id": obs_match.group(1),
                "tool": tool_match.group(1) if tool_match else "unknown",
                "grounding_quality": grounding_match.group(1).strip() if grounding_match else "",
                "confidence": confidence_match.group(1) if confidence_match else "",
                "artifacts": artifacts_match.group(1).strip() if artifacts_match else "-",
                "claim": claim_match.group(1).strip() if claim_match else "",
                "limitations": limitation_match.group(1).strip() if limitation_match else "-",
            }
        )
    return entries


def _option_letter_map(options: Sequence[str]) -> dict[str, str]:
    mapping = {}
    for index, option in enumerate(options):
        text = str(option).strip()
        match = re.match(r"^([A-Za-z])(?:[\.)]\s*|\s+|$)", text)
        letter = match.group(1).upper() if match else chr(ord("A") + index)
        mapping[letter] = text
    return mapping


def _normalize_supported_option(value: Any, *, option_map: Mapping[str, str]) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    letter_match = re.search(r"\b(?:option\s*)?([A-Za-z])\b", text, flags=re.IGNORECASE)
    if letter_match:
        letter = letter_match.group(1).upper()
        if not option_map or letter in option_map:
            return letter
    for letter, option_text in option_map.items():
        if text == option_text or text.lower() in option_text.lower() or option_text.lower() in text.lower():
            return letter
    return None


def _supported_option_from_claim(claim: str) -> str | None:
    match = re.search(
        r"\b(?:supported\s+option|supports?|matches|chooses?|answer(?:s)?|option)\s*:?\s+(?:option\s*)?([A-Za-z])\b",
        claim,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()
    return None


def _bare_option_from_claim(
    claim: str,
    *,
    option_map: Mapping[str, str] | None = None,
) -> str | None:
    match = re.match(r"^\s*([A-Za-z])[\.)]\s+\S", claim, flags=re.DOTALL)
    if not match:
        return None
    letter = match.group(1).upper()
    if option_map is not None:
        return letter if not option_map or letter in option_map else None
    return letter if letter in {"A", "B", "C", "D"} else None


def _claim_without_legacy_worker_vote(claim: str) -> str:
    text = str(claim).strip()
    bare_match = re.match(r"^\s*[A-Za-z][\.)]\s+(.*)$", text, flags=re.DOTALL)
    if bare_match:
        return bare_match.group(1).strip()

    fact_lines = []
    vote_line = re.compile(
        r"^\s*(?:supported\s+option|supported_option|supported\s+option\s+letter|"
        r"supported_option_letter|answer_option|answer|option)\s*:?\s*(?:option\s*)?[A-Za-z]\b",
        flags=re.IGNORECASE,
    )
    metadata_line = re.compile(r"^\s*confidence\s*:", flags=re.IGNORECASE)
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or vote_line.search(line) or metadata_line.search(line):
            continue
        line = re.sub(r"^\s*claim\s*:\s*", "", line, flags=re.IGNORECASE).strip()
        if line:
            fact_lines.append(line)
    return " ".join(fact_lines).strip() or text


def _supported_option_from_relations(value: Any, *, option_map: Mapping[str, str]) -> str | None:
    relations = _candidate_option_relations(value)
    support_relations = [
        relation
        for relation in relations
        if str(relation.get("relation", "")).strip().lower() in {"support", "supports", "supported"}
    ]
    if not support_relations:
        return None
    support_relations.sort(key=lambda relation: (-float(relation.get("strength", 0.0) or 0.0), str(relation.get("option", ""))))
    return _normalize_supported_option(support_relations[0].get("option"), option_map=option_map)


def _candidate_option_relations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _tool_emits_candidate_hints_only(tool_name: Any) -> bool:
    return str(tool_name) == "global_gist"


def _project_target_row_to_option(row: Mapping[str, Any], *, registry_obj: Any) -> dict[str, Any]:
    if registry_obj is None:
        return dict(row)
    if row.get("supported_option") or _candidate_option_relations(row.get("candidate_option_relations")):
        return dict(row)
    ordered_target_refs = _ordered_target_refs_from_evidence_row(row)
    if ordered_target_refs:
        option_id = _option_for_ordered_target_refs(ordered_target_refs, registry_obj=registry_obj)
        if option_id:
            return _project_option_relation(
                row,
                option_id=option_id,
                target_ref="ordered_sequence",
                assigned_by="target_registry_sequence_projection",
            )
    target_ref = _target_ref_from_evidence_row(row)
    if not target_ref:
        return dict(row)
    try:
        options = tuple(registry_obj.options_for_target(target_ref))
    except (AttributeError, KeyError):
        return dict(row)
    if len(options) != 1:
        return dict(row)
    option_id = str(getattr(options[0], "option_id", "")).strip().upper()
    if not option_id:
        return dict(row)
    return _project_option_relation(
        row,
        option_id=option_id,
        target_ref=target_ref,
        assigned_by="target_registry_projection",
    )


def _project_option_relation(
    row: Mapping[str, Any],
    *,
    option_id: str,
    target_ref: str,
    assigned_by: str,
) -> dict[str, Any]:
    confidence = float(row.get("confidence", 0.0) or 0.0)
    relation = {
        "option": option_id,
        "relation": "support",
        "strength": confidence,
        "observation_id": str(row.get("obs_id") or row.get("observation_id") or ""),
        "evidence_id": str(row.get("evidence_id", "")),
        "target_ref": target_ref,
        "grounding_quality": str(row.get("grounding_quality", "")),
        "answer_grade": True,
        "assigned_by": assigned_by,
    }
    projected = dict(row)
    projected["candidate_option_relations"] = _merge_candidate_option_relations(
        _candidate_option_relations(projected.get("candidate_option_relations")),
        [relation],
    )
    projected["supported_option"] = option_id
    return projected


def _ordered_target_refs_from_evidence_row(row: Mapping[str, Any]) -> tuple[str, ...]:
    candidates = [
        row.get("ordered_target_refs"),
        row.get("ordered_targets"),
    ]
    binding = row.get("evidence_binding")
    if isinstance(binding, Mapping):
        candidates.extend([binding.get("ordered_target_refs"), binding.get("ordered_targets")])
    sequence = row.get("ordered_transcript_sequence")
    if isinstance(sequence, Mapping):
        candidates.extend([sequence.get("ordered_target_refs"), sequence.get("ordered_targets")])
    for candidate in candidates:
        if not isinstance(candidate, Sequence) or isinstance(candidate, (str, bytes)):
            continue
        refs = tuple(str(ref).strip() for ref in candidate if str(ref).strip())
        if refs and all(_TARGET_REF_RE.fullmatch(ref) for ref in refs):
            return refs
    return ()


def _option_for_ordered_target_refs(ordered_target_refs: Sequence[str], *, registry_obj: Any) -> str:
    try:
        options_by_id = getattr(registry_obj, "options_by_id", {})
        option_values = options_by_id.values() if isinstance(options_by_id, Mapping) else tuple(options_by_id)
    except (AttributeError, TypeError):
        return ""
    matches = [
        str(getattr(option, "option_id", "")).strip().upper()
        for option in option_values
        if tuple(str(ref) for ref in getattr(option, "target_sequence", ())) == tuple(ordered_target_refs)
    ]
    unique = sorted({match for match in matches if match})
    return unique[0] if len(unique) == 1 else ""


def _target_ref_from_evidence_row(row: Mapping[str, Any]) -> str:
    candidates = [
        row.get("target_id"),
        row.get("target_ref"),
        row.get("event_label"),
        row.get("entity"),
    ]
    binding = row.get("evidence_binding")
    if isinstance(binding, Mapping):
        candidates.extend([binding.get("target_id"), binding.get("target_ref")])
    for candidate in candidates:
        text = str(candidate or "").strip()
        if _TARGET_REF_RE.fullmatch(text):
            return text
    return ""


def _candidate_option_relations_from_supported_option(observation: Observation) -> list[dict[str, Any]]:
    raw_output = observation.raw_output if isinstance(observation.raw_output, Mapping) else {}
    option = (
        raw_output.get("supported_option")
        or raw_output.get("supported_option_letter")
        or raw_output.get("answer_option")
        or _first_item(raw_output.get("supported_options"))
    )
    if not option:
        return []
    return [
        {
            "option": str(option),
            "relation": "support",
            "strength": float(observation.confidence),
            "observation_id": observation.observation_id,
            "rationale": "Tool output reported this supported option.",
            "assigned_by": observation.tool,
        }
    ]


def _observation_relation_count(observation: Mapping[str, Any]) -> int:
    raw_output = observation.get("raw_output", {})
    if not isinstance(raw_output, Mapping):
        return 0
    return len(_candidate_option_relations(raw_output.get("candidate_option_relations")))


def _ledger_records_by_observation(records: Sequence[EvidenceRecord]) -> dict[str, list[EvidenceRecord]]:
    by_observation: dict[str, list[EvidenceRecord]] = {}
    for record in records:
        if record.stage != "ledger" or not record.observation_id:
            continue
        by_observation.setdefault(record.observation_id, []).append(record)
    return by_observation


def _mapped_relation_parent(
    relation: Mapping[str, Any],
    *,
    observation_id: str,
    evidence_by_id: Mapping[str, EvidenceRecord],
    ledger_by_observation: Mapping[str, Sequence[EvidenceRecord]],
) -> EvidenceRecord | None:
    parent_id = _relation_parent_evidence_id(relation)
    if parent_id:
        parent_record = evidence_by_id.get(parent_id)
        if parent_record is None or parent_record.stage not in {"ledger", "distilled"}:
            return None
        if parent_record.observation_id and parent_record.observation_id != observation_id:
            return None
        return parent_record
    ledger_records = ledger_by_observation.get(observation_id, [])
    return ledger_records[-1] if ledger_records else None


def _relation_parent_evidence_id(relation: Mapping[str, Any]) -> str:
    return str(
        relation.get("parent_evidence_id")
        or relation.get("ledger_evidence_id")
        or relation.get("parent_id")
        or ""
    ).strip()


def _relation_option_letter(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"\b(?:option\s*)?([A-Za-z])\b", text, flags=re.IGNORECASE)
    return match.group(1).upper() if match else text[:1].upper()


def _candidate_option_relation_present(
    existing: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
) -> bool:
    candidate_key = _candidate_option_relation_exact_key(candidate)
    return any(_candidate_option_relation_exact_key(relation) == candidate_key for relation in existing)


def _candidate_option_relation_exact_key(relation: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(relation.get("option", "")).strip().upper(),
        str(relation.get("relation", "")).strip().lower(),
        str(relation.get("observation_id", "")).strip(),
        str(relation.get("assigned_by", "")).strip().lower(),
        _relation_parent_evidence_id(relation),
    )


def _mapped_evidence_record(
    *,
    workspace: EvidenceWorkspace,
    observation: Mapping[str, Any],
    relation: Mapping[str, Any],
    parent_record: EvidenceRecord,
) -> EvidenceRecord:
    confidence = _relation_confidence(relation, default=parent_record.confidence)
    return EvidenceRecord(
        evidence_id=workspace.next_evidence_id("mapped"),
        stage="mapped",
        parent_id=parent_record.evidence_id,
        tool=str(observation.get("tool") or parent_record.tool),
        observation_id=str(observation.get("observation_id") or parent_record.observation_id or ""),
        frame_set_id=parent_record.frame_set_id,
        content={
            "candidate_option_relation": dict(relation),
            "parent_stage": parent_record.stage,
            "source_claim": str(observation.get("claim", "")),
        },
        grounding_quality=str(parent_record.grounding_quality),  # type: ignore[arg-type]
        confidence=confidence,
        created_at=time.time(),
    )


def _relation_confidence(relation: Mapping[str, Any], *, default: float) -> float:
    value = relation.get("strength", relation.get("confidence", default))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _relations_for_observation(
    relations: Sequence[Mapping[str, Any]],
    *,
    observation_id: str,
    default_observation_id: str,
    assigned_by: str,
) -> list[dict[str, Any]]:
    scoped: list[dict[str, Any]] = []
    for relation in relations:
        relation_observation_id = str(
            relation.get("observation_id") or relation.get("obs_id") or relation.get("citation") or ""
        ).strip()
        if relation_observation_id and relation_observation_id != observation_id:
            continue
        if not relation_observation_id and observation_id != default_observation_id:
            continue
        scoped_relation = dict(relation)
        scoped_relation["observation_id"] = observation_id
        scoped_relation["assigned_by"] = str(scoped_relation.get("assigned_by") or assigned_by)
        scoped.append(scoped_relation)
    return scoped


def _reject_answer_agent_relations_for_observation(
    *,
    raw_output: Mapping[str, Any],
    relations: Sequence[Mapping[str, Any]],
) -> bool:
    if not any(_relation_assigned_by_answer_agent(relation) for relation in relations):
        return False
    integrity = str(raw_output.get("observation_integrity") or "").strip().lower()
    if integrity in {"unverifiable", "internally_inconsistent"}:
        return True
    has_ordered_visible = bool(raw_output.get("ordered_visible_in_window") or raw_output.get("ordered_visible"))
    grounding_quality = str(raw_output.get("grounding_quality") or "").strip().lower()
    return has_ordered_visible and grounding_quality == "weak"


def _relation_assigned_by_answer_agent(relation: Mapping[str, Any]) -> bool:
    assigned_by = str(relation.get("assigned_by") or "").strip().lower()
    return assigned_by == "answer_agent" or assigned_by.startswith("answer_agent_")


def _merge_candidate_option_relations(
    existing: Sequence[Mapping[str, Any]],
    incoming: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen = set()
    for relation in list(existing) + list(incoming):
        if not isinstance(relation, Mapping):
            continue
        normalized = dict(relation)
        key = (
            str(normalized.get("option", "")).strip().upper(),
            str(normalized.get("relation", "")).strip().lower(),
            str(normalized.get("observation_id", "")).strip(),
            str(normalized.get("assigned_by", "")).strip().lower(),
        )
        if key in seen:
            for merged_relation in merged:
                merged_key = (
                    str(merged_relation.get("option", "")).strip().upper(),
                    str(merged_relation.get("relation", "")).strip().lower(),
                    str(merged_relation.get("observation_id", "")).strip(),
                    str(merged_relation.get("assigned_by", "")).strip().lower(),
                )
                if merged_key == key:
                    _merge_missing_relation_fields(merged_relation, normalized)
                    break
            continue
        seen.add(key)
        merged.append(normalized)
    return merged


def _merge_missing_relation_fields(target: dict[str, Any], incoming: Mapping[str, Any]) -> None:
    for key, value in incoming.items():
        if value in (None, "", []):
            continue
        if target.get(key) in (None, "", []):
            target[key] = value


def _has_worker_option_vote(
    *,
    tool_name: str,
    raw_output: Mapping[str, Any],
    claim: str,
    option_map: Mapping[str, str] | None = None,
) -> bool:
    if tool_name not in EvidenceWorkspace.LOCAL_WORKER_EVIDENCE_TOOLS:
        return False
    if _candidate_option_relations(raw_output.get("candidate_option_relations")):
        return False
    vote_fields = ["supported_option", "supported_option_letter", "answer_option", "supported_options"]
    if any(raw_output.get(field) for field in vote_fields):
        return True
    return _bare_option_from_claim(claim, option_map=option_map) is not None or _supported_option_from_claim(claim) is not None


def _first_item(value: Any) -> Any:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value[0] if value else None
    return value


def _normalize_evidence_row(row: Mapping[str, Any], *, evidence_id: str | None = None) -> dict[str, Any]:
    time_range = _evidence_row_time_range(row)
    t_start = time_range[0] if time_range is not None else _optional_float(row.get("t_start"))
    t_end = time_range[1] if time_range is not None else _optional_float(row.get("t_end"))
    payload = EvidenceRowV2(
        evidence_id=str(evidence_id or row.get("evidence_id", "")),
        obs_id=str(row.get("obs_id") or row.get("observation_id") or ""),
        tool=str(row.get("tool", "")),
        segment_id=str(row.get("segment_id", "")),
        t_start=t_start,
        t_end=t_end,
        entity=str(row.get("entity") or row.get("event_label") or ""),
        time_range=time_range,
        supported_option=(None if row.get("supported_option") in (None, "") else str(row.get("supported_option"))),
        event_label=str(row.get("event_label") or row.get("entity") or ""),
        claim=str(row.get("claim", "")),
        confidence=float(row.get("confidence", 0.0) or 0.0),
        grounding_quality=str(row.get("grounding_quality") or "weak"),
        candidate_option_relations=_candidate_option_relations(row.get("candidate_option_relations")),
        confidence_signal=str(row.get("confidence_signal", "")),
        mutex_group_id=str(row.get("mutex_group_id", "")),
        legacy_worker_vote=bool(row.get("legacy_worker_vote", False)),
        limitations=str(row.get("limitations", "")),
        artifact=str(row.get("artifact", "")),
        **_evidence_provenance_fields(row),
    ).to_dict()
    if payload["time_range"] is None and payload["t_start"] is not None and payload["t_end"] is not None:
        payload["time_range"] = [payload["t_start"], payload["t_end"]]
    for field in (
        "evidence_type",
        "evidence_grade",
        "support_status",
        "source_observation_id",
        "source_tool",
    ):
        value = row.get(field)
        if value not in (None, ""):
            payload[field] = str(value)
    if row.get("requires_visual_verification") is not None:
        payload["requires_visual_verification"] = bool(row.get("requires_visual_verification"))
    anchors = row.get("anchors_for_vlm")
    if isinstance(anchors, Sequence) and not isinstance(anchors, (str, bytes)):
        payload["anchors_for_vlm"] = [dict(anchor) for anchor in anchors if isinstance(anchor, Mapping)]
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        payload["metadata"] = dict(metadata)
    return payload


def _normalize_timeline_row(row: Mapping[str, Any]) -> dict[str, Any]:
    observed_at_sec = _optional_float(row.get("observed_at_sec"))
    window = row.get("window")
    normalized_window: list[float] | None = None
    if isinstance(window, Sequence) and not isinstance(window, (str, bytes)) and len(window) >= 2:
        normalized_window = [float(window[0]), float(window[1])]
    payload = {
        "obs_id": str(row.get("obs_id") or row.get("observation_id") or ""),
        "entity": str(row.get("entity") or row.get("event_label") or ""),
        "observed_at_sec": observed_at_sec,
        "window": normalized_window,
        "confidence_signal": str(row.get("confidence_signal", "")),
        "claim": str(row.get("claim", "")),
    }
    if not payload["confidence_signal"]:
        payload["confidence_signal"] = "confirmed" if observed_at_sec is not None else "window_only"
    for field in ("grounding_quality", "source", "candidate_id", "segment_id"):
        value = row.get(field)
        if value not in (None, ""):
            payload[field] = str(value)
    if row.get("requires_visual_verification") is not None:
        payload["requires_visual_verification"] = bool(row.get("requires_visual_verification"))
    return payload


def _compact_recent_tool_payload(value: Any, *, max_string_chars: int = 500, max_items: int = 8) -> Any:
    return _compact_recent_tool_payload_value(
        value,
        max_string_chars=max_string_chars,
        max_items=max_items,
        path="raw_output",
        seen_strings={},
        seen_items=set(),
    )


def _compact_recent_tool_payload_value(
    value: Any,
    *,
    max_string_chars: int,
    max_items: int,
    path: str,
    seen_strings: dict[str, str],
    seen_items: set[str],
) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _compact_recent_tool_payload_value(
                item,
                max_string_chars=max_string_chars,
                max_items=max_items,
                path=f"{path}.{key}",
                seen_strings=seen_strings,
                seen_items=seen_items,
            )
            for key, item in _planner_payload_items(value, max_items=max_items)
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        compacted = []
        for index, item in enumerate(list(value)[:max_items]):
            child = _compact_recent_tool_payload_value(
                item,
                max_string_chars=max_string_chars,
                max_items=max_items,
                path=f"{path}[{index}]",
                seen_strings=seen_strings,
                seen_items=seen_items,
            )
            item_key = _compact_payload_dedupe_key(child)
            if item_key in seen_items:
                continue
            seen_items.add(item_key)
            compacted.append(child)
        return compacted
    if isinstance(value, str):
        normalized = _compact_payload_text_key(value)
        if len(normalized) >= max(80, max_string_chars // 2):
            first_path = seen_strings.get(normalized)
            if first_path:
                return f"[duplicate of {first_path}]"
            seen_strings[normalized] = path
        if len(value) > max_string_chars:
            return value[:max_string_chars] + f"... [truncated {len(value) - max_string_chars} chars]"
    return value


def _compact_payload_text_key(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _planner_payload_items(value: Mapping[Any, Any], *, max_items: int) -> list[tuple[Any, Any]]:
    priority = {
        "visual_caption": 0,
        "caption": 1,
        "summary": 2,
        "text": 3,
        "claim": 4,
    }

    def sort_key(item: tuple[Any, Any]) -> tuple[int, str]:
        key, child = item
        key_text = str(key)
        if key_text in priority:
            return (priority[key_text], key_text)
        if isinstance(child, Sequence) and not isinstance(child, (str, bytes)):
            return (20, key_text)
        if isinstance(child, Mapping):
            return (15, key_text)
        return (10, key_text)

    return sorted(list(value.items())[:max_items], key=sort_key)


def _compact_payload_dedupe_key(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return repr(value)


def _normalize_hypothesis_slots(slots: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    normalized: dict[str, dict[str, str]] = {}
    for raw_name, raw_slot in slots.items():
        slot_name = str(raw_name).strip()
        if not slot_name:
            continue
        if isinstance(raw_slot, Mapping):
            normalized[slot_name] = _normalize_hypothesis_slot(raw_slot)
        else:
            normalized[slot_name] = _normalize_hypothesis_slot({"status": raw_slot})
    return normalized


def _normalize_hypothesis_slot(slot: Mapping[str, Any]) -> dict[str, str]:
    status = str(slot.get("status", "empty")).strip().lower()
    if status not in {"empty", "partial", "satisfied"}:
        status = "empty"
    return {
        "status": status,
        "evidence_obs_id": str(slot.get("evidence_obs_id", "")).strip(),
    }


def _timeline_sort_key(row: Mapping[str, Any]) -> tuple[float, str]:
    observed_at_sec = row.get("observed_at_sec")
    if observed_at_sec is not None:
        return (float(observed_at_sec), str(row.get("obs_id", "")))
    window = row.get("window")
    if isinstance(window, Sequence) and not isinstance(window, (str, bytes)) and window:
        return (float(window[0]), str(row.get("obs_id", "")))
    return (float("inf"), str(row.get("obs_id", "")))


def _observed_at_sec(raw_output: Mapping[str, Any]) -> float | None:
    for key in ["observed_at_sec", "timestamp_sec", "time_sec", "t_sec", "timestamp"]:
        value = raw_output.get(key)
        if value is None or value == "":
            continue
        return float(value)
    return None


def _timeline_confidence_signal(
    *,
    raw_output: Mapping[str, Any],
    observation_signal: str,
    observed_at_sec: float | None,
) -> str:
    signal = str(observation_signal or raw_output.get("confidence_signal", "")).strip().lower()
    if signal in {"unsupported", "degenerate"}:
        return "unsupported"
    grounding_quality = str(raw_output.get("grounding_quality", "")).strip().lower()
    if grounding_quality in {"weak", "inferred", "external_knowledge"}:
        return "unsupported"
    if grounding_quality == "visually_confirmed" and observed_at_sec is not None:
        return "visually_confirmed"
    return "confirmed" if observed_at_sec is not None else "window_only"


def _evidence_row_from_observation(
    *,
    observation: Observation,
    evidence_record: EvidenceRecord,
    claim: str,
    grounding_quality: str,
    confidence: float,
) -> dict[str, Any]:
    raw_output = observation.raw_output if isinstance(observation.raw_output, Mapping) else {}
    row = _evidence_row_from_observation_mapping(
        observation={
            "observation_id": observation.observation_id,
            "tool": observation.tool,
            "claim": observation.claim,
            "confidence": confidence,
            "regions": observation.regions,
            "limitations": observation.limitations,
            "confidence_signal": observation.confidence_signal,
            "raw_output": raw_output,
            "input_artifacts": observation.input_artifacts,
        },
        evidence_id=evidence_record.evidence_id,
    )
    row["claim"] = claim
    row["grounding_quality"] = grounding_quality
    row["frame_set_id"] = evidence_record.frame_set_id or ""
    return row


def _evidence_row_from_observation_mapping(
    *,
    observation: Mapping[str, Any],
    evidence_id: str,
) -> dict[str, Any]:
    raw_output = observation.get("raw_output", {})
    if not isinstance(raw_output, Mapping):
        raw_output = {}
    display_claim = str(observation.get("claim", ""))
    if _has_worker_option_vote(
        tool_name=str(observation.get("tool", "")),
        raw_output=raw_output,
        claim=display_claim,
    ):
        display_claim = _claim_without_legacy_worker_vote(display_claim)
    time_range = _observation_time_range(observation)
    event_label = _observation_event_label(raw_output=raw_output, claim=display_claim)
    row = {
        "evidence_id": str(evidence_id),
        "obs_id": str(observation.get("observation_id", "")),
        "tool": str(observation.get("tool", "")),
        "segment_id": _observation_segment_id(observation),
        "time_range": time_range,
        "t_start": time_range[0] if time_range is not None else None,
        "t_end": time_range[1] if time_range is not None else None,
        "entity": event_label,
        "event_label": event_label,
        "claim": display_claim,
        "confidence": float(observation.get("confidence", 0.0) or 0.0),
        "grounding_quality": _grounding_quality(
            raw_output=raw_output,
            limitations=str(observation.get("limitations", "")),
            confidence_signal=str(observation.get("confidence_signal", "")),
        ),
        "candidate_option_relations": _candidate_option_relations(raw_output.get("candidate_option_relations")),
        "confidence_signal": str(observation.get("confidence_signal", "") or raw_output.get("confidence_signal", "")),
        "mutex_group_id": str(raw_output.get("mutex_group_id", "")),
        "legacy_worker_vote": _has_worker_option_vote(
            tool_name=str(observation.get("tool", "")),
            raw_output=raw_output,
            claim=str(observation.get("claim", "")),
        ),
        "limitations": str(observation.get("limitations", "")),
        "artifact": _first_item(observation.get("input_artifacts")) or "",
        **_evidence_provenance_fields(raw_output),
    }
    option_source = (
        raw_output.get("supported_option")
        or raw_output.get("supported_option_letter")
        or raw_output.get("answer_option")
        or _first_item(raw_output.get("supported_options"))
    )
    if option_source:
        row["supported_option"] = str(option_source)
    return row


def _evidence_row_time_range(row: Mapping[str, Any]) -> list[float] | None:
    time_range = row.get("time_range")
    if isinstance(time_range, Sequence) and not isinstance(time_range, (str, bytes)) and len(time_range) >= 2:
        return [float(time_range[0]), float(time_range[1])]
    start = row.get("t_start")
    end = row.get("t_end")
    if start is None:
        start = row.get("start_sec")
    if end is None:
        end = row.get("end_sec")
    if start is not None and end is not None:
        return [float(start), float(end)]
    return None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _evidence_provenance_fields(source: Mapping[str, Any]) -> dict[str, Any]:
    citation_provenance = source.get("citation_provenance", {})
    if not isinstance(citation_provenance, Mapping):
        citation_provenance = {}
    evidence_binding = source.get("evidence_binding", {})
    if not isinstance(evidence_binding, Mapping):
        evidence_binding = {}
    evidence_binding = dict(evidence_binding)
    target_ref = source.get("target_ref") or source.get("target_id")
    if target_ref:
        evidence_binding.setdefault("target_ref", str(target_ref))
        evidence_binding.setdefault("target_id", str(target_ref))
    ordered_refs = source.get("ordered_target_refs") or source.get("ordered_targets")
    if isinstance(ordered_refs, Sequence) and not isinstance(ordered_refs, (str, bytes)):
        evidence_binding.setdefault(
            "ordered_target_refs",
            [str(ref) for ref in ordered_refs if str(ref or "").strip()],
        )
    return {
        "source_segment_id": str(source.get("source_segment_id") or ""),
        "raw_asr_ref": source.get("raw_asr_ref", ""),
        "visual_caption_source": str(source.get("visual_caption_source") or ""),
        "citation_provenance": dict(citation_provenance),
        "evidence_binding": evidence_binding,
    }


def _observation_segment_id(observation: Observation | Mapping[str, Any]) -> str:
    if isinstance(observation, Observation):
        raw_output = observation.raw_output if isinstance(observation.raw_output, Mapping) else {}
        regions = observation.regions
    else:
        raw_output = observation.get("raw_output", {})
        regions = observation.get("regions", [])
    if isinstance(raw_output, Mapping) and raw_output.get("segment_id"):
        return str(raw_output.get("segment_id"))
    if isinstance(regions, Sequence) and not isinstance(regions, (str, bytes)):
        for region in regions:
            if isinstance(region, Mapping) and region.get("segment_id"):
                return str(region.get("segment_id"))
    return ""


def _observation_time_range(observation: Mapping[str, Any]) -> list[float] | None:
    raw_output = observation.get("raw_output", {})
    if isinstance(raw_output, Mapping):
        time_range = raw_output.get("time_range")
        if isinstance(time_range, Sequence) and not isinstance(time_range, (str, bytes)) and len(time_range) >= 2:
            return [float(time_range[0]), float(time_range[1])]
        start = raw_output.get("start_sec")
        end = raw_output.get("end_sec")
        if start is not None and end is not None:
            return [float(start), float(end)]

    regions = observation.get("regions", [])
    if isinstance(regions, Sequence) and not isinstance(regions, (str, bytes)):
        for region in regions:
            if not isinstance(region, Mapping):
                continue
            start = region.get("start_sec")
            end = region.get("end_sec")
            if start is not None and end is not None:
                return [float(start), float(end)]
    return None


def _observation_event_label(*, raw_output: Mapping[str, Any], claim: str) -> str:
    for key in ["event_label", "event", "event_name", "sequence_item", "visible_event"]:
        value = _first_item(raw_output.get(key))
        if value is not None and str(value).strip():
            return str(value).strip()
    match = re.search(r"\bevent:\s*([^.;|]+)", claim, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _caption_timeline_entity(claim: str, *, max_chars: int = 220) -> str:
    text = " ".join(str(claim or "").split())
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def _grounding_quality(*, raw_output: Mapping[str, Any], limitations: str, confidence_signal: str = "") -> str:
    signal = str(confidence_signal or raw_output.get("confidence_signal", "")).strip().lower()
    if signal == "unsupported":
        return "inferred"
    if signal == "degenerate":
        return "weak"
    explicit = str(raw_output.get("grounding_quality", "")).strip().lower()
    if explicit in {
        "global_sparse",
        "visually_confirmed",
        "indexed_transcript",
        "query_global_context",
        "inferred",
        "external_knowledge",
        "weak",
    }:
        return explicit

    text = limitations.lower()
    if "external knowledge" in text or "outside knowledge" in text or "world knowledge" in text:
        return "external_knowledge"
    if "infer" in text or "guess" in text or "deduc" in text or "assum" in text:
        return "inferred"
    weak_markers = [
        "not directly visible",
        "not visible",
        "lacks explicit",
        "lack explicit",
        "no explicit",
        "ambiguous",
        "unclear",
        "low resolution",
        "blur",
        "limited",
        "weak",
    ]
    if any(marker in text for marker in weak_markers):
        return "weak"
    return "visually_confirmed"


def _sort_evidence_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grounding_rank = {
        "global_sparse": 0,
        "visually_confirmed": 0,
        "indexed_transcript": 0,
        "inferred": 1,
        "weak": 2,
        "external_knowledge": 3,
    }
    return sorted(
        rows,
        key=lambda row: (
            grounding_rank.get(str(row.get("grounding_quality", "weak")), 9),
            -float(row.get("confidence", 0.0) or 0.0),
            str(row.get("obs_id", "")),
        ),
    )


def _option_sort_key(option: str) -> tuple[int, str]:
    if option == "unassigned":
        return (1, option)
    return (0, option)


def _format_compact_entry(entry: Mapping[str, Any]) -> str:
    confidence = f" | confidence: {entry['confidence']}" if entry.get("confidence") else ""
    grounding = f" | gq: {entry['grounding_quality']}" if entry.get("grounding_quality") else ""
    limitations = entry.get("limitations") or "-"
    raw_output = entry.get("raw_output", {})
    integrity = ""
    support_note = ""
    if isinstance(raw_output, Mapping):
        integrity_value = str(raw_output.get("observation_integrity") or "").strip()
        if integrity_value:
            integrity = f" | integrity: {integrity_value}"
        if integrity_value in {"unverifiable", "internally_inconsistent"}:
            support_note = " | support: not answer-grade"
    return (
        f"- `{entry['observation_id']}` | tool: `{entry.get('tool', 'unknown')}`"
        f"{grounding}{confidence}{integrity}{support_note} | "
        f"claim: {entry.get('claim', '')} | limitations: {limitations}"
    )


def _format_context_only_entry(entry: Mapping[str, Any]) -> str:
    limitations = entry.get("limitations") or "-"
    tool_name = str(entry.get("tool", "unknown"))
    claim = str(entry.get("claim", "")).strip()
    if _tool_emits_candidate_hints_only(tool_name):
        clipped = (claim[:480] + "...") if len(claim) > 480 else claim
        marker = f"{tool_name} topic hint (one-shot, already executed)"
        return (
            f"- `{entry['observation_id']}` | tool: `{tool_name}` | "
            f"{marker} | claim: {clipped or '(empty)'} | limitations: {limitations}"
        )
    return (
        f"- `{entry['observation_id']}` | tool: `{tool_name}` | "
        f"context hint only; not answer support | claim: {claim} | limitations: {limitations}"
    )


def _format_navigation_entry(entry: Mapping[str, Any]) -> str:
    tool_name = str(entry.get("tool", "unknown"))
    if tool_name in {"target_coverage", "read_segment_detail", "search_segments", "ground_question", "locate_targets_in_segment"}:
        claim = _compact_text(str(entry.get("claim", "")), limit=720)
        suffix = _navigation_entry_suffix(entry)
        return f"- {entry['observation_id']}: {tool_name} | claim: {claim or '(empty)'}{suffix}"
    return f"- {entry['observation_id']}: {tool_name}"


def _attach_observation_payloads(
    entries: Sequence[Mapping[str, Any]],
    *,
    observations: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    by_id = {str(observation.get("observation_id", "")): observation for observation in observations}
    enriched = []
    for entry in entries:
        observation = by_id.get(str(entry.get("observation_id", "")))
        if observation is None:
            enriched.append(dict(entry))
            continue
        merged = dict(entry)
        raw_output = observation.get("raw_output", {})
        if isinstance(raw_output, Mapping):
            merged["raw_output"] = dict(raw_output)
        regions = observation.get("regions", [])
        if isinstance(regions, Sequence) and not isinstance(regions, (str, bytes)):
            merged["regions"] = list(regions)
        enriched.append(merged)
    return enriched


def _navigation_entry_suffix(entry: Mapping[str, Any]) -> str:
    if str(entry.get("tool", "")) != "locate_targets_in_segment":
        return ""
    raw_output = entry.get("raw_output", {})
    if not isinstance(raw_output, Mapping):
        return ""
    recommended_actions = raw_output.get("recommended_next_actions")
    if isinstance(recommended_actions, Sequence) and not isinstance(recommended_actions, (str, bytes)):
        for action in recommended_actions:
            if not isinstance(action, Mapping):
                continue
            if str(action.get("route_kind") or "") != "focused_ordered_list_vision":
                continue
            args = action.get("args")
            if not isinstance(args, Mapping):
                continue
            segment_id = str(args.get("segment_id") or "").strip()
            start_sec = args.get("start_sec")
            end_sec = args.get("end_sec")
            nframes = args.get("nframes")
            return (
                " | next: vision_read"
                f"(segment_id={segment_id}, start_sec={start_sec}, end_sec={end_sec}, nframes={nframes})"
            )
    verify_args = raw_output.get("verify_call_args")
    if not isinstance(verify_args, Mapping) or not verify_args:
        return ""
    compact_args = _compact_json(verify_args, limit=900)
    if not compact_args:
        return ""
    return f" | next: verify_segment_anchors verify_call_args={compact_args}"


def _compact_json(value: Any, *, limit: int = 900) -> str:
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        return ""
    return _compact_text(encoded, limit=limit)


def _format_rawish_entry(entry: Mapping[str, Any]) -> str:
    artifacts = entry.get("artifacts") or "-"
    limitations = entry.get("limitations") or "-"
    confidence = entry.get("confidence") or "0.00"
    line = (
        f"- `{entry['observation_id']}` | tool: `{entry.get('tool', 'unknown')}` | "
        f"confidence: {confidence} | artifacts: {artifacts} | "
        f"claim: {entry.get('claim', '')} | limitations: {limitations}"
    )
    raw_output = entry.get("raw_output", {})
    if isinstance(raw_output, Mapping) and raw_output:
        compact_raw = _compact_json(raw_output, limit=600)
        if compact_raw:
            line += f" | raw_output: {compact_raw}"
    return line


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
