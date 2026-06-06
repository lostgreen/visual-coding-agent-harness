"""Evidence workspace for visual tool results."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .agents.contracts import CONTRACT_VERSION, BudgetReason, EvidenceStage, GroundingQuality, SamplingPolicy
from .schemas import EvidenceRowV2


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
    }
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
    CONTEXT_ONLY_TOOLS = {"query_context"}
    ANSWER_EVIDENCE_TOOLS = (VISUAL_EVIDENCE_TOOLS - CONTEXT_ONLY_TOOLS) | {
        "caption_image",
        "caption_region",
        "ocr_region",
        "qa_region",
        "inspect_region",
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
        ]:
            child.mkdir(parents=True, exist_ok=True)

        for filename in [
            "observations.jsonl",
            "trace.jsonl",
            "evidence.jsonl",
            "evidence_table.jsonl",
            "map_proposals.jsonl",
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
        observation = Observation(
            observation_id=self._next_observation_id(),
            tool=tool_name,
            claim=claim,
            confidence=confidence,
            input_artifacts=list(input_artifacts),
            regions=list(regions),
            limitations=limitations,
            confidence_signal=confidence_signal or str((raw_output or {}).get("confidence_signal", "")),
            raw_output=dict(raw_output or {}),
            created_at=_utc_now(),
            frame_set_id=frame_set_id,
        )
        self._append_jsonl("observations.jsonl", asdict(observation))
        return observation

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

    def append_timeline_from_observation(self, observation: Observation) -> dict[str, Any] | None:
        if observation.tool not in {"vision_read", "inspect_segment"}:
            return None
        raw_output = observation.raw_output if isinstance(observation.raw_output, Mapping) else {}
        entity = _observation_event_label(raw_output=raw_output, claim=observation.claim)
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
                entry for entry in entries if str(entry.get("tool", "")) not in self.CONTEXT_ONLY_TOOLS
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
                f"- {entry['observation_id']}: {entry.get('tool', 'unknown')}"
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
                supported_option = _normalize_supported_option(option_source, option_map=option_map)
            group_key = supported_option or "unassigned"
            groups.setdefault(group_key, [])

            row = EvidenceRowV2(
                obs_id=str(observation.get("observation_id", "")),
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
                if isinstance(row, Mapping) and str(row.get("grounding_quality", "")) in {"visually_confirmed", "global_sparse"}
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
        """Check whether cited observation ids include answer-facing visual evidence."""

        cited = {str(item) for item in citations}
        if not cited:
            return False
        for observation in self._read_observation_dicts():
            if str(observation.get("observation_id", "")) not in cited:
                continue
            tool_name = str(observation.get("tool", ""))
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
        total_mapped = len(self.mapped_evidence_records())
        payload: dict[str, Any] = {
            "schema_version": "EvidenceChainsV1",
            "workspace_root": self.root.as_posix(),
            "chain_count": len(chains),
            "total_mapped_evidence": total_mapped,
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
        confidence_match = re.search(r"confidence:\s*([0-9.]+)", line)
        artifacts_match = re.search(r"artifacts:\s*(.*?)\s*\|\s*claim:", line)
        claim_match = re.search(r"claim:\s*(.*?)\s*\|\s*limitations:", line)
        limitation_match = re.search(r"limitations:\s*(.*)$", line)
        entries.append(
            {
                "observation_id": obs_match.group(1),
                "tool": tool_match.group(1) if tool_match else "unknown",
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
    ).to_dict()
    if payload["time_range"] is None and payload["t_start"] is not None and payload["t_end"] is not None:
        payload["time_range"] = [payload["t_start"], payload["t_end"]]
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
    return payload


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
        "segment_id": str(raw_output.get("segment_id", "")),
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
    limitations = entry.get("limitations") or "-"
    return (
        f"- `{entry['observation_id']}` | tool: `{entry.get('tool', 'unknown')}`{confidence} | "
        f"claim: {entry.get('claim', '')} | limitations: {limitations}"
    )


def _format_context_only_entry(entry: Mapping[str, Any]) -> str:
    limitations = entry.get("limitations") or "-"
    return (
        f"- `{entry['observation_id']}` | tool: `{entry.get('tool', 'unknown')}` | "
        f"context hint only; not answer support | claim: {entry.get('claim', '')} | limitations: {limitations}"
    )


def _format_rawish_entry(entry: Mapping[str, Any]) -> str:
    artifacts = entry.get("artifacts") or "-"
    limitations = entry.get("limitations") or "-"
    confidence = entry.get("confidence") or "0.00"
    return (
        f"- `{entry['observation_id']}` | tool: `{entry.get('tool', 'unknown')}` | "
        f"confidence: {confidence} | artifacts: {artifacts} | "
        f"claim: {entry.get('claim', '')} | limitations: {limitations}"
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
