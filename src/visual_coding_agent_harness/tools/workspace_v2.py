"""Workspace-first v2 tool surface."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Optional, Sequence

from ..backends.base import BackendRequest, VisionLanguageBackend
from ..core.registry import DuplicateGuardPolicy, ToolRegistry, ToolRuntimeSpec, tool
from ..video.map import (
    IndexRefiner,
    VideoMap,
    VideoMapSegment,
    VideoMapStore,
    _resolve_search_modalities,
    search_modality_limitations,
)
from ..workspace import EvidenceWorkspace
from .frame_cache import FrameSampler
from .tool_specs import (
    _normalize_read_clip,
    _normalize_read_segment,
    _read_segment_semantic_key,
    _normalize_workspace_v2_answer,
    _normalize_workspace_v2_list,
    _normalize_workspace_v2_search,
    _normalize_workspace_v2_synthesize_memory,
    _normalize_workspace_v2_verify,
)
from .workspace_primitives import build_workspace_primitives_registry


ANSWER_SUPPORTING_KINDS = frozenset({"answer_support", "synthesized_support", "answer_conflict_resolved"})
VERIFY_WINDOW_DEFAULT_FPS = 2.0
VERIFY_WINDOW_MAX_FRAMES = 128


def build_workspace_v2_registry(
    *,
    video_map: VideoMap | VideoMapStore,
    backend: VisionLanguageBackend,
    workspace: Optional[EvidenceWorkspace] = None,
    include_workspace_primitives: bool = True,
    index_refiner: IndexRefiner | None = None,
    frame_sampler: FrameSampler | None = None,
) -> ToolRegistry:
    """Build the compact v2 registry: read_clip/search/list/verify/answer plus dispositions."""

    video_map_store = video_map if isinstance(video_map, VideoMapStore) else VideoMapStore(video_map)
    artifact_root = workspace.root / "artifacts" / "index_refinement" if workspace is not None else None
    segment_read_service = SegmentReadService(
        video_map_store=video_map_store,
        backend=backend,
        index_refiner=index_refiner or IndexRefiner(backend=backend, artifact_root=artifact_root),
        frame_sampler=frame_sampler,
        workspace=workspace,
    )
    registry = ToolRegistry()

    @tool(name="read_segment", description="Progressively read a video segment index, refinement, or verified evidence.")
    def read_segment(
        segment_id: str,
        mode: str = "index",
        sub_window: Mapping[str, float] | None = None,
        resolution: str = "medium",
        evidence_mode: str = "visual",
        focus: Sequence[str] = (),
    ) -> Mapping[str, object]:
        if mode == "index":
            return segment_read_service.read_index(segment_id=segment_id)
        if mode == "refine":
            return segment_read_service.refine_index(
                segment_id=segment_id,
                sub_window=sub_window,
                resolution=resolution,
                focus=focus,
            )
        if mode == "verify":
            return segment_read_service.verify_evidence(
                segment_id=segment_id,
                sub_window=sub_window,
                evidence_mode=evidence_mode,
                focus=focus,
            )
        raise ValueError(f"read_segment_failed: unknown mode={mode}")

    @tool(name="scan_segment", description="Ask an IndexScout worker to turn one raw segment index into candidate verification windows.")
    def scan_segment(
        segment_id: str,
        question: str = "",
        options: Mapping[str, Any] | None = None,
        scan_goal: str = "",
        preferred_modalities: Sequence[str] = (),
        max_candidates: int = 3,
    ) -> Mapping[str, object]:
        return segment_read_service.scan_segment(
            segment_id=segment_id,
            question=question,
            options=options or {},
            scan_goal=scan_goal,
            preferred_modalities=preferred_modalities,
            max_candidates=max_candidates,
        )

    @tool(name="verify_window", description="Ask an EvidenceVerifier worker to read one candidate window and return local facts with anchors.")
    def verify_window(
        candidate_id: str = "",
        segment_id: str = "",
        time_range: Sequence[float] | Mapping[str, float] | None = None,
        evidence_mode: str = "multimodal",
        focus: Sequence[str] = (),
        checks: Sequence[Mapping[str, Any] | str] = (),
        verification_targets: Sequence[Mapping[str, Any] | str] = (),
        sampling: Mapping[str, Any] | None = None,
    ) -> Mapping[str, object]:
        return segment_read_service.verify_window(
            candidate_id=candidate_id,
            segment_id=segment_id,
            time_range=time_range,
            evidence_mode=evidence_mode,
            focus=focus,
            checks=checks,
            verification_targets=verification_targets,
            sampling=sampling,
        )

    @tool(name="read_clip", description="Read facts from a video clip without choosing an answer option.")
    def read_clip(
        scope: Mapping[str, Any],
        focus: Sequence[str] = (),
        sampling: Mapping[str, Any] | None = None,
    ) -> Mapping[str, object]:
        return _read_clip_evidence(
            video_map_store=video_map_store,
            backend=backend,
            frame_sampler=frame_sampler,
            scope=scope,
            focus=focus,
            sampling=sampling,
            tool_name="read_clip",
        )

    @tool(name="search", description="Search substrate indexes and return candidate-only hits.")
    def search(
        query: str,
        modality: str | Sequence[str] = (),
        scope: Mapping[str, Any] | None = None,
        top_k: int = 5,
    ) -> Mapping[str, object]:
        current = video_map_store.current
        modalities = _modalities_arg(modality)
        candidates = current.search(query=query, top_k=int(top_k), modalities=modalities)
        scope_filter = dict(scope or {})
        if scope_filter:
            candidates = [
                result
                for result in candidates
                if _segment_matches_scope(result.segment, scope_filter)
            ]
        results: list[dict[str, object]] = []
        anchors: list[dict[str, object]] = []
        for index, result in enumerate(candidates, start=1):
            match = _first_match(result.to_dict())
            anchor_id = f"anch_search_{result.segment.segment_id}_{index:03d}"
            time_range = [float(result.segment.start_sec), float(result.segment.end_sec)]
            excerpt = str(match.get("evidence") or result.segment.compact_text())
            row = {
                "hit_id": f"hit_{index:03d}",
                "modality": str(match.get("modality") or _first_modality(result.matched_fields)),
                "segment_id": result.segment.segment_id,
                "time_range": time_range,
                "excerpt": excerpt,
                "matched_terms": list(match.get("matched_terms", [])),
                "candidate_anchor_id": anchor_id,
                "support_status": "candidate_only",
                "needs_local_read": True,
                "must_commit_as": "retrieval_candidate",
                "cannot_final_cite": True,
                "recommended_next_tool": "read_segment",
                "recommended_mode": "verify",
                "recommended_scope": {"segment_id": result.segment.segment_id, "sub_window": {"start_sec": time_range[0], "end_sec": time_range[1]}},
                "score": result.score,
            }
            results.append(row)
            anchors.append(
                {
                    "anchor_id": anchor_id,
                    "observation_id": "__pending__",
                    "source_kind": "retrieval_hit",
                    "segment_id": result.segment.segment_id,
                    "start_sec": time_range[0],
                    "end_sec": time_range[1],
                    "field_path": "results",
                    "excerpt": excerpt,
                    "modality": row["modality"],
                }
            )
        return {
            "claim": f"Search for '{query}' returned {len(results)} candidate hit(s).",
            "confidence": 0.85 if results else 0.2,
            "results": results,
            "regions": results,
            "produced_anchors": anchors,
            "limitations": "Search results are candidate_only and cannot be final citations until committed.",
            "modalities_used": _resolve_search_modalities(modalities),
            "scope": scope_filter,
        }

    @tool(name="list", description="List substrate or committed workspace state.")
    def list_tool(kind: str, filter: Mapping[str, Any] | None = None) -> Mapping[str, object]:
        current = video_map_store.current
        kind_text = str(kind or "").strip()
        filter_payload = dict(filter or {})
        if kind_text in {"segments", "segment"}:
            items = [
                dict(segment.to_dict())
                for segment in current.segments
                if _row_matches_filter(segment.to_dict(), filter_payload)
            ]
        elif kind_text in {"memory", "entities", "events", "relations", "attributes", "pinned_anchors", "observations_by_id", "plan", "open_questions"}:
            items = workspace.read_workspace_section(kind_text, filter=filter_payload) if workspace is not None else []
        else:
            raise ValueError(f"workspace_v2_list_failed: unknown kind={kind_text}")
        return {
            "claim": f"Listed {len(items)} item(s) for {kind_text}.",
            "confidence": 1.0,
            "items": items,
            "regions": [{"kind": kind_text, "items": items}],
            "limitations": "Cheap structured listing; no video frames inspected.",
        }

    @tool(name="verify", description="Phase-1 provenance gate for citations and committed state.")
    def verify(claim: str, against: Mapping[str, Any]) -> Mapping[str, object]:
        del claim
        citations = [str(item).strip() for item in _sequence_items(against.get("citations")) if str(item).strip()]
        if not citations:
            reason = "verify requires at least one citation"
            return {
                "claim": f"Provenance gate rejected: {reason}",
                "confidence": 0.0,
                "accepted": False,
                "phase": "provenance_gate",
                "reason": reason,
                "citations": [],
                "limitations": "Phase 1 checks provenance only; semantic entailment is not judged.",
            }
        accepted, reason = _verify_citations(
            workspace,
            citations,
            require_memory=bool(against.get("final")),
            require_answer_support=bool(against.get("final")),
        )
        return {
            "claim": "Provenance gate accepted." if accepted else f"Provenance gate rejected: {reason}",
            "confidence": 1.0 if accepted else 0.0,
            "accepted": accepted,
            "phase": "provenance_gate",
            "reason": reason,
            "citations": citations,
            "limitations": "Phase 1 checks provenance only; semantic entailment is not judged.",
        }

    @tool(name="synthesize_memory", description="Derive planner memory from committed workspace memory.")
    def synthesize_memory(
        claim: str,
        supports: Sequence[str],
        derived_from: Sequence[str] = (),
        evidence_obs_ids: Sequence[str] = (),
        confidence: str = "medium",
        supports_option: str = "",
        tags: Sequence[str] = (),
    ) -> Mapping[str, object]:
        if workspace is None:
            raise ValueError("synthesize_memory_failed: workspace is required")
        support_ids = _unique_nonempty(supports)
        derived_ids = _unique_nonempty(derived_from)
        if not support_ids:
            raise ValueError("synthesize_memory_failed: supports must include committed memory ids")

        memory_by_id = {entry.entry_id: entry for entry in workspace.memory_entries()}
        missing = [memory_id for memory_id in support_ids + derived_ids if memory_id not in memory_by_id]
        if missing:
            raise ValueError("synthesize_memory_failed: unknown memory id: " + ", ".join(missing))
        unsupported_supports = [
            memory_id
            for memory_id in support_ids
            if memory_by_id[memory_id].kind not in ANSWER_SUPPORTING_KINDS
        ]
        if unsupported_supports:
            kinds = ", ".join(f"{memory_id}:{memory_by_id[memory_id].kind}" for memory_id in unsupported_supports)
            raise ValueError("synthesize_memory_failed: supporting memory kind required for " + kinds)

        del evidence_obs_ids

        anchor_ids = _unique_nonempty(
            anchor.anchor_id
            for memory_id in support_ids
            for anchor in memory_by_id[memory_id].anchors
        )
        if not anchor_ids:
            raise ValueError("synthesize_memory_failed: supports do not contain anchors")
        lineage = _unique_nonempty([*support_ids, *derived_ids])
        provenance_obs_ids = _provenance_observation_ids(memory_by_id, lineage)
        entry = workspace.write_memory(
            kind="synthesized_support",
            claim=str(claim),
            anchors=[{"anchor_id": anchor_id} for anchor_id in anchor_ids],
            supports_option=supports_option,
            confidence=confidence,
            previous_memory_refs=lineage,
            tags=tags,
            role="planner",
            layer="derived",
            metadata={
                "tool": "synthesize_memory",
                "supports": support_ids,
                "derived_from": derived_ids,
                "evidence_obs_ids": provenance_obs_ids,
            },
        )
        return {
            "claim": entry.claim,
            "confidence": 1.0 if entry.confidence == "high" else 0.75,
            "memory_id": entry.entry_id,
            "citations": [entry.entry_id],
            "anchor_ids": [anchor.anchor_id for anchor in entry.anchors],
            "previous_memory_refs": list(entry.previous_memory_refs),
            "limitations": "Derived only from committed workspace memory; no new raw observation was inspected.",
        }

    @tool(name="answer", description="Validate and return a final answer with committed workspace citations.")
    def answer(text: str, citations: Sequence[str], confidence: str = "medium") -> Mapping[str, object]:
        citation_ids = [str(item).strip() for item in citations if str(item).strip()]
        accepted, reason = _verify_citations(
            workspace,
            citation_ids,
            require_memory=True,
            require_answer_support=True,
        )
        if not accepted:
            raise ValueError(f"answer_validation_failed: {reason}")
        return {
            "claim": str(text),
            "confidence": 1.0 if confidence == "high" else 0.75,
            "answer": str(text),
            "citations": citation_ids,
            "answer_confidence": str(confidence),
            "accepted": True,
            "limitations": "Final answer cites committed planner-authored workspace memory.",
        }

    registry.register(
        ToolRuntimeSpec(
            tool_spec=read_segment,
            argument_normalizer=_normalize_read_segment,
            semantic_key_builder=_read_segment_semantic_key,
            duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
            commit_required_predicate=_read_segment_verify_has_evidence,
        )
    )
    registry.register(
        ToolRuntimeSpec(
            tool_spec=scan_segment,
            semantic_key_builder=_static_key("scan_segment"),
            duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
        )
    )
    registry.register(
        ToolRuntimeSpec(
            tool_spec=verify_window,
            semantic_key_builder=_static_key("verify_window"),
            duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
            commit_required=True,
        )
    )
    registry.register(
        ToolRuntimeSpec(
            tool_spec=read_clip,
            argument_normalizer=_normalize_read_clip,
            semantic_key_builder=_static_key("read_clip"),
            duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
            commit_required=True,
        )
    )
    registry.register(
        ToolRuntimeSpec(
            tool_spec=search,
            argument_normalizer=_normalize_workspace_v2_search,
            semantic_key_builder=_static_key("search"),
            duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
            commit_required_predicate=_search_has_evidence,
        )
    )
    registry.register(
        ToolRuntimeSpec(
            tool_spec=list_tool,
            argument_normalizer=_normalize_workspace_v2_list,
            semantic_key_builder=_static_key("list"),
            duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
        )
    )
    registry.register(
        ToolRuntimeSpec(
            tool_spec=verify,
            argument_normalizer=_normalize_workspace_v2_verify,
            semantic_key_builder=_static_key("verify"),
            duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
            commit_required_predicate=_verify_has_rejection_evidence,
        )
    )
    registry.register(
        ToolRuntimeSpec(
            tool_spec=synthesize_memory,
            argument_normalizer=_normalize_workspace_v2_synthesize_memory,
            semantic_key_builder=_static_key("synthesize_memory"),
            duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
        )
    )
    registry.register(
        ToolRuntimeSpec(
            tool_spec=answer,
            argument_normalizer=_normalize_workspace_v2_answer,
            duplicate_guard_policy=DuplicateGuardPolicy.OFF,
        )
    )
    if include_workspace_primitives:
        registry.extend(build_workspace_primitives_registry(workspace=workspace))
    return registry


class SegmentReadService:
    def __init__(
        self,
        *,
        video_map_store: VideoMapStore,
        backend: VisionLanguageBackend,
        index_refiner: IndexRefiner,
        frame_sampler: FrameSampler | None = None,
        workspace: EvidenceWorkspace | None = None,
    ) -> None:
        self.video_map_store = video_map_store
        self.backend = backend
        self.index_refiner = index_refiner
        self.frame_sampler = frame_sampler
        self.workspace = workspace
        self._indexed_roots: set[str] = {
            _root_segment_id(segment)
            for segment in self.video_map_store.current.segments
            if getattr(segment, "index_level", "root") == "root"
        }

    def read_index(self, *, segment_id: str) -> Mapping[str, object]:
        current = self.video_map_store.current
        segment = current.get(segment_id)
        self._indexed_roots.add(_root_segment_id(segment))
        children = [
            child
            for child in current.segments
            if child.parent_segment_id == segment.segment_id and child.index_level == "refined"
        ]
        return {
            "claim": f"Read navigation index for {segment.segment_id}.",
            "confidence": 1.0,
            "mode": "index",
            "segment_id": segment.segment_id,
            "index_level": segment.index_level,
            "root_summary": segment.low_fps_caption,
            "timeline_beats": [beat.to_dict() for beat in segment.timeline_beats],
            "asr_cue_count": len(segment.asr_sentences),
            "ocr_hint_count": len(segment.ocr_frames),
            "children": [child.segment_id for child in children],
            "regions": [segment.to_dict()],
            "produced_anchors": [],
            "commit_required": False,
            "limitations": ["navigation only; not answer evidence"],
        }

    def scan_segment(
        self,
        *,
        segment_id: str,
        question: str,
        options: Mapping[str, Any],
        scan_goal: str,
        preferred_modalities: Sequence[str],
        max_candidates: int,
    ) -> Mapping[str, object]:
        current = self.video_map_store.current
        segment = current.get(segment_id)
        self._indexed_roots.add(_root_segment_id(segment))
        max_candidates = max(0, min(8, int(max_candidates or 3)))
        if self.workspace is not None:
            self.workspace.write_trace_event(
                "index_scout_dispatched",
                {
                    "segment_id": segment.segment_id,
                    "start_sec": segment.start_sec,
                    "end_sec": segment.end_sec,
                    "max_candidates": max_candidates,
                    "preferred_modalities": [str(item) for item in preferred_modalities],
                },
            )
        raw_index = _raw_segment_index_payload(segment)
        prompt = _scan_segment_prompt(
            question=question,
            options=options,
            scan_goal=scan_goal,
            preferred_modalities=preferred_modalities,
            max_candidates=max_candidates,
            raw_index=raw_index,
        )
        scan_notes = ""
        try:
            response = self.backend.generate(
                BackendRequest(
                    task="replan",
                    prompt=prompt,
                    max_new_tokens=1024,
                    temperature=0.0,
                    metadata={
                        "worker": "IndexScout",
                        "tool": "scan_segment",
                        "segment_id": segment.segment_id,
                    },
                )
            )
            payload = _parse_json_object(response.text)
            candidates = _candidate_windows_from_payload(
                payload,
                segment=segment,
                max_candidates=max_candidates,
                default_goal=scan_goal or question,
                preferred_modalities=preferred_modalities,
            )
            scan_notes = str(payload.get("scan_notes") or "")
        except Exception as exc:
            candidates = _fallback_candidate_windows(
                segment=segment,
                max_candidates=max_candidates,
                default_goal=scan_goal or question,
                preferred_modalities=preferred_modalities,
            )
            scan_notes = f"IndexScout fallback used after invalid worker output: {exc}"
        claim = _candidate_windows_claim(segment.segment_id, candidates)
        return {
            "claim": claim,
            "confidence": 1.0,
            "worker": "IndexScout",
            "mode": "scan_segment",
            "segment_id": segment.segment_id,
            "candidate_windows": candidates,
            "regions": [
                {
                    "segment_id": segment.segment_id,
                    "start_sec": segment.start_sec,
                    "end_sec": segment.end_sec,
                    "worker": "IndexScout",
                }
            ],
            "produced_anchors": [],
            "commit_required": False,
            "limitations": ["navigation only; candidate windows require verify_window before answer use"],
            "scan_notes": scan_notes,
        }

    def refine_index(
        self,
        *,
        segment_id: str,
        sub_window: Mapping[str, float] | None,
        resolution: str,
        focus: Sequence[str],
    ) -> Mapping[str, object]:
        parent = self.video_map_store.current.get(segment_id)
        self._require_index_read(parent, mode="refine")
        start_sec, end_sec = _sub_window_range(sub_window, parent, mode="refine", require_explicit=True)
        patch = self.index_refiner.refine(
            self.video_map_store,
            parent_segment_id=segment_id,
            requested_start_sec=start_sec,
            requested_end_sec=end_sec,
            resolution=_resolution(resolution),
            focus=focus,
        )
        if self.workspace is not None:
            self.workspace.write_trace_event(
                "index_refinement_cache_hit" if patch.cache_hit else "index_refinement_created",
                {
                    "patch_id": patch.patch_id,
                    "parent_segment_id": segment_id,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "resolution": resolution,
                },
            )
        return {
            "claim": f"Refined navigation index for {segment_id} [{start_sec:.1f}, {end_sec:.1f}]s.",
            "confidence": 1.0,
            "mode": "refine",
            "patch": patch.to_dict(),
            "regions": [child.to_dict() for child in patch.children],
            "produced_anchors": [],
            "commit_required": False,
            "limitations": ["navigation only; not answer evidence"],
        }

    def verify_evidence(
        self,
        *,
        segment_id: str,
        sub_window: Mapping[str, float] | None,
        evidence_mode: str,
        focus: Sequence[str],
    ) -> Mapping[str, object]:
        parent = self.video_map_store.current.get(segment_id)
        self._require_index_read(parent, mode="verify")
        start_sec, end_sec = _sub_window_range(sub_window, parent, mode="verify", require_explicit=True)
        if self.workspace is not None:
            self.workspace.write_trace_event(
                "segment_verify_dispatched",
                {
                    "segment_id": segment_id,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "evidence_mode": evidence_mode,
                },
            )
        focus_items = [str(evidence_mode).strip(), *[str(item).strip() for item in focus if str(item).strip()]]
        result = _read_clip_evidence(
            video_map_store=self.video_map_store,
            backend=self.backend,
            frame_sampler=self.frame_sampler,
            scope={"segment_id": segment_id, "time_range": [start_sec, end_sec]},
            focus=focus_items,
            sampling={},
            tool_name="read_segment",
        )
        return {**result, "mode": "verify", "evidence_mode": evidence_mode}

    def verify_window(
        self,
        *,
        candidate_id: str,
        segment_id: str,
        time_range: Sequence[float] | Mapping[str, float] | None,
        evidence_mode: str,
        focus: Sequence[str],
        checks: Sequence[Mapping[str, Any] | str] = (),
        verification_targets: Sequence[Mapping[str, Any] | str] = (),
        sampling: Mapping[str, Any] | None,
    ) -> Mapping[str, object]:
        candidate = self._resolve_candidate(candidate_id=candidate_id, segment_id=segment_id, time_range=time_range)
        resolved_segment_id = str(candidate["segment_id"])
        start_sec = float(candidate["start_sec"])
        end_sec = float(candidate["end_sec"])
        parent = self.video_map_store.current.get(resolved_segment_id)
        if start_sec < float(parent.start_sec) or end_sec > float(parent.end_sec) or end_sec <= start_sec:
            raise ValueError("verify_window_failed: candidate time_range must be non-empty and within its root segment")
        sampling_payload = _verification_sampling_payload(sampling, start_sec=start_sec, end_sec=end_sec)
        targets = _normalize_verification_targets((*_sequence_items(checks), *_sequence_items(verification_targets)))
        if self.workspace is not None:
            self.workspace.write_trace_event(
                "evidence_verifier_dispatched",
                {
                    "candidate_id": str(candidate.get("candidate_id") or ""),
                    "segment_id": resolved_segment_id,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "evidence_mode": evidence_mode,
                    "sampling": dict(sampling_payload),
                    "verification_targets": targets,
                },
            )
        focus_items = [
            str(evidence_mode).strip(),
            str(candidate.get("verification_goal") or "").strip(),
            *[str(item).strip() for item in focus if str(item).strip()],
        ]
        result = _read_clip_evidence(
            video_map_store=self.video_map_store,
            backend=self.backend,
            frame_sampler=self.frame_sampler,
            scope={"segment_id": resolved_segment_id, "time_range": [start_sec, end_sec]},
            focus=focus_items,
            verification_targets=targets,
            sampling=sampling_payload,
            tool_name="verify_window",
        )
        return {
            **result,
            "worker": "EvidenceVerifier",
            "mode": "verify_window",
            "candidate_id": str(candidate.get("candidate_id") or candidate_id or ""),
            "source_beat_ids": list(candidate.get("source_beat_ids") or ()),
            "verification_goal": str(candidate.get("verification_goal") or ""),
            "verification_targets": targets,
            "evidence_mode": evidence_mode,
        }

    def _require_index_read(self, segment: VideoMapSegment, *, mode: str) -> None:
        root_id = _root_segment_id(segment)
        if root_id not in self._indexed_roots:
            raise ValueError(
                f"read_segment_failed: requires_index_read before mode={mode} for root_segment_id={root_id}"
            )

    def _resolve_candidate(
        self,
        *,
        candidate_id: str,
        segment_id: str,
        time_range: Sequence[float] | Mapping[str, float] | None,
    ) -> dict[str, object]:
        normalized_candidate_id = str(candidate_id or "").strip()
        if normalized_candidate_id and self.workspace is not None:
            for observation in reversed(self.workspace.read_observations()):
                raw_output = observation.raw_output if isinstance(observation.raw_output, Mapping) else {}
                for candidate in _mapping_items(raw_output.get("candidate_windows")):
                    if str(candidate.get("candidate_id") or "") == normalized_candidate_id:
                        return dict(candidate)
        if not segment_id:
            raise ValueError("verify_window_failed: candidate_id was not found; provide segment_id and time_range")
        start_sec, end_sec = _time_range_argument(time_range)
        return {
            "candidate_id": normalized_candidate_id,
            "segment_id": str(segment_id),
            "start_sec": start_sec,
            "end_sec": end_sec,
            "source_beat_ids": [],
            "verification_goal": "; ".join(str(item).strip() for item in ()),
        }


def _read_clip_evidence(
    *,
    video_map_store: VideoMapStore,
    backend: VisionLanguageBackend,
    frame_sampler: FrameSampler | None = None,
    scope: Mapping[str, Any],
    focus: Sequence[str],
    verification_targets: Sequence[Mapping[str, Any]] = (),
    sampling: Mapping[str, Any] | None,
    tool_name: str,
) -> Mapping[str, object]:
    current = video_map_store.current
    segment = _segment_from_scope(current, scope)
    start_sec, end_sec = _time_range_from_scope(scope, segment)
    focus_items = [str(item).strip() for item in focus if str(item).strip()]
    targets = _normalize_verification_targets(verification_targets)
    ask_for = "; ".join(focus_items) or "Return concise visible/audio/OCR facts from this clip."
    if targets:
        ask_for = "\n".join(
            [
                ask_for,
                "Verification targets:",
                *_format_verification_target_lines(targets),
                "For each target, return only local findings for this clip and distinguish present, absent, and uncertain.",
            ]
        )
    sampling_payload = dict(sampling or {})
    requested_nframes = int(sampling_payload.get("nframes") or 8)
    requested_nframes = max(1, requested_nframes)
    frame_paths = (
        tuple(frame_sampler(current.video_path, float(start_sec), float(end_sec), requested_nframes))
        if frame_sampler is not None
        else ()
    )
    media_path = None if frame_paths else current.video_path
    response = backend.generate(
        BackendRequest(
            task="vision_read",
            prompt=_read_clip_prompt(
                segment_id=segment.segment_id,
                start_sec=start_sec,
                end_sec=end_sec,
                focus=ask_for,
            ),
            media_path=media_path,
            media_type="video",
            frames=frame_paths,
            max_new_tokens=384,
            metadata={
                "tool": tool_name,
                "segment_id": segment.segment_id,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "focus": focus_items,
                "verification_targets": targets,
                "sampling": sampling_payload,
                "nframes": len(frame_paths) if frame_paths else requested_nframes,
            },
        )
    )
    fact_text = str(response.text or "").strip()
    raw_backend = dict(response.raw or {})
    fallback_source_kind = _source_kind_from_backend(raw_backend, focus_items)
    facts = _facts_from_backend_response(
        text=fact_text,
        raw_backend=raw_backend,
        fallback_source_kind=fallback_source_kind,
        time_range=[start_sec, end_sec],
    )
    base_anchor_id = f"clip_anch_{segment.segment_id}_{int(start_sec * 1000):08d}_{int(end_sec * 1000):08d}"
    produced_anchors = []
    for index, fact in enumerate(facts):
        source_kind = str(fact.get("source_kind") or fallback_source_kind)
        anchor_id = base_anchor_id if len(facts) == 1 else f"{base_anchor_id}_{index + 1:02d}"
        produced_anchors.append(
            {
                "anchor_id": anchor_id,
                "observation_id": "__pending__",
                "source_kind": source_kind,
                "segment_id": segment.segment_id,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "field_path": f"facts[{index}].text",
                "excerpt": str(fact.get("text") or ""),
                "frame_refs": list(fact.get("frames_used") or []),
                "modality": _anchor_modality(source_kind),
            }
        )
    return {
        "claim": fact_text,
        "confidence": 0.74,
        "facts": facts,
        "regions": [
            {
                "segment_id": segment.segment_id,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "focus": focus_items,
            }
        ],
        "time_range": [start_sec, end_sec],
        "candidate_anchor_ids": [str(anchor["anchor_id"]) for anchor in produced_anchors],
        "produced_anchors": produced_anchors,
        "limitations": "Facts-only clip read; no answer option vote is emitted.",
        "verification_targets": targets,
        "raw_backend": raw_backend,
        "raw": {**raw_backend, "verification_targets": targets},
    }


def _sub_window_range(
    sub_window: Mapping[str, float] | None,
    segment: VideoMapSegment,
    *,
    mode: str,
    require_explicit: bool = False,
) -> tuple[float, float]:
    payload = dict(sub_window or {})
    if require_explicit and ("start_sec" not in payload or "end_sec" not in payload):
        raise ValueError(
            f"read_segment_failed: mode={mode} requires explicit sub_window start_sec/end_sec from a DVC beat"
        )
    start_sec = float(payload.get("start_sec", segment.start_sec))
    end_sec = float(payload.get("end_sec", segment.end_sec))
    if start_sec < segment.start_sec or end_sec > segment.end_sec or end_sec <= start_sec:
        raise ValueError("read_segment_failed: sub_window must be non-empty and within the root segment")
    return start_sec, end_sec


def _raw_segment_index_payload(segment: VideoMapSegment) -> Mapping[str, object]:
    return {
        "segment_id": segment.segment_id,
        "time_range": [float(segment.start_sec), float(segment.end_sec)],
        "root_summary": segment.low_fps_caption,
        "entities": list(segment.entities),
        "dense_video_caption": [beat.to_dict() for beat in segment.timeline_beats],
        "asr_cues": [dict(item) for item in segment.asr_sentences],
        "ocr_spans": [dict(item) for item in segment.ocr_frames],
    }


def _scan_segment_prompt(
    *,
    question: str,
    options: Mapping[str, Any],
    scan_goal: str,
    preferred_modalities: Sequence[str],
    max_candidates: int,
    raw_index: Mapping[str, object],
) -> str:
    payload = {
        "question": str(question or ""),
        "options": {str(key): str(value) for key, value in dict(options or {}).items()},
        "scan_goal": str(scan_goal or ""),
        "preferred_modalities": [str(item) for item in preferred_modalities],
        "max_candidates": int(max_candidates),
        "raw_segment_index": raw_index,
    }
    return (
        "You are IndexScout, an index-navigation worker.\n"
        "Read the raw index for ONE segment and propose candidate time windows for later verification.\n"
        "Do not answer the question. Do not choose A/B/C/D. Do not treat captions as final evidence.\n"
        "Return exactly one JSON object with keys: candidates, scan_notes.\n"
        "Each candidate must include time_range [start_sec,end_sec], source_beat_ids, entities, "
        "verification_goal, recommended_evidence_mode, and priority.\n\n"
        f"INPUT:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        loaded = json.loads(stripped[start : end + 1])
    if not isinstance(loaded, dict):
        raise ValueError("expected JSON object")
    return dict(loaded)


def _candidate_windows_from_payload(
    payload: Mapping[str, Any],
    *,
    segment: VideoMapSegment,
    max_candidates: int,
    default_goal: str,
    preferred_modalities: Sequence[str],
) -> list[dict[str, object]]:
    raw_candidates = payload.get("candidates")
    if raw_candidates is None:
        raw_candidates = payload.get("candidate_windows")
    candidates = []
    for item in _mapping_items(raw_candidates):
        try:
            start_sec, end_sec = _time_range_argument(item.get("time_range"))
        except ValueError:
            continue
        if start_sec < float(segment.start_sec) or end_sec > float(segment.end_sec) or end_sec <= start_sec:
            continue
        index = len(candidates) + 1
        candidates.append(
            _candidate_window(
                segment=segment,
                index=index,
                start_sec=start_sec,
                end_sec=end_sec,
                source_beat_ids=_string_sequence(item.get("source_beat_ids")),
                entities=_string_sequence(item.get("entities")),
                verification_goal=str(item.get("verification_goal") or default_goal or "Verify local facts in this window."),
                recommended_evidence_mode=str(
                    item.get("recommended_evidence_mode")
                    or item.get("evidence_mode")
                    or _preferred_evidence_mode(preferred_modalities)
                ),
                priority=int(item.get("priority") or index),
            )
        )
        if len(candidates) >= max_candidates:
            break
    return candidates or _fallback_candidate_windows(
        segment=segment,
        max_candidates=max_candidates,
        default_goal=default_goal,
        preferred_modalities=preferred_modalities,
    )


def _fallback_candidate_windows(
    *,
    segment: VideoMapSegment,
    max_candidates: int,
    default_goal: str,
    preferred_modalities: Sequence[str],
) -> list[dict[str, object]]:
    if max_candidates <= 0:
        return []
    candidates: list[dict[str, object]] = []
    beats = list(segment.timeline_beats or ())
    if not beats:
        beats = [
            type(
                "_FallbackBeat",
                (),
                {
                    "beat_id": f"{segment.segment_id}_root",
                    "start_sec": float(segment.start_sec),
                    "end_sec": float(segment.end_sec),
                    "entity_hints": tuple(segment.entities),
                },
            )()
        ]
    for index, beat in enumerate(beats[:max_candidates], start=1):
        candidates.append(
            _candidate_window(
                segment=segment,
                index=index,
                start_sec=float(getattr(beat, "start_sec", segment.start_sec)),
                end_sec=float(getattr(beat, "end_sec", segment.end_sec)),
                source_beat_ids=[str(getattr(beat, "beat_id", f"{segment.segment_id}_b{index:02d}"))],
                entities=[str(item) for item in getattr(beat, "entity_hints", ()) or ()],
                verification_goal=default_goal or "Verify local facts in this window.",
                recommended_evidence_mode=_preferred_evidence_mode(preferred_modalities),
                priority=index,
            )
        )
    return candidates


def _candidate_window(
    *,
    segment: VideoMapSegment,
    index: int,
    start_sec: float,
    end_sec: float,
    source_beat_ids: Sequence[str],
    entities: Sequence[str],
    verification_goal: str,
    recommended_evidence_mode: str,
    priority: int,
) -> dict[str, object]:
    candidate_id = f"cand_{segment.segment_id}_{index:03d}"
    return {
        "candidate_id": candidate_id,
        "segment_id": segment.segment_id,
        "time_range": [float(start_sec), float(end_sec)],
        "start_sec": float(start_sec),
        "end_sec": float(end_sec),
        "source_beat_ids": _unique_nonempty(source_beat_ids),
        "entities": _unique_nonempty(entities),
        "verification_goal": str(verification_goal),
        "recommended_evidence_mode": str(recommended_evidence_mode or "multimodal"),
        "priority": int(priority),
        "status": "pending",
    }


def _candidate_windows_claim(segment_id: str, candidates: Sequence[Mapping[str, object]]) -> str:
    if not candidates:
        return f"IndexScout found no candidate windows for {segment_id}."
    parts = []
    for candidate in candidates:
        time_range = candidate.get("time_range") or [candidate.get("start_sec"), candidate.get("end_sec")]
        start_sec, end_sec = _time_range_argument(time_range)
        goal = str(candidate.get("verification_goal") or "").strip()
        parts.append(f"{candidate.get('candidate_id')} [{start_sec:.1f}-{end_sec:.1f}s] {goal}".strip())
    return f"IndexScout candidate windows for {segment_id}: " + "; ".join(parts)


def _preferred_evidence_mode(preferred_modalities: Sequence[str]) -> str:
    values = _unique_nonempty(preferred_modalities)
    if not values:
        return "multimodal"
    if len(values) == 1:
        return values[0]
    return "multimodal"


def _time_range_argument(value: Sequence[float] | Mapping[str, float] | None) -> tuple[float, float]:
    if isinstance(value, Mapping):
        start = value.get("start_sec", value.get("start"))
        end = value.get("end_sec", value.get("end"))
        if start is not None and end is not None:
            return float(start), float(end)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    raise ValueError("time_range must include start/end seconds")


def _verification_sampling_payload(
    sampling: Mapping[str, Any] | None,
    *,
    start_sec: float,
    end_sec: float,
) -> dict[str, object]:
    payload = dict(sampling or {})
    if "nframes" in payload:
        payload["nframes"] = min(VERIFY_WINDOW_MAX_FRAMES, max(1, int(payload["nframes"])))
        return payload
    max_frames = min(VERIFY_WINDOW_MAX_FRAMES, max(1, int(payload.get("max_frames") or VERIFY_WINDOW_MAX_FRAMES)))
    fps = float(payload.get("fps") or VERIFY_WINDOW_DEFAULT_FPS)
    payload["max_frames"] = max_frames
    payload["fps"] = fps
    if fps > 0:
        duration = max(0.1, float(end_sec) - float(start_sec))
        payload["nframes"] = max(1, min(max_frames, int(duration * fps + 0.999)))
    else:
        payload["nframes"] = max(1, max_frames)
    return payload


def _mapping_items(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _string_sequence(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _resolution(value: str) -> Any:
    text = str(value or "medium").strip()
    if text not in {"coarse", "medium", "dense"}:
        raise ValueError("read_segment_failed: resolution must be coarse, medium, or dense")
    return text


def _segment_from_scope(video_map: VideoMap, scope: Mapping[str, Any]) -> VideoMapSegment:
    segment_id = str(scope.get("segment_id") or "").strip()
    if segment_id:
        return video_map.get(segment_id)
    time_range = scope.get("time_range")
    if isinstance(time_range, Sequence) and not isinstance(time_range, (str, bytes)) and len(time_range) >= 2:
        start_sec = float(time_range[0])
        for segment in video_map.segments:
            if float(segment.start_sec) <= start_sec <= float(segment.end_sec):
                return segment
    if video_map.segments:
        return video_map.segments[0]
    raise ValueError("read_clip_failed: no segments are indexed")


def _root_segment_id(segment: VideoMapSegment) -> str:
    if segment.root_segment_id:
        return str(segment.root_segment_id)
    if segment.index_level == "root":
        return str(segment.segment_id)
    if segment.parent_segment_id:
        return str(segment.parent_segment_id)
    return str(segment.segment_id)


def _time_range_from_scope(scope: Mapping[str, Any], segment: VideoMapSegment) -> tuple[float, float]:
    time_range = scope.get("time_range")
    if isinstance(time_range, Sequence) and not isinstance(time_range, (str, bytes)) and len(time_range) >= 2:
        return float(time_range[0]), float(time_range[1])
    start_sec = scope.get("start_sec")
    end_sec = scope.get("end_sec")
    if start_sec is not None and end_sec is not None:
        return float(start_sec), float(end_sec)
    return float(segment.start_sec), float(segment.end_sec)


def _read_clip_prompt(*, segment_id: str, start_sec: float, end_sec: float, focus: str) -> str:
    return (
        "Read only factual visual/audio/OCR evidence from this clip. "
        "Do not choose an answer option. "
        f"Segment {segment_id} [{start_sec:.1f}, {end_sec:.1f}], focus: {focus}"
    )


def _normalize_verification_targets(targets: Sequence[Mapping[str, Any] | str] | None) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in _sequence_items(targets):
        if isinstance(item, Mapping):
            claim = str(item.get("claim") or item.get("question") or item.get("text") or "").strip()
            if not claim:
                continue
            target_id = str(item.get("id") or item.get("target_id") or item.get("name") or "").strip()
            polarity = str(item.get("polarity") or item.get("kind") or "presence").strip()
        else:
            claim = str(item or "").strip()
            if not claim:
                continue
            target_id = ""
            polarity = "presence"
        if not target_id:
            target_id = f"target_{len(normalized) + 1}"
        normalized.append({"id": target_id, "claim": claim, "polarity": polarity or "presence"})
    return normalized


def _format_verification_target_lines(targets: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = []
    for target in targets:
        target_id = str(target.get("id") or "").strip()
        claim = str(target.get("claim") or "").strip()
        polarity = str(target.get("polarity") or "presence").strip()
        if claim:
            lines.append(f"- {target_id}: {claim} (polarity={polarity})")
    return lines


def _source_kind_from_focus(focus: Sequence[str]) -> str:
    text = " ".join(focus).lower()
    if "asr" in text or "audio" in text or "narrat" in text:
        return "audio_fact"
    if "ocr" in text or "text" in text or "label" in text:
        return "ocr_fact"
    if "time" in text or "before" in text or "after" in text:
        return "temporal_fact"
    return "visual_fact"


def _source_kind_from_backend(raw_backend: Mapping[str, Any], focus: Sequence[str]) -> str:
    source_kind = str(raw_backend.get("source_kind") or "").strip()
    if source_kind:
        return source_kind
    facts = raw_backend.get("facts")
    if isinstance(facts, Sequence) and not isinstance(facts, (str, bytes)):
        for item in facts:
            if isinstance(item, Mapping):
                source_kind = str(item.get("source_kind") or "").strip()
                if source_kind:
                    return source_kind
    return _source_kind_from_focus(focus)


def _facts_from_backend_response(
    *,
    text: str,
    raw_backend: Mapping[str, Any],
    fallback_source_kind: str,
    time_range: Sequence[float],
) -> list[dict[str, Any]]:
    raw_facts = raw_backend.get("facts")
    facts: list[dict[str, Any]] = []
    if isinstance(raw_facts, Sequence) and not isinstance(raw_facts, (str, bytes)):
        for item in raw_facts:
            if not isinstance(item, Mapping):
                continue
            fact_text = str(item.get("text") or "").strip()
            if not fact_text:
                continue
            facts.append(
                {
                    "text": fact_text,
                    "source_kind": str(item.get("source_kind") or fallback_source_kind),
                    "confidence": float(item.get("confidence", 0.74) or 0.74),
                    "time_range": list(item.get("time_range") or time_range),
                    "frames_used": list(item.get("frames_used") or []),
                }
            )
    if not facts:
        for sentence in _split_fact_sentences(text):
            facts.append(
                {
                    "text": sentence,
                    "source_kind": fallback_source_kind,
                    "confidence": 0.74,
                    "time_range": list(time_range),
                    "frames_used": [],
                }
            )
    return facts


def _split_fact_sentences(text: str) -> list[str]:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return []
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]
    return parts or [normalized]


def _anchor_modality(source_kind: str) -> str:
    if source_kind == "audio_fact":
        return "asr"
    if source_kind == "ocr_fact":
        return "ocr"
    return "visual"


def _modalities_arg(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value if str(item))
    return ()


def _segment_matches_scope(segment: VideoMapSegment, scope: Mapping[str, Any]) -> bool:
    segment_id = str(scope.get("segment_id") or "").strip()
    if segment_id and segment.segment_id != segment_id:
        return False
    time_range = scope.get("time_range")
    if isinstance(time_range, Sequence) and not isinstance(time_range, (str, bytes)) and len(time_range) >= 2:
        start_sec = float(time_range[0])
        end_sec = float(time_range[1])
        return float(segment.end_sec) >= start_sec and float(segment.start_sec) <= end_sec
    return True


def _first_match(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    matches = payload.get("matches", [])
    if isinstance(matches, Sequence) and not isinstance(matches, (str, bytes)):
        for item in matches:
            if isinstance(item, Mapping):
                return item
    return {}


def _first_modality(fields: Sequence[str]) -> str:
    field = str(fields[0]) if fields else ""
    if field == "asr_text":
        return "asr"
    if field == "ocr_text":
        return "ocr"
    if field == "entities":
        return "entities"
    return "caption"


def _search_has_evidence(output: Mapping[str, Any]) -> bool:
    results = output.get("results") or ()
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        return False
    for row in results:
        if not isinstance(row, Mapping):
            continue
        excerpt = str(row.get("excerpt") or "").strip()
        segment_id = str(row.get("segment_id") or "").strip()
        if excerpt and excerpt != segment_id:
            return True
    return False


def _read_segment_verify_has_evidence(output: Mapping[str, Any]) -> bool:
    if str(output.get("mode") or "") != "verify":
        return False
    anchors = output.get("produced_anchors") or ()
    return isinstance(anchors, Sequence) and not isinstance(anchors, (str, bytes)) and bool(anchors)


def _verify_has_rejection_evidence(output: Mapping[str, Any]) -> bool:
    if bool(output.get("accepted")):
        return False
    reason = str(output.get("reason") or output.get("claim") or "").strip()
    citations = output.get("citations") or ()
    return bool(reason or citations)


def _row_matches_filter(row: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    for key, value in expected.items():
        if row.get(str(key)) != value:
            return False
    return True


def _sequence_items(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _unique_nonempty(values: Sequence[Any]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _verify_citations(
    workspace: Optional[EvidenceWorkspace],
    citations: Sequence[str],
    *,
    require_memory: bool,
    require_answer_support: bool = False,
) -> tuple[bool, str]:
    if workspace is None:
        return False, "workspace is required"
    memory_by_id = {entry.entry_id: entry for entry in workspace.memory_entries()}
    known_memory = set(memory_by_id)
    known_anchors = {str(anchor.get("anchor_id", "")) for anchor in workspace.read_pinned_anchors()}
    known_entities = {str(row.get("entity_id", "")) for row in workspace.read_workspace_section("entities")}
    known_events = {str(row.get("event_id", "")) for row in workspace.read_workspace_section("events")}
    known_relations = {str(row.get("relation_id", "")) for row in workspace.read_workspace_section("relations")}
    known_observations = {observation.observation_id for observation in workspace.read_observations()}
    known = known_memory | known_anchors | known_entities | known_events | known_relations | known_observations
    unknown = [citation for citation in citations if citation not in known]
    if unknown:
        return False, "unknown citation: " + ", ".join(unknown)
    cited_memory = [memory_by_id[citation] for citation in citations if citation in memory_by_id]
    if require_memory and not cited_memory:
        return False, "final answer requires at least one planner-authored memory citation"
    if require_answer_support:
        non_memory = [citation for citation in citations if citation not in memory_by_id]
        if non_memory:
            return False, "final answer citations must be memory ids: " + ", ".join(non_memory)
        unsupported = [entry for entry in cited_memory if entry.kind not in ANSWER_SUPPORTING_KINDS]
        if unsupported:
            details = ", ".join(f"{entry.entry_id}:{entry.kind}" for entry in unsupported)
            return False, (
                "unsupported final citation memory kind: "
                + details
                + "; allowed: "
                + "/".join(sorted(ANSWER_SUPPORTING_KINDS))
            )
    return True, ""


def _provenance_observation_ids(memory_by_id: Mapping[str, Any], memory_ids: Sequence[str]) -> list[str]:
    observation_ids: set[str] = set()
    for memory_id in memory_ids:
        entry = memory_by_id[memory_id]
        for anchor in entry.anchors:
            if anchor.observation_id:
                observation_ids.add(str(anchor.observation_id))
        metadata = dict(entry.metadata or {})
        disposition_observation_id = str(metadata.get("disposition_observation_id") or "").strip()
        if disposition_observation_id:
            observation_ids.add(disposition_observation_id)
        for item in _sequence_items(metadata.get("evidence_obs_ids")):
            obs_id = str(item).strip()
            if obs_id:
                observation_ids.add(obs_id)
    return sorted(observation_ids)


def _static_key(tool_name: str):
    def build(_ctx: Any, request: Any) -> str:
        return f"{tool_name}:{request.arguments}"

    return build
