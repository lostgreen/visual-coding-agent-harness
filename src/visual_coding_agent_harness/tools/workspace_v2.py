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
    _normalize_workspace_v2_answer,
    _normalize_workspace_v2_synthesize_memory,
)
from .workspace_primitives import build_workspace_primitives_registry


ANSWER_SUPPORTING_KINDS = frozenset(
    {"answer_support", "caption_support", "visual_support", "synthesized_support", "answer_conflict_resolved"}
)
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
    """Build the compact v2 registry: explore -> verify_window -> memory -> answer."""

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

    @tool(name="explore", description="Find candidate windows from the video index. Navigation only; never answer evidence.")
    def explore(
        query: str = "",
        targets: Sequence[Mapping[str, Any] | str] = (),
        scope: Mapping[str, Any] | None = None,
        modalities: str | Sequence[str] = (),
        top_k: int = 8,
        window_sec: float = 20.0,
        purpose: str = "candidate_discovery",
        original_question: str = "",
        answer_options: Mapping[str, Any] | Sequence[str] | None = None,
    ) -> Mapping[str, object]:
        return segment_read_service.explore(
            query=query,
            targets=targets,
            scope=scope or {},
            modalities=_modalities_arg(modalities),
            top_k=top_k,
            window_sec=window_sec,
            purpose=purpose,
            original_question=original_question,
            answer_options=answer_options or {},
        )

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
        candidate_key: str = "",
        candidate_id: str = "",
        source_observation_id: str = "",
        segment_id: str = "",
        time_range: Sequence[float] | Mapping[str, float] | None = None,
        evidence_mode: str = "multimodal",
        focus: Sequence[str] = (),
        checks: Sequence[Mapping[str, Any] | str] = (),
        verification_targets: Sequence[Mapping[str, Any] | str] = (),
        sampling: Mapping[str, Any] | None = None,
    ) -> Mapping[str, object]:
        return segment_read_service.verify_window(
            candidate_key=candidate_key,
            candidate_id=candidate_id,
            source_observation_id=source_observation_id,
            segment_id=segment_id,
            time_range=time_range,
            evidence_mode=evidence_mode,
            focus=focus,
            checks=checks,
            verification_targets=verification_targets,
            sampling=sampling,
        )

    @tool(name="read_workspace", description="Read a bounded section from the durable workspace.")
    def read_workspace(section: str, filter: Mapping[str, Any] | None = None) -> Mapping[str, object]:
        rows = workspace.read_workspace_section(section, filter=filter or {}) if workspace is not None else []
        return {
            "claim": f"Read {len(rows)} row(s) from workspace section {section}.",
            "confidence": 1.0,
            "items": rows,
            "regions": [{"section": section, "rows": rows, "observations": rows if section == "observations_by_id" else []}],
            "limitations": "Cheap workspace read; no video frames inspected.",
        }

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
                "source_tool": "synthesize_memory",
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

    # workspace_v2 intentionally exposes a small planner tool surface:
    # explore -> verify_window -> memory -> answer. Legacy read_clip/read_segment/
    # search/scan_segment/verify tools are not registered on the active planner path.
    registry.register(
        ToolRuntimeSpec(
            tool_spec=explore,
            argument_normalizer=_normalize_workspace_v2_explore,
            semantic_key_builder=_canonical_tool_key("explore"),
            duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
            commit_required_predicate=_explore_has_candidates,
        )
    )
    registry.register(
        ToolRuntimeSpec(
            tool_spec=verify_window,
            argument_normalizer=_normalize_workspace_v2_verify_window,
            semantic_key_builder=_canonical_tool_key("verify_window"),
            duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
            commit_required=True,
        )
    )
    registry.register(
        ToolRuntimeSpec(
            tool_spec=read_workspace,
            argument_normalizer=_normalize_workspace_v2_read_workspace,
            semantic_key_builder=_canonical_tool_key("read_workspace"),
            duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
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
        registry.extend(build_workspace_primitives_registry(workspace=workspace, include=("commit",)))
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

    def explore(
        self,
        *,
        query: str,
        targets: Sequence[Mapping[str, Any] | str],
        scope: Mapping[str, Any],
        modalities: Sequence[str],
        top_k: int,
        window_sec: float,
        purpose: str,
        original_question: str = "",
        answer_options: Mapping[str, Any] | Sequence[str] | None = None,
    ) -> Mapping[str, object]:
        current = self.video_map_store.current
        normalized_targets = _normalize_explore_targets(targets)
        original_question = str(original_question or "").strip()
        normalized_options = _normalize_answer_options(answer_options)
        scope_filter = dict(scope or {})
        top_k = max(1, min(16, int(top_k or 8)))
        source_observation_id = _predicted_source_observation_id(self.workspace)
        resolved_modalities = _resolve_search_modalities(_explore_search_modalities(modalities))
        search_query = str(query or "").strip() or " ".join(str(target.get("claim") or "") for target in normalized_targets)
        search_results = list(current.search(query=search_query, top_k=top_k, modalities=resolved_modalities))
        if scope_filter:
            search_results = [
                result
                for result in search_results
                if _segment_matches_scope(result.segment, scope_filter)
            ]
        segments = [result.segment for result in search_results]
        if not segments:
            segments = [
                segment
                for segment in current.segments
                if getattr(segment, "index_level", "root") == "root" and _segment_matches_scope(segment, scope_filter)
            ][:top_k]
        result_by_segment = {result.segment.segment_id: result for result in search_results}
        candidate_windows: list[dict[str, object]] = []
        candidate_anchors: list[dict[str, object]] = []
        for segment in segments:
            if len(candidate_windows) >= top_k:
                break
            result = result_by_segment.get(segment.segment_id)
            index = len(candidate_windows) + 1
            time_range = _explore_time_range(segment, window_sec=window_sec)
            candidate_id = f"cand_{index:04d}"
            candidate_key = f"{source_observation_id}:{candidate_id}"
            match = _first_match(result.to_dict()) if result is not None else {}
            matched_terms = [str(item) for item in _sequence_items(match.get("matched_terms")) if str(item)]
            source_modalities = _unique_nonempty(
                [
                    str(match.get("modality") or ""),
                    *([_first_modality(result.matched_fields)] if result is not None else []),
                    *resolved_modalities,
                ]
            )
            matched_target_ids = _matched_target_ids(segment, matched_terms=matched_terms, targets=normalized_targets)
            rationale = str(result.relevance_reason if result is not None else "").strip()
            if not rationale:
                rationale = "Scoped index window selected for verification."
            candidate = {
                "candidate_key": candidate_key,
                "candidate_id": candidate_id,
                "source_observation_id": source_observation_id,
                "segment_id": segment.segment_id,
                "time_range": time_range,
                "start_sec": time_range[0],
                "end_sec": time_range[1],
                "matched_targets": matched_target_ids,
                "matched_terms": matched_terms,
                "source_modalities": source_modalities,
                "source_beat_ids": _beat_ids_for_window(segment, time_range),
                "entities": list(segment.entities),
                "verification_goal": search_query or "Verify local facts in this window.",
                "recommended_evidence_mode": "multimodal",
                "rationale": rationale,
                "status": "pending_verification",
            }
            candidate_windows.append(candidate)
            candidate_anchors.append(
                {
                    "anchor_id": f"anch_explore_{source_observation_id}_{candidate_id}",
                    "observation_id": "__pending__",
                    "source_kind": "retrieval_hit",
                    "segment_id": segment.segment_id,
                    "start_sec": time_range[0],
                    "end_sec": time_range[1],
                    "field_path": "candidate_windows",
                    "excerpt": rationale,
                    "modality": source_modalities[0] if source_modalities else "index",
                }
            )
        caption_hits = _caption_hits_for_explore(search_results=search_results, segments=segments, top_k=top_k)
        reasoning = _run_explore_caption_reasoning(
            backend=self.backend,
            query=search_query,
            targets=normalized_targets,
            caption_hits=caption_hits,
            candidate_windows=candidate_windows,
            source_observation_id=source_observation_id,
            original_question=original_question,
            answer_options=normalized_options,
        )
        if reasoning is not None:
            mode = str(reasoning.get("mode") or "").strip()
            facts = _normalize_caption_facts(reasoning.get("facts"))
            answer_mapping = _normalize_answer_mapping(reasoning.get("answer_mapping"))
            query_analysis = _normalize_query_analysis(reasoning.get("query_analysis"), query=search_query, answer_options=normalized_options)
            question_condition = _normalize_question_condition(reasoning.get("question_condition"), original_question=original_question)
            condition_match = _normalize_condition_match(reasoning.get("condition_match"), default_matches=not bool(original_question))
            caption_anchors = _normalize_caption_anchors(
                reasoning.get("anchors"),
                facts=facts,
                source_observation_id=source_observation_id,
            )
            if mode in {"caption_fact", "mixed"} and (facts or caption_anchors or str(reasoning.get("claim") or "").strip()):
                candidate_payload = (
                    _normalize_candidate_windows(
                        reasoning.get("candidate_windows"),
                        fallback=candidate_windows,
                        source_observation_id=source_observation_id,
                    )
                    if mode == "mixed"
                    else _normalize_candidate_windows(
                        reasoning.get("candidate_windows"),
                        fallback=(),
                        source_observation_id=source_observation_id,
                    )
                )
                produced_anchors = [*caption_anchors, *candidate_anchors_for_windows(candidate_payload)]
                support_status = str(
                    reasoning.get("support_status")
                    or ("caption_supported" if mode == "caption_fact" else "partial_caption_supported")
                )
                claim = str(reasoning.get("claim") or _caption_fact_claim(facts) or "Explore found caption-level evidence.").strip()
                payload = {
                    "claim": claim,
                    "confidence": _bounded_confidence(reasoning.get("confidence"), default=0.72),
                    "mode": mode,
                    "support_status": support_status,
                    "needs_visual_verify": bool(reasoning.get("needs_visual_verify", mode == "mixed")),
                    "purpose": str(purpose or "caption_reasoning"),
                    "query": search_query,
                    "original_question": original_question,
                    "answer_options": normalized_options,
                    "targets": normalized_targets,
                    "scope": scope_filter,
                    "modalities_used": list(resolved_modalities),
                    "facts": facts,
                    "anchors": caption_anchors,
                    "query_analysis": query_analysis,
                    "question_condition": question_condition,
                    "condition_match": condition_match,
                    "answer_mapping": answer_mapping,
                    "candidate_windows": candidate_payload,
                    "regions": [*caption_anchors, *candidate_payload],
                    "produced_anchors": produced_anchors,
                    "notes": ["Caption-level evidence may be committed; candidate windows still require verify_window."],
                    "limitations": str(reasoning.get("limitations") or "Caption/index reasoning only; no new visual verification was performed."),
                    "raw": {"explore_reasoning": reasoning, "caption_hits": caption_hits},
                }
                return _validated_caption_explore_payload(payload)
        return {
            "claim": f"Explore found {len(candidate_windows)} candidate window(s) for verification.",
            "confidence": 0.85 if candidate_windows else 0.2,
            "support_status": "candidate_only",
            "cannot_final_cite": True,
            "mode": "candidate_discovery",
            "purpose": str(purpose or "candidate_discovery"),
            "query": search_query,
            "original_question": original_question,
            "answer_options": normalized_options,
            "targets": normalized_targets,
            "scope": scope_filter,
            "modalities_used": list(resolved_modalities),
            "candidate_windows": candidate_windows,
            "regions": candidate_windows,
            "facts": [],
            "anchors": [],
            "query_analysis": _normalize_query_analysis({}, query=search_query, answer_options=normalized_options),
            "question_condition": _normalize_question_condition({}, original_question=original_question),
            "condition_match": {"matches_original_question": False, "match_level": "unknown", "reason": "Candidate discovery requires verification."},
            "answer_mapping": {"supports_option": None, "opposes_options": [], "reason": None},
            "needs_visual_verify": True,
            "produced_anchors": candidate_anchors,
            "notes": ["Candidate windows are navigation only. Use verify_window before committing answer_support."],
            "limitations": "Explore results are navigation only and candidate_only; they cannot support final answers.",
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
        candidate_key: str,
        candidate_id: str,
        source_observation_id: str,
        segment_id: str,
        time_range: Sequence[float] | Mapping[str, float] | None,
        evidence_mode: str,
        focus: Sequence[str],
        checks: Sequence[Mapping[str, Any] | str] = (),
        verification_targets: Sequence[Mapping[str, Any] | str] = (),
        sampling: Mapping[str, Any] | None,
    ) -> Mapping[str, object]:
        candidate = self._resolve_candidate(
            candidate_key=candidate_key,
            candidate_id=candidate_id,
            source_observation_id=source_observation_id,
            segment_id=segment_id,
            time_range=time_range,
        )
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
            "candidate_key": str(candidate.get("candidate_key") or ""),
            "source_observation_id": str(candidate.get("source_observation_id") or source_observation_id or ""),
            "segment_id": resolved_segment_id,
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
        candidate_key: str,
        candidate_id: str,
        source_observation_id: str,
        segment_id: str,
        time_range: Sequence[float] | Mapping[str, float] | None,
    ) -> dict[str, object]:
        normalized_candidate_key = str(candidate_key or "").strip()
        normalized_candidate_id = str(candidate_id or "").strip()
        normalized_source_observation_id = str(source_observation_id or "").strip()
        if normalized_candidate_key and self.workspace is not None:
            for observation in self.workspace.read_observations():
                raw_output = observation.raw_output if isinstance(observation.raw_output, Mapping) else {}
                for candidate in _mapping_items(raw_output.get("candidate_windows")):
                    if str(candidate.get("candidate_key") or "") == normalized_candidate_key:
                        return dict(candidate)
            raise ValueError(f"verify_window_failed: candidate_key not found: {normalized_candidate_key}")
        if normalized_candidate_id and self.workspace is not None:
            matches = []
            for observation in self.workspace.read_observations():
                if normalized_source_observation_id and observation.observation_id != normalized_source_observation_id:
                    continue
                raw_output = observation.raw_output if isinstance(observation.raw_output, Mapping) else {}
                for candidate in _mapping_items(raw_output.get("candidate_windows")):
                    if str(candidate.get("candidate_id") or "") == normalized_candidate_id:
                        matches.append(dict(candidate))
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(
                    "verify_window_failed: candidate_id is ambiguous; provide candidate_key or source_observation_id"
                )
            if normalized_source_observation_id:
                raise ValueError(
                    f"verify_window_failed: candidate_id={normalized_candidate_id} not found in {normalized_source_observation_id}"
                )
        if not segment_id:
            raise ValueError(
                "verify_window_failed: candidate was not found; provide candidate_key, "
                "candidate_id+source_observation_id, or segment_id+time_range"
            )
        start_sec, end_sec = _time_range_argument(time_range)
        return {
            "candidate_id": normalized_candidate_id,
            "candidate_key": normalized_candidate_key,
            "source_observation_id": normalized_source_observation_id,
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
    verification_results = _verification_results_from_backend(
        targets=targets,
        raw_backend=raw_backend,
        facts=facts,
        produced_anchors=produced_anchors,
        segment_id=segment.segment_id,
        time_range=[start_sec, end_sec],
    )
    return {
        "claim": fact_text,
        "confidence": 0.74,
        "facts": facts,
        "verification_results": verification_results,
        "summary": _verification_summary(verification_results, fallback=fact_text),
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
        "raw": {**raw_backend, "verification_targets": targets, "verification_results": verification_results},
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


def _predicted_source_observation_id(workspace: EvidenceWorkspace | None) -> str:
    if workspace is None:
        return "obs_pending"
    return workspace._next_observation_id()  # noqa: SLF001 - candidate keys must match the imminent observation id.


def _caption_hits_for_explore(
    *,
    search_results: Sequence[Any],
    segments: Sequence[VideoMapSegment],
    top_k: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    result_by_segment = {result.segment.segment_id: result for result in search_results}
    for segment in segments[: max(1, int(top_k or 8))]:
        result = result_by_segment.get(segment.segment_id)
        for source_kind, text in (
            ("dense_caption", segment.low_fps_caption),
            ("asr", segment.asr_text),
            ("ocr", segment.ocr_text),
            ("index_summary", " ".join(beat.summary for beat in segment.timeline_beats if getattr(beat, "summary", ""))),
        ):
            excerpt = str(text or "").strip()
            if not excerpt:
                continue
            key = (segment.segment_id, source_kind, excerpt)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "source_kind": source_kind,
                    "segment_id": segment.segment_id,
                    "time_range": [float(segment.start_sec), float(segment.end_sec)],
                    "excerpt": excerpt,
                    "score": float(getattr(result, "score", 0.0) or 0.0) if result is not None else 0.0,
                    "matched_fields": list(getattr(result, "matched_fields", ()) or ()),
                }
            )
    return rows


def _run_explore_caption_reasoning(
    *,
    backend: VisionLanguageBackend,
    query: str,
    targets: Sequence[Mapping[str, object]],
    caption_hits: Sequence[Mapping[str, object]],
    candidate_windows: Sequence[Mapping[str, object]],
    source_observation_id: str,
    original_question: str = "",
    answer_options: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    if not caption_hits:
        return None
    prompt = _explore_caption_reasoning_prompt(
        query=query,
        targets=targets,
        caption_hits=caption_hits,
        candidate_windows=candidate_windows,
        original_question=original_question,
        answer_options=answer_options or {},
    )
    response = backend.generate(
        BackendRequest(
            task="explore_caption_reasoning",
            prompt=prompt,
            max_new_tokens=1200,
            temperature=0.0,
            metadata={
                "tool": "explore",
                "source_observation_id": source_observation_id,
                "caption_hit_count": len(caption_hits),
                "candidate_window_count": len(candidate_windows),
            },
        )
    )
    payload: dict[str, Any]
    if isinstance(response.raw, Mapping) and response.raw.get("mode"):
        payload = dict(response.raw)
    else:
        try:
            payload = _parse_json_object(response.text)
        except ValueError:
            return None
    mode = str(payload.get("mode") or "").strip()
    if mode not in {"caption_fact", "candidate_discovery", "mixed"}:
        return None
    return payload


def _explore_caption_reasoning_prompt(
    *,
    query: str,
    targets: Sequence[Mapping[str, object]],
    caption_hits: Sequence[Mapping[str, object]],
    candidate_windows: Sequence[Mapping[str, object]],
    original_question: str = "",
    answer_options: Mapping[str, str] | None = None,
) -> str:
    return (
        "You are the explore subagent. Use only dense captions, ASR, OCR, and index summaries below. "
        "Do not use outside knowledge and do not inspect video frames. "
        "Decide whether the original question can be answered at caption/index level, or whether candidate windows need verify_window.\n"
        "The planner query is only a retrieval hint. The original question defines what counts as answer evidence.\n"
        "Before returning caption_fact, check whether the fact directly answers the original question condition, "
        "whether it is merely related background, and whether it belongs to a later or earlier event than the one asked.\n"
        "For multiple-choice questions, determine whether the planner query is option-biased. "
        "Do not return caption_supported merely because a caption matches one option; it must also satisfy the original question condition.\n"
        "Return exactly one JSON object with keys: mode, support_status, claim, confidence, facts, anchors, "
        "candidate_windows, query_analysis, question_condition, condition_match, answer_mapping, needs_visual_verify, cannot_final_cite, limitations.\n"
        "Allowed mode values: caption_fact, candidate_discovery, mixed. "
        "Allowed support_status values: caption_supported, partial_caption_supported, candidate_only, uncertain, not_found.\n\n"
        f"Original question: {original_question}\n"
        "Answer options:\n"
        + json.dumps(dict(answer_options or {}), ensure_ascii=False)
        + "\n"
        f"Planner query: {query}\n"
        "Targets:\n"
        + json.dumps(list(targets), ensure_ascii=False)
        + "\nCaption/ASR/OCR/index hits:\n"
        + json.dumps(list(caption_hits), ensure_ascii=False)
        + "\nCandidate windows:\n"
        + json.dumps(list(candidate_windows), ensure_ascii=False)
    )


def _normalize_caption_facts(value: Any) -> list[dict[str, object]]:
    facts: list[dict[str, object]] = []
    for item in _mapping_items(value):
        claim = str(item.get("claim") or item.get("text") or "").strip()
        if not claim:
            continue
        time_range = _optional_time_range(item.get("time_range"))
        facts.append(
            {
                "claim": claim,
                "confidence": _bounded_confidence(item.get("confidence"), default=0.7),
                "source_kind": _caption_source_kind(item.get("source_kind")),
                "segment_id": str(item.get("segment_id") or "").strip() or None,
                "time_range": time_range,
                "excerpt": str(item.get("excerpt") or claim).strip(),
                "supports_option": str(item.get("supports_option") or "").strip() or None,
                "opposes_options": [str(option).strip() for option in _sequence_items(item.get("opposes_options")) if str(option).strip()],
            }
        )
    return facts


def _normalize_caption_anchors(
    value: Any,
    *,
    facts: Sequence[Mapping[str, object]],
    source_observation_id: str,
) -> list[dict[str, object]]:
    raw_anchors = _mapping_items(value)
    if not raw_anchors:
        raw_anchors = [
            {
                "source_kind": fact.get("source_kind"),
                "segment_id": fact.get("segment_id"),
                "time_range": fact.get("time_range"),
                "excerpt": fact.get("excerpt") or fact.get("claim"),
                "reliability": "medium",
            }
            for fact in facts
            if fact.get("segment_id") and fact.get("time_range")
        ]
    anchors: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_anchors, start=1):
        segment_id = str(item.get("segment_id") or "").strip()
        time_range = _optional_time_range(item.get("time_range"))
        excerpt = str(item.get("excerpt") or "").strip()
        if not segment_id or time_range is None or not excerpt:
            continue
        anchor_id = str(item.get("anchor_id") or f"anch_caption_{source_observation_id}_{index:03d}").strip()
        if not anchor_id or anchor_id in seen:
            continue
        seen.add(anchor_id)
        source_kind = _caption_source_kind(item.get("source_kind"))
        anchors.append(
            {
                "anchor_id": anchor_id,
                "observation_id": "__pending__",
                "source_kind": source_kind,
                "kind": source_kind,
                "segment_id": segment_id,
                "start_sec": float(time_range[0]),
                "end_sec": float(time_range[1]),
                "time_range": [float(time_range[0]), float(time_range[1])],
                "field_path": "facts",
                "excerpt": excerpt,
                "reliability": _caption_reliability(item.get("reliability")),
                "modality": source_kind,
            }
        )
    return anchors


def _normalize_candidate_windows(
    value: Any,
    *,
    fallback: Sequence[Mapping[str, object]],
    source_observation_id: str,
) -> list[dict[str, object]]:
    raw_candidates = _mapping_items(value) or [dict(item) for item in fallback]
    candidates: list[dict[str, object]] = []
    for index, item in enumerate(raw_candidates, start=1):
        candidate = dict(item)
        candidate_id = str(candidate.get("candidate_id") or f"cand_{index:04d}").strip()
        candidate["candidate_id"] = candidate_id
        candidate.setdefault("source_observation_id", source_observation_id)
        candidate.setdefault("candidate_key", f"{source_observation_id}:{candidate_id}")
        if "time_range" not in candidate:
            start_sec = candidate.get("start_sec")
            end_sec = candidate.get("end_sec")
            if start_sec is not None and end_sec is not None:
                candidate["time_range"] = [float(start_sec), float(end_sec)]
        if "time_range" in candidate:
            time_range = _time_range_argument(candidate["time_range"])  # type: ignore[arg-type]
            candidate["time_range"] = [float(time_range[0]), float(time_range[1])]
            candidate.setdefault("start_sec", float(time_range[0]))
            candidate.setdefault("end_sec", float(time_range[1]))
        candidate.setdefault("status", "pending_verification")
        candidates.append(candidate)
    return candidates


def candidate_anchors_for_windows(candidate_windows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    anchors: list[dict[str, object]] = []
    for candidate in candidate_windows:
        candidate_key = str(candidate.get("candidate_key") or "")
        candidate_id = str(candidate.get("candidate_id") or "")
        source_observation_id = str(candidate.get("source_observation_id") or "obs_pending")
        time_range = candidate.get("time_range") or [candidate.get("start_sec"), candidate.get("end_sec")]
        try:
            start_sec, end_sec = _time_range_argument(time_range)  # type: ignore[arg-type]
        except ValueError:
            continue
        excerpt = str(candidate.get("rationale") or candidate.get("verification_goal") or "Candidate window requires verification.").strip()
        anchors.append(
            {
                "anchor_id": f"anch_explore_{source_observation_id}_{candidate_id or len(anchors) + 1:>04}",
                "observation_id": "__pending__",
                "source_kind": "retrieval_hit",
                "segment_id": str(candidate.get("segment_id") or ""),
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "field_path": "candidate_windows",
                "excerpt": excerpt,
                "modality": "index",
                "candidate_key": candidate_key,
            }
        )
    return anchors


def _caption_fact_claim(facts: Sequence[Mapping[str, object]]) -> str:
    if not facts:
        return ""
    return "; ".join(str(fact.get("claim") or "").strip() for fact in facts if str(fact.get("claim") or "").strip())


def _normalize_answer_mapping(value: Any) -> dict[str, object]:
    payload = dict(value or {}) if isinstance(value, Mapping) else {}
    return {
        "supports_option": str(payload.get("supports_option") or "").strip() or None,
        "neutral_options": [str(option).strip() for option in _sequence_items(payload.get("neutral_options")) if str(option).strip()],
        "opposes_options": [str(option).strip() for option in _sequence_items(payload.get("opposes_options")) if str(option).strip()],
        "reason": str(payload.get("reason") or "").strip() or None,
    }


def _normalize_answer_options(value: Mapping[str, Any] | Sequence[str] | None) -> dict[str, str]:
    if isinstance(value, Mapping):
        return {str(key).strip().upper()[:1]: str(item).strip() for key, item in value.items() if str(key).strip() and str(item).strip()}
    options: dict[str, str] = {}
    for index, item in enumerate(_sequence_items(value), start=1):
        text = str(item or "").strip()
        if not text:
            continue
        if len(text) >= 2 and text[0].upper() in "ABCDEFGH" and text[1] in {".", ")"}:
            options[text[0].upper()] = text[2:].strip()
        else:
            options[chr(ord("A") + len(options))] = text
    return options


def _normalize_query_analysis(value: Any, *, query: str, answer_options: Mapping[str, str]) -> dict[str, object]:
    payload = dict(value or {}) if isinstance(value, Mapping) else {}
    biased = payload.get("is_option_biased")
    biased_option = str(payload.get("biased_toward_option") or "").strip().upper()[:1] or None
    if biased is None:
        biased_option = _query_biased_toward_option(query, answer_options)
        biased = bool(biased_option)
    return {
        "is_option_biased": bool(biased),
        "biased_toward_option": biased_option,
        "reason": str(payload.get("reason") or ("Query overlaps mostly with one answer option." if biased else "No strong option-only bias detected.")).strip() or None,
    }


def _normalize_question_condition(value: Any, *, original_question: str) -> dict[str, object]:
    payload = dict(value or {}) if isinstance(value, Mapping) else {}
    return {
        "condition_text": str(payload.get("condition_text") or _question_condition_text(original_question) or "").strip() or None,
        "condition_type": str(payload.get("condition_type") or _condition_type(original_question) or "").strip() or None,
        "required_focus": str(payload.get("required_focus") or "").strip() or None,
    }


def _normalize_condition_match(value: Any, *, default_matches: bool = False) -> dict[str, object]:
    payload = dict(value or {}) if isinstance(value, Mapping) else {}
    match_level = str(payload.get("match_level") or "unknown").strip()
    if match_level not in {"direct", "related", "related_but_wrong_scope", "contradiction", "unknown"}:
        match_level = "unknown"
    if default_matches and not payload:
        match_level = "direct"
    return {
        "matches_original_question": bool(payload.get("matches_original_question")) if "matches_original_question" in payload else bool(default_matches),
        "match_level": match_level,
        "reason": str(payload.get("reason") or "").strip() or None,
    }


def _validated_caption_explore_payload(payload: Mapping[str, Any]) -> dict[str, object]:
    result = dict(payload)
    facts = [dict(item) for item in _mapping_items(result.get("facts"))]
    anchors = [dict(item) for item in _mapping_items(result.get("anchors"))]
    produced_anchors = [dict(item) for item in _mapping_items(result.get("produced_anchors"))]
    answer_mapping = _normalize_answer_mapping(result.get("answer_mapping"))
    condition_match = _normalize_condition_match(result.get("condition_match"))
    query_analysis = _normalize_query_analysis(
        result.get("query_analysis"),
        query=str(result.get("query") or ""),
        answer_options=_normalize_answer_options(result.get("answer_options") if isinstance(result.get("answer_options"), Mapping) else {}),
    )
    original_question = str(result.get("original_question") or "").strip()
    task_type = _task_type_for_question(original_question or str(result.get("query") or ""))
    visual_required = _task_requires_visual_verification(task_type)
    grounded = _caption_fact_is_grounded(result, facts=facts, anchors=anchors, produced_anchors=produced_anchors)
    condition_ok = bool(condition_match.get("matches_original_question")) and str(condition_match.get("match_level") or "") == "direct"

    if not grounded:
        result.update(
            {
                "mode": "candidate_discovery",
                "support_status": "uncertain",
                "cannot_final_cite": True,
                "needs_visual_verify": True,
                "facts": [],
                "anchors": [],
                "produced_anchors": candidate_anchors_for_windows(_mapping_items(result.get("candidate_windows"))),
                "answer_mapping": _clear_answer_support(answer_mapping, reason="Caption fact downgraded because it lacks grounded facts/anchors."),
                "condition_match": condition_match,
                "query_analysis": query_analysis,
                "task_type": task_type,
                "caption_fact_downgraded": True,
            }
        )
        return result

    if not condition_ok or visual_required:
        result["mode"] = "mixed"
        result["support_status"] = "partial_caption_supported"
        result["cannot_final_cite"] = True
        result["needs_visual_verify"] = True
        answer_mapping = _clear_answer_support(
            answer_mapping,
            reason=(
                "Option match removed because evidence does not satisfy original question condition."
                if not condition_ok
                else f"Task type {task_type} requires visual verification."
            ),
        )
        for fact in facts:
            fact["supports_option"] = None
        result["facts"] = facts
        result["caption_fact_downgraded"] = True
    else:
        result["cannot_final_cite"] = False
        result["needs_visual_verify"] = bool(result.get("needs_visual_verify", False))
        result["caption_fact_downgraded"] = False

    if query_analysis.get("is_option_biased") and not condition_ok:
        answer_mapping = _clear_answer_support(answer_mapping, reason="Option-biased query cannot support an answer without direct condition match.")
        result["cannot_final_cite"] = True
        result["needs_visual_verify"] = True
    if query_analysis.get("is_option_biased"):
        result["confidence"] = min(float(result.get("confidence") or 0.0), 0.74)
    if answer_mapping.get("supports_option") and not condition_ok:
        answer_mapping = _clear_answer_support(answer_mapping, reason="Option match removed because evidence does not satisfy original question condition.")
    result["answer_mapping"] = answer_mapping
    result["condition_match"] = condition_match
    result["query_analysis"] = query_analysis
    result["task_type"] = task_type
    return result


def _caption_fact_is_grounded(
    payload: Mapping[str, Any],
    *,
    facts: Sequence[Mapping[str, Any]],
    anchors: Sequence[Mapping[str, Any]],
    produced_anchors: Sequence[Mapping[str, Any]],
) -> bool:
    claim = str(payload.get("claim") or "").strip()
    query = str(payload.get("query") or "").strip()
    if claim and query and _norm_text(claim) == _norm_text(query):
        return False
    if not facts and not anchors and not produced_anchors:
        return False
    for item in [*facts, *anchors, *produced_anchors]:
        if str(item.get("excerpt") or item.get("claim") or item.get("text") or "").strip():
            return True
        if item.get("time_range") is not None or item.get("segment_id") is not None:
            return True
    return False


def _clear_answer_support(answer_mapping: Mapping[str, object], *, reason: str) -> dict[str, object]:
    mapping = dict(answer_mapping)
    existing_reason = str(mapping.get("reason") or "").strip()
    mapping["supports_option"] = None
    mapping["reason"] = (existing_reason + " " + reason).strip() if existing_reason else reason
    return mapping


def _query_biased_toward_option(query: str, answer_options: Mapping[str, str]) -> str | None:
    query_tokens = set(_word_tokens(query))
    if not query_tokens or not answer_options:
        return None
    scores: list[tuple[str, int]] = []
    for option, text in answer_options.items():
        tokens = set(_word_tokens(text))
        scores.append((option, len(query_tokens & tokens)))
    scores.sort(key=lambda item: item[1], reverse=True)
    if not scores or scores[0][1] < 2:
        return None
    runner_up = scores[1][1] if len(scores) > 1 else 0
    if scores[0][1] >= runner_up + 2:
        return scores[0][0]
    return None


def _question_condition_text(question: str) -> str:
    lines = str(question or "").splitlines()
    text = lines[0].strip() if lines else ""
    if not text:
        return ""
    lower = text.lower()
    for marker in ("when ", "after ", "before ", "during ", "shown as ", "how many ", "what object ", "which action ", "what event "):
        index = lower.find(marker)
        if index >= 0:
            return text[index:].rstrip(" ?")
    return text.rstrip(" ?")


def _condition_type(question: str) -> str:
    lower = str(question or "").lower()
    if any(term in lower for term in ("how many", "number of", "count")):
        return "counting"
    if any(term in lower for term in ("when", "after", "before", "during", "first", "last")):
        return "temporal_event"
    if any(term in lower for term in ("where", "left", "right", "above", "below", "between")):
        return "spatial_relation"
    return "narrative_fact"


def _task_type_for_question(question: str) -> str:
    lower = str(question or "").lower()
    if any(term in lower for term in ("how many", "number of", "count")):
        return "counting"
    if any(term in lower for term in ("order", "sequence", "first", "last", "before", "after")):
        return "ordering"
    if any(term in lower for term in ("visible", "appears", "object", "which item")):
        return "object_presence"
    if any(term in lower for term in ("where", "left", "right", "above", "below", "next to")):
        return "spatial_relation"
    if any(term in lower for term in ("score", "scoreboard", "ui", "screen text")):
        return "scoreboard"
    if any(term in lower for term in ("read", "text", "ocr", "word")):
        return "visual_text"
    if any(term in lower for term in ("action", "gesture", "move", "opens", "closes")):
        return "fine_action"
    return "mcq_narrative_fact" if _normalize_answer_options_from_question(question) else "narrative_fact"


def _task_requires_visual_verification(task_type: str) -> bool:
    return task_type in {"counting", "object_presence", "spatial_relation", "scoreboard", "fine_action", "visual_text"}


def _normalize_answer_options_from_question(question: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for line in str(question or "").splitlines():
        stripped = line.strip()
        if len(stripped) >= 3 and stripped[0].upper() in "ABCDEFGH" and stripped[1] in {".", ")"}:
            options[stripped[0].upper()] = stripped[2:].strip()
    return options


def _word_tokens(text: str) -> list[str]:
    return [token for token in _norm_text(text).split() if len(token) > 2]


def _norm_text(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in str(text or "")).split())


def _caption_source_kind(value: Any) -> str:
    text = str(value or "dense_caption").strip()
    if text in {"dense_caption", "asr", "ocr", "index_summary"}:
        return text
    return "dense_caption"


def _caption_reliability(value: Any) -> str:
    text = str(value or "medium").strip()
    if text in {"high", "medium", "low"}:
        return text
    return "medium"


def _bounded_confidence(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    return max(0.0, min(1.0, number))


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y"}


def _normalize_explore_targets(targets: Sequence[Mapping[str, Any] | str] | None) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for item in _sequence_items(targets):
        if isinstance(item, Mapping):
            claim = str(item.get("claim") or item.get("question") or item.get("text") or "").strip()
            aliases = [str(alias).strip() for alias in _sequence_items(item.get("aliases")) if str(alias).strip()]
            target_id = str(item.get("target_id") or item.get("id") or item.get("name") or "").strip()
        else:
            claim = str(item or "").strip()
            aliases = []
            target_id = ""
        if not claim and not aliases:
            continue
        if not target_id:
            target_id = f"target_{len(normalized) + 1}"
        normalized.append({"target_id": target_id, "claim": claim, "aliases": aliases})
    return normalized


def _explore_search_modalities(modalities: Sequence[str]) -> tuple[str, ...]:
    values = [str(item).strip().lower() for item in modalities if str(item).strip()]
    if not values:
        return ()
    mapped = []
    for value in values:
        if value == "index":
            mapped.extend(["caption", "entities"])
        else:
            mapped.append(value)
    return tuple(mapped)


def _explore_time_range(segment: VideoMapSegment, *, window_sec: float) -> list[float]:
    beats = list(segment.timeline_beats or ())
    if beats:
        beat = beats[0]
        return [float(getattr(beat, "start_sec", segment.start_sec)), float(getattr(beat, "end_sec", segment.end_sec))]
    duration = max(0.1, float(segment.end_sec) - float(segment.start_sec))
    width = min(duration, max(0.1, float(window_sec or duration)))
    return [float(segment.start_sec), float(segment.start_sec) + width]


def _beat_ids_for_window(segment: VideoMapSegment, time_range: Sequence[float]) -> list[str]:
    start_sec, end_sec = _time_range_argument(time_range)
    beat_ids = []
    for beat in segment.timeline_beats or ():
        beat_start = float(getattr(beat, "start_sec", segment.start_sec))
        beat_end = float(getattr(beat, "end_sec", segment.end_sec))
        if beat_end >= start_sec and beat_start <= end_sec:
            beat_ids.append(str(getattr(beat, "beat_id", "")))
    return _unique_nonempty(beat_ids)


def _matched_target_ids(
    segment: VideoMapSegment,
    *,
    matched_terms: Sequence[str],
    targets: Sequence[Mapping[str, object]],
) -> list[str]:
    if not targets:
        return []
    haystack = " ".join([segment.compact_text(), " ".join(matched_terms)]).lower()
    matched = []
    for target in targets:
        target_id = str(target.get("target_id") or "").strip()
        terms = [str(target.get("claim") or ""), *[str(item) for item in _sequence_items(target.get("aliases"))]]
        if any(term.strip().lower() and term.strip().lower() in haystack for term in terms):
            matched.append(target_id)
    return _unique_nonempty(matched)


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


def _normalize_verification_targets(targets: Sequence[Mapping[str, Any] | str] | None) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for item in _sequence_items(targets):
        if isinstance(item, Mapping):
            claim = str(item.get("claim") or item.get("question") or item.get("text") or "").strip()
            if not claim:
                continue
            target_id = str(item.get("target_id") or item.get("id") or item.get("target_ref") or item.get("name") or "").strip()
            polarity = str(item.get("polarity") or item.get("kind") or "presence").strip()
            expected_evidence = [
                str(value).strip()
                for value in _sequence_items(
                    item.get("expected_evidence") or item.get("evidence_modes") or item.get("modalities")
                )
                if str(value).strip()
            ]
        else:
            claim = str(item or "").strip()
            if not claim:
                continue
            target_id = ""
            polarity = "presence"
            expected_evidence = []
        if not target_id:
            target_id = f"target_{len(normalized) + 1}"
        normalized.append(
            {
                "target_id": target_id,
                "claim": claim,
                "polarity": polarity or "presence",
                "expected_evidence": expected_evidence,
            }
        )
    return normalized


def _format_verification_target_lines(targets: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = []
    for target in targets:
        target_id = str(target.get("target_id") or target.get("id") or "").strip()
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


def _verification_results_from_backend(
    *,
    targets: Sequence[Mapping[str, Any]],
    raw_backend: Mapping[str, Any],
    facts: Sequence[Mapping[str, Any]],
    produced_anchors: Sequence[Mapping[str, Any]],
    segment_id: str,
    time_range: Sequence[float],
) -> list[dict[str, Any]]:
    if not targets:
        return []
    scope = {"segment_id": str(segment_id), "time_range": [float(time_range[0]), float(time_range[1])]}
    raw_results = raw_backend.get("verification_results")
    if isinstance(raw_results, Sequence) and not isinstance(raw_results, (str, bytes)):
        normalized = []
        for index, item in enumerate(raw_results):
            if not isinstance(item, Mapping):
                continue
            target = targets[index] if index < len(targets) else {}
            normalized.append(
                _verification_result(
                    target_id=str(item.get("target_id") or item.get("id") or target.get("target_id") or "unknown"),
                    claim=str(item.get("claim") or target.get("claim") or ""),
                    verdict=str(item.get("verdict") or "uncertain"),
                    scope=dict(item.get("scope", {}) or scope) if isinstance(item.get("scope", {}), Mapping) else scope,
                    anchor_ids=[str(value) for value in _sequence_items(item.get("anchor_ids")) if str(value)],
                    source_kind=str(item.get("source_kind") or "multimodal_fact"),
                    confidence=float(item.get("confidence", 0.5) or 0.5),
                    rationale=str(item.get("rationale") or ""),
                )
            )
        if normalized:
            return normalized

    raw_facts = raw_backend.get("facts")
    if (
        isinstance(raw_facts, Sequence)
        and not isinstance(raw_facts, (str, bytes))
        and len(raw_facts) == len(targets)
    ):
        aligned = []
        for index, target in enumerate(targets):
            fact = facts[index] if index < len(facts) else {}
            anchor = produced_anchors[index] if index < len(produced_anchors) else {}
            aligned.append(
                _verification_result(
                    target_id=str(target.get("target_id") or f"target_{index + 1}"),
                    claim=str(target.get("claim") or ""),
                    verdict=str(fact.get("verdict") or "supported"),
                    scope=scope,
                    anchor_ids=[str(anchor.get("anchor_id"))] if anchor.get("anchor_id") else [],
                    source_kind=str(fact.get("source_kind") or "multimodal_fact"),
                    confidence=float(fact.get("confidence", 0.74) or 0.74),
                    rationale=str(fact.get("text") or ""),
                )
            )
        return aligned

    return [
        _verification_result(
            target_id=str(target.get("target_id") or f"target_{index + 1}"),
            claim=str(target.get("claim") or ""),
            verdict="uncertain",
            scope=scope,
            anchor_ids=[],
            source_kind="multimodal_fact",
            confidence=0.5,
            rationale="Backend output was not structured enough to assign this check a supported verdict.",
        )
        for index, target in enumerate(targets)
    ]


def _verification_result(
    *,
    target_id: str,
    claim: str,
    verdict: str,
    scope: Mapping[str, Any],
    anchor_ids: Sequence[str],
    source_kind: str,
    confidence: float,
    rationale: str,
) -> dict[str, Any]:
    normalized_verdict = str(verdict or "uncertain").strip()
    if normalized_verdict not in {"supported", "contradicted", "not_found_in_window", "uncertain"}:
        normalized_verdict = "uncertain"
    return {
        "target_id": str(target_id or "unknown"),
        "claim": str(claim or ""),
        "verdict": normalized_verdict,
        "scope": dict(scope),
        "anchor_ids": [str(anchor_id) for anchor_id in anchor_ids if str(anchor_id)],
        "source_kind": str(source_kind or "multimodal_fact"),
        "confidence": float(confidence),
        "rationale": str(rationale or ""),
    }


def _verification_summary(results: Sequence[Mapping[str, Any]], *, fallback: str) -> str:
    if not results:
        return str(fallback or "")
    counts: dict[str, int] = {}
    for result in results:
        verdict = str(result.get("verdict") or "uncertain")
        counts[verdict] = counts.get(verdict, 0) + 1
    return ", ".join(f"{verdict}={count}" for verdict, count in sorted(counts.items()))


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
    segment_ids = {str(item).strip() for item in _sequence_items(scope.get("segment_ids")) if str(item).strip()}
    if segment_ids and segment.segment_id not in segment_ids:
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


def _explore_has_candidates(output: Mapping[str, Any]) -> bool:
    if str(output.get("mode") or "") in {"caption_fact", "mixed"}:
        facts = output.get("facts") or ()
        anchors = output.get("anchors") or output.get("produced_anchors") or ()
        if isinstance(facts, Sequence) and not isinstance(facts, (str, bytes)) and bool(facts):
            return True
        if isinstance(anchors, Sequence) and not isinstance(anchors, (str, bytes)) and bool(anchors):
            return True
    candidates = output.get("candidate_windows") or ()
    return isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)) and bool(candidates)


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


def _normalize_workspace_v2_explore(_ctx: Any, request: Any) -> Mapping[str, Any]:
    args = dict(request.arguments)
    scope = dict(args.get("scope", {}) or {}) if isinstance(args.get("scope"), Mapping) else {}
    segment_ids = args.get("segment_ids")
    if segment_ids is None:
        segment_ids = args.get("segments")
    if segment_ids is None and args.get("segment_id") is not None:
        segment_ids = [args.get("segment_id")]
    if segment_ids is not None:
        scope["segment_ids"] = [str(item).strip() for item in _sequence_items(segment_ids) if str(item).strip()]
    time_range = _optional_time_range(args.get("time_range") or args.get("range"))
    if time_range is not None:
        scope["time_range"] = time_range
    modalities = args.get("modalities")
    if modalities is None:
        modalities = args.get("modality")
    if modalities is None:
        modalities = args.get("evidence_modes")
    return {
        "query": str(args.get("query") or args.get("focus") or args.get("question") or ""),
        "targets": list(_sequence_items(args.get("targets") or args.get("checks") or args.get("facts"))),
        "scope": scope,
        "modalities": _modalities_arg(modalities),
        "top_k": int(args.get("top_k") or args.get("max_candidates") or 8),
        "window_sec": float(args.get("window_sec") or args.get("window_seconds") or 20.0),
        "purpose": str(args.get("purpose") or "candidate_discovery"),
        "original_question": str(args.get("original_question") or ""),
        "answer_options": args.get("answer_options") or args.get("options") or {},
    }


def _normalize_workspace_v2_verify_window(_ctx: Any, request: Any) -> Mapping[str, Any]:
    args = dict(request.arguments)
    candidate_value = str(args.get("candidate") or "").strip()
    candidate_key = str(args.get("candidate_key") or "").strip()
    candidate_id = str(args.get("candidate_id") or "").strip()
    if candidate_value:
        if ":" in candidate_value and not candidate_key:
            candidate_key = candidate_value
        elif not candidate_id:
            candidate_id = candidate_value
    checks = args.get("checks")
    if checks is None:
        checks = args.get("verification_targets")
    if checks is None:
        checks = args.get("targets")
    if checks is None:
        checks = args.get("facts")
    modalities = args.get("evidence_mode")
    if modalities is None:
        modalities = args.get("modalities")
    if modalities is None:
        modalities = args.get("evidence_modes")
    evidence_values = _modalities_arg(modalities)
    sampling = dict(args.get("sampling", {}) or {}) if isinstance(args.get("sampling"), Mapping) else {}
    if args.get("max_frames") is not None:
        sampling["max_frames"] = int(args["max_frames"])
    if args.get("nframes") is not None:
        sampling["nframes"] = int(args["nframes"])
    return {
        "candidate_key": candidate_key,
        "candidate_id": candidate_id,
        "source_observation_id": str(args.get("source_observation_id") or args.get("obs_id") or ""),
        "segment_id": str(args.get("segment_id") or args.get("segment") or ""),
        "time_range": _optional_time_range(args.get("time_range") or args.get("range")),
        "evidence_mode": ",".join(evidence_values) if evidence_values else "multimodal",
        "focus": tuple(str(item).strip() for item in _sequence_items(args.get("focus")) if str(item).strip()),
        "checks": _normalize_verification_targets(checks),
        "sampling": sampling,
    }


def _normalize_workspace_v2_read_workspace(_ctx: Any, request: Any) -> Mapping[str, Any]:
    args = dict(request.arguments)
    return {
        "section": str(args.get("section") or args.get("kind") or ""),
        "filter": dict(args.get("filter", {}) or {}) if isinstance(args.get("filter"), Mapping) else {},
    }


def _optional_time_range(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        start_sec, end_sec = _time_range_argument(value)
    except ValueError:
        return None
    return [start_sec, end_sec]


def _canonical_tool_key(tool_name: str):
    def build(_ctx: Any, request: Any) -> str:
        return f"{tool_name}:{json.dumps(request.arguments, sort_keys=True, ensure_ascii=False, default=str)}"

    return build


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
        for entry in cited_memory:
            provenance_reason = _invalid_final_citation_provenance(workspace, entry, memory_by_id, seen=set())
            if provenance_reason:
                return False, provenance_reason
    return True, ""


def _invalid_final_citation_provenance(
    workspace: EvidenceWorkspace,
    entry: Any,
    memory_by_id: Mapping[str, Any],
    *,
    seen: set[str],
) -> str:
    if entry.entry_id in seen:
        return ""
    seen.add(entry.entry_id)
    metadata = dict(entry.metadata or {})
    source_tool = str(metadata.get("source_tool") or metadata.get("tool") or "").strip()
    removed_legacy_tools = {
        "read_clip",
        "read_segment",
        "scan_segment",
        "search",
        "list",
        "verify",
        "bind_asr_claim",
        "target_coverage",
        "ground_question",
        "vision_read",
        "verify_segment_anchors",
        "global_gist",
        "query_context",
        "caption_segment",
    }
    if not source_tool:
        return f"final citation {entry.entry_id} lacks source provenance"
    if source_tool in removed_legacy_tools:
        return f"final citation {entry.entry_id} comes from removed legacy tool {source_tool}"
    if entry.kind == "caption_support":
        if source_tool != "explore":
            return f"final citation {entry.entry_id} must come from explore caption provenance"
        if _truthy(metadata.get("cannot_final_cite")):
            return f"final citation {entry.entry_id} is marked non-final-citable"
        if _truthy(metadata.get("requires_visual_verify")):
            return f"final citation {entry.entry_id} requires visual verification"
        if str(metadata.get("task_type") or "") in {"counting", "object_presence", "spatial_relation", "scoreboard", "fine_action", "visual_text"}:
            return f"final citation {entry.entry_id} task requires visual_support"
        condition_match = metadata.get("condition_match")
        if isinstance(condition_match, Mapping):
            if not bool(condition_match.get("matches_original_question")):
                return f"final citation {entry.entry_id} does not match the original question condition"
            if str(condition_match.get("match_level") or "") != "direct":
                return f"final citation {entry.entry_id} is not a direct condition match"
        elif "question_condition_match" in metadata and not _truthy(metadata.get("question_condition_match")):
            return f"final citation {entry.entry_id} does not match the original question condition"
        elif "question_condition_match" not in metadata:
            return f"final citation {entry.entry_id} lacks condition-match provenance"
        if entry.supports_option is None:
            answer_mapping = metadata.get("answer_mapping")
            mapped_option = answer_mapping.get("supports_option") if isinstance(answer_mapping, Mapping) else None
            if not str(mapped_option or "").strip():
                return f"final citation {entry.entry_id} lacks explicit option support"
        support_status = str(metadata.get("support_status") or "").strip()
        if support_status != "caption_supported":
            return f"final citation {entry.entry_id} has non-caption support status {support_status}"
        return ""
    if entry.kind == "visual_support":
        if source_tool != "verify_window":
            return f"final citation {entry.entry_id} must come from verify_window provenance"
        verdict = str(metadata.get("verdict") or "").strip()
        if verdict and verdict not in {"supported", "not_found_in_window"}:
            return f"final citation {entry.entry_id} has unsupported visual verdict {verdict}"
        return ""
    if entry.kind in {"answer_support", "answer_conflict_resolved"}:
        if source_tool == "explore":
            mode = str(metadata.get("mode") or "").strip()
            support_status = str(metadata.get("support_status") or "").strip()
            if mode not in {"caption_fact", "mixed"} or support_status not in {"caption_supported", "partial_caption_supported"}:
                return f"final citation {entry.entry_id} must come from caption-supported explore provenance"
            if _truthy(metadata.get("requires_visual_verify")):
                return f"final citation {entry.entry_id} requires visual verification"
            return ""
        if source_tool != "verify_window":
            return f"final citation {entry.entry_id} must come from verify_window provenance"
        if entry.kind == "answer_support" and str(metadata.get("verdict") or "") not in {"", "supported"}:
            return f"final citation {entry.entry_id} has non-support verdict {metadata.get('verdict')}"
        return ""
    if entry.kind == "synthesized_support":
        if source_tool != "synthesize_memory":
            return f"final citation {entry.entry_id} must record synthesize_memory provenance"
        refs = [str(ref).strip() for ref in entry.previous_memory_refs if str(ref).strip()]
        if not refs:
            refs = [str(ref).strip() for ref in _sequence_items(metadata.get("supports")) if str(ref).strip()]
        if not refs:
            return f"final citation {entry.entry_id} lacks synthesized source memory refs"
        for ref in refs:
            source = memory_by_id.get(ref)
            if source is None:
                return f"final citation {entry.entry_id} references unknown source memory {ref}"
            reason = _invalid_final_citation_provenance(workspace, source, memory_by_id, seen=seen)
            if reason:
                return reason
        return ""
    return f"final citation {entry.entry_id} has unsupported provenance kind {entry.kind}"


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
