"""Workspace-first v2 tool surface."""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Sequence

from ..backends.base import BackendRequest, VisionLanguageBackend
from ..registry import DuplicateGuardPolicy, ToolRegistry, ToolRuntimeSpec, tool
from ..video_map import VideoMap, VideoMapSegment, VideoMapStore, _resolve_search_modalities, search_modality_limitations
from ..workspace import EvidenceWorkspace
from .runtime_specs import (
    _normalize_read_clip,
    _normalize_workspace_v2_answer,
    _normalize_workspace_v2_list,
    _normalize_workspace_v2_search,
    _normalize_workspace_v2_synthesize_memory,
    _normalize_workspace_v2_verify,
)
from .workspace_primitives import build_workspace_primitives_registry


ANSWER_SUPPORTING_KINDS = frozenset({"answer_support", "synthesized_support", "answer_conflict_resolved"})


def build_workspace_v2_registry(
    *,
    video_map: VideoMap | VideoMapStore,
    backend: VisionLanguageBackend,
    workspace: Optional[EvidenceWorkspace] = None,
    include_workspace_primitives: bool = True,
) -> ToolRegistry:
    """Build the compact v2 registry: read_clip/search/list/verify/answer plus dispositions."""

    video_map_store = video_map if isinstance(video_map, VideoMapStore) else VideoMapStore(video_map)
    registry = ToolRegistry()

    @tool(name="read_clip", description="Read facts from a video clip without choosing an answer option.")
    def read_clip(
        scope: Mapping[str, Any],
        focus: Sequence[str] = (),
        sampling: Mapping[str, Any] | None = None,
    ) -> Mapping[str, object]:
        current = video_map_store.current
        segment = _segment_from_scope(current, scope)
        start_sec, end_sec = _time_range_from_scope(scope, segment)
        focus_items = [str(item).strip() for item in focus if str(item).strip()]
        ask_for = "; ".join(focus_items) or "Return concise visible/audio/OCR facts from this clip."
        sampling_payload = dict(sampling or {})
        response = backend.generate(
            BackendRequest(
                task="vision_read",
                prompt=_read_clip_prompt(
                    segment_id=segment.segment_id,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    focus=ask_for,
                ),
                media_path=current.video_path,
                media_type="video",
                max_new_tokens=384,
                metadata={
                    "tool": "read_clip",
                    "segment_id": segment.segment_id,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "focus": focus_items,
                    "sampling": sampling_payload,
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
            "raw_backend": raw_backend,
        }

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
                "needs_local_read": False,
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
