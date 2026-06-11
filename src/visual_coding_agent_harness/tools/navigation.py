"""Navigation tools over a structured VideoMap workspace."""

from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from ..agents.grounding.lexicon import FUTURE_PROSPECTIVE_MARKERS
from ..agents.transcript_binder import TranscriptEvidenceBinder
from ..contracts import (
    ClaimModality,
    ClaimRelation,
    TargetSpec,
    build_ordered_transcript_sequence,
    ordered_sequence_exact_option,
)
from ..registry import ToolError, ToolRegistry, tool
from ..text_norm import token_spans, unique_tokens
from ..video_map import VideoMap, VideoMapSegment, VideoMapStore, _resolve_search_modalities, search_modality_limitations
from ..workspace import EvidenceWorkspace, MapUpdateProposal


_TARGET_REF_RE = re.compile(r"^T[1-9]\d*$")
_FUTURE_PROSPECTIVE_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(marker).replace(r"\ ", r"\s+") for marker in FUTURE_PROSPECTIVE_MARKERS)
    + r")\b",
    flags=re.IGNORECASE,
)


def build_video_navigation_registry(
    video_map: VideoMap | VideoMapStore,
    *,
    workspace: EvidenceWorkspace | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    video_map_store = video_map if isinstance(video_map, VideoMapStore) else VideoMapStore(video_map)

    @tool(name="video_ls", description="Build a compact map-first overview of the indexed video workspace.")
    def video_ls(query: str = "", max_segments: int = 16, top_k: int = 5) -> Mapping[str, object]:
        current = video_map_store.current
        indexed_fields = _available_indexes(current.segments)
        overview = current.overview(query=query, max_segments=max_segments, top_k=top_k)
        candidate_ids = [str(candidate["segment_id"]) for candidate in overview["candidates"]]
        candidate_text = ", ".join(candidate_ids) if candidate_ids else "none"
        claim = (
            f"map-first video_ls: Video {current.video_path} has {len(current.segments)} segments "
            f"over {current.duration_sec:.1f} seconds. Available indexes: {', '.join(indexed_fields) or 'none'}. "
            f"Candidate segments: {candidate_text}."
        )
        return {
            "claim": claim,
            "confidence": 1.0,
            "input_artifacts": [current.video_path],
            "regions": [
                {
                    "segment_count": len(current.segments),
                    "duration_sec": current.duration_sec,
                    "available_indexes": indexed_fields,
                }
            ],
            "coverage": overview["coverage"],
            "outline": overview["outline"],
            "candidates": overview["candidates"],
            "recommended_next_tools": overview["recommended_next_tools"],
            "raw_video_map": current.to_dict(),
        }

    @tool(
        name="search_segments",
        description="Search indexed video segments by text query. Valid modalities: caption|asr|ocr|entities.",
    )
    def search_segments(
        query: str,
        top_k: int = 5,
        modalities: Sequence[str] = (),
        additional_targets: Sequence[str] = (),
    ) -> Mapping[str, object]:
        current = video_map_store.current
        query = _query_with_additional_targets(query=query, additional_targets=additional_targets)
        results = current.search(query=query, top_k=top_k, modalities=modalities)
        if results:
            ids = ", ".join(result.segment.segment_id for result in results)
            claim = f"Search for '{query}' returned candidate segments: {ids}."
        else:
            claim = f"Search for '{query}' returned no candidate segments."
        return {
            "claim": claim,
            "confidence": 0.85 if results else 0.2,
            "input_artifacts": [current.video_path],
            "regions": [result.to_dict() for result in results],
            "modalities": _modality_results(current=current, query=query, top_k=top_k, modalities=modalities),
            "limitations": _search_segments_limitations(modalities),
        }

    @tool(name="target_coverage", description="Build a target-to-segment coverage matrix from indexed caption/ASR/OCR/entity fields.")
    def target_coverage(
        targets: Sequence[str] = (),
        target_refs: Sequence[str] = (),
        additional_targets: Sequence[str] = (),
        top_k: int = 3,
        modalities: Sequence[str] = (),
        group_by_option: bool = False,
    ) -> Mapping[str, object]:
        _reject_additional_targets(additional_targets)
        current = video_map_store.current
        rows = []
        coverage_targets = _coverage_target_specs(targets=targets, target_refs=target_refs, workspace=workspace)
        for index, coverage_target in enumerate(coverage_targets, start=1):
            target_ref = str(coverage_target.get("target_ref") or "").strip()
            query_id = "" if target_ref else f"Q{index}"
            display_id = target_ref or query_id
            candidates = _coverage_candidates_for_target(
                current=current,
                target=coverage_target["target"],
                aliases=coverage_target.get("aliases", ()),
                top_k=top_k,
                modalities=modalities,
            )
            status = "candidate" if candidates else "missing"
            row = {
                "target_id": display_id,
                "target": coverage_target["target"],
                "status": status,
                "candidates": candidates,
                "missing_confirmation": not bool(candidates),
                "source": "target_registry" if target_ref else "free_text_query",
            }
            if target_ref:
                row["target_ref"] = target_ref
            else:
                row["query_id"] = query_id
            rows.append(row)
        option_coverage = _option_coverage_rows(rows=rows, workspace=workspace) if group_by_option else []
        summary = "; ".join(
            f"{row['target_id']} {row['target']}: "
            + (
                ", ".join(candidate["segment_id"] for candidate in row["candidates"])
                if row["candidates"]
                else "missing"
            )
            for row in rows
        )
        return {
            "claim": f"Target coverage matrix: {summary or 'no targets supplied'}.",
            "confidence": 1.0,
            "input_artifacts": [current.video_path],
            "coverage": rows,
            "option_coverage": option_coverage,
            "modalities_used": _resolve_search_modalities(modalities),
            "limitations": " ".join(
                [
                    *search_modality_limitations(modalities),
                    "Index coverage only; use read_segment_detail and visual tools to confirm facts before final answers.",
                ]
            ),
        }

    @tool(name="ground_question", description="Ground a question or event into candidate video windows without answering.")
    def ground_question(query: str, top_k: int = 5, modalities: Sequence[str] = ()) -> Mapping[str, object]:
        current = video_map_store.current
        normalized_query = _normalize_grounding_query(query)
        results = current.search(query=normalized_query, top_k=top_k, modalities=modalities) if normalized_query else []
        if not results:
            results = current.anchor_segments(max_segments=top_k)
        candidates = [_grounding_candidate(result) for result in results]
        ids = ", ".join(str(candidate["segment_id"]) for candidate in candidates) if candidates else "none"
        return {
            "claim": f"Grounding query '{query}' returned candidate windows: {ids}.",
            "confidence": max([float(candidate["confidence"]) for candidate in candidates] or [0.0]),
            "input_artifacts": [current.video_path],
            "regions": candidates,
            "candidates": candidates,
            "normalized_query": normalized_query,
            "recommended_next_tools": [
                {
                    "tool": "vision_read",
                    "args": {
                        "segment_id": candidate["segment_id"],
                        "start_sec": candidate["start_sec"],
                        "end_sec": candidate["end_sec"],
                        "ask_for": query,
                    },
                    "reason": "Read typed visual facts from this grounded candidate window.",
                }
                for candidate in candidates
            ],
            "modalities_used": _resolve_search_modalities(modalities),
            "limitations": " ".join(
                [
                    *search_modality_limitations(modalities),
                    "Grounding only localizes candidates from indexes; it does not choose MCQ options or produce final answers.",
                ]
            ),
        }

    @tool(name="read_segment", description="Read compact indexed metadata for one segment.")
    def read_segment(segment_id: str) -> Mapping[str, object]:
        current = video_map_store.current
        invalid = _invalid_segment_result(current=current, segment_id=segment_id)
        if invalid is not None:
            return invalid
        segment = current.get(segment_id)
        claim = _segment_claim(segment)
        return {
            "claim": claim,
            "confidence": 1.0,
            "input_artifacts": [current.video_path],
            "regions": [segment.to_dict()],
        }

    @tool(name="read_segment_detail", description="Read full indexed details for one segment, including target hits.")
    def read_segment_detail(
        segment_id: str,
        targets: Sequence[str] = (),
        target_refs: Sequence[str] = (),
        additional_targets: Sequence[str] = (),
        promote_answer_evidence: bool = False,
        option_targets: Mapping[str, Sequence[str]] | None = None,
    ) -> Mapping[str, object]:
        _reject_additional_targets(additional_targets)
        current = video_map_store.current
        invalid = _invalid_segment_result(current=current, segment_id=segment_id)
        if invalid is not None:
            return invalid
        segment = current.get(segment_id)
        resolved_option_targets = _normalize_option_targets(option_targets or {})
        binding_targets, binding_relations = _resolve_binding_specs(
            target_refs=target_refs,
            workspace=workspace,
        )
        if promote_answer_evidence and not binding_targets:
            binding_targets, binding_relations = _all_registry_binding_specs(workspace=workspace)
        target_ref_texts = [target.canonical_text for target in binding_targets]
        if target_ref_texts:
            resolved_targets = _unique_nonempty_texts(target_ref_texts)
        else:
            resolved_targets = _detail_targets(
                targets=[
                    *list(targets),
                    *_flatten_option_targets(resolved_option_targets),
                ],
                workspace=workspace,
            )
        target_hits = [
            _target_hit_for_segment(segment=segment, target=str(target))
            for target in resolved_targets
            if str(target).strip()
        ]
        target_matches = [_detail_target_match(hit) for hit in target_hits if bool(hit.get("matched"))]
        unmatched_targets = [str(hit.get("target", "")) for hit in target_hits if not bool(hit.get("matched"))]
        nav_digest = _segment_nav_digest(segment=segment, target_matches=target_matches)
        answer_evidence_rows = (
            []
            if promote_answer_evidence
            else _answer_evidence_rows_from_indexed_detail(
                segment=segment,
                option_targets=resolved_option_targets,
            )
        )
        evidence_bindings: list[Mapping[str, object]] = []
        relation_bindings: list[Mapping[str, object]] = []
        if promote_answer_evidence and binding_targets:
            binding_result = _bound_transcript_binding_result(
                segment=segment,
                targets=binding_targets,
                relations=binding_relations,
            )
            binding_payload = binding_result.to_dict()
            evidence_bindings = [
                dict(binding)
                for binding in binding_payload.get("evidence_bindings", [])
                if isinstance(binding, Mapping)
            ]
            relation_bindings = [
                dict(binding)
                for binding in binding_payload.get("relation_bindings", [])
                if isinstance(binding, Mapping)
            ]
            answer_evidence_rows = [
                *answer_evidence_rows,
                *_answer_evidence_rows_from_bound_targets(
                    segment=segment,
                    targets=binding_targets,
                    relations=binding_relations,
                    workspace=workspace,
                    binding_result=binding_result,
                ),
            ]
        result = {
            "claim": f"{_segment_detail_claim(segment, target_hits=target_hits)} {nav_digest}",
            "confidence": 1.0,
            "input_artifacts": [current.video_path],
            "segment_id": segment.segment_id,
            "start_sec": float(segment.start_sec),
            "end_sec": float(segment.end_sec),
            "visual_caption": segment.low_fps_caption,
            "asr_summary": segment.asr_text,
            "raw_asr_excerpt": segment.asr_text,
            "ocr_text": segment.ocr_text,
            "entities": list(segment.entities),
            "keyframe_paths": list(segment.keyframe_paths),
            "target_hits": target_hits,
            "target_matches": target_matches,
            "unmatched_targets": unmatched_targets,
            "evidence_bindings": evidence_bindings,
            "relation_bindings": relation_bindings,
            "answer_evidence_rows": answer_evidence_rows,
            "promote_answer_evidence": bool(promote_answer_evidence),
            "target_refs": [target.target_id for target in binding_targets],
            "nav_digest": nav_digest,
            "regions": [segment.to_dict()],
            "recommended_next_tools": _detail_recommended_next_tools(segment=segment, target_matches=target_matches),
            "limitations": "Indexed segment detail only; call vision_read or caption_segment for fresh visual evidence.",
        }
        if workspace is not None:
            return workspace.promote_textual_answer_evidence_payload(
                tool_name="read_segment_detail",
                raw_output=result,
            )
        return result

    @tool(name="locate_targets_in_segment", description="Text-only target locator over ASR/OCR/caption indexes for one segment.")
    def locate_targets_in_segment(
        segment_id: str,
        targets: Sequence[str] = (),
        target_refs: Sequence[str] = (),
        additional_targets: Sequence[str] = (),
        top_k_per_target: int = 3,
    ) -> Mapping[str, object]:
        _reject_additional_targets(additional_targets)
        current = video_map_store.current
        invalid = _invalid_segment_result(current=current, segment_id=segment_id)
        if invalid is not None:
            return invalid
        segment = current.get(segment_id)
        binding_targets, _binding_relations = _resolve_binding_specs(target_refs=target_refs, workspace=workspace)
        target_ref_texts = [target.canonical_text for target in binding_targets]
        if target_ref_texts:
            resolved_targets = _unique_nonempty_texts(target_ref_texts)
        else:
            resolved_targets = _detail_targets(
                targets=list(targets),
                workspace=workspace,
            )
        candidates = _locate_target_candidates(
            segment=segment,
            targets=resolved_targets,
            top_k_per_target=top_k_per_target,
        )
        anchors = _merge_locate_candidates(segment=segment, candidates=candidates)
        ordered_list_timeline_rows = _ordered_list_timeline_rows(
            segment=segment,
            candidates=candidates,
        )
        ordered_transcript_rows = _ordered_transcript_answer_evidence_rows(
            segment=segment,
            targets=binding_targets,
            workspace=workspace,
        )
        focused_vision_call_args = _focused_vision_call_args_for_ordered_candidate(
            segment=segment,
            candidates=candidates,
        )
        recommended_next_actions = _locate_recommended_next_actions(
            focused_vision_call_args=focused_vision_call_args,
            candidates=candidates,
            target_refs=_ordered_candidate_target_refs(candidates=candidates, workspace=workspace),
            ordered_transcript_rows=ordered_transcript_rows,
            segment_id=segment.segment_id,
        )
        if workspace is not None and recommended_next_actions:
            for action in recommended_next_actions:
                route_kind = str(action.get("route_kind") or "")
                if route_kind in {"focused_ordered_list_vision", "ordered_list_transcript_complete"}:
                    workspace.write_trace_event(
                        "ordered_transcript_candidate_detected"
                        if route_kind == "ordered_list_transcript_complete"
                        else "ordered_list_candidate_detected",
                        {
                            "segment_id": segment.segment_id,
                            "candidate_id": str(action.get("candidate_id") or ""),
                            "target_refs": list(action.get("target_refs") or []),
                        },
                    )
                if action.get("route_kind") == "ordered_list_transcript_complete":
                    workspace.write_trace_event(
                        "ordered_transcript_sequence_supported",
                        {
                            "segment_id": segment.segment_id,
                            "candidate_id": str(action.get("candidate_id") or ""),
                            "target_refs": list(action.get("target_refs") or []),
                        },
                    )
                if action.get("route_kind") == "focused_ordered_list_vision":
                    workspace.write_trace_event(
                        "focused_ordered_list_vision_recommended",
                        {
                            "segment_id": segment.segment_id,
                            "candidate_id": str(action.get("candidate_id") or ""),
                            "args": dict(action.get("args") or {}),
                        },
                    )
        verify_call_args = {
            "segment_id": segment.segment_id,
            "anchors": anchors,
            "targets": list(resolved_targets),
        } if anchors else {}
        target_text = ", ".join(str(target) for target in resolved_targets) or "none"
        candidate_text = ", ".join(
            _locate_candidate_label(candidate) for candidate in candidates
        )
        claim = (
            f"locate_targets_in_segment({segment.segment_id}) searched {len(resolved_targets)} target(s): "
            f"{target_text}. Candidate anchors: {candidate_text or 'none'}."
        )
        return {
            "claim": claim,
            "confidence": 1.0 if candidates else 0.35,
            "input_artifacts": [current.video_path],
            "segment_id": segment.segment_id,
            "start_sec": float(segment.start_sec),
            "end_sec": float(segment.end_sec),
            "targets": list(resolved_targets),
            "candidates": candidates,
            "anchors_for_vlm": anchors,
            "ordered_list_timeline_rows": ordered_list_timeline_rows,
            "answer_evidence_rows": ordered_transcript_rows,
            "focused_vision_call_args": focused_vision_call_args,
            "verify_call_args": verify_call_args,
            "recommended_next_actions": recommended_next_actions,
            "regions": [
                {
                    "segment_id": segment.segment_id,
                    "start_sec": float(segment.start_sec),
                    "end_sec": float(segment.end_sec),
                    "candidates": candidates,
                    "anchors_for_vlm": anchors,
                    "ordered_list_timeline_rows": ordered_list_timeline_rows,
                    "answer_evidence_rows": ordered_transcript_rows,
                    "focused_vision_call_args": focused_vision_call_args,
                    "verify_call_args": verify_call_args,
                    "recommended_next_actions": recommended_next_actions,
                }
            ],
            "recommended_next_tools": _locate_recommended_next_tools(
                focused_vision_call_args=focused_vision_call_args,
                verify_call_args=verify_call_args,
            ),
            "limitations": (
                "Text-only locator (ASR/OCR/visual_caption/entities); does NOT confirm visual presence. "
                "For ordered-list candidates, call the focused vision_read next to confirm the single visible scene; "
                "call verify_segment_anchors on anchors_for_vlm when target-level visual verification is needed."
            ),
        }

    @tool(name="expand_window", description="Return a bounded temporal window around a segment.")
    def expand_window(segment_id: str, before_sec: float = 30.0, after_sec: float = 30.0) -> Mapping[str, object]:
        current = video_map_store.current
        invalid = _invalid_segment_result(current=current, segment_id=segment_id)
        if invalid is not None:
            return invalid
        segment = current.get(segment_id)
        start_sec = max(0.0, segment.start_sec - before_sec)
        end_sec = min(current.duration_sec, segment.end_sec + after_sec)
        return {
            "claim": (
                f"Expanded {segment.segment_id} from [{segment.start_sec:.1f}, {segment.end_sec:.1f}] "
                f"to [{start_sec:.1f}, {end_sec:.1f}] seconds."
            ),
            "confidence": 1.0,
            "input_artifacts": [current.video_path],
            "regions": [
                {
                    "segment_id": segment.segment_id,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "source_start_sec": segment.start_sec,
                    "source_end_sec": segment.end_sec,
                }
            ],
        }

    @tool(name="zoom", description="Materialize finer child segments for a coarse VideoMap segment.")
    def zoom(segment_id: str, target_granularity_sec: float = 60.0) -> Mapping[str, object]:
        current = video_map_store.current
        invalid = _invalid_segment_result(current=current, segment_id=segment_id)
        if invalid is not None:
            return invalid
        parent = current.get(segment_id)
        children = video_map_store.materialize_zoom(
            segment_id,
            target_granularity_sec=target_granularity_sec,
        )
        child_ids = ", ".join(child.segment_id for child in children)
        return {
            "claim": (
                f"Materialized {len(children)} child segment{'s' if len(children) != 1 else ''} "
                f"from {segment_id}: {child_ids}."
            ),
            "confidence": 1.0,
            "input_artifacts": [current.video_path],
            "regions": [
                {
                    "segment_id": parent.segment_id,
                    "start_sec": parent.start_sec,
                    "end_sec": parent.end_sec,
                    "target_granularity_sec": target_granularity_sec,
                    "child_segments": [child.to_dict() for child in children],
                }
            ],
            "recommended_next_tools": [
                {
                    "tool": "inspect_segment",
                    "args": {
                        "segment_id": child.segment_id,
                        "start_sec": child.start_sec,
                        "end_sec": child.end_sec,
                    },
                    "reason": "Delegate localized inspection on this finer temporal window.",
                }
                for child in children
            ],
        }

    @tool(name="commit_map_proposals", description="Apply pending map update proposals to the mutable VideoMap.")
    def commit_map_proposals(limit: int = 8, min_confidence: float = 0.0) -> Mapping[str, object]:
        if workspace is None:
            raise ValueError("commit_map_proposals requires an EvidenceWorkspace")
        pending = workspace.load_pending_proposals()
        applied = []
        skipped = []
        for proposal in pending[: max(0, int(limit))]:
            if proposal.confidence < float(min_confidence):
                skipped.append({"proposal_id": proposal.proposal_id, "reason": "below_min_confidence"})
                continue
            try:
                updated = _apply_map_update_proposal(video_map_store=video_map_store, proposal=proposal)
            except ValueError as exc:
                skipped.append({"proposal_id": proposal.proposal_id, "reason": str(exc)})
                continue
            committed = workspace.mark_proposal_committed(proposal.proposal_id)
            applied.append(
                {
                    "proposal_id": proposal.proposal_id,
                    "segment_id": updated.segment_id,
                    "update_type": proposal.update_type,
                    "committed_at": committed.committed_at if committed else None,
                }
            )
        return {
            "claim": f"Committed {len(applied)} map proposal(s); skipped {len(skipped)}.",
            "confidence": 1.0,
            "input_artifacts": [],
            "regions": [{"applied": applied, "skipped": skipped}],
            "applied": applied,
            "skipped": skipped,
            "limitations": "Applies audited proposal payloads only; no model inference is performed.",
        }

    registry.register(video_ls)
    registry.register(search_segments)
    registry.register(target_coverage)
    registry.register(ground_question)
    registry.register(read_segment)
    registry.register(read_segment_detail)
    registry.register(locate_targets_in_segment)
    registry.register(expand_window)
    registry.register(zoom)
    registry.register(commit_map_proposals)
    return registry


def _apply_map_update_proposal(*, video_map_store: VideoMapStore, proposal: MapUpdateProposal) -> VideoMapSegment:
    payload = dict(proposal.payload)
    allowed = {
        "low_fps_caption": payload.get("low_fps_caption"),
        "asr_text": payload.get("asr_text"),
        "ocr_text": payload.get("ocr_text"),
        "entities": payload.get("entities"),
        "keyframe_paths": payload.get("keyframe_paths"),
        "embedding_refs": payload.get("embedding_refs"),
    }
    if all(value in (None, "", []) for value in allowed.values()):
        raise ValueError("proposal has no supported map update fields")
    return video_map_store.update_segment(
        proposal.target_segment_id,
        low_fps_caption=_optional_text(allowed["low_fps_caption"]),
        asr_text=_optional_text(allowed["asr_text"]),
        ocr_text=_optional_text(allowed["ocr_text"]),
        entities=_optional_text_list(allowed["entities"]),
        keyframe_paths=_optional_text_list(allowed["keyframe_paths"]),
        embedding_refs=_optional_text_list(allowed["embedding_refs"]),
    )


def _optional_text(value: object) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_text_list(value: object) -> list[str] | None:
    if value in (None, ""):
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value if str(item)]
    text = str(value).strip()
    return [text] if text else None


def _current_video_map(video_map: VideoMap | VideoMapStore) -> VideoMap:
    if isinstance(video_map, VideoMapStore):
        return video_map.current
    return video_map


def _available_indexes(segments: Sequence[VideoMapSegment]) -> Sequence[str]:
    indexes = []
    checks = [
        ("keyframes", lambda segment: bool(segment.keyframe_paths)),
        ("captions", lambda segment: bool(segment.low_fps_caption)),
        ("asr", lambda segment: bool(segment.asr_text)),
        ("ocr", lambda segment: bool(segment.ocr_text)),
        ("entities", lambda segment: bool(segment.entities)),
        ("embeddings", lambda segment: bool(segment.embedding_refs)),
    ]
    for name, predicate in checks:
        if any(predicate(segment) for segment in segments):
            indexes.append(name)
    return indexes


def _invalid_segment_result(*, current: VideoMap, segment_id: object) -> Mapping[str, object] | None:
    requested = str(segment_id or "").strip()
    valid_ids = [segment.segment_id for segment in current.segments]
    if requested and requested in set(valid_ids):
        return None
    return {
        "ok": False,
        "error_code": "invalid_segment_id",
        "requested_segment_id": requested,
        "valid_segment_ids": valid_ids,
        "claim": (
            f"Invalid segment_id '{requested}'. Valid segment_ids: "
            + (", ".join(valid_ids[:16]) if valid_ids else "(none)")
        ),
        "confidence": 0.0,
        "input_artifacts": [current.video_path],
        "regions": [],
        "grounding_quality": "invalid",
        "limitations": "Segment id was rejected exactly; no substitute segment was selected.",
    }


_GROUNDING_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "with",
    "without",
    "for",
    "from",
    "by",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "these",
    "those",
    "his",
    "her",
    "their",
    "its",
    "how",
    "what",
    "which",
    "where",
    "who",
    "when",
    "why",
    "video",
    "question",
    "questions",
    "option",
    "options",
    "answer",
    "letter",
    "exactly",
    "one",
    "short",
    "reason",
    "outside",
    "knowledge",
    "unless",
    "directly",
    "supported",
    "evidence",
    "according",
    "use",
    "not",
    "multiple",
    "choice",
    "first",
}


def _normalize_grounding_query(query: str) -> str:
    text = re.sub(r"^\s*[A-H][.)]\s*", "", str(query)).strip()
    tokens = []
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", text):
        lowered = token.lower().strip("'")
        if len(lowered) < 3 or lowered in _GROUNDING_STOPWORDS:
            continue
        tokens.append(token)
    return " ".join(tokens)


def _modality_results(
    *,
    current: VideoMap,
    query: str,
    top_k: int,
    modalities: Sequence[str],
) -> Mapping[str, Sequence[Mapping[str, object]]]:
    channels = _resolve_search_modalities(modalities)
    grouped = {}
    for channel in channels:
        results = current.search(query=query, top_k=top_k, modalities=[channel])
        grouped[channel] = [result.to_dict() for result in results]
    return grouped


def _search_segments_limitations(modalities: Sequence[str]) -> str:
    base_limitations = (
        "Training-free VideoMap retrieval over caption/ASR/OCR/entity indexes; "
        "embedding retrieval can replace the scoring backend without changing this contract."
    )
    return " ".join([*search_modality_limitations(modalities), base_limitations])


def _grounding_candidate(result: object) -> Mapping[str, object]:
    segment = getattr(result, "segment")
    matches = getattr(result, "matches", []) or []
    modalities = []
    for match in matches:
        if not isinstance(match, Mapping):
            continue
        modality = str(match.get("modality", "") or match.get("field", "")).strip()
        if modality and modality not in modalities:
            modalities.append(modality)
    matched_fields = [str(field) for field in getattr(result, "matched_fields", []) or []]
    reason = str(getattr(result, "relevance_reason", "") or "").strip()
    if not reason:
        reason = _relevance_reason(matched_fields)
    return {
        "segment_id": segment.segment_id,
        "start_sec": float(segment.start_sec),
        "end_sec": float(segment.end_sec),
        "reason": reason,
        "modality": ", ".join(modalities) or (matched_fields[0] if matched_fields else "timeline_anchor"),
        "confidence": float(getattr(result, "score", 0.0) or 0.0),
        "matched_fields": matched_fields,
        "matches": [dict(match) for match in matches if isinstance(match, Mapping)],
    }


def _coverage_candidate(result: object) -> Mapping[str, object]:
    segment = getattr(result, "segment")
    matches = [dict(match) for match in getattr(result, "matches", []) or [] if isinstance(match, Mapping)]
    best_match = max(
        matches,
        key=lambda match: float(match.get("score", 0.0) or 0.0),
        default={},
    )
    best_score = float(best_match.get("score", 0.0) or 0.0)
    return {
        "segment_id": segment.segment_id,
        "start_sec": float(segment.start_sec),
        "end_sec": float(segment.end_sec),
        "score": float(getattr(result, "score", 0.0) or 0.0),
        "matched_fields": [str(field) for field in getattr(result, "matched_fields", []) or []],
        "matches": matches,
        "source": str(best_match.get("field") or best_match.get("modality") or ""),
        "snippet": str(best_match.get("evidence", "")),
        "directness": _target_directness(best_score),
        "summary": segment.compact_text(),
        "relevance_reason": str(getattr(result, "relevance_reason", "") or ""),
    }


def _coverage_candidates_for_target(
    *,
    current: VideoMap,
    target: object,
    aliases: object,
    top_k: int,
    modalities: Sequence[str],
) -> list[Mapping[str, object]]:
    queries = _unique_nonempty_texts([str(target), *_coerce_text_sequence(aliases)])
    candidates: list[Mapping[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for query in queries:
        for result in current.search(query=query, top_k=top_k, modalities=modalities):
            candidate = dict(_coverage_candidate(result))
            key = (str(candidate.get("segment_id", "")), str(candidate.get("source", "")))
            if key in seen:
                continue
            seen.add(key)
            candidate["query"] = query
            candidates.append(candidate)
            if len(candidates) >= max(1, int(top_k or 1)):
                return candidates
    return candidates


def _coverage_target_specs(
    *,
    targets: Sequence[str],
    target_refs: Sequence[str],
    workspace: EvidenceWorkspace | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen_refs: set[str] = set()
    binding_targets, _relations = _resolve_binding_specs(target_refs=target_refs, workspace=workspace)
    for target in binding_targets:
        if target.target_id in seen_refs:
            continue
        seen_refs.add(target.target_id)
        rows.append(
            {
                "target_ref": target.target_id,
                "target": target.canonical_text,
                "aliases": tuple(str(alias) for alias in target.aliases),
            }
        )
    for target in _unique_nonempty_texts(targets):
        rows.append({"target": target, "aliases": ()})
    return rows


def _option_coverage_rows(*, rows: Sequence[Mapping[str, object]], workspace: EvidenceWorkspace | None) -> list[dict[str, object]]:
    registry = getattr(workspace, "target_registry", None) if workspace is not None else None
    if registry is None:
        return []
    rows_by_ref = {str(row.get("target_ref", "")): row for row in rows if str(row.get("target_ref", ""))}
    option_rows: list[dict[str, object]] = []
    options_by_id = getattr(registry, "options_by_id", {})
    for option_id in sorted(str(key) for key in options_by_id):
        option = registry.option_for(option_id)
        target_sequence = [str(target_id) for target_id in option.target_sequence]
        if not target_sequence or any(target_id not in rows_by_ref for target_id in target_sequence):
            continue
        option_rows.append(
            {
                "option": option.option_id,
                "target_refs": target_sequence,
                "targets": [str(rows_by_ref[target_id].get("target", "")) for target_id in target_sequence],
                "coverage": [dict(rows_by_ref[target_id]) for target_id in target_sequence],
            }
        )
    return option_rows


def _relevance_reason(matched_fields: Sequence[str]) -> str:
    fields = [str(field) for field in matched_fields if str(field)]
    if not fields:
        return "fallback timeline anchor"
    return "matched indexed field(s): " + ", ".join(fields)


def _segment_claim(segment: VideoMapSegment) -> str:
    parts = [
        f"{segment.segment_id} covers {segment.start_sec:.1f}-{segment.end_sec:.1f}s.",
        f"caption: {segment.low_fps_caption}" if segment.low_fps_caption else "",
        f"ASR: {segment.asr_text}" if segment.asr_text else "",
        f"OCR: {segment.ocr_text}" if segment.ocr_text else "",
        f"entities: {', '.join(segment.entities)}" if segment.entities else "",
    ]
    return " ".join(part for part in parts if part)


def _segment_detail_claim(segment: VideoMapSegment, *, target_hits: Sequence[Mapping[str, object]]) -> str:
    hit_targets = [str(hit.get("target")) for hit in target_hits if bool(hit.get("matched"))]
    parts = [
        f"{segment.segment_id} detail covers {segment.start_sec:.1f}-{segment.end_sec:.1f}s.",
        "visual caption available" if segment.low_fps_caption else "",
        "ASR summary available" if segment.asr_text else "",
        "OCR available" if segment.ocr_text else "",
        f"entities: {', '.join(segment.entities)}" if segment.entities else "",
        f"target hits: {', '.join(hit_targets)}" if hit_targets else "target hits: none",
    ]
    return " ".join(part for part in parts if part)


def _segment_nav_digest(
    *,
    segment: VideoMapSegment,
    target_matches: Sequence[Mapping[str, object]],
) -> str:
    matched = ", ".join(str(match.get("target", "")) for match in target_matches if str(match.get("target", "")))
    parts = [
        f"targets={matched}" if matched else "",
        f"visual={_detail_evidence_snippet(segment.low_fps_caption, max_chars=120)}" if segment.low_fps_caption else "",
        f"asr={_detail_evidence_snippet(segment.asr_text, max_chars=160)}" if segment.asr_text else "",
        f"ocr={_detail_evidence_snippet(segment.ocr_text, max_chars=120)}" if segment.ocr_text else "",
    ]
    digest = " | ".join(part for part in parts if part)
    return f"nav_digest: {digest}" if digest else "nav_digest: (no indexed detail)"


def _detail_targets(*, targets: Sequence[str], workspace: EvidenceWorkspace | None) -> list[str]:
    explicit = _unique_nonempty_texts(targets)
    if workspace is None:
        return explicit
    inherited = _coverage_targets(workspace)
    if explicit:
        return _unique_nonempty_texts([*explicit, *inherited])
    return inherited


def _normalize_option_targets(option_targets: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    for option, targets in option_targets.items():
        letter = str(option).strip().upper()[:1]
        if not letter:
            continue
        if isinstance(targets, (str, bytes)):
            values = [str(targets)]
        elif isinstance(targets, Sequence):
            values = [str(target) for target in targets]
        else:
            continue
        cleaned = _unique_nonempty_texts(values)
        if cleaned:
            normalized[letter] = cleaned
    return normalized


def _flatten_option_targets(option_targets: Mapping[str, Sequence[str]]) -> list[str]:
    flattened: list[str] = []
    for targets in option_targets.values():
        if isinstance(targets, Sequence) and not isinstance(targets, (str, bytes)):
            flattened.extend(str(target) for target in targets)
    return _unique_nonempty_texts(flattened)


def _resolve_binding_specs(
    *,
    target_refs: Sequence[str],
    workspace: EvidenceWorkspace | None,
) -> tuple[list[TargetSpec], list[ClaimRelation]]:
    registry = getattr(workspace, "target_registry", None) if workspace is not None else None
    if registry is None:
        if any(str(ref or "").strip() for ref in target_refs):
            raise ToolError("target_refs require a workspace TargetRegistry")
        return [], []
    selected_targets: list[TargetSpec] = []
    selected_target_ids: set[str] = set()
    relation_ids: set[str] = set()
    for raw_ref in target_refs:
        ref = str(raw_ref or "").strip()
        if not ref:
            continue
        if not _TARGET_REF_RE.fullmatch(ref):
            raise ToolError(f"Unknown target_ref: {ref}")
        try:
            target = registry.resolve_target_ref(ref)
        except KeyError as exc:
            raise ToolError(f"Unknown target_ref: {ref}") from exc
        if target.target_id not in selected_target_ids:
            selected_targets.append(target)
            selected_target_ids.add(target.target_id)

    relations_by_id = getattr(registry, "relations_by_id", {})
    selected_relations: list[ClaimRelation] = []
    for relation in relations_by_id.values():
        if not isinstance(relation, ClaimRelation):
            continue
        if relation.relation_id in relation_ids or (
            relation.source_target_id in selected_target_ids
            and relation.destination_target_id in selected_target_ids
        ):
            selected_relations.append(relation)
    return selected_targets, selected_relations


def _all_registry_binding_specs(
    *,
    workspace: EvidenceWorkspace | None,
) -> tuple[list[TargetSpec], list[ClaimRelation]]:
    registry = getattr(workspace, "target_registry", None) if workspace is not None else None
    targets_by_id = getattr(registry, "targets_by_id", None)
    if not isinstance(targets_by_id, Mapping):
        return [], []
    targets = [target for target in targets_by_id.values() if isinstance(target, TargetSpec)]
    target_ids = {target.target_id for target in targets}
    relations_by_id = getattr(registry, "relations_by_id", {})
    relations = [
        relation
        for relation in getattr(relations_by_id, "values", lambda: ())()
        if isinstance(relation, ClaimRelation)
        and relation.source_target_id in target_ids
        and relation.destination_target_id in target_ids
    ]
    return targets, relations


def _answer_evidence_rows_from_indexed_detail(
    *,
    segment: VideoMapSegment,
    option_targets: Mapping[str, Sequence[str]],
) -> list[Mapping[str, object]]:
    if not option_targets:
        return []
    asr_sources = [
        source for source in _locate_text_sources(segment) if str(source.get("source", "")).startswith("asr")
    ]
    if not asr_sources:
        return []

    rows: list[Mapping[str, object]] = []
    for option, targets in option_targets.items():
        ordered_targets = _unique_nonempty_texts([str(target) for target in targets])
        if len(ordered_targets) >= 2:
            sequence = _ordered_indexed_matches(targets=ordered_targets, sources=asr_sources)
            if len(sequence) >= len(ordered_targets):
                rows.append(
                    _indexed_asr_evidence_row(
                        segment=segment,
                        target=" -> ".join(ordered_targets[:6]),
                        match=sequence[-1],
                        claim=(
                            f"Indexed ASR in {segment.segment_id} presents a target sequence in order: "
                            + " -> ".join(ordered_targets[:6])
                        ),
                        confidence=0.6,
                        assigned_by="asr_cue_sequence",
                    )
                )
        for target in ordered_targets:
            match = _best_indexed_asr_match(target=target, sources=asr_sources)
            if match is None:
                continue
            rows.append(
                _indexed_asr_evidence_row(
                    segment=segment,
                    target=target,
                    match=match,
                    claim=(
                        f"Indexed ASR in {segment.segment_id} directly mentions target '{target}'. "
                        f"Snippet: {match.get('snippet', '')}"
                    ),
                    confidence=min(0.6, float(match.get("confidence", 0.0) or 0.0)),
                    assigned_by="asr_cue_detail",
                )
            )
    return rows


def _answer_evidence_rows_from_bound_targets(
    *,
    segment: VideoMapSegment,
    targets: Sequence[TargetSpec],
    relations: Sequence[ClaimRelation],
    workspace: EvidenceWorkspace | None,
    binding_result: Any | None = None,
) -> list[Mapping[str, object]]:
    if not _bound_transcript_text(segment):
        return []
    result = binding_result or _bound_transcript_binding_result(
        segment=segment,
        targets=targets,
        relations=relations,
    )
    relation_payloads = [asdict(binding) for binding in result.relation_bindings]
    supported_relations = [
        relation for relation in relation_payloads if str(relation.get("status", "")).lower() == "supported"
    ]
    rows: list[Mapping[str, object]] = []
    for binding in result.evidence_bindings:
        if binding.status != "supported":
            continue
        binding_payload = asdict(binding)
        binding_payload["claim_modality"] = binding.claim_modality.value
        binding_payload["segment_id"] = segment.segment_id
        binding_payload["relation_bindings"] = [
            relation
            for relation in supported_relations
            if _relation_touches_target(
                relation_id=str(relation.get("relation_id", "")),
                target_id=binding.target_id,
                relations=relations,
            )
        ]
        rows.append(
            {
                "evidence_id": binding.evidence_id,
                "tool": "transcript_evidence_binder",
                "segment_id": segment.segment_id,
                "time_range": [float(segment.start_sec), float(segment.end_sec)],
                "event_label": binding.target_id,
                "claim": (
                    f"TranscriptEvidenceBinder marked {binding.target_id} as supported in "
                    f"{segment.segment_id}: {binding.snippet}"
                ),
                "confidence": 0.88,
                "grounding_quality": "indexed_transcript",
                "confidence_signal": "explicit transcript binding",
                "limitations": "Conservative transcript binding over indexed ASR; no model-level semantic inference.",
                "source": binding.source,
                "snippet": binding.snippet,
                "evidence_binding": binding_payload,
            }
        )
    return rows


def _bound_transcript_binding_result(
    *,
    segment: VideoMapSegment,
    targets: Sequence[TargetSpec],
    relations: Sequence[ClaimRelation],
) -> Any:
    text = _bound_transcript_text(segment)
    start_sec = _bound_transcript_start_sec(segment)
    return TranscriptEvidenceBinder().bind(
        text=text,
        targets=targets,
        relations=relations,
        segment_id=segment.segment_id,
        start_sec=start_sec,
        source="indexed_transcript",
    )


def _bound_transcript_text(segment: VideoMapSegment) -> str:
    sentence_texts = [
        str(sentence.get("text") or "").strip()
        for sentence in (getattr(segment, "asr_sentences", ()) or ())
        if isinstance(sentence, Mapping) and str(sentence.get("text") or "").strip()
    ]
    if sentence_texts:
        return " ".join(sentence_texts)
    return str(segment.asr_text or "").strip()


def _bound_transcript_start_sec(segment: VideoMapSegment) -> float:
    starts = [
        float(sentence.get("start_sec", segment.start_sec) or segment.start_sec)
        for sentence in (getattr(segment, "asr_sentences", ()) or ())
        if isinstance(sentence, Mapping) and str(sentence.get("text") or "").strip()
    ]
    return min(starts) if starts else float(segment.start_sec)


def _ordered_transcript_answer_evidence_rows(
    *,
    segment: VideoMapSegment,
    targets: Sequence[TargetSpec],
    workspace: EvidenceWorkspace | None,
) -> list[Mapping[str, object]]:
    if not targets:
        return []
    sequence = build_ordered_transcript_sequence(
        text=segment.asr_text,
        targets=targets,
        segment_id=segment.segment_id,
        start_sec=float(segment.start_sec),
        end_sec=float(segment.end_sec),
    )
    if sequence is None:
        return []
    if sequence.status != "supported":
        return []
    sequence_payload = sequence.to_dict()
    target_refs = list(sequence.ordered_target_refs)
    evidence_id = str(sequence_payload.get("evidence_id") or f"seq_{segment.segment_id}")
    row = {
        "evidence_id": evidence_id,
        "tool": "ordered_transcript_sequence",
        "segment_id": segment.segment_id,
        "time_range": [float(segment.start_sec), float(segment.end_sec)],
        "event_label": "ordered_transcript_sequence",
        "claim": (
            f"Indexed transcript in {segment.segment_id} gives a complete contiguous ordered list "
            "over target refs: " + " -> ".join(target_refs)
        ),
        "confidence": sequence.confidence,
        "grounding_quality": "indexed_transcript",
        "confidence_signal": "complete contiguous transcript ordered list",
        "limitations": "Order is derived from ASR text position in one contiguous enumeration; visual corroboration is optional unless explicitly required.",
        "source": sequence.source,
        "snippet": sequence.snippet,
        "ordered_target_refs": target_refs,
        "ordered_transcript_sequence": sequence_payload,
        "sequence_binding": {
            "evidence_id": evidence_id,
            "status": "supported",
            "source": sequence.source,
            "snippet": sequence.snippet,
            "ordered_target_refs": target_refs,
        },
        "evidence_binding": {
            "evidence_id": evidence_id,
            "status": "supported",
            "claim_modality": ClaimModality.NARRATED_FACT.value,
            "target_id": "ordered_sequence",
            "segment_id": segment.segment_id,
            "source": sequence.source,
            "snippet": sequence.snippet,
            "ordered_target_refs": target_refs,
        },
    }
    registry = getattr(workspace, "target_registry", None) if workspace is not None else None
    options_by_id = getattr(registry, "options_by_id", {}) if registry is not None else {}
    supported_option = ordered_sequence_exact_option(sequence, options_by_id)
    if supported_option:
        row["supported_option"] = supported_option
        row["evidence_binding"]["supported_option"] = supported_option
    return [row]


def _relation_touches_target(
    *,
    relation_id: str,
    target_id: str,
    relations: Sequence[ClaimRelation],
) -> bool:
    for relation in relations:
        if relation.relation_id != relation_id:
            continue
        return target_id in {relation.source_target_id, relation.destination_target_id}
    return False


def _best_indexed_asr_match(
    *,
    target: str,
    sources: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    matches = [
        match
        for match in _indexed_asr_matches_for_target(target=target, sources=sources, top_k=5)
        if str(match.get("source", "")).startswith("asr")
    ]
    return matches[0] if matches else None


def _ordered_indexed_matches(
    *,
    targets: Sequence[str],
    sources: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    ordered: list[Mapping[str, object]] = []
    cursor = (-1.0, -1)
    for target in targets:
        candidates = [
            match
            for match in _indexed_asr_matches_for_target(target=target, sources=sources, top_k=8)
            if str(match.get("source", "")).startswith("asr")
        ]
        selected = None
        for match in candidates:
            position = (float(match.get("start_sec", 0.0) or 0.0), int(match.get("match_start", 0) or 0))
            if position > cursor:
                selected = match
                cursor = position
                break
        if selected is None:
            return ordered
        ordered.append(selected)
    return ordered


def _indexed_asr_matches_for_target(
    *,
    target: str,
    sources: Sequence[Mapping[str, object]],
    top_k: int,
) -> list[Mapping[str, object]]:
    matches: list[Mapping[str, object]] = []
    for alias in _indexed_asr_target_aliases(target):
        matches.extend(_find_target_text_matches(target=alias, sources=sources, top_k=top_k))
        matches.extend(_find_stemmed_target_text_matches(target=alias, sources=sources, top_k=top_k))
    deduped = _dedupe_locate_matches([dict(match) for match in matches])
    return sorted(
        deduped,
        key=lambda item: (
            int(item.get("source_priority", 9)),
            int(item.get("match_priority", 9)),
            float(item.get("start_sec", 0.0) or 0.0),
            int(item.get("match_start", 0) or 0),
            -float(item.get("confidence", 0.0) or 0.0),
        ),
    )[: max(1, int(top_k or 1))]


def _indexed_asr_target_aliases(target: str) -> list[str]:
    return _unique_nonempty_texts([str(target)])


def _indexed_asr_evidence_row(
    *,
    segment: VideoMapSegment,
    target: str,
    match: Mapping[str, object],
    claim: str,
    confidence: float,
    assigned_by: str,
) -> Mapping[str, object]:
    return {
        "tool": "asr_cue_detail",
        "segment_id": segment.segment_id,
        "time_range": [float(match.get("start_sec", segment.start_sec) or segment.start_sec), float(match.get("end_sec", segment.end_sec) or segment.end_sec)],
        "event_label": target,
        "claim": claim,
        "confidence": float(confidence),
        "grounding_quality": "indexed_transcript",
        "confidence_signal": "indexed transcript cue",
        "limitations": "Derived from indexed ASR cue text; use visual tools for non-narrated visual appearance details.",
        "artifact": segment.video_path if hasattr(segment, "video_path") else "",
        "source": str(match.get("source", "")),
        "snippet": str(match.get("snippet", "")),
    }


def _coverage_targets(workspace: EvidenceWorkspace) -> list[str]:
    for observation in reversed(workspace.read_observations(tool_name="target_coverage")):
        coverage = observation.raw_output.get("coverage", [])
        if not isinstance(coverage, Sequence) or isinstance(coverage, (str, bytes)):
            continue
        inherited = _unique_nonempty_texts(
            str(row.get("target", ""))
            for row in coverage
            if isinstance(row, Mapping)
        )
        if inherited:
            return inherited
    return []


def _detail_target_match(hit: Mapping[str, object]) -> Mapping[str, object]:
    matches = hit.get("matches", [])
    best_match = {}
    if isinstance(matches, Sequence) and not isinstance(matches, (str, bytes)):
        best_match = max(
            [dict(match) for match in matches if isinstance(match, Mapping)],
            key=lambda match: float(match.get("score", 0.0) or 0.0),
            default={},
        )
    score = float(best_match.get("score", 0.0) or 0.0)
    return {
        "target": str(hit.get("target", "")),
        "source": str(best_match.get("field") or best_match.get("modality") or ""),
        "snippet": str(best_match.get("evidence", "")),
        "score": score,
        "directness": _target_directness(score),
    }


def _target_directness(score: float) -> str:
    if score >= 0.75:
        return "direct_mention"
    if score >= 0.4:
        return "possible_mention"
    return "weak_overlap"


def _detail_recommended_next_tools(
    *,
    segment: VideoMapSegment,
    target_matches: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    if target_matches:
        ask_for = (
            "Openly verify the concrete visible artworks, objects, onscreen text, and narrated events in this segment. "
            "Use the indexed target hints only as possible search cues, not as a yes/no checklist: "
            + "; ".join(str(match.get("target", "")) for match in target_matches if str(match.get("target", "")))
        )
    else:
        ask_for = (
            "Openly describe the concrete visible artworks, objects, onscreen text, and narrated events in this segment. "
            "Do not answer a multiple-choice option; report observations in presentation order."
        )
    return [
        {
            "tool": "vision_read",
            "args": {
                "segment_id": segment.segment_id,
                "start_sec": float(segment.start_sec),
                "end_sec": float(segment.end_sec),
                "ask_for": ask_for,
            },
            "reason": "Use after reading the cheap detail pack when a fresh visual grounding check is needed.",
        }
    ]


def _target_hit_for_segment(*, segment: VideoMapSegment, target: str) -> Mapping[str, object]:
    target_text = str(target).strip()
    target_terms = _target_tokens(target_text)
    field_hits = []
    for field_name, field_value in _detail_search_fields(segment).items():
        value_terms = _target_tokens(field_value)
        overlap = target_terms.intersection(value_terms)
        if not overlap:
            continue
        field_hits.append(
            {
                "field": field_name,
                "modality": _detail_field_modality(field_name),
                "matched_terms": sorted(overlap),
                "evidence": _detail_evidence_snippet(field_value),
                "score": round(len(overlap) / max(len(target_terms), 1), 3),
            }
        )
    return {
        "target": target_text,
        "matched": bool(field_hits),
        "fields": [str(hit["field"]) for hit in field_hits],
        "matches": field_hits,
    }


def _locate_target_candidates(
    *,
    segment: VideoMapSegment,
    targets: Sequence[str],
    top_k_per_target: int = 3,
) -> list[Mapping[str, object]]:
    candidates: list[Mapping[str, object]] = []
    sources = _locate_text_sources(segment)
    per_target_limit = max(1, int(top_k_per_target or 1))
    for target_index, target in enumerate([str(item).strip() for item in targets if str(item).strip()], start=1):
        matches = _find_target_text_matches(target=target, sources=sources, top_k=per_target_limit)
        for match_index, match in enumerate(matches, start=1):
            candidate_id = _locate_candidate_id(
                segment_id=segment.segment_id,
                kind="target",
                index=target_index,
                match=match,
                target=target,
                match_index=match_index,
            )
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "query_id": f"Q{target_index}",
                    "target": target,
                    "source": match["source"],
                    "match_type": match["match_type"],
                    "source_span_start": int(match.get("source_span_start", match.get("match_start", 0)) or 0),
                    "source_span_end": int(match.get("source_span_end", match.get("match_end", 0)) or 0),
                    "timestamp_start": match.get("timestamp_start"),
                    "timestamp_end": match.get("timestamp_end"),
                    "forward_reference": bool(match.get("forward_reference", False)),
                    "start_sec": float(match["start_sec"]),
                    "end_sec": float(match["end_sec"]),
                    "snippet": match["snippet"],
                    "confidence": float(match["confidence"]),
                    "temporal_density": float(match.get("temporal_density", 0.0) or 0.0),
                    "directness": _locate_match_directness(
                        match_type=str(match["match_type"]),
                        confidence=float(match["confidence"]),
                    ),
                }
            )
    for ordered_index, ordered in enumerate(_ordered_list_candidates(segment=segment, sources=sources, targets=targets), start=1):
        candidate_id = _locate_candidate_id(
            segment_id=segment.segment_id,
            kind="ordered_list",
            index=ordered_index,
            match=ordered,
            target="ordered_list",
            match_index=ordered_index,
        )
        ordered_candidate = dict(ordered)
        ordered_candidate["candidate_id"] = candidate_id
        candidates.append(ordered_candidate)
    return candidates


def _locate_candidate_id(
    *,
    segment_id: str,
    kind: str,
    index: int,
    match: Mapping[str, object],
    target: str,
    match_index: int,
) -> str:
    source = re.sub(r"[^A-Za-z0-9]+", "_", str(match.get("source") or "source")).strip("_") or "source"
    start = str(round(float(match.get("start_sec", 0.0) or 0.0), 3)).replace(".", "_")
    char_start = int(match.get("match_start", match.get("start_char", 0)) or 0)
    target_key = re.sub(r"[^A-Za-z0-9]+", "_", _target_text_key(str(target)))[:32].strip("_") or "target"
    segment_key = re.sub(r"[^A-Za-z0-9]+", "_", str(segment_id or "segment")).strip("_") or "segment"
    kind_key = re.sub(r"[^A-Za-z0-9]+", "_", str(kind or "candidate")).strip("_") or "candidate"
    return f"cand_{segment_key}_{kind_key}_{index:02d}_{match_index:02d}_{source}_{start}_{char_start}_{target_key}"


def _locate_text_sources(segment: VideoMapSegment) -> list[Mapping[str, object]]:
    sources: list[Mapping[str, object]] = []
    for sentence in getattr(segment, "asr_sentences", ()) or ():
        if not isinstance(sentence, Mapping):
            continue
        text = str(sentence.get("text") or "").strip()
        if not text:
            continue
        sources.append(
            {
                "source": "asr_sentence",
                "start_sec": float(sentence.get("start_sec", segment.start_sec) or segment.start_sec),
                "end_sec": float(sentence.get("end_sec", segment.end_sec) or segment.end_sec),
                "text": text,
            }
        )
    for frame in getattr(segment, "ocr_frames", ()) or ():
        if not isinstance(frame, Mapping):
            continue
        text = str(frame.get("text") or "").strip()
        if not text:
            continue
        timestamp = float(frame.get("timestamp_sec", segment.start_sec) or segment.start_sec)
        sources.append(
            {
                "source": "ocr_frame",
                "start_sec": timestamp,
                "end_sec": timestamp,
                "text": text,
            }
        )
    fallback_sources = (
        ("asr_text", segment.asr_text),
        ("ocr_text", segment.ocr_text),
        ("visual_caption", segment.low_fps_caption),
        ("entities", " ".join(segment.entities)),
    )
    existing = {(str(item["source"]), str(item["text"])) for item in sources}
    for source_name, text_value in fallback_sources:
        text = str(text_value or "").strip()
        if not text or (source_name, text) in existing:
            continue
        sources.append(
            {
                "source": source_name,
                "start_sec": float(segment.start_sec),
                "end_sec": float(segment.end_sec),
                "text": text,
            }
        )
    return sorted(
        sources,
        key=lambda item: (
            _locate_source_priority(str(item.get("source", ""))),
            float(item["start_sec"]),
            str(item["source"]),
        ),
    )


def _locate_source_priority(source: str) -> int:
    return {
        "asr_sentence": 0,
        "ocr_frame": 1,
        "visual_caption": 2,
        "ocr_text": 3,
        "entities": 4,
        "asr_text": 5,
    }.get(source, 9)


def _best_target_text_match(
    *,
    target: str,
    sources: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    matches = _find_target_text_matches(target=target, sources=sources, top_k=1)
    return matches[0] if matches else None


def _find_target_text_matches(
    *,
    target: str,
    sources: Sequence[Mapping[str, object]],
    top_k: int = 3,
) -> list[Mapping[str, object]]:
    aliases = _target_alias_patterns(target)
    matches: list[dict[str, object]] = []
    for source in sources:
        text = str(source.get("text") or "")
        for alias in aliases:
            pattern = alias["pattern"]
            if not hasattr(pattern, "finditer"):
                continue
            for match in pattern.finditer(text):
                token_tuple = tuple(str(token) for token in alias.get("tokens", ()) if str(token))
                if bool(alias.get("requires_context")) and not _single_token_context_allowed(
                    target_token=token_tuple[0] if token_tuple else str(target).lower(),
                    text=text,
                    start=match.start(),
                    end=match.end(),
                ):
                    continue
                matches.append(
                    {
                        "source": str(source.get("source") or ""),
                        "source_priority": _locate_source_priority(str(source.get("source") or "")),
                        "match_type": str(alias["match_type"]),
                        "match_priority": _locate_match_priority(str(alias["match_type"])),
                        "match_start": int(match.start()),
                        "match_end": int(match.end()),
                        "source_span_start": int(match.start()),
                        "source_span_end": int(match.end()),
                        "timestamp_start": (
                            float(source.get("start_sec", 0.0) or 0.0)
                            if _source_has_concrete_timestamp(source)
                            else None
                        ),
                        "timestamp_end": (
                            float(source.get("end_sec", source.get("start_sec", 0.0)) or 0.0)
                            if _source_has_concrete_timestamp(source)
                            else None
                        ),
                        "forward_reference": _is_forward_reference_span(
                            text=text,
                            start=int(match.start()),
                            end=int(match.end()),
                            has_concrete_coanchor=_source_has_concrete_timestamp(source),
                        ),
                        "start_sec": float(source.get("start_sec", 0.0) or 0.0),
                        "end_sec": float(source.get("end_sec", source.get("start_sec", 0.0)) or 0.0),
                        "snippet": _match_snippet(text, start=match.start(), end=match.end()),
                        "confidence": float(alias["confidence"]),
                    }
                )
    matches = _dedupe_locate_matches(matches)
    if any(int(match.get("source_priority", 9)) <= 1 for match in matches):
        matches = [match for match in matches if int(match.get("source_priority", 9)) <= 1]
    for match in matches:
        match["temporal_density"] = _temporal_density(match=match, matches=matches)
    ranked = sorted(
        matches,
        key=lambda item: (
            int(item.get("source_priority", 9)),
            int(item.get("match_priority", 9)),
            -float(item.get("temporal_density", 0.0) or 0.0),
            -float(item.get("confidence", 0.0) or 0.0),
            float(item.get("start_sec", 0.0) or 0.0),
        ),
    )
    return ranked[: max(1, int(top_k or 1))]


def _find_stemmed_target_text_matches(
    *,
    target: str,
    sources: Sequence[Mapping[str, object]],
    top_k: int = 3,
) -> list[Mapping[str, object]]:
    target_terms = _target_tokens(target)
    if not target_terms:
        return []
    min_overlap = min(2, len(target_terms))
    matches: list[dict[str, object]] = []
    for source in sources:
        text = str(source.get("text") or "")
        spans = list(token_spans(text))
        if not spans:
            continue
        by_term: dict[str, list[tuple[int, int]]] = {}
        for term, start, end in spans:
            by_term.setdefault(term, []).append((start, end))
        overlap = target_terms.intersection(by_term)
        if len(overlap) < min_overlap:
            continue
        selected_spans = [span for term in sorted(overlap) for span in by_term.get(term, [])]
        if not selected_spans:
            continue
        start = min(span[0] for span in selected_spans)
        end = max(span[1] for span in selected_spans)
        confidence = min(0.6, len(overlap) / max(len(target_terms), 1))
        matches.append(
            {
                "source": str(source.get("source") or ""),
                "source_priority": _locate_source_priority(str(source.get("source") or "")),
                "match_type": "stemmed_token_overlap",
                "match_priority": _locate_match_priority("stemmed_token_overlap"),
                "match_start": start,
                "match_end": end,
                "source_span_start": start,
                "source_span_end": end,
                "timestamp_start": (
                    float(source.get("start_sec", 0.0) or 0.0)
                    if _source_has_concrete_timestamp(source)
                    else None
                ),
                "timestamp_end": (
                    float(source.get("end_sec", source.get("start_sec", 0.0)) or 0.0)
                    if _source_has_concrete_timestamp(source)
                    else None
                ),
                "forward_reference": _is_forward_reference_span(
                    text=text,
                    start=start,
                    end=end,
                    has_concrete_coanchor=_source_has_concrete_timestamp(source),
                ),
                "start_sec": float(source.get("start_sec", 0.0) or 0.0),
                "end_sec": float(source.get("end_sec", source.get("start_sec", 0.0)) or 0.0),
                "snippet": _match_snippet(text, start=start, end=end),
                "confidence": confidence,
                "matched_terms": sorted(overlap),
            }
        )
    matches = _dedupe_locate_matches(matches)
    return sorted(
        matches,
        key=lambda item: (
            int(item.get("source_priority", 9)),
            int(item.get("match_priority", 9)),
            -float(item.get("confidence", 0.0) or 0.0),
            float(item.get("start_sec", 0.0) or 0.0),
        ),
    )[: max(1, int(top_k or 1))]


def _ordered_list_candidates(
    *,
    segment: VideoMapSegment,
    sources: Sequence[Mapping[str, object]],
    targets: Sequence[str],
    max_gap_sec: float = 3.0,
    max_window_sec: float = 45.0,
) -> list[Mapping[str, object]]:
    target_list = [str(target).strip() for target in targets if str(target).strip()]
    if len(target_list) < 3:
        return []
    candidates: list[dict[str, object]] = []
    for start_index, first in enumerate(sources):
        pieces = []
        piece_spans: list[Mapping[str, object]] = []
        window_start = float(first.get("start_sec", segment.start_sec) or segment.start_sec)
        window_end = float(first.get("end_sec", window_start) or window_start)
        previous_end = window_end
        source_names: list[str] = []
        for source in sources[start_index:]:
            source_start = float(source.get("start_sec", segment.start_sec) or segment.start_sec)
            source_end = float(source.get("end_sec", source_start) or source_start)
            if pieces and source_start - previous_end > max_gap_sec:
                break
            effective_window_start = _ordered_list_effective_start_sec(
                text=" ".join(pieces),
                source_spans=piece_spans,
                targets=target_list,
                fallback_sec=window_start,
            )
            if pieces and source_end - effective_window_start > max_window_sec:
                break
            text = str(source.get("text") or "").strip()
            if not text:
                continue
            combined_start = sum(len(piece) + 1 for piece in pieces)
            pieces.append(text)
            piece_spans.append(
                {
                    "combined_start": combined_start,
                    "combined_end": combined_start + len(text),
                    "source_start_sec": source_start,
                    "source_end_sec": source_end,
                    "source_text_len": len(text),
                    "source": str(source.get("source") or ""),
                    "text": text,
                    "has_concrete_timestamp": _source_has_concrete_timestamp(source),
                }
            )
            source_name = str(source.get("source") or "")
            if source_name and source_name not in source_names:
                source_names.append(source_name)
            window_end = max(window_end, source_end)
            previous_end = source_end
            combined = " ".join(pieces)
            list_match = _preferred_ordered_list_match(combined, target_list, source_spans=piece_spans)
            ordered_targets = list(list_match.get("ordered_targets", [])) if list_match else []
            if len(ordered_targets) < min(2, len(target_list)):
                continue
            order_source = str(list_match.get("order_source", "text"))
            directness = "ordered_list_navigation" if order_source == "quoted_list" else "text_position_inference"
            route_kind = "ordered_list" if len(ordered_targets) == len(target_list) else "partial_ordered_list"
            if order_source == "quoted_list":
                confidence = 0.98 if len(ordered_targets) == len(target_list) else 0.82
            else:
                confidence = min(0.6, 0.72 + 0.02 * len(ordered_targets))
            list_start_sec, list_end_sec = _combined_char_window(
                int(list_match.get("start_char", 0)),
                int(list_match.get("end_char", len(combined))),
                source_spans=piece_spans,
                fallback_start_sec=window_start,
                fallback_end_sec=window_end,
            )
            candidates.append(
                {
                    "candidate_id": "",
                    "target_id": "ordered_list",
                    "target": "ordered target list",
                    "source": "+".join(source_names) or str(first.get("source") or ""),
                    "match_type": "ordered_list_mention",
                    "start_sec": round(list_start_sec, 3),
                    "end_sec": round(max(list_start_sec, list_end_sec), 3),
                    "snippet": _detail_evidence_snippet(combined, max_chars=240),
                    "confidence": confidence,
                    "temporal_density": float(len(ordered_targets)),
                    "directness": directness,
                    "ordered_targets": ordered_targets,
                    "ordered_target_hits": list(list_match.get("ordered_target_hits", [])),
                    "route_kind": route_kind,
                    "text_span_window": [round(list_start_sec, 3), round(max(list_start_sec, list_end_sec), 3)],
                    "quoted_target_count": int(list_match.get("quoted_target_count", 0)),
                    "target_span_chars": int(list_match.get("target_span_chars", 0)),
                    "list_order_source": order_source,
                    "single_sentence_quoted": bool(list_match.get("single_sentence_quoted", False)),
                }
            )
    return _dedupe_ordered_list_candidates(candidates, target_count=len(target_list))


def _preferred_ordered_list_match(
    text: str,
    targets: Sequence[str],
    *,
    source_spans: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    quoted_mentions = _quoted_target_mentions(text=text, targets=targets, source_spans=source_spans)
    quoted_mentions = [mention for mention in quoted_mentions if not bool(mention.get("forward_reference", False))]
    quoted_window = _best_quoted_target_window(quoted_mentions, text=str(text or ""), target_count=len(targets))
    if quoted_window is not None:
        return quoted_window

    positioned: list[Mapping[str, object]] = []
    for target in targets:
        position = _target_first_position(
            text=text,
            target=target,
            allow_list_context=True,
        )
        if position is None:
            continue
        end = position + len(str(target))
        hit = _ordered_target_hit(
            text=text,
            target=str(target),
            start=position,
            end=end,
            source_spans=source_spans,
        )
        if bool(hit.get("forward_reference", False)):
            continue
        positioned.append(hit)
    if not positioned:
        return None
    positioned = sorted(positioned, key=lambda item: int(item["source_span_start"]))
    return {
        "ordered_targets": [str(hit["target"]) for hit in positioned],
        "ordered_target_hits": positioned,
        "start_char": int(positioned[0]["source_span_start"]),
        "end_char": int(positioned[-1]["source_span_end"]),
        "quoted_target_count": 0,
        "target_span_chars": max(0, int(positioned[-1]["source_span_end"]) - int(positioned[0]["source_span_start"])),
        "order_source": "text_first_position",
    }


def _best_quoted_target_window(
    mentions: Sequence[Mapping[str, object]],
    *,
    text: str,
    target_count: int,
) -> Mapping[str, object] | None:
    min_unique = min(2, int(target_count))
    if len(mentions) < min_unique:
        return None
    best: tuple[tuple[int, int, int, int], Mapping[str, object]] | None = None
    for start_index in range(len(mentions)):
        for end_index in range(start_index, len(mentions)):
            window = mentions[start_index : end_index + 1]
            if not _quoted_window_has_list_continuity(text=text, window=window):
                continue
            ordered_targets = _unique_ordered_targets(window)
            unique_count = len(ordered_targets)
            if unique_count < min_unique:
                continue
            start_char = int(window[0]["start"])
            end_char = int(window[-1]["end"])
            span = max(0, end_char - start_char)
            duplicate_count = max(0, len(window) - unique_count)
            rank = (-unique_count, span, duplicate_count, start_char)
            candidate = {
                "ordered_targets": ordered_targets,
                "ordered_target_hits": _unique_ordered_target_hits(window),
                "start_char": start_char,
                "end_char": end_char,
                "quoted_target_count": len(window),
                "target_span_chars": span,
                "order_source": "quoted_list",
                "single_sentence_quoted": "." not in str(text[start_char:end_char]),
            }
            if best is None or rank < best[0]:
                best = (rank, candidate)
    return best[1] if best is not None else None


def _quoted_window_has_list_continuity(*, text: str, window: Sequence[Mapping[str, object]]) -> bool:
    for left, right in zip(window, window[1:]):
        bridge = str(text or "")[int(left.get("end", 0)) : int(right.get("start", 0))]
        if re.search(r"[.!?]\s+", bridge):
            return False
    return True


def _quoted_target_mentions(
    *,
    text: str,
    targets: Sequence[str],
    source_spans: Sequence[Mapping[str, object]] = (),
) -> list[Mapping[str, object]]:
    target_by_key = {
        _target_text_key(target): str(target).strip()
        for target in targets
        if str(target).strip()
    }
    mentions: list[Mapping[str, object]] = []
    for match in re.finditer(r"[\"“]([^\"”]+)[\"”]", str(text or "")):
        quoted_key = _target_text_key(match.group(1))
        target = target_by_key.get(quoted_key)
        if not target:
            continue
        mentions.append(
            _ordered_target_hit(
                text=str(text or ""),
                target=target,
                start=int(match.start(1)),
                end=int(match.end(1)),
                source_spans=source_spans,
            )
        )
    return sorted(mentions, key=lambda item: int(item["start"]))


def _unique_ordered_targets(mentions: Sequence[Mapping[str, object]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for mention in mentions:
        target = str(mention.get("target", "")).strip()
        key = _target_text_key(target)
        if not target or key in seen:
            continue
        seen.add(key)
        ordered.append(target)
    return ordered


def _unique_ordered_target_hits(mentions: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    ordered: list[Mapping[str, object]] = []
    seen: set[str] = set()
    for mention in sorted(mentions, key=lambda item: int(item.get("source_span_start", item.get("start", 0)))):
        target = str(mention.get("target", "")).strip()
        key = _target_text_key(target)
        if not target or key in seen:
            continue
        seen.add(key)
        ordered.append(dict(mention))
    return ordered


def _ordered_target_hit(
    *,
    text: str,
    target: str,
    start: int,
    end: int,
    source_spans: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    source_span = _source_span_for_combined_char(int(start), source_spans=source_spans)
    has_concrete_timestamp = bool(source_span.get("has_concrete_timestamp", False))
    forward_reference = _is_forward_reference_span(
        text=text,
        start=int(start),
        end=int(end),
        has_concrete_coanchor=has_concrete_timestamp,
    )
    timestamp_start: float | None = None
    timestamp_end: float | None = None
    if has_concrete_timestamp:
        timestamp_start = float(source_span.get("source_start_sec", 0.0) or 0.0)
        timestamp_end = float(source_span.get("source_end_sec", timestamp_start) or timestamp_start)
    return {
        "start": int(start),
        "end": int(end),
        "target": str(target).strip(),
        "source_span_start": int(start),
        "source_span_end": int(end),
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
        "forward_reference": forward_reference,
    }


def _ordered_list_effective_start_sec(
    *,
    text: str,
    source_spans: Sequence[Mapping[str, object]],
    targets: Sequence[str],
    fallback_sec: float,
) -> float:
    mentions = _quoted_target_mentions(text=text, targets=targets)
    if len(mentions) < min(2, len(targets)):
        return float(fallback_sec)
    return _combined_char_time(
        int(mentions[0]["start"]),
        source_spans=source_spans,
        fallback_sec=fallback_sec,
    )


def _combined_char_time(
    char_index: int,
    *,
    source_spans: Sequence[Mapping[str, object]],
    fallback_sec: float,
) -> float:
    for span in source_spans:
        combined_start = int(span.get("combined_start", 0))
        combined_end = int(span.get("combined_end", combined_start))
        if not (combined_start <= int(char_index) <= combined_end):
            continue
        source_start = float(span.get("source_start_sec", fallback_sec) or fallback_sec)
        source_end = float(span.get("source_end_sec", source_start) or source_start)
        source_len = max(1, int(span.get("source_text_len", combined_end - combined_start) or 1))
        offset = min(max(0, int(char_index) - combined_start), source_len)
        return source_start + (source_end - source_start) * (offset / source_len)
    return float(fallback_sec)


def _combined_char_window(
    start_char: int,
    end_char: int,
    *,
    source_spans: Sequence[Mapping[str, object]],
    fallback_start_sec: float,
    fallback_end_sec: float,
) -> tuple[float, float]:
    start_span = _source_span_for_combined_char(int(start_char), source_spans=source_spans)
    end_span = _source_span_for_combined_char(max(int(start_char), int(end_char) - 1), source_spans=source_spans)
    if not start_span or not end_span:
        return float(fallback_start_sec), float(fallback_end_sec)
    return (
        float(start_span.get("source_start_sec", fallback_start_sec) or fallback_start_sec),
        float(end_span.get("source_end_sec", fallback_end_sec) or fallback_end_sec),
    )


def _source_span_for_combined_char(
    char_index: int,
    *,
    source_spans: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    for span in source_spans:
        combined_start = int(span.get("combined_start", 0))
        combined_end = int(span.get("combined_end", combined_start))
        if combined_start <= int(char_index) <= combined_end:
            return span
    return {}


def _source_has_concrete_timestamp(source: Mapping[str, object]) -> bool:
    source_name = str(source.get("source") or "")
    if source_name != "ocr_frame":
        return False
    return float(source.get("start_sec", 0.0) or 0.0) == float(source.get("end_sec", 0.0) or 0.0)


def _is_forward_reference_span(
    *,
    text: str,
    start: int,
    end: int,
    has_concrete_coanchor: bool,
) -> bool:
    if has_concrete_coanchor:
        return False
    sentence_start, sentence_end = _sentence_bounds(text=text, start=start, end=end)
    sentence = str(text or "")[sentence_start:sentence_end]
    relative_start = max(0, int(start) - sentence_start)
    relative_end = max(relative_start, int(end) - sentence_start)
    for marker in _FUTURE_PROSPECTIVE_RE.finditer(sentence):
        if marker.end() <= relative_start and relative_start - marker.end() <= 32:
            return True
        if marker.start() >= relative_end and marker.start() - relative_end <= 32:
            return True
    return False


def _sentence_bounds(*, text: str, start: int, end: int) -> tuple[int, int]:
    value = str(text or "")
    left = max(value.rfind(".", 0, int(start)), value.rfind("!", 0, int(start)), value.rfind("?", 0, int(start)))
    right_candidates = [position for position in (value.find(".", int(end)), value.find("!", int(end)), value.find("?", int(end))) if position >= 0]
    sentence_start = 0 if left < 0 else left + 1
    sentence_end = min(right_candidates) + 1 if right_candidates else len(value)
    return sentence_start, sentence_end


def _target_text_key(text: str) -> str:
    return " ".join(_target_token_list(text))


def _targets_in_text_order(
    text: str,
    targets: Sequence[str],
    *,
    allow_list_context: bool = False,
) -> list[str]:
    positioned: list[tuple[int, str]] = []
    for target in targets:
        position = _target_first_position(
            text=text,
            target=target,
            allow_list_context=allow_list_context,
        )
        if position is not None:
            positioned.append((position, str(target)))
    return [target for _, target in sorted(positioned, key=lambda item: item[0])]


def _target_first_position(*, text: str, target: str, allow_list_context: bool = False) -> int | None:
    positions = []
    for alias in _target_alias_patterns(target):
        pattern = alias["pattern"]
        if not hasattr(pattern, "finditer"):
            continue
        for match in pattern.finditer(text):
            token_tuple = tuple(str(token) for token in alias.get("tokens", ()) if str(token))
            if (
                bool(alias.get("requires_context"))
                and not allow_list_context
                and not _single_token_context_allowed(
                    target_token=token_tuple[0] if token_tuple else str(target).lower(),
                    text=text,
                    start=match.start(),
                    end=match.end(),
                )
            ):
                continue
            positions.append(int(match.start()))
    return min(positions) if positions else None


def _dedupe_ordered_list_candidates(
    candidates: Sequence[Mapping[str, object]],
    *,
    target_count: int,
) -> list[Mapping[str, object]]:
    ranked = sorted(
        [dict(candidate) for candidate in candidates],
        key=lambda candidate: (
            -len(candidate.get("ordered_targets", []) if isinstance(candidate.get("ordered_targets"), list) else []),
            -int(candidate.get("quoted_target_count", 0) or 0),
            float(candidate.get("target_span_chars", 0) or 0),
            float(candidate.get("start_sec", 0.0) or 0.0),
            float(candidate.get("end_sec", 0.0) or 0.0),
        ),
    )
    kept: list[dict[str, object]] = []
    seen: set[tuple[float, tuple[str, ...]]] = set()
    for candidate in ranked:
        ordered_targets = candidate.get("ordered_targets", [])
        if not isinstance(ordered_targets, list):
            continue
        key = (round(float(candidate.get("start_sec", 0.0) or 0.0), 3), tuple(str(item) for item in ordered_targets))
        if key in seen:
            continue
        seen.add(key)
        kept.append(candidate)
        if len(ordered_targets) == target_count:
            break
    return sorted(kept, key=lambda candidate: float(candidate.get("start_sec", 0.0) or 0.0))


def _ordered_list_timeline_rows(
    *,
    segment: VideoMapSegment,
    candidates: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        if str(candidate.get("match_type", "")) != "ordered_list_mention":
            continue
        if str(candidate.get("directness", "")) != "ordered_list_navigation":
            continue
        ordered_targets = candidate.get("ordered_targets", [])
        if not isinstance(ordered_targets, Sequence) or isinstance(ordered_targets, (str, bytes)):
            continue
        targets = [str(target).strip() for target in ordered_targets if str(target).strip()]
        if len(targets) < 2:
            continue
        ordered_hits = candidate.get("ordered_target_hits", [])
        if not isinstance(ordered_hits, Sequence) or isinstance(ordered_hits, (str, bytes)):
            continue
        hit_rows = [dict(hit) for hit in ordered_hits if isinstance(hit, Mapping)]
        if len(hit_rows) != len(targets):
            continue
        if any(hit.get("timestamp_start") is None for hit in hit_rows):
            continue
        start_sec = float(candidate.get("start_sec", segment.start_sec) or segment.start_sec)
        end_sec = float(candidate.get("end_sec", start_sec) or start_sec)
        claim = (
            f"Indexed transcript ordered list in {segment.segment_id} mentions targets in order: "
            + " -> ".join(targets)
        )
        for target, hit in zip(targets, hit_rows):
            rows.append(
                {
                    "entity": target,
                    "observed_at_sec": round(float(hit["timestamp_start"]), 3),
                    "window": [start_sec, end_sec],
                    "confidence_signal": "text_inferred",
                    "claim": claim,
                    "grounding_quality": "indexed_transcript",
                    "requires_visual_verification": True,
                    "source": str(candidate.get("source", "")),
                    "candidate_id": str(candidate.get("candidate_id", "")),
                }
            )
    return rows


def _focused_vision_call_args_for_ordered_candidate(
    *,
    segment: VideoMapSegment,
    candidates: Sequence[Mapping[str, object]],
    pad_sec: float = 2.0,
) -> dict[str, object]:
    ordered_candidates = [
        candidate
        for candidate in candidates
        if str(candidate.get("match_type", "")) == "ordered_list_mention"
    ]
    if not ordered_candidates:
        return {}
    ranked = sorted(
        ordered_candidates,
        key=lambda candidate: (
            -len(candidate.get("ordered_targets", []) if isinstance(candidate.get("ordered_targets", []), list) else []),
            not bool(candidate.get("single_sentence_quoted", False)),
            -float(candidate.get("confidence", 0.0) or 0.0),
            float(candidate.get("end_sec", segment.end_sec) or segment.end_sec)
            - float(candidate.get("start_sec", segment.start_sec) or segment.start_sec),
        ),
    )
    selected = ranked[0]
    start_sec = max(float(segment.start_sec), float(selected.get("start_sec", segment.start_sec) or segment.start_sec) - pad_sec)
    end_sec = min(float(segment.end_sec), float(selected.get("end_sec", start_sec) or start_sec) + pad_sec)
    if end_sec <= start_sec:
        return {}
    if end_sec - start_sec >= 60.0:
        midpoint = (start_sec + end_sec) / 2.0
        start_sec = max(float(segment.start_sec), midpoint - 29.5)
        end_sec = min(float(segment.end_sec), midpoint + 29.5)
    ordered_targets = selected.get("ordered_targets", [])
    target_count = len(ordered_targets) if isinstance(ordered_targets, Sequence) and not isinstance(ordered_targets, (str, bytes)) else 0
    return {
        "segment_id": segment.segment_id,
        "start_sec": round(start_sec, 3),
        "end_sec": round(end_sec, 3),
        "ask_for": (
            "Inspect only this focused scene. Describe the visible artworks and scene transitions "
            "in first-visible timestamp order. Identify only targets supported by visible or narrated "
            "evidence. Do not choose an option. Do not infer names that are not shown or narrated."
        ),
        "event_label": f"focused_ordered_list_candidate_{target_count}_items",
        "nframes": 128,
    }


def _locate_recommended_next_actions(
    *,
    focused_vision_call_args: Mapping[str, object],
    candidates: Sequence[Mapping[str, object]],
    target_refs: Sequence[str],
    ordered_transcript_rows: Sequence[Mapping[str, object]] = (),
    segment_id: str = "",
) -> list[dict[str, object]]:
    ordered_candidate = _selected_ordered_list_candidate(candidates)
    if not ordered_candidate:
        return []
    ordered_targets = ordered_candidate.get("ordered_targets", [])
    ordered_target_texts = [
        str(target).strip()
        for target in (
            ordered_targets
            if isinstance(ordered_targets, Sequence) and not isinstance(ordered_targets, (str, bytes))
            else []
        )
        if str(target).strip()
    ]
    refs = [str(ref).strip() for ref in target_refs if str(ref).strip()]
    complete_rows = [row for row in ordered_transcript_rows if isinstance(row, Mapping)]
    if str(ordered_candidate.get("route_kind") or "") == "partial_ordered_list":
        return [
            {
                "candidate_id": _ordered_candidate_stable_id(
                    segment_id=segment_id,
                    candidate=ordered_candidate,
                    route_kind="partial_ordered_list",
                ),
                "candidate_type": "ordered_list",
                "route_kind": "partial_ordered_list",
                "tool": "locate_targets_in_segment",
                "target_refs": refs,
                "ordered_targets": ordered_target_texts,
                "args": {
                    "segment_id": str(segment_id or ordered_candidate.get("segment_id") or ""),
                    "target_refs": refs,
                },
                "reason": "Only a partial ordered list was found in this text span; do not treat it as a supported sequence relation.",
            }
        ]
    if complete_rows and refs:
        row = dict(complete_rows[0])
        return [
            {
                "candidate_id": _ordered_candidate_stable_id(
                    segment_id=segment_id,
                    candidate=ordered_candidate,
                    route_kind="ordered_list_transcript_complete",
                ),
                "candidate_type": "ordered_list",
                "route_kind": "ordered_list_transcript_complete",
                "tool": "read_segment_detail",
                "target_refs": refs,
                "ordered_targets": ordered_target_texts,
                "evidence_id": str(row.get("evidence_id") or ""),
                "args": {
                    "segment_id": str(segment_id or ordered_candidate.get("segment_id") or ""),
                    "target_refs": refs,
                    "promote_answer_evidence": True,
                },
            }
        ]
    if not focused_vision_call_args:
        return []
    return [
            {
                "candidate_id": _ordered_candidate_stable_id(
                    segment_id=segment_id,
                    candidate=ordered_candidate,
                    route_kind="focused_ordered_list_vision",
                ),
                "candidate_type": "ordered_list",
                "candidate_kind": "ordered_list_visual_candidate",
                "route_kind": "focused_ordered_list_vision",
                "tool": "vision_read",
            "target_refs": refs,
            "ordered_targets": ordered_target_texts,
            "args": dict(focused_vision_call_args),
        }
    ]


def _ordered_candidate_stable_id(
    *,
    segment_id: str,
    candidate: Mapping[str, object],
    route_kind: str,
) -> str:
    raw_id = str(candidate.get("candidate_id") or "").strip()
    start = str(candidate.get("start_sec", "")).replace(".", "_")
    end = str(candidate.get("end_sec", "")).replace(".", "_")
    if raw_id and raw_id.startswith("cand_") and str(segment_id):
        return f"cand_{segment_id}_{route_kind}_{raw_id}_{start}_{end}"
    if raw_id:
        return raw_id
    return f"cand_{segment_id or 'segment'}_{route_kind}_{start}_{end}"


def _locate_recommended_next_tools(
    *,
    focused_vision_call_args: Mapping[str, object],
    verify_call_args: Mapping[str, object],
) -> list[dict[str, object]]:
    recommendations: list[dict[str, object]] = []
    if focused_vision_call_args:
        recommendations.append(
            {
                "tool": "vision_read",
                "args": dict(focused_vision_call_args),
                "reason": "Confirm the ordered-list candidate in a focused single-scene visual window.",
            }
        )
    if verify_call_args:
        recommendations.append(
            {
                "tool": "verify_segment_anchors",
                "args": dict(verify_call_args),
                "reason": "Verify text-located anchors visually before using them as evidence.",
            }
        )
    return recommendations


def _selected_ordered_list_candidate(candidates: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    ordered_candidates = [
        candidate
        for candidate in candidates
        if str(candidate.get("match_type", "")) == "ordered_list_mention"
    ]
    if not ordered_candidates:
        return {}
    return sorted(
        ordered_candidates,
        key=lambda candidate: (
            -len(candidate.get("ordered_targets", []) if isinstance(candidate.get("ordered_targets", []), list) else []),
            not bool(candidate.get("single_sentence_quoted", False)),
            -float(candidate.get("confidence", 0.0) or 0.0),
            float(candidate.get("end_sec", 0.0) or 0.0) - float(candidate.get("start_sec", 0.0) or 0.0),
        ),
    )[0]


def _ordered_candidate_target_refs(
    *,
    candidates: Sequence[Mapping[str, object]],
    workspace: EvidenceWorkspace | None,
) -> list[str]:
    registry = getattr(workspace, "target_registry", None) if workspace is not None else None
    if registry is None:
        return []
    selected = _selected_ordered_list_candidate(candidates)
    ordered_targets = selected.get("ordered_targets", [])
    if not isinstance(ordered_targets, Sequence) or isinstance(ordered_targets, (str, bytes)):
        return []
    refs: list[str] = []
    for target in ordered_targets:
        matches = registry.targets_for_canonical(str(target))
        if len(matches) != 1:
            return []
        refs.append(matches[0].target_id)
    return refs


def _dedupe_locate_matches(matches: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    ranked = sorted(
        [dict(match) for match in matches],
        key=lambda item: (
            int(item.get("source_priority", 9)),
            float(item.get("start_sec", 0.0) or 0.0),
            int(item.get("match_start", 0)),
            int(item.get("match_priority", 9)),
            -float(item.get("confidence", 0.0) or 0.0),
        ),
    )
    kept: list[dict[str, object]] = []
    for match in ranked:
        overlaps_existing = False
        for existing in kept:
            if str(match.get("source")) != str(existing.get("source")):
                continue
            if float(match.get("start_sec", 0.0) or 0.0) != float(existing.get("start_sec", 0.0) or 0.0):
                continue
            if _ranges_overlap(
                int(match.get("match_start", 0)),
                int(match.get("match_end", 0)),
                int(existing.get("match_start", 0)),
                int(existing.get("match_end", 0)),
            ):
                overlaps_existing = True
                break
        if not overlaps_existing:
            kept.append(match)
    return kept


def _ranges_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return max(start_a, start_b) < min(end_a, end_b)


def _target_alias_patterns(target: str) -> list[Mapping[str, object]]:
    tokens = _target_token_list(target)
    if not tokens:
        return []
    aliases: list[tuple[str, str, float, Sequence[str], bool]] = []
    if len(tokens) > 1:
        aliases.append(("full_name", "full_name", 0.95, tokens, False))
    if tokens[0] == "the" and len(tokens) > 1:
        aliases.append(("cleaned_name", "cleaned_name", 0.85, tokens[1:], False))
    if len(tokens) == 1:
        aliases.append(("contextual_single_name", "contextual_single_name", 0.55, tokens, True))

    patterns: list[Mapping[str, object]] = []
    seen: set[str] = set()
    for key, match_type, confidence, alias_tokens, requires_context in aliases:
        token_tuple = tuple(str(token) for token in alias_tokens if str(token))
        if not token_tuple or str((key, token_tuple)) in seen:
            continue
        seen.add(str((key, token_tuple)))
        patterns.append(
            {
                "match_type": match_type,
                "confidence": confidence,
                "tokens": token_tuple,
                "requires_context": requires_context,
                "pattern": re.compile(_token_sequence_regex(token_tuple), flags=re.IGNORECASE),
            }
        )
    return patterns


def _locate_match_priority(match_type: str) -> int:
    return {
        "full_name": 0,
        "cleaned_name": 1,
        "phrase_alias": 2,
        "contextual_single_name": 3,
        "stemmed_token_overlap": 4,
    }.get(str(match_type), 9)


def _temporal_density(*, match: Mapping[str, object], matches: Sequence[Mapping[str, object]], radius_sec: float = 20.0) -> float:
    start = float(match.get("start_sec", 0.0) or 0.0)
    return float(
        sum(
            1
            for other in matches
            if abs(float(other.get("start_sec", 0.0) or 0.0) - start) <= radius_sec
        )
    )


def _single_token_context_allowed(*, target_token: str, text: str, start: int, end: int, context_chars: int = 120) -> bool:
    token = str(target_token or "").lower()
    left = max(0, int(start) - context_chars)
    right = min(len(text), int(end) + context_chars)
    context = " ".join(str(text[left:right]).lower().split())
    context_tokens = set(re.findall(r"[a-z0-9]+", context))
    generic_stopwords = {
        "a",
        "an",
        "and",
        "as",
        "by",
        "for",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "video",
        "with",
    }
    informative_context = {
        item
        for item in context_tokens
        if item != token and len(item) >= 4 and item not in generic_stopwords
    }
    return len(informative_context) >= 2


def _locate_match_directness(*, match_type: str, confidence: float) -> str:
    if match_type == "ordered_list_mention":
        return "ordered_list_navigation"
    if match_type == "contextual_single_name":
        return "routing_only_low_confidence"
    if confidence >= 0.75:
        return "direct_mention"
    if confidence >= 0.4:
        return "possible_mention"
    return "weak_overlap"


def _token_sequence_regex(tokens: Sequence[str]) -> str:
    escaped = [re.escape(str(token)) for token in tokens if str(token)]
    if not escaped:
        return r"$^"
    return r"\b" + r"[\W_]+".join(escaped) + r"\b"


def _target_token_list(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9]+", str(text or ""))]


def _match_snippet(text: str, *, start: int, end: int, context_chars: int = 80) -> str:
    left = max(0, int(start) - context_chars)
    right = min(len(text), int(end) + context_chars)
    prefix = "..." if left > 0 else ""
    suffix = "..." if right < len(text) else ""
    return prefix + " ".join(text[left:right].split()) + suffix


def _merge_locate_candidates(
    *,
    segment: VideoMapSegment,
    candidates: Sequence[Mapping[str, object]],
    merge_gap_sec: float = 15.0,
    padding_sec: float = 5.0,
) -> list[Mapping[str, object]]:
    sorted_candidates = sorted(candidates, key=lambda item: (float(item.get("start_sec", 0.0)), str(item.get("target_id", ""))))
    anchors: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for candidate in sorted_candidates:
        start_sec = float(candidate.get("start_sec", segment.start_sec) or segment.start_sec)
        end_sec = float(candidate.get("end_sec", start_sec) or start_sec)
        if current is None or start_sec - float(current["end_sec"]) > merge_gap_sec:
            current = {
                "anchor_id": f"anchor_{len(anchors) + 1:04d}",
                "segment_id": segment.segment_id,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "targets": [],
                "candidate_ids": [],
                "reason": "",
            }
            anchors.append(current)
        current["end_sec"] = max(float(current["end_sec"]), end_sec)
        targets = current["targets"]
        candidate_ids = current["candidate_ids"]
        candidate_targets = _candidate_anchor_targets(candidate)
        if isinstance(targets, list):
            for target in candidate_targets:
                if target and target not in targets:
                    targets.append(target)
        if isinstance(candidate_ids, list):
            candidate_ids.append(str(candidate.get("candidate_id", "")))
    for anchor in anchors:
        anchor["start_sec"] = max(float(segment.start_sec), float(anchor["start_sec"]) - padding_sec)
        anchor["end_sec"] = min(float(segment.end_sec), float(anchor["end_sec"]) + padding_sec)
        anchor["reason"] = (
            "Text locator found target mention(s) in indexed ASR/OCR/caption sources: "
            + ", ".join(str(target) for target in anchor.get("targets", []))
        )
    return anchors


def _candidate_anchor_targets(candidate: Mapping[str, object]) -> list[str]:
    ordered_targets = candidate.get("ordered_targets")
    if isinstance(ordered_targets, Sequence) and not isinstance(ordered_targets, (str, bytes)):
        return _unique_nonempty_texts(str(target) for target in ordered_targets)
    return _unique_nonempty_texts([str(candidate.get("target", ""))])


def _locate_candidate_label(candidate: Mapping[str, object]) -> str:
    if str(candidate.get("match_type", "")) == "ordered_list_mention":
        ordered_targets = candidate.get("ordered_targets", [])
        if isinstance(ordered_targets, Sequence) and not isinstance(ordered_targets, (str, bytes)):
            return f"ordered_list({len(ordered_targets)} targets)@{float(candidate['start_sec']):.1f}s"
    return f"{candidate['target']}@{float(candidate['start_sec']):.1f}s"


def _target_tokens(text: str) -> set[str]:
    return unique_tokens(text)


def _detail_search_fields(segment: VideoMapSegment) -> Mapping[str, str]:
    return {
        "low_fps_caption": segment.low_fps_caption,
        "asr_text": segment.asr_text,
        "ocr_text": segment.ocr_text,
        "entities": " ".join(segment.entities),
    }


def _detail_field_modality(field_name: str) -> str:
    return {
        "low_fps_caption": "caption",
        "asr_text": "asr",
        "ocr_text": "ocr",
        "entities": "entities",
    }.get(field_name, field_name)


def _detail_evidence_snippet(text: str, max_chars: int = 160) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _unique_nonempty_texts(values: Sequence[str]) -> list[str]:
    items = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        items.append(text)
        seen.add(text)
    return items


def _reject_additional_targets(additional_targets: Sequence[str]) -> None:
    if _unique_nonempty_texts(additional_targets):
        raise ToolError("invalid_tool_args(reason_code=additional_targets_not_allowed)")


def _query_with_additional_targets(*, query: str, additional_targets: Sequence[str]) -> str:
    extras = _unique_nonempty_texts(additional_targets)
    if not extras:
        return str(query)
    base = str(query or "").strip()
    suffix = " ".join(extras)
    return f"{base} {suffix}".strip()


def _coerce_text_sequence(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [str(value)]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return [str(value)]
