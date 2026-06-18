"""Runtime metadata installers for video exploration tools."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from ..protocol import ToolRequest
from ..registry import DuplicateGuardPolicy, ToolError, ToolRegistry
from ..agents.runtime.lifecycle import RunContext


_CORE_RUNTIME_SPEC_TOOLS = (
    "bind_asr_claim",
    "target_coverage",
    "ground_question",
    "read_segment_detail",
    "vision_read",
    "verify_segment_anchors",
    "verify_ledger_answer",
    "global_gist",
    "query_context",
    "search_segments",
    "caption_segment",
    "read_timeline_sorted",
    "write_memory",
    "read_workspace",
    "commit_observation",
    "reject_observation",
    "defer_observation",
    "no_commit_needed",
    "read_clip",
    "search",
    "list",
    "verify",
    "synthesize_memory",
    "answer",
)


def install_video_runtime_specs(registry: ToolRegistry, *, required: bool = False) -> ToolRegistry:
    """Attach lifecycle metadata to the real video exploration tools."""

    missing: list[str] = []
    _replace(
        registry,
        "bind_asr_claim",
        missing=missing,
        argument_normalizer=_normalize_bind_asr_claim,
        semantic_key_builder=_key_from_normalizer("bind_asr_claim", _normalize_bind_asr_claim),
        duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
    )
    _replace(
        registry,
        "target_coverage",
        missing=missing,
        argument_normalizer=_normalize_target_coverage,
        semantic_key_builder=_key_from_normalizer("target_coverage", _normalize_target_coverage),
        duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
    )
    _replace(
        registry,
        "ground_question",
        missing=missing,
        argument_normalizer=_normalize_ground_question,
        semantic_key_builder=_key_from_normalizer("ground_question", _normalize_ground_question),
        duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
    )
    _replace(
        registry,
        "search_segments",
        missing=missing,
        argument_normalizer=_normalize_search_segments,
        semantic_key_builder=_key_from_normalizer("search_segments", _normalize_search_segments),
        duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
    )
    _replace(
        registry,
        "read_segment_detail",
        missing=missing,
        argument_normalizer=_normalize_read_segment_detail,
        semantic_key_builder=_key_from_normalizer("read_segment_detail", _normalize_read_segment_detail),
        duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
    )
    _replace(
        registry,
        "vision_read",
        missing=missing,
        argument_normalizer=_normalize_vision_read,
        semantic_key_builder=_key_from_normalizer("vision_read", _normalize_vision_read),
        duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
        commit_required=True,
    )
    _replace(
        registry,
        "verify_segment_anchors",
        missing=missing,
        argument_normalizer=_normalize_verify_segment_anchors,
        semantic_key_builder=_key_from_normalizer("verify_segment_anchors", _normalize_verify_segment_anchors),
        duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
        commit_required=True,
    )
    _replace(
        registry,
        "verify_ledger_answer",
        missing=missing,
        argument_normalizer=_normalize_verify_ledger_answer,
        semantic_key_builder=_key_from_normalizer("verify_ledger_answer", _normalize_verify_ledger_answer),
        duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
    )
    _replace(
        registry,
        "global_gist",
        missing=missing,
        argument_normalizer=_normalize_global_gist,
        semantic_key_builder=_key_from_normalizer("global_gist", _normalize_global_gist),
        duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
    )
    _replace(
        registry,
        "query_context",
        missing=missing,
        argument_normalizer=_normalize_query_context,
        semantic_key_builder=_key_from_normalizer("query_context", _normalize_query_context),
        duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
    )
    _replace(
        registry,
        "caption_segment",
        missing=missing,
        argument_normalizer=_normalize_caption_segment,
        semantic_key_builder=_key_from_normalizer("caption_segment", _normalize_caption_segment),
        duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
    )
    _replace(
        registry,
        "read_timeline_sorted",
        missing=missing,
        argument_normalizer=_normalize_no_args,
        semantic_key_builder=_read_timeline_sorted_key,
        duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
    )
    _replace(
        registry,
        "write_memory",
        missing=missing,
        argument_normalizer=_normalize_write_memory,
        semantic_key_builder=_key_from_normalizer("write_memory", _normalize_write_memory),
        duplicate_guard_policy=DuplicateGuardPolicy.OFF,
    )
    _replace(
        registry,
        "read_workspace",
        missing=missing,
        argument_normalizer=_normalize_read_workspace,
        semantic_key_builder=_key_from_normalizer("read_workspace", _normalize_read_workspace),
        duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
    )
    _replace(
        registry,
        "commit_observation",
        missing=missing,
        argument_normalizer=_normalize_commit_observation,
        semantic_key_builder=_key_from_normalizer("commit_observation", _normalize_commit_observation),
        duplicate_guard_policy=DuplicateGuardPolicy.OFF,
    )
    _replace(
        registry,
        "reject_observation",
        missing=missing,
        argument_normalizer=_normalize_observation_reason_disposition,
        semantic_key_builder=_key_from_normalizer("reject_observation", _normalize_observation_reason_disposition),
        duplicate_guard_policy=DuplicateGuardPolicy.OFF,
    )
    _replace(
        registry,
        "defer_observation",
        missing=missing,
        argument_normalizer=_normalize_defer_observation,
        semantic_key_builder=_key_from_normalizer("defer_observation", _normalize_defer_observation),
        duplicate_guard_policy=DuplicateGuardPolicy.OFF,
    )
    _replace(
        registry,
        "no_commit_needed",
        missing=missing,
        argument_normalizer=_normalize_observation_reason_disposition,
        semantic_key_builder=_key_from_normalizer("no_commit_needed", _normalize_observation_reason_disposition),
        duplicate_guard_policy=DuplicateGuardPolicy.OFF,
    )
    _replace(
        registry,
        "read_clip",
        missing=missing,
        argument_normalizer=_normalize_read_clip,
        semantic_key_builder=_key_from_normalizer("read_clip", _normalize_read_clip),
        duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
        commit_required=True,
    )
    _replace(
        registry,
        "search",
        missing=missing,
        argument_normalizer=_normalize_workspace_v2_search,
        semantic_key_builder=_key_from_normalizer("search", _normalize_workspace_v2_search),
        duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
    )
    _replace(
        registry,
        "list",
        missing=missing,
        argument_normalizer=_normalize_workspace_v2_list,
        semantic_key_builder=_key_from_normalizer("list", _normalize_workspace_v2_list),
        duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
    )
    _replace(
        registry,
        "verify",
        missing=missing,
        argument_normalizer=_normalize_workspace_v2_verify,
        semantic_key_builder=_key_from_normalizer("verify", _normalize_workspace_v2_verify),
        duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
        commit_required=True,
    )
    _replace(
        registry,
        "synthesize_memory",
        missing=missing,
        argument_normalizer=_normalize_workspace_v2_synthesize_memory,
        semantic_key_builder=_key_from_normalizer("synthesize_memory", _normalize_workspace_v2_synthesize_memory),
        duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
    )
    _replace(
        registry,
        "answer",
        missing=missing,
        argument_normalizer=_normalize_workspace_v2_answer,
        semantic_key_builder=_key_from_normalizer("answer", _normalize_workspace_v2_answer),
        duplicate_guard_policy=DuplicateGuardPolicy.OFF,
    )
    if required:
        required_missing = [tool_name for tool_name in _CORE_RUNTIME_SPEC_TOOLS if tool_name in missing]
        if required_missing:
            raise ToolError("Missing required runtime spec tools: " + ", ".join(required_missing))
    return registry


def _replace(registry: ToolRegistry, name: str, *, missing: list[str], **updates: Any) -> None:
    try:
        registry.replace_runtime_spec(name, **updates)
    except ToolError:
        missing.append(name)
        return


def _key_from_normalizer(tool_name: str, normalizer: Any):
    def build(ctx: RunContext, request: ToolRequest) -> str:
        normalized = normalizer(ctx, request)
        return f"{tool_name}:{_canonical_json(normalized)}"

    return build


def _normalize_bind_asr_claim(_ctx: RunContext, request: ToolRequest) -> Mapping[str, Any]:
    args = dict(request.arguments)
    return {
        "segment_id": _text(args.get("segment_id")),
        "target_refs": _string_list(args.get("target_refs")),
    }


def _normalize_target_coverage(_ctx: RunContext, request: ToolRequest) -> Mapping[str, Any]:
    args = dict(request.arguments)
    return {
        "targets": _string_list(args.get("targets")),
        "target_refs": _string_list(args.get("target_refs")),
        "additional_targets": _string_list(args.get("additional_targets")),
        "top_k": _positive_int(args.get("top_k"), default=3),
        "modalities": _string_list(args.get("modalities")),
        "group_by_option": _bool(args.get("group_by_option")),
    }


def _normalize_ground_question(_ctx: RunContext, request: ToolRequest) -> Mapping[str, Any]:
    args = dict(request.arguments)
    return {
        "query": _text(args.get("query")),
        "top_k": _positive_int(args.get("top_k"), default=5),
        "modalities": _string_list(args.get("modalities")),
    }


def _normalize_search_segments(_ctx: RunContext, request: ToolRequest) -> Mapping[str, Any]:
    args = dict(request.arguments)
    return {
        "query": _text(args.get("query")),
        "top_k": _positive_int(args.get("top_k"), default=5),
        "modalities": _string_list(args.get("modalities")),
        "additional_targets": _string_list(args.get("additional_targets")),
    }


def _normalize_read_segment_detail(_ctx: RunContext, request: ToolRequest) -> Mapping[str, Any]:
    args = dict(request.arguments)
    return {
        "segment_id": _text(args.get("segment_id")),
        "targets": _string_list(args.get("targets")),
        "target_refs": _string_list(args.get("target_refs")),
        "additional_targets": _string_list(args.get("additional_targets")),
        "promote_answer_evidence": _bool(args.get("promote_answer_evidence")),
        "option_targets": _string_list_mapping(args.get("option_targets")),
    }


def _normalize_vision_read(_ctx: RunContext, request: ToolRequest) -> Mapping[str, Any]:
    args = dict(request.arguments)
    normalized: dict[str, Any] = {
        "video_path": _text(args.get("video_path")),
        "segment_id": _text(args.get("segment_id")),
        "start_sec": _float(args.get("start_sec")),
        "end_sec": _float(args.get("end_sec")),
        "ask_for": _text(args.get("ask_for")),
        "additional_targets": _string_list(args.get("additional_targets")),
        "event_label": _text(args.get("event_label")),
        "max_pixels": _positive_int(args.get("max_pixels"), default=360 * 420),
        "fps": _float(args.get("fps")),
        "mutex_group_id": _text(args.get("mutex_group_id")),
        "mutex_option_x": _text(args.get("mutex_option_x")),
        "mutex_option_x_text": _text(args.get("mutex_option_x_text")),
        "mutex_option_y": _text(args.get("mutex_option_y")),
        "mutex_option_y_text": _text(args.get("mutex_option_y_text")),
    }
    if args.get("nframes") is not None:
        normalized["nframes"] = _positive_int(args.get("nframes"), default=8)
    return normalized


def _normalize_global_gist(_ctx: RunContext, request: ToolRequest) -> Mapping[str, Any]:
    args = dict(request.arguments)
    normalized: dict[str, Any] = {
        "video_path": _text(args.get("video_path")),
        "question": _text(args.get("question")),
        "duration_sec": _float(args.get("duration_sec")),
        "max_pixels": _positive_int(args.get("max_pixels"), default=151200),
        "sample_offset_sec": _float(args.get("sample_offset_sec")),
    }
    if args.get("nframes") is not None:
        normalized["nframes"] = _positive_int(args.get("nframes"), default=8)
    return normalized


def _normalize_query_context(_ctx: RunContext, request: ToolRequest) -> Mapping[str, Any]:
    args = dict(request.arguments)
    normalized: dict[str, Any] = {
        "video_path": _text(args.get("video_path")),
        "query": _text(args.get("query")),
        "scope": _text(args.get("scope")) or "full_video",
        "max_pixels": _positive_int(args.get("max_pixels"), default=151200),
    }
    if args.get("duration_sec") is not None:
        normalized["duration_sec"] = _float(args.get("duration_sec"))
    if args.get("nframes") is not None:
        normalized["nframes"] = _positive_int(args.get("nframes"), default=8)
    return normalized


def _normalize_caption_segment(_ctx: RunContext, request: ToolRequest) -> Mapping[str, Any]:
    args = dict(request.arguments)
    normalized: dict[str, Any] = {
        "video_path": _text(args.get("video_path")),
        "segment_id": _text(args.get("segment_id")),
        "start_sec": _float(args.get("start_sec")),
        "end_sec": _float(args.get("end_sec")),
        "question": _text(args.get("question")) or "Describe this video segment.",
        "max_pixels": _positive_int(args.get("max_pixels"), default=360 * 420),
        "fps": _float(args.get("fps")),
    }
    if args.get("nframes") is not None:
        normalized["nframes"] = _positive_int(args.get("nframes"), default=8)
    return normalized


def _normalize_write_memory(_ctx: RunContext, request: ToolRequest) -> Mapping[str, Any]:
    args = dict(request.arguments)
    return {
        "kind": _text(args.get("kind")) or "note",
        "claim": _text(args.get("claim")),
        "anchor_ids": [
            _text(anchor.get("anchor_id"))
            for anchor in _mapping_list(args.get("anchors"))
            if _text(anchor.get("anchor_id"))
        ],
        "supports_option": _text(args.get("supports_option")),
        "confidence": _text(args.get("confidence")) or "medium",
        "previous_memory_refs": _string_list(args.get("previous_memory_refs")),
        "tags": _string_list(args.get("tags")),
        "role": _text(args.get("role")),
        "layer": _text(args.get("layer")),
        "embedding_refs": _string_list(args.get("embedding_refs")),
        "metadata": _canonical_value(args.get("metadata")) if isinstance(args.get("metadata"), Mapping) else {},
    }


def _normalize_read_workspace(_ctx: RunContext, request: ToolRequest) -> Mapping[str, Any]:
    args = dict(request.arguments)
    return {
        "section": _text(args.get("section")),
        "filter": _canonical_value(args.get("filter")) if isinstance(args.get("filter"), Mapping) else {},
    }


def _normalize_commit_observation(_ctx: RunContext, request: ToolRequest) -> Mapping[str, Any]:
    args = dict(request.arguments)
    return {
        "observation_id": _text(args.get("observation_id")),
        "writes": _canonical_value(args.get("writes")) if isinstance(args.get("writes"), Mapping) else {},
    }


def _normalize_observation_reason_disposition(_ctx: RunContext, request: ToolRequest) -> Mapping[str, Any]:
    args = dict(request.arguments)
    return {
        "observation_id": _text(args.get("observation_id")),
        "reason": _text(args.get("reason")),
    }


def _normalize_defer_observation(_ctx: RunContext, request: ToolRequest) -> Mapping[str, Any]:
    args = dict(request.arguments)
    return {
        "observation_id": _text(args.get("observation_id")),
        "until": _text(args.get("until")),
        "reason": _text(args.get("reason")),
    }


def _normalize_read_clip(_ctx: RunContext, request: ToolRequest) -> Mapping[str, Any]:
    args = dict(request.arguments)
    scope = args.get("scope") if isinstance(args.get("scope"), Mapping) else {}
    sampling = args.get("sampling") if isinstance(args.get("sampling"), Mapping) else {}
    return {
        "scope": _canonical_value(scope),
        "focus": _string_list(args.get("focus")),
        "sampling": _canonical_value(sampling),
    }


def _normalize_workspace_v2_search(_ctx: RunContext, request: ToolRequest) -> Mapping[str, Any]:
    args = dict(request.arguments)
    scope = args.get("scope") if isinstance(args.get("scope"), Mapping) else {}
    return {
        "query": _text(args.get("query")),
        "modality": _string_list(args.get("modality")),
        "scope": _canonical_value(scope),
        "top_k": _positive_int(args.get("top_k"), default=5),
    }


def _normalize_workspace_v2_list(_ctx: RunContext, request: ToolRequest) -> Mapping[str, Any]:
    args = dict(request.arguments)
    filter_payload = args.get("filter") if isinstance(args.get("filter"), Mapping) else {}
    return {
        "kind": _text(args.get("kind")),
        "filter": _canonical_value(filter_payload),
    }


def _normalize_workspace_v2_verify(_ctx: RunContext, request: ToolRequest) -> Mapping[str, Any]:
    args = dict(request.arguments)
    against = args.get("against") if isinstance(args.get("against"), Mapping) else {}
    return {
        "claim": _text(args.get("claim")),
        "against": _canonical_value(against),
    }


def _normalize_workspace_v2_synthesize_memory(_ctx: RunContext, request: ToolRequest) -> Mapping[str, Any]:
    args = dict(request.arguments)
    return {
        "claim": _text(args.get("claim")),
        "supports": _string_list(args.get("supports")),
        "derived_from": _string_list(args.get("derived_from")),
        "evidence_obs_ids": _string_list(args.get("evidence_obs_ids")),
        "confidence": _text(args.get("confidence")) or "medium",
        "supports_option": _text(args.get("supports_option")),
        "tags": _string_list(args.get("tags")),
    }


def _normalize_workspace_v2_answer(_ctx: RunContext, request: ToolRequest) -> Mapping[str, Any]:
    args = dict(request.arguments)
    return {
        "text": _text(args.get("text")),
        "citations": _string_list(args.get("citations")),
        "confidence": _text(args.get("confidence")) or "medium",
    }


def _normalize_no_args(_ctx: RunContext, _request: ToolRequest) -> Mapping[str, Any]:
    return {}


def _read_timeline_sorted_key(ctx: RunContext, request: ToolRequest) -> str:
    del request
    entries = ctx.workspace.read_timeline_sorted() if ctx.workspace is not None else []
    tail = entries[-1] if entries else {}
    tail_key = _canonical_json(tail if isinstance(tail, Mapping) else {})
    return f"read_timeline_sorted:{len(entries)}:{tail_key}"


def _normalize_verify_segment_anchors(_ctx: RunContext, request: ToolRequest) -> Mapping[str, Any]:
    args = dict(request.arguments)
    normalized: dict[str, Any] = {
        "video_path": _text(args.get("video_path")),
        "segment_id": _text(args.get("segment_id")),
        "anchors": _mapping_list(args.get("anchors")),
        "question": _text(args.get("question")),
        "targets": _string_list(args.get("targets")),
        "target_refs": _string_list(args.get("target_refs")),
        "additional_targets": _string_list(args.get("additional_targets")),
        "start_sec": _float(args.get("start_sec")),
        "end_sec": _float(args.get("end_sec")),
        "max_pixels": _positive_int(args.get("max_pixels"), default=360 * 420),
        "fps": _float(args.get("fps")),
    }
    if args.get("nframes") is not None:
        normalized["nframes"] = _positive_int(args.get("nframes"), default=8)
    return normalized


def _normalize_verify_ledger_answer(_ctx: RunContext, request: ToolRequest) -> Mapping[str, Any]:
    args = dict(request.arguments)
    return {
        "answer": _text(args.get("answer")),
        "ledger_text": _text(args.get("ledger_text")),
        "question": _text(args.get("question")),
        "candidate_options": _string_list(args.get("candidate_options")),
        "target_refs": _string_list(args.get("target_refs")),
        "min_score": _float(args.get("min_score"), default=0.6),
        "required_citations": _string_list(args.get("required_citations")),
        "requires_visual_evidence": _bool(args.get("requires_visual_evidence"), default=True),
    }


def _text(value: Any) -> str:
    return str(value or "").strip()


def _string_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw_items: Sequence[Any] = (value,)
    elif isinstance(value, Sequence):
        raw_items = value
    else:
        raw_items = (value,)
    items: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = _text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        items.append(text)
    return tuple(items)


def _string_list_mapping(value: Any) -> Mapping[str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): _string_list(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}


def _mapping_list(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    items: list[Mapping[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            items.append({str(key): _canonical_value(child) for key, child in item.items()})
    return tuple(items)


def _positive_int(value: Any, *, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, number)


def _float(value: Any, *, default: float = 0.0) -> float:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return default


def _bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(child) for key, child in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_canonical_value(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
