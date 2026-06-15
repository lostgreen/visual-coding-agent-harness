"""Runtime metadata installers for video exploration tools."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from ..protocol import ToolRequest
from ..registry import DuplicateGuardPolicy, ToolError, ToolRegistry
from ..agents.runtime.lifecycle import RunContext


def install_video_runtime_specs(registry: ToolRegistry) -> ToolRegistry:
    """Attach lifecycle metadata to the real video exploration tools."""

    _replace(
        registry,
        "bind_asr_claim",
        argument_normalizer=_normalize_bind_asr_claim,
        semantic_key_builder=_key_from_normalizer("bind_asr_claim", _normalize_bind_asr_claim),
        duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
    )
    _replace(
        registry,
        "target_coverage",
        argument_normalizer=_normalize_target_coverage,
        semantic_key_builder=_key_from_normalizer("target_coverage", _normalize_target_coverage),
        duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
    )
    _replace(
        registry,
        "ground_question",
        argument_normalizer=_normalize_ground_question,
        semantic_key_builder=_key_from_normalizer("ground_question", _normalize_ground_question),
        duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
    )
    _replace(
        registry,
        "read_segment_detail",
        argument_normalizer=_normalize_read_segment_detail,
        semantic_key_builder=_key_from_normalizer("read_segment_detail", _normalize_read_segment_detail),
        duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
    )
    _replace(
        registry,
        "vision_read",
        argument_normalizer=_normalize_vision_read,
        semantic_key_builder=_key_from_normalizer("vision_read", _normalize_vision_read),
        duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
    )
    _replace(
        registry,
        "verify_segment_anchors",
        argument_normalizer=_normalize_verify_segment_anchors,
        semantic_key_builder=_key_from_normalizer("verify_segment_anchors", _normalize_verify_segment_anchors),
        duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
    )
    _replace(
        registry,
        "verify_ledger_answer",
        argument_normalizer=_normalize_verify_ledger_answer,
        semantic_key_builder=_key_from_normalizer("verify_ledger_answer", _normalize_verify_ledger_answer),
        duplicate_guard_policy=DuplicateGuardPolicy.STRICT,
    )
    return registry


def _replace(registry: ToolRegistry, name: str, **updates: Any) -> None:
    try:
        registry.replace_runtime_spec(name, **updates)
    except ToolError:
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
