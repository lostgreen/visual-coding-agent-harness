"""Segment Inspector subagent boundary for long-video context control."""

from __future__ import annotations

import json
import re
from typing import Mapping, Optional, Sequence

from ..agents.contracts import resolve_nframes
from ..agents.open_questions import exploration_question
from ..agents.output_quality import confidence_signal_from_text
from ..backends.base import BackendRequest, VisionLanguageBackend
from ..registry import ToolRegistry, tool
from ..workspace import EvidenceWorkspace
from .segments import ClipExtractor, _clip_output_path, _extract_clip_ffmpeg


_MAX_VERIFY_ANCHOR_UNION_SEC = 45.0


def build_segment_inspector_registry(
    backend: VisionLanguageBackend,
    *,
    workspace: Optional[EvidenceWorkspace] = None,
    extract_clips: bool = False,
    clip_extractor: Optional[ClipExtractor] = None,
) -> ToolRegistry:
    registry = ToolRegistry()

    @tool(name="inspect_segment", description="Delegate localized visual inspection to an isolated Segment Inspector.")
    def inspect_segment(
        video_path: str,
        segment_id: str,
        start_sec: float,
        end_sec: float,
        question: str,
        candidate_options: Sequence[str] = (),
        nframes: int | None = None,
        max_pixels: int = 360 * 420,
        fps: float = 0.0,
    ) -> Mapping[str, object]:
        sanitized_question = exploration_question(question)
        return _run_inspector(
            backend=backend,
            video_path=video_path,
            segment_id=segment_id,
            start_sec=start_sec,
            end_sec=end_sec,
            question=sanitized_question,
            candidate_options=(),
            original_question=question,
            original_candidate_options=candidate_options,
            nframes=nframes,
            max_pixels=max_pixels,
            fps=fps,
            workspace=workspace,
            extract_clips=extract_clips,
            clip_extractor=clip_extractor,
        )

    @tool(name="vision_read", description="Read typed visual facts from one localized time window without option voting.")
    def vision_read(
        video_path: str,
        segment_id: str,
        start_sec: float,
        end_sec: float,
        ask_for: str,
        event_label: str = "",
        nframes: int | None = None,
        max_pixels: int = 360 * 420,
        fps: float = 0.0,
        mutex_group_id: str = "",
        mutex_option_x: str = "",
        mutex_option_x_text: str = "",
        mutex_option_y: str = "",
        mutex_option_y_text: str = "",
    ) -> Mapping[str, object]:
        mutex_prompt = ""
        if mutex_option_x_text and mutex_option_y_text:
            mutex_prompt = _mutex_read_prompt(
                segment_id=segment_id,
                start_sec=float(start_sec),
                end_sec=float(end_sec),
                option_x=mutex_option_x or "X",
                option_x_text=mutex_option_x_text,
                option_y=mutex_option_y or "Y",
                option_y_text=mutex_option_y_text,
            )
        result = dict(
            _run_inspector(
                backend=backend,
                video_path=video_path,
                segment_id=segment_id,
                start_sec=start_sec,
                end_sec=end_sec,
                question=_sanitize_vision_read_ask_for(ask_for),
                candidate_options=(),
                original_question=ask_for,
                nframes=nframes,
                max_pixels=max_pixels,
                fps=fps,
                workspace=workspace,
                extract_clips=extract_clips,
                clip_extractor=clip_extractor,
                task_name="vision_read",
                prompt_style="vision_read",
                prompt_override=mutex_prompt,
            )
        )
        resolved_event = event_label or ask_for
        time_range = [float(start_sec), float(end_sec)]
        confidence_signal = str(result.get("confidence_signal", ""))
        grounding_quality = str(result.get("grounding_quality") or "visually_confirmed")
        polarity = "unknown" if confidence_signal == "unsupported" else "present"
        result.update(
            {
                "facts": [
                    {
                        "fact": str(result.get("claim", "")),
                        "event_label": resolved_event,
                        "polarity": polarity,
                        "time_range": time_range,
                        "grounding_quality": grounding_quality,
                    }
                ],
                "event_label": resolved_event,
                "time_range": time_range,
                "grounding_quality": grounding_quality,
            }
        )
        if confidence_signal:
            result["confidence_signal"] = confidence_signal
        ordered_visible = _ordered_visible_from_verification_text(str(result.get("claim", "")), {})
        if ordered_visible:
            result["ordered_visible_in_window"] = ordered_visible
        if mutex_group_id:
            result["mutex_group_id"] = str(mutex_group_id)
        if mutex_prompt:
            supported_option = _mutex_supported_option(
                str(result.get("claim", "")),
                option_x=mutex_option_x or "X",
                option_y=mutex_option_y or "Y",
            )
            result["supported_option"] = supported_option
            if supported_option:
                result["candidate_option_relations"] = [
                    {
                        "option": supported_option,
                        "relation": "support",
                        "strength": float(result.get("confidence", 0.0) or 0.0),
                        "assigned_by": "vision_read_mutex",
                    }
                ]
        return result

    @tool(name="verify_segment_anchors", description="Verify text-located target anchors with a focused visual read.")
    def verify_segment_anchors(
        video_path: str,
        segment_id: str,
        anchors: Sequence[Mapping[str, object]],
        question: str = "",
        targets: Sequence[str] = (),
        start_sec: float = 0.0,
        end_sec: float = 0.0,
        nframes: int | None = 8,
        max_pixels: int = 360 * 420,
        fps: float = 0.0,
    ) -> Mapping[str, object]:
        anchor_list = [dict(anchor) for anchor in anchors if isinstance(anchor, Mapping)]
        _validate_anchor_segment_ids(segment_id=segment_id, anchors=anchor_list)
        anchor_groups = _anchor_verify_groups(
            anchors=anchor_list,
            fallback_start=float(start_sec),
            fallback_end=float(end_sec),
            max_union_sec=_MAX_VERIFY_ANCHOR_UNION_SEC,
        )
        raw_results: list[dict[str, object]] = []
        confirmations: list[dict[str, object]] = []
        rejections: list[dict[str, object]] = []
        timeline_rows: list[dict[str, object]] = []
        ordered_visible_in_window: list[str] = []
        verify_windows: list[dict[str, object]] = []
        for anchor_group in anchor_groups:
            window_start, window_end = _anchor_union_window(
                anchors=anchor_group,
                fallback_start=float(start_sec),
                fallback_end=float(end_sec),
            )
            prompt = _verify_segment_anchors_prompt(
                segment_id=segment_id,
                anchors=anchor_group,
                question=question,
                targets=targets,
            )
            raw_result = dict(
                _run_inspector(
                    backend=backend,
                    video_path=video_path,
                    segment_id=segment_id,
                    start_sec=window_start,
                    end_sec=window_end,
                    question=question or "Verify target anchors.",
                    candidate_options=(),
                    nframes=nframes,
                    max_pixels=max_pixels,
                    fps=fps,
                    workspace=workspace,
                    extract_clips=extract_clips,
                    clip_extractor=clip_extractor,
                    task_name="verify_segment_anchors",
                    prompt_style="vision_read",
                    prompt_override=prompt,
                    max_new_tokens=768,
                    nframes_tool_cap=8,
                )
            )
            raw_results.append(raw_result)
            parsed = _parse_anchor_verification(raw_result.get("claim", ""))
            group_confirmations = parsed["confirmations"]
            group_ordered_visible = [
                str(item).strip()
                for item in parsed.get("ordered_visible_in_window", [])
                if str(item).strip()
            ]
            ordered_visible_in_window.extend(
                item for item in group_ordered_visible if item not in ordered_visible_in_window
            )
            confirmations.extend(group_confirmations)
            rejections.extend(parsed["rejections"])
            timeline_rows.extend(
                _timeline_rows_from_confirmations(
                    confirmations=group_confirmations,
                    segment_id=segment_id,
                    fallback_window=[window_start, window_end],
                    ordered_visible=group_ordered_visible,
                )
            )
            verify_windows.append(
                {
                    "start_sec": window_start,
                    "end_sec": window_end,
                    "anchor_ids": [
                        str(anchor.get("anchor_id", ""))
                        for anchor in anchor_group
                        if str(anchor.get("anchor_id", ""))
                    ],
                    "targets": _anchor_group_targets(anchor_group),
                }
            )
        merged_result = dict(raw_results[0]) if raw_results else {
            "input_artifacts": [video_path],
            "regions": [],
            "raw_backend": {},
        }
        merged_result["input_artifacts"] = [
            artifact
            for result in raw_results
            for artifact in result.get("input_artifacts", [])
            if isinstance(artifact, str)
        ]
        merged_result["regions"] = [
            region
            for result in raw_results
            for region in result.get("regions", [])
            if isinstance(region, Mapping)
        ]
        merged_result["raw_backend"] = [dict(result.get("raw_backend", {})) for result in raw_results]
        claim = (
            f"verify_segment_anchors({segment_id}, {len(anchor_list)} anchors): "
            f"confirmed {len(confirmations)} / rejected {len(rejections)}."
        )
        if timeline_rows:
            claim += " Confirmed targets: " + ", ".join(str(row["entity"]) for row in timeline_rows) + "."
        split_note = (
            f" split {len(anchor_groups)} anchor windows because union exceeded {_MAX_VERIFY_ANCHOR_UNION_SEC:.0f}s."
            if len(anchor_groups) > 1
            else ""
        )
        merged_result.update(
            {
                "claim": claim,
                "confidence": 0.82 if confirmations else 0.45,
                "confirmations": confirmations,
                "rejections": rejections,
                "timeline_rows": timeline_rows,
                "ordered_visible_in_window": ordered_visible_in_window,
                "anchors": anchor_list,
                "targets": list(targets),
                "verify_windows": verify_windows,
                "grounding_quality": "visually_confirmed" if confirmations else "inferred",
                "limitations": (
                    "Focused VLM verification over locator-proposed anchors; use confirmations, not rejected targets, as evidence."
                    + split_note
                ),
            }
        )
        return merged_result

    registry.register(inspect_segment)
    registry.register(vision_read)
    registry.register(verify_segment_anchors)
    return registry


def _run_inspector(
    *,
    backend: VisionLanguageBackend,
    video_path: str,
    segment_id: str,
    start_sec: float,
    end_sec: float,
    question: str,
    candidate_options: Sequence[str],
    nframes: int | None,
    max_pixels: int,
    fps: float,
    workspace: Optional[EvidenceWorkspace],
    extract_clips: bool,
    clip_extractor: Optional[ClipExtractor],
    task_name: str = "inspect_segment",
    prompt_style: str = "inspect_segment",
    prompt_override: str = "",
    original_question: str | None = None,
    original_candidate_options: Sequence[str] = (),
    max_new_tokens: int = 512,
    nframes_tool_cap: int | None = None,
) -> Mapping[str, object]:
    resolved_nframes, _ = resolve_nframes(nframes, tool_cap=nframes_tool_cap)
    metadata = {
        "tool_role": "segment_inspector",
        "context_tier": "isolated_subagent",
        "segment_id": segment_id,
        "start_sec": float(start_sec),
        "end_sec": float(end_sec),
        "question": question,
        "candidate_options": list(candidate_options),
        "nframes": int(resolved_nframes),
        "max_pixels": int(max_pixels),
    }
    if fps > 0:
        metadata["fps"] = float(fps)
    max_new_tokens = 256
    if task_name == "vision_read":
        metadata["repetition_penalty"] = 1.15
        metadata["no_repeat_ngram_size"] = 6
        max_new_tokens = 384

    media_path = video_path
    input_artifacts = [f"{video_path}#t={float(start_sec):.3f},{float(end_sec):.3f}"]
    limitations = (
        "Inspector distilled one localized observation; intermediate visual reasoning stays outside the main context."
    )

    if extract_clips:
        if workspace is None:
            raise ValueError("extract_clips=True requires an EvidenceWorkspace")
        output_path = _clip_output_path(
            workspace=workspace,
            segment_id=segment_id,
            start_sec=float(start_sec),
            end_sec=float(end_sec),
        )
        extractor = clip_extractor or _extract_clip_ffmpeg
        clip_path = extractor(video_path, str(output_path), float(start_sec), float(end_sec))
        media_path = clip_path
        input_artifacts = [clip_path]
        metadata["source_video_path"] = video_path
        metadata["clip_path"] = clip_path
        limitations = (
            "Inspector used a physical segment clip and returned one distilled observation; "
            "fine-grained motion may still be limited by sampling."
        )

    response = backend.generate(
        BackendRequest(
            task=task_name,
            prompt=prompt_override
            or (
                _vision_read_prompt(
                    segment_id=segment_id,
                    start_sec=float(start_sec),
                    end_sec=float(end_sec),
                    ask_for=question,
                )
                if prompt_style == "vision_read"
                else _inspector_prompt(
                    segment_id=segment_id,
                    start_sec=float(start_sec),
                    end_sec=float(end_sec),
                    question=question,
                    candidate_options=candidate_options,
                )
            ),
            media_path=media_path,
            media_type="video",
            max_new_tokens=max_new_tokens,
            metadata=metadata,
        )
    )
    claim = response.text.strip()
    confidence_signal = confidence_signal_from_text(claim)
    grounding_quality = "inferred" if confidence_signal == "unsupported" else ""
    result: dict[str, object] = {
        "claim": claim,
        "confidence": 0.74,
        "input_artifacts": input_artifacts,
        "regions": [metadata],
        "limitations": limitations,
        "raw_backend": dict(response.raw),
    }
    if confidence_signal:
        result["confidence_signal"] = confidence_signal
    if grounding_quality:
        result["grounding_quality"] = grounding_quality
    return result


def _inspector_prompt(
    *,
    segment_id: str,
    start_sec: float,
    end_sec: float,
    question: str,
    candidate_options: Sequence[str],
) -> str:
    prompt = (
        "You are a Segment Inspector subagent for long-video reasoning.\n"
        "Your context is intentionally isolated from the master planner.\n"
        "Inspect only the provided time window and do not rely on outside video context.\n"
        "Return one distilled local observation: visible facts, confidence, and limitation.\n"
        "Use candidate options only to understand what facts to look for.\n"
        "Do not choose an option. Do not emit supported_option, answer_option, or final_answer.\n"
        "Do not include step-by-step reasoning or raw frame descriptions unless essential evidence.\n"
        f"Segment: {segment_id} [{start_sec:.3f}s, {end_sec:.3f}s]\n"
        f"Question: {question}"
    )
    if candidate_options:
        options_text = "\n".join(str(option) for option in candidate_options)
        prompt += f"\nCandidate options:\n{options_text}"
    return prompt


def _vision_read_prompt(
    *,
    segment_id: str,
    start_sec: float,
    end_sec: float,
    ask_for: str,
) -> str:
    return (
        "You are a v4 VisionAgent reading one localized long-video window.\n"
        "Return typed visual facts only: fact, event label if present, polarity, timestamp, and limitation.\n"
        "If multiple requested items are visibly confirmed in this window, add ORDERED_VISIBLE: item1 -> item2 -> item3 using first-visible order.\n"
        "Do not choose an option. Do not emit supported_option, answer_option, or final_answer.\n"
        "Do not use outside video context or external knowledge.\n"
        f"Segment: {segment_id} [{start_sec:.3f}s, {end_sec:.3f}s]\n"
        f"Ask for: {ask_for}"
    )


def _verify_segment_anchors_prompt(
    *,
    segment_id: str,
    anchors: Sequence[Mapping[str, object]],
    question: str,
    targets: Sequence[str],
) -> str:
    lines = [
        "You are a focused visual verifier for target anchors in one video segment.",
        "Inspect only the supplied anchor windows. Use the anchor reason only as a hint, not as proof.",
        "Confirm targets only when visible, narrated, OCR-visible, or visually identifiable in this window.",
        "Return JSON only with keys confirmations and rejections.",
        "Each confirmation should include target, relative_sec if possible, observed_at_sec if possible, and evidence.",
        "Each rejection should include target and reason.",
        "After JSON, add ORDERED_VISIBLE: item1 -> item2 -> item3 using only confirmed visible targets in first-visible order.",
        "Use relative seconds within each anchor when possible; do not choose or compare multiple-choice options.",
        f"Segment: {segment_id}",
    ]
    if question:
        lines.append(f"Question context: {question}")
    if targets:
        lines.append("Unordered target names: " + "; ".join(str(target) for target in targets))
    lines.append("Anchors:")
    for index, anchor in enumerate(anchors, start=1):
        anchor_targets = anchor.get("targets", [])
        target_text = (
            "; ".join(str(target) for target in anchor_targets)
            if isinstance(anchor_targets, Sequence) and not isinstance(anchor_targets, (str, bytes))
            else str(anchor_targets)
        )
        lines.append(
            f"- A{index}: id={anchor.get('anchor_id', '')} "
            f"[{float(anchor.get('start_sec', 0.0) or 0.0):.3f}s, {float(anchor.get('end_sec', 0.0) or 0.0):.3f}s]; "
            f"targets={target_text}; reason={anchor.get('reason', '')}"
        )
    return "\n".join(lines)


def _anchor_union_window(
    *,
    anchors: Sequence[Mapping[str, object]],
    fallback_start: float,
    fallback_end: float,
) -> tuple[float, float]:
    starts = [float(anchor.get("start_sec", 0.0) or 0.0) for anchor in anchors]
    ends = [float(anchor.get("end_sec", 0.0) or 0.0) for anchor in anchors]
    starts = [value for value in starts if value > 0 or fallback_start <= 0]
    ends = [value for value in ends if value > 0]
    start = min(starts) if starts else float(fallback_start)
    end = max(ends) if ends else float(fallback_end or start)
    if end < start:
        end = start
    return start, end


def _validate_anchor_segment_ids(*, segment_id: str, anchors: Sequence[Mapping[str, object]]) -> None:
    requested_segment_id = str(segment_id or "").strip()
    if not requested_segment_id:
        return
    mismatches = []
    for anchor in anchors:
        anchor_segment_id = str(anchor.get("segment_id", "") or "").strip()
        if anchor_segment_id and anchor_segment_id != requested_segment_id:
            mismatches.append(
                {
                    "anchor_id": str(anchor.get("anchor_id", "") or ""),
                    "anchor_segment_id": anchor_segment_id,
                }
            )
    if not mismatches:
        return
    details = ", ".join(
        f"{item['anchor_id'] or '<unnamed>'}:{item['anchor_segment_id']}" for item in mismatches
    )
    raise ValueError(
        f"verify_segment_anchors anchor segment_id mismatch for {requested_segment_id}: {details}"
    )


def _anchor_verify_groups(
    *,
    anchors: Sequence[Mapping[str, object]],
    fallback_start: float,
    fallback_end: float,
    max_union_sec: float,
) -> list[list[Mapping[str, object]]]:
    anchor_list = [dict(anchor) for anchor in anchors if isinstance(anchor, Mapping)]
    if not anchor_list:
        return [[{"start_sec": fallback_start, "end_sec": fallback_end, "targets": [], "reason": "fallback window"}]]
    union_start, union_end = _anchor_union_window(
        anchors=anchor_list,
        fallback_start=fallback_start,
        fallback_end=fallback_end,
    )
    if union_end - union_start <= float(max_union_sec):
        return [anchor_list]
    return [[anchor] for anchor in sorted(anchor_list, key=lambda item: float(item.get("start_sec", fallback_start) or fallback_start))]


def _anchor_group_targets(anchors: Sequence[Mapping[str, object]]) -> list[str]:
    targets: list[str] = []
    for anchor in anchors:
        anchor_targets = anchor.get("targets", [])
        if isinstance(anchor_targets, Sequence) and not isinstance(anchor_targets, (str, bytes)):
            values = [str(target).strip() for target in anchor_targets]
        else:
            values = [str(anchor_targets).strip()]
        for value in values:
            if value and value not in targets:
                targets.append(value)
    return targets


def _parse_anchor_verification(text: object) -> dict[str, object]:
    raw_text = str(text or "")
    parsed = _json_object_from_text(raw_text)
    confirmations = parsed.get("confirmations", []) if isinstance(parsed, Mapping) else []
    rejections = parsed.get("rejections", []) if isinstance(parsed, Mapping) else []
    ordered_visible = _ordered_visible_from_verification_text(raw_text, parsed)
    return {
        "confirmations": [dict(item) for item in confirmations if isinstance(item, Mapping)],
        "rejections": [dict(item) for item in rejections if isinstance(item, Mapping)],
        "ordered_visible_in_window": ordered_visible,
    }


def _ordered_visible_from_verification_text(text: str, parsed: Mapping[str, object]) -> list[str]:
    for key in ("ordered_visible_in_window", "ordered_visible", "ORDERED_VISIBLE"):
        value = parsed.get(key) if isinstance(parsed, Mapping) else None
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return _split_ordered_visible_items(value)
    match = re.search(r"ORDERED_VISIBLE\s*:\s*(.+)", str(text or ""), flags=re.IGNORECASE)
    if not match:
        return []
    return _split_ordered_visible_items(match.group(1))


def _split_ordered_visible_items(value: str) -> list[str]:
    return [
        item.strip().strip("\"'")
        for item in re.split(r"\s*(?:->|→|,|;)\s*", str(value or "").strip())
        if item.strip().strip("\"'")
    ]


def _json_object_from_text(text: str) -> Mapping[str, object]:
    stripped = str(text or "").strip()
    if not stripped:
        return {}
    try:
        loaded = json.loads(stripped)
        return loaded if isinstance(loaded, Mapping) else {}
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end <= start:
            return {}
        try:
            loaded = json.loads(stripped[start : end + 1])
            return loaded if isinstance(loaded, Mapping) else {}
        except json.JSONDecodeError:
            return {}


def _timeline_rows_from_confirmations(
    *,
    confirmations: Sequence[Mapping[str, object]],
    segment_id: str,
    fallback_window: Sequence[float],
    ordered_visible: Sequence[str] = (),
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ordered_keys = [_target_order_key(target) for target in ordered_visible if str(target).strip()]
    ordered_rank = {key: index for index, key in enumerate(ordered_keys)}
    ordered_confirmations = sorted(
        list(confirmations),
        key=lambda confirmation: (
            ordered_rank.get(_target_order_key(str(confirmation.get("target", ""))), len(ordered_rank)),
            _optional_float(confirmation.get("observed_at_sec"))
            if _optional_float(confirmation.get("observed_at_sec")) is not None
            else float("inf"),
        ),
    )
    start = float(fallback_window[0]) if len(fallback_window) >= 2 else 0.0
    end = float(fallback_window[1]) if len(fallback_window) >= 2 else start
    step = max(0.001, (end - start) / max(1, len(ordered_confirmations) - 1)) if ordered_confirmations else 0.001
    for index, confirmation in enumerate(ordered_confirmations):
        target = str(confirmation.get("target", "")).strip()
        if not target:
            continue
        observed_at = _optional_float(confirmation.get("observed_at_sec"))
        if observed_at is None:
            relative = _optional_float(confirmation.get("relative_sec"))
            if relative is not None and fallback_window:
                observed_at = float(fallback_window[0]) + relative
        if observed_at is None and ordered_keys:
            observed_at = start + index * step
        rows.append(
            {
                "segment_id": segment_id,
                "entity": target,
                "observed_at_sec": observed_at,
                "window": [float(fallback_window[0]), float(fallback_window[1])] if len(fallback_window) >= 2 else [],
                "confidence_signal": "visually_confirmed",
                "claim": str(confirmation.get("evidence") or confirmation.get("claim") or ""),
            }
        )
    return rows


def _target_order_key(value: str) -> str:
    return re.sub(r"\W+", " ", str(value or "").lower()).strip()


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sanitize_vision_read_ask_for(ask_for: str) -> str:
    return exploration_question(ask_for)


def _sanitize_inspect_segment_question(
    question: str,
    *,
    candidate_options: Sequence[str],
) -> tuple[str, Sequence[str]]:
    return exploration_question(question), ()


def _looks_like_mcq(text: str) -> bool:
    option_lines = re.findall(r"(?m)^\s*[A-H][\).]\s+\S+", str(text))
    if len(option_lines) >= 2:
        return True
    return bool(re.search(r"\b(?:which|what|why|how)\b", str(text), flags=re.IGNORECASE)) and len(option_lines) >= 1


def _candidate_options_look_like_full_mcq(candidate_options: Sequence[str]) -> bool:
    option_lines = [
        str(option)
        for option in candidate_options
        if re.match(r"\s*[A-H][\).]\s+\S+", str(option), flags=re.IGNORECASE)
    ]
    if len(option_lines) < 3:
        return False
    joined = "\n".join(option_lines)
    return len(joined) > 180 or any("," in line or '"' in line for line in option_lines)


def _mutex_read_prompt(
    *,
    segment_id: str,
    start_sec: float,
    end_sec: float,
    option_x: str,
    option_x_text: str,
    option_y: str,
    option_y_text: str,
) -> str:
    return (
        "You are a v4 VisionAgent reading one localized long-video window.\n"
        f"In this window, is option {option_x} (`{option_x_text}`) true, "
        f"OR option {option_y} (`{option_y_text}`) true, OR NEITHER true?\n"
        "Cite only visible frames. If no visible evidence supports either, return NEITHER.\n"
        "Return one label first: the option letter, or NEITHER, followed by a short visual justification.\n"
        "Do not use outside video context or external knowledge.\n"
        f"Segment: {segment_id} [{start_sec:.3f}s, {end_sec:.3f}s]"
    )


def _mutex_supported_option(claim: str, *, option_x: str, option_y: str) -> str:
    text = str(claim).strip()
    if re.search(r"\bneither\b", text, flags=re.IGNORECASE):
        return ""
    for option in (option_x, option_y):
        normalized = str(option).strip().upper()[:1]
        if normalized and re.match(rf"\s*(?:option\s+)?{re.escape(normalized)}\b", text, flags=re.IGNORECASE):
            return normalized
    return ""
