"""Segment Inspector subagent boundary for long-video context control."""

from __future__ import annotations

import re
from typing import Mapping, Optional, Sequence

from ..agents.contracts import resolve_nframes
from ..agents.open_questions import exploration_question
from ..agents.output_quality import confidence_signal_from_text
from ..backends.base import BackendRequest, VisionLanguageBackend
from ..registry import ToolRegistry, tool
from ..workspace import EvidenceWorkspace
from .segments import ClipExtractor, _clip_output_path, _extract_clip_ffmpeg


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

    registry.register(inspect_segment)
    registry.register(vision_read)
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
) -> Mapping[str, object]:
    resolved_nframes, _ = resolve_nframes(nframes)
    metadata_candidate_options = list(original_candidate_options or candidate_options)
    metadata = {
        "tool_role": "segment_inspector",
        "context_tier": "isolated_subagent",
        "segment_id": segment_id,
        "start_sec": float(start_sec),
        "end_sec": float(end_sec),
        "question": question,
        "candidate_options": metadata_candidate_options,
        "nframes": int(resolved_nframes),
        "max_pixels": int(max_pixels),
    }
    if original_question is not None and str(original_question or "").strip() != str(question or "").strip():
        metadata["original_question"] = original_question
    if original_candidate_options:
        metadata["original_candidate_options"] = list(original_candidate_options)
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
        "Do not choose an option. Do not emit supported_option, answer_option, or final_answer.\n"
        "Do not use outside video context or external knowledge.\n"
        f"Segment: {segment_id} [{start_sec:.3f}s, {end_sec:.3f}s]\n"
        f"Ask for: {ask_for}"
    )


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
