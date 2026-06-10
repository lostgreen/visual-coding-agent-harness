"""Dual-source SceneIndex builder for VideoMME subtitles and video captions."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from ...backends.base import BackendRequest, VisionLanguageBackend
from ...tools.frame_cache import FrameSampler
from ...video_index import SceneIndex, VideoSegment, fixed_window_scene_index
from .scene_index_cache import SceneIndexCache


SCENE_INDEX_BUILDER_SCHEMA_VERSION = "dual_source_scene_index_v4"

ClipExtractor = Callable[[str, str, float, float], str]


@dataclass(frozen=True)
class SubtitleCue:
    start_sec: float
    end_sec: float
    text: str
    cue_id: str = ""


class SceneIndexBuilder:
    def __init__(
        self,
        *,
        backend: VisionLanguageBackend,
        text_model_id: str,
        vl_model_id: str,
        window_sec: float = 30.0,
        caption_nframes: int = 8,
        cache: Optional[SceneIndexCache] = None,
        clip_root: Optional[Path | str] = None,
        clip_extractor: Optional[ClipExtractor] = None,
        frame_sampler: Optional[FrameSampler] = None,
        schema_version: str = SCENE_INDEX_BUILDER_SCHEMA_VERSION,
    ) -> None:
        self.backend = backend
        self.text_model_id = text_model_id
        self.vl_model_id = vl_model_id
        self.window_sec = window_sec
        self.caption_nframes = caption_nframes
        self.cache = cache
        self.clip_root = Path(clip_root) if clip_root is not None else None
        self.clip_extractor = clip_extractor or _extract_clip_ffmpeg
        self.frame_sampler = frame_sampler
        self.schema_version = schema_version

    def build(
        self,
        *,
        video_id: str,
        video_path: str,
        duration_sec: float,
        subtitle_cues: Sequence[SubtitleCue],
    ) -> SceneIndex:
        cues = [_coerce_cue(cue) for cue in subtitle_cues]
        cache_key = self.cache_key(
            video_id=video_id,
            video_path=video_path,
            duration_sec=duration_sec,
            subtitle_cues=cues,
        )
        if self.cache is not None:
            cached = self.cache.load(cache_key)
            if cached is not None:
                return cached

        base = fixed_window_scene_index(
            video_path=video_path,
            duration_sec=duration_sec,
            window_sec=self.window_sec,
            source="dual_source_scene_index",
        )
        segments = []
        for segment in base.segments:
            segment_cues = _cues_for_segment(cues, segment)
            asr_data = self._summarize_subtitles(segment=segment, cues=segment_cues)
            visual_data = self._caption_scene(video_id=video_id, video_path=video_path, segment=segment)
            map_summary = self._summarize_scene_map(segment=segment, asr_data=asr_data, visual_data=visual_data)
            segments.append(
                _merge_segment(
                    segment,
                    asr_data=asr_data,
                    visual_data=visual_data,
                    map_summary=map_summary,
                    asr_source=f"summarize_subtitle_segment:{self.text_model_id}",
                    visual_source=f"caption_scene_segment:{self.vl_model_id}",
                )
            )

        scene_index = SceneIndex(video_path=video_path, duration_sec=duration_sec, segments=segments)
        if self.cache is not None:
            self.cache.store(cache_key, scene_index)
        return scene_index

    def cache_key(
        self,
        *,
        video_id: str,
        video_path: str,
        duration_sec: float,
        subtitle_cues: Sequence[SubtitleCue],
    ) -> str:
        parts = {
            "schema_version": self.schema_version,
            "video_id": video_id,
            "video_path": video_path,
            "duration_sec": round(float(duration_sec), 3),
            "window_sec": round(float(self.window_sec), 3),
            "caption_nframes": int(self.caption_nframes),
            "visual_clip_policy": (
                "precomputed_2fps_frames"
                if self.frame_sampler is not None
                else "physical_clip"
                if self.clip_root is not None
                else "whole_video_metadata"
            ),
            "subtitle_hash": subtitle_hash(subtitle_cues),
            "text_model_id": self.text_model_id,
            "vl_model_id": self.vl_model_id,
        }
        if self.cache is not None:
            return self.cache.key_for(parts)
        payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _summarize_subtitles(self, *, segment: VideoSegment, cues: Sequence[SubtitleCue]) -> Mapping[str, Any]:
        cue_text = "\n".join(f"{cue.cue_id or idx}: {cue.text}" for idx, cue in enumerate(cues, start=1))
        response = self.backend.generate(
            BackendRequest(
                task="summarize_subtitle_segment",
                prompt=(
                    "Summarize only the subtitle/ASR content for this fixed video segment. "
                    "Return JSON with summary, entities, topic_tags, confidence, raw_asr_ref. "
                    "Do not include answer options or candidate option relations.\n"
                    f"Segment: {segment.segment_id} {segment.start_sec:.3f}-{segment.end_sec:.3f}s\n"
                    f"Subtitles:\n{cue_text}"
                ),
                max_new_tokens=256,
                metadata={
                    "segment_id": segment.segment_id,
                    "start_sec": segment.start_sec,
                    "end_sec": segment.end_sec,
                    "cue_ids": [cue.cue_id for cue in cues if cue.cue_id],
                    "model_id": self.text_model_id,
                },
            )
        )
        data = _parse_lenient_json(response.text)
        if not data:
            data = {"summary": response.text}
        return {
            "summary": _clean_generated_text(data.get("summary") or data.get("asr_summary") or "", "summary", "asr_summary"),
            "entities": _clean_list(data.get("entities")),
            "topic_tags": _clean_list(data.get("topic_tags") or data.get("tags")),
            "confidence": _clean_float(data.get("confidence")),
            "raw_asr_ref": _clean_text(
                data.get("raw_asr_ref") or ",".join(cue.cue_id for cue in cues if cue.cue_id)
            ),
            "asr_sentences": _cue_sentence_rows(cues),
        }

    def _caption_scene(self, *, video_id: str, video_path: str, segment: VideoSegment) -> Mapping[str, Any]:
        media_path: str | None = video_path
        media_type = "video"
        frame_paths: tuple[str, ...] = ()
        metadata: dict[str, Any] = {
            "segment_id": segment.segment_id,
            "start_sec": segment.start_sec,
            "end_sec": segment.end_sec,
            "nframes": int(self.caption_nframes),
            "model_id": self.vl_model_id,
        }
        if self.frame_sampler is not None:
            frame_paths = tuple(
                self.frame_sampler(video_path, float(segment.start_sec), float(segment.end_sec), int(self.caption_nframes))
            )
            if frame_paths:
                media_path = None
                media_type = "image"
                metadata["source_video_path"] = video_path
                metadata["frame_cache_policy"] = "precomputed_2fps"
                metadata["frame_count"] = len(frame_paths)
        if not frame_paths and self.clip_root is not None:
            clip_path = _clip_output_path(clip_root=self.clip_root, video_id=video_id, segment=segment)
            media_path = self.clip_extractor(video_path, str(clip_path), segment.start_sec, segment.end_sec)
            metadata["source_video_path"] = video_path
            metadata["clip_path"] = media_path
        response = self.backend.generate(
            BackendRequest(
                task="caption_scene_segment",
                prompt=(
                    "Caption only the visual content in this fixed video segment. "
                    "Return JSON with caption, stage_tags, entities, grounding_quality. "
                    "Do not include answer options or candidate option relations.\n"
                    f"Segment: {segment.segment_id} {segment.start_sec:.3f}-{segment.end_sec:.3f}s"
                ),
                media_path=media_path,
                media_type=media_type,
                frames=frame_paths,
                max_new_tokens=256,
                metadata=metadata,
            )
        )
        data = _parse_lenient_json(response.text)
        if not data:
            data = {"caption": response.text}
        return {
            "caption": _clean_generated_text(data.get("caption") or data.get("visual_caption") or "", "caption", "visual_caption"),
            "stage_tags": _clean_list(data.get("stage_tags")),
            "entities": _clean_list(data.get("entities")),
            "grounding_quality": _clean_text(data.get("grounding_quality") or ""),
        }

    def _summarize_scene_map(
        self,
        *,
        segment: VideoSegment,
        asr_data: Mapping[str, Any],
        visual_data: Mapping[str, Any],
    ) -> str:
        response = self.backend.generate(
            BackendRequest(
                task="summarize_scene_map_segment",
                prompt=(
                    "Create one short planner map line for this fixed video segment. "
                    "Use the visual caption as the primary signal and subtitle/ASR only as supplementary context. "
                    "Return JSON with summary only. The summary must be one sentence, under 22 words, "
                    "and must not mention answer options, candidate answers, Tags, Entities, Visual, or ASR labels.\n"
                    f"Segment: {segment.segment_id} {segment.start_sec:.3f}-{segment.end_sec:.3f}s\n"
                    f"Visual caption: {_clean_text(visual_data.get('caption') or '')}\n"
                    f"Subtitle/ASR summary: {_clean_text(asr_data.get('summary') or '')}"
                ),
                max_new_tokens=96,
                metadata={
                    "segment_id": segment.segment_id,
                    "start_sec": segment.start_sec,
                    "end_sec": segment.end_sec,
                    "model_id": self.text_model_id,
                },
            )
        )
        data = _parse_lenient_json(response.text)
        summary = _clean_generated_text(data.get("summary") if data else response.text, "summary", "map_summary")
        return summary or _fallback_map_summary(asr_data=asr_data, visual_data=visual_data)


def subtitle_hash(cues: Sequence[SubtitleCue]) -> str:
    normalized = [
        {
            "cue_id": cue.cue_id,
            "start_sec": round(float(cue.start_sec), 3),
            "end_sec": round(float(cue.end_sec), 3),
            "text": _clean_text(cue.text),
        }
        for cue in cues
    ]
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _merge_segment(
    segment: VideoSegment,
    *,
    asr_data: Mapping[str, Any],
    visual_data: Mapping[str, Any],
    map_summary: str,
    asr_source: str,
    visual_source: str,
) -> VideoSegment:
    visual_caption = _clean_generated_text(visual_data.get("caption") or "", "caption", "visual_caption")
    asr_summary = _clean_generated_text(asr_data.get("summary") or "", "summary", "asr_summary")
    return VideoSegment(
        segment_id=segment.segment_id,
        start_sec=segment.start_sec,
        end_sec=segment.end_sec,
        keyframe_path=segment.keyframe_path,
        low_fps_caption=visual_caption,
        source="dual_source_scene_index",
        source_segment_id=segment.segment_id,
        visual_caption=visual_caption,
        visual_caption_source=visual_source,
        asr_summary=asr_summary,
        asr_summary_source=asr_source,
        map_summary=_clean_generated_text(map_summary, "summary", "map_summary"),
        raw_asr_ref=_clean_text(asr_data.get("raw_asr_ref") or ""),
        stage_tags=tuple(_clean_list(visual_data.get("stage_tags"))),
        entities=tuple(
            _unique([*_clean_list(asr_data.get("entities")), *_clean_list(visual_data.get("entities"))])
        ),
        topic_tags=tuple(_clean_list(asr_data.get("topic_tags"))),
        confidence=_clean_float(asr_data.get("confidence")),
        grounding_quality=_clean_text(visual_data.get("grounding_quality") or ""),
        citation_provenance={"asr": "subtitle", "visual": "video"},
        asr_sentences=tuple(dict(item) for item in asr_data.get("asr_sentences") or ()),
    )


def _clip_output_path(*, clip_root: Path, video_id: str, segment: VideoSegment) -> Path:
    safe_video_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(video_id)).strip("_") or "video"
    safe_segment_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", segment.segment_id).strip("_") or "segment"
    start_ms = int(round(segment.start_sec * 1000))
    end_ms = int(round(segment.end_sec * 1000))
    return clip_root / safe_video_id / f"{safe_video_id}_{safe_segment_id}_{start_ms}_{end_ms}.mp4"


def _extract_clip_ffmpeg(video_path: str, output_path: str, start_sec: float, end_sec: float) -> str:
    output = Path(output_path)
    if output.exists() and output.stat().st_size > 0:
        return str(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.001, float(end_sec) - float(start_sec))
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{float(start_sec):.3f}",
        "-i",
        video_path,
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        "-movflags",
        "+faststart",
        output_path,
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to extract scene-index clips") from exc
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or "").strip().splitlines()[-3:]
        raise RuntimeError(f"ffmpeg failed to extract scene-index clip: {' | '.join(message)}") from exc
    return output_path


def _parse_lenient_json(text: str) -> Mapping[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _cues_for_segment(cues: Sequence[SubtitleCue], segment: VideoSegment) -> list[SubtitleCue]:
    return [
        cue
        for cue in cues
        if cue.start_sec < segment.end_sec and cue.end_sec > segment.start_sec
    ]


def _coerce_cue(cue: SubtitleCue) -> SubtitleCue:
    if isinstance(cue, SubtitleCue):
        return cue
    if isinstance(cue, Mapping):
        return SubtitleCue(
            start_sec=float(cue["start_sec"]),
            end_sec=float(cue.get("end_sec", cue["start_sec"])),
            text=str(cue["text"]),
            cue_id=str(cue.get("cue_id") or ""),
        )
    if isinstance(cue, tuple):
        if len(cue) == 2:
            return SubtitleCue(start_sec=float(cue[0]), end_sec=float(cue[0]), text=str(cue[1]))
        return SubtitleCue(start_sec=float(cue[0]), end_sec=float(cue[1]), text=str(cue[2]))
    return SubtitleCue(
        start_sec=float(getattr(cue, "start_sec")),
        end_sec=float(getattr(cue, "end_sec", getattr(cue, "start_sec"))),
        text=str(getattr(cue, "text")),
        cue_id=str(getattr(cue, "cue_id", "") or ""),
    )


def _cue_sentence_rows(cues: Sequence[SubtitleCue]) -> tuple[Mapping[str, object], ...]:
    return tuple(
        {
            "cue_id": str(cue.cue_id),
            "start_sec": float(cue.start_sec),
            "end_sec": float(cue.end_sec),
            "text": _clean_text(cue.text),
        }
        for cue in cues
        if _clean_text(cue.text)
    )


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _clean_generated_text(value: Any, *preferred_keys: str) -> str:
    text = _clean_text(value)
    if not text or not (text.startswith("{") and text.endswith("}")):
        return text
    data = _parse_lenient_json(text)
    for key in preferred_keys:
        if key in data:
            nested = _clean_text(data.get(key))
            if nested != text:
                return _clean_generated_text(nested, *preferred_keys)
    return text


def _fallback_map_summary(*, asr_data: Mapping[str, Any], visual_data: Mapping[str, Any]) -> str:
    visual = _clean_generated_text(visual_data.get("caption") or "", "caption", "visual_caption")
    asr = _clean_generated_text(asr_data.get("summary") or "", "summary", "asr_summary")
    return _clean_text(visual or asr or "no coarse caption yet")


def _clean_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    else:
        try:
            items = list(value)
        except TypeError:
            items = [value]
    return _unique(_clean_text(item) for item in items)


def _clean_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _unique(values: Sequence[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        text = _clean_text(value)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result
