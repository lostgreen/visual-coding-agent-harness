"""Root DVC SceneIndex builder for VideoMME navigation indexes."""

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
from ...video_index import SceneIndex, TimelineBeat, VideoSegment, fixed_window_scene_index
from .scene_index_cache import SceneIndexCache


SCENE_INDEX_BUILDER_SCHEMA_VERSION = "dvc_root_v1"

ClipExtractor = Callable[[str, str, float, float], str]


@dataclass(frozen=True)
class SubtitleCue:
    start_sec: float
    end_sec: float
    text: str
    cue_id: str = ""


@dataclass(frozen=True)
class RootIndexPolicy:
    root_window_sec: float = 300.0
    frame_cache_fps: float = 1.0
    max_pixels_per_frame: int = 360 * 420
    max_beats_per_root: int = 8


class SceneIndexBuilder:
    def __init__(
        self,
        *,
        backend: VisionLanguageBackend,
        text_model_id: str,
        vl_model_id: str,
        window_sec: float = 300.0,
        caption_nframes: int = 8,
        root_policy: Optional[RootIndexPolicy] = None,
        cache: Optional[SceneIndexCache] = None,
        clip_root: Optional[Path | str] = None,
        clip_extractor: Optional[ClipExtractor] = None,
        frame_sampler: Optional[FrameSampler] = None,
        schema_version: str = SCENE_INDEX_BUILDER_SCHEMA_VERSION,
    ) -> None:
        self.backend = backend
        self.text_model_id = text_model_id
        self.vl_model_id = vl_model_id
        self.root_policy = root_policy or RootIndexPolicy(root_window_sec=float(window_sec))
        self.window_sec = self.root_policy.root_window_sec
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
            source="dvc_root_v1",
        )
        segments = []
        for segment in base.segments:
            segment_cues = _cues_for_segment(cues, segment)
            root_data = self._build_root_dvc(
                video_id=video_id,
                video_path=video_path,
                segment=segment,
                cues=segment_cues,
            )
            segments.append(
                _merge_root_segment(
                    segment,
                    root_data=root_data,
                    cues=segment_cues,
                    root_source=f"build_root_dvc_index:{self.vl_model_id}",
                    max_beats=self.root_policy.max_beats_per_root,
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
            "root_window_sec": round(float(self.root_policy.root_window_sec), 3),
            "frame_cache_fps": round(float(self.root_policy.frame_cache_fps), 3),
            "max_pixels_per_frame": int(self.root_policy.max_pixels_per_frame),
            "subtitle_hash": subtitle_hash(subtitle_cues),
            "vl_model_id": self.vl_model_id,
        }
        if self.cache is not None:
            return self.cache.key_for(parts)
        payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _build_root_dvc(
        self,
        *,
        video_id: str,
        video_path: str,
        segment: VideoSegment,
        cues: Sequence[SubtitleCue],
    ) -> Mapping[str, Any]:
        media_path, media_type, frame_paths, metadata = self._root_media(
            video_id=video_id,
            video_path=video_path,
            segment=segment,
        )
        cue_text = "\n".join(
            f"{cue.cue_id or idx} [{cue.start_sec:.3f}-{cue.end_sec:.3f}s]: {cue.text}"
            for idx, cue in enumerate(cues, start=1)
        )
        response = self.backend.generate(
            BackendRequest(
                task="build_root_dvc_index",
                prompt=(
                    "You are building a navigation-only index for a five-minute video interval.\n\n"
                    "You receive chronologically ordered video frames sampled at 1 FPS when available, "
                    "and timestamped subtitle / ASR cues for the same interval.\n\n"
                    "Return only JSON using this schema:\n"
                    "{\n"
                    '  "root_summary": "one sentence navigation summary",\n'
                    '  "beats": [\n'
                    "    {\n"
                    '      "start_offset_sec": 0.0,\n'
                    '      "end_offset_sec": 10.0,\n'
                    '      "summary": "visible or spoken content in this beat",\n'
                    '      "entity_hints": ["optional"],\n'
                    '      "modality_hints": ["visual|asr|ocr|temporal"]\n'
                    "    }\n"
                    "  ],\n"
                    '  "topic_tags": ["optional"],\n'
                    '  "limitations": ["optional"]\n'
                    "}\n"
                    "Use 1 to MAX_BEATS chronological non-overlapping beats. For each beat, provide "
                    "start_offset_sec, end_offset_sec, and summary relative to this interval. Summarize only visible "
                    "or spoken content. Mark useful cues as visual, asr, ocr, or temporal. Do not answer a downstream "
                    "question. This is a navigation index, not final evidence.\n\n"
                    f"MAX_BEATS: {self.root_policy.max_beats_per_root}\n"
                    f"Segment: {segment.segment_id} {segment.start_sec:.3f}-{segment.end_sec:.3f}s\n"
                    f"Subtitles / ASR cues:\n{cue_text}"
                ),
                media_path=media_path,
                media_type=media_type,
                frames=frame_paths,
                max_new_tokens=512,
                metadata={
                    **metadata,
                    "schema_version": self.schema_version,
                    "segment_id": segment.segment_id,
                    "start_sec": segment.start_sec,
                    "end_sec": segment.end_sec,
                    "cue_ids": [cue.cue_id for cue in cues if cue.cue_id],
                    "model_id": self.vl_model_id,
                    "frame_cache_fps": self.root_policy.frame_cache_fps,
                    "max_pixels_per_frame": self.root_policy.max_pixels_per_frame,
                    "max_beats_per_root": self.root_policy.max_beats_per_root,
                },
            )
        )
        data = _parse_lenient_json(response.text)
        if not data:
            raise ValueError(f"Root DVC response for {segment.segment_id} was not valid JSON")
        return data

    def _root_media(
        self,
        *,
        video_id: str,
        video_path: str,
        segment: VideoSegment,
    ) -> tuple[str | None, str, tuple[str, ...], dict[str, Any]]:
        media_path: str | None = video_path
        media_type = "video"
        frame_paths: tuple[str, ...] = ()
        metadata: dict[str, Any] = {
            "source_video_path": video_path,
            "root_window_sec": self.root_policy.root_window_sec,
        }
        if self.frame_sampler is None:
            raise ValueError("Root DVC requires a precomputed frame cache frame_sampler")
        max_frames = int(round(max(1.0, segment.end_sec - segment.start_sec) * self.root_policy.frame_cache_fps))
        frame_paths = tuple(self.frame_sampler(video_path, float(segment.start_sec), float(segment.end_sec), max_frames))
        if not frame_paths:
            raise ValueError("Root DVC requires non-empty cached frames from frame_sampler")
        media_path = None
        metadata["frame_cache_policy"] = "precomputed_2fps"
        metadata["frame_count"] = len(frame_paths)
        return media_path, media_type, frame_paths, metadata

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


def _merge_root_segment(
    segment: VideoSegment,
    *,
    root_data: Mapping[str, Any],
    cues: Sequence[SubtitleCue],
    root_source: str,
    max_beats: int,
) -> VideoSegment:
    root_summary = _clean_generated_text(
        root_data.get("root_summary") or root_data.get("summary") or "",
        "root_summary",
        "summary",
    )
    beats = _normalize_root_beats(
        segment=segment,
        beats=root_data.get("beats") or root_data.get("timeline_beats") or (),
        max_beats=max_beats,
    )
    if not root_summary and beats:
        root_summary = beats[0].summary
    if not root_summary:
        raise ValueError(f"Root DVC response for {segment.segment_id} did not include root_summary")

    beat_entities = [hint for beat in beats for hint in beat.entity_hints]
    root_entities = _clean_list(root_data.get("entities") or root_data.get("entity_hints"))
    return VideoSegment(
        segment_id=segment.segment_id,
        start_sec=segment.start_sec,
        end_sec=segment.end_sec,
        keyframe_path=segment.keyframe_path,
        low_fps_caption=root_summary,
        source="dvc_root_v1",
        source_segment_id=segment.segment_id,
        visual_caption=root_summary,
        visual_caption_source=root_source,
        asr_summary="",
        asr_summary_source="subtitle_cues",
        map_summary=root_summary,
        raw_asr_ref=_clean_text(root_data.get("raw_asr_ref") or ",".join(cue.cue_id for cue in cues if cue.cue_id)),
        entities=tuple(_unique([*root_entities, *beat_entities])),
        topic_tags=tuple(_clean_list(root_data.get("topic_tags") or root_data.get("tags"))),
        confidence=_clean_float(root_data.get("confidence")),
        citation_provenance={"index": "navigation_only", "asr": "subtitle", "visual": "video"},
        asr_sentences=_cue_sentence_rows(cues),
        limitations=tuple(_clean_list(root_data.get("limitations"))),
        index_level="root",
        root_segment_id=segment.segment_id,
        timeline_beats=beats,
        refinement_state="coarse",
        index_provenance={
            "schema_version": "dvc_root_v1",
            "source": root_source,
            "navigation_only": True,
        },
    )


def _normalize_root_beats(
    *,
    segment: VideoSegment,
    beats: Any,
    max_beats: int | None = None,
) -> tuple[TimelineBeat, ...]:
    try:
        raw_beats = list(beats)
    except TypeError as exc:
        raise ValueError(f"Root DVC beats for {segment.segment_id} must be a list") from exc
    if max_beats is not None:
        raw_beats = raw_beats[:max_beats]

    normalized: list[TimelineBeat] = []
    last_end = segment.start_sec
    for index, item in enumerate(raw_beats, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"Root DVC beat {index} for {segment.segment_id} must be an object")
        start = _beat_abs_time(segment=segment, item=item, key="start")
        end = _beat_abs_time(segment=segment, item=item, key="end")
        start = max(segment.start_sec, min(segment.end_sec, start))
        end = max(segment.start_sec, min(segment.end_sec, end))
        if end <= start:
            raise ValueError(f"Root DVC beat {index} for {segment.segment_id} has empty duration")
        if start < last_end:
            raise ValueError(f"Root DVC beats for {segment.segment_id} must be chronological and non-overlapping")
        summary = _beat_summary(item)
        if not summary:
            last_end = end
            continue
        normalized.append(
            TimelineBeat(
                beat_id=_clean_text(item.get("beat_id") or f"{segment.segment_id}_b{index:02d}"),
                start_sec=round(start, 3),
                end_sec=round(end, 3),
                summary=summary,
                entity_hints=tuple(_clean_list(item.get("entity_hints"))),
                modality_hints=tuple(_clean_modalities(item.get("modality_hints"))),
                confidence=_clean_float(item.get("confidence")),
                frame_refs=tuple(_clean_list(item.get("frame_refs"))),
                limitations=tuple(_clean_list(item.get("limitations"))),
            )
        )
        last_end = end
    return tuple(normalized)


def _beat_summary(item: Mapping[str, Any]) -> str:
    for key in ("summary", "description", "caption", "text", "event", "content"):
        summary = _clean_generated_text(item.get(key) or "", key)
        if summary:
            return summary
    return ""


def _beat_abs_time(*, segment: VideoSegment, item: Mapping[str, Any], key: str) -> float:
    absolute_key = f"{key}_sec"
    offset_key = f"{key}_offset_sec"
    if offset_key in item:
        return segment.start_sec + float(item[offset_key])
    return float(item[absolute_key])


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


def _clean_modalities(value: Any) -> list[str]:
    allowed = {"visual", "asr", "ocr", "temporal"}
    return [item for item in _clean_list(value) if item in allowed]


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
