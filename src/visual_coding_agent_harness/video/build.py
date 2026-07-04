"""Builders and adapters for the multi_v3 layered video index."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Sequence

from .pipeline import compose_scene_thumb, compose_shot_grid
from .index import Frame, Scene, SceneIndex, Shot, VideoIndex, VideoSegment
from .pipeline import sample_shot_frames
from .pipeline import aggregate_shot_ranges_by_duration
from .pipeline import detect_shots_ffmpeg, detect_shots_uniform


FramePathBuilder = Callable[[VideoSegment], Sequence[str]]
ShotGridBuilder = Callable[[Shot, Sequence[Frame]], str]
SceneThumbBuilder = Callable[[Scene, Sequence[Shot]], str]
ShotDetector = Callable[[str, float], Sequence[tuple[float, float]]]
KeyframeSampler = Callable[[str, float, float, int, Path], Sequence[Frame]]


def build_video_index_from_video(
    video_path: str,
    duration_sec: float,
    *,
    artifact_dir: Path,
    asr_cues: Sequence[Any] = (),
    shot_detector: ShotDetector | None = None,
    keyframe_sampler: KeyframeSampler | None = None,
    source_segments: Sequence[VideoSegment] = (),
    frames_per_shot: int = 6,
    scene_max_sec: float = 600.0,
    render_grid: bool = True,
) -> VideoIndex:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    shot_ranges = tuple((float(start), float(end)) for start, end in _detect_shot_ranges(video_path, duration_sec, detector=shot_detector))
    if not shot_ranges:
        shot_ranges = tuple(detect_shots_uniform(duration_sec, window=max(1.0, min(float(duration_sec), 15.0))))
    groups = aggregate_shot_ranges_by_duration(shot_ranges, max_scene_sec=scene_max_sec)
    scenes = []
    sampler = keyframe_sampler or _default_keyframe_sampler
    for scene_number, group in enumerate(groups, start=1):
        scene_id = f"sc{scene_number:02d}"
        shots = []
        for shot_number, (start_sec, end_sec) in enumerate(group, start=1):
            shot_id = f"{scene_id}_sh{shot_number:03d}"
            inherited_segments = _segments_for_range(source_segments, start_sec=float(start_sec), end_sec=float(end_sec))
            frames = tuple(
                _with_frame_id(frame, shot_id=shot_id, frame_number=index)
                for index, frame in enumerate(
                    sampler(str(video_path), float(start_sec), float(end_sec), int(frames_per_shot), artifact_dir / "frames" / shot_id),
                    start=1,
                )
            )
            shot = Shot(
                shot_id=shot_id,
                scene_id=scene_id,
                start_sec=float(start_sec),
                end_sec=float(end_sec),
                frames=frames,
                visual_caption=_visual_caption_for_segments(inherited_segments),
                asr_text=_asr_text_for_range(asr_cues, start_sec=float(start_sec), end_sec=float(end_sec)),
                ocr_lines=(),
                entities=_entities_for_segments(inherited_segments),
                lowres_grid_path="",
                source_segment_id=shot_id,
            )
            grid_path = _compose_shot_grid_or_empty(shot=shot, frames=frames, artifact_dir=artifact_dir) if render_grid else ""
            shots.append(replace(shot, lowres_grid_path=grid_path))
        scene_start = min(start for start, _end in group)
        scene_end = max(end for _start, end in group)
        scene_segments = _segments_for_range(source_segments, start_sec=float(scene_start), end_sec=float(scene_end))
        scene_entities = _entities_for_segments(scene_segments) or _unique_text(item for shot in shots for item in shot.entities)
        scene_topics = _topics_for_segments(scene_segments)
        scene = Scene(
            scene_id=scene_id,
            start_sec=float(scene_start),
            end_sec=float(scene_end),
            title=_video_scene_title(scene_number, scene_topics),
            summary=_scene_summary_from_shots(shots, source_segments=scene_segments),
            shots=tuple(shots),
            dominant_entities=scene_entities,
            dominant_topics=scene_topics,
            scene_thumb_path="",
            source_segment_id=scene_id,
        )
        thumb_path = (
            str(compose_scene_thumb(scene, tuple(shots), artifact_dir / "scenes" / f"{scene_id}_thumb.jpg"))
            if render_grid
            else ""
        )
        scenes.append(replace(scene, scene_thumb_path=thumb_path))
    return VideoIndex(video_path=str(video_path), duration_sec=float(duration_sec), scenes=tuple(scenes))


def build_video_index_from_scene_index(
    scene_index: SceneIndex,
    *,
    artifact_dir: Path,
    shot_frame_paths: FramePathBuilder | None = None,
    shot_grid_builder: ShotGridBuilder | None = None,
    scene_thumb_builder: SceneThumbBuilder | None = None,
    render_grid: bool = True,
) -> VideoIndex:
    """Create the v3 hierarchy from the current flat SceneIndex.

    The initial migration keeps one legacy segment as one scene and one shot.
    Dedicated shot detection can later replace this adapter without changing the
    Reasoner/Investigator contracts.
    """

    artifact_dir.mkdir(parents=True, exist_ok=True)
    scenes = []
    for scene_number, segment in enumerate(scene_index.segments, start=1):
        scene_id = f"sc{scene_number:02d}"
        shot_id = f"{scene_id}_sh001"
        frame_paths = tuple(shot_frame_paths(segment) if shot_frame_paths else _default_frame_paths(segment))
        frames = tuple(
            Frame(frame_id=f"{shot_id}_fr{frame_number:03d}", time_sec=_frame_time(segment, frame_number, len(frame_paths)), thumb_path=path)
            for frame_number, path in enumerate(frame_paths, start=1)
        )
        shot = Shot(
            shot_id=shot_id,
            scene_id=scene_id,
            start_sec=segment.start_sec,
            end_sec=segment.end_sec,
            frames=frames,
            visual_caption=segment.visual_caption or segment.low_fps_caption or segment.map_summary,
            asr_text=_segment_asr_text(segment),
            ocr_lines=_segment_ocr_lines(segment),
            entities=segment.entities,
            lowres_grid_path="",
            source_segment_id=segment.segment_id,
        )
        if shot_grid_builder:
            grid_path = shot_grid_builder(shot, frames)
        elif render_grid:
            grid_path = _compose_shot_grid_or_empty(shot=shot, frames=frames, artifact_dir=artifact_dir)
        else:
            grid_path = ""
        shot = replace(shot, lowres_grid_path=grid_path)
        scene = Scene(
            scene_id=scene_id,
            start_sec=segment.start_sec,
            end_sec=segment.end_sec,
            title=_scene_title(segment),
            summary=_scene_summary(segment),
            shots=(shot,),
            dominant_entities=segment.entities,
            dominant_topics=segment.topic_tags,
            scene_thumb_path="",
            source_segment_id=segment.segment_id,
        )
        if scene_thumb_builder:
            thumb_path = scene_thumb_builder(scene, (shot,))
        elif render_grid:
            thumb_path = str(compose_scene_thumb(scene, (shot,), artifact_dir / "scenes" / f"{scene_id}_thumb.jpg"))
        else:
            thumb_path = segment.keyframe_path or ""
        scenes.append(replace(scene, scene_thumb_path=thumb_path))
    return VideoIndex(video_path=scene_index.video_path, duration_sec=scene_index.duration_sec, scenes=tuple(scenes))


def _compose_shot_grid_or_empty(*, shot: Shot, frames: Sequence[Frame], artifact_dir: Path) -> str:
    if not frames:
        return ""
    try:
        return str(compose_shot_grid(frames, artifact_dir / "shots" / f"{shot.shot_id}_grid.jpg"))
    except (OSError, ValueError):
        return ""


def _detect_shot_ranges(
    video_path: str,
    duration_sec: float,
    *,
    detector: ShotDetector | None,
) -> Sequence[tuple[float, float]]:
    if detector is not None:
        return detector(str(video_path), float(duration_sec))
    try:
        return detect_shots_ffmpeg(str(video_path))
    except RuntimeError:
        return detect_shots_uniform(float(duration_sec), window=15.0)


def _default_keyframe_sampler(
    video_path: str,
    start_sec: float,
    end_sec: float,
    n_frames: int,
    out_dir: Path,
) -> Sequence[Frame]:
    return sample_shot_frames(
        video_path,
        start_sec,
        end_sec,
        n_frames=n_frames,
        out_dir=out_dir,
    )


def _with_frame_id(frame: Frame, *, shot_id: str, frame_number: int) -> Frame:
    return replace(frame, frame_id=f"{shot_id}_fr{frame_number:03d}")


def _asr_text_for_range(cues: Sequence[Any], *, start_sec: float, end_sec: float) -> str:
    lines = []
    for cue in cues:
        cue_start = float(_field(cue, "start_sec", 0.0) or 0.0)
        cue_end = float(_field(cue, "end_sec", cue_start) or cue_start)
        if cue_end < start_sec or cue_start > end_sec:
            continue
        text = str(_field(cue, "text", "") or "").strip()
        if text:
            lines.append(text)
    return " ".join(lines)


def _segments_for_range(segments: Sequence[VideoSegment], *, start_sec: float, end_sec: float) -> tuple[VideoSegment, ...]:
    overlapping = []
    for segment in segments:
        segment_start = float(segment.start_sec)
        segment_end = float(segment.end_sec)
        if segment_end <= start_sec or segment_start >= end_sec:
            continue
        overlapping.append(segment)
    return tuple(overlapping)


def _visual_caption_for_segments(segments: Sequence[VideoSegment]) -> str:
    return _bounded_text(" ".join(_unique_text(_segment_visual_text(segment) for segment in segments)), 360)


def _segment_visual_text(segment: VideoSegment) -> str:
    return segment.visual_caption or segment.low_fps_caption or segment.map_summary


def _entities_for_segments(segments: Sequence[VideoSegment]) -> tuple[str, ...]:
    return _unique_text(item for segment in segments for item in segment.entities)


def _topics_for_segments(segments: Sequence[VideoSegment]) -> tuple[str, ...]:
    return _unique_text(item for segment in segments for item in segment.topic_tags)


def _video_scene_title(scene_number: int, topics: Sequence[str]) -> str:
    if topics:
        return f"Scene {scene_number} ({', '.join(topics[:3])})"
    return f"Scene {scene_number}"


def _scene_summary_from_shots(shots: Sequence[Shot], *, source_segments: Sequence[VideoSegment] = ()) -> str:
    captions = " ".join(shot.visual_caption for shot in shots if shot.visual_caption).strip()
    if captions:
        return _bounded_text(captions, 360)
    segment_summary = " ".join(_unique_text(_scene_summary(segment) for segment in source_segments)).strip()
    if segment_summary:
        return _bounded_text(segment_summary, 360)
    asr = " ".join(shot.asr_text for shot in shots if shot.asr_text).strip()
    if asr:
        return _bounded_text(asr, 360)
    return f"{len(shots)} shot visual scene."


def _bounded_text(text: str, max_chars: int) -> str:
    text = " ".join(str(text or "").split())
    max_chars = max(0, int(max_chars))
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    sentence_end = max(text.rfind(".", 0, max_chars - 3), text.rfind("。", 0, max_chars - 3))
    if sentence_end >= max(0, max_chars // 2):
        return text[: sentence_end + 1]
    return text[: max_chars - 3].rstrip() + "..."


def _unique_text(values: Any) -> tuple[str, ...]:
    seen = set()
    result = []
    for value in values:
        text = str(value or "").strip()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return tuple(result)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _default_frame_paths(segment: VideoSegment) -> tuple[str, ...]:
    if segment.keyframe_path:
        return (segment.keyframe_path,)
    return ()


def _frame_time(segment: VideoSegment, frame_number: int, frame_count: int) -> float:
    if frame_count <= 1:
        return float(segment.start_sec)
    span = max(0.0, float(segment.end_sec) - float(segment.start_sec))
    return round(float(segment.start_sec) + span * float(frame_number - 1) / float(frame_count - 1), 3)


def _scene_title(segment: VideoSegment) -> str:
    summary = _scene_summary(segment)
    if not summary:
        return segment.segment_id
    return summary.split(".")[0][:80].strip() or segment.segment_id


def _scene_summary(segment: VideoSegment) -> str:
    return (
        segment.map_summary
        or segment.low_fps_caption
        or segment.visual_caption
        or segment.asr_summary
        or _segment_asr_text(segment)
        or "No scene summary available."
    )


def _segment_asr_text(segment: VideoSegment) -> str:
    if segment.asr_summary:
        return segment.asr_summary
    lines = []
    for item in segment.asr_sentences:
        if isinstance(item, dict) and item.get("text"):
            lines.append(str(item["text"]))
    return " ".join(lines)


def _segment_ocr_lines(segment: VideoSegment) -> tuple[str, ...]:
    lines = []
    for item in segment.ocr_frames:
        if isinstance(item, dict) and item.get("text"):
            lines.append(str(item["text"]))
    return tuple(lines)
