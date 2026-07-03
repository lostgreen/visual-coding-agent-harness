"""Builders and adapters for the multi_v3 layered video index."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

from .index import Frame, Scene, SceneIndex, Shot, VideoIndex, VideoSegment


FramePathBuilder = Callable[[VideoSegment], Sequence[str]]
ShotGridBuilder = Callable[[Shot, Sequence[Frame]], str]
SceneThumbBuilder = Callable[[Scene, Sequence[Shot]], str]


def build_video_index_from_scene_index(
    scene_index: SceneIndex,
    *,
    artifact_dir: Path,
    shot_frame_paths: FramePathBuilder | None = None,
    shot_grid_builder: ShotGridBuilder | None = None,
    scene_thumb_builder: SceneThumbBuilder | None = None,
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
        grid_path = (
            shot_grid_builder(shot, frames)
            if shot_grid_builder
            else str(artifact_dir / "shots" / f"{shot_id}_lowres_grid.json")
        )
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
        thumb_path = (
            scene_thumb_builder(scene, (shot,))
            if scene_thumb_builder
            else (segment.keyframe_path or str(artifact_dir / "scenes" / f"{scene_id}_thumb.json"))
        )
        scenes.append(replace(scene, scene_thumb_path=thumb_path))
    return VideoIndex(video_path=scene_index.video_path, duration_sec=scene_index.duration_sec, scenes=tuple(scenes))


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
