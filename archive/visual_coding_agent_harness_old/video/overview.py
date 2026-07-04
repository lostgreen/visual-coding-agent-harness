"""Chapter-level overview artifacts for the multi_v3 Reasoner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .index import Frame
from .pipeline import compose_shot_grid
from visual_coding_agent_harness.workspace.video_workspace import VideoWorkspace


@dataclass(frozen=True)
class SceneTimelineOverview:
    grid_image_path: str
    manifest_path: str

    @property
    def grid_path(self) -> str:
        return self.grid_image_path


def build_scene_timeline_overview(index: VideoWorkspace, *, output_dir: Path, cols: int = 8) -> SceneTimelineOverview:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "scene_timeline_grid.json"
    grid_image_path = output_dir / "scene_timeline_grid.jpg"
    payload = {
        "video_path": index.video_path,
        "duration_sec": index.duration_sec,
        "cols": int(cols),
        "chapters": [
            {
                "chapter_id": chapter.chapter_id,
                "start_sec": chapter.start_sec,
                "end_sec": chapter.end_sec,
                "title": chapter.title,
                "thumb_path": chapter.thumb_path,
                "beat_count": len(chapter.beat_ids),
            }
            for chapter in index.chapters
        ],
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    frames = tuple(
        Frame(frame_id=chapter.chapter_id, time_sec=chapter.start_sec, thumb_path=chapter.thumb_path)
        for chapter in index.chapters
        if chapter.thumb_path
    )
    if frames:
        compose_shot_grid(frames, grid_image_path, cols=cols)
    else:
        grid_image_path.write_bytes(b"")
    return SceneTimelineOverview(grid_image_path=str(grid_image_path), manifest_path=str(manifest_path))
