"""Scene-level overview artifacts for the multi_v3 Reasoner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .pipeline import compose_scene_timeline_grid
from .index import VideoIndex


@dataclass(frozen=True)
class SceneTimelineOverview:
    grid_image_path: str
    manifest_path: str

    @property
    def grid_path(self) -> str:
        return self.grid_image_path


def build_scene_timeline_overview(index: VideoIndex, *, output_dir: Path, cols: int = 8) -> SceneTimelineOverview:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "scene_timeline_grid.json"
    grid_image_path = output_dir / "scene_timeline_grid.jpg"
    payload = {
        "video_path": index.video_path,
        "duration_sec": index.duration_sec,
        "cols": int(cols),
        "scenes": [
            {
                "scene_id": scene.scene_id,
                "start_sec": scene.start_sec,
                "end_sec": scene.end_sec,
                "title": scene.title,
                "summary": scene.summary,
                "thumb_path": scene.scene_thumb_path,
                "entities": list(scene.dominant_entities),
            }
            for scene in index.scenes
        ],
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    compose_scene_timeline_grid(index.scenes, grid_image_path, cols=cols)
    return SceneTimelineOverview(grid_image_path=str(grid_image_path), manifest_path=str(manifest_path))
