from __future__ import annotations

import json
from pathlib import Path

import pytest

from visual_coding_agent_harness.video.build import build_video_index_from_scene_index
from visual_coding_agent_harness.video.index import Frame, Scene, SceneIndex, Shot, VideoIndex, VideoSegment
from visual_coding_agent_harness.video.overview import build_scene_timeline_overview


def test_video_index_validates_hierarchy_boundaries() -> None:
    with pytest.raises(ValueError, match="within scene"):
        Scene(
            scene_id="sc01",
            start_sec=10.0,
            end_sec=20.0,
            title="Out of range",
            summary="Shot starts before scene.",
            shots=(
                Shot(
                    shot_id="sc01_sh001",
                    scene_id="sc01",
                    start_sec=5.0,
                    end_sec=12.0,
                    frames=(),
                    visual_caption="",
                    asr_text="",
                    ocr_lines=(),
                    entities=(),
                    lowres_grid_path="",
                ),
            ),
        )


def test_build_video_index_from_legacy_scene_index_creates_scene_shot_frame_layers(tmp_path: Path) -> None:
    source = SceneIndex(
        video_path="/videos/demo.mp4",
        duration_sec=65.0,
        segments=[
            VideoSegment(
                segment_id="seg_0001",
                start_sec=0.0,
                end_sec=30.0,
                keyframe_path="/frames/0001.jpg",
                low_fps_caption="A chef chops onions.",
                asr_summary="The narrator mentions onions.",
                ocr_frames=({"time_sec": 4.0, "text": "ONION"},),
                entities=("chef", "onion"),
                topic_tags=("cooking",),
            ),
            VideoSegment(
                segment_id="seg_0002",
                start_sec=30.0,
                end_sec=65.0,
                keyframe_path="/frames/0002.jpg",
                visual_caption="The dish is plated.",
                asr_sentences=({"start_sec": 33.0, "end_sec": 35.0, "text": "The dish is ready."},),
            ),
        ],
    )

    index = build_video_index_from_scene_index(
        source,
        artifact_dir=tmp_path,
        shot_frame_paths=lambda segment: (f"/thumbs/{segment.segment_id}_a.jpg", f"/thumbs/{segment.segment_id}_b.jpg"),
        shot_grid_builder=lambda shot, frames: str(tmp_path / f"{shot.shot_id}.grid.jpg"),
        scene_thumb_builder=lambda scene, shots: str(tmp_path / f"{scene.scene_id}.thumb.jpg"),
    )

    assert isinstance(index, VideoIndex)
    assert [scene.scene_id for scene in index.scenes] == ["sc01", "sc02"]
    assert index.scenes[0].shots[0].shot_id == "sc01_sh001"
    assert [frame.frame_id for frame in index.scenes[0].shots[0].frames] == ["sc01_sh001_fr001", "sc01_sh001_fr002"]
    assert index.scenes[0].shots[0].lowres_grid_path.endswith("sc01_sh001.grid.jpg")
    assert index.scenes[0].scene_thumb_path.endswith("sc01.thumb.jpg")
    assert "onions" in index.scenes[0].summary
    assert index.to_scene_index().segments[0].segment_id == "seg_0001"
    assert VideoIndex.from_dict(index.to_dict()) == index


def test_scene_timeline_overview_writes_manifest_with_scene_thumbs(tmp_path: Path) -> None:
    frame = Frame(frame_id="fr1", time_sec=2.0, thumb_path="/thumbs/fr1.jpg")
    index = VideoIndex(
        video_path="/videos/demo.mp4",
        duration_sec=12.0,
        scenes=(
            Scene(
                scene_id="sc01",
                start_sec=0.0,
                end_sec=12.0,
                title="Intro",
                summary="A host introduces the clip.",
                shots=(
                    Shot(
                        shot_id="sc01_sh001",
                        scene_id="sc01",
                        start_sec=0.0,
                        end_sec=12.0,
                        frames=(frame,),
                        visual_caption="A host in a studio.",
                        asr_text="Welcome.",
                        ocr_lines=(),
                        entities=("host",),
                        lowres_grid_path="/grids/sc01_sh001.jpg",
                    ),
                ),
                dominant_entities=("host",),
                dominant_topics=("intro",),
                scene_thumb_path="/thumbs/sc01.jpg",
            ),
        ),
    )

    overview = build_scene_timeline_overview(index, output_dir=tmp_path, cols=4)

    manifest = json.loads(Path(overview.manifest_path).read_text(encoding="utf-8"))
    assert overview.grid_path.endswith("scene_timeline_grid.json")
    assert manifest["cols"] == 4
    assert manifest["scenes"][0]["scene_id"] == "sc01"
    assert manifest["scenes"][0]["thumb_path"] == "/thumbs/sc01.jpg"
