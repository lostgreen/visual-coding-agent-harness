from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from visual_coding_agent_harness.video import _keyframes as keyframe_module
from visual_coding_agent_harness.video.build import build_video_index_from_scene_index, build_video_index_from_video
from visual_coding_agent_harness.video._artifacts import compose_scene_timeline_grid
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


def test_default_scene_index_adapter_renders_real_grid_and_thumb_images(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.jpg"
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(frame_path)
    source = SceneIndex(
        video_path="/videos/demo.mp4",
        duration_sec=10.0,
        segments=[
            VideoSegment(
                segment_id="seg_0001",
                start_sec=0.0,
                end_sec=10.0,
                keyframe_path=str(frame_path),
                visual_caption="A red frame.",
            )
        ],
    )

    index = build_video_index_from_scene_index(source, artifact_dir=tmp_path / "artifacts")

    grid_path = Path(index.scenes[0].shots[0].lowres_grid_path)
    thumb_path = Path(index.scenes[0].scene_thumb_path)
    assert grid_path.suffix == ".jpg"
    assert thumb_path.suffix == ".jpg"
    with Image.open(grid_path) as grid:
        assert grid.size == (320, 180)
    with Image.open(thumb_path) as thumb:
        assert thumb.size == (320, 180)


def test_build_video_index_from_video_uses_detected_shots_and_sampled_frames(tmp_path: Path) -> None:
    def fake_keyframes(video_path: str, start_sec: float, end_sec: float, n_frames: int, out_dir: Path):
        del video_path, start_sec, end_sec
        out_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for index in range(1, min(3, n_frames) + 1):
            path = out_dir / f"frame_{index:03d}.jpg"
            Image.new("RGB", (24, 24), color=(index * 30, 80, 120)).save(path)
            frames.append(Frame(frame_id=f"tmp_{index}", time_sec=float(index), thumb_path=str(path)))
        return tuple(frames)

    index = build_video_index_from_video(
        "/videos/demo.mp4",
        60.0,
        artifact_dir=tmp_path,
        shot_detector=lambda _video_path, _duration: ((0.0, 10.0), (10.0, 30.0), (30.0, 60.0)),
        keyframe_sampler=fake_keyframes,
        frames_per_shot=3,
    )

    assert len(index.scenes) == 1
    assert len(index.scenes[0].shots) == 3
    for shot in index.scenes[0].shots:
        assert len(shot.frames) == 3
        assert Path(shot.lowres_grid_path).exists()
        with Image.open(shot.lowres_grid_path) as grid:
            assert grid.size == (960, 180)


def test_sample_shot_frames_can_preserve_source_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_extract_frame(*, video_path: str, time_sec: float, output_path: Path) -> None:
        del video_path, time_sec
        Image.new("RGB", (800, 450), color=(120, 80, 40)).save(output_path)

    monkeypatch.setattr(keyframe_module, "_extract_frame", fake_extract_frame)

    frames = keyframe_module.sample_shot_frames(
        "/videos/demo.mp4",
        0.0,
        2.0,
        n_frames=1,
        out_dir=tmp_path / "verify_frames",
        size=None,
    )

    with Image.open(frames[0].thumb_path) as image:
        assert image.size == (800, 450)


def test_build_video_index_from_video_inherits_scene_index_semantics(tmp_path: Path) -> None:
    def fake_keyframes(video_path: str, start_sec: float, end_sec: float, n_frames: int, out_dir: Path):
        del video_path, start_sec, end_sec, n_frames
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "frame_001.jpg"
        Image.new("RGB", (24, 24), color=(80, 120, 160)).save(path)
        return (Frame(frame_id="tmp", time_sec=1.0, thumb_path=str(path)),)

    source_segments = (
        VideoSegment(
            segment_id="seg_0001",
            start_sec=0.0,
            end_sec=10.0,
            visual_caption="A chef chops onions on a counter.",
            entities=("chef", "onion"),
            topic_tags=("cooking",),
        ),
        VideoSegment(
            segment_id="seg_0002",
            start_sec=10.0,
            end_sec=20.0,
            low_fps_caption="A plated dish is shown.",
            entities=("dish",),
            topic_tags=("plating",),
        ),
    )

    index = build_video_index_from_video(
        "/videos/demo.mp4",
        20.0,
        artifact_dir=tmp_path,
        shot_detector=lambda _video_path, _duration: ((0.0, 10.0), (10.0, 20.0)),
        keyframe_sampler=fake_keyframes,
        source_segments=source_segments,
        frames_per_shot=1,
    )

    assert index.scenes[0].shots[0].visual_caption == "A chef chops onions on a counter."
    assert index.scenes[0].shots[0].entities == ("chef", "onion")
    assert index.scenes[0].shots[1].visual_caption == "A plated dish is shown."
    assert index.scenes[0].dominant_entities == ("chef", "onion", "dish")
    assert index.scenes[0].dominant_topics == ("cooking", "plating")
    assert "cooking" in index.scenes[0].title
    assert "chef chops onions" in index.scenes[0].summary


def test_placeholder_timeline_grid_does_not_draw_text(tmp_path: Path) -> None:
    scene = Scene(
        scene_id="sc99",
        start_sec=0.0,
        end_sec=1.0,
        title="Scene title should not be drawn",
        summary="Summary should not be drawn either",
        shots=(),
        scene_thumb_path="",
    )

    out_path = compose_scene_timeline_grid((scene,), tmp_path / "timeline.jpg")

    with Image.open(out_path) as image:
        colors = image.convert("RGB").getcolors(maxcolors=10_000)
    assert len(colors or ()) == 1
    assert colors[0][0] == 320 * 180


def test_scene_timeline_overview_writes_manifest_and_real_grid_image(tmp_path: Path) -> None:
    thumb_path = tmp_path / "sc01.jpg"
    Image.new("RGB", (32, 32), color=(0, 0, 255)).save(thumb_path)
    frame = Frame(frame_id="fr1", time_sec=2.0, thumb_path=str(thumb_path))
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
                scene_thumb_path=str(thumb_path),
            ),
        ),
    )

    overview = build_scene_timeline_overview(index, output_dir=tmp_path, cols=4)

    manifest = json.loads(Path(overview.manifest_path).read_text(encoding="utf-8"))
    assert overview.grid_path.endswith("scene_timeline_grid.jpg")
    assert overview.grid_image_path == overview.grid_path
    assert overview.manifest_path.endswith("scene_timeline_grid.json")
    with Image.open(overview.grid_image_path) as grid:
        assert grid.size == (320, 180)
    assert manifest["cols"] == 4
    assert manifest["scenes"][0]["scene_id"] == "sc01"
    assert manifest["scenes"][0]["thumb_path"] == str(thumb_path)
