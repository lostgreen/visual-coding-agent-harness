from __future__ import annotations

import json
from pathlib import Path

import pytest

from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse
from visual_coding_agent_harness.evals.videomme.scene_index_builder import (
    RootIndexPolicy,
    SCENE_INDEX_BUILDER_SCHEMA_VERSION,
    SceneIndexBuilder,
    SubtitleCue,
    subtitle_hash,
)
from visual_coding_agent_harness.evals.videomme.scene_index_cache import SceneIndexCache
from visual_coding_agent_harness.video_index import SceneIndex, VideoSegment
from visual_coding_agent_harness.video_map import VideoMap


class RecordingBackend:
    def __init__(self) -> None:
        self.requests: list[BackendRequest] = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        if request.task == "build_root_dvc_index":
            start_sec = float(request.metadata["start_sec"])
            end_sec = float(request.metadata["end_sec"])
            duration = end_sec - start_sec
            split = min(12.0, duration)
            beats = [
                {
                    "start_offset_sec": 0.0,
                    "end_offset_sec": split,
                    "summary": "A narrator introduces the aircraft museum.",
                    "entity_hints": ["narrator", "aircraft"],
                    "modality_hints": ["asr"],
                    "confidence": 0.82,
                }
            ]
            if split < duration:
                beats.append(
                    {
                        "start_offset_sec": split,
                        "end_offset_sec": duration,
                        "summary": "A silver plane is shown inside a hangar.",
                        "entity_hints": ["plane", "hangar"],
                        "modality_hints": ["visual"],
                    }
                )
            return BackendResponse(
                text=json.dumps(
                    {
                        "root_summary": "Museum aircraft intro with a silver plane in a hangar.",
                        "beats": beats,
                        "entities": ["narrator", "aircraft"],
                        "topic_tags": ["museum"],
                        "confidence": 0.82,
                        "limitations": ["navigation only"],
                    }
                )
            )
        raise AssertionError(f"unexpected task: {request.task}")


def _frame_sampler(video_path: str, start_sec: float, end_sec: float, max_frames: int) -> list[str]:
    del video_path, end_sec
    return [f"/frames/video-1/{int(start_sec):04d}_{index:03d}.jpg" for index in range(max(1, min(max_frames, 2)))]


def test_builder_creates_single_call_root_dvc_manifest() -> None:
    backend = RecordingBackend()
    builder = SceneIndexBuilder(
        backend=backend,
        text_model_id="text-mini",
        vl_model_id="vl-mini",
        window_sec=30.0,
        caption_nframes=12,
        frame_sampler=_frame_sampler,
    )

    scene_index = builder.build(
        video_id="video-1",
        video_path="/tmp/video-1.mp4",
        duration_sec=30.0,
        subtitle_cues=[
            SubtitleCue(start_sec=0.0, end_sec=4.0, text="Welcome to the aircraft museum.", cue_id="cue-1"),
            SubtitleCue(start_sec=8.0, end_sec=9.0, text="This plane is silver.", cue_id="cue-2"),
        ],
    )

    assert [request.task for request in backend.requests] == ["build_root_dvc_index"]
    assert backend.requests[0].metadata["schema_version"] == "dvc_root_v1"
    assert backend.requests[0].metadata["frame_cache_fps"] == 1.0
    assert backend.requests[0].metadata["max_pixels_per_frame"] == 360 * 420
    segment = scene_index.segments[0]
    assert segment.source_segment_id == "seg_0001"
    assert segment.map_summary == "Museum aircraft intro with a silver plane in a hangar."
    assert segment.low_fps_caption == "Museum aircraft intro with a silver plane in a hangar."
    assert segment.visual_caption_source == "build_root_dvc_index:vl-mini"
    assert segment.asr_sentences == (
        {"cue_id": "cue-1", "start_sec": 0.0, "end_sec": 4.0, "text": "Welcome to the aircraft museum."},
        {"cue_id": "cue-2", "start_sec": 8.0, "end_sec": 9.0, "text": "This plane is silver."},
    )
    assert [beat.summary for beat in segment.timeline_beats] == [
        "A narrator introduces the aircraft museum.",
        "A silver plane is shown inside a hangar.",
    ]
    assert segment.timeline_beats[0].start_sec == 0.0
    assert segment.timeline_beats[1].end_sec == 30.0
    assert segment.timeline_beats[0].modality_hints == ("asr",)
    assert segment.topic_tags == ("museum",)
    assert segment.confidence == 0.82
    assert segment.citation_provenance["index"] == "navigation_only"


def test_builder_requires_frame_cache_instead_of_physical_clip_fallback(tmp_path) -> None:
    backend = RecordingBackend()
    builder = SceneIndexBuilder(
        backend=backend,
        text_model_id="text-mini",
        vl_model_id="vl-mini",
        window_sec=10.0,
        clip_root=tmp_path / "clips",
    )

    with pytest.raises(ValueError, match="frame cache"):
        builder.build(
            video_id="video 1",
            video_path="/tmp/video-1.mp4",
            duration_sec=25.0,
            subtitle_cues=[SubtitleCue(start_sec=0.0, end_sec=1.0, text="Museum aircraft.", cue_id="cue-1")],
        )

    assert backend.requests == []


def test_builder_prefers_frame_cache_over_physical_clips(tmp_path) -> None:
    backend = RecordingBackend()
    sampled = []
    extracted = []

    def fake_frame_sampler(video_path: str, start_sec: float, end_sec: float, max_frames: int) -> list[str]:
        sampled.append((video_path, start_sec, end_sec, max_frames))
        return [
            f"/frames/video-1/{int(start_sec):04d}_a.jpg",
            f"/frames/video-1/{int(start_sec):04d}_b.jpg",
        ]

    def fake_extract(video_path: str, output_path: str, start_sec: float, end_sec: float) -> str:
        extracted.append((video_path, output_path, start_sec, end_sec))
        return output_path

    builder = SceneIndexBuilder(
        backend=backend,
        text_model_id="text-mini",
        vl_model_id="vl-mini",
        window_sec=10.0,
        caption_nframes=12,
        clip_root=tmp_path / "clips",
        clip_extractor=fake_extract,
        frame_sampler=fake_frame_sampler,
    )

    builder.build(
        video_id="video 1",
        video_path="/tmp/video-1.mp4",
        duration_sec=25.0,
        subtitle_cues=[SubtitleCue(start_sec=0.0, end_sec=1.0, text="Museum aircraft.", cue_id="cue-1")],
    )

    visual_requests = [request for request in backend.requests if request.task == "build_root_dvc_index"]

    assert extracted == []
    assert [(round(call[1], 3), round(call[2], 3), call[3]) for call in sampled] == [
        (0.0, 10.0, 10),
        (10.0, 20.0, 10),
        (20.0, 25.0, 5),
    ]
    assert all(request.media_path is None for request in visual_requests)
    assert all(request.media_type == "video" for request in visual_requests)
    assert [list(request.frames) for request in visual_requests] == [
        ["/frames/video-1/0000_a.jpg", "/frames/video-1/0000_b.jpg"],
        ["/frames/video-1/0010_a.jpg", "/frames/video-1/0010_b.jpg"],
        ["/frames/video-1/0020_a.jpg", "/frames/video-1/0020_b.jpg"],
    ]
    assert visual_requests[1].metadata["source_video_path"] == "/tmp/video-1.mp4"
    assert visual_requests[1].metadata["frame_cache_policy"] == "precomputed_2fps"


def test_summary_uses_one_line_map_not_full_dual_source_detail() -> None:
    backend = RecordingBackend()
    builder = SceneIndexBuilder(backend=backend, text_model_id="text-mini", vl_model_id="vl-mini")
    builder.frame_sampler = _frame_sampler

    scene_index = builder.build(
        video_id="video-1",
        video_path="/tmp/video-1.mp4",
        duration_sec=30.0,
        subtitle_cues=[SubtitleCue(start_sec=0.0, end_sec=1.0, text="Museum aircraft.", cue_id="cue-1")],
    )

    summary = scene_index.summary(max_caption_chars=80)

    assert "seg_0001 [0.0-30.0s] Museum aircraft intro with a silver plane in a hangar." in summary
    assert "ASR:" not in summary
    assert "Visual:" not in summary
    assert "Tags:" not in summary
    assert "Entities:" not in summary
    assert "supported_option" not in summary
    assert "answer_option" not in summary
    assert "candidate_option_relations" not in summary


class NestedJsonBackend(RecordingBackend):
    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        if request.task == "build_root_dvc_index":
            return BackendResponse(text=json.dumps({"root_summary": json.dumps({"summary": "Clean one-line map."})}))
        raise AssertionError(f"unexpected task: {request.task}")


def test_builder_unwraps_nested_generated_json_fields() -> None:
    backend = NestedJsonBackend()
    builder = SceneIndexBuilder(backend=backend, text_model_id="text-mini", vl_model_id="vl-mini")
    builder.frame_sampler = _frame_sampler

    scene_index = builder.build(
        video_id="video-1",
        video_path="/tmp/video-1.mp4",
        duration_sec=30.0,
        subtitle_cues=[SubtitleCue(start_sec=0.0, end_sec=1.0, text="Museum aircraft.", cue_id="cue-1")],
    )

    segment = scene_index.segments[0]

    assert segment.map_summary == "Clean one-line map."
    assert "summary" not in scene_index.summary()


def test_scene_index_cache_avoids_duplicate_backend_calls(tmp_path) -> None:
    backend = RecordingBackend()
    cache = SceneIndexCache(tmp_path)
    builder = SceneIndexBuilder(
        backend=backend,
        cache=cache,
        text_model_id="text-mini",
        vl_model_id="vl-mini",
        window_sec=30.0,
        frame_sampler=_frame_sampler,
    )
    cues = [SubtitleCue(start_sec=0.0, end_sec=1.0, text="Museum aircraft.", cue_id="cue-1")]

    first = builder.build(video_id="video-1", video_path="/tmp/video-1.mp4", duration_sec=30.0, subtitle_cues=cues)
    second = builder.build(video_id="video-1", video_path="/tmp/video-1.mp4", duration_sec=30.0, subtitle_cues=cues)

    assert len(backend.requests) == 1
    assert first.to_dict() == second.to_dict()


def test_root_dvc_policy_caps_timeline_beats() -> None:
    backend = RecordingBackend()
    builder = SceneIndexBuilder(
        backend=backend,
        text_model_id="text-mini",
        vl_model_id="vl-mini",
        window_sec=30.0,
        root_policy=RootIndexPolicy(root_window_sec=30.0, max_beats_per_root=1),
        frame_sampler=_frame_sampler,
    )

    scene_index = builder.build(
        video_id="video-1",
        video_path="/tmp/video-1.mp4",
        duration_sec=30.0,
        subtitle_cues=[SubtitleCue(start_sec=0.0, end_sec=1.0, text="Museum aircraft.", cue_id="cue-1")],
    )

    assert len(scene_index.segments[0].timeline_beats) == 1


def test_root_dvc_tolerates_beat_summary_aliases_and_drops_unusable_beats() -> None:
    class AliasBackend(RecordingBackend):
        def generate(self, request: BackendRequest) -> BackendResponse:
            self.requests.append(request)
            return BackendResponse(
                text=json.dumps(
                    {
                        "root_summary": "A city street scene unfolds.",
                        "timeline_beats": [
                            {
                                "start_offset_sec": 0.0,
                                "end_offset_sec": 8.0,
                                "description": "People cross a busy intersection.",
                                "modality_hints": ["visual"],
                            },
                            {
                                "start_offset_sec": 8.0,
                                "end_offset_sec": 12.0,
                                "entity_hints": ["traffic light"],
                            },
                        ],
                    }
                )
            )

    backend = AliasBackend()
    builder = SceneIndexBuilder(
        backend=backend,
        text_model_id="text-mini",
        vl_model_id="vl-mini",
        window_sec=30.0,
        frame_sampler=_frame_sampler,
    )

    scene_index = builder.build(
        video_id="video-1",
        video_path="/tmp/video-1.mp4",
        duration_sec=30.0,
        subtitle_cues=[],
    )

    assert [beat.summary for beat in scene_index.segments[0].timeline_beats] == [
        "People cross a busy intersection."
    ]


def test_root_dvc_cache_key_uses_stable_policy_fields() -> None:
    backend = RecordingBackend()
    builder = SceneIndexBuilder(
        backend=backend,
        text_model_id="text-mini",
        vl_model_id="vl-mini",
        window_sec=30.0,
        caption_nframes=99,
    )
    cues = [SubtitleCue(start_sec=0.0, end_sec=1.0, text="Museum aircraft.", cue_id="cue-1")]

    key = builder.cache_key(video_id="video-1", video_path="/tmp/video-1.mp4", duration_sec=30.0, subtitle_cues=cues)

    assert SCENE_INDEX_BUILDER_SCHEMA_VERSION == "dvc_root_v1"
    assert key == SceneIndexCache(Path("/tmp/unused")).key_for(
        {
            "schema_version": "dvc_root_v1",
            "video_id": "video-1",
            "video_path": "/tmp/video-1.mp4",
            "duration_sec": 30.0,
            "root_window_sec": 30.0,
            "frame_cache_fps": 1.0,
            "max_pixels_per_frame": 360 * 420,
            "subtitle_hash": subtitle_hash(cues),
            "vl_model_id": "vl-mini",
        }
    )


def test_old_scene_index_fixtures_load_with_default_index_fields() -> None:
    payload = {
        "video_path": "/tmp/video-1.mp4",
        "duration_sec": 30.0,
        "segments": [
            {
                "segment_id": "seg_0001",
                "start_sec": 0.0,
                "end_sec": 30.0,
                "map_summary": "old map",
            }
        ],
    }

    scene_index = SceneIndex.from_dict(payload)
    segment = scene_index.segments[0]

    assert segment.index_level == "root"
    assert segment.refinement_state == "coarse"
    assert segment.timeline_beats == ()


def test_video_map_from_scene_index_preserves_timeline_beats() -> None:
    scene_index = SceneIndex(
        video_path="/tmp/video-1.mp4",
        duration_sec=30.0,
        segments=[
            VideoSegment.from_dict(
                {
                    "segment_id": "seg_0001",
                    "start_sec": 0.0,
                    "end_sec": 30.0,
                    "map_summary": "root summary",
                    "timeline_beats": [
                        {
                            "beat_id": "seg_0001_b01",
                            "start_sec": 0.0,
                            "end_sec": 10.0,
                            "summary": "opening",
                            "modality_hints": ["visual"],
                        }
                    ],
                }
            )
        ],
    )

    video_map = VideoMap.from_scene_index(scene_index)
    segment = video_map.segments[0]

    assert segment.index_level == "root"
    assert segment.timeline_beats[0].beat_id == "seg_0001_b01"
    assert segment.to_dict()["timeline_beats"][0]["summary"] == "opening"
