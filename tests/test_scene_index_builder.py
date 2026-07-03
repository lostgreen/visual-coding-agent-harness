from __future__ import annotations

import json
from pathlib import Path
import threading
import time

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
from visual_coding_agent_harness.video.index import SceneIndex, VideoSegment
from visual_coding_agent_harness.video._map import VideoMap


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


class ManyBeatBackend:
    def __init__(self, beat_count: int) -> None:
        self.beat_count = beat_count
        self.requests: list[BackendRequest] = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        if request.task != "build_root_dvc_index":
            raise AssertionError(f"unexpected task: {request.task}")
        start_sec = float(request.metadata["start_sec"])
        end_sec = float(request.metadata["end_sec"])
        step = (end_sec - start_sec) / float(self.beat_count)
        beats = [
            {
                "start_sec": start_sec + step * index,
                "end_sec": start_sec + step * (index + 1),
                "summary": f"Beat {index + 1}.",
            }
            for index in range(self.beat_count)
        ]
        return BackendResponse(text=json.dumps({"root_summary": "Many beat summary.", "beats": beats}))


class FailsAfterFirstRootBackend(RecordingBackend):
    def generate(self, request: BackendRequest) -> BackendResponse:
        if request.task == "build_root_dvc_index" and len(self.requests) >= 1:
            raise RuntimeError("simulated root caption failure")
        return super().generate(request)


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
    assert backend.requests[0].metadata["schema_version"] == "dvc_root_v3"
    assert backend.requests[0].metadata["frame_cache_fps"] == 0.5
    assert backend.requests[0].metadata["max_pixels_per_frame"] == 360 * 420
    assert backend.requests[0].metadata["max_pixels"] == 360 * 420
    assert backend.requests[0].metadata["fps"] == 0.5
    assert backend.requests[0].metadata["nframes"] == 2
    assert "max_beats_per_root" not in backend.requests[0].metadata
    assert backend.requests[0].max_new_tokens == RootIndexPolicy().max_new_tokens
    assert "MAX_BEATS" not in backend.requests[0].prompt
    assert '"scene": "where the video is and what is visible"' in backend.requests[0].prompt
    assert '"event": "the simple action, narration point, or state change"' in backend.requests[0].prompt
    assert "Do not collapse the interval into only one broad overview" in backend.requests[0].prompt
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
        (0.0, 10.0, 5),
        (10.0, 20.0, 5),
        (20.0, 25.0, 2),
    ]
    assert all(request.media_path is None for request in visual_requests)
    assert all(request.media_type == "video" for request in visual_requests)
    assert [list(request.frames) for request in visual_requests] == [
        ["/frames/video-1/0000_a.jpg", "/frames/video-1/0000_b.jpg"],
        ["/frames/video-1/0010_a.jpg", "/frames/video-1/0010_b.jpg"],
        ["/frames/video-1/0020_a.jpg", "/frames/video-1/0020_b.jpg"],
    ]
    assert visual_requests[1].metadata["source_video_path"] == "/tmp/video-1.mp4"
    assert visual_requests[1].metadata["frame_cache_policy"] == "root_policy_fps"


def test_builder_root_dvc_uses_policy_fps_for_frame_sampling() -> None:
    backend = RecordingBackend()
    sampled = []

    def fake_frame_sampler(video_path: str, start_sec: float, end_sec: float, max_frames: int) -> list[str]:
        sampled.append((video_path, start_sec, end_sec, max_frames))
        return [f"/frames/video-1/{index:03d}.jpg" for index in range(max_frames)]

    builder = SceneIndexBuilder(
        backend=backend,
        text_model_id="text-mini",
        vl_model_id="vl-mini",
        window_sec=10.0,
        frame_sampler=fake_frame_sampler,
    )

    builder.build(
        video_id="video-1",
        video_path="/tmp/video-1.mp4",
        duration_sec=10.0,
        subtitle_cues=[],
    )

    assert sampled == [("/tmp/video-1.mp4", 0.0, 10.0, 5)]
    assert backend.requests[0].metadata["nframes"] == 5


def test_builder_preserves_all_root_dvc_beats_without_policy_cap() -> None:
    backend = ManyBeatBackend(beat_count=15)
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

    assert len(scene_index.segments[0].timeline_beats) == 15
    assert scene_index.segments[0].timeline_beats[-1].summary == "Beat 15."


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


def test_scene_index_cache_persists_and_resumes_partial_roots(tmp_path) -> None:
    cache = SceneIndexCache(tmp_path)
    failing_backend = FailsAfterFirstRootBackend()
    first_builder = SceneIndexBuilder(
        backend=failing_backend,
        cache=cache,
        text_model_id="text-mini",
        vl_model_id="vl-mini",
        window_sec=30.0,
        frame_sampler=_frame_sampler,
    )
    cache_key = first_builder.cache_key(
        video_id="video-1",
        video_path="/tmp/video-1.mp4",
        duration_sec=60.0,
        subtitle_cues=(),
    )

    with pytest.raises(RuntimeError, match="simulated root caption failure"):
        first_builder.build(
            video_id="video-1",
            video_path="/tmp/video-1.mp4",
            duration_sec=60.0,
            subtitle_cues=[],
        )

    partial = cache.load(cache_key)
    assert partial is not None
    assert partial.segments[0].source == SCENE_INDEX_BUILDER_SCHEMA_VERSION
    assert partial.segments[0].timeline_beats
    assert partial.segments[1].source == SCENE_INDEX_BUILDER_SCHEMA_VERSION
    assert not partial.segments[1].timeline_beats

    resume_backend = RecordingBackend()
    resumed = SceneIndexBuilder(
        backend=resume_backend,
        cache=cache,
        text_model_id="text-mini",
        vl_model_id="vl-mini",
        window_sec=30.0,
        frame_sampler=_frame_sampler,
    ).build(
        video_id="video-1",
        video_path="/tmp/video-1.mp4",
        duration_sec=60.0,
        subtitle_cues=[],
    )

    assert len(resume_backend.requests) == 1
    assert all(segment.source == SCENE_INDEX_BUILDER_SCHEMA_VERSION for segment in resumed.segments)
    assert all(segment.timeline_beats for segment in resumed.segments)


def test_scene_index_cache_rebuilds_damaged_root_dvc_caption(tmp_path) -> None:
    cache = SceneIndexCache(tmp_path)
    builder = SceneIndexBuilder(
        backend=RecordingBackend(),
        cache=cache,
        text_model_id="text-mini",
        vl_model_id="vl-mini",
        window_sec=30.0,
        frame_sampler=_frame_sampler,
    )
    cues = [SubtitleCue(start_sec=0.0, end_sec=1.0, text="Museum aircraft.", cue_id="cue-1")]
    cache_key = builder.cache_key(video_id="video-1", video_path="/tmp/video-1.mp4", duration_sec=30.0, subtitle_cues=cues)
    damaged = SceneIndex(
        video_path="/tmp/video-1.mp4",
        duration_sec=30.0,
        segments=[
            VideoSegment(
                segment_id="seg_0001",
                start_sec=0.0,
                end_sec=30.0,
                map_summary='{ "root_summary": "truncated caption',
                low_fps_caption='{ "root_summary": "truncated caption',
                source="dvc_root_v1",
                index_level="root",
            )
        ],
    )
    cache.store(cache_key, damaged)

    rebuilt = builder.build(video_id="video-1", video_path="/tmp/video-1.mp4", duration_sec=30.0, subtitle_cues=cues)

    assert len(builder.backend.requests) == 1
    assert rebuilt.segments[0].map_summary == "Museum aircraft intro with a silver plane in a hangar."


def test_root_dvc_policy_does_not_cap_timeline_beats() -> None:
    backend = ManyBeatBackend(beat_count=15)
    builder = SceneIndexBuilder(
        backend=backend,
        text_model_id="text-mini",
        vl_model_id="vl-mini",
        window_sec=30.0,
        root_policy=RootIndexPolicy(root_window_sec=30.0),
        frame_sampler=_frame_sampler,
    )

    scene_index = builder.build(
        video_id="video-1",
        video_path="/tmp/video-1.mp4",
        duration_sec=30.0,
        subtitle_cues=[SubtitleCue(start_sec=0.0, end_sec=1.0, text="Museum aircraft.", cue_id="cue-1")],
    )

    segment = scene_index.segments[0]
    assert len(segment.timeline_beats) == 15
    assert segment.timeline_beats[-1].summary == "Beat 15."


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
                                "scene": "A camera faces the curb beside the crosswalk.",
                                "event": "Traffic pauses while pedestrians finish crossing.",
                                "modality_hints": ["visual", "temporal"],
                            },
                            {
                                "start_offset_sec": 12.0,
                                "end_offset_sec": 16.0,
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
        "People cross a busy intersection.",
        "Scene: A camera faces the curb beside the crosswalk. Event: Traffic pauses while pedestrians finish crossing.",
        "Scene: root segment coverage. Event: A city street scene unfolds.",
    ]


def test_root_dvc_drops_empty_duration_beat_instead_of_failing_root_caption() -> None:
    class EmptyBeatBackend(RecordingBackend):
        def generate(self, request: BackendRequest) -> BackendResponse:
            self.requests.append(request)
            return BackendResponse(
                text=json.dumps(
                    {
                        "root_summary": "The root interval has a complete caption.",
                        "beats": [
                            {
                                "start_offset_sec": 30.0,
                                "end_offset_sec": 30.0,
                                "summary": "This malformed beat has no duration.",
                            },
                            {
                                "start_offset_sec": 30.0,
                                "end_offset_sec": 40.0,
                                "summary": "A valid follow-up beat remains usable.",
                            },
                        ],
                    }
                )
            )

    scene_index = SceneIndexBuilder(
        backend=EmptyBeatBackend(),
        text_model_id="text-mini",
        vl_model_id="vl-mini",
        window_sec=60.0,
        frame_sampler=_frame_sampler,
    ).build(
        video_id="video-1",
        video_path="/tmp/video-1.mp4",
        duration_sec=60.0,
        subtitle_cues=[],
    )

    segment = scene_index.segments[0]
    assert segment.map_summary == "The root interval has a complete caption."
    assert [beat.summary for beat in segment.timeline_beats] == [
        "Scene: root segment coverage. Event: The root interval has a complete caption.",
        "A valid follow-up beat remains usable.",
        "Scene: root segment coverage. Event: The root interval has a complete caption.",
    ]
    assert [(beat.start_sec, beat.end_sec) for beat in segment.timeline_beats] == [(0.0, 30.0), (30.0, 40.0), (40.0, 60.0)]
    assert "root_dvc_dropped_invalid_beats" in segment.limitations
    assert "root_dvc_synthesized_coverage_beats" in segment.limitations


def test_root_dvc_synthesizes_root_summary_from_beats_when_missing() -> None:
    class BeatsOnlyBackend(RecordingBackend):
        def generate(self, request: BackendRequest) -> BackendResponse:
            self.requests.append(request)
            return BackendResponse(
                text=json.dumps(
                    {
                        "beats": [
                            {
                                "start_offset_sec": 0.0,
                                "end_offset_sec": 10.0,
                                "summary": "A map introduces the region.",
                            },
                            {
                                "start_offset_sec": 10.0,
                                "end_offset_sec": 30.0,
                                "summary": "Narration explains a political border change.",
                            },
                        ]
                    }
                )
            )

    scene_index = SceneIndexBuilder(
        backend=BeatsOnlyBackend(),
        text_model_id="text-mini",
        vl_model_id="vl-mini",
        window_sec=30.0,
        frame_sampler=_frame_sampler,
    ).build(
        video_id="video-1",
        video_path="/tmp/video-1.mp4",
        duration_sec=30.0,
        subtitle_cues=[],
    )

    segment = scene_index.segments[0]
    assert segment.map_summary == "A map introduces the region. Narration explains a political border change."
    assert "root_dvc_missing_root_summary" in segment.limitations
    assert "root_dvc_synthesized_root_summary" not in segment.limitations


def test_root_dvc_synthesizes_root_summary_from_cues_when_payload_has_no_caption() -> None:
    class EmptyPayloadBackend(RecordingBackend):
        def generate(self, request: BackendRequest) -> BackendResponse:
            self.requests.append(request)
            return BackendResponse(
                text=json.dumps(
                    {
                        "beats": [
                            {
                                "start_offset_sec": 0.0,
                                "end_offset_sec": 0.0,
                                "summary": "",
                            }
                        ]
                    }
                )
            )

    scene_index = SceneIndexBuilder(
        backend=EmptyPayloadBackend(),
        text_model_id="text-mini",
        vl_model_id="vl-mini",
        window_sec=30.0,
        frame_sampler=_frame_sampler,
    ).build(
        video_id="video-1",
        video_path="/tmp/video-1.mp4",
        duration_sec=30.0,
        subtitle_cues=[
            SubtitleCue(start_sec=1.0, end_sec=3.0, text="The speaker describes the first scene.", cue_id="cue-1"),
            SubtitleCue(start_sec=8.0, end_sec=9.0, text="A second event is mentioned.", cue_id="cue-2"),
        ],
    )

    segment = scene_index.segments[0]
    assert segment.map_summary == "The speaker describes the first scene. A second event is mentioned."
    assert len(segment.timeline_beats) == 1
    assert segment.timeline_beats[0].start_sec == 0.0
    assert segment.timeline_beats[0].end_sec == 30.0
    assert segment.timeline_beats[0].limitations == ("root_dvc_synthesized_coverage_gap",)
    assert "Scene: root segment coverage. Event: The speaker describes the first scene." in segment.timeline_beats[0].summary
    assert "root_dvc_missing_root_summary" in segment.limitations
    assert "root_dvc_synthesized_root_summary" in segment.limitations
    assert "root_dvc_dropped_invalid_beats" in segment.limitations
    assert "root_dvc_synthesized_coverage_beats" in segment.limitations


def test_root_dvc_uses_plain_text_response_as_degraded_root_summary() -> None:
    class PlainTextBackend(RecordingBackend):
        def generate(self, request: BackendRequest) -> BackendResponse:
            self.requests.append(request)
            return BackendResponse(text="The clip shows a presenter walking through a city square.")

    backend = PlainTextBackend()
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

    segment = scene_index.segments[0]
    assert segment.map_summary == "The clip shows a presenter walking through a city square."
    assert len(segment.timeline_beats) == 1
    assert segment.timeline_beats[0].summary == (
        "Scene: root segment coverage. Event: The clip shows a presenter walking through a city square."
    )
    assert "root_dvc_synthesized_coverage_beats" in segment.limitations
    assert "root_dvc_non_json_response" in segment.limitations


def test_root_dvc_extracts_caption_from_truncated_json_without_polluting_map_summary() -> None:
    class TruncatedJsonBackend(RecordingBackend):
        def generate(self, request: BackendRequest) -> BackendResponse:
            self.requests.append(request)
            return BackendResponse(
                text=(
                    '{ "root_summary": "The clip surveys the empire collapse with maps and narration.", '
                    '"beats": [ { "start_offset_sec": 0, "end_offset_sec": 30, "summary": "opening map'
                )
            )

    backend = TruncatedJsonBackend()
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

    segment = scene_index.segments[0]
    assert segment.map_summary == "The clip surveys the empire collapse with maps and narration."
    assert not segment.map_summary.lstrip().startswith("{")
    assert len(segment.timeline_beats) == 1
    assert segment.timeline_beats[0].summary == (
        "Scene: root segment coverage. Event: The clip surveys the empire collapse with maps and narration."
    )
    assert "root_dvc_synthesized_coverage_beats" in segment.limitations
    assert "root_dvc_truncated_json_response" in segment.limitations


def test_root_dvc_rejects_unusable_jsonish_response_instead_of_using_it_as_caption() -> None:
    class BadJsonBackend(RecordingBackend):
        def generate(self, request: BackendRequest) -> BackendResponse:
            self.requests.append(request)
            return BackendResponse(text='{ "beats": [ { "summary": "opening map" ')

    builder = SceneIndexBuilder(
        backend=BadJsonBackend(),
        text_model_id="text-mini",
        vl_model_id="vl-mini",
        window_sec=30.0,
        frame_sampler=_frame_sampler,
    )

    with pytest.raises(ValueError, match="not valid JSON"):
        builder.build(
            video_id="video-1",
            video_path="/tmp/video-1.mp4",
            duration_sec=30.0,
            subtitle_cues=[],
        )


def test_root_dvc_uses_root_caption_alias_when_root_summary_is_missing() -> None:
    class CaptionAliasBackend(RecordingBackend):
        def generate(self, request: BackendRequest) -> BackendResponse:
            self.requests.append(request)
            return BackendResponse(text=json.dumps({"caption": "A complete root caption from the model."}))

    scene_index = SceneIndexBuilder(
        backend=CaptionAliasBackend(),
        text_model_id="text-mini",
        vl_model_id="vl-mini",
        window_sec=30.0,
        frame_sampler=_frame_sampler,
    ).build(
        video_id="video-1",
        video_path="/tmp/video-1.mp4",
        duration_sec=30.0,
        subtitle_cues=[],
    )

    assert scene_index.segments[0].map_summary == "A complete root caption from the model."


def test_root_dvc_cache_key_uses_stable_policy_fields() -> None:
    backend = RecordingBackend()
    builder = SceneIndexBuilder(
        backend=backend,
        text_model_id="text-mini",
        vl_model_id="vl-mini",
        window_sec=30.0,
    )
    cues = [SubtitleCue(start_sec=0.0, end_sec=1.0, text="Museum aircraft.", cue_id="cue-1")]

    key = builder.cache_key(video_id="video-1", video_path="/tmp/video-1.mp4", duration_sec=30.0, subtitle_cues=cues)

    assert SCENE_INDEX_BUILDER_SCHEMA_VERSION == "dvc_root_v3"
    assert key == SceneIndexCache(Path("/tmp/unused")).key_for(
        {
            "schema_version": "dvc_root_v3",
            "video_id": "video-1",
            "video_path": "/tmp/video-1.mp4",
            "duration_sec": 30.0,
            "root_window_sec": 30.0,
            "frame_cache_fps": 0.5,
            "max_pixels_per_frame": 360 * 420,
            "max_new_tokens": 6144,
            "subtitle_hash": subtitle_hash(cues),
            "vl_model_id": "vl-mini",
        }
    )


def test_root_dvc_ignores_legacy_or_beatless_cache(tmp_path: Path) -> None:
    backend = RecordingBackend()
    cache = SceneIndexCache(tmp_path / "scene_cache")
    builder = SceneIndexBuilder(
        backend=backend,
        text_model_id="text-mini",
        vl_model_id="vl-mini",
        window_sec=30.0,
        cache=cache,
        frame_sampler=_frame_sampler,
    )
    cues: list[SubtitleCue] = []
    cache_key = builder.cache_key(
        video_id="video-1",
        video_path="/tmp/video-1.mp4",
        duration_sec=30.0,
        subtitle_cues=cues,
    )
    cache.store(
        cache_key,
        SceneIndex(
            video_path="/tmp/video-1.mp4",
            duration_sec=30.0,
            segments=[
                VideoSegment(
                    segment_id="seg_0001",
                    start_sec=0.0,
                    end_sec=30.0,
                    source="dvc_root_v1",
                    map_summary="legacy root summary without beats",
                    low_fps_caption="legacy root summary without beats",
                )
            ],
        ),
    )

    scene_index = builder.build(
        video_id="video-1",
        video_path="/tmp/video-1.mp4",
        duration_sec=30.0,
        subtitle_cues=cues,
    )

    assert [request.task for request in backend.requests] == ["build_root_dvc_index"]
    assert scene_index.segments[0].timeline_beats


def test_builder_can_build_root_dvc_segments_concurrently() -> None:
    class ConcurrentBackend(RecordingBackend):
        def __init__(self) -> None:
            super().__init__()
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def generate(self, request: BackendRequest) -> BackendResponse:
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(0.02)
                return super().generate(request)
            finally:
                with self.lock:
                    self.active -= 1

    backend = ConcurrentBackend()
    builder = SceneIndexBuilder(
        backend=backend,
        text_model_id="text-mini",
        vl_model_id="vl-mini",
        window_sec=30.0,
        frame_sampler=_frame_sampler,
        root_concurrency=3,
    )

    scene_index = builder.build(
        video_id="video-1",
        video_path="/tmp/video-1.mp4",
        duration_sec=90.0,
        subtitle_cues=[],
    )

    assert [segment.segment_id for segment in scene_index.segments] == ["seg_0001", "seg_0002", "seg_0003"]
    assert backend.max_active > 1


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
