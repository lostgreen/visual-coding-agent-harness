from __future__ import annotations

import json
from pathlib import Path

from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse
from visual_coding_agent_harness.evals.videomme.scene_index_builder import (
    SceneIndexBuilder,
    SubtitleCue,
)
from visual_coding_agent_harness.evals.videomme.scene_index_cache import SceneIndexCache


class RecordingBackend:
    def __init__(self) -> None:
        self.requests: list[BackendRequest] = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        if request.task == "summarize_subtitle_segment":
            return BackendResponse(
                text=json.dumps(
                    {
                        "summary": "Narrator describes the museum aircraft.",
                        "entities": ["narrator", "aircraft"],
                        "topic_tags": ["museum"],
                        "confidence": 0.82,
                        "raw_asr_ref": "cue-1,cue-2",
                        "supported_option": "A",
                    }
                )
            )
        if request.task == "caption_scene_segment":
            return BackendResponse(
                text=(
                    "prefix "
                    + json.dumps(
                        {
                            "caption": "A wide shot shows a silver plane in a hangar.",
                            "stage_tags": ["wide shot"],
                            "entities": ["plane", "hangar"],
                            "grounding_quality": "medium",
                            "answer_option": "B",
                        }
                    )
                )
            )
        raise AssertionError(f"unexpected task: {request.task}")


def test_builder_keeps_asr_and_visual_sources_separate() -> None:
    backend = RecordingBackend()
    builder = SceneIndexBuilder(
        backend=backend,
        text_model_id="text-mini",
        vl_model_id="vl-mini",
        window_sec=30.0,
        caption_nframes=12,
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

    assert [request.task for request in backend.requests] == [
        "summarize_subtitle_segment",
        "caption_scene_segment",
    ]
    assert backend.requests[1].metadata["nframes"] == 12
    segment = scene_index.segments[0]
    assert segment.source_segment_id == "seg_0001"
    assert segment.asr_summary == "Narrator describes the museum aircraft."
    assert segment.asr_summary_source == "summarize_subtitle_segment:text-mini"
    assert segment.visual_caption == "A wide shot shows a silver plane in a hangar."
    assert segment.visual_caption_source == "caption_scene_segment:vl-mini"
    assert segment.raw_asr_ref == "cue-1,cue-2"
    assert segment.stage_tags == ("wide shot",)
    assert segment.entities == ("narrator", "aircraft", "plane", "hangar")
    assert segment.topic_tags == ("museum",)
    assert segment.confidence == 0.82
    assert segment.grounding_quality == "medium"
    assert segment.citation_provenance["asr"] == "subtitle"
    assert segment.citation_provenance["visual"] == "video"


def test_builder_captions_physical_segment_clips(tmp_path) -> None:
    backend = RecordingBackend()
    calls = []

    def fake_extract(video_path: str, output_path: str, start_sec: float, end_sec: float) -> str:
        calls.append((video_path, output_path, start_sec, end_sec))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("clip", encoding="utf-8")
        return output_path

    builder = SceneIndexBuilder(
        backend=backend,
        text_model_id="text-mini",
        vl_model_id="vl-mini",
        window_sec=10.0,
        clip_root=tmp_path / "clips",
        clip_extractor=fake_extract,
    )

    builder.build(
        video_id="video 1",
        video_path="/tmp/video-1.mp4",
        duration_sec=25.0,
        subtitle_cues=[SubtitleCue(start_sec=0.0, end_sec=1.0, text="Museum aircraft.", cue_id="cue-1")],
    )

    visual_requests = [request for request in backend.requests if request.task == "caption_scene_segment"]

    assert len(visual_requests) == 3
    assert [(round(call[2], 3), round(call[3], 3)) for call in calls] == [(0.0, 10.0), (10.0, 20.0), (20.0, 25.0)]
    assert all(call[0] == "/tmp/video-1.mp4" for call in calls)
    assert all(request.media_path != "/tmp/video-1.mp4" for request in visual_requests)
    assert [request.media_path for request in visual_requests] == [call[1] for call in calls]
    assert visual_requests[1].metadata["source_video_path"] == "/tmp/video-1.mp4"
    assert visual_requests[1].metadata["clip_path"] == calls[1][1]
    assert "video_1_seg_0002_10000_20000.mp4" in calls[1][1]


def test_summary_uses_dual_source_fields_and_does_not_leak_options() -> None:
    backend = RecordingBackend()
    builder = SceneIndexBuilder(backend=backend, text_model_id="text-mini", vl_model_id="vl-mini")

    scene_index = builder.build(
        video_id="video-1",
        video_path="/tmp/video-1.mp4",
        duration_sec=30.0,
        subtitle_cues=[SubtitleCue(start_sec=0.0, end_sec=1.0, text="Museum aircraft.", cue_id="cue-1")],
    )

    summary = scene_index.summary(max_caption_chars=40)

    assert "ASR: Narrator describes the museum aircraft." in summary
    assert "Visual: A wide shot shows a silver plane in a..." in summary
    assert summary.index("Visual:") < summary.index("ASR:")
    assert "Tags: museum, wide shot" in summary
    assert "Entities: narrator, aircraft, plane, hangar" in summary
    assert "supported_option" not in summary
    assert "answer_option" not in summary
    assert "candidate_option_relations" not in summary


def test_scene_index_cache_avoids_duplicate_backend_calls(tmp_path) -> None:
    backend = RecordingBackend()
    cache = SceneIndexCache(tmp_path)
    builder = SceneIndexBuilder(
        backend=backend,
        cache=cache,
        text_model_id="text-mini",
        vl_model_id="vl-mini",
        window_sec=30.0,
    )
    cues = [SubtitleCue(start_sec=0.0, end_sec=1.0, text="Museum aircraft.", cue_id="cue-1")]

    first = builder.build(video_id="video-1", video_path="/tmp/video-1.mp4", duration_sec=30.0, subtitle_cues=cues)
    second = builder.build(video_id="video-1", video_path="/tmp/video-1.mp4", duration_sec=30.0, subtitle_cues=cues)

    assert len(backend.requests) == 2
    assert first.to_dict() == second.to_dict()
