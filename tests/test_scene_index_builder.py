from __future__ import annotations

import json

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
