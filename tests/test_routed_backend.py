from __future__ import annotations

from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse
from visual_coding_agent_harness.backends.routed import RoutedBackend


class RecordingBackend:
    def __init__(self, name: str) -> None:
        self.name = name
        self.requests = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        return BackendResponse(text=f"{self.name}:{request.task}", raw={"backend": self.name})


def test_routed_backend_sends_text_tasks_to_text_backend():
    text = RecordingBackend("text")
    vl = RecordingBackend("vl")
    backend = RoutedBackend(text_backend=text, vl_backend=vl)

    for task in [
        "replan",
        "answer_from_evidence",
        "asr_claim_binding",
        "verify_from_evidence",
        "summarize_subtitle_segment",
        "summarize_scene_map_segment",
    ]:
        response = backend.generate(BackendRequest(task=task, prompt="text only"))
        assert response.text == f"text:{task}"
        assert response.raw == {"backend": "text", "route_backend": "text"}

    assert [request.task for request in text.requests] == [
        "replan",
        "answer_from_evidence",
        "asr_claim_binding",
        "verify_from_evidence",
        "summarize_subtitle_segment",
        "summarize_scene_map_segment",
    ]
    assert vl.requests == []


def test_routed_backend_sends_media_requests_to_vl_backend():
    text = RecordingBackend("text")
    vl = RecordingBackend("vl")
    backend = RoutedBackend(text_backend=text, vl_backend=vl)

    media_response = backend.generate(
        BackendRequest(task="answer_from_evidence", prompt="look", media_path="/tmp/frame.png", media_type="image")
    )
    frames_response = backend.generate(BackendRequest(task="verify_from_evidence", prompt="look", frames=["a.png"]))

    assert media_response.raw == {"backend": "vl", "route_backend": "vl"}
    assert frames_response.raw == {"backend": "vl", "route_backend": "vl"}
    assert text.requests == []
    assert [request.task for request in vl.requests] == ["answer_from_evidence", "verify_from_evidence"]


def test_routed_backend_sends_vision_tasks_to_vl_backend_without_media():
    text = RecordingBackend("text")
    vl = RecordingBackend("vl")
    backend = RoutedBackend(text_backend=text, vl_backend=vl)

    for task in ["caption_scene_segment", "vision_read", "caption_segment", "qa_video"]:
        response = backend.generate(BackendRequest(task=task, prompt="visual task"))
        assert response.text == f"vl:{task}"
        assert response.raw == {"backend": "vl", "route_backend": "vl"}

    assert text.requests == []
    assert [request.task for request in vl.requests] == [
        "caption_scene_segment",
        "vision_read",
        "caption_segment",
        "qa_video",
    ]
