from __future__ import annotations

from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse, VisionLanguageBackend
from visual_coding_agent_harness.backends.qwen_vl import QwenVLBackend
from visual_coding_agent_harness.tools.inspector import build_segment_inspector_registry


class RecordingBackend(VisionLanguageBackend):
    def __init__(self) -> None:
        self.requests: list[BackendRequest] = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        return BackendResponse(text="A short visual fact.", raw={"task": request.task})


class FakeBatch(dict):
    def __init__(self) -> None:
        super().__init__(input_ids=[[1, 2, 3]])
        self.input_ids = self["input_ids"]
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class FakeProcessor:
    def __init__(self) -> None:
        self.messages = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.messages = messages
        assert tokenize is False
        assert add_generation_prompt is True
        return "chat text"

    def __call__(self, *, text, images, videos, padding, return_tensors):
        assert text == ["chat text"]
        assert images == []
        assert videos == ["video-input"]
        assert padding is True
        assert return_tensors == "pt"
        return FakeBatch()

    def batch_decode(self, token_ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert token_ids == [[4, 5]]
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return [" decoded answer "]


class FakeModel:
    device = "fake-device"

    def __init__(self) -> None:
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [[1, 2, 3, 4, 5]]


def test_vision_read_requests_repetition_control_and_larger_token_budget():
    backend = RecordingBackend()
    registry = build_segment_inspector_registry(backend)

    registry.execute(
        "vision_read",
        {
            "video_path": "/videos/demo.mp4",
            "segment_id": "seg_0001",
            "start_sec": 1.0,
            "end_sec": 8.0,
            "ask_for": "Read the on-screen title.",
        },
    )

    request = backend.requests[0]
    assert request.task == "vision_read"
    assert request.max_new_tokens >= 384
    assert request.metadata["repetition_penalty"] >= 1.1
    assert request.metadata["no_repeat_ngram_size"] >= 6


def test_qwen_vl_generate_forwards_decoding_metadata_without_temperature_none(monkeypatch):
    model = FakeModel()
    processor = FakeProcessor()
    backend = QwenVLBackend(model=model, processor=processor)
    monkeypatch.setattr(
        "visual_coding_agent_harness.backends.qwen_vl._process_vision_info",
        lambda messages: ([], ["video-input"]),
    )

    response = backend.generate(
        BackendRequest(
            task="vision_read",
            prompt="Read visible text.",
            media_path="/videos/demo.mp4",
            media_type="video",
            max_new_tokens=384,
            metadata={
                "nframes": 16,
                "max_pixels": 151200,
                "repetition_penalty": 1.15,
                "no_repeat_ngram_size": 6,
                "top_p": 0.9,
                "top_k": 50,
            },
        )
    )

    assert response.text == "decoded answer"
    assert model.generate_kwargs["max_new_tokens"] == 384
    assert model.generate_kwargs["do_sample"] is False
    assert "temperature" not in model.generate_kwargs
    assert model.generate_kwargs["repetition_penalty"] == 1.15
    assert model.generate_kwargs["no_repeat_ngram_size"] == 6
    assert model.generate_kwargs["top_p"] == 0.9
    assert model.generate_kwargs["top_k"] == 50


def test_qwen_vl_generate_clamps_nframes_to_backend_interval(monkeypatch):
    model = FakeModel()
    processor = FakeProcessor()
    backend = QwenVLBackend(model=model, processor=processor)
    seen_nframes = []

    def process_with_dynamic_cap(messages):
        video_item = messages[0]["content"][0]
        seen_nframes.append(video_item.get("nframes"))
        if video_item.get("nframes") == 64:
            raise ValueError("nframes should in interval [2, 7], but got 64.")
        return [], ["video-input"]

    monkeypatch.setattr(
        "visual_coding_agent_harness.backends.qwen_vl._process_vision_info",
        process_with_dynamic_cap,
    )

    response = backend.generate(
        BackendRequest(
            task="vision_read",
            prompt="Read visible text.",
            media_path="/videos/demo.mp4",
            media_type="video",
            max_new_tokens=384,
            metadata={"nframes": 64, "max_pixels": 151200},
        )
    )

    assert response.text == "decoded answer"
    assert seen_nframes == [64, 7]
    assert processor.messages[0]["content"][0]["nframes"] == 7
