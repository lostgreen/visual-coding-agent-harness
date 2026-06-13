from __future__ import annotations

import sys
import types

from visual_coding_agent_harness.backends.base import BackendRequest
from visual_coding_agent_harness.backends.qwen_vl import _message_content, _resolve_qwen_model_class


class FakeBatch(dict):
    def __init__(self) -> None:
        super().__init__(input_ids=[[1, 2, 3]])
        self.input_ids = self["input_ids"]
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class FakeQwen35Processor:
    loaded_path = None

    def __init__(self) -> None:
        self.chat_messages = None
        self.chat_template_kwargs = None
        self.decoded_tokens = None

    @classmethod
    def from_pretrained(cls, model_path):
        cls.loaded_path = model_path
        return cls()

    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt,
        tokenize,
        return_dict,
        return_tensors,
        chat_template_kwargs=None,
    ):
        self.chat_messages = messages
        self.chat_template_kwargs = chat_template_kwargs
        assert add_generation_prompt is True
        assert tokenize is True
        assert return_dict is True
        assert return_tensors == "pt"
        return FakeBatch()

    def decode(self, token_ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        self.decoded_tokens = list(token_ids)
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return "  <think>private</think>\n\nvisible evidence  "


class FakeQwen35Model:
    loaded_path = None
    loaded_kwargs = None
    device = "fake-device"

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        cls.loaded_path = model_path
        cls.loaded_kwargs = kwargs
        return cls()

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [[1, 2, 3, 4, 5]]


def test_qwen_vl_backend_serializes_frame_requests_as_video_frame_list() -> None:
    content = _message_content(
        BackendRequest(
            task="caption_segment",
            prompt="Describe the selected frames.",
            media_type="video",
            frames=("/frames/demo/frame_000000003.jpg", "/frames/demo/frame_000000004.jpg"),
            metadata={"nframes": 64, "max_pixels": 151200},
        )
    )

    assert content == [
        {
            "type": "video",
            "video": ["/frames/demo/frame_000000003.jpg", "/frames/demo/frame_000000004.jpg"],
            "nframes": 64,
            "max_pixels": 151200,
        },
        {"type": "text", "text": "Describe the selected frames."},
    ]


def test_qwen_vl_model_class_supports_qwen35_multimodal_lm(monkeypatch) -> None:
    class FakeMultimodalLM:
        pass

    module = types.SimpleNamespace(AutoModelForMultimodalLM=FakeMultimodalLM)
    monkeypatch.setitem(sys.modules, "transformers", module)

    assert _resolve_qwen_model_class() is FakeMultimodalLM


def test_qwen35_vl_backend_uses_official_multimodal_chat_template(monkeypatch) -> None:
    module = types.SimpleNamespace(
        AutoProcessor=FakeQwen35Processor,
        AutoModelForMultimodalLM=FakeQwen35Model,
    )
    monkeypatch.setitem(sys.modules, "transformers", module)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16=object(), float16=object(), float32=object()))

    from visual_coding_agent_harness.backends.qwen_vl import QwenVLBackend

    backend = QwenVLBackend.from_pretrained(
        "/m2v_intern/xuboshen/models/Qwen3.5-9B",
        device_map="cpu",
        torch_dtype="auto",
    )
    response = backend.generate(
        BackendRequest(
            task="vision_read",
            prompt="Describe visible evidence.",
            media_path="/tmp/frame.png",
            media_type="image",
            max_new_tokens=12,
            temperature=0.0,
        )
    )

    assert FakeQwen35Processor.loaded_path == "/m2v_intern/xuboshen/models/Qwen3.5-9B"
    assert FakeQwen35Model.loaded_path == "/m2v_intern/xuboshen/models/Qwen3.5-9B"
    assert FakeQwen35Model.loaded_kwargs == {"device_map": "cpu"}
    assert backend.processor.chat_messages == [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/tmp/frame.png"},
                {"type": "text", "text": "Describe visible evidence."},
            ],
        }
    ]
    assert backend.processor.chat_template_kwargs == {"enable_thinking": False}
    assert backend.processor.decoded_tokens == [4, 5]
    assert backend.model.generate_kwargs["max_new_tokens"] == 12
    assert backend.model.generate_kwargs["do_sample"] is False
    assert response.text == "visible evidence"
    assert response.raw == {"model": "FakeQwen35Model", "task": "vision_read", "model_family": "qwen3.5"}
