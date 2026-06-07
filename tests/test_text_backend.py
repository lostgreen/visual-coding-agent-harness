from __future__ import annotations

import sys
import types

import pytest

from visual_coding_agent_harness.backends.base import BackendRequest


class FakeBatch(dict):
    def __init__(self) -> None:
        super().__init__(input_ids=[[10, 11, 12]])
        self.input_ids = self["input_ids"]
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class FakeTokenizer:
    loaded_path = None

    def __init__(self) -> None:
        self.chat_messages = None
        self.tokenized_text = None
        self.decoded_tokens = None

    @classmethod
    def from_pretrained(cls, model_path):
        cls.loaded_path = model_path
        return cls()

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.chat_messages = messages
        assert tokenize is False
        assert add_generation_prompt is True
        return f"CHAT:{messages[0]['content']}"

    def __call__(self, text, *, return_tensors):
        self.tokenized_text = text
        assert return_tensors == "pt"
        return FakeBatch()

    def decode(self, token_ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        self.decoded_tokens = list(token_ids)
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return "  final answer  "


class FakeModel:
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
        return [[10, 11, 12, 20, 21]]


@pytest.fixture
def fake_transformers(monkeypatch):
    module = types.SimpleNamespace(AutoTokenizer=FakeTokenizer, AutoModelForCausalLM=FakeModel)
    monkeypatch.setitem(sys.modules, "transformers", module)
    return module


def test_qwen_text_from_pretrained_loads_tokenizer_model_and_generates(fake_transformers):
    from visual_coding_agent_harness.backends.qwen_text import QwenTextBackend

    backend = QwenTextBackend.from_pretrained(
        "qwen-text",
        device_map="cpu",
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
    )

    response = backend.generate(
        BackendRequest(
            task="answer_from_evidence",
            prompt="Answer using evidence only.",
            max_new_tokens=7,
            temperature=0.25,
        )
    )

    assert FakeTokenizer.loaded_path == "qwen-text"
    assert FakeModel.loaded_path == "qwen-text"
    assert FakeModel.loaded_kwargs == {
        "device_map": "cpu",
        "attn_implementation": "flash_attention_2",
    }
    assert backend.tokenizer.chat_messages == [{"role": "user", "content": "Answer using evidence only."}]
    assert backend.tokenizer.tokenized_text == "CHAT:Answer using evidence only."
    assert backend.tokenizer.decoded_tokens == [20, 21]
    assert backend.model.generate_kwargs["max_new_tokens"] == 7
    assert backend.model.generate_kwargs["do_sample"] is True
    assert backend.model.generate_kwargs["temperature"] == 0.25
    assert response.text == "final answer"
    assert response.raw == {"backend": "qwen_text", "task": "answer_from_evidence"}


def test_qwen_text_rejects_media_requests(fake_transformers):
    from visual_coding_agent_harness.backends.qwen_text import QwenTextBackend

    backend = QwenTextBackend.from_pretrained("qwen-text")

    with pytest.raises(ValueError, match="text-only"):
        backend.generate(BackendRequest(task="caption_segment", prompt="Describe.", media_path="/tmp/video.mp4"))

    with pytest.raises(ValueError, match="text-only"):
        backend.generate(BackendRequest(task="caption_frames", prompt="Describe.", frames=["frame-1.png"]))
