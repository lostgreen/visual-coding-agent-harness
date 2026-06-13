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


class FakeQwen35Processor:
    loaded_path = None

    def __init__(self) -> None:
        self.chat_messages = None
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
    ):
        self.chat_messages = messages
        assert add_generation_prompt is True
        assert tokenize is True
        assert return_dict is True
        assert return_tensors == "pt"
        return FakeBatch()

    def decode(self, token_ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        self.decoded_tokens = list(token_ids)
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return (
            "  Thinking Process:\n"
            "1. Analyze the planner request.\n\n"
            "{\"status\":\"final\",\"answer\":\"A\"}  "
        )


class FakeQwen35Model(FakeModel):
    loaded_path = None
    loaded_kwargs = None


class ForbiddenCausalLM:
    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        raise AssertionError("Qwen3.5 planner should not load AutoModelForCausalLM")


@pytest.fixture
def fake_transformers(monkeypatch):
    module = types.SimpleNamespace(AutoTokenizer=FakeTokenizer, AutoModelForCausalLM=FakeModel)
    monkeypatch.setitem(sys.modules, "transformers", module)
    return module


@pytest.fixture
def fake_qwen35_transformers(monkeypatch):
    module = types.SimpleNamespace(
        AutoTokenizer=FakeTokenizer,
        AutoModelForCausalLM=ForbiddenCausalLM,
        AutoProcessor=FakeQwen35Processor,
        AutoModelForMultimodalLM=FakeQwen35Model,
    )
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


def test_qwen35_text_planner_uses_multimodal_processor_and_disables_thinking(fake_qwen35_transformers):
    from visual_coding_agent_harness.backends.qwen_text import QwenTextBackend

    backend = QwenTextBackend.from_pretrained(
        "/m2v_intern/xuboshen/models/Qwen3.5-9B",
        device_map="cpu",
        torch_dtype="auto",
    )

    response = backend.generate(
        BackendRequest(
            task="replan",
            prompt="Return planner JSON only.",
            max_new_tokens=17,
            temperature=0.0,
        )
    )

    assert FakeQwen35Processor.loaded_path == "/m2v_intern/xuboshen/models/Qwen3.5-9B"
    assert FakeQwen35Model.loaded_path == "/m2v_intern/xuboshen/models/Qwen3.5-9B"
    assert FakeQwen35Model.loaded_kwargs == {"device_map": "cpu"}
    assert backend.processor.chat_messages[0]["role"] == "system"
    assert "Do not explain your reasoning" in backend.processor.chat_messages[0]["content"][0]["text"]
    assert backend.processor.chat_messages[1]["role"] == "user"
    user_text = backend.processor.chat_messages[1]["content"][0]["text"]
    assert user_text.startswith("Return planner JSON only.")
    assert "Return only one parseable JSON object" in user_text
    assert backend.processor.decoded_tokens == [20, 21]
    assert backend.model.generate_kwargs["max_new_tokens"] == 17
    assert backend.model.generate_kwargs["do_sample"] is False
    assert "temperature" not in backend.model.generate_kwargs
    assert response.text == '{"status":"final","answer":"A"}'
    assert response.raw == {"backend": "qwen_text", "task": "replan", "model_family": "qwen3.5"}


def test_qwen35_text_planner_caps_structured_json_generation_budget(fake_qwen35_transformers):
    from visual_coding_agent_harness.backends.qwen_text import QwenTextBackend

    backend = QwenTextBackend.from_pretrained(
        "/m2v_intern/xuboshen/models/Qwen3.5-9B",
        device_map="cpu",
        torch_dtype="auto",
    )

    backend.generate(
        BackendRequest(
            task="ground_question",
            prompt="Return a grounding plan JSON only.",
            max_new_tokens=4096,
            temperature=0.0,
        )
    )

    assert backend.model.generate_kwargs["max_new_tokens"] == 768
    assert backend.model.generate_kwargs["max_time"] == 90.0
    assert backend.model.generate_kwargs["do_sample"] is False
