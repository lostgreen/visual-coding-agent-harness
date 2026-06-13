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


class CountingQwen35Model(FakeQwen35Model):
    def __init__(self) -> None:
        self.generate_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        self.generate_kwargs = kwargs
        return [[10, 11, 12, 20, 21]]


class RepairingQwen35Processor(FakeQwen35Processor):
    def __init__(self) -> None:
        super().__init__()
        self.messages_by_call = []
        self.responses = [
            'analysis first {"status":"continue","program":[{"tool":"vision_read","args":{"segment_id":"seg_0001"}}',
            '{"status":"continue","program":[],"rationale":"repair"}',
        ]

    def apply_chat_template(self, messages, **kwargs):
        self.messages_by_call.append(messages)
        return super().apply_chat_template(messages, **kwargs)

    def decode(self, token_ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        self.decoded_tokens = list(token_ids)
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return self.responses.pop(0)


class FailingRepairQwen35Processor(RepairingQwen35Processor):
    def __init__(self) -> None:
        super().__init__()
        self.responses = [
            "original analysis without json",
            "repair analysis still without json",
        ]


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
            task="replan",
            prompt="Return planner JSON only.",
            max_new_tokens=4096,
            temperature=0.0,
        )
    )

    assert backend.model.generate_kwargs["max_new_tokens"] == 4096
    assert backend.model.generate_kwargs["max_time"] == 180.0
    assert backend.model.generate_kwargs["do_sample"] is False


def test_qwen35_text_planner_repairs_unparseable_structured_json(monkeypatch):
    module = types.SimpleNamespace(
        AutoTokenizer=FakeTokenizer,
        AutoModelForCausalLM=ForbiddenCausalLM,
        AutoProcessor=RepairingQwen35Processor,
        AutoModelForMultimodalLM=CountingQwen35Model,
    )
    monkeypatch.setitem(sys.modules, "transformers", module)

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
            max_new_tokens=4096,
            temperature=0.0,
        )
    )

    assert response.text == '{"status":"continue","program":[],"rationale":"repair"}'
    assert len(backend.model.generate_calls) == 2
    repair_user_text = backend.processor.messages_by_call[-1][1]["content"][0]["text"]
    assert "Draft response to repair" in repair_user_text
    assert backend.model.generate_calls[-1]["max_new_tokens"] == 512
    assert backend.model.generate_calls[-1]["max_time"] == 45.0


def test_qwen35_text_planner_keeps_original_when_json_repair_fails(monkeypatch):
    module = types.SimpleNamespace(
        AutoTokenizer=FakeTokenizer,
        AutoModelForCausalLM=ForbiddenCausalLM,
        AutoProcessor=FailingRepairQwen35Processor,
        AutoModelForMultimodalLM=CountingQwen35Model,
    )
    monkeypatch.setitem(sys.modules, "transformers", module)

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
            max_new_tokens=4096,
            temperature=0.0,
        )
    )

    assert response.text == "original analysis without json"
    assert len(backend.model.generate_calls) == 2


def test_qwen35_structured_json_normalizer_requires_answer_shape():
    from visual_coding_agent_harness.backends.qwen_text import _normalized_structured_json_output

    repaired = _normalized_structured_json_output(
        'analysis {"answer":"need_more_evidence","citations":[],"missing_evidence":["count missing"],"confidence":0}',
        task="answer_from_evidence",
    )

    assert repaired == '{"answer":"need_more_evidence","citations":[],"missing_evidence":["count missing"],"confidence":0}'
    assert _normalized_structured_json_output('{"args":{"query":"not an answer"}}', task="answer_from_evidence") is None


def test_qwen35_strip_thinking_removes_reasoning_prefix_before_json():
    from visual_coding_agent_harness.backends.qwen_text import _strip_qwen_thinking

    assert _strip_qwen_thinking(
        "The user wants me to inspect evidence first.\n"
        "{\"status\":\"continue\",\"program\":[]}"
    ) == '{"status":"continue","program":[]}'
    assert _strip_qwen_thinking(
        "draft </think>\n{\"status\":\"final\",\"answer\":\"C\"}"
    ) == '{"status":"final","answer":"C"}'
