"""Text-only Qwen backend adapter."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Optional

from .base import BackendRequest, BackendResponse


@dataclass
class QwenTextBackend:
    model: Any
    tokenizer: Any | None = None
    processor: Any | None = None
    torch_dtype: Optional[Any] = None
    model_family: str = "qwen"

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        attn_implementation: Optional[str] = None,
    ) -> "QwenTextBackend":
        kwargs: dict[str, Any] = {"device_map": device_map}
        dtype = _resolve_dtype(torch_dtype)
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        if _is_qwen35_model_path(model_path):
            from transformers import AutoProcessor

            model_class = _resolve_qwen35_model_class()
            processor = AutoProcessor.from_pretrained(model_path)
            model = model_class.from_pretrained(model_path, **kwargs)
            return cls(model=model, processor=processor, torch_dtype=dtype, model_family="qwen3.5")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        return cls(model=model, tokenizer=tokenizer, torch_dtype=dtype)

    def generate(self, request: BackendRequest) -> BackendResponse:
        if request.media_path or request.frames:
            raise ValueError("QwenTextBackend is text-only and cannot handle media requests")
        if self.processor is not None:
            return self._generate_with_processor(request)

        if self.tokenizer is None:
            raise RuntimeError("QwenTextBackend requires either tokenizer or processor")
        messages = [{"role": "user", "content": request.prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = _move_inputs_to_model(inputs, self.model)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            do_sample=request.temperature > 0,
            temperature=request.temperature if request.temperature > 0 else None,
        )
        input_ids = inputs.input_ids if hasattr(inputs, "input_ids") else inputs["input_ids"]
        trimmed_ids = generated_ids[0][len(input_ids[0]) :]
        output_text = self.tokenizer.decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return BackendResponse(
            text=_strip_qwen_thinking(output_text).strip(),
            raw={"backend": "qwen_text", "task": request.task},
        )

    def _generate_with_processor(self, request: BackendRequest) -> BackendResponse:
        messages = _qwen35_messages(request)
        inputs = _apply_qwen35_chat_template(self.processor, messages)
        inputs = _move_inputs_to_model(inputs, self.model)
        max_new_tokens, max_time = _qwen35_structured_generation_budget(request)
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": request.temperature > 0,
        }
        if max_time is not None:
            generation_kwargs["max_time"] = max_time
        if request.temperature > 0:
            generation_kwargs["temperature"] = request.temperature
        generated_ids = self.model.generate(**generation_kwargs)
        input_ids = inputs.input_ids if hasattr(inputs, "input_ids") else inputs["input_ids"]
        trimmed_ids = generated_ids[0][len(input_ids[0]) :]
        output_text = self.processor.decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return BackendResponse(
            text=_strip_qwen_thinking(output_text).strip(),
            raw={"backend": "qwen_text", "task": request.task, "model_family": self.model_family},
        )


def _is_qwen35_model_path(model_path: str) -> bool:
    normalized = str(model_path).lower().replace("_", ".")
    return bool(re.search(r"(?:^|[/.-])qwen3\.5(?:-|/|$)", normalized))


def _resolve_qwen35_model_class() -> Any:
    import transformers

    for name in ["AutoModelForMultimodalLM", "AutoModelForImageTextToText"]:
        model_class = getattr(transformers, name, None)
        if model_class is not None:
            return model_class
    raise ImportError("Qwen3.5 text planner requires AutoModelForMultimodalLM-compatible transformers")


def _apply_qwen35_chat_template(
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    processor_kwargs: dict[str, Any] | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    if processor_kwargs:
        kwargs["processor_kwargs"] = processor_kwargs
    return processor.apply_chat_template(messages, **kwargs)


def _qwen35_messages(request: BackendRequest) -> list[dict[str, Any]]:
    prompt = request.prompt
    if request.task in _JSON_TEXT_TASKS:
        prompt = (
            f"{prompt}\n\n"
            "IMPORTANT FOR THIS RESPONSE: Return only one parseable JSON object. "
            "No prose, no bullets, no markdown, no analysis. "
            "The first non-whitespace character must be `{`."
        )
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Follow the user's requested output format exactly. "
                        "Do not explain your reasoning, restate context, or add markdown. "
                        "If JSON is requested, output only parseable JSON."
                    ),
                }
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]


_JSON_TEXT_TASKS = {
    "replan",
    "answer_from_evidence",
    "ground_question",
    "rewrite_exploration_question",
    "verify_from_evidence",
    "asr_claim_binding",
}

_QWEN35_JSON_MAX_NEW_TOKENS = 768
_QWEN35_JSON_MAX_TIME_SECONDS = 90.0


def _qwen35_structured_generation_budget(request: BackendRequest) -> tuple[int, float | None]:
    if request.task not in _JSON_TEXT_TASKS:
        return request.max_new_tokens, None
    max_new_tokens = min(request.max_new_tokens, _QWEN35_JSON_MAX_NEW_TOKENS)
    max_time = request.metadata.get("max_time")
    if max_time is None:
        max_time = _QWEN35_JSON_MAX_TIME_SECONDS
    return max_new_tokens, float(max_time)


def _strip_qwen_thinking(text: str) -> str:
    cleaned = re.sub(r"^\s*<think>.*?</think>\s*", "", str(text), flags=re.DOTALL | re.IGNORECASE)
    if re.match(r"^\s*thinking process\s*:", cleaned, flags=re.IGNORECASE):
        json_start = _first_json_start(cleaned)
        if json_start is not None:
            return cleaned[json_start:]
    return cleaned


def _first_json_start(text: str) -> int | None:
    candidates = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
    return min(candidates) if candidates else None


def _resolve_dtype(torch_dtype: str) -> Any:
    if torch_dtype == "auto":
        return None
    import torch

    if torch_dtype == "bfloat16":
        return torch.bfloat16
    if torch_dtype == "float16":
        return torch.float16
    if torch_dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {torch_dtype}")


def _move_inputs_to_model(inputs: Any, model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None and hasattr(inputs, "to"):
        return inputs.to(device)
    return inputs
