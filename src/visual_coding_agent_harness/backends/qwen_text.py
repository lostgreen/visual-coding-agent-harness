"""Text-only Qwen backend adapter."""

from __future__ import annotations

from dataclasses import dataclass
import json
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
        messages = _text_messages_for_request(request)
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
        output_text = self._generate_with_processor_messages(messages, request)
        cleaned_text = _strip_qwen_thinking(output_text).strip()
        if request.task in _JSON_TEXT_TASKS:
            normalized = _normalized_structured_json_output(cleaned_text, task=request.task)
            if normalized is None:
                original_text = cleaned_text
                repair_text = self._repair_structured_json_output(request=request, draft=cleaned_text)
                repaired_text = _strip_qwen_thinking(repair_text).strip()
                normalized = _normalized_structured_json_output(repaired_text, task=request.task)
                cleaned_text = repaired_text if normalized is not None else original_text
            if normalized is not None:
                cleaned_text = normalized
        return BackendResponse(
            text=cleaned_text,
            raw={"backend": "qwen_text", "task": request.task, "model_family": self.model_family},
        )

    def _generate_with_processor_messages(self, messages: list[dict[str, Any]], request: BackendRequest) -> str:
        inputs = _apply_qwen35_chat_template(
            self.processor,
            messages,
            enable_thinking=False if request.task in _JSON_TEXT_TASKS else None,
        )
        inputs = _move_inputs_to_model(inputs, self.model)
        max_new_tokens, max_time = _qwen35_structured_generation_budget(request)
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": request.temperature > 0,
        }
        stopping_criteria = _structured_json_stopping_criteria(
            processor=self.processor,
            inputs=inputs,
            task=request.task,
        )
        if stopping_criteria is not None:
            generation_kwargs["stopping_criteria"] = stopping_criteria
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
        return str(output_text)

    def _repair_structured_json_output(self, *, request: BackendRequest, draft: str) -> str:
        repair_request = BackendRequest(
            task=request.task,
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            max_new_tokens=min(512, max(128, request.max_new_tokens)),
            temperature=0.0,
            metadata={**dict(request.metadata), "max_time": 45.0},
        )
        return self._generate_with_processor_messages(
            _qwen35_json_repair_messages(request=request, draft=draft),
            repair_request,
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
    enable_thinking: bool | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    if processor_kwargs:
        kwargs["processor_kwargs"] = processor_kwargs
    if enable_thinking is not None:
        for thinking_kwargs in (
            {"chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}},
            {"enable_thinking": bool(enable_thinking)},
        ):
            try:
                return processor.apply_chat_template(messages, **kwargs, **thinking_kwargs)
            except TypeError:
                continue
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
    system_text = (
        "Follow the user's requested output format exactly. "
        "Do not explain your reasoning, restate context, or add markdown. "
        "If JSON is requested, output only parseable JSON."
    )
    system_prompt = str(getattr(request, "system_prompt", "") or "").strip()
    if system_prompt:
        system_text = f"{system_prompt}\n\n{system_text}"
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_text,
                }
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]


def _text_messages_for_request(request: BackendRequest) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    system_prompt = str(getattr(request, "system_prompt", "") or "").strip()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": request.prompt})
    return messages


_JSON_TEXT_TASKS = {
    "replan",
    "answer_from_evidence",
    "ground_question",
    "rewrite_exploration_question",
    "verify_from_evidence",
    "asr_claim_binding",
}

_QWEN35_JSON_MAX_NEW_TOKENS = 4096
_QWEN35_JSON_MAX_TIME_SECONDS = 180.0


def _qwen35_structured_generation_budget(request: BackendRequest) -> tuple[int, float | None]:
    if request.task not in _JSON_TEXT_TASKS:
        return request.max_new_tokens, None
    max_new_tokens = min(request.max_new_tokens, _QWEN35_JSON_MAX_NEW_TOKENS)
    max_time = request.metadata.get("max_time")
    if max_time is None:
        max_time = _QWEN35_JSON_MAX_TIME_SECONDS
    return max_new_tokens, float(max_time)


def _structured_json_stopping_criteria(*, processor: Any, inputs: Any, task: str) -> Any | None:
    if task not in _JSON_TEXT_TASKS:
        return None
    try:
        import torch
        import transformers
    except Exception:
        return None
    stopping_base = getattr(transformers, "StoppingCriteria", None)
    stopping_list = getattr(transformers, "StoppingCriteriaList", None)
    if stopping_base is None or stopping_list is None:
        return None
    input_ids = inputs.input_ids if hasattr(inputs, "input_ids") else inputs.get("input_ids")
    try:
        prompt_length = int(input_ids.shape[-1])
    except Exception:
        try:
            prompt_length = len(input_ids[0])
        except Exception:
            return None

    class StopOnStructuredJson(stopping_base):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            self.tail_tokens = 4096

        def __call__(self, input_ids: Any, scores: Any = None, **kwargs: Any) -> bool:
            del scores, kwargs
            if not torch.is_tensor(input_ids) or input_ids.ndim < 2:
                return False
            generated_ids = input_ids[0, prompt_length:]
            if generated_ids.numel() <= 0:
                return False
            tail_ids = generated_ids[-self.tail_tokens :]
            try:
                tail_text = processor.decode(
                    tail_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            except Exception:
                return False
            return _normalized_structured_json_output(_strip_qwen_thinking(str(tail_text)), task=task) is not None

    return stopping_list([StopOnStructuredJson()])


def _qwen35_json_repair_messages(*, request: BackendRequest, draft: str) -> list[dict[str, Any]]:
    task_hint = _json_task_shape_hint(request.task)
    original_prompt = _compact_for_repair(request.prompt, limit=4000)
    draft_excerpt = _compact_for_repair(draft, limit=4000)
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You repair structured agent outputs. Return exactly one complete, parseable JSON object. "
                        "No prose, no markdown, no analysis, no code fences."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Task name: {request.task}\n"
                        f"{task_hint}\n\n"
                        "Original instruction:\n"
                        f"{original_prompt}\n\n"
                        "Draft response to repair:\n"
                        f"{draft_excerpt}\n\n"
                        "Return only the repaired JSON object now."
                    ),
                }
            ],
        },
    ]


def _json_task_shape_hint(task: str) -> str:
    if task == "replan":
        return (
            'For replan, use either {"status":"continue","program":[],"rationale":"..."} '
            'or {"status":"final","answer":"A","citations":[],"confidence":0.0}.'
        )
    if task == "answer_from_evidence":
        return (
            'For answer_from_evidence, use {"answer":"need_more_evidence","rationale":"...",'
            '"citations":[],"candidate_option_relations":[],"missing_evidence":["..."],"confidence":0.0} '
            'or the same schema with an option-letter answer and cited observation ids.'
        )
    return "Preserve the JSON schema requested by the original instruction."


def _compact_for_repair(text: str, *, limit: int) -> str:
    compact = str(text or "").strip()
    if len(compact) <= limit:
        return compact
    half = max(1, limit // 2)
    return f"{compact[:half]}\n...[truncated]...\n{compact[-half:]}"


def _normalized_structured_json_output(text: str, *, task: str) -> str | None:
    candidates: list[tuple[int, str]] = []
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", str(text or "")):
        start = match.start()
        try:
            payload, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        score = _structured_json_score(payload, task=task)
        if score <= 0:
            continue
        candidates.append((score, json.dumps(payload, ensure_ascii=False, separators=(",", ":"))))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _structured_json_score(payload: dict[str, Any], *, task: str) -> int:
    if task == "replan":
        if payload.get("status") in {"continue", "final"}:
            return 100
        if "program" in payload or "answer" in payload:
            return 60
        return 0
    if task == "ground_question":
        score = 0
        for key in ("route", "recommended_skill", "subjects", "targets", "relations"):
            if key in payload:
                score += 20
        return score
    if task == "answer_from_evidence":
        if "answer" in payload:
            return 100
        if "candidate_option_relations" in payload or "missing_evidence" in payload:
            return 60
        return 0
    return 50


def _strip_qwen_thinking(text: str) -> str:
    cleaned = re.sub(r"<think\b.*?</think>\s*", "", str(text), flags=re.DOTALL | re.IGNORECASE)
    closing_think = cleaned.lower().rfind("</think>")
    if closing_think >= 0:
        cleaned = cleaned[closing_think + len("</think>") :].lstrip()
    json_start = _first_json_start(cleaned)
    if json_start is not None and (
        json_start == 0
        or re.match(r"^\s*thinking process\s*:", cleaned, flags=re.IGNORECASE)
        or _looks_like_qwen_reasoning_prefix(cleaned[:json_start])
    ):
        return cleaned[json_start:]
    return cleaned


def _first_json_start(text: str) -> int | None:
    candidates = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
    return min(candidates) if candidates else None


def _looks_like_qwen_reasoning_prefix(text: str) -> bool:
    prefix = str(text or "").strip().lower()
    if not prefix:
        return False
    return any(
        marker in prefix
        for marker in (
            "the user",
            "i need",
            "i should",
            "looking at",
            "current state",
            "analyze",
            "reasoning",
        )
    )


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
