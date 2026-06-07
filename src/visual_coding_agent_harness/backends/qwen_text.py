"""Text-only Qwen backend adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .base import BackendRequest, BackendResponse


@dataclass
class QwenTextBackend:
    model: Any
    tokenizer: Any
    torch_dtype: Optional[Any] = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        attn_implementation: Optional[str] = None,
    ) -> "QwenTextBackend":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs: dict[str, Any] = {"device_map": device_map}
        dtype = _resolve_dtype(torch_dtype)
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        return cls(model=model, tokenizer=tokenizer, torch_dtype=dtype)

    def generate(self, request: BackendRequest) -> BackendResponse:
        if request.media_path or request.frames:
            raise ValueError("QwenTextBackend is text-only and cannot handle media requests")

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
            text=output_text.strip(),
            raw={"backend": "qwen_text", "task": request.task},
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
