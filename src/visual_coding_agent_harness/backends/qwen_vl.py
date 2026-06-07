"""Qwen-VL backend adapter.

The adapter is intentionally small: it owns one loaded model/processor pair and
serves both the main agent planner call and VLM-backed tools.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Optional

from .base import BackendRequest, BackendResponse


@dataclass
class QwenVLBackend:
    model: Any
    processor: Any
    device: Optional[str] = None
    torch_dtype: Optional[Any] = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        attn_implementation: Optional[str] = None,
    ) -> "QwenVLBackend":
        import torch
        from transformers import AutoProcessor

        model_class = _resolve_qwen_model_class()
        dtype = _resolve_dtype(torch_dtype=torch_dtype, torch_module=torch)
        kwargs: dict[str, Any] = {"device_map": device_map}
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        model = model_class.from_pretrained(model_path, **kwargs)
        processor = AutoProcessor.from_pretrained(model_path)
        return cls(model=model, processor=processor, torch_dtype=dtype)

    def generate(self, request: BackendRequest) -> BackendResponse:
        messages = [
            {
                "role": "user",
                "content": _message_content(request),
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = _process_vision_info_with_nframes_clamp(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = _move_inputs_to_model(inputs, self.model)
        generation_kwargs = {
            **inputs,
            "max_new_tokens": request.max_new_tokens,
            "do_sample": request.temperature > 0,
        }
        if request.temperature > 0:
            generation_kwargs["temperature"] = request.temperature
        for key in ("repetition_penalty", "no_repeat_ngram_size", "top_p", "top_k"):
            if key in request.metadata:
                generation_kwargs[key] = request.metadata[key]
        generated_ids = self.model.generate(**generation_kwargs)
        generated_trimmed = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return BackendResponse(
            text=output_text.strip(),
            raw={"model": self.model.__class__.__name__, "task": request.task},
        )


def _resolve_qwen_model_class() -> Any:
    import transformers

    for name in [
        "Qwen3VLForConditionalGeneration",
        "AutoModelForImageTextToText",
        "Qwen2_5_VLForConditionalGeneration",
    ]:
        model_class = getattr(transformers, name, None)
        if model_class is not None:
            return model_class
    raise ImportError("No Qwen-VL compatible model class found in transformers")


def _resolve_dtype(*, torch_dtype: str, torch_module: Any) -> Any:
    if torch_dtype == "auto":
        return None
    if torch_dtype == "bfloat16":
        return torch_module.bfloat16
    if torch_dtype == "float16":
        return torch_module.float16
    if torch_dtype == "float32":
        return torch_module.float32
    raise ValueError(f"Unsupported torch dtype: {torch_dtype}")


def _message_content(request: BackendRequest) -> list[Mapping[str, Any]]:
    content: list[Mapping[str, Any]] = []
    if request.media_path:
        if request.media_type == "video":
            video_item: dict[str, Any] = {"type": "video", "video": request.media_path}
            for key in ["nframes", "fps", "max_pixels", "min_pixels"]:
                if key in request.metadata:
                    video_item[key] = request.metadata[key]
            content.append(video_item)
        elif request.media_type == "image":
            content.append({"type": "image", "image": request.media_path})
    content.append({"type": "text", "text": request.prompt})
    return content


def _process_vision_info_with_nframes_clamp(messages: list[Mapping[str, Any]]) -> tuple[Any, Any]:
    try:
        return _process_vision_info(messages)
    except ValueError as exc:
        interval = _nframes_interval_from_error(str(exc))
        if interval is None or not _clamp_video_nframes(messages, interval=interval):
            raise
        return _process_vision_info(messages)


def _nframes_interval_from_error(message: str) -> tuple[int, int] | None:
    match = re.search(r"nframes should in interval \[(\d+),\s*(\d+)\]", str(message))
    if not match:
        return None
    low = int(match.group(1))
    high = int(match.group(2))
    if low > high:
        return None
    return low, high


def _clamp_video_nframes(messages: list[Mapping[str, Any]], *, interval: tuple[int, int]) -> bool:
    low, high = interval
    changed = False
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "video" or "nframes" not in item:
                continue
            original = int(item["nframes"])
            clamped = max(low, min(high, original))
            if clamped != original:
                item["nframes"] = clamped
                changed = True
    return changed


def _process_vision_info(messages: list[Mapping[str, Any]]) -> tuple[Any, Any]:
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise ImportError("qwen_vl_utils is required for Qwen-VL media preprocessing") from exc
    return process_vision_info(messages)


def _move_inputs_to_model(inputs: Any, model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return inputs.to(device)
    return inputs
