from __future__ import annotations

import base64
import os
from pathlib import Path
import random
import threading
import time
from typing import Any, Mapping, Sequence

import requests
import yaml


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "on", "supported"}
    return bool(value)


def _seed_support_status(value: Any) -> str:
    if value is None or value == "":
        return "unknown"
    if _as_bool(value):
        return "supported"
    if isinstance(value, str) and value.strip().casefold() in {"unknown", "unreported", "not_reported"}:
        return "unknown"
    return "unsupported"


def _request_id(response: Any, payload: Mapping[str, Any]) -> str:
    headers = getattr(response, "headers", {}) or {}
    for key in ("x-request-id", "request-id", "x-amzn-requestid"):
        value = headers.get(key) or headers.get(key.title())
        if value:
            return str(value)
    return str(payload.get("id", "") or "")


def _image_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


class ImageAttachmentError(RuntimeError):
    def __init__(self, message: str, metadata: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.metadata = dict(metadata)


class OpenAICompatibleClient:
    """Small OpenAI-compatible client with reproducibility metadata."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.base = str(config["base"]).rstrip("/")
        self.model = str(config["model"])
        self.api_key = str(config["api_key"])
        self.api_type = str(config.get("type", "openai_compatible") or "openai_compatible").casefold()
        self.user_key = str(config.get("user_key", "") or "")
        self.biz_scene = str(config.get("biz_scene", "") or "")
        self.timeout = float(config.get("timeout", 300))
        self.max_retries = max(0, int(config.get("max_retries", 5)))
        self.retry_base_sec = max(0.0, float(config.get("retry_base_sec", 1.0)))
        self.retry_max_sec = max(self.retry_base_sec, float(config.get("retry_max_sec", 30.0)))
        self.retry_jitter = max(0.0, min(1.0, float(config.get("retry_jitter", 0.2))))
        self.max_dropped_images = max(0, int(config.get("max_dropped_images", 0) or 0))
        self.temperature = _optional_float(config.get("temperature"))
        self.top_p = _optional_float(config.get("top_p"))
        self.provider_reported_seed_support = _seed_support_status(
            config.get(
                "provider_reported_seed_support",
                config.get("provider_seed_supported", config.get("supports_seed")),
            )
        )
        self.provider_seed_supported = self.provider_reported_seed_support == "supported"
        self._configured_seed = _optional_int(config.get("seed"))
        self._thread_state = threading.local()
        for key, value in (config.get("proxy_env") or {}).items():
            os.environ[str(key)] = str(value)

    @classmethod
    def from_yaml(cls, path: Path, *, section: str | None = None) -> "OpenAICompatibleClient":
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"API config must be a mapping: {path}")
        selected = payload.get(section) if section else payload
        if section and selected is None and {"base", "model", "api_key"}.issubset(payload):
            selected = payload
        if not isinstance(selected, Mapping):
            raise ValueError(f"API config {path} has no '{section}' section")
        return cls(selected)

    def chat(
        self,
        prompt: str,
        *,
        image_paths: Sequence[str] = (),
        image_labels: Sequence[str] = (),
        prompt_position: str = "first",
        max_tokens: int = 900,
        _retry_truncation: bool = True,
    ) -> str:
        requested_paths = tuple(str(path) for path in image_paths)
        requested_labels = tuple(str(label) for label in image_labels)
        if requested_labels and len(requested_labels) != len(requested_paths):
            raise ValueError("image_labels must match image_paths length")
        position = str(prompt_position or "first").strip().casefold()
        if position not in {"first", "last"}:
            raise ValueError("prompt_position must be 'first' or 'last'")
        attached_paths = tuple(path for path in requested_paths if Path(path).is_file())
        dropped_paths = tuple(path for path in requested_paths if not Path(path).is_file())
        attachment_metadata = {
            "images_requested": len(requested_paths),
            "images_attached": len(attached_paths),
            "images_dropped": len(dropped_paths),
            "image_attachment_warning": bool(dropped_paths),
            "image_label_count": len(requested_labels),
            "prompt_position": position,
        }
        if len(dropped_paths) > self.max_dropped_images:
            self.last_response_metadata = {
                **attachment_metadata,
                "finish_reason": "image_attachment_failed",
                "requested_completion_tokens": int(max_tokens),
                "dropped_image_paths": list(dropped_paths),
            }
            raise ImageAttachmentError(
                f"Image attachment failed: {len(dropped_paths)} of {len(requested_paths)} requested images are missing",
                self.last_response_metadata,
            )

        label_by_path = dict(zip(requested_paths, requested_labels)) if requested_labels else {}
        content: list[dict[str, Any]] = []
        if position == "first":
            content.append({"type": "text", "text": prompt})
        for path in attached_paths:
            if requested_labels:
                content.append({"type": "text", "text": label_by_path[path]})
            content.append({"type": "image_url", "image_url": {"url": _image_data_url(Path(path))}})
        if position == "last":
            content.append({"type": "text", "text": prompt})
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
        }
        if "gpt-5" in self.model.casefold():
            body["max_completion_tokens"] = int(max_tokens)
        else:
            body["max_tokens"] = int(max_tokens)
            body["temperature"] = 0 if self.temperature is None else self.temperature
        if self.temperature is not None and "gpt-5" in self.model.casefold():
            body["temperature"] = self.temperature
        if self.top_p is not None:
            body["top_p"] = self.top_p
        requested_seed = self.requested_seed
        if self.provider_seed_supported and requested_seed is not None:
            body["seed"] = requested_seed
        if self.api_type in {"gemini_gateway", "kuaishou_gateway"}:
            body["stream"] = False

        retryable = {408, 409, 429, 500, 502, 503, 504}
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base}/chat/completions",
                    headers=self._headers(),
                    json=body,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"API request failed after {attempt + 1} attempts: {type(exc).__name__}"
                    ) from exc
                time.sleep(self._retry_delay(attempt))
                continue
            if response.status_code >= 400:
                if response.status_code in retryable and attempt < self.max_retries:
                    time.sleep(self._retry_delay(attempt, retry_after=response.headers.get("Retry-After")))
                    continue
                snippet = response.text[:500].replace(self.api_key, "<redacted>")
                if self.user_key:
                    snippet = snippet.replace(self.user_key, "<redacted>")
                raise RuntimeError(f"HTTP {response.status_code}: {snippet}")

            payload = response.json()
            choice = payload["choices"][0]
            usage = payload.get("usage") or {}
            details = usage.get("completion_tokens_details") or {}
            answer = str((choice.get("message") or {}).get("content") or "")
            self.last_response_metadata = {
                "finish_reason": str(choice.get("finish_reason") or ""),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": details.get("reasoning_tokens"),
                "content_chars": len(answer),
                "requested_completion_tokens": int(max_tokens),
                "provider_request_id": _request_id(response, payload),
                "retry_count": attempt,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "requested_seed": requested_seed,
                "provider_seed_supported": self.provider_seed_supported,
                "provider_reported_seed_support": self.provider_reported_seed_support,
                **attachment_metadata,
            }
            if str(choice.get("finish_reason") or "").casefold() != "length" or not _retry_truncation:
                self.last_response_metadata["truncated_then_retried"] = False
                self.last_response_metadata["truncation_retry_count"] = 0
                return answer

            first_metadata = self.last_response_metadata
            retried = self.chat(
                prompt,
                image_paths=requested_paths,
                image_labels=requested_labels,
                prompt_position=position,
                max_tokens=max(4096, int(max_tokens) * 2),
                _retry_truncation=False,
            )
            retry_metadata = self.last_response_metadata
            combined = {
                **retry_metadata,
                "truncated_then_retried": True,
                "truncation_retry_count": 1,
                "initial_truncated_response": first_metadata,
                "retry_count": int(first_metadata.get("retry_count", 0) or 0)
                + int(retry_metadata.get("retry_count", 0) or 0),
            }
            for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
                values = (first_metadata.get(key), retry_metadata.get(key))
                if any(isinstance(value, (int, float)) for value in values):
                    combined[key] = sum(int(value) for value in values if isinstance(value, (int, float)))
            self.last_response_metadata = combined
            return retried
        raise RuntimeError("API retry loop exhausted")

    @property
    def last_response_metadata(self) -> dict[str, Any]:
        return dict(getattr(self._thread_state, "last_response_metadata", {}) or {})

    @last_response_metadata.setter
    def last_response_metadata(self, value: Mapping[str, Any]) -> None:
        self._thread_state.last_response_metadata = dict(value)

    @property
    def requested_seed(self) -> int | None:
        return getattr(self._thread_state, "requested_seed", self._configured_seed)

    def set_requested_seed(self, seed: int | None) -> None:
        self._thread_state.requested_seed = None if seed is None else int(seed)

    @property
    def replay_settings(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "requested_seed": self.requested_seed,
            "provider_seed_supported": self.provider_seed_supported,
            "provider_reported_seed_support": self.provider_reported_seed_support,
        }

    def _headers(self) -> dict[str, str]:
        base = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_type in {"gemini_gateway", "kuaishou_gateway"}:
            return {
                **base,
                "x-api-key": self.api_key,
                "x-ks-user-key": self.user_key,
                "x-ks-llm-model": self.model,
                "x-ks-biz-scene": self.biz_scene,
            }
        return {**base, "Authorization": f"Bearer {self.api_key}"}

    def _retry_delay(self, attempt: int, *, retry_after: str | None = None) -> float:
        delay = self.retry_base_sec * (2 ** max(0, int(attempt)))
        try:
            delay = max(delay, float(retry_after)) if retry_after is not None else delay
        except (TypeError, ValueError):
            pass
        delay = min(self.retry_max_sec, delay)
        if self.retry_jitter:
            delay *= random.uniform(1.0 - self.retry_jitter, 1.0 + self.retry_jitter)
        return max(0.0, delay)
