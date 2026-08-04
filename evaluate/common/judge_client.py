from __future__ import annotations

import os
from pathlib import Path
import random
import threading
import time
from typing import Any, Mapping
from urllib.parse import urlparse

import requests
import yaml


def _optional_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    return float(value)


class OpenAICompatibleJudgeClient:
    """Evaluation-only Chat Completions client with usage provenance."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.base = str(config["base"]).rstrip("/")
        self.model = str(config["model"])
        self.api_key = str(config["api_key"])
        self.api_type = str(
            config.get("type", "openai_compatible") or "openai_compatible"
        ).casefold()
        self.user_key = str(config.get("user_key", "") or "")
        self.biz_scene = str(config.get("biz_scene", "") or "")
        self.timeout = float(config.get("timeout", 300))
        self.max_retries = max(0, int(config.get("max_retries", 3)))
        self.retry_base_sec = max(0.0, float(config.get("retry_base_sec", 1.0)))
        self.retry_max_sec = max(
            self.retry_base_sec,
            float(config.get("retry_max_sec", 30.0)),
        )
        self.retry_jitter = max(
            0.0,
            min(1.0, float(config.get("retry_jitter", 0.2))),
        )
        self.temperature = _optional_float(config.get("temperature"), 0.0)
        self.top_p = _optional_float(config.get("top_p"))
        self._thread_state = threading.local()
        for key, value in (config.get("proxy_env") or {}).items():
            os.environ[str(key)] = str(value)

    @classmethod
    def from_yaml(
        cls,
        path: Path,
        *,
        section: str | None = None,
    ) -> "OpenAICompatibleJudgeClient":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"API config must be a mapping: {path}")
        selected = payload.get(section) if section else payload
        if section and selected is None and {"base", "model", "api_key"}.issubset(payload):
            selected = payload
        if not isinstance(selected, Mapping):
            raise ValueError(f"API config {path} has no '{section}' section")
        return cls(selected)

    @property
    def endpoint_family(self) -> str:
        if self.api_type != "openai_compatible":
            return self.api_type
        return urlparse(self.base).hostname or "openai_compatible"

    @property
    def last_response_metadata(self) -> dict[str, Any]:
        return dict(getattr(self._thread_state, "last_response_metadata", {}) or {})

    @last_response_metadata.setter
    def last_response_metadata(self, value: Mapping[str, Any]) -> None:
        self._thread_state.last_response_metadata = dict(value)

    def chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_completion_tokens: int = 4096,
        _retry_truncation: bool = True,
    ) -> str:
        token_budget = max(4096, int(max_completion_tokens))
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": str(system_prompt)},
                {"role": "user", "content": str(user_prompt)},
            ],
            "temperature": self.temperature,
        }
        if "gpt-5" in self.model.casefold():
            body["max_completion_tokens"] = token_budget
        else:
            body["max_tokens"] = token_budget
        if self.top_p is not None:
            body["top_p"] = self.top_p

        payload, retry_count, request_id = self._post(body)
        choice = payload["choices"][0]
        usage = payload.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        answer = str((choice.get("message") or {}).get("content") or "")
        metadata = {
            "finish_reason": str(choice.get("finish_reason") or ""),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": details.get("reasoning_tokens"),
            "content_chars": len(answer),
            "requested_completion_tokens": token_budget,
            "provider_request_id": request_id,
            "retry_count": retry_count,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "truncated_then_retried": False,
            "truncation_retry_count": 0,
        }
        self.last_response_metadata = metadata
        if str(choice.get("finish_reason") or "").casefold() != "length" or not _retry_truncation:
            return answer

        first_metadata = metadata
        retried = self.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_completion_tokens=max(4096, token_budget * 2),
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
                combined[key] = sum(
                    int(value)
                    for value in values
                    if isinstance(value, (int, float))
                )
        self.last_response_metadata = combined
        return retried

    def _post(self, body: Mapping[str, Any]) -> tuple[dict[str, Any], int, str]:
        retryable = {408, 409, 429, 500, 502, 503, 504}
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base}/chat/completions",
                    headers=self._headers(),
                    json=dict(body),
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"Judge API request failed after {attempt + 1} attempts: {type(exc).__name__}"
                    ) from exc
                time.sleep(self._retry_delay(attempt))
                continue
            if response.status_code >= 400:
                if response.status_code in retryable and attempt < self.max_retries:
                    time.sleep(
                        self._retry_delay(
                            attempt,
                            retry_after=response.headers.get("Retry-After"),
                        )
                    )
                    continue
                snippet = response.text[:500].replace(self.api_key, "<redacted>")
                if self.user_key:
                    snippet = snippet.replace(self.user_key, "<redacted>")
                raise RuntimeError(f"Judge HTTP {response.status_code}: {snippet}")
            payload = response.json()
            request_id = ""
            for key in ("x-request-id", "request-id", "x-amzn-requestid"):
                if response.headers.get(key):
                    request_id = str(response.headers[key])
                    break
            request_id = request_id or str(payload.get("id", "") or "")
            return dict(payload), attempt, request_id
        raise RuntimeError("Judge API retry loop exhausted")

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
        return max(0.0, delay * random.uniform(1.0 - self.retry_jitter, 1.0 + self.retry_jitter))
