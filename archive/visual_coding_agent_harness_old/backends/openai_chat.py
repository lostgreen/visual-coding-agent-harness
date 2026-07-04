"""OpenAI-compatible text backend for persistent planner services."""

from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import mimetypes
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Mapping

from .base import BackendRequest, BackendResponse
from .qwen_text import _JSON_TEXT_TASKS, _normalized_structured_json_output, _strip_qwen_thinking

_OPENAI_COMPATIBLE_USER_AGENT = "OpenAI/Python 1.0.0"


class _OpenAIChatRequestError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.retry_attempts = 0
        self.request_attempts = 1


@dataclass
class OpenAIChatTextBackend:
    """Backend for OpenAI-compatible chat completions, with optional frame inputs."""

    api_base: str = ""
    model: str = ""
    api_key: str = "EMPTY"
    timeout: float = 180.0
    api_type: str = "openai_compatible"
    api_version: str = ""
    api_base_env: str = ""
    model_env: str = ""
    api_key_env: str = ""
    api_version_env: str = ""
    user_key_env: str = ""
    biz_scene_env: str = ""
    user_key: str = ""
    biz_scene: str = ""
    allow_media: bool = False
    thinking_token_budget: int | None = None
    enable_thinking: bool | None = False
    proxy_env: Mapping[str, str] = field(default_factory=dict)
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    max_retries: int = 5
    retry_initial_delay: float = 1.0
    retry_max_delay: float = 16.0

    def __post_init__(self) -> None:
        self.api_type = _normalized_api_type(self.api_type)
        if _is_azure_api_type(self.api_type):
            self.api_base_env = self.api_base_env or "ENDPOINT_URL"
            self.model_env = self.model_env or "DEPLOYMENT_NAME"
            self.api_key_env = self.api_key_env or "AZURE_OPENAI_API_KEY"
            self.api_version_env = self.api_version_env or "AZURE_OPENAI_API_VERSION"
            self.api_base = self.api_base or os.getenv(self.api_base_env, "")
            self.model = self.model or os.getenv(self.model_env, "")
            if not self.api_key or self.api_key == "EMPTY":
                self.api_key = os.getenv(self.api_key_env, "")
            self.api_version = self.api_version or os.getenv(self.api_version_env, "") or "2025-01-01-preview"
        elif _is_gemini_gateway_api_type(self.api_type):
            self.api_base_env = self.api_base_env or "GEMINI_API_BASE"
            self.model_env = self.model_env or "GEMINI_MODEL"
            self.api_key_env = self.api_key_env or "GEMINI_API_KEY"
            self.user_key_env = self.user_key_env or "GEMINI_USER_KEY"
            self.biz_scene_env = self.biz_scene_env or "GEMINI_BIZ_SCENE"
            self.api_base = self.api_base or os.getenv(self.api_base_env, "")
            self.model = self.model or os.getenv(self.model_env, "")
            if not self.api_key or self.api_key == "EMPTY":
                self.api_key = os.getenv(self.api_key_env, "")
            self.user_key = self.user_key or os.getenv(self.user_key_env, "")
            self.biz_scene = self.biz_scene or os.getenv(self.biz_scene_env, "")
        else:
            if self.api_base_env and not self.api_base:
                self.api_base = os.getenv(self.api_base_env, "")
            if self.model_env and not self.model:
                self.model = os.getenv(self.model_env, "")
            if self.api_key_env and (not self.api_key or self.api_key == "EMPTY"):
                self.api_key = os.getenv(self.api_key_env, self.api_key)
            if self.api_version_env and not self.api_version:
                self.api_version = os.getenv(self.api_version_env, "")

    def generate(self, request: BackendRequest) -> BackendResponse:
        if (request.media_path or request.frames) and not self.allow_media:
            raise ValueError("OpenAIChatTextBackend media support is disabled for this route")
        body = self._request_body(request)
        thinking_budget_fallback = False
        retry_attempts = 0
        request_attempts = 0
        try:
            payload, stats = self._post_json_with_retries(body)
            retry_attempts += int(stats["retry_attempts"])
            request_attempts += int(stats["request_attempts"])
        except RuntimeError as exc:
            if not _should_retry_without_thinking_budget(body, exc):
                raise
            retry_attempts += int(getattr(exc, "retry_attempts", 0))
            request_attempts += int(getattr(exc, "request_attempts", 1))
            thinking_budget_fallback = True
            payload, stats = self._post_json_with_retries(_disable_thinking_budget_body(body))
            retry_attempts += int(stats["retry_attempts"])
            request_attempts += int(stats["request_attempts"])
        text, raw_flags = _extract_message_text(payload)
        cleaned_text = _strip_qwen_thinking(text).strip()
        if request.task in _JSON_TEXT_TASKS:
            normalized = _normalized_structured_json_output(cleaned_text, task=request.task)
            if normalized is not None:
                cleaned_text = normalized
        return BackendResponse(
            text=cleaned_text,
            raw={
                "backend": "openai_chat",
                "task": request.task,
                "model": self.model,
                "api_type": self.api_type,
                "thinking_budget_fallback": thinking_budget_fallback,
                "retry_attempts": retry_attempts,
                "request_attempts": request_attempts,
                "max_retries": int(max(0, self.max_retries)),
                **raw_flags,
                **self._raw_endpoint_flags(),
            },
        )

    def _request_body(self, request: BackendRequest) -> dict[str, Any]:
        if _is_azure_api_type(self.api_type):
            body: dict[str, Any] = {
                "messages": _messages_for_request(
                    request,
                    system_role="developer",
                    typed_content=True,
                    include_media=self.allow_media,
                ),
                "max_completion_tokens": int(max(1, request.max_new_tokens)),
            }
        else:
            body = {
                "model": self.model,
                "messages": _messages_for_request(request, typed_content=self.allow_media, include_media=self.allow_media),
                "max_tokens": int(max(1, request.max_new_tokens)),
                "temperature": float(request.temperature),
            }
            if _is_gemini_gateway_api_type(self.api_type):
                body["stream"] = False
        extra_body = dict(self.extra_body)
        metadata_extra = request.metadata.get("extra_body")
        if isinstance(metadata_extra, Mapping):
            extra_body.update(dict(metadata_extra))
        body.update(extra_body)
        if _is_azure_api_type(self.api_type):
            return body

        metadata_has_thinking_budget = "thinking_token_budget" in request.metadata
        thinking_budget = request.metadata.get("thinking_token_budget", self.thinking_token_budget)
        enable_thinking = request.metadata.get("enable_thinking", self.enable_thinking)
        chat_template_kwargs = dict(body.get("chat_template_kwargs") or {})
        metadata_chat_kwargs = request.metadata.get("chat_template_kwargs")
        if isinstance(metadata_chat_kwargs, Mapping):
            chat_template_kwargs.update(dict(metadata_chat_kwargs))
        if enable_thinking is not None and "enable_thinking" not in chat_template_kwargs:
            chat_template_kwargs["enable_thinking"] = bool(enable_thinking)
        elif thinking_budget is not None and "enable_thinking" not in chat_template_kwargs:
            chat_template_kwargs["enable_thinking"] = int(thinking_budget) > 0
        effective_enable_thinking = chat_template_kwargs.get("enable_thinking", enable_thinking)
        if thinking_budget is not None and (metadata_has_thinking_budget or effective_enable_thinking is not False):
            body["thinking_token_budget"] = int(thinking_budget)
        if chat_template_kwargs:
            body["chat_template_kwargs"] = chat_template_kwargs
        return body

    def _post_json_with_retries(self, body: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
        retry_attempts = 0
        while True:
            try:
                payload = self._post_json_once(body)
                return payload, {
                    "retry_attempts": retry_attempts,
                    "request_attempts": retry_attempts + 1,
                }
            except _OpenAIChatRequestError as exc:
                if retry_attempts >= int(max(0, self.max_retries)) or not exc.retryable:
                    exc.retry_attempts = retry_attempts
                    exc.request_attempts = retry_attempts + 1
                    raise
                self._sleep(self._retry_delay(retry_attempts))
                retry_attempts += 1

    def _post_json_once(self, body: Mapping[str, Any]) -> dict[str, Any]:
        self._validate_endpoint_config()
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self._chat_completions_url(),
            data=data,
            headers=self._headers(),
            method="POST",
        )
        try:
            with _temporary_environ(self.proxy_env):
                with urllib.request.urlopen(request, timeout=float(self.timeout)) as response:
                    payload_bytes = response.read()
        except urllib.error.HTTPError as exc:
            raise _OpenAIChatRequestError(
                _format_http_error(exc),
                status_code=int(exc.code),
                retryable=_is_retryable_http_status(int(exc.code)),
            ) from exc
        except urllib.error.URLError as exc:
            raise _OpenAIChatRequestError(
                f"OpenAI-compatible planner request failed: {exc.reason}",
                retryable=True,
            ) from exc
        try:
            payload = json.loads(payload_bytes.decode("utf-8"))
        except Exception as exc:
            raise _OpenAIChatRequestError(
                "OpenAI-compatible planner returned invalid JSON",
                retryable=True,
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError("OpenAI-compatible planner returned a non-object JSON payload")
        return payload

    def _retry_delay(self, retry_attempts: int) -> float:
        initial = max(0.0, float(self.retry_initial_delay))
        delay = initial * (2 ** max(0, int(retry_attempts)))
        return min(delay, max(0.0, float(self.retry_max_delay)))

    def _sleep(self, seconds: float) -> None:
        if seconds <= 0:
            return
        time.sleep(seconds)

    def _chat_completions_url(self) -> str:
        if _is_azure_api_type(self.api_type):
            return _azure_chat_completions_url(
                endpoint=self.api_base,
                deployment=self.model,
                api_version=self.api_version,
            )
        return _chat_completions_url(self.api_base)

    def _headers(self) -> dict[str, str]:
        base_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": _OPENAI_COMPATIBLE_USER_AGENT,
        }
        if _is_azure_api_type(self.api_type):
            return {
                **base_headers,
                "api-key": self.api_key,
            }
        if _is_gemini_gateway_api_type(self.api_type):
            return {
                **base_headers,
                "x-api-key": self.api_key,
                "x-ks-user-key": self.user_key,
                "x-ks-llm-model": self.model,
                "x-ks-biz-scene": self.biz_scene,
            }
        return {
            **base_headers,
            "Authorization": f"Bearer {self.api_key}",
        }

    def _validate_endpoint_config(self) -> None:
        if _is_gemini_gateway_api_type(self.api_type):
            missing = []
            if not self.api_base:
                missing.append(self.api_base_env or "GEMINI_API_BASE")
            if not self.model:
                missing.append(self.model_env or "GEMINI_MODEL")
            if not self.api_key:
                missing.append(self.api_key_env or "GEMINI_API_KEY")
            if not self.user_key:
                missing.append(self.user_key_env or "GEMINI_USER_KEY")
            if not self.biz_scene:
                missing.append(self.biz_scene_env or "GEMINI_BIZ_SCENE")
            if missing:
                raise RuntimeError(
                    "Gemini gateway planner configuration is missing environment variable(s): "
                    + ", ".join(missing)
                )
            return
        if not _is_azure_api_type(self.api_type):
            if not self.api_base:
                raise RuntimeError("OpenAI-compatible planner api_base is required")
            if not self.model:
                raise RuntimeError("OpenAI-compatible planner model is required")
            return
        missing = []
        if not self.api_base:
            missing.append(self.api_base_env or "ENDPOINT_URL")
        if not self.model:
            missing.append(self.model_env or "DEPLOYMENT_NAME")
        if not self.api_key:
            missing.append(self.api_key_env or "AZURE_OPENAI_API_KEY")
        if missing:
            raise RuntimeError(
                "Azure OpenAI planner configuration is missing environment variable(s): "
                + ", ".join(missing)
            )

    def _raw_endpoint_flags(self) -> dict[str, Any]:
        if _is_azure_api_type(self.api_type):
            return {
                "api_base_set": bool(self.api_base),
                "api_base_env": self.api_base_env,
                "model_env": self.model_env,
                "api_key_set": bool(self.api_key),
                "api_key_env": self.api_key_env,
                "api_version": self.api_version,
                "api_version_env": self.api_version_env,
            }
        if _is_gemini_gateway_api_type(self.api_type):
            return {
                "api_base_set": bool(self.api_base),
                "api_base_env": self.api_base_env,
                "model_env": self.model_env,
                "api_key_set": bool(self.api_key),
                "api_key_env": self.api_key_env,
                "user_key_set": bool(self.user_key),
                "user_key_env": self.user_key_env,
                "biz_scene_set": bool(self.biz_scene),
                "biz_scene_env": self.biz_scene_env,
            }
        return {"api_base": _normalized_api_base(self.api_base)}


def _messages_for_request(
    request: BackendRequest,
    *,
    system_role: str = "system",
    typed_content: bool = False,
    include_media: bool = False,
) -> list[dict[str, Any]]:
    prompt = str(request.prompt)
    if request.task in _JSON_TEXT_TASKS:
        prompt = (
            f"{prompt}\n\n"
            "IMPORTANT FOR THIS RESPONSE: Return only one parseable JSON object. "
            "No prose, no bullets, no markdown, no analysis. "
            "The first non-whitespace character must be `{`."
        )
    messages = [
        {
            "role": system_role,
            "content": _message_content(
                "Follow the user's requested output format exactly. "
                "Do not explain your reasoning, restate context, or add markdown. "
                "If JSON is requested, output only parseable JSON.",
                typed=typed_content,
            ),
        },
        {"role": "user", "content": _message_content_for_request(request, prompt=prompt, typed=typed_content, include_media=include_media)},
    ]
    system_prompt = str(getattr(request, "system_prompt", "") or "").strip()
    if system_prompt:
        messages.insert(0, {"role": system_role, "content": _message_content(system_prompt, typed=typed_content)})
    return messages


def _message_content(text: str, *, typed: bool) -> str | list[dict[str, str]]:
    if not typed:
        return text
    return [{"type": "text", "text": text}]


def _message_content_for_request(
    request: BackendRequest,
    *,
    prompt: str,
    typed: bool,
    include_media: bool,
) -> str | list[dict[str, Any]]:
    if not typed:
        return prompt
    content: list[dict[str, Any]] = []
    if include_media:
        content.extend(_media_content_for_request(request))
    content.append({"type": "text", "text": prompt})
    return content


def _media_content_for_request(request: BackendRequest) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for frame_path in request.frames:
        content.append(_image_url_content(str(frame_path)))
    if request.media_path and request.media_type == "image":
        content.append(_image_url_content(str(request.media_path)))
    elif request.media_path and request.media_type == "video" and not request.frames:
        raise ValueError("OpenAIChatTextBackend media requests for video require sampled frame paths")
    return content


def _image_url_content(path: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": _data_url_for_file(path)}}


def _data_url_for_file(path: str) -> str:
    file_path = Path(path)
    mime_type = mimetypes.guess_type(str(file_path))[0] or "image/jpeg"
    encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


@contextmanager
def _temporary_environ(values: Mapping[str, str]):
    if not values:
        yield
        return
    previous: dict[str, str | None] = {}
    try:
        for key, value in values.items():
            env_key = str(key)
            previous[env_key] = os.environ.get(env_key)
            os.environ[env_key] = str(value)
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _extract_message_text(payload: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("OpenAI-compatible planner response did not include choices")
    first_choice = choices[0]
    if not isinstance(first_choice, Mapping):
        raise RuntimeError("OpenAI-compatible planner response choice is not an object")
    message = first_choice.get("message", {})
    if not isinstance(message, Mapping):
        raise RuntimeError("OpenAI-compatible planner response message is not an object")
    content = _coerce_content(message.get("content", ""))
    reasoning_content = message.get("reasoning_content")
    if content == "" and isinstance(first_choice.get("text"), str):
        content = str(first_choice["text"])
    return content, {
        "reasoning_content_present": bool(reasoning_content),
        "finish_reason": first_choice.get("finish_reason", ""),
        "usage": payload.get("usage", {}),
    }


def _coerce_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(str(item["text"]))
        return "".join(parts)
    return str(content)


def _chat_completions_url(api_base: str) -> str:
    base = str(api_base).rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _azure_chat_completions_url(*, endpoint: str, deployment: str, api_version: str) -> str:
    base = str(endpoint).rstrip("/")
    quoted_deployment = urllib.parse.quote(str(deployment), safe="")
    query = urllib.parse.urlencode({"api-version": str(api_version)})
    return f"{base}/openai/deployments/{quoted_deployment}/chat/completions?{query}"


def _normalized_api_base(api_base: str) -> str:
    base = str(api_base).rstrip("/")
    if base.endswith("/chat/completions"):
        return base[: -len("/chat/completions")]
    return base


def _normalized_api_type(api_type: str) -> str:
    normalized = str(api_type or "openai_compatible").strip().lower().replace("-", "_")
    if normalized in {"openai", "openai_compatible", "vllm", "compatible"}:
        return "openai_compatible"
    if normalized in {"azure", "azure_openai"}:
        return "azure_openai"
    if normalized in {"gemini", "gemini_gateway", "ks_gateway", "kigress_gateway"}:
        return "gemini_gateway"
    return normalized


def _is_azure_api_type(api_type: str) -> bool:
    return _normalized_api_type(api_type) == "azure_openai"


def _is_gemini_gateway_api_type(api_type: str) -> bool:
    return _normalized_api_type(api_type) == "gemini_gateway"


def _is_retryable_http_status(status_code: int) -> bool:
    return status_code in {408, 409, 425, 429, 500, 502, 503, 504} or 520 <= status_code <= 599


def _format_http_error(exc: urllib.error.HTTPError) -> str:
    try:
        payload = json.loads(exc.read().decode("utf-8"))
    except Exception:
        payload = None
    detail = ""
    if isinstance(payload, Mapping):
        error = payload.get("error")
        if isinstance(error, Mapping):
            detail = str(error.get("message") or error)
        else:
            detail = str(payload)
    if not detail:
        detail = str(exc.reason)
    return f"OpenAI-compatible planner request failed with HTTP {exc.code}: {detail}"


def _should_retry_without_thinking_budget(body: Mapping[str, Any], exc: RuntimeError) -> bool:
    if "thinking_token_budget" not in body:
        return False
    message = str(exc).lower()
    if "thinking_token_budget" not in message:
        return False
    return any(
        marker in message
        for marker in (
            "extra",
            "not permitted",
            "unknown",
            "unrecognized",
            "unexpected",
            "unsupported",
            "invalid",
        )
    )


def _disable_thinking_budget_body(body: Mapping[str, Any]) -> dict[str, Any]:
    retry_body = dict(body)
    retry_body.pop("thinking_token_budget", None)
    chat_template_kwargs = dict(retry_body.get("chat_template_kwargs") or {})
    chat_template_kwargs["enable_thinking"] = False
    retry_body["chat_template_kwargs"] = chat_template_kwargs
    return retry_body
