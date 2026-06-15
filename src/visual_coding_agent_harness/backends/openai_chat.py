"""OpenAI-compatible text backend for persistent planner services."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import urllib.error
import urllib.request
from typing import Any, Mapping

from .base import BackendRequest, BackendResponse
from .qwen_text import _JSON_TEXT_TASKS, _normalized_structured_json_output, _strip_qwen_thinking


@dataclass
class OpenAIChatTextBackend:
    """Text-only backend for vLLM/OpenAI-compatible chat completions."""

    api_base: str
    model: str
    api_key: str = "EMPTY"
    timeout: float = 180.0
    thinking_token_budget: int | None = None
    enable_thinking: bool | None = False
    extra_body: Mapping[str, Any] = field(default_factory=dict)

    def generate(self, request: BackendRequest) -> BackendResponse:
        if request.media_path or request.frames:
            raise ValueError("OpenAIChatTextBackend is text-only and cannot handle media requests")
        body = self._request_body(request)
        thinking_budget_fallback = False
        try:
            payload = self._post_json(body)
        except RuntimeError as exc:
            if not _should_retry_without_thinking_budget(body, exc):
                raise
            thinking_budget_fallback = True
            payload = self._post_json(_disable_thinking_budget_body(body))
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
                "api_base": _normalized_api_base(self.api_base),
                "thinking_budget_fallback": thinking_budget_fallback,
                **raw_flags,
            },
        )

    def _request_body(self, request: BackendRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": _messages_for_request(request),
            "max_tokens": int(max(1, request.max_new_tokens)),
            "temperature": float(request.temperature),
        }
        extra_body = dict(self.extra_body)
        metadata_extra = request.metadata.get("extra_body")
        if isinstance(metadata_extra, Mapping):
            extra_body.update(dict(metadata_extra))
        body.update(extra_body)

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

    def _post_json(self, body: Mapping[str, Any]) -> dict[str, Any]:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            _chat_completions_url(self.api_base),
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=float(self.timeout)) as response:
                payload_bytes = response.read()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(_format_http_error(exc)) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI-compatible planner request failed: {exc.reason}") from exc
        try:
            payload = json.loads(payload_bytes.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError("OpenAI-compatible planner returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("OpenAI-compatible planner returned a non-object JSON payload")
        return payload


def _messages_for_request(request: BackendRequest) -> list[dict[str, str]]:
    prompt = str(request.prompt)
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
            "content": (
                "Follow the user's requested output format exactly. "
                "Do not explain your reasoning, restate context, or add markdown. "
                "If JSON is requested, output only parseable JSON."
            ),
        },
        {"role": "user", "content": prompt},
    ]


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


def _normalized_api_base(api_base: str) -> str:
    base = str(api_base).rstrip("/")
    if base.endswith("/chat/completions"):
        return base[: -len("/chat/completions")]
    return base


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
