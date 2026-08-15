from __future__ import annotations

import base64
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import random
import threading
import time
from typing import Any, Mapping, Sequence

from PIL import Image, ImageOps
import requests
import yaml


_DEFAULT_IMAGE_PAYLOAD_LIMIT_BYTES = 18_000_000
_IMAGE_REENCODE_STEPS = (
    (1536, 85),
    (1280, 85),
    (1024, 82),
    (896, 80),
    (768, 78),
    (640, 75),
    (512, 72),
    (384, 68),
    (256, 65),
    (192, 62),
    (128, 60),
)


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


def _prepared_image_data_urls(
    paths: Sequence[Path],
    *,
    payload_limit_bytes: int,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    original_urls = tuple(_image_data_url(path) for path in paths)
    original_bytes = sum(len(url.encode("ascii")) for url in original_urls)
    metadata: dict[str, Any] = {
        "image_payload_bytes_original": original_bytes,
        "image_payload_bytes_attached": original_bytes,
        "image_payload_limit_bytes": payload_limit_bytes or None,
        "images_reencoded": 0,
        "image_reencode_max_edge": None,
        "image_reencode_quality": None,
    }
    if not payload_limit_bytes or original_bytes <= payload_limit_bytes:
        return original_urls, metadata

    last_bytes = original_bytes
    for max_edge, quality in _IMAGE_REENCODE_STEPS:
        urls = tuple(
            _reencoded_image_data_url(path, max_edge=max_edge, quality=quality)
            for path in paths
        )
        attached_bytes = sum(len(url.encode("ascii")) for url in urls)
        last_bytes = attached_bytes
        if attached_bytes <= payload_limit_bytes:
            return urls, {
                **metadata,
                "image_payload_bytes_attached": attached_bytes,
                "images_reencoded": len(paths),
                "image_reencode_max_edge": max_edge,
                "image_reencode_quality": quality,
            }

    raise ImageAttachmentError(
        "Image attachments exceed the configured payload budget after re-encoding",
        {
            **metadata,
            "image_payload_bytes_attached": last_bytes,
            "images_reencoded": len(paths),
            "finish_reason": "image_payload_budget_exceeded",
        },
    )


def _reencoded_image_data_url(path: Path, *, max_edge: int, quality: int) -> str:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


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
        self.max_image_payload_bytes = max(
            0,
            int(
                config.get(
                    "max_image_payload_bytes",
                    _DEFAULT_IMAGE_PAYLOAD_LIMIT_BYTES,
                )
                or 0
            ),
        )
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
        attachment_metadata: dict[str, Any] = {
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

        try:
            image_urls, payload_metadata = _prepared_image_data_urls(
                tuple(Path(path) for path in attached_paths),
                payload_limit_bytes=self.max_image_payload_bytes,
            )
        except ImageAttachmentError as exc:
            self.last_response_metadata = {
                **attachment_metadata,
                **exc.metadata,
                "requested_completion_tokens": int(max_tokens),
            }
            raise
        attachment_metadata.update(payload_metadata)

        label_by_path = dict(zip(requested_paths, requested_labels)) if requested_labels else {}
        content: list[dict[str, Any]] = []
        if position == "first":
            content.append({"type": "text", "text": prompt})
        for path, image_url in zip(attached_paths, image_urls):
            if requested_labels:
                content.append({"type": "text", "text": label_by_path[path]})
            content.append({"type": "image_url", "image_url": {"url": image_url}})
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


class MatchedResponseReplayError(RuntimeError):
    """Raised when a pre-treatment matched response cannot be replayed exactly."""


class MatchedResponseSession:
    """Shared lifecycle and accounting for per-role matched response clients."""

    def __init__(self, *, mode: str) -> None:
        normalized = str(mode or "").strip().casefold()
        if normalized not in {"record", "replay"}:
            raise ValueError("matched response mode must be record or replay")
        self.mode = normalized
        self.active = True
        self.deactivation_reason = ""
        self._counts = {
            "recorded": {},
            "replayed": {},
            "live_after_treatment": {},
        }
        self._mismatch_count = 0
        self._lock = threading.Lock()

    def deactivate(self, reason: str) -> None:
        with self._lock:
            if not self.active:
                return
            self.active = False
            self.deactivation_reason = str(reason or "").strip()

    def note(self, kind: str, namespace: str) -> None:
        with self._lock:
            counts = self._counts[str(kind)]
            role = str(namespace or "").strip()
            counts[role] = int(counts.get(role, 0)) + 1

    def note_mismatch(self) -> None:
        with self._lock:
            self._mismatch_count += 1

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            counts = {
                kind: dict(sorted(values.items()))
                for kind, values in self._counts.items()
            }
            return {
                "schema_version": "MatchedPreTreatmentResponseV1",
                "mode": self.mode,
                "active": bool(self.active),
                "deactivation_reason": self.deactivation_reason,
                **counts,
                "recorded_count": sum(counts["recorded"].values()),
                "replayed_count": sum(counts["replayed"].values()),
                "live_after_treatment_count": sum(
                    counts["live_after_treatment"].values()
                ),
                "mismatch_count": int(self._mismatch_count),
            }


class MatchedResponseCacheClient:
    """Record or replay exact API responses until the shared session deactivates."""

    def __init__(
        self,
        delegate: OpenAICompatibleClient,
        *,
        root: Path,
        mode: str,
        namespace: str,
        session: MatchedResponseSession,
    ) -> None:
        normalized_mode = str(mode or "").strip().casefold()
        if normalized_mode not in {"record", "replay"}:
            raise ValueError("matched response mode must be record or replay")
        if session.mode != normalized_mode:
            raise ValueError("matched response client/session mode mismatch")
        role = str(namespace or "").strip().casefold()
        if not role:
            raise ValueError("matched response namespace is required")
        self.delegate = delegate
        self.root = Path(root)
        self.mode = normalized_mode
        self.namespace = role
        self.session = session
        self._sequence = 0
        self._last_response_metadata: dict[str, Any] = {}
        if self.mode == "record":
            (self.root / self.namespace).mkdir(parents=True, exist_ok=True)
        elif not (self.root / self.namespace).is_dir():
            raise FileNotFoundError(
                f"missing matched response role fixture: {self.root / self.namespace}"
            )

    @property
    def model(self) -> str:
        return self.delegate.model

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
        if not self.session.active:
            response = self.delegate.chat(
                prompt,
                image_paths=image_paths,
                image_labels=image_labels,
                prompt_position=prompt_position,
                max_tokens=max_tokens,
                _retry_truncation=_retry_truncation,
            )
            self._last_response_metadata = {
                **self.delegate.last_response_metadata,
                "matched_response_cache_mode": self.mode,
                "matched_response_cache_active": False,
                "matched_response_cache_hit": False,
            }
            self.session.note("live_after_treatment", self.namespace)
            return response

        self._sequence += 1
        request = _matched_response_request(
            namespace=self.namespace,
            model=self.model,
            prompt=prompt,
            image_paths=image_paths,
            image_labels=image_labels,
            prompt_position=prompt_position,
            max_tokens=max_tokens,
        )
        request_digest = _stable_json_digest(request)
        fixture_path = (
            self.root / self.namespace / f"{self._sequence:06d}.json"
        )
        if self.mode == "replay":
            if not fixture_path.is_file():
                self.session.note_mismatch()
                raise MatchedResponseReplayError(
                    f"missing matched response fixture: {self.namespace}/{self._sequence:06d}"
                )
            fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
            if str(fixture.get("request_digest", "")) != request_digest:
                self.session.note_mismatch()
                raise MatchedResponseReplayError(
                    f"matched response request mismatch: {self.namespace}/{self._sequence:06d}"
                )
            response = str(fixture.get("response", "") or "")
            self._last_response_metadata = {
                **dict(fixture.get("response_metadata") or {}),
                "matched_response_cache_mode": self.mode,
                "matched_response_cache_active": True,
                "matched_response_cache_hit": True,
                "matched_response_sequence": self._sequence,
                "matched_response_request_digest": request_digest,
            }
            self.session.note("replayed", self.namespace)
            return response

        response = self.delegate.chat(
            prompt,
            image_paths=image_paths,
            image_labels=image_labels,
            prompt_position=prompt_position,
            max_tokens=max_tokens,
            _retry_truncation=_retry_truncation,
        )
        response_metadata = dict(self.delegate.last_response_metadata)
        fixture = {
            "schema_version": "MatchedPreTreatmentResponseEntryV1",
            "namespace": self.namespace,
            "sequence": self._sequence,
            "request": request,
            "request_digest": request_digest,
            "response": response,
            "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
            "response_metadata": response_metadata,
        }
        if fixture_path.exists():
            existing = json.loads(fixture_path.read_text(encoding="utf-8"))
            if (
                str(existing.get("request_digest", "")) != request_digest
                or str(existing.get("response_sha256", ""))
                != fixture["response_sha256"]
            ):
                self.session.note_mismatch()
                raise MatchedResponseReplayError(
                    f"matched response record conflict: {self.namespace}/{self._sequence:06d}"
                )
        else:
            _atomic_json_write(fixture_path, fixture)
        self._last_response_metadata = {
            **response_metadata,
            "matched_response_cache_mode": self.mode,
            "matched_response_cache_active": True,
            "matched_response_cache_hit": False,
            "matched_response_sequence": self._sequence,
            "matched_response_request_digest": request_digest,
        }
        self.session.note("recorded", self.namespace)
        return response

    @property
    def last_response_metadata(self) -> dict[str, Any]:
        return dict(self._last_response_metadata)

    @property
    def replay_settings(self) -> dict[str, Any]:
        return {
            **self.delegate.replay_settings,
            "matched_response_cache_mode": self.mode,
        }

    def set_requested_seed(self, seed: int | None) -> None:
        self.delegate.set_requested_seed(seed)


def _matched_response_request(
    *,
    namespace: str,
    model: str,
    prompt: str,
    image_paths: Sequence[str],
    image_labels: Sequence[str],
    prompt_position: str,
    max_tokens: int,
) -> dict[str, Any]:
    image_digests = []
    for raw_path in image_paths:
        path = Path(raw_path)
        image_digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
    return {
        "namespace": str(namespace),
        "model": str(model),
        "prompt_sha256": hashlib.sha256(str(prompt).encode("utf-8")).hexdigest(),
        "image_sha256": image_digests,
        "image_labels": [str(value) for value in image_labels],
        "prompt_position": str(prompt_position),
        "max_tokens": int(max_tokens),
    }


def _stable_json_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
