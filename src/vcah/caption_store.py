from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any, Mapping, Sequence

from vcah.caption_schema import (
    CaptionChunkV1,
    CaptionPassageV1,
    chunk_from_dict,
    chunk_to_dict,
    passage_from_dict,
    passage_to_dict,
)


CAPTION_STORE_VERSION = 1
ACTIVE_CAPTION_CONFIG_VERSION = 1
VALID_STATUSES = frozenset({"pending", "running", "success", "failed"})


class CaptionStore:
    def __init__(
        self,
        asset_root: Path,
        config_digest: str,
        *,
        eager_exports: bool = True,
    ) -> None:
        self.asset_root = Path(asset_root)
        self.config_digest = str(config_digest)
        self.eager_exports = bool(eager_exports)
        self.root = self.asset_root / "captions"
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / f"state.{self.config_digest}.json"
        self.chunks_path = self.root / f"chunks.{self.config_digest}.jsonl"
        self.passages_path = self.root / f"passages.{self.config_digest}.jsonl"
        self._lock = threading.RLock()
        self._state = self._load_state()

    def recover_interrupted(self) -> int:
        with self._lock:
            recovered = 0
            for record in self._state["records"].values():
                if record.get("status") == "running":
                    record["status"] = "pending"
                    recovered += 1
            if recovered:
                self._persist()
            return recovered

    def prepare(self, cache_key: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            key = str(cache_key)
            records = self._state["records"]
            if key not in records:
                records[key] = {
                    "status": "pending",
                    "attempt_count": 0,
                    "last_error": None,
                    "retry_after": None,
                    "metadata": dict(metadata),
                    "chunk": None,
                    "passages": [],
                }
                self._persist()
            return dict(records[key])

    def begin(self, cache_key: str) -> bool:
        with self._lock:
            record = self._record(cache_key)
            if record["status"] == "success":
                return False
            record["status"] = "running"
            record["attempt_count"] = int(record.get("attempt_count", 0) or 0) + 1
            record["last_error"] = None
            record["retry_after"] = None
            self._persist()
            return True

    def mark_success(
        self,
        cache_key: str,
        chunk: CaptionChunkV1,
        passages: Sequence[CaptionPassageV1],
    ) -> None:
        with self._lock:
            record = self._record(cache_key)
            record["status"] = "success"
            record["last_error"] = None
            record["retry_after"] = None
            record["chunk"] = chunk_to_dict(chunk)
            record["passages"] = [passage_to_dict(passage) for passage in passages]
            self._persist()

    def mark_failed(self, cache_key: str, error: str, *, retry_after: str | None = None) -> None:
        with self._lock:
            record = self._record(cache_key)
            record["status"] = "failed"
            record["last_error"] = str(error)[:1000]
            record["retry_after"] = retry_after
            self._persist()

    def record(self, cache_key: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._state["records"].get(str(cache_key))
            return dict(value) if isinstance(value, Mapping) else None

    def successful_chunks(self) -> tuple[CaptionChunkV1, ...]:
        with self._lock:
            chunks = [
                chunk_from_dict(record["chunk"])
                for record in self._state["records"].values()
                if record.get("status") == "success" and isinstance(record.get("chunk"), Mapping)
            ]
            return tuple(sorted(chunks, key=lambda chunk: (chunk.virtual_start_sec, chunk.caption_id)))

    def successful_records(self) -> tuple[tuple[str, CaptionChunkV1], ...]:
        with self._lock:
            records = [
                (key, chunk_from_dict(record["chunk"]))
                for key, record in self._state["records"].items()
                if record.get("status") == "success" and isinstance(record.get("chunk"), Mapping)
            ]
            return tuple(
                sorted(records, key=lambda item: (item[1].virtual_start_sec, item[1].caption_id))
            )

    def replace_successful_records(
        self,
        replacements: Mapping[
            str,
            tuple[CaptionChunkV1, Sequence[CaptionPassageV1]],
        ],
    ) -> int:
        with self._lock:
            for cache_key, (chunk, passages) in replacements.items():
                record = self._record(cache_key)
                if record.get("status") != "success":
                    raise ValueError(f"Caption record is not successful: {cache_key}")
                record["chunk"] = chunk_to_dict(chunk)
                record["passages"] = [passage_to_dict(passage) for passage in passages]
            if replacements:
                self._persist()
            return len(replacements)

    def successful_passages(self) -> tuple[CaptionPassageV1, ...]:
        with self._lock:
            passages = [
                passage_from_dict(payload)
                for record in self._state["records"].values()
                if record.get("status") == "success"
                for payload in record.get("passages", ())
                if isinstance(payload, Mapping)
            ]
            return tuple(
                sorted(
                    passages,
                    key=lambda passage: (
                        passage.virtual_start_sec,
                        passage.caption_id,
                        passage.ordinal,
                    ),
                )
            )

    def status_counts(self) -> dict[str, int]:
        with self._lock:
            counts = {status: 0 for status in sorted(VALID_STATUSES)}
            for record in self._state["records"].values():
                status = str(record.get("status", "pending"))
                counts[status] = counts.get(status, 0) + 1
            return counts

    def write_run_summary(self, run_id: str, payload: Mapping[str, Any]) -> Path:
        with self._lock:
            path = self.root / f"run.{_safe_name(run_id)}.json"
            suffix = 1
            while path.exists():
                path = self.root / f"run.{_safe_name(run_id)}.{suffix}.json"
                suffix += 1
            _atomic_write_json(
                path,
                {
                    "schema_version": CAPTION_STORE_VERSION,
                    "config_digest": self.config_digest,
                    **dict(payload),
                },
            )
            return path

    def flush_exports(self) -> None:
        with self._lock:
            self._write_exports()

    def _record(self, cache_key: str) -> dict[str, Any]:
        key = str(cache_key)
        record = self._state["records"].get(key)
        if not isinstance(record, dict):
            raise KeyError(f"Unknown caption cache key: {key}")
        return record

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {
                "schema_version": CAPTION_STORE_VERSION,
                "config_digest": self.config_digest,
                "records": {},
            }
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        if str(payload.get("config_digest")) != self.config_digest:
            raise ValueError(f"Caption state digest mismatch: {self.state_path}")
        records = payload.get("records")
        if not isinstance(records, dict):
            raise ValueError(f"Caption state records must be a mapping: {self.state_path}")
        for key, record in records.items():
            if not isinstance(record, dict) or record.get("status") not in VALID_STATUSES:
                raise ValueError(f"Invalid caption state record: {key}")
        return dict(payload)

    def _persist(self) -> None:
        _atomic_write_json(self.state_path, self._state)
        if self.eager_exports:
            self._write_exports()

    def _write_exports(self) -> None:
        _atomic_write_jsonl(
            self.chunks_path,
            (chunk_to_dict(chunk) for chunk in self.successful_chunks()),
        )
        _atomic_write_jsonl(
            self.passages_path,
            (passage_to_dict(passage) for passage in self.successful_passages()),
        )


def resolve_caption_passages_path(
    asset_root: Path,
    *,
    config_digest: str | None = None,
) -> tuple[Path, str]:
    captions_root = Path(asset_root) / "captions"
    if config_digest:
        resolved_digest = str(config_digest)
        path = captions_root / f"passages.{resolved_digest}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Caption passages not found: {path}")
        return path, resolved_digest

    active_path = captions_root / "active.json"
    if active_path.is_file():
        try:
            active = json.loads(active_path.read_text(encoding="utf-8"))
            resolved_digest = str(active["config_digest"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid active caption config: {active_path}") from exc
        path = captions_root / f"passages.{resolved_digest}.jsonl"
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Active caption passages not found: {path}")
        return path, resolved_digest

    candidates = tuple(
        path
        for path in sorted(captions_root.glob("passages.*.jsonl"))
        if path.stat().st_size > 0
    )
    if not candidates:
        raise FileNotFoundError(f"No caption passages found under {captions_root}")
    if len(candidates) != 1:
        raise ValueError("Multiple caption passage caches exist; select or activate a config digest")
    path = candidates[0]
    resolved_digest = path.name[len("passages.") : -len(".jsonl")]
    return path, resolved_digest


def activate_caption_config(
    asset_root: Path,
    config_digest: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    path, resolved_digest = resolve_caption_passages_path(
        asset_root,
        config_digest=config_digest,
    )
    if path.stat().st_size <= 0:
        raise ValueError(f"Cannot activate an empty caption passage cache: {path}")
    active_path = Path(asset_root) / "captions" / "active.json"
    _atomic_write_json(
        active_path,
        {
            "schema_version": ACTIVE_CAPTION_CONFIG_VERSION,
            "config_digest": resolved_digest,
            "passages_path": str(path),
            "metadata": dict(metadata or {}),
        },
    )
    return active_path


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_write_jsonl(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in str(value))
