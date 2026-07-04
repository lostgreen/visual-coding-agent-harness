"""Observation memo storage for the hot workspace layer."""

from __future__ import annotations

from dataclasses import dataclass, replace
import fcntl
import json
from pathlib import Path
from typing import Any

from visual_coding_agent_harness.contracts.playbook import Playbook


@dataclass(frozen=True)
class ObservationMemo:
    memo_id: str
    beat_id: str
    observation: str
    source_query_id: str
    source_playbook: Playbook
    created_at: str
    verified_by: tuple[str, ...] = ()
    invalidated_by: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.memo_id:
            raise ValueError("memo_id is required")
        if not self.beat_id:
            raise ValueError("beat_id is required")
        if not self.observation.strip():
            raise ValueError("observation is required")
        object.__setattr__(self, "source_playbook", Playbook.parse(self.source_playbook))
        object.__setattr__(self, "verified_by", _text_tuple(self.verified_by))
        object.__setattr__(self, "invalidated_by", _text_tuple(self.invalidated_by))

    def to_dict(self) -> dict[str, object]:
        return {
            "memo_id": self.memo_id,
            "beat_id": self.beat_id,
            "observation": self.observation,
            "source_query_id": self.source_query_id,
            "source_playbook": self.source_playbook.value,
            "created_at": self.created_at,
            "verified_by": list(self.verified_by),
            "invalidated_by": list(self.invalidated_by),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ObservationMemo":
        return cls(
            memo_id=str(payload.get("memo_id") or ""),
            beat_id=str(payload.get("beat_id") or ""),
            observation=str(payload.get("observation") or ""),
            source_query_id=str(payload.get("source_query_id") or ""),
            source_playbook=Playbook.parse(payload.get("source_playbook")),
            created_at=str(payload.get("created_at") or ""),
            verified_by=_text_tuple(payload.get("verified_by") or ()),
            invalidated_by=_text_tuple(payload.get("invalidated_by") or ()),
        )


class MemoStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(self, memo: ObservationMemo) -> None:
        self._append_record({"op": "append", "memo": memo.to_dict()})

    def get(self, beat_id: str) -> tuple[ObservationMemo, ...]:
        return tuple(memo for memo in self.load() if memo.beat_id == beat_id)

    def invalidate(self, memo_id: str, *, by_evidence_id: str) -> None:
        self._append_record({"op": "invalidate", "memo_id": str(memo_id), "by_evidence_id": str(by_evidence_id)})

    def load(self) -> tuple[ObservationMemo, ...]:
        if not self.path.exists():
            return ()
        memos: dict[str, ObservationMemo] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("op") == "append" and isinstance(record.get("memo"), dict):
                memo = ObservationMemo.from_dict(record["memo"])
                memos[memo.memo_id] = memo
            elif record.get("op") == "invalidate":
                memo_id = str(record.get("memo_id") or "")
                memo = memos.get(memo_id)
                if memo is not None:
                    invalidated_by = tuple(dict.fromkeys((*memo.invalidated_by, str(record.get("by_evidence_id") or ""))))
                    memos[memo_id] = replace(memo, invalidated_by=invalidated_by)
        return tuple(memo for memo in memos.values() if not memo.invalidated_by)

    def _append_record(self, record: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _text_tuple(value: object) -> tuple[str, ...]:
    if value is None or isinstance(value, (str, bytes)):
        values = () if value is None else (value,)
    else:
        try:
            values = tuple(value)  # type: ignore[arg-type]
        except TypeError:
            values = (value,)
    return tuple(text for item in values if (text := str(item).strip()))
