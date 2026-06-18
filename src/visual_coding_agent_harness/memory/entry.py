"""Planner-written memory entries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from .anchor import SourceAnchor


MemoryKind = Literal["note", "support", "conflict", "reject", "hypothesis", "open_question"]
MemoryConfidence = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class MemoryEntry:
    entry_id: str
    round_number: int
    kind: MemoryKind
    claim: str
    anchors: tuple[SourceAnchor, ...]
    supports_option: str | None = None
    confidence: MemoryConfidence = "medium"
    previous_memory_refs: tuple[str, ...] = ()
    superseded_by: str | None = None
    tags: tuple[str, ...] = ()
    created_at_sec: float = 0.0
    role: str | None = None
    layer: str | None = None
    embedding_refs: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry_id", str(self.entry_id))
        object.__setattr__(self, "round_number", int(self.round_number))
        object.__setattr__(self, "kind", _memory_kind(self.kind))
        object.__setattr__(self, "claim", str(self.claim or ""))
        object.__setattr__(self, "anchors", tuple(self.anchors))
        object.__setattr__(self, "supports_option", _optional_str(self.supports_option))
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(self, "previous_memory_refs", tuple(str(item) for item in self.previous_memory_refs))
        object.__setattr__(self, "superseded_by", _optional_str(self.superseded_by))
        object.__setattr__(self, "tags", tuple(str(item) for item in self.tags))
        object.__setattr__(self, "role", _optional_str(self.role))
        object.__setattr__(self, "layer", _optional_str(self.layer))
        object.__setattr__(self, "embedding_refs", tuple(str(item) for item in self.embedding_refs))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MemoryEntry":
        anchors = payload.get("anchors", ())
        return cls(
            entry_id=str(payload.get("entry_id", "")),
            round_number=int(payload.get("round_number", 0) or 0),
            kind=_memory_kind(payload.get("kind", "note")),
            claim=str(payload.get("claim", "") or ""),
            anchors=tuple(SourceAnchor.from_mapping(anchor) for anchor in _mapping_sequence(anchors)),
            supports_option=_optional_str(payload.get("supports_option")),
            confidence=_confidence(payload.get("confidence", "medium")),
            previous_memory_refs=tuple(str(item) for item in _sequence(payload.get("previous_memory_refs"))),
            superseded_by=_optional_str(payload.get("superseded_by")),
            tags=tuple(str(item) for item in _sequence(payload.get("tags"))),
            created_at_sec=float(payload.get("created_at_sec", 0.0) or 0.0),
            role=_optional_str(payload.get("role")),
            layer=_optional_str(payload.get("layer")),
            embedding_refs=tuple(str(item) for item in _sequence(payload.get("embedding_refs"))),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "round_number": self.round_number,
            "kind": self.kind,
            "claim": self.claim,
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "supports_option": self.supports_option,
            "confidence": self.confidence,
            "previous_memory_refs": list(self.previous_memory_refs),
            "superseded_by": self.superseded_by,
            "tags": list(self.tags),
            "created_at_sec": self.created_at_sec,
            "role": self.role,
            "layer": self.layer,
            "embedding_refs": list(self.embedding_refs),
            "metadata": dict(self.metadata),
        }


def _memory_kind(value: Any) -> MemoryKind:
    text = str(value or "note")
    if text in {"note", "support", "conflict", "reject", "hypothesis", "open_question"}:
        return text  # type: ignore[return-value]
    raise ValueError(f"memory_validation_failed: unknown kind={text}")


def _confidence(value: Any) -> MemoryConfidence:
    text = str(value or "medium")
    if text in {"high", "medium", "low"}:
        return text  # type: ignore[return-value]
    raise ValueError(f"memory_validation_failed: unknown confidence={text}")


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or value is None:
        return ()
    try:
        return tuple(value)
    except TypeError:
        return ()


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    return tuple(item for item in _sequence(value) if isinstance(item, Mapping))
