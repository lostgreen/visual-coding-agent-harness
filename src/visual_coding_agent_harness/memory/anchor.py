"""Provenance anchors for tool observations."""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Mapping


SourceKind = str


def normalized_text(text: str) -> str:
    """Normalize text for provenance substring checks."""

    return " ".join(unicodedata.normalize("NFKC", str(text)).split())


def excerpt_hash(text: str) -> str:
    return hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceAnchor:
    anchor_id: str
    observation_id: str
    source_kind: SourceKind
    segment_id: str | None = None
    cue_id: str | None = None
    region_id: str | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    field_path: str = ""
    excerpt: str = ""
    excerpt_hash: str = ""
    embedding_ref: str | None = None
    frame_refs: tuple[str, ...] = ()
    modality: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchor_id", str(self.anchor_id))
        object.__setattr__(self, "observation_id", str(self.observation_id))
        object.__setattr__(self, "source_kind", str(self.source_kind))
        object.__setattr__(self, "segment_id", _optional_str(self.segment_id))
        object.__setattr__(self, "cue_id", _optional_str(self.cue_id))
        object.__setattr__(self, "region_id", _optional_str(self.region_id))
        object.__setattr__(self, "field_path", str(self.field_path or ""))
        object.__setattr__(self, "excerpt", str(self.excerpt or ""))
        object.__setattr__(self, "excerpt_hash", str(self.excerpt_hash or excerpt_hash(self.excerpt)))
        object.__setattr__(self, "embedding_ref", _optional_str(self.embedding_ref))
        object.__setattr__(self, "frame_refs", tuple(str(item) for item in self.frame_refs))
        object.__setattr__(self, "modality", _optional_str(self.modality))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SourceAnchor":
        return cls(
            anchor_id=str(payload.get("anchor_id", "")),
            observation_id=str(payload.get("observation_id", "")),
            source_kind=str(payload.get("source_kind", "")),
            segment_id=_optional_str(payload.get("segment_id")),
            cue_id=_optional_str(payload.get("cue_id")),
            region_id=_optional_str(payload.get("region_id")),
            start_sec=_optional_float(payload.get("start_sec")),
            end_sec=_optional_float(payload.get("end_sec")),
            field_path=str(payload.get("field_path", "") or ""),
            excerpt=str(payload.get("excerpt", "") or ""),
            excerpt_hash=str(payload.get("excerpt_hash", "") or ""),
            embedding_ref=_optional_str(payload.get("embedding_ref")),
            frame_refs=tuple(str(item) for item in _sequence(payload.get("frame_refs"))),
            modality=_optional_str(payload.get("modality")),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "observation_id": self.observation_id,
            "source_kind": self.source_kind,
            "segment_id": self.segment_id,
            "cue_id": self.cue_id,
            "region_id": self.region_id,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "field_path": self.field_path,
            "excerpt": self.excerpt,
            "excerpt_hash": self.excerpt_hash,
            "embedding_ref": self.embedding_ref,
            "frame_refs": list(self.frame_refs),
            "modality": self.modality,
            "metadata": dict(self.metadata),
        }


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or value is None:
        return ()
    try:
        return tuple(value)
    except TypeError:
        return ()
