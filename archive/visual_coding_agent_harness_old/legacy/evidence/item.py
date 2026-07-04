"""Evidence-facing view of committed workspace memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..memory import MemoryEntry, SourceAnchor

EvidenceItemPolarity = str

_SUPPORT_KINDS = frozenset({"visual_support", "answer_support", "caption_support", "synthesized_support"})
_REFUTE_KINDS = frozenset({"answer_conflict", "answer_conflict_resolved", "contradiction", "contradicting", "conflict"})


@dataclass(frozen=True)
class EvidenceItem:
    """Semantic evidence item derived from a MemoryEntry."""

    evidence_id: str
    option_id: str | None
    polarity: EvidenceItemPolarity
    claim: str
    segment_id: str | None
    time_range: tuple[float, float] | None
    modality: str
    confidence: str
    source_need_id: str | None
    tags: tuple[str, ...]
    memory_kind: str

    @classmethod
    def from_memory_entry(cls, entry: MemoryEntry) -> "EvidenceItem":
        anchor = _primary_anchor(entry.anchors)
        return cls(
            evidence_id=entry.entry_id,
            option_id=entry.supports_option,
            polarity=_polarity_from_memory(entry.kind, entry.metadata),
            claim=entry.claim,
            segment_id=anchor.segment_id if anchor is not None else None,
            time_range=_time_range(anchor),
            modality=_anchor_modality(anchor),
            confidence=entry.confidence,
            source_need_id=_source_need_id(entry.metadata),
            tags=tuple(entry.tags),
            memory_kind=entry.kind,
        )


def _primary_anchor(anchors: tuple[SourceAnchor, ...]) -> SourceAnchor | None:
    return anchors[0] if anchors else None


def _time_range(anchor: SourceAnchor | None) -> tuple[float, float] | None:
    if anchor is None or anchor.start_sec is None or anchor.end_sec is None:
        return None
    return float(anchor.start_sec), float(anchor.end_sec)


def _anchor_modality(anchor: SourceAnchor | None) -> str:
    if anchor is None:
        return ""
    if anchor.modality:
        return anchor.modality
    source_kind = anchor.source_kind.lower()
    if "audio" in source_kind or "asr" in source_kind:
        return "asr"
    if "ocr" in source_kind:
        return "ocr"
    if "visual" in source_kind or "caption" in source_kind:
        return "visual"
    return source_kind


def _polarity_from_memory(kind: str, metadata: Mapping[str, object]) -> EvidenceItemPolarity:
    verdict = str(metadata.get("verdict") or "").strip().lower()
    if kind in _SUPPORT_KINDS:
        return "supports"
    if kind in _REFUTE_KINDS and (not verdict or verdict == "contradicted"):
        return "refutes"
    if kind == "local_negative" or verdict == "not_found_in_window":
        return "absent"
    if kind == "verification_uncertain" or verdict == "uncertain":
        return "inconclusive"
    return "inconclusive"


def _source_need_id(metadata: Mapping[str, object]) -> str | None:
    for key in ("source_need_id", "sub_goal_id", "need_id"):
        value = metadata.get(key)
        text = str(value or "").strip()
        if text:
            return text
    return None
