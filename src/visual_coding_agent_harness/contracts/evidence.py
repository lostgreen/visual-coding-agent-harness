"""Evidence contracts for the active multi_v3 path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from .report import CandidateShot, Finding, VerifyRequest


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    claim: str
    stance: Literal["supports", "refutes"]
    modality: Literal["frame", "asr", "ocr"]
    time_sec: float
    pointer: str
    verbatim: str
    query_id: str
    beat_id: str

    def __post_init__(self) -> None:
        if not self.verbatim.strip():
            raise ValueError("EvidenceRecord.verbatim must be non-empty")
        if self.modality == "frame" and not self.pointer.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            raise ValueError("frame evidence requires image pointer")
        object.__setattr__(self, "time_sec", float(self.time_sec))

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "claim": self.claim,
            "stance": self.stance,
            "modality": self.modality,
            "time_sec": float(self.time_sec),
            "pointer": self.pointer,
            "verbatim": self.verbatim,
            "query_id": self.query_id,
            "beat_id": self.beat_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceRecord":
        return cls(
            evidence_id=str(value.get("evidence_id") or ""),
            claim=str(value.get("claim") or ""),
            stance=str(value.get("stance") or "supports"),  # type: ignore[arg-type]
            modality=str(value.get("modality") or "frame"),  # type: ignore[arg-type]
            time_sec=float(value.get("time_sec", 0.0) or 0.0),
            pointer=str(value.get("pointer") or ""),
            verbatim=str(value.get("verbatim") or ""),
            query_id=str(value.get("query_id") or ""),
            beat_id=str(value.get("beat_id") or ""),
        )


__all__ = ["CandidateShot", "EvidenceRecord", "Finding", "VerifyRequest"]
