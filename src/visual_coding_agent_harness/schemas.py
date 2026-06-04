"""Shared v4 typed evidence schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class CandidateOptionRelation:
    option: str
    relation: str
    strength: float = 0.0
    assigned_by: str = ""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CandidateOptionRelation":
        return cls(
            option=str(payload.get("option", "")),
            relation=str(payload.get("relation", "")),
            strength=float(payload.get("strength", 0.0) or 0.0),
            assigned_by=str(payload.get("assigned_by", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GroundingCandidate:
    window: Mapping[str, Any]
    score: float = 0.0
    relevance_reason: str = ""
    channel: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisionFact:
    fact: str
    event_label: str = ""
    polarity: str = "present"
    time_range: Sequence[float] | None = None
    grounding_quality: str = "visually_confirmed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceRowV2:
    obs_id: str
    tool: str
    claim: str
    confidence: float
    grounding_quality: str
    supported_option: str | None = None
    time_range: Sequence[float] | None = None
    event_label: str = ""
    candidate_option_relations: Sequence[Mapping[str, Any]] = field(default_factory=list)
    legacy_worker_vote: bool = False
    limitations: str = ""
    artifact: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnswerAgentDecision:
    option: str | None
    rationale: str = ""
    citations: Sequence[str] = field(default_factory=list)
    option_relations: Mapping[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    need_more_evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalVerifierResult:
    verdict: str
    expected_events: Sequence[str] = field(default_factory=list)
    observed_events: Sequence[Mapping[str, Any]] = field(default_factory=list)
    matched_events: Sequence[Mapping[str, Any]] = field(default_factory=list)
    reasons: Sequence[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
