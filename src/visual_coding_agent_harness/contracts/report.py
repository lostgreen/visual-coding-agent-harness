"""Investigation report contracts for the multi_v3 loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence


ReportStatus = Literal["satisfied", "partial", "empty", "infeasible"]


@dataclass(frozen=True)
class CandidateShot:
    shot_id: str
    score: float
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {"shot_id": self.shot_id, "score": float(self.score), "reason": self.reason}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateShot":
        return cls(shot_id=str(value.get("shot_id") or ""), score=float(value.get("score", 0.0) or 0.0), reason=str(value.get("reason") or ""))


@dataclass(frozen=True)
class VerifyRequest:
    shot_id: str
    time_range: tuple[float, float]
    focus_claim: str
    sampling: Mapping[str, object] = field(default_factory=dict)
    checks: Sequence[Mapping[str, object]] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.shot_id:
            raise ValueError("shot_id is required")
        start, end = self.time_range
        if float(end) < float(start):
            raise ValueError("time_range end must be greater than or equal to start")
        object.__setattr__(self, "time_range", (float(start), float(end)))
        object.__setattr__(self, "sampling", dict(self.sampling))
        object.__setattr__(self, "checks", tuple(dict(item) for item in self.checks))

    def to_dict(self) -> dict[str, object]:
        return {
            "shot_id": self.shot_id,
            "time_range": list(self.time_range),
            "focus_claim": self.focus_claim,
            "sampling": dict(self.sampling),
            "checks": [dict(item) for item in self.checks],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VerifyRequest":
        time_range = _sequence(value.get("time_range"))
        return cls(
            shot_id=str(value.get("shot_id") or ""),
            time_range=(
                float(time_range[0]) if len(time_range) >= 1 else 0.0,
                float(time_range[1]) if len(time_range) >= 2 else 0.0,
            ),
            focus_claim=str(value.get("focus_claim") or ""),
            sampling=_mapping(value.get("sampling")),
            checks=tuple(_mapping(item) for item in _sequence(value.get("checks"))),
        )


@dataclass(frozen=True)
class Finding:
    finding_id: str
    query_id: str
    shot_id: str
    summary: str
    supports_options: Sequence[str] = field(default_factory=tuple)
    refutes_options: Sequence[str] = field(default_factory=tuple)
    citation_ids: Sequence[str] = field(default_factory=tuple)
    confidence: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "supports_options", _text_tuple(self.supports_options))
        object.__setattr__(self, "refutes_options", _text_tuple(self.refutes_options))
        object.__setattr__(self, "citation_ids", _text_tuple(self.citation_ids))
        if self.confidence is not None:
            object.__setattr__(self, "confidence", float(self.confidence))

    def to_dict(self) -> dict[str, object]:
        return {
            "finding_id": self.finding_id,
            "query_id": self.query_id,
            "shot_id": self.shot_id,
            "summary": self.summary,
            "supports_options": list(self.supports_options),
            "refutes_options": list(self.refutes_options),
            "citation_ids": list(self.citation_ids),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Finding":
        return cls(
            finding_id=str(value.get("finding_id") or ""),
            query_id=str(value.get("query_id") or ""),
            shot_id=str(value.get("shot_id") or ""),
            summary=str(value.get("summary") or ""),
            supports_options=_text_tuple(value.get("supports_options") or ()),
            refutes_options=_text_tuple(value.get("refutes_options") or ()),
            citation_ids=_text_tuple(value.get("citation_ids") or ()),
            confidence=float(value["confidence"]) if value.get("confidence") is not None else None,
        )


@dataclass(frozen=True)
class InvestigationReport:
    query_id: str
    status: ReportStatus
    findings: Sequence[Finding] = field(default_factory=tuple)
    explored_shots: Sequence[str] = field(default_factory=tuple)
    verified_shots: Sequence[str] = field(default_factory=tuple)
    unresolved: Sequence[str] = field(default_factory=tuple)
    cost: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "findings", tuple(self.findings))
        object.__setattr__(self, "explored_shots", _text_tuple(self.explored_shots))
        object.__setattr__(self, "verified_shots", _text_tuple(self.verified_shots))
        object.__setattr__(self, "unresolved", _text_tuple(self.unresolved))
        object.__setattr__(self, "cost", {str(key): int(value) for key, value in dict(self.cost).items()})

    def to_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "status": self.status,
            "findings": [finding.to_dict() for finding in self.findings],
            "explored_shots": list(self.explored_shots),
            "verified_shots": list(self.verified_shots),
            "unresolved": list(self.unresolved),
            "cost": dict(self.cost),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InvestigationReport":
        return cls(
            query_id=str(value.get("query_id") or ""),
            status=str(value.get("status") or "empty"),  # type: ignore[arg-type]
            findings=tuple(Finding.from_dict(item) for item in _sequence(value.get("findings")) if isinstance(item, Mapping)),
            explored_shots=_text_tuple(value.get("explored_shots") or ()),
            verified_shots=_text_tuple(value.get("verified_shots") or ()),
            unresolved=_text_tuple(value.get("unresolved") or ()),
            cost={str(key): int(val) for key, val in _mapping(value.get("cost")).items()},
        )


@dataclass(frozen=True)
class DigestItem:
    query_id: str
    goal_id: str
    status: ReportStatus
    summary: str
    supports_options: Sequence[str] = field(default_factory=tuple)
    refutes_options: Sequence[str] = field(default_factory=tuple)
    citation_ids: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "supports_options", _text_tuple(self.supports_options))
        object.__setattr__(self, "refutes_options", _text_tuple(self.refutes_options))
        object.__setattr__(self, "citation_ids", _text_tuple(self.citation_ids))

    @classmethod
    def from_report(cls, report: InvestigationReport, *, goal_id: str) -> "DigestItem":
        findings = tuple(report.findings)
        summary = " ".join(finding.summary for finding in findings if finding.summary).strip()
        return cls(
            query_id=report.query_id,
            goal_id=goal_id,
            status=report.status,
            summary=summary or "; ".join(report.unresolved),
            supports_options=_unique(item for finding in findings for item in finding.supports_options),
            refutes_options=_unique(item for finding in findings for item in finding.refutes_options),
            citation_ids=_unique(item for finding in findings for item in finding.citation_ids),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "goal_id": self.goal_id,
            "status": self.status,
            "summary": self.summary,
            "supports_options": list(self.supports_options),
            "refutes_options": list(self.refutes_options),
            "citation_ids": list(self.citation_ids),
        }


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> tuple[Any, ...]:
    if value is None or isinstance(value, (str, bytes)):
        return ()
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return ()


def _text_tuple(value: object) -> tuple[str, ...]:
    if value is None or isinstance(value, (str, bytes)):
        values = () if value is None else (value,)
    else:
        try:
            values = tuple(value)  # type: ignore[arg-type]
        except TypeError:
            values = (value,)
    return tuple(text for item in values if (text := str(item).strip()))


def _unique(values: Sequence[str] | Any) -> tuple[str, ...]:
    seen = set()
    result = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return tuple(result)
