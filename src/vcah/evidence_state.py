from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping, Sequence


EvidenceKind = Literal[
    "generic",
    "text_exact",
    "ui_text",
    "persistent_state",
    "transient_event",
    "relation",
]
ObligationStatus = Literal[
    "open",
    "candidate_found",
    "observed",
    "contested",
    "satisfied",
    "unresolved",
]

EVIDENCE_KINDS = {
    "generic",
    "text_exact",
    "ui_text",
    "persistent_state",
    "transient_event",
    "relation",
}
OBLIGATION_STATUSES = {
    "open",
    "candidate_found",
    "observed",
    "contested",
    "satisfied",
    "unresolved",
}


@dataclass(frozen=True)
class EvidenceObligation:
    requirement_id: str
    observable_goal: str
    evidence_kind: EvidenceKind = "generic"
    temporal_relation: str | None = None
    depends_on: tuple[str, ...] = ()
    answer_bearing: bool = True

    def __post_init__(self) -> None:
        requirement_id = str(self.requirement_id or "").strip()
        observable_goal = str(self.observable_goal or "").strip()
        if not requirement_id:
            raise ValueError("obligation_requires_requirement_id")
        if not observable_goal:
            raise ValueError(f"obligation_requires_observable_goal:{requirement_id}")
        evidence_kind = str(self.evidence_kind or "generic").strip().casefold()
        if evidence_kind not in EVIDENCE_KINDS:
            raise ValueError(f"invalid_evidence_kind:{evidence_kind}")
        depends_on = _ids(self.depends_on)
        if requirement_id in depends_on:
            raise ValueError(f"obligation_cannot_depend_on_itself:{requirement_id}")
        temporal_relation = str(self.temporal_relation or "").strip().casefold() or None
        object.__setattr__(self, "requirement_id", requirement_id)
        object.__setattr__(self, "observable_goal", observable_goal)
        object.__setattr__(self, "evidence_kind", evidence_kind)
        object.__setattr__(self, "temporal_relation", temporal_relation)
        object.__setattr__(self, "depends_on", depends_on)
        object.__setattr__(self, "answer_bearing", bool(self.answer_bearing))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "EvidenceObligation":
        return cls(
            requirement_id=str(value.get("requirement_id", value.get("id", "")) or ""),
            observable_goal=str(value.get("observable_goal", value.get("goal", "")) or ""),
            evidence_kind=str(value.get("evidence_kind", "generic") or "generic"),  # type: ignore[arg-type]
            temporal_relation=value.get("temporal_relation"),  # type: ignore[arg-type]
            depends_on=tuple(value.get("depends_on", ()) or ()),  # type: ignore[arg-type]
            answer_bearing=bool(value.get("answer_bearing", True)),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceObligationState:
    requirement_id: str
    status: ObligationStatus = "open"
    supporting_claim_ids: tuple[str, ...] = ()
    supporting_attempt_ids: tuple[str, ...] = ()
    residual_uncertainty: str = ""

    def __post_init__(self) -> None:
        requirement_id = str(self.requirement_id or "").strip()
        if not requirement_id:
            raise ValueError("obligation_state_requires_requirement_id")
        status = str(self.status or "open").strip().casefold()
        if status not in OBLIGATION_STATUSES:
            raise ValueError(f"invalid_obligation_status:{status}")
        object.__setattr__(self, "requirement_id", requirement_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "supporting_claim_ids", _ids(self.supporting_claim_ids))
        object.__setattr__(self, "supporting_attempt_ids", _ids(self.supporting_attempt_ids))
        object.__setattr__(self, "residual_uncertainty", str(self.residual_uncertainty or "").strip())

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "EvidenceObligationState":
        return cls(
            requirement_id=str(value.get("requirement_id", value.get("id", "")) or ""),
            status=str(value.get("status", "open") or "open"),  # type: ignore[arg-type]
            supporting_claim_ids=tuple(value.get("supporting_claim_ids", ()) or ()),  # type: ignore[arg-type]
            supporting_attempt_ids=tuple(value.get("supporting_attempt_ids", ()) or ()),  # type: ignore[arg-type]
            residual_uncertainty=str(value.get("residual_uncertainty", "") or ""),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _ids(values: Sequence[object]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            text
            for value in values
            if (text := str(value or "").strip())
        )
    )
