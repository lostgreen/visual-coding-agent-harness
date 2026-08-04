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
CUE_STATUSES = {"unverified", "verified", "rejected"}


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


@dataclass(frozen=True)
class InterpretationItem:
    item_id: str
    time_anchor: tuple[float, float]
    text: str
    item_kind: str = "observation"

    def __post_init__(self) -> None:
        item_id = str(self.item_id or "").strip()
        text = str(self.text or "").strip()
        if not item_id:
            raise ValueError("interpretation_item_requires_item_id")
        if not text:
            raise ValueError(f"interpretation_item_requires_text:{item_id}")
        if len(self.time_anchor) != 2:
            raise ValueError(f"interpretation_item_requires_time_anchor:{item_id}")
        start, end = sorted((float(self.time_anchor[0]), float(self.time_anchor[1])))
        object.__setattr__(self, "item_id", item_id)
        object.__setattr__(self, "time_anchor", (start, end))
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "item_kind", str(self.item_kind or "observation").strip().casefold())

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "InterpretationItem":
        return cls(
            item_id=str(value.get("item_id", value.get("id", "")) or ""),
            time_anchor=tuple(value.get("time_anchor", ()) or ()),  # type: ignore[arg-type]
            text=str(value.get("text", value.get("description", "")) or ""),
            item_kind=str(value.get("item_kind", value.get("kind", "observation")) or "observation"),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ObservationCue:
    cue_id: str
    attempt_id: str
    interpretation_id: str
    item_id: str
    source_frame_ref: str
    virtual_time: float
    cue_kind: str

    def __post_init__(self) -> None:
        for field_name in (
            "cue_id",
            "attempt_id",
            "interpretation_id",
            "item_id",
            "source_frame_ref",
            "cue_kind",
        ):
            value = str(getattr(self, field_name) or "").strip()
            if not value:
                raise ValueError(f"observation_cue_requires_{field_name}")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "virtual_time", float(self.virtual_time))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ObservationCue":
        return cls(
            cue_id=str(value.get("cue_id", value.get("id", "")) or ""),
            attempt_id=str(value.get("attempt_id", "") or ""),
            interpretation_id=str(value.get("interpretation_id", "") or ""),
            item_id=str(value.get("item_id", "") or ""),
            source_frame_ref=str(value.get("source_frame_ref", "") or ""),
            virtual_time=float(value.get("virtual_time", 0.0) or 0.0),
            cue_kind=str(value.get("cue_kind", "observation") or "observation"),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ObservationCueState:
    cue_id: str
    status: Literal["unverified", "verified", "rejected"] = "unverified"
    verification_attempt_id: str = ""
    verification_interpretation_id: str = ""
    verification_item_id: str = ""

    def __post_init__(self) -> None:
        cue_id = str(self.cue_id or "").strip()
        status = str(self.status or "unverified").strip().casefold()
        if not cue_id:
            raise ValueError("cue_state_requires_cue_id")
        if status not in CUE_STATUSES:
            raise ValueError(f"invalid_cue_status:{status}")
        object.__setattr__(self, "cue_id", cue_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "verification_attempt_id", str(self.verification_attempt_id or "").strip())
        object.__setattr__(
            self,
            "verification_interpretation_id",
            str(self.verification_interpretation_id or "").strip(),
        )
        object.__setattr__(self, "verification_item_id", str(self.verification_item_id or "").strip())

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ObservationCueState":
        return cls(
            cue_id=str(value.get("cue_id", value.get("id", "")) or ""),
            status=str(value.get("status", "unverified") or "unverified"),  # type: ignore[arg-type]
            verification_attempt_id=str(value.get("verification_attempt_id", "") or ""),
            verification_interpretation_id=str(value.get("verification_interpretation_id", "") or ""),
            verification_item_id=str(value.get("verification_item_id", "") or ""),
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
