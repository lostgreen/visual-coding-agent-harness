from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from vcah.evidence_state import (
    EvidenceObligation,
    EvidenceObligationState,
    InterpretationItem,
    ObservationCue,
    ObservationCueState,
)
from vcah.temporal_scope import TemporalScope


ClaimSource = Literal["premise", "observation", "derived", "hypothesis"]
ClaimStatus = Literal["active", "superseded", "contested", "retracted"]
ClaimConfidence = Literal["high", "medium", "low"]

_CLAIM_SOURCES = {"premise", "observation", "derived", "hypothesis"}
_CLAIM_STATUSES = {"active", "superseded", "contested", "retracted"}
_CLAIM_CONFIDENCES = {"high", "medium", "low"}
_OP_TYPES = {
    "add_claim",
    "supersede",
    "set_status",
    "link_conflict",
    "note_interval",
    "update_entity",
    "add_obligation",
    "set_obligation_status",
    "add_temporal_scope",
    "set_cue_status",
}


def prompt_digest(prompt: str) -> str:
    return hashlib.sha256(str(prompt or "").encode("utf-8")).hexdigest()[:20]


def stable_attempt_id(
    *,
    source_video_ids: Sequence[str] = (),
    frame_refs: Sequence[str] = (),
    frame_times: Sequence[float] = (),
    inspected_ranges: Sequence[Sequence[float]] = (),
    sampling_fps: float = 0.0,
    modality: str = "visual",
) -> str:
    """Return a prompt-independent identity for the inspected source material."""
    normalized_times = sorted({round(float(value), 6) for value in frame_times})
    normalized_ranges = sorted(
        {
            (round(min(float(item[0]), float(item[1])), 6), round(max(float(item[0]), float(item[1])), 6))
            for item in inspected_ranges
            if len(item) == 2
        }
    )
    material_refs = (
        [f"time:{value:.6f}" for value in normalized_times]
        if normalized_times
        else sorted({str(value).strip() for value in frame_refs if str(value).strip()})
    )
    payload = {
        "source_video_ids": sorted({str(value).strip() for value in source_video_ids if str(value).strip()}),
        "material_refs": material_refs,
        "inspected_ranges": [] if material_refs else normalized_ranges,
        "sampling_fps": round(max(0.0, float(sampling_fps or 0.0)), 6),
        "modality": str(modality or "visual").strip().casefold(),
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return f"attempt_{hashlib.sha256(encoded).hexdigest()[:24]}"


def evidence_attempt_id(record: Any) -> str:
    existing = str(getattr(record, "observation_id", "") or "").strip()
    if existing.startswith("attempt_"):
        return existing
    lineage = tuple(getattr(record, "source_lineage", ()) or ())
    source_video_ids = tuple(
        str(item.get("source_video_id", "") or "")
        for item in lineage
        if isinstance(item, Mapping)
    )
    ranges: list[Sequence[float]] = []
    start = getattr(record, "start_sec", None)
    end = getattr(record, "end_sec", None)
    if start is not None and end is not None:
        ranges.append((float(start), float(end)))
    return stable_attempt_id(
        source_video_ids=source_video_ids,
        frame_refs=tuple(getattr(record, "frame_refs", ()) or ()),
        inspected_ranges=ranges,
        sampling_fps=float(getattr(record, "sampling_fps", 0.0) or 0.0),
        modality=str(getattr(record, "modality", "visual") or "visual"),
    )


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    source: ClaimSource
    cites: tuple[str, ...] = ()
    derived_from: tuple[str, ...] = ()
    time_anchor: tuple[float, float] | None = None
    status: ClaimStatus = "active"
    superseded_by: str | None = None
    conflicts_with: tuple[str, ...] = ()
    confidence: ClaimConfidence = "medium"
    entity_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    interpretation_id: str = ""
    interpretation_item_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", str(self.claim_id or "").strip())
        object.__setattr__(self, "text", str(self.text or "").strip())
        source = str(self.source or "hypothesis").strip().casefold()
        if source not in _CLAIM_SOURCES:
            raise ValueError(f"invalid_claim_source:{source}")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "cites", _ids(self.cites))
        object.__setattr__(self, "derived_from", _ids(self.derived_from))
        object.__setattr__(self, "time_anchor", _time_range(self.time_anchor))
        status = str(self.status or "active").strip().casefold()
        if status not in _CLAIM_STATUSES:
            raise ValueError(f"invalid_claim_status:{status}")
        object.__setattr__(self, "status", status)
        superseded_by = str(self.superseded_by or "").strip() or None
        object.__setattr__(self, "superseded_by", superseded_by)
        object.__setattr__(self, "conflicts_with", _ids(self.conflicts_with))
        confidence = str(self.confidence or "medium").strip().casefold()
        if confidence not in _CLAIM_CONFIDENCES:
            raise ValueError(f"invalid_claim_confidence:{confidence}")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "entity_ids", _ids(self.entity_ids))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        object.__setattr__(self, "interpretation_id", str(self.interpretation_id or "").strip())
        object.__setattr__(
            self,
            "interpretation_item_id",
            str(self.interpretation_item_id or "").strip(),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Claim":
        return cls(
            claim_id=str(value.get("claim_id", value.get("id", "")) or ""),
            text=str(value.get("text", value.get("claim", "")) or ""),
            source=str(value.get("source", "hypothesis") or "hypothesis"),  # type: ignore[arg-type]
            cites=tuple(value.get("cites", ()) or ()),
            derived_from=tuple(value.get("derived_from", ()) or ()),
            time_anchor=value.get("time_anchor"),
            status=str(value.get("status", "active") or "active"),  # type: ignore[arg-type]
            superseded_by=value.get("superseded_by"),
            conflicts_with=tuple(value.get("conflicts_with", ()) or ()),
            confidence=str(value.get("confidence", "medium") or "medium"),  # type: ignore[arg-type]
            entity_ids=tuple(value.get("entity_ids", ()) or ()),
            metadata=dict(value.get("metadata", {}) or {}),
            interpretation_id=str(
                value.get("interpretation_id", dict(value.get("metadata", {}) or {}).get("interpretation_id", ""))
                or ""
            ),
            interpretation_item_id=str(
                value.get(
                    "interpretation_item_id",
                    value.get("item_id", dict(value.get("metadata", {}) or {}).get("item_id", "")),
                )
                or ""
            ),
        )


@dataclass(frozen=True)
class Entity:
    entity_id: str
    description: str = ""
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_id", str(self.entity_id or "").strip())
        object.__setattr__(self, "description", str(self.description or "").strip())
        object.__setattr__(self, "aliases", _ids(self.aliases))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Entity":
        return cls(
            entity_id=str(value.get("entity_id", value.get("id", "")) or ""),
            description=str(value.get("description", "") or ""),
            aliases=tuple(value.get("aliases", ()) or ()),
        )


@dataclass(frozen=True)
class IntervalNote:
    time_range: tuple[float, float]
    label: str
    claim_ids: tuple[str, ...] = ()
    role: str = "supporting"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = _time_range(self.time_range)
        if normalized is None:
            raise ValueError("IntervalNote requires a two-value time_range")
        object.__setattr__(self, "time_range", normalized)
        object.__setattr__(self, "label", str(self.label or "").strip())
        if not self.label:
            raise ValueError("interval_note_requires_label")
        object.__setattr__(self, "claim_ids", _ids(self.claim_ids))
        role = str(self.role or "supporting").strip().casefold()
        if role not in {"candidate", "supporting", "negative"}:
            raise ValueError(f"invalid_interval_role:{role}")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "IntervalNote":
        time_range = value.get("time_range")
        if time_range is None and value.get("start_sec") is not None and value.get("end_sec") is not None:
            time_range = (value["start_sec"], value["end_sec"])
        return cls(
            time_range=tuple(time_range or ()),  # type: ignore[arg-type]
            label=str(value.get("label", "") or ""),
            claim_ids=tuple(value.get("claim_ids", ()) or ()),
            role=str(value.get("role", value.get("interval_role", "supporting")) or "supporting"),
            metadata=dict(value.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class WorkspaceApplyResult:
    accepted: bool
    revision: int
    applied_count: int = 0
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnswerValidation:
    passed: bool
    reason: str
    supporting_claim_ids: tuple[str, ...] = ()
    cited_attempt_ids: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkingDocument:
    claims: dict[str, Claim] = field(default_factory=dict)
    entities: dict[str, Entity] = field(default_factory=dict)
    timeline: list[IntervalNote] = field(default_factory=list)
    obligations: dict[str, EvidenceObligation] = field(default_factory=dict)
    obligation_states: dict[str, EvidenceObligationState] = field(default_factory=dict)
    temporal_scopes: dict[str, TemporalScope] = field(default_factory=dict)
    cue_states: dict[str, ObservationCueState] = field(default_factory=dict)
    revision: int = 0
    active_claim_limit: int = 60

    @classmethod
    def with_question_premise(cls, question: str, *, active_claim_limit: int = 60) -> "WorkingDocument":
        premise = Claim(
            claim_id="premise:question",
            text=str(question or "").strip(),
            source="premise",
            confidence="high",
        )
        return cls(
            claims={premise.claim_id: premise} if premise.text else {},
            active_claim_limit=max(1, int(active_claim_limit)),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WorkingDocument":
        raw_claims = value.get("claims", {}) or {}
        claim_values = raw_claims.values() if isinstance(raw_claims, Mapping) else raw_claims
        claims = {
            claim.claim_id: claim
            for item in claim_values
            if isinstance(item, Mapping)
            and (claim := Claim.from_mapping(item)).claim_id
        }
        raw_entities = value.get("entities", {}) or {}
        entity_values = raw_entities.values() if isinstance(raw_entities, Mapping) else raw_entities
        entities = {
            entity.entity_id: entity
            for item in entity_values
            if isinstance(item, Mapping)
            and (entity := Entity.from_mapping(item)).entity_id
        }
        return cls(
            claims=claims,
            entities=entities,
            timeline=[
                IntervalNote.from_mapping(item)
                for item in tuple(value.get("timeline", ()) or ())
                if isinstance(item, Mapping)
            ],
            obligations={
                obligation.requirement_id: obligation
                for item in _mapping_values(value.get("obligations"))
                if isinstance(item, Mapping)
                and (obligation := EvidenceObligation.from_mapping(item)).requirement_id
            },
            obligation_states={
                state.requirement_id: state
                for item in _mapping_values(value.get("obligation_states"))
                if isinstance(item, Mapping)
                and (state := EvidenceObligationState.from_mapping(item)).requirement_id
            },
            temporal_scopes={
                scope.scope_id: scope
                for item in _mapping_values(value.get("temporal_scopes"))
                if isinstance(item, Mapping)
                and (scope := TemporalScope.from_mapping(item)).scope_id
            },
            cue_states={
                state.cue_id: state
                for item in _mapping_values(value.get("cue_states"))
                if isinstance(item, Mapping)
                and (state := ObservationCueState.from_mapping(item)).cue_id
            },
            revision=max(0, int(value.get("revision", 0) or 0)),
            active_claim_limit=max(1, int(value.get("active_claim_limit", 60) or 60)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "WorkingDocumentV5",
            "revision": self.revision,
            "active_claim_limit": self.active_claim_limit,
            "claims": {claim_id: asdict(claim) for claim_id, claim in sorted(self.claims.items())},
            "entities": {entity_id: asdict(entity) for entity_id, entity in sorted(self.entities.items())},
            "timeline": [asdict(item) for item in self.timeline],
            "obligations": {
                requirement_id: obligation.to_dict()
                for requirement_id, obligation in sorted(self.obligations.items())
            },
            "obligation_states": {
                requirement_id: state.to_dict()
                for requirement_id, state in sorted(self.obligation_states.items())
            },
            "temporal_scopes": {
                scope_id: scope.to_dict()
                for scope_id, scope in sorted(self.temporal_scopes.items())
            },
            "cue_states": {
                cue_id: state.to_dict()
                for cue_id, state in sorted(self.cue_states.items())
            },
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(f"{destination.suffix}.tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

    def apply_ops(
        self,
        operations: Sequence[Mapping[str, Any]],
        *,
        observation_ids: Sequence[str],
        observation_rows: Sequence[Mapping[str, Any]] = (),
        require_item_provenance: bool = False,
    ) -> WorkspaceApplyResult:
        ops = tuple(dict(item) for item in operations if isinstance(item, Mapping))
        if not ops:
            return WorkspaceApplyResult(True, self.revision, 0, ())
        staged = WorkingDocument.from_mapping(self.to_dict())
        errors: list[str] = []
        for index, operation in enumerate(ops, start=1):
            try:
                staged._apply_one(operation)
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"op[{index}]: {exc}")
                break
        if not errors:
            errors.extend(
                staged.validate(
                    observation_ids=observation_ids,
                    observation_rows=observation_rows,
                    require_item_provenance=require_item_provenance,
                )
            )
        if errors:
            return WorkspaceApplyResult(False, self.revision, 0, tuple(errors))
        self.claims = staged.claims
        self.entities = staged.entities
        self.timeline = staged.timeline
        self.obligations = staged.obligations
        self.obligation_states = staged.obligation_states
        self.temporal_scopes = staged.temporal_scopes
        self.cue_states = staged.cue_states
        self.revision += 1
        return WorkspaceApplyResult(True, self.revision, len(ops), ())

    def validate(
        self,
        *,
        observation_ids: Sequence[str],
        observation_rows: Sequence[Mapping[str, Any]] = (),
        require_item_provenance: bool = False,
    ) -> tuple[str, ...]:
        known_observations = {str(item) for item in observation_ids if str(item)}
        item_index = _observation_item_index(observation_rows)
        cue_index = _observation_cue_index(observation_rows)
        errors: list[str] = []
        active_count = sum(claim.status in {"active", "contested"} for claim in self.claims.values())
        if active_count > self.active_claim_limit:
            errors.append(f"active_claim_limit_exceeded:{active_count}>{self.active_claim_limit}")
        for claim_id, claim in self.claims.items():
            if not claim_id:
                errors.append("claim_id_missing")
                continue
            if not claim.text:
                errors.append(f"claim_text_missing:{claim_id}")
            missing_cites = tuple(cite for cite in claim.cites if cite not in known_observations)
            if missing_cites:
                errors.append(f"claim_cites_unknown_attempt:{claim_id}:{','.join(missing_cites)}")
            if claim.source == "observation" and not claim.cites:
                errors.append(f"observation_claim_requires_cites:{claim_id}")
            if claim.source == "observation" and require_item_provenance:
                errors.extend(_observation_claim_binding_errors(claim, item_index))
            if claim.source == "derived" and not claim.derived_from:
                errors.append(f"derived_claim_requires_parents:{claim_id}")
            missing_parents = tuple(parent for parent in claim.derived_from if parent not in self.claims)
            if missing_parents:
                errors.append(f"claim_parent_missing:{claim_id}:{','.join(missing_parents)}")
            missing_conflicts = tuple(other for other in claim.conflicts_with if other not in self.claims)
            if missing_conflicts:
                errors.append(f"claim_conflict_missing:{claim_id}:{','.join(missing_conflicts)}")
            missing_entities = tuple(entity_id for entity_id in claim.entity_ids if entity_id not in self.entities)
            if missing_entities:
                errors.append(f"claim_entity_missing:{claim_id}:{','.join(missing_entities)}")
            if claim.status == "superseded":
                if not claim.superseded_by:
                    errors.append(f"superseded_claim_requires_successor:{claim_id}")
                elif claim.superseded_by not in self.claims:
                    errors.append(f"superseding_claim_missing:{claim_id}:{claim.superseded_by}")
            elif claim.superseded_by:
                errors.append(f"non_superseded_claim_has_successor:{claim_id}")
        errors.extend(self._cycle_errors())
        for index, note in enumerate(self.timeline):
            missing = tuple(claim_id for claim_id in note.claim_ids if claim_id not in self.claims)
            if missing:
                errors.append(f"timeline_claim_missing:{index}:{','.join(missing)}")
        for requirement_id, obligation in self.obligations.items():
            missing_dependencies = tuple(
                dependency
                for dependency in obligation.depends_on
                if dependency not in self.obligations
            )
            if missing_dependencies:
                errors.append(
                    f"obligation_dependency_missing:{requirement_id}:{','.join(missing_dependencies)}"
                )
            state = self.obligation_states.get(requirement_id)
            if state is None:
                errors.append(f"obligation_state_missing:{requirement_id}")
                continue
            missing_claims = tuple(
                claim_id
                for claim_id in state.supporting_claim_ids
                if claim_id not in self.claims
            )
            if missing_claims:
                errors.append(
                    f"obligation_supporting_claim_missing:{requirement_id}:{','.join(missing_claims)}"
                )
            missing_attempts = tuple(
                attempt_id
                for attempt_id in state.supporting_attempt_ids
                if attempt_id not in known_observations
            )
            if missing_attempts:
                errors.append(
                    f"obligation_supporting_attempt_missing:{requirement_id}:{','.join(missing_attempts)}"
                )
            if state.status == "satisfied" and not state.supporting_claim_ids:
                errors.append(f"satisfied_obligation_requires_claim:{requirement_id}")
            if state.status == "satisfied" and not state.supporting_attempt_ids:
                errors.append(f"satisfied_obligation_requires_attempt:{requirement_id}")
            if state.status == "satisfied":
                for dependency in obligation.depends_on:
                    dependency_state = self.obligation_states.get(dependency)
                    if dependency_state is None or dependency_state.status != "satisfied":
                        errors.append(
                            f"obligation_dependency_unsatisfied:{requirement_id}:{dependency}"
                        )
                        continue
                    dependency_claims = set(dependency_state.supporting_claim_ids)
                    if dependency_claims and not any(
                        dependency_claims.intersection(
                            self._claim_lineage_ids(claim_id)
                        )
                        for claim_id in state.supporting_claim_ids
                    ):
                        errors.append(
                            f"obligation_dependency_lineage_missing:{requirement_id}:{dependency}"
                        )
            if state.status == "unresolved" and not state.residual_uncertainty:
                errors.append(f"unresolved_obligation_requires_uncertainty:{requirement_id}")
        orphan_states = tuple(
            requirement_id
            for requirement_id in self.obligation_states
            if requirement_id not in self.obligations
        )
        if orphan_states:
            errors.append(f"obligation_definition_missing:{','.join(orphan_states)}")
        errors.extend(self._obligation_cycle_errors())
        for scope_id, scope in self.temporal_scopes.items():
            if scope.anchor_requirement_id not in self.obligations:
                errors.append(
                    f"temporal_scope_anchor_obligation_missing:{scope_id}:{scope.anchor_requirement_id}"
                )
            if scope.target_requirement_id not in self.obligations:
                errors.append(
                    f"temporal_scope_target_obligation_missing:{scope_id}:{scope.target_requirement_id}"
                )
        for cue_id, state in self.cue_states.items():
            if cue_id not in cue_index:
                errors.append(f"cue_definition_missing:{cue_id}")
                continue
            if state.status in {"verified", "rejected"}:
                if not (
                    state.verification_attempt_id
                    and state.verification_interpretation_id
                    and state.verification_item_id
                ):
                    errors.append(f"cue_verification_item_missing:{cue_id}")
                    continue
                item = item_index.get(
                    (
                        state.verification_attempt_id,
                        state.verification_interpretation_id,
                        state.verification_item_id,
                    )
                )
                if item is None:
                    errors.append(f"cue_verification_item_unknown:{cue_id}")
                elif not _anchors_overlap(
                    tuple(item["time_anchor"]),
                    (float(cue_index[cue_id]["virtual_time"]),) * 2,
                ):
                    errors.append(f"cue_verification_time_mismatch:{cue_id}")
        return tuple(dict.fromkeys(errors))

    def obligation_summary(self) -> dict[str, Any]:
        answer_bearing_ids = tuple(
            requirement_id
            for requirement_id, obligation in self.obligations.items()
            if obligation.answer_bearing
        )
        satisfied = tuple(
            requirement_id
            for requirement_id in answer_bearing_ids
            if self.obligation_states.get(requirement_id)
            and self.obligation_states[requirement_id].status == "satisfied"
        )
        unresolved = tuple(
            requirement_id
            for requirement_id in answer_bearing_ids
            if self.obligation_states.get(requirement_id)
            and self.obligation_states[requirement_id].status == "unresolved"
        )
        closed = set((*satisfied, *unresolved))
        open_ids = tuple(
            requirement_id
            for requirement_id in answer_bearing_ids
            if requirement_id not in closed
        )
        return {
            "answer_bearing_obligation_count": len(answer_bearing_ids),
            "satisfied_obligation_count": len(satisfied),
            "open_obligation_count_at_answer": len(open_ids),
            "unresolved_obligation_count_at_answer": len(unresolved),
            "obligation_coverage_rate": (
                len(satisfied) / len(answer_bearing_ids) if answer_bearing_ids else 0.0
            ),
            "answer_bearing_requirement_ids": list(answer_bearing_ids),
            "satisfied_requirement_ids": list(satisfied),
            "open_requirement_ids": list(open_ids),
            "unresolved_requirement_ids": list(unresolved),
        }

    def provenance_summary(
        self,
        observation_rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        item_index = _observation_item_index(observation_rows)
        claims = tuple(
            claim
            for claim in self.claims.values()
            if claim.source == "observation"
            and claim.status in {"active", "contested"}
        )
        bound = tuple(
            claim
            for claim in claims
            if not _observation_claim_binding_errors(claim, item_index)
        )
        known_item_ids = {key[2] for key in item_index}
        dangling = sum(
            bool(claim.interpretation_item_id)
            and claim.interpretation_item_id not in known_item_ids
            for claim in claims
        )
        return {
            "observation_claim_count": len(claims),
            "item_bound_observation_claim_count": len(bound),
            "observation_claim_item_binding_rate": (
                len(bound) / len(claims) if claims else 0.0
            ),
            "dangling_interpretation_item_count": dangling,
            "dangling_interpretation_item_rate": (
                dangling / len(claims) if claims else 0.0
            ),
        }

    def cue_summary(
        self,
        observation_rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        cue_ids = tuple(_observation_cue_index(observation_rows))
        verified = tuple(
            cue_id
            for cue_id in cue_ids
            if self.cue_states.get(cue_id)
            and self.cue_states[cue_id].status == "verified"
        )
        rejected = tuple(
            cue_id
            for cue_id in cue_ids
            if self.cue_states.get(cue_id)
            and self.cue_states[cue_id].status == "rejected"
        )
        unverified = tuple(
            cue_id for cue_id in cue_ids if cue_id not in {*verified, *rejected}
        )
        decided = len(verified) + len(rejected)
        return {
            "observation_cue_count": len(cue_ids),
            "verified_cue_count": len(verified),
            "rejected_cue_count": len(rejected),
            "unverified_cue_count": len(unverified),
            "cue_verification_rate": len(verified) / decided if decided else 0.0,
            "cue_rejection_rate": len(rejected) / decided if decided else 0.0,
            "verified_cue_ids": list(verified),
            "rejected_cue_ids": list(rejected),
        }

    def validate_answer(
        self,
        supporting_claim_ids: Sequence[str],
        *,
        observation_ids: Sequence[str],
        supporting_observation_ids: Sequence[str] | None = None,
        require_obligation_coverage: bool = False,
        observation_rows: Sequence[Mapping[str, Any]] = (),
        require_item_provenance: bool = False,
    ) -> AnswerValidation:
        support = _ids(supporting_claim_ids)
        if not support:
            return AnswerValidation(False, "supporting_claims_missing")
        document_errors = self.validate(
            observation_ids=observation_ids,
            observation_rows=observation_rows,
            require_item_provenance=require_item_provenance,
        )
        missing = tuple(claim_id for claim_id in support if claim_id not in self.claims)
        inactive = tuple(
            claim_id
            for claim_id in support
            if claim_id in self.claims and self.claims[claim_id].status in {"superseded", "retracted"}
        )
        hypothetical = tuple(
            claim_id
            for claim_id in support
            if claim_id in self.claims and self.claims[claim_id].source == "hypothesis"
        )
        uncertain = tuple(
            claim_id
            for claim_id in support
            if claim_id in self.claims
            and (
                self.claims[claim_id].status == "contested"
                or self.claims[claim_id].confidence == "low"
            )
        )
        conflicted = tuple(
            claim_id
            for claim_id in support
            if claim_id in self.claims
            and any(
                other in self.claims
                and self.claims[other].status in {"active", "contested"}
                for other in self.claims[claim_id].conflicts_with
            )
        )
        errors = [*document_errors]
        if missing:
            errors.append(f"supporting_claim_missing:{','.join(missing)}")
        if inactive:
            errors.append(f"supporting_claim_inactive:{','.join(inactive)}")
        if hypothetical:
            errors.append(f"supporting_claim_hypothetical:{','.join(hypothetical)}")
        if uncertain:
            errors.append(f"supporting_claim_uncertain:{','.join(uncertain)}")
        if conflicted:
            errors.append(f"supporting_claim_conflicted:{','.join(conflicted)}")
        cited_attempts = tuple(
            dict.fromkeys(
                cite
                for claim_id in support
                if claim_id in self.claims
                for cite in self._claim_attempts(claim_id)
            )
        )
        if not cited_attempts:
            errors.append("supporting_claims_require_observation")
        eligible_attempts = (
            {str(item) for item in supporting_observation_ids}
            if supporting_observation_ids is not None
            else set(observation_ids)
        )
        ineligible_attempts = tuple(
            attempt_id for attempt_id in cited_attempts if attempt_id not in eligible_attempts
        )
        if ineligible_attempts:
            errors.append(
                f"supporting_claims_cite_candidate_or_negative:{','.join(ineligible_attempts)}"
            )
        if require_obligation_coverage:
            answer_bearing = tuple(
                obligation
                for obligation in self.obligations.values()
                if obligation.answer_bearing
            )
            if not answer_bearing:
                errors.append("answer_bearing_obligations_missing")
            for obligation in answer_bearing:
                state = self.obligation_states.get(obligation.requirement_id)
                if state is None:
                    errors.append(
                        f"answer_bearing_obligation_state_missing:{obligation.requirement_id}"
                    )
                    continue
                if state.status == "satisfied":
                    obligation_claims = set(state.supporting_claim_ids)
                    closed_by_answer = bool(obligation_claims.intersection(support)) or any(
                        obligation_claims.intersection(self._claim_lineage_ids(claim_id))
                        for claim_id in support
                    )
                    if not closed_by_answer:
                        errors.append(
                            f"answer_support_does_not_close_obligation:{obligation.requirement_id}"
                        )
                elif state.status == "unresolved":
                    if not state.residual_uncertainty:
                        errors.append(
                            f"unresolved_obligation_requires_uncertainty:{obligation.requirement_id}"
                        )
                else:
                    errors.append(
                        f"open_answer_bearing_obligation:{obligation.requirement_id}:{state.status}"
                    )
        if errors:
            return AnswerValidation(False, errors[0].split(":", 1)[0], support, cited_attempts, tuple(errors))
        return AnswerValidation(True, "reference_integrity_verified", support, cited_attempts, ())

    def _apply_one(self, operation: Mapping[str, Any]) -> None:
        op_type = str(operation.get("op", operation.get("type", "")) or "").strip().casefold()
        if op_type not in _OP_TYPES:
            raise ValueError(f"unsupported_workspace_op:{op_type or 'missing'}")
        if op_type == "add_claim":
            payload = operation.get("claim") if isinstance(operation.get("claim"), Mapping) else operation
            claim = Claim.from_mapping(payload)
            if not claim.claim_id:
                raise ValueError("add_claim requires claim_id")
            if claim.source == "premise":
                raise ValueError("premise_source_is_framework_managed")
            if claim.claim_id in self.claims:
                raise ValueError(f"claim_already_exists:{claim.claim_id}")
            self.claims[claim.claim_id] = claim
            return
        if op_type == "supersede":
            claim_id = _required_id(operation, "claim_id")
            successor_id = _required_id(operation, "superseded_by", "successor_id")
            self._require_editable_claim(claim_id)
            self._require_claim(successor_id)
            self.claims[claim_id] = replace(
                self.claims[claim_id],
                status="superseded",
                superseded_by=successor_id,
            )
            return
        if op_type == "set_status":
            claim_id = _required_id(operation, "claim_id")
            self._require_editable_claim(claim_id)
            status = str(operation.get("status", "") or "").strip().casefold()
            if status not in {"active", "contested", "retracted"}:
                raise ValueError(f"invalid_claim_status:{status or 'missing'}")
            self.claims[claim_id] = replace(
                self.claims[claim_id],
                status=status,  # type: ignore[arg-type]
                superseded_by=None,
            )
            return
        if op_type == "link_conflict":
            left_id = _required_id(operation, "claim_id", "left_claim_id")
            right_id = _required_id(operation, "other_claim_id", "right_claim_id", "conflicts_with")
            self._require_claim(left_id)
            self._require_claim(right_id)
            if left_id == right_id:
                raise ValueError("claim_cannot_conflict_with_itself")
            left = self.claims[left_id]
            right = self.claims[right_id]
            self.claims[left_id] = replace(left, conflicts_with=_ids((*left.conflicts_with, right_id)))
            self.claims[right_id] = replace(right, conflicts_with=_ids((*right.conflicts_with, left_id)))
            return
        if op_type == "note_interval":
            note_payload = operation.get("note") if isinstance(operation.get("note"), Mapping) else operation
            self.timeline.append(IntervalNote.from_mapping(note_payload))
            return
        if op_type == "add_obligation":
            payload = (
                operation.get("obligation")
                if isinstance(operation.get("obligation"), Mapping)
                else operation
            )
            obligation = EvidenceObligation.from_mapping(payload)
            if obligation.requirement_id in self.obligations:
                raise ValueError(
                    f"obligation_already_exists:{obligation.requirement_id}"
                )
            self.obligations[obligation.requirement_id] = obligation
            self.obligation_states[obligation.requirement_id] = EvidenceObligationState(
                requirement_id=obligation.requirement_id
            )
            return
        if op_type == "set_obligation_status":
            requirement_id = _required_id(operation, "requirement_id", "obligation_id")
            if requirement_id not in self.obligations:
                raise ValueError(f"obligation_missing:{requirement_id}")
            self.obligation_states[requirement_id] = EvidenceObligationState(
                requirement_id=requirement_id,
                status=str(operation.get("status", "open") or "open"),  # type: ignore[arg-type]
                supporting_claim_ids=tuple(operation.get("supporting_claim_ids", ()) or ()),
                supporting_attempt_ids=tuple(operation.get("supporting_attempt_ids", ()) or ()),
                residual_uncertainty=str(operation.get("residual_uncertainty", "") or ""),
            )
            return
        if op_type == "add_temporal_scope":
            payload = (
                operation.get("temporal_scope")
                if isinstance(operation.get("temporal_scope"), Mapping)
                else operation
            )
            scope = TemporalScope.from_mapping(payload)
            if scope.scope_id in self.temporal_scopes:
                raise ValueError(f"temporal_scope_already_exists:{scope.scope_id}")
            self.temporal_scopes[scope.scope_id] = scope
            return
        if op_type == "set_cue_status":
            cue_id = _required_id(operation, "cue_id")
            self.cue_states[cue_id] = ObservationCueState(
                cue_id=cue_id,
                status=str(operation.get("status", "unverified") or "unverified"),  # type: ignore[arg-type]
                verification_attempt_id=str(
                    operation.get("verification_attempt_id", "") or ""
                ),
                verification_interpretation_id=str(
                    operation.get("verification_interpretation_id", "") or ""
                ),
                verification_item_id=str(
                    operation.get("verification_item_id", "") or ""
                ),
            )
            return
        entity_payload = operation.get("entity") if isinstance(operation.get("entity"), Mapping) else operation
        entity = Entity.from_mapping(entity_payload)
        if not entity.entity_id:
            raise ValueError("update_entity requires entity_id")
        existing = self.entities.get(entity.entity_id)
        self.entities[entity.entity_id] = Entity(
            entity_id=entity.entity_id,
            description=entity.description or (existing.description if existing else ""),
            aliases=_ids((*(existing.aliases if existing else ()), *entity.aliases)),
        )

    def _require_claim(self, claim_id: str) -> None:
        if claim_id not in self.claims:
            raise ValueError(f"claim_missing:{claim_id}")

    def _require_editable_claim(self, claim_id: str) -> None:
        self._require_claim(claim_id)
        if self.claims[claim_id].source == "premise":
            raise ValueError("premise_is_framework_managed")

    def _cycle_errors(self) -> tuple[str, ...]:
        errors: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(claim_id: str) -> None:
            if claim_id in visiting:
                errors.append(f"derived_claim_cycle:{claim_id}")
                return
            if claim_id in visited or claim_id not in self.claims:
                return
            visiting.add(claim_id)
            for parent in self.claims[claim_id].derived_from:
                visit(parent)
            visiting.remove(claim_id)
            visited.add(claim_id)

        for claim_id in self.claims:
            visit(claim_id)
        return tuple(errors)

    def _obligation_cycle_errors(self) -> tuple[str, ...]:
        errors: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(requirement_id: str) -> None:
            if requirement_id in visiting:
                errors.append(f"obligation_dependency_cycle:{requirement_id}")
                return
            if requirement_id in visited or requirement_id not in self.obligations:
                return
            visiting.add(requirement_id)
            for dependency in self.obligations[requirement_id].depends_on:
                visit(dependency)
            visiting.remove(requirement_id)
            visited.add(requirement_id)

        for requirement_id in self.obligations:
            visit(requirement_id)
        return tuple(errors)

    def _claim_attempts(self, claim_id: str, seen: set[str] | None = None) -> tuple[str, ...]:
        visited = set(seen or ())
        if claim_id in visited or claim_id not in self.claims:
            return ()
        visited.add(claim_id)
        claim = self.claims[claim_id]
        return tuple(
            dict.fromkeys(
                (
                    *claim.cites,
                    *(
                        cite
                        for parent in claim.derived_from
                        for cite in self._claim_attempts(parent, visited)
                    ),
                )
            )
        )

    def _claim_lineage_ids(self, claim_id: str, seen: set[str] | None = None) -> tuple[str, ...]:
        visited = set(seen or ())
        if claim_id in visited or claim_id not in self.claims:
            return ()
        visited.add(claim_id)
        claim = self.claims[claim_id]
        return tuple(
            dict.fromkeys(
                (
                    *claim.derived_from,
                    *(
                        ancestor
                        for parent in claim.derived_from
                        for ancestor in self._claim_lineage_ids(parent, visited)
                    ),
                )
            )
        )


class ObservationLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=False)
        self.rows: list[dict[str, Any]] = []

    @property
    def attempt_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(str(row["attempt_id"]) for row in self.rows))

    @property
    def interpretation_item_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                str(item.get("item_id", "") or "")
                for row in self.rows
                for item in tuple(row.get("interpretation_items", ()) or ())
                if isinstance(item, Mapping) and str(item.get("item_id", "") or "")
            )
        )

    @property
    def cue_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                str(cue.get("cue_id", "") or "")
                for row in self.rows
                for cue in tuple(row.get("observation_cues", ()) or ())
                if isinstance(cue, Mapping) and str(cue.get("cue_id", "") or "")
            )
        )

    def append_attempt(
        self,
        attempt: Any,
        *,
        round_id: int | str,
        source_lineage: Sequence[Mapping[str, Any]] = (),
    ) -> Mapping[str, Any]:
        source_video_ids = tuple(
            dict.fromkeys(
                (
                    *tuple(getattr(attempt, "source_video_ids", ()) or ()),
                    *(
                        str(item.get("source_video_id", "") or "")
                        for item in source_lineage
                        if isinstance(item, Mapping)
                    ),
                )
            )
        )
        inspected_ranges = tuple(getattr(attempt, "inspected_ranges", ()) or ())
        frame_refs = tuple(getattr(attempt, "frame_refs", ()) or ())
        frame_times = tuple(getattr(attempt, "attached_frame_times", ()) or ())
        sampling_fps = float(getattr(attempt, "sampling_fps", 0.0) or 0.0)
        modality = str(
            getattr(attempt, "modality", "")
            or dict(getattr(attempt, "sampling_config", {}) or {}).get("modality", "visual")
            or "visual"
        )
        supplied_attempt_id = str(getattr(attempt, "attempt_id", "") or "").strip()
        attempt_id = stable_attempt_id(
            source_video_ids=source_video_ids,
            frame_refs=frame_refs,
            frame_times=frame_times,
            inspected_ranges=inspected_ranges,
            sampling_fps=sampling_fps,
            modality=modality,
        )
        if supplied_attempt_id and supplied_attempt_id != attempt_id:
            raise ValueError(
                f"attempt_id does not match inspected material: {supplied_attempt_id} != {attempt_id}"
            )
        raw_output = str(getattr(attempt, "raw_output", "") or "")
        digest = str(getattr(attempt, "prompt_digest", "") or "").strip()
        interpretation_material = json.dumps(
            [attempt_id, digest, raw_output, str(round_id), len(self.rows)],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        interpretation_id = (
            "interpretation_"
            + hashlib.sha256(interpretation_material.encode("utf-8")).hexdigest()[:20]
        )
        interpretation_items = tuple(
            item
            if isinstance(item, InterpretationItem)
            else InterpretationItem.from_mapping(item)
            for item in tuple(getattr(attempt, "interpretation_items", ()) or ())
            if isinstance(item, (InterpretationItem, Mapping))
        )
        observation_cues = _bound_observation_cues(
            attempt_id=attempt_id,
            interpretation_id=interpretation_id,
            items=interpretation_items,
            frame_refs=frame_refs,
            frame_times=frame_times,
        )
        row = {
            "schema_version": "ObservationInterpretationV1",
            "attempt_id": attempt_id,
            "interpretation_id": interpretation_id,
            "task_id": str(getattr(attempt, "task_id", "") or ""),
            "round_id": str(round_id),
            "requested_range": list(getattr(attempt, "requested_range", ()) or ()),
            "inspected_ranges": [list(item) for item in inspected_ranges],
            "frame_refs": list(frame_refs),
            "frame_times": [float(item) for item in frame_times],
            "sampling_fps": sampling_fps,
            "sampling_config": dict(getattr(attempt, "sampling_config", {}) or {}),
            "modality": modality.strip().casefold(),
            "evidence_role": str(getattr(attempt, "evidence_role", "unclassified") or "unclassified"),
            "interpretation_purpose": str(
                getattr(attempt, "interpretation_purpose", "primary") or "primary"
            ),
            "prompt_digest": digest,
            "raw_output": raw_output,
            "parse_status": str(getattr(attempt, "parse_status", "unknown") or "unknown"),
            "execution_status": str(
                getattr(attempt, "execution_status", "completed") or "completed"
            ),
            "source_video_ids": [item for item in source_video_ids if item],
            "source_lineage": [dict(item) for item in source_lineage],
            "interpretation_items": [item.to_dict() for item in interpretation_items],
            "observation_cues": [cue.to_dict() for cue in observation_cues],
        }
        self.rows.append(row)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return row

    def catalog(self) -> tuple[dict[str, Any], ...]:
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in self.rows:
            grouped.setdefault(str(row["attempt_id"]), []).append(row)
        catalog = []
        for attempt_id, rows in grouped.items():
            first = rows[0]
            raw_outputs = tuple(str(row.get("raw_output", "") or "") for row in rows)
            catalog.append(
                {
                    "attempt_id": attempt_id,
                    "time_ranges": first.get("inspected_ranges") or [first.get("requested_range")],
                    "sampling_fps": first.get("sampling_fps", 0.0),
                    "modality": first.get("modality", ""),
                    "evidence_role": first.get("evidence_role", "unclassified"),
                    "interpretation_purposes": [
                        str(row.get("interpretation_purpose", "primary") or "primary")
                        for row in rows
                    ],
                    "frame_count": len(tuple(first.get("frame_refs", ()) or ())),
                    "interpretation_count": len(rows),
                    "interpretation_item_count": sum(
                        len(tuple(row.get("interpretation_items", ()) or ()))
                        for row in rows
                    ),
                    "observation_cue_count": sum(
                        len(tuple(row.get("observation_cues", ()) or ()))
                        for row in rows
                    ),
                    "interpretation_item_previews": [
                        {
                            "interpretation_id": row.get("interpretation_id", ""),
                            "item_id": item.get("item_id", ""),
                            "time_anchor": item.get("time_anchor", ()),
                            "item_kind": item.get("item_kind", ""),
                            "text": _compact_text(str(item.get("text", "") or ""), 180),
                        }
                        for row in rows
                        for item in tuple(row.get("interpretation_items", ()) or ())[:8]
                        if isinstance(item, Mapping)
                    ][:12],
                    "observation_cues": [
                        dict(cue)
                        for row in rows
                        for cue in tuple(row.get("observation_cues", ()) or ())
                        if isinstance(cue, Mapping)
                    ][:12],
                    "interpretation_previews": [_compact_text(text, 320) for text in raw_outputs if text][:3],
                    "source_video_ids": list(first.get("source_video_ids", ()) or ()),
                }
            )
        return tuple(catalog)

    def read(
        self,
        *,
        attempt_ids: Sequence[str] = (),
        time_range: Sequence[float] | None = None,
        max_entries: int = 12,
    ) -> tuple[Mapping[str, Any], ...]:
        requested = {str(item) for item in attempt_ids if str(item)}
        normalized_range = _time_range(time_range)
        rows = []
        for row in self.rows:
            if requested and str(row.get("attempt_id", "")) not in requested:
                continue
            if normalized_range is not None and not _row_overlaps(row, normalized_range):
                continue
            rows.append(dict(row))
        return tuple(rows[-max(1, int(max_entries)):])

    def coverage_ledger(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "attempt_id": row["attempt_id"],
                "time_ranges": row.get("inspected_ranges") or [row.get("requested_range")],
                "sampling_fps": row.get("sampling_fps", 0.0),
                "modality": row.get("modality", ""),
                "evidence_role": row.get("evidence_role", "unclassified"),
                "execution_status": row.get("execution_status", ""),
            }
            for row in self.catalog_source_rows()
        )

    def catalog_source_rows(self) -> tuple[Mapping[str, Any], ...]:
        by_attempt: dict[str, Mapping[str, Any]] = {}
        for row in self.rows:
            by_attempt.setdefault(str(row["attempt_id"]), row)
        return tuple(by_attempt.values())


def render_working_view(
    document: WorkingDocument,
    observations: ObservationLog,
    *,
    requested_observations: Sequence[Mapping[str, Any]] = (),
    feedback: Mapping[str, Any] | None = None,
    requested_observation_chars: int = 2400,
    max_observations_per_round: int = 8,
) -> str:
    lines = [f"WORKING DOCUMENT revision={document.revision}"]
    if feedback:
        lines.extend(("WORKSPACE FEEDBACK", json.dumps(dict(feedback), ensure_ascii=False, separators=(",", ":"))))
    if document.entities:
        lines.append("ENTITIES")
        for entity in document.entities.values():
            aliases = f" aliases={list(entity.aliases)}" if entity.aliases else ""
            lines.append(f"- [{entity.entity_id}] {entity.description}{aliases}")
    active = [claim for claim in document.claims.values() if claim.status in {"active", "contested"}]
    history = [claim for claim in document.claims.values() if claim.status in {"superseded", "retracted"}]
    lines.append("ACTIVE CLAIMS")
    for claim in sorted(active, key=_claim_sort_key):
        lines.append(_render_claim(claim))
    if history:
        lines.append("CLAIM HISTORY")
        for claim in sorted(history, key=_claim_sort_key):
            successor = f" -> {claim.superseded_by}" if claim.superseded_by else ""
            lines.append(f"- [{claim.claim_id}] {claim.status}{successor}: {_compact_text(claim.text, 180)}")
    if document.timeline:
        lines.append("TIMELINE")
        for note in sorted(document.timeline, key=lambda item: item.time_range):
            lines.append(
                f"- {note.time_range[0]:.3f}-{note.time_range[1]:.3f}s role={note.role} "
                f"{note.label} claims={list(note.claim_ids)}"
            )
    lines.append("TEMPORAL SCOPES")
    lines.append(
        json.dumps(
            [scope.to_dict() for scope in document.temporal_scopes.values()],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    lines.append("EVIDENCE OBLIGATIONS")
    if not document.obligations:
        lines.append("[]")
    else:
        lines.append(
            json.dumps(
                [
                    {
                        **obligation.to_dict(),
                        "state": document.obligation_states[requirement_id].to_dict(),
                    }
                    for requirement_id, obligation in document.obligations.items()
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    lines.append("OBSERVATION CUE STATES")
    lines.append(
        json.dumps(
            {
                cue_id: state.to_dict()
                for cue_id, state in sorted(document.cue_states.items())
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    lines.append("COVERAGE LEDGER")
    coverage = observations.coverage_ledger()
    lines.append(json.dumps(coverage, ensure_ascii=False, separators=(",", ":")))
    lines.append("OBSERVATION CATALOG")
    lines.append(json.dumps(observations.catalog(), ensure_ascii=False, separators=(",", ":")))
    if requested_observations:
        lines.append("REQUESTED OBSERVATION PREVIEWS")
        selected = tuple(requested_observations)[-max(1, int(max_observations_per_round)):]
        for row in selected:
            raw = _compact_text(
                str(row.get("raw_output", "") or ""),
                max(1, int(requested_observation_chars)),
            )
            lines.append(
                f"- [{row.get('attempt_id', '')}/{row.get('interpretation_id', '')}] "
                f"modality={row.get('modality', '')} ranges={row.get('inspected_ranges', ())} "
                f"raw_pointer={observations.path}::{row.get('interpretation_id', '')} {raw}"
            )
    return "\n".join(lines)


def append_workspace_history(
    path: str | Path,
    *,
    round_id: int | str,
    operations: Sequence[Mapping[str, Any]],
    result: WorkspaceApplyResult,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema_version": "WorkingDocumentRevisionV1",
        "round_id": str(round_id),
        "operations": [dict(item) for item in operations],
        "result": result.to_dict(),
    }
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _ids(values: Sequence[Any]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _mapping_values(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Mapping):
        return tuple(value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    return ()


def _time_range(value: Sequence[float] | None) -> tuple[float, float] | None:
    if value is None or len(value) != 2:
        return None
    start, end = sorted((float(value[0]), float(value[1])))
    return start, end


def _required_id(value: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        candidate = str(value.get(key, "") or "").strip()
        if candidate:
            return candidate
    raise ValueError(f"missing_id:{'/'.join(keys)}")


def _observation_item_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    items: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        attempt_id = str(row.get("attempt_id", "") or "").strip()
        interpretation_id = str(row.get("interpretation_id", "") or "").strip()
        if not attempt_id or not interpretation_id:
            continue
        for raw_item in tuple(row.get("interpretation_items", ()) or ()):
            if not isinstance(raw_item, Mapping):
                continue
            try:
                item = InterpretationItem.from_mapping(raw_item)
            except (TypeError, ValueError):
                continue
            key = (attempt_id, interpretation_id, item.item_id)
            items[key] = {
                "attempt_id": attempt_id,
                "interpretation_id": interpretation_id,
                **item.to_dict(),
            }
    return items


def _observation_cue_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    cues: dict[str, dict[str, Any]] = {}
    for row in rows:
        for raw_cue in tuple(row.get("observation_cues", ()) or ()):
            if not isinstance(raw_cue, Mapping):
                continue
            try:
                cue = ObservationCue.from_mapping(raw_cue)
            except (TypeError, ValueError):
                continue
            cues[cue.cue_id] = cue.to_dict()
    return cues


def _observation_claim_binding_errors(
    claim: Claim,
    item_index: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> tuple[str, ...]:
    errors: list[str] = []
    if not claim.interpretation_id:
        errors.append(f"observation_claim_requires_interpretation_id:{claim.claim_id}")
    if not claim.interpretation_item_id:
        errors.append(f"observation_claim_requires_interpretation_item_id:{claim.claim_id}")
    if errors:
        return tuple(errors)

    matches = tuple(
        item
        for (attempt_id, interpretation_id, item_id), item in item_index.items()
        if item_id == claim.interpretation_item_id
        and interpretation_id == claim.interpretation_id
        and attempt_id in claim.cites
    )
    if not matches:
        same_item = tuple(
            key
            for key in item_index
            if key[2] == claim.interpretation_item_id
        )
        if not same_item:
            errors.append(
                f"observation_claim_item_unknown:{claim.claim_id}:{claim.interpretation_item_id}"
            )
        elif not any(key[1] == claim.interpretation_id for key in same_item):
            errors.append(
                f"observation_claim_interpretation_mismatch:{claim.claim_id}:{claim.interpretation_id}"
            )
        else:
            errors.append(
                f"observation_claim_attempt_mismatch:{claim.claim_id}:{','.join(claim.cites)}"
            )
        return tuple(errors)

    if claim.time_anchor is not None and not any(
        _anchors_overlap(claim.time_anchor, tuple(item["time_anchor"]))
        for item in matches
    ):
        errors.append(f"observation_claim_time_anchor_mismatch:{claim.claim_id}")
    return tuple(errors)


def _bound_observation_cues(
    *,
    attempt_id: str,
    interpretation_id: str,
    items: Sequence[InterpretationItem],
    frame_refs: Sequence[str],
    frame_times: Sequence[float],
) -> tuple[ObservationCue, ...]:
    sampled_frames = tuple(
        (str(frame_ref), float(frame_time))
        for frame_ref, frame_time in zip(frame_refs, frame_times)
        if str(frame_ref).strip()
    )
    cues: list[ObservationCue] = []
    seen: set[str] = set()
    for item in items:
        start, end = item.time_anchor
        if abs(start - end) > 1e-6:
            continue
        matched = next(
            (
                (frame_ref, sampled_time)
                for frame_ref, sampled_time in sampled_frames
                if abs(sampled_time - start) <= 1e-6
            ),
            None,
        )
        if matched is None:
            continue
        frame_ref, sampled_time = matched
        cue_id = "cue_" + hashlib.sha256(
            json.dumps(
                [
                    attempt_id,
                    interpretation_id,
                    item.item_id,
                    frame_ref,
                    round(sampled_time, 6),
                ],
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:24]
        if cue_id in seen:
            continue
        seen.add(cue_id)
        cues.append(
            ObservationCue(
                cue_id=cue_id,
                attempt_id=attempt_id,
                interpretation_id=interpretation_id,
                item_id=item.item_id,
                source_frame_ref=frame_ref,
                virtual_time=sampled_time,
                cue_kind=item.item_kind,
            )
        )
    return tuple(cues)


def _anchors_overlap(
    left: tuple[float, float],
    right: tuple[float, float],
) -> bool:
    return min(left[1], right[1]) + 1e-6 >= max(left[0], right[0])


def _compact_text(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def _claim_sort_key(claim: Claim) -> tuple[float, float, str]:
    anchor = claim.time_anchor or (float("inf"), float("inf"))
    return anchor[0], anchor[1], claim.claim_id


def _render_claim(claim: Claim) -> str:
    anchor = f" time={claim.time_anchor[0]:.3f}-{claim.time_anchor[1]:.3f}s" if claim.time_anchor else ""
    citations = f" cites={list(claim.cites)}" if claim.cites else ""
    parents = f" derived_from={list(claim.derived_from)}" if claim.derived_from else ""
    conflicts = f" conflicts={list(claim.conflicts_with)}" if claim.conflicts_with else ""
    entities = f" entities={list(claim.entity_ids)}" if claim.entity_ids else ""
    interpretation = (
        f" interpretation={claim.interpretation_id}/{claim.interpretation_item_id}"
        if claim.interpretation_id or claim.interpretation_item_id
        else ""
    )
    metadata = f" metadata={json.dumps(dict(claim.metadata), ensure_ascii=False, separators=(',', ':'))}" if claim.metadata else ""
    return (
        f"- [{claim.claim_id}] source={claim.source} status={claim.status} confidence={claim.confidence}"
        f"{anchor}{citations}{parents}{conflicts}{entities}{interpretation}{metadata}: {claim.text}"
    )


def _row_overlaps(row: Mapping[str, Any], requested: tuple[float, float]) -> bool:
    ranges = tuple(row.get("inspected_ranges", ()) or ())
    if not ranges and row.get("requested_range"):
        ranges = (row["requested_range"],)
    for item in ranges:
        normalized = _time_range(item)
        if normalized is not None and min(normalized[1], requested[1]) >= max(normalized[0], requested[0]):
            return True
    return False
