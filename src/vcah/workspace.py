from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from vcah.evidence_state import EvidenceObligation, EvidenceObligationState


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
            revision=max(0, int(value.get("revision", 0) or 0)),
            active_claim_limit=max(1, int(value.get("active_claim_limit", 60) or 60)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "WorkingDocumentV3",
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
            errors.extend(staged.validate(observation_ids=observation_ids))
        if errors:
            return WorkspaceApplyResult(False, self.revision, 0, tuple(errors))
        self.claims = staged.claims
        self.entities = staged.entities
        self.timeline = staged.timeline
        self.obligations = staged.obligations
        self.obligation_states = staged.obligation_states
        self.revision += 1
        return WorkspaceApplyResult(True, self.revision, len(ops), ())

    def validate(self, *, observation_ids: Sequence[str]) -> tuple[str, ...]:
        known_observations = {str(item) for item in observation_ids if str(item)}
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

    def validate_answer(
        self,
        supporting_claim_ids: Sequence[str],
        *,
        observation_ids: Sequence[str],
        supporting_observation_ids: Sequence[str] | None = None,
        require_obligation_coverage: bool = False,
    ) -> AnswerValidation:
        support = _ids(supporting_claim_ids)
        if not support:
            return AnswerValidation(False, "supporting_claims_missing")
        document_errors = self.validate(observation_ids=observation_ids)
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
        row = {
            "schema_version": "ObservationInterpretationV1",
            "attempt_id": attempt_id,
            "interpretation_id": f"interpretation_{hashlib.sha256(interpretation_material.encode('utf-8')).hexdigest()[:20]}",
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
    metadata = f" metadata={json.dumps(dict(claim.metadata), ensure_ascii=False, separators=(',', ':'))}" if claim.metadata else ""
    return (
        f"- [{claim.claim_id}] source={claim.source} status={claim.status} confidence={claim.confidence}"
        f"{anchor}{citations}{parents}{conflicts}{entities}{metadata}: {claim.text}"
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
