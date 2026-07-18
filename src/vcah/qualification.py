from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Callable, Mapping, Sequence

from vcah.provenance import heuristic_provenance, normalize_provenance, provenance_is_admissible


class RequirementStatus(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    CONFLICTED = "conflicted"
    UNKNOWN = "unknown"
    BLOCKED_UNRESOLVED = "blocked_unresolved"
    BLOCKED_CONFLICTED = "blocked_conflicted"
    NOT_APPLICABLE = "not_applicable"
    # Compatibility alias for callers that only distinguish a generic blocked state.
    BLOCKED = "blocked_unresolved"


@dataclass(frozen=True)
class QualificationRequirement:
    requirement_id: str
    predicate: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    scope: str = "candidate"
    quantifier: str = "exists"
    dependency_ids: tuple[str, ...] = ()
    required: bool = True
    evaluator: str = "builtin"

    def __post_init__(self) -> None:
        object.__setattr__(self, "requirement_id", str(self.requirement_id or "").strip())
        object.__setattr__(self, "predicate", str(self.predicate or "custom").strip().casefold())
        object.__setattr__(self, "arguments", dict(self.arguments or {}))
        object.__setattr__(self, "scope", str(self.scope or "candidate").strip().casefold())
        object.__setattr__(self, "quantifier", str(self.quantifier or "exists").strip().casefold())
        object.__setattr__(
            self,
            "dependency_ids",
            tuple(dict.fromkeys(str(item).strip() for item in self.dependency_ids if str(item).strip())),
        )
        object.__setattr__(self, "required", bool(self.required))
        object.__setattr__(self, "evaluator", str(self.evaluator or "builtin").strip().casefold())


@dataclass(frozen=True)
class RequirementEvaluation:
    requirement_id: str
    status: RequirementStatus
    fact_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    reason: str = ""
    blocked_by: tuple[str, ...] = ()
    provenance: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "status": self.status.value,
            "fact_ids": list(self.fact_ids),
            "evidence_ids": list(self.evidence_ids),
            "reason": self.reason,
            "blocked_by": list(self.blocked_by),
            "provenance": [dict(item) for item in self.provenance],
        }


@dataclass(frozen=True)
class OptionPredicate:
    predicate_id: str
    option_id: str
    subject_role: str
    attribute: str
    operator: str
    value: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicate_id": self.predicate_id,
            "option_id": self.option_id,
            "subject_role": self.subject_role,
            "attribute": self.attribute,
            "operator": self.operator,
            "value": self.value,
        }


Evaluator = Callable[[QualificationRequirement, Mapping[str, Any]], RequirementEvaluation]


def _dependency_short_circuit(
    dependencies: Sequence[RequirementEvaluation],
) -> tuple[RequirementStatus, str] | None:
    statuses = {item.status for item in dependencies}
    if statuses.intersection({RequirementStatus.CONTRADICTED, RequirementStatus.NOT_APPLICABLE}):
        return (
            RequirementStatus.NOT_APPLICABLE,
            "A terminal dependency was contradicted, so this requirement is not applicable.",
        )
    if statuses.intersection({RequirementStatus.CONFLICTED, RequirementStatus.BLOCKED_CONFLICTED}):
        return RequirementStatus.BLOCKED_CONFLICTED, "A required dependency is conflicted."
    if statuses.intersection({RequirementStatus.UNKNOWN, RequirementStatus.BLOCKED_UNRESOLVED}):
        return RequirementStatus.BLOCKED_UNRESOLVED, "A required dependency remains unresolved."
    return None


def evaluate_requirement_graph(
    requirements: Sequence[QualificationRequirement],
    context: Mapping[str, Any],
    evaluators: Mapping[str, Evaluator] | None = None,
) -> tuple[RequirementEvaluation, ...]:
    available = dict(_BUILTIN_EVALUATORS)
    available.update(dict(evaluators or {}))
    pending = {requirement.requirement_id: requirement for requirement in requirements}
    results: dict[str, RequirementEvaluation] = {}
    while pending:
        progressed = False
        for requirement_id, requirement in tuple(pending.items()):
            unresolved = tuple(item for item in requirement.dependency_ids if item not in results)
            if unresolved:
                continue
            dependencies = tuple(results[item] for item in requirement.dependency_ids)
            blocked_by = tuple(
                item for item in requirement.dependency_ids
                if results[item].status is not RequirementStatus.SUPPORTED
            )
            short_circuit = _dependency_short_circuit(dependencies)
            if short_circuit is not None:
                status, reason = short_circuit
                results[requirement_id] = RequirementEvaluation(
                    requirement_id,
                    status,
                    reason=reason,
                    blocked_by=blocked_by,
                )
            else:
                evaluator = available.get(requirement.evaluator) or available.get(requirement.predicate)
                results[requirement_id] = (
                    evaluator(requirement, context)
                    if evaluator is not None
                    else RequirementEvaluation(
                        requirement_id,
                        RequirementStatus.UNKNOWN,
                        reason=f"No evaluator is registered for predicate {requirement.predicate}.",
                    )
                )
            del pending[requirement_id]
            progressed = True
        if progressed:
            continue
        for requirement_id, requirement in pending.items():
            results[requirement_id] = RequirementEvaluation(
                requirement_id,
                RequirementStatus.BLOCKED_UNRESOLVED,
                reason="Requirement dependency cycle or missing dependency.",
                blocked_by=tuple(requirement.dependency_ids),
            )
        break
    return tuple(results[requirement.requirement_id] for requirement in requirements)


def qualification_status(
    requirements: Sequence[QualificationRequirement],
    evaluations: Sequence[RequirementEvaluation],
) -> str:
    required_ids = {requirement.requirement_id for requirement in requirements if requirement.required}
    evaluated_required_ids = {
        evaluation.requirement_id
        for evaluation in evaluations
        if evaluation.requirement_id in required_ids
    }
    statuses = {
        evaluation.status
        for evaluation in evaluations
        if evaluation.requirement_id in required_ids
    }
    if RequirementStatus.CONTRADICTED in statuses:
        return "unqualified_precondition"
    if statuses.intersection({RequirementStatus.CONFLICTED, RequirementStatus.BLOCKED_CONFLICTED}):
        return "conflicted"
    if statuses.intersection({RequirementStatus.UNKNOWN, RequirementStatus.BLOCKED_UNRESOLVED}) or evaluated_required_ids != required_ids:
        return "incomplete"
    return "qualified"


def requirement_telemetry(evaluations: Sequence[RequirementEvaluation]) -> dict[str, Any]:
    counts = {status.value: 0 for status in RequirementStatus}
    for evaluation in evaluations:
        counts[evaluation.status.value] += 1
    blocked_unresolved = counts.get(RequirementStatus.BLOCKED_UNRESOLVED.value, 0)
    blocked_conflicted = counts.get(RequirementStatus.BLOCKED_CONFLICTED.value, 0)
    return {
        "total": len(evaluations),
        **counts,
        "blocked": blocked_unresolved + blocked_conflicted,
        "unresolved_dependency_ids": [
            evaluation.requirement_id
            for evaluation in evaluations
            if evaluation.status in {
                RequirementStatus.UNKNOWN,
                RequirementStatus.CONFLICTED,
                RequirementStatus.BLOCKED_UNRESOLVED,
                RequirementStatus.BLOCKED_CONFLICTED,
            }
        ],
    }


def qualify_event_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    require_state_precondition: bool = False,
) -> dict[str, Any]:
    qualified: list[dict[str, Any]] = []
    unqualified: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    conflicted: list[dict[str, Any]] = []
    all_evaluations: list[RequirementEvaluation] = []
    for candidate in candidates:
        requirements = event_candidate_requirements(
            candidate,
            require_state_precondition=require_state_precondition,
        )
        evaluations = evaluate_requirement_graph(requirements, {})
        all_evaluations.extend(evaluations)
        status = qualification_status(requirements, evaluations)
        row = {
            **dict(candidate),
            "qualification_status": status,
            "requirement_ids": [requirement.requirement_id for requirement in requirements],
            "requirement_evaluations": [evaluation.to_dict() for evaluation in evaluations],
        }
        if status == "qualified":
            qualified.append(row)
        elif status == "unqualified_precondition":
            unqualified.append(row)
        elif status == "conflicted":
            conflicted.append(row)
        else:
            incomplete.append(row)
    return {
        "qualified_events": qualified,
        "unqualified_events": unqualified,
        "incomplete_events": incomplete,
        "conflicted_events": conflicted,
        "requirement_graph": requirement_telemetry(all_evaluations),
    }


def event_candidate_requirements(
    candidate: Mapping[str, Any],
    *,
    require_state_precondition: bool = False,
) -> tuple[QualificationRequirement, ...]:
    candidate_id = str(candidate.get("candidate_id", "") or "candidate")
    evidence_ids = tuple(candidate.get("evidence_ids", ()) or ())
    provenance = tuple(candidate.get("provenance", ()) or ()) or (
        heuristic_provenance(
            fact_ids=(candidate_id,),
            evidence_ids=evidence_ids,
            derivation="candidate_fields_without_independent_witness",
        ),
    )
    distinctness_provenance = tuple(candidate.get("distinctness_provenance", ()) or ()) or (
        heuristic_provenance(
            fact_ids=(candidate_id,),
            evidence_ids=evidence_ids,
            derivation="distinct_occurrence_without_canonical_reconciliation",
        ),
    )
    qualification_provenance = tuple(candidate.get("qualification_provenance", ()) or ()) or (
        heuristic_provenance(
            fact_ids=(candidate_id,),
            evidence_ids=evidence_ids,
            derivation="qualification_fields_without_independent_witness",
        ),
    )
    qualification_provenance_by_field = {
        str(field_name): tuple(value or ())
        for field_name, value in dict(
            candidate.get("qualification_provenance_by_field", {}) or {}
        ).items()
    }

    def provenance_for(field_name: str) -> tuple[Any, ...]:
        return qualification_provenance_by_field.get(field_name, qualification_provenance)

    candidate_status = str(candidate.get("candidate_status", "observed_candidate") or "observed_candidate")
    observed_status = "conflicted" if candidate_status == "conflicted_candidate" else "supported"
    requirements = [
        QualificationRequirement(
            f"req_{candidate_id}_observed",
            "observed",
            {"status": observed_status, "evidence_ids": evidence_ids, "provenance": provenance},
            scope="candidate",
        ),
        QualificationRequirement(
            f"req_{candidate_id}_distinct",
            "distinct_occurrence",
            {
                "status": "conflicted" if candidate_status == "conflicted_candidate" else "supported",
                "evidence_ids": evidence_ids,
                "provenance": distinctness_provenance,
            },
            scope="candidate",
            dependency_ids=(f"req_{candidate_id}_observed",),
        ),
    ]
    if bool(candidate.get("focal_position_transition")):
        observations = dict(candidate.get("qualification_observations", {}) or {})
        preconditions = tuple(candidate.get("preconditions_met_observations", ()) or ())
        if False in preconditions:
            observations["required_prior_state"] = "contradicted"
        elif True in preconditions or any(
            re.search(r"\b(?:rank(?:ed)?\s*(?:=|is|was)?\s*1|first place|in the lead|leading)\b", str(value).casefold())
            for value in tuple(candidate.get("states_before", ()) or ())
        ):
            observations["required_prior_state"] = "supported"
        if not _recognized_observation_status(observations.get("transition")) and tuple(
            candidate.get("transitions", ()) or ()
        ):
            observations["transition"] = "supported"
        if not _recognized_observation_status(observations.get("same_subject")) and "camera_holder" in set(
            candidate.get("participant_ids", ()) or ()
        ):
            observations["same_subject"] = "supported"
        if not _recognized_observation_status(observations.get("episode_boundary")) and not (
            bool(candidate.get("continues_from_previous")) or bool(candidate.get("continues_to_next"))
        ):
            observations["episode_boundary"] = "supported"
        requirement_specs = (
            ("prior_state", "state_before", "required_prior_state", observations.get("required_prior_state")),
            ("transition", "state_transition", "transition", observations.get("transition")),
            ("same_subject", "same_entity", "same_subject", observations.get("same_subject")),
            ("episode_boundary", "distinct_occurrence", "episode_boundary", observations.get("episode_boundary")),
        )
        dependency = f"req_{candidate_id}_distinct"
        for suffix, predicate, provenance_field, status in requirement_specs:
            requirement_id = f"req_{candidate_id}_{suffix}"
            requirements.append(QualificationRequirement(
                requirement_id,
                predicate,
                {
                    "status": _normalized_observation_status(status),
                    "evidence_ids": evidence_ids,
                    "provenance": provenance_for(provenance_field),
                },
                scope="event",
                dependency_ids=(dependency,),
            ))
            dependency = requirement_id
    elif require_state_precondition:
        observations = tuple(candidate.get("preconditions_met_observations", ()) or ())
        status = (
            "contradicted" if False in observations
            else "supported" if observations and all(value is True for value in observations)
            else "unknown"
        )
        requirements.append(QualificationRequirement(
            f"req_{candidate_id}_prior_state",
            "state_before",
            {
                "status": status,
                "evidence_ids": evidence_ids,
                "provenance": provenance_for("required_prior_state"),
            },
            scope="event",
            dependency_ids=(f"req_{candidate_id}_distinct",),
        ))
    return tuple(requirements)


def parse_option_predicates(options: Mapping[str, str]) -> dict[str, tuple[OptionPredicate, ...]]:
    return {
        option: tuple(_parse_option_predicate(option, text))
        for option, text in options.items()
    }


def apply_observation_to_requirements(
    observation: Mapping[str, Any],
    requirements: Sequence[QualificationRequirement],
) -> tuple[RequirementEvaluation, ...]:
    target_ids = {
        str(item).strip()
        for item in tuple(observation.get("target_requirement_ids", ()) or ())
        if str(item).strip()
    }
    if not target_ids:
        return ()
    proposed = dict(observation.get("requirement_results", {}) or {})
    scoped = tuple(requirement for requirement in requirements if requirement.requirement_id in target_ids)
    contextualized = tuple(
        QualificationRequirement(
            requirement.requirement_id,
            requirement.predicate,
            {
                **dict(requirement.arguments),
                "status": proposed.get(requirement.requirement_id, "unknown"),
                "evidence_ids": tuple(observation.get("evidence_ids", ()) or ()),
                "provenance": tuple(observation.get("provenance", ()) or ()),
            },
            requirement.scope,
            requirement.quantifier,
            tuple(item for item in requirement.dependency_ids if item in target_ids),
            requirement.required,
            requirement.evaluator,
        )
        for requirement in scoped
    )
    return evaluate_requirement_graph(contextualized, {})


def _parse_option_predicate(option_id: str, text: str) -> list[OptionPredicate]:
    normalized = " ".join(str(text or "").casefold().split())
    predicates: list[OptionPredicate] = []
    brand = "red_bull" if re.search(r"\bred[ -]?bull\b", normalized) else ""
    if brand:
        predicates.append(_option_predicate(option_id, "helmet_brand", brand, len(predicates)))
    colors = re.findall(
        r"\b(?:black|blue|brown|gray|grey|green|orange|pink|purple|red|tan|white|yellow)\b",
        normalized,
    )
    color = colors[-1] if colors else ""
    color = "gray" if color == "grey" else color
    if color and re.search(r"\bhelmet\b", normalized) and not brand:
        predicates.append(_option_predicate(option_id, "helmet_color", color, len(predicates)))
    if color and re.search(r"\b(?:clothes|clothing|jacket|jersey|shirt|suit|top)\b", normalized):
        predicates.append(_option_predicate(option_id, "clothing_color", color, len(predicates)))
    count = _option_count(normalized) if not predicates else None
    if count is not None:
        predicates.append(_option_predicate(option_id, "count", count, len(predicates), subject_role="query_fact"))
    return predicates


def _option_predicate(
    option_id: str,
    attribute: str,
    value: Any,
    index: int,
    subject_role: str = "target_participant",
) -> OptionPredicate:
    return OptionPredicate(
        f"option_{option_id}_pred_{index + 1:02d}",
        option_id,
        subject_role,
        attribute,
        "equals",
        value,
    )


def _option_count(text: str) -> int | None:
    match = re.search(r"\b\d+\b", text)
    if match:
        return int(match.group(0))
    words = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    return next((value for word, value in words.items() if re.search(rf"\b{word}\b", text)), None)


def _normalized_observation_status(value: Any) -> str:
    return {
        True: "supported",
        False: "contradicted",
        "supported": "supported",
        "verified": "supported",
        "satisfied": "supported",
        "refuted": "contradicted",
        "contradicted": "contradicted",
        "conflicted": "conflicted",
    }.get(value, "unknown")


def _recognized_observation_status(value: Any) -> bool:
    return _normalized_observation_status(value) != "unknown"


def _argument_status(
    requirement: QualificationRequirement,
    context: Mapping[str, Any],
) -> RequirementEvaluation:
    value = requirement.arguments.get("status", "unknown")
    status = {
        True: RequirementStatus.SUPPORTED,
        False: RequirementStatus.CONTRADICTED,
        "supported": RequirementStatus.SUPPORTED,
        "verified": RequirementStatus.SUPPORTED,
        "contradicted": RequirementStatus.CONTRADICTED,
        "refuted": RequirementStatus.CONTRADICTED,
        "conflicted": RequirementStatus.CONFLICTED,
    }.get(value, RequirementStatus.UNKNOWN)
    provenance = normalize_provenance(requirement.arguments.get("provenance", ()))
    if status is not RequirementStatus.UNKNOWN and not provenance_is_admissible(provenance):
        return RequirementEvaluation(
            requirement.requirement_id,
            RequirementStatus.UNKNOWN,
            evidence_ids=_strings(requirement.arguments.get("evidence_ids", ())),
            reason=f"{requirement.predicate} has no admissible provenance.",
            provenance=provenance,
        )
    return RequirementEvaluation(
        requirement.requirement_id,
        status,
        fact_ids=_strings(requirement.arguments.get("fact_ids", ())),
        evidence_ids=_strings(requirement.arguments.get("evidence_ids", ())),
        reason=str(requirement.arguments.get("reason", "") or f"{requirement.predicate} is {status.value}."),
        provenance=provenance,
    )


def _custom_requirement(
    requirement: QualificationRequirement,
    context: Mapping[str, Any],
) -> RequirementEvaluation:
    facts = tuple(context.get("narrative_facts", ()) or ())
    definition = str(requirement.arguments.get("definition", "") or "").strip().casefold()
    matching = [
        fact for fact in facts
        if str(fact.get("predicate", "") or "").strip().casefold() == definition
        and str(fact.get("qualification_status", "") or "") == "qualified"
        and provenance_is_admissible(fact.get("provenance", ()))
    ]
    if not matching:
        return RequirementEvaluation(
            requirement.requirement_id,
            RequirementStatus.UNKNOWN,
            reason="Custom predicates require a qualified narrative inference fact.",
        )
    return RequirementEvaluation(
        requirement.requirement_id,
        RequirementStatus.SUPPORTED,
        fact_ids=_strings(fact.get("fact_id", "") for fact in matching),
        evidence_ids=_strings(
            evidence_id
            for fact in matching
            for evidence_id in tuple(fact.get("evidence_ids", ()) or ())
        ),
        reason="A qualified narrative inference fact matches the custom predicate.",
        provenance=tuple(
            row
            for fact in matching
            for row in normalize_provenance(fact.get("provenance", ()))
        ),
    )


def _strings(values: Any) -> tuple[str, ...]:
    if isinstance(values, str):
        values = (values,)
    return tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


_BUILTIN_EVALUATORS: dict[str, Evaluator] = {
    "builtin": _argument_status,
    "observed": _argument_status,
    "event_type": _argument_status,
    "participant_role": _argument_status,
    "state_before": _argument_status,
    "state_after": _argument_status,
    "state_transition": _argument_status,
    "same_entity": _argument_status,
    "distinct_occurrence": _argument_status,
    "within_scope": _argument_status,
    "temporal_before": _argument_status,
    "temporal_after": _argument_status,
    "temporal_min": _argument_status,
    "temporal_max": _argument_status,
    "ordinal_member": _argument_status,
    "attribute_match": _argument_status,
    "coverage_complete": _argument_status,
    "qualified_absence": _argument_status,
    "narrative_relation": _argument_status,
    "option_predicate_match": _argument_status,
    "custom": _custom_requirement,
}
