from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


SUFFICIENCY_OPERATION = "assess_sufficiency"
SUFFICIENCY_VERDICTS = frozenset({"sufficient", "insufficient"})
CONSTRAINT_SUPPORT_STATUSES = frozenset(
    {"supported", "partial", "unknown", "contradicted"}
)
QUESTION_CRITICAL_CONSTRAINT_TYPES = frozenset(
    {
        "action",
        "identity",
        "event",
        "relation",
        "temporal",
        "state",
        "attribute",
        "object",
        "location",
        "order",
        "outcome",
    }
)
MAX_CONSTRAINTS = 6
MAX_CONSTRAINT_DESCRIPTION_CHARS = 240
MAX_EVIDENCE_PASSAGES = 3


@dataclass(frozen=True)
class OccurrenceSufficiencyDecision:
    set_id: str
    verdict: str
    constraints_checked: tuple[dict[str, Any], ...]
    sufficient_occurrence_ids: tuple[str, ...]
    declared_verdict: str
    verdict_normalized: bool
    implicit_unknown_support_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "set_id": self.set_id,
            "verdict": self.verdict,
            "constraints_checked": [
                {
                    **constraint,
                    "support": [
                        dict(row)
                        for row in tuple(constraint.get("support", ()) or ())
                    ],
                }
                for constraint in self.constraints_checked
            ],
            "sufficient_occurrence_ids": list(self.sufficient_occurrence_ids),
            "declared_verdict": self.declared_verdict,
            "verdict_normalized": self.verdict_normalized,
            "implicit_unknown_support_count": self.implicit_unknown_support_count,
        }

    def to_operation(self) -> dict[str, Any]:
        return {"op": SUFFICIENCY_OPERATION, **self.to_dict()}


def validate_sufficiency_operation(
    operation: Mapping[str, Any],
    *,
    set_id: str,
    candidates: Mapping[str, Mapping[str, Any]],
    viable_occurrence_ids: Sequence[str],
    operation_index: int,
) -> tuple[OccurrenceSufficiencyDecision | None, list[dict[str, Any]]]:
    verdict = str(operation.get("verdict", "") or "").strip().casefold()
    if verdict not in SUFFICIENCY_VERDICTS:
        return None, [
            _error(
                "occurrence_sufficiency_verdict_invalid",
                operation_index,
                set_id=set_id,
                verdict=verdict,
            )
        ]

    raw_constraints = operation.get("constraints_checked")
    if not isinstance(raw_constraints, Sequence) or isinstance(
        raw_constraints, (str, bytes)
    ):
        raw_constraints = ()
    if not 1 <= len(raw_constraints) <= MAX_CONSTRAINTS:
        return None, [
            _error(
                "occurrence_sufficiency_constraints_required",
                operation_index,
                set_id=set_id,
                minimum=1,
                maximum=MAX_CONSTRAINTS,
            )
        ]

    viable_ids = tuple(dict.fromkeys(str(value) for value in viable_occurrence_ids))
    viable_set = set(viable_ids)
    normalized_constraints: list[dict[str, Any]] = []
    support_by_candidate: dict[str, list[str]] = {
        occurrence_id: [] for occurrence_id in viable_ids
    }
    seen_constraint_ids: set[str] = set()
    errors: list[dict[str, Any]] = []
    for constraint_index, raw_constraint in enumerate(raw_constraints):
        if not isinstance(raw_constraint, Mapping):
            errors.append(
                _error(
                    "occurrence_sufficiency_constraint_must_be_object",
                    operation_index,
                    set_id=set_id,
                    constraint_index=constraint_index,
                )
            )
            continue
        constraint_id = str(raw_constraint.get("constraint_id", "") or "").strip()
        constraint_type = str(
            raw_constraint.get("constraint_type", "") or ""
        ).strip().casefold()
        description = str(raw_constraint.get("description", "") or "").strip()
        if not constraint_id or constraint_id in seen_constraint_ids:
            errors.append(
                _error(
                    "occurrence_sufficiency_constraint_id_invalid",
                    operation_index,
                    set_id=set_id,
                    constraint_index=constraint_index,
                    constraint_id=constraint_id,
                )
            )
            continue
        seen_constraint_ids.add(constraint_id)
        if constraint_type not in QUESTION_CRITICAL_CONSTRAINT_TYPES:
            errors.append(
                _error(
                    "occurrence_sufficiency_constraint_type_invalid",
                    operation_index,
                    set_id=set_id,
                    constraint_index=constraint_index,
                    constraint_type=constraint_type,
                )
            )
        if not description or len(description) > MAX_CONSTRAINT_DESCRIPTION_CHARS:
            errors.append(
                _error(
                    "occurrence_sufficiency_constraint_description_invalid",
                    operation_index,
                    set_id=set_id,
                    constraint_index=constraint_index,
                    description_chars=len(description),
                )
            )

        raw_support = raw_constraint.get("support")
        if not isinstance(raw_support, Sequence) or isinstance(
            raw_support, (str, bytes)
        ):
            raw_support = ()
        support_rows: list[dict[str, Any]] = []
        seen_support_ids: set[str] = set()
        for support_index, raw_row in enumerate(raw_support):
            if not isinstance(raw_row, Mapping):
                errors.append(
                    _error(
                        "occurrence_sufficiency_support_must_be_object",
                        operation_index,
                        set_id=set_id,
                        constraint_index=constraint_index,
                        support_index=support_index,
                    )
                )
                continue
            occurrence_id = str(raw_row.get("occurrence_id", "") or "").strip()
            status = str(raw_row.get("status", "") or "").strip().casefold()
            if occurrence_id not in viable_set or occurrence_id in seen_support_ids:
                errors.append(
                    _error(
                        "occurrence_sufficiency_support_candidate_invalid",
                        operation_index,
                        set_id=set_id,
                        constraint_index=constraint_index,
                        occurrence_id=occurrence_id,
                    )
                )
                continue
            seen_support_ids.add(occurrence_id)
            if status not in CONSTRAINT_SUPPORT_STATUSES:
                errors.append(
                    _error(
                        "occurrence_sufficiency_support_status_invalid",
                        operation_index,
                        set_id=set_id,
                        constraint_index=constraint_index,
                        occurrence_id=occurrence_id,
                        status=status,
                    )
                )
                continue
            evidence_ids = _normalized_strings(
                raw_row.get("evidence_passage_ids"), limit=MAX_EVIDENCE_PASSAGES
            )
            visible_passage_ids = {
                str(value)
                for value in tuple(
                    candidates.get(occurrence_id, {}).get("passage_ids", ()) or ()
                )
                if str(value)
            }
            if any(value not in visible_passage_ids for value in evidence_ids):
                errors.append(
                    _error(
                        "occurrence_sufficiency_evidence_not_visible",
                        operation_index,
                        set_id=set_id,
                        constraint_index=constraint_index,
                        occurrence_id=occurrence_id,
                    )
                )
                continue
            if status == "supported" and not evidence_ids:
                errors.append(
                    _error(
                        "occurrence_sufficiency_supported_requires_evidence",
                        operation_index,
                        set_id=set_id,
                        constraint_index=constraint_index,
                        occurrence_id=occurrence_id,
                    )
                )
                continue
            support_by_candidate[occurrence_id].append(status)
            support_rows.append(
                {
                    "occurrence_id": occurrence_id,
                    "status": status,
                    "evidence_passage_ids": list(evidence_ids),
                }
            )
        missing_ids = [
            occurrence_id
            for occurrence_id in viable_ids
            if occurrence_id not in seen_support_ids
        ]
        for occurrence_id in missing_ids:
            support_by_candidate[occurrence_id].append("unknown")
            support_rows.append(
                {
                    "occurrence_id": occurrence_id,
                    "status": "unknown",
                    "evidence_passage_ids": [],
                }
            )
        normalized_constraints.append(
            {
                "constraint_id": constraint_id,
                "constraint_type": constraint_type,
                "description": description,
                "support": support_rows,
                "implicit_unknown_occurrence_ids": missing_ids,
            }
        )

    if errors:
        return None, errors
    sufficient_ids = tuple(
        occurrence_id
        for occurrence_id in viable_ids
        if len(support_by_candidate[occurrence_id]) == len(normalized_constraints)
        and all(
            status == "supported" for status in support_by_candidate[occurrence_id]
        )
    )
    expected_verdict = "sufficient" if sufficient_ids else "insufficient"
    return (
        OccurrenceSufficiencyDecision(
            set_id=set_id,
            verdict=expected_verdict,
            constraints_checked=tuple(normalized_constraints),
            sufficient_occurrence_ids=sufficient_ids,
            declared_verdict=verdict,
            verdict_normalized=verdict != expected_verdict,
            implicit_unknown_support_count=sum(
                len(tuple(row.get("implicit_unknown_occurrence_ids", ()) or ()))
                for row in normalized_constraints
            ),
        ),
        [],
    )


def _normalized_strings(value: Any, *, limit: int) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(
        dict.fromkeys(
            str(item or "").strip()
            for item in value
            if str(item or "").strip()
        )
    )[:limit]


def _error(code: str, operation_index: int, **fields: Any) -> dict[str, Any]:
    return {
        "code": code,
        "occurrence_op_index": operation_index,
        **fields,
    }
