from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence


EVIDENCE_OPERATION = "declare_occurrence_evidence"
LEGACY_SUFFICIENCY_OPERATION = "assess_sufficiency"
# Runtime code imports this name for the active protocol operation.
SUFFICIENCY_OPERATION = EVIDENCE_OPERATION
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
REFERENT_IDENTIFYING_CONSTRAINT_TYPES = frozenset(
    {
        "action",
        "identity",
        "event",
        "relation",
        "temporal",
        "location",
        "order",
    }
)
ANSWER_TARGET_CONSTRAINT_TYPES = frozenset(
    {"state", "attribute", "object", "outcome"}
)
if (
    REFERENT_IDENTIFYING_CONSTRAINT_TYPES | ANSWER_TARGET_CONSTRAINT_TYPES
    != QUESTION_CRITICAL_CONSTRAINT_TYPES
    or REFERENT_IDENTIFYING_CONSTRAINT_TYPES & ANSWER_TARGET_CONSTRAINT_TYPES
):
    raise RuntimeError("occurrence sufficiency constraint taxonomy is incomplete")

MAX_CONSTRAINTS = 6
MAX_CONSTRAINT_DESCRIPTION_CHARS = 240
MAX_EVIDENCE_PASSAGES = 3
DEFAULT_SUFFICIENCY_CANDIDATE_LIMIT = 5
EVIDENCE_SUPPORT_CONTRACT = "rule_blind_sparse_positive_evidence_v1"
SUFFICIENCY_SUPPORT_CONTRACT = EVIDENCE_SUPPORT_CONTRACT
SUFFICIENCY_AGGREGATION_RULE = "unique_supported_count_margin"
MIN_SUFFICIENCY_SUPPORT_MARGIN = 1

_FORBIDDEN_GATE_FIELDS = frozenset(
    {
        "verdict",
        "declared_verdict",
        "sufficient_occurrence_ids",
        "winner",
        "winner_occurrence_id",
        "support_count_by_occurrence",
        "best_support_count",
        "runner_up_support_count",
        "margin",
        "aggregation_rule",
        "minimum_support_margin",
    }
)
_EVIDENCE_OPERATION_FIELDS = frozenset(
    {"op", "type", "set_id", "locator_attempt_id", "constraints"}
)
_EVIDENCE_CONSTRAINT_FIELDS = frozenset(
    {"constraint_id", "constraint_type", "description", "supported_candidates"}
)
_EVIDENCE_SUPPORT_FIELDS = frozenset(
    {"occurrence_id", "evidence_passage_ids"}
)


@dataclass(frozen=True)
class OccurrenceEvidenceReport:
    set_id: str
    constraints: tuple[dict[str, Any], ...]
    implicit_unknown_support_count: int
    scope_occurrence_ids: tuple[str, ...]
    out_of_scope_occurrence_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "OccurrenceEvidenceReportV1",
            "set_id": self.set_id,
            "constraints": [
                {
                    **constraint,
                    "supported_candidates": [
                        dict(row)
                        for row in tuple(
                            constraint.get("supported_candidates", ()) or ()
                        )
                    ],
                }
                for constraint in self.constraints
            ],
            "implicit_unknown_support_count": self.implicit_unknown_support_count,
            "scope_occurrence_ids": list(self.scope_occurrence_ids),
            "out_of_scope_occurrence_ids": list(
                self.out_of_scope_occurrence_ids
            ),
            "support_complete": True,
            "support_contract": EVIDENCE_SUPPORT_CONTRACT,
            "rule_blind": True,
            "model_verdict_present": False,
        }

    def to_operation(self) -> dict[str, Any]:
        return {
            "op": EVIDENCE_OPERATION,
            **self.to_dict(),
            "evidence_report_digest": self.digest,
        }

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class OccurrenceSufficiencyDecision:
    set_id: str
    verdict: str
    sufficient_occurrence_ids: tuple[str, ...]
    support_count_by_occurrence: tuple[tuple[str, int], ...]
    best_support_count: int
    runner_up_support_count: int
    evidence_report_digest: str

    @property
    def winner_occurrence_id(self) -> str:
        return (
            self.sufficient_occurrence_ids[0]
            if self.sufficient_occurrence_ids
            else ""
        )

    @property
    def support_margin(self) -> int:
        return self.best_support_count - self.runner_up_support_count

    @property
    def resolution_op(self) -> str:
        return "select" if self.verdict == "sufficient" else "no_match"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "OccurrenceSufficiencyGateDecisionV1",
            "set_id": self.set_id,
            "verdict": self.verdict,
            "winner_occurrence_id": self.winner_occurrence_id or None,
            "sufficient_occurrence_ids": list(self.sufficient_occurrence_ids),
            "support_contract": EVIDENCE_SUPPORT_CONTRACT,
            "aggregation_rule": SUFFICIENCY_AGGREGATION_RULE,
            "minimum_support_margin": MIN_SUFFICIENCY_SUPPORT_MARGIN,
            "support_count_by_occurrence": dict(
                self.support_count_by_occurrence
            ),
            "best_support_count": self.best_support_count,
            "runner_up_support_count": self.runner_up_support_count,
            "support_margin": self.support_margin,
            "resolution_op": self.resolution_op,
            "decision_owner": "runtime",
            "model_verdict_present": False,
            "evidence_report_digest": self.evidence_report_digest,
        }


def validate_evidence_operation(
    operation: Mapping[str, Any],
    *,
    set_id: str,
    candidates: Mapping[str, Mapping[str, Any]],
    viable_occurrence_ids: Sequence[str],
    out_of_scope_occurrence_ids: Sequence[str] = (),
    operation_index: int,
) -> tuple[OccurrenceEvidenceReport | None, list[dict[str, Any]]]:
    forbidden_fields = sorted(_FORBIDDEN_GATE_FIELDS & set(operation))
    if forbidden_fields:
        return None, [
            _error(
                "occurrence_evidence_forbidden_gate_field",
                operation_index,
                set_id=set_id,
                forbidden_fields=forbidden_fields,
            )
        ]
    unexpected_fields = sorted(set(operation) - _EVIDENCE_OPERATION_FIELDS)
    if unexpected_fields:
        return None, [
            _error(
                "occurrence_evidence_operation_field_invalid",
                operation_index,
                set_id=set_id,
                unexpected_fields=unexpected_fields,
            )
        ]

    raw_constraints = operation.get("constraints")
    if not isinstance(raw_constraints, Sequence) or isinstance(
        raw_constraints, (str, bytes)
    ):
        raw_constraints = ()
    if not 1 <= len(raw_constraints) <= MAX_CONSTRAINTS:
        return None, [
            _error(
                "occurrence_evidence_constraints_required",
                operation_index,
                set_id=set_id,
                minimum=1,
                maximum=MAX_CONSTRAINTS,
            )
        ]

    viable_ids = tuple(dict.fromkeys(str(value) for value in viable_occurrence_ids))
    viable_set = set(viable_ids)
    normalized_constraints: list[dict[str, Any]] = []
    seen_constraint_ids: set[str] = set()
    errors: list[dict[str, Any]] = []
    for constraint_index, raw_constraint in enumerate(raw_constraints):
        if not isinstance(raw_constraint, Mapping):
            errors.append(
                _error(
                    "occurrence_evidence_constraint_must_be_object",
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
        unexpected_constraint_fields = sorted(
            set(raw_constraint) - _EVIDENCE_CONSTRAINT_FIELDS
        )
        if unexpected_constraint_fields:
            errors.append(
                _error(
                    "occurrence_evidence_constraint_field_invalid",
                    operation_index,
                    set_id=set_id,
                    constraint_index=constraint_index,
                    unexpected_fields=unexpected_constraint_fields,
                )
            )
        if not constraint_id or constraint_id in seen_constraint_ids:
            errors.append(
                _error(
                    "occurrence_evidence_constraint_id_invalid",
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
                    "occurrence_evidence_constraint_type_invalid",
                    operation_index,
                    set_id=set_id,
                    constraint_index=constraint_index,
                    constraint_type=constraint_type,
                )
            )
        if not description or len(description) > MAX_CONSTRAINT_DESCRIPTION_CHARS:
            errors.append(
                _error(
                    "occurrence_evidence_constraint_description_invalid",
                    operation_index,
                    set_id=set_id,
                    constraint_index=constraint_index,
                    description_chars=len(description),
                )
            )

        raw_rows = raw_constraint.get("supported_candidates")
        if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
            raw_rows = ()
        supported_rows: list[dict[str, Any]] = []
        seen_occurrence_ids: set[str] = set()
        for support_index, raw_row in enumerate(raw_rows):
            if not isinstance(raw_row, Mapping):
                errors.append(
                    _error(
                        "occurrence_evidence_support_must_be_object",
                        operation_index,
                        set_id=set_id,
                        constraint_index=constraint_index,
                        support_index=support_index,
                    )
                )
                continue
            occurrence_id = str(raw_row.get("occurrence_id", "") or "").strip()
            unexpected_support_fields = sorted(
                set(raw_row) - _EVIDENCE_SUPPORT_FIELDS
            )
            if unexpected_support_fields:
                errors.append(
                    _error(
                        "occurrence_evidence_support_field_invalid",
                        operation_index,
                        set_id=set_id,
                        constraint_index=constraint_index,
                        support_index=support_index,
                        unexpected_fields=unexpected_support_fields,
                    )
                )
            if occurrence_id not in viable_set or occurrence_id in seen_occurrence_ids:
                errors.append(
                    _error(
                        "occurrence_evidence_support_candidate_invalid",
                        operation_index,
                        set_id=set_id,
                        constraint_index=constraint_index,
                        occurrence_id=occurrence_id,
                    )
                )
                continue
            seen_occurrence_ids.add(occurrence_id)
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
                        "occurrence_evidence_passage_not_visible",
                        operation_index,
                        set_id=set_id,
                        constraint_index=constraint_index,
                        occurrence_id=occurrence_id,
                    )
                )
                continue
            if not evidence_ids:
                errors.append(
                    _error(
                        "occurrence_evidence_support_requires_passage",
                        operation_index,
                        set_id=set_id,
                        constraint_index=constraint_index,
                        occurrence_id=occurrence_id,
                    )
                )
                continue
            supported_rows.append(
                {
                    "occurrence_id": occurrence_id,
                    "evidence_passage_ids": list(evidence_ids),
                }
            )
        normalized_constraints.append(
            {
                "constraint_id": constraint_id,
                "constraint_type": constraint_type,
                "description": description,
                "supported_candidates": supported_rows,
                "implicit_unknown_occurrence_ids": [
                    occurrence_id
                    for occurrence_id in viable_ids
                    if occurrence_id not in seen_occurrence_ids
                ],
            }
        )

    if errors:
        return None, errors
    return (
        OccurrenceEvidenceReport(
            set_id=set_id,
            constraints=tuple(normalized_constraints),
            implicit_unknown_support_count=sum(
                len(tuple(row.get("implicit_unknown_occurrence_ids", ()) or ()))
                for row in normalized_constraints
            ),
            scope_occurrence_ids=viable_ids,
            out_of_scope_occurrence_ids=tuple(
                dict.fromkeys(
                    str(value)
                    for value in out_of_scope_occurrence_ids
                    if str(value)
                )
            ),
        ),
        [],
    )


def aggregate_occurrence_evidence(
    report: OccurrenceEvidenceReport,
) -> OccurrenceSufficiencyDecision:
    counts = {occurrence_id: 0 for occurrence_id in report.scope_occurrence_ids}
    for constraint in report.constraints:
        for row in tuple(constraint.get("supported_candidates", ()) or ()):
            occurrence_id = str(row.get("occurrence_id", "") or "")
            if occurrence_id in counts:
                counts[occurrence_id] += 1
    support_counts = tuple(counts.items())
    ranked_counts = sorted(
        support_counts,
        key=lambda item: (-item[1], report.scope_occurrence_ids.index(item[0])),
    )
    best_support_count = ranked_counts[0][1] if ranked_counts else 0
    runner_up_support_count = ranked_counts[1][1] if len(ranked_counts) > 1 else 0
    leaders = tuple(
        occurrence_id
        for occurrence_id, count in ranked_counts
        if count == best_support_count
    )
    sufficient_ids = (
        leaders
        if best_support_count > 0
        and len(leaders) == 1
        and best_support_count - runner_up_support_count
        >= MIN_SUFFICIENCY_SUPPORT_MARGIN
        else ()
    )
    return OccurrenceSufficiencyDecision(
        set_id=report.set_id,
        verdict="sufficient" if sufficient_ids else "insufficient",
        sufficient_occurrence_ids=sufficient_ids,
        support_count_by_occurrence=support_counts,
        best_support_count=best_support_count,
        runner_up_support_count=runner_up_support_count,
        evidence_report_digest=report.digest,
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
