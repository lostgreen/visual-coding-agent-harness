from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Mapping, Sequence

from vcah.occurrence_negative_sidecar import stable_digest
from vcah.occurrence_visual_geometry import (
    PAIR_CATEGORIES,
    VISUAL_GEOMETRY_CONTRACT,
)


VISUAL_TIE_ANATOMY_CONTRACT = "wp15_1_zero_model_tie_loss_anatomy_v1"
DOMINANT_TIE_SHARE_MIN = 0.75
MAX_LOSS_CASES_FOR_FOLLOWUP = 1

TIE_CLASSES = (
    "both_supported_dominated",
    "neither_supported_dominated",
    "mixed_cancellation",
    "balanced_other",
)


def build_visual_tie_anatomy_report(
    geometry_report: Mapping[str, Any],
    *,
    expected_cases: int,
    expected_pairs: int,
    dominant_tie_share_min: float = DOMINANT_TIE_SHARE_MIN,
    max_loss_cases_for_followup: int = MAX_LOSS_CASES_FOR_FOLLOWUP,
) -> dict[str, Any]:
    geometry = geometry_report.get("paired_support_geometry", {})
    pair_rows = tuple(
        dict(row)
        for row in tuple(
            geometry.get("by_constraint", ())
            if isinstance(geometry, Mapping)
            else ()
        )
        if isinstance(row, Mapping)
    )
    by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        by_case[str(row.get("case_id", "") or "")].append(row)

    case_rows = []
    outcomes = Counter()
    tie_classes = Counter()
    for case_id in sorted(case_id for case_id in by_case if case_id):
        rows = by_case[case_id]
        counts = Counter(str(row.get("support_geometry", "") or "") for row in rows)
        matched_supported = (
            counts["matched_only_supported"] + counts["both_supported"]
        )
        mismatched_supported = (
            counts["mismatched_only_supported"] + counts["both_supported"]
        )
        margin = matched_supported - mismatched_supported
        outcome = "win" if margin > 0 else "loss" if margin < 0 else "tie"
        outcomes[outcome] += 1
        residual_class = None
        if outcome == "tie":
            residual_class = classify_tie(counts)
            tie_classes[residual_class] += 1
        elif outcome == "loss":
            residual_class = "reversed_unary_support"
        constraint_types = Counter(
            str(row.get("constraint_type", "") or "") for row in rows
        )
        case_rows.append(
            {
                "case_id": case_id,
                "constraint_count": len(rows),
                "matched_supported_count": matched_supported,
                "mismatched_supported_count": mismatched_supported,
                "margin": margin,
                "outcome": outcome,
                "support_geometry_counts": {
                    category: counts[category] for category in PAIR_CATEGORIES
                },
                "constraint_type_counts": dict(sorted(constraint_types.items())),
                "residual_class": residual_class,
                "constraints": [
                    {
                        "constraint_id": str(row.get("constraint_id", "") or ""),
                        "constraint_type": str(
                            row.get("constraint_type", "") or ""
                        ),
                        "support_geometry": str(
                            row.get("support_geometry", "") or ""
                        ),
                    }
                    for row in sorted(
                        rows,
                        key=lambda value: (
                            str(value.get("constraint_type", "") or ""),
                            str(value.get("constraint_id", "") or ""),
                        ),
                    )
                ],
            }
        )

    type_rows = []
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    for row in pair_rows:
        by_type[str(row.get("constraint_type", "") or "")][
            str(row.get("support_geometry", "") or "")
        ] += 1
    for constraint_type, counts in sorted(by_type.items()):
        total = sum(counts.values())
        matched_only = counts["matched_only_supported"]
        both = counts["both_supported"]
        mismatched_only = counts["mismatched_only_supported"]
        neither = counts["neither_supported"]
        type_rows.append(
            {
                "constraint_type": constraint_type,
                "count": total,
                "matched_only_count": matched_only,
                "both_supported_count": both,
                "mismatched_only_count": mismatched_only,
                "neither_supported_count": neither,
                "matched_only_rate": _rate(matched_only, total),
                "both_supported_rate": _rate(both, total),
                "mismatched_only_rate": _rate(mismatched_only, total),
                "neither_supported_rate": _rate(neither, total),
                "ambiguity_rate": _rate(both + neither, total),
                "discriminative_minus_adverse_rate": _rate(
                    matched_only - mismatched_only, total
                ),
            }
        )

    tie_count = outcomes["tie"]
    required_dominant_ties = (
        math.ceil(tie_count * dominant_tie_share_min) if tie_count else None
    )
    both_dominated = tie_classes["both_supported_dominated"]
    neither_dominated = tie_classes["neither_supported_dominated"]
    loss_count = outcomes["loss"]
    loss_gate = loss_count <= max_loss_cases_for_followup
    both_branch = bool(
        required_dominant_ties
        and both_dominated >= required_dominant_ties
        and loss_gate
    )
    neither_branch = bool(
        required_dominant_ties
        and neither_dominated >= required_dominant_ties
        and loss_gate
    )

    source_joint = geometry_report.get("case_level_joint_evidence", {})
    source_outcomes_match = bool(
        isinstance(source_joint, Mapping)
        and source_joint.get("wins") == outcomes["win"]
        and source_joint.get("ties") == outcomes["tie"]
        and source_joint.get("losses") == outcomes["loss"]
    )
    structural_checks = {
        "wp15_0_contract_matches": geometry_report.get("contract")
        == VISUAL_GEOMETRY_CONTRACT,
        "wp15_0_structural_gate_passed": geometry_report.get(
            "structural_gate_passed"
        )
        is True,
        "case_count_matches_expected": len(case_rows) == expected_cases,
        "pair_count_matches_expected": len(pair_rows) == expected_pairs,
        "pair_categories_valid": all(
            row.get("support_geometry") in PAIR_CATEGORIES for row in pair_rows
        ),
        "case_margins_reconcile": source_outcomes_match,
        "tie_classes_complete": sum(tie_classes.values()) == tie_count,
        "type_counts_reconcile": sum(row["count"] for row in type_rows)
        == len(pair_rows),
        "no_type_weights_learned": True,
        "zero_model_calls": True,
        "endpoint_values_not_validity_gates": True,
    }
    structural_gate_passed = bool(case_rows) and all(structural_checks.values())
    if not structural_gate_passed:
        decision = "STOP_INVALID_WP15_1_INPUT"
    elif both_branch:
        decision = "PROPOSE_WP16_A_COMPARATIVE_DISCRIMINATIVE_PROBE"
    elif neither_branch:
        decision = "PROPOSE_WP16_B_CONSTRAINT_CONDITIONED_ACQUISITION"
    else:
        decision = "STOP_CURRENT_VISUAL_INTEGRATION_LINE"

    return {
        "schema_version": "MMLifelongOccurrenceVisualTieAnatomyReportV1",
        "contract": VISUAL_TIE_ANATOMY_CONTRACT,
        "study": "WP15-1 zero-model tie and loss anatomy",
        "scope": "frozen39 mechanism-development exploratory diagnostic",
        "zero_model_calls": True,
        "qa_judge_run": False,
        "source_geometry_digest": stable_digest(geometry_report),
        "structural_checks": structural_checks,
        "structural_gate_passed": structural_gate_passed,
        "case_level": {
            "case_count": len(case_rows),
            "wins": outcomes["win"],
            "ties": outcomes["tie"],
            "losses": outcomes["loss"],
            "cases": case_rows,
        },
        "residual_anatomy": {
            "tie_count": tie_count,
            "loss_count": loss_count,
            "tie_class_counts": {
                tie_class: tie_classes[tie_class] for tie_class in TIE_CLASSES
            },
            "tie_class_rates": {
                tie_class: _rate(tie_classes[tie_class], tie_count)
                for tie_class in TIE_CLASSES
            },
            "loss_case_count": loss_count,
        },
        "constraint_type_discriminativeness": type_rows,
        "branch_thresholds": {
            "dominant_tie_share_min": dominant_tie_share_min,
            "required_dominant_ties": required_dominant_ties,
            "max_loss_cases_for_followup": max_loss_cases_for_followup,
            "frozen_before_wp15_1_anatomy": True,
            "type_weights_learned": False,
        },
        "branch_checks": {
            "both_supported_dominated_branch": both_branch,
            "neither_supported_dominated_branch": neither_branch,
            "loss_gate_passed": loss_gate,
        },
        "decision": decision,
        "underpowered": True,
        "day_test140_accessed": False,
        "week_accessed": False,
    }


def classify_tie(counts: Mapping[str, int]) -> str:
    both = int(counts.get("both_supported", 0) or 0)
    neither = int(counts.get("neither_supported", 0) or 0)
    cancellation = int(counts.get("matched_only_supported", 0) or 0) + int(
        counts.get("mismatched_only_supported", 0) or 0
    )
    if both > max(neither, cancellation):
        return "both_supported_dominated"
    if neither > max(both, cancellation):
        return "neither_supported_dominated"
    if cancellation > max(both, neither):
        return "mixed_cancellation"
    return "balanced_other"


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
