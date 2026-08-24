from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from vcah.occurrence_anchor_recall import ANCHOR_RECALL_KS
from vcah.occurrence_field_index import OCCURRENCE_FIELDS


FIELD_ABLATION_REPORT_CONTRACT = "WP16-5-oracle-field-ablation-report-v1"


def build_field_ablation_report(
    cases: Sequence[Mapping[str, Any]],
    *,
    expected_cases: int,
    variant_order: Sequence[str],
) -> dict[str, Any]:
    rows = tuple(dict(case) for case in cases)
    variants = tuple(dict.fromkeys(str(value) for value in variant_order))
    case_ids = tuple(str(case.get("case_id", "") or "") for case in rows)
    checks = {
        "expected_case_count": len(rows) == int(expected_cases),
        "unique_case_ids": len(set(case_ids)) == len(case_ids),
        "single_anchor_subset_excludes_0115": all(
            not case_id.endswith("-0115") for case_id in case_ids
        ),
        "fields_frozen_before_ablation_outcomes": all(
            bool(case.get("fields_frozen_before_ablation_outcomes")) for case in rows
        ),
        "oracle_gold_occurrence_only_recorded": all(
            bool(case.get("oracle_gold_occurrence_only")) for case in rows
        ),
        "target_evidence_and_answer_excluded": all(
            bool(case.get("target_evidence_and_answer_excluded")) for case in rows
        ),
        "field_set_complete": all(
            set(case.get("field_names", ())) == set(OCCURRENCE_FIELDS)
            for case in rows
        ),
        "one_oracle_passage_per_case": all(
            bool(case.get("oracle_passage_id")) for case in rows
        ),
        "query_variants_complete": all(
            set(dict(case.get("ranks", {}))) == set(variants) for case in rows
        ),
        "rank_domain_valid": all(
            rank is None or _valid_rank(rank)
            for case in rows
            for rank in dict(case.get("ranks", {})).values()
        ),
    }
    summaries = {variant: _variant_summary(rows, variant) for variant in variants}
    baseline = "caption_only_hybrid"
    concat = "caption_plus_all_hybrid"
    separated = "field_rrf_hybrid"
    target = math.ceil(0.8 * max(1, len(rows)))
    concat_at5 = _recall_count(summaries.get(concat, {}), 5)
    separated_at5 = _recall_count(summaries.get(separated, {}), 5)
    if concat_at5 >= target:
        decision = "ORACLE_CONCAT_FIELDS_SUFFICIENT"
    elif separated_at5 >= target:
        decision = "ORACLE_FIELD_SEPARATION_REQUIRED"
    else:
        decision = "ORACLE_FIELDS_INSUFFICIENT"

    comparisons = {
        f"{variant}_vs_{baseline}_at5": _paired_at_k(
            rows,
            treatment=variant,
            control=baseline,
            k=5,
        )
        for variant in variants
        if variant != baseline
    }
    single_fields = {
        field_name: _recall_count(
            summaries.get(f"caption_plus_{field_name}_hybrid", {}), 5
        )
        for field_name in OCCURRENCE_FIELDS
    }
    best_single_count = max(single_fields.values(), default=0)
    best_single_fields = sorted(
        field_name
        for field_name, count in single_fields.items()
        if count == best_single_count
    )
    return {
        "contract": FIELD_ABLATION_REPORT_CONTRACT,
        "case_count": len(rows),
        "case_ids": list(case_ids),
        "structural_checks": checks,
        "structural_gate_passed": all(checks.values()),
        "endpoint_values_are_gates": False,
        "oracle_upper_bound": True,
        "decision": decision,
        "decision_thresholds": {"target_recall_at5_count": target},
        "variants": summaries,
        "comparisons": comparisons,
        "per_case": [
            {
                "case_id": str(case.get("case_id", "") or ""),
                "anchor_description": str(
                    case.get("anchor_description", "") or ""
                ),
                "oracle_passage_id": str(
                    case.get("oracle_passage_id", "") or ""
                ),
                "ranks": dict(case.get("ranks", {})),
            }
            for case in rows
        ],
        "diagnostics": {
            "single_field_hybrid_at5": single_fields,
            "best_single_field_at5_count": best_single_count,
            "best_single_fields": best_single_fields,
            "concat_all_at5_count": concat_at5,
            "field_separated_at5_count": separated_at5,
            "manual_query_parser_is_oracle": True,
            "manual_gold_document_fields_are_oracle": True,
            "non_gold_field_false_positives_are_not_measured": True,
        },
    }


def _variant_summary(
    cases: Sequence[Mapping[str, Any]],
    variant: str,
) -> dict[str, Any]:
    ranks = tuple(dict(case.get("ranks", {})).get(variant) for case in cases)
    valid = tuple(int(rank) for rank in ranks if _valid_rank(rank))
    return {
        "recall": {
            f"at_{k}": {
                "count": sum(_valid_rank(rank) and int(rank) <= k for rank in ranks),
                "case_count": len(cases),
                "rate": (
                    sum(_valid_rank(rank) and int(rank) <= k for rank in ranks)
                    / len(cases)
                    if cases
                    else None
                ),
            }
            for k in ANCHOR_RECALL_KS
        },
        "mean_reciprocal_rank": (
            sum(1.0 / rank for rank in valid) / len(cases) if cases else None
        ),
        "not_found_count": sum(rank is None for rank in ranks),
    }


def _paired_at_k(
    cases: Sequence[Mapping[str, Any]],
    *,
    treatment: str,
    control: str,
    k: int,
) -> dict[str, Any]:
    buckets = {"recovered": [], "regressed": [], "retained": [], "missed": []}
    for case in cases:
        ranks = dict(case.get("ranks", {}))
        control_hit = _valid_rank(ranks.get(control)) and int(ranks[control]) <= k
        treatment_hit = _valid_rank(ranks.get(treatment)) and int(ranks[treatment]) <= k
        case_id = str(case.get("case_id", "") or "")
        if treatment_hit and not control_hit:
            buckets["recovered"].append(case_id)
        elif control_hit and not treatment_hit:
            buckets["regressed"].append(case_id)
        elif control_hit and treatment_hit:
            buckets["retained"].append(case_id)
        else:
            buckets["missed"].append(case_id)
    return {
        "recovered_case_ids": buckets["recovered"],
        "regressed_case_ids": buckets["regressed"],
        "retained_case_ids": buckets["retained"],
        "missed_case_ids": buckets["missed"],
        "net_recovery": len(buckets["recovered"]) - len(buckets["regressed"]),
    }


def _recall_count(summary: Mapping[str, Any], k: int) -> int:
    return int(
        dict(dict(summary.get("recall", {})).get(f"at_{k}", {})).get(
            "count", 0
        )
        or 0
    )


def _valid_rank(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1
