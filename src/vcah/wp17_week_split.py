"""Outcome-independent Week development and query-holdout splits for WP17."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence


WP17_WEEK_SPLIT_SCHEMA = "MMLifelongWP17WeekQuerySplitV1"
WP17_WEEK_SPLIT_PROTOCOL_SCHEMA = "MMLifelongWP17WeekQuerySplitProtocolV1"
_PERSISTED_CASE_FIELDS = ("case_id", "question_type", "case_sha256")


def build_week_query_manifests(
    cases: Sequence[Mapping[str, Any]],
    *,
    dev_count: int = 60,
    expected_count: int = 200,
    seed: int = 20260830,
) -> dict[str, dict[str, Any]]:
    rows = _normalize_cases(cases)
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} Week cases, found {len(rows)}")
    if not 0 < dev_count < expected_count:
        raise ValueError("Week development count must be between zero and the universe size")

    type_counts = Counter(row["question_type"] for row in rows)
    quotas = _largest_remainder_quotas(type_counts, total=expected_count, selected=dev_count)
    by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_type[row["question_type"]].append(row)

    dev_ids: set[str] = set()
    for question_type, group in by_type.items():
        ranked = sorted(
            group,
            key=lambda row: (
                _selection_rank(seed, question_type, row["case_id"]),
                row["case_id"],
            ),
        )
        dev_ids.update(row["case_id"] for row in ranked[: quotas[question_type]])

    dev_rows = sorted(
        (row for row in rows if row["case_id"] in dev_ids),
        key=lambda row: row["case_id"],
    )
    holdout_rows = sorted(
        (row for row in rows if row["case_id"] not in dev_ids),
        key=lambda row: row["case_id"],
    )
    universe_ids = sorted(row["case_id"] for row in rows)
    common = {
        "schema_version": WP17_WEEK_SPLIT_SCHEMA,
        "benchmark_subset": "week",
        "selection_strategy": "question_type_stratified_sha256_rank",
        "selection_seed": int(seed),
        "selection_source_fields": ["case_id", "question_type"],
        "selection_is_outcome_independent": True,
        "agent_visible_benchmark_annotations": False,
        "question_answer_clue_fields_persisted": False,
        "week_universe_count": len(rows),
        "week_universe_case_id_digest": _digest(universe_ids),
    }
    dev_manifest = {
        **common,
        "protocol_role": "cross_domain_method_development",
        "method_selection_allowed": True,
        "eligible_for_final_query_holdout_claim": False,
        "selected_count": len(dev_rows),
        "question_type_quota": {
            key: {
                "universe": type_counts[key],
                "development": quotas[key],
                "holdout": type_counts[key] - quotas[key],
            }
            for key in sorted(type_counts)
        },
        "cases": dev_rows,
    }
    holdout_manifest = {
        **common,
        "protocol_role": "final_query_holdout_after_method_freeze",
        "method_selection_allowed": False,
        "eligible_for_final_query_holdout_claim": True,
        "eligible_for_unseen_video_claim": False,
        "shared_video_corpus_with_development": True,
        "access_only_after_method_freeze": True,
        "selected_count": len(holdout_rows),
        "cases": holdout_rows,
    }
    checks = _checks(
        dev_manifest,
        holdout_manifest,
        expected_count=expected_count,
        dev_count=dev_count,
    )
    protocol = {
        "schema_version": WP17_WEEK_SPLIT_PROTOCOL_SCHEMA,
        "decision": (
            "WP17_WEEK_QUERY_SPLIT_FROZEN"
            if all(checks.values())
            else "WP17_WEEK_QUERY_SPLIT_FAILED"
        ),
        "structural_gate_passed": all(checks.values()),
        "checks": checks,
        "development_count": len(dev_rows),
        "holdout_count": len(holdout_rows),
        "selection_seed": int(seed),
        "month_video_dependency": False,
        "claim_policy": (
            "Week-dev60 may guide method selection. Week-holdout140 remains sealed "
            "until method freeze and supports only a query-level holdout claim because "
            "both partitions share the same Week video corpus."
        ),
    }
    return {
        "week_dev": dev_manifest,
        "week_holdout": holdout_manifest,
        "protocol": protocol,
    }


def _normalize_cases(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in cases:
        case_id = str(raw.get("case_id", "")).strip()
        if not case_id:
            raise ValueError("Week split contains an empty case ID")
        if case_id in seen:
            raise ValueError(f"Week split contains duplicate case ID {case_id}")
        seen.add(case_id)
        rows.append(
            {
                "case_id": case_id,
                "question_type": str(raw.get("question_type") or "Unknown"),
                "case_sha256": str(raw.get("case_sha256") or ""),
            }
        )
    return rows


def _largest_remainder_quotas(
    counts: Mapping[str, int], *, total: int, selected: int
) -> dict[str, int]:
    quotas = {
        key: int(count) * int(selected) // int(total)
        for key, count in counts.items()
    }
    remaining = int(selected) - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda key: (
            -(int(counts[key]) * int(selected) % int(total)),
            key,
        ),
    )
    for key in order[:remaining]:
        quotas[key] += 1
    if sum(quotas.values()) != selected:
        raise ValueError("Week question-type quotas do not sum to the development count")
    if any(quotas[key] > counts[key] for key in counts):
        raise ValueError("Week question-type quota exceeds its stratum size")
    return quotas


def _selection_rank(seed: int, question_type: str, case_id: str) -> str:
    payload = f"{int(seed)}\0{question_type}\0{case_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _checks(
    dev: Mapping[str, Any],
    holdout: Mapping[str, Any],
    *,
    expected_count: int,
    dev_count: int,
) -> dict[str, bool]:
    dev_rows = tuple(dev["cases"])
    holdout_rows = tuple(holdout["cases"])
    dev_ids = {str(row["case_id"]) for row in dev_rows}
    holdout_ids = {str(row["case_id"]) for row in holdout_rows}
    persisted_fields = set(_PERSISTED_CASE_FIELDS)
    checks = {
        "development_count_exact": len(dev_rows) == dev_count,
        "holdout_count_exact": len(holdout_rows) == expected_count - dev_count,
        "partitions_disjoint": not (dev_ids & holdout_ids),
        "partitions_complete": len(dev_ids | holdout_ids) == expected_count,
        "case_fields_are_whitelisted": all(
            set(row) == persisted_fields for row in dev_rows + holdout_rows
        ),
        "development_allows_method_selection": dev.get("method_selection_allowed")
        is True,
        "holdout_forbids_method_selection": holdout.get("method_selection_allowed")
        is False,
        "holdout_is_query_not_video_claim": holdout.get(
            "eligible_for_unseen_video_claim"
        )
        is False,
        "shared_video_corpus_declared": holdout.get(
            "shared_video_corpus_with_development"
        )
        is True,
        "benchmark_annotations_hidden_from_agent": dev.get(
            "agent_visible_benchmark_annotations"
        )
        is False
        and holdout.get("agent_visible_benchmark_annotations") is False,
    }
    return checks


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
