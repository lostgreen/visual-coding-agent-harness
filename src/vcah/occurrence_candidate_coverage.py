from __future__ import annotations

from collections import Counter
from math import ceil
from statistics import mean
from typing import Any, Mapping, Sequence

from vcah.caption_occurrence import (
    DEFAULT_OCCURRENCE_GAP_SEC,
    build_caption_occurrence_set,
)


CANDIDATE_COVERAGE_CONTRACT = "WP16-0-candidate-coverage-v1"
RECALL_KS = (1, 3, 5, 10, 20)
OBSERVED_RECALL_KS = (1, 3, 5)
BOUNDARY_NEAR_MISS_SEC = 2.0
DOMINANT_FAILURE_SHARE = 0.60

FAILURE_CATEGORIES = (
    "retrieved_then_pruned_or_retired",
    "retrieved_but_outside_topK",
    "representation_or_boundary_mismatch",
    "never_retrieved_top20",
)


def build_candidate_coverage_report(
    cases: Sequence[Mapping[str, Any]],
    *,
    expected_cases: int,
    expected_candidate_present: int,
    expected_candidate_absent: int,
) -> dict[str, Any]:
    normalized = tuple(
        sorted((dict(case) for case in cases), key=lambda row: row["case_id"])
    )
    structural_errors = _structural_errors(
        normalized,
        expected_cases=expected_cases,
        expected_candidate_present=expected_candidate_present,
        expected_candidate_absent=expected_candidate_absent,
    )
    case_rows = tuple(_case_row(case) for case in normalized)
    present_rows = tuple(row for row in case_rows if row["candidate_present_final"])
    absent_rows = tuple(row for row in case_rows if not row["candidate_present_final"])

    category_counts = Counter(
        str(row["failure_category"])
        for row in absent_rows
        if row.get("failure_category")
    )
    category_summary = {
        category: {
            "count": category_counts[category],
            "rate": _rate(category_counts[category], len(absent_rows)),
            "case_ids": [
                row["case_id"]
                for row in absent_rows
                if row["failure_category"] == category
            ],
        }
        for category in FAILURE_CATEGORIES
    }
    coverage_count = sum(
        category_counts[category] for category in FAILURE_CATEGORIES[:2]
    )
    query_or_representation_count = sum(
        category_counts[category] for category in FAILURE_CATEGORIES[2:]
    )
    dominance_count = ceil(DOMINANT_FAILURE_SHARE * len(absent_rows))
    structural_gate_passed = not structural_errors
    if not structural_gate_passed:
        decision = "INVALID_WP16_0_AUDIT"
    elif coverage_count >= dominance_count:
        decision = "PROPOSE_COVERAGE_PRESERVING_DIVERSE_OCCURRENCE_SET"
    elif query_or_representation_count >= dominance_count:
        decision = "PROPOSE_OCCURRENCE_AWARE_QUERY_OR_REPRESENTATION"
    else:
        decision = "STOP_MIXED_FAILURES_NO_UNIFIED_PATCH"

    crowding = _crowding_summary(normalized)
    query_coverage = _query_coverage_summary(absent_rows)
    observed_depths = Counter(
        int(packet.get("observed_top_k", 0) or 0)
        for case in normalized
        for packet in tuple(case.get("packets", ()) or ())
    )
    return {
        "schema_version": 1,
        "contract": CANDIDATE_COVERAGE_CONTRACT,
        "decision": decision,
        "structural_gate_passed": structural_gate_passed,
        "structural_errors": structural_errors,
        "protocol": {
            "observed_packet_depth_counts": {
                str(depth): count for depth, count in sorted(observed_depths.items())
            },
            "counterfactual_replay_depth": 20,
            "recall_ks": list(RECALL_KS),
            "boundary_near_miss_sec": BOUNDARY_NEAR_MISS_SEC,
            "occurrence_gap_sec": DEFAULT_OCCURRENCE_GAP_SEC,
            "dominant_failure_share": DOMINANT_FAILURE_SHARE,
            "endpoint_values_are_gates": False,
            "k_tuned_on_outcomes": False,
        },
        "cohort": {
            "case_count": len(case_rows),
            "candidate_present_count": len(present_rows),
            "candidate_absent_count": len(absent_rows),
            "candidate_present_case_ids": [row["case_id"] for row in present_rows],
            "candidate_absent_case_ids": [row["case_id"] for row in absent_rows],
        },
        "recall": {
            "final_scoped": _recall_summary(
                case_rows, "final_best_gold_rank", OBSERVED_RECALL_KS
            ),
            "observed_trajectory": _identified_recall_summary(
                case_rows, "observed_best_gold_rank", RECALL_KS
            ),
            "counterfactual_top20": _recall_summary(
                case_rows, "replay_best_gold_rank", RECALL_KS
            ),
            "observed_recall_uses_depth_eligible_denominators": True,
        },
        "candidate_absent_failure_partition": {
            "categories": category_summary,
            "partition_complete": sum(category_counts.values()) == len(absent_rows),
        },
        "duplicate_crowding": crowding,
        "query_coverage": query_coverage,
        "branch_evidence": {
            "dominance_case_count_required": dominance_count,
            "coverage_preservation_case_count": coverage_count,
            "coverage_preservation_rate": _rate(coverage_count, len(absent_rows)),
            "query_or_representation_case_count": query_or_representation_count,
            "query_or_representation_rate": _rate(
                query_or_representation_count, len(absent_rows)
            ),
            "coverage_cases_with_observed_top5_crowding": sum(
                bool(row["observed_top5_slots_lost"])
                for row in absent_rows
                if row["failure_category"] in FAILURE_CATEGORIES[:2]
            ),
            "never_retrieved_cases_with_query_episode_collapse": sum(
                bool(row["query_episode_collapse"])
                for row in absent_rows
                if row["failure_category"] == "never_retrieved_top20"
            ),
        },
        "case_level": list(case_rows),
        "limitations": [
            "Observed packets have heterogeneous depths; each observed Recall@K uses only cases with a recorded packet of depth at least K.",
            "The depth-20 replay is admitted only when deterministic replay exactly reproduces every packet at its recorded retrieval depth.",
            "Query text is never emitted; exact normalized-query diversity and retrieved-episode collapse are reported instead of semantic-template judgments.",
            "A two-second interval gap is labeled boundary-near-miss only and is not treated as proof of a representation failure.",
        ],
    }


def _structural_errors(
    cases: Sequence[Mapping[str, Any]],
    *,
    expected_cases: int,
    expected_candidate_present: int,
    expected_candidate_absent: int,
) -> list[str]:
    errors: list[str] = []
    case_ids = [str(case.get("case_id", "") or "") for case in cases]
    if len(cases) != expected_cases:
        errors.append(f"case_count:{len(cases)}!={expected_cases}")
    if len(set(case_ids)) != len(case_ids) or any(not case_id for case_id in case_ids):
        errors.append("case_ids_not_unique_and_nonempty")
    present_count = 0
    for case in cases:
        case_id = str(case.get("case_id", "") or "")
        clues = tuple(case.get("clues", ()) or ())
        packets = tuple(case.get("packets", ()) or ())
        final_candidates = tuple(case.get("final_candidates", ()) or ())
        if not clues:
            errors.append(f"{case_id}:missing_clues")
        if not packets:
            errors.append(f"{case_id}:missing_retrieval_packets")
        if not final_candidates:
            errors.append(f"{case_id}:missing_final_candidate_set")
        if not bool(case.get("replay_available")):
            errors.append(f"{case_id}:missing_depth20_replay")
        for index, packet in enumerate(packets):
            if int(packet.get("observed_top_k", 0) or 0) < 1:
                errors.append(f"{case_id}:packet_{index}:invalid_observed_depth")
            if not bool(packet.get("recorded_depth_replay_match")):
                errors.append(
                    f"{case_id}:packet_{index}:recorded_depth_replay_mismatch"
                )
            if not bool(packet.get("index_digest_match")):
                errors.append(f"{case_id}:packet_{index}:index_digest_mismatch")
        if _best_gold_rank(final_candidates, clues) is not None:
            present_count += 1
    absent_count = len(cases) - present_count
    if present_count != expected_candidate_present:
        errors.append(
            f"candidate_present_count:{present_count}!={expected_candidate_present}"
        )
    if absent_count != expected_candidate_absent:
        errors.append(
            f"candidate_absent_count:{absent_count}!={expected_candidate_absent}"
        )
    return errors


def _case_row(case: Mapping[str, Any]) -> dict[str, Any]:
    case_id = str(case.get("case_id", "") or "")
    clues = tuple(case.get("clues", ()) or ())
    packets = tuple(case.get("packets", ()) or ())
    final_candidates = tuple(case.get("final_candidates", ()) or ())
    observed_candidates = tuple(
        candidate
        for packet in packets
        for candidate in tuple(packet.get("observed_candidates", ()) or ())
        if isinstance(candidate, Mapping)
    )
    replay_candidates = tuple(
        candidate
        for packet in packets
        for candidate in tuple(packet.get("replay_candidates", ()) or ())
        if isinstance(candidate, Mapping)
    )
    final_rank = _best_gold_rank(final_candidates, clues)
    observed_rank = _best_gold_rank(observed_candidates, clues)
    replay_rank = _best_gold_rank(replay_candidates, clues)
    nearest_gap = _nearest_gap((*observed_candidates, *replay_candidates), clues)
    candidate_present = final_rank is not None
    historical_gold_set_ids = [
        str(packet.get("attempt_id", "") or "")
        for packet in packets
        if _best_gold_rank(tuple(packet.get("observed_candidates", ()) or ()), clues)
        is not None
    ]
    failure_category = ""
    failure_subcategory = ""
    if not candidate_present:
        if observed_rank is not None:
            failure_category = "retrieved_then_pruned_or_retired"
            retired_ids = set(case.get("retired_set_ids", ()) or ())
            failure_subcategory = (
                "retired"
                if any(set_id in retired_ids for set_id in historical_gold_set_ids)
                else "not_in_final_scope"
            )
        elif replay_rank is not None:
            failure_category = "retrieved_but_outside_topK"
            observed_max_depth = max(
                (int(packet.get("observed_top_k", 0) or 0) for packet in packets),
                default=0,
            )
            failure_subcategory = (
                "counterfactual_rank_beyond_max_observed_depth"
                if replay_rank > observed_max_depth
                else "counterfactual_depth_induced_prefix_shift"
            )
        elif nearest_gap is not None and nearest_gap <= BOUNDARY_NEAR_MISS_SEC:
            failure_category = "representation_or_boundary_mismatch"
            failure_subcategory = "boundary_near_miss"
        else:
            failure_category = "never_retrieved_top20"
            failure_subcategory = "representation_not_identifiable"

    observed_slots_lost = sum(
        _packet_slot_loss(packet, source="observed", k=5) for packet in packets
    )
    replay_slots_lost = {
        str(k): sum(
            _packet_slot_loss(packet, source="replay", k=k) for packet in packets
        )
        for k in RECALL_KS
    }
    query_episode_count = _episode_count(
        tuple(case.get("query_top1_candidates", ()) or ())
    )
    query_context_count = int(case.get("query_context_count", 0) or 0)
    query_episode_collapse = bool(
        query_context_count >= 2
        and query_episode_count == 1
        and len(tuple(case.get("query_top1_candidates", ()) or ()))
        == query_context_count
    )
    return {
        "case_id": case_id,
        "candidate_present_final": candidate_present,
        "final_best_gold_rank": final_rank,
        "observed_best_gold_rank": observed_rank,
        "replay_best_gold_rank": replay_rank,
        "observed_max_depth": max(
            (int(packet.get("observed_top_k", 0) or 0) for packet in packets),
            default=0,
        ),
        "historical_gold_set_ids": historical_gold_set_ids,
        "nearest_nonoverlap_gap_sec": nearest_gap,
        "failure_category": failure_category,
        "failure_subcategory": failure_subcategory,
        "observed_top5_slots_lost": observed_slots_lost,
        "replay_slots_lost_by_k": replay_slots_lost,
        "normalized_query_count": int(case.get("normalized_query_count", 0) or 0),
        "query_context_count": query_context_count,
        "query_top1_episode_count": query_episode_count,
        "query_episode_collapse": query_episode_collapse,
        "semantic_template_collapse_identifiable": False,
    }


def _recall_summary(
    rows: Sequence[Mapping[str, Any]], rank_key: str, ks: Sequence[int]
) -> dict[str, Any]:
    return {
        f"at_{k}": {
            "count": sum(
                isinstance(row.get(rank_key), int) and int(row[rank_key]) <= k
                for row in rows
            ),
            "rate": _rate(
                sum(
                    isinstance(row.get(rank_key), int) and int(row[rank_key]) <= k
                    for row in rows
                ),
                len(rows),
            ),
            "eligible_case_count": len(rows),
            "cohort_coverage": 1.0 if rows else None,
        }
        for k in ks
    }


def _identified_recall_summary(
    rows: Sequence[Mapping[str, Any]], rank_key: str, ks: Sequence[int]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for k in ks:
        eligible = [
            row for row in rows if int(row.get("observed_max_depth", 0) or 0) >= k
        ]
        count = sum(
            isinstance(row.get(rank_key), int) and int(row[rank_key]) <= k
            for row in eligible
        )
        summary[f"at_{k}"] = {
            "count": count,
            "rate": _rate(count, len(eligible)),
            "eligible_case_count": len(eligible),
            "cohort_coverage": _rate(len(eligible), len(rows)),
        }
    return summary


def _crowding_summary(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    observed = _crowding_at_k(cases, source="observed", k=5)
    replay = {f"at_{k}": _crowding_at_k(cases, source="replay", k=k) for k in RECALL_KS}
    return {
        "definition": "raw retrieval slots collapsed into occurrence clusters with the frozen 120-second gap",
        "observed_first5": observed,
        "counterfactual_top20": replay,
    }


def _crowding_at_k(
    cases: Sequence[Mapping[str, Any]], *, source: str, k: int
) -> dict[str, Any]:
    hit_slots = 0
    occurrence_clusters = 0
    affected_cases = 0
    for case in cases:
        case_loss = 0
        for packet in tuple(case.get("packets", ()) or ()):
            hits = _ranked_hits(packet, source=source, k=k)
            if not hits:
                continue
            cluster_count = len(
                tuple(build_caption_occurrence_set(hits).get("candidates", ()) or ())
            )
            hit_slots += len(hits)
            occurrence_clusters += cluster_count
            case_loss += max(0, len(hits) - cluster_count)
        affected_cases += bool(case_loss)
    slots_lost = max(0, hit_slots - occurrence_clusters)
    return {
        "hit_slots": hit_slots,
        "occurrence_clusters": occurrence_clusters,
        "slots_consumed_by_same_occurrence": slots_lost,
        "slot_loss_rate": _rate(slots_lost, hit_slots),
        "affected_case_count": affected_cases,
        "affected_case_rate": _rate(affected_cases, len(cases)),
    }


def _query_coverage_summary(absent_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not absent_rows:
        return {
            "candidate_absent_case_count": 0,
            "mean_normalized_query_count": None,
            "mean_top1_episode_count": None,
            "episode_collapse_count": 0,
            "episode_collapse_rate": None,
        }
    collapsed = sum(bool(row["query_episode_collapse"]) for row in absent_rows)
    return {
        "candidate_absent_case_count": len(absent_rows),
        "mean_normalized_query_count": mean(
            int(row["normalized_query_count"]) for row in absent_rows
        ),
        "mean_query_context_count": mean(
            int(row["query_context_count"]) for row in absent_rows
        ),
        "mean_top1_episode_count": mean(
            int(row["query_top1_episode_count"]) for row in absent_rows
        ),
        "episode_collapse_count": collapsed,
        "episode_collapse_rate": _rate(collapsed, len(absent_rows)),
        "semantic_template_collapse_identifiable": False,
    }


def _packet_slot_loss(packet: Mapping[str, Any], *, source: str, k: int) -> int:
    hits = _ranked_hits(packet, source=source, k=k)
    if not hits:
        return 0
    clusters = tuple(build_caption_occurrence_set(hits).get("candidates", ()) or ())
    return max(0, len(hits) - len(clusters))


def _ranked_hits(
    packet: Mapping[str, Any], *, source: str, k: int
) -> tuple[Mapping[str, Any], ...]:
    key = "observed_hits" if source == "observed" else "replay_hits"
    hits = tuple(
        hit for hit in tuple(packet.get(key, ()) or ()) if isinstance(hit, Mapping)
    )
    return tuple(
        hit
        for _, hit in sorted(
            (
                (int(hit.get("rank", index + 1) or index + 1), hit)
                for index, hit in enumerate(hits)
            ),
            key=lambda item: item[0],
        )
        if int(hit.get("rank", 1) or 1) <= k
    )


def _best_gold_rank(
    candidates: Sequence[Mapping[str, Any]], clues: Sequence[Sequence[float]]
) -> int | None:
    ranks = [
        int(candidate.get("rank", index + 1) or index + 1)
        for index, candidate in enumerate(candidates)
        if _candidate_is_gold(candidate, clues)
    ]
    return min(ranks) if ranks else None


def _candidate_is_gold(
    candidate: Mapping[str, Any], clues: Sequence[Sequence[float]]
) -> bool:
    interval = _interval(candidate.get("time_range", ()))
    return bool(
        interval
        and any(
            (clue_interval := _interval(clue)) is not None
            and _overlap(interval, clue_interval) > 0
            for clue in clues
        )
    )


def _nearest_gap(
    candidates: Sequence[Mapping[str, Any]], clues: Sequence[Sequence[float]]
) -> float | None:
    gaps = []
    for candidate in candidates:
        interval = _interval(candidate.get("time_range", ()))
        if interval is None:
            continue
        for clue in clues:
            clue_interval = _interval(clue)
            if clue_interval is not None:
                gaps.append(_interval_gap(interval, clue_interval))
    positive = [gap for gap in gaps if gap > 0]
    return min(positive) if positive else None


def _episode_count(candidates: Sequence[Mapping[str, Any]]) -> int:
    rows = [
        candidate
        for candidate in candidates
        if _interval(candidate.get("time_range", ())) is not None
    ]
    if not rows:
        return 0
    parents = list(range(len(rows)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if _same_episode(rows[left], rows[right]):
                union(left, right)
    return len({root(index) for index in range(len(rows))})


def _same_episode(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_sources = set(left.get("source_video_ids", ()) or ())
    right_sources = set(right.get("source_video_ids", ()) or ())
    if left_sources and right_sources and not left_sources.intersection(right_sources):
        return False
    left_segments = set(left.get("segment_ids", ()) or ())
    right_segments = set(right.get("segment_ids", ()) or ())
    if (
        not left_sources
        and not right_sources
        and left_segments
        and right_segments
        and not left_segments.intersection(right_segments)
    ):
        return False
    left_interval = _interval(left.get("time_range", ()))
    right_interval = _interval(right.get("time_range", ()))
    return bool(
        left_interval
        and right_interval
        and _interval_gap(left_interval, right_interval) <= DEFAULT_OCCURRENCE_GAP_SEC
    )


def _interval(value: Any) -> tuple[float, float] | None:
    try:
        if len(value) != 2:
            return None
        start, end = sorted((float(value[0]), float(value[1])))
    except (TypeError, ValueError):
        return None
    return (start, end) if end > start else None


def _overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def _interval_gap(left: tuple[float, float], right: tuple[float, float]) -> float:
    if _overlap(left, right) > 0:
        return 0.0
    return min(abs(left[1] - right[0]), abs(right[1] - left[0]))


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
