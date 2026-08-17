from __future__ import annotations

from vcah.occurrence_candidate_coverage import build_candidate_coverage_report


CLUE = [[100.0, 110.0]]


def _candidate(
    occurrence_id: str,
    start: float,
    end: float,
    *,
    rank: int = 1,
) -> dict:
    return {
        "occurrence_id": occurrence_id,
        "rank": rank,
        "time_range": [start, end],
        "source_video_ids": ["video-a"],
        "segment_ids": ["segment-a"],
        "passage_ids": [f"passage-{occurrence_id}"],
    }


def _hit(passage_id: str, start: float, end: float, *, rank: int) -> dict:
    return {
        "passage_id": passage_id,
        "caption_id": f"caption-{passage_id}",
        "rank": rank,
        "fused_score": 1.0 / rank,
        "virtual_start_sec": start,
        "virtual_end_sec": end,
        "metadata": {
            "source_video_ids": ["video-a"],
            "source_segments": ["segment-a"],
        },
    }


def _packet(
    attempt_id: str,
    observed_candidates: list[dict],
    replay_candidates: list[dict] | None = None,
    *,
    prefix_match: bool = True,
) -> dict:
    observed_hits = [
        _hit(
            f"{attempt_id}-hit-{index}",
            200.0 + index * 10,
            205.0 + index * 10,
            rank=index,
        )
        for index in range(1, 6)
    ]
    replay_hits = observed_hits + [
        _hit(
            f"{attempt_id}-hit-{index}",
            400.0 + index * 10,
            405.0 + index * 10,
            rank=index,
        )
        for index in range(6, 21)
    ]
    return {
        "attempt_id": attempt_id,
        "observed_top_k": 5,
        "recorded_depth_replay_match": prefix_match,
        "index_digest_match": True,
        "observed_hits": observed_hits,
        "observed_candidates": observed_candidates,
        "replay_hits": replay_hits,
        "replay_candidates": replay_candidates or observed_candidates,
    }


def _case(
    case_id: str,
    *,
    final: list[dict] | None = None,
    observed: list[dict] | None = None,
    replay: list[dict] | None = None,
    retired: bool = False,
    prefix_match: bool = True,
    query_top1: list[dict] | None = None,
) -> dict:
    far = [_candidate(f"{case_id}-far", 300.0, 310.0)]
    attempt_id = f"{case_id}-attempt"
    return {
        "case_id": case_id,
        "clues": CLUE,
        "packets": [
            _packet(
                attempt_id,
                observed or far,
                replay or observed or far,
                prefix_match=prefix_match,
            )
        ],
        "final_candidates": final or far,
        "retired_set_ids": [attempt_id] if retired else [],
        "normalized_query_count": 2,
        "query_context_count": 2,
        "query_top1_candidates": query_top1
        or [
            _candidate(f"{case_id}-q1", 300.0, 305.0),
            _candidate(f"{case_id}-q2", 306.0, 311.0),
        ],
        "replay_available": True,
    }


def test_candidate_coverage_partitions_absent_cases_without_overclaiming() -> None:
    cases = [
        _case(
            "present",
            final=[_candidate("present-gold", 101.0, 105.0)],
            observed=[_candidate("present-gold", 101.0, 105.0)],
        ),
        _case(
            "retired",
            observed=[_candidate("retired-gold", 102.0, 106.0, rank=2)],
            retired=True,
        ),
        _case(
            "outside",
            replay=[_candidate("outside-gold", 103.0, 107.0, rank=6)],
        ),
        _case(
            "boundary",
            observed=[_candidate("boundary-near", 90.0, 98.0)],
            replay=[_candidate("boundary-near", 90.0, 98.0)],
        ),
        _case("never"),
    ]

    report = build_candidate_coverage_report(
        cases,
        expected_cases=5,
        expected_candidate_present=1,
        expected_candidate_absent=4,
    )

    assert report["structural_gate_passed"] is True
    assert report["decision"] == "STOP_MIXED_FAILURES_NO_UNIFIED_PATCH"
    counts = report["candidate_absent_failure_partition"]["categories"]
    assert {key: value["count"] for key, value in counts.items()} == {
        "retrieved_then_pruned_or_retired": 1,
        "retrieved_but_outside_topK": 1,
        "representation_or_boundary_mismatch": 1,
        "never_retrieved_top20": 1,
    }
    assert report["candidate_absent_failure_partition"]["partition_complete"] is True
    assert report["recall"]["final_scoped"]["at_1"]["count"] == 1
    assert report["recall"]["observed_trajectory"]["at_3"]["count"] == 2
    assert report["recall"]["counterfactual_top20"]["at_10"]["count"] == 3
    assert report["recall"]["observed_recall_uses_depth_eligible_denominators"] is True


def test_candidate_coverage_selects_only_a_dominant_frozen_branch() -> None:
    coverage_cases = [
        _case(
            f"retired-{index}",
            observed=[_candidate(f"gold-{index}", 101.0, 105.0)],
            retired=True,
        )
        for index in range(3)
    ] + [_case("never-1"), _case("never-2")]
    coverage = build_candidate_coverage_report(
        coverage_cases,
        expected_cases=5,
        expected_candidate_present=0,
        expected_candidate_absent=5,
    )
    assert coverage["decision"] == "PROPOSE_COVERAGE_PRESERVING_DIVERSE_OCCURRENCE_SET"

    query_cases = [
        _case("never-1"),
        _case("never-2"),
        _case("never-3"),
        _case(
            "retired-1",
            observed=[_candidate("gold-1", 101.0, 105.0)],
            retired=True,
        ),
        _case(
            "retired-2",
            observed=[_candidate("gold-2", 101.0, 105.0)],
            retired=True,
        ),
    ]
    query = build_candidate_coverage_report(
        query_cases,
        expected_cases=5,
        expected_candidate_present=0,
        expected_candidate_absent=5,
    )
    assert query["decision"] == "PROPOSE_OCCURRENCE_AWARE_QUERY_OR_REPRESENTATION"


def test_candidate_coverage_detects_raw_hit_slot_crowding() -> None:
    case = _case("crowded")
    packet = case["packets"][0]
    packet["observed_hits"] = [
        _hit("near-a", 200.0, 205.0, rank=1),
        _hit("near-b", 210.0, 215.0, rank=2),
        _hit("far-a", 500.0, 505.0, rank=3),
        _hit("far-b", 700.0, 705.0, rank=4),
        _hit("far-c", 900.0, 905.0, rank=5),
    ]
    packet["replay_hits"] = list(packet["observed_hits"])
    report = build_candidate_coverage_report(
        [case],
        expected_cases=1,
        expected_candidate_present=0,
        expected_candidate_absent=1,
    )
    observed = report["duplicate_crowding"]["observed_first5"]
    assert observed["hit_slots"] == 5
    assert observed["occurrence_clusters"] == 4
    assert observed["slots_consumed_by_same_occurrence"] == 1
    assert observed["affected_case_count"] == 1


def test_candidate_coverage_rejects_nonreproducible_replay() -> None:
    report = build_candidate_coverage_report(
        [_case("bad-prefix", prefix_match=False)],
        expected_cases=1,
        expected_candidate_present=0,
        expected_candidate_absent=1,
    )
    assert report["structural_gate_passed"] is False
    assert report["decision"] == "INVALID_WP16_0_AUDIT"
    assert any(
        "recorded_depth_replay_mismatch" in error
        for error in report["structural_errors"]
    )
