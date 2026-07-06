from __future__ import annotations

from vcah.sampling import adaptive_sample_plan
from vcah.types import ClaimContract, CoverageSegment


def test_adaptive_sampler_adds_candidate_centered_burst() -> None:
    plan = adaptive_sample_plan(
        claim_contract=ClaimContract(required_scope="window", required_observability=("visual",)),
        candidate_times_sec=(100.0,),
        burst_delta_sec=3.0,
    )

    assert plan.frame_times_sec == (97.0, 100.0, 103.0)
    assert plan.reason == "candidate_burst"


def test_adaptive_sampler_prioritizes_uncovered_intervals() -> None:
    plan = adaptive_sample_plan(
        claim_contract=ClaimContract(required_scope="multi_window", required_observability=("visual",)),
        current_coverage=(
            CoverageSegment("win_1", 0.0, 10.0, "visual", 1.0),
            CoverageSegment("win_2", 10.0, 20.0, "visual", 0.25),
        ),
        uncertainty_reason="insufficient_scope",
    )

    assert [(window.start_sec, window.end_sec) for window in plan.windows] == [(10.0, 20.0)]
    assert plan.reason == "insufficient_scope"
    assert plan.coverage_manifest[1].request_id == "win_2"
