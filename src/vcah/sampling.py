from __future__ import annotations

from dataclasses import dataclass

from vcah.types import ClaimContract, CoverageSegment, Window


@dataclass(frozen=True)
class SamplingPlan:
    windows: tuple[Window, ...]
    frame_times_sec: tuple[float, ...]
    reason: str
    coverage_manifest: tuple[CoverageSegment, ...] = ()


def adaptive_sample_plan(
    *,
    claim_contract: ClaimContract,
    candidate_times_sec: tuple[float, ...] = (),
    current_coverage: tuple[CoverageSegment, ...] = (),
    uncertainty_reason: str = "",
    burst_delta_sec: float = 2.0,
) -> SamplingPlan:
    times: list[float] = []
    for candidate in candidate_times_sec:
        center = float(candidate)
        for delta in (-float(burst_delta_sec), 0.0, float(burst_delta_sec)):
            value = max(0.0, center + delta)
            if value not in times:
                times.append(value)

    windows = tuple(_uncovered_windows(current_coverage))
    reason = uncertainty_reason or ("candidate_burst" if times else "baseline")
    if claim_contract.required_scope in {"multi_window", "full_video"} and not windows and current_coverage:
        reason = "coverage_satisfied"
    return SamplingPlan(
        windows=windows,
        frame_times_sec=tuple(sorted(times)),
        reason=reason,
        coverage_manifest=current_coverage,
    )


def _uncovered_windows(coverage: tuple[CoverageSegment, ...]) -> tuple[Window, ...]:
    return tuple(Window(segment.start_sec, segment.end_sec) for segment in coverage if segment.coverage < 0.8)
