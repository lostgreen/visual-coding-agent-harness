from __future__ import annotations

from dataclasses import asdict, dataclass

from vcah.evidence_state import EVIDENCE_KINDS

from vcah.types import ClaimContract, CoverageSegment, Window


@dataclass(frozen=True)
class SamplingPlan:
    windows: tuple[Window, ...]
    frame_times_sec: tuple[float, ...]
    reason: str
    coverage_manifest: tuple[CoverageSegment, ...] = ()


@dataclass(frozen=True)
class EvidenceSamplingProfile:
    evidence_kind: str
    fps: float
    max_frames: int
    max_window_sec: float | None = None
    same_material_second_read: bool = False
    min_probe_count: int = 0
    max_probe_count: int = 0
    perception_mode: str = "visual"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evidence_sampling_profile(
    evidence_kind: str,
    *,
    requested_fps: float = 0.5,
) -> EvidenceSamplingProfile:
    kind = str(evidence_kind or "generic").strip().casefold()
    if kind not in EVIDENCE_KINDS:
        raise ValueError(f"unsupported evidence_kind: {kind}")
    base_fps = _supported_fps(requested_fps)
    profiles = {
        "generic": EvidenceSamplingProfile(kind, base_fps, 96),
        "text_exact": EvidenceSamplingProfile(
            kind,
            2.0,
            72,
            max_window_sec=36.0,
            same_material_second_read=True,
            perception_mode="exact_text",
        ),
        "ui_text": EvidenceSamplingProfile(
            kind,
            2.0,
            40,
            max_window_sec=20.0,
            same_material_second_read=True,
            perception_mode="ui_read",
        ),
        "persistent_state": EvidenceSamplingProfile(
            kind,
            0.5,
            12,
            min_probe_count=6,
            max_probe_count=12,
            perception_mode="state_probe",
        ),
        "transient_event": EvidenceSamplingProfile(
            kind,
            2.0,
            32,
            max_window_sec=16.0,
            perception_mode="cue_refinement",
        ),
        "relation": EvidenceSamplingProfile(
            kind,
            max(1.0, base_fps),
            64,
            perception_mode="relation_sides",
        ),
    }
    return profiles[kind]


def bounded_profile_range(
    start_sec: float,
    end_sec: float,
    profile: EvidenceSamplingProfile,
) -> tuple[float, float]:
    start, end = sorted((float(start_sec), float(end_sec)))
    limit = profile.max_window_sec
    if limit is None or end - start <= limit:
        return start, end
    center = (start + end) / 2.0
    half = float(limit) / 2.0
    return center - half, center + half


def probe_coverage_requirement(
    frame_limit: int,
    profile: EvidenceSamplingProfile,
) -> int:
    if profile.evidence_kind != "persistent_state":
        return 0
    return min(
        max(1, int(frame_limit)),
        max(1, int(profile.min_probe_count)),
    )


def _supported_fps(value: float) -> float:
    requested = float(value or 0.5)
    return min((0.5, 1.0, 2.0), key=lambda item: abs(item - requested))


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
