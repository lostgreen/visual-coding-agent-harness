from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any

from visual_coding_agent_harness.core.contracts import CONTRACT_VERSION


@dataclass
class RunSummary:
    # identity
    run_id: str
    case_ids: list[str]
    timestamp: str
    git_commit: str
    contract_version: str

    # accuracy
    accuracy: float
    final_rate: float
    need_more_evidence_rate: float
    unsupported_final_rate: float
    low_confidence_final_rate: float

    # evidence quality
    evidence_provenance_completeness: float
    tool_nframes_compliance: float
    legacy_worker_vote_rows: int
    direct_regressions: int

    # followup
    followup_success_rate: float
    avg_followups_per_case: float
    saturation_termination_rate: float

    # context
    context_budget_overflow_count: int
    avg_tokens_per_turn: int

    # phase D diagnostics
    unsupported_citation_rate: float
    mutex_conflict_detection_count: int
    timeline_completeness: float
    degenerate_observation_rate: float
    normalization_notes_per_round: float

    # diagnostics
    route_violations: int
    nframes_histogram: dict[str, dict[int, int]] = field(default_factory=dict)
    map_reflux_commit_count: int = 0
    query_context_usage_rate: float = 0.0
    training_trajectory_exported: bool = False
    per_case: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def with_defaults(cls, run_id: str, case_ids: list[str] | tuple[str, ...]) -> "RunSummary":
        return cls(
            run_id=str(run_id),
            case_ids=[str(case_id) for case_id in case_ids],
            timestamp=datetime.now(timezone.utc).isoformat(),
            git_commit="",
            contract_version=CONTRACT_VERSION,
            accuracy=0.0,
            final_rate=0.0,
            need_more_evidence_rate=0.0,
            unsupported_final_rate=0.0,
            low_confidence_final_rate=0.0,
            evidence_provenance_completeness=0.0,
            tool_nframes_compliance=0.0,
            legacy_worker_vote_rows=0,
            direct_regressions=0,
            followup_success_rate=0.0,
            avg_followups_per_case=0.0,
            saturation_termination_rate=0.0,
            context_budget_overflow_count=0,
            avg_tokens_per_turn=0,
            unsupported_citation_rate=0.0,
            mutex_conflict_detection_count=0,
            timeline_completeness=0.0,
            degenerate_observation_rate=0.0,
            normalization_notes_per_round=0.0,
            route_violations=0,
            nframes_histogram={},
            map_reflux_commit_count=0,
            query_context_usage_rate=0.0,
            training_trajectory_exported=False,
            per_case=[],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunSummary":
        allowed = {item.name for item in fields(cls)}
        values = {key: value for key, value in payload.items() if key in allowed}
        if "nframes_histogram" in values:
            values["nframes_histogram"] = _normalize_histogram(values["nframes_histogram"])
        defaults = cls.with_defaults(
            str(values.get("run_id", "")),
            values.get("case_ids", []) if isinstance(values.get("case_ids", []), list) else [],
        ).to_dict()
        defaults.update(values)
        return cls(**defaults)


def validate(summary: RunSummary) -> list[str]:
    errors = []
    if summary.unsupported_final_rate > 0.0:
        errors.append("unsupported_final_rate must be <= 0.0")
    if summary.unsupported_citation_rate > 0.0:
        errors.append("unsupported_citation_rate must be <= 0.0")
    if summary.legacy_worker_vote_rows != 0:
        errors.append("legacy_worker_vote_rows must be 0")
    if summary.route_violations != 0:
        errors.append("route_violations must be 0")
    if not 0.0 <= summary.tool_nframes_compliance <= 1.0:
        errors.append("tool_nframes_compliance must be in [0.0, 1.0]")
    if not 0.0 <= summary.accuracy <= 1.0:
        errors.append("accuracy must be in [0.0, 1.0]")
    return errors


def _normalize_histogram(value: Any) -> dict[str, dict[int, int]]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, dict[int, int]] = {}
    for tool, counts in value.items():
        if not isinstance(counts, dict):
            continue
        normalized[str(tool)] = {int(frames): int(count) for frames, count in counts.items()}
    return normalized
