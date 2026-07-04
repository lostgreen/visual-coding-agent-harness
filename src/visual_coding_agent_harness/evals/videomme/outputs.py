"""VideoMME output schemas, exports, and Markdown rendering."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    raw_choice_accuracy: float
    grounded_choice_accuracy: float
    final_rate: float
    cited_answer_rate: float
    need_more_evidence_rate: float
    unsupported_final_rate: float
    low_confidence_final_rate: float
    unvalidated_guess_rate: float

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
    option_biased_first_query_rate: float
    wrong_scope_caption_fact_rate: float
    caption_fact_downgrade_rate: float
    caption_support_final_rate: float
    visual_required_but_caption_final_rate: float
    planner_recovery_hint_rate: float
    repeated_explore_rate: float
    ledger_pending_candidate_rate: float

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
            raw_choice_accuracy=0.0,
            grounded_choice_accuracy=0.0,
            final_rate=0.0,
            cited_answer_rate=0.0,
            need_more_evidence_rate=0.0,
            unsupported_final_rate=0.0,
            low_confidence_final_rate=0.0,
            unvalidated_guess_rate=0.0,
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
            option_biased_first_query_rate=0.0,
            wrong_scope_caption_fact_rate=0.0,
            caption_fact_downgrade_rate=0.0,
            caption_support_final_rate=0.0,
            visual_required_but_caption_final_rate=0.0,
            planner_recovery_hint_rate=0.0,
            repeated_explore_rate=0.0,
            ledger_pending_candidate_rate=0.0,
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


def export_multi_v3_trajectory(
    investigator_workspace: Any,
    *,
    question: str,
    video_path: str,
    final: Mapping[str, Any],
    reward_tags: Sequence[str] = (),
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    root = _workspace_root(investigator_workspace)
    actions, observations = _trajectory_actions_and_observations(root)
    payload = {
        "schema_version": "LongVideoAgentTrajectoryV1",
        "state": {
            "question": str(question),
            "video_path": str(video_path),
            "workspace_root": root.as_posix(),
            "observation_count": len(observations),
        },
        "actions": actions,
        "observations": observations,
        "final": dict(final),
        "verifier_result": {"status": str(final.get("status", ""))},
        "reward_tags": [str(item) for item in reward_tags],
    }
    path = Path(output_path) if output_path is not None else root.parent / "artifacts" / "trajectories" / "longvideoagent_trajectory.json"
    _write_json(path, payload)
    return payload


def export_multi_v3_evidence_chains(
    investigator_workspace: Any,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    root = _workspace_root(investigator_workspace)
    findings = _ledger_findings(root)
    chains = []
    for finding in findings:
        finding_id = str(finding.get("finding_id", "")).strip()
        if not finding_id:
            continue
        chains.append(
            {
                "chain_id": finding_id,
                "records": [
                    {
                        "evidence_id": finding_id,
                        "stage": "verified",
                        "tool": "multi_v3_verify",
                        "observation_id": finding_id,
                        "content": {
                            "claim": str(finding.get("summary", "")),
                            "shot_id": str(finding.get("shot_id", "")),
                            "query_id": str(finding.get("query_id", "")),
                        },
                        "confidence": finding.get("confidence"),
                    }
                ],
            }
        )
    payload = {"schema_version": "EvidenceChainsV1", "chain_count": len(chains), "chains": chains}
    path = Path(output_path) if output_path is not None else root.parent / "artifacts" / "evidence_chains" / "evidence_chains.json"
    _write_json(path, payload)
    return payload


def export_multi_v3_exploration_records(
    investigator_workspace: Any,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    root = _workspace_root(investigator_workspace)
    records = [_exploration_record(root, query_dir) for query_dir in _query_dirs(root)]
    path = (
        Path(output_path)
        if output_path is not None
        else root.parent / "artifacts" / "exploration_records" / "exploration_records.jsonl"
    )
    _write_jsonl(path, records)
    return {
        "schema_version": "MultiV3ExplorationRecordsV1",
        "path": path.as_posix(),
        "record_count": len(records),
        "records": records,
    }


def export_multi_v3_training_trajectory(
    investigator_workspace: Any,
    *,
    case_id: str,
    question: str,
    options: Sequence[str] = (),
    ground_truth: str | None = None,
    final_decision: str = "",
    selected_option: str | None = None,
    is_correct: bool | None = None,
    output_path: str | Path,
) -> dict[str, Any]:
    root = _workspace_root(investigator_workspace)
    tool_calls, tool_results = _training_tool_io(root)
    evidence_chain_ids = [[str(item.get("finding_id", ""))] for item in _ledger_findings(root) if item.get("finding_id")]
    payload = {
        "schema_version": "TrainingTrajectoryV1",
        "contract_version": "multi_v3",
        "run_id": root.parent.name,
        "case_id": str(case_id),
        "question": str(question),
        "options": [str(item) for item in options],
        "ground_truth": str(ground_truth) if ground_truth is not None else None,
        "final_decision": str(final_decision),
        "final_decision_owner": "model" if str(final_decision) == "final" else "",
        "diagnostic_errors": [],
        "framework_fallback_used": False,
        "final_diagnostics": {},
        "model_evidence_sufficiency": "",
        "format_repair_count": 0,
        "no_model_final": str(final_decision) not in {"final", "low_confidence_final"},
        "selected_option": str(selected_option) if selected_option else None,
        "is_correct": is_correct,
        "evidence_chain_ids": evidence_chain_ids,
        "frame_set_ids": [],
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "planner_turns": [],
        "planner_plans": [],
        "route_repairs": [],
        "context_budget_reports": [],
        "followup_history": [],
        "workspace_root": root.as_posix(),
        "trajectory_path": str(output_path),
    }
    _write_json(Path(output_path), payload)
    return payload


def export_multi_v3_workspace_round_log(
    investigator_workspace: Any,
    *,
    question: str,
    video_path: str,
    final: Mapping[str, Any],
    trajectory_path: str | Path,
    evidence_chains_path: str | Path,
    log_root: str | Path,
    output_path: str | Path = "workspace_round_log.md",
) -> Mapping[str, Any]:
    root = _workspace_root(investigator_workspace)
    output_root = Path(log_root)
    output_file = Path(output_path)
    if not output_file.is_absolute():
        output_file = output_root / output_file
    coverage = _read_json(root / "coverage.json")
    lines = [
        "# Workspace Round Log",
        "",
        "## Summary",
        f"- workspace_root: {root.as_posix()}",
        f"- log_root: {output_root.as_posix()}",
        f"- question: {question}",
        f"- video_path: {video_path}",
        f"- final_status: {final.get('status', '')}",
        f"- final_answer: {final.get('answer', '')}",
        f"- final_citations: {', '.join(str(item) for item in final.get('citations', []) or []) or '(none)'}",
        f"- trajectory_json: {trajectory_path}",
        f"- evidence_chains_json: {evidence_chains_path}",
        "",
        "## Coverage",
        f"- explored_shots: {', '.join(coverage.get('explored_shots', []) or []) or '(none)'}",
        f"- verified_shots: {', '.join(coverage.get('verified_shots', []) or []) or '(none)'}",
        "",
        "## Queries",
        "",
    ]
    query_dirs = _query_dirs(root)
    if not query_dirs:
        lines.append("(none)")
    for query_dir in query_dirs:
        report = _read_json(query_dir / "report.json")
        lines.append(f"### {query_dir.name}")
        lines.append(f"- status: {report.get('status', '')}")
        lines.append(f"- explored: {', '.join(report.get('explored_shots', []) or []) or '(none)'}")
        lines.append(f"- verified: {', '.join(report.get('verified_shots', []) or []) or '(none)'}")
        lines.append("")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return {
        "path": output_file.as_posix(),
        "relative_path": output_file.name,
        "log_root": output_root.as_posix(),
        "round_count": len(query_dirs),
        "planner_prompt_count": 0,
    }


def multi_v3_tools_and_segments(investigator_workspace: Any) -> tuple[list[str], list[str]]:
    root = _workspace_root(investigator_workspace)
    tools: list[str] = []
    segments: list[str] = []
    for query_dir in _query_dirs(root):
        explore = _read_json(query_dir / "explore.json")
        if explore:
            tools.append("multi_v3_explore")
            for candidate in explore.get("candidates", []) or []:
                if isinstance(candidate, Mapping) and candidate.get("shot_id"):
                    segments.append(str(candidate["shot_id"]))
        for verify_path in sorted(query_dir.glob("verify_*.json")):
            verify = _read_json(verify_path)
            tools.append("multi_v3_verify")
            shot_id = str(verify.get("shot_id", "")).strip()
            if shot_id:
                segments.append(shot_id)
    return tools, sorted(set(segments))


def multi_v3_backend_call_counters(investigator_workspace: Any) -> dict[str, int]:
    tools, _segments = multi_v3_tools_and_segments(investigator_workspace)
    return {
        "root_index_backend_calls": 0,
        "refinement_backend_calls": tools.count("multi_v3_explore"),
        "verify_backend_calls": tools.count("multi_v3_verify"),
    }


def _trajectory_actions_and_observations(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    actions: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for query_dir in _query_dirs(root):
        request = _read_json(query_dir / "request.json")
        explore = _read_json(query_dir / "explore.json")
        if explore:
            actions.append(
                _action(
                    actions,
                    tool="multi_v3_explore",
                    arguments={"query_id": query_dir.name, "request": request, "candidates": explore.get("candidates", [])},
                )
            )
        for verify_path in sorted(query_dir.glob("verify_*.json")):
            verify = _read_json(verify_path)
            findings = [item for item in verify.get("findings", []) or [] if isinstance(item, Mapping)]
            observation = _observation_from_finding(findings[0]) if findings else None
            action = _action(
                actions,
                tool="multi_v3_verify",
                arguments={"query_id": query_dir.name, "shot_id": verify.get("shot_id", "")},
            )
            if observation is not None:
                action["observation_id"] = observation["observation_id"]
                action["observation"] = observation
            actions.append(action)
            observations.extend(_observation_from_finding(finding) for finding in findings)
    return actions, observations


def _training_tool_io(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for query_dir in _query_dirs(root):
        for verify_path in sorted(query_dir.glob("verify_*.json")):
            verify = _read_json(verify_path)
            findings = [item for item in verify.get("findings", []) or [] if isinstance(item, Mapping)]
            for finding in findings:
                step = len(calls) + 1
                shot_id = str(verify.get("shot_id", "") or finding.get("shot_id", ""))
                calls.append(
                    {
                        "step": step,
                        "source_round": 0,
                        "tool": "multi_v3_verify",
                        "arguments": {"query_id": query_dir.name, "shot_id": shot_id},
                    }
                )
                results.append(
                    {
                        "step": step,
                        "source_round": 0,
                        "tool": "multi_v3_verify",
                        "observation_id": str(finding.get("finding_id", "")),
                        "claim": str(finding.get("summary", "")),
                        "confidence": finding.get("confidence"),
                        "grounding_quality": "multi_v3_verified",
                        "limitations": "",
                        "time_range": [],
                        "mode": "verify",
                        "evidence_mode": "verify",
                        "evidence_record_ids": [str(finding.get("finding_id", ""))],
                        "visible_in_planner_rounds": [],
                    }
                )
    if not calls:
        for query_dir in _query_dirs(root):
            request = _read_json(query_dir / "request.json")
            calls.append(
                {
                    "step": len(calls) + 1,
                    "source_round": 0,
                    "tool": "multi_v3_explore",
                    "arguments": {"query_id": query_dir.name, "request": request},
                }
            )
    return calls, results


def _exploration_record(root: Path, query_dir: Path) -> dict[str, Any]:
    request = _read_json(query_dir / "request.json")
    explore = _read_json(query_dir / "explore.json")
    report = _read_json(query_dir / "report.json")
    verify_paths = sorted(query_dir.glob("verify_*.json"))
    verifications = [_verification_record(root, path) for path in verify_paths]
    candidates = [
        _candidate_record(candidate)
        for candidate in explore.get("candidates", []) or []
        if isinstance(candidate, Mapping)
    ]
    report_findings = [
        _finding_record(finding)
        for finding in report.get("findings", []) or []
        if isinstance(finding, Mapping)
    ]
    return {
        "schema_version": "MultiV3ExplorationRecordV1",
        "query_id": query_dir.name,
        "request": {
            "goal_id": str(request.get("goal_id", "")),
            "natural_query": str(request.get("natural_query", "")),
            "scope": _mapping_value(request.get("scope")),
            "expected_evidence": str(request.get("expected_evidence", "")),
            "budget": _mapping_value(request.get("budget")),
        },
        "explore": {
            "candidate_count": len(candidates),
            "candidates": candidates,
        },
        "verify": verifications,
        "report": {
            "status": str(report.get("status", "")),
            "finding_count": len(report_findings),
            "findings": report_findings,
            "explored_shots": [str(item) for item in _sequence_value(report.get("explored_shots"))],
            "verified_shots": [str(item) for item in _sequence_value(report.get("verified_shots"))],
            "unresolved": _sequence_value(report.get("unresolved")),
            "cost": _mapping_value(report.get("cost")),
        },
        "artifacts": {
            "request": _workspace_relative(root, query_dir / "request.json"),
            "explore": _workspace_relative(root, query_dir / "explore.json"),
            "report": _workspace_relative(root, query_dir / "report.json"),
            "verify": [_workspace_relative(root, path) for path in verify_paths],
        },
    }


def _verification_record(root: Path, path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    findings = [
        _finding_record(finding)
        for finding in payload.get("findings", []) or []
        if isinstance(finding, Mapping)
    ]
    return {
        "artifact": _workspace_relative(root, path),
        "shot_id": str(payload.get("shot_id", "")),
        "finding_count": len(findings),
        "findings": findings,
    }


def _candidate_record(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "shot_id": str(candidate.get("shot_id", "")),
        "score": candidate.get("score"),
        "reason": str(candidate.get("reason", "")),
    }


def _finding_record(finding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "finding_id": str(finding.get("finding_id", "")),
        "query_id": str(finding.get("query_id", "")),
        "shot_id": str(finding.get("shot_id", "")),
        "summary": str(finding.get("summary", "")),
        "supports_options": [str(item) for item in finding.get("supports_options", []) or []],
        "refutes_options": [str(item) for item in finding.get("refutes_options", []) or []],
        "citation_ids": [str(item) for item in finding.get("citation_ids", []) or []],
        "confidence": finding.get("confidence"),
    }


def _workspace_relative(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _mapping_value(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence_value(value: Any) -> list[Any]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return []
    return list(value)


def _action(actions: Sequence[Mapping[str, Any]], *, tool: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    step = len(actions) + 1
    return {
        "index": step,
        "type": "tool_use",
        "step": step,
        "tool": tool,
        "arguments": dict(arguments),
        "created_at": "",
    }


def _observation_from_finding(finding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "observation_id": str(finding.get("finding_id", "")),
        "tool": "multi_v3_verify",
        "claim": str(finding.get("summary", "")),
        "confidence": finding.get("confidence"),
        "regions": [{"shot_id": str(finding.get("shot_id", "")), "query_id": str(finding.get("query_id", ""))}],
        "limitations": "",
        "confidence_signal": "multi_v3_verify",
        "raw_output": {
            "supports_options": list(finding.get("supports_options", []) or []),
            "refutes_options": list(finding.get("refutes_options", []) or []),
            "citation_ids": list(finding.get("citation_ids", []) or []),
        },
    }


def _ledger_findings(root: Path) -> list[Mapping[str, Any]]:
    return _read_jsonl(root / "evidence_ledger.jsonl")


def _query_dirs(root: Path) -> list[Path]:
    queries = root / "queries"
    return sorted(path for path in queries.iterdir() if path.is_dir()) if queries.exists() else []


def _workspace_root(value: Any) -> Path:
    return Path(getattr(value, "root", value))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        return []
    rows: list[Mapping[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=True, sort_keys=True) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def render_trajectory_markdown(
    trajectory: Mapping[str, Any],
    *,
    trajectory_path: str | Path | None = None,
) -> str:
    """Render a compact TrainingTrajectory JSON artifact as a readable trace."""
    case_id = _text(trajectory.get("case_id", "unknown"))
    lines: list[str] = [
        f"# Trajectory {case_id}",
        "",
        "## Summary",
        f"- trajectory_json: {_text(trajectory_path) if trajectory_path else '(memory)'}",
        f"- question: {_text(trajectory.get('question', ''))}",
    ]
    options = trajectory.get("options", [])
    if isinstance(options, Sequence) and not isinstance(options, (str, bytes)) and options:
        lines.append("- options:")
        for option in options:
            lines.append(f"  - {_text(option)}")
    lines.extend(
        [
            f"- ground_truth: {_text(trajectory.get('ground_truth', ''))}",
            f"- selected_option: {_text(trajectory.get('selected_option', ''))}",
            f"- is_correct: {_text(trajectory.get('is_correct', ''))}",
            f"- final_decision: {_text(trajectory.get('final_decision', ''))}",
            "",
        ]
    )

    planner_turns = _items(trajectory.get("planner_turns", []))
    tool_calls_by_round = _items_by_round(trajectory.get("tool_calls", []))
    tool_results_by_round = _items_by_round(trajectory.get("tool_results", []))
    plans_by_round = _items_by_round(trajectory.get("planner_plans", []))
    repairs_by_round = _items_by_round(trajectory.get("route_repairs", []))

    if planner_turns:
        for turn in planner_turns:
            round_number = _round(turn)
            lines.extend(_render_round(
                turn,
                round_number=round_number,
                trajectory=trajectory,
                calls=tool_calls_by_round.get(round_number, []),
                results=tool_results_by_round.get(round_number, []),
                plans=plans_by_round.get(round_number, []),
                repairs=repairs_by_round.get(round_number, []),
            ))
    else:
        lines.extend(
            [
                "## Planner turns",
                "No planner turns were recorded. This run likely finalized through a non-planner route before the iterative planner loop.",
                "",
            ]
        )

    non_planner_calls = tool_calls_by_round.get(0, [])
    non_planner_results = tool_results_by_round.get(0, [])
    if non_planner_calls or non_planner_results:
        lines.extend(
            [
                "## Non-planner tool activity",
                "",
            ]
        )
        lines.extend(_render_tool_calls(non_planner_calls))
        lines.extend(_render_tool_results(non_planner_results))

    return "\n".join(lines).rstrip() + "\n"


def write_trajectory_markdown(
    trajectory_json: str | Path,
    *,
    output_path: str | Path | None = None,
) -> Path:
    trajectory_path = Path(trajectory_json)
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    if not isinstance(trajectory, Mapping):
        raise ValueError(f"Expected object JSON in {trajectory_path}")
    if output_path is None:
        output = trajectory_path.with_suffix(".md")
    else:
        output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_trajectory_markdown(trajectory, trajectory_path=trajectory_path), encoding="utf-8")
    return output


def _render_round(
    turn: Mapping[str, Any],
    *,
    round_number: int,
    trajectory: Mapping[str, Any],
    calls: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    plans: Sequence[Mapping[str, Any]],
    repairs: Sequence[Mapping[str, Any]],
) -> list[str]:
    lines = [
        f"## Round {round_number}",
        "",
        "### Planner input",
        *_render_planner_input_summary(turn),
        "",
        "### Planner output",
        *_render_planner_output_summary(turn),
        "",
    ]
    lines.extend(_render_planner_plans(plans))
    lines.extend(_render_route_repairs(repairs))
    lines.extend(_render_tool_calls(calls))
    lines.extend(_render_tool_results(results))
    return lines


def _render_planner_plans(plans: Sequence[Mapping[str, Any]]) -> list[str]:
    if not plans:
        return ["### Planner parsed plan", "(none)", ""]
    lines = ["### Planner parsed plan"]
    for plan in plans:
        lines.append(f"- rationale: {_text(plan.get('rationale', ''))}")
        lines.append("- program:")
        lines.extend(_indented_json_lines(plan.get("program", []), indent="  "))
    lines.append("")
    return lines


def _render_route_repairs(repairs: Sequence[Mapping[str, Any]]) -> list[str]:
    if not repairs:
        return ["### Route repairs", "(none)", ""]
    lines = ["### Route repairs"]
    for repair in repairs:
        requested = _text(repair.get("requested_tool", ""))
        resolved = _text(repair.get("resolved_tool", ""))
        reason = _text(repair.get("reason", ""))
        lines.append(f"- {requested} -> {resolved}: {reason}")
    lines.append("")
    return lines


def _render_tool_calls(calls: Sequence[Mapping[str, Any]]) -> list[str]:
    if not calls:
        return ["### Tool calls", "(none)", ""]
    lines = ["### Tool calls"]
    for call in calls:
        lines.append(f"- step {_text(call.get('step', ''))}: {_text(call.get('tool', ''))}")
        lines.extend(_indented_json_lines(call.get("arguments", {}), indent="  "))
    lines.append("")
    return lines


def _render_tool_results(results: Sequence[Mapping[str, Any]]) -> list[str]:
    if not results:
        return ["### Tool results", "(none)", ""]
    lines = ["### Tool results"]
    for result in results:
        lines.extend(
            [
                f"- step {_text(result.get('step', ''))}: {_text(result.get('tool', ''))} / {_text(result.get('observation_id', ''))}",
                f"  - claim: {_text(result.get('claim', ''))}",
                f"  - confidence: {_text(result.get('confidence', ''))}",
                f"  - grounding_quality: {_text(result.get('grounding_quality', ''))}",
                f"  - limitations: {_text(result.get('limitations', ''))}",
                f"  - time_range: {_text(result.get('time_range', ''))}",
                f"  - mode: {_text(result.get('mode', ''))}",
                f"  - evidence_mode: {_text(result.get('evidence_mode', ''))}",
                f"  - evidence_record_ids: {_text(result.get('evidence_record_ids', []))}",
                f"  - visible_in_planner_rounds: {_text(result.get('visible_in_planner_rounds', []))}",
            ]
        )
        facts = result.get("facts", [])
        if isinstance(facts, Sequence) and not isinstance(facts, (str, bytes)) and facts:
            lines.append("  - facts:")
            lines.extend(_indented_json_lines(facts, indent="    "))
        relations = result.get("candidate_option_relations", [])
        if isinstance(relations, Sequence) and not isinstance(relations, (str, bytes)) and relations:
            lines.append("  - candidate_option_relations:")
            lines.extend(_indented_json_lines(relations, indent="    "))
        produced_anchors = result.get("produced_anchors", [])
        if isinstance(produced_anchors, Sequence) and not isinstance(produced_anchors, (str, bytes)) and produced_anchors:
            lines.append("  - produced_anchors:")
            lines.extend(_indented_json_lines(produced_anchors, indent="    "))
        candidate_anchor_ids = result.get("candidate_anchor_ids", [])
        if isinstance(candidate_anchor_ids, Sequence) and not isinstance(candidate_anchor_ids, (str, bytes)) and candidate_anchor_ids:
            lines.append("  - candidate_anchor_ids:")
            lines.extend(_indented_json_lines(candidate_anchor_ids, indent="    "))
        regions = result.get("regions", [])
        if isinstance(regions, Sequence) and not isinstance(regions, (str, bytes)) and regions:
            lines.append("  - regions:")
            lines.extend(_indented_json_lines(regions, indent="    "))
    lines.append("")
    return lines


def _planner_output_text(trajectory: Mapping[str, Any], turn: Mapping[str, Any]) -> str:
    del trajectory
    return _text(turn.get("response_excerpt", ""))


def _render_planner_input_summary(turn: Mapping[str, Any]) -> list[str]:
    lines = [_artifact_summary_line("prompt_artifact", turn.get("prompt_artifact", {}))]
    evidence_ids = turn.get("evidence_observation_ids", [])
    lines.append(f"- evidence_observation_ids: {_text(evidence_ids)}")
    lines.append(f"- evidence_snapshot_chars: {_text(turn.get('evidence_snapshot_chars', 0))}")
    lines.append(f"- empty_evidence_claim_count: {_text(turn.get('empty_evidence_claim_count', 0))}")
    return lines


def _render_planner_output_summary(turn: Mapping[str, Any]) -> list[str]:
    lines = [_artifact_summary_line("response_artifact", turn.get("response_artifact", {}))]
    output = _planner_output_text({}, turn).strip()
    lines.append(_fenced(output or "(no public planner action summary recorded)"))
    return lines


def _artifact_summary_line(label: str, artifact: Any) -> str:
    payload = artifact if isinstance(artifact, Mapping) else {}
    path = _text(payload.get("path", ""))
    sha = _text(payload.get("sha256", ""))
    chars = _text(payload.get("chars", payload.get("stored_chars", "")))
    return f"- {label}: path={path or '(missing)'} chars={chars or '0'} sha256={sha[:12] if sha else '(missing)'}"


def _artifact_text(trajectory: Mapping[str, Any], artifact: Any) -> str:
    path = _artifact_path(trajectory, artifact)
    if path is None:
        return "(missing artifact path)"
    if not path.exists():
        return f"(missing artifact file: {path.as_posix()})"
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"(artifact is not valid UTF-8: {path.as_posix()})"


def _artifact_path(trajectory: Mapping[str, Any], artifact: Any) -> Path | None:
    payload = artifact if isinstance(artifact, Mapping) else {}
    artifact_path = _text(payload.get("path", "")).strip()
    if not artifact_path:
        return None
    path = Path(artifact_path)
    if path.is_absolute():
        return path
    workspace_root = _text(trajectory.get("workspace_root", "")).strip()
    return (Path(workspace_root) / path) if workspace_root else path


def _items(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _items_by_round(value: Any) -> dict[int, list[Mapping[str, Any]]]:
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for item in _items(value):
        grouped.setdefault(_round(item), []).append(item)
    return grouped


def _round(item: Mapping[str, Any]) -> int:
    value = item.get("round", item.get("source_round", 0))
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _indented_json_lines(value: Any, *, indent: str) -> list[str]:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    return [f"{indent}{line}" for line in encoded.splitlines()]


def _fenced(text: str) -> str:
    longest = 0
    current = 0
    for char in text:
        if char == "`":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{text.rstrip()}\n{fence}"


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render a TrainingTrajectory JSON artifact as readable Markdown.")
    parser.add_argument("trajectory_json", type=Path, help="Path to a TrainingTrajectory JSON artifact.")
    parser.add_argument("-o", "--output", type=Path, help="Output Markdown path. Defaults to TRAJECTORY.md.")
    args = parser.parse_args(argv)

    output_path = write_trajectory_markdown(args.trajectory_json, output_path=args.output)
    print(output_path.as_posix())


if __name__ == "__main__":
    main()
