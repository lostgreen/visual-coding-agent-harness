"""Export helpers for multi_v3 sidecar workspaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


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
