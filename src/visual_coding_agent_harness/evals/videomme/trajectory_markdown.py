from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


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
