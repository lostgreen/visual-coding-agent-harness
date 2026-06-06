#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


VISUAL_RESULT_TOOLS = {
    "global_gist",
    "query_context",
    "caption_segment",
    "qa_segment",
    "inspect_segment",
    "vision_read",
    "caption_image",
    "caption_region",
    "ocr_region",
    "qa_region",
    "inspect_region",
    "verify_local_claim",
}


def audit_trajectory(path: Path) -> tuple[int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return 2, "invalid trajectory: root payload is not an object"
    if payload.get("schema_version") != "TrainingTrajectoryV1":
        return 2, f"invalid schema: {payload.get('schema_version')}"
    tool_calls = payload.get("tool_calls", [])
    tool_results = payload.get("tool_results", [])
    planner_turns = payload.get("planner_turns", [])
    planner_plans = payload.get("planner_plans", [])
    route_repairs = payload.get("route_repairs", [])
    evidence_chain_ids = payload.get("evidence_chain_ids", [])
    final_decision = str(payload.get("final_decision", ""))
    errors = []
    if not isinstance(tool_calls, list) or not tool_calls:
        errors.append("missing tool_calls")
    if isinstance(tool_results, list) and tool_results:
        errors.extend(_tool_result_errors(tool_calls=tool_calls, tool_results=tool_results, planner_turns=planner_turns))
    if isinstance(planner_turns, list) and planner_turns:
        errors.extend(_planner_turn_errors(planner_turns))
    if final_decision in {"final", "low_confidence_final"} and not evidence_chain_ids:
        errors.append("final decision has no evidence chains")
    visible_count, visibility_total = _tool_result_visibility_counts(tool_results)
    lines = [
        f"case_id: {payload.get('case_id', '')}",
        f"final_decision: {final_decision}",
        f"selected_option: {payload.get('selected_option', '')}",
        f"is_correct: {payload.get('is_correct', '')}",
        f"tool_calls: {len(tool_calls) if isinstance(tool_calls, list) else 0}",
        f"tool_results: {len(tool_results) if isinstance(tool_results, list) else 0}",
        f"planner_turns: {len(planner_turns) if isinstance(planner_turns, list) else 0}",
        f"planner_plans: {len(planner_plans) if isinstance(planner_plans, list) else 0}",
        f"route_repairs: {len(route_repairs) if isinstance(route_repairs, list) else 0}",
        f"tool_result_visibility: {visible_count}/{visibility_total}",
        f"evidence_chains: {len(evidence_chain_ids) if isinstance(evidence_chain_ids, list) else 0}",
        f"frame_sets: {len(payload.get('frame_set_ids', []) or [])}",
        f"context_budget_reports: {len(payload.get('context_budget_reports', []) or [])}",
    ]
    if errors:
        lines.append("errors: " + "; ".join(errors))
        return 1, "\n".join(lines)
    return 0, "\n".join(lines)


def _tool_result_errors(
    *,
    tool_calls: Any,
    tool_results: Sequence[Any],
    planner_turns: Any,
) -> list[str]:
    errors: list[str] = []
    if isinstance(tool_calls, list) and len(tool_calls) != len(tool_results):
        errors.append(f"tool_calls/tool_results mismatch: {len(tool_calls)} != {len(tool_results)}")
    planner_rounds = _planner_rounds(planner_turns)
    for result in tool_results:
        if not isinstance(result, Mapping):
            errors.append("invalid tool_result entry")
            continue
        tool_name = str(result.get("tool", ""))
        observation_id = str(result.get("observation_id", ""))
        if tool_name in VISUAL_RESULT_TOOLS and not str(result.get("claim", "")).strip():
            errors.append(f"empty claim for {observation_id or tool_name}")
        if _should_be_visible_later(result=result, planner_rounds=planner_rounds):
            visible_rounds = _int_list(result.get("visible_in_planner_rounds", []))
            source_round = _optional_int(result.get("source_round"))
            later_visible = (
                any(round_number > source_round for round_number in visible_rounds)
                if source_round is not None
                else bool(visible_rounds)
            )
            if not later_visible:
                errors.append(f"not visible in later planner prompt: {observation_id or tool_name}")
    return errors


def _planner_turn_errors(planner_turns: Sequence[Any]) -> list[str]:
    errors: list[str] = []
    for turn in planner_turns:
        if not isinstance(turn, Mapping):
            errors.append("invalid planner_turn entry")
            continue
        prompt_artifact = turn.get("prompt_artifact", {})
        if isinstance(prompt_artifact, Mapping) and not str(prompt_artifact.get("sha256", "")).strip():
            errors.append(f"planner round {turn.get('round', '?')} has no prompt sha256")
        empty_claim_count = _optional_int(turn.get("empty_evidence_claim_count")) or 0
        if empty_claim_count > 0:
            errors.append(f"planner round {turn.get('round', '?')} has empty evidence claim lines: {empty_claim_count}")
    return errors


def _tool_result_visibility_counts(tool_results: Any) -> tuple[int, int]:
    if not isinstance(tool_results, list):
        return 0, 0
    total = 0
    visible = 0
    for result in tool_results:
        if not isinstance(result, Mapping):
            continue
        total += 1
        if _int_list(result.get("visible_in_planner_rounds", [])):
            visible += 1
    return visible, total


def _should_be_visible_later(*, result: Mapping[str, Any], planner_rounds: Sequence[int]) -> bool:
    if not planner_rounds:
        return False
    source_round = _optional_int(result.get("source_round"))
    if source_round is None:
        return True
    return any(round_number > source_round for round_number in planner_rounds)


def _planner_rounds(planner_turns: Any) -> list[int]:
    if not isinstance(planner_turns, Sequence) or isinstance(planner_turns, (str, bytes)):
        return []
    rounds = []
    for turn in planner_turns:
        if isinstance(turn, Mapping):
            value = _optional_int(turn.get("round"))
            if value is not None:
                rounds.append(value)
    return sorted(set(rounds))


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    items = []
    for item in value:
        parsed = _optional_int(item)
        if parsed is not None:
            items.append(parsed)
    return items


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a compact TrainingTrajectory JSON artifact.")
    parser.add_argument("trajectory_json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    status, output = audit_trajectory(args.trajectory_json)
    print(output)
    raise SystemExit(status)


if __name__ == "__main__":
    main(sys.argv[1:])
