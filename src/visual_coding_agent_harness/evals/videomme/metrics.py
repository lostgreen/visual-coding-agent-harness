from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from visual_coding_agent_harness.workspace import EvidenceWorkspace


INCOMPLETE_STATUSES = {
    "max_rounds_reached",
    "protocol_repair_exhausted",
    "evidence_repair_exhausted",
    "incomplete",
    "error",
    "failed",
}
VISUAL_SEGMENT_TOOLS = {"inspect_segment", "caption_segment", "qa_segment", "caption_segments", "vision_read"}
GROUNDING_WEIGHTS = {
    "global_sparse": 0.35,
    "visually_confirmed": 1.0,
    "indexed_transcript": 0.85,
    "inferred": 0.35,
    "weak": 0.2,
    "external_knowledge": 0.1,
}
WEAK_GROUNDING = {"global_sparse", "inferred", "weak", "external_knowledge"}


def build_report(summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cases = [_case_report(case, summary_path=summary_path) for case in summary.get("cases", [])]
    strategies = sorted({strategy for case in cases for strategy in case["strategies"]})
    return {
        "summary_path": str(summary_path),
        "strategies": {strategy: _strategy_report(cases, strategy) for strategy in strategies},
        "cases": cases,
    }


def _case_report(case: Mapping[str, Any], *, summary_path: Path) -> dict[str, Any]:
    raw_artifacts = case.get("raw_artifacts", {}) if isinstance(case.get("raw_artifacts", {}), Mapping) else {}
    return {
        "question_id": str(case.get("question_id", "")),
        "gt": str(case.get("gt", "")),
        "strategies": {
            strategy: _case_strategy_report(
                strategy=strategy,
                raw=raw,
                raw_artifacts=raw_artifacts,
                summary_path=summary_path,
                question=str(case.get("question", case.get("question_excerpt", ""))),
                options=case.get("options", []),
            )
            for strategy, raw in case.get("strategies", {}).items()
        },
    }


def _case_strategy_report(
    *,
    strategy: str,
    raw: Mapping[str, Any],
    raw_artifacts: Mapping[str, Any],
    summary_path: Path,
    question: str,
    options: Sequence[str] | Any,
) -> dict[str, Any]:
    workspace_path = _workspace_path(strategy=strategy, raw_artifacts=raw_artifacts, summary_path=summary_path)
    trace = _trace_summary(workspace_path)
    tool_sequence = trace["tool_sequence"] or [str(tool) for tool in raw.get("tools", [])]
    unique_segments = trace["unique_inspected_segments"] or _unique(str(segment) for segment in raw.get("segments", []))
    seconds = _seconds(raw)
    final = _is_final(strategy, raw)
    incomplete = _is_incomplete(strategy, raw, final=final)
    citations = _citations(raw, trace=trace)
    choice = str(raw.get("choice", ""))
    arbitration = _arbitration_report(
        workspace_path=workspace_path,
        question=question,
        options=options,
        choice=choice,
        citations=citations,
        final=final,
    )
    return {
        "choice": choice,
        "correct": bool(raw.get("correct", False)),
        "status": str(raw.get("status", "")),
        "seconds": seconds,
        "final": final,
        "incomplete": incomplete,
        "tool_sequence": tool_sequence,
        "unique_inspected_segments": unique_segments,
        "citations": citations,
        "citation_count": int(raw.get("citation_count", 0) or len(citations) or 0),
        "workspace": str(workspace_path) if workspace_path is not None else "",
        "error": str(raw.get("error", "")),
        "unsupported_citation_final": bool(trace["unsupported_citation_final"]),
        "mutex_conflict_detection_count": int(trace["mutex_conflict_detection_count"]),
        "timeline_completeness": trace["timeline_completeness"],
        "degenerate_observation_count": int(trace["degenerate_observation_count"]),
        "observation_count": int(trace["observation_count"]),
        "normalization_note_count": int(trace["normalization_note_count"]),
        "normalization_round_count": int(trace["normalization_round_count"]),
        "planner_recovery_hint_count": int(trace["planner_recovery_hint_count"]),
        "repeated_explore_count": int(trace["repeated_explore_count"]),
        "ledger_pending_candidate_count": int(trace["ledger_pending_candidate_count"]),
        "grounded_correct": bool(raw.get("correct", False)) and final and int(raw.get("citation_count", 0) or len(citations) or 0) > 0,
        **arbitration,
    }


def _strategy_report(cases: Sequence[Mapping[str, Any]], strategy: str) -> dict[str, Any]:
    rows = [case["strategies"][strategy] for case in cases if strategy in case["strategies"]]
    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    grounded_correct = sum(1 for row in rows if row["grounded_correct"])
    finals = sum(1 for row in rows if row["final"])
    cited_finals = sum(1 for row in rows if row["final"] and int(row["citation_count"]) > 0)
    incomplete = sum(1 for row in rows if row["incomplete"])
    conflict = sum(1 for row in rows if row["has_conflict"])
    final_with_conflict = sum(1 for row in rows if row["final_with_conflict"])
    unsupported_final = sum(1 for row in rows if row["unsupported_final"])
    traced_final_rows = [row for row in rows if row["final"]]
    unsupported_citation = sum(1 for row in traced_final_rows if row["unsupported_citation_final"])
    mutex_conflicts = sum(int(row["mutex_conflict_detection_count"]) for row in rows)
    timeline_scores = [float(row["timeline_completeness"]) for row in rows if row["timeline_completeness"] is not None]
    degenerate_observations = sum(int(row["degenerate_observation_count"]) for row in rows)
    observations = sum(int(row["observation_count"]) for row in rows)
    normalization_notes = sum(int(row["normalization_note_count"]) for row in rows)
    normalization_rounds = sum(int(row["normalization_round_count"]) for row in rows)
    planner_recovery_hints = sum(int(row["planner_recovery_hint_count"]) for row in rows)
    repeated_explores = sum(int(row["repeated_explore_count"]) for row in rows)
    ledger_pending_candidates = sum(int(row["ledger_pending_candidate_count"]) for row in rows)
    legacy_worker_vote_rows = sum(int(row["legacy_worker_vote_rows"]) for row in rows)
    consistency_rows = [row for row in rows if row["option_support_consistency"] is not None]
    consistent = sum(1 for row in consistency_rows if row["option_support_consistency"])
    seconds = [row["seconds"] for row in rows if row["seconds"] is not None]
    return {
        "accuracy": f"{correct}/{total}",
        "raw_choice_accuracy": f"{correct}/{total}",
        "grounded_choice_accuracy": f"{grounded_correct}/{total}",
        "accuracy_rate": correct / total if total else 0.0,
        "grounded_choice_accuracy_rate": grounded_correct / total if total else 0.0,
        "cited_answer_rate": cited_finals / total if total else 0.0,
        "final_rate": finals / total if total else 0.0,
        "incomplete_rate": incomplete / total if total else 0.0,
        "conflict_rate": conflict / total if total else 0.0,
        "final_with_conflict_rate": final_with_conflict / total if total else 0.0,
        "unsupported_final_rate": unsupported_final / total if total else 0.0,
        "unsupported_citation_rate": unsupported_citation / len(traced_final_rows) if traced_final_rows else 0.0,
        "mutex_conflict_detection_count": mutex_conflicts,
        "timeline_completeness": sum(timeline_scores) / len(timeline_scores) if timeline_scores else 0.0,
        "degenerate_observation_rate": degenerate_observations / observations if observations else 0.0,
        "normalization_notes_per_round": (
            normalization_notes / normalization_rounds if normalization_rounds else 0.0
        ),
        "planner_recovery_hint_rate": planner_recovery_hints / total if total else 0.0,
        "repeated_explore_rate": repeated_explores / total if total else 0.0,
        "ledger_pending_candidate_rate": ledger_pending_candidates / total if total else 0.0,
        "legacy_worker_vote_rows": legacy_worker_vote_rows,
        "option_support_consistency_rate": consistent / len(consistency_rows) if consistency_rows else 0.0,
        "avg_seconds": round(sum(seconds) / len(seconds), 3) if seconds else None,
    }


def _trace_summary(workspace_path: Path | None) -> dict[str, Any]:
    default = {
        "tool_sequence": [],
        "unique_inspected_segments": [],
        "final_citations": [],
        "unsupported_citation_final": False,
        "mutex_conflict_detection_count": 0,
        "timeline_completeness": None,
        "degenerate_observation_count": 0,
        "observation_count": 0,
        "normalization_note_count": 0,
        "normalization_round_count": 0,
        "planner_recovery_hint_count": 0,
        "repeated_explore_count": 0,
        "ledger_pending_candidate_count": 0,
    }
    if workspace_path is None:
        return default
    trace_path = workspace_path / "trace.jsonl"
    if not trace_path.exists():
        return default
    observations = _load_observations(workspace_path)
    observations_by_id = {
        str(row.get("observation_id", "")): row
        for row in observations
        if row.get("observation_id")
    }
    tools = []
    inspected_segments = []
    final_citations = []
    unsupported_citation_final = False
    mutex_conflict_detection_count = 0
    timeline_scores = []
    degenerate_observation_ids = set()
    anonymous_degenerate_events = 0
    normalization_note_count = 0
    normalization_round_count = 0
    planner_recovery_hint_count = 0
    repeated_explore_count = 0
    ledger_pending_candidate_count = 0
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_type = event.get("type")
            payload = event.get("payload", {}) if isinstance(event.get("payload", {}), Mapping) else {}
            if event_type == "iterative_final":
                final_citations = [str(item) for item in payload.get("citations", [])]
                if any(
                    _observation_confidence_signal(observations_by_id.get(citation, {})) == "unsupported"
                    for citation in final_citations
                ):
                    unsupported_citation_final = True
                continue
            if event_type == "iterative_answer_agent" and str(payload.get("status", "")) == "need_more_evidence":
                if "mutex_conflict" in _payload_text(payload).lower():
                    mutex_conflict_detection_count += 1
            elif event_type in {"iterative_final_blocked", "answer_agent_need_more_evidence"}:
                if "mutex_conflict" in _payload_text(payload).lower():
                    mutex_conflict_detection_count += 1
            if event_type == "iterative_timeline_temporal_decision":
                explicit = _explicit_completeness_score(payload)
                if explicit is not None:
                    timeline_scores.append(explicit)
                elif payload.get("matched_events"):
                    timeline_scores.append(1.0)
                continue
            if event_type == "timeline_ordering_missing_entity":
                explicit = _explicit_completeness_score(payload)
                if explicit is not None:
                    timeline_scores.append(explicit)
                else:
                    targets = _string_list(payload.get("target_facts", []))
                    missing = set(_string_list(payload.get("missing_entities", [])))
                    if targets:
                        timeline_scores.append(sum(1 for target in targets if target not in missing) / len(targets))
                continue
            if event_type == "tool_output_degenerate":
                observation_id = str(payload.get("observation_id", "")).strip()
                if observation_id:
                    degenerate_observation_ids.add(observation_id)
                else:
                    anonymous_degenerate_events += 1
                continue
            if event_type == "iterative_normalization_empty":
                notes = payload.get("notes", [])
                if isinstance(notes, Sequence) and not isinstance(notes, (str, bytes)):
                    normalization_round_count += 1
                    normalization_note_count += len(notes)
                continue
            if event_type == "planner_recovery_hint_emitted":
                planner_recovery_hint_count += 1
                continue
            if event_type == "repeated_explore_detected":
                repeated_explore_count += 1
                continue
            if event_type != "tool_use":
                continue
            tool = str(payload.get("tool", ""))
            if tool:
                tools.append(tool)
            arguments = payload.get("arguments", {}) if isinstance(payload.get("arguments", {}), Mapping) else {}
            segment_id = arguments.get("segment_id")
            if segment_id and tool in VISUAL_SEGMENT_TOOLS:
                inspected_segments.append(str(segment_id))
    return {
        "tool_sequence": tools,
        "unique_inspected_segments": _unique(inspected_segments),
        "final_citations": final_citations,
        "unsupported_citation_final": unsupported_citation_final,
        "mutex_conflict_detection_count": mutex_conflict_detection_count,
        "timeline_completeness": sum(timeline_scores) / len(timeline_scores) if timeline_scores else None,
        "degenerate_observation_count": len(degenerate_observation_ids) + anonymous_degenerate_events,
        "observation_count": len(observations),
        "normalization_note_count": normalization_note_count,
        "normalization_round_count": normalization_round_count,
        "planner_recovery_hint_count": planner_recovery_hint_count,
        "repeated_explore_count": repeated_explore_count,
        "ledger_pending_candidate_count": ledger_pending_candidate_count,
    }


def _workspace_path(*, strategy: str, raw_artifacts: Mapping[str, Any], summary_path: Path) -> Path | None:
    workspaces = raw_artifacts.get("workspaces", {}) if isinstance(raw_artifacts.get("workspaces", {}), Mapping) else {}
    if strategy in workspaces:
        return _resolve_path(Path(str(workspaces[strategy])), summary_path=summary_path)
    return None


def _resolve_path(path: Path, *, summary_path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return summary_path.parent / path


def _is_final(strategy: str, raw: Mapping[str, Any]) -> bool:
    status = str(raw.get("status", "")).lower()
    return status == "final"


def _is_incomplete(strategy: str, raw: Mapping[str, Any], *, final: bool) -> bool:
    status = str(raw.get("status", "")).lower()
    if status in INCOMPLETE_STATUSES or raw.get("error"):
        return True
    if not final:
        return True
    return False


def _citations(raw: Mapping[str, Any], *, trace: Mapping[str, list[str]]) -> list[str]:
    raw_citations = raw.get("citations", [])
    if isinstance(raw_citations, Sequence) and not isinstance(raw_citations, (str, bytes)):
        citations = [str(item) for item in raw_citations]
        if citations:
            return citations
    return [str(item) for item in trace.get("final_citations", [])]


def _arbitration_report(
    *,
    workspace_path: Path | None,
    question: str,
    options: Sequence[str] | Any,
    choice: str,
    citations: Sequence[str],
    final: bool,
) -> dict[str, Any]:
    default = {
        "has_conflict": False,
        "conflict_options": [],
        "option_support": {},
        "top_supported_option": "",
        "option_support_consistency": None,
        "final_with_conflict": False,
        "unsupported_final": False,
        "legacy_worker_vote_rows": 0,
    }
    if workspace_path is None or not workspace_path.exists():
        return default

    table = EvidenceWorkspace(workspace_path).evidence_table(
        question=question,
        options=options if isinstance(options, Sequence) and not isinstance(options, (str, bytes)) else [],
    )
    support = _weighted_option_support(table)
    legacy_worker_vote_rows = _legacy_worker_vote_rows(table)
    supported_options = [option for option, score in support.items() if option != "unassigned" and score > 0]
    has_conflict = len(supported_options) >= 2
    top_supported_option = _top_supported_option(support)
    normalized_choice = choice.strip().upper()[:1]
    option_support_consistency = (
        top_supported_option == normalized_choice if final and top_supported_option and normalized_choice else None
    )
    rows_by_obs = {str(row["obs_id"]): row for row in table.get("rows", [])}
    cited_rows = [rows_by_obs[citation] for citation in citations if citation in rows_by_obs]
    unsupported_final = bool(final and (not cited_rows or all(_is_weak_row(row) for row in cited_rows)))
    final_with_conflict = bool(
        final
        and has_conflict
        and (
            option_support_consistency is False
            or _has_uncited_well_grounded_conflict(
                table=table,
                choice=normalized_choice,
                citations=citations,
            )
        )
    )
    return {
        "has_conflict": has_conflict,
        "conflict_options": supported_options,
        "option_support": {option: round(score, 3) for option, score in support.items()},
        "top_supported_option": top_supported_option,
        "option_support_consistency": option_support_consistency,
        "final_with_conflict": final_with_conflict,
        "unsupported_final": unsupported_final,
        "legacy_worker_vote_rows": legacy_worker_vote_rows,
    }


def _weighted_option_support(table: Mapping[str, Any]) -> dict[str, float]:
    support: dict[str, float] = {}
    groups = table.get("groups", {}) if isinstance(table.get("groups", {}), Mapping) else {}
    for option, rows in groups.items():
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            continue
        score = 0.0
        for row in rows:
            if isinstance(row, Mapping):
                score += float(row.get("confidence", 0.0) or 0.0) * GROUNDING_WEIGHTS.get(
                    str(row.get("grounding_quality", "weak")),
                    0.2,
                )
        support[str(option)] = score
    return {option: score for option, score in support.items() if score > 0}


def _legacy_worker_vote_rows(table: Mapping[str, Any]) -> int:
    rows = table.get("rows", []) if isinstance(table.get("rows", []), Sequence) else []
    return sum(1 for row in rows if isinstance(row, Mapping) and row.get("legacy_worker_vote"))


def _top_supported_option(support: Mapping[str, float]) -> str:
    candidates = [(option, score) for option, score in support.items() if option != "unassigned" and score > 0]
    if not candidates:
        return ""
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return candidates[0][0]


def _is_weak_row(row: Mapping[str, Any]) -> bool:
    return str(row.get("grounding_quality", "weak")) in WEAK_GROUNDING


def _has_uncited_well_grounded_conflict(
    *,
    table: Mapping[str, Any],
    choice: str,
    citations: Sequence[str],
) -> bool:
    cited = set(citations)
    groups = table.get("groups", {}) if isinstance(table.get("groups", {}), Mapping) else {}
    for option, rows in groups.items():
        if option in {"unassigned", choice}:
            continue
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if str(row.get("obs_id", "")) in cited:
                continue
            if str(row.get("grounding_quality")) == "visually_confirmed" and float(row.get("confidence", 0.0) or 0.0) >= 0.75:
                return True
    return False


def _load_observations(workspace_path: Path) -> list[dict[str, Any]]:
    path = workspace_path / "observations.jsonl"
    if not path.exists():
        return []
    observations = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                observations.append(payload)
    return observations


def _observation_confidence_signal(observation: Mapping[str, Any]) -> str:
    signal = str(observation.get("confidence_signal", "")).strip().lower()
    if signal:
        return signal
    raw_output = observation.get("raw_output", {})
    if isinstance(raw_output, Mapping):
        return str(raw_output.get("confidence_signal", "")).strip().lower()
    return ""


def _explicit_completeness_score(payload: Mapping[str, Any]) -> float | None:
    if "required_slots" not in payload and "satisfied_slots" not in payload:
        return None
    required = _numeric_slot_count(payload.get("required_slots"))
    if required <= 0:
        return None
    satisfied = _numeric_slot_count(payload.get("satisfied_slots"))
    return max(0.0, min(1.0, satisfied / required))


def _numeric_slot_count(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    return 0


def _payload_text(payload: Any) -> str:
    if isinstance(payload, Mapping):
        return " ".join(_payload_text(value) for value in payload.values())
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        return " ".join(_payload_text(value) for value in payload)
    return str(payload)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value if str(item)]


def _seconds(raw: Mapping[str, Any]) -> float | None:
    value = raw.get("seconds")
    if value is None:
        return None
    return float(value)


def _unique(values: Sequence[str] | Any) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# VideoMME Metrics",
        "",
        "| Strategy | Accuracy | Legacy Worker Votes | Final Rate | Incomplete Rate | Unsupported Citations | Mutex Conflicts | Timeline Completeness | Degenerate Obs | Norm Notes/Round | Avg Sec |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy, metrics in report["strategies"].items():
        lines.append(
            "| "
            + " | ".join(
                    [
                        strategy,
                        str(metrics["accuracy"]),
                        str(metrics["legacy_worker_vote_rows"]),
                        _pct(metrics["final_rate"]),
                        _pct(metrics["incomplete_rate"]),
                        _pct(metrics["unsupported_citation_rate"]),
                        str(metrics["mutex_conflict_detection_count"]),
                        _pct(metrics["timeline_completeness"]),
                        _pct(metrics["degenerate_observation_rate"]),
                        _fmt(metrics["normalization_notes_per_round"]),
                        _fmt(metrics["avg_seconds"]),
                    ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Case | Strategy | Status | Choice | Tools | Inspected | Citations | Sec |",
            "| --- | --- | --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for case in report["cases"]:
        for strategy, detail in case["strategies"].items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        case["question_id"],
                        strategy,
                        detail["status"] or ("final" if detail["final"] else ""),
                        detail["choice"],
                        " -> ".join(detail["tool_sequence"]),
                        ", ".join(detail["unique_inspected_segments"]),
                        str(detail["citation_count"]),
                        _fmt(detail["seconds"]),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report compact metrics from a VideoMME eval summary.")
    parser.add_argument("summary_json", type=Path)
    parser.add_argument("--json", action="store_true", help="Emit structured JSON instead of markdown.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    report = build_report(args.summary_json)
    if args.json:
        print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    else:
        print(render_markdown(report))


if __name__ == "__main__":
    main()
