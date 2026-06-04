from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from visual_coding_agent_harness.workspace import EvidenceWorkspace


INCOMPLETE_STATUSES = {"max_rounds_reached", "incomplete", "error", "failed"}
VISUAL_SEGMENT_TOOLS = {"inspect_segment", "caption_segment", "qa_segment", "caption_segments"}
GROUNDING_WEIGHTS = {
    "global_sparse": 1.0,
    "visually_confirmed": 1.0,
    "inferred": 0.35,
    "weak": 0.2,
    "external_knowledge": 0.1,
}
WEAK_GROUNDING = {"inferred", "weak", "external_knowledge"}


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
    direct_seconds = _seconds(case.get("strategies", {}).get("direct_full_video", {}))
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
                direct_seconds=direct_seconds,
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
    direct_seconds: float | None,
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
        "walltime_vs_direct": _ratio(seconds, direct_seconds),
        "workspace": str(workspace_path) if workspace_path is not None else "",
        "error": str(raw.get("error", "")),
        **arbitration,
    }


def _strategy_report(cases: Sequence[Mapping[str, Any]], strategy: str) -> dict[str, Any]:
    rows = [case["strategies"][strategy] for case in cases if strategy in case["strategies"]]
    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    finals = sum(1 for row in rows if row["final"])
    incomplete = sum(1 for row in rows if row["incomplete"])
    conflict = sum(1 for row in rows if row["has_conflict"])
    final_with_conflict = sum(1 for row in rows if row["final_with_conflict"])
    unsupported_final = sum(1 for row in rows if row["unsupported_final"])
    legacy_worker_vote_rows = sum(int(row["legacy_worker_vote_rows"]) for row in rows)
    consistency_rows = [row for row in rows if row["option_support_consistency"] is not None]
    consistent = sum(1 for row in consistency_rows if row["option_support_consistency"])
    seconds = [row["seconds"] for row in rows if row["seconds"] is not None]
    ratios = [row["walltime_vs_direct"] for row in rows if row["walltime_vs_direct"] is not None]
    direct_regressions = _direct_regressions(cases=cases, strategy=strategy)
    return {
        "accuracy": f"{correct}/{total}",
        "accuracy_rate": correct / total if total else 0.0,
        "direct_regressions": direct_regressions,
        "final_rate": finals / total if total else 0.0,
        "incomplete_rate": incomplete / total if total else 0.0,
        "conflict_rate": conflict / total if total else 0.0,
        "final_with_conflict_rate": final_with_conflict / total if total else 0.0,
        "unsupported_final_rate": unsupported_final / total if total else 0.0,
        "legacy_worker_vote_rows": legacy_worker_vote_rows,
        "option_support_consistency_rate": consistent / len(consistency_rows) if consistency_rows else 0.0,
        "avg_seconds": round(sum(seconds) / len(seconds), 3) if seconds else None,
        "avg_walltime_vs_direct": round(sum(ratios) / len(ratios), 3) if ratios else None,
    }


def _direct_regressions(*, cases: Sequence[Mapping[str, Any]], strategy: str) -> int:
    if strategy == "direct_full_video":
        return 0
    regressions = 0
    for case in cases:
        strategies = case.get("strategies", {}) if isinstance(case.get("strategies", {}), Mapping) else {}
        direct = strategies.get("direct_full_video")
        row = strategies.get(strategy)
        if not isinstance(direct, Mapping) or not isinstance(row, Mapping):
            continue
        if bool(direct.get("correct")) and not bool(row.get("correct")):
            regressions += 1
    return regressions


def _trace_summary(workspace_path: Path | None) -> dict[str, list[str]]:
    if workspace_path is None:
        return {"tool_sequence": [], "unique_inspected_segments": [], "final_citations": []}
    trace_path = workspace_path / "trace.jsonl"
    if not trace_path.exists():
        return {"tool_sequence": [], "unique_inspected_segments": [], "final_citations": []}
    tools = []
    inspected_segments = []
    final_citations = []
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
    }


def _workspace_path(*, strategy: str, raw_artifacts: Mapping[str, Any], summary_path: Path) -> Path | None:
    workspaces = raw_artifacts.get("workspaces", {}) if isinstance(raw_artifacts.get("workspaces", {}), Mapping) else {}
    if strategy in workspaces:
        return _resolve_path(Path(str(workspaces[strategy])), summary_path=summary_path)
    legacy_keys = {
        "empty_index_loop": "empty_workspace",
        "subtitle_index_loop": "subtitle_workspace",
        "agent_v2": "agent_v2_workspace",
    }
    legacy = raw_artifacts.get(legacy_keys.get(strategy, ""))
    if legacy:
        return _resolve_path(Path(str(legacy)), summary_path=summary_path)
    return None


def _resolve_path(path: Path, *, summary_path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return summary_path.parent / path


def _is_final(strategy: str, raw: Mapping[str, Any]) -> bool:
    status = str(raw.get("status", "")).lower()
    if strategy == "direct_full_video":
        return status not in INCOMPLETE_STATUSES and not raw.get("error")
    return status == "final"


def _is_incomplete(strategy: str, raw: Mapping[str, Any], *, final: bool) -> bool:
    status = str(raw.get("status", "")).lower()
    if status in INCOMPLETE_STATUSES or raw.get("error"):
        return True
    if strategy != "direct_full_video" and not final:
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


def _seconds(raw: Mapping[str, Any]) -> float | None:
    value = raw.get("seconds")
    if value is None:
        return None
    return float(value)


def _ratio(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or baseline <= 0:
        return None
    return round(value / baseline, 3)


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
        "| Strategy | Accuracy | Direct Regressions | Legacy Worker Votes | Final Rate | Incomplete Rate | Avg Sec | Avg vs Direct |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy, metrics in report["strategies"].items():
        lines.append(
            "| "
            + " | ".join(
                    [
                        strategy,
                        str(metrics["accuracy"]),
                        str(metrics["direct_regressions"]),
                        str(metrics["legacy_worker_vote_rows"]),
                        _pct(metrics["final_rate"]),
                    _pct(metrics["incomplete_rate"]),
                    _fmt(metrics["avg_seconds"]),
                    _fmt(metrics["avg_walltime_vs_direct"]),
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
