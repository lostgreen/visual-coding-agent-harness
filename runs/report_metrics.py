from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


INCOMPLETE_STATUSES = {"max_rounds_reached", "incomplete", "error", "failed"}
VISUAL_SEGMENT_TOOLS = {"inspect_segment", "caption_segment", "qa_segment", "caption_segments"}


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
) -> dict[str, Any]:
    workspace_path = _workspace_path(strategy=strategy, raw_artifacts=raw_artifacts, summary_path=summary_path)
    trace = _trace_summary(workspace_path)
    tool_sequence = trace["tool_sequence"] or [str(tool) for tool in raw.get("tools", [])]
    unique_segments = trace["unique_inspected_segments"] or _unique(str(segment) for segment in raw.get("segments", []))
    seconds = _seconds(raw)
    final = _is_final(strategy, raw)
    incomplete = _is_incomplete(strategy, raw, final=final)
    return {
        "choice": str(raw.get("choice", "")),
        "correct": bool(raw.get("correct", False)),
        "status": str(raw.get("status", "")),
        "seconds": seconds,
        "final": final,
        "incomplete": incomplete,
        "tool_sequence": tool_sequence,
        "unique_inspected_segments": unique_segments,
        "citation_count": int(raw.get("citation_count", 0) or 0),
        "walltime_vs_direct": _ratio(seconds, direct_seconds),
        "workspace": str(workspace_path) if workspace_path is not None else "",
        "error": str(raw.get("error", "")),
    }


def _strategy_report(cases: Sequence[Mapping[str, Any]], strategy: str) -> dict[str, Any]:
    rows = [case["strategies"][strategy] for case in cases if strategy in case["strategies"]]
    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    finals = sum(1 for row in rows if row["final"])
    incomplete = sum(1 for row in rows if row["incomplete"])
    seconds = [row["seconds"] for row in rows if row["seconds"] is not None]
    ratios = [row["walltime_vs_direct"] for row in rows if row["walltime_vs_direct"] is not None]
    return {
        "accuracy": f"{correct}/{total}",
        "accuracy_rate": correct / total if total else 0.0,
        "final_rate": finals / total if total else 0.0,
        "incomplete_rate": incomplete / total if total else 0.0,
        "avg_seconds": round(sum(seconds) / len(seconds), 3) if seconds else None,
        "avg_walltime_vs_direct": round(sum(ratios) / len(ratios), 3) if ratios else None,
    }


def _trace_summary(workspace_path: Path | None) -> dict[str, list[str]]:
    if workspace_path is None:
        return {"tool_sequence": [], "unique_inspected_segments": []}
    trace_path = workspace_path / "trace.jsonl"
    if not trace_path.exists():
        return {"tool_sequence": [], "unique_inspected_segments": []}
    tools = []
    inspected_segments = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "tool_use":
                continue
            payload = event.get("payload", {}) if isinstance(event.get("payload", {}), Mapping) else {}
            tool = str(payload.get("tool", ""))
            if tool:
                tools.append(tool)
            arguments = payload.get("arguments", {}) if isinstance(payload.get("arguments", {}), Mapping) else {}
            segment_id = arguments.get("segment_id")
            if segment_id and tool in VISUAL_SEGMENT_TOOLS:
                inspected_segments.append(str(segment_id))
    return {"tool_sequence": tools, "unique_inspected_segments": _unique(inspected_segments)}


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
        "| Strategy | Accuracy | Final Rate | Incomplete Rate | Avg Sec | Avg vs Direct |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy, metrics in report["strategies"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    strategy,
                    str(metrics["accuracy"]),
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
