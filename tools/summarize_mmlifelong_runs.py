#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


def collect_evaluations(run_roots: Sequence[Path]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in run_roots:
        candidates = (root,) if root.name == "mmlifelong_metrics.json" else root.rglob("mmlifelong_metrics.json")
        for path in candidates:
            resolved = path.resolve()
            if resolved in seen or not path.is_file():
                continue
            seen.add(resolved)
            evaluation = json.loads(path.read_text(encoding="utf-8"))
            config_path = path.parent / "run_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
            rows.append({"path": str(path), "evaluation": evaluation, "config": config})
    return tuple(rows)


def aggregate_evaluations(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        evaluation = row.get("evaluation", {})
        config = row.get("config", {})
        digest = str(evaluation.get("config_digest", config.get("config_digest", "unknown")))
        grouped[digest].append(row)
    aggregates = []
    for digest, group in sorted(grouped.items()):
        evaluations = [row["evaluation"] for row in group]
        config = dict(group[0].get("config", {}) or {})
        aggregates.append(
            {
                "config_digest": digest,
                "caption_index_mode": config.get("caption_index_mode", "unknown"),
                "answer_policy": config.get("answer_policy", "unknown"),
                "case_count": len(group),
                "Acc": _optional_mean(row.get("accuracy_score") for row in evaluations),
                "answer_rate": _mean_path(evaluations, "agent", "answer_rate"),
                "reference_valid_rate": _mean_path(evaluations, "agent", "reference_valid_rate"),
                "Ref@60": _mean_path(evaluations, "ref", "Ref@60"),
                "Ref@300": _mean_path(evaluations, "ref", "Ref@300"),
                "Ref@600": _mean_path(evaluations, "ref", "Ref@600"),
                "ClueRecall@5": _mean_path(evaluations, "retrieval", "ClueRecall@5"),
                "AllCluesRecall@5": _mean_path(evaluations, "retrieval", "AllCluesRecall@5"),
                "avg_rounds": _mean_path(evaluations, "agent", "rounds"),
                "avg_caption_searches": _mean_path(evaluations, "agent", "caption_searches"),
                "avg_caption_material_attempts": _mean_path(
                    evaluations,
                    "agent",
                    "caption_material_attempts",
                ),
                "caption_result_novelty_rate": _mean_path(
                    evaluations,
                    "agent",
                    "caption_result_novelty_rate",
                ),
                "avg_caption_result_set_reuses": _mean_path(
                    evaluations,
                    "agent",
                    "caption_result_set_reuse_count",
                ),
                "avg_occurrence_candidates": _mean_path(
                    evaluations,
                    "agent",
                    "caption_occurrence_candidate_count",
                ),
                "avg_unique_visual_material_attempts": _mean_path(
                    evaluations,
                    "agent",
                    "unique_visual_material_attempts",
                ),
                "avg_visual_interpretations": _mean_path(
                    evaluations,
                    "agent",
                    "visual_interpretation_count",
                ),
                "avg_visual_confirmations": _mean_path(evaluations, "agent", "visual_confirmations"),
                "avg_visual_frames": _mean_path(evaluations, "agent", "visual_frames_inspected"),
            }
        )
    return tuple(aggregates)


def render_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = (
        "caption_index_mode",
        "case_count",
        "Acc",
        "answer_rate",
        "reference_valid_rate",
        "Ref@60",
        "Ref@300",
        "Ref@600",
        "ClueRecall@5",
        "caption_result_novelty_rate",
        "avg_occurrence_candidates",
        "avg_unique_visual_material_attempts",
        "avg_visual_interpretations",
        "avg_rounds",
        "avg_visual_frames",
    )
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_format_cell(row.get(column)) for column in columns)
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    raw = collect_evaluations(tuple(Path(value) for value in args.run_root))
    aggregates = aggregate_evaluations(raw)
    payload = {"schema_version": "MMLifelongAggregateV1", "run_count": len(raw), "rows": aggregates}
    markdown = render_markdown(aggregates)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.out_md:
        Path(args.out_md).write_text(markdown, encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _mean_path(rows: Sequence[Mapping[str, Any]], section: str, key: str) -> float | None:
    return _optional_mean(
        row.get(section, {}).get(key)
        for row in rows
        if isinstance(row.get(section), Mapping)
    )


def _optional_mean(values: Any) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return mean(numeric) if numeric else None


def _format_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate MM-Lifelong case metrics into a paper-style table.")
    parser.add_argument("--run-root", action="append", required=True)
    parser.add_argument("--out-json")
    parser.add_argument("--out-md")
    return parser.parse_args()


if __name__ == "__main__":
    main()
