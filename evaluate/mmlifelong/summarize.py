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
        candidates = (
            (root,)
            if root.name == "mmlifelong_eval.json"
            else root.rglob("mmlifelong_eval.json")
        )
        for path in candidates:
            resolved = path.resolve()
            if resolved in seen or not path.is_file():
                continue
            seen.add(resolved)
            evaluation = json.loads(path.read_text(encoding="utf-8"))
            run_dir = path.parent.parent
            config_path = run_dir / "run_config.json"
            runtime_path = run_dir / "runtime_summary.json"
            provenance_path = path.parent / "eval_provenance.json"
            config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
            runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.is_file() else {}
            provenance = (
                json.loads(provenance_path.read_text(encoding="utf-8"))
                if provenance_path.is_file()
                else {}
            )
            rows.append(
                {
                    "path": str(path),
                    "evaluation": evaluation,
                    "runtime": runtime,
                    "config": config,
                    "provenance": provenance,
                }
            )
    return tuple(rows)


def aggregate_evaluations(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        evaluation = row.get("evaluation", {})
        config = row.get("config", {})
        provenance = row.get("provenance", {})
        evaluator_digest = (
            provenance.get("evaluator_revision", "unknown")
            if isinstance(provenance, Mapping)
            else "unknown"
        )
        runtime_digest = evaluation.get(
            "config_digest",
            config.get("config_digest", "unknown"),
        )
        digest = f"{runtime_digest}:{evaluator_digest}"
        grouped[digest].append(row)
    aggregates = []
    for digest, group in sorted(grouped.items()):
        evaluations = [row["evaluation"] for row in group]
        runtimes = [
            row.get("runtime", {}).get(
                "runtime_metrics",
                row["evaluation"].get("agent", {}),
            )
            for row in group
        ]
        config = dict(group[0].get("config", {}) or {})
        aggregates.append(
            {
                "config_digest": digest,
                "caption_index_mode": config.get("caption_index_mode", "unknown"),
                "answer_policy": config.get("answer_policy", "unknown"),
                "case_count": len(group),
                "Acc": _optional_mean(_answer_score(row) for row in evaluations),
                "answer_rate": _optional_mean(row.get("answer_rate") for row in runtimes),
                "reference_valid_rate": _optional_mean(
                    row.get("reference_valid_rate") for row in runtimes
                ),
                "Ref@60": _optional_mean(_ref_score(row, 60) for row in evaluations),
                "Ref@300": _optional_mean(_ref_score(row, 300) for row in evaluations),
                "Ref@600": _optional_mean(_ref_score(row, 600) for row in evaluations),
                "ClueRecall@5": _mean_path(evaluations, "retrieval", "ClueRecall@5"),
                "AllCluesRecall@5": _mean_path(evaluations, "retrieval", "AllCluesRecall@5"),
                "avg_rounds": _optional_mean(row.get("rounds") for row in runtimes),
                "avg_caption_searches": _optional_mean(
                    row.get("caption_searches") for row in runtimes
                ),
                "avg_caption_material_attempts": _optional_mean(
                    row.get("caption_material_attempts") for row in runtimes
                ),
                "caption_result_novelty_rate": _optional_mean(
                    row.get("caption_result_novelty_rate") for row in runtimes
                ),
                "avg_caption_result_set_reuses": _optional_mean(
                    row.get("caption_result_set_reuse_count") for row in runtimes
                ),
                "avg_occurrence_candidates": _optional_mean(
                    row.get("caption_occurrence_candidate_count") for row in runtimes
                ),
                "avg_unique_visual_material_attempts": _optional_mean(
                    row.get("unique_visual_material_attempts") for row in runtimes
                ),
                "avg_visual_interpretations": _optional_mean(
                    row.get("visual_interpretation_count") for row in runtimes
                ),
                "avg_visual_confirmations": _optional_mean(
                    row.get("visual_confirmations") for row in runtimes
                ),
                "avg_visual_frames": _optional_mean(
                    row.get("visual_frames_inspected") for row in runtimes
                ),
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
    payload = {"schema_version": "MMLifelongAggregateV2", "run_count": len(raw), "rows": aggregates}
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


def _answer_score(evaluation: Mapping[str, Any]) -> Any:
    answer = evaluation.get("answer")
    if isinstance(answer, Mapping):
        return answer.get("score")
    return evaluation.get("accuracy_score")


def _ref_score(evaluation: Mapping[str, Any], bucket: int) -> Any:
    grounding = evaluation.get("reference_grounding")
    if isinstance(grounding, Mapping):
        return grounding.get(f"ref_{int(bucket)}")
    legacy = evaluation.get("ref")
    if isinstance(legacy, Mapping):
        return legacy.get(f"Ref@{int(bucket)}")
    return None


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
