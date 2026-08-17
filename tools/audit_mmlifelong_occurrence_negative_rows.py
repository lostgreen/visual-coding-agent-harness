#!/usr/bin/env python3
"""Prepare and summarize a blinded audit of negative-sidecar rows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from statistics import mean
from typing import Any, Mapping, Sequence

from vcah.occurrence_negative_sidecar import (
    load_negative_sidecar_snapshot,
    stable_digest,
)


VALID_VERDICTS = frozenset({"true_contradiction", "false_contradiction", "unclear"})


def prepare_audit(
    repeats: Mapping[str, Path],
    *,
    positive_run_root: Path,
    replay_fixture_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    blind_items: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    case_ids = tuple(
        sorted(
            {
                path.parent.name
                for root in repeats.values()
                for path in Path(root).glob("cases/*/sidecar_result.json")
            }
        )
    )
    snapshots = {
        case_id: load_negative_sidecar_snapshot(
            Path(positive_run_root) / "cases" / case_id,
            replay_fixture_path=(
                Path(replay_fixture_root) / "cases" / f"{case_id}.json"
            ),
        )
        for case_id in case_ids
    }
    for repeat_label, root in repeats.items():
        for path in sorted(Path(root).glob("cases/*/sidecar_result.json")):
            result = _read_json(path)
            case_id = str(result.get("case_id", path.parent.name) or path.parent.name)
            snapshot = snapshots[case_id]
            constraints = {
                str(row["constraint_id"]): row for row in snapshot.constraints
            }
            candidates = {
                str(row.get("occurrence_id", "") or ""): row
                for row in snapshot.candidates
            }
            for row_index, row in enumerate(
                tuple(result.get("contradiction_rows", ()) or ())
            ):
                if not isinstance(row, Mapping):
                    continue
                constraint_id = str(row.get("constraint_id", "") or "")
                occurrence_id = str(row.get("occurrence_id", "") or "")
                if constraint_id not in constraints or occurrence_id not in candidates:
                    raise ValueError(f"{case_id}: audited row is outside frozen snapshot")
                cited_ids = tuple(
                    str(value)
                    for value in tuple(row.get("evidence_passage_ids", ()) or ())
                    if str(value)
                )
                item_id = stable_digest(
                    {
                        "repeat": repeat_label,
                        "case_id": case_id,
                        "row_index": row_index,
                        "constraint_id": constraint_id,
                        "occurrence_id": occurrence_id,
                        "cited_ids": cited_ids,
                    }
                )[:24]
                candidate = candidates[occurrence_id]
                blind_items.append(
                    {
                        "audit_item_id": item_id,
                        "question": snapshot.question,
                        "options": snapshot.options,
                        "constraint": {
                            "constraint_type": constraints[constraint_id][
                                "constraint_type"
                            ],
                            "description": constraints[constraint_id]["description"],
                        },
                        "candidate_label": "candidate-"
                        + stable_digest(occurrence_id)[:10],
                        "candidate_passages": [
                            {
                                "passage_id": str(
                                    passage.get("passage_id", "") or ""
                                ),
                                "time_range": list(
                                    passage.get("time_range", ()) or ()
                                ),
                                "caption_excerpt": str(
                                    passage.get("caption_excerpt", "") or ""
                                ),
                                "cited": str(passage.get("passage_id", "") or "")
                                in cited_ids,
                            }
                            for passage in tuple(
                                candidate.get("representative_passages", ()) or ()
                            )
                            if isinstance(passage, Mapping)
                        ],
                        "audit_question": (
                            "Do the cited visible passages directly contradict this "
                            "candidate on the stated constraint?"
                        ),
                        "allowed_verdicts": sorted(VALID_VERDICTS),
                    }
                )
                key_rows.append(
                    {
                        "audit_item_id": item_id,
                        "repeat_label": repeat_label,
                        "case_id": case_id,
                        "constraint_id": constraint_id,
                        "constraint_type": constraints[constraint_id][
                            "constraint_type"
                        ],
                        "occurrence_id_digest": stable_digest(occurrence_id),
                        "cited_passage_count": len(cited_ids),
                    }
                )
    blind_items.sort(key=lambda row: row["audit_item_id"])
    key_rows.sort(key=lambda row: row["audit_item_id"])
    return (
        {
            "schema_version": "MMLifelongOccurrenceNegativeRowBlindAuditV1",
            "item_count": len(blind_items),
            "items": blind_items,
            "blinding_checks": {
                "selection_outcomes_absent": True,
                "reference_labels_absent": True,
            },
        },
        {
            "schema_version": "MMLifelongOccurrenceNegativeRowAuditKeyV1",
            "item_count": len(key_rows),
            "rows": key_rows,
            "blind_items_digest": stable_digest(blind_items),
        },
    )


def analyze_judgments(
    key: Mapping[str, Any],
    judgments: Mapping[str, Any],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    key_rows = {
        str(row.get("audit_item_id", "") or ""): row
        for row in tuple(key.get("rows", ()) or ())
        if isinstance(row, Mapping)
    }
    judgment_rows = {
        str(row.get("audit_item_id", "") or ""): row
        for row in tuple(judgments.get("judgments", ()) or ())
        if isinstance(row, Mapping)
    }
    unknown_ids = sorted(set(judgment_rows) - set(key_rows))
    if unknown_ids:
        raise ValueError("judgments contain unknown audit item IDs")
    normalized: list[dict[str, Any]] = []
    invalid_ids: list[str] = []
    for item_id, row in judgment_rows.items():
        verdict = str(row.get("verdict", "") or "").strip().casefold()
        if verdict not in VALID_VERDICTS:
            invalid_ids.append(item_id)
            continue
        normalized.append({**dict(key_rows[item_id]), "verdict": verdict})
    adjudicated = [row for row in normalized if row["verdict"] != "unclear"]
    true_count = sum(row["verdict"] == "true_contradiction" for row in adjudicated)
    by_type = {
        constraint_type: _precision_summary(
            [
                row
                for row in normalized
                if row["constraint_type"] == constraint_type
            ]
        )
        for constraint_type in sorted(
            {str(row["constraint_type"]) for row in normalized}
        )
    }
    return {
        "schema_version": "MMLifelongOccurrenceNegativeRowAuditAnalysisV1",
        "expected_item_count": len(key_rows),
        "judgment_count": len(judgment_rows),
        "valid_judgment_count": len(normalized),
        "invalid_judgment_item_ids": invalid_ids,
        "missing_judgment_count": len(set(key_rows) - set(judgment_rows)),
        "unclear_count": sum(row["verdict"] == "unclear" for row in normalized),
        "true_contradiction_count": true_count,
        "false_contradiction_count": len(adjudicated) - true_count,
        "row_precision": true_count / len(adjudicated) if adjudicated else None,
        "row_precision_wilson95": _wilson_interval(true_count, len(adjudicated)),
        "case_cluster_bootstrap": _case_cluster_bootstrap(
            normalized, samples=bootstrap_samples, seed=seed
        ),
        "by_constraint_type": by_type,
        "complete": (
            len(normalized) == len(key_rows)
            and not invalid_ids
            and not (set(key_rows) - set(judgment_rows))
        ),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# WP12 Negative-Row Quality Audit",
        "",
        f"Complete: **{report['complete']}**",
        "",
        (
            f"Judgments: {report['valid_judgment_count']}/"
            f"{report['expected_item_count']}; unclear: {report['unclear_count']}."
        ),
        (
            "Row precision: "
            f"{_fmt(report['row_precision'])} "
            f"{_fmt_ci(report['row_precision_wilson95'])}."
        ),
        (
            "Case-cluster bootstrap 95% CI: "
            f"{_fmt_ci(report['case_cluster_bootstrap']['ci95'])}."
        ),
        "",
        "| Constraint type | True | False | Unclear | Precision |",
        "|---|---:|---:|---:|---:|",
    ]
    for constraint_type, row in report["by_constraint_type"].items():
        lines.append(
            f"| {constraint_type} | {row['true_count']} | {row['false_count']} | "
            f"{row['unclear_count']} | {_fmt(row['precision'])} |"
        )
    return "\n".join(lines) + "\n"


def _precision_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    adjudicated = [row for row in rows if row["verdict"] != "unclear"]
    true_count = sum(row["verdict"] == "true_contradiction" for row in adjudicated)
    false_count = len(adjudicated) - true_count
    return {
        "true_count": true_count,
        "false_count": false_count,
        "unclear_count": len(rows) - len(adjudicated),
        "precision": true_count / len(adjudicated) if adjudicated else None,
        "wilson95": _wilson_interval(true_count, len(adjudicated)),
    }


def _case_cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]], *, samples: int, seed: int
) -> dict[str, Any]:
    by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    case_ids = sorted(by_case)
    if not case_ids or samples <= 0:
        return {"samples": samples, "valid_samples": 0, "ci95": [None, None]}
    rng = random.Random(seed)
    precisions: list[float] = []
    for _ in range(samples):
        sampled = [rng.choice(case_ids) for _ in case_ids]
        sampled_rows = [row for case_id in sampled for row in by_case[case_id]]
        adjudicated = [row for row in sampled_rows if row["verdict"] != "unclear"]
        if not adjudicated:
            continue
        precisions.append(
            mean(row["verdict"] == "true_contradiction" for row in adjudicated)
        )
    precisions.sort()
    return {
        "samples": samples,
        "valid_samples": len(precisions),
        "ci95": [
            _quantile(precisions, 0.025),
            _quantile(precisions, 0.975),
        ],
    }


def _wilson_interval(successes: int, total: int) -> list[float | None]:
    if total <= 0:
        return [None, None]
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return [max(0.0, center - spread), min(1.0, center + spread)]


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower] * (1 - weight) + values[upper] * weight)


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def _fmt_ci(values: Sequence[Any]) -> str:
    items = tuple(values or ())
    if len(items) != 2 or any(value is None for value in items):
        return "[NA, NA]"
    return f"[{_fmt(items[0])}, {_fmt(items[1])}]"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _parse_repeat(value: str) -> tuple[str, Path]:
    label, separator, path = str(value).partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("repeat must be LABEL=PATH")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--repeat", action="append", type=_parse_repeat, required=True)
    prepare.add_argument("--positive-run-root", required=True)
    prepare.add_argument("--replay-fixture-root", required=True)
    prepare.add_argument("--output-items-json", required=True)
    prepare.add_argument("--output-key-json", required=True)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--key-json", required=True)
    analyze.add_argument("--judgments-json", required=True)
    analyze.add_argument("--bootstrap-samples", type=int, default=10000)
    analyze.add_argument("--seed", type=int, default=20260817)
    analyze.add_argument("--output-json", required=True)
    analyze.add_argument("--output-md", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        blind, key = prepare_audit(
            dict(args.repeat),
            positive_run_root=Path(args.positive_run_root),
            replay_fixture_root=Path(args.replay_fixture_root),
        )
        _write_json(Path(args.output_items_json), blind)
        _write_json(Path(args.output_key_json), key)
        print(f"NEGATIVE_ROW_AUDIT_PREPARED items={blind['item_count']}", flush=True)
        return
    report = analyze_judgments(
        _read_json(Path(args.key_json)),
        _read_json(Path(args.judgments_json)),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    _write_json(Path(args.output_json), report)
    Path(args.output_md).write_text(render_markdown(report), encoding="utf-8")
    print(
        f"NEGATIVE_ROW_AUDIT_ANALYZED complete={report['complete']} "
        f"precision={_fmt(report['row_precision'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
