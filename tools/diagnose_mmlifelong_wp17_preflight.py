#!/usr/bin/env python3
"""Run the zero-API WP17-0 surface and admission preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from vcah.occurrence_negative_sidecar import file_sha256
from vcah.wp17_preflight import (
    WP17_PREFLIGHT_CONTRACT,
    build_wp17_preflight_report,
)


def run(args: argparse.Namespace) -> Path:
    a3_root = Path(args.a3_root)
    manifest_path = a3_root / "run_manifest.json"
    runtime_report_path = a3_root / "entity_occurrence_report.json"
    manifest = _read_json(manifest_path)
    runtime_report = _read_json(runtime_report_path)
    surface_path = Path(args.surface_spec)
    surface_spec = _read_json(surface_path)
    if surface_spec.get("contract") != WP17_PREFLIGHT_CONTRACT:
        raise ValueError("WP17 surface spec contract mismatch")
    cases = dict(surface_spec.get("cases", {}) or {})
    if len(cases) != int(args.expected_cases):
        raise ValueError("WP17 surface case count mismatch")

    selection_path = Path(str(manifest.get("selection_path", "")))
    selection_rows = _read_jsonl(selection_path)
    batch_paths = tuple(sorted((a3_root / "batch_results").rglob("*.json")))
    batch_results = tuple(_read_json(path) for path in batch_paths)
    parsed_rows = tuple(
        dict(row)
        for result in batch_results
        for row in tuple(result.get("parsed_rows", ()) or ())
        if isinstance(row, Mapping)
    )
    checks = {
        "a3_runtime_structural_gate_passed": bool(
            runtime_report.get("gates", {}).get("structural_gate_passed")
        ),
        "a3_source_commit_exact": manifest.get("source_commit")
        == str(args.expected_source_commit),
        "a3_actual_model_exact": manifest.get("actual_model")
        == str(args.expected_model),
        "a3_question_gold_blind": all(
            manifest.get(key) is False
            for key in (
                "question_visible_to_model",
                "options_visible_to_model",
                "answer_visible_to_model",
                "caption_text_visible_to_model",
                "official_intervals_visible_to_model",
            )
        ),
        "batch_count_exact": len(batch_results)
        == int(runtime_report.get("counts", {}).get("batch_results", -1)),
        "all_batches_successful_and_parsed": bool(batch_results)
        and all(
            result.get("status") == "success"
            and result.get("parse_status") == "success"
            for result in batch_results
        ),
        "raw_responses_and_prompts_not_persisted": all(
            result.get("raw_response_persisted") is False
            and result.get("prompt_persisted") is False
            for result in batch_results
        ),
        "parsed_row_count_exact": len(parsed_rows)
        == int(runtime_report.get("counts", {}).get("parsed_entity_rows", -1)),
        "surface_spec_is_diagnostic_only": surface_spec.get("diagnostic_only")
        is True,
        "surface_spec_is_evaluation_only": surface_spec.get("evaluation_only")
        is True,
        "no_day_test140_or_week": manifest.get("day_test140_accessed") is False
        and manifest.get("week_accessed") is False,
    }
    report = build_wp17_preflight_report(
        case_specs=cases,
        selection_rows=selection_rows,
        parsed_rows=parsed_rows,
        merge_gap_sec=float(manifest.get("occurrence_gap_sec", 60.0)),
        structural_checks=checks,
    )
    report["provenance"] = {
        "a3_root": str(a3_root),
        "a3_manifest_sha256": file_sha256(manifest_path),
        "a3_runtime_report_sha256": file_sha256(runtime_report_path),
        "selection_path": str(selection_path),
        "selection_sha256": file_sha256(selection_path),
        "batch_result_count": len(batch_paths),
        "surface_spec_path": str(surface_path),
        "surface_spec_sha256": file_sha256(surface_path),
        "expected_source_commit": str(args.expected_source_commit),
        "expected_model": str(args.expected_model),
        "analysis_model_calls": 0,
        "day_test140_accessed": False,
        "week_accessed": False,
    }
    out_json = Path(args.out_json)
    _write_json(out_json, report)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_render_markdown(report), encoding="utf-8")
    print(
        "WP17_PREFLIGHT_DONE "
        f"decision={report['decision']} cases={report['case_count']} "
        f"pending_visual={report['pending_visual_review_count']} "
        f"gate={str(report['structural_gate_passed']).lower()}",
        flush=True,
    )
    return out_json


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# MM-Lifelong WP17-0 构建前诊断",
        "",
        f"- 决策：`{report['decision']}`",
        f"- 结构门：`{str(report['structural_gate_passed']).lower()}`",
        f"- 待人工像素复核：`{report['pending_visual_review_count']}`",
        "- 本分析重放既有结构化 OCR rows；model/retrieval/QA/judge 调用均为 0。",
        "",
        "| Case | Frames | Parsed rows | Pre/Post target | Pixel | 分类 | 最近 OCR surface |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for row in report["case_level"]:
        closest = ", ".join(
            f"{value['text']} (d={value['edit_distance']})"
            for value in row["closest_observed_surfaces"][:2]
        ) or "-"
        lines.append(
            f"| {row['case_id']} | {row['selected_frame_count']} | "
            f"{row['parsed_row_count']} | "
            f"{row['pre_admission_target_match_count']}/"
            f"{row['current_admission_target_match_count']} | "
            f"{row['pixel_status']} | {row['category']} | {closest} |"
        )
    lines.extend(
        [
            "",
            "## Admission sweep",
            "",
            "| Support | Lexical | High-value singleton | Target cases | Occurrences |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["admission_grid"]:
        lines.append(
            f"| {row['support']} | "
            f"{str(row['lexical_filter_enabled']).lower()} | "
            f"{str(row['high_value_singleton_enabled']).lower()} | "
            f"{row['target_covered_case_count']} | "
            f"{row['admitted_occurrence_count']} |"
        )
    lines.extend(
        [
            "",
            "Admission sweep 是 diagnostic，不修改既有 WP16 endpoint，也不作为结构 gate。",
            "",
        ]
    )
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"expected JSON object row: {path}")
        rows.append(dict(payload))
    return tuple(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a3-root", required=True)
    parser.add_argument("--surface-spec", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--expected-cases", type=int, default=8)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    return parser.parse_args()


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
