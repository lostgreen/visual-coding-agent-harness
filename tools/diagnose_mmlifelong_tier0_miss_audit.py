#!/usr/bin/env python3
"""Audit WP16-7 misses with blind OCR on frozen official intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from vcah.change_triggered_entity_coverage import build_tier0_miss_audit_report
from vcah.change_triggered_entity_occurrence import CHANGE_TRIGGERED_ENTITY_CONTRACT
from vcah.occurrence_field_index import normalize_occurrence_fields
from vcah.occurrence_negative_sidecar import file_sha256


DIAGNOSTIC_ARM = "a3_tier0_diagnostic"


def run(args: argparse.Namespace) -> Path:
    paths = {
        "protocol": Path(args.protocol_spec),
        "field": Path(args.field_spec),
        "relation": Path(args.relation_spec),
        "coverage": Path(args.coverage_report),
    }
    expected_hashes = {
        "protocol": str(args.expected_protocol_sha256),
        "field": str(args.expected_field_spec_sha256),
        "relation": str(args.expected_relation_spec_sha256),
        "coverage": str(args.expected_coverage_report_sha256),
    }
    for label, path in paths.items():
        _require_sha(path, expected_hashes[label], label)
    protocol = _read_json(paths["protocol"])
    field = _read_json(paths["field"])
    relation = _read_json(paths["relation"])
    coverage = _read_json(paths["coverage"])
    if protocol.get("contract") != CHANGE_TRIGGERED_ENTITY_CONTRACT:
        raise ValueError("WP16-7 protocol contract mismatch")

    annotation = dict(protocol.get("anchor_text_expected_annotation", {}) or {})
    annotation_cases = dict(annotation.get("cases", {}) or {})
    field_cases = dict(field.get("cases", {}) or {})
    relation_cases = dict(relation.get("cases", {}) or {})
    miss_ids = tuple(
        str(value)
        for value in tuple(
            coverage.get("tier0_miss_audit_required_case_ids", ()) or ()
        )
    )
    if len(miss_ids) != int(args.expected_cases) or len(set(miss_ids)) != len(
        miss_ids
    ):
        raise ValueError("frozen Tier-0 miss case set mismatch")
    if any(
        case_id not in annotation_cases
        or case_id not in field_cases
        or case_id not in relation_cases
        for case_id in miss_ids
    ):
        raise ValueError("miss case lacks frozen annotation, field, or relation")

    case_rows = []
    for case_id in miss_ids:
        fields = normalize_occurrence_fields(dict(field_cases[case_id]))
        case_rows.append(
            {
                "case_id": case_id,
                "anchor_text_expected": str(
                    annotation_cases[case_id].get("value", "")
                ),
                "entity_query": list(fields["entity"]["query_terms"]),
                "anchor_intervals": list(
                    relation_cases[case_id].get("anchor_intervals", ()) or ()
                ),
            }
        )

    a3_root = Path(args.a3_root)
    manifest = _read_json(a3_root / "run_manifest.json")
    runtime_report = _read_json(a3_root / "entity_occurrence_report.json")
    occurrences = _read_jsonl(a3_root / "entity_occurrences.jsonl")
    sampling_root = Path(str(manifest.get("sampling_root", "")))
    diagnostic_manifest = _read_json(sampling_root / "tier0_manifest.json")
    diagnostic_sampling = _read_json(sampling_root / "sampling_report.json")
    selected = _read_jsonl(
        sampling_root / "selections" / f"{DIAGNOSTIC_ARM}.jsonl"
    )
    selected_case_ids = {
        case_id
        for row in selected
        for case_id in tuple(row.get("diagnostic_case_ids", ()) or ())
    }
    protocol_gap = float(protocol["admission"]["occurrence_merge_gap_sec"])
    structural_checks = {
        "frozen_coverage_structural_gate_passed": bool(
            coverage.get("structural_gate_passed")
        ),
        "frozen_coverage_requires_tier0_miss_audit": coverage.get("decision")
        == "NO_GO_PENDING_TIER0_MISS_AUDIT",
        "coverage_source_commit_exact": coverage.get("provenance", {}).get(
            "expected_source_commit"
        )
        == str(args.expected_endpoint_source_commit),
        "a3_runtime_structural_gate_passed": bool(
            runtime_report.get("gates", {}).get("structural_gate_passed")
        ),
        "a3_arm_exact": manifest.get("arm") == DIAGNOSTIC_ARM
        and runtime_report.get("arm") == DIAGNOSTIC_ARM,
        "a3_source_commit_exact": manifest.get("source_commit")
        == str(args.expected_a3_source_commit),
        "a3_model_exact": manifest.get("actual_model") == str(args.expected_model),
        "protocol_sha_exact": manifest.get("protocol_sha256")
        == expected_hashes["protocol"]
        and diagnostic_manifest.get("protocol_sha256")
        == expected_hashes["protocol"],
        "relation_sha_exact": diagnostic_manifest.get("relation_spec_sha256")
        == expected_hashes["relation"],
        "coverage_report_sha_exact": diagnostic_manifest.get(
            "frozen_coverage_report_sha256"
        )
        == expected_hashes["coverage"],
        "miss_case_set_exact": tuple(
            diagnostic_manifest.get("diagnostic_case_ids", ()) or ()
        )
        == miss_ids
        and selected_case_ids == set(miss_ids),
        "diagnostic_sampling_structural_gate_passed": bool(
            diagnostic_sampling.get("gates", {}).get("structural_gate_passed")
        ),
        "official_intervals_visible_only_to_sampler": diagnostic_manifest.get(
            "official_intervals_visible_to_sampling"
        )
        is True
        and manifest.get("official_intervals_visible_to_model") is False,
        "question_options_answer_caption_blind_to_model": all(
            manifest.get(key) is False
            for key in (
                "question_visible_to_model",
                "options_visible_to_model",
                "answer_visible_to_model",
                "caption_text_visible_to_model",
            )
        ),
        "diagnostic_not_endpoint_or_upper_bound": diagnostic_manifest.get(
            "diagnostic_only"
        )
        is True
        and diagnostic_manifest.get("endpoint_evaluation") is False
        and diagnostic_manifest.get("upper_bound_claim") is False,
        "selected_frame_count_exact": len(selected)
        == int(manifest.get("selected_frame_count", -1)),
        "shared_occurrence_admission": float(
            manifest.get("occurrence_gap_sec", -1.0)
        )
        == protocol_gap,
        "complete_occurrence_lineage": all(
            _occurrence_lineage_valid(row) for row in occurrences
        ),
        "retrieval_qa_judge_not_run": runtime_report.get("retrieval_run") is False
        and runtime_report.get("qa_run") is False
        and int(runtime_report.get("judge_calls", -1)) == 0,
        "day_test140_and_week_not_accessed": manifest.get("day_test140_accessed")
        is False
        and manifest.get("week_accessed") is False,
    }
    report = build_tier0_miss_audit_report(
        case_rows=case_rows,
        diagnostic_occurrences=occurrences,
        structural_checks=structural_checks,
    )
    report["provenance"] = {
        "protocol_path": str(paths["protocol"]),
        "protocol_sha256": expected_hashes["protocol"],
        "field_spec_path": str(paths["field"]),
        "field_spec_sha256": expected_hashes["field"],
        "relation_spec_path": str(paths["relation"]),
        "relation_spec_sha256": expected_hashes["relation"],
        "coverage_report": str(paths["coverage"]),
        "coverage_report_sha256": expected_hashes["coverage"],
        "a3_root": str(a3_root),
        "expected_endpoint_source_commit": str(
            args.expected_endpoint_source_commit
        ),
        "expected_a3_source_commit": str(args.expected_a3_source_commit),
        "expected_model": str(args.expected_model),
        "generative_model_calls_during_evaluation": 0,
        "vlm_calls_during_evaluation": 0,
        "judge_calls": 0,
        "day_test140_accessed": False,
        "week_accessed": False,
    }
    out_json = Path(args.out_json)
    _write_json(out_json, report)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_render_markdown(report), encoding="utf-8")
    strict = report["strict_text_expected_yes"]
    print(
        "TIER0_MISS_AUDIT_DONE "
        f"decision={report['decision']} "
        f"recovered={strict['recovered_count']}/{strict['case_count']} "
        f"gate={str(report['structural_gate_passed']).lower()}",
        flush=True,
    )
    return out_json


def _occurrence_lineage_valid(row: Mapping[str, Any]) -> bool:
    return (
        row.get("contract") == CHANGE_TRIGGERED_ENTITY_CONTRACT
        and bool(str(row.get("occurrence_id", "") or ""))
        and bool(str(row.get("text", "") or ""))
        and isinstance(row.get("occurrence_start_sec"), (int, float))
        and isinstance(row.get("occurrence_end_sec"), (int, float))
        and bool(tuple(row.get("frame_ids", ()) or ()))
        and int(row.get("support_count", 0) or 0) >= 1
    )


def _render_markdown(report: Mapping[str, Any]) -> str:
    strict = report["strict_text_expected_yes"]
    lines = [
        "# WP16-7 Tier-0 未命中审计",
        "",
        f"- 决策：`{report['decision']}`",
        f"- 结构门：`{str(report['structural_gate_passed']).lower()}`",
        f"- 预期有文字的 case 中，密集区间 OCR 恢复：`{strict['recovered_count']}/{strict['case_count']}`",
        "- 该结果只用于区分采样/reader 问题与未恢复的可见实体，不是 endpoint，也不是上界。",
        "",
        "| Case | 文字预期 | 密集 OCR 命中 | 诊断分类 | 中文解释 |",
        "|---|---:|---:|---|---|",
    ]
    explanations = {
        "ui_text_exists_reader_or_resolution_failure": "官方区间内能恢复目标实体；A1/A2 的失败来自采样或读取分辨率。",
        "no_ui_text_visual_event_or_state": "该锚点原本主要是视觉事件/状态，不应期待实体文字召回。",
        "annotation_uncertain": "是否存在稳定 UI 文字本来就不确定，不能据此判 reader。",
        "other": "官方区间密集 OCR 仍未恢复目标词；可能无稳定文字，也可能 reader 仍不足。",
    }
    for row in report["case_level"]:
        category = row["category"]
        lines.append(
            f"| {row['case_id']} | {row['anchor_text_expected']} | "
            f"{row['diagnostic_match_count']} | {category} | {explanations[category]} |"
        )
    lines.extend(["", "未运行 retrieval、QA 或 judge。", ""])
    return "\n".join(lines)


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"expected JSON object row: {path}")
            rows.append(dict(payload))
    return tuple(rows)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _require_sha(path: Path, expected: str, label: str) -> None:
    if file_sha256(path) != str(expected):
        raise ValueError(f"{label} SHA256 mismatch")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-spec", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--field-spec", required=True)
    parser.add_argument("--expected-field-spec-sha256", required=True)
    parser.add_argument("--relation-spec", required=True)
    parser.add_argument("--expected-relation-spec-sha256", required=True)
    parser.add_argument("--coverage-report", required=True)
    parser.add_argument("--expected-coverage-report-sha256", required=True)
    parser.add_argument("--a3-root", required=True)
    parser.add_argument("--expected-endpoint-source-commit", required=True)
    parser.add_argument("--expected-a3-source-commit", required=True)
    parser.add_argument("--expected-model", default="pa/gmn-2.5-pr")
    parser.add_argument("--expected-cases", type=int, default=10)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    return parser.parse_args()


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
