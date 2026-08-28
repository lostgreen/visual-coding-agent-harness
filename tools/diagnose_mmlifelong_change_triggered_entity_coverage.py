#!/usr/bin/env python3
"""Evaluate WP16-7 entity coverage without running retrieval or QA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from vcah.change_triggered_entity_coverage import (
    build_change_triggered_coverage_report,
)
from vcah.change_triggered_entity_occurrence import (
    CHANGE_TRIGGERED_ENTITY_CONTRACT,
)
from vcah.occurrence_field_index import normalize_occurrence_fields
from vcah.occurrence_negative_sidecar import file_sha256


def run(args: argparse.Namespace) -> Path:
    paths = {
        "protocol": Path(args.protocol_spec),
        "field": Path(args.field_spec),
        "relation": Path(args.relation_spec),
    }
    expected_hashes = {
        "protocol": str(args.expected_protocol_sha256),
        "field": str(args.expected_field_spec_sha256),
        "relation": str(args.expected_relation_spec_sha256),
    }
    for label, path in paths.items():
        _require_sha(path, expected_hashes[label], label)
    protocol = _read_json(paths["protocol"])
    if protocol.get("contract") != CHANGE_TRIGGERED_ENTITY_CONTRACT:
        raise ValueError("WP16-7 protocol contract mismatch")

    annotation = dict(protocol.get("anchor_text_expected_annotation", {}) or {})
    annotation_cases = dict(annotation.get("cases", {}) or {})
    field_cases = dict(_read_json(paths["field"]).get("cases", {}) or {})
    relation_cases = dict(_read_json(paths["relation"]).get("cases", {}) or {})
    case_ids = tuple(annotation_cases)
    if len(case_ids) != int(args.expected_cases):
        raise ValueError("frozen annotation case count mismatch")
    if any(
        case_id not in field_cases or case_id not in relation_cases
        for case_id in case_ids
    ):
        raise ValueError("frozen case is missing a field or relation specification")

    case_specs = []
    for case_id in case_ids:
        fields = normalize_occurrence_fields(dict(field_cases[case_id]))
        relation = dict(relation_cases[case_id])
        case_specs.append(
            {
                "case_id": case_id,
                "anchor_text_expected": str(annotation_cases[case_id].get("value", "")),
                "entity_query": list(fields["entity"]["query_terms"]),
                "anchor_intervals": list(relation.get("anchor_intervals", ()) or ()),
            }
        )

    roots = {
        "a1_uniform": Path(args.a1_root),
        "a2_change": Path(args.a2_root),
    }
    arm_data = {arm: _load_arm(root) for arm, root in roots.items()}
    expected_model = str(args.expected_model)
    expected_commit = str(args.expected_source_commit)
    protocol_gap = float(protocol["admission"]["occurrence_merge_gap_sec"])
    tolerance = float(protocol["endpoint_policy"]["coverage_anchor_tolerance_sec"])
    manifests = {arm: value["manifest"] for arm, value in arm_data.items()}
    reports = {arm: value["report"] for arm, value in arm_data.items()}
    occurrences = {arm: value["occurrences"] for arm, value in arm_data.items()}
    selected_counts = {
        arm: int(manifests[arm].get("selected_frame_count", -1)) for arm in roots
    }
    structural_checks = {
        "protocol_frozen_before_outcomes": bool(
            protocol.get("protocol_frozen_before_outcomes")
        ),
        "annotation_frozen_before_outcomes": bool(
            annotation.get("frozen_before_a1_a2_outcomes")
        ),
        "arm_runtime_structural_gates_passed": all(
            bool(reports[arm].get("gates", {}).get("structural_gate_passed"))
            for arm in roots
        ),
        "arm_labels_exact": all(
            manifests[arm].get("arm") == arm and reports[arm].get("arm") == arm
            for arm in roots
        ),
        "source_commit_exact": all(
            manifests[arm].get("source_commit") == expected_commit for arm in roots
        ),
        "actual_model_exact": all(
            manifests[arm].get("actual_model") == expected_model for arm in roots
        ),
        "protocol_sha_exact": all(
            manifests[arm].get("protocol_sha256") == expected_hashes["protocol"]
            for arm in roots
        ),
        "shared_sampling_run": len(
            {str(manifests[arm].get("sampling_report_sha256", "")) for arm in roots}
        )
        == 1,
        "exact_budget_equality": len(set(selected_counts.values())) == 1
        and min(selected_counts.values()) > 0,
        "shared_occurrence_admission": all(
            float(manifests[arm].get("occurrence_gap_sec", -1.0)) == protocol_gap
            for arm in roots
        ),
        "complete_occurrence_lineage": all(
            _occurrence_lineage_valid(row)
            for arm_rows in occurrences.values()
            for row in arm_rows
        ),
        "retrieval_qa_judge_not_run": all(
            reports[arm].get("retrieval_run") is False
            and reports[arm].get("qa_run") is False
            and int(reports[arm].get("judge_calls", -1)) == 0
            for arm in roots
        ),
    }
    report = build_change_triggered_coverage_report(
        case_specs=case_specs,
        arm_occurrences=occurrences,
        tolerance_sec=tolerance,
        structural_checks=structural_checks,
    )
    report["tier0_miss_audit_required_case_ids"] = [
        row["case_id"]
        for row in report["case_level"]
        if not row["arms"]["a2_change"]["covered"]
    ]
    report["provenance"] = {
        "protocol_path": str(paths["protocol"]),
        "protocol_sha256": expected_hashes["protocol"],
        "field_spec_path": str(paths["field"]),
        "field_spec_sha256": expected_hashes["field"],
        "relation_spec_path": str(paths["relation"]),
        "relation_spec_sha256": expected_hashes["relation"],
        "a1_root": str(roots["a1_uniform"]),
        "a2_root": str(roots["a2_change"]),
        "expected_source_commit": expected_commit,
        "expected_model": expected_model,
        "generative_model_calls_during_evaluation": 0,
        "vlm_calls_during_evaluation": 0,
        "judge_calls": 0,
        "day_test140_accessed": False,
        "week_accessed": False,
    }
    if args.historical_a0_report:
        report["historical_a0_fixed3"] = _historical_a0_summary(
            Path(args.historical_a0_report),
            expected_sha256=str(args.expected_historical_a0_sha256),
            case_ids=case_ids,
            yes_case_ids={
                row["case_id"]
                for row in case_specs
                if row["anchor_text_expected"] == "yes"
            },
        )

    out_json = Path(args.out_json)
    _write_json(out_json, report)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_render_markdown(report), encoding="utf-8")
    strict = report["strict_text_expected_yes"]
    print(
        "CHANGE_ENTITY_COVERAGE_DONE "
        f"decision={report['decision']} "
        f"a1={strict['a1_uniform']['count']}/{strict['a1_uniform']['case_count']} "
        f"a2={strict['a2_change']['count']}/{strict['a2_change']['case_count']} "
        f"gate={str(report['structural_gate_passed']).lower()}",
        flush=True,
    )
    return out_json


def _load_arm(root: Path) -> dict[str, Any]:
    return {
        "manifest": _read_json(root / "run_manifest.json"),
        "report": _read_json(root / "entity_occurrence_report.json"),
        "occurrences": _read_jsonl(root / "entity_occurrences.jsonl"),
    }


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


def _historical_a0_summary(
    path: Path,
    *,
    expected_sha256: str,
    case_ids: Sequence[str],
    yes_case_ids: set[str],
) -> dict[str, Any]:
    if not expected_sha256:
        raise ValueError("historical A0 SHA256 is required when A0 is supplied")
    _require_sha(path, expected_sha256, "historical A0 report")
    source = _read_json(path)
    by_case = {
        str(row.get("case_id", "")): bool(row.get("gold_anchor_entity_covered"))
        for row in tuple(source.get("case_level", ()) or ())
        if isinstance(row, Mapping)
    }
    if not set(case_ids) <= set(by_case):
        raise ValueError("historical A0 report does not cover frozen10")
    strict_count = sum(by_case[case_id] for case_id in yes_case_ids)
    all_count = sum(by_case[case_id] for case_id in case_ids)
    return {
        "role": "historical_reproduction_only",
        "admission_not_claimed_identical": True,
        "strict_text_expected_yes": {
            "count": strict_count,
            "case_count": len(yes_case_ids),
            "rate": strict_count / len(yes_case_ids),
        },
        "all_case_secondary": {
            "count": all_count,
            "case_count": len(case_ids),
            "rate": all_count / len(case_ids),
        },
        "source_path": str(path),
        "source_sha256": expected_sha256,
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    strict = report["strict_text_expected_yes"]
    secondary = report["all_case_secondary"]
    paired = report["paired_a2_minus_a1"]
    lines = [
        "# WP16-7 Change-triggered Entity Coverage",
        "",
        f"- 决策：`{report['decision']}`",
        f"- 结构门：`{str(report['structural_gate_passed']).lower()}`",
        "- 主分母：冻结前标为 `anchor_text_expected=yes` 的 8 个 case",
        f"- A1 uniform：`{strict['a1_uniform']['count']}/{strict['a1_uniform']['case_count']}`",
        f"- A2 change：`{strict['a2_change']['count']}/{strict['a2_change']['case_count']}`",
        f"- A2-A1：`{paired['count_delta']}` case；W/T/L = `{paired['wins_ties_losses']['wins']}/{paired['wins_ties_losses']['ties']}/{paired['wins_ties_losses']['losses']}`",
        f"- McNemar exact p（仅报告）：`{paired['mcnemar_exact_two_sided_p']:.4f}`",
        f"- 全 10 case 次要覆盖：A1 `{secondary['a1_uniform']['count']}/10`，A2 `{secondary['a2_change']['count']}/10`",
        "",
        "| Case | 预期有文字 | 实体查询 | A1 | A2 |",
        "|---|---:|---|---:|---:|",
    ]
    for row in report["case_level"]:
        query = " / ".join(row["entity_query"])
        lines.append(
            f"| {row['case_id']} | {row['anchor_text_expected']} | {query} | "
            f"{'yes' if row['arms']['a1_uniform']['covered'] else 'no'} | "
            f"{'yes' if row['arms']['a2_change']['covered'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "端点值没有参与结构门。当前 frozen10 样本量不足；未命中的 case 需要再做 Tier-0 区间诊断，不能把 A3 当成上界。",
            "",
        ]
    )
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
    parser.add_argument("--a1-root", required=True)
    parser.add_argument("--a2-root", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-model", default="pa/gmn-2.5-pr")
    parser.add_argument("--expected-cases", type=int, default=10)
    parser.add_argument("--historical-a0-report")
    parser.add_argument("--expected-historical-a0-sha256", default="")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    return parser.parse_args()


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
