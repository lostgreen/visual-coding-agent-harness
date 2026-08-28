#!/usr/bin/env python3
"""Select frozen official-interval Tier-0 frames for WP16-7 miss diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from vcah.change_triggered_entity_occurrence import (
    CHANGE_TRIGGERED_ENTITY_CONTRACT,
    select_interval_diagnostic,
    write_jsonl,
)
from vcah.occurrence_negative_sidecar import file_sha256, stable_digest


DIAGNOSTIC_ARM = "a3_tier0_diagnostic"


def run(args: argparse.Namespace) -> Path:
    protocol_path = Path(args.protocol_spec)
    relation_path = Path(args.relation_spec)
    _require_sha(protocol_path, args.expected_protocol_sha256, "protocol")
    _require_sha(relation_path, args.expected_relation_spec_sha256, "relation")
    protocol = _read_json(protocol_path)
    relation = _read_json(relation_path)
    if protocol.get("contract") != CHANGE_TRIGGERED_ENTITY_CONTRACT:
        raise ValueError("WP16-7 protocol contract mismatch")

    sampling_root = Path(args.source_sampling_root)
    source_manifest_path = sampling_root / "tier0_manifest.json"
    source_report_path = sampling_root / "sampling_report.json"
    source_manifest = _read_json(source_manifest_path)
    source_report = _read_json(source_report_path)
    if not bool(source_report.get("gates", {}).get("structural_gate_passed")):
        raise ValueError("source sampling structural gate did not pass")
    if source_manifest.get("protocol_sha256") != str(
        args.expected_protocol_sha256
    ):
        raise ValueError("source sampling protocol SHA mismatch")

    coverage_path = Path(args.coverage_report)
    coverage = _read_json(coverage_path)
    if not bool(coverage.get("structural_gate_passed")):
        raise ValueError("frozen coverage structural gate did not pass")
    miss_ids = tuple(
        dict.fromkeys(
            str(value)
            for value in tuple(
                coverage.get("tier0_miss_audit_required_case_ids", ()) or ()
            )
            if str(value)
        )
    )
    if len(miss_ids) != int(args.expected_cases):
        raise ValueError("Tier-0 miss case count mismatch")

    relation_cases = dict(relation.get("cases", {}) or {})
    missing_relations = tuple(case_id for case_id in miss_ids if case_id not in relation_cases)
    if missing_relations:
        raise ValueError(f"miss cases lack relation intervals: {missing_relations}")
    intervals_by_case = {
        case_id: tuple(relation_cases[case_id].get("anchor_intervals", ()) or ())
        for case_id in miss_ids
    }

    observations = []
    score_paths = []
    for segment in tuple(source_manifest.get("segments", ()) or ()):
        score_path = Path(str(segment.get("score_path", "")))
        if not score_path.is_file():
            raise FileNotFoundError(f"missing Tier-0 score file: {score_path}")
        if file_sha256(score_path) != str(segment.get("score_sha256", "")):
            raise ValueError(f"Tier-0 score SHA mismatch: {score_path}")
        rows = _read_jsonl(score_path)
        if len(rows) != int(segment.get("observation_count", -1)):
            raise ValueError(f"Tier-0 score count mismatch: {score_path}")
        observations.extend(rows)
        score_paths.append(str(score_path))
    if len(observations) != int(source_manifest.get("observation_count", -1)):
        raise ValueError("source Tier-0 observation count mismatch")

    selected = select_interval_diagnostic(
        observations,
        intervals_by_case=intervals_by_case,
    )
    sampled_case_ids = {
        case_id
        for row in selected
        for case_id in tuple(row.get("diagnostic_case_ids", ()) or ())
    }
    out_root = Path(args.out_root)
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"diagnostic output is not empty: {out_root}")
    selection_path = out_root / "selections" / f"{DIAGNOSTIC_ARM}.jsonl"
    write_jsonl(selection_path, selected)

    tier0_manifest = {
        "schema_version": "MMLifelongTier0MissDiagnosticManifestV1",
        "contract": CHANGE_TRIGGERED_ENTITY_CONTRACT,
        "source_commit": str(args.source_commit),
        "protocol_path": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "relation_spec_path": str(relation_path),
        "relation_spec_sha256": file_sha256(relation_path),
        "workspace_root": str(source_manifest.get("workspace_root", "")),
        "workspace_id": str(source_manifest.get("workspace_id", "")),
        "asset_root": str(source_manifest.get("asset_root", "")),
        "tier0_fps": float(source_manifest.get("tier0_fps", 0.0)),
        "tier0_width": int(source_manifest.get("tier0_width", 0)),
        "tier0_height": int(source_manifest.get("tier0_height", 0)),
        "segment_count": int(source_manifest.get("segment_count", 0)),
        "observation_count": len(observations),
        "source_score_paths": score_paths,
        "source_sampling_root": str(sampling_root),
        "source_sampling_manifest_sha256": file_sha256(source_manifest_path),
        "source_sampling_report_sha256": file_sha256(source_report_path),
        "frozen_coverage_report": str(coverage_path),
        "frozen_coverage_report_sha256": file_sha256(coverage_path),
        "diagnostic_case_ids": list(miss_ids),
        "selected_frame_count": len(selected),
        "selection_digest": stable_digest(selected),
        "official_intervals_visible_to_sampling": True,
        "question_visible_to_sampling": False,
        "options_visible_to_sampling": False,
        "answer_visible_to_sampling": False,
        "caption_visible_to_sampling": False,
        "diagnostic_only": True,
        "endpoint_evaluation": False,
        "upper_bound_claim": False,
        "video_copied": False,
        "dense_frames_persisted": False,
        "day_test140_accessed": False,
        "week_accessed": False,
    }
    _write_json(out_root / "tier0_manifest.json", tier0_manifest)
    checks = {
        "source_sampling_structural_gate_passed": True,
        "frozen_coverage_structural_gate_passed": True,
        "protocol_sha_exact": tier0_manifest["protocol_sha256"]
        == str(args.expected_protocol_sha256),
        "relation_sha_exact": tier0_manifest["relation_spec_sha256"]
        == str(args.expected_relation_spec_sha256),
        "miss_case_count_exact": len(miss_ids) == int(args.expected_cases),
        "every_miss_case_has_a_selected_frame": sampled_case_ids == set(miss_ids),
        "selection_uses_existing_tier0_observations": all(
            row.get("contract") == CHANGE_TRIGGERED_ENTITY_CONTRACT
            for row in selected
        ),
        "selection_arm_exact": all(
            row.get("selection_arm") == DIAGNOSTIC_ARM for row in selected
        ),
        "diagnostic_not_endpoint": tier0_manifest["endpoint_evaluation"] is False,
        "diagnostic_not_upper_bound": tier0_manifest["upper_bound_claim"] is False,
        "question_options_answer_caption_blind": all(
            tier0_manifest[key] is False
            for key in (
                "question_visible_to_sampling",
                "options_visible_to_sampling",
                "answer_visible_to_sampling",
                "caption_visible_to_sampling",
            )
        ),
        "zero_video_or_dense_frame_copy": tier0_manifest["video_copied"] is False
        and tier0_manifest["dense_frames_persisted"] is False,
    }
    checks["structural_gate_passed"] = all(checks.values())
    report = {
        "schema_version": "MMLifelongTier0MissDiagnosticSamplingReportV1",
        "contract": CHANGE_TRIGGERED_ENTITY_CONTRACT,
        "decision": (
            "DIAGNOSTIC_SAMPLING_READY"
            if checks["structural_gate_passed"]
            else "STRUCTURAL_FAILURE"
        ),
        "counts": {
            "miss_cases": len(miss_ids),
            "source_tier0_observations": len(observations),
            "selected_frames": len(selected),
        },
        "selection_paths": {DIAGNOSTIC_ARM: str(selection_path)},
        "selection_sha256": {DIAGNOSTIC_ARM: file_sha256(selection_path)},
        "gates": checks,
        "diagnostic_only": True,
        "endpoint_values_were_not_computed": True,
        "retrieval_run": False,
        "qa_run": False,
        "judge_calls": 0,
    }
    report_path = out_root / "sampling_report.json"
    _write_json(report_path, report)
    (out_root / "sampling_report.md").write_text(
        _render_markdown(report), encoding="utf-8"
    )
    print(
        "TIER0_MISS_DIAGNOSTIC_READY "
        f"cases={len(miss_ids)} frames={len(selected)} "
        f"gate={str(checks['structural_gate_passed']).lower()}",
        flush=True,
    )
    return report_path


def _render_markdown(report: Mapping[str, Any]) -> str:
    counts = report["counts"]
    return "\n".join(
        [
            "# WP16-7 Tier-0 Miss Diagnostic Sampling",
            "",
            f"- Decision: `{report['decision']}`",
            f"- Miss cases: `{counts['miss_cases']}`",
            f"- Reused Tier-0 observations: `{counts['source_tier0_observations']}`",
            f"- Official-interval frames: `{counts['selected_frames']}`",
            f"- Structural gate: `{str(report['gates']['structural_gate_passed']).lower()}`",
            "",
            "This oracle-interval sample is diagnostic only. It is neither an endpoint nor an upper bound.",
            "",
        ]
    )


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
    parser.add_argument("--source-sampling-root", required=True)
    parser.add_argument("--coverage-report", required=True)
    parser.add_argument("--protocol-spec", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--relation-spec", required=True)
    parser.add_argument("--expected-relation-spec-sha256", required=True)
    parser.add_argument("--expected-cases", type=int, default=10)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out-root", required=True)
    return parser.parse_args()


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
