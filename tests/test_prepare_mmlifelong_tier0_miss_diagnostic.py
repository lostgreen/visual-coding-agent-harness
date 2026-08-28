from __future__ import annotations

from argparse import Namespace
import importlib.util
import json
from pathlib import Path

from vcah.change_triggered_entity_occurrence import CHANGE_TRIGGERED_ENTITY_CONTRACT
from vcah.occurrence_negative_sidecar import file_sha256


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "prepare_mmlifelong_tier0_miss_diagnostic.py"
)
SPEC = importlib.util.spec_from_file_location("tier0_miss_diagnostic", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_prepare_diagnostic_reuses_scores_and_marks_oracle_sampling(
    tmp_path: Path,
) -> None:
    protocol_path = tmp_path / "protocol.json"
    relation_path = tmp_path / "relation.json"
    coverage_path = tmp_path / "coverage.json"
    score_path = tmp_path / "source" / "tier0_scores" / "segment.jsonl"
    rows = [
        {
            "contract": CHANGE_TRIGGERED_ENTITY_CONTRACT,
            "segment_id": "segment-a",
            "source_video_id": "video-a",
            "source_path": "/existing/video.mp4",
            "tier0_frame_index": index,
            "source_time_sec": float(index),
            "virtual_time_sec": float(index),
            "global_change_score": 0.0,
            "ui_change_score": 0.0,
            "selection_score": 0.0,
        }
        for index in range(6)
    ]
    _write_json(protocol_path, {"contract": CHANGE_TRIGGERED_ENTITY_CONTRACT})
    _write_json(
        relation_path,
        {
            "cases": {
                "case-a": {"anchor_intervals": [[1.0, 2.0]]},
                "case-b": {"anchor_intervals": [[4.0, 5.0]]},
            }
        },
    )
    _write_json(
        coverage_path,
        {
            "structural_gate_passed": True,
            "tier0_miss_audit_required_case_ids": ["case-a", "case-b"],
        },
    )
    _write_jsonl(score_path, rows)
    source_root = tmp_path / "source"
    _write_json(
        source_root / "tier0_manifest.json",
        {
            "protocol_sha256": file_sha256(protocol_path),
            "workspace_root": "/workspace",
            "workspace_id": "workspace-a",
            "asset_root": "/assets",
            "tier0_fps": 1.0,
            "tier0_width": 160,
            "tier0_height": 90,
            "segment_count": 1,
            "observation_count": len(rows),
            "segments": [
                {
                    "score_path": str(score_path),
                    "score_sha256": file_sha256(score_path),
                    "observation_count": len(rows),
                }
            ],
        },
    )
    _write_json(
        source_root / "sampling_report.json",
        {"gates": {"structural_gate_passed": True}},
    )
    out_root = tmp_path / "diagnostic"
    report_path = diagnostic.run(
        Namespace(
            source_sampling_root=str(source_root),
            coverage_report=str(coverage_path),
            protocol_spec=str(protocol_path),
            expected_protocol_sha256=file_sha256(protocol_path),
            relation_spec=str(relation_path),
            expected_relation_spec_sha256=file_sha256(relation_path),
            expected_cases=2,
            source_commit="commit-a",
            out_root=str(out_root),
        )
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        (out_root / "tier0_manifest.json").read_text(encoding="utf-8")
    )
    selection = [
        json.loads(line)
        for line in (
            out_root / "selections" / "a3_tier0_diagnostic.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert report["decision"] == "DIAGNOSTIC_SAMPLING_READY"
    assert [row["tier0_frame_index"] for row in selection] == [1, 2, 4, 5]
    assert manifest["official_intervals_visible_to_sampling"] is True
    assert manifest["question_visible_to_sampling"] is False
    assert manifest["diagnostic_only"] is True
    assert manifest["endpoint_evaluation"] is False
    assert manifest["upper_bound_claim"] is False
