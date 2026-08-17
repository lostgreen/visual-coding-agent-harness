from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from vcah.occurrence_negative_sidecar import file_sha256, stable_digest
from vcah.occurrence_visual_probe import (
    VISUAL_PROBE_CONTRACT,
    audit_visual_probe_manifest,
    build_case_probe_plan,
    finalize_case_probe_plan,
    load_visual_probe_source,
    parse_visual_probe_response,
    visual_probe_prompt,
)
from vcah.virtual_video import VirtualVideoManifest, VirtualVideoSegment


ROOT = Path(__file__).resolve().parents[1]
ANALYZER_PATH = ROOT / "tools" / "analyze_mmlifelong_occurrence_visual_probe.py"
SPEC = importlib.util.spec_from_file_location("occurrence_visual_probe_analysis", ANALYZER_PATH)
assert SPEC is not None and SPEC.loader is not None
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def _source_files(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run" / "cases" / "case-1"
    run_dir.mkdir(parents=True)
    (run_dir / "case.json").write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "question": "Which visit is referenced?",
                "asset_ref": "/asset/game",
            }
        ),
        encoding="utf-8",
    )
    trace = [
        {
            "type": "reasoner_decision",
            "occurrence_ops": [
                {
                    "op": "assess_sufficiency",
                    "set_id": "set-1",
                    "constraints_checked": [
                        {
                            "constraint_id": "c-identity",
                            "constraint_type": "identity",
                            "description": "The visitor wears a red coat.",
                            "support": [],
                        },
                        {
                            "constraint_id": "c-event",
                            "constraint_type": "event",
                            "description": "The visitor opens the gate.",
                            "support": [],
                        },
                        {
                            "constraint_id": "c-outcome",
                            "constraint_type": "outcome",
                            "description": "The score after the visit.",
                            "support": [],
                        },
                    ],
                }
            ],
        },
        {
            "type": "occurrence_sufficiency_decision",
            "set_id": "set-1",
            "scope_occurrence_ids": ["occ-gold", "occ-hard-negative"],
            "verdict": "sufficient",
        },
    ]
    (run_dir / "runtime_summary.json").write_text(
        json.dumps({"trace": trace}), encoding="utf-8"
    )
    occurrence_set = {
        "candidates": [
            {
                "occurrence_id": "occ-gold",
                "time_range": [10.0, 20.0],
                "rank": 2,
                "max_score": 0.8,
                "source_video_ids": ["video-a"],
            },
            {
                "occurrence_id": "occ-hard-negative",
                "time_range": [50.0, 60.0],
                "rank": 1,
                "max_score": 0.9,
                "source_video_ids": ["video-a"],
            },
        ]
    }
    (run_dir / "observation_log.jsonl").write_text(
        json.dumps(
            {
                "attempt_id": "set-1",
                "sampling_config": {"occurrence_set": occurrence_set},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    evaluation_path = tmp_path / "evaluation_case.json"
    evaluation_path.write_text(
        json.dumps({"case_id": "case-1", "clue_intervals": [[12.0, 15.0]]}),
        encoding="utf-8",
    )
    return run_dir, evaluation_path


def _manifest() -> VirtualVideoManifest:
    return VirtualVideoManifest(
        workspace_id="workspace",
        segments=(
            VirtualVideoSegment(
                segment_id="seg-a",
                source_video_id="video-a",
                source_path="/videos/a.mp4",
                source_start_sec=0.0,
                source_end_sec=100.0,
                virtual_start_sec=0.0,
                virtual_end_sec=100.0,
            ),
            VirtualVideoSegment(
                segment_id="seg-b",
                source_video_id="video-b",
                source_path="/videos/b.mp4",
                source_start_sec=0.0,
                source_end_sec=100.0,
                virtual_start_sec=100.0,
                virtual_end_sec=200.0,
            ),
        ),
    )


def _materialized_manifest(tmp_path: Path) -> dict:
    run_dir, evaluation_path = _source_files(tmp_path)
    source = load_visual_probe_source(
        run_dir, evaluation_record_path=evaluation_path
    )
    plan = build_case_probe_plan(source, manifest=_manifest(), seed=20260817)
    assert plan["eligible"] is True
    assert len(plan["constraints"]) == 2
    assert {row["pair_kind"] for row in plan["windows"]} == {
        "matched",
        "mismatched",
        "null",
    }
    null_window = next(row for row in plan["windows"] if row["pair_kind"] == "null")
    assert null_window["source_video_id"] == "video-b"
    materialized = {}
    for window in plan["windows"]:
        frames = []
        start_sec, end_sec = window["time_range"]
        for index in range(8):
            path = Path("frames") / window["visual_observation_id"] / f"{index}.jpg"
            (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / path).write_bytes(b"frame")
            frames.append(
                {
                    "frame_id": f"{window['visual_observation_id']}-fr-{index}",
                    "path": str(path),
                    "virtual_time_sec": start_sec
                    + (end_sec - start_sec) * (index + 0.5) / 8,
                    "segment_id": "seg",
                    "source_video_id": "video",
                }
            )
        materialized[window["visual_observation_id"]] = {
            "executed": True,
            "frames": frames,
        }
    finalized = finalize_case_probe_plan(
        plan, materialized_windows=materialized
    )
    return {
        "contract": VISUAL_PROBE_CONTRACT,
        "engineering_thresholds": {
            "matched_minus_mismatched_support_rate_min": 0.4,
            "null_support_rate_max": 0.15,
            "endpoint_values_are_structural_gates": False,
        },
        "cases": [finalized],
    }


def test_visual_probe_source_plan_and_provenance_are_mechanically_bound(
    tmp_path: Path,
) -> None:
    manifest = _materialized_manifest(tmp_path)
    audit = audit_visual_probe_manifest(manifest, root=tmp_path)
    assert audit["structural_gate_passed"] is True
    assert audit["counts"]["item_count"] == 6
    assert audit["counts"]["visual_observation_count"] == 3
    assert audit["counts"]["silent_locator_drop_count"] == 0

    broken = json.loads(json.dumps(manifest))
    broken["cases"][0]["items"][0]["occurrence_id"] = "wrong-occurrence"
    broken_audit = audit_visual_probe_manifest(broken, root=tmp_path)
    assert broken_audit["structural_gate_passed"] is False
    assert broken_audit["counts"]["wrong_occurrence_binding_count"] == 1


def test_visual_probe_prompt_is_blind_and_parser_is_strict(tmp_path: Path) -> None:
    manifest = _materialized_manifest(tmp_path)
    item = manifest["cases"][0]["items"][0]
    prompt = visual_probe_prompt(item)
    assert item["constraint_description"] in prompt
    assert item["case_id"] not in prompt
    assert item["pair_kind"] not in prompt
    assert item["occurrence_id"] not in prompt
    assert parse_visual_probe_response('{"verdict":"supported"}') == "supported"
    assert parse_visual_probe_response('```json\n{"verdict":"unknown"}\n```') == "unknown"
    assert parse_visual_probe_response('{"verdict":"supported","answer":"A"}') is None
    assert parse_visual_probe_response('{"verdict":"maybe"}') is None


def test_visual_probe_analysis_separates_endpoints_from_validity(tmp_path: Path) -> None:
    manifest = _materialized_manifest(tmp_path)
    probe_root = tmp_path / "probe"
    probe_root.mkdir()
    # The synthetic frame paths were created relative to tmp_path.
    manifest_for_root = json.loads(json.dumps(manifest))
    for case in manifest_for_root["cases"]:
        for window in case["windows"]:
            for frame in window["frames"]:
                source = tmp_path / frame["path"]
                target = probe_root / frame["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
    manifest_path = probe_root / "probe_manifest.json"
    manifest_path.write_text(json.dumps(manifest_for_root), encoding="utf-8")
    results = []
    for item in manifest_for_root["cases"][0]["items"]:
        verdict = "supported" if item["pair_kind"] == "matched" else "unknown"
        results.append(
            {
                "item_id": item["item_id"],
                "visual_observation_id": item["visual_observation_id"],
                "status": "success",
                "verdict": verdict,
                "actual_model": "runtime-visual",
                "item_digest": stable_digest(item),
                "prompt_persisted": False,
                "raw_response_persisted": False,
            }
        )
    report = ANALYZER.build_report(
        manifest_for_root,
        results,
        probe_root=probe_root,
        run_manifest={
            "actual_model": "runtime-visual",
            "probe_manifest_sha256": file_sha256(manifest_path),
            "agent_behavior_changed": False,
        },
        expected_model="runtime-visual",
        expected_items=6,
        bootstrap_samples=100,
        seed=7,
    )
    assert report["structural_gate_passed"] is True
    assert report["primary"]["matched_supported_rate"] == 1.0
    assert report["primary"]["mismatched_supported_rate"] == 0.0
    assert report["primary"]["null_supported_rate"] == 0.0
    assert report["verifier_discriminability_passed"] is True
    assert report["decision"] == "PROCEED_TO_WP14_3_SHADOW"
    assert report["efficacy_claim_allowed"] is False
