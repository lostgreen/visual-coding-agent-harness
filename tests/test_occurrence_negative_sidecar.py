from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from vcah.occurrence_negative_sidecar import (
    load_negative_sidecar_snapshot,
    negative_sidecar_prompt,
    parse_negative_sidecar_response,
    validate_negative_sidecar_output,
)


ANALYZER_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "analyze_mmlifelong_occurrence_negative_sidecar.py"
)
SPEC = importlib.util.spec_from_file_location(
    "negative_sidecar_analysis", ANALYZER_PATH
)
assert SPEC and SPEC.loader
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)

AUDIT_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "audit_mmlifelong_occurrence_negative_sidecar.py"
)
AUDIT_SPEC = importlib.util.spec_from_file_location(
    "negative_sidecar_audit", AUDIT_PATH
)
assert AUDIT_SPEC and AUDIT_SPEC.loader
AUDIT = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(AUDIT)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _snapshot(tmp_path: Path):
    case_id = "case-1"
    run_dir = tmp_path / "run" / "cases" / case_id
    _write_json(
        run_dir / "case.json",
        {
            "case_id": case_id,
            "question": "Which event happened first?",
            "options": {"A": "Open", "B": "Close"},
        },
    )
    _write_json(
        run_dir / "runtime_summary.json",
        {
            "trace": [
                {
                    "type": "occurrence_evidence_declaration",
                    "set_id": "set-1",
                    "scope_occurrence_ids": ["occ-1", "occ-2"],
                    "constraints": [
                        {
                            "constraint_id": "c-1",
                            "constraint_type": "event",
                            "description": "The opening event occurs.",
                            "supported_candidates": [
                                {
                                    "occurrence_id": "occ-1",
                                    "evidence_passage_ids": ["p-1"],
                                }
                            ],
                            "contradicted_candidates": [],
                        }
                    ],
                },
                {
                    "type": "occurrence_sufficiency_decision",
                    "set_id": "set-1",
                },
            ]
        },
    )
    fixture_path = tmp_path / "fixtures" / "cases" / f"{case_id}.json"
    _write_json(
        fixture_path,
        {
            "case_id": case_id,
            "packets": [
                {
                    "packet": {
                        "occurrence_set": {
                            "attempt_id": "set-1",
                            "candidates": [
                                {
                                    "occurrence_id": "occ-1",
                                    "rank": 1,
                                    "time_range": [0, 5],
                                    "passage_ids": ["p-1"],
                                },
                                {
                                    "occurrence_id": "occ-2",
                                    "rank": 2,
                                    "time_range": [10, 15],
                                    "passage_ids": ["p-2"],
                                },
                            ],
                        },
                        "hits": [
                            {
                                "passage_id": "p-1",
                                "virtual_start_sec": 0,
                                "virtual_end_sec": 5,
                                "text": "The door remains closed.",
                            },
                            {
                                "passage_id": "p-2",
                                "virtual_start_sec": 10,
                                "virtual_end_sec": 15,
                                "text": "The door opens.",
                            },
                        ],
                    }
                }
            ],
        },
    )
    return load_negative_sidecar_snapshot(
        run_dir,
        replay_fixture_path=fixture_path,
    )


def test_snapshot_strips_positive_and_selection_state(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    payload = snapshot.model_payload()
    serialized = json.dumps(payload, sort_keys=True)

    assert [row["occurrence_id"] for row in snapshot.candidates] == ["occ-1", "occ-2"]
    assert "supported_candidates" not in serialized
    assert "contradicted_candidates" not in serialized
    assert "winner_occurrence_id" not in serialized
    assert "selected_occurrence_ids" not in serialized
    assert "support_count_by_occurrence" not in serialized
    assert "margin" not in serialized
    prompt = negative_sidecar_prompt(snapshot)
    assert "positive support" in prompt
    assert '"contradictions"' in prompt


def test_negative_output_validation_is_scope_and_passage_bound(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    valid = parse_negative_sidecar_response(
        json.dumps(
            {
                "contradictions": [
                    {
                        "constraint_id": "c-1",
                        "occurrence_id": "occ-1",
                        "evidence_passage_ids": ["p-1"],
                    }
                ]
            }
        )
    )
    rows, errors = validate_negative_sidecar_output(valid, snapshot=snapshot)
    assert not errors
    assert rows[0]["constraint_type"] == "event"

    invalid = {
        "contradictions": [
            {
                "constraint_id": "c-1",
                "occurrence_id": "occ-1",
                "evidence_passage_ids": ["p-2"],
            }
        ],
        "winner": "occ-1",
    }
    rows, errors = validate_negative_sidecar_output(invalid, snapshot=snapshot)
    assert not rows
    assert "negative_sidecar_top_level_field_invalid" in errors
    assert "negative_sidecar_passage_not_visible" in errors


def test_input_audit_reconstructs_without_model_calls(tmp_path: Path) -> None:
    _snapshot(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, {"cases": [{"case_id": "case-1"}]})

    report = AUDIT.build_audit(
        positive_run_root=tmp_path / "run",
        replay_fixture_root=tmp_path / "fixtures",
        case_manifest=manifest_path,
        expected_cases=1,
    )

    assert report["structural_gate_passed"] is True
    assert report["successful_snapshot_count"] == 1
    assert report["model_calls_used"] is False


def test_two_repeat_analysis_qualifies_only_at_frozen_working_point() -> None:
    false_ids = [f"negative-{index:02d}" for index in range(12)]
    positive_ids = [f"positive-{index:02d}" for index in range(8)]
    case_ids = [*false_ids, *positive_ids]
    frozen_rows = []
    repeats = {"r1": {}, "r2": {}}
    for index, case_id in enumerate(case_ids):
        candidate_present = case_id in positive_ids
        winner = f"{case_id}-winner"
        frozen_rows.append(
            {
                "arm": "a4",
                "case_id": case_id,
                "candidate_recall_resolved_set": candidate_present,
                "final_resolution": "selected",
                "selected_occurrence_ids": [winner],
                "osa_strict": candidate_present,
            }
        )
        contradicted = not candidate_present and index < 5
        contradiction_rows = (
            [
                {
                    "constraint_id": "identity-1",
                    "constraint_type": "identity",
                    "occurrence_id": winner,
                    "evidence_passage_ids": [f"passage-{index:02d}"],
                }
            ]
            if contradicted
            else []
        )
        for label in repeats:
            repeats[label][case_id] = {
                "case_id": case_id,
                "repeat_label": label,
                "status": "success",
                "actual_model": "test-model",
                "live_model_call": True,
                "snapshot_digest": f"snapshot-{case_id}",
                "contradiction_rows": contradiction_rows,
                "no_oracle_input_gate_passed": True,
                "positive_support_visible_to_model": False,
                "selection_state_visible_to_model": False,
                "workspace_write_enabled": False,
                "reasoner_context_write_enabled": False,
                "raw_response_persisted": False,
                "prompt_persisted": False,
            }

    report = ANALYZER.build_report(
        repeats,
        frozen_rows=frozen_rows,
        expected_cases=20,
    )

    assert report["structural_gates"]["passed"] is True
    assert report["per_repeat"]["r1"]["false_winner_contradicted_count"] == 5
    assert report["per_repeat"]["r1"]["positive_winner_contradicted_count"] == 0
    assert report["stability"]["winner_flag_exact_agreement"] is True
    assert report["decision"] == "QUALIFIES_FOR_HARD_GUARD_EXPERIMENT"

    canary = ANALYZER.build_report(
        repeats,
        frozen_rows=frozen_rows,
        expected_cases=20,
        expected_false_commits=12,
        expected_positive_commits=8,
        structural_only=True,
    )
    assert canary["structural_gates"]["passed"] is True
    assert canary["decision"] == "STRUCTURAL_CANARY_ONLY"
