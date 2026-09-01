from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "diagnose_mmlifelong_wp17_slot_v10.py"
SPEC = importlib.util.spec_from_file_location("wp17_slot_v10_diagnosis", MODULE_PATH)
assert SPEC and SPEC.loader
DIAGNOSE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSE)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _row(arm: str, *, abstain: bool = False) -> dict:
    return {
        "status": "success",
        "arm": arm,
        "slot_transaction_abstained": abstain,
        "ser_endpoint_eligible": not abstain,
        "illegal_operation_count": int(abstain),
        "model_output": {
            "structured_event_record": {
                "entities": ["Yin Tiger"],
                "events": ["boss fight"],
                "state_changes": ["gourd upgrade"],
                "relations": [],
                "occurrence_refs": [],
                "summary": "The player starts a boss fight with Yin Tiger.",
            },
            "slot_operations": [],
        },
        "attempts": (
            [
                {
                    "status": "validation_failed",
                    "failure_code": "slot_validation_error",
                    "repair_contract": {"details": {}},
                }
            ]
            if abstain
            else []
        ),
        "lifecycle_events": (
            [{"slot": "occurrence_counter", "operation": "write"}]
            if arm == "e1c2" and not abstain
            else []
        ),
        "capsule": {"slots": []},
    }


def test_forensics_separates_raw_and_committed_ser(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    segment_id = "seg-1"
    _write(run_root / "run_manifest.json", {"source_commit": "old"})
    _write(
        run_root / "run_summary.json",
        {"complete": True, "successes": 3},
    )
    _write(run_root / "segments" / segment_id / "e1c0.json", _row("e1c0"))
    _write(run_root / "segments" / segment_id / "e1c1.json", _row("e1c1"))
    _write(
        run_root / "segments" / segment_id / "e1c2.json",
        _row("e1c2", abstain=True),
    )
    construction = {
        "segments": [
            {
                "segment_id": segment_id,
                "virtual_start_sec": 0,
                "virtual_end_sec": 120,
            }
        ]
    }
    analysis = {
        "cases": {
            "case-1": {
                "anchor_intervals": [[10, 20]],
                "entity_terms": ["Yin Tiger"],
                "event_terms": ["boss fight"],
                "state_terms": ["gourd upgrade"],
                "occurrence_terms": [],
                "ordinal_terms": [],
            }
        }
    }

    report = DIAGNOSE.build_report(
        run_root=run_root,
        construction_protocol=construction,
        analysis_protocol=analysis,
        source_commit="new",
    )

    raw = report["strict_lexical_exploratory_coverage"]["raw_ser"]["per_arm"]
    committed = report["strict_lexical_exploratory_coverage"]["committed_memory"]["per_arm"]
    assert raw["e1c2"]["anchor"]["count"] == 1
    assert committed["e1c2"]["anchor"]["count"] == 0
    assert report["failure_fingerprints"]["missing_structured_details"] == 1
    assert report["counts"]["e1c2_abstain_rate"] == 1.0
    assert report["model_calls"] == 0


def test_forensics_reports_canonical_evidence_without_surface_prose(tmp_path: Path) -> None:
    evidence_path = tmp_path / "evidence.jsonl"
    evidence_path.write_text(
        json.dumps(
            {
                "surface": "Chapter Five",
                "normalized_surface": "chapter five",
                "surfaces": [],
                "start_sec": 100,
                "end_sec": 101,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = DIAGNOSE._canonical_evidence_summary(
        evidence_path,
        terms=("Chapter Five", "Flaming Mountains"),
        before_sec=200,
    )

    assert result["terms"][0]["hits_before_boundary"] == 1
    assert result["terms"][1]["total_hits"] == 0
    assert "surface" not in json.dumps(result)
