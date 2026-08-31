from __future__ import annotations

from argparse import Namespace
import importlib.util
import json
from pathlib import Path

from vcah.occurrence_negative_sidecar import file_sha256


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREPARE = _load(
    "prepare_wp17_slot_endpoint_analysis",
    ROOT / "tools" / "prepare_mmlifelong_wp17_slot_endpoint_analysis.py",
)
ANALYZE = _load(
    "analyze_wp17_slot_endpoint_analysis",
    ROOT / "tools" / "analyze_mmlifelong_wp17_slot_construction.py",
)


def test_protocol_marks_0115_posthoc_and_preserves_frozen10() -> None:
    fields = {
        "cases": {
            f"case-{index}": {
                field: {"document_terms": [f"{field}-{index}"]}
                for field in ("entity", "event", "state")
            }
            for index in range(10)
        }
    }
    timeline = {
        "cases": {
            **{
                f"case-{index}": {"anchor_intervals": [[index, index + 1]]}
                for index in range(10)
            },
            "mmlifelong-game-test-0115": {"anchor_intervals": [[100, 101]]},
        }
    }
    posthoc = {
        "cases": {
            "mmlifelong-game-test-0115": {
                "entity_terms": ["entity"],
                "event_terms": ["event"],
                "state_terms": ["state"],
                "occurrence_terms": ["again"],
                "ordinal_terms": ["second"],
            }
        }
    }

    protocol = PREPARE.build_protocol(
        fields,
        timeline,
        posthoc,
        field_spec_sha256="f" * 64,
        timeline_spec_sha256="t" * 64,
        posthoc_spec_sha256="p" * 64,
        construction_protocol_sha256="c" * 64,
        source_commit="commit",
    )

    assert protocol["case_count"] == 11
    assert protocol["cases"]["case-0"]["annotation_timing"] == (
        "frozen_before_construction_outcomes"
    )
    assert protocol["cases"]["mmlifelong-game-test-0115"][
        "annotation_timing"
    ].startswith("post_hoc")
    assert protocol["unbiased_publication_claim_allowed"] is False


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _result(segment_id: str, arm: str, summary: str, *, abstain: bool = False) -> dict:
    return {
        "segment_id": segment_id,
        "arm": arm,
        "status": "success",
        "ser_endpoint_eligible": not abstain,
        "slot_transaction_abstained": abstain,
        "history_token_count": 0 if arm == "e1c0" else 20,
        "validation_retry_count": 0,
        "attempts": [],
        "model_output": {
            "observations": [],
            "slot_operations": [],
            "structured_event_record": {
                "entities": [],
                "events": [],
                "state_changes": [],
                "relations": [],
                "occurrence_refs": [],
                "summary": summary,
            },
        },
        "lifecycle_events": [],
    }


def test_analysis_scores_abstain_as_zero_and_reports_paired_effect(tmp_path: Path) -> None:
    segment_id = "segment-1"
    construction_protocol = {
        "segments": [
            {
                "segment_id": segment_id,
                "virtual_start_sec": 0,
                "virtual_end_sec": 120,
            }
        ]
    }
    construction_path = tmp_path / "construction.json"
    _write(construction_path, construction_protocol)
    construction_sha = file_sha256(construction_path)
    analysis_protocol = {
        "contract": ANALYZE.ANALYSIS_PROTOCOL_CONTRACT,
        "construction_protocol_sha256": construction_sha,
        "effects": {"bootstrap_samples": 100, "bootstrap_seed": 7},
        "cases": {
            "case-1": {
                "anchor_intervals": [[10, 20]],
                "entity_terms": ["Yin Tiger"],
                "event_terms": ["boss fight"],
                "state_terms": ["gourd upgrade"],
                "occurrence_terms": [],
                "ordinal_terms": [],
                "annotation_timing": "frozen_before_construction_outcomes",
            }
        },
    }
    analysis_path = tmp_path / "analysis.json"
    _write(analysis_path, analysis_protocol)
    audit_path = tmp_path / "audit.json"
    _write(
        audit_path,
        {
            "decision": "PASSED",
            "structural_gate_passed": True,
            "endpoint_values_evaluated": False,
        },
    )
    run_root = tmp_path / "run"
    _write(
        run_root / "run_manifest.json",
        {
            "expected_result_count": 3,
            "protocol_manifest_sha256": construction_sha,
        },
    )
    _write(run_root / "run_summary.json", {"complete": True, "successes": 3})
    _write(
        run_root / "segments" / segment_id / "e1c0.json",
        _result(segment_id, "e1c0", "A generic scene."),
    )
    _write(
        run_root / "segments" / segment_id / "e1c1.json",
        _result(segment_id, "e1c1", "Yin Tiger boss fight."),
    )
    _write(
        run_root / "segments" / segment_id / "e1c2.json",
        _result(
            segment_id,
            "e1c2",
            "Yin Tiger boss fight and gourd upgrade.",
            abstain=True,
        ),
    )

    report_path = ANALYZE.run(
        Namespace(
            analysis_protocol=str(analysis_path),
            expected_analysis_protocol_sha256=file_sha256(analysis_path),
            construction_protocol=str(construction_path),
            expected_construction_protocol_sha256=construction_sha,
            run_root=str(run_root),
            structural_audit=str(audit_path),
            out_root=str(tmp_path / "report"),
            bootstrap_samples=100,
            seed=7,
        )
    )
    report = json.loads(report_path.read_text())

    assert report["arm_metrics"]["e1c1"]["anchor_representation_coverage"][
        "rate"
    ] == 1.0
    assert report["arm_metrics"]["e1c2"]["anchor_representation_coverage"][
        "rate"
    ] == 0.0
    assert report["paired_effects"]["e1c2_minus_e1c1"][
        "anchor_representation_coverage"
    ]["paired_delta"] == -1.0
