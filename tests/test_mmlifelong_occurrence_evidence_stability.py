from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "analyze_mmlifelong_occurrence_evidence_stability.py"
)
SPEC = importlib.util.spec_from_file_location("evidence_stability", MODULE_PATH)
assert SPEC and SPEC.loader
STABILITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STABILITY)


def _case(*, supported: set[tuple[str, str, str]], winner: str, gate: str) -> dict:
    return {
        "supported_rows": supported,
        "strict_supported_rows": {
            (*row[:2], "description", row[2], ("p1",)) for row in supported
        },
        "candidate_passage_rows": {
            (row[0], row[2], "p1") for row in supported
        },
        "support_counts": {"occ_1": int(winner == "occ_1"), "occ_2": 0},
        "winner": winner,
        "gate": gate,
        "evidence_declaration_valid": True,
        "mechanical_gate_valid": True,
        "full_structural_valid": True,
        "working_method_valid": True,
        "answer_present": True,
        "terminal_occurrence_failure_count": 0,
        "evidence_event_count": 1,
        "gate_event_count": 1,
    }


def test_stability_requires_both_repeats_to_pass_preregistered_guardrails() -> None:
    stable = _case(
        supported={("set_1", "identity", "occ_1")},
        winner="occ_1",
        gate="sufficient",
    )
    runs = {
        "repeat_1": {"c1": stable, "c2": stable},
        "repeat_2": {"c1": stable, "c2": stable},
    }
    passing = {
        label: {
            "false_commit_rate": 0.25,
            "commit_recall": 0.70,
            "osa_given_commit": 1.0,
        }
        for label in runs
    }

    report = STABILITY.build_stability_report(
        runs,
        repeat_labels=("repeat_1", "repeat_2"),
        expected_cases=2,
        performance=passing,
        baseline_supported_row_agreement=0.80,
    )

    assert report["stability_passed"] is True
    assert report["performance_guardrails_passed"] is True
    assert report["working_method_passed"] is True
    assert report["metrics"]["gate_agreement"] == 1.0

    failing = {**passing, "repeat_2": {**passing["repeat_2"], "false_commit_rate": 0.31}}
    failed = STABILITY.build_stability_report(
        runs,
        repeat_labels=("repeat_1", "repeat_2"),
        expected_cases=2,
        performance=failing,
        baseline_supported_row_agreement=0.80,
    )
    assert failed["performance_guardrails_passed"] is False
    assert failed["working_method_passed"] is False


def test_stability_reports_gate_and_winner_drift() -> None:
    selected = _case(
        supported={("set_1", "identity", "occ_1")},
        winner="occ_1",
        gate="sufficient",
    )
    abstained = _case(supported=set(), winner="", gate="insufficient")

    report = STABILITY.build_stability_report(
        {"repeat_1": {"c1": abstained}, "repeat_2": {"c1": selected}},
        repeat_labels=("repeat_1", "repeat_2"),
        expected_cases=1,
    )

    assert report["metrics"]["gate_agreement"] == 0.0
    assert report["metrics"]["winner_agreement"] == 0.0
    assert report["metrics"]["no_match_to_selected_case_count"] == 1
    assert report["gate_drift_case_ids"] == ["c1"]
    assert report["working_method_passed"] is False


def test_stability_excludes_missing_events_from_mechanism_denominators() -> None:
    stable = _case(
        supported={("set_1", "identity", "occ_1")},
        winner="occ_1",
        gate="sufficient",
    )
    missing = {
        **stable,
        "supported_rows": set(),
        "strict_supported_rows": set(),
        "candidate_passage_rows": set(),
        "support_counts": {},
        "winner": "",
        "gate": "",
        "evidence_declaration_valid": False,
        "mechanical_gate_valid": False,
        "full_structural_valid": False,
        "working_method_valid": False,
        "answer_present": False,
        "evidence_event_count": 0,
        "gate_event_count": 0,
    }
    passing = {
        label: {
            "false_commit_rate": 0.25,
            "commit_recall": 0.70,
            "osa_given_commit": 1.0,
        }
        for label in ("repeat_1", "repeat_2")
    }

    report = STABILITY.build_stability_report(
        {
            "repeat_1": {"valid": stable, "missing": stable},
            "repeat_2": {"valid": stable, "missing": missing},
        },
        repeat_labels=("repeat_1", "repeat_2"),
        expected_cases=2,
        performance=passing,
    )

    assert report["aligned_case_count"] == 2
    assert report["validity"]["evidence_valid_pair_count"] == 1
    assert report["validity"]["working_method_valid_pair_count"] == 1
    assert report["validity"]["missing_evidence_event_count"] == {
        "repeat_1": 0,
        "repeat_2": 1,
    }
    assert report["metrics"]["supported_row_jaccard_macro"] == 1.0
    assert report["per_case"]["missing"]["supported_row_jaccard"] is None
    assert report["structural_reliability_passed"] is False
    assert report["working_method_passed"] is False


def test_scope_size_diagnostic_stratifies_gate_errors_without_changing_r5() -> None:
    runs = {
        "repeat_1": {
            "absent": {
                "support_counts": {"occ_1": 1},
                "gate": "sufficient",
            },
            "present": {
                "support_counts": {"occ_1": 2, "occ_2": 1},
                "gate": "sufficient",
            },
        }
    }
    performance_cases = {
        "repeat_1": {
            "absent": {"false_commit": True, "false_abstention": None},
            "present": {"false_commit": None, "false_abstention": False},
        }
    }

    diagnostic = STABILITY.build_scope_size_diagnostic(
        runs,
        repeat_labels=("repeat_1",),
        performance_cases=performance_cases,
    )

    size_one = diagnostic["by_run"]["repeat_1"]["by_scope_size"]["1"]
    assert size_one["n"] == 1
    assert size_one["false_commit_rate"] == 1.0
    assert size_one["mean_best_support_count"] == 1.0
    assert size_one["mean_winner_margin"] == 1.0

    size_two = diagnostic["by_run"]["repeat_1"]["by_scope_size"]["2"]
    assert size_two["n"] == 1
    assert size_two["commit_recall"] == 1.0
    assert size_two["mean_total_support_count"] == 3.0
    assert size_two["mean_winner_margin"] == 1.0
    assert diagnostic["aggregation_changed"] is False
