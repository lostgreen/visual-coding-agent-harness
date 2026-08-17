from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vcah.occurrence_negative_sidecar import (
    NegativeSidecarOracleLeakError,
    NegativeSidecarScopeMismatchError,
    assert_negative_sidecar_payload,
    load_negative_sidecar_snapshot,
    negative_sidecar_prompt,
    parse_negative_sidecar_response,
    positive_source_manifest_digest,
    replay_source_manifest_digest,
    scan_persisted_json_surface,
    validate_negative_sidecar_output,
    validate_negative_sidecar_output_detailed,
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

RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "run_mmlifelong_occurrence_negative_sidecar.py"
)
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "negative_sidecar_runner", RUNNER_PATH
)
assert RUNNER_SPEC and RUNNER_SPEC.loader
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)

ROW_AUDIT_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "audit_mmlifelong_occurrence_negative_rows.py"
)
ROW_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "negative_sidecar_row_audit", ROW_AUDIT_PATH
)
assert ROW_AUDIT_SPEC and ROW_AUDIT_SPEC.loader
ROW_AUDIT = importlib.util.module_from_spec(ROW_AUDIT_SPEC)
ROW_AUDIT_SPEC.loader.exec_module(ROW_AUDIT)

ROW_JUDGE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "evaluate_mmlifelong_occurrence_negative_rows.py"
)
ROW_JUDGE_SPEC = importlib.util.spec_from_file_location(
    "negative_sidecar_row_judge", ROW_JUDGE_PATH
)
assert ROW_JUDGE_SPEC and ROW_JUDGE_SPEC.loader
ROW_JUDGE = importlib.util.module_from_spec(ROW_JUDGE_SPEC)
ROW_JUDGE_SPEC.loader.exec_module(ROW_JUDGE)


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
                    "type": "occurrence_sufficiency_gate_decision",
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
    result = validate_negative_sidecar_output_detailed(invalid, snapshot=snapshot)
    assert result.status == "validation_failed"
    assert not result.rows
    assert result.unknown_field_count == 1
    assert "negative_sidecar_unknown_top_level_fields_dropped" in result.warning_codes
    assert result.dropped_rows[0]["error_codes"] == [
        "negative_sidecar_passage_not_visible"
    ]


def test_negative_output_keeps_valid_rows_and_strips_unknown_fields(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    result = validate_negative_sidecar_output_detailed(
        {
            "contradictions": [
                {
                    "constraint_id": "c-1",
                    "occurrence_id": "occ-1",
                    "evidence_passage_ids": ["p-1"],
                    "reason": "extra prose",
                },
                {
                    "constraint_id": "c-1",
                    "occurrence_id": "outside",
                    "evidence_passage_ids": ["p-x"],
                },
            ],
            "comment": "ignored",
        },
        snapshot=snapshot,
    )

    assert result.status == "partial_valid"
    assert len(result.rows) == 1
    assert result.rows[0]["occurrence_id"] == "occ-1"
    assert result.unknown_field_count == 2
    assert result.dropped_rows[0]["row_index"] == 1
    assert "negative_sidecar_candidate_invalid" in result.error_codes


def test_payload_blacklist_reports_recursive_path() -> None:
    with pytest.raises(NegativeSidecarOracleLeakError, match=r"\$\.nested\.verdict"):
        assert_negative_sidecar_payload({"nested": {"verdict": "selected"}})


def test_snapshot_requires_decision_and_records_packet_fallback(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    assert snapshot.packet_match_mode == "exact"
    runtime_path = tmp_path / "run" / "cases" / "case-1" / "runtime_summary.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["trace"] = [
        row
        for row in runtime["trace"]
        if row["type"]
        not in {
            "occurrence_sufficiency_gate_decision",
            "occurrence_sufficiency_decision",
        }
    ]
    _write_json(runtime_path, runtime)
    with pytest.raises(
        NegativeSidecarScopeMismatchError, match="no frozen sufficiency"
    ):
        load_negative_sidecar_snapshot(
            tmp_path / "run" / "cases" / "case-1",
            replay_fixture_path=tmp_path / "fixtures" / "cases" / "case-1.json",
        )

    runtime["trace"].append(
        {"type": "occurrence_sufficiency_gate_decision", "set_id": "set-1"}
    )
    _write_json(runtime_path, runtime)
    fixture_path = tmp_path / "fixtures" / "cases" / "case-1.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    fixture["packets"][0]["packet"]["occurrence_set"]["attempt_id"] = "other-set"
    _write_json(fixture_path, fixture)
    fallback = load_negative_sidecar_snapshot(
        tmp_path / "run" / "cases" / "case-1",
        replay_fixture_path=fixture_path,
    )
    assert fallback.packet_match_mode == "fallback"
    assert fallback.packet_attempt_id == "other-set"


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
    assert report["checks"]["all_packet_matches_exact"] is True


def test_independent_audit_recomputes_sources_and_detects_mutation(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    positive_root = tmp_path / "run"
    replay_root = tmp_path / "fixtures"
    positive_digest = positive_source_manifest_digest(positive_root, ["case-1"])
    replay_digest = replay_source_manifest_digest(replay_root, ["case-1"])
    roots = {"r1": tmp_path / "r1", "r2": tmp_path / "r2"}
    for label, root in roots.items():
        _write_json(
            root / "run_manifest.json",
            {
                "positive_root_unmodified": True,
                "positive_source_manifest_before": positive_digest,
                "positive_source_manifest_after": positive_digest,
                "replay_source_manifest_digest": replay_digest,
                "temperature": 0,
                "top_p": 1,
                "requested_seed": 7,
                "provider_seed_supported": True,
                "provider_reported_seed_support": "supported",
                "response_format": {"type": "json_object"},
            },
        )
        _write_json(
            root / "cases" / "case-1" / "sidecar_result.json",
            {
                "schema_version": "MMLifelongOccurrenceNegativeSidecarCaseV1",
                "case_id": "case-1",
                "repeat_label": label,
                "status": "success",
                "actual_model": "test-model",
                "snapshot_digest": snapshot.legacy_digest,
                "model_response_digest": "response-digest",
                "source_digests": {
                    "case_sha256": snapshot.source_case_sha256,
                    "runtime_sha256": snapshot.source_runtime_sha256,
                    "replay_fixture_sha256": snapshot.replay_fixture_sha256,
                },
                "packet_match_mode": "exact",
                "attempt_count": 1,
                "attempt_history": [{"attempt_index": 1, "status": "success"}],
                "status_history": ["success"],
                "resumed_from_failure": False,
                "contradiction_rows": [],
            },
        )

    audit, candidates = ANALYZER.build_independent_audit(
        roots,
        positive_run_root=positive_root,
        replay_fixture_root=replay_root,
        case_ids=["case-1"],
    )
    assert audit["passed"] is True
    assert candidates["case-1"] == ("occ-1", "occ-2")

    case_path = positive_root / "cases" / "case-1" / "case.json"
    case = json.loads(case_path.read_text(encoding="utf-8"))
    case["question"] = "Mutated after the run"
    _write_json(case_path, case)
    mutated, _ = ANALYZER.build_independent_audit(
        roots,
        positive_run_root=positive_root,
        replay_fixture_root=replay_root,
        case_ids=["case-1"],
    )
    assert mutated["passed"] is False
    assert mutated["checks"]["r1_snapshot_digest_recomputed_matches"] is False
    assert mutated["checks"]["r2_positive_root_unmodified_recorded"] is False


def test_persisted_surface_scan_is_observational(tmp_path: Path) -> None:
    _write_json(tmp_path / "safe.json", {"prompt_digest": "abc"})
    assert scan_persisted_json_surface(tmp_path)["passed"] is True
    _write_json(tmp_path / "unsafe.json", {"raw_response": "hidden"})
    report = scan_persisted_json_surface(tmp_path)
    assert report["passed"] is False
    assert report["violations"][0]["kind"] == "forbidden_key"


class _SequenceClient:
    model = "test-model"

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.last_response_metadata = {
            "finish_reason": "stop",
            "completion_tokens": 10,
            "reasoning_tokens": 0,
        }

    def chat(self, *_: object, **__: object) -> str:
        return self.responses.pop(0)


def test_runner_retries_json_once_and_preserves_resume_history(tmp_path: Path) -> None:
    _snapshot(tmp_path)
    out_root = tmp_path / "out"
    args = SimpleNamespace(
        out_root=str(out_root),
        positive_run_root=str(tmp_path / "run"),
        replay_fixture_root=str(tmp_path / "fixtures"),
        repeat_label="r1",
        max_completion_tokens=4096,
        resume=False,
    )
    first = RUNNER._run_one(
        "case-1", _SequenceClient(["not-json", "still-not-json"]), args
    )
    assert first["status"] == "validation_failed"
    assert first["parse_retry_count"] == 1
    assert first["status_history"] == ["invalid_json", "validation_failed"]

    args.resume = True
    valid = json.dumps({"contradictions": []})
    second = RUNNER._run_one("case-1", _SequenceClient([valid]), args)
    assert second["status"] == "success"
    assert second["resumed_from_failure"] is True
    assert second["attempt_count"] == 3
    assert second["first_attempt_status"] == "invalid_json"


def test_row_quality_audit_is_blinded_and_case_clustered(tmp_path: Path) -> None:
    _snapshot(tmp_path)
    roots = {"r1": tmp_path / "r1", "r2": tmp_path / "r2"}
    contradiction = {
        "constraint_id": "c-1",
        "constraint_type": "event",
        "occurrence_id": "occ-1",
        "evidence_passage_ids": ["p-1"],
    }
    for label, root in roots.items():
        _write_json(
            root / "cases" / "case-1" / "sidecar_result.json",
            {
                "case_id": "case-1",
                "repeat_label": label,
                "contradiction_rows": [contradiction],
            },
        )
    blind, key = ROW_AUDIT.prepare_audit(
        roots,
        positive_run_root=tmp_path / "run",
        replay_fixture_root=tmp_path / "fixtures",
        frozen_rows=[
            {
                "arm": "a4",
                "case_id": "case-1",
                "candidate_recall_resolved_set": False,
                "final_resolution": "selected",
                "selected_occurrence_ids": ["occ-1"],
                "osa_strict": False,
            }
        ],
        reliability_fraction=0.5,
        seed=7,
    )
    serialized = json.dumps(blind, sort_keys=True)
    assert blind["item_count"] == 2
    assert "case-1" not in serialized
    assert "repeat_label" not in serialized
    assert "winner" not in serialized
    assert "gold" not in serialized
    primary = [
        {
            "audit_item_id": key["rows"][0]["audit_item_id"],
            "verdict": "true_contradiction",
        },
        {
            "audit_item_id": key["rows"][1]["audit_item_id"],
            "verdict": "false_contradiction",
        },
    ]
    primary_by_id = {row["audit_item_id"]: row for row in primary}
    reliability_id = key["reliability_sample_item_ids"][0]
    judgments = {
        "judgment_protocol_digest": key["judgment_protocol_digest"],
        "judgments": [
            *primary,
        ],
        "reliability_judgments": [primary_by_id[reliability_id]],
    }
    report = ROW_AUDIT.analyze_judgments(key, judgments, bootstrap_samples=100, seed=7)
    assert report["complete"] is True
    assert report["row_precision"] == 0.5
    assert report["by_constraint_type"]["event"]["true_count"] == 1
    assert report["unique_semantic_claim_count"] == 1
    assert report["duplicate_emitted_row_count"] == 1
    assert report["discordant_duplicate_claim_count"] == 1
    assert report["unique_semantic_claim_precision"]["precision"] is None
    assert report["reliability"]["cohen_kappa"] == 1.0
    assert report["winner_discrimination"]["available"] is True


def test_row_audit_reconnects_validated_claims_to_frozen_winners() -> None:
    winner_cases = []
    key_rows = []
    judgments = []
    for case_id, winner_class in (
        ("false-1", "false_winner"),
        ("false-2", "false_winner"),
        ("positive-1", "candidate_present_winner"),
        ("positive-2", "candidate_present_winner"),
    ):
        winner_cases.append(
            {
                "case_id": case_id,
                "winner_class": winner_class,
                "strict_correct": winner_class == "candidate_present_winner",
                "selected_occurrence_id_digest": f"winner-{case_id}",
            }
        )
        for repeat in ("r1", "r2"):
            item_id = f"{repeat}-{case_id}"
            key_rows.append(
                {
                    "audit_item_id": item_id,
                    "repeat_label": repeat,
                    "case_id": case_id,
                    "constraint_type": "identity",
                    "semantic_claim_digest": item_id,
                    "targets_selected_winner": True,
                }
            )
            judgments.append(
                {
                    "audit_item_id": item_id,
                    "verdict": (
                        "true_contradiction"
                        if winner_class == "false_winner"
                        else "false_contradiction"
                    ),
                }
            )
    report = ROW_AUDIT.analyze_judgments(
        {
            "rows": key_rows,
            "winner_cases": winner_cases,
            "reliability_sample_item_ids": [],
        },
        {"judgments": judgments},
        bootstrap_samples=100,
        seed=7,
    )
    discrimination = report["winner_discrimination"]
    assert report["complete"] is True
    assert discrimination["validated_discrimination_established"] is True
    for repeat in discrimination["per_repeat"].values():
        assert repeat["validated"]["false_hit_count"] == 2
        assert repeat["validated"]["candidate_present_hit_count"] == 0
        assert repeat["validated"]["false_candidate_gap"] == 1.0


def test_blind_row_judge_uses_one_item_without_persisting_prose(tmp_path: Path) -> None:
    item = {
        "audit_item_id": "blind-1",
        "question": "What happens?",
        "options": {"A": "Open", "B": "Close"},
        "constraint": {"constraint_type": "event", "description": "It opens."},
        "candidate_label": "candidate-1",
        "candidate_passages": [
            {
                "passage_id": "p-1",
                "time_range": [0, 1],
                "caption_excerpt": "It remains closed.",
                "cited": True,
            }
        ],
        "audit_question": "Does the cited passage directly contradict it?",
        "allowed_verdicts": sorted(ROW_AUDIT.VALID_VERDICTS),
    }
    task = ROW_JUDGE._task("blind-1", item, kind="primary")
    prompt = ROW_JUDGE._judgment_prompt(
        item,
        protocol=ROW_AUDIT.BLIND_JUDGMENT_PROTOCOL,
        instance_nonce=task["task_id"],
    )
    assert "blind-1" not in prompt
    assert "Absence of support is not contradiction" in prompt

    class _Client:
        model = "judge-model"
        last_response_metadata = {}

        def chat(self, *_args, **_kwargs):
            self.last_response_metadata = {
                "finish_reason": "stop",
                "completion_tokens": 8,
            }
            return '{"verdict":"true_contradiction"}'

    result = ROW_JUDGE._judge_task(
        task,
        _Client(),
        {
            "judgment_protocol": ROW_AUDIT.BLIND_JUDGMENT_PROTOCOL,
            "judgment_protocol_digest": "protocol-digest",
        },
        SimpleNamespace(
            out_root=str(tmp_path / "judge"),
            resume=False,
            judge_max_retries=2,
            max_completion_tokens=4096,
        ),
    )
    persisted = json.loads(
        (tmp_path / "judge" / "tasks" / f"{task['task_id']}.json").read_text()
    )
    assert result["status"] == "success"
    assert persisted["verdict"] == "true_contradiction"
    assert persisted["raw_response_persisted"] is False
    assert "raw_response" not in persisted
    assert "prompt" not in persisted


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
                "model_response_digest": f"response-{case_id}",
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
        frozen_rows=[
            *frozen_rows,
            {
                "arm": "a4",
                "case_id": "unused-frozen-control",
                "candidate_recall_resolved_set": False,
                "final_resolution": "no_match",
                "selected_occurrence_ids": [],
                "osa_strict": False,
            },
        ],
        expected_cases=20,
        independent_audit={"checks": {"synthetic_audit": True}, "passed": True},
        bootstrap_samples=200,
        permutation_samples=200,
        seed=7,
    )

    assert report["structural_gates"]["passed"] is True
    assert report["per_repeat"]["r1"]["false_winner_contradicted_count"] == 5
    assert report["per_repeat"]["r1"]["positive_winner_contradicted_count"] == 0
    assert report["stability"]["winner_flag_exact_agreement"] is True
    assert report["stability"]["winner_flag_cohen_kappa"] == 1.0
    assert report["decision"] == "NEGATIVE_ROW_QUALITY_AUDIT_REQUIRED"

    canary = ANALYZER.build_report(
        repeats,
        frozen_rows=frozen_rows,
        expected_cases=20,
        expected_false_commits=12,
        expected_positive_commits=8,
        structural_only=True,
        independent_audit={"checks": {"synthetic_audit": True}, "passed": True},
        bootstrap_samples=20,
        permutation_samples=20,
        seed=7,
    )
    assert canary["structural_gates"]["passed"] is True
    assert canary["decision"] == "STRUCTURAL_CANARY_ONLY"
