from __future__ import annotations

from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any


def _load_batch_module() -> Any:
    path = Path(__file__).resolve().parents[1] / "tools" / "run_mmlifelong_batch.py"
    spec = importlib.util.spec_from_file_location("mmlifelong_batch", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BATCH = _load_batch_module()


def test_stratified_selection_balances_types_and_spreads_timeline() -> None:
    cases = tuple(
        {
            "case_id": f"{question_type}-{index}",
            "question_type": question_type,
            "selection_coordinate": float(index),
            "case_workspace": f"/{question_type}/{index}",
        }
        for question_type in ("A", "B", "C")
        for index in range(5)
    )

    selected = BATCH.select_stratified_cases(cases, 6)

    assert Counter(case["question_type"] for case in selected) == {"A": 2, "B": 2, "C": 2}
    assert {
        case["case_id"] for case in selected if case["question_type"] == "A"
    } == {"A-1", "A-3"}


def test_stratified_selection_caps_limit_at_available_cases() -> None:
    cases = (
        {
            "case_id": "only-case",
            "question_type": "A",
            "selection_coordinate": 0.0,
            "case_workspace": "/A/0",
        },
    )

    selected = BATCH.select_stratified_cases(cases, 20)

    assert [case["case_id"] for case in selected] == ["only-case"]


def test_recorded_fixture_is_bound_to_the_matching_case(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixtures"
    fixture_path = fixture_root / "cases" / "case-0072.json"
    fixture_path.parent.mkdir(parents=True)
    fixture_path.write_text("{}\n", encoding="utf-8")
    args = SimpleNamespace(
        config="api.yaml",
        reasoner_config="reasoner.yaml",
        investigator_config="investigator.yaml",
        reasoner_section="reasoner_api",
        investigator_section="investigator_api",
        answer_policy="benchmark_best_effort",
        controller_mode="frozen_baseline",
        controller_evidence_visibility="none",
        measurement_control="none",
        evidence_control_mode="shadow",
        evidence_state_mode="llm_authored",
        max_rounds=6,
        max_investigations=12,
        max_tasks_per_round=4,
        control_retry_budget=0,
        caption_index_mode="hybrid",
        caption_query_strategy="joint",
        caption_config_digest="digest",
        embedding_model="embedding",
        embedding_device="cpu",
        embedding_batch_size=64,
        embedding_revision=None,
        recorded_fixture_root=str(fixture_root),
        oracle_arm="o0",
        oracle_intervention_root=None,
        occurrence_method_arm="a1",
        occurrence_replay_root=str(fixture_root),
        occurrence_replay_record_root=None,
        occurrence_replay_prime=False,
        matched_response_record_root=None,
        matched_response_replay_root=None,
    )
    case = {
        "case_id": "case-0072",
        "case_workspace": "/cases/case-0072",
    }

    command = BATCH._case_command(case, args, tmp_path / "out")

    index = command.index("--recorded-decisions")
    assert command[index + 1] == str(fixture_path)
    assert command[command.index("--reasoner-config") + 1] == "reasoner.yaml"
    assert command[command.index("--investigator-config") + 1] == "investigator.yaml"
    assert command[command.index("--occurrence-method-arm") + 1] == "a1"
    assert command[command.index("--occurrence-replay-fixture") + 1] == str(
        fixture_path
    )

    args.recorded_fixture_root = None
    args.occurrence_replay_prime = True
    primed_command = BATCH._case_command(case, args, tmp_path / "primed-out")
    assert "--occurrence-replay-prime" in primed_command

    matched_root = tmp_path / "matched"
    args.matched_response_record_root = str(matched_root)
    matched_command = BATCH._case_command(case, args, tmp_path / "matched-out")
    assert matched_command[matched_command.index("--matched-response-record") + 1] == str(
        matched_root / "cases" / "case-0072"
    )


def test_batch_subprocess_env_includes_repository_and_src() -> None:
    environment = BATCH._subprocess_env()
    entries = environment["PYTHONPATH"].split(BATCH.os.pathsep)
    repository_root = str(Path(BATCH.__file__).resolve().parents[1])

    assert entries[:2] == [repository_root, f"{repository_root}/src"]


def test_occurrence_replay_manifest_binds_case_files(tmp_path: Path) -> None:
    fixture = tmp_path / "cases" / "case-1.json"
    fixture.parent.mkdir(parents=True)
    fixture.write_text(
        json.dumps({"packets": [{"packet": {}}, {"packet": {}}]}),
        encoding="utf-8",
    )

    manifest_path = BATCH._write_occurrence_replay_manifest(
        tmp_path,
        ({"case_id": "case-1"},),
        caption_config_digest="caption",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["case_count"] == 1
    assert manifest["caption_config_digest"] == "caption"
    assert manifest["cases"][0]["case_id"] == "case-1"
    assert manifest["cases"][0]["packet_count"] == 2
    assert len(manifest["cases"][0]["sha256"]) == 64


def test_matched_response_manifest_binds_role_sequences(tmp_path: Path) -> None:
    case_root = tmp_path / "cases" / "case-1"
    for role in ("reasoner", "investigator"):
        role_root = case_root / role
        role_root.mkdir(parents=True)
        if role == "reasoner":
            (role_root / "000001.json").write_text(
                json.dumps({"role": role, "sequence": 1}),
                encoding="utf-8",
            )

    manifest_path = BATCH._write_matched_response_manifest(
        tmp_path,
        ({"case_id": "case-1"},),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "MatchedPreTreatmentResponseManifestV1"
    assert manifest["case_count"] == 1
    assert manifest["cases"][0]["role_counts"] == {
        "investigator": 0,
        "reasoner": 1,
    }
    assert len(manifest["cases"][0]["digest"]) == 64
