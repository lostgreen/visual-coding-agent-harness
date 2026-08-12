from __future__ import annotations

from collections import Counter
import importlib.util
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
    )
    case = {
        "case_id": "case-0072",
        "case_workspace": "/cases/case-0072",
    }

    command = BATCH._case_command(case, args, tmp_path / "out")

    index = command.index("--recorded-decisions")
    assert command[index + 1] == str(fixture_path)


def test_batch_subprocess_env_includes_repository_and_src() -> None:
    environment = BATCH._subprocess_env()
    entries = environment["PYTHONPATH"].split(BATCH.os.pathsep)
    repository_root = str(Path(BATCH.__file__).resolve().parents[1])

    assert entries[:2] == [repository_root, f"{repository_root}/src"]
