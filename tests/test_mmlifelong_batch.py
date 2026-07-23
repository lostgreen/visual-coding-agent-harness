from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import sys
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
            "target_virtual_interval": [float(index * 100), float(index * 100 + 10)],
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
            "target_virtual_interval": [0.0, 1.0],
            "case_workspace": "/A/0",
        },
    )

    selected = BATCH.select_stratified_cases(cases, 20)

    assert [case["case_id"] for case in selected] == ["only-case"]
