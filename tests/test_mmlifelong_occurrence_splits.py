from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any


def _load_module() -> Any:
    path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "prepare_mmlifelong_occurrence_splits.py"
    )
    spec = importlib.util.spec_from_file_location("mmlifelong_occurrence_splits", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SPLITS = _load_module()


def _cases(prefix: str, count: int) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "case_id": f"{prefix}-{index:04d}",
            "question_type": "Event" if index % 2 else "State",
            "case_sha256": f"digest-{prefix}-{index}",
        }
        for index in range(count)
    )


def test_occurrence_split_is_disjoint_complete_and_external_week() -> None:
    day = _cases("day", 10)
    week = _cases("week", 8)
    dev_source = {"cases": [{"case_id": row["case_id"]} for row in day[:3]]}

    manifests = SPLITS.build_manifests(
        day,
        week,
        dev_source,
        day_dev_source_sha256="source-digest",
        expected_day_count=10,
        expected_week_count=8,
    )

    val_ids = {row["case_id"] for row in manifests["day_val"]["cases"]}
    test_ids = {row["case_id"] for row in manifests["day_test"]["cases"]}
    assert val_ids.isdisjoint(test_ids)
    assert val_ids | test_ids == {row["case_id"] for row in day}
    assert manifests["day_val"]["selected_count"] == 3
    assert manifests["day_test"]["selected_count"] == 7
    assert manifests["day_test"]["historical_diagnostic_exposure"] is True
    assert manifests["week_external"]["selected_count"] == 8
    assert manifests["week_external"]["method_selection_allowed"] is False
    assert manifests["week_external"]["eligible_for_final_external_claim"] is True


def test_occurrence_split_rejects_missing_development_case() -> None:
    try:
        SPLITS.build_manifests(
            _cases("day", 4),
            _cases("week", 4),
            {"cases": [{"case_id": "day-missing"}]},
            day_dev_source_sha256="source-digest",
            expected_day_count=4,
            expected_week_count=4,
        )
    except ValueError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("expected a missing development case to fail")
