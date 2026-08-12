from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any


def _load_module() -> Any:
    path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "prepare_mmlifelong_cross_model_subset.py"
    )
    spec = importlib.util.spec_from_file_location("mmlifelong_cross_subset", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SUBSET = _load_module()


def _cases() -> tuple[dict[str, Any], ...]:
    rows = []
    for question_type in ("Event", "State"):
        for clue_bucket, clue_count in (("single", 1), ("multi", 2)):
            for temporal_index, center in enumerate((0.1, 0.5, 0.9)):
                for replica in range(3):
                    case_id = (
                        f"{question_type}-{clue_bucket}-{temporal_index}-{replica}"
                    )
                    rows.append(
                        {
                            "case_id": case_id,
                            "question_type": question_type,
                            "clue_count": clue_count,
                            "clue_count_bucket": clue_bucket,
                            "normalized_clue_center": center,
                            "clue_duration_sec": 5.0,
                            "source_digest": f"digest-{case_id}",
                        }
                    )
    return tuple(rows)


def test_subset_is_deterministic_and_covers_all_nonempty_strata() -> None:
    first = SUBSET.build_manifest(_cases(), limit=24, seed=17)
    second = SUBSET.build_manifest(tuple(reversed(_cases())), limit=24, seed=17)

    assert first == second
    assert first["selected_count"] == 24
    assert len(first["stratum_selected_counts"]) == 12
    assert all(count >= 1 for count in first["stratum_selected_counts"].values())
    assert first["selection_is_outcome_independent"] is True
    assert not any(
        forbidden in row
        for row in first["cases"]
        for forbidden in SUBSET.FORBIDDEN_SELECTION_FIELDS
    )


def test_subset_seed_changes_members_without_changing_stratum_counts() -> None:
    first = SUBSET.build_manifest(_cases(), limit=24, seed=17)
    second = SUBSET.build_manifest(_cases(), limit=24, seed=18)

    assert {row["case_id"] for row in first["cases"]} != {
        row["case_id"] for row in second["cases"]
    }
    assert first["stratum_selected_counts"] == second["stratum_selected_counts"]
