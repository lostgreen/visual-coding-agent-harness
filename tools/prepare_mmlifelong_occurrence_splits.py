#!/usr/bin/env python3
"""Freeze Day development/test and Week external-validation manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "MMLifelongOccurrenceSplitV1"


def load_case_rows(case_root: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for case_path in sorted(Path(case_root).glob("*/case.json")):
        payload = _read_json(case_path)
        case_id = str(payload.get("case_id", "")).strip()
        if not case_id:
            raise ValueError(f"missing case_id: {case_path}")
        rows.append(
            {
                "case_id": case_id,
                "question_type": str(payload.get("question_type") or "Unknown"),
                "case_sha256": _file_sha256(case_path),
            }
        )
    if not rows:
        raise FileNotFoundError(f"no prepared cases under {case_root}")
    case_ids = [row["case_id"] for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"duplicate case IDs under {case_root}")
    return tuple(rows)


def build_manifests(
    day_cases: Sequence[Mapping[str, Any]],
    week_cases: Sequence[Mapping[str, Any]],
    day_dev_source: Mapping[str, Any],
    *,
    day_dev_source_sha256: str,
    expected_day_count: int = 200,
    expected_week_count: int = 200,
) -> dict[str, dict[str, Any]]:
    day_by_id = _index_cases(day_cases, subset="day")
    week_by_id = _index_cases(week_cases, subset="week")
    if len(day_by_id) != expected_day_count:
        raise ValueError(
            f"expected {expected_day_count} Day cases, found {len(day_by_id)}"
        )
    if len(week_by_id) != expected_week_count:
        raise ValueError(
            f"expected {expected_week_count} Week cases, found {len(week_by_id)}"
        )

    raw_dev_cases = day_dev_source.get("cases", ())
    if not isinstance(raw_dev_cases, Sequence) or isinstance(
        raw_dev_cases, (str, bytes)
    ):
        raise ValueError("Day development source must contain a cases list")
    dev_ids = [
        str(row.get("case_id", "")).strip()
        for row in raw_dev_cases
        if isinstance(row, Mapping)
    ]
    if not dev_ids or any(not case_id for case_id in dev_ids):
        raise ValueError("Day development source contains an empty case ID")
    if len(dev_ids) != len(set(dev_ids)):
        raise ValueError("Day development source contains duplicate case IDs")
    missing = sorted(set(dev_ids) - set(day_by_id))
    if missing:
        raise ValueError(f"Day development cases are missing: {', '.join(missing)}")

    test_ids = sorted(set(day_by_id) - set(dev_ids))
    dev_rows = [dict(day_by_id[case_id]) for case_id in dev_ids]
    test_rows = [dict(day_by_id[case_id]) for case_id in test_ids]
    week_rows = [dict(week_by_id[case_id]) for case_id in sorted(week_by_id)]
    universe_digest = _digest(sorted(day_by_id))
    common = {
        "schema_version": SCHEMA_VERSION,
        "selection_is_outcome_independent": True,
        "agent_visible_benchmark_annotations": False,
        "day_universe_count": len(day_by_id),
        "day_universe_case_id_digest": universe_digest,
        "day_dev_source_sha256": day_dev_source_sha256,
    }
    manifests = {
        "day_val": {
            **common,
            "benchmark_subset": "day",
            "protocol_role": "development_validation",
            "selection_strategy": "reuse_frozen_stratified_day_subset",
            "historical_diagnostic_exposure": True,
            "eligible_for_final_external_claim": False,
            "selected_count": len(dev_rows),
            "cases": dev_rows,
        },
        "day_test": {
            **common,
            "benchmark_subset": "day",
            "protocol_role": "retrospective_internal_test",
            "selection_strategy": "complement_of_frozen_day_validation_subset",
            "historical_diagnostic_exposure": True,
            "eligible_for_final_external_claim": False,
            "selected_count": len(test_rows),
            "cases": test_rows,
        },
        "week_external": {
            "schema_version": SCHEMA_VERSION,
            "benchmark_subset": "week",
            "protocol_role": "external_validation_after_method_freeze",
            "selection_strategy": "all_prepared_week_test_cases",
            "selection_is_outcome_independent": True,
            "agent_visible_benchmark_annotations": False,
            "method_selection_allowed": False,
            "eligible_for_final_external_claim": True,
            "selected_count": len(week_rows),
            "week_universe_count": len(week_by_id),
            "week_universe_case_id_digest": _digest(sorted(week_by_id)),
            "cases": week_rows,
        },
    }
    _validate_partition(manifests, expected_day_count=expected_day_count)
    return manifests


def build_protocol(manifests: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "MMLifelongOccurrenceProtocolV1",
        "day_split": {
            "validation_count": manifests["day_val"]["selected_count"],
            "test_count": manifests["day_test"]["selected_count"],
            "validation_fraction": round(
                float(manifests["day_val"]["selected_count"])
                / float(manifests["day_val"]["day_universe_count"]),
                6,
            ),
            "test_fraction": round(
                float(manifests["day_test"]["selected_count"])
                / float(manifests["day_test"]["day_universe_count"]),
                6,
            ),
            "partition_is_disjoint_and_complete": True,
            "historical_diagnostic_exposure": True,
        },
        "week_external": {
            "case_count": manifests["week_external"]["selected_count"],
            "run_only_after_method_freeze": True,
            "method_changes_after_observing_results": False,
        },
        "claim_policy": (
            "Use Day validation for method selection, Day test as retrospective "
            "internal evidence, and Week as the external validation result."
        ),
    }


def _index_cases(
    cases: Sequence[Mapping[str, Any]], *, subset: str
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for raw in cases:
        row = dict(raw)
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            raise ValueError(f"{subset} contains an empty case ID")
        if case_id in rows:
            raise ValueError(f"{subset} contains duplicate case ID {case_id}")
        rows[case_id] = row
    return rows


def _validate_partition(
    manifests: Mapping[str, Mapping[str, Any]], *, expected_day_count: int
) -> None:
    val_ids = {str(row["case_id"]) for row in manifests["day_val"]["cases"]}
    test_ids = {str(row["case_id"]) for row in manifests["day_test"]["cases"]}
    if val_ids & test_ids:
        raise ValueError("Day validation and test manifests overlap")
    if len(val_ids | test_ids) != expected_day_count:
        raise ValueError("Day validation and test manifests do not cover the universe")


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day-case-root", required=True)
    parser.add_argument("--week-case-root", required=True)
    parser.add_argument("--day-dev-source", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--expected-day-count", type=int, default=200)
    parser.add_argument("--expected-week-count", type=int, default=200)
    args = parser.parse_args()

    source_path = Path(args.day_dev_source)
    manifests = build_manifests(
        load_case_rows(Path(args.day_case_root)),
        load_case_rows(Path(args.week_case_root)),
        _read_json(source_path),
        day_dev_source_sha256=_file_sha256(source_path),
        expected_day_count=args.expected_day_count,
        expected_week_count=args.expected_week_count,
    )
    out_root = Path(args.out_root)
    paths = {
        "day_val": out_root / "day_val60.json",
        "day_test": out_root / "day_test140.json",
        "week_external": out_root / "week_external200.json",
        "protocol": out_root / "protocol.json",
    }
    for name in ("day_val", "day_test", "week_external"):
        _write_json(paths[name], manifests[name])
    protocol = build_protocol(manifests)
    protocol["manifest_sha256"] = {
        name: _file_sha256(paths[name])
        for name in ("day_val", "day_test", "week_external")
    }
    _write_json(paths["protocol"], protocol)
    print(
        json.dumps(
            {
                "day_val": manifests["day_val"]["selected_count"],
                "day_test": manifests["day_test"]["selected_count"],
                "week_external": manifests["week_external"]["selected_count"],
                "partition_valid": True,
                "out_root": str(out_root),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
