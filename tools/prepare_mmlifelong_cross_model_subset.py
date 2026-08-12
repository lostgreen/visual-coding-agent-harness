#!/usr/bin/env python3
"""Prepare a score-independent stratified subset for cross-model transfer."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


FORBIDDEN_SELECTION_FIELDS = (
    "answer",
    "gold",
    "reference_answer",
    "score",
    "correct",
    "oracle_gain",
)


def discover_cases(case_root: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for case_path in sorted(Path(case_root).glob("*/case.json")):
        run_dir = case_path.parent
        evaluation_path = run_dir / "evaluation_case.json"
        if not evaluation_path.is_file():
            raise FileNotFoundError(evaluation_path)
        case = _read_json(case_path)
        evaluation = _read_json(evaluation_path)
        metadata = evaluation.get("evaluation_metadata", {})
        metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        intervals = tuple(
            (float(value[0]), float(value[1]))
            for value in tuple(evaluation.get("clue_intervals", ()) or ())
            if isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and len(value) == 2
        )
        if not intervals:
            raise ValueError(f"missing clue intervals: {evaluation_path}")
        timeline_digest = None
        duration = float(metadata.get("total_seconds", 0.0) or 0.0)
        if duration <= 0.0:
            asset_ref = Path(str(case.get("asset_ref", "")))
            timeline_path = asset_ref / "virtual_timeline.json"
            if not timeline_path.is_file():
                raise FileNotFoundError(timeline_path)
            timeline = _read_json(timeline_path)
            duration = float(timeline.get("duration_sec", 0.0) or 0.0)
            timeline_digest = _file_sha256(timeline_path)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError(f"invalid total_seconds: {evaluation_path}")
        center = sum((start + end) / 2.0 for start, end in intervals) / len(intervals)
        normalized_center = min(1.0, max(0.0, center / duration))
        rows.append(
            {
                "case_id": str(case["case_id"]),
                "question_type": str(
                    case.get("question_type")
                    or metadata.get("question_type")
                    or "Unknown"
                ),
                "clue_count": len(intervals),
                "clue_count_bucket": "single" if len(intervals) == 1 else "multi",
                "normalized_clue_center": normalized_center,
                "clue_duration_sec": sum(max(0.0, end - start) for start, end in intervals),
                "source_digest": _digest(
                    {
                        "case_id": str(case["case_id"]),
                        "question_type": str(
                            case.get("question_type")
                            or metadata.get("question_type")
                            or "Unknown"
                        ),
                        "clue_intervals": [list(value) for value in intervals],
                        "total_seconds": duration,
                        "timeline_digest": timeline_digest,
                    }
                ),
            }
        )
    if not rows:
        raise FileNotFoundError(f"no prepared cases under {case_root}")
    return tuple(rows)


def select_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    seed: int,
) -> tuple[tuple[dict[str, Any], ...], dict[str, float]]:
    requested = min(int(limit), len(cases))
    if requested <= 0:
        raise ValueError("limit must be positive")
    sorted_centers = sorted(float(case["normalized_clue_center"]) for case in cases)
    thresholds = {
        "q1": _quantile(sorted_centers, 1.0 / 3.0),
        "q2": _quantile(sorted_centers, 2.0 / 3.0),
    }
    enriched = []
    for raw in cases:
        row = dict(raw)
        center = float(row["normalized_clue_center"])
        row["timeline_quantile"] = (
            "early"
            if center <= thresholds["q1"]
            else "middle"
            if center <= thresholds["q2"]
            else "late"
        )
        row["stratum"] = " | ".join(
            (
                str(row["question_type"]),
                str(row["clue_count_bucket"]),
                str(row["timeline_quantile"]),
            )
        )
        enriched.append(row)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        grouped[str(row["stratum"])].append(row)
    if requested < len(grouped):
        raise ValueError(
            f"limit {requested} is smaller than non-empty stratum count {len(grouped)}"
        )
    quotas = {stratum: 1 for stratum in grouped}
    remaining = requested - len(grouped)
    capacities = {stratum: len(rows) - 1 for stratum, rows in grouped.items()}
    total_capacity = sum(capacities.values())
    if remaining and not total_capacity:
        raise ValueError("no capacity for remaining subset quota")
    raw_allocations = {
        stratum: (remaining * capacity / total_capacity if total_capacity else 0.0)
        for stratum, capacity in capacities.items()
    }
    for stratum, value in raw_allocations.items():
        quotas[stratum] += min(capacities[stratum], int(math.floor(value)))
    unallocated = requested - sum(quotas.values())
    order = sorted(
        grouped,
        key=lambda stratum: (
            -(raw_allocations[stratum] - math.floor(raw_allocations[stratum])),
            _seeded_hash(seed, stratum),
            stratum,
        ),
    )
    while unallocated:
        changed = False
        for stratum in order:
            if quotas[stratum] >= len(grouped[stratum]):
                continue
            quotas[stratum] += 1
            unallocated -= 1
            changed = True
            if not unallocated:
                break
        if not changed:
            raise ValueError("could not allocate requested subset size")

    selected: list[dict[str, Any]] = []
    for stratum in sorted(grouped):
        ranked = sorted(
            grouped[stratum],
            key=lambda row: (_seeded_hash(seed, str(row["case_id"])), row["case_id"]),
        )
        selected.extend(ranked[: quotas[stratum]])
    return (
        tuple(sorted(selected, key=lambda row: str(row["case_id"]))),
        thresholds,
    )


def build_manifest(
    cases: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    seed: int,
) -> dict[str, Any]:
    selected, thresholds = select_cases(cases, limit=limit, seed=seed)
    return {
        "schema_version": "MMLifelongCrossModelSubsetV1",
        "selection_strategy": (
            "question_type_x_clue_count_x_normalized_clue_center_tertiles"
        ),
        "selection_is_outcome_independent": True,
        "forbidden_selection_fields": list(FORBIDDEN_SELECTION_FIELDS),
        "seed": int(seed),
        "candidate_count": len(cases),
        "selected_count": len(selected),
        "timeline_quantile_thresholds": thresholds,
        "candidate_source_digest": _digest(
            sorted(str(case["source_digest"]) for case in cases)
        ),
        "stratum_candidate_counts": dict(
            sorted(Counter(str(case["stratum"]) for case in _enrich(cases, thresholds)).items())
        ),
        "stratum_selected_counts": dict(
            sorted(Counter(str(case["stratum"]) for case in selected).items())
        ),
        "cases": [
            {
                "case_id": row["case_id"],
                "question_type": row["question_type"],
                "clue_count": row["clue_count"],
                "clue_count_bucket": row["clue_count_bucket"],
                "timeline_quantile": row["timeline_quantile"],
                "normalized_clue_center": round(
                    float(row["normalized_clue_center"]), 6
                ),
                "clue_duration_sec": round(float(row["clue_duration_sec"]), 3),
                "stratum": row["stratum"],
                "source_digest": row["source_digest"],
            }
            for row in selected
        ],
    }


def _enrich(
    cases: Sequence[Mapping[str, Any]], thresholds: Mapping[str, float]
) -> tuple[dict[str, Any], ...]:
    rows = []
    for value in cases:
        row = dict(value)
        center = float(row["normalized_clue_center"])
        row["timeline_quantile"] = (
            "early"
            if center <= thresholds["q1"]
            else "middle"
            if center <= thresholds["q2"]
            else "late"
        )
        row["stratum"] = " | ".join(
            (
                str(row["question_type"]),
                str(row["clue_count_bucket"]),
                str(row["timeline_quantile"]),
            )
        )
        rows.append(row)
    return tuple(rows)


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires values")
    position = probability * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(values[lower])
    fraction = position - lower
    return float(values[lower]) * (1.0 - fraction) + float(values[upper]) * fraction


def _seeded_hash(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


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


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", required=True)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = build_manifest(
        discover_cases(Path(args.case_root)),
        limit=args.limit,
        seed=args.seed,
    )
    _write(Path(args.out), json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "selected_count": manifest["selected_count"],
                "stratum_count": len(manifest["stratum_selected_counts"]),
                "candidate_source_digest": manifest["candidate_source_digest"],
                "out": args.out,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
