#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmarks.mmlifelong.adapter import (
    evaluation_record_from_dataset,
    runtime_question_from_case,
)
from benchmarks.schema import RuntimeQuestion
from vcah.caption_schema import stable_digest
from vcah.replay import file_checksum
from vcah.virtual_video import VirtualVideoWorkspace


def prepare_oracle_day(
    *,
    legacy_case_root: Path,
    dataset_root: Path,
    out_root: Path,
    caption_config_digest: str,
    experiment_seed: int,
) -> Mapping[str, Any]:
    output = Path(out_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"oracle preparation output is not empty: {output}")
    case_paths = tuple(sorted(Path(legacy_case_root).glob("*/case.json")))
    if not case_paths:
        raise FileNotFoundError(f"no legacy MM-Lifelong cases under {legacy_case_root}")
    cases_root = output / "cases"
    interventions_root = output / "interventions" / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)
    interventions_root.mkdir(parents=True, exist_ok=True)

    prepared: list[dict[str, Any]] = []
    asset_roots: set[str] = set()
    source_manifest_digests: set[str] = set()
    question_type_counts: Counter[str] = Counter()
    clue_count = 0
    for source_case_path in case_paths:
        source = VirtualVideoWorkspace.load(source_case_path.parent)
        source_index = source.case.metadata.get("source_index")
        runtime = runtime_question_from_case(
            {
                "case_id": source.case.case_id,
                "question": source.case.question,
                "options": source.case.options,
                "question_type": source.case.question_type,
                "subset": source.case.subset,
                "split": source.case.split,
                "runtime_metadata": source.case.metadata,
            }
        )
        evaluation = evaluation_record_from_dataset(
            Path(dataset_root),
            case_id=source.case.case_id,
            subset=source.case.subset,
            split=source.case.split,
            source_index=source_index,
        )
        if (
            source.case.gold_answer
            and _normalized_text(source.case.gold_answer)
            != _normalized_text(evaluation.reference_answer)
        ):
            raise ValueError(f"reference answer mismatch for {source.case.case_id}")
        normalized_clues = tuple(
            _normalize_clue_interval(
                interval,
                duration_sec=source.manifest.duration_sec,
            )
            for interval in evaluation.clue_intervals
        )
        if not normalized_clues:
            raise ValueError(f"case has no clue intervals: {source.case.case_id}")
        if source.case.gold_clue_intervals and len(source.case.gold_clue_intervals) != len(
            normalized_clues
        ):
            raise ValueError(f"legacy/evaluator clue count mismatch for {source.case.case_id}")

        case_output = cases_root / source.case.case_id
        case_output.mkdir()
        runtime_payload = runtime.to_dict()
        runtime_payload["asset_ref"] = str(source.asset_root.resolve())
        RuntimeQuestion.from_mapping(runtime_payload)
        _write_json(case_output / "case.json", runtime_payload)
        _write_json(case_output / "evaluation_case.json", evaluation.to_dict())

        manifest_path = source.asset_root / "virtual_timeline.json"
        source_manifest_digest = str(file_checksum(manifest_path)["sha256"])
        intervention = {
            "schema_version": "MMLifelongOracleInterventionV1",
            "case_id": source.case.case_id,
            "normalized_clue_intervals": [list(item) for item in normalized_clues],
            "clue_interval_digest": stable_digest(
                [list(item) for item in normalized_clues]
            ),
            "experiment_seed": int(experiment_seed),
            "caption_config_digest": str(caption_config_digest),
            "source_manifest_digest": source_manifest_digest,
        }
        _write_json(
            interventions_root / f"{source.case.case_id}.json",
            intervention,
        )
        asset_roots.add(str(source.asset_root.resolve()))
        source_manifest_digests.add(source_manifest_digest)
        question_type = str(source.case.question_type or "Unknown")
        question_type_counts[question_type] += 1
        clue_count += len(normalized_clues)
        prepared.append(
            {
                "case_id": source.case.case_id,
                "question_type": question_type,
                "clue_count": len(normalized_clues),
                "runtime_digest": stable_digest(runtime_payload),
                "intervention_digest": stable_digest(intervention),
            }
        )

    summary = {
        "schema_version": "MMLifelongOraclePreparedDayV1",
        "case_count": len(prepared),
        "clue_count": clue_count,
        "question_type_counts": dict(sorted(question_type_counts.items())),
        "caption_config_digest": str(caption_config_digest),
        "experiment_seed": int(experiment_seed),
        "asset_roots": sorted(asset_roots),
        "source_manifest_digests": sorted(source_manifest_digests),
        "runtime_gold_separation": "passed",
        "cases": prepared,
    }
    _write_json(output / "manifest.json", summary)
    return summary


def _normalize_clue_interval(
    value: Sequence[float],
    *,
    duration_sec: float,
) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"expected [start, end], got {value!r}")
    start_sec, end_sec = float(value[0]), float(value[1])
    if not math.isfinite(start_sec) or not math.isfinite(end_sec):
        raise ValueError(f"clue interval must be finite: {value!r}")
    if end_sec < start_sec:
        start_sec, end_sec = end_sec, start_sec
    elif end_sec == start_sec:
        if start_sec >= duration_sec:
            start_sec = max(0.0, duration_sec - 0.001)
            end_sec = duration_sec
        else:
            end_sec = start_sec + 0.001
    start_sec, end_sec = round(start_sec, 3), round(end_sec, 3)
    if start_sec < 0.0 or end_sec > duration_sec or end_sec <= start_sec:
        raise ValueError(f"clue interval is outside [0, {duration_sec}]: {value!r}")
    return start_sec, end_sec


def _normalized_text(value: str) -> str:
    return " ".join(str(value).casefold().split())


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split legacy Day cases and build answer-free oracle manifests."
    )
    parser.add_argument("--legacy-case-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--caption-config-digest", required=True)
    parser.add_argument("--experiment-seed", type=int, default=20260811)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = prepare_oracle_day(
        legacy_case_root=Path(args.legacy_case_root),
        dataset_root=Path(args.dataset_root),
        out_root=Path(args.out_root),
        caption_config_digest=args.caption_config_digest,
        experiment_seed=args.experiment_seed,
    )
    print(
        json.dumps(
            {
                "case_count": summary["case_count"],
                "clue_count": summary["clue_count"],
                "manifest": str(Path(args.out_root) / "manifest.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
