#!/usr/bin/env python3
"""Prepare the frozen, blind WP14 visual discriminability probe."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping

from vcah.occurrence_negative_sidecar import (
    file_sha256,
    positive_source_manifest_digest,
    replay_source_manifest_digest,
)
from vcah.occurrence_visual_probe import (
    VISUAL_PROBE_CONTRACT,
    VISUAL_PROBE_FPS,
    VISUAL_PROBE_FRAME_CAP,
    audit_visual_probe_manifest,
    build_case_probe_plan,
    finalize_case_probe_plan,
    load_visual_probe_source,
)
from vcah.virtual_video import VirtualVideoWorkspace, materialize_window_frames


MAX_WORKERS = 16


def prepare_probe(args: argparse.Namespace) -> tuple[Path, Path]:
    out_root = Path(args.out_root)
    if out_root.exists() and any(out_root.iterdir()) and not args.resume:
        raise FileExistsError(f"probe output is not empty: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    case_ids = _manifest_case_ids(Path(args.case_manifest))
    if args.case_ids:
        requested = set(args.case_ids)
        missing = sorted(requested - set(case_ids))
        if missing:
            raise ValueError("requested IDs are outside the manifest")
        case_ids = tuple(case_id for case_id in case_ids if case_id in requested)
    if args.expected_cases is not None and len(case_ids) != args.expected_cases:
        raise ValueError(
            f"expected {args.expected_cases} cases, selected {len(case_ids)}"
        )
    if not case_ids:
        raise ValueError("no visual probe cases selected")

    positive_root = Path(args.positive_run_root)
    replay_root = Path(args.replay_fixture_root)
    evaluation_root = Path(args.evaluation_record_root)
    positive_before = positive_source_manifest_digest(positive_root, case_ids)
    replay_digest = replay_source_manifest_digest(replay_root, case_ids)

    plans: dict[str, dict[str, Any]] = {}
    for case_id in case_ids:
        source = load_visual_probe_source(
            positive_root / "cases" / case_id,
            evaluation_record_path=evaluation_root / case_id / "evaluation_case.json",
        )
        workspace = VirtualVideoWorkspace.load(evaluation_root / case_id)
        plans[case_id] = build_case_probe_plan(
            source,
            manifest=workspace.manifest,
            seed=int(args.seed),
        )
    eligible_ids = tuple(
        case_id for case_id in case_ids if bool(plans[case_id].get("eligible"))
    )
    if (
        args.expected_eligible_cases is not None
        and len(eligible_ids) != args.expected_eligible_cases
    ):
        raise ValueError(
            "expected "
            f"{args.expected_eligible_cases} eligible cases, found {len(eligible_ids)}"
        )

    workers = max(1, min(MAX_WORKERS, int(args.workers), len(eligible_ids) or 1))
    finalized: dict[str, dict[str, Any]] = {
        case_id: plans[case_id] for case_id in case_ids if case_id not in eligible_ids
    }
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _materialize_case,
                case_id,
                plans[case_id],
                evaluation_root=evaluation_root,
                out_root=out_root,
            ): case_id
            for case_id in eligible_ids
        }
        for future in as_completed(futures):
            case_id = futures[future]
            finalized[case_id] = future.result()
            print(
                f"VISUAL_PROBE_PREP case_id={case_id} "
                f"observations={len(finalized[case_id].get('windows', ()))}",
                flush=True,
            )

    positive_after = positive_source_manifest_digest(positive_root, case_ids)
    ordered_cases = [finalized[case_id] for case_id in case_ids]
    exclusion_counts = Counter(
        str(reason)
        for row in ordered_cases
        for reason in tuple(row.get("exclusion_reasons", ()) or ())
    )
    manifest = {
        "schema_version": "MMLifelongVisualDiscriminabilityProbeManifestV1",
        "contract": VISUAL_PROBE_CONTRACT,
        "study": "WP14-1 provenance and WP14-2 blind visual discriminability",
        "source_commit": str(args.source_commit),
        "case_manifest_sha256": file_sha256(Path(args.case_manifest)),
        "positive_run_root": str(positive_root),
        "evaluation_record_root": str(evaluation_root),
        "replay_fixture_root": str(replay_root),
        "positive_source_manifest_before": positive_before,
        "positive_source_manifest_after": positive_after,
        "positive_root_unmodified": positive_before == positive_after,
        "replay_source_manifest_digest": replay_digest,
        "selected_case_count": len(case_ids),
        "eligible_case_count": len(eligible_ids),
        "excluded_case_count": len(case_ids) - len(eligible_ids),
        "exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
        "visual_profile": {
            "sampling": "uniform_chronological",
            "fps": VISUAL_PROBE_FPS,
            "max_frames": VISUAL_PROBE_FRAME_CAP,
            "profile_frozen_before_outcomes": True,
            "profile_sweep_performed": False,
        },
        "pair_policy": {
            "matched": "highest clue-overlap frozen candidate",
            "mismatched": "highest retrieval-score same-question non-gold candidate",
            "null": "deterministic non-overlapping window, preferring a different source video",
        },
        "blinding": {
            "gold_visible_to_model": False,
            "pair_kind_visible_to_model": False,
            "case_id_visible_to_model": False,
            "r5_winner_visible_to_model": False,
            "r5_margin_visible_to_model": False,
            "false_commit_visible_to_model": False,
            "question_visible_to_model": False,
            "options_or_answer_visible_to_model": False,
        },
        "engineering_thresholds": {
            "matched_minus_mismatched_support_rate_min": 0.40,
            "null_support_rate_max": 0.15,
            "endpoint_values_are_structural_gates": False,
        },
        "agent_behavior_changed": False,
        "workspace_write_enabled": False,
        "reasoner_context_write_enabled": False,
        "day_test140_accessed": False,
        "week_accessed": False,
        "seed": int(args.seed),
        "cases": ordered_cases,
    }
    manifest_path = out_root / "probe_manifest.json"
    _write_json_atomic(manifest_path, manifest)
    audit = audit_visual_probe_manifest(manifest, root=out_root)
    audit_path = out_root / "provenance_audit.json"
    _write_json_atomic(audit_path, audit)
    if not audit["structural_gate_passed"]:
        raise SystemExit(1)
    return manifest_path, audit_path


def _materialize_case(
    case_id: str,
    plan: Mapping[str, Any],
    *,
    evaluation_root: Path,
    out_root: Path,
) -> dict[str, Any]:
    source_workspace = VirtualVideoWorkspace.load(evaluation_root / case_id)
    probe_root = out_root / "materialized" / case_id
    probe_root.mkdir(parents=True, exist_ok=True)
    workspace = replace(
        source_workspace,
        root_dir=probe_root,
        frame_manifest=probe_root / "frame_manifest.jsonl",
    )
    materialized: dict[str, dict[str, Any]] = {}
    for window in tuple(plan.get("windows", ()) or ()):
        if not isinstance(window, Mapping):
            continue
        observation_id = str(window.get("visual_observation_id", "") or "")
        interval = tuple(window.get("time_range", ()) or ())
        try:
            frames = materialize_window_frames(
                workspace,
                float(interval[0]),
                float(interval[1]),
                query_id=observation_id,
                fps=VISUAL_PROBE_FPS,
                max_frames=VISUAL_PROBE_FRAME_CAP,
            )
            materialized[observation_id] = {
                "executed": True,
                "frames": [
                    {
                        "frame_id": frame.frame_id,
                        "path": str(Path(frame.path).relative_to(out_root)),
                        "virtual_time_sec": frame.virtual_time_sec,
                        "segment_id": frame.segment_id,
                        "source_video_id": frame.source_video_id,
                    }
                    for frame in frames
                ],
            }
        except Exception as exc:
            materialized[observation_id] = {
                "executed": False,
                "frames": [],
                "failure_type": type(exc).__name__,
            }
    return finalize_case_probe_plan(plan, materialized_windows=materialized)


def _manifest_case_ids(path: Path) -> tuple[str, ...]:
    manifest = _read_json(path)
    case_ids = tuple(
        str(row.get("case_id", "") or "")
        for row in tuple(manifest.get("cases", ()) or ())
        if isinstance(row, Mapping) and str(row.get("case_id", "") or "")
    )
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("case manifest must contain unique case IDs")
    return case_ids


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-run-root", required=True)
    parser.add_argument("--replay-fixture-root", required=True)
    parser.add_argument("--evaluation-record-root", required=True)
    parser.add_argument("--case-manifest", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--expected-cases", type=int)
    parser.add_argument("--expected-eligible-cases", type=int)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    manifest_path, audit_path = prepare_probe(_parse_args())
    print(
        f"VISUAL_PROBE_PREP_DONE manifest={manifest_path} audit={audit_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
