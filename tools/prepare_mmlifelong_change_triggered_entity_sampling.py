#!/usr/bin/env python3
"""Build exact-budget uniform and change-triggered frame manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from vcah.change_triggered_entity_occurrence import (
    CHANGE_TRIGGERED_ENTITY_CONTRACT,
    scan_segment_change_observations,
    select_change_budget,
    select_uniform_budget,
    write_jsonl,
)
from vcah.occurrence_negative_sidecar import file_sha256, stable_digest
from vcah.virtual_video import VirtualVideoSegment, VirtualVideoWorkspace


def run(args: argparse.Namespace) -> Path:
    protocol_path = Path(args.protocol_spec)
    _require_sha(protocol_path, args.expected_protocol_sha256, "protocol")
    protocol = _read_json(protocol_path)
    if protocol.get("contract") != CHANGE_TRIGGERED_ENTITY_CONTRACT:
        raise ValueError("WP16-7 protocol contract mismatch")
    if not bool(protocol.get("protocol_frozen_before_outcomes")):
        raise ValueError("WP16-7 protocol was not frozen before outcomes")

    workspace = VirtualVideoWorkspace.load(Path(args.workspace_root))
    segments = _selected_segments(
        workspace.manifest.segments,
        segment_ids=tuple(args.segment_ids or ()),
    )
    if not segments:
        raise ValueError("no virtual video segments selected")
    out_root = Path(args.out_root)
    if out_root.exists() and any(out_root.iterdir()) and not args.resume:
        raise FileExistsError(f"sampling output is not empty: {out_root}")
    score_root = out_root / "tier0_scores"
    score_root.mkdir(parents=True, exist_ok=True)

    observations: list[dict[str, Any]] = []
    segment_rows = []
    for completed, segment in enumerate(segments, start=1):
        score_path = score_root / f"{_segment_key(segment)}.jsonl"
        rows = _read_jsonl(score_path) if args.resume and score_path.is_file() else ()
        reused = bool(rows) and _valid_reusable_segment(rows, segment=segment)
        if not reused:
            rows = scan_segment_change_observations(
                segment,
                fps=float(args.tier0_fps),
                width=int(args.tier0_width),
                height=int(args.tier0_height),
                ffmpeg_executable=str(args.ffmpeg_executable),
            )
            write_jsonl(score_path, rows)
        observations.extend(dict(row) for row in rows)
        segment_rows.append(
            {
                "segment_id": segment.segment_id,
                "source_video_id": segment.source_video_id,
                "virtual_start_sec": segment.virtual_start_sec,
                "virtual_end_sec": segment.virtual_end_sec,
                "observation_count": len(rows),
                "score_path": str(score_path),
                "score_sha256": file_sha256(score_path),
                "resume_reused": reused,
            }
        )
        print(
            "TIER0_SEGMENT_DONE "
            f"completed={completed}/{len(segments)} "
            f"segment={segment.segment_id} observations={len(rows)} "
            f"reused={str(reused).lower()}",
            flush=True,
        )
    if not observations:
        raise RuntimeError("Tier-0 scan produced zero observations")

    budget = int(args.budget)
    uniform = select_uniform_budget(observations, budget=budget)
    changed = select_change_budget(
        observations,
        budget=budget,
        coverage_bin_sec=float(args.coverage_bin_sec),
        min_spacing_sec=float(args.min_spacing_sec),
    )
    selection_root = out_root / "selections"
    uniform_path = selection_root / "a1_uniform.jsonl"
    change_path = selection_root / "a2_change.jsonl"
    write_jsonl(uniform_path, uniform)
    write_jsonl(change_path, changed)

    tier0_manifest = {
        "schema_version": "MMLifelongTier0ChangeManifestV1",
        "contract": CHANGE_TRIGGERED_ENTITY_CONTRACT,
        "source_commit": str(args.source_commit),
        "protocol_path": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "workspace_root": str(Path(args.workspace_root)),
        "workspace_id": workspace.workspace_id,
        "asset_root": str(workspace.asset_root),
        "tier0_fps": float(args.tier0_fps),
        "tier0_width": int(args.tier0_width),
        "tier0_height": int(args.tier0_height),
        "ffmpeg_executable": str(args.ffmpeg_executable),
        "segment_count": len(segments),
        "observation_count": len(observations),
        "observation_digest": stable_digest(
            [
                {
                    "segment_id": row["segment_id"],
                    "tier0_frame_index": row["tier0_frame_index"],
                    "selection_score": row["selection_score"],
                }
                for row in observations
            ]
        ),
        "segments": segment_rows,
        "question_visible_to_sampling": False,
        "options_visible_to_sampling": False,
        "answer_visible_to_sampling": False,
        "official_intervals_visible_to_sampling": False,
        "caption_visible_to_sampling": False,
        "video_copied": False,
        "dense_frames_persisted": False,
        "day_test140_accessed": False,
        "week_accessed": False,
    }
    _write_json(out_root / "tier0_manifest.json", tier0_manifest)
    checks = {
        "exact_shared_tier0_frame_universe": all(
            row.get("contract") == CHANGE_TRIGGERED_ENTITY_CONTRACT
            for row in observations
        ),
        "a1_exact_budget": len(uniform) == budget,
        "a2_exact_budget": len(changed) == budget,
        "a1_a2_exact_budget_equality": len(uniform) == len(changed),
        "question_and_gold_blind_sampling": all(
            tier0_manifest[key] is False
            for key in (
                "question_visible_to_sampling",
                "options_visible_to_sampling",
                "answer_visible_to_sampling",
                "official_intervals_visible_to_sampling",
                "caption_visible_to_sampling",
            )
        ),
        "zero_video_copy": tier0_manifest["video_copied"] is False,
        "zero_dense_frame_persistence": tier0_manifest["dense_frames_persisted"]
        is False,
    }
    checks["structural_gate_passed"] = all(checks.values())
    report = {
        "schema_version": "MMLifelongChangeTriggeredSamplingReportV1",
        "contract": CHANGE_TRIGGERED_ENTITY_CONTRACT,
        "decision": (
            "SAMPLING_READY"
            if checks["structural_gate_passed"]
            else "STRUCTURAL_FAILURE"
        ),
        "counts": {
            "segments": len(segments),
            "tier0_observations": len(observations),
            "a1_selected_frames": len(uniform),
            "a2_selected_frames": len(changed),
            "a2_coverage_bin_peaks": sum(
                row["selection_reason"] == "coverage_bin_peak" for row in changed
            ),
            "a2_ranked_change_peaks": sum(
                row["selection_reason"] == "ranked_change_peak" for row in changed
            ),
            "a2_spacing_fallback": sum(
                row["selection_reason"] == "exact_budget_spacing_fallback"
                for row in changed
            ),
        },
        "selection_paths": {
            "a1_uniform": str(uniform_path),
            "a2_change": str(change_path),
        },
        "selection_sha256": {
            "a1_uniform": file_sha256(uniform_path),
            "a2_change": file_sha256(change_path),
        },
        "gates": checks,
        "endpoint_values_were_not_computed": True,
        "retrieval_run": False,
        "qa_run": False,
        "judge_calls": 0,
    }
    report_path = out_root / "sampling_report.json"
    _write_json(report_path, report)
    (out_root / "sampling_report.md").write_text(
        _render_markdown(report), encoding="utf-8"
    )
    print(
        "CHANGE_SAMPLING_DONE "
        f"observations={len(observations)} budget={budget} "
        f"gate={str(checks['structural_gate_passed']).lower()}",
        flush=True,
    )
    return report_path


def _selected_segments(
    segments: Sequence[VirtualVideoSegment],
    *,
    segment_ids: Sequence[str],
) -> tuple[VirtualVideoSegment, ...]:
    requested = tuple(dict.fromkeys(str(value) for value in segment_ids if str(value)))
    if not requested:
        return tuple(segments)
    by_id = {segment.segment_id: segment for segment in segments}
    missing = tuple(value for value in requested if value not in by_id)
    if missing:
        raise ValueError(f"unknown segment IDs: {', '.join(missing)}")
    return tuple(by_id[value] for value in requested)


def _valid_reusable_segment(
    rows: Sequence[Mapping[str, Any]],
    *,
    segment: VirtualVideoSegment,
) -> bool:
    return (
        bool(rows)
        and all(
            row.get("contract") == CHANGE_TRIGGERED_ENTITY_CONTRACT
            and row.get("segment_id") == segment.segment_id
            and isinstance(row.get("tier0_frame_index"), int)
            for row in rows
        )
        and int(rows[-1]["tier0_frame_index"]) == len(rows) - 1
    )


def _segment_key(segment: VirtualVideoSegment) -> str:
    return hashlib.sha256(segment.segment_id.encode("utf-8")).hexdigest()[:16]


def _render_markdown(report: Mapping[str, Any]) -> str:
    counts = report["counts"]
    return "\n".join(
        [
            "# WP16-7 Tier-0 Sampling",
            "",
            f"- Decision: `{report['decision']}`",
            f"- Segments / observations: `{counts['segments']} / {counts['tier0_observations']}`",
            f"- A1 / A2 selected frames: `{counts['a1_selected_frames']} / {counts['a2_selected_frames']}`",
            f"- A2 bin peaks / ranked peaks / fallback: `{counts['a2_coverage_bin_peaks']} / {counts['a2_ranked_change_peaks']} / {counts['a2_spacing_fallback']}`",
            f"- Structural gate: `{str(report['gates']['structural_gate_passed']).lower()}`",
            "",
            "No video copies or dense frame files were produced. Endpoint values were not computed.",
            "",
        ]
    )


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"expected JSON object row: {path}")
            rows.append(dict(payload))
    return tuple(rows)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _require_sha(path: Path, expected: str, label: str) -> None:
    if file_sha256(path) != str(expected):
        raise ValueError(f"{label} SHA256 mismatch")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--protocol-spec", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--tier0-fps", type=float, default=1.0)
    parser.add_argument("--tier0-width", type=int, default=160)
    parser.add_argument("--tier0-height", type=int, default=90)
    parser.add_argument("--ffmpeg-executable", default="ffmpeg")
    parser.add_argument("--coverage-bin-sec", type=float, default=300.0)
    parser.add_argument("--min-spacing-sec", type=float, default=2.0)
    parser.add_argument("--segment-ids", nargs="*", default=())
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
