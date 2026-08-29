#!/usr/bin/env python3
"""Build the zero-API WP17-1 OCR track and evidence-store canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from vcah.occurrence_negative_sidecar import file_sha256
from vcah.wp17_memory_construction import build_ocr_tracks, build_track_evidence
from vcah.wp17_preflight import diagnostic_frame_label, surface_matches


def run(args: argparse.Namespace) -> Path:
    a3_root = Path(args.a3_root)
    manifest_path = a3_root / "run_manifest.json"
    runtime_report_path = a3_root / "entity_occurrence_report.json"
    manifest = _read_json(manifest_path)
    runtime_report = _read_json(runtime_report_path)
    selection_path = Path(str(manifest["selection_path"]))
    selection_rows = _read_jsonl(selection_path)
    frame_metadata = {
        diagnostic_frame_label(row): {
            "frame_id": diagnostic_frame_label(row),
            "virtual_time_sec": float(row["virtual_time_sec"]),
            "source_time_sec": float(row["source_time_sec"]),
            "segment_id": str(row["segment_id"]),
            "source_video_id": str(row["source_video_id"]),
        }
        for row in selection_rows
    }
    batch_paths = tuple(sorted((a3_root / "batch_results").rglob("*.json")))
    batch_results = tuple(_read_json(path) for path in batch_paths)
    parsed_rows = tuple(
        dict(row)
        for result in batch_results
        for row in tuple(result.get("parsed_rows", ()) or ())
        if isinstance(row, Mapping)
    )
    build = build_ocr_tracks(
        parsed_rows,
        frame_metadata=frame_metadata,
        max_gap_sec=float(args.max_gap_sec),
        default_reader_source=str(args.reader_source),
    )
    tracks = tuple(build["tracks"])
    evidence = build_track_evidence(tracks)

    surface_spec = _read_json(Path(args.surface_spec)) if args.surface_spec else None
    surface_rows = _evaluate_surfaces(tracks, surface_spec, selection_rows)
    checks = {
        "source_commit_exact": manifest.get("source_commit")
        == str(args.expected_source_commit),
        "actual_model_exact": manifest.get("actual_model") == str(args.expected_model),
        "a3_runtime_structural_gate_passed": bool(
            runtime_report.get("gates", {}).get("structural_gate_passed")
        ),
        "question_gold_blind": all(
            manifest.get(key) is False
            for key in (
                "question_visible_to_model",
                "options_visible_to_model",
                "answer_visible_to_model",
                "caption_text_visible_to_model",
                "official_intervals_visible_to_model",
            )
        ),
        "batch_count_exact": len(batch_results)
        == int(runtime_report.get("counts", {}).get("batch_results", -1)),
        "all_batches_successful_and_parsed": bool(batch_results)
        and all(
            result.get("status") == "success"
            and result.get("parse_status") == "success"
            for result in batch_results
        ),
        "parsed_row_count_exact": len(parsed_rows) == int(args.expected_parsed_rows),
        "raw_responses_and_prompts_not_persisted": all(
            result.get("raw_response_persisted") is False
            and result.get("prompt_persisted") is False
            for result in batch_results
        ),
        "track_build_structural_gate_passed": bool(build["structural_gate_passed"]),
        "evidence_count_equals_track_count": len(evidence) == len(tracks),
        "evidence_ids_unique": len({row["evidence_id"] for row in evidence})
        == len(evidence),
        "no_day_test140_or_week": manifest.get("day_test140_accessed") is False
        and manifest.get("week_accessed") is False,
    }
    checks["structural_gate_passed"] = all(checks.values())
    report = {
        "schema_version": "MMLifelongWP17OCRTrackReportV1",
        "decision": (
            "WP17_1_TRACK_CANARY_READY"
            if checks["structural_gate_passed"]
            else "STRUCTURAL_FAILURE"
        ),
        "counts": {
            **dict(build["counts"]),
            "evidence_records": len(evidence),
            "strict_surface_cases": len(surface_rows),
            "strict_surface_cases_represented": sum(
                row["represented"] for row in surface_rows
            ),
        },
        "surface_diagnostic": surface_rows,
        "gates": checks,
        "structural_gate_passed": checks["structural_gate_passed"],
        "model_calls_during_track_build": 0,
        "admission_filter_applied": False,
        "diagnostic_rows_are_official_interval_selected": True,
        "dense_timeline_claimed": False,
        "provenance": {
            "a3_root": str(a3_root),
            "a3_manifest_sha256": file_sha256(manifest_path),
            "a3_runtime_report_sha256": file_sha256(runtime_report_path),
            "selection_sha256": file_sha256(selection_path),
            "batch_result_count": len(batch_paths),
            "expected_source_commit": str(args.expected_source_commit),
            "expected_model": str(args.expected_model),
            "reader_source": str(args.reader_source),
            "max_gap_sec": float(args.max_gap_sec),
        },
    }

    out_root = Path(args.out_root)
    targets = {
        "tracks": out_root / "ocr_tracks.jsonl",
        "evidence": out_root / "evidence_store.jsonl",
        "report": out_root / "wp17_ocr_track_report.json",
        "markdown": out_root / "wp17_ocr_track_report.md",
    }
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError(f"WP17 track outputs already exist: {existing}")
    out_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(targets["tracks"], tracks)
    _write_jsonl(targets["evidence"], evidence)
    _write_json(targets["report"], report)
    targets["markdown"].write_text(_render_markdown(report), encoding="utf-8")
    print(
        "WP17_TRACKS_DONE "
        f"decision={report['decision']} rows={len(parsed_rows)} "
        f"tracks={len(tracks)} evidence={len(evidence)} "
        f"gate={str(report['structural_gate_passed']).lower()}",
        flush=True,
    )
    return targets["report"]


def _evaluate_surfaces(
    tracks: Sequence[Mapping[str, Any]],
    surface_spec: Mapping[str, Any] | None,
    selection_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not surface_spec:
        return []
    case_frame_ids: dict[str, set[str]] = {}
    for selection in selection_rows:
        frame_id = diagnostic_frame_label(selection)
        for case_id in tuple(selection.get("diagnostic_case_ids", ()) or ()):
            case_frame_ids.setdefault(str(case_id), set()).add(frame_id)
    rows = []
    for case_id, raw in sorted(dict(surface_spec.get("cases", {}) or {}).items()):
        expected = tuple(dict(raw).get("expected_surfaces", ()) or ())
        observed = [
            str(surface.get("surface", "") or "")
            for track in tracks
            if case_frame_ids.get(str(case_id), set())
            & {str(value) for value in tuple(track.get("support_frame_ids", ()) or ())}
            for surface in tuple(track.get("surfaces", ()) or ())
        ]
        rows.append(
            {
                "case_id": str(case_id),
                "represented": any(surface_matches(value, expected) for value in observed),
            }
        )
    return rows


def _render_markdown(report: Mapping[str, Any]) -> str:
    counts = dict(report["counts"])
    lines = [
        "# MM-Lifelong WP17-1 OCR Track Canary",
        "",
        f"- 决策：`{report['decision']}`",
        f"- 结构门：`{str(report['structural_gate_passed']).lower()}`",
        f"- OCR observations / tracks / evidence：`{counts['assigned_observations']} / {counts['tracks']} / {counts['evidence_records']}`",
        f"- Multi-frame / singleton tracks：`{counts['multi_frame_tracks']} / {counts['singleton_tracks']}`",
        f"- 重复表面被分成多个 track：`{counts['surfaces_with_multiple_tracks']}`",
        f"- 严格表面 representation：`{counts['strict_surface_cases_represented']}/{counts['strict_surface_cases']}`",
        "- 本 canary 重放既有 question-blind OCR rows，未调用模型，也不宣称覆盖完整 timeline。",
        "- Track 层不执行 lexical/admission 过滤；所有非空 observation 必须且只能归属一个 track。",
        "",
    ]
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"expected JSON object row: {path}")
            rows.append(dict(payload))
    return tuple(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a3-root", required=True)
    parser.add_argument("--surface-spec")
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--expected-parsed-rows", type=int, required=True)
    parser.add_argument("--reader-source", default="gemini25_pro_ppio")
    parser.add_argument("--max-gap-sec", type=float, default=3.0)
    parser.add_argument("--out-root", required=True)
    return parser.parse_args()


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
