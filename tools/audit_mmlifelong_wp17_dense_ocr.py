#!/usr/bin/env python3
"""Audit WP17-1 dense OCR target representation without model calls."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from vcah.occurrence_negative_sidecar import file_sha256
from vcah.wp17_preflight import (
    compact_surface,
    levenshtein_distance,
    surface_matches,
)


NEAR_MATCH_MAX_EDIT_DISTANCE = 2
NEAR_MATCH_MAX_DISTANCE_RATIO = 0.67


def build_dense_ocr_audit(
    *,
    tracks: Sequence[Mapping[str, Any]],
    dense_report: Mapping[str, Any],
    dense_manifest: Mapping[str, Any],
    timeline: Mapping[str, Any],
    surface_spec: Mapping[str, Any],
    preflight_report: Mapping[str, Any],
    a3_runtime_report: Mapping[str, Any],
    track_line_count: int,
    evidence_line_count: int,
    expected_dense_source_commit: str,
    audit_source_commit: str,
) -> dict[str, Any]:
    cases = {
        str(case_id): dict(row)
        for case_id, row in dict(surface_spec.get("cases", {}) or {}).items()
    }
    preflight_cases = {
        str(row["case_id"]): dict(row)
        for row in tuple(preflight_report.get("case_level", ()) or ())
    }
    case_windows: dict[str, list[tuple[float, float]]] = {
        case_id: [] for case_id in cases
    }
    timeline_case_ids: set[str] = set()
    for raw in tuple(timeline.get("windows", ()) or ()):
        window = dict(raw)
        interval = (
            float(window["virtual_start_sec"]),
            float(window["virtual_end_sec"]),
        )
        for raw_case_id in tuple(window.get("case_ids", ()) or ()):
            case_id = str(raw_case_id)
            timeline_case_ids.add(case_id)
            if case_id in case_windows:
                case_windows[case_id].append(interval)

    case_rows = []
    for case_id, spec in sorted(cases.items()):
        expected = tuple(spec.get("expected_surfaces", ()) or ())
        aliases = tuple(spec.get("diagnostic_alias_surfaces", ()) or ())
        targets = expected + aliases
        windows = tuple(case_windows.get(case_id, ()))
        relevant = tuple(
            track
            for track in tracks
            if any(_track_overlaps(track, interval) for interval in windows)
        )
        surface_counts: Counter[str] = Counter()
        strict_tracks = []
        near_tracks = []
        for track in relevant:
            surfaces = tuple(track.get("surfaces", ()) or ())
            values = tuple(
                str(dict(surface).get("surface", "") or "")
                for surface in surfaces
                if str(dict(surface).get("surface", "") or "")
            )
            for surface in surfaces:
                row = dict(surface)
                text = str(row.get("surface", "") or "")
                if text:
                    surface_counts[text] += int(row.get("count", 1) or 1)
            if any(surface_matches(value, expected) for value in values):
                strict_tracks.append(track)
            elif any(_near_surface_match(value, targets) for value in values):
                near_tracks.append(track)

        strict_hit = bool(strict_tracks)
        near_hit = bool(near_tracks)
        alias_aware_hit = strict_hit or near_hit
        preflight = preflight_cases.get(case_id, {})
        a3_strict_hit = int(
            preflight.get("pre_admission_target_match_count", 0) or 0
        ) > 0
        a3_alias_aware_hit = a3_strict_hit or str(
            preflight.get("category", "") or ""
        ) in {"reader_near_miss", "alias_mismatch"}
        pixel_status = str(
            dict(spec.get("visual_audit", {}) or {}).get("pixel_status", "pending")
        )
        closest = _closest_surface(surface_counts, targets)
        matching_tracks = strict_tracks + near_tracks
        case_rows.append(
            {
                "case_id": case_id,
                "canonical_entity": str(spec.get("canonical_entity", "") or ""),
                "pixel_status": pixel_status,
                "window_count": len(windows),
                "observed_track_count": len(relevant),
                "observed_surface_count": len(surface_counts),
                "paddle_strict_hit": strict_hit,
                "paddle_alias_aware_hit": alias_aware_hit,
                "paddle_outcome": (
                    "strict_hit"
                    if strict_hit
                    else "near_miss"
                    if near_hit
                    else "miss"
                ),
                "strict_matching_track_count": len(strict_tracks),
                "near_only_matching_track_count": len(near_tracks),
                "matching_support_frame_count": sum(
                    int(track.get("support_frame_count", 0) or 0)
                    for track in matching_tracks
                ),
                "closest_observed_surface": closest,
                "a3_gemini_strict_hit": a3_strict_hit,
                "a3_gemini_alias_aware_hit": a3_alias_aware_hit,
                "a3_gemini_outcome": str(preflight.get("category", "missing")),
                "strict_complementarity": _complementarity_label(
                    paddle=strict_hit, a3=a3_strict_hit
                ),
                "alias_aware_complementarity": _complementarity_label(
                    paddle=alias_aware_hit, a3=a3_alias_aware_hit
                ),
            }
        )

    visible = tuple(row for row in case_rows if row["pixel_status"] == "visible")
    paddle_strict_hits = sum(row["paddle_strict_hit"] for row in visible)
    paddle_alias_hits = sum(row["paddle_alias_aware_hit"] for row in visible)
    a3_strict_hits = sum(row["a3_gemini_strict_hit"] for row in visible)
    a3_alias_hits = sum(row["a3_gemini_alias_aware_hit"] for row in visible)
    visible_count = len(visible)
    paddle_strict_recall = _rate(paddle_strict_hits, visible_count)
    a3_strict_recall = _rate(a3_strict_hits, visible_count)
    strict_complementarity = Counter(
        row["strict_complementarity"] for row in visible
    )
    alias_complementarity = Counter(
        row["alias_aware_complementarity"] for row in visible
    )
    dense_counts = dict(dense_report.get("counts", {}) or {})
    dense_track_counts = dict(dense_report.get("track_counts", {}) or {})
    a3_counts = dict(a3_runtime_report.get("counts", {}) or {})
    checks = {
        "dense_structural_gate_passed": bool(
            dense_report.get("structural_gate_passed")
        ),
        "dense_source_commit_exact": dense_manifest.get("source_commit")
        == str(expected_dense_source_commit),
        "timeline_structural_gate_passed": bool(
            timeline.get("structural_gate_passed")
        ),
        "surface_spec_is_evaluation_only": surface_spec.get("evaluation_only")
        is True,
        "surface_spec_hidden_from_reader": surface_spec.get("visible_to_ocr_model")
        is False,
        "preflight_structural_gate_passed": bool(
            preflight_report.get("structural_gate_passed")
        ),
        "a3_runtime_structural_gate_passed": bool(
            dict(a3_runtime_report.get("gates", {}) or {}).get(
                "structural_gate_passed"
            )
        ),
        "surface_case_set_matches_preflight": set(cases) == set(preflight_cases),
        "surface_cases_covered_by_timeline": set(cases) <= timeline_case_ids,
        "all_surface_cases_have_windows": all(case_windows.values()),
        "pixel_visible_case_set_nonempty": visible_count > 0,
        "track_count_matches_report": int(track_line_count)
        == int(dense_counts.get("tracks", -1)),
        "evidence_count_matches_report": int(evidence_line_count)
        == int(dense_counts.get("evidence_records", -1)),
        "zero_dense_gemini_calls": int(dense_report.get("model_calls", -1)) == 0,
        "no_day_test140_or_week": dense_manifest.get("day_test140_accessed")
        is False
        and dense_manifest.get("week_accessed") is False,
        "paddle_strict_recall_at_least_80pct_of_a3_diagnostic": (
            paddle_strict_recall + 1e-12 >= 0.8 * a3_strict_recall
        ),
    }
    checks["structural_and_promotion_gate_passed"] = all(checks.values())
    report = {
        "schema_version": "MMLifelongWP17DenseOCRAuditV1",
        "contract": "WP17-1-dense-ocr-target-representation-audit-v1",
        "decision": (
            "WP17_1_DENSE_OCR_AUDIT_READY"
            if checks["structural_and_promotion_gate_passed"]
            else "WP17_1_DENSE_OCR_AUDIT_STOP"
        ),
        "counts": {
            "surface_cases": len(case_rows),
            "pixel_visible_cases": visible_count,
            "pixel_absent_cases": sum(
                row["pixel_status"] == "absent" for row in case_rows
            ),
            "paddle_strict_hits": paddle_strict_hits,
            "paddle_alias_aware_hits": paddle_alias_hits,
            "a3_gemini_strict_hits": a3_strict_hits,
            "a3_gemini_alias_aware_hits": a3_alias_hits,
        },
        "representation": {
            "paddle_dense": {
                "strict_recall": paddle_strict_recall,
                "alias_aware_recall": _rate(paddle_alias_hits, visible_count),
            },
            "a3_gemini_diagnostic": {
                "strict_recall": a3_strict_recall,
                "alias_aware_recall": _rate(a3_alias_hits, visible_count),
            },
            "strict_complementarity": dict(sorted(strict_complementarity.items())),
            "alias_aware_complementarity": dict(
                sorted(alias_complementarity.items())
            ),
        },
        "fragmentation": {
            "dense_tracks": int(dense_counts.get("tracks", 0) or 0),
            "normalized_surfaces": int(
                dense_track_counts.get("normalized_surfaces", 0) or 0
            ),
            "surfaces_with_multiple_tracks": int(
                dense_track_counts.get("surfaces_with_multiple_tracks", 0) or 0
            ),
            "strict_target_matching_tracks": sum(
                row["strict_matching_track_count"] for row in visible
            ),
            "near_only_target_matching_tracks": sum(
                row["near_only_matching_track_count"] for row in visible
            ),
            "visible_cases_with_multiple_target_tracks": sum(
                row["strict_matching_track_count"]
                + row["near_only_matching_track_count"]
                > 1
                for row in visible
            ),
        },
        "cost": {
            "paddle_frames": int(dense_counts.get("frames", 0) or 0),
            "paddle_views_per_frame": int(
                dense_counts.get("views_per_frame", 0) or 0
            ),
            "paddle_reader_calls": int(dense_counts.get("reader_calls", 0) or 0),
            "paddle_gemini_calls": 0,
            "a3_selected_frames": int(a3_counts.get("selected_frames", 0) or 0),
            "a3_gemini_calls": int(a3_counts.get("model_calls", 0) or 0),
            "scope_comparable": False,
        },
        "case_level": case_rows,
        "matching_policy": {
            "strict": "WP17-0 surface_matches",
            "alias_aware": (
                "strict or frozen near-miss rule: edit_distance<=2 and "
                "distance_ratio<=0.67"
            ),
            "pixel_absent_excluded_from_reader_recall": True,
        },
        "gates": checks,
        "structural_and_promotion_gate_passed": checks[
            "structural_and_promotion_gate_passed"
        ],
        "endpoint_values_were_not_reader_inputs": True,
        "model_calls_during_audit": 0,
        "retrieval_run": False,
        "qa_run": False,
        "judge_calls": 0,
        "provenance": {
            "dense_data_source_commit": str(expected_dense_source_commit),
            "audit_source_commit": str(audit_source_commit),
        },
    }
    return report


def _track_overlaps(
    track: Mapping[str, Any], interval: tuple[float, float]
) -> bool:
    start, end = interval
    return not (
        float(track.get("end_sec", 0.0)) < start
        or float(track.get("start_sec", 0.0)) > end
    )


def _near_surface_match(text: Any, expected_surfaces: Sequence[Any]) -> bool:
    observed = compact_surface(text)
    if not observed:
        return False
    for value in expected_surfaces:
        expected = compact_surface(value)
        if not expected:
            continue
        distance = levenshtein_distance(observed, expected)
        ratio = distance / max(len(observed), len(expected))
        if (
            distance <= NEAR_MATCH_MAX_EDIT_DISTANCE
            and ratio <= NEAR_MATCH_MAX_DISTANCE_RATIO
        ):
            return True
    return False


def _closest_surface(
    surface_counts: Mapping[str, int], expected_surfaces: Sequence[Any]
) -> dict[str, Any] | None:
    expected = tuple(
        (str(value), compact_surface(value))
        for value in expected_surfaces
        if compact_surface(value)
    )
    candidates = []
    for text, count in surface_counts.items():
        observed = compact_surface(text)
        if not observed or not expected:
            continue
        target, compact_target = min(
            expected,
            key=lambda item: (
                levenshtein_distance(observed, item[1]),
                item[0],
            ),
        )
        distance = levenshtein_distance(observed, compact_target)
        candidates.append(
            {
                "surface": str(text)[:80],
                "count": int(count),
                "closest_expected_surface": target,
                "edit_distance": distance,
                "distance_ratio": round(
                    distance / max(len(observed), len(compact_target)), 4
                ),
            }
        )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            float(row["distance_ratio"]),
            int(row["edit_distance"]),
            -int(row["count"]),
            str(row["surface"]),
        ),
    )


def _complementarity_label(*, paddle: bool, a3: bool) -> str:
    if paddle and a3:
        return "both"
    if paddle:
        return "paddle_only"
    if a3:
        return "a3_only"
    return "neither"


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _render_markdown(report: Mapping[str, Any]) -> str:
    counts = dict(report["counts"])
    representation = dict(report["representation"])
    paddle = dict(representation["paddle_dense"])
    a3 = dict(representation["a3_gemini_diagnostic"])
    cost = dict(report["cost"])
    fragmentation = dict(report["fragmentation"])
    lines = [
        "# MM-Lifelong WP17-1 Dense OCR Target Audit",
        "",
        f"- 决策：`{report['decision']}`",
        f"- Gate：`{str(report['structural_and_promotion_gate_passed']).lower()}`",
        f"- 可见像素 case：`{counts['pixel_visible_cases']}`；pixel-absent：`{counts['pixel_absent_cases']}`（不计 reader failure）",
        f"- Paddle strict / alias-aware recall：`{counts['paddle_strict_hits']}/{counts['pixel_visible_cases']}` / `{counts['paddle_alias_aware_hits']}/{counts['pixel_visible_cases']}`",
        f"- A3 Gemini strict / alias-aware recall：`{counts['a3_gemini_strict_hits']}/{counts['pixel_visible_cases']}` / `{counts['a3_gemini_alias_aware_hits']}/{counts['pixel_visible_cases']}`",
        f"- Recall（Paddle strict vs A3 strict）：`{paddle['strict_recall']:.3f}` vs `{a3['strict_recall']:.3f}`",
        f"- OCR 成本：`{cost['paddle_frames']}` frames，`{cost['paddle_reader_calls']}` local reader calls，`0` Gemini calls。A3 诊断为 `{cost['a3_selected_frames']}` frames / `{cost['a3_gemini_calls']}` Gemini calls；两者 scope 不同，不做直接成本比值。",
        f"- Track fragmentation：`{fragmentation['surfaces_with_multiple_tracks']}/{fragmentation['normalized_surfaces']}` normalized surfaces 被拆成多个 tracks。",
        "",
        "| Case | Pixel | Paddle | A3 Gemini | Strict pairing | Closest Paddle surface |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["case_level"]:
        closest = row.get("closest_observed_surface")
        closest_text = (
            str(dict(closest).get("surface", "")) if isinstance(closest, Mapping) else ""
        )
        lines.append(
            "| {case} | {pixel} | {paddle} | {a3} | {pair} | {closest} |".format(
                case=_escape_markdown(str(row["case_id"]).removeprefix("mmlifelong-game-test-")),
                pixel=_escape_markdown(str(row["pixel_status"])),
                paddle=_escape_markdown(str(row["paddle_outcome"])),
                a3=_escape_markdown(str(row["a3_gemini_outcome"])),
                pair=_escape_markdown(str(row["strict_complementarity"])),
                closest=_escape_markdown(closest_text or "-"),
            )
        )
    lines.extend(
        (
            "",
            "严格命中沿用 WP17-0 `surface_matches`。Alias-aware 只增加此前已冻结的 near-miss 规则（edit distance <=2、normalized ratio <=0.67），没有按本轮结果调参。",
            "",
        )
    )
    return "\n".join(lines)


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError(f"expected JSON object row: {path}")
                rows.append(dict(payload))
    return tuple(rows)


def _count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> Path:
    dense_root = Path(args.dense_root)
    track_path = dense_root / "ocr_tracks.jsonl"
    evidence_path = dense_root / "evidence_store.jsonl"
    dense_report_path = dense_root / "wp17_dense_ocr_report.json"
    dense_manifest_path = dense_root / "run_manifest.json"
    timeline_path = Path(args.timeline_manifest)
    surface_path = Path(args.surface_spec)
    preflight_path = Path(args.preflight_report)
    a3_runtime_path = Path(args.a3_runtime_report)
    report = build_dense_ocr_audit(
        tracks=_read_jsonl(track_path),
        dense_report=_read_json(dense_report_path),
        dense_manifest=_read_json(dense_manifest_path),
        timeline=_read_json(timeline_path),
        surface_spec=_read_json(surface_path),
        preflight_report=_read_json(preflight_path),
        a3_runtime_report=_read_json(a3_runtime_path),
        track_line_count=_count_jsonl(track_path),
        evidence_line_count=_count_jsonl(evidence_path),
        expected_dense_source_commit=str(args.expected_dense_source_commit),
        audit_source_commit=str(args.audit_source_commit),
    )
    report["provenance"].update(
        {
            "timeline_sha256": file_sha256(timeline_path),
            "surface_spec_sha256": file_sha256(surface_path),
            "preflight_report_sha256": file_sha256(preflight_path),
            "dense_report_sha256": file_sha256(dense_report_path),
            "dense_manifest_sha256": file_sha256(dense_manifest_path),
            "a3_runtime_report_sha256": file_sha256(a3_runtime_path),
        }
    )
    out_root = Path(args.out_root)
    report_path = out_root / "wp17_dense_ocr_audit.json"
    markdown_path = out_root / "wp17_dense_ocr_audit.md"
    if report_path.exists() or markdown_path.exists():
        raise FileExistsError("WP17 dense OCR audit output already exists")
    _write_json(report_path, report)
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    print(
        "WP17_DENSE_OCR_AUDIT_DONE "
        f"decision={report['decision']} "
        f"strict={report['counts']['paddle_strict_hits']}/"
        f"{report['counts']['pixel_visible_cases']} "
        f"alias={report['counts']['paddle_alias_aware_hits']}/"
        f"{report['counts']['pixel_visible_cases']} "
        f"gate={str(report['structural_and_promotion_gate_passed']).lower()}",
        flush=True,
    )
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-root", required=True)
    parser.add_argument("--timeline-manifest", required=True)
    parser.add_argument("--surface-spec", required=True)
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--a3-runtime-report", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--expected-dense-source-commit", required=True)
    parser.add_argument("--audit-source-commit", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
