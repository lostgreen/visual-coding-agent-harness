from __future__ import annotations

from collections import Counter
import hashlib
import re
import unicodedata
from typing import Any, Mapping, Sequence

from vcah.change_triggered_entity_occurrence import admit_entity_occurrences
from vcah.occurrence_entity_sidecar import HIGH_VALUE_UI_REGIONS


WP17_PREFLIGHT_CONTRACT = "WP17-0-memory-construction-preflight-v1"
_SURFACE_CHARACTER_RE = re.compile(r"[0-9a-z\u4e00-\u9fff]")


def diagnostic_frame_label(row: Mapping[str, Any]) -> str:
    segment = hashlib.sha256(str(row["segment_id"]).encode("utf-8")).hexdigest()[:8]
    return f"frame_{segment}_{int(row['tier0_frame_index']):06d}"


def select_surface_review_rows(
    *,
    case_id: str,
    selection_rows: Sequence[Mapping[str, Any]],
    parsed_rows: Sequence[Mapping[str, Any]],
    max_frames: int = 8,
) -> tuple[dict[str, Any], ...]:
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    candidates = sorted(
        (
            dict(row)
            for row in selection_rows
            if case_id
            in {
                str(value)
                for value in tuple(row.get("diagnostic_case_ids", ()) or ())
            }
        ),
        key=lambda row: (
            float(row["virtual_time_sec"]),
            int(row["tier0_frame_index"]),
        ),
    )
    if not candidates:
        raise ValueError(f"no selected frames for case: {case_id}")

    frame_scores: Counter[str] = Counter()
    for row in parsed_rows:
        label = str(row.get("frame_label", "") or "")
        if not label:
            continue
        frame_scores[label] += 1
        if str(row.get("ui_region", "") or "") in HIGH_VALUE_UI_REGIONS:
            frame_scores[label] += 3

    last = len(candidates) - 1
    anchors = {0, last, last // 4, last // 2, (3 * last) // 4}
    selected = {index for index in anchors if 0 <= index <= last}
    ranked = sorted(
        range(len(candidates)),
        key=lambda index: (
            -frame_scores[diagnostic_frame_label(candidates[index])],
            float(candidates[index]["virtual_time_sec"]),
        ),
    )
    for index in ranked:
        if len(selected) >= min(max_frames, len(candidates)):
            break
        selected.add(index)

    result = []
    for index in sorted(selected)[:max_frames]:
        row = dict(candidates[index])
        row["frame_label"] = diagnostic_frame_label(row)
        row["surface_review_score"] = frame_scores[row["frame_label"]]
        result.append(row)
    return tuple(result)


def compact_surface(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return "".join(_SURFACE_CHARACTER_RE.findall(text))


def surface_matches(text: Any, expected_surfaces: Sequence[Any]) -> bool:
    observed = compact_surface(text)
    if not observed:
        return False
    for value in expected_surfaces:
        expected = compact_surface(value)
        if not expected:
            continue
        if observed == expected:
            return True
        if min(len(observed), len(expected)) >= 2 and (
            observed in expected or expected in observed
        ):
            return True
    return False


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    prior = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    prior[right_index] + 1,
                    prior[right_index - 1] + (left_char != right_char),
                )
            )
        prior = current
    return prior[-1]


def build_wp17_preflight_report(
    *,
    case_specs: Mapping[str, Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
    parsed_rows: Sequence[Mapping[str, Any]],
    merge_gap_sec: float,
    structural_checks: Mapping[str, bool],
) -> dict[str, Any]:
    cases = {str(key): dict(value) for key, value in case_specs.items()}
    if not cases:
        raise ValueError("WP17 preflight requires case specs")
    case_frame_labels: dict[str, set[str]] = {case_id: set() for case_id in cases}
    frame_metadata: dict[str, dict[str, Any]] = {}
    for raw in selection_rows:
        label = diagnostic_frame_label(raw)
        if label in frame_metadata:
            raise ValueError(f"duplicate diagnostic frame label: {label}")
        frame_metadata[label] = {
            "frame_id": label,
            "virtual_time_sec": float(raw["virtual_time_sec"]),
            "source_time_sec": float(raw["source_time_sec"]),
            "segment_id": str(raw["segment_id"]),
            "source_video_id": str(raw["source_video_id"]),
        }
        for case_id in tuple(raw.get("diagnostic_case_ids", ()) or ()):
            if str(case_id) in case_frame_labels:
                case_frame_labels[str(case_id)].add(label)

    normalized_rows = tuple(dict(row) for row in parsed_rows)
    unknown_row_labels = sorted(
        {
            str(row.get("frame_label", "") or "")
            for row in normalized_rows
            if str(row.get("frame_label", "") or "") not in frame_metadata
        }
    )
    admission_grid = _build_admission_grid(
        cases=cases,
        case_frame_labels=case_frame_labels,
        rows=normalized_rows,
        frame_metadata=frame_metadata,
        merge_gap_sec=merge_gap_sec,
    )
    current = next(
        row
        for row in admission_grid
        if row["support"] == 2
        and row["lexical_filter_enabled"] is True
        and row["high_value_singleton_enabled"] is True
    )

    case_rows = []
    category_counts: Counter[str] = Counter()
    for case_id, spec in sorted(cases.items()):
        labels = case_frame_labels[case_id]
        expected = tuple(spec.get("expected_surfaces", ()) or ())
        alias_surfaces = tuple(spec.get("diagnostic_alias_surfaces", ()) or ())
        if not expected:
            raise ValueError(f"{case_id}: expected_surfaces cannot be empty")
        observed = tuple(
            row
            for row in normalized_rows
            if str(row.get("frame_label", "") or "") in labels
        )
        pre_matches = tuple(
            row for row in observed if surface_matches(row.get("text", ""), expected)
        )
        alias_matches = tuple(
            row
            for row in observed
            if alias_surfaces
            and surface_matches(row.get("text", ""), alias_surfaces)
            and not surface_matches(row.get("text", ""), expected)
        )
        post_matches = int(current["case_target_match_counts"].get(case_id, 0))
        visual_status = str(
            dict(spec.get("visual_audit", {}) or {}).get("pixel_status", "pending")
        )
        closest = _closest_surfaces(observed, expected)
        category = _diagnostic_category(
            pre_match_count=len(pre_matches),
            post_match_count=post_matches,
            alias_match_count=len(alias_matches),
            visual_status=visual_status,
            closest=closest,
        )
        category_counts[category] += 1
        surface_counts = Counter(
            (
                str(row.get("text", "") or ""),
                str(row.get("ui_region", "other") or "other"),
            )
            for row in observed
            if str(row.get("text", "") or "")
        )
        case_rows.append(
            {
                "case_id": case_id,
                "canonical_entity": str(spec.get("canonical_entity", "") or ""),
                "expected_surfaces": list(expected),
                "selected_frame_count": len(labels),
                "parsed_row_count": len(observed),
                "pre_admission_target_match_count": len(pre_matches),
                "current_admission_target_match_count": post_matches,
                "diagnostic_alias_match_count": len(alias_matches),
                "pixel_status": visual_status,
                "category": category,
                "closest_observed_surfaces": closest,
                "top_observed_surfaces": [
                    {"text": text, "ui_region": region, "count": count}
                    for (text, region), count in surface_counts.most_common(8)
                ],
            }
        )

    checks = {str(key): bool(value) for key, value in structural_checks.items()}
    checks.update(
        {
            "all_case_specs_have_selected_frames": all(case_frame_labels.values()),
            "all_parsed_rows_have_frame_lineage": not unknown_row_labels,
            "admission_grid_complete": len(admission_grid) == 12,
            "current_policy_present_once": sum(
                row["support"] == 2
                and row["lexical_filter_enabled"] is True
                and row["high_value_singleton_enabled"] is True
                for row in admission_grid
            )
            == 1,
        }
    )
    checks["structural_gate_passed"] = all(checks.values())
    pending_visual = sum(row["pixel_status"] == "pending" for row in case_rows)
    decision = (
        "STRUCTURAL_FAILURE"
        if not checks["structural_gate_passed"]
        else "READY_FOR_PIXEL_REVIEW"
        if pending_visual
        else "WP17_0_SURFACE_AUDIT_READY"
    )
    return {
        "schema_version": "MMLifelongWP17PreflightReportV1",
        "contract": WP17_PREFLIGHT_CONTRACT,
        "decision": decision,
        "case_count": len(case_rows),
        "case_level": case_rows,
        "category_counts": dict(sorted(category_counts.items())),
        "admission_grid": admission_grid,
        "pending_visual_review_count": pending_visual,
        "gates": checks,
        "structural_gate_passed": checks["structural_gate_passed"],
        "diagnostic_only": True,
        "endpoint_values_were_not_structural_gates": True,
        "model_calls_during_analysis": 0,
        "retrieval_run": False,
        "qa_run": False,
        "judge_calls": 0,
    }


def _build_admission_grid(
    *,
    cases: Mapping[str, Mapping[str, Any]],
    case_frame_labels: Mapping[str, set[str]],
    rows: Sequence[Mapping[str, Any]],
    frame_metadata: Mapping[str, Mapping[str, Any]],
    merge_gap_sec: float,
) -> list[dict[str, Any]]:
    results = []
    for support in (1, 2, 3):
        for lexical_filter_enabled in (False, True):
            for high_value_singleton_enabled in (False, True):
                admission = admit_entity_occurrences(
                    rows,
                    frame_metadata=frame_metadata,
                    merge_gap_sec=merge_gap_sec,
                    multi_frame_min_support=support,
                    high_value_regions=(
                        tuple(sorted(HIGH_VALUE_UI_REGIONS))
                        if high_value_singleton_enabled
                        else ()
                    ),
                    lexical_filter_enabled=lexical_filter_enabled,
                )
                occurrences = tuple(admission["occurrences"])
                match_counts = {}
                for case_id, spec in cases.items():
                    labels = case_frame_labels[case_id]
                    expected = tuple(spec.get("expected_surfaces", ()) or ())
                    match_counts[case_id] = sum(
                        surface_matches(row.get("text", ""), expected)
                        and bool(
                            labels
                            & {
                                str(value)
                                for value in tuple(row.get("frame_labels", ()) or ())
                            }
                        )
                        for row in occurrences
                    )
                results.append(
                    {
                        "support": support,
                        "lexical_filter_enabled": lexical_filter_enabled,
                        "high_value_singleton_enabled": high_value_singleton_enabled,
                        "admitted_occurrence_count": len(occurrences),
                        "target_covered_case_count": sum(
                            count > 0 for count in match_counts.values()
                        ),
                        "case_target_match_counts": dict(sorted(match_counts.items())),
                        "rejection_counts": dict(admission["rejection_counts"]),
                    }
                )
    return results


def _closest_surfaces(
    rows: Sequence[Mapping[str, Any]], expected_surfaces: Sequence[Any]
) -> list[dict[str, Any]]:
    expected = tuple(
        (str(value), compact_surface(value))
        for value in expected_surfaces
        if compact_surface(value)
    )
    counts: Counter[str] = Counter(
        str(row.get("text", "") or "")
        for row in rows
        if str(row.get("text", "") or "")
    )
    candidates = []
    for text, count in counts.items():
        observed = compact_surface(text)
        if not observed or not expected:
            continue
        alias, alias_compact = min(
            expected,
            key=lambda value: (
                levenshtein_distance(observed, value[1]),
                value[0],
            ),
        )
        distance = levenshtein_distance(observed, alias_compact)
        candidates.append(
            {
                "text": text,
                "count": count,
                "closest_expected_surface": alias,
                "edit_distance": distance,
                "distance_ratio": round(
                    distance / max(len(observed), len(alias_compact)), 4
                ),
            }
        )
    return sorted(
        candidates,
        key=lambda row: (
            float(row["distance_ratio"]),
            int(row["edit_distance"]),
            -int(row["count"]),
            str(row["text"]),
        ),
    )[:5]


def _diagnostic_category(
    *,
    pre_match_count: int,
    post_match_count: int,
    alias_match_count: int,
    visual_status: str,
    closest: Sequence[Mapping[str, Any]],
) -> str:
    if pre_match_count:
        return "target_surface_admitted" if post_match_count else "admission_rejection"
    if alias_match_count:
        return "alias_mismatch"
    if visual_status == "absent":
        return "pixel_absent"
    if visual_status == "visible":
        near = bool(closest) and int(closest[0].get("edit_distance", 99)) <= 2 and float(
            closest[0].get("distance_ratio", 1.0)
        ) <= 0.67
        return "reader_near_miss" if near else "reader_miss"
    if visual_status == "uncertain":
        return "pixel_visibility_uncertain"
    return "pending_pixel_review"
