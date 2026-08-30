"""Frozen protocol construction for the WP17 memory 2x2 experiment."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence


WP17_2_PROTOCOL_CONTRACT = "WP17-2-memory-construction-factorial-protocol-v1"
WP17_2_MANIFEST_CONTRACT = "WP17-2-memory-construction-factorial-manifest-v1"
WP17_2_ARMS = ("c00", "c01", "c10", "c11")


def build_wp17_2_protocol_manifest(
    protocol: Mapping[str, Any],
    *,
    timeline: Mapping[str, Any],
    dense_report: Mapping[str, Any],
    dense_audit: Mapping[str, Any],
    input_sha256: Mapping[str, str],
    source_commit: str,
) -> dict[str, Any]:
    if protocol.get("contract") != WP17_2_PROTOCOL_CONTRACT:
        raise ValueError("WP17-2 protocol contract mismatch")
    scope = dict(protocol.get("scope", {}) or {})
    segment_duration = float(scope["segment_duration_sec"])
    if segment_duration <= 0.0:
        raise ValueError("WP17-2 segment duration must be positive")
    arms = tuple(str(row["arm"]) for row in protocol.get("arms", ()))
    if arms != WP17_2_ARMS:
        raise ValueError("WP17-2 arm order must be C00/C01/C10/C11")

    segments: list[dict[str, Any]] = []
    case_windows: dict[str, list[str]] = {}
    window_segment_ids: dict[str, list[str]] = {}
    for raw_window in tuple(timeline.get("windows", ()) or ()):
        window = dict(raw_window)
        window_id = str(window["window_id"])
        start = float(window["virtual_start_sec"])
        end = float(window["virtual_end_sec"])
        count = int(math.ceil((end - start) / segment_duration))
        ids = []
        for local_index in range(count):
            segment_start = start + local_index * segment_duration
            segment_end = min(end, segment_start + segment_duration)
            segment_id = f"wp17mc_{window_id}_seg_{local_index:04d}"
            arm_offset = len(segments) % len(WP17_2_ARMS)
            arm_order = list(WP17_2_ARMS[arm_offset:] + WP17_2_ARMS[:arm_offset])
            segments.append(
                {
                    "segment_id": segment_id,
                    "window_id": window_id,
                    "segment_ordinal": len(segments),
                    "window_segment_ordinal": local_index,
                    "virtual_start_sec": round(segment_start, 3),
                    "virtual_end_sec": round(segment_end, 3),
                    "frame_sampling_fps": float(scope["frame_sampling_fps"]),
                    "arm_execution_order": arm_order,
                }
            )
            ids.append(segment_id)
        window_segment_ids[window_id] = ids
        for case_id in tuple(window.get("case_ids", ()) or ()):
            case_windows.setdefault(str(case_id), []).append(window_id)

    canary_segment_ids = {}
    for case_id in tuple(scope.get("canary_case_ids", ()) or ()):
        windows = case_windows.get(str(case_id), [])
        if not windows:
            continue
        ids = window_segment_ids[windows[0]]
        canary_segment_ids[str(case_id)] = ids[len(ids) // 2]

    base_calls = len(segments) * len(WP17_2_ARMS)
    hard_cap = int(scope["model_call_hard_cap"])
    expected_sha = dict(scope.get("expected_input_sha256", {}) or {})
    checks = {
        "protocol_frozen_before_wp17_2_outcomes": protocol.get(
            "protocol_frozen_before_wp17_2_outcomes"
        )
        is True,
        "timeline_structural_gate_passed": bool(
            timeline.get("structural_gate_passed")
        ),
        "dense_ocr_structural_gate_passed": bool(
            dense_report.get("structural_gate_passed")
        ),
        "dense_ocr_audit_gate_passed": bool(
            dense_audit.get("structural_and_promotion_gate_passed")
        ),
        "input_sha256_exact": expected_sha == dict(input_sha256),
        "four_factorial_arms_exact": arms == WP17_2_ARMS,
        "segment_count_exact": len(segments)
        == int(scope["expected_segment_count"]),
        "base_call_count_exact": base_calls == int(scope["expected_base_calls"]),
        "hard_cap_covers_base_calls": hard_cap >= base_calls,
        "hard_cap_exact": hard_cap == int(scope["model_call_hard_cap"]),
        "canary_segment_count_exact": len(set(canary_segment_ids.values()))
        == int(scope["expected_canary_segment_count"]),
        "question_gold_official_intervals_hidden": all(
            protocol.get("construction_input_visibility", {}).get(key) is False
            for key in (
                "question",
                "options",
                "gold_answer",
                "official_intervals",
                "evaluation_aliases",
                "case_ids",
            )
        ),
        "cross_treatment_response_replay_disabled": protocol.get(
            "matched_control", {}
        ).get("cross_treatment_response_replay")
        is False,
        "c10_c11_share_frozen_ocr_packet": protocol.get(
            "matched_control", {}
        ).get("c10_c11_share_frozen_ocr_packet")
        is True,
        "state_stores_are_arm_isolated": protocol.get("matched_control", {}).get(
            "state_stores_are_arm_isolated"
        )
        is True,
        "endpoint_values_not_structural_gates": protocol.get(
            "endpoint_values_are_structural_gates"
        )
        is False,
        "no_day_test140_or_week": protocol.get("scope", {}).get(
            "day_test140_accessed"
        )
        is False
        and protocol.get("scope", {}).get("week_accessed") is False,
        "source_paths_not_persisted": "source_path"
        not in json.dumps(segments, ensure_ascii=False),
        "zero_model_calls_during_freeze": True,
    }
    checks["structural_gate_passed"] = all(checks.values())
    return {
        "schema_version": "MMLifelongWP17Memory2x2ManifestV1",
        "contract": WP17_2_MANIFEST_CONTRACT,
        "decision": (
            "WP17_2_PROTOCOL_FROZEN"
            if checks["structural_gate_passed"]
            else "WP17_2_PROTOCOL_FREEZE_FAILED"
        ),
        "counts": {
            "windows": len(tuple(timeline.get("windows", ()) or ())),
            "segments": len(segments),
            "arms": len(WP17_2_ARMS),
            "base_model_calls": base_calls,
            "model_call_hard_cap": hard_cap,
            "canary_segments": len(set(canary_segment_ids.values())),
        },
        "segments": segments,
        "canary_segment_ids_by_evaluation_case": canary_segment_ids,
        "arms": [dict(row) for row in protocol["arms"]],
        "matched_control": dict(protocol["matched_control"]),
        "output_contract": dict(protocol["output_contract"]),
        "structural_gates": list(protocol["structural_gates"]),
        "construction_endpoints": list(protocol["construction_endpoints"]),
        "development_decisions": dict(protocol["development_decisions"]),
        "gates": checks,
        "structural_gate_passed": checks["structural_gate_passed"],
        "construction_input_visibility": dict(
            protocol["construction_input_visibility"]
        ),
        "model_calls_during_freeze": 0,
        "model_calls_launched": False,
        "provenance": {
            "source_commit": str(source_commit),
            "input_sha256": dict(input_sha256),
            "parent_dense_data_commit": str(
                protocol["provenance"]["parent_dense_data_commit"]
            ),
            "parent_dense_audit_commit": str(
                protocol["provenance"]["parent_dense_audit_commit"]
            ),
        },
    }

