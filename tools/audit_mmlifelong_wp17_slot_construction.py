#!/usr/bin/env python3
"""Independently replay and audit a WP17 slot-construction run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from vcah.occurrence_negative_sidecar import file_sha256
from vcah.virtual_video import VirtualVideoWorkspace
from vcah.wp17_slot_memory import SlotMemoryState, validate_construction_output
from vcah.wp17_slot_protocol import WP17_3_MANIFEST_CONTRACT
from vcah.wp17_slot_runner import build_asr_packet, build_ocr_packet, frame_evidence_ids


def run(args: argparse.Namespace) -> Path:
    protocol_path = Path(args.protocol_manifest)
    if file_sha256(protocol_path) != str(args.expected_protocol_sha256):
        raise ValueError("WP17 slot audit protocol SHA mismatch")
    protocol = _read_json(protocol_path)
    if protocol.get("contract") != WP17_3_MANIFEST_CONTRACT:
        raise ValueError("WP17 slot audit protocol contract mismatch")
    run_root = Path(args.run_root)
    run_manifest = _read_json(run_root / "run_manifest.json")
    summary = _read_json(run_root / "run_summary.json")
    mode = str(run_manifest["mode"])
    all_segments = tuple(dict(row) for row in protocol["segments"])
    if mode == "canary":
        selected = set(str(value) for value in protocol["canary_segment_chain"])
        segments = tuple(row for row in all_segments if str(row["segment_id"]) in selected)
    else:
        segments = all_segments
    expected_results = len(segments) * 3

    workspace = VirtualVideoWorkspace.load(Path(args.workspace_root))
    asr_cues = workspace.read_asr_virtual_cues()
    evidence_rows = _read_jsonl(Path(args.dense_root) / "evidence_store.jsonl")
    result_rows: list[dict[str, Any]] = []
    replay_errors: list[dict[str, str]] = []
    slot_state = SlotMemoryState("e1c2", token_budget=600)
    active_window = ""
    context_pairs = []

    for segment in segments:
        segment_id = str(segment["segment_id"])
        window_id = str(segment["window_id"])
        if window_id != active_window:
            active_window = window_id
            slot_state = SlotMemoryState("e1c2", token_budget=600)
        start = float(segment["virtual_start_sec"])
        end = float(segment["virtual_end_sec"])
        ocr_packet = build_ocr_packet(evidence_rows, start_sec=start, end_sec=end)
        asr_packet = build_asr_packet(
            asr_cues,
            segment_id=segment_id,
            start_sec=start,
            end_sec=end,
        )
        allowed = (
            frame_evidence_ids(segment_id, int(segment["max_frames"]))
            + tuple(str(row["evidence_id"]) for row in ocr_packet)
            + tuple(str(row["evidence_id"]) for row in asr_packet)
        )
        by_arm = {}
        for arm in tuple(segment["arm_execution_order"]):
            path = run_root / "segments" / segment_id / f"{arm}.json"
            if not path.is_file():
                replay_errors.append(
                    {"segment_id": segment_id, "arm": str(arm), "error": "missing_result"}
                )
                continue
            row = _read_json(path)
            result_rows.append(row)
            by_arm[str(arm)] = row
            if row.get("status") != "success":
                replay_errors.append(
                    {"segment_id": segment_id, "arm": str(arm), "error": "non_success"}
                )
                continue
            try:
                normalized = validate_construction_output(
                    dict(row["model_output"]),
                    arm=str(arm),
                    segment_id=segment_id,
                    allowed_evidence_ids=allowed,
                    state=slot_state if str(arm) == "e1c2" else None,
                )
            except Exception as exc:
                replay_errors.append(
                    {
                        "segment_id": segment_id,
                        "arm": str(arm),
                        "error": type(exc).__name__,
                    }
                )
                continue
            if str(arm) == "e1c2" and normalized.get("state_digest") != row.get("state_digest"):
                replay_errors.append(
                    {"segment_id": segment_id, "arm": str(arm), "error": "state_digest"}
                )
        if "e1c1" in by_arm and "e1c2" in by_arm:
            context_pairs.append(
                {
                    "segment_id": segment_id,
                    "c1_tokens": int(by_arm["e1c1"].get("history_token_count", 0) or 0),
                    "c1_limit": int(by_arm["e1c1"].get("history_token_limit", 0) or 0),
                    "c2_tokens": int(by_arm["e1c2"].get("history_token_count", 0) or 0),
                }
            )

    safe_metadata_rows = [
        dict(attempt.get("response_metadata", {}) or {})
        for row in result_rows
        for attempt in tuple(row.get("attempts", ()) or ())
        if attempt.get("status") == "success"
    ]
    all_serialized = json.dumps(
        {"manifest": run_manifest, "summary": summary, "results": result_rows},
        ensure_ascii=False,
        sort_keys=True,
    ).casefold()
    forbidden_persisted_keys = (
        '"api_key"',
        '"authorization"',
        '"bridge_url"',
        '"raw_response"',
        '"source_path"',
        '"question"',
        '"options"',
        '"gold_answer"',
        '"official_intervals"',
        '"case_id"',
        '"case_ids"',
    )
    c1_mean = mean(row["c1_tokens"] for row in context_pairs) if context_pairs else 0.0
    c2_mean = mean(row["c2_tokens"] for row in context_pairs) if context_pairs else 0.0
    gates = {
        "protocol_structural_gate_passed": bool(protocol.get("structural_gate_passed")),
        "protocol_sha_exact": run_manifest.get("protocol_manifest_sha256")
        == file_sha256(protocol_path),
        "actual_model_exact": run_manifest.get("actual_model")
        == protocol["model_policy"]["actual_model"],
        "source_commit_exact": run_manifest.get("source_commit")
        == protocol.get("provenance", {}).get("source_commit"),
        "frame_preprocessing_exact": run_manifest.get("image_preprocessing")
        == protocol["evidence_policy"]["frame_preprocessing"]
        and all(
            row.get("image_preprocessing")
            == protocol["evidence_policy"]["frame_preprocessing"]
            for row in result_rows
        ),
        "result_count_exact": len(result_rows) == expected_results,
        "all_results_success": len(result_rows) == expected_results
        and all(row.get("status") == "success" for row in result_rows),
        "summary_complete": summary.get("complete") is True
        and int(summary.get("successes", 0)) == expected_results,
        "model_call_cap_respected": int(summary.get("model_calls", 0))
        <= int(summary.get("model_call_hard_cap", -1)),
        "independent_transaction_replay_passed": not replay_errors,
        "three_arm_input_digests_exact": all(
            len(
                {
                    json.dumps(row.get("input_digests", {}), sort_keys=True)
                    for row in result_rows
                    if row.get("segment_id") == segment["segment_id"]
                }
            )
            == 1
            for segment in segments
        ),
        "context_limit_pairing_exact": bool(context_pairs)
        and all(row["c1_limit"] == row["c2_tokens"] for row in context_pairs),
        "context_mean_budget_gap_within_10pct": abs(c1_mean - c2_mean) / 600.0 <= 0.10,
        "capsules_within_600_tokens": all(
            row.get("capsule") is None
            or (
                row["capsule"].get("within_budget") is True
                and int(row["capsule"].get("token_count", 601)) <= 600
            )
            for row in result_rows
        ),
        "response_usage_metadata_keys_present": len(safe_metadata_rows) == expected_results
        and all(
            all(
                key in metadata
                for key in (
                    "finish_reason",
                    "prompt_tokens",
                    "completion_tokens",
                    "reasoning_tokens",
                )
            )
            for metadata in safe_metadata_rows
        ),
        "raw_response_not_persisted": run_manifest.get("raw_model_response_persisted")
        is False
        and all(row.get("raw_model_response_persisted") is False for row in result_rows),
        "temporary_frames_not_retained": run_manifest.get("temporary_frames_retained")
        is False
        and all(row.get("temporary_frames_retained") is False for row in result_rows),
        "forbidden_fields_not_persisted": not any(
            value in all_serialized for value in forbidden_persisted_keys
        ),
        "question_gold_interval_blind": all(
            run_manifest.get(key) is False
            for key in (
                "question_visible",
                "options_visible",
                "answer_visible",
                "official_intervals_visible",
                "case_ids_visible",
            )
        ),
        "endpoint_values_not_gates": protocol.get("gates", {}).get(
            "endpoint_values_not_structural_gates"
        )
        is True,
    }
    gates["structural_gate_passed"] = all(gates.values())
    report = {
        "schema_version": "MMLifelongWP17SlotConstructionAuditV1",
        "contract": "WP17-3-slot-construction-audit-v1",
        "decision": (
            "WP17_3_SLOT_CANARY_PASSED"
            if gates["structural_gate_passed"] and mode == "canary"
            else "WP17_3_SLOT_RUN_PASSED"
            if gates["structural_gate_passed"]
            else "WP17_3_SLOT_STRUCTURAL_STOP"
        ),
        "mode": mode,
        "counts": {
            "segments": len(segments),
            "expected_results": expected_results,
            "results": len(result_rows),
            "successes": sum(row.get("status") == "success" for row in result_rows),
            "model_calls": int(summary.get("model_calls", 0)),
            "replay_errors": len(replay_errors),
        },
        "context_budget": {
            "c1_mean_tokens": round(c1_mean, 3),
            "c2_mean_tokens": round(c2_mean, 3),
            "absolute_mean_gap": round(abs(c1_mean - c2_mean), 3),
            "budget": 600,
        },
        "replay_error_fingerprints": replay_errors[:5],
        "gates": gates,
        "structural_gate_passed": gates["structural_gate_passed"],
        "endpoint_values_evaluated": False,
        "model_calls": 0,
    }
    out_root = Path(args.out_root)
    report_path = out_root / "wp17_slot_construction_audit.json"
    markdown_path = out_root / "wp17_slot_construction_audit.md"
    if report_path.exists() or markdown_path.exists():
        raise FileExistsError("WP17 slot audit output already exists")
    out_root.mkdir(parents=True, exist_ok=True)
    _write_json(report_path, report)
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    print(
        "WP17_SLOT_AUDIT_DONE "
        f"decision={report['decision']} results={len(result_rows)}/{expected_results} "
        f"replay_errors={len(replay_errors)} gate={str(gates['structural_gate_passed']).lower()} "
        "model_calls=0",
        flush=True,
    )
    return report_path


def _render_markdown(report: Mapping[str, Any]) -> str:
    counts = dict(report["counts"])
    context = dict(report["context_budget"])
    return "\n".join(
        (
            "# MM-Lifelong WP17-3 Slot Construction Audit",
            "",
            f"- Decision: `{report['decision']}`",
            f"- Structural gate: `{str(report['structural_gate_passed']).lower()}`",
            f"- Results / expected: `{counts['results']} / {counts['expected_results']}`",
            f"- Independent replay errors: `{counts['replay_errors']}`",
            f"- C1 / C2 mean history tokens: `{context['c1_mean_tokens']} / {context['c2_mean_tokens']}`",
            "- Endpoint values were not evaluated or used as gates.",
            "",
        )
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError(f"expected JSONL object: {path}")
                rows.append(dict(payload))
    return tuple(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-manifest", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--dense-root", required=True)
    parser.add_argument("--out-root", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
