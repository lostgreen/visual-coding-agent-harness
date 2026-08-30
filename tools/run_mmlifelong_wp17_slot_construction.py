#!/usr/bin/env python3
"""Run the frozen question-blind WP17 120-second construction arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

from vcah.captioning import materialize_caption_frames, preprocess_caption_images
from vcah.model_client import OpenAICompatibleClient
from vcah.occurrence_negative_sidecar import file_sha256, safe_response_metadata, stable_digest
from vcah.virtual_video import VirtualVideoWorkspace
from vcah.wp17_slot_memory import (
    WP17_BUDGET_TOKENIZER,
    WP17_CAPSULE_PROVENANCE_CONTRACT,
    WP17_MAX_OBSERVATIONS,
    WP17_MAX_OUTPUT_JSON_CHARS,
    WP17_MAX_STRUCTURED_EVENT_ITEMS,
    WP17_SLOT_CAPSULE_CONTRACT,
    WP17_TARGET_OBSERVATION_EVIDENCE_IDS,
    SlotMemoryState,
    SlotTransactionError,
    budget_token_count,
    parse_transaction_response,
    tail_budget_text,
    validate_construction_output,
)
from vcah.wp17_slot_protocol import WP17_3_MANIFEST_CONTRACT
from vcah.wp17_slot_runner import (
    WP17_EVIDENCE_ALIAS_CONTRACT,
    WP17_OCR_AGGREGATION_CONTRACT,
    alias_current_evidence,
    build_asr_packet,
    build_ocr_packet,
    construction_prompt,
    frame_evidence_ids,
    packet_digest,
)


def run(args: argparse.Namespace) -> Path:
    protocol_path = Path(args.protocol_manifest)
    if file_sha256(protocol_path) != str(args.expected_protocol_sha256):
        raise ValueError("WP17-3 protocol manifest SHA mismatch")
    protocol = _read_json(protocol_path)
    if protocol.get("contract") != WP17_3_MANIFEST_CONTRACT:
        raise ValueError("WP17-3 protocol manifest contract mismatch")
    if not protocol.get("structural_gate_passed"):
        raise RuntimeError("WP17-3 protocol structural gate did not pass")
    if str(args.source_commit) != str(protocol.get("provenance", {}).get("source_commit", "")):
        raise ValueError("WP17-3 runtime source commit mismatch")
    mode = str(args.mode).strip().casefold()
    if mode not in {"canary", "full"}:
        raise ValueError("mode must be canary or full")

    out_root = Path(args.out_root)
    if out_root.exists() and any(out_root.iterdir()) and not args.resume:
        raise FileExistsError(f"WP17 slot output is not empty: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    result_root = out_root / "segments"
    result_root.mkdir(exist_ok=True)

    workspace = VirtualVideoWorkspace.load(Path(args.workspace_root))
    dense_root = Path(args.dense_root)
    evidence_path = dense_root / "evidence_store.jsonl"
    dense_report_path = dense_root / "wp17_dense_ocr_report.json"
    if not evidence_path.is_file() or not dense_report_path.is_file():
        raise FileNotFoundError("WP17 dense OCR evidence/report is incomplete")
    evidence_rows = _read_jsonl(evidence_path)
    asr_cues = workspace.read_asr_virtual_cues()
    client = OpenAICompatibleClient.from_yaml(Path(args.config), section=args.section)
    expected_model = str(protocol["model_policy"]["actual_model"])
    if client.model != expected_model:
        raise ValueError("WP17 slot actual model mismatch")
    max_completion_tokens = max(4096, int(args.max_completion_tokens))
    history_budget = int(protocol["state_policy"]["history_token_budget"])
    preprocessing_policy = dict(protocol["evidence_policy"]["frame_preprocessing"])

    all_segments = tuple(dict(row) for row in protocol["segments"])
    if mode == "canary":
        selected_ids = set(str(value) for value in protocol["canary_segment_chain"])
        segments = tuple(row for row in all_segments if str(row["segment_id"]) in selected_ids)
        hard_cap = int(protocol["counts"]["canary_model_call_hard_cap"])
    else:
        segments = all_segments
        hard_cap = int(protocol["counts"]["model_call_hard_cap"])
    expected_results = len(segments) * int(protocol["counts"]["arms"])
    if not segments:
        raise ValueError("WP17 slot run selected no segments")

    run_manifest = {
        "schema_version": "MMLifelongWP17SlotConstructionRunV3",
        "contract": "WP17-3-slot-construction-run-v3",
        "mode": mode,
        "source_commit": str(args.source_commit),
        "protocol_manifest_sha256": file_sha256(protocol_path),
        "dense_report_sha256": file_sha256(dense_report_path),
        "dense_evidence_sha256": file_sha256(evidence_path),
        "workspace_manifest_sha256": file_sha256(workspace.asset_root / "virtual_timeline.json"),
        "asr_packet_source_sha256": file_sha256(workspace.asr_virtual_cues),
        "config_sha256": file_sha256(Path(args.config)),
        "api_section": str(args.section),
        "actual_model": client.model,
        "max_completion_tokens": max_completion_tokens,
        "response_format": {"type": "json_object"},
        "budget_tokenizer": WP17_BUDGET_TOKENIZER,
        "history_token_budget": history_budget,
        "slot_capsule_contract": WP17_SLOT_CAPSULE_CONTRACT,
        "capsule_provenance_projection_contract": WP17_CAPSULE_PROVENANCE_CONTRACT,
        "image_preprocessing": preprocessing_policy,
        "ocr_aggregation_contract": WP17_OCR_AGGREGATION_CONTRACT,
        "evidence_alias_contract": WP17_EVIDENCE_ALIAS_CONTRACT,
        "output_limits": {
            "max_observations": WP17_MAX_OBSERVATIONS,
            "target_evidence_ids_per_observation": WP17_TARGET_OBSERVATION_EVIDENCE_IDS,
            "max_evidence_ids_per_observation": None,
            "max_structured_event_items_per_field": WP17_MAX_STRUCTURED_EVENT_ITEMS,
            "max_json_chars": WP17_MAX_OUTPUT_JSON_CHARS,
        },
        "segment_count": len(segments),
        "expected_result_count": expected_results,
        "model_call_hard_cap": hard_cap,
        "raw_model_response_persisted": False,
        "source_paths_persisted": False,
        "temporary_frames_retained": False,
        "question_visible": False,
        "options_visible": False,
        "answer_visible": False,
        "official_intervals_visible": False,
        "case_ids_visible": False,
        "day_test140_accessed": False,
        "week_outcomes_accessed": False,
    }
    manifest_path = out_root / "run_manifest.json"
    if manifest_path.is_file():
        prior_manifest = _read_json(manifest_path)
        for key in (
            "mode",
            "source_commit",
            "protocol_manifest_sha256",
            "dense_report_sha256",
            "dense_evidence_sha256",
            "workspace_manifest_sha256",
            "config_sha256",
            "actual_model",
            "max_completion_tokens",
            "ocr_aggregation_contract",
            "evidence_alias_contract",
            "slot_capsule_contract",
            "capsule_provenance_projection_contract",
            "output_limits",
        ):
            if prior_manifest.get(key) != run_manifest.get(key):
                raise RuntimeError(f"WP17 slot resume manifest mismatch: {key}")
    else:
        _write_json_atomic(manifest_path, run_manifest)

    prior_summary = _read_json(out_root / "run_summary.json") if (out_root / "run_summary.json").is_file() else {}
    call_count = int(prior_summary.get("model_calls", 0) or 0)
    results: dict[tuple[str, str], dict[str, Any]] = {}
    slot_state = SlotMemoryState("e1c2", token_budget=history_budget)
    previous_caption = ""
    active_window = ""

    for segment in segments:
        segment_id = str(segment["segment_id"])
        window_id = str(segment["window_id"])
        if window_id != active_window:
            active_window = window_id
            slot_state = SlotMemoryState("e1c2", token_budget=history_budget)
            previous_caption = ""
        start = float(segment["virtual_start_sec"])
        end = float(segment["virtual_end_sec"])
        canonical_ocr_packet = build_ocr_packet(
            evidence_rows,
            segment_id=segment_id,
            start_sec=start,
            end_sec=end,
        )
        canonical_asr_packet = build_asr_packet(
            asr_cues,
            segment_id=segment_id,
            start_sec=start,
            end_sec=end,
        )
        prior_capsule = slot_state.capsule()
        c2_context = str(prior_capsule["context"])
        c2_tokens = int(prior_capsule["token_count"])
        c1_context = tail_budget_text(previous_caption, max_tokens=c2_tokens)
        c1_tokens = budget_token_count(c1_context)
        histories = {
            "e1c0": ("", 0, 0),
            "e1c1": (c1_context, c1_tokens, c2_tokens),
            "e1c2": (c2_context, c2_tokens, history_budget),
        }

        with tempfile.TemporaryDirectory(prefix="wp17-slot-frames-") as frame_dir:
            frames = materialize_caption_frames(
                workspace.manifest,
                start,
                end,
                out_dir=Path(frame_dir),
                fps=float(segment["frame_sampling_fps"]),
                max_frames=int(segment["max_frames"]),
                extraction_mode="fps_batch",
            )
            preprocessing = preprocess_caption_images(
                frames,
                width=int(preprocessing_policy["width"]),
                height=int(preprocessing_policy["height"]),
                jpeg_quality=int(preprocessing_policy["jpeg_quality"]),
            )
            image_paths = tuple(frame.path for frame in frames)
            canonical_frame_ids = frame_evidence_ids(segment_id, len(frames))
            (
                prompt_frame_ids,
                prompt_ocr_packet,
                prompt_asr_packet,
                evidence_alias_map,
            ) = alias_current_evidence(
                canonical_frame_ids,
                canonical_ocr_packet,
                canonical_asr_packet,
            )
            image_labels = tuple(
                f"local_time_sec={float(frame.virtual_time_sec) - start:.3f} evidence_id={evidence_id}"
                for frame, evidence_id in zip(frames, prompt_frame_ids)
            )
            frame_digest = _frame_packet_digest(image_paths)
            input_digests = {
                "frame_packet": frame_digest,
                "ocr_packet": packet_digest(prompt_ocr_packet),
                "asr_packet": packet_digest(prompt_asr_packet),
                "evidence_alias_map": stable_digest(evidence_alias_map),
            }
            prompt_evidence_ids = tuple(evidence_alias_map)
            canonical_evidence_ids = tuple(evidence_alias_map.values())

            frozen_histories = {arm: histories[arm] for arm in histories}
            for arm in tuple(segment["arm_execution_order"]):
                arm = str(arm)
                history, history_tokens, history_limit = frozen_histories[arm]
                result_path = result_root / segment_id / f"{arm}.json"
                history_digest = stable_digest(history)
                if args.resume and result_path.is_file():
                    prior = _read_json(result_path)
                    if prior.get("status") == "success":
                        if prior.get("input_digests") != input_digests or prior.get("history_digest") != history_digest:
                            raise RuntimeError("WP17 slot resume input/history digest mismatch")
                        normalized = validate_construction_output(
                            dict(prior["model_output"]),
                            arm=arm,
                            segment_id=segment_id,
                            allowed_evidence_ids=canonical_evidence_ids,
                            state=slot_state if arm == "e1c2" else None,
                            enforce_output_size=False,
                        )
                        if arm == "e1c2" and normalized.get("state_digest") != prior.get("state_digest"):
                            raise RuntimeError("WP17 slot resume state digest mismatch")
                        if arm == "e1c1":
                            previous_caption = str(
                                normalized["structured_event_record"]["summary"]
                            )
                        results[(segment_id, arm)] = prior
                        _write_summary(out_root, results, expected_results, call_count, hard_cap)
                        continue

                result, consumed = _run_one(
                    client=client,
                    arm=arm,
                    segment_id=segment_id,
                    duration_sec=end - start,
                    image_paths=image_paths,
                    image_labels=image_labels,
                    frame_ids=prompt_frame_ids,
                    ocr_packet=prompt_ocr_packet,
                    asr_packet=prompt_asr_packet,
                    history=history,
                    history_tokens=history_tokens,
                    history_limit=history_limit,
                    input_digests=input_digests,
                    allowed_evidence_ids=prompt_evidence_ids,
                    evidence_id_map=evidence_alias_map,
                    state=slot_state if arm == "e1c2" else None,
                    max_completion_tokens=max_completion_tokens,
                    remaining_calls=hard_cap - call_count,
                )
                call_count += consumed
                result.update(
                    {
                        "window_id": window_id,
                        "segment_ordinal": int(segment["segment_ordinal"]),
                        "window_segment_ordinal": int(segment["window_segment_ordinal"]),
                        "history_digest": history_digest,
                        "history_token_count": history_tokens,
                        "history_token_limit": history_limit,
                        "history_tokenizer": WP17_BUDGET_TOKENIZER,
                        "input_digests": input_digests,
                        "frame_count": len(frames),
                        "ocr_source_evidence_count": sum(
                            int(row.get("source_evidence_count", 0) or 0)
                            for row in canonical_ocr_packet
                        ),
                        "ocr_aggregate_count": len(canonical_ocr_packet),
                        "prompt_evidence_alias_count": len(evidence_alias_map),
                        "image_preprocessing": {
                            key: preprocessing[key]
                            for key in ("width", "height", "jpeg_quality")
                        },
                        "raw_model_response_persisted": False,
                        "temporary_frames_retained": False,
                        "source_paths_persisted": False,
                    }
                )
                _write_json_atomic(result_path, result)
                results[(segment_id, arm)] = result
                if result["status"] != "success":
                    _write_summary(out_root, results, expected_results, call_count, hard_cap)
                    raise RuntimeError(
                        f"WP17 slot arm failed: {segment_id}/{arm}/{result.get('failure_code', 'unknown')}"
                    )
                if arm == "e1c1":
                    previous_caption = str(
                        result["model_output"]["structured_event_record"]["summary"]
                    )
                _write_summary(out_root, results, expected_results, call_count, hard_cap)
                print(
                    "WP17_SLOT_DONE "
                    f"completed={sum(row.get('status') == 'success' for row in results.values())}/{expected_results} "
                    f"segment={segment_id} arm={arm} status=success calls={call_count}/{hard_cap}",
                    flush=True,
                )
    summary_path = _write_summary(out_root, results, expected_results, call_count, hard_cap)
    if sum(row.get("status") == "success" for row in results.values()) != expected_results:
        raise SystemExit(1)
    return summary_path


def _run_one(
    *,
    client: OpenAICompatibleClient,
    arm: str,
    segment_id: str,
    duration_sec: float,
    image_paths: Sequence[str],
    image_labels: Sequence[str],
    frame_ids: Sequence[str],
    ocr_packet: Sequence[Mapping[str, Any]],
    asr_packet: Sequence[Mapping[str, Any]],
    history: str,
    history_tokens: int,
    history_limit: int,
    input_digests: Mapping[str, str],
    allowed_evidence_ids: Sequence[str],
    evidence_id_map: Mapping[str, str],
    state: SlotMemoryState | None,
    max_completion_tokens: int,
    remaining_calls: int,
) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    attempts = []
    repair_error = ""
    consumed = 0
    for attempt_index in range(2):
        if consumed >= remaining_calls:
            return (
                {
                    "schema_version": "MMLifelongWP17SlotConstructionResultV2",
                    "segment_id": segment_id,
                    "arm": arm,
                    "status": "failed",
                    "failure_code": "model_call_hard_cap_exhausted",
                    "attempts": attempts,
                    "duration_sec": round(time.monotonic() - started, 3),
                },
                consumed,
            )
        prompt = construction_prompt(
            arm=arm,
            segment_duration_sec=duration_sec,
            frame_ids=frame_ids,
            ocr_packet=ocr_packet,
            asr_packet=asr_packet,
            history_context=history,
            history_token_count=history_tokens,
            history_token_limit=history_limit,
            repair_error=repair_error,
        )
        prompt_digest = stable_digest(prompt)
        consumed += 1
        try:
            raw = client.chat(
                prompt,
                image_paths=image_paths,
                image_labels=image_labels,
                prompt_position="last",
                max_tokens=max_completion_tokens,
                response_format={"type": "json_object"},
            )
            metadata = safe_response_metadata(client.last_response_metadata)
            extra_calls = int(metadata.get("truncation_retry_count", 0) or 0)
            consumed += extra_calls
        except Exception as exc:
            attempts.append(
                {
                    "attempt_index": attempt_index + 1,
                    "status": "model_failed",
                    "failure_code": type(exc).__name__,
                    "prompt_digest": prompt_digest,
                }
            )
            return (
                {
                    "schema_version": "MMLifelongWP17SlotConstructionResultV2",
                    "segment_id": segment_id,
                    "arm": arm,
                    "status": "failed",
                    "failure_code": "model_failed",
                    "attempts": attempts,
                    "duration_sec": round(time.monotonic() - started, 3),
                },
                consumed,
            )
        parsed = parse_transaction_response(raw)
        if parsed is None:
            validation_error = "response_not_json_object"
        else:
            model_output_json_chars = len(
                json.dumps(parsed, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            )
            try:
                normalized = validate_construction_output(
                    parsed,
                    arm=arm,
                    segment_id=segment_id,
                    allowed_evidence_ids=allowed_evidence_ids,
                    state=state,
                    evidence_id_map=evidence_id_map,
                )
            except SlotTransactionError as exc:
                validation_error = str(exc)
            else:
                attempts.append(
                    {
                        "attempt_index": attempt_index + 1,
                        "status": "success",
                        "prompt_digest": prompt_digest,
                        "response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                        "response_metadata": metadata,
                    }
                )
                return (
                    {
                        "schema_version": "MMLifelongWP17SlotConstructionResultV2",
                        "segment_id": segment_id,
                        "arm": arm,
                        "status": "success",
                        "actual_model": client.model,
                        "input_digests": dict(input_digests),
                        "model_output": {
                            "contract": normalized["contract"],
                            "observations": normalized["observations"],
                            "slot_operations": normalized["slot_operations"],
                            "structured_event_record": normalized[
                                "structured_event_record"
                            ],
                        },
                        "model_output_json_chars": model_output_json_chars,
                        "state_digest": normalized.get("state_digest"),
                        "capsule": normalized.get("capsule"),
                        "lifecycle_events": normalized.get("lifecycle_events", []),
                        "long_term_ledger_count": normalized.get(
                            "long_term_ledger_count", 0
                        ),
                        "attempts": attempts,
                        "attempt_count": len(attempts),
                        "validation_retry_count": attempt_index,
                        "duration_sec": round(time.monotonic() - started, 3),
                    },
                    consumed,
                )
        attempts.append(
            {
                "attempt_index": attempt_index + 1,
                "status": "validation_failed",
                "failure_code": validation_error[:240],
                "prompt_digest": prompt_digest,
                "response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "response_metadata": metadata,
            }
        )
        repair_error = validation_error
    return (
        {
            "schema_version": "MMLifelongWP17SlotConstructionResultV2",
            "segment_id": segment_id,
            "arm": arm,
            "status": "failed",
            "failure_code": "validation_retry_exhausted",
            "attempts": attempts,
            "duration_sec": round(time.monotonic() - started, 3),
        },
        consumed,
    )


def _write_summary(
    out_root: Path,
    results: Mapping[tuple[str, str], Mapping[str, Any]],
    expected_results: int,
    model_calls: int,
    hard_cap: int,
) -> Path:
    statuses: dict[str, int] = {}
    for row in results.values():
        status = str(row.get("status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1
    summary = {
        "schema_version": "MMLifelongWP17SlotConstructionSummaryV2",
        "expected_results": int(expected_results),
        "completed_results": len(results),
        "successes": statuses.get("success", 0),
        "failures": sum(value for key, value in statuses.items() if key != "success"),
        "status_counts": statuses,
        "model_calls": int(model_calls),
        "model_call_hard_cap": int(hard_cap),
        "complete": statuses.get("success", 0) == int(expected_results),
    }
    path = Path(out_root) / "run_summary.json"
    _write_json_atomic(path, summary)
    return path


def _frame_packet_digest(paths: Sequence[str]) -> str:
    digests = [file_sha256(Path(path)) for path in paths]
    return stable_digest(digests)


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


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(path).with_name(f".{Path(path).name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-manifest", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--dense-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--section", default="planner_api")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--mode", choices=("canary", "full"), default="canary")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
