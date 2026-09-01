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
    WP17_SLOT_REPAIR_CONTRACT,
    WP17_TARGET_OBSERVATION_EVIDENCE_IDS,
    SlotMemoryState,
    SlotTransactionError,
    budget_token_count,
    parse_transaction_response,
    tail_budget_text,
    validate_construction_base,
    validate_construction_output,
)
from vcah.wp17_slot_protocol import WP17_3_MANIFEST_CONTRACT
from vcah.wp17_slot_continuation import (
    WP17_SLOT_CONTINUATION_CONTRACT,
    cumulative_experiment_model_calls,
    index_continuation_entries,
)
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
    continuation_values = (
        args.parent_run_root,
        args.continuation_plan,
        args.expected_continuation_plan_sha256,
    )
    if any(continuation_values) and not all(continuation_values):
        raise ValueError("WP17 continuation requires parent root, plan, and expected plan SHA")
    continuation_plan: dict[str, Any] | None = None
    continuation_entries: dict[tuple[str, str], dict[str, Any]] = {}
    continuation_plan_sha = ""
    parent_root: Path | None = None
    parent_manifest_path: Path | None = None
    parent_summary_path: Path | None = None
    parent_summary: dict[str, Any] = {}
    if all(continuation_values):
        parent_root = Path(args.parent_run_root)
        parent_manifest_path = parent_root / "run_manifest.json"
        parent_summary_path = parent_root / "run_summary.json"
        continuation_plan_path = Path(args.continuation_plan)
        continuation_plan_sha = file_sha256(continuation_plan_path)
        if continuation_plan_sha != str(args.expected_continuation_plan_sha256):
            raise ValueError("WP17 continuation plan SHA mismatch")
        continuation_plan = _read_json(continuation_plan_path)
        continuation_entries = index_continuation_entries(continuation_plan)
        continuation_contract = dict(protocol.get("continuation", {}) or {})
        if continuation_contract.get("contract") != WP17_SLOT_CONTINUATION_CONTRACT:
            raise ValueError("WP17 continuation protocol contract mismatch")
        if continuation_contract.get("plan_sha256") != continuation_plan_sha:
            raise ValueError("WP17 continuation protocol/plan mismatch")
        if continuation_plan.get("source_commit") != str(args.source_commit):
            raise ValueError("WP17 continuation plan source commit mismatch")
        if file_sha256(parent_manifest_path) != continuation_plan.get(
            "parent_run_manifest_sha256"
        ):
            raise ValueError("WP17 continuation parent manifest SHA mismatch")
        if file_sha256(parent_summary_path) != continuation_plan.get(
            "parent_run_summary_sha256"
        ):
            raise ValueError("WP17 continuation parent summary SHA mismatch")
        parent_manifest = _read_json(parent_manifest_path)
        if parent_manifest.get("protocol_manifest_sha256") != continuation_plan.get(
            "parent_protocol_sha256"
        ):
            raise ValueError("WP17 continuation parent protocol provenance mismatch")
        parent_summary = _read_json(parent_summary_path)
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
    if continuation_plan is not None:
        expected_continuation_keys = {
            (str(segment["segment_id"]), arm)
            for segment in segments
            for arm in ("e1c0", "e1c1", "e1c2")
        }
        if set(continuation_entries) != expected_continuation_keys:
            raise ValueError("WP17 continuation plan does not cover every result")

    run_manifest = {
        "schema_version": "MMLifelongWP17SlotConstructionRunV4",
        "contract": "WP17-3-slot-construction-run-v4",
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
        "slot_repair_contract": WP17_SLOT_REPAIR_CONTRACT,
        "maximum_attempts_per_result": 3,
        "transaction_abstain_preserves_state": True,
        "transaction_abstain_ser_endpoint_eligible": False,
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
        "model_output_json_chars_contract": (
            "compact_parsed_response_before_evidence_alias_canonicalization"
        ),
        "persisted_model_output_json_chars_contract": (
            "compact_persisted_response_after_evidence_alias_canonicalization"
        ),
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
    if continuation_plan is not None:
        run_manifest["continuation"] = {
            "contract": WP17_SLOT_CONTINUATION_CONTRACT,
            "plan_sha256": continuation_plan_sha,
            "parent_protocol_sha256": continuation_plan.get(
                "parent_protocol_sha256"
            ),
            "parent_run_manifest_sha256": continuation_plan.get(
                "parent_run_manifest_sha256"
            ),
            "parent_run_summary_sha256": continuation_plan.get(
                "parent_run_summary_sha256"
            ),
            "parent_source_commit": continuation_plan.get("parent_source_commit"),
            "parent_model_calls": cumulative_experiment_model_calls(parent_summary),
            "planned_reuse_results": int(
                continuation_plan.get("counts", {}).get("reuse", 0)
            ),
            "planned_rerun_results": int(
                continuation_plan.get("counts", {}).get("rerun", 0)
            ),
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
            "slot_repair_contract",
            "maximum_attempts_per_result",
            "transaction_abstain_preserves_state",
            "transaction_abstain_ser_endpoint_eligible",
            "capsule_provenance_projection_contract",
            "output_limits",
            "continuation",
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
    c1_chain_gap = False
    active_window = ""

    for segment in segments:
        segment_id = str(segment["segment_id"])
        window_id = str(segment["window_id"])
        if window_id != active_window:
            active_window = window_id
            slot_state = SlotMemoryState("e1c2", token_budget=history_budget)
            previous_caption = ""
            c1_chain_gap = False
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
        c1_context = tail_budget_text(previous_caption, max_tokens=history_budget)
        c1_tokens = budget_token_count(c1_context)
        histories = {
            "e1c0": ("", 0, 0),
            "e1c1": (c1_context, c1_tokens, history_budget),
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
                        normalized = _replay_success_row(
                            prior,
                            arm=arm,
                            segment_id=segment_id,
                            input_digests=input_digests,
                            history_digest=history_digest,
                            allowed_evidence_ids=canonical_evidence_ids,
                            state=slot_state if arm == "e1c2" else None,
                        )
                        if arm == "e1c1":
                            previous_caption = str(
                                normalized["structured_event_record"]["summary"]
                            )
                            c1_chain_gap = False
                        results[(segment_id, arm)] = prior
                        _write_summary(
                            out_root,
                            results,
                            expected_results,
                            call_count,
                            hard_cap,
                            continuation=run_manifest.get("continuation"),
                        )
                        continue

                continuation_entry = continuation_entries.get((segment_id, arm))
                if continuation_entry and continuation_entry["action"] == "reuse":
                    assert parent_root is not None
                    parent_path = parent_root / "segments" / segment_id / f"{arm}.json"
                    parent_sha = file_sha256(parent_path)
                    if parent_sha != continuation_entry.get("parent_result_sha256"):
                        raise RuntimeError("WP17 continuation parent result SHA mismatch")
                    parent_row = _read_json(parent_path)
                    normalized = _replay_success_row(
                        parent_row,
                        arm=arm,
                        segment_id=segment_id,
                        input_digests=input_digests,
                        history_digest=history_digest,
                        allowed_evidence_ids=canonical_evidence_ids,
                        state=slot_state if arm == "e1c2" else None,
                    )
                    reused = dict(parent_row)
                    reused["continuation_provenance"] = {
                        "action": "reuse",
                        "plan_sha256": continuation_plan_sha,
                        "parent_result_sha256": parent_sha,
                    }
                    _write_json_atomic(result_path, reused)
                    if arm == "e1c1":
                        previous_caption = str(
                            normalized["structured_event_record"]["summary"]
                        )
                        c1_chain_gap = False
                    results[(segment_id, arm)] = reused
                    _write_summary(
                        out_root,
                        results,
                        expected_results,
                        call_count,
                        hard_cap,
                        continuation=run_manifest.get("continuation"),
                    )
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
                        "history_chain_gap": bool(c1_chain_gap) if arm == "e1c1" else False,
                        "history_degenerate_to_no_context": bool(
                            arm == "e1c1" and c1_chain_gap and not history
                        ),
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
                if continuation_entry:
                    result["continuation_provenance"] = {
                        "action": "rerun",
                        "plan_sha256": continuation_plan_sha,
                        "parent_result_sha256": continuation_entry.get(
                            "parent_result_sha256"
                        ),
                        "reasons": list(continuation_entry.get("reasons", ())),
                    }
                _write_json_atomic(result_path, result)
                results[(segment_id, arm)] = result
                if arm == "e1c1":
                    if result["status"] == "success":
                        previous_caption = str(
                            result["model_output"]["structured_event_record"]["summary"]
                        )
                        c1_chain_gap = False
                    else:
                        previous_caption = ""
                        c1_chain_gap = True
                _write_summary(
                    out_root,
                    results,
                    expected_results,
                    call_count,
                    hard_cap,
                    continuation=run_manifest.get("continuation"),
                )
                print(
                    "WP17_SLOT_DONE "
                    f"completed={sum(row.get('status') == 'success' for row in results.values())}/{expected_results} "
                    f"segment={segment_id} arm={arm} status={result['status']} calls={call_count}/{hard_cap}",
                    flush=True,
                )
    summary_path = _write_summary(
        out_root,
        results,
        expected_results,
        call_count,
        hard_cap,
        continuation=run_manifest.get("continuation"),
    )
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
    attempts: list[dict[str, Any]] = []
    repair_contract: dict[str, Any] | None = None
    repair_mode = "base"
    abstain_base: dict[str, Any] | None = None
    abstain_model_output_json_chars = 0
    illegal_operation_contracts: list[dict[str, Any]] = []
    consumed = 0
    for attempt_index in range(3):
        if consumed >= remaining_calls:
            break
        prompt = construction_prompt(
            arm=arm,
            segment_duration_sec=duration_sec,
            frame_ids=frame_ids,
            ocr_packet=ocr_packet,
            asr_packet=asr_packet,
            history_context=history,
            history_token_count=history_tokens,
            history_token_limit=history_limit,
            repair_contract=repair_contract,
            repair_mode=repair_mode,
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
            if arm == "e1c2" and abstain_base is not None:
                break
            return (
                {
                    "schema_version": "MMLifelongWP17SlotConstructionResultV3",
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
            finish_reason = str(metadata.get("finish_reason", "") or "")
            failure_code = (
                "response_truncated" if finish_reason == "length" else "response_malformed"
            )
            attempts.append(
                {
                    "attempt_index": attempt_index + 1,
                    "status": "validation_failed",
                    "failure_code": failure_code,
                    "prompt_digest": prompt_digest,
                    "response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                    "response_metadata": metadata,
                }
            )
            repair_mode = "serialization"
            continue
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
                failure_code = exc.code
                semantic_repair = exc.repair_contract()
                try:
                    base = validate_construction_base(
                        parsed,
                        allowed_evidence_ids=allowed_evidence_ids,
                        evidence_id_map=evidence_id_map,
                    )
                except SlotTransactionError:
                    base = None
                if arm == "e1c2" and base is not None:
                    abstain_base = base
                    abstain_model_output_json_chars = model_output_json_chars
                    illegal_operation_contracts.append(semantic_repair)
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
                        "schema_version": "MMLifelongWP17SlotConstructionResultV3",
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
                        "slot_transaction_abstained": False,
                        "ser_endpoint_eligible": True,
                        "ser_trust_status": "trusted",
                        "illegal_operation_count": len(illegal_operation_contracts),
                        "duration_sec": round(time.monotonic() - started, 3),
                    },
                    consumed,
                )
        attempts.append(
            {
                "attempt_index": attempt_index + 1,
                "status": "validation_failed",
                "failure_code": failure_code,
                "repair_contract": semantic_repair,
                "prompt_digest": prompt_digest,
                "response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "response_metadata": metadata,
            }
        )
        repair_contract = semantic_repair
        repair_mode = "semantic"
        if attempt_index > 0 and abstain_base is not None:
            break
    if arm == "e1c2" and state is not None and abstain_base is not None:
        abstained_model_output = {
            "contract": abstain_base["contract"],
            "observations": abstain_base["observations"],
            "slot_operations": [],
            "structured_event_record": abstain_base["structured_event_record"],
        }
        abstain_event = {
            "event": "slot_transaction_abstain",
            "segment_id": segment_id,
            "operation": "transaction_abstain",
            "state_changed": False,
            "ser_endpoint_eligible": False,
        }
        return (
            {
                "schema_version": "MMLifelongWP17SlotConstructionResultV3",
                "segment_id": segment_id,
                "arm": arm,
                "status": "success",
                "actual_model": client.model,
                "input_digests": dict(input_digests),
                "model_output": abstained_model_output,
                "model_output_json_chars": abstain_model_output_json_chars,
                "persisted_model_output_json_chars": len(
                    json.dumps(
                        abstained_model_output,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                ),
                "state_digest": state.digest(),
                "capsule": state.capsule(),
                "lifecycle_events": [abstain_event],
                "long_term_ledger_count": len(state.ledger),
                "attempts": attempts,
                "attempt_count": len(attempts),
                "validation_retry_count": max(0, len(attempts) - 1),
                "slot_transaction_abstained": True,
                "ser_endpoint_eligible": False,
                "ser_trust_status": "untrusted_for_endpoint",
                "illegal_operation_count": len(illegal_operation_contracts),
                "illegal_operation_contracts": illegal_operation_contracts,
                "duration_sec": round(time.monotonic() - started, 3),
            },
            consumed,
        )
    terminal_code = (
        "model_call_hard_cap_exhausted"
        if consumed >= remaining_calls
        else "validation_retry_exhausted"
    )
    return (
        {
            "schema_version": "MMLifelongWP17SlotConstructionResultV3",
            "segment_id": segment_id,
            "arm": arm,
            "status": "failed",
            "failure_code": terminal_code,
            "attempts": attempts,
            "illegal_operation_count": len(illegal_operation_contracts),
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
    *,
    continuation: Mapping[str, Any] | None = None,
) -> Path:
    statuses: dict[str, int] = {}
    for row in results.values():
        status = str(row.get("status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1
    abstentions = sum(
        row.get("slot_transaction_abstained") is True for row in results.values()
    )
    ser_ineligible = sum(
        row.get("ser_endpoint_eligible") is False for row in results.values()
    )
    illegal_operations = sum(
        int(row.get("illegal_operation_count", 0) or 0) for row in results.values()
    )
    summary = {
        "schema_version": "MMLifelongWP17SlotConstructionSummaryV3",
        "expected_results": int(expected_results),
        "completed_results": len(results),
        "successes": statuses.get("success", 0),
        "failures": sum(value for key, value in statuses.items() if key != "success"),
        "status_counts": statuses,
        "model_calls": int(model_calls),
        "model_call_hard_cap": int(hard_cap),
        "slot_transaction_abstentions": abstentions,
        "ser_endpoint_ineligible": ser_ineligible,
        "illegal_operation_attempts": illegal_operations,
        "complete": statuses.get("success", 0) == int(expected_results),
    }
    if continuation is not None:
        reused = sum(
            row.get("continuation_provenance", {}).get("action") == "reuse"
            for row in results.values()
        )
        rerun = sum(
            row.get("continuation_provenance", {}).get("action") == "rerun"
            for row in results.values()
        )
        parent_calls = int(continuation.get("parent_model_calls", 0) or 0)
        summary["continuation"] = {
            "contract": WP17_SLOT_CONTINUATION_CONTRACT,
            "plan_sha256": continuation.get("plan_sha256"),
            "reused_results": reused,
            "rerun_results": rerun,
            "planned_reuse_results": int(
                continuation.get("planned_reuse_results", 0) or 0
            ),
            "planned_rerun_results": int(
                continuation.get("planned_rerun_results", 0) or 0
            ),
            "parent_model_calls": parent_calls,
            "continuation_model_calls": int(model_calls),
            "total_experiment_model_calls": parent_calls + int(model_calls),
        }
    path = Path(out_root) / "run_summary.json"
    _write_json_atomic(path, summary)
    return path


def _replay_success_row(
    prior: Mapping[str, Any],
    *,
    arm: str,
    segment_id: str,
    input_digests: Mapping[str, str],
    history_digest: str,
    allowed_evidence_ids: Sequence[str],
    state: SlotMemoryState | None,
) -> dict[str, Any]:
    if prior.get("status") != "success":
        raise RuntimeError("WP17 continuation cannot reuse a non-success result")
    if (
        prior.get("input_digests") != dict(input_digests)
        or prior.get("history_digest") != history_digest
    ):
        raise RuntimeError("WP17 slot replay input/history digest mismatch")
    if prior.get("slot_transaction_abstained") is True:
        normalized = validate_construction_base(
            dict(prior["model_output"]),
            allowed_evidence_ids=allowed_evidence_ids,
            enforce_output_size=False,
        )
        if arm != "e1c2" or state is None:
            raise RuntimeError("WP17 transaction abstain is only valid for E1C2")
        normalized["state_digest"] = state.digest()
    else:
        normalized = validate_construction_output(
            dict(prior["model_output"]),
            arm=arm,
            segment_id=segment_id,
            allowed_evidence_ids=allowed_evidence_ids,
            state=state,
            enforce_output_size=False,
        )
    if arm == "e1c2" and normalized.get("state_digest") != prior.get("state_digest"):
        raise RuntimeError("WP17 slot replay state digest mismatch")
    return dict(normalized)


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
    parser.add_argument("--parent-run-root")
    parser.add_argument("--continuation-plan")
    parser.add_argument("--expected-continuation-plan-sha256")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
