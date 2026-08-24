#!/usr/bin/env python3
"""Build a blind, question-independent OCR entity sidecar over Caption passages."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from vcah.caption_lexical_index import CaptionLexicalIndex
from vcah.caption_schema import CaptionPassageV1
from vcah.model_client import OpenAICompatibleClient
from vcah.occurrence_entity_sidecar import (
    GLOBAL_ENTITY_SIDECAR_CONTRACT,
    admitted_entity_row_valid,
    admit_global_entity_rows,
    fixed3_passage_targets,
    global_entity_duplicate_stats,
    global_entity_ocr_prompt,
    parse_global_entity_ocr_response_diagnostic,
    select_hashed_passages,
)
from vcah.occurrence_negative_sidecar import (
    file_sha256,
    safe_response_metadata,
    stable_digest,
)
from vcah.virtual_video import VirtualVideoWorkspace, materialize_window_frames


MAX_WORKERS = 16


def run(args: argparse.Namespace) -> Path:
    protocol_path = Path(args.protocol_spec)
    _require_sha(protocol_path, args.expected_protocol_sha256, "protocol spec")
    protocol = _read_json(protocol_path)
    if protocol.get("contract") != GLOBAL_ENTITY_SIDECAR_CONTRACT:
        raise ValueError("global entity protocol contract mismatch")
    if not bool(protocol.get("protocol_frozen_before_outcomes")):
        raise ValueError("global entity protocol was not frozen before outcomes")

    workspace = VirtualVideoWorkspace.load(Path(args.workspace_root))
    lexical = CaptionLexicalIndex.from_asset_root(
        workspace.asset_root,
        config_digest=str(args.caption_config_digest),
    )
    if len(lexical.passages) != int(args.expected_passages):
        raise ValueError(
            f"expected {args.expected_passages} passages, found {len(lexical.passages)}"
        )
    selected = _selected_passages(lexical.passages, args=args, protocol=protocol)
    if len(selected) != int(args.expected_selected_passages):
        raise ValueError(
            f"expected {args.expected_selected_passages} selected passages, "
            f"found {len(selected)}"
        )

    out_root = Path(args.out_root)
    if out_root.exists() and any(out_root.iterdir()) and not args.resume:
        raise FileExistsError(f"global entity output is not empty: {out_root}")
    (out_root / "results").mkdir(parents=True, exist_ok=True)
    (out_root / "frames").mkdir(parents=True, exist_ok=True)

    probe_client = OpenAICompatibleClient.from_yaml(
        Path(args.config), section=str(args.section)
    )
    if str(probe_client.model) != str(args.expected_model):
        raise ValueError(
            f"actual model mismatch: {probe_client.model} != {args.expected_model}"
        )
    thread_state = threading.local()

    def client_for_worker() -> OpenAICompatibleClient:
        client = getattr(thread_state, "client", None)
        if client is None:
            client = OpenAICompatibleClient.from_yaml(
                Path(args.config), section=str(args.section)
            )
            if str(client.model) != str(args.expected_model):
                raise ValueError("worker model mismatch")
            thread_state.client = client
        return client

    selection = dict(protocol.get("sampling", {}).get("canary_selection", {}))
    run_manifest = {
        "schema_version": "MMLifelongGlobalEntityOCRRunV1",
        "contract": GLOBAL_ENTITY_SIDECAR_CONTRACT,
        "source_commit": str(args.source_commit),
        "protocol_spec_sha256": file_sha256(protocol_path),
        "caption_config_digest": str(args.caption_config_digest),
        "caption_index_digest": lexical.index_digest,
        "caption_passage_count": len(lexical.passages),
        "selection_mode": str(args.selection_mode),
        "selection_seed": str(selection.get("seed", "")),
        "selected_passage_count": len(selected),
        "selected_passage_digest": stable_digest(
            [passage.passage_id for passage in selected]
        ),
        "sampling_strategy": "fixed_3_per_caption_passage_v1",
        "actual_model": str(probe_client.model),
        "config_sha256": file_sha256(Path(args.config)),
        "api_section": str(args.section),
        "workers": max(1, min(MAX_WORKERS, int(args.workers))),
        "max_completion_tokens": max(4096, int(args.max_completion_tokens)),
        "question_visible_to_model": False,
        "options_visible_to_model": False,
        "answer_visible_to_model": False,
        "official_intervals_visible_to_model": False,
        "caption_text_visible_to_model": False,
        "raw_response_persisted": False,
        "prompt_persisted": False,
        "day_test140_accessed": False,
        "week_accessed": False,
    }
    _write_json_atomic(out_root / "run_manifest.json", run_manifest)

    results: dict[str, dict[str, Any]] = {}
    workers = max(1, min(MAX_WORKERS, int(args.workers), len(selected)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_passage,
                passage,
                workspace=workspace,
                out_root=out_root,
                client_for_worker=client_for_worker,
                args=args,
            ): passage.passage_id
            for passage in selected
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            passage_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "schema_version": "MMLifelongGlobalEntityOCRPassageV1",
                    "contract": GLOBAL_ENTITY_SIDECAR_CONTRACT,
                    "passage_id": passage_id,
                    "status": "orchestrator_failed",
                    "error_type": type(exc).__name__,
                    "admitted_entities": [],
                }
            results[passage_id] = result
            if completed == len(selected) or completed % 25 == 0:
                summary = _progress_summary(results, expected=len(selected))
                _write_json_atomic(out_root / "batch_summary.json", summary)
                print(
                    "GLOBAL_ENTITY_PROGRESS "
                    f"completed={summary['completed_count']}/{summary['selected_count']} "
                    f"success={summary['success_count']} "
                    f"failed={summary['failure_count']} "
                    f"entities={summary['admitted_entity_count']}",
                    flush=True,
                )
    progress = _progress_summary(results, expected=len(selected))
    _write_json_atomic(out_root / "batch_summary.json", progress)
    if progress["failure_count"]:
        raise SystemExit(1)

    report = _finalize_sidecar(
        selected,
        results,
        run_manifest=run_manifest,
        out_root=out_root,
        protocol=protocol,
    )
    report_path = out_root / "global_entity_report.json"
    _write_json_atomic(report_path, report)
    (out_root / "global_entity_report.md").write_text(
        _render_markdown(report), encoding="utf-8"
    )
    print(
        "GLOBAL_ENTITY_DONE "
        f"passages={report['counts']['successful_passages']} "
        f"frames={report['counts']['sampled_frames']} "
        f"entities={report['counts']['admitted_entities']} "
        f"gate={str(report['gates']['structural_gate_passed']).lower()}",
        flush=True,
    )
    return report_path


def _selected_passages(
    passages: Sequence[CaptionPassageV1],
    *,
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
) -> tuple[CaptionPassageV1, ...]:
    mode = str(args.selection_mode)
    if mode == "full":
        return tuple(passages)
    if mode != "hash20":
        raise ValueError("selection mode must be hash20 or full")
    canary = dict(protocol.get("sampling", {}).get("canary_selection", {}))
    count = int(canary.get("exact_passage_count", 0) or 0)
    return select_hashed_passages(
        passages,
        seed=str(canary.get("seed", "")),
        count=count,
    )


def _run_passage(
    passage: CaptionPassageV1,
    *,
    workspace: VirtualVideoWorkspace,
    out_root: Path,
    client_for_worker: Callable[[], OpenAICompatibleClient],
    args: argparse.Namespace,
) -> dict[str, Any]:
    key = hashlib.sha256(passage.passage_id.encode("utf-8")).hexdigest()
    result_path = out_root / "results" / key[:2] / f"{key}.json"
    frame_manifest_path = out_root / "frames" / key / "frame_manifest.json"
    frames = _load_reusable_frames(frame_manifest_path) if args.resume else ()
    if not frames:
        frames = _materialize_fixed3_frames(
            workspace,
            passage,
            frame_root=out_root / "frames" / key,
        )
        _write_json_atomic(
            frame_manifest_path,
            {
                "schema_version": "MMLifelongGlobalEntityOCRFramesV1",
                "contract": GLOBAL_ENTITY_SIDECAR_CONTRACT,
                "passage_id": passage.passage_id,
                "virtual_start_sec": passage.virtual_start_sec,
                "virtual_end_sec": passage.virtual_end_sec,
                "frames": list(frames),
            },
        )
    labels = tuple(str(frame["frame_label"]) for frame in frames)
    paths = tuple(str(frame["path"]) for frame in frames)
    prompt = global_entity_ocr_prompt(labels)
    prompt_digest = stable_digest(prompt)
    frame_digest = stable_digest(
        [
            {
                "frame_label": frame["frame_label"],
                "virtual_time_sec": frame["virtual_time_sec"],
                "path_sha256": file_sha256(Path(frame["path"])),
            }
            for frame in frames
        ]
    )
    prior = _read_json(result_path) if args.resume and result_path.is_file() else None
    if (
        prior is not None
        and prior.get("status") == "success"
        and prior.get("actual_model") == str(args.expected_model)
        and prior.get("prompt_digest") == prompt_digest
        and prior.get("frame_digest") == frame_digest
    ):
        return {**prior, "resume_reused_success": True}

    client = client_for_worker()
    started = time.monotonic()
    attempts: list[dict[str, Any]] = []
    parsed: tuple[dict[str, Any], ...] | None = None
    parse_diagnostic: dict[str, Any] = {}
    raw = ""
    response_metadata: dict[str, Any] = {}
    for parse_attempt in range(2):
        call_prompt = prompt
        if parse_attempt:
            call_prompt += (
                "\nThe previous response violated the schema. Emit each allowed frame_label "
                "exactly once and return JSON only."
            )
        try:
            raw = client.chat(
                call_prompt,
                image_paths=paths,
                image_labels=tuple(
                    f"{label} (Fixed-3 sample {index + 1})"
                    for index, label in enumerate(labels)
                ),
                max_tokens=max(4096, int(args.max_completion_tokens)),
                response_format={"type": "json_object"},
            )
            response_metadata = safe_response_metadata(client.last_response_metadata)
        except Exception as exc:
            attempts.append(
                {
                    "attempt_index": parse_attempt + 1,
                    "status": "model_failed",
                    "error_type": type(exc).__name__,
                }
            )
            break
        parse_diagnostic = parse_global_entity_ocr_response_diagnostic(
            raw,
            allowed_frame_labels=labels,
        )
        raw_rows = parse_diagnostic.get("rows")
        parsed = tuple(raw_rows) if isinstance(raw_rows, Sequence) else None
        attempts.append(
            {
                "attempt_index": parse_attempt + 1,
                "status": "success" if parsed is not None else "invalid_json",
                "parse_status": str(parse_diagnostic.get("status", "invalid")),
                "normalization_counts": dict(
                    parse_diagnostic.get("normalization_counts", {})
                ),
                "model_response_digest": stable_digest(raw),
                "response_metadata": response_metadata,
            }
        )
        if parsed is not None:
            break
    frame_by_label = {str(frame["frame_label"]): frame for frame in frames}
    admission = admit_global_entity_rows(
        parsed or (),
        passage_id=passage.passage_id,
        frame_metadata=frame_by_label,
    )
    result = {
        "schema_version": "MMLifelongGlobalEntityOCRPassageV1",
        "contract": GLOBAL_ENTITY_SIDECAR_CONTRACT,
        "passage_id": passage.passage_id,
        "caption_id": passage.caption_id,
        "virtual_start_sec": passage.virtual_start_sec,
        "virtual_end_sec": passage.virtual_end_sec,
        "status": "success" if parsed is not None else attempts[-1]["status"],
        "actual_model": str(client.model),
        "frame_count": len(frames),
        "frame_digest": frame_digest,
        "prompt_digest": prompt_digest,
        "model_response_digest": stable_digest(raw),
        "parsed_entity_candidate_count": len(parsed or ()),
        "candidate_unique_text_count": int(admission["candidate_unique_text_count"]),
        "admitted_entities": list(admission["admitted_rows"]),
        "admitted_entity_count": len(admission["admitted_rows"]),
        "rejection_counts": dict(admission["rejection_counts"]),
        "parse_status": str(parse_diagnostic.get("status", "invalid")),
        "normalization_counts": dict(
            parse_diagnostic.get("normalization_counts", {})
        ),
        "attempt_count": len(attempts),
        "attempt_history": attempts,
        "response_metadata": response_metadata,
        "duration_sec": round(time.monotonic() - started, 3),
        "resume_reused_success": False,
        "question_visible_to_model": False,
        "options_visible_to_model": False,
        "answer_visible_to_model": False,
        "official_intervals_visible_to_model": False,
        "caption_text_visible_to_model": False,
        "raw_response_persisted": False,
        "prompt_persisted": False,
    }
    _write_json_atomic(result_path, result)
    return result


def _materialize_fixed3_frames(
    workspace: VirtualVideoWorkspace,
    passage: CaptionPassageV1,
    *,
    frame_root: Path,
) -> tuple[dict[str, Any], ...]:
    materialization_workspace = replace(
        workspace,
        root_dir=frame_root,
        frame_manifest=frame_root / "frame_manifest.jsonl",
    )
    rows: list[dict[str, Any]] = []
    for index, target in enumerate(fixed3_passage_targets(passage), start=1):
        virtual_time = float(target["virtual_time_sec"])
        sampled = materialize_window_frames(
            materialization_workspace,
            virtual_time,
            virtual_time + 0.001,
            query_id=f"fixed3_{index:02d}",
            fps=0.5,
            max_frames=1,
        )
        frame = sampled[0]
        rows.append(
            {
                "frame_label": f"frame_{index:02d}",
                "frame_id": frame.frame_id,
                "path": str(Path(frame.path).resolve()),
                "virtual_time_sec": frame.virtual_time_sec,
                "requested_virtual_time_sec": virtual_time,
                "sample_positions": list(target["sample_positions"]),
                "segment_id": frame.segment_id,
                "source_video_id": frame.source_video_id,
                "source_time_sec": frame.source_time_sec,
            }
        )
    return tuple(rows)


def _load_reusable_frames(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        return ()
    payload = _read_json(path)
    frames = tuple(
        dict(row)
        for row in tuple(payload.get("frames", ()) or ())
        if isinstance(row, Mapping) and Path(str(row.get("path", ""))).is_file()
    )
    return frames if len(frames) == len(tuple(payload.get("frames", ()) or ())) else ()


def _progress_summary(
    results: Mapping[str, Mapping[str, Any]], *, expected: int
) -> dict[str, Any]:
    success = sum(row.get("status") == "success" for row in results.values())
    return {
        "schema_version": "MMLifelongGlobalEntityOCRBatchSummaryV1",
        "contract": GLOBAL_ENTITY_SIDECAR_CONTRACT,
        "selected_count": int(expected),
        "completed_count": len(results),
        "success_count": success,
        "failure_count": len(results) - success,
        "answer_count": 0,
        "admitted_entity_count": sum(
            int(row.get("admitted_entity_count", 0) or 0) for row in results.values()
        ),
    }


def _finalize_sidecar(
    selected: Sequence[CaptionPassageV1],
    results: Mapping[str, Mapping[str, Any]],
    *,
    run_manifest: Mapping[str, Any],
    out_root: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    ordered_results = tuple(results[passage.passage_id] for passage in selected)
    entity_rows = tuple(
        dict(entity)
        for result in ordered_results
        for entity in tuple(result.get("admitted_entities", ()) or ())
        if isinstance(entity, Mapping)
    )
    _write_jsonl_atomic(out_root / "entity_sidecar.jsonl", entity_rows)
    rejection_counts: Counter[str] = Counter()
    normalization_counts: Counter[str] = Counter()
    for result in ordered_results:
        rejection_counts.update(dict(result.get("rejection_counts", {}) or {}))
        normalization_counts.update(dict(result.get("normalization_counts", {}) or {}))
    durations = [float(result.get("duration_sec", 0.0) or 0.0) for result in ordered_results]
    entity_counts = [
        int(result.get("admitted_entity_count", 0) or 0) for result in ordered_results
    ]
    duplicate_stats = global_entity_duplicate_stats(entity_rows)
    selected_digest = stable_digest([passage.passage_id for passage in selected])
    expected_digest = str(run_manifest.get("selected_passage_digest", ""))
    all_valid = all(admitted_entity_row_valid(row) for row in entity_rows)
    full_success = all(result.get("status") == "success" for result in ordered_results)
    actual_models = {str(result.get("actual_model", "")) for result in ordered_results}
    fixed3_valid = all(
        int(result.get("frame_count", 0) or 0)
        == len(fixed3_passage_targets(passage))
        for passage, result in zip(selected, ordered_results)
    )
    blind_valid = all(
        result.get(key) is False
        for result in ordered_results
        for key in (
            "question_visible_to_model",
            "options_visible_to_model",
            "answer_visible_to_model",
            "official_intervals_visible_to_model",
            "caption_text_visible_to_model",
            "raw_response_persisted",
            "prompt_persisted",
        )
    )
    gates = {
        "selection_is_exact": len(selected)
        == int(run_manifest.get("selected_passage_count", 0) or 0),
        "selection_digest_matches": selected_digest == expected_digest,
        "all_selected_passages_complete": full_success,
        "actual_model_matches": actual_models
        == {str(run_manifest.get("actual_model", ""))},
        "fixed3_lineage_valid": fixed3_valid,
        "blind_prompt_contract_valid": blind_valid,
        "all_results_parse": all(
            result.get("parse_status") == "success" for result in ordered_results
        ),
        "all_admitted_rows_satisfy_runtime_policy": all_valid,
        "zero_numeric_only_or_blocked_admissions": all_valid,
    }
    gates["structural_gate_passed"] = all(gates.values())
    counts = {
        "selected_passages": len(selected),
        "successful_passages": sum(
            result.get("status") == "success" for result in ordered_results
        ),
        "sampled_frames": sum(
            int(result.get("frame_count", 0) or 0) for result in ordered_results
        ),
        "model_calls": sum(
            int(result.get("attempt_count", 0) or 0) for result in ordered_results
        ),
        "parse_retries": sum(
            max(0, int(result.get("attempt_count", 0) or 0) - 1)
            for result in ordered_results
        ),
        "passages_with_entities": sum(value > 0 for value in entity_counts),
        "admitted_entities": len(entity_rows),
        "parsed_entity_candidates": sum(
            int(result.get("parsed_entity_candidate_count", 0) or 0)
            for result in ordered_results
        ),
    }
    density = {
        "mean_admitted_entities_per_passage": (
            statistics.fmean(entity_counts) if entity_counts else 0.0
        ),
        "p95_admitted_entities_per_passage": _quantile(entity_counts, 0.95),
        "maximum_admitted_entities_per_passage": max(entity_counts, default=0),
        "mean_model_duration_sec": statistics.fmean(durations) if durations else 0.0,
        "p95_model_duration_sec": _quantile(durations, 0.95),
    }
    return {
        "schema_version": "MMLifelongGlobalEntityOCRReportV1",
        "contract": GLOBAL_ENTITY_SIDECAR_CONTRACT,
        "decision": (
            "GLOBAL_ENTITY_SIDECAR_READY"
            if gates["structural_gate_passed"]
            else "STRUCTURAL_FAILURE"
        ),
        "selection_mode": run_manifest["selection_mode"],
        "counts": counts,
        "density": density,
        "duplicate_stats": duplicate_stats,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "normalization_counts": dict(sorted(normalization_counts.items())),
        "gates": gates,
        "provenance": dict(run_manifest),
        "protocol_decision_values_are_not_structural_gates": True,
        "endpoint_values_available_only_after_frozen10_evaluation": True,
        "protocol_digest": stable_digest(protocol),
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    counts = report["counts"]
    density = report["density"]
    duplicate = report["duplicate_stats"]
    return "\n".join(
        [
            "# WP16-6A Global OCR Entity Sidecar",
            "",
            f"- Decision: `{report['decision']}`",
            f"- Selection: `{report['selection_mode']}`",
            f"- Passages: `{counts['successful_passages']}/{counts['selected_passages']}`",
            f"- Frames: `{counts['sampled_frames']}`",
            f"- Model calls / retries: `{counts['model_calls']} / {counts['parse_retries']}`",
            f"- Admitted entities: `{counts['admitted_entities']}` in `{counts['passages_with_entities']}` passages",
            f"- Mean / P95 entities per passage: `{density['mean_admitted_entities_per_passage']:.3f} / {density['p95_admitted_entities_per_passage']:.3f}`",
            f"- Duplicate entity text rate: `{duplicate['duplicate_entity_rate']:.4f}`",
            f"- Structural gate: `{str(report['gates']['structural_gate_passed']).lower()}`",
            "",
            "Endpoint values are not structural gates. This extraction report does not use questions, answers, official intervals, bounded search, QA, or a judge.",
            "",
        ]
    )


def _quantile(values: Sequence[float | int], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = round(max(0.0, min(1.0, float(fraction))) * (len(ordered) - 1))
    return ordered[index]


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(target)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _require_sha(path: Path, expected: str, label: str) -> None:
    if file_sha256(path) != str(expected):
        raise ValueError(f"{label} SHA256 mismatch")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--caption-config-digest", required=True)
    parser.add_argument("--expected-passages", type=int, default=2960)
    parser.add_argument("--protocol-spec", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--selection-mode", choices=("hash20", "full"), required=True)
    parser.add_argument("--expected-selected-passages", type=int, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--section", default="planner_api")
    parser.add_argument("--expected-model", default="pa/gmn-2.5-pr")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
