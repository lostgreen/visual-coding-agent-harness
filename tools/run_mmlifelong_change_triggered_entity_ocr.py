#!/usr/bin/env python3
"""Run blind entity OCR on exact-budget WP16-7 frame selections."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import threading
import time
from typing import Any, Callable, Mapping, Sequence, TypeVar

from vcah.change_triggered_entity_occurrence import (
    CHANGE_TRIGGERED_ENTITY_CONTRACT,
    admit_entity_occurrences,
    write_jsonl,
)
from vcah.model_client import OpenAICompatibleClient
from vcah.occurrence_entity_sidecar import (
    MAX_GLOBAL_ENTITY_ROWS_PER_FRAME,
    admitted_entity_row_valid,
    global_entity_ocr_prompt,
    parse_global_entity_ocr_response_diagnostic,
)
from vcah.occurrence_negative_sidecar import (
    file_sha256,
    safe_response_metadata,
    stable_digest,
)
from vcah.virtual_video import VirtualVideoSegment, VirtualVideoWorkspace


MAX_WORKERS = 16
ALLOWED_ARMS = ("a1_uniform", "a2_change")
T = TypeVar("T")
R = TypeVar("R")


def run(args: argparse.Namespace) -> Path:
    protocol_path = Path(args.protocol_spec)
    _require_sha(protocol_path, args.expected_protocol_sha256, "protocol")
    protocol = _read_json(protocol_path)
    if protocol.get("contract") != CHANGE_TRIGGERED_ENTITY_CONTRACT:
        raise ValueError("WP16-7 protocol contract mismatch")
    arm = str(args.arm)
    if arm not in ALLOWED_ARMS:
        raise ValueError("arm must be a1_uniform or a2_change")

    sampling_root = Path(args.sampling_root)
    sampling_report = _read_json(sampling_root / "sampling_report.json")
    tier0_manifest = _read_json(sampling_root / "tier0_manifest.json")
    if not bool(sampling_report.get("gates", {}).get("structural_gate_passed")):
        raise ValueError("sampling structural gate did not pass")
    if tier0_manifest.get("protocol_sha256") != str(args.expected_protocol_sha256):
        raise ValueError("sampling protocol SHA mismatch")
    selection_path = sampling_root / "selections" / f"{arm}.jsonl"
    selection = _read_jsonl(selection_path)
    if len(selection) != int(args.expected_selected_frames):
        raise ValueError("selected frame count mismatch")
    if any(row.get("selection_arm") != arm for row in selection):
        raise ValueError("selection manifest arm mismatch")

    workspace = VirtualVideoWorkspace.load(Path(args.workspace_root))
    segments = {segment.segment_id: segment for segment in workspace.manifest.segments}
    unknown_segments = sorted(
        {str(row.get("segment_id", "")) for row in selection} - set(segments)
    )
    if unknown_segments:
        raise ValueError(f"selection contains unknown segments: {unknown_segments}")

    out_root = Path(args.out_root)
    if out_root.exists() and any(out_root.iterdir()) and not args.resume:
        raise FileExistsError(f"OCR output is not empty: {out_root}")
    result_root = out_root / "batch_results"
    temp_root = out_root / "temporary_frames"
    result_root.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)

    probe_client = OpenAICompatibleClient.from_yaml(
        Path(args.config), section=str(args.section)
    )
    if str(probe_client.model) != str(args.expected_model):
        raise ValueError("actual OCR model mismatch")
    thread_state = threading.local()

    def client_for_worker() -> OpenAICompatibleClient:
        client = getattr(thread_state, "client", None)
        if client is None:
            client = OpenAICompatibleClient.from_yaml(
                Path(args.config), section=str(args.section)
            )
            if str(client.model) != str(args.expected_model):
                raise ValueError("worker OCR model mismatch")
            thread_state.client = client
        return client

    run_manifest = {
        "schema_version": "MMLifelongChangeTriggeredEntityOCRRunV1",
        "contract": CHANGE_TRIGGERED_ENTITY_CONTRACT,
        "source_commit": str(args.source_commit),
        "arm": arm,
        "protocol_path": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "sampling_root": str(sampling_root),
        "sampling_report_sha256": file_sha256(sampling_root / "sampling_report.json"),
        "tier0_manifest_sha256": file_sha256(sampling_root / "tier0_manifest.json"),
        "selection_path": str(selection_path),
        "selection_sha256": file_sha256(selection_path),
        "selected_frame_count": len(selection),
        "actual_model": str(probe_client.model),
        "config_sha256": file_sha256(Path(args.config)),
        "api_section": str(args.section),
        "workers": max(1, min(MAX_WORKERS, int(args.workers))),
        "batch_size": max(1, min(16, int(args.batch_size))),
        "max_completion_tokens": max(4096, int(args.max_completion_tokens)),
        "occurrence_gap_sec": float(args.occurrence_gap_sec),
        "ffmpeg_executable": str(args.ffmpeg_executable),
        "question_visible_to_model": False,
        "options_visible_to_model": False,
        "answer_visible_to_model": False,
        "official_intervals_visible_to_model": False,
        "caption_text_visible_to_model": False,
        "raw_response_persisted": False,
        "prompt_persisted": False,
        "video_copied": False,
        "dense_frames_persisted": False,
        "day_test140_accessed": False,
        "week_accessed": False,
    }
    _write_json(out_root / "run_manifest.json", run_manifest)

    all_results: list[dict[str, Any]] = []
    by_segment: dict[str, list[dict[str, Any]]] = {}
    for row in selection:
        by_segment.setdefault(str(row["segment_id"]), []).append(dict(row))
    segment_ids = tuple(sorted(by_segment))
    progress_lock = threading.Lock()
    progress_completed = 0

    def run_segment(segment_id: str) -> tuple[str, int, tuple[dict[str, Any], ...]]:
        nonlocal progress_completed
        segment = segments[segment_id]
        rows = tuple(
            sorted(
                by_segment[segment_id],
                key=lambda row: int(row["tier0_frame_index"]),
            )
        )
        results = _run_segment_batches(
            segment,
            rows,
            arm=arm,
            result_root=result_root,
            temp_root=temp_root,
            client_for_worker=client_for_worker,
            args=args,
            batch_workers=1,
        )
        with progress_lock:
            progress_completed += 1
            print(
                "CHANGE_OCR_SEGMENT_DONE "
                f"completed={progress_completed}/{len(segment_ids)} "
                f"segment={segment_id} frames={len(rows)} batches={len(results)}",
                flush=True,
            )
        return segment_id, len(rows), results

    segment_results = _ordered_parallel_map(
        segment_ids,
        run_segment,
        workers=int(args.workers),
    )
    for _segment_id, _frame_count, results in segment_results:
        all_results.extend(results)

    frame_metadata = {
        _frame_label(row): {
            "frame_id": _frame_label(row),
            "virtual_time_sec": float(row["virtual_time_sec"]),
            "source_time_sec": float(row["source_time_sec"]),
            "segment_id": str(row["segment_id"]),
            "source_video_id": str(row["source_video_id"]),
            "selection_arm": arm,
            "selection_reason": str(row["selection_reason"]),
        }
        for row in selection
    }
    parsed_rows = tuple(
        dict(row)
        for result in all_results
        for row in tuple(result.get("parsed_rows", ()) or ())
        if isinstance(row, Mapping)
    )
    admitted = admit_entity_occurrences(
        parsed_rows,
        frame_metadata=frame_metadata,
        merge_gap_sec=float(args.occurrence_gap_sec),
    )
    occurrences = tuple(admitted["occurrences"])
    occurrence_path = out_root / "entity_occurrences.jsonl"
    write_jsonl(occurrence_path, occurrences)

    success = all(result.get("status") == "success" for result in all_results)
    actual_models = {str(result.get("actual_model", "")) for result in all_results}
    temp_images = tuple(temp_root.rglob("*.jpg")) if temp_root.exists() else ()
    gates = {
        "sampling_structural_gate_passed": bool(
            sampling_report.get("gates", {}).get("structural_gate_passed")
        ),
        "selected_frame_count_exact": len(selection)
        == int(args.expected_selected_frames),
        "selection_arm_exact": all(
            row.get("selection_arm") == arm for row in selection
        ),
        "actual_model_matches": actual_models == {str(args.expected_model)},
        "all_batches_complete": success,
        "all_results_parse": all(
            result.get("parse_status") == "success" for result in all_results
        ),
        "blind_prompt_contract_valid": all(
            run_manifest[key] is False
            for key in (
                "question_visible_to_model",
                "options_visible_to_model",
                "answer_visible_to_model",
                "official_intervals_visible_to_model",
                "caption_text_visible_to_model",
                "raw_response_persisted",
                "prompt_persisted",
            )
        ),
        "all_occurrences_satisfy_runtime_admission": all(
            admitted_entity_row_valid(row) for row in occurrences
        ),
        "zero_video_copy": run_manifest["video_copied"] is False,
        "zero_dense_frame_persistence": run_manifest["dense_frames_persisted"] is False,
        "zero_temporary_frame_files": not temp_images,
    }
    gates["structural_gate_passed"] = all(gates.values())
    durations = [float(row.get("duration_sec", 0.0) or 0.0) for row in all_results]
    report = {
        "schema_version": "MMLifelongChangeTriggeredEntityOCRReportV1",
        "contract": CHANGE_TRIGGERED_ENTITY_CONTRACT,
        "decision": (
            "ENTITY_OCCURRENCES_READY"
            if gates["structural_gate_passed"]
            else "STRUCTURAL_FAILURE"
        ),
        "arm": arm,
        "counts": {
            "selected_frames": len(selection),
            "batch_results": len(all_results),
            "successful_batches": sum(
                result.get("status") == "success" for result in all_results
            ),
            "model_calls": sum(
                int(result.get("attempt_count", 0) or 0) for result in all_results
            ),
            "parse_retries": sum(
                max(0, int(result.get("attempt_count", 0) or 0) - 1)
                for result in all_results
            ),
            "parsed_entity_rows": len(parsed_rows),
            "admitted_entity_occurrences": len(occurrences),
        },
        "density": {
            "mean_batch_duration_sec": (
                statistics.fmean(durations) if durations else 0.0
            ),
            "mean_occurrences_per_1000_frames": (
                1000.0 * len(occurrences) / len(selection) if selection else 0.0
            ),
        },
        "rejection_counts": dict(admitted["rejection_counts"]),
        "gates": gates,
        "provenance": run_manifest,
        "endpoint_values_were_not_computed": True,
        "retrieval_run": False,
        "qa_run": False,
        "judge_calls": 0,
    }
    report_path = out_root / "entity_occurrence_report.json"
    _write_json(report_path, report)
    (out_root / "entity_occurrence_report.md").write_text(
        _render_markdown(report), encoding="utf-8"
    )
    print(
        "CHANGE_OCR_DONE "
        f"arm={arm} frames={len(selection)} occurrences={len(occurrences)} "
        f"gate={str(gates['structural_gate_passed']).lower()}",
        flush=True,
    )
    return report_path


def _run_segment_batches(
    segment: VirtualVideoSegment,
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    result_root: Path,
    temp_root: Path,
    client_for_worker: Callable[[], OpenAICompatibleClient],
    args: argparse.Namespace,
    batch_workers: int | None = None,
) -> tuple[dict[str, Any], ...]:
    batch_size = max(1, min(16, int(args.batch_size)))
    batches = tuple(
        tuple(dict(row) for row in rows[start : start + batch_size])
        for start in range(0, len(rows), batch_size)
    )
    reusable: dict[int, dict[str, Any]] = {}
    pending: list[int] = []
    for index, batch in enumerate(batches):
        result_path = _batch_result_path(result_root, arm=arm, batch=batch)
        prior = (
            _read_json(result_path) if args.resume and result_path.is_file() else None
        )
        if prior is not None and _reusable_batch_result(
            prior,
            batch=batch,
            expected_model=str(args.expected_model),
        ):
            reusable[index] = {**prior, "resume_reused_success": True}
        else:
            pending.append(index)
    if not pending:
        segment_temp = temp_root / _safe_name(segment.segment_id)
        if segment_temp.exists():
            shutil.rmtree(segment_temp)
        return tuple(reusable[index] for index in range(len(batches)))

    segment_temp = temp_root / _safe_name(segment.segment_id)
    frame_paths = _materialize_selected_segment_frames(
        segment,
        rows,
        out_dir=segment_temp,
        max_image_edge=int(args.max_image_edge),
        ffmpeg_executable=str(args.ffmpeg_executable),
    )
    path_by_identity = {
        _selection_identity(row): path for row, path in zip(rows, frame_paths)
    }
    requested_workers = (
        int(args.workers) if batch_workers is None else int(batch_workers)
    )
    workers = max(1, min(MAX_WORKERS, requested_workers, len(pending)))
    generated: dict[int, dict[str, Any]] = {}
    if workers == 1:
        for index in pending:
            generated[index] = _run_batch(
                batches[index],
                image_paths=tuple(
                    path_by_identity[_selection_identity(row)]
                    for row in batches[index]
                ),
                arm=arm,
                result_path=_batch_result_path(
                    result_root, arm=arm, batch=batches[index]
                ),
                client_for_worker=client_for_worker,
                args=args,
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_batch,
                    batches[index],
                    image_paths=tuple(
                        path_by_identity[_selection_identity(row)]
                        for row in batches[index]
                    ),
                    arm=arm,
                    result_path=_batch_result_path(
                        result_root, arm=arm, batch=batches[index]
                    ),
                    client_for_worker=client_for_worker,
                    args=args,
                ): index
                for index in pending
            }
            for future in as_completed(futures):
                generated[futures[future]] = future.result()
    if all(result.get("status") == "success" for result in generated.values()):
        shutil.rmtree(segment_temp)
    combined = {**reusable, **generated}
    return tuple(combined[index] for index in range(len(batches)))


def _ordered_parallel_map(
    values: Sequence[T],
    function: Callable[[T], R],
    *,
    workers: int,
) -> tuple[R, ...]:
    worker_count = max(1, min(MAX_WORKERS, int(workers), len(values)))
    if worker_count == 1:
        return tuple(function(value) for value in values)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return tuple(executor.map(function, values))


def _run_batch(
    batch: Sequence[Mapping[str, Any]],
    *,
    image_paths: Sequence[Path],
    arm: str,
    result_path: Path,
    client_for_worker: Callable[[], OpenAICompatibleClient],
    args: argparse.Namespace,
) -> dict[str, Any]:
    client = client_for_worker()
    labels = tuple(_frame_label(row) for row in batch)
    prompt = global_entity_ocr_prompt(labels)
    prompt_digest = stable_digest(prompt)
    selection_digest = _batch_selection_digest(batch)
    started = time.monotonic()
    attempts = []
    parsed: tuple[dict[str, Any], ...] | None = None
    diagnostic: dict[str, Any] = {}
    response_metadata: dict[str, Any] = {}
    for attempt in range(2):
        call_prompt = prompt
        if attempt:
            call_prompt += (
                "\nThe previous response violated the schema. Emit each allowed "
                "frame_label exactly once and return JSON only."
            )
            if diagnostic.get("status") == "too_many_rows":
                call_prompt += (
                    " Keep each frame's entities array to at most "
                    f"{MAX_GLOBAL_ENTITY_ROWS_PER_FRAME} entries. Retain only "
                    "stable named-entity text visibly supported by that frame."
                )
        try:
            raw = client.chat(
                call_prompt,
                image_paths=tuple(str(path) for path in image_paths),
                image_labels=tuple(
                    f"{label} (blind global frame {index + 1})"
                    for index, label in enumerate(labels)
                ),
                max_tokens=max(4096, int(args.max_completion_tokens)),
                response_format={"type": "json_object"},
            )
            response_metadata = safe_response_metadata(client.last_response_metadata)
        except Exception as exc:
            attempts.append(
                {
                    "attempt_index": attempt + 1,
                    "status": "model_failed",
                    "error_type": type(exc).__name__,
                }
            )
            break
        diagnostic = parse_global_entity_ocr_response_diagnostic(
            raw,
            allowed_frame_labels=labels,
        )
        raw_rows = diagnostic.get("rows")
        parsed = tuple(raw_rows) if isinstance(raw_rows, Sequence) else None
        attempts.append(
            {
                "attempt_index": attempt + 1,
                "status": "success" if parsed is not None else "invalid_json",
                "parse_status": str(diagnostic.get("status", "invalid")),
                "normalization_counts": dict(
                    diagnostic.get("normalization_counts", {})
                ),
                "response_metadata": response_metadata,
                "model_response_digest": stable_digest(raw),
            }
        )
        if parsed is not None:
            break
    result = {
        "schema_version": "MMLifelongChangeTriggeredEntityOCRBatchV1",
        "contract": CHANGE_TRIGGERED_ENTITY_CONTRACT,
        "arm": arm,
        "status": "success" if parsed is not None else attempts[-1]["status"],
        "actual_model": str(client.model),
        "frame_labels": list(labels),
        "selection_digest": selection_digest,
        "prompt_digest": prompt_digest,
        "parsed_rows": list(parsed or ()),
        "parsed_entity_row_count": len(parsed or ()),
        "parse_status": str(diagnostic.get("status", "invalid")),
        "normalization_counts": dict(diagnostic.get("normalization_counts", {})),
        "attempt_count": len(attempts),
        "attempt_history": attempts,
        "response_metadata": response_metadata,
        "duration_sec": round(time.monotonic() - started, 3),
        "resume_reused_success": False,
        "raw_response_persisted": False,
        "prompt_persisted": False,
    }
    _write_json(result_path, result)
    return result


def _materialize_selected_segment_frames(
    segment: VirtualVideoSegment,
    rows: Sequence[Mapping[str, Any]],
    *,
    out_dir: Path,
    max_image_edge: int,
    ffmpeg_executable: str = "ffmpeg",
) -> tuple[Path, ...]:
    selected = tuple(
        sorted(
            (dict(row) for row in rows), key=lambda row: int(row["tier0_frame_index"])
        )
    )
    indexes = tuple(int(row["tier0_frame_index"]) for row in selected)
    if len(set(indexes)) != len(indexes):
        raise ValueError("selected segment frames contain duplicate Tier-0 indexes")
    output_root = Path(out_dir)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    select_expression = "+".join(f"eq(n\\,{index})" for index in indexes)
    duration = float(segment.source_end_sec) - float(segment.source_start_sec)
    output_pattern = output_root / "frame_%06d.jpg"
    command = [
        str(ffmpeg_executable),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{float(segment.source_start_sec):.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(segment.source_path),
        "-vf",
        (
            f"fps=1.00000000,select='{select_expression}',"
            f"scale={int(max_image_edge)}:{int(max_image_edge)}:"
            "force_original_aspect_ratio=decrease"
        ),
        "-vsync",
        "0",
        "-q:v",
        "2",
        str(output_pattern),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(120.0, duration * 2.0),
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"ffmpeg executable is unavailable: {ffmpeg_executable}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("selected-frame ffmpeg extraction timed out") from exc
    paths = tuple(sorted(output_root.glob("frame_*.jpg")))
    if completed.returncode != 0 or len(paths) != len(selected):
        tail = " | ".join((completed.stderr or "").strip().splitlines()[-2:])
        raise RuntimeError(
            "selected-frame extraction mismatch: "
            f"expected {len(selected)}, found {len(paths)}; {tail}"
        )
    return paths


def _reusable_batch_result(
    result: Mapping[str, Any],
    *,
    batch: Sequence[Mapping[str, Any]],
    expected_model: str,
) -> bool:
    return (
        result.get("status") == "success"
        and result.get("parse_status") == "success"
        and result.get("actual_model") == expected_model
        and result.get("selection_digest") == _batch_selection_digest(batch)
        and tuple(result.get("frame_labels", ()) or ())
        == tuple(_frame_label(row) for row in batch)
    )


def _batch_result_path(
    root: Path,
    *,
    arm: str,
    batch: Sequence[Mapping[str, Any]],
) -> Path:
    digest = _batch_selection_digest(batch)
    return Path(root) / arm / digest[:2] / f"{digest}.json"


def _batch_selection_digest(batch: Sequence[Mapping[str, Any]]) -> str:
    return stable_digest(
        [
            {
                "frame_label": _frame_label(row),
                "virtual_time_sec": float(row["virtual_time_sec"]),
                "source_time_sec": float(row["source_time_sec"]),
                "selection_reason": str(row["selection_reason"]),
            }
            for row in batch
        ]
    )


def _frame_label(row: Mapping[str, Any]) -> str:
    segment = hashlib.sha256(str(row["segment_id"]).encode("utf-8")).hexdigest()[:8]
    return f"frame_{segment}_{int(row['tier0_frame_index']):06d}"


def _selection_identity(row: Mapping[str, Any]) -> str:
    return f"{row['segment_id']}:{int(row['tier0_frame_index'])}"


def _safe_name(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def _render_markdown(report: Mapping[str, Any]) -> str:
    counts = report["counts"]
    return "\n".join(
        [
            "# WP16-7 Entity Occurrence Extraction",
            "",
            f"- Decision: `{report['decision']}`",
            f"- Arm: `{report['arm']}`",
            f"- Selected frames / batches: `{counts['selected_frames']} / {counts['batch_results']}`",
            f"- Model calls / retries: `{counts['model_calls']} / {counts['parse_retries']}`",
            f"- Parsed rows / admitted occurrences: `{counts['parsed_entity_rows']} / {counts['admitted_entity_occurrences']}`",
            f"- Structural gate: `{str(report['gates']['structural_gate_passed']).lower()}`",
            "",
            "Only normalized OCR rows and occurrence lineage were persisted. Selected images were temporary; retrieval, QA, and judge evaluation were not run.",
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
    parser.add_argument("--sampling-root", required=True)
    parser.add_argument("--protocol-spec", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--arm", choices=ALLOWED_ARMS, required=True)
    parser.add_argument("--expected-selected-frames", type=int, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--section", default="planner_api")
    parser.add_argument("--expected-model", default="pa/gmn-2.5-pr")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-image-edge", type=int, default=1280)
    parser.add_argument("--ffmpeg-executable", default="ffmpeg")
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--occurrence-gap-sec", type=float, default=60.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
