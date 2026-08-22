#!/usr/bin/env python3
"""Run a blind Gemini OCR probe and an oracle-localized retrieval diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path
import threading
import time
from typing import Any, Mapping, Sequence

from vcah.caption_lexical_index import CaptionLexicalIndex
from vcah.caption_occurrence import build_caption_occurrence_set
from vcah.model_client import OpenAICompatibleClient
from vcah.occurrence_negative_sidecar import (
    file_sha256,
    safe_response_metadata,
    stable_digest,
)
from vcah.occurrence_ocr import (
    GEMINI_OCR_CONTRACT,
    OCR_PROMPT_VARIANTS,
    deduplicate_ocr_rows,
    enrich_caption_passages_with_ocr,
    fuse_caption_hit_ranks,
    gemini_ocr_prompt,
    ocr_sidecar_passages,
    ocr_query_overlap,
    parse_gemini_ocr_response_diagnostic,
)
from vcah.virtual_video import VirtualVideoWorkspace, materialize_window_frames


MAX_WORKERS = 16
RECALL_KS = (1, 3, 5)


def run(args: argparse.Namespace) -> Path:
    manifest_path = Path(args.case_manifest)
    if file_sha256(manifest_path) != str(args.expected_manifest_sha256):
        raise ValueError("case manifest SHA256 mismatch")
    frozen_ids = _manifest_case_ids(_read_json_value(manifest_path))
    selected_ids = _select_case_ids(frozen_ids, tuple(args.case_ids or ()))
    if len(selected_ids) != int(args.expected_cases):
        raise ValueError(
            f"expected {args.expected_cases} selected cases, found {len(selected_ids)}"
        )
    variants = tuple(dict.fromkeys(str(value) for value in args.prompt_variants))
    if any(variant not in OCR_PROMPT_VARIANTS for variant in variants):
        raise ValueError("unsupported OCR prompt variant")
    if str(args.selected_variant) not in variants:
        raise ValueError("selected retrieval variant must be among prompt variants")

    out_root = Path(args.out_root)
    if out_root.exists() and any(out_root.iterdir()) and not args.resume:
        raise FileExistsError(f"OCR output is not empty: {out_root}")
    (out_root / "cases").mkdir(parents=True, exist_ok=True)
    client = OpenAICompatibleClient.from_yaml(Path(args.config), section=args.section)
    if str(args.expected_model) and str(client.model) != str(args.expected_model):
        raise ValueError(
            f"actual model mismatch: {client.model} != {args.expected_model}"
        )
    run_manifest = {
        "schema_version": "MMLifelongGeminiOCRRunV1",
        "contract": GEMINI_OCR_CONTRACT,
        "source_commit": str(args.source_commit),
        "case_manifest_sha256": file_sha256(manifest_path),
        "selected_case_count": len(selected_ids),
        "selected_case_ids": list(selected_ids),
        "prompt_variants": list(variants),
        "selected_retrieval_variant": str(args.selected_variant),
        "actual_model": str(client.model),
        "config_sha256": file_sha256(Path(args.config)),
        "api_section": str(args.section),
        "fps": float(args.fps),
        "max_frames_per_case": int(args.max_frames),
        "max_completion_tokens": max(4096, int(args.max_completion_tokens)),
        "workers": max(1, min(MAX_WORKERS, int(args.workers))),
        "localization": "official_clue_intervals_oracle_diagnostic",
        "official_intervals_visible_to_model": False,
        "question_visible_to_model": False,
        "options_visible_to_model": False,
        "answer_visible_to_model": False,
        "raw_response_persisted": False,
        "prompt_persisted": False,
        "retrieval_claim_is_formal_improvement": False,
        "day_test140_accessed": False,
        "week_accessed": False,
    }
    _write_json_atomic(out_root / "run_manifest.json", run_manifest)

    frame_manifests: dict[str, dict[str, Any]] = {}
    for case_id in selected_ids:
        frame_manifests[case_id] = _materialize_case(
            case_id,
            run_root=Path(args.run_root),
            evaluation_record_root=Path(args.evaluation_record_root),
            out_root=out_root,
            fps=float(args.fps),
            max_frames=int(args.max_frames),
        )
        _write_json_atomic(
            out_root / "cases" / case_id / "frame_manifest.json",
            frame_manifests[case_id],
        )

    work_items = tuple(
        (case_id, variant) for case_id in selected_ids for variant in variants
    )
    results: dict[tuple[str, str], dict[str, Any]] = {}
    lock = threading.Lock()
    workers = max(1, min(MAX_WORKERS, int(args.workers), len(work_items)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_one,
                case_id,
                variant,
                frame_manifest=frame_manifests[case_id],
                out_root=out_root,
                client=client,
                args=args,
            ): (case_id, variant)
            for case_id, variant in work_items
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "schema_version": "MMLifelongGeminiOCRCaseResultV1",
                    "contract": GEMINI_OCR_CONTRACT,
                    "case_id": key[0],
                    "prompt_variant": key[1],
                    "status": "orchestrator_failed",
                    "error_type": type(exc).__name__,
                }
            with lock:
                results[key] = result
                _write_progress(out_root, work_items, results, run_manifest)
            print(
                f"OCR_DONE case_id={key[0]} variant={key[1]} "
                f"status={result['status']} lines={len(result.get('ocr_rows', ()))}",
                flush=True,
            )
    _write_progress(out_root, work_items, results, run_manifest)
    if sum(row.get("status") == "success" for row in results.values()) != len(
        work_items
    ):
        raise SystemExit(1)

    report = _build_report(
        selected_ids,
        variants,
        results,
        frame_manifests,
        run_root=Path(args.run_root),
        selected_variant=str(args.selected_variant),
        retrieval_top_k=int(args.retrieval_top_k),
    )
    report["provenance"] = run_manifest
    report_path = out_root / "ocr_retrieval_report.json"
    _write_json_atomic(report_path, report)
    (out_root / "ocr_retrieval_report.md").write_text(
        _render_markdown(report), encoding="utf-8"
    )
    print(
        "OCR_EXPERIMENT_DONE "
        f"cases={len(selected_ids)} model={client.model} "
        f"baseline_r5={report['retrieval']['baseline']['at_5']['count']} "
        f"enriched_r5={report['retrieval']['ocr_fused']['at_5']['count']} "
        "claim=oracle_diagnostic_only",
        flush=True,
    )
    return report_path


def _materialize_case(
    case_id: str,
    *,
    run_root: Path,
    evaluation_record_root: Path,
    out_root: Path,
    fps: float,
    max_frames: int,
) -> dict[str, Any]:
    workspace = VirtualVideoWorkspace.load(run_root / "cases" / case_id)
    record = _read_json(evaluation_record_root / case_id / "evaluation_case.json")
    intervals = tuple(
        interval
        for raw in tuple(record.get("clue_intervals", ()) or ())
        if (interval := _optional_interval(raw)) is not None
    )
    if not intervals:
        raise ValueError(f"{case_id}: no official clue intervals")
    frame_root = out_root / "frames" / case_id
    frame_root.mkdir(parents=True, exist_ok=True)
    materialization_workspace = replace(
        workspace,
        root_dir=frame_root,
        frame_manifest=frame_root / "frame_manifest.jsonl",
    )
    cap = max(len(intervals), int(max_frames))
    base, extra = divmod(cap, len(intervals))
    frames: list[dict[str, Any]] = []
    for interval_index, (start_sec, end_sec) in enumerate(intervals):
        interval_cap = max(1, base + (1 if interval_index < extra else 0))
        sampled = materialize_window_frames(
            materialization_workspace,
            start_sec,
            end_sec,
            query_id=f"ocr_{case_id}_{interval_index:02d}",
            fps=fps,
            max_frames=interval_cap,
        )
        for frame in sampled:
            frames.append(
                {
                    "path": str(Path(frame.path).resolve()),
                    "virtual_time_sec": frame.virtual_time_sec,
                    "segment_id": frame.segment_id,
                    "source_video_id": frame.source_video_id,
                    "source_time_sec": frame.source_time_sec,
                    "clue_interval_index": interval_index,
                }
            )
    frames.sort(key=lambda row: (row["virtual_time_sec"], row["path"]))
    for index, frame in enumerate(frames, start=1):
        frame["frame_label"] = f"frame_{index:02d}"
    return {
        "schema_version": "MMLifelongGeminiOCRFrameManifestV1",
        "contract": GEMINI_OCR_CONTRACT,
        "case_id": case_id,
        "localization": "official_clue_intervals_oracle_diagnostic",
        "clue_intervals": [list(interval) for interval in intervals],
        "frame_count": len(frames),
        "frames": frames,
    }


def _run_one(
    case_id: str,
    variant: str,
    *,
    frame_manifest: Mapping[str, Any],
    out_root: Path,
    client: OpenAICompatibleClient,
    args: argparse.Namespace,
) -> dict[str, Any]:
    result_path = out_root / "cases" / case_id / f"ocr.{variant}.json"
    frames = tuple(
        dict(frame)
        for frame in tuple(frame_manifest.get("frames", ()) or ())
        if isinstance(frame, Mapping)
    )
    labels = tuple(str(frame["frame_label"]) for frame in frames)
    paths = tuple(str(frame["path"]) for frame in frames)
    prompt = gemini_ocr_prompt(labels, variant=variant)
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
        and prior.get("actual_model") == client.model
        and prior.get("prompt_digest") == stable_digest(prompt)
        and prior.get("frame_digest") == frame_digest
    ):
        return {**prior, "resume_reused_success": True}
    started = time.monotonic()
    attempts: list[dict[str, Any]] = []
    parsed = None
    parse_diagnostic: dict[str, Any] = {}
    raw = ""
    response_metadata: dict[str, Any] = {}
    for parse_attempt in range(2):
        call_prompt = prompt
        if parse_attempt:
            call_prompt += (
                "\nThe previous response violated the schema. Emit every allowed frame_label "
                "exactly once and return JSON only."
            )
        try:
            raw = client.chat(
                call_prompt,
                image_paths=paths,
                image_labels=tuple(
                    f"{label} (chronological frame {index + 1})"
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
        parse_diagnostic = parse_gemini_ocr_response_diagnostic(
            raw, allowed_frame_labels=labels
        )
        parsed_rows = parse_diagnostic.get("rows")
        parsed = tuple(parsed_rows) if isinstance(parsed_rows, Sequence) else None
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
    ocr_rows = (
        deduplicate_ocr_rows(parsed or (), frame_metadata=frame_by_label)
        if parsed is not None
        else ()
    )
    result = {
        "schema_version": "MMLifelongGeminiOCRCaseResultV1",
        "contract": GEMINI_OCR_CONTRACT,
        "case_id": case_id,
        "prompt_variant": variant,
        "status": "success" if parsed is not None else attempts[-1]["status"],
        "actual_model": str(client.model),
        "frame_count": len(frames),
        "frame_digest": frame_digest,
        "prompt_digest": stable_digest(prompt),
        "model_response_digest": stable_digest(raw),
        "ocr_rows": list(ocr_rows),
        "ocr_unique_line_count": len(ocr_rows),
        "ocr_nonempty": bool(ocr_rows),
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
        "raw_response_persisted": False,
        "prompt_persisted": False,
    }
    _write_json_atomic(result_path, result)
    return result


def _build_report(
    case_ids: Sequence[str],
    variants: Sequence[str],
    results: Mapping[tuple[str, str], Mapping[str, Any]],
    frame_manifests: Mapping[str, Mapping[str, Any]],
    *,
    run_root: Path,
    selected_variant: str,
    retrieval_top_k: int,
) -> dict[str, Any]:
    variant_summary: dict[str, Any] = {}
    queries_by_case: dict[str, tuple[str, ...]] = {}
    packets_by_case: dict[str, tuple[dict[str, Any], ...]] = {}
    for case_id in case_ids:
        packets = _caption_packets(run_root / "cases" / case_id)
        packets_by_case[case_id] = packets
        queries_by_case[case_id] = tuple(
            dict.fromkeys(
                str(query)
                for packet in packets
                for query in tuple(packet.get("queries", ()) or ())
                if str(query).strip()
            )
        )
    for variant in variants:
        rows = [results[(case_id, variant)] for case_id in case_ids]
        overlaps = [
            ocr_query_overlap(row.get("ocr_rows", ()), queries_by_case[case_id])
            for case_id, row in zip(case_ids, rows)
        ]
        variant_summary[variant] = {
            "success_count": sum(row.get("status") == "success" for row in rows),
            "nonempty_case_count": sum(bool(row.get("ocr_rows")) for row in rows),
            "unique_line_count": sum(len(row.get("ocr_rows", ())) for row in rows),
            "query_overlap_case_count": sum(
                overlap["matched_token_count"] > 0 for overlap in overlaps
            ),
            "matched_query_token_count": sum(
                int(overlap["matched_token_count"]) for overlap in overlaps
            ),
            "parse_retry_count": sum(
                max(0, int(row.get("attempt_count", 0) or 0) - 1) for row in rows
            ),
        }

    case_rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        run_dir = run_root / "cases" / case_id
        workspace = VirtualVideoWorkspace.load(run_dir)
        packets = packets_by_case[case_id]
        selected = results[(case_id, selected_variant)]
        ocr_rows = tuple(selected.get("ocr_rows", ()) or ())
        overlap = ocr_query_overlap(ocr_rows, queries_by_case[case_id])
        clues = tuple(
            tuple(float(value) for value in interval)
            for interval in tuple(
                frame_manifests[case_id].get("clue_intervals", ()) or ()
            )
        )
        baseline_best: int | None = None
        fused_best: int | None = None
        bound_passage_ids: set[str] = set()
        for packet in packets:
            config_digest = str(packet.get("config_digest", "") or "")
            lexical = CaptionLexicalIndex.from_asset_root(
                workspace.asset_root, config_digest=config_digest
            )
            enriched_passages = enrich_caption_passages_with_ocr(
                lexical.passages, ocr_rows
            )
            bound_passage_ids.update(
                passage.passage_id
                for passage in enriched_passages
                if passage.metadata.get("ocr_contract") == GEMINI_OCR_CONTRACT
            )
            ocr_index = CaptionLexicalIndex(
                ocr_sidecar_passages(enriched_passages),
                config_digest=stable_digest(
                    {
                        "base": config_digest,
                        "contract": GEMINI_OCR_CONTRACT,
                        "case_id": case_id,
                        "variant": selected_variant,
                    }
                ),
            )
            queries = tuple(str(value) for value in packet.get("queries", ()) or ())
            baseline_hits = tuple(packet.get("hits", ()) or ())[:retrieval_top_k]
            ocr_hits = ocr_index.search(
                queries,
                top_k=max(20, retrieval_top_k * 4),
                time_range=_optional_interval(packet.get("time_range")),
                segment_ids=tuple(packet.get("segment_ids", ()) or ()),
                expand_neighbors=0,
            )
            fused_hits = fuse_caption_hit_ranks(
                baseline_hits,
                ocr_hits,
                top_k=retrieval_top_k,
            )
            baseline_set = packet.get("occurrence_set")
            if not isinstance(baseline_set, Mapping):
                baseline_set = build_caption_occurrence_set(baseline_hits)
            fused_set = build_caption_occurrence_set(fused_hits)
            baseline_best = _minimum_rank(
                baseline_best,
                _best_candidate_rank(
                    tuple(baseline_set.get("candidates", ()) or ()), clues
                ),
            )
            fused_best = _minimum_rank(
                fused_best,
                _best_candidate_rank(
                    tuple(fused_set.get("candidates", ()) or ()), clues
                ),
            )
        case_rows.append(
            {
                "case_id": case_id,
                "frame_count": int(frame_manifests[case_id].get("frame_count", 0)),
                "ocr_unique_line_count": len(ocr_rows),
                "ocr_bound_passage_count": len(bound_passage_ids),
                "query_overlap": overlap,
                "baseline_best_occurrence_rank": baseline_best,
                "ocr_fused_best_occurrence_rank": fused_best,
                "recovered_at_5": baseline_best is None and fused_best is not None,
                "regressed_at_5": baseline_best is not None and fused_best is None,
            }
        )
    return {
        "schema_version": "MMLifelongGeminiOCRRetrievalReportV1",
        "contract": GEMINI_OCR_CONTRACT,
        "decision": "ORACLE_LOCALIZED_OCR_DIAGNOSTIC_ONLY",
        "selected_variant": selected_variant,
        "case_count": len(case_rows),
        "prompt_variant_summary": variant_summary,
        "retrieval": {
            "baseline": _recall_summary(case_rows, "baseline_best_occurrence_rank"),
            "ocr_fused": _recall_summary(
                case_rows, "ocr_fused_best_occurrence_rank"
            ),
            "recovered_case_ids": [
                row["case_id"] for row in case_rows if row["recovered_at_5"]
            ],
            "regressed_case_ids": [
                row["case_id"] for row in case_rows if row["regressed_at_5"]
            ],
            "fusion": "baseline_caption_rank + OCR-only lexical rank, RRF k0=60",
        },
        "case_level": case_rows,
        "validity": {
            "model_was_question_blind": True,
            "model_was_answer_blind": True,
            "raw_model_response_persisted": False,
            "gold_intervals_used_to_select_frames": True,
            "formal_retrieval_improvement_claim_allowed": False,
            "required_next_step": (
                "Build a question-independent global OCR index before claiming retrieval gain."
            ),
        },
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    baseline = report["retrieval"]["baseline"]
    enriched = report["retrieval"]["ocr_fused"]
    lines = [
        "# MM-Lifelong Gemini 2.5 Pro OCR 诊断",
        "",
        "## 结论边界",
        "",
        "本轮使用官方 clue interval 选帧，因此只能判断 Gemini OCR 能否读出关键 UI 实体，不能作为正式检索提升。模型未看到题目、选项或答案，原始回复未落盘。",
        "",
        "## Prompt 对比",
        "",
        "| 版本 | 成功 case | 非空 OCR case | 去重文本行 | 与冻结 query 有交集的 case | 匹配 token | parse retry |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, row in report["prompt_variant_summary"].items():
        lines.append(
            f"| {variant} | {row['success_count']} | {row['nonempty_case_count']} | "
            f"{row['unique_line_count']} | {row['query_overlap_case_count']} | "
            f"{row['matched_query_token_count']} | {row['parse_retry_count']} |"
        )
    lines.extend(
        [
            "",
            "## Oracle-localized 检索诊断",
            "",
            "| 检索 | R@1 | R@3 | R@5 |",
            "|---|---:|---:|---:|",
            f"| Caption baseline | {_fmt_recall(baseline['at_1'])} | {_fmt_recall(baseline['at_3'])} | {_fmt_recall(baseline['at_5'])} |",
            f"| Caption + OCR RRF | {_fmt_recall(enriched['at_1'])} | {_fmt_recall(enriched['at_3'])} | {_fmt_recall(enriched['at_5'])} |",
            "",
            f"Recovered: {', '.join(report['retrieval']['recovered_case_ids']) or 'none'}",
            "",
            f"Regressed: {', '.join(report['retrieval']['regressed_case_ids']) or 'none'}",
            "",
            "## Case 级摘要",
            "",
            "| Case | Frames | OCR lines | Bound passages | Query overlap | Baseline rank | OCR rank |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["case_level"]:
        lines.append(
            f"| {row['case_id']} | {row['frame_count']} | {row['ocr_unique_line_count']} | "
            f"{row['ocr_bound_passage_count']} | {row['query_overlap']['matched_token_count']} | "
            f"{row['baseline_best_occurrence_rank'] or 'NA'} | "
            f"{row['ocr_fused_best_occurrence_rank'] or 'NA'} |"
        )
    lines.extend(
        [
            "",
            "## 下一步",
            "",
            "只有在问题无关的全局抽帧/OCR 索引上复现提升，才能把这里的诊断转成正式 retrieval 结果。",
        ]
    )
    return "\n".join(lines) + "\n"


def _caption_packets(run_dir: Path) -> tuple[dict[str, Any], ...]:
    packets: list[dict[str, Any]] = []
    for observation in _read_jsonl(run_dir / "observation_log.jsonl"):
        config = observation.get("sampling_config")
        if not isinstance(config, Mapping) or not isinstance(
            config.get("occurrence_set"), Mapping
        ):
            continue
        raw_output = observation.get("raw_output")
        if isinstance(raw_output, str) and raw_output.strip():
            raw_output = json.loads(raw_output)
        payload = raw_output if isinstance(raw_output, Mapping) else {}
        pointer = str(payload.get("raw_output_pointer", "") or "")
        if not pointer:
            continue
        path = Path(pointer)
        if not path.is_absolute():
            path = run_dir / path
        path = path.resolve()
        if path.parent != (run_dir / "caption_search").resolve():
            raise ValueError(f"{run_dir.name}: caption packet escaped case root")
        packets.append(_read_json(path))
    if not packets:
        raise ValueError(f"{run_dir.name}: no frozen caption packets")
    return tuple(packets)


def _best_candidate_rank(
    candidates: Sequence[Any], clues: Sequence[tuple[float, float]]
) -> int | None:
    ranks = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        interval = _optional_interval(candidate.get("time_range"))
        if interval is None:
            continue
        if any(_interval_overlap(interval, clue) > 0.0 for clue in clues):
            ranks.append(max(1, int(candidate.get("rank", 1) or 1)))
    return min(ranks) if ranks else None


def _recall_summary(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    return {
        f"at_{k}": {
            "count": sum(
                isinstance(row.get(key), int) and int(row[key]) <= k for row in rows
            ),
            "case_count": len(rows),
            "rate": (
                sum(
                    isinstance(row.get(key), int) and int(row[key]) <= k
                    for row in rows
                )
                / len(rows)
                if rows
                else 0.0
            ),
        }
        for k in RECALL_KS
    }


def _write_progress(
    out_root: Path,
    work_items: Sequence[tuple[str, str]],
    results: Mapping[tuple[str, str], Mapping[str, Any]],
    run_manifest: Mapping[str, Any],
) -> Path:
    statuses = Counter(str(row.get("status", "") or "") for row in results.values())
    payload = {
        "schema_version": "MMLifelongGeminiOCRBatchSummaryV1",
        "contract": GEMINI_OCR_CONTRACT,
        "selected_count": len(work_items),
        "completed_count": len(results),
        "success_count": statuses["success"],
        "failure_count": len(results) - statuses["success"],
        "status_counts": dict(sorted(statuses.items())),
        "actual_model": run_manifest["actual_model"],
        "raw_response_persisted": False,
    }
    path = out_root / "batch_summary.json"
    _write_json_atomic(path, payload)
    return path


def _select_case_ids(
    frozen_ids: Sequence[str], requested: Sequence[str]
) -> tuple[str, ...]:
    if not requested:
        return tuple(frozen_ids)
    aliases = {case_id.rsplit("-", 1)[-1]: case_id for case_id in frozen_ids}
    normalized = []
    for value in requested:
        case_id = str(value)
        resolved = case_id if case_id in frozen_ids else aliases.get(case_id, "")
        if not resolved:
            raise ValueError(f"requested case is outside the frozen manifest: {value}")
        normalized.append(resolved)
    return tuple(case_id for case_id in frozen_ids if case_id in set(normalized))


def _manifest_case_ids(payload: Any) -> tuple[str, ...]:
    raw = payload.get("case_ids", payload.get("cases", ())) if isinstance(payload, Mapping) else payload
    return tuple(
        str(row.get("case_id", row.get("id", "")) if isinstance(row, Mapping) else row)
        for row in tuple(raw or ())
        if str(row.get("case_id", row.get("id", "")) if isinstance(row, Mapping) else row)
    )


def _optional_interval(value: Any) -> tuple[float, float] | None:
    try:
        if value is None or len(value) != 2:
            return None
        start, end = sorted((float(value[0]), float(value[1])))
    except (TypeError, ValueError):
        return None
    return (start, end) if end > start else None


def _interval_overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def _minimum_rank(left: int | None, right: int | None) -> int | None:
    values = [value for value in (left, right) if value is not None]
    return min(values) if values else None


def _fmt_recall(row: Mapping[str, Any]) -> str:
    return f"{100.0 * float(row['rate']):.2f}% ({row['count']}/{row['case_count']})"


def _read_json(path: Path) -> dict[str, Any]:
    value = _read_json_value(path)
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _read_json_value(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    return tuple(
        dict(value)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and isinstance((value := json.loads(line)), Mapping)
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--evaluation-record-root", required=True)
    parser.add_argument("--case-manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--case-ids", nargs="*")
    parser.add_argument("--config", required=True)
    parser.add_argument("--section", default="planner_api")
    parser.add_argument("--expected-model", default="pa/gmn-2.5-pr")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--prompt-variants", nargs="+", default=["generic_v0", "ui_aware_v1"]
    )
    parser.add_argument("--selected-variant", default="ui_aware_v1")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=10)
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
