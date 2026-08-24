#!/usr/bin/env python3
"""Evaluate a global OCR entity sidecar on the frozen WP16-4/5 Anchor subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from vcah.caption_hybrid_search import CaptionHybridSearch
from vcah.caption_lexical_index import CaptionLexicalIndex
from vcah.caption_schema import CaptionHitV1, CaptionPassageV1
from vcah.caption_semantic_index import CaptionSemanticIndex
from vcah.embedding_adapter import SentenceTransformerEmbeddingAdapter
from vcah.global_entity_recall import build_global_entity_recall_report
from vcah.occurrence_entity_sidecar import (
    GLOBAL_ENTITY_SIDECAR_CONTRACT,
    admitted_entity_row_valid,
    build_entity_sidecar_passages,
    global_entity_duplicate_stats,
)
from vcah.occurrence_field_index import normalize_occurrence_fields
from vcah.occurrence_negative_sidecar import file_sha256
from vcah.occurrence_ocr import fuse_caption_hit_ranks, ocr_text_has_query_evidence
from vcah.virtual_video import VirtualVideoWorkspace


def run(args: argparse.Namespace) -> Path:
    paths = {
        "manifest": Path(args.case_manifest),
        "protocol": Path(args.protocol_spec),
        "relation": Path(args.relation_spec),
        "anchor_query": Path(args.anchor_query_spec),
        "field": Path(args.field_spec),
    }
    expected = {
        "manifest": args.expected_manifest_sha256,
        "protocol": args.expected_protocol_sha256,
        "relation": args.expected_relation_spec_sha256,
        "anchor_query": args.expected_anchor_query_spec_sha256,
        "field": args.expected_field_spec_sha256,
    }
    for label, path in paths.items():
        _require_sha(path, expected[label], label)
    protocol = _read_json(paths["protocol"])
    if protocol.get("contract") != GLOBAL_ENTITY_SIDECAR_CONTRACT:
        raise ValueError("WP16-6 protocol contract mismatch")

    case_ids = tuple(dict.fromkeys(str(value) for value in args.case_ids))
    manifest_ids = set(_manifest_case_ids(_read_json_value(paths["manifest"])))
    if len(case_ids) != int(args.expected_cases) or not set(case_ids) <= manifest_ids:
        raise ValueError("selected cases do not match frozen manifest subset")
    relation_cases = dict(_read_json(paths["relation"]).get("cases", {}) or {})
    query_cases = dict(_read_json(paths["anchor_query"]).get("cases", {}) or {})
    field_cases = dict(_read_json(paths["field"]).get("cases", {}) or {})
    if any(
        case_id not in relation_cases
        or case_id not in query_cases
        or case_id not in field_cases
        for case_id in case_ids
    ):
        raise ValueError("selected case is missing a frozen WP16 specification")

    sidecar_root = Path(args.sidecar_root)
    extraction_report = _read_json(sidecar_root / "global_entity_report.json")
    extraction_manifest = _read_json(sidecar_root / "run_manifest.json")
    entity_rows = _read_jsonl(sidecar_root / "entity_sidecar.jsonl")
    extraction_gate = _validate_extraction(
        extraction_report,
        extraction_manifest,
        entity_rows,
        expected_source_commit=str(args.expected_sidecar_source_commit),
        expected_model=str(args.expected_ocr_model),
        expected_protocol_sha256=str(args.expected_protocol_sha256),
        expected_passages=int(args.expected_passages),
    )

    first_workspace = VirtualVideoWorkspace.load(Path(args.run_root) / case_ids[0])
    lexical = CaptionLexicalIndex.from_asset_root(
        first_workspace.asset_root,
        config_digest=str(args.caption_config_digest),
    )
    if len(lexical.passages) != int(args.expected_passages):
        raise ValueError("Caption passage count mismatch")
    adapter = SentenceTransformerEmbeddingAdapter(
        str(args.embedding_model),
        revision=str(args.embedding_revision),
        device=str(args.embedding_device),
        normalize=True,
        batch_size=max(1, int(args.embedding_batch_size)),
    )
    dense = CaptionSemanticIndex.from_asset_root(
        first_workspace.asset_root,
        adapter=adapter,
        config_digest=str(args.caption_config_digest),
    )
    baseline_search = CaptionHybridSearch(lexical, dense, query_strategy="joint")
    sidecar_passages = build_entity_sidecar_passages(lexical.passages, entity_rows)
    entity_lexical = CaptionLexicalIndex(
        sidecar_passages,
        config_digest=f"{args.caption_config_digest}:wp16-6-global-entity",
    )
    passage_by_id = {passage.passage_id: passage for passage in lexical.passages}
    rows_by_passage: dict[str, list[dict[str, Any]]] = {}
    for entity in entity_rows:
        rows_by_passage.setdefault(str(entity.get("passage_id", "")), []).append(entity)

    case_rows = []
    for case_id in case_ids:
        workspace = VirtualVideoWorkspace.load(Path(args.run_root) / case_id)
        if workspace.asset_root.resolve() != first_workspace.asset_root.resolve():
            raise ValueError("frozen10 cases do not share one global Caption asset")
        relation = dict(relation_cases[case_id])
        query_case = dict(query_cases[case_id])
        fields = normalize_occurrence_fields(dict(field_cases[case_id]))
        anchor_query = str(query_case.get("anchor_query", "") or "").strip()
        entity_query = tuple(fields["entity"]["query_terms"])
        intervals = tuple(relation.get("anchor_intervals", ()) or ())
        if not anchor_query or not entity_query or not intervals:
            raise ValueError(f"{case_id}: incomplete frozen query/anchor specification")
        top_k = max(1, int(args.top_k))
        baseline_hits = baseline_search.search((anchor_query,), top_k=top_k)
        entity_hits = entity_lexical.search(entity_query, top_k=top_k)
        fused_hits = fuse_caption_hit_ranks(
            baseline_hits,
            entity_hits,
            top_k=top_k,
            rrf_k0=max(1, int(args.rrf_k0)),
            baseline_weight=1.0,
            ocr_weight=1.0,
        )
        matching_passage_ids = {
            passage_id
            for passage_id, passage_rows in rows_by_passage.items()
            if any(
                ocr_text_has_query_evidence(
                    str(entity.get("text", "") or ""), entity_query
                )
                for entity in passage_rows
            )
        }
        gold_matching = {
            passage_id
            for passage_id in matching_passage_ids
            if passage_id in passage_by_id
            and _overlaps_anchor(passage_by_id[passage_id], intervals)
        }
        case_rows.append(
            {
                "case_id": case_id,
                "anchor_query": anchor_query,
                "entity_query": list(entity_query),
                "anchor_intervals": [list(value) for value in intervals],
                "baseline_rank": _hit_rank(baseline_hits, intervals),
                "entity_rank": _hit_rank(entity_hits, intervals),
                "fused_rank": _fused_rank(
                    fused_hits,
                    passage_by_id=passage_by_id,
                    anchor_intervals=intervals,
                ),
                "gold_anchor_entity_covered": bool(gold_matching),
                "same_entity_occurrence_count": len(matching_passage_ids),
                "gold_entity_occurrence_count": len(gold_matching),
                "non_gold_entity_document_rate": (
                    (len(matching_passage_ids) - len(gold_matching))
                    / len(matching_passage_ids)
                    if matching_passage_ids
                    else 0.0
                ),
            }
        )

    duplicate_stats = global_entity_duplicate_stats(entity_rows)
    report = build_global_entity_recall_report(
        case_rows,
        expected_cases=int(args.expected_cases),
        extraction_gate_passed=extraction_gate,
        duplicate_stats=duplicate_stats,
    )
    report["provenance"] = {
        "label": str(args.provenance),
        "source_run_root": str(Path(args.run_root)),
        "sidecar_root": str(sidecar_root),
        **{f"{label}_path": str(path) for label, path in paths.items()},
        **{f"{label}_sha256": file_sha256(path) for label, path in paths.items()},
        "caption_config_digest": str(args.caption_config_digest),
        "caption_index_digest": lexical.index_digest,
        "entity_index_digest": entity_lexical.index_digest,
        "embedding": dict(adapter.manifest),
        "top_k": max(1, int(args.top_k)),
        "rrf_k0": max(1, int(args.rrf_k0)),
        "generative_model_calls_during_evaluation": 0,
        "vlm_calls_during_evaluation": 0,
        "judge_calls": 0,
        "day_test140_accessed": False,
        "week_accessed": False,
    }
    out_json = Path(args.out_json)
    _write_json(out_json, report)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_render_markdown(report), encoding="utf-8")
    print(
        "GLOBAL_ENTITY_RECALL_DONE "
        f"decision={report['decision']} "
        f"structural={str(report['structural_gate_passed']).lower()} "
        f"coverage={report['entity_coverage']['count']}/{report['case_count']} "
        f"r5={report['retrieval']['caption_entity_rrf']['at_5']['count']}/{report['case_count']}",
        flush=True,
    )
    return out_json


def _validate_extraction(
    report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_source_commit: str,
    expected_model: str,
    expected_protocol_sha256: str,
    expected_passages: int,
) -> bool:
    checks = (
        report.get("contract") == GLOBAL_ENTITY_SIDECAR_CONTRACT,
        bool(report.get("gates", {}).get("structural_gate_passed")),
        manifest.get("selection_mode") == "full",
        int(manifest.get("selected_passage_count", 0) or 0) == expected_passages,
        manifest.get("source_commit") == expected_source_commit,
        manifest.get("actual_model") == expected_model,
        manifest.get("protocol_spec_sha256") == expected_protocol_sha256,
        manifest.get("question_visible_to_model") is False,
        manifest.get("answer_visible_to_model") is False,
        manifest.get("official_intervals_visible_to_model") is False,
        manifest.get("caption_text_visible_to_model") is False,
        manifest.get("day_test140_accessed") is False,
        manifest.get("week_accessed") is False,
        int(report.get("counts", {}).get("admitted_entities", -1)) == len(rows),
        all(admitted_entity_row_valid(row) for row in rows),
    )
    return all(checks)


def _hit_rank(
    hits: Sequence[CaptionHitV1], anchor_intervals: Sequence[Sequence[float]]
) -> int | None:
    for hit in hits:
        if _overlaps_anchor(hit, anchor_intervals):
            return int(hit.rank)
    return None


def _fused_rank(
    hits: Sequence[Mapping[str, Any]],
    *,
    passage_by_id: Mapping[str, CaptionPassageV1],
    anchor_intervals: Sequence[Sequence[float]],
) -> int | None:
    for fallback_rank, hit in enumerate(hits, start=1):
        passage = passage_by_id.get(str(hit.get("passage_id", "") or ""))
        if passage is not None and _overlaps_anchor(passage, anchor_intervals):
            return int(hit.get("rank", fallback_rank) or fallback_rank)
    return None


def _overlaps_anchor(
    passage: CaptionPassageV1 | CaptionHitV1,
    anchor_intervals: Sequence[Sequence[float]],
) -> bool:
    start = float(passage.virtual_start_sec)
    end = float(passage.virtual_end_sec)
    for value in anchor_intervals:
        if len(value) == 2 and min(end, float(value[1])) > max(start, float(value[0])):
            return True
    return False


def _render_markdown(report: Mapping[str, Any]) -> str:
    retrieval = report["retrieval"]
    coverage = report["entity_coverage"]
    false_positive = report["false_positive_diagnostics"]
    lines = [
        "# WP16-6A Global OCR Entity Anchor Recall",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Structural gate: `{str(report['structural_gate_passed']).lower()}`",
        f"- Gold anchor entity coverage: `{coverage['count']}/{coverage['case_count']}`",
        f"- Mean same-entity occurrences: `{false_positive['mean_same_entity_occurrence_count']:.3f}`",
        f"- Mean non-gold entity document rate: `{false_positive['mean_non_gold_entity_document_rate']:.4f}`",
        "",
        "| Retrieval | R@1 | R@3 | R@5 | R@10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("Caption baseline", "caption_baseline"),
        ("Entity lexical", "entity_lexical"),
        ("Caption + Entity RRF", "caption_entity_rrf"),
    ):
        row = retrieval[key]
        lines.append(
            f"| {label} | {_fmt(row['at_1'])} | {_fmt(row['at_3'])} | "
            f"{_fmt(row['at_5'])} | {_fmt(row['at_10'])} |"
        )
    lines.extend(
        [
            "",
            "| Case | Coverage | Same-entity docs | Non-gold rate | Base rank | Entity rank | Fused rank |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["case_level"]:
        lines.append(
            f"| {row['case_id']} | {int(bool(row['gold_anchor_entity_covered']))} | "
            f"{row['same_entity_occurrence_count']} | {row['non_gold_entity_document_rate']:.3f} | "
            f"{row['baseline_rank'] or 'NA'} | {row['entity_rank'] or 'NA'} | {row['fused_rank'] or 'NA'} |"
        )
    lines.extend(
        [
            "",
            "Endpoint values were not structural gates. No bounded search, QA, or judge was run.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(row: Mapping[str, Any]) -> str:
    return f"{row['count']}/{row['case_count']}"


def _manifest_case_ids(payload: Any) -> tuple[str, ...]:
    raw = (
        payload.get("case_ids", payload.get("cases", ()))
        if isinstance(payload, Mapping)
        else payload
    )
    return tuple(
        str(
            row.get("case_id", row.get("id", ""))
            if isinstance(row, Mapping)
            else row
        )
        for row in tuple(raw or ())
        if str(
            row.get("case_id", row.get("id", ""))
            if isinstance(row, Mapping)
            else row
        )
    )


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    return tuple(
        dict(value)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and isinstance((value := json.loads(line)), Mapping)
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = _read_json_value(path)
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _read_json_value(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require_sha(path: Path, expected: str, label: str) -> None:
    if file_sha256(path) != str(expected):
        raise ValueError(f"{label} SHA256 mismatch")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--case-manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--protocol-spec", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--relation-spec", required=True)
    parser.add_argument("--expected-relation-spec-sha256", required=True)
    parser.add_argument("--anchor-query-spec", required=True)
    parser.add_argument("--expected-anchor-query-spec-sha256", required=True)
    parser.add_argument("--field-spec", required=True)
    parser.add_argument("--expected-field-spec-sha256", required=True)
    parser.add_argument("--case-ids", nargs="+", required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--sidecar-root", required=True)
    parser.add_argument("--expected-sidecar-source-commit", required=True)
    parser.add_argument("--expected-ocr-model", default="pa/gmn-2.5-pr")
    parser.add_argument("--expected-passages", type=int, default=2960)
    parser.add_argument("--caption-config-digest", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rrf-k0", type=int, default=60)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-revision", required=True)
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    return parser.parse_args()


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
