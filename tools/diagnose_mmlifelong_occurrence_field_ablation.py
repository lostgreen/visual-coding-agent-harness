#!/usr/bin/env python3
"""Measure the oracle value of entity/event/state occurrence fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.diagnose_mmlifelong_occurrence_candidate_coverage import (
    _manifest_case_ids,
)
from tools.run_mmlifelong_gemini_ocr import _caption_packets
from vcah.caption_hybrid_search import CaptionHybridSearch
from vcah.caption_lexical_index import CaptionLexicalIndex
from vcah.caption_schema import CaptionHitV1, CaptionPassageV1, stable_digest
from vcah.caption_semantic_index import CaptionSemanticIndex
from vcah.embedding_adapter import SentenceTransformerEmbeddingAdapter
from vcah.occurrence_field_ablation import build_field_ablation_report
from vcah.occurrence_field_index import (
    OCCURRENCE_FIELDS,
    augment_oracle_occurrence_passage,
    normalize_occurrence_fields,
    occurrence_field_queries,
    oracle_field_passage,
    reciprocal_rank_fusion,
    select_oracle_occurrence_passage,
)
from vcah.virtual_video import VirtualVideoWorkspace


VARIANTS = (
    "caption_only_hybrid",
    "caption_plus_entity_lexical",
    "caption_plus_entity_hybrid",
    "caption_plus_event_lexical",
    "caption_plus_event_hybrid",
    "caption_plus_state_lexical",
    "caption_plus_state_hybrid",
    "caption_plus_all_lexical",
    "caption_plus_all_hybrid",
    "field_rrf_lexical",
    "field_rrf_hybrid",
)


def load_field_ablation_cases(
    run_root: Path,
    *,
    relation_spec: Mapping[str, Any],
    query_spec: Mapping[str, Any],
    field_spec: Mapping[str, Any],
    adapter: SentenceTransformerEmbeddingAdapter,
    case_ids: Sequence[str],
    top_k: int,
    rrf_k0: int,
) -> tuple[dict[str, Any], ...]:
    relation_cases = dict(relation_spec.get("cases", {}) or {})
    query_cases = dict(query_spec.get("cases", {}) or {})
    field_cases = dict(field_spec.get("cases", {}) or {})
    field_policy = dict(field_spec.get("annotation_policy", {}) or {})
    frozen = bool(field_spec.get("fields_frozen_before_ablation_outcomes"))
    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        relation = relation_cases.get(case_id)
        query_case = query_cases.get(case_id)
        field_case = field_cases.get(case_id)
        if not all(isinstance(value, Mapping) for value in (relation, query_case, field_case)):
            raise ValueError(f"missing field ablation specification: {case_id}")
        normalized_fields = normalize_occurrence_fields(field_case)
        run_dir = Path(run_root) / "cases" / case_id
        workspace = VirtualVideoWorkspace.load(run_dir)
        packets = _caption_packets(run_dir)
        config_digests = {
            str(packet.get("config_digest", "") or "") for packet in packets
        }
        if len(config_digests) != 1 or not all(config_digests):
            raise ValueError(f"expected one frozen Caption config: {case_id}")
        config_digest = next(iter(config_digests))
        lexical = CaptionLexicalIndex.from_asset_root(
            workspace.asset_root,
            config_digest=config_digest,
        )
        dense = CaptionSemanticIndex.from_asset_root(
            workspace.asset_root,
            adapter=adapter,
            config_digest=config_digest,
        )
        if tuple(p.passage_id for p in lexical.passages) != tuple(
            p.passage_id for p in dense.passages
        ):
            raise ValueError(f"Caption index passage mismatch: {case_id}")
        anchor_intervals = tuple(relation.get("anchor_intervals", ()) or ())
        oracle_passage = select_oracle_occurrence_passage(
            lexical.passages,
            anchor_intervals,
        )
        if oracle_passage is None:
            raise ValueError(f"oracle anchor has no Caption passage: {case_id}")
        anchor_query = str(query_case.get("anchor_query", "") or "").strip()
        if not anchor_query:
            raise ValueError(f"empty anchor query: {case_id}")
        ranks = _case_ranks(
            lexical=lexical,
            dense=dense,
            adapter=adapter,
            oracle_passage=oracle_passage,
            fields=normalized_fields,
            anchor_query=(anchor_query,),
            anchor_intervals=anchor_intervals,
            top_k=top_k,
            rrf_k0=rrf_k0,
        )
        rows.append(
            {
                "case_id": case_id,
                "anchor_description": str(
                    field_case.get("anchor_description", "") or ""
                ),
                "anchor_intervals": [
                    list(value)
                    for value in tuple(relation.get("anchor_intervals", ()) or ())
                ],
                "oracle_passage_id": oracle_passage.passage_id,
                "field_names": list(normalized_fields),
                "fields_frozen_before_ablation_outcomes": frozen,
                "oracle_gold_occurrence_only": bool(
                    field_policy.get("gold_occurrence_only")
                ),
                "target_evidence_and_answer_excluded": bool(
                    field_policy.get(
                        "target_evidence_and_reference_answer_values_excluded"
                    )
                ),
                "ranks": ranks,
            }
        )
    return tuple(rows)


def _case_ranks(
    *,
    lexical: CaptionLexicalIndex,
    dense: CaptionSemanticIndex,
    adapter: SentenceTransformerEmbeddingAdapter,
    oracle_passage: CaptionPassageV1,
    fields: Mapping[str, Mapping[str, Sequence[str]]],
    anchor_query: Sequence[str],
    anchor_intervals: Sequence[Sequence[float]],
    top_k: int,
    rrf_k0: int,
) -> dict[str, int | None]:
    baseline_hybrid = CaptionHybridSearch(lexical, dense, query_strategy="joint")
    ranks: dict[str, int | None] = {
        "caption_only_hybrid": _hit_rank(
            baseline_hybrid.search(anchor_query, top_k=top_k),
            anchor_intervals,
        )
    }
    for field_name in OCCURRENCE_FIELDS:
        augmented = augment_oracle_occurrence_passage(
            lexical.passages,
            oracle_passage_id=oracle_passage.passage_id,
            fields=fields,
            selected_fields=(field_name,),
        )
        field_lexical, field_hybrid = _augmented_indexes(
            augmented,
            base_dense=dense,
            adapter=adapter,
            oracle_passage_id=oracle_passage.passage_id,
            label=f"concat-{field_name}",
        )
        ranks[f"caption_plus_{field_name}_lexical"] = _hit_rank(
            field_lexical.search(anchor_query, top_k=top_k),
            anchor_intervals,
        )
        ranks[f"caption_plus_{field_name}_hybrid"] = _hit_rank(
            field_hybrid.search(anchor_query, top_k=top_k),
            anchor_intervals,
        )

    augmented_all = augment_oracle_occurrence_passage(
        lexical.passages,
        oracle_passage_id=oracle_passage.passage_id,
        fields=fields,
        selected_fields=OCCURRENCE_FIELDS,
    )
    all_lexical, all_hybrid = _augmented_indexes(
        augmented_all,
        base_dense=dense,
        adapter=adapter,
        oracle_passage_id=oracle_passage.passage_id,
        label="concat-all",
    )
    ranks["caption_plus_all_lexical"] = _hit_rank(
        all_lexical.search(anchor_query, top_k=top_k),
        anchor_intervals,
    )
    ranks["caption_plus_all_hybrid"] = _hit_rank(
        all_hybrid.search(anchor_query, top_k=top_k),
        anchor_intervals,
    )

    depth = max(50, top_k * 5)
    caption_lexical_ids = _hit_ids(
        lexical.search(
            anchor_query,
            top_k=depth,
            per_caption_limit=depth,
            temporal_iou_threshold=1.01,
        )
    )
    caption_hybrid_ids = _hit_ids(
        baseline_hybrid.search(
            anchor_query,
            top_k=depth,
            per_caption_limit=depth,
            temporal_iou_threshold=1.01,
        )
    )
    queries = occurrence_field_queries(fields)
    lexical_rankings: dict[str, tuple[str, ...]] = {
        "caption": caption_lexical_ids
    }
    hybrid_rankings: dict[str, tuple[str, ...]] = {"caption": caption_hybrid_ids}
    for field_name in OCCURRENCE_FIELDS:
        passage = oracle_field_passage(
            oracle_passage,
            field_name=field_name,
            fields=fields,
        )
        config_digest = stable_digest(
            {
                "base": lexical.config_digest,
                "field": field_name,
                "passage": passage.text,
            }
        )
        field_lexical = CaptionLexicalIndex((passage,), config_digest=config_digest)
        field_dense = CaptionSemanticIndex.build(
            (passage,),
            adapter=adapter,
            config_digest=config_digest,
        )
        field_hybrid = CaptionHybridSearch(field_lexical, field_dense)
        lexical_rankings[field_name] = _hit_ids(
            field_lexical.search(queries[field_name], top_k=1)
        )
        hybrid_rankings[field_name] = _hit_ids(
            field_hybrid.search(queries[field_name], top_k=1)
        )
    ranks["field_rrf_lexical"] = _passage_ids_overlap_rank(
        reciprocal_rank_fusion(lexical_rankings, rrf_k0=rrf_k0),
        passages=lexical.passages,
        anchor_intervals=anchor_intervals,
    )
    ranks["field_rrf_hybrid"] = _passage_ids_overlap_rank(
        reciprocal_rank_fusion(hybrid_rankings, rrf_k0=rrf_k0),
        passages=lexical.passages,
        anchor_intervals=anchor_intervals,
    )
    return ranks


def _augmented_indexes(
    passages: Sequence[CaptionPassageV1],
    *,
    base_dense: CaptionSemanticIndex,
    adapter: SentenceTransformerEmbeddingAdapter,
    oracle_passage_id: str,
    label: str,
) -> tuple[CaptionLexicalIndex, CaptionHybridSearch]:
    config_digest = stable_digest(
        {
            "base": base_dense.config_digest,
            "contract": "WP16-5-oracle-field-ablation-v1",
            "label": label,
            "oracle_passage_id": oracle_passage_id,
        }
    )
    lexical = CaptionLexicalIndex(passages, config_digest=config_digest)
    vectors = np.asarray(base_dense.vectors, dtype=np.float32).copy()
    passage_index = {
        passage.passage_id: index for index, passage in enumerate(passages)
    }
    oracle_index = passage_index.get(str(oracle_passage_id))
    if oracle_index is None:
        raise ValueError("augmented oracle passage is absent")
    vectors[oracle_index] = np.asarray(
        adapter.embed_documents([passages[oracle_index].text])[0],
        dtype=np.float32,
    )
    dense = CaptionSemanticIndex(
        passages,
        vectors,
        adapter=adapter,
        config_digest=config_digest,
    )
    return lexical, CaptionHybridSearch(lexical, dense, query_strategy="joint")


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# MM-Lifelong WP16-5 Oracle Field Ablation",
        "",
        "## 结论",
        "",
        f"`{report['decision']}`",
        "",
        "本轮只测 entity/event/state 对 Anchor Recall 的 oracle 上界，不运行 evidence search 或 QA。",
        "",
        "## Anchor Recall",
        "",
        "| Variant | R@1 | R@3 | R@5 | R@10 | MRR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, summary in report["variants"].items():
        recalls = [
            summary["recall"][f"at_{k}"]["count"] for k in (1, 3, 5, 10)
        ]
        lines.append(
            f"| `{name}` | {recalls[0]}/{report['case_count']} | "
            f"{recalls[1]}/{report['case_count']} | {recalls[2]}/{report['case_count']} | "
            f"{recalls[3]}/{report['case_count']} | "
            f"{summary['mean_reciprocal_rank']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 逐题排名",
            "",
            "| Case | Base | Entity H | Event H | State H | All L | All H | Field RRF H |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case in report["per_case"]:
        ranks = case["ranks"]

        def rank(name: str) -> str:
            value = ranks.get(name)
            return str(value) if value is not None else "-"

        lines.append(
            f"| `{case['case_id'][-4:]}` | {rank('caption_only_hybrid')} | "
            f"{rank('caption_plus_entity_hybrid')} | "
            f"{rank('caption_plus_event_hybrid')} | "
            f"{rank('caption_plus_state_hybrid')} | "
            f"{rank('caption_plus_all_lexical')} | "
            f"{rank('caption_plus_all_hybrid')} | "
            f"{rank('field_rrf_hybrid')} |"
        )
    lines.extend(
        [
            "",
            "## 有效性边界",
            "",
            f"- Structural gate: {report['structural_gate_passed']}",
            "- Gold interval 用于选择并增强一个 oracle occurrence passage。",
            "- Query fields 是人工 parser 上界；document fields 是 perfect extraction 上界。",
            "- Non-gold field false positives 未测，因此不能把结果当部署性能。",
            "- Target evidence 与参考答案值没有进入字段。",
            "- Endpoint 不是结构 gate；frozen10 underpowered。",
            "- Generative/VLM/judge calls 均为 0。",
        ]
    )
    return "\n".join(lines) + "\n"


def _hit_rank(
    hits: Sequence[CaptionHitV1],
    anchor_intervals: Sequence[Sequence[float]],
) -> int | None:
    for hit in hits:
        if _overlaps_anchor(
            hit.virtual_start_sec,
            hit.virtual_end_sec,
            anchor_intervals,
        ):
            return int(hit.rank)
    return None


def _hit_ids(hits: Sequence[CaptionHitV1]) -> tuple[str, ...]:
    return tuple(hit.passage_id for hit in hits)


def _passage_ids_overlap_rank(
    passage_ids: Sequence[str],
    *,
    passages: Sequence[CaptionPassageV1],
    anchor_intervals: Sequence[Sequence[float]],
) -> int | None:
    passage_by_id = {passage.passage_id: passage for passage in passages}
    for rank, passage_id in enumerate(passage_ids, start=1):
        passage = passage_by_id.get(str(passage_id))
        if passage is not None and _overlaps_anchor(
            passage.virtual_start_sec,
            passage.virtual_end_sec,
            anchor_intervals,
        ):
            return rank
    return None


def _overlaps_anchor(
    start: float,
    end: float,
    anchor_intervals: Sequence[Sequence[float]],
) -> bool:
    left_start, left_end = sorted((float(start), float(end)))
    for value in anchor_intervals:
        if len(value) != 2:
            continue
        right_start, right_end = sorted((float(value[0]), float(value[1])))
        if min(left_end, right_end) - max(left_start, right_start) > 0.0:
            return True
    return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--case-manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--relation-spec", required=True)
    parser.add_argument("--expected-relation-spec-sha256", required=True)
    parser.add_argument("--anchor-query-spec", required=True)
    parser.add_argument("--expected-anchor-query-spec-sha256", required=True)
    parser.add_argument("--field-spec", required=True)
    parser.add_argument("--expected-field-spec-sha256", required=True)
    parser.add_argument("--case-ids", nargs="+", required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
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
    args = _parse_args()
    paths = {
        "manifest": Path(args.case_manifest),
        "relation_spec": Path(args.relation_spec),
        "anchor_query_spec": Path(args.anchor_query_spec),
        "field_spec": Path(args.field_spec),
    }
    expected = {
        "manifest": args.expected_manifest_sha256,
        "relation_spec": args.expected_relation_spec_sha256,
        "anchor_query_spec": args.expected_anchor_query_spec_sha256,
        "field_spec": args.expected_field_spec_sha256,
    }
    for label, path in paths.items():
        _require_sha(path, expected[label], label)
    case_ids = tuple(dict.fromkeys(str(value) for value in args.case_ids))
    manifest_ids = set(_manifest_case_ids(_read_json_value(paths["manifest"])))
    if len(case_ids) != int(args.expected_cases) or not set(case_ids) <= manifest_ids:
        raise ValueError("selected cases do not match frozen manifest subset")
    adapter = SentenceTransformerEmbeddingAdapter(
        str(args.embedding_model),
        revision=str(args.embedding_revision),
        device=str(args.embedding_device),
        normalize=True,
        batch_size=max(1, int(args.embedding_batch_size)),
    )
    cases = load_field_ablation_cases(
        Path(args.run_root),
        relation_spec=_read_json(paths["relation_spec"]),
        query_spec=_read_json(paths["anchor_query_spec"]),
        field_spec=_read_json(paths["field_spec"]),
        adapter=adapter,
        case_ids=case_ids,
        top_k=max(1, int(args.top_k)),
        rrf_k0=max(1, int(args.rrf_k0)),
    )
    report = build_field_ablation_report(
        cases,
        expected_cases=int(args.expected_cases),
        variant_order=VARIANTS,
    )
    report["provenance"] = {
        "label": str(args.provenance),
        "source_run_root": str(Path(args.run_root)),
        **{
            f"{label}_path": str(path)
            for label, path in paths.items()
        },
        **{
            f"{label}_sha256": _sha256(path)
            for label, path in paths.items()
        },
        "embedding": dict(adapter.manifest),
        "top_k": max(1, int(args.top_k)),
        "rrf_k0": max(1, int(args.rrf_k0)),
        "generative_model_calls": 0,
        "vlm_calls": 0,
        "judge_calls": 0,
    }
    _write_json(Path(args.out_json), report)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(report), encoding="utf-8")
    best = report["diagnostics"]["best_single_field_at5_count"]
    print(
        "FIELD_ABLATION_DONE "
        f"decision={report['decision']} "
        f"structural={report['structural_gate_passed']} "
        f"best_single_at5={best}/{report['case_count']}",
        flush=True,
    )


def _require_sha(path: Path, expected: str, label: str) -> None:
    if _sha256(path) != str(expected):
        raise ValueError(f"{label} SHA256 mismatch")


def _read_json(path: Path) -> dict[str, Any]:
    value = _read_json_value(path)
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _read_json_value(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
