#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from vcah.caption_hybrid_search import CaptionHybridSearch
from vcah.caption_lexical_index import CaptionLexicalIndex
from vcah.caption_occurrence import cluster_caption_occurrences
from vcah.caption_schema import CaptionHitV1
from vcah.caption_semantic_index import CaptionSemanticIndex
from vcah.embedding_adapter import SentenceTransformerEmbeddingAdapter


def evaluate_hits(
    hits: Sequence[CaptionHitV1 | Mapping[str, Any]],
    clue_intervals: Sequence[Sequence[float]],
) -> dict[str, Any]:
    clues = tuple(
        tuple(sorted((float(value[0]), float(value[1]))))
        for value in clue_intervals
        if len(value) == 2
    )
    normalized_hits = tuple(_hit_interval(hit) for hit in hits)
    recalled = sum(
        any(_overlap(clue, interval) for interval in normalized_hits)
        for clue in clues
    )
    occurrences = cluster_caption_occurrences(hits)
    occurrence_recalled = sum(
        any(
            _overlap(
                clue,
                (candidate.virtual_start_sec, candidate.virtual_end_sec),
            )
            for candidate in occurrences
        )
        for clue in clues
    )
    clue_count = len(clues)
    candidate_count = len(hits)
    return {
        "candidate_clue_recall": recalled / clue_count if clue_count else 0.0,
        "occurrence_candidate_recall": (
            occurrence_recalled / clue_count if clue_count else 0.0
        ),
        "candidate_count": candidate_count,
        "occurrence_candidate_count": len(occurrences),
        "recall_per_candidate": (
            (recalled / clue_count) / candidate_count
            if clue_count and candidate_count
            else 0.0
        ),
        "recalled_clue_count": recalled,
        "clue_count": clue_count,
    }


def select_promotion(
    variants: Sequence[Mapping[str, Any]],
    *,
    baseline_name: str = "current_hybrid",
) -> dict[str, Any]:
    baseline = next(
        (dict(row) for row in variants if row.get("name") == baseline_name),
        None,
    )
    if baseline is None:
        raise ValueError(f"baseline variant missing: {baseline_name}")
    baseline_recall = float(baseline.get("candidate_clue_recall", 0.0) or 0.0)
    eligible = [
        dict(row)
        for row in variants
        if row.get("status", "ok") == "ok"
        and float(row.get("candidate_clue_recall", 0.0) or 0.0) > baseline_recall
        and float(row.get("occurrence_candidate_recall", 0.0) or 0.0)
        >= float(row.get("candidate_clue_recall", 0.0) or 0.0)
    ]
    selected = min(
        eligible,
        key=lambda row: (
            -float(row.get("candidate_clue_recall", 0.0) or 0.0),
            int(row.get("candidate_count", 0) or 0),
            -float(row.get("recall_per_candidate", 0.0) or 0.0),
            str(row.get("name", "")),
        ),
        default=None,
    )
    return {
        "promotion_eligible": selected is not None,
        "selected_variant": str(selected.get("name", "")) if selected else "",
        "baseline_candidate_clue_recall": baseline_recall,
        "selected_candidate_clue_recall": (
            float(selected.get("candidate_clue_recall", 0.0) or 0.0)
            if selected
            else baseline_recall
        ),
        "reason": (
            "offline_candidate_recall_improved"
            if selected
            else "no_offline_candidate_recall_improvement"
        ),
    }


def run_ablation(
    *,
    asset_root: Path,
    config_digest: str,
    queries: Sequence[str],
    target_queries: Sequence[str],
    clue_intervals: Sequence[Sequence[float]],
    embedding_model: str,
    embedding_revision: str | None = None,
    multilingual_embedding_model: str = "",
    multilingual_embedding_revision: str | None = None,
    device: str = "cpu",
    batch_size: int = 64,
) -> dict[str, Any]:
    base_queries = _queries(queries)
    # The production searcher accepts at most five independent queries. Put the
    # target-side additions first so the ablation actually exercises them.
    union_queries = _queries((*target_queries, *queries))
    adapter = SentenceTransformerEmbeddingAdapter(
        embedding_model,
        revision=embedding_revision,
        device=device,
        normalize=True,
        batch_size=batch_size,
    )
    lexical = CaptionLexicalIndex.from_asset_root(
        asset_root,
        config_digest=config_digest,
    )
    dense = CaptionSemanticIndex.from_asset_root(
        asset_root,
        adapter=adapter,
        config_digest=lexical.config_digest,
    )
    current = CaptionHybridSearch(lexical, dense, query_strategy="rema")
    variants: list[dict[str, Any]] = []

    def evaluate(
        name: str,
        search: CaptionHybridSearch,
        variant_queries: Sequence[str],
        *,
        top_k: int,
        expand_neighbors: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        hits = search.search(
            variant_queries,
            top_k=top_k,
            expand_neighbors=expand_neighbors,
        )
        variants.append(
            {
                "name": name,
                "status": "ok",
                "queries": list(variant_queries),
                "top_k": top_k,
                "expand_neighbors": expand_neighbors,
                "index_digest": search.index_digest,
                **evaluate_hits(hits, clue_intervals),
                "hit_intervals": [list(_hit_interval(hit)) for hit in hits],
                "hit_passage_ids": [hit.passage_id for hit in hits],
                **dict(metadata or {}),
            }
        )

    evaluate("current_hybrid", current, base_queries, top_k=12, expand_neighbors=0)
    evaluate("target_query_union", current, union_queries, top_k=12, expand_neighbors=0)
    evaluate("neighbor_expansion", current, union_queries, top_k=12, expand_neighbors=1)
    evaluate("higher_candidate_k", current, union_queries, top_k=32, expand_neighbors=1)

    augmented_passages, title_coverage = _with_source_titles(
        asset_root,
        lexical.passages,
    )
    title_lexical = CaptionLexicalIndex(
        augmented_passages,
        config_digest=lexical.config_digest,
    )
    title_search = CaptionHybridSearch(title_lexical, dense, query_strategy="rema")
    evaluate(
        "source_title_metadata",
        title_search,
        union_queries,
        top_k=32,
        expand_neighbors=1,
        metadata={"source_title_passage_coverage": title_coverage},
    )

    if multilingual_embedding_model:
        multilingual_adapter = SentenceTransformerEmbeddingAdapter(
            multilingual_embedding_model,
            revision=multilingual_embedding_revision,
            device=device,
            normalize=True,
            batch_size=batch_size,
        )
        multilingual_dense = CaptionSemanticIndex.from_asset_root(
            asset_root,
            adapter=multilingual_adapter,
            config_digest=lexical.config_digest,
        )
        multilingual_search = CaptionHybridSearch(
            title_lexical,
            multilingual_dense,
            query_strategy="rema",
        )
        evaluate(
            "multilingual_embedding",
            multilingual_search,
            union_queries,
            top_k=32,
            expand_neighbors=1,
            metadata={
                "embedding": dict(multilingual_adapter.manifest),
                "source_title_passage_coverage": title_coverage,
            },
        )
    else:
        variants.append(
            {
                "name": "multilingual_embedding",
                "status": "not_run",
                "reason": "multilingual_embedding_model_not_configured",
            }
        )
    promotion = select_promotion(variants)
    return {
        "schema_version": "MGERLocatorAblationV1",
        "asset_root": str(Path(asset_root).resolve()),
        "caption_config_digest": lexical.config_digest,
        "base_queries": list(base_queries),
        "target_queries": list(_queries(target_queries)),
        "clue_intervals": [list(value) for value in clue_intervals],
        "embedding": dict(adapter.manifest),
        "variants": variants,
        "promotion": promotion,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the MGER Phase 4 offline Caption locator ablation."
    )
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--config-digest", required=True)
    parser.add_argument("--query", action="append", required=True)
    parser.add_argument("--target-query", action="append", default=[])
    parser.add_argument(
        "--clue-interval",
        action="append",
        nargs=2,
        type=float,
        required=True,
        metavar=("START", "END"),
    )
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-revision")
    parser.add_argument("--multilingual-embedding-model", default="")
    parser.add_argument("--multilingual-embedding-revision")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    result = run_ablation(
        asset_root=args.asset_root,
        config_digest=args.config_digest,
        queries=args.query,
        target_queries=args.target_query,
        clue_intervals=args.clue_interval,
        embedding_model=args.embedding_model,
        embedding_revision=args.embedding_revision,
        multilingual_embedding_model=args.multilingual_embedding_model,
        multilingual_embedding_revision=args.multilingual_embedding_revision,
        device=args.device,
        batch_size=args.batch_size,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


def _with_source_titles(asset_root: Path, passages: Sequence[Any]) -> tuple[tuple[Any, ...], float]:
    timeline = json.loads((Path(asset_root) / "virtual_timeline.json").read_text(encoding="utf-8"))
    titles = {
        str(segment.get("segment_id", "") or ""): str(
            segment.get("source_video_id")
            or segment.get("metadata", {}).get("relative_source_path")
            or ""
        )
        for segment in tuple(timeline.get("segments", ()) or ())
        if isinstance(segment, Mapping)
    }
    augmented = []
    covered = 0
    for passage in passages:
        segment_ids = tuple(passage.metadata.get("source_segments", ()) or ())
        passage_titles = tuple(
            dict.fromkeys(titles.get(str(segment_id), "") for segment_id in segment_ids)
        )
        passage_titles = tuple(title for title in passage_titles if title)
        if passage_titles:
            covered += 1
        augmented.append(
            replace(
                passage,
                text=(
                    f"{passage.text}\nSource title: {' | '.join(passage_titles)}"
                    if passage_titles
                    else passage.text
                ),
                metadata={
                    **dict(passage.metadata),
                    "source_titles": list(passage_titles),
                },
            )
        )
    return tuple(augmented), covered / len(passages) if passages else 0.0


def _queries(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))[:8]


def _hit_interval(hit: CaptionHitV1 | Mapping[str, Any]) -> tuple[float, float]:
    if isinstance(hit, CaptionHitV1):
        return hit.virtual_start_sec, hit.virtual_end_sec
    raw_range = tuple(hit.get("range", ()) or ())
    if len(raw_range) == 2:
        return tuple(sorted((float(raw_range[0]), float(raw_range[1]))))
    return tuple(
        sorted(
            (
                float(hit.get("virtual_start_sec", 0.0) or 0.0),
                float(hit.get("virtual_end_sec", 0.0) or 0.0),
            )
        )
    )


def _overlap(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return min(left[1], right[1]) >= max(left[0], right[0])


if __name__ == "__main__":
    raise SystemExit(main())
