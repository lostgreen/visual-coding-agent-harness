from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Sequence

from vcah.caption_lexical_index import CaptionLexicalIndex, normalize_caption_query
from vcah.caption_schema import CaptionHitV1, stable_digest
from vcah.caption_semantic_index import (
    CaptionSemanticIndex,
    expand_passage_neighbors,
    passage_interval_iou,
)
from vcah.embedding_adapter import TextEmbeddingAdapter


class CaptionHybridSearch:
    def __init__(
        self,
        lexical: CaptionLexicalIndex,
        dense: CaptionSemanticIndex,
        *,
        lexical_weight: float = 1.0,
        dense_weight: float = 1.0,
        rrf_k0: int = 60,
        query_strategy: str = "joint",
    ) -> None:
        if lexical.config_digest != dense.config_digest:
            raise ValueError("lexical and dense caption indexes use different caption configs")
        lexical_ids = tuple(passage.passage_id for passage in lexical.passages)
        dense_ids = tuple(passage.passage_id for passage in dense.passages)
        if lexical_ids != dense_ids:
            raise ValueError("lexical and dense caption indexes contain different passages")
        self.lexical = lexical
        self.dense = dense
        self.passages = dense.passages
        self.config_digest = dense.config_digest
        self.lexical_weight = max(0.0, float(lexical_weight))
        self.dense_weight = max(0.0, float(dense_weight))
        self.rrf_k0 = max(1, int(rrf_k0))
        self.query_strategy = str(query_strategy or "joint").strip().casefold()
        if self.query_strategy not in {"joint", "rema"}:
            raise ValueError(f"unsupported hybrid query strategy: {self.query_strategy}")
        if self.lexical_weight == 0.0 and self.dense_weight == 0.0:
            raise ValueError("hybrid search requires at least one positive rank weight")
        digest_payload = {
            "index_mode": "hybrid_rrf",
            "lexical_index_digest": lexical.index_digest,
            "dense_index_digest": dense.index_digest,
            "lexical_weight": self.lexical_weight,
            "dense_weight": self.dense_weight,
            "rrf_k0": self.rrf_k0,
        }
        if self.query_strategy != "joint":
            digest_payload["query_strategy"] = self.query_strategy
        self.index_digest = stable_digest(digest_payload)

    @classmethod
    def from_asset_root(
        cls,
        asset_root: Path,
        *,
        adapter: TextEmbeddingAdapter,
        config_digest: str | None = None,
        rebuild_dense: bool = False,
        lexical_weight: float = 1.0,
        dense_weight: float = 1.0,
        rrf_k0: int = 60,
        query_strategy: str = "joint",
    ) -> "CaptionHybridSearch":
        lexical = CaptionLexicalIndex.from_asset_root(asset_root, config_digest=config_digest)
        dense = CaptionSemanticIndex.from_asset_root(
            asset_root,
            adapter=adapter,
            config_digest=lexical.config_digest,
            rebuild=rebuild_dense,
        )
        return cls(
            lexical,
            dense,
            lexical_weight=lexical_weight,
            dense_weight=dense_weight,
            rrf_k0=rrf_k0,
            query_strategy=query_strategy,
        )

    def search(
        self,
        queries: Sequence[str],
        *,
        top_k: int = 12,
        time_range: tuple[float, float] | None = None,
        segment_ids: Sequence[str] = (),
        expand_neighbors: int = 0,
        per_caption_limit: int = 3,
        temporal_iou_threshold: float = 0.9,
    ) -> tuple[CaptionHitV1, ...]:
        if self.query_strategy == "rema":
            return self._search_independent(
                queries,
                top_k=top_k,
                time_range=time_range,
                segment_ids=segment_ids,
                expand_neighbors=expand_neighbors,
                per_caption_limit=per_caption_limit,
                temporal_iou_threshold=temporal_iou_threshold,
            )
        return self._search_joint(
            queries,
            top_k=top_k,
            time_range=time_range,
            segment_ids=segment_ids,
            expand_neighbors=expand_neighbors,
            per_caption_limit=per_caption_limit,
            temporal_iou_threshold=temporal_iou_threshold,
        )

    def _search_joint(
        self,
        queries: Sequence[str],
        *,
        top_k: int,
        time_range: tuple[float, float] | None,
        segment_ids: Sequence[str],
        expand_neighbors: int,
        per_caption_limit: int,
        temporal_iou_threshold: float,
    ) -> tuple[CaptionHitV1, ...]:
        result_limit = max(1, int(top_k))
        candidate_limit = max(20, result_limit * 4)
        lexical_hits = self.lexical.search(
            queries,
            top_k=candidate_limit,
            time_range=time_range,
            segment_ids=segment_ids,
            expand_neighbors=0,
            per_caption_limit=max(per_caption_limit, candidate_limit),
            temporal_iou_threshold=1.01,
        )
        dense_hits = self.dense.search(
            queries,
            top_k=candidate_limit,
            time_range=time_range,
            segment_ids=segment_ids,
            expand_neighbors=0,
            per_caption_limit=max(per_caption_limit, candidate_limit),
            temporal_iou_threshold=1.01,
        )
        lexical_by_id = {hit.passage_id: hit for hit in lexical_hits}
        dense_by_id = {hit.passage_id: hit for hit in dense_hits}
        passage_by_id = {passage.passage_id: passage for passage in self.passages}
        scores: dict[str, float] = {}
        for hit in lexical_hits:
            scores[hit.passage_id] = scores.get(hit.passage_id, 0.0) + (
                self.lexical_weight / (self.rrf_k0 + hit.rank)
            )
        for hit in dense_hits:
            scores[hit.passage_id] = scores.get(hit.passage_id, 0.0) + (
                self.dense_weight / (self.rrf_k0 + hit.rank)
            )
        ordered = sorted(
            scores,
            key=lambda passage_id: (
                -scores[passage_id],
                passage_by_id[passage_id].virtual_start_sec,
                passage_id,
            ),
        )
        selected_ids: list[str] = []
        caption_counts: Counter[str] = Counter()
        for passage_id in ordered:
            passage = passage_by_id[passage_id]
            if caption_counts[passage.caption_id] >= max(1, int(per_caption_limit)):
                continue
            if any(
                passage_interval_iou(passage, passage_by_id[other_id])
                >= float(temporal_iou_threshold)
                for other_id in selected_ids
            ):
                continue
            selected_ids.append(passage_id)
            caption_counts[passage.caption_id] += 1
            if len(selected_ids) >= result_limit:
                break
        hits = [
            self._hit(
                passage_id,
                rank=rank,
                fused_score=scores[passage_id],
                lexical_hit=lexical_by_id.get(passage_id),
                dense_hit=dense_by_id.get(passage_id),
            )
            for rank, passage_id in enumerate(selected_ids, start=1)
        ]
        if expand_neighbors > 0:
            hits = expand_passage_neighbors(
                self.passages,
                hits,
                distance=int(expand_neighbors),
                time_range=time_range,
                segment_ids=segment_ids,
                index_digest=self.index_digest,
                config_digest=self.config_digest,
            )
        return tuple(
            CaptionHitV1(**{**asdict(hit), "rank": rank})
            for rank, hit in enumerate(hits, start=1)
        )

    def _search_independent(
        self,
        queries: Sequence[str],
        *,
        top_k: int,
        time_range: tuple[float, float] | None,
        segment_ids: Sequence[str],
        expand_neighbors: int,
        per_caption_limit: int,
        temporal_iou_threshold: float,
    ) -> tuple[CaptionHitV1, ...]:
        normalized_queries = tuple(
            dict.fromkeys(
                normalize_caption_query(query)
                for query in queries
                if normalize_caption_query(query)
            )
        )[:5]
        if not normalized_queries:
            return ()
        result_limit = max(1, int(top_k))
        per_query_hits = tuple(
            self._search_joint(
                (query,),
                top_k=result_limit,
                time_range=time_range,
                segment_ids=segment_ids,
                expand_neighbors=0,
                per_caption_limit=per_caption_limit,
                temporal_iou_threshold=temporal_iou_threshold,
            )
            for query in normalized_queries
        )
        matches: dict[str, list[dict[str, Any]]] = {}
        for query, hits in zip(normalized_queries, per_query_hits):
            for hit in hits:
                matches.setdefault(hit.passage_id, []).append(
                    {"query": query, "rank": hit.rank}
                )

        passage_by_id = {passage.passage_id: passage for passage in self.passages}
        selected: list[CaptionHitV1] = []
        seen: set[str] = set()
        caption_counts: Counter[str] = Counter()
        max_depth = max((len(hits) for hits in per_query_hits), default=0)
        for depth in range(max_depth):
            for hits in per_query_hits:
                if depth >= len(hits):
                    continue
                hit = hits[depth]
                if hit.passage_id in seen:
                    continue
                passage = passage_by_id[hit.passage_id]
                if caption_counts[passage.caption_id] >= max(1, int(per_caption_limit)):
                    continue
                if any(
                    passage_interval_iou(passage, passage_by_id[existing.passage_id])
                    >= float(temporal_iou_threshold)
                    for existing in selected
                ):
                    continue
                global_rank = len(selected) + 1
                selected.append(
                    CaptionHitV1(
                        **{
                            **asdict(hit),
                            "rank": global_rank,
                            "fused_score": 1.0 / (self.rrf_k0 + global_rank),
                            "metadata": {
                                **dict(hit.metadata),
                                "query_strategy": "rema",
                                "balanced_rank": global_rank,
                                "component_fused_score": hit.fused_score,
                                "query_matches": matches.get(hit.passage_id, ()),
                            },
                        }
                    )
                )
                seen.add(hit.passage_id)
                caption_counts[passage.caption_id] += 1
                if len(selected) >= result_limit:
                    break
            if len(selected) >= result_limit:
                break
        hits = selected
        if expand_neighbors > 0:
            hits = expand_passage_neighbors(
                self.passages,
                hits,
                distance=int(expand_neighbors),
                time_range=time_range,
                segment_ids=segment_ids,
                index_digest=self.index_digest,
                config_digest=self.config_digest,
            )
        return tuple(
            CaptionHitV1(**{**asdict(hit), "rank": rank})
            for rank, hit in enumerate(hits, start=1)
        )

    def query_fingerprint(
        self,
        queries: Sequence[str],
        *,
        top_k: int,
        time_range: tuple[float, float] | None,
        expand_neighbors: int,
        segment_ids: Sequence[str] = (),
    ) -> str:
        payload = {
            "index_mode": "hybrid",
            "index_digest": self.index_digest,
            "queries": [normalize_caption_query(query) for query in queries],
            "top_k": int(top_k),
            "time_range": list(time_range) if time_range else None,
            "expand_neighbors": int(expand_neighbors),
            "segment_ids": sorted(
                {str(item).strip() for item in segment_ids if str(item).strip()}
            ),
        }
        if self.query_strategy != "joint":
            payload["query_strategy"] = self.query_strategy
        return stable_digest(payload)

    def save_manifest(self, asset_root: Path) -> Path:
        path = (
            Path(asset_root)
            / "captions"
            / "hybrid"
            / self.config_digest
            / f"index.{self.index_digest[:20]}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "index_digest": self.index_digest,
                        "config_digest": self.config_digest,
                        "passage_count": len(self.passages),
                        "lexical_index_digest": self.lexical.index_digest,
                        "dense_index_digest": self.dense.index_digest,
                        "lexical_weight": self.lexical_weight,
                        "dense_weight": self.dense_weight,
                        "rrf_k0": self.rrf_k0,
                        "query_strategy": self.query_strategy,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        return path

    def _hit(
        self,
        passage_id: str,
        *,
        rank: int,
        fused_score: float,
        lexical_hit: CaptionHitV1 | None,
        dense_hit: CaptionHitV1 | None,
    ) -> CaptionHitV1:
        source = lexical_hit or dense_hit
        if source is None:
            raise ValueError(f"hybrid passage is missing from component results: {passage_id}")
        return CaptionHitV1(
            passage_id=source.passage_id,
            caption_id=source.caption_id,
            rank=rank,
            lexical_score=lexical_hit.lexical_score if lexical_hit else None,
            dense_score=dense_hit.dense_score if dense_hit else None,
            fused_score=fused_score,
            virtual_start_sec=source.virtual_start_sec,
            virtual_end_sec=source.virtual_end_sec,
            wall_clock_begin=source.wall_clock_begin,
            wall_clock_end=source.wall_clock_end,
            text=source.text,
            interval_precision=source.interval_precision,
            source_pointer=source.source_pointer,
            metadata={
                **dict(source.metadata),
                "index_digest": self.index_digest,
                "index_mode": "hybrid",
                "query_strategy": self.query_strategy,
                "lexical_rank": lexical_hit.rank if lexical_hit else None,
                "dense_rank": dense_hit.rank if dense_hit else None,
            },
        )
