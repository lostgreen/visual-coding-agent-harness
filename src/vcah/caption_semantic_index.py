from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from vcah.caption_lexical_index import normalize_caption_query
from vcah.caption_schema import CaptionHitV1, CaptionPassageV1, passage_from_dict, stable_digest
from vcah.caption_store import resolve_caption_passages_path
from vcah.embedding_adapter import TextEmbeddingAdapter, normalize_rows


SEMANTIC_INDEX_SCHEMA_VERSION = 1


class CaptionSemanticIndex:
    def __init__(
        self,
        passages: Sequence[CaptionPassageV1],
        vectors: np.ndarray,
        *,
        adapter: TextEmbeddingAdapter,
        config_digest: str,
        cache_manifest: Mapping[str, Any] | None = None,
    ) -> None:
        self.passages = tuple(passages)
        self.adapter = adapter
        self.config_digest = str(config_digest)
        matrix = np.asarray(vectors, dtype=np.float32)
        expected = (len(self.passages), int(adapter.dimension))
        if matrix.shape != expected:
            raise ValueError(f"caption embedding shape mismatch: expected {expected}, got {matrix.shape}")
        self.vectors = normalize_rows(matrix) if len(matrix) else matrix
        self.adapter_manifest = _adapter_manifest(adapter)
        self.passages_digest = stable_digest([asdict(passage) for passage in self.passages])
        self.index_digest = stable_digest(
            {
                "schema_version": SEMANTIC_INDEX_SCHEMA_VERSION,
                "config_digest": self.config_digest,
                "passages_digest": self.passages_digest,
                "adapter": self.adapter_manifest,
                "cosine_normalize": True,
            }
        )
        self.cache_manifest = dict(cache_manifest or {})

    @classmethod
    def build(
        cls,
        passages: Sequence[CaptionPassageV1],
        *,
        adapter: TextEmbeddingAdapter,
        config_digest: str,
    ) -> "CaptionSemanticIndex":
        vectors = adapter.embed_documents([passage.text for passage in passages])
        return cls(passages, vectors, adapter=adapter, config_digest=config_digest)

    @classmethod
    def from_asset_root(
        cls,
        asset_root: Path,
        *,
        adapter: TextEmbeddingAdapter,
        config_digest: str | None = None,
        rebuild: bool = False,
    ) -> "CaptionSemanticIndex":
        passages, resolved_digest, _ = load_caption_passages(asset_root, config_digest=config_digest)
        adapter_manifest = _adapter_manifest(adapter)
        adapter_digest = str(adapter_manifest["adapter_digest"])
        dense_root = Path(asset_root) / "captions" / "dense" / resolved_digest
        manifest_path = dense_root / f"index.{adapter_digest[:20]}.json"
        matrix_path = dense_root / f"vectors.{adapter_digest[:20]}.npy"
        passages_digest = stable_digest([asdict(passage) for passage in passages])
        expected_manifest = {
            "schema_version": SEMANTIC_INDEX_SCHEMA_VERSION,
            "config_digest": resolved_digest,
            "passages_digest": passages_digest,
            "passage_count": len(passages),
            "adapter": adapter_manifest,
            "cosine_normalize": True,
        }
        matrix: np.ndarray | None = None
        manifest: Mapping[str, Any] | None = None
        if not rebuild and manifest_path.is_file() and matrix_path.is_file():
            candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
            if all(candidate.get(key) == value for key, value in expected_manifest.items()):
                loaded = np.load(matrix_path, allow_pickle=False)
                if loaded.shape == (len(passages), int(adapter.dimension)):
                    matrix = np.asarray(loaded, dtype=np.float32)
                    manifest = candidate
        if matrix is None:
            matrix = np.asarray(
                adapter.embed_documents([passage.text for passage in passages]),
                dtype=np.float32,
            )
            index = cls(passages, matrix, adapter=adapter, config_digest=resolved_digest)
            manifest = {
                **expected_manifest,
                "index_digest": index.index_digest,
                "matrix_path": str(matrix_path),
                "matrix_dtype": "float32",
                "matrix_shape": list(index.vectors.shape),
            }
            dense_root.mkdir(parents=True, exist_ok=True)
            _atomic_save_numpy(matrix_path, index.vectors)
            _atomic_write_json(manifest_path, manifest)
            index.cache_manifest = dict(manifest)
            return index
        index = cls(
            passages,
            matrix,
            adapter=adapter,
            config_digest=resolved_digest,
            cache_manifest=manifest,
        )
        if str(manifest.get("index_digest", "")) != index.index_digest:
            raise ValueError("cached caption semantic index digest does not match its contents")
        return index

    def search(
        self,
        queries: Sequence[str],
        *,
        top_k: int = 12,
        time_range: tuple[float, float] | None = None,
        expand_neighbors: int = 0,
        per_caption_limit: int = 3,
        temporal_iou_threshold: float = 0.9,
    ) -> tuple[CaptionHitV1, ...]:
        normalized_queries = tuple(
            dict.fromkeys(normalize_caption_query(query) for query in queries if normalize_caption_query(query))
        )[:5]
        if not normalized_queries or not self.passages:
            return ()
        query_vectors = np.asarray(self.adapter.embed_queries(normalized_queries), dtype=np.float32)
        if query_vectors.shape != (len(normalized_queries), int(self.adapter.dimension)):
            raise ValueError("query embedding shape does not match the configured adapter")
        query_vectors = normalize_rows(query_vectors)
        scores = np.max(query_vectors @ self.vectors.T, axis=0)
        allowed = tuple(
            index
            for index, passage in enumerate(self.passages)
            if interval_in_time_range(passage, time_range)
        )
        ordered = sorted(
            allowed,
            key=lambda index: (
                -float(scores[index]),
                self.passages[index].virtual_start_sec,
                self.passages[index].passage_id,
            ),
        )
        selected: list[int] = []
        caption_counts: Counter[str] = Counter()
        for index in ordered:
            passage = self.passages[index]
            if caption_counts[passage.caption_id] >= max(1, int(per_caption_limit)):
                continue
            if any(
                passage_interval_iou(passage, self.passages[existing]) >= float(temporal_iou_threshold)
                for existing in selected
            ):
                continue
            selected.append(index)
            caption_counts[passage.caption_id] += 1
            if len(selected) >= max(1, int(top_k)):
                break
        hits = [
            self._hit(index, rank=rank, score=float(scores[index]))
            for rank, index in enumerate(selected, start=1)
        ]
        if expand_neighbors > 0:
            hits = expand_passage_neighbors(
                self.passages,
                hits,
                distance=int(expand_neighbors),
                time_range=time_range,
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
    ) -> str:
        return stable_digest(
            {
                "index_mode": "dense",
                "index_digest": self.index_digest,
                "queries": [normalize_caption_query(query) for query in queries],
                "top_k": int(top_k),
                "time_range": list(time_range) if time_range else None,
                "expand_neighbors": int(expand_neighbors),
            }
        )

    def save_manifest(self, asset_root: Path) -> Path:
        adapter_digest = str(self.adapter_manifest["adapter_digest"])
        path = (
            Path(asset_root)
            / "captions"
            / "dense"
            / self.config_digest
            / f"index.{adapter_digest[:20]}.json"
        )
        if not path.is_file():
            _atomic_write_json(
                path,
                {
                    "schema_version": SEMANTIC_INDEX_SCHEMA_VERSION,
                    "config_digest": self.config_digest,
                    "passages_digest": self.passages_digest,
                    "passage_count": len(self.passages),
                    "adapter": self.adapter_manifest,
                    "cosine_normalize": True,
                    "index_digest": self.index_digest,
                    "matrix_shape": list(self.vectors.shape),
                },
            )
        return path

    def _hit(self, index: int, *, rank: int, score: float) -> CaptionHitV1:
        passage = self.passages[index]
        return CaptionHitV1(
            passage_id=passage.passage_id,
            caption_id=passage.caption_id,
            rank=rank,
            lexical_score=None,
            dense_score=score,
            fused_score=score,
            virtual_start_sec=passage.virtual_start_sec,
            virtual_end_sec=passage.virtual_end_sec,
            wall_clock_begin=_optional_text(passage.metadata.get("wall_clock_begin")),
            wall_clock_end=_optional_text(passage.metadata.get("wall_clock_end")),
            text=passage.text,
            interval_precision=str(passage.metadata.get("interval_precision", "chunk")),
            source_pointer=f"caption://{self.config_digest}/{passage.passage_id}",
            metadata={"index_digest": self.index_digest, "index_mode": "dense"},
        )


def load_caption_passages(
    asset_root: Path,
    *,
    config_digest: str | None = None,
) -> tuple[tuple[CaptionPassageV1, ...], str, Path]:
    path, resolved_digest = resolve_caption_passages_path(
        asset_root,
        config_digest=config_digest,
    )
    passages = tuple(
        passage_from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return passages, resolved_digest, path


def interval_in_time_range(
    passage: CaptionPassageV1,
    time_range: tuple[float, float] | None,
) -> bool:
    if time_range is None:
        return True
    start, end = sorted((float(time_range[0]), float(time_range[1])))
    return passage.virtual_end_sec > start and passage.virtual_start_sec < end


def passage_interval_iou(left: CaptionPassageV1, right: CaptionPassageV1) -> float:
    intersection = max(
        0.0,
        min(left.virtual_end_sec, right.virtual_end_sec)
        - max(left.virtual_start_sec, right.virtual_start_sec),
    )
    union = max(left.virtual_end_sec, right.virtual_end_sec) - min(
        left.virtual_start_sec,
        right.virtual_start_sec,
    )
    return intersection / union if union > 0.0 else 0.0


def expand_passage_neighbors(
    passages: Sequence[CaptionPassageV1],
    hits: Sequence[CaptionHitV1],
    *,
    distance: int,
    time_range: tuple[float, float] | None,
    index_digest: str,
    config_digest: str,
) -> list[CaptionHitV1]:
    by_id = {passage.passage_id: passage for passage in passages}
    by_caption_ordinal = {
        (passage.caption_id, passage.ordinal): passage
        for passage in passages
    }
    expanded: list[CaptionHitV1] = []
    seen: set[str] = set()
    for hit in hits:
        if hit.passage_id not in seen:
            expanded.append(hit)
            seen.add(hit.passage_id)
        source = by_id.get(hit.passage_id)
        if source is None:
            continue
        for offset in range(-max(0, int(distance)), max(0, int(distance)) + 1):
            if offset == 0:
                continue
            neighbor = by_caption_ordinal.get((source.caption_id, source.ordinal + offset))
            if neighbor is None or neighbor.passage_id in seen:
                continue
            if not interval_in_time_range(neighbor, time_range):
                continue
            expanded.append(
                CaptionHitV1(
                    passage_id=neighbor.passage_id,
                    caption_id=neighbor.caption_id,
                    rank=len(expanded) + 1,
                    lexical_score=None,
                    dense_score=None,
                    fused_score=max(0.0, hit.fused_score * 0.5),
                    virtual_start_sec=neighbor.virtual_start_sec,
                    virtual_end_sec=neighbor.virtual_end_sec,
                    wall_clock_begin=_optional_text(neighbor.metadata.get("wall_clock_begin")),
                    wall_clock_end=_optional_text(neighbor.metadata.get("wall_clock_end")),
                    text=neighbor.text,
                    interval_precision=str(neighbor.metadata.get("interval_precision", "chunk")),
                    source_pointer=f"caption://{config_digest}/{neighbor.passage_id}",
                    metadata={
                        "index_digest": index_digest,
                        "neighbor_of": hit.passage_id,
                        "candidate_only": True,
                    },
                )
            )
            seen.add(neighbor.passage_id)
    return expanded


def _adapter_manifest(adapter: TextEmbeddingAdapter) -> dict[str, Any]:
    payload = dict(adapter.manifest)
    payload.setdefault("model_id", str(adapter.model_id))
    payload.setdefault("model_version", str(adapter.model_version))
    payload.setdefault("dimension", int(adapter.dimension))
    payload.setdefault("normalize", bool(adapter.normalize))
    payload["adapter_digest"] = str(
        payload.get("adapter_digest")
        or stable_digest({key: value for key, value in payload.items() if key != "device"})
    )
    return payload


def _atomic_save_numpy(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as handle:
        temp_path = Path(handle.name)
        np.save(handle, np.asarray(matrix, dtype=np.float32), allow_pickle=False)
    os.replace(temp_path, path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        suffix=".json",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp_path, path)


def _optional_text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None
