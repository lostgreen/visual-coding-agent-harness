from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vcah.caption_schema import CaptionPassageV1, passage_to_dict
from vcah.caption_semantic_index import CaptionSemanticIndex
from vcah.embedding_adapter import SentenceTransformerEmbeddingAdapter


class FakeSemanticAdapter:
    model_id = "fixture-semantic"
    model_version = "v1"
    dimension = 3
    normalize = True

    def __init__(self) -> None:
        self.document_calls = 0
        self.query_calls = 0

    @property
    def manifest(self) -> Mapping[str, Any]:
        return {
            "backend": "fixture",
            "model_id": self.model_id,
            "model_version": self.model_version,
            "dimension": self.dimension,
            "normalize": self.normalize,
            "adapter_digest": "fixture-semantic-v1",
        }

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        self.document_calls += 1
        return np.asarray([self._vector(text) for text in texts], dtype=np.float32)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        self.query_calls += 1
        return np.asarray([self._vector(text) for text in texts], dtype=np.float32)

    @staticmethod
    def _vector(text: str) -> tuple[float, float, float]:
        vectors = {
            "An automobile is parked beside the road.": (1.0, 0.0, 0.0),
            "A singer performs under bright stage lights.": (0.0, 1.0, 0.0),
            "The word car appears on a poster.": (0.0, 0.0, 1.0),
            "stationary car": (1.0, 0.0, 0.0),
            "stage performance": (0.0, 1.0, 0.0),
        }
        return vectors.get(text, (0.0, 0.0, 1.0))


def passages() -> tuple[CaptionPassageV1, ...]:
    return (
        CaptionPassageV1(
            passage_id="cap:p0",
            caption_id="cap",
            text="An automobile is parked beside the road.",
            virtual_start_sec=0.0,
            virtual_end_sec=10.0,
            anchor_virtual_sec=0.0,
            ordinal=0,
        ),
        CaptionPassageV1(
            passage_id="cap:p1",
            caption_id="cap",
            text="A singer performs under bright stage lights.",
            virtual_start_sec=10.0,
            virtual_end_sec=20.0,
            anchor_virtual_sec=10.0,
            ordinal=1,
        ),
        CaptionPassageV1(
            passage_id="cap:p2",
            caption_id="cap",
            text="The word car appears on a poster.",
            virtual_start_sec=20.0,
            virtual_end_sec=30.0,
            anchor_virtual_sec=20.0,
            ordinal=2,
        ),
    )


def write_passages(asset_root: Path, config_digest: str = "caption-config") -> Path:
    path = asset_root / "captions" / f"passages.{config_digest}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        "".join(json.dumps(passage_to_dict(item), sort_keys=True) + "\n" for item in passages()),
        encoding="utf-8",
    )
    return path


def test_dense_semantic_paraphrase_finds_nonlexical_match() -> None:
    adapter = FakeSemanticAdapter()
    index = CaptionSemanticIndex.build(passages(), adapter=adapter, config_digest="fixture")

    hits = index.search(("stationary car",), top_k=2)

    assert hits[0].passage_id == "cap:p0"
    assert hits[0].dense_score == 1.0
    assert hits[0].lexical_score is None
    assert adapter.document_calls == 1
    assert adapter.query_calls == 1


def test_dense_index_cache_is_bound_to_embedding_identity(tmp_path: Path) -> None:
    write_passages(tmp_path)
    adapter = FakeSemanticAdapter()

    first = CaptionSemanticIndex.from_asset_root(tmp_path, adapter=adapter)
    second = CaptionSemanticIndex.from_asset_root(tmp_path, adapter=adapter)

    assert first.index_digest == second.index_digest
    assert adapter.document_calls == 1
    manifest_path = first.save_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["adapter"]["model_id"] == "fixture-semantic"
    assert manifest["adapter"]["model_version"] == "v1"
    assert manifest["adapter"]["normalize"] is True
    assert manifest["matrix_shape"] == [3, 3]


def test_dense_time_filter_and_neighbor_expansion() -> None:
    index = CaptionSemanticIndex.build(
        passages(),
        adapter=FakeSemanticAdapter(),
        config_digest="fixture",
    )

    filtered = index.search(("stationary car",), top_k=1, time_range=(10.0, 30.0))
    expanded = index.search(("stationary car",), top_k=1, expand_neighbors=1)

    assert filtered[0].passage_id != "cap:p0"
    assert [hit.passage_id for hit in expanded] == ["cap:p0", "cap:p1"]
    assert expanded[1].metadata["candidate_only"] is True


class FakeSentenceModel:
    def get_sentence_embedding_dimension(self) -> int:
        return 2

    def encode(self, texts: Sequence[str], **_: Any) -> np.ndarray:
        return np.asarray([[3.0, 4.0] for _ in texts], dtype=np.float32)


def test_sentence_transformer_adapter_normalizes_real_model_output() -> None:
    adapter = SentenceTransformerEmbeddingAdapter("fixture/model", model=FakeSentenceModel())

    matrix = adapter.embed_queries(("query",))

    assert matrix.shape == (1, 2)
    assert np.linalg.norm(matrix[0]) == np.float32(1.0)
    assert adapter.manifest["backend"] == "sentence-transformers"
