"""In-memory visual semantic index over cold workspace beats."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, TYPE_CHECKING

import numpy as np

from visual_coding_agent_harness.backends.embedding import EmbeddingBackend

if TYPE_CHECKING:  # pragma: no cover
    from visual_coding_agent_harness.workspace.video_workspace import Beat


@dataclass(frozen=True)
class BeatHit:
    beat_id: str
    score: float
    modality: Literal["visual", "text"]


class VisualIndex:
    def __init__(self, backend: EmbeddingBackend) -> None:
        self.backend = backend
        self._beat_ids: tuple[str, ...] = ()
        self._embeddings = np.zeros((0, int(getattr(backend, "embedding_dim", 0) or 0)), dtype=np.float32)

    def build(self, beats: Sequence["Beat"]) -> None:
        self._beat_ids = tuple(beat.beat_id for beat in beats)
        if not self._beat_ids:
            self._embeddings = np.zeros((0, int(getattr(self.backend, "embedding_dim", 0) or 0)), dtype=np.float32)
            return
        embeddings = np.asarray(self.backend.encode_images([beat.keyframe_path for beat in beats]), dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(self._beat_ids):
            raise ValueError("encode_images must return an (N, D) array")
        self._embeddings = _l2_normalize(embeddings)

    def search(self, query: str, k: int = 20) -> tuple[BeatHit, ...]:
        if not self._beat_ids or self._embeddings.size == 0 or k <= 0:
            return ()
        query_vec = np.asarray(self.backend.encode_text((query,)), dtype=np.float32)
        if query_vec.ndim != 2 or query_vec.shape[0] != 1:
            raise ValueError("encode_text must return a (1, D) array for one query")
        query_vec = _l2_normalize(query_vec)
        scores = self._embeddings @ query_vec[0]
        order = sorted(range(len(scores)), key=lambda idx: (-float(scores[idx]), self._beat_ids[idx]))
        return tuple(
            BeatHit(beat_id=self._beat_ids[idx], score=float(scores[idx]), modality="visual")
            for idx in order[: max(0, int(k))]
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            beat_ids=np.asarray(self._beat_ids, dtype=str),
            embeddings=np.asarray(self._embeddings, dtype=np.float32),
        )

    @classmethod
    def load(cls, path: Path, backend: EmbeddingBackend) -> "VisualIndex":
        index = cls(backend)
        payload = np.load(Path(path), allow_pickle=False)
        index._beat_ids = tuple(str(item) for item in payload["beat_ids"].tolist())
        index._embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        return index


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return array / norms
