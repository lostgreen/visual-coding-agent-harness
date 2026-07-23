from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


class TextEmbeddingAdapter(Protocol):
    model_id: str
    model_version: str
    dimension: int
    normalize: bool

    @property
    def manifest(self) -> Mapping[str, Any]:
        ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        ...

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        ...


@dataclass(frozen=True)
class EmbeddingIdentity:
    backend: str
    model_id: str
    model_version: str
    dimension: int
    normalize: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "dimension": self.dimension,
            "normalize": self.normalize,
        }

    @property
    def digest(self) -> str:
        serialized = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class SentenceTransformerEmbeddingAdapter:
    """Real text embeddings backed by sentence-transformers."""

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        device: str = "cpu",
        normalize: bool = True,
        batch_size: int = 64,
        model: Any | None = None,
    ) -> None:
        self.model_id = str(model_id or "").strip()
        if not self.model_id:
            raise ValueError("embedding model_id is required")
        self.device = str(device or "cpu")
        self.normalize = bool(normalize)
        self.batch_size = max(1, int(batch_size))
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for SentenceTransformerEmbeddingAdapter"
                ) from exc
            kwargs: dict[str, Any] = {"device": self.device}
            if revision:
                kwargs["revision"] = str(revision)
            model = SentenceTransformer(self.model_id, **kwargs)
        self._model = model
        dimension = model.get_sentence_embedding_dimension()
        if dimension is None or int(dimension) <= 0:
            raise ValueError("embedding model did not report a positive dimension")
        self.dimension = int(dimension)
        self.model_version = _resolved_model_version(model, revision)
        self.identity = EmbeddingIdentity(
            backend="sentence-transformers",
            model_id=self.model_id,
            model_version=self.model_version,
            dimension=self.dimension,
            normalize=self.normalize,
        )

    @property
    def manifest(self) -> Mapping[str, Any]:
        return {
            **self.identity.to_dict(),
            "adapter_digest": self.identity.digest,
            "device": self.device,
            "batch_size": self.batch_size,
        }

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        values = tuple(str(text) for text in texts)
        if not values:
            return np.empty((0, self.dimension), dtype=np.float32)
        encoded = self._model.encode(
            list(values),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        matrix = np.asarray(encoded, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape != (len(values), self.dimension):
            raise ValueError(
                f"embedding shape mismatch: expected {(len(values), self.dimension)}, got {matrix.shape}"
            )
        return normalize_rows(matrix) if self.normalize else matrix


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("embedding matrix must be two-dimensional")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float32).eps)


def _resolved_model_version(model: Any, requested_revision: str | None) -> str:
    try:
        first_module = model._first_module()
        auto_model = getattr(first_module, "auto_model", None)
        config = getattr(auto_model, "config", None)
        commit_hash = str(getattr(config, "_commit_hash", "") or "").strip()
        if commit_hash:
            return commit_hash
    except (AttributeError, TypeError):
        pass
    return str(requested_revision or "default")
