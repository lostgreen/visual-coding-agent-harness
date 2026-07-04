from __future__ import annotations

import hashlib
import os
from typing import Any, Mapping, Sequence

import numpy as np

from vcah.types import ToolAction


class ModelClient:
    """Thin model wrapper configured by environment variables."""

    embedding_dim = 8

    def __init__(self) -> None:
        self.controller_model = os.getenv("VCAH_CONTROLLER_MODEL", "local-scripted")
        self.vision_model = os.getenv("VCAH_VISION_MODEL", "local-placeholder")
        self.embed_model = os.getenv("VCAH_EMBED_MODEL", "local-hash")
        self.transcribe_model = os.getenv("VCAH_TRANSCRIBE_MODEL", "none")

    def controller(self, question: str, index_digest: str, memory_digest: str, evidence_digest: str) -> ToolAction:
        del index_digest
        if not memory_digest:
            return ToolAction(type="search_text", query=question)
        if "ev_" in evidence_digest:
            first_id = evidence_digest.split()[0]
            return ToolAction(type="answer", answer="See verified evidence.", citations=(first_id,))
        return ToolAction(type="focus_clip", beat_id="")

    def vision(self, image_paths: Sequence[str], prompt: str) -> str:
        del image_paths
        return prompt

    def embed_text(self, queries: Sequence[str]) -> np.ndarray:
        return np.asarray([_hash_embedding(query, self.embedding_dim) for query in queries], dtype=np.float32)

    def embed_image(self, paths: Sequence[str]) -> np.ndarray:
        return np.asarray([_hash_embedding(path, self.embedding_dim) for path in paths], dtype=np.float32)

    def transcribe(self, video_path: str) -> tuple[Mapping[str, Any], ...]:
        del video_path
        return ()


class ScriptedModel(ModelClient):
    def __init__(self, actions: Sequence[Mapping[str, Any]] = ()) -> None:
        super().__init__()
        self._actions = [ToolAction.from_mapping(action) for action in actions]
        self._cursor = 0

    def controller(self, question: str, index_digest: str, memory_digest: str, evidence_digest: str) -> ToolAction:
        del question, index_digest, memory_digest, evidence_digest
        if self._cursor >= len(self._actions):
            return ToolAction(type="answer", answer="Insufficient verified evidence.", citations=())
        action = self._actions[self._cursor]
        self._cursor += 1
        return action


def _hash_embedding(text: str, dim: int) -> list[float]:
    digest = hashlib.sha256(str(text).encode("utf-8")).digest()
    values = [float(digest[index] - 127) for index in range(max(1, int(dim)))]
    norm = sum(value * value for value in values) ** 0.5 or 1.0
    return [value / norm for value in values]
