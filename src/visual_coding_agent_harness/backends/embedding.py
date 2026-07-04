"""Embedding backends for cold visual workspace indexes."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


class EmbeddingBackend(Protocol):
    embedding_dim: int

    def encode_images(self, paths: Sequence[str]) -> np.ndarray:
        """Return one embedding row per image path."""

    def encode_text(self, queries: Sequence[str]) -> np.ndarray:
        """Return one embedding row per text query."""


class SigLIPBackend:
    """Lazy SigLIP adapter used when a real visual semantic index is requested."""

    def __init__(self, model_id: str = "google/siglip-so400m-patch14-384", device: str = "cuda") -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:  # pragma: no cover - optional heavy dependency.
            raise RuntimeError("SigLIPBackend requires torch and transformers to be installed") from exc

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device)
        self.model.eval()
        self.embedding_dim = int(getattr(getattr(self.model, "config", None), "projection_dim", 0) or 0)

    def encode_images(self, paths: Sequence[str]) -> np.ndarray:
        from PIL import Image
        import torch

        images = []
        for path in paths:
            with Image.open(Path(path)) as image:
                images.append(image.convert("RGB").copy())
        if not images:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return features.detach().cpu().numpy().astype(np.float32)

    def encode_text(self, queries: Sequence[str]) -> np.ndarray:
        import torch

        if not queries:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        inputs = self.processor(text=list(queries), padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        return features.detach().cpu().numpy().astype(np.float32)
