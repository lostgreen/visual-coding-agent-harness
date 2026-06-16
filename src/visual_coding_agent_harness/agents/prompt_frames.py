"""Prompt frame rendering with explicit retention semantics."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from hashlib import sha256

from .runtime_capabilities import PromptRetentionMode


@dataclass(frozen=True)
class PromptFrame:
    frame_id: str
    title: str
    body: str
    version: str
    cacheable: bool = True

    @cached_property
    def digest(self) -> str:
        return sha256(self.body.encode("utf-8")).hexdigest()[:12]

    def render_full(self, *, replacing: bool = False) -> str:
        marker = " replacing" if replacing else ""
        return f"# {self.title} [v={self.version}, digest={self.digest}{marker}]\n{self.body.strip()}\n"

    def render_reference(self) -> str:
        return f"# {self.title} [loaded, v={self.version}, digest={self.digest}]"


class PromptFrameLedger:
    def __init__(self, *, mode: PromptRetentionMode) -> None:
        self.mode = mode
        self._loaded: dict[str, str] = {}

    def take(self, frame: PromptFrame) -> str:
        prior = self._loaded.get(frame.frame_id)
        changed = prior is not None and prior != frame.digest

        if self.mode is not PromptRetentionMode.STICKY_REFERENCE:
            self._loaded[frame.frame_id] = frame.digest
            return frame.render_full(replacing=changed)

        if prior is None or changed:
            self._loaded[frame.frame_id] = frame.digest
            return frame.render_full(replacing=changed)

        return frame.render_reference()

    def invalidate(self, frame_id: str) -> None:
        self._loaded.pop(frame_id, None)

    def snapshot(self) -> dict[str, str]:
        return dict(self._loaded)
