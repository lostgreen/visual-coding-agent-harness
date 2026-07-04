"""Frame contract for active two-speed video workspaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class Frame:
    frame_id: str
    time_sec: float
    thumb_path: str
    thumb_embedding: bytes = b""
    ocr_text: str = ""

    def to_dict(self) -> Mapping[str, object]:
        return {
            "frame_id": self.frame_id,
            "time_sec": float(self.time_sec),
            "thumb_path": self.thumb_path,
            "thumb_embedding": self.thumb_embedding.hex(),
            "ocr_text": self.ocr_text,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Frame":
        embedding = value.get("thumb_embedding") or ""
        return cls(
            frame_id=str(value.get("frame_id") or ""),
            time_sec=float(value.get("time_sec", 0.0) or 0.0),
            thumb_path=str(value.get("thumb_path") or ""),
            thumb_embedding=bytes.fromhex(str(embedding)) if str(embedding) else b"",
            ocr_text=str(value.get("ocr_text") or ""),
        )


__all__ = ["Frame"]
