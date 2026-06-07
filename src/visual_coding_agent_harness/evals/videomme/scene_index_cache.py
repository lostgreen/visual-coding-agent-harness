"""Stable on-disk cache for VideoMME scene indexes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional

from ...video_index import SceneIndex


class SceneIndexCache:
    def __init__(self, cache_dir: Path | str) -> None:
        self.cache_dir = Path(cache_dir)

    def key_for(self, parts: Mapping[str, Any]) -> str:
        payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def load(self, key: str) -> Optional[SceneIndex]:
        path = self._path_for(key)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return SceneIndex.from_dict(data["scene_index"])

    def store(self, key: str, scene_index: SceneIndex) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._path_for(key)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps({"scene_index": scene_index.to_dict()}, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(path)

    def _path_for(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"
