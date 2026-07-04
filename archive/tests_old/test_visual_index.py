from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from visual_coding_agent_harness.workspace.video_workspace import Beat
from visual_coding_agent_harness.workspace.visual_index import VisualIndex


class MockEmbeddingBackend:
    embedding_dim = 3

    def encode_images(self, paths: Sequence[str]) -> np.ndarray:
        rows = []
        for path in paths:
            name = Path(path).stem
            if "blue" in name:
                rows.append([0.0, 0.0, 1.0])
            elif "green" in name:
                rows.append([0.0, 1.0, 0.0])
            else:
                rows.append([1.0, 0.0, 0.0])
        return np.asarray(rows, dtype=np.float32)

    def encode_text(self, queries: Sequence[str]) -> np.ndarray:
        rows = []
        for query in queries:
            if "blue" in query:
                rows.append([0.0, 0.0, 1.0])
            elif "green" in query:
                rows.append([0.0, 1.0, 0.0])
            else:
                rows.append([1.0, 0.0, 0.0])
        return np.asarray(rows, dtype=np.float32)


def _beat(beat_id: str, keyframe: str) -> Beat:
    return Beat(
        beat_id=beat_id,
        chapter_id="ch01",
        start_sec=0.0,
        end_sec=1.0,
        keyframe_path=keyframe,
        asr_verbatim="",
        ocr_verbatim=(),
        shot_ids=(beat_id.replace("bt", "sh"),),
    )


def test_visual_index_searches_by_cosine_and_roundtrips(tmp_path: Path) -> None:
    backend = MockEmbeddingBackend()
    index = VisualIndex(backend)
    index.build(
        (
            _beat("bt001", str(tmp_path / "red.jpg")),
            _beat("bt002", str(tmp_path / "blue.jpg")),
            _beat("bt003", str(tmp_path / "green.jpg")),
        )
    )

    assert [hit.beat_id for hit in index.search("blue frame", k=2)] == ["bt002", "bt001"]

    path = tmp_path / "visual_index.npz"
    index.save(path)
    loaded = VisualIndex.load(path, backend)

    assert [hit.beat_id for hit in loaded.search("blue frame", k=2)] == ["bt002", "bt001"]
