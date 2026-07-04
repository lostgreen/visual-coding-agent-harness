from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from vcah.index import ColdIndex, build_cold_index
from vcah.model import ModelClient
from vcah.types import Frame


class ColorModel:
    embedding_dim = 3

    def embed_image(self, paths: Sequence[str]) -> np.ndarray:
        rows = []
        for path in paths:
            with Image.open(path) as image:
                pixel = image.convert("RGB").resize((1, 1)).getpixel((0, 0))
            channel = int(max(range(3), key=lambda idx: pixel[idx]))
            row = [0.0, 0.0, 0.0]
            row[channel] = 1.0
            rows.append(row)
        return np.asarray(rows, dtype=np.float32)

    def embed_text(self, queries: Sequence[str]) -> np.ndarray:
        rows = []
        for query in queries:
            query = query.lower()
            if "blue" in query:
                rows.append([0.0, 0.0, 1.0])
            elif "green" in query:
                rows.append([0.0, 1.0, 0.0])
            else:
                rows.append([1.0, 0.0, 0.0])
        return np.asarray(rows, dtype=np.float32)


def _sampler(video_path: str, start_sec: float, end_sec: float, n_frames: int, out_dir: Path) -> tuple[Frame, ...]:
    del video_path, end_sec, n_frames
    out_dir.mkdir(parents=True, exist_ok=True)
    color = (240, 20, 20) if start_sec < 60 else ((20, 220, 30) if start_sec < 90 else (20, 40, 230))
    path = out_dir / f"frame_{int(start_sec):03d}.jpg"
    Image.new("RGB", (32, 18), color=color).save(path)
    return (Frame(frame_id="fr001", time_sec=start_sec, path=str(path)),)


def test_cold_index_builds_chapters_beats_indexes_and_artifacts(tmp_path: Path) -> None:
    cold = build_cold_index(
        "/videos/demo.mp4",
        duration_sec=120.0,
        run_dir=tmp_path,
        model=ColorModel(),
        asr_cues=({"start": 0.0, "end": 40.0, "text": "the shipyard closed in 1982"},),
        ocr_lines=((101.0, "BLUE SIGN"),),
        range_detector=lambda _video_path, _duration: ((0.0, 120.0),),
        keyframe_sampler=_sampler,
        max_range_sec=30.0,
        max_beat_sec=30.0,
    )

    assert isinstance(cold, ColdIndex)
    assert len(cold.beats) == 4
    assert all((beat.end_sec - beat.start_sec) <= 30.0 for beat in cold.beats)
    assert cold.search_text("shipyard")[0].beat_id == "bt00001"
    assert cold.search_visual("blue frame", k=1)[0].beat_id == "bt00004"
    assert cold.window("bt00002", before_sec=5.0, after_sec=25.0)[-1].beat_id == "bt00003"
    assert "chapters" in cold.timeline_digest()

    assert (tmp_path / "cold_index" / "index.json").exists()
    assert (tmp_path / "cold_index" / "diagnostics.json").exists()
    assert (tmp_path / "cold_index" / "visual_index.npz").exists()
    assert (tmp_path / "cold_index" / "timeline.jpg").exists()

    payload = json.loads((tmp_path / "cold_index" / "diagnostics.json").read_text(encoding="utf-8"))
    assert payload["beat_count"] == 4
    assert payload["max_beat_sec"] <= 30.0


def test_cold_index_roundtrips_without_vlm_captions(tmp_path: Path) -> None:
    cold = build_cold_index(
        "/videos/demo.mp4",
        duration_sec=8.0,
        run_dir=tmp_path,
        model=ColorModel(),
        asr_cues=({"start": 0.0, "end": 4.0, "text": "red crane enters"},),
        range_detector=lambda _video_path, _duration: ((0.0, 4.0), (4.0, 8.0)),
        keyframe_sampler=_sampler,
    )

    loaded = ColdIndex.load(tmp_path / "cold_index", model=ColorModel())

    assert loaded.search_text('"red crane"')[0].beat_id == cold.beats[0].beat_id
    assert "caption" not in (tmp_path / "cold_index" / "index.json").read_text(encoding="utf-8").lower()


def test_local_hash_visual_backend_is_marked_as_placeholder(tmp_path: Path) -> None:
    cold = build_cold_index(
        "/videos/demo.mp4",
        duration_sec=4.0,
        run_dir=tmp_path,
        model=ModelClient(),
        range_detector=lambda _video_path, _duration: ((0.0, 4.0),),
        keyframe_sampler=_sampler,
    )

    diagnostics = json.loads((tmp_path / "cold_index" / "diagnostics.json").read_text(encoding="utf-8"))

    assert diagnostics["embedding_backend"] == "local-hash"
    assert "placeholder_visual_embedding_backend" in diagnostics["warnings"]
    assert cold.search_visual("anything visual") == ()
