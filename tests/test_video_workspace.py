from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from visual_coding_agent_harness.video.index import Frame
from visual_coding_agent_harness.workspace.text_index import InvertedIndex
from visual_coding_agent_harness.workspace.video_workspace import Beat, Chapter, VideoWorkspace, build_video_workspace
from visual_coding_agent_harness.workspace.visual_index import VisualIndex


class ColorEmbeddingBackend:
    embedding_dim = 3

    def encode_images(self, paths: Sequence[str]) -> np.ndarray:
        rows = []
        for path in paths:
            with Image.open(path) as image:
                pixel = image.convert("RGB").resize((1, 1)).getpixel((0, 0))
            channel = int(max(range(3), key=lambda idx: pixel[idx]))
            row = [0.0, 0.0, 0.0]
            row[channel] = 1.0
            rows.append(row)
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


def _sampler(video_path: str, start_sec: float, end_sec: float, n_frames: int, out_dir: Path) -> tuple[Frame, ...]:
    del video_path, end_sec, n_frames
    out_dir.mkdir(parents=True, exist_ok=True)
    color = (240, 20, 20) if start_sec < 4 else ((20, 220, 30) if start_sec < 6 else (20, 40, 230))
    path = out_dir / f"shot_{int(start_sec)}.jpg"
    Image.new("RGB", (32, 18), color=color).save(path)
    return (Frame(frame_id="tmp", time_sec=start_sec, thumb_path=str(path)),)


def test_build_video_workspace_creates_cold_indexes_and_roundtrips(tmp_path: Path) -> None:
    workspace = build_video_workspace(
        "/videos/demo.mp4",
        8.0,
        artifact_dir=tmp_path / "workspace",
        asr_cues=({"start": 0.0, "end": 4.0, "text": "the shipyard closed in 1982"},),
        ocr_lines_by_time=((6.5, "BLUE SIGN"),),
        embedding_backend=ColorEmbeddingBackend(),
        shot_detector=lambda _video_path, _duration: ((0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0)),
        keyframe_sampler=_sampler,
    )

    assert len(workspace.chapters) >= 1
    assert len(workspace.beats) < 4
    assert workspace.search_text("shipyard")[0].beat_id == "bt00001"
    assert workspace.search_visual("blue frame", k=1)[0].beat_id == workspace.beats[-1].beat_id
    assert workspace.window(workspace.beats[0].beat_id, after=1) == workspace.beats[:2]
    assert workspace.beats_in_chapters((workspace.chapters[0].chapter_id,))
    assert "entities" not in workspace.beat_metadata_text((workspace.beats[0].beat_id,))

    workspace.save(tmp_path / "saved")
    loaded = VideoWorkspace.load(tmp_path / "saved", embedding_backend=ColorEmbeddingBackend())

    assert loaded.timeline_text()
    assert loaded.search_text('"shipyard closed"')[0].beat_id == "bt00001"


def test_timeline_text_can_fill_missing_titles_without_vlm(tmp_path: Path) -> None:
    workspace = build_video_workspace(
        "/videos/demo.mp4",
        4.0,
        artifact_dir=tmp_path / "workspace",
        asr_cues=({"start": 0.0, "end": 4.0, "text": "shipyard shipyard harbor"},),
        embedding_backend=ColorEmbeddingBackend(),
        shot_detector=lambda _video_path, _duration: ((0.0, 2.0), (2.0, 4.0)),
        keyframe_sampler=_sampler,
    )

    assert workspace.chapters[0].title is None
    text = workspace.timeline_text(fill_missing_titles=True)

    assert workspace.chapters[0].title is not None
    assert "shipyard" in text.lower()


def test_window_time_returns_beats_overlapping_requested_seconds() -> None:
    beats = (
        Beat("bt00001", "ch01", 0.0, 10.0, "", "", (), ("sh001",)),
        Beat("bt00002", "ch01", 10.0, 20.0, "", "", (), ("sh002",)),
        Beat("bt00003", "ch01", 40.0, 50.0, "", "", (), ("sh003",)),
    )
    workspace = VideoWorkspace(
        video_path="/videos/demo.mp4",
        duration_sec=50.0,
        chapters=(Chapter("ch01", 0.0, 50.0, tuple(beat.beat_id for beat in beats), ""),),
        beats=beats,
        text_index=InvertedIndex(),
        visual_index=VisualIndex(ColorEmbeddingBackend()),
    )

    window = workspace.window_time("bt00002", before_sec=5.0, after_sec=25.0)

    assert tuple(beat.beat_id for beat in window) == ("bt00001", "bt00002", "bt00003")
