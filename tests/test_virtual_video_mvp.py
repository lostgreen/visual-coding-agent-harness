from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from vcah.types import Frame
from vcah.virtual_index import build_virtual_beat_index
from vcah.virtual_video import (
    VirtualVideoCase,
    VirtualVideoManifest,
    VirtualVideoSegment,
    VirtualVideoWorkspace,
    load_srt_as_virtual_cues,
    materialize_highfps_window,
    materialize_lowfps_frame_cache,
    virtual_to_source_windows,
)


class ColorModel:
    embedding_dim = 3
    embed_model = "color-test"
    allow_placeholder_visual = True

    def embed_image(self, paths: Sequence[str]) -> np.ndarray:
        rows = []
        for path in paths:
            with Image.open(path) as image:
                red, green, blue = image.convert("RGB").resize((1, 1)).getpixel((0, 0))
            row = [0.0, 0.0, 0.0]
            row[int(max(range(3), key=lambda idx: (red, green, blue)[idx]))] = 1.0
            rows.append(row)
        return np.asarray(rows, dtype=np.float32)

    def embed_text(self, queries: Sequence[str]) -> np.ndarray:
        rows = []
        for query in queries:
            text = query.casefold()
            rows.append([0.0, 0.0, 1.0] if "blue" in text else [1.0, 0.0, 0.0])
        return np.asarray(rows, dtype=np.float32)


def _fake_sampler(video_path: str, start_sec: float, end_sec: float, n_frames: int, out_dir: Path) -> tuple[Frame, ...]:
    del end_sec
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(max(1, int(n_frames))):
        time_sec = round(float(start_sec) + index * 0.5, 3)
        color = (20, 40, 230) if "target" in video_path else (230, 30, 20)
        path = out_dir / f"{Path(video_path).stem}_{time_sec:.3f}_{index:03d}.jpg"
        Image.new("RGB", (32, 18), color=color).save(path)
        frames.append(Frame(frame_id=f"fr{index:03d}", time_sec=time_sec, path=str(path)))
    return tuple(frames)


def _workspace(tmp_path: Path) -> VirtualVideoWorkspace:
    manifest = VirtualVideoManifest(
        workspace_id="case-1",
        segments=(
            VirtualVideoSegment("seg_a", "distractor", "distractor.mp4", 10.0, 14.0, 0.0, 4.0, "distractor"),
            VirtualVideoSegment("seg_b", "target", "target.mp4", 100.0, 106.0, 4.0, 10.0, "target"),
        ),
    )
    case = VirtualVideoCase(
        case_id="case-1",
        question="What number is visible on the target jersey?",
        options={"A": "7", "B": "11"},
        gold="B",
        target_segment_id="seg_b",
        target_virtual_interval=(5.0, 8.0),
    )
    return VirtualVideoWorkspace.create(tmp_path / "case-1", manifest=manifest, case=case)


def test_virtual_timeline_maps_cross_segment_windows_and_srt_cues(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    srt_path = tmp_path / "target.srt"
    srt_path.write_text(
        "1\n00:01:41,000 --> 00:01:42,500\nNishida turns around\n\n"
        "2\n00:01:45,000 --> 00:01:46,000\nnumber on the back\n",
        encoding="utf-8",
    )

    windows = virtual_to_source_windows(workspace.manifest, 3.0, 6.0)
    cues = load_srt_as_virtual_cues(srt_path, workspace.manifest.segments[1])

    assert [(item.segment_id, item.source_start_sec, item.source_end_sec) for item in windows] == [
        ("seg_a", 13.0, 14.0),
        ("seg_b", 100.0, 102.0),
    ]
    assert cues == (
        {"start": 5.0, "end": 6.5, "text": "Nishida turns around", "segment_id": "seg_b", "source_video_id": "target", "source_start": 101.0, "source_end": 102.5},
        {"start": 9.0, "end": 10.0, "text": "number on the back", "segment_id": "seg_b", "source_video_id": "target", "source_start": 105.0, "source_end": 106.0},
    )


def test_lowfps_and_highfps_manifests_keep_separate_lineage(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    low_frames = materialize_lowfps_frame_cache(workspace, fps=1.0, sampler=_fake_sampler)
    high_frames = materialize_highfps_window(workspace, 5.0, 6.0, query_id="q1", fps=2.0, sampler=_fake_sampler)

    low_manifest = workspace.root_dir / "frame_manifest.jsonl"
    high_manifest = workspace.root_dir / "observations" / "highfps_frame_manifest.jsonl"
    low_rows = [json.loads(line) for line in low_manifest.read_text(encoding="utf-8").splitlines()]
    high_rows = [json.loads(line) for line in high_manifest.read_text(encoding="utf-8").splitlines()]

    assert low_frames and high_frames
    assert {row["fps_level"] for row in low_rows} == {"low"}
    assert {row["fps_level"] for row in high_rows} == {"high"}
    assert all("observations" not in row["path"] for row in low_rows)
    assert all(row["query_id"] == "q1" for row in high_rows)
    assert high_rows[0]["virtual_time_sec"] == 5.0
    assert high_rows[0]["source_video_id"] == "target"
    assert high_rows[0]["source_time_sec"] == 101.0


def test_virtual_beat_index_uses_thumbnail_as_cold_keyframe_and_virtual_times(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    frames = materialize_lowfps_frame_cache(workspace, fps=1.0, sampler=_fake_sampler)
    workspace.write_asr_virtual_cues(({"start": 4.5, "end": 5.5, "text": "blue target jersey", "segment_id": "seg_b"},))

    result = build_virtual_beat_index(workspace, frames, model=ColorModel(), beat_sec=3.0)

    assert result.cold_index.search_text("target jersey")[0].beat_id == "bt00003"
    assert result.cold_index.search_visual("blue", k=1)[0].beat_id == "bt00003"
    assert result.cold_index.beats[0].start_sec == 0.0
    assert result.cold_index.beats[0].frame_times[0] == 0.0
    assert result.cold_index.beats[0].keyframe_path.endswith("_grid.jpg")
    assert Path(result.cold_index.beats[0].keyframe_path).exists()
    assert result.beat_index_path.exists()
    beat_rows = json.loads(result.beat_index_path.read_text(encoding="utf-8"))["beats"]
    assert beat_rows[2]["source_lineage"][0]["source_video_id"] == "target"
