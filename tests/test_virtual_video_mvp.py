from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Sequence

import numpy as np
from PIL import Image
import pytest

import vcah.video as video
from vcah.types import Frame
from vcah.virtual_index import build_virtual_beat_index, build_workspace_overview
from vcah.virtual_video import (
    VirtualVideoCase,
    VirtualVideoManifest,
    VirtualVideoSegment,
    VirtualVideoWorkspace,
    load_srt_as_virtual_cues,
    materialize_lowfps_frame_cache,
    materialize_window_frames,
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


def test_ffmpeg_frame_extraction_uses_concurrency_guard_and_timeout(monkeypatch, tmp_path: Path) -> None:
    class Guard:
        def __init__(self) -> None:
            self.entries = 0

        def __enter__(self) -> None:
            self.entries += 1

        def __exit__(self, exc_type, exc, traceback) -> None:
            del exc_type, exc, traceback

    guard = Guard()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((list(command), dict(kwargs)))
        Image.new("RGB", (32, 18), color=(40, 60, 80)).save(command[-1])
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(video, "_FFMPEG_SEMAPHORE", guard, raising=False)
    monkeypatch.setattr(video.subprocess, "run", fake_run)
    output_path = tmp_path / "frame.jpg"

    video._extract_frame("source.mp4", 12.5, output_path)

    assert guard.entries == 1
    assert calls[0][1]["timeout"] == 30.0
    assert "-nostdin" in calls[0][0]
    assert output_path.exists()


def test_ffmpeg_frame_extraction_retries_bounded_timeouts(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append((list(command), dict(kwargs)))
        raise subprocess.TimeoutExpired(command, kwargs.get("timeout"))

    monkeypatch.setattr(video.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="timed out"):
        video._extract_frame("source.mp4", 12.5, tmp_path / "frame.jpg")

    assert len(calls) == 2


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


def _constant_name_sampler(video_path: str, start_sec: float, end_sec: float, n_frames: int, out_dir: Path) -> tuple[Frame, ...]:
    del video_path, end_sec, n_frames
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "frame_001.jpg"
    Image.new("RGB", (32, 18), color=(60, 120, 180)).save(path)
    return (Frame(frame_id="fr001", time_sec=round(float(start_sec), 3), path=str(path)),)


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


def test_lowfps_cache_and_window_sampling_manifests_keep_separate_lineage(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    low_frames = materialize_lowfps_frame_cache(workspace, fps=1.0, sampler=_fake_sampler)
    cached_frames = materialize_window_frames(workspace, 5.0, 6.0, query_id="q1", fps=1.0, sampler=_fake_sampler)
    sampled_frames = materialize_window_frames(workspace, 5.0, 6.0, query_id="q2", fps=2.0, sampler=_fake_sampler)

    low_manifest = workspace.root_dir / "frame_manifest.jsonl"
    window_manifest = workspace.root_dir / "observations" / "window_frame_manifest.jsonl"
    low_rows = [json.loads(line) for line in low_manifest.read_text(encoding="utf-8").splitlines()]
    window_rows = [json.loads(line) for line in window_manifest.read_text(encoding="utf-8").splitlines()]

    assert low_frames and cached_frames and sampled_frames
    assert all(frame.fps_level == "low" for frame in cached_frames)
    assert {row["fps_level"] for row in low_rows} == {"low"}
    assert {row["fps_level"] for row in window_rows} == {"window"}
    assert all("observations" not in row["path"] for row in low_rows)
    assert all(row["query_id"] == "q2" for row in window_rows)
    assert window_rows[0]["virtual_time_sec"] == 5.0
    assert window_rows[0]["source_video_id"] == "target"
    assert window_rows[0]["source_time_sec"] == 101.0


def test_materialized_frame_paths_are_unique_with_constant_sampler_names(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    low_frames = materialize_lowfps_frame_cache(workspace, fps=1.0, sampler=_constant_name_sampler)
    window_frames = materialize_window_frames(workspace, 5.0, 7.0, query_id="q_unique", fps=2.0, sampler=_constant_name_sampler)

    assert len({frame.path for frame in low_frames}) == len(low_frames)
    assert len({frame.path for frame in window_frames}) == len(window_frames)


def test_window_sampling_uniformly_covers_full_window_when_capped(tmp_path: Path) -> None:
    manifest = VirtualVideoManifest(
        workspace_id="long",
        segments=(VirtualVideoSegment("seg_long", "long", "long.mp4", 0.0, 600.0, 0.0, 600.0),),
    )
    case = VirtualVideoCase(
        case_id="long",
        question="What happens near the end?",
        options={"A": "start", "B": "end"},
        gold="B",
        target_segment_id="seg_long",
        target_virtual_interval=(580.0, 600.0),
    )
    workspace = VirtualVideoWorkspace.create(tmp_path / "long", manifest=manifest, case=case)

    frames = materialize_window_frames(workspace, 0.0, 600.0, query_id="q_long", fps=2.0, max_frames=64, sampler=_fake_sampler)

    times = [frame.virtual_time_sec for frame in frames]
    assert len(times) == 64
    assert times[0] == 0.0
    assert times[-1] >= 599.0
    assert times[32] > 300.0


def test_window_sampling_avoids_exact_source_duration_endpoint(tmp_path: Path) -> None:
    manifest = VirtualVideoManifest(
        workspace_id="endpoint",
        segments=(VirtualVideoSegment("seg_end", "source", "source.mp4", 0.0, 10.0, 0.0, 10.0),),
    )
    case = VirtualVideoCase(
        case_id="endpoint",
        question="What happens at the end?",
        options={"A": "nothing", "B": "an event"},
        gold="B",
        target_segment_id="seg_end",
        target_virtual_interval=(9.0, 10.0),
    )
    workspace = VirtualVideoWorkspace.create(tmp_path / "endpoint", manifest=manifest, case=case)
    sampled: list[float] = []
    attempted: list[float] = []

    def endpoint_strict_sampler(
        video_path: str,
        start_sec: float,
        end_sec: float,
        n_frames: int,
        out_dir: Path,
    ) -> tuple[Frame, ...]:
        del video_path, end_sec, n_frames
        attempted.append(start_sec)
        if start_sec > 9.0:
            raise RuntimeError("tail frames are not decodable")
        sampled.append(start_sec)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"frame_{start_sec:.3f}.jpg"
        Image.new("RGB", (16, 9), color=(0, 0, 0)).save(path)
        return (Frame(path.stem, start_sec, str(path)),)

    frames = materialize_window_frames(
        workspace,
        0.0,
        10.0,
        query_id="q_endpoint",
        fps=2.0,
        max_frames=3,
        sampler=endpoint_strict_sampler,
    )

    assert len(frames) == 3
    assert attempted[-3:] == [9.9, 9.4, 8.9]
    assert sampled[-1] == 8.9
    assert frames[-1].virtual_time_sec == 8.9


def test_near_equivalent_window_reuses_existing_observation_frames(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    sampled_times: list[float] = []

    def counting_sampler(video_path: str, start_sec: float, end_sec: float, n_frames: int, out_dir: Path) -> tuple[Frame, ...]:
        sampled_times.append(float(start_sec))
        return _constant_name_sampler(video_path, start_sec, end_sec, n_frames, out_dir)

    first = materialize_window_frames(
        workspace,
        4.0,
        10.0,
        query_id="q_first",
        fps=2.0,
        max_frames=64,
        sampler=counting_sampler,
    )
    first_call_count = len(sampled_times)
    reused = materialize_window_frames(
        workspace,
        4.2,
        9.8,
        query_id="q_nearby",
        fps=1.0,
        max_frames=8,
        sampler=counting_sampler,
    )

    assert first_call_count > 0
    assert len(sampled_times) == first_call_count
    assert reused
    assert {frame.path for frame in reused}.issubset({frame.path for frame in first})
    assert {frame.query_id for frame in reused} == {"q_first"}


def test_virtual_beat_index_uses_thumbnail_as_cold_keyframe_and_virtual_times(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    frames = materialize_lowfps_frame_cache(workspace, fps=1.0, sampler=_fake_sampler)
    workspace.write_asr_virtual_cues(({"start": 4.5, "end": 5.5, "text": "blue target jersey", "segment_id": "seg_b"},))

    result = build_virtual_beat_index(workspace, frames, model=ColorModel(), beat_sec=3.0)

    assert result.cold_index.search_text("target jersey")[0].beat_id == "bt00003"
    assert result.cold_index.search_visual("blue", k=1)[0].beat_id == "bt00003"
    assert result.cold_index.beats[0].start_sec == 0.0
    assert result.cold_index.beats[0].frame_times[0] == 0.0
    assert result.cold_index.beats[0].keyframe_path.endswith("_q0.jpg")
    assert Path(result.cold_index.beats[0].keyframe_path).exists()
    assert result.beat_index_path.exists()
    beat_rows = json.loads(result.beat_index_path.read_text(encoding="utf-8"))["beats"]
    assert beat_rows[2]["source_lineage"][0]["source_video_id"] == "target"
    assert len(beat_rows[2]["thumbnail_grid_paths"]) == 1
    assert all(Path(path).exists() for path in beat_rows[2]["thumbnail_grid_paths"])
    with Image.open(beat_rows[2]["thumbnail_grid_paths"][0]) as image:
        assert image.size == (480, 90)


def test_workspace_overview_caps_initial_segment_thumbnails_at_40(tmp_path: Path) -> None:
    segments = []
    virtual_start = 0.0
    for index in range(45):
        segments.append(
            VirtualVideoSegment(
                segment_id=f"seg_{index:04d}",
                source_video_id=f"vid_{index:04d}",
                source_path=f"video_{index:04d}.mp4",
                source_start_sec=0.0,
                source_end_sec=2.0,
                virtual_start_sec=virtual_start,
                virtual_end_sec=virtual_start + 2.0,
            )
        )
        virtual_start += 2.0
    manifest = VirtualVideoManifest(workspace_id="many", segments=tuple(segments))
    case = VirtualVideoCase(
        case_id="many",
        question="Which segment contains the relevant visual detail?",
        options={"A": "one", "B": "two"},
        gold="A",
        target_segment_id="seg_0000",
        target_virtual_interval=(0.0, 1.0),
    )
    workspace = VirtualVideoWorkspace.create(tmp_path / "many", manifest=manifest, case=case)
    frames = materialize_lowfps_frame_cache(workspace, fps=1.0, sampler=_fake_sampler)

    overview = build_workspace_overview(workspace, frames, thumbnail_budget=40)

    assert overview["workspace_duration_sec"] == 90.0
    assert len(overview["segment_overviews"]) <= 40
    assert overview["thumbnail_count"] <= 40
    assert overview["segment_overviews"][0]["kind"] == "page"
    assert len(overview["segment_overviews"][0]["segment_ids"]) > 1
    assert Path(overview["segment_overviews"][0]["overview_thumbnail_grid_path"]).exists()
    with Image.open(overview["segment_overviews"][0]["overview_thumbnail_grid_path"]) as image:
        assert image.size[0] <= 640
        assert image.size[1] <= 360
    assert overview["available_tools"] == ["open_segment", "inspect_window"]
