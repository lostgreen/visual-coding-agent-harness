from __future__ import annotations

from pathlib import Path

from visual_coding_agent_harness.tools.frame_cache import (
    FrameCache,
    FrameSample,
    build_extract_frame_cache_command,
)


def test_frame_cache_samples_only_requested_time_window() -> None:
    cache = FrameCache(
        video_path="/videos/demo.mp4",
        frame_dir=Path("/frames/demo"),
        fps=2.0,
        frames=tuple(
            FrameSample(timestamp_sec=index * 0.5, path=f"/frames/demo/frame_{index + 1:09d}.jpg")
            for index in range(8)
        ),
    )

    selected = cache.sample(start_sec=1.0, end_sec=2.1, max_frames=3)

    assert [round(frame.timestamp_sec, 3) for frame in selected] == [1.0, 1.5, 2.0]
    assert [Path(frame.path).name for frame in selected] == [
        "frame_000000003.jpg",
        "frame_000000004.jpg",
        "frame_000000005.jpg",
    ]


def test_frame_cache_uniformly_downsamples_large_windows() -> None:
    cache = FrameCache(
        video_path="/videos/demo.mp4",
        frame_dir=Path("/frames/demo"),
        fps=2.0,
        frames=tuple(
            FrameSample(timestamp_sec=index * 0.5, path=f"/frames/demo/frame_{index + 1:09d}.jpg")
            for index in range(10)
        ),
    )

    selected = cache.sample(start_sec=0.0, end_sec=5.0, max_frames=4)

    assert [round(frame.timestamp_sec, 3) for frame in selected] == [0.0, 1.5, 3.0, 4.5]


def test_extract_frame_cache_command_caps_video_at_two_fps() -> None:
    command = build_extract_frame_cache_command(
        video_path="/videos/demo.mp4",
        output_pattern="/frames/demo/frame_%09d.jpg",
        fps=2.0,
    )

    assert command == [
        "ffmpeg",
        "-y",
        "-i",
        "/videos/demo.mp4",
        "-vf",
        "fps=2.0",
        "-q:v",
        "2",
        "/frames/demo/frame_%09d.jpg",
    ]
