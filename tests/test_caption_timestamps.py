from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image
import pytest

import vcah.captioning as captioning
from vcah.caption_schema import (
    CaptionChunkV1,
    count_repaired_timestamp_tokens,
    parse_timestamp_anchors,
    split_caption_passages,
)
from vcah.captioning import (
    CaptionGenerationConfig,
    _caption_batch_timeout_sec,
    _caption_frame_workers,
    caption_cache_digest,
    materialize_caption_frames,
    run_caption_generation,
    source_manifest_digest,
)
from vcah.types import Frame
from vcah.virtual_video import VirtualVideoManifest, VirtualVideoSegment


class FixtureGenerator:
    model = "fixture-vlm"
    provider = "fixture"

    def __init__(self, *, fail_once: bool = False) -> None:
        self.calls = 0
        self.fail_once = fail_once

    def generate(
        self,
        image_paths: Sequence[str],
        prompt: str,
        *,
        image_labels: Sequence[str] = (),
    ) -> str:
        self.calls += 1
        assert image_paths
        assert len(image_labels) == len(image_paths)
        assert "Frame timestamps in attachment order" in prompt
        if self.fail_once and self.calls == 1:
            raise RuntimeError("transient fixture error")
        return "[00:00:01] The player opens a red door. [00:00:03] The player walks through it."

    @property
    def last_response_metadata(self) -> Mapping[str, Any]:
        return {
            "finish_reason": "stop",
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "reasoning_tokens": 0,
        }


class InvalidAnchorGenerator(FixtureGenerator):
    def generate(
        self,
        image_paths: Sequence[str],
        prompt: str,
        *,
        image_labels: Sequence[str] = (),
    ) -> str:
        self.calls += 1
        assert image_paths
        assert len(image_labels) == len(image_paths)
        assert "Frame timestamps in attachment order" in prompt
        return "[00:00:01] A visible event occurs. [01:01:06] A timestamp is malformed."


class OfficialRemaFixtureGenerator(FixtureGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.seen_prompt = ""

    def generate(
        self,
        image_paths: Sequence[str],
        prompt: str,
        *,
        image_labels: Sequence[str] = (),
    ) -> str:
        self.calls += 1
        self.seen_prompt = prompt
        assert "Frame timestamps in attachment order" not in prompt
        assert len(image_labels) == len(image_paths)
        sizes = []
        for path in image_paths:
            with Image.open(path) as image:
                sizes.append(image.size)
        assert all(size == (64, 36) for size in sizes)
        return "[00:00:01] A visible event occurs."


def _manifest(tmp_path: Path) -> VirtualVideoManifest:
    return VirtualVideoManifest(
        workspace_id="mmlifelong-game",
        segments=(
            VirtualVideoSegment("seg_0001", "one", str(tmp_path / "one.mp4"), 0.0, 4.0, 0.0, 4.0),
            VirtualVideoSegment("seg_0002", "two", str(tmp_path / "two.mp4"), 0.0, 6.0, 4.0, 10.0),
        ),
    )


def _asset_root(tmp_path: Path) -> tuple[Path, VirtualVideoManifest]:
    asset_root = tmp_path / "assets" / "game"
    asset_root.mkdir(parents=True)
    manifest = _manifest(tmp_path)
    (asset_root / "virtual_timeline.json").write_text(
        json.dumps(
            {
                "workspace_id": manifest.workspace_id,
                "duration_sec": manifest.duration_sec,
                "segments": [asdict(segment) for segment in manifest.segments],
            }
        ),
        encoding="utf-8",
    )
    return asset_root, manifest


def _sampler(
    video_path: str,
    start_sec: float,
    end_sec: float,
    n_frames: int,
    out_dir: Path,
) -> tuple[Frame, ...]:
    del video_path, end_sec, n_frames
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "frame.jpg"
    Image.new("RGB", (16, 9), color=(20, 60, 100)).save(path)
    return (Frame(path.stem, start_sec, str(path)),)


def test_deterministic_timestamp_shift_keeps_twenty_anchors_in_range() -> None:
    text = " ".join(f"[00:00:{index:02d}] event {index}." for index in range(20))

    anchors = parse_timestamp_anchors(text, chunk_start_sec=100.0, chunk_end_sec=120.0)

    assert len(anchors) == 20
    assert anchors[0].virtual_sec == 100.0
    assert anchors[-1].virtual_sec == 119.0
    assert all(100.0 <= anchor.virtual_sec <= 120.0 for anchor in anchors)


def test_timestamp_parser_rejects_anchor_outside_chunk() -> None:
    with pytest.raises(ValueError, match="outside chunk"):
        parse_timestamp_anchors(
            "[00:00:11] too late",
            chunk_start_sec=50.0,
            chunk_end_sec=60.0,
        )


def test_rema_timestamp_parser_repairs_duplicated_minute_in_hour_field() -> None:
    text = (
        "[01:01:43] The player opens a chest. "
        "[02:02:08] A large battle starts. "
        "[02:13] The battle continues. "
        "[03:35:00] The player leaves the cave. "
        "[02:22:21] Another event occurs. "
        "[00:24:00] An early event is restated."
    )

    with pytest.raises(ValueError, match="outside chunk"):
        parse_timestamp_anchors(text, chunk_start_sec=19800.0, chunk_end_sec=20100.0)

    anchors = parse_timestamp_anchors(
        text,
        chunk_start_sec=19800.0,
        chunk_end_sec=20100.0,
        repair_duplicate_minute_hour=True,
        repair_short_timestamp=True,
    )

    assert [anchor.local_sec for anchor in anchors] == [
        103.0,
        128.0,
        133.0,
        215.0,
        142.0,
        24.0,
    ]
    assert [anchor.virtual_sec for anchor in anchors] == [
        19903.0,
        19928.0,
        19933.0,
        20015.0,
        19942.0,
        19824.0,
    ]
    assert [anchor.label for anchor in anchors] == [
        "[00:01:43]",
        "[00:02:08]",
        "[00:02:13]",
        "[00:03:35]",
        "[00:02:22]",
        "[00:00:24]",
    ]
    assert count_repaired_timestamp_tokens(text, chunk_duration_sec=300.0) == 6


def test_passage_split_uses_anchor_intervals_and_chunk_fallback() -> None:
    anchored_text = "[00:00:02] Door opens. [00:00:05] Person enters."
    anchored = CaptionChunkV1(
        caption_id="cap-a",
        subset="game",
        virtual_start_sec=100.0,
        virtual_end_sec=110.0,
        source_segments=("seg_0001",),
        wall_clock_begin=None,
        wall_clock_end=None,
        text_raw=anchored_text,
        text_normalized=anchored_text,
        timestamp_anchors=parse_timestamp_anchors(
            anchored_text,
            chunk_start_sec=100.0,
            chunk_end_sec=110.0,
        ),
        model="fixture",
        provider="fixture",
        prompt_digest="p",
        generation_config_digest="g",
        source_manifest_digest="m",
        created_at="now",
    )
    fallback = CaptionChunkV1(
        **{
            **asdict(anchored),
            "caption_id": "cap-b",
            "text_raw": "Door opens. Person enters.",
            "text_normalized": "Door opens. Person enters.",
            "timestamp_anchors": (),
        }
    )

    anchored_passages = split_caption_passages(anchored)
    fallback_passages = split_caption_passages(fallback)

    assert [(item.virtual_start_sec, item.virtual_end_sec) for item in anchored_passages] == [
        (102.0, 105.0),
        (105.0, 110.0),
    ]
    assert all(item.metadata["interval_precision"] == "anchor" for item in anchored_passages)
    assert len(fallback_passages) == 2
    assert all(item.metadata["interval_precision"] == "chunk" for item in fallback_passages)


def test_caption_frame_sampling_crosses_segments_in_virtual_order(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    frames = materialize_caption_frames(
        manifest,
        0.0,
        10.0,
        out_dir=tmp_path / "frames",
        fps=1.0,
        max_frames=5,
        sampler=_sampler,
    )

    assert [frame.virtual_time_sec for frame in frames] == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert [frame.segment_id for frame in frames] == [
        "seg_0001",
        "seg_0001",
        "seg_0002",
        "seg_0002",
        "seg_0002",
    ]
    assert len({frame.virtual_time_sec for frame in frames}) == len(frames)


def test_caption_frame_worker_env_requires_a_positive_integer(monkeypatch) -> None:
    monkeypatch.setenv("VCAH_CAPTION_FRAME_WORKERS", "8")
    assert _caption_frame_workers() == 8

    monkeypatch.setenv("VCAH_CAPTION_FRAME_WORKERS", "invalid")
    assert _caption_frame_workers() == 4

    monkeypatch.setenv("VCAH_CAPTION_FRAME_WORKERS", "0")
    assert _caption_frame_workers() == 1


def test_caption_batch_timeout_env_has_a_bounded_default(monkeypatch) -> None:
    monkeypatch.setenv("VCAH_CAPTION_BATCH_TIMEOUT_SEC", "900")
    assert _caption_batch_timeout_sec() == 900.0

    monkeypatch.setenv("VCAH_CAPTION_BATCH_TIMEOUT_SEC", "invalid")
    assert _caption_batch_timeout_sec() == 600.0

    monkeypatch.setenv("VCAH_CAPTION_BATCH_TIMEOUT_SEC", "10")
    assert _caption_batch_timeout_sec() == 60.0


def test_caption_batch_sampling_decodes_each_contiguous_segment_once(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = []

    def fake_batch(video_path, *, start_sec, frame_count, fps, out_dir):
        calls.append((video_path, start_sec, frame_count, fps))
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for index in range(1, frame_count + 1):
            path = out_dir / f"frame_{index:06d}.jpg"
            Image.new("RGB", (16, 9), color=(20, 60, 100)).save(path)
            paths.append(path)
        return tuple(paths)

    monkeypatch.setattr(captioning, "_extract_caption_frame_batch", fake_batch)

    frames = materialize_caption_frames(
        _manifest(tmp_path),
        0.0,
        10.0,
        out_dir=tmp_path / "batch-frames",
        fps=1.0,
        max_frames=5,
        extraction_mode="fps_batch",
    )

    assert [frame.virtual_time_sec for frame in frames] == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert [frame.segment_id for frame in frames] == [
        "seg_0001",
        "seg_0001",
        "seg_0002",
        "seg_0002",
        "seg_0002",
    ]
    assert [(call[2], call[3]) for call in calls] == [(2, 0.5), (3, 0.5)]


def test_caption_batch_sampling_fills_missing_boundary_frame_with_exact_seek(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def short_batch(video_path, *, start_sec, frame_count, fps, out_dir):
        del video_path, start_sec, fps
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for index in range(1, frame_count):
            path = out_dir / f"frame_{index:06d}.jpg"
            Image.new("RGB", (16, 9), color=(20, 60, 100)).save(path)
            paths.append(path)
        return tuple(paths)

    monkeypatch.setattr(captioning, "_extract_caption_frame_batch", short_batch)
    monkeypatch.setattr(captioning, "sample_frames", _sampler)

    frames = materialize_caption_frames(
        _manifest(tmp_path),
        0.0,
        10.0,
        out_dir=tmp_path / "batch-boundary-frames",
        fps=1.0,
        max_frames=5,
        extraction_mode="fps_batch",
    )

    assert len(frames) == 5
    assert all(Path(frame.path).is_file() for frame in frames)
    assert sum("fallback_" in frame.path for frame in frames) == 2


def test_caption_run_retries_resumes_and_prompt_change_invalidates_cache(tmp_path: Path) -> None:
    asset_root, manifest = _asset_root(tmp_path)
    fixed_now = lambda: datetime(2026, 7, 21, tzinfo=timezone.utc)
    config = CaptionGenerationConfig(
        model="fixture-vlm",
        provider="fixture",
        chunk_sec=5.0,
        sample_fps=1.0,
        max_frames=3,
        max_retries=1,
    )
    generator = FixtureGenerator(fail_once=True)

    first = run_caption_generation(
        asset_root,
        config,
        generator,
        max_chunks=1,
        sampler=_sampler,
        now=fixed_now,
    )
    resumed = run_caption_generation(
        asset_root,
        config,
        generator,
        max_chunks=1,
        sampler=_sampler,
        now=fixed_now,
    )
    changed = CaptionGenerationConfig(
        model="fixture-vlm",
        provider="fixture",
        prompt=config.prompt + " Mention colors.",
        chunk_sec=5.0,
        sample_fps=1.0,
        max_frames=3,
        max_retries=0,
    )
    changed_generator = FixtureGenerator()
    changed_result = run_caption_generation(
        asset_root,
        changed,
        changed_generator,
        max_chunks=1,
        sampler=_sampler,
        now=fixed_now,
    )

    assert first.generated_chunks == 1
    assert generator.calls == 2
    assert resumed.generated_chunks == 0
    assert resumed.skipped_success_chunks == 1
    assert resumed.run_summary_path != first.run_summary_path
    assert len(first.store.successful_passages()) == 2
    assert changed_result.config_digest != first.config_digest
    assert changed_generator.calls == 1
    assert caption_cache_digest(config, source_manifest_digest(manifest)) == first.config_digest


def test_caption_run_workers_preserve_resume_state_and_clean_frames(tmp_path: Path) -> None:
    asset_root, _ = _asset_root(tmp_path)
    config = CaptionGenerationConfig(
        model="fixture-vlm",
        provider="fixture",
        chunk_sec=5.0,
        sample_fps=1.0,
        max_frames=3,
        max_retries=0,
    )
    generator = FixtureGenerator()

    first = run_caption_generation(
        asset_root,
        config,
        generator,
        workers=2,
        keep_frames=False,
        sampler=_sampler,
    )
    resumed = run_caption_generation(
        asset_root,
        config,
        generator,
        workers=2,
        keep_frames=False,
        sampler=_sampler,
    )
    summary = json.loads(first.run_summary_path.read_text(encoding="utf-8"))
    frame_root = first.store.root / "frames" / first.config_digest

    assert first.generated_chunks == 2
    assert first.failed_chunks == 0
    assert first.store.status_counts()["success"] == 2
    assert generator.calls == 2
    assert not tuple(frame_root.rglob("*.jpg"))
    assert summary["workers"] == 2
    assert summary["keep_frames"] is False
    assert summary["usage_totals"]["completion_tokens"] == 40
    assert resumed.generated_chunks == 0
    assert resumed.skipped_success_chunks == 2


def test_caption_run_keeps_valid_anchors_when_one_timestamp_is_out_of_range(tmp_path: Path) -> None:
    asset_root, _ = _asset_root(tmp_path)
    config = CaptionGenerationConfig(
        model="fixture-vlm",
        provider="fixture",
        chunk_sec=5.0,
        sample_fps=1.0,
        max_frames=3,
        max_retries=0,
    )
    result = run_caption_generation(
        asset_root,
        config,
        InvalidAnchorGenerator(),
        max_chunks=1,
        sampler=_sampler,
    )
    chunk = result.store.successful_chunks()[0]

    assert result.generated_chunks == 1
    assert result.failed_chunks == 0
    assert [anchor.local_sec for anchor in chunk.timestamp_anchors] == [1.0]
    assert chunk.metadata["timestamp_parse_status"] == "filtered_invalid"
    assert chunk.metadata["timestamp_token_count"] == 2
    assert chunk.metadata["valid_timestamp_anchor_count"] == 1
    assert "outside chunk" in chunk.metadata["timestamp_parse_warning"]


def test_caption_run_supports_official_rema_image_and_prompt_contract(tmp_path: Path) -> None:
    asset_root, manifest = _asset_root(tmp_path)
    prompt = "Official ReMA prompt text."
    config = CaptionGenerationConfig(
        model="fixture-vlm",
        provider="fixture",
        prompt=prompt,
        chunk_sec=5.0,
        sample_fps=1.0,
        max_frames=3,
        max_retries=0,
        image_width=64,
        image_height=36,
        jpeg_quality=75,
        append_timestamp_map=False,
        timestamp_shift_mode="deterministic_rema_v3",
    )
    generator = OfficialRemaFixtureGenerator()

    result = run_caption_generation(
        asset_root,
        config,
        generator,
        max_chunks=1,
        sampler=_sampler,
    )
    chunk = result.store.successful_chunks()[0]
    preprocessing = chunk.metadata["image_preprocessing"]
    default_config = CaptionGenerationConfig(model="fixture-vlm", provider="fixture")

    assert generator.seen_prompt == prompt
    assert preprocessing["width"] == 64
    assert preprocessing["height"] == 36
    assert preprocessing["jpeg_quality"] == 75
    assert preprocessing["jpeg_bytes"] > 0
    assert preprocessing["estimated_base64_bytes"] >= preprocessing["jpeg_bytes"]
    assert chunk.metadata["timestamp_repair_count"] == 0
    assert config.generation_config_digest != default_config.generation_config_digest
    assert caption_cache_digest(config, source_manifest_digest(manifest)) == result.config_digest
