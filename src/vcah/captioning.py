from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping, Protocol, Sequence

from PIL import Image

from vcah.caption_schema import (
    CaptionChunkV1,
    count_repaired_timestamp_tokens,
    count_timestamp_tokens,
    normalize_caption_text,
    parse_timestamp_anchors,
    split_caption_passages,
    stable_digest,
    text_digest,
)
from vcah.caption_store import CaptionStore
from vcah.model_client import OpenAICompatibleClient
from vcah.types import Frame
from vcah.video import sample_frames
from vcah.virtual_video import (
    FrameSampler,
    VirtualFrameRef,
    VirtualVideoManifest,
    VirtualVideoSegment,
    virtual_to_source_windows,
)


DEFAULT_CAPTION_PROMPT = """Describe the attached video frames in chronological order.
You are a multimodal video understanding assistant. Generate a detailed caption for this video chunk.

Requirements:
1. Analyze visible actions, expressions, scene elements, objects, people, state changes, counts, and event order.
2. Describe visible text, including subtitles, signs, menus, notifications, labels, and HUD elements.
3. Each attached image is immediately preceded by its local timestamp. Put one of those timestamps in [HH:MM:SS] format at the start of each important sentence or segment, where [00:00:00] is the start of this chunk. Include at most 10 timestamps.
4. Use chronological natural language, at least one sentence per timestamped segment, without repeating information.
5. Do not speculate; describe only what is directly observable in the attached frames.

Return only the timestamped chronological caption."""
class CaptionGenerator(Protocol):
    model: str
    provider: str

    def generate(
        self,
        image_paths: Sequence[str],
        prompt: str,
        *,
        image_labels: Sequence[str] = (),
    ) -> str:
        ...

    @property
    def last_response_metadata(self) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class CaptionGenerationConfig:
    model: str
    provider: str
    prompt: str = DEFAULT_CAPTION_PROMPT
    chunk_sec: float = 300.0
    sample_fps: float = 1.0
    max_frames: int = 300
    timestamp_shift_mode: str = "deterministic"
    max_retries: int = 2
    max_tokens: int = 1800
    image_width: int | None = None
    image_height: int | None = None
    jpeg_quality: int | None = None
    append_timestamp_map: bool = True
    frame_extraction_mode: str = "seek"

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", str(self.model))
        object.__setattr__(self, "provider", str(self.provider))
        object.__setattr__(self, "prompt", str(self.prompt))
        object.__setattr__(self, "chunk_sec", float(self.chunk_sec))
        object.__setattr__(self, "sample_fps", float(self.sample_fps))
        object.__setattr__(self, "max_frames", int(self.max_frames))
        object.__setattr__(self, "max_retries", max(0, int(self.max_retries)))
        object.__setattr__(self, "max_tokens", max(1, int(self.max_tokens)))
        object.__setattr__(self, "append_timestamp_map", bool(self.append_timestamp_map))
        object.__setattr__(self, "frame_extraction_mode", str(self.frame_extraction_mode))
        image_values = (self.image_width, self.image_height, self.jpeg_quality)
        if any(value is not None for value in image_values):
            if not all(value is not None for value in image_values):
                raise ValueError("Caption image width, height, and JPEG quality must be set together")
            object.__setattr__(self, "image_width", int(self.image_width))
            object.__setattr__(self, "image_height", int(self.image_height))
            object.__setattr__(self, "jpeg_quality", int(self.jpeg_quality))
            if self.image_width <= 0 or self.image_height <= 0:
                raise ValueError("Caption image dimensions must be positive")
            if not 1 <= self.jpeg_quality <= 95:
                raise ValueError("Caption JPEG quality must be between 1 and 95")
        if self.chunk_sec <= 0.0:
            raise ValueError("Caption chunk_sec must be positive")
        if self.sample_fps <= 0.0:
            raise ValueError("Caption sample_fps must be positive")
        if self.max_frames <= 0:
            raise ValueError("Caption max_frames must be positive")
        if self.timestamp_shift_mode not in {
            "deterministic",
            "deterministic_rema",
            "deterministic_rema_v2",
            "deterministic_rema_v3",
        }:
            raise ValueError("Unsupported deterministic timestamp shifting mode")
        if self.frame_extraction_mode not in {"seek", "fps_batch"}:
            raise ValueError("Caption frame extraction mode must be 'seek' or 'fps_batch'")

    @property
    def prompt_digest(self) -> str:
        return text_digest(self.prompt)

    @property
    def generation_config_digest(self) -> str:
        payload: dict[str, Any] = {
            "chunk_sec": self.chunk_sec,
            "sample_fps": self.sample_fps,
            "max_frames": self.max_frames,
            "timestamp_shift_mode": self.timestamp_shift_mode,
            "max_tokens": self.max_tokens,
        }
        if self.image_width is not None:
            payload["image_preprocessing"] = {
                "width": self.image_width,
                "height": self.image_height,
                "jpeg_quality": self.jpeg_quality,
            }
        if not self.append_timestamp_map:
            payload["append_timestamp_map"] = False
        if self.frame_extraction_mode != "seek":
            payload["frame_extraction_mode"] = self.frame_extraction_mode
        return stable_digest(payload)


@dataclass(frozen=True)
class CaptionChunkSpec:
    ordinal: int
    virtual_start_sec: float
    virtual_end_sec: float
    source_segments: tuple[str, ...]
    wall_clock_begin: str | None
    wall_clock_end: str | None
    cache_key: str


@dataclass(frozen=True)
class CaptionRunResult:
    config_digest: str
    store: CaptionStore
    run_summary_path: Path
    requested_chunks: int
    generated_chunks: int
    skipped_success_chunks: int
    failed_chunks: int

    def summary(self) -> dict[str, Any]:
        return {
            "config_digest": self.config_digest,
            "requested_chunks": self.requested_chunks,
            "generated_chunks": self.generated_chunks,
            "skipped_success_chunks": self.skipped_success_chunks,
            "failed_chunks": self.failed_chunks,
            "status_counts": self.store.status_counts(),
            "chunks_path": str(self.store.chunks_path),
            "passages_path": str(self.store.passages_path),
            "run_summary_path": str(self.run_summary_path),
        }


class OpenAICompatibleCaptionGenerator:
    def __init__(self, api: OpenAICompatibleClient, *, provider: str, max_tokens: int) -> None:
        self.api = api
        self.model = api.model
        self.provider = str(provider)
        self.max_tokens = int(max_tokens)

    def generate(
        self,
        image_paths: Sequence[str],
        prompt: str,
        *,
        image_labels: Sequence[str] = (),
    ) -> str:
        return self.api.chat(
            prompt,
            image_paths=image_paths,
            image_labels=image_labels,
            prompt_position="last",
            max_tokens=self.max_tokens,
        )

    @property
    def last_response_metadata(self) -> Mapping[str, Any]:
        return self.api.last_response_metadata


def run_caption_generation(
    asset_root: Path,
    config: CaptionGenerationConfig,
    generator: CaptionGenerator,
    *,
    resume: bool = True,
    max_chunks: int | None = None,
    start_chunk: int = 0,
    workers: int = 1,
    keep_frames: bool = True,
    sampler: FrameSampler | None = None,
    now: Callable[[], datetime] | None = None,
) -> CaptionRunResult:
    asset_root = Path(asset_root)
    if str(generator.model) != config.model or str(generator.provider) != config.provider:
        raise ValueError("Caption generator model/provider does not match the generation config")
    manifest = load_asset_manifest(asset_root)
    manifest_digest = source_manifest_digest(manifest)
    config_digest = caption_cache_digest(config, manifest_digest)
    store = CaptionStore(asset_root, config_digest, eager_exports=False)
    if store.state_path.exists() and not resume and store.status_counts()["success"]:
        raise FileExistsError(f"Caption cache already contains successful chunks: {store.state_path}")
    recovered = store.recover_interrupted() if resume else 0
    specs = caption_chunk_specs(manifest, config, manifest_digest=manifest_digest)
    start_index = max(0, int(start_chunk))
    selected = specs[start_index:]
    if max_chunks is not None:
        selected = selected[: max(0, int(max_chunks))]

    clock = now or (lambda: datetime.now(timezone.utc))
    generated = 0
    skipped = 0
    failed = 0
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
    pending_specs: list[CaptionChunkSpec] = []
    for spec in selected:
        store.prepare(spec.cache_key, asdict(spec))
        record = store.record(spec.cache_key) or {}
        if record.get("status") == "success":
            skipped += 1
            continue
        if int(record.get("attempt_count", 0) or 0) > config.max_retries:
            failed += 1
            continue
        pending_specs.append(spec)

    def generate_spec(spec: CaptionChunkSpec) -> tuple[str, dict[str, int]]:
        record = store.record(spec.cache_key) or {}
        chunk_usage = {key: 0 for key in usage_totals}
        frame_root = store.root / "frames" / config_digest / spec.cache_key[:16]
        while int(record.get("attempt_count", 0) or 0) <= config.max_retries:
            store.begin(spec.cache_key)
            try:
                frames = materialize_caption_frames(
                    manifest,
                    spec.virtual_start_sec,
                    spec.virtual_end_sec,
                    out_dir=frame_root,
                    fps=config.sample_fps,
                    max_frames=config.max_frames,
                    sampler=sampler,
                    extraction_mode=config.frame_extraction_mode,
                )
                image_preprocessing: Mapping[str, Any] = {}
                if config.image_width is not None:
                    image_preprocessing = preprocess_caption_images(
                        frames,
                        width=config.image_width,
                        height=config.image_height,
                        jpeg_quality=config.jpeg_quality or 75,
                    )
                prompt = (
                    caption_prompt(config.prompt, frames, chunk_start_sec=spec.virtual_start_sec)
                    if config.append_timestamp_map
                    else config.prompt.strip()
                )
                image_labels = tuple(
                    _format_timestamp(frame.virtual_time_sec - spec.virtual_start_sec)
                    for frame in frames
                )
                text_raw = generator.generate(
                    tuple(frame.path for frame in frames),
                    prompt,
                    image_labels=image_labels,
                )
                timestamp_parse_status = "strict"
                timestamp_parse_warning = ""
                repair_rema_timestamps = config.timestamp_shift_mode.startswith("deterministic_rema")
                try:
                    anchors = parse_timestamp_anchors(
                        text_raw,
                        chunk_start_sec=spec.virtual_start_sec,
                        chunk_end_sec=spec.virtual_end_sec,
                        strict=True,
                        repair_duplicate_minute_hour=repair_rema_timestamps,
                        repair_short_timestamp=repair_rema_timestamps,
                    )
                except ValueError as exc:
                    anchors = parse_timestamp_anchors(
                        text_raw,
                        chunk_start_sec=spec.virtual_start_sec,
                        chunk_end_sec=spec.virtual_end_sec,
                        strict=False,
                        repair_duplicate_minute_hour=repair_rema_timestamps,
                        repair_short_timestamp=repair_rema_timestamps,
                    )
                    timestamp_parse_status = "filtered_invalid" if anchors else "chunk_fallback"
                    timestamp_parse_warning = str(exc)[:300]
                response_metadata = dict(generator.last_response_metadata)
                for key in chunk_usage:
                    value = response_metadata.get(key)
                    if isinstance(value, (int, float)):
                        chunk_usage[key] += int(value)
                created_at = clock().astimezone(timezone.utc).isoformat()
                caption_id = f"cap_{spec.ordinal:06d}_{spec.cache_key[:12]}"
                chunk = CaptionChunkV1(
                    caption_id=caption_id,
                    subset=_manifest_subset(manifest),
                    virtual_start_sec=spec.virtual_start_sec,
                    virtual_end_sec=spec.virtual_end_sec,
                    source_segments=spec.source_segments,
                    wall_clock_begin=spec.wall_clock_begin,
                    wall_clock_end=spec.wall_clock_end,
                    text_raw=text_raw,
                    text_normalized=normalize_caption_text(text_raw),
                    timestamp_anchors=anchors,
                    model=config.model,
                    provider=config.provider,
                    prompt_digest=config.prompt_digest,
                    generation_config_digest=config.generation_config_digest,
                    source_manifest_digest=manifest_digest,
                    created_at=created_at,
                    metadata={
                        "cache_key": spec.cache_key,
                        "frame_count": len(frames),
                        "frame_virtual_times": [frame.virtual_time_sec for frame in frames],
                        "image_preprocessing": dict(image_preprocessing),
                        "frames_retained": bool(keep_frames),
                        "timestamp_parse_status": timestamp_parse_status,
                        "timestamp_parse_warning": timestamp_parse_warning,
                        "timestamp_token_count": count_timestamp_tokens(text_raw),
                        "timestamp_repair_count": count_repaired_timestamp_tokens(
                            text_raw,
                            chunk_duration_sec=spec.virtual_end_sec - spec.virtual_start_sec,
                        )
                        if repair_rema_timestamps
                        else 0,
                        "valid_timestamp_anchor_count": len(anchors),
                        "response_metadata": response_metadata,
                    },
                )
                store.mark_success(spec.cache_key, chunk, split_caption_passages(chunk))
                return "generated", chunk_usage
            except Exception as exc:
                store.mark_failed(spec.cache_key, f"{type(exc).__name__}: {exc}")
                record = store.record(spec.cache_key) or {}
                if int(record.get("attempt_count", 0) or 0) > config.max_retries:
                    return "failed", chunk_usage
            finally:
                if not keep_frames:
                    _cleanup_caption_frame_dir(frame_root)
        return "failed", chunk_usage

    worker_count = max(1, int(workers))
    if worker_count == 1:
        outcomes = (generate_spec(spec) for spec in pending_specs)
        for status, chunk_usage in outcomes:
            generated += status == "generated"
            failed += status == "failed"
            for key, value in chunk_usage.items():
                usage_totals[key] += value
    else:
        with ThreadPoolExecutor(max_workers=min(worker_count, max(1, len(pending_specs)))) as executor:
            futures = tuple(executor.submit(generate_spec, spec) for spec in pending_specs)
            for future in as_completed(futures):
                status, chunk_usage = future.result()
                generated += status == "generated"
                failed += status == "failed"
                for key, value in chunk_usage.items():
                    usage_totals[key] += value

    store.flush_exports()
    finished_at = clock().astimezone(timezone.utc)
    run_id = finished_at.strftime("%Y%m%dT%H%M%SZ")
    run_summary_path = store.write_run_summary(
        run_id,
        {
            "asset_root": str(asset_root),
            "model": config.model,
            "provider": config.provider,
            "prompt_digest": config.prompt_digest,
            "generation_config_digest": config.generation_config_digest,
            "source_manifest_digest": manifest_digest,
            "requested_chunks": len(selected),
            "generated_chunks": generated,
            "skipped_success_chunks": skipped,
            "failed_chunks": failed,
            "recovered_running_chunks": recovered,
            "workers": worker_count,
            "keep_frames": bool(keep_frames),
            "status_counts": store.status_counts(),
            "usage_totals": usage_totals,
            "finished_at": finished_at.isoformat(),
        },
    )
    return CaptionRunResult(
        config_digest=config_digest,
        store=store,
        run_summary_path=run_summary_path,
        requested_chunks=len(selected),
        generated_chunks=generated,
        skipped_success_chunks=skipped,
        failed_chunks=failed,
    )


def load_asset_manifest(asset_root: Path) -> VirtualVideoManifest:
    path = Path(asset_root) / "virtual_timeline.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return VirtualVideoManifest(
        workspace_id=str(payload["workspace_id"]),
        duration_sec=float(payload.get("duration_sec", 0.0)),
        segments=tuple(VirtualVideoSegment(**item) for item in payload.get("segments", ())),
    )


def source_manifest_digest(manifest: VirtualVideoManifest) -> str:
    return stable_digest(
        {
            "workspace_id": manifest.workspace_id,
            "duration_sec": manifest.duration_sec,
            "segments": [asdict(segment) for segment in manifest.segments],
        }
    )


def caption_cache_digest(config: CaptionGenerationConfig, manifest_digest: str) -> str:
    return stable_digest(
        {
            "source_manifest_digest": manifest_digest,
            "model": config.model,
            "provider": config.provider,
            "prompt_digest": config.prompt_digest,
            "generation_config_digest": config.generation_config_digest,
        }
    )


def caption_chunk_specs(
    manifest: VirtualVideoManifest,
    config: CaptionGenerationConfig,
    *,
    manifest_digest: str | None = None,
) -> tuple[CaptionChunkSpec, ...]:
    source_digest = manifest_digest or source_manifest_digest(manifest)
    specs: list[CaptionChunkSpec] = []
    start = 0.0
    ordinal = 0
    while start < manifest.duration_sec:
        end = min(manifest.duration_sec, start + config.chunk_sec)
        windows = virtual_to_source_windows(manifest, start, end)
        if not windows:
            raise ValueError(f"Caption chunk [{start}, {end}] does not map to a source segment")
        source_segments = tuple(dict.fromkeys(window.segment_id for window in windows))
        segment_map = {segment.segment_id: segment for segment in manifest.segments}
        first_segment = segment_map[source_segments[0]]
        last_segment = segment_map[source_segments[-1]]
        cache_key = stable_digest(
            {
                "source_manifest_digest": source_digest,
                "virtual_start_sec": round(start, 3),
                "virtual_end_sec": round(end, 3),
                "model": config.model,
                "provider": config.provider,
                "prompt_digest": config.prompt_digest,
                "generation_config_digest": config.generation_config_digest,
            }
        )
        specs.append(
            CaptionChunkSpec(
                ordinal=ordinal,
                virtual_start_sec=round(start, 3),
                virtual_end_sec=round(end, 3),
                source_segments=source_segments,
                wall_clock_begin=first_segment.wall_clock_begin,
                wall_clock_end=last_segment.wall_clock_end,
                cache_key=cache_key,
            )
        )
        start = end
        ordinal += 1
    return tuple(specs)


def materialize_caption_frames(
    manifest: VirtualVideoManifest,
    start_sec: float,
    end_sec: float,
    *,
    out_dir: Path,
    fps: float,
    max_frames: int,
    sampler: FrameSampler | None = None,
    extraction_mode: str = "seek",
) -> tuple[VirtualFrameRef, ...]:
    start = float(start_sec)
    end = float(end_sec)
    if end <= start:
        raise ValueError("Caption frame interval must have positive duration")
    count = max(1, min(int(max_frames), int(math.ceil((end - start) * float(fps)))))
    times = _uniform_times(start, end, count)
    mode = str(extraction_mode)
    if mode == "fps_batch":
        if sampler is not None:
            raise ValueError("fps_batch caption extraction does not accept a custom sampler")
        return _materialize_caption_frames_batch(
            manifest,
            times,
            out_dir=out_dir,
            sampling_fps=float(fps),
        )
    if mode != "seek":
        raise ValueError(f"Unsupported caption frame extraction mode: {mode}")
    sampler = sampler or sample_frames
    segments = {segment.segment_id: segment for segment in manifest.segments}

    def materialize(item: tuple[int, float]) -> VirtualFrameRef:
        index, virtual_time = item
        windows = virtual_to_source_windows(manifest, virtual_time, min(end, virtual_time + 0.001))
        if not windows and virtual_time > start:
            windows = virtual_to_source_windows(manifest, virtual_time - 0.001, virtual_time)
        if not windows:
            raise ValueError(f"Caption frame at {virtual_time} does not map to a source segment")
        window = windows[0]
        segment = segments[window.segment_id]
        source_time = segment.source_start_sec + (virtual_time - segment.virtual_start_sec)
        source_time = min(source_time, max(segment.source_start_sec, segment.source_end_sec - 0.1))
        frame_dir = Path(out_dir) / window.segment_id / f"frame_{index:06d}"
        sampled = tuple(sampler(window.source_path, source_time, source_time, 1, frame_dir))
        if not sampled:
            raise RuntimeError(f"Caption sampler returned no frame at {virtual_time}")
        frame: Frame = sampled[0]
        return VirtualFrameRef(
            frame_id=f"caption_{index:06d}",
            path=str(frame.path),
            virtual_time_sec=round(virtual_time, 3),
            segment_id=window.segment_id,
            source_video_id=window.source_video_id,
            source_path=window.source_path,
            source_time_sec=round(source_time, 3),
            fps_level="caption",
            query_id="caption",
            sampling_fps=float(fps),
        )

    with ThreadPoolExecutor(max_workers=min(_caption_frame_workers(), len(times))) as executor:
        return tuple(executor.map(materialize, enumerate(times, start=1)))


def _caption_frame_workers() -> int:
    try:
        return max(1, int(os.environ.get("VCAH_CAPTION_FRAME_WORKERS", "4")))
    except ValueError:
        return 4


def _caption_batch_timeout_sec() -> float:
    try:
        return max(60.0, float(os.environ.get("VCAH_CAPTION_BATCH_TIMEOUT_SEC", "600")))
    except ValueError:
        return 600.0


def _materialize_caption_frames_batch(
    manifest: VirtualVideoManifest,
    times: Sequence[float],
    *,
    out_dir: Path,
    sampling_fps: float,
) -> tuple[VirtualFrameRef, ...]:
    segments = {segment.segment_id: segment for segment in manifest.segments}
    mapped: list[tuple[int, float, VirtualVideoSegment, float]] = []
    for index, virtual_time in enumerate(times, start=1):
        windows = virtual_to_source_windows(manifest, virtual_time, virtual_time + 0.001)
        if not windows and index > 1:
            windows = virtual_to_source_windows(manifest, virtual_time - 0.001, virtual_time)
        if not windows:
            raise ValueError(f"Caption frame at {virtual_time} does not map to a source segment")
        segment = segments[windows[0].segment_id]
        source_time = segment.source_start_sec + (virtual_time - segment.virtual_start_sec)
        source_time = min(source_time, max(segment.source_start_sec, segment.source_end_sec - 0.1))
        mapped.append((index, virtual_time, segment, source_time))

    groups: list[list[tuple[int, float, VirtualVideoSegment, float]]] = []
    for item in mapped:
        if not groups:
            groups.append([item])
            continue
        previous = groups[-1][-1]
        same_segment = item[2].segment_id == previous[2].segment_id
        virtual_step = item[1] - previous[1]
        source_step = item[3] - previous[3]
        if same_segment and abs(source_step - virtual_step) <= 0.01:
            groups[-1].append(item)
        else:
            groups.append([item])

    refs: list[VirtualFrameRef] = []
    for group_index, group in enumerate(groups, start=1):
        step = group[1][1] - group[0][1] if len(group) > 1 else 1.0 / sampling_fps
        output_fps = 1.0 / max(0.001, step)
        paths = _extract_caption_frame_batch(
            group[0][2].source_path,
            start_sec=group[0][3],
            frame_count=len(group),
            fps=output_fps,
            out_dir=Path(out_dir) / f"batch_{group_index:04d}_{group[0][2].segment_id}",
        )
        if len(paths) < len(group):
            completed_paths = list(paths)
            for missing_index, item in enumerate(group[len(paths) :], start=len(paths) + 1):
                fallback_dir = (
                    Path(out_dir)
                    / f"batch_{group_index:04d}_{group[0][2].segment_id}"
                    / f"fallback_{missing_index:06d}"
                )
                sampled = tuple(sample_frames(item[2].source_path, item[3], item[3], 1, fallback_dir))
                if not sampled:
                    raise RuntimeError(f"Caption fallback sampler returned no frame at {item[1]}")
                completed_paths.append(Path(sampled[0].path))
            paths = tuple(completed_paths)
        for (index, virtual_time, segment, source_time), path in zip(group, paths):
            refs.append(
                VirtualFrameRef(
                    frame_id=f"caption_{index:06d}",
                    path=str(path),
                    virtual_time_sec=round(virtual_time, 3),
                    segment_id=segment.segment_id,
                    source_video_id=segment.source_video_id,
                    source_path=segment.source_path,
                    source_time_sec=round(source_time, 3),
                    fps_level="caption",
                    query_id="caption",
                    sampling_fps=float(sampling_fps),
                )
            )
    return tuple(sorted(refs, key=lambda frame: frame.virtual_time_sec))


def _extract_caption_frame_batch(
    video_path: str,
    *,
    start_sec: float,
    frame_count: int,
    fps: float,
    out_dir: Path,
) -> tuple[Path, ...]:
    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for stale in output_root.glob("frame_*.jpg"):
        stale.unlink(missing_ok=True)
    output_pattern = output_root / "frame_%06d.jpg"
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{float(start_sec):.3f}",
        "-i",
        str(video_path),
        "-vf",
        f"fps={float(fps):.8f}",
        "-frames:v",
        str(int(frame_count)),
        "-q:v",
        "2",
        str(output_pattern),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=_caption_batch_timeout_sec(),
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to batch-sample Caption frames") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("ffmpeg timed out while batch-sampling Caption frames") from exc
    paths = tuple(sorted(output_root.glob("frame_*.jpg")))
    if completed.returncode != 0 or len(paths) > int(frame_count):
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-2:]
        raise RuntimeError(
            f"ffmpeg Caption batch failed with {len(paths)}/{int(frame_count)} frames: {' | '.join(tail)}"
        )
    return paths


def _cleanup_caption_frame_dir(root: Path) -> None:
    path = Path(root)
    if not path.is_dir():
        return
    descendants = tuple(path.rglob("*"))
    for candidate in descendants:
        if candidate.is_file():
            candidate.unlink(missing_ok=True)
    for candidate in sorted(
        (item for item in descendants if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        try:
            candidate.rmdir()
        except OSError:
            pass
    try:
        path.rmdir()
    except OSError:
        pass


def preprocess_caption_images(
    frames: Sequence[VirtualFrameRef],
    *,
    width: int,
    height: int,
    jpeg_quality: int,
) -> dict[str, Any]:
    size = (int(width), int(height))
    quality = int(jpeg_quality)
    paths = tuple(Path(frame.path) for frame in frames)
    source_bytes = sum(path.stat().st_size for path in paths)

    def preprocess(path: Path) -> None:
        temporary = path.with_name(f".{path.name}.preprocessed.tmp")
        with Image.open(path) as image:
            resized = image.convert("RGB").resize(size, Image.Resampling.BICUBIC)
            resized.save(temporary, format="JPEG", quality=quality)
        temporary.replace(path)

    with ThreadPoolExecutor(max_workers=min(16, max(1, len(paths)))) as executor:
        tuple(executor.map(preprocess, paths))
    jpeg_bytes = sum(path.stat().st_size for path in paths)
    return {
        "width": size[0],
        "height": size[1],
        "jpeg_quality": quality,
        "source_bytes": source_bytes,
        "jpeg_bytes": jpeg_bytes,
        "estimated_base64_bytes": 4 * ((jpeg_bytes + 2) // 3),
    }


def caption_prompt(
    base_prompt: str,
    frames: Sequence[VirtualFrameRef],
    *,
    chunk_start_sec: float,
) -> str:
    timestamp_map = ", ".join(
        f"{index}={_format_timestamp(frame.virtual_time_sec - float(chunk_start_sec))}"
        for index, frame in enumerate(frames, start=1)
    )
    return f"{str(base_prompt).strip()}\n\nFrame timestamps in attachment order: {timestamp_map}"


def _uniform_times(start: float, end: float, count: int) -> tuple[float, ...]:
    if count <= 1:
        return (round(start, 3),)
    step = (end - start) / count
    return tuple(round(start + index * step, 3) for index in range(count))


def _format_timestamp(seconds: float) -> str:
    value = max(0, int(round(float(seconds))))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"


def _manifest_subset(manifest: VirtualVideoManifest) -> str:
    prefix = "mmlifelong-"
    return manifest.workspace_id[len(prefix) :] if manifest.workspace_id.startswith(prefix) else manifest.workspace_id
