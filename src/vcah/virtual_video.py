from __future__ import annotations

from dataclasses import asdict, dataclass, field
import html
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence, TypeVar

from vcah.types import Frame
from vcah.video import sample_frames


FrameSampler = Callable[[str, float, float, int, Path], Sequence[Frame]]
T = TypeVar("T")
SRT_TIME_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+(?P<end>\d{2}:\d{2}:\d{2},\d{3})"
)
HTML_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class VirtualVideoSegment:
    segment_id: str
    source_video_id: str
    source_path: str
    source_start_sec: float
    source_end_sec: float
    virtual_start_sec: float
    virtual_end_sec: float
    role: str = "content"

    @property
    def duration_sec(self) -> float:
        return max(0.0, float(self.virtual_end_sec) - float(self.virtual_start_sec))


@dataclass(frozen=True)
class VirtualVideoManifest:
    workspace_id: str
    segments: tuple[VirtualVideoSegment, ...]
    duration_sec: float = 0.0

    def __post_init__(self) -> None:
        segments = tuple(_segment(item) for item in self.segments)
        object.__setattr__(self, "segments", segments)
        duration = float(self.duration_sec or 0.0)
        if duration <= 0.0 and segments:
            duration = max(float(segment.virtual_end_sec) for segment in segments)
        object.__setattr__(self, "duration_sec", round(duration, 3))

    def segment(self, segment_id: str) -> VirtualVideoSegment:
        for segment in self.segments:
            if segment.segment_id == segment_id:
                return segment
        raise ValueError(f"Unknown virtual segment: {segment_id}")


@dataclass(frozen=True)
class VirtualVideoCase:
    case_id: str
    question: str
    options: Mapping[str, str]
    gold: str
    target_segment_id: str
    target_virtual_interval: tuple[float, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", {str(k): str(v) for k, v in dict(self.options).items()})
        start, end = self.target_virtual_interval
        object.__setattr__(self, "target_virtual_interval", (float(start), float(end)))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class VirtualFrameRef:
    frame_id: str
    path: str
    virtual_time_sec: float
    segment_id: str
    source_video_id: str
    source_path: str
    source_time_sec: float
    fps_level: str
    query_id: str = ""
    sampling_fps: float = 0.0


@dataclass(frozen=True)
class SourceWindow:
    segment_id: str
    source_video_id: str
    source_path: str
    source_start_sec: float
    source_end_sec: float
    virtual_start_sec: float
    virtual_end_sec: float


@dataclass(frozen=True)
class VirtualVideoWorkspace:
    workspace_id: str
    root_dir: Path
    manifest: VirtualVideoManifest
    case: VirtualVideoCase
    frame_manifest: Path
    asr_virtual_cues: Path
    cold_index_dir: Path

    @classmethod
    def create(
        cls,
        root_dir: Path,
        *,
        manifest: VirtualVideoManifest,
        case: VirtualVideoCase,
    ) -> "VirtualVideoWorkspace":
        root = Path(root_dir)
        root.mkdir(parents=True, exist_ok=True)
        workspace = cls(
            workspace_id=manifest.workspace_id,
            root_dir=root,
            manifest=manifest,
            case=case,
            frame_manifest=root / "frame_manifest.jsonl",
            asr_virtual_cues=root / "asr_virtual_cues.json",
            cold_index_dir=root / "cold_index",
        )
        workspace.save()
        if not workspace.frame_manifest.exists():
            workspace.frame_manifest.write_text("", encoding="utf-8")
        if not workspace.asr_virtual_cues.exists():
            workspace.asr_virtual_cues.write_text("[]\n", encoding="utf-8")
        return workspace

    @classmethod
    def load(cls, root_dir: Path) -> "VirtualVideoWorkspace":
        root = Path(root_dir)
        manifest_payload = json.loads((root / "virtual_timeline.json").read_text(encoding="utf-8"))
        case_payload = json.loads((root / "case.json").read_text(encoding="utf-8"))
        manifest = VirtualVideoManifest(
            workspace_id=str(manifest_payload["workspace_id"]),
            duration_sec=float(manifest_payload.get("duration_sec", 0.0)),
            segments=tuple(VirtualVideoSegment(**item) for item in manifest_payload.get("segments", ())),
        )
        case = VirtualVideoCase(
            case_id=str(case_payload["case_id"]),
            question=str(case_payload["question"]),
            options=dict(case_payload.get("options", {})),
            gold=str(case_payload.get("gold", "")),
            target_segment_id=str(case_payload.get("target_segment_id", "")),
            target_virtual_interval=tuple(case_payload.get("target_virtual_interval", (0.0, 0.0))),  # type: ignore[arg-type]
            metadata=dict(case_payload.get("metadata", {})),
        )
        return cls(
            workspace_id=manifest.workspace_id,
            root_dir=root,
            manifest=manifest,
            case=case,
            frame_manifest=root / "frame_manifest.jsonl",
            asr_virtual_cues=root / "asr_virtual_cues.json",
            cold_index_dir=root / "cold_index",
        )

    def save(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        (self.root_dir / "virtual_timeline.json").write_text(
            json.dumps(_manifest_payload(self.manifest), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (self.root_dir / "case.json").write_text(
            json.dumps(_case_payload(self.case), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def write_asr_virtual_cues(self, cues: Sequence[Mapping[str, Any]]) -> None:
        self.asr_virtual_cues.write_text(
            json.dumps([dict(item) for item in cues], ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def read_asr_virtual_cues(self) -> tuple[dict[str, Any], ...]:
        if not self.asr_virtual_cues.exists():
            return ()
        return tuple(dict(item) for item in json.loads(self.asr_virtual_cues.read_text(encoding="utf-8") or "[]"))

    def read_frame_manifest(self) -> tuple[VirtualFrameRef, ...]:
        if not self.frame_manifest.exists():
            return ()
        rows = [json.loads(line) for line in self.frame_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        return tuple(VirtualFrameRef(**row) for row in rows)

    def read_window_frame_manifest(self) -> tuple[VirtualFrameRef, ...]:
        path = self.root_dir / "observations" / "window_frame_manifest.jsonl"
        if not path.exists():
            return ()
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return tuple(VirtualFrameRef(**row) for row in rows)


def virtual_to_source_windows(manifest: VirtualVideoManifest, start_sec: float, end_sec: float) -> tuple[SourceWindow, ...]:
    start = float(start_sec)
    end = float(end_sec)
    windows: list[SourceWindow] = []
    for segment in manifest.segments:
        overlap_start = max(start, float(segment.virtual_start_sec))
        overlap_end = min(end, float(segment.virtual_end_sec))
        if overlap_end <= overlap_start:
            continue
        source_start = float(segment.source_start_sec) + (overlap_start - float(segment.virtual_start_sec))
        source_end = float(segment.source_start_sec) + (overlap_end - float(segment.virtual_start_sec))
        windows.append(
            SourceWindow(
                segment_id=segment.segment_id,
                source_video_id=segment.source_video_id,
                source_path=segment.source_path,
                source_start_sec=round(source_start, 3),
                source_end_sec=round(source_end, 3),
                virtual_start_sec=round(overlap_start, 3),
                virtual_end_sec=round(overlap_end, 3),
            )
        )
    return tuple(windows)


def load_srt_as_virtual_cues(srt_path: Path, segment: VirtualVideoSegment) -> tuple[dict[str, Any], ...]:
    if not Path(srt_path).exists():
        return ()
    cues: list[dict[str, Any]] = []
    current_start: float | None = None
    current_end: float | None = None
    text_lines: list[str] = []

    def flush() -> None:
        nonlocal current_start, current_end, text_lines
        if current_start is None or current_end is None:
            text_lines = []
            return
        text = " ".join(_clean_srt_text(line) for line in text_lines if _clean_srt_text(line)).strip()
        if text and current_end >= segment.source_start_sec and current_start <= segment.source_end_sec:
            clipped_start = max(float(segment.source_start_sec), current_start)
            clipped_end = min(float(segment.source_end_sec), current_end)
            virtual_start = float(segment.virtual_start_sec) + (clipped_start - float(segment.source_start_sec))
            virtual_end = float(segment.virtual_start_sec) + (clipped_end - float(segment.source_start_sec))
            cues.append(
                {
                    "start": round(virtual_start, 3),
                    "end": round(virtual_end, 3),
                    "text": text,
                    "segment_id": segment.segment_id,
                    "source_video_id": segment.source_video_id,
                    "source_start": round(clipped_start, 3),
                    "source_end": round(clipped_end, 3),
                }
            )
        current_start = None
        current_end = None
        text_lines = []

    for raw in Path(srt_path).read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        if line.isdigit():
            continue
        match = SRT_TIME_RE.match(line)
        if match:
            flush()
            current_start = _parse_srt_time(match.group("start"))
            current_end = _parse_srt_time(match.group("end"))
            continue
        text_lines.append(line)
    flush()
    return tuple(cues)


def materialize_lowfps_frame_cache(
    workspace: VirtualVideoWorkspace,
    *,
    fps: float = 0.5,
    sampler: FrameSampler | None = None,
) -> tuple[VirtualFrameRef, ...]:
    sampler = sampler or sample_frames
    step = 1.0 / max(0.001, float(fps))
    rows: list[VirtualFrameRef] = []
    failures: list[dict[str, Any]] = []
    failures_path = workspace.root_dir / "frame_sampling_failures.jsonl"
    failures_path.unlink(missing_ok=True)
    frame_index = 1
    for segment in workspace.manifest.segments:
        duration = max(0.0, float(segment.virtual_end_sec) - float(segment.virtual_start_sec))
        count = max(1, int(duration * float(fps))) if duration > 0 else 0
        segment_frame_count = 0
        for offset_index in range(count):
            virtual_time = round(float(segment.virtual_start_sec) + offset_index * step, 3)
            if virtual_time >= float(segment.virtual_end_sec):
                virtual_time = round(float(segment.virtual_end_sec) - 0.001, 3)
            source_time = round(float(segment.source_start_sec) + (virtual_time - float(segment.virtual_start_sec)), 3)
            frame_dir = workspace.root_dir / "frames" / "low" / segment.segment_id / f"lo_{frame_index:06d}"
            try:
                frame = tuple(sampler(segment.source_path, source_time, source_time, 1, frame_dir))[0]
            except RuntimeError as exc:
                failures.append(
                    {
                        "error": str(exc),
                        "fps_level": "low",
                        "sampling_fps": float(fps),
                        "segment_id": segment.segment_id,
                        "source_path": segment.source_path,
                        "source_time_sec": source_time,
                        "source_video_id": segment.source_video_id,
                        "virtual_time_sec": virtual_time,
                    }
                )
                continue
            rows.append(
                VirtualFrameRef(
                    frame_id=f"lo_{frame_index:06d}",
                    path=str(frame.path),
                    virtual_time_sec=virtual_time,
                    segment_id=segment.segment_id,
                    source_video_id=segment.source_video_id,
                    source_path=segment.source_path,
                    source_time_sec=source_time,
                    fps_level="low",
                    sampling_fps=float(fps),
                )
            )
            frame_index += 1
            segment_frame_count += 1
        if count and segment_frame_count == 0:
            _write_jsonl(failures_path, failures)
            raise RuntimeError(f"Failed to sample any low-fps frames for segment {segment.segment_id}")
    if failures:
        _write_jsonl(failures_path, failures)
    _write_jsonl(workspace.frame_manifest, (asdict(row) for row in rows))
    return tuple(rows)


def materialize_highfps_window(
    workspace: VirtualVideoWorkspace,
    start_sec: float,
    end_sec: float,
    *,
    query_id: str,
    fps: float = 2.0,
    max_frames: int = 32,
    sampler: FrameSampler | None = None,
) -> tuple[VirtualFrameRef, ...]:
    return materialize_window_frames(
        workspace,
        start_sec,
        end_sec,
        query_id=query_id,
        fps=fps,
        max_frames=max_frames,
        sampler=sampler,
    )


def materialize_window_frames(
    workspace: VirtualVideoWorkspace,
    start_sec: float,
    end_sec: float,
    *,
    query_id: str,
    fps: float = 0.5,
    max_frames: int = 64,
    sampler: FrameSampler | None = None,
    phase_offset_sec: float = 0.0,
) -> tuple[VirtualFrameRef, ...]:
    requested_fps = float(fps)
    if requested_fps not in {0.5, 1.0, 2.0}:
        raise ValueError("inspect_window fps must be one of 0.5, 1.0, or 2.0")
    cap = max(1, min(512, int(max_frames)))
    cached = tuple(
        frame
        for frame in workspace.read_frame_manifest()
        if float(start_sec) <= frame.virtual_time_sec <= float(end_sec) and frame.fps_level == "low"
    )
    cache_fps = cached[0].sampling_fps if cached else 0.0
    phase_offset = max(0.0, float(phase_offset_sec or 0.0))
    if not phase_offset and cached and abs(cache_fps - requested_fps) < 1e-6:
        return _select_frame_refs(cached, cap)

    observed = () if phase_offset else _reusable_window_frames(
        workspace,
        float(start_sec),
        float(end_sec),
        requested_fps=requested_fps,
    )
    if observed:
        return select_uniform_items(observed, cap)

    sampler = sampler or sample_frames
    observations = workspace.root_dir / "observations"
    rows: list[VirtualFrameRef] = []
    failures: list[dict[str, Any]] = []
    for frame_index, virtual_time in enumerate(
        _uniform_times(
            float(start_sec),
            float(end_sec),
            requested_fps,
            cap,
            phase_offset_sec=phase_offset,
        ),
        start=1,
    ):
        window = _source_window_for_time(workspace.manifest, virtual_time)
        if window is None:
            continue
        segment = next(item for item in workspace.manifest.segments if item.segment_id == window.segment_id)
        source_time = window.source_start_sec + (virtual_time - window.virtual_start_sec)
        if source_time > segment.source_end_sec - 0.1:
            source_time = max(segment.source_start_sec, segment.source_end_sec - 0.1)
        source_time = round(source_time, 3)
        out_dir = observations / str(query_id) / window.segment_id / f"win_{frame_index:06d}"
        try:
            frame, source_time = _sample_window_frame_with_tail_backoff(
                sampler,
                window,
                source_time,
                out_dir,
                source_start_sec=segment.source_start_sec,
                source_end_sec=segment.source_end_sec,
            )
        except RuntimeError as exc:
            failures.append(
                {
                    "error": str(exc),
                    "fps_level": "window",
                    "frame_index": frame_index,
                    "query_id": str(query_id),
                    "requested_virtual_time_sec": virtual_time,
                    "sampling_fps": requested_fps,
                    "segment_id": window.segment_id,
                    "source_path": window.source_path,
                    "source_time_sec": source_time,
                    "source_video_id": window.source_video_id,
                }
            )
            continue
        sampled_virtual_time = round(
            window.virtual_start_sec + (source_time - window.source_start_sec),
            3,
        )
        rows.append(
            VirtualFrameRef(
                frame_id=f"win_{query_id}_{frame_index:06d}",
                path=str(frame.path),
                virtual_time_sec=sampled_virtual_time,
                segment_id=window.segment_id,
                source_video_id=window.source_video_id,
                source_path=window.source_path,
                source_time_sec=source_time,
                fps_level="window",
                query_id=str(query_id),
                sampling_fps=requested_fps,
            )
        )
    if failures:
        _append_jsonl(observations / "window_sampling_failures.jsonl", failures)
    if not rows:
        detail = failures[-1]["error"] if failures else "no source window overlapped the request"
        raise RuntimeError(f"Failed to sample any frames for query {query_id}: {detail}")
    manifest_path = observations / "window_frame_manifest.jsonl"
    _append_jsonl(manifest_path, (asdict(row) for row in rows))
    return tuple(rows)


def _sample_window_frame_with_tail_backoff(
    sampler: FrameSampler,
    window: SourceWindow,
    source_time: float,
    out_dir: Path,
    *,
    source_start_sec: float,
    source_end_sec: float,
) -> tuple[Frame, float]:
    candidates = [float(source_time)]
    if source_end_sec - source_time <= 2.0:
        candidates.extend(max(source_start_sec, source_time - delta) for delta in (0.5, 1.0, 2.0))
    last_error: RuntimeError | None = None
    for candidate in dict.fromkeys(round(value, 3) for value in candidates):
        try:
            frame = tuple(sampler(window.source_path, candidate, candidate, 1, out_dir))[0]
        except RuntimeError as exc:
            last_error = exc
            continue
        return frame, candidate
    if last_error is not None:
        raise last_error
    raise RuntimeError("Frame sampler returned no decodable frame")


def _uniform_times(
    start_sec: float,
    end_sec: float,
    fps: float,
    max_frames: int,
    *,
    phase_offset_sec: float = 0.0,
) -> tuple[float, ...]:
    duration = max(0.0, end_sec - start_sec)
    phase = max(0.0, float(phase_offset_sec or 0.0))
    if phase:
        step = 1.0 / max(float(fps), 1e-6)
        shifted = tuple(
            round(start_sec + phase + index * step, 3)
            for index in range(max(1, int(duration * fps) + 1))
            if start_sec + phase + index * step <= end_sec + 1e-9
        )
        if shifted:
            return select_uniform_items(shifted, max_frames)
    count = min(max(1, int(duration * fps)), max(1, int(max_frames)))
    if count == 1:
        return (round((start_sec + end_sec) / 2.0, 3),)
    span = end_sec - start_sec
    return tuple(round(start_sec + index * span / (count - 1), 3) for index in range(count))


def _source_window_for_time(manifest: VirtualVideoManifest, virtual_time_sec: float) -> SourceWindow | None:
    windows = virtual_to_source_windows(manifest, virtual_time_sec, virtual_time_sec + 0.001)
    if windows:
        return windows[0]
    adjusted = max(0.0, virtual_time_sec - 0.001)
    windows = virtual_to_source_windows(manifest, adjusted, virtual_time_sec)
    return windows[-1] if windows else None


def _select_frame_refs(frames: tuple[VirtualFrameRef, ...], max_frames: int) -> tuple[VirtualFrameRef, ...]:
    return select_uniform_items(frames, max_frames)


def select_uniform_items(items: Sequence[T], max_items: int) -> tuple[T, ...]:
    values = tuple(items)
    limit = max(1, int(max_items))
    if len(values) <= limit:
        return values
    if limit == 1:
        return (values[len(values) // 2],)
    indexes = [round(index * (len(values) - 1) / (limit - 1)) for index in range(limit)]
    return tuple(values[int(index)] for index in indexes)


def _reusable_window_frames(
    workspace: VirtualVideoWorkspace,
    start_sec: float,
    end_sec: float,
    *,
    requested_fps: float,
    min_source_iou: float = 0.8,
) -> tuple[VirtualFrameRef, ...]:
    groups: dict[str, list[VirtualFrameRef]] = {}
    for frame in workspace.read_window_frame_manifest():
        if frame.query_id and frame.sampling_fps + 1e-6 >= float(requested_fps):
            groups.setdefault(frame.query_id, []).append(frame)
    requested = virtual_to_source_windows(workspace.manifest, start_sec, end_sec)
    candidates = []
    for frames in groups.values():
        ordered = tuple(sorted(frames, key=lambda frame: frame.virtual_time_sec))
        iou = _frame_source_iou(ordered, requested)
        if iou >= float(min_source_iou):
            candidates.append((iou, -ordered[0].sampling_fps, ordered))
    if not candidates:
        return ()
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _frame_source_iou(frames: Sequence[VirtualFrameRef], requested: Sequence[SourceWindow]) -> float:
    existing: dict[tuple[str, str], tuple[float, float]] = {}
    for frame in frames:
        key = (frame.segment_id, frame.source_video_id)
        current = existing.get(key)
        if current is None:
            existing[key] = (frame.source_time_sec, frame.source_time_sec)
        else:
            existing[key] = (min(current[0], frame.source_time_sec), max(current[1], frame.source_time_sec))
    requested_ranges = {
        (window.segment_id, window.source_video_id): (window.source_start_sec, window.source_end_sec)
        for window in requested
    }
    keys = set(existing) | set(requested_ranges)
    intersection = 0.0
    union = 0.0
    for key in keys:
        observed = existing.get(key)
        wanted = requested_ranges.get(key)
        if observed is None:
            union += max(0.0, wanted[1] - wanted[0]) if wanted is not None else 0.0
            continue
        if wanted is None:
            union += max(0.0, observed[1] - observed[0])
            continue
        overlap = max(0.0, min(observed[1], wanted[1]) - max(observed[0], wanted[0]))
        intersection += overlap
        union += max(observed[1], wanted[1]) - min(observed[0], wanted[0])
    return intersection / union if union > 0.0 else 0.0


def _segment(value: VirtualVideoSegment | Mapping[str, Any]) -> VirtualVideoSegment:
    if isinstance(value, VirtualVideoSegment):
        return value
    return VirtualVideoSegment(**dict(value))


def _manifest_payload(manifest: VirtualVideoManifest) -> dict[str, Any]:
    return {
        "workspace_id": manifest.workspace_id,
        "duration_sec": manifest.duration_sec,
        "segments": [asdict(segment) for segment in manifest.segments],
    }


def _case_payload(case: VirtualVideoCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "question": case.question,
        "options": dict(case.options),
        "gold": case.gold,
        "target_segment_id": case.target_segment_id,
        "target_virtual_interval": list(case.target_virtual_interval),
        "metadata": dict(case.metadata),
    }


def _write_jsonl(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _append_jsonl(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _parse_srt_time(value: str) -> float:
    hours, minutes, seconds_ms = value.split(":")
    seconds, millis = seconds_ms.split(",")
    return int(hours) * 3600.0 + int(minutes) * 60.0 + int(seconds) + int(millis) / 1000.0


def _clean_srt_text(value: str) -> str:
    return " ".join(HTML_RE.sub(" ", html.unescape(str(value))).split())
