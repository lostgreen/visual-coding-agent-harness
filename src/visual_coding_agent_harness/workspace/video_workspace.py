"""Two-speed cold workspace facade for long-video search."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Callable, Sequence

from visual_coding_agent_harness.backends.embedding import EmbeddingBackend
from visual_coding_agent_harness.video.index import Frame
from visual_coding_agent_harness.video.pipeline import detect_shots_ffmpeg, detect_shots_uniform, sample_shot_frames, shots_to_beats
from visual_coding_agent_harness.workspace.text_index import InvertedIndex
from visual_coding_agent_harness.workspace.visual_index import BeatHit, VisualIndex


ShotDetector = Callable[[str, float], Sequence[tuple[float, float]]]
KeyframeSampler = Callable[[str, float, float, int, Path], Sequence[Frame]]
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True)
class Beat:
    beat_id: str
    chapter_id: str
    start_sec: float
    end_sec: float
    keyframe_path: str
    asr_verbatim: str
    ocr_verbatim: tuple[str, ...]
    shot_ids: tuple[str, ...]
    micro_caption: str | None = None

    def __post_init__(self) -> None:
        if self.end_sec < self.start_sec:
            raise ValueError("Beat end_sec must be greater than or equal to start_sec")
        object.__setattr__(self, "start_sec", float(self.start_sec))
        object.__setattr__(self, "end_sec", float(self.end_sec))
        object.__setattr__(self, "ocr_verbatim", _text_tuple(self.ocr_verbatim))
        object.__setattr__(self, "shot_ids", _text_tuple(self.shot_ids))


@dataclass(frozen=True)
class Chapter:
    chapter_id: str
    start_sec: float
    end_sec: float
    beat_ids: tuple[str, ...]
    thumb_path: str
    title: str | None = None

    def __post_init__(self) -> None:
        if self.end_sec < self.start_sec:
            raise ValueError("Chapter end_sec must be greater than or equal to start_sec")
        object.__setattr__(self, "start_sec", float(self.start_sec))
        object.__setattr__(self, "end_sec", float(self.end_sec))
        object.__setattr__(self, "beat_ids", _text_tuple(self.beat_ids))


@dataclass
class VideoWorkspace:
    video_path: str
    duration_sec: float
    chapters: tuple[Chapter, ...]
    beats: tuple[Beat, ...]
    text_index: InvertedIndex
    visual_index: VisualIndex
    memos: dict[str, object] = field(default_factory=dict)
    evidence: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        self.duration_sec = float(self.duration_sec)
        self.chapters = tuple(self.chapters)
        self.beats = tuple(self.beats)
        self.evidence = tuple(self.evidence)

    def search_text(self, query: str, *, modality: Sequence[str] = ("asr", "ocr")) -> tuple[BeatHit, ...]:
        return self.text_index.search(query, modality=modality)

    def search_visual(self, query: str, k: int = 20) -> tuple[BeatHit, ...]:
        return self.visual_index.search(query, k=k)

    def get_beat(self, beat_id: str) -> Beat:
        for beat in self.beats:
            if beat.beat_id == beat_id:
                return beat
        raise ValueError(f"Unknown beat_id: {beat_id}")

    def get_chapter(self, chapter_id: str) -> Chapter:
        for chapter in self.chapters:
            if chapter.chapter_id == chapter_id:
                return chapter
        raise ValueError(f"Unknown chapter_id: {chapter_id}")

    def window(self, beat_id: str, *, before: int = 2, after: int = 2) -> tuple[Beat, ...]:
        index = next((idx for idx, beat in enumerate(self.beats) if beat.beat_id == beat_id), None)
        if index is None:
            raise ValueError(f"Unknown beat_id: {beat_id}")
        start = max(0, index - max(0, int(before)))
        end = min(len(self.beats), index + max(0, int(after)) + 1)
        return self.beats[start:end]

    def beats_in_chapters(self, chapter_ids: Sequence[str]) -> tuple[Beat, ...]:
        allowed = set(_text_tuple(chapter_ids))
        return tuple(beat for beat in self.beats if beat.chapter_id in allowed)

    def timeline_text(self, max_chapters: int = 40, *, fill_missing_titles: bool = False) -> str:
        if not self.chapters:
            return "(no chapters indexed)"
        if fill_missing_titles:
            self._fill_missing_titles()
        lines = []
        for chapter in self.chapters[: max(0, int(max_chapters))]:
            title = chapter.title or "(title pending)"
            lines.append(
                f"{chapter.chapter_id}  [{_clock(chapter.start_sec)}-{_clock(chapter.end_sec)}]  "
                f"{title:<32} | {len(chapter.beat_ids)} beats"
            )
        remaining = len(self.chapters) - max(0, int(max_chapters))
        if remaining > 0:
            lines.append(f"... {remaining} more chapters omitted")
        return "\n".join(lines)

    def _fill_missing_titles(self) -> None:
        updated = []
        for chapter in self.chapters:
            if chapter.title is not None:
                updated.append(chapter)
                continue
            beats = self.beats_in_chapters((chapter.chapter_id,))
            title = _chapter_title_from_beats(beats)
            updated.append(replace(chapter, title=title))
        self.chapters = tuple(updated)

    def beat_metadata_text(self, beat_ids: Sequence[str]) -> str:
        lines = []
        for beat_id in _text_tuple(beat_ids):
            beat = self.get_beat(beat_id)
            parts = [f"{beat.beat_id}  [{_clock(beat.start_sec)}-{_clock(beat.end_sec)}]"]
            if beat.asr_verbatim:
                parts.append(f'asr: "{_bounded(beat.asr_verbatim, 220)}"')
            if beat.ocr_verbatim:
                parts.append(f"ocr: ({'; '.join(_bounded(item, 120) for item in beat.ocr_verbatim)})")
            lines.append("  ".join(parts))
        return "\n".join(lines)

    def save(self, artifact_dir: Path) -> None:
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "video_path": self.video_path,
            "duration_sec": self.duration_sec,
            "chapters": [asdict(chapter) for chapter in self.chapters],
            "beats": [asdict(beat) for beat in self.beats],
        }
        (artifact_dir / "workspace.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        self.text_index.save(artifact_dir / "text_index.json")
        self.visual_index.save(artifact_dir / "visual_index.npz")

    @classmethod
    def load(cls, artifact_dir: Path, embedding_backend: EmbeddingBackend) -> "VideoWorkspace":
        artifact_dir = Path(artifact_dir)
        payload = json.loads((artifact_dir / "workspace.json").read_text(encoding="utf-8"))
        beats = tuple(Beat(**item) for item in payload.get("beats", ()))
        chapters = tuple(Chapter(**item) for item in payload.get("chapters", ()))
        return cls(
            video_path=str(payload.get("video_path") or ""),
            duration_sec=float(payload.get("duration_sec", 0.0) or 0.0),
            chapters=chapters,
            beats=beats,
            text_index=InvertedIndex.load(artifact_dir / "text_index.json"),
            visual_index=VisualIndex.load(artifact_dir / "visual_index.npz", embedding_backend),
        )


def build_video_workspace(
    video_path: str,
    duration_sec: float,
    *,
    artifact_dir: Path,
    asr_cues: Sequence[Any] = (),
    ocr_lines_by_time: Sequence[Any] = (),
    embedding_backend: EmbeddingBackend,
    max_chapters: int = 40,
    shot_detector: ShotDetector | None = None,
    keyframe_sampler: KeyframeSampler | None = None,
) -> VideoWorkspace:
    """Build the cold Tier 0 workspace without VLM calls."""

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    shot_ranges = tuple((float(start), float(end)) for start, end in _detect_shot_ranges(video_path, duration_sec, detector=shot_detector))
    if not shot_ranges:
        shot_ranges = tuple(detect_shots_uniform(duration_sec, window=max(1.0, min(float(duration_sec), 15.0))))
    sampler = keyframe_sampler or _sample_one_keyframe
    shot_frames: list[tuple[Frame, ...]] = []
    keyframes: list[str] = []
    for shot_number, (start_sec, end_sec) in enumerate(shot_ranges, start=1):
        frames = tuple(sampler(str(video_path), float(start_sec), float(end_sec), 1, artifact_dir / "shot_keyframes" / f"sh{shot_number:05d}"))
        shot_frames.append(frames)
        keyframes.append(frames[0].thumb_path if frames else "")
    groups = shots_to_beats(shot_ranges, keyframes, sim_threshold=0.85, max_beat_sec=60.0)
    beats_without_chapters = tuple(
        _make_beat(
            beat_number=beat_number,
            group=group,
            shot_ranges=shot_ranges,
            shot_frames=shot_frames,
            asr_cues=asr_cues,
            ocr_lines_by_time=ocr_lines_by_time,
        )
        for beat_number, group in enumerate(groups, start=1)
    )
    chapters, beats = _assign_chapters(beats_without_chapters, duration_sec=float(duration_sec), max_chapters=max_chapters)
    text_index = InvertedIndex()
    for beat in beats:
        text_index.add(beat.beat_id, beat.asr_verbatim, modality="asr")
        text_index.add(beat.beat_id, " ".join(beat.ocr_verbatim), modality="ocr")
    visual_index = VisualIndex(embedding_backend)
    visual_index.build(beats)
    return VideoWorkspace(
        video_path=str(video_path),
        duration_sec=float(duration_sec),
        chapters=chapters,
        beats=beats,
        text_index=text_index,
        visual_index=visual_index,
    )


def _make_beat(
    *,
    beat_number: int,
    group: Sequence[int],
    shot_ranges: Sequence[tuple[float, float]],
    shot_frames: Sequence[Sequence[Frame]],
    asr_cues: Sequence[Any],
    ocr_lines_by_time: Sequence[Any],
) -> Beat:
    start_sec = min(shot_ranges[index][0] for index in group)
    end_sec = max(shot_ranges[index][1] for index in group)
    first_frames = tuple(shot_frames[group[0]]) if group else ()
    return Beat(
        beat_id=f"bt{beat_number:05d}",
        chapter_id="",
        start_sec=float(start_sec),
        end_sec=float(end_sec),
        keyframe_path=first_frames[0].thumb_path if first_frames else "",
        asr_verbatim=_asr_text_for_range(asr_cues, start_sec=float(start_sec), end_sec=float(end_sec)),
        ocr_verbatim=_ocr_lines_for_range(ocr_lines_by_time, start_sec=float(start_sec), end_sec=float(end_sec)),
        shot_ids=tuple(f"sh{index + 1:05d}" for index in group),
    )


def _assign_chapters(beats: Sequence[Beat], *, duration_sec: float, max_chapters: int) -> tuple[tuple[Chapter, ...], tuple[Beat, ...]]:
    if not beats:
        return (), ()
    limit = max(1, int(max_chapters))
    target_sec = max(60.0, float(duration_sec) / float(limit))
    groups: list[list[Beat]] = []
    current: list[Beat] = []
    current_start = float(beats[0].start_sec)
    for beat in beats:
        if current and len(groups) + 1 < limit and float(beat.end_sec) - current_start > target_sec:
            groups.append(current)
            current = []
            current_start = float(beat.start_sec)
        current.append(beat)
    if current:
        groups.append(current)
    while len(groups) > limit:
        tail = groups.pop()
        groups[-1].extend(tail)

    chapters: list[Chapter] = []
    assigned: list[Beat] = []
    for chapter_number, group in enumerate(groups, start=1):
        chapter_id = f"ch{chapter_number:02d}"
        beat_ids = tuple(beat.beat_id for beat in group)
        chapters.append(
            Chapter(
                chapter_id=chapter_id,
                start_sec=group[0].start_sec,
                end_sec=group[-1].end_sec,
                beat_ids=beat_ids,
                thumb_path=group[0].keyframe_path,
                title=None,
            )
        )
        assigned.extend(replace(beat, chapter_id=chapter_id) for beat in group)
    return tuple(chapters), tuple(assigned)


def _detect_shot_ranges(video_path: str, duration_sec: float, *, detector: ShotDetector | None) -> Sequence[tuple[float, float]]:
    if detector is not None:
        return detector(str(video_path), float(duration_sec))
    try:
        return detect_shots_ffmpeg(str(video_path))
    except RuntimeError:
        return detect_shots_uniform(float(duration_sec), window=15.0)


def _sample_one_keyframe(video_path: str, start_sec: float, end_sec: float, n_frames: int, out_dir: Path) -> Sequence[Frame]:
    del n_frames
    return sample_shot_frames(video_path, start_sec, end_sec, n_frames=1, out_dir=out_dir)


def _asr_text_for_range(cues: Sequence[Any], *, start_sec: float, end_sec: float) -> str:
    lines = []
    for cue in cues:
        cue_start = float(_field(cue, "start_sec", _field(cue, "start", 0.0)) or 0.0)
        cue_end = float(_field(cue, "end_sec", _field(cue, "end", cue_start)) or cue_start)
        if cue_end < start_sec or cue_start > end_sec:
            continue
        text = str(_field(cue, "text", "") or "").strip()
        if text:
            lines.append(text)
    return " ".join(lines)


def _ocr_lines_for_range(lines_by_time: Sequence[Any], *, start_sec: float, end_sec: float) -> tuple[str, ...]:
    lines = []
    for item in lines_by_time:
        if isinstance(item, dict):
            time_sec = float(item.get("time_sec", item.get("time", 0.0)) or 0.0)
            text = str(item.get("text", "") or "").strip()
        else:
            try:
                time_raw, text_raw = item
            except (TypeError, ValueError):
                continue
            time_sec = float(time_raw)
            text = str(text_raw or "").strip()
        if start_sec <= time_sec <= end_sec and text:
            lines.append(text)
    return tuple(lines)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _text_tuple(value: object) -> tuple[str, ...]:
    if value is None or isinstance(value, (str, bytes)):
        values = () if value is None else (value,)
    else:
        try:
            values = tuple(value)  # type: ignore[arg-type]
        except TypeError:
            values = (value,)
    return tuple(text for item in values if (text := str(item).strip()))


def _clock(seconds: float) -> str:
    seconds_int = max(0, int(round(float(seconds))))
    minutes, sec = divmod(seconds_int, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _bounded(text: str, max_chars: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, int(max_chars) - 3)].rstrip() + "..."


def _chapter_title_from_beats(beats: Sequence[Beat]) -> str:
    stop = {"the", "and", "that", "this", "with", "from", "there", "here", "then", "than", "into", "onto"}
    counts: Counter[str] = Counter()
    for beat in beats:
        for token in _text_tuple(_TOKEN_RE.findall(beat.asr_verbatim.casefold())):
            if len(token) > 2 and token not in stop:
                counts[token] += 1
    if not counts:
        return "untitled chapter"
    return " / ".join(token for token, _count in counts.most_common(3))
