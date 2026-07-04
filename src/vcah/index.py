from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, replace
import json
from math import log
from pathlib import Path
import re
from statistics import median
from typing import Any, Callable, Sequence

import numpy as np

from vcah.model import ModelClient
from vcah.types import Beat, Chapter, Frame, Hit, IndexDiagnostics
from vcah.video import detect_frame_ranges_uniform, frame_ranges_to_beats, render_timeline_grid, sample_frames


RangeDetector = Callable[[str, float], Sequence[tuple[float, float]]]
KeyframeSampler = Callable[[str, float, float, int, Path], Sequence[Frame]]
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


class TextIndex:
    def __init__(self) -> None:
        self.documents: dict[str, dict[str, str]] = {}
        self._postings: dict[str, dict[str, dict[str, int]]] = {"asr": defaultdict(dict), "ocr": defaultdict(dict)}
        self._tokens: dict[str, dict[str, tuple[str, ...]]] = {}

    def add(self, beat_id: str, text: str, *, modality: str) -> None:
        if modality not in ("asr", "ocr"):
            raise ValueError("modality must be 'asr' or 'ocr'")
        beat_id = str(beat_id)
        self.documents.setdefault(beat_id, {})[modality] = str(text or "")
        tokens = _tokenize(text)
        self._tokens.setdefault(beat_id, {})[modality] = tokens
        counts: dict[str, int] = defaultdict(int)
        for token in tokens:
            counts[token] += 1
        for token, count in counts.items():
            self._postings[modality][token][beat_id] = count

    def search(self, query: str, *, modality: Sequence[str] = ("asr", "ocr")) -> tuple[Hit, ...]:
        modalities = tuple(item for item in modality if item in ("asr", "ocr"))
        if not modalities:
            return ()
        phrase = _quoted_phrase(query)
        if phrase is not None:
            return self._search_phrase(phrase, modalities)
        return self._search_terms(query, modalities)

    @classmethod
    def from_documents(cls, documents: dict[str, dict[str, str]]) -> "TextIndex":
        index = cls()
        for beat_id, by_modality in documents.items():
            for modality, text in by_modality.items():
                if modality in ("asr", "ocr"):
                    index.add(beat_id, text, modality=modality)
        return index

    def _search_terms(self, query: str, modalities: Sequence[str]) -> tuple[Hit, ...]:
        terms = _tokenize(query)
        if not terms:
            return ()
        scores: dict[str, float] = defaultdict(float)
        total_docs = max(1, len(self.documents))
        for modality in modalities:
            for term in terms:
                postings = self._postings[modality].get(term, {})
                if not postings:
                    continue
                weight = log(1.0 + total_docs / max(1, len(postings)))
                for beat_id, tf in postings.items():
                    scores[beat_id] += float(tf) * weight
        return _rank(scores, modality="text")

    def _search_phrase(self, phrase: str, modalities: Sequence[str]) -> tuple[Hit, ...]:
        phrase_tokens = _tokenize(phrase)
        scores: dict[str, float] = defaultdict(float)
        for beat_id, by_modality in self._tokens.items():
            for modality in modalities:
                tokens = by_modality.get(modality, ())
                if _contains_phrase(tokens, phrase_tokens):
                    scores[beat_id] += float(len(phrase_tokens))
        return _rank(scores, modality="text")


class VisualIndex:
    def __init__(self, model: ModelClient) -> None:
        self.model = model
        self.beat_ids: tuple[str, ...] = ()
        self.embeddings = np.zeros((0, int(getattr(model, "embedding_dim", 0) or 0)), dtype=np.float32)

    def build(self, beats: Sequence[Beat]) -> None:
        self.beat_ids = tuple(beat.beat_id for beat in beats)
        if not self.beat_ids:
            return
        rows = np.asarray(self.model.embed_image([beat.keyframe_path for beat in beats]), dtype=np.float32)
        if rows.ndim != 2 or rows.shape[0] != len(self.beat_ids):
            raise ValueError("embed_image must return an (N, D) array")
        self.embeddings = _l2_normalize(rows)

    def search(self, query: str, *, k: int = 20) -> tuple[Hit, ...]:
        if not self.beat_ids or self.embeddings.size == 0 or k <= 0:
            return ()
        query_vec = np.asarray(self.model.embed_text((query,)), dtype=np.float32)
        if query_vec.ndim != 2 or query_vec.shape[0] != 1:
            raise ValueError("embed_text must return a (1, D) array")
        scores = self.embeddings @ _l2_normalize(query_vec)[0]
        order = sorted(range(len(scores)), key=lambda idx: (-float(scores[idx]), self.beat_ids[idx]))
        return tuple(Hit(self.beat_ids[idx], float(scores[idx]), "visual") for idx in order[: max(0, int(k))])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, beat_ids=np.asarray(self.beat_ids, dtype=str), embeddings=self.embeddings)

    @classmethod
    def load(cls, path: Path, model: ModelClient) -> "VisualIndex":
        index = cls(model)
        payload = np.load(path, allow_pickle=False)
        index.beat_ids = tuple(str(item) for item in payload["beat_ids"].tolist())
        index.embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        return index


class ColdIndex:
    def __init__(
        self,
        *,
        video_path: str,
        duration_sec: float,
        chapters: Sequence[Chapter],
        beats: Sequence[Beat],
        text_index: TextIndex,
        visual_index: VisualIndex,
        diagnostics: IndexDiagnostics,
    ) -> None:
        self.video_path = str(video_path)
        self.duration_sec = float(duration_sec)
        self.chapters = tuple(chapters)
        self.beats = tuple(beats)
        self.text_index = text_index
        self.visual_index = visual_index
        self.diagnostics = diagnostics

    def search_text(self, query: str) -> tuple[Hit, ...]:
        return self.text_index.search(query)

    def search_visual(self, query: str, *, k: int = 20) -> tuple[Hit, ...]:
        return self.visual_index.search(query, k=k)

    def get_beat(self, beat_id: str) -> Beat:
        for beat in self.beats:
            if beat.beat_id == beat_id:
                return beat
        raise ValueError(f"Unknown beat_id: {beat_id}")

    def window(self, beat_id: str, *, before_sec: float = 60.0, after_sec: float = 60.0) -> tuple[Beat, ...]:
        target = self.get_beat(beat_id)
        start = max(0.0, target.start_sec - max(0.0, float(before_sec)))
        end = target.end_sec + max(0.0, float(after_sec))
        return tuple(beat for beat in self.beats if beat.end_sec >= start and beat.start_sec <= end)

    def timeline_digest(self) -> str:
        lines = [f"{len(self.chapters)} chapters, {len(self.beats)} beats"]
        for chapter in self.chapters[:12]:
            lines.append(
                f"{chapter.chapter_id} [{_clock(chapter.start_sec)}-{_clock(chapter.end_sec)}] "
                f"{len(chapter.beat_ids)} beats"
            )
        return "\n".join(lines)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "video_path": self.video_path,
            "duration_sec": self.duration_sec,
            "chapters": [asdict(chapter) for chapter in self.chapters],
            "beats": [asdict(beat) for beat in self.beats],
            "text_documents": self.text_index.documents,
        }
        (path / "index.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        (path / "diagnostics.json").write_text(
            json.dumps(asdict(self.diagnostics), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.visual_index.save(path / "visual_index.npz")

    @classmethod
    def load(cls, path: Path, *, model: ModelClient | None = None) -> "ColdIndex":
        path = Path(path)
        model = model or ModelClient()
        payload = json.loads((path / "index.json").read_text(encoding="utf-8"))
        diagnostics = IndexDiagnostics(**json.loads((path / "diagnostics.json").read_text(encoding="utf-8")))
        return cls(
            video_path=str(payload["video_path"]),
            duration_sec=float(payload["duration_sec"]),
            chapters=tuple(Chapter(**item) for item in payload.get("chapters", ())),
            beats=tuple(Beat(**item) for item in payload.get("beats", ())),
            text_index=TextIndex.from_documents(payload.get("text_documents", {})),
            visual_index=VisualIndex.load(path / "visual_index.npz", model),
            diagnostics=diagnostics,
        )


def build_cold_index(
    video_path: str,
    *,
    duration_sec: float,
    run_dir: Path,
    model: ModelClient | None = None,
    asr_cues: Sequence[Any] = (),
    ocr_lines: Sequence[Any] = (),
    range_detector: RangeDetector | None = None,
    keyframe_sampler: KeyframeSampler | None = None,
    max_chapters: int = 40,
    max_range_sec: float = 60.0,
    max_beat_sec: float = 60.0,
    index_mode: str = "fast",
) -> ColdIndex:
    model = model or ModelClient()
    cold_dir = Path(run_dir) / "cold_index"
    cold_dir.mkdir(parents=True, exist_ok=True)
    raw_ranges = tuple(range_detector(video_path, duration_sec) if range_detector else detect_frame_ranges_uniform(duration_sec, window=15.0))
    frame_ranges = _normalize_ranges(raw_ranges, duration_sec=duration_sec, max_range_sec=max_range_sec)
    if not frame_ranges:
        frame_ranges = detect_frame_ranges_uniform(duration_sec, window=min(15.0, max(1.0, duration_sec)))
    sampler = keyframe_sampler or sample_frames
    frames_by_range = tuple(
        tuple(sampler(video_path, start, end, 1, cold_dir / "frames" / f"rg{idx:05d}"))
        for idx, (start, end) in enumerate(frame_ranges, start=1)
    )
    keyframes = tuple(frames[0].path if frames else "" for frames in frames_by_range)
    groups = frame_ranges_to_beats(frame_ranges, keyframes, max_beat_sec=max_beat_sec)
    beats = tuple(
        _make_beat(idx, group, frame_ranges, frames_by_range, asr_cues=asr_cues, ocr_lines=ocr_lines)
        for idx, group in enumerate(groups, start=1)
    )
    chapters, beats = _assign_chapters(beats, duration_sec=duration_sec, max_chapters=max_chapters)
    text_index = TextIndex()
    for beat in beats:
        text_index.add(beat.beat_id, beat.asr_text, modality="asr")
        text_index.add(beat.beat_id, " ".join(beat.ocr_text), modality="ocr")
    visual_index = VisualIndex(model)
    visual_index.build(beats)
    cold = ColdIndex(
        video_path=video_path,
        duration_sec=duration_sec,
        chapters=chapters,
        beats=beats,
        text_index=text_index,
        visual_index=visual_index,
        diagnostics=_diagnostics(duration_sec, chapters, beats, visual_index, index_mode),
    )
    cold.save(cold_dir)
    render_timeline_grid([beat.keyframe_path for beat in beats], cold_dir / "timeline.jpg")
    return cold


def _make_beat(
    number: int,
    group: Sequence[int],
    frame_ranges: Sequence[tuple[float, float]],
    frames_by_range: Sequence[Sequence[Frame]],
    *,
    asr_cues: Sequence[Any],
    ocr_lines: Sequence[Any],
) -> Beat:
    start_sec = min(frame_ranges[index][0] for index in group)
    end_sec = max(frame_ranges[index][1] for index in group)
    frame_paths = tuple(frame.path for index in group for frame in frames_by_range[index])
    return Beat(
        beat_id=f"bt{number:05d}",
        chapter_id="",
        start_sec=start_sec,
        end_sec=end_sec,
        keyframe_path=frame_paths[0] if frame_paths else "",
        asr_text=_text_for_range(asr_cues, start_sec=start_sec, end_sec=end_sec),
        ocr_text=_ocr_for_range(ocr_lines, start_sec=start_sec, end_sec=end_sec),
        frame_paths=frame_paths,
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
    chapters = []
    assigned = []
    for number, group in enumerate(groups, start=1):
        chapter_id = f"ch{number:02d}"
        beat_ids = tuple(beat.beat_id for beat in group)
        chapters.append(Chapter(chapter_id, group[0].start_sec, group[-1].end_sec, beat_ids, group[0].keyframe_path))
        assigned.extend(replace(beat, chapter_id=chapter_id) for beat in group)
    return tuple(chapters), tuple(assigned)


def _diagnostics(
    duration_sec: float,
    chapters: Sequence[Chapter],
    beats: Sequence[Beat],
    visual_index: VisualIndex,
    index_mode: str,
) -> IndexDiagnostics:
    durations = tuple(max(0.0, beat.end_sec - beat.start_sec) for beat in beats)
    norms = np.linalg.norm(visual_index.embeddings, axis=1) if visual_index.embeddings.size else np.asarray([], dtype=np.float32)
    warnings = []
    if visual_index.embeddings.shape[1] <= 1:
        warnings.append("visual_index_dim_lte_1")
    if norms.size and float(norms.mean()) <= 0.0:
        warnings.append("visual_embedding_norm_mean_lte_0")
    return IndexDiagnostics(
        duration_sec=float(duration_sec),
        chapter_count=len(chapters),
        beat_count=len(beats),
        median_beat_sec=float(median(durations)) if durations else 0.0,
        max_beat_sec=float(max(durations)) if durations else 0.0,
        visual_index_dim=int(visual_index.embeddings.shape[1]) if visual_index.embeddings.ndim == 2 else 0,
        visual_embedding_norm_mean=float(norms.mean()) if norms.size else 0.0,
        index_mode=str(index_mode or "fast"),
        warnings=tuple(warnings),
    )


def _normalize_ranges(
    raw_ranges: Sequence[tuple[float, float]],
    *,
    duration_sec: float,
    max_range_sec: float,
) -> tuple[tuple[float, float], ...]:
    duration = max(0.0, float(duration_sec))
    max_range = max(0.1, float(max_range_sec))
    clipped = sorted((max(0.0, min(duration, float(start))), max(0.0, min(duration, float(end)))) for start, end in raw_ranges)
    ranges: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in clipped:
        if end <= start:
            continue
        if start > cursor:
            ranges.extend(_split_range(cursor, start, max_range))
        clipped_start = max(start, cursor)
        if end > clipped_start:
            ranges.extend(_split_range(clipped_start, end, max_range))
            cursor = max(cursor, end)
    if cursor < duration:
        ranges.extend(_split_range(cursor, duration, max_range))
    return tuple(ranges)


def _split_range(start: float, end: float, max_range: float) -> tuple[tuple[float, float], ...]:
    ranges = []
    cursor = float(start)
    while cursor < end:
        next_end = min(float(end), cursor + max_range)
        ranges.append((round(cursor, 3), round(next_end, 3)))
        cursor = next_end
    return tuple(ranges)


def _text_for_range(cues: Sequence[Any], *, start_sec: float, end_sec: float) -> str:
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


def _ocr_for_range(lines: Sequence[Any], *, start_sec: float, end_sec: float) -> tuple[str, ...]:
    result = []
    for item in lines:
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
            result.append(text)
    return tuple(result)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _rank(scores: dict[str, float], *, modality: str) -> tuple[Hit, ...]:
    return tuple(Hit(beat_id, float(score), modality) for beat_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0])) if score > 0)


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in _TOKEN_RE.finditer(str(text or "")))


def _quoted_phrase(query: str) -> str | None:
    match = re.search(r'"([^"]+)"', str(query or ""))
    return match.group(1) if match else None


def _contains_phrase(tokens: Sequence[str], phrase_tokens: Sequence[str]) -> bool:
    if len(phrase_tokens) > len(tokens):
        return False
    width = len(phrase_tokens)
    return any(tuple(tokens[index : index + width]) == tuple(phrase_tokens) for index in range(0, len(tokens) - width + 1))


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return array / norms


def _clock(seconds: float) -> str:
    minutes, sec = divmod(max(0, int(round(float(seconds)))), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}" if hours else f"{minutes:02d}:{sec:02d}"
