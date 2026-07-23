from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
from typing import Any, Mapping, Sequence


CAPTION_SCHEMA_VERSION = 1
TIMESTAMP_RE = re.compile(
    r"\[(?:(?P<hours>\d{1,2}):)?(?P<minutes>\d{1,2}):(?P<seconds>\d{2}(?:\.\d{1,3})?)\]"
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")


@dataclass(frozen=True)
class CaptionAnchorV1:
    label: str
    local_sec: float
    virtual_sec: float
    sentence: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "local_sec", round(float(self.local_sec), 3))
        object.__setattr__(self, "virtual_sec", round(float(self.virtual_sec), 3))
        if self.sentence is not None:
            object.__setattr__(self, "sentence", _normalize_text(self.sentence) or None)


@dataclass(frozen=True)
class CaptionChunkV1:
    caption_id: str
    subset: str
    virtual_start_sec: float
    virtual_end_sec: float
    source_segments: tuple[str, ...]
    wall_clock_begin: str | None
    wall_clock_end: str | None
    text_raw: str
    text_normalized: str
    timestamp_anchors: tuple[CaptionAnchorV1, ...]
    model: str
    provider: str
    prompt_digest: str
    generation_config_digest: str
    source_manifest_digest: str
    created_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "virtual_start_sec", round(float(self.virtual_start_sec), 3))
        object.__setattr__(self, "virtual_end_sec", round(float(self.virtual_end_sec), 3))
        object.__setattr__(self, "source_segments", tuple(str(value) for value in self.source_segments))
        object.__setattr__(
            self,
            "timestamp_anchors",
            tuple(_anchor(value) for value in self.timestamp_anchors),
        )
        object.__setattr__(self, "text_raw", str(self.text_raw))
        object.__setattr__(self, "text_normalized", _normalize_text(self.text_normalized))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.virtual_end_sec <= self.virtual_start_sec:
            raise ValueError("Caption chunk end must be greater than start")


@dataclass(frozen=True)
class CaptionPassageV1:
    passage_id: str
    caption_id: str
    text: str
    virtual_start_sec: float
    virtual_end_sec: float
    anchor_virtual_sec: float | None
    ordinal: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _normalize_text(self.text))
        object.__setattr__(self, "virtual_start_sec", round(float(self.virtual_start_sec), 3))
        object.__setattr__(self, "virtual_end_sec", round(float(self.virtual_end_sec), 3))
        if self.anchor_virtual_sec is not None:
            object.__setattr__(self, "anchor_virtual_sec", round(float(self.anchor_virtual_sec), 3))
        object.__setattr__(self, "ordinal", int(self.ordinal))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.virtual_end_sec < self.virtual_start_sec:
            raise ValueError("Caption passage end cannot be before start")


@dataclass(frozen=True)
class CaptionHitV1:
    passage_id: str
    caption_id: str
    rank: int
    lexical_score: float | None
    dense_score: float | None
    fused_score: float
    virtual_start_sec: float
    virtual_end_sec: float
    wall_clock_begin: str | None
    wall_clock_end: str | None
    text: str
    interval_precision: str
    source_pointer: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rank", max(1, int(self.rank)))
        if self.lexical_score is not None:
            object.__setattr__(self, "lexical_score", float(self.lexical_score))
        if self.dense_score is not None:
            object.__setattr__(self, "dense_score", float(self.dense_score))
        object.__setattr__(self, "fused_score", float(self.fused_score))
        object.__setattr__(self, "virtual_start_sec", round(float(self.virtual_start_sec), 3))
        object.__setattr__(self, "virtual_end_sec", round(float(self.virtual_end_sec), 3))
        object.__setattr__(self, "text", _normalize_text(self.text))
        object.__setattr__(self, "metadata", dict(self.metadata))


def parse_timestamp_anchors(
    text: str,
    *,
    chunk_start_sec: float,
    chunk_end_sec: float,
    strict: bool = True,
    repair_duplicate_minute_hour: bool = False,
    repair_short_timestamp: bool = False,
) -> tuple[CaptionAnchorV1, ...]:
    raw = str(text)
    chunk_start = float(chunk_start_sec)
    chunk_end = float(chunk_end_sec)
    duration = max(0.0, chunk_end - chunk_start)
    matches = tuple(TIMESTAMP_RE.finditer(raw))
    anchors: list[CaptionAnchorV1] = []
    for index, match in enumerate(matches):
        local_sec = _timestamp_seconds(match)
        repaired = match.group("hours") is None and repair_short_timestamp
        if match.group("hours") is None and not repair_short_timestamp:
            if strict:
                raise ValueError(f"Caption anchor {match.group(0)} must use HH:MM:SS format")
            continue
        if repair_duplicate_minute_hour:
            local_sec, repaired_duplicate = _repair_duplicate_minute_hour(match, local_sec, duration)
            repaired = repaired or repaired_duplicate
            local_sec, repaired_shifted = _repair_shifted_minute_second(match, local_sec, duration)
            repaired = repaired or repaired_shifted
        if local_sec < 0.0 or local_sec > duration + 0.001:
            if strict:
                raise ValueError(
                    f"Caption anchor {match.group(0)} is outside chunk duration {duration:.3f}s"
                )
            continue
        sentence_end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        sentence = _normalize_text(raw[match.end() : sentence_end]).lstrip("-: ") or None
        anchors.append(
            CaptionAnchorV1(
                label=_timestamp_label(local_sec) if repaired else match.group(0),
                local_sec=local_sec,
                virtual_sec=min(chunk_end, chunk_start + local_sec),
                sentence=sentence,
            )
        )
    return tuple(anchors)


def split_caption_passages(chunk: CaptionChunkV1) -> tuple[CaptionPassageV1, ...]:
    passages: list[CaptionPassageV1] = []
    anchors = chunk.timestamp_anchors
    if anchors:
        for index, anchor in enumerate(anchors):
            next_sec = anchors[index + 1].virtual_sec if index + 1 < len(anchors) else chunk.virtual_end_sec
            start, end = _passage_interval(
                anchor.virtual_sec,
                next_sec,
                chunk_start=chunk.virtual_start_sec,
                chunk_end=chunk.virtual_end_sec,
            )
            text = anchor.sentence or anchor.label
            passages.append(
                CaptionPassageV1(
                    passage_id=f"{chunk.caption_id}:p{index:04d}",
                    caption_id=chunk.caption_id,
                    text=text,
                    virtual_start_sec=start,
                    virtual_end_sec=end,
                    anchor_virtual_sec=anchor.virtual_sec,
                    ordinal=index,
                    metadata={
                        "interval_precision": "anchor",
                        "source_segments": list(chunk.source_segments),
                        "wall_clock_begin": chunk.wall_clock_begin,
                        "wall_clock_end": chunk.wall_clock_end,
                    },
                )
            )
        return tuple(passages)

    sentences = tuple(
        sentence
        for sentence in (_normalize_text(value) for value in SENTENCE_SPLIT_RE.split(chunk.text_normalized))
        if sentence
    )
    for index, sentence in enumerate(sentences or (chunk.text_normalized,)):
        if not sentence:
            continue
        passages.append(
            CaptionPassageV1(
                passage_id=f"{chunk.caption_id}:p{index:04d}",
                caption_id=chunk.caption_id,
                text=sentence,
                virtual_start_sec=chunk.virtual_start_sec,
                virtual_end_sec=chunk.virtual_end_sec,
                anchor_virtual_sec=None,
                ordinal=index,
                metadata={
                    "interval_precision": "chunk",
                    "source_segments": list(chunk.source_segments),
                    "wall_clock_begin": chunk.wall_clock_begin,
                    "wall_clock_end": chunk.wall_clock_end,
                },
            )
        )
    return tuple(passages)


def chunk_to_dict(chunk: CaptionChunkV1) -> dict[str, Any]:
    payload = asdict(chunk)
    payload["schema_version"] = CAPTION_SCHEMA_VERSION
    return payload


def chunk_from_dict(payload: Mapping[str, Any]) -> CaptionChunkV1:
    values = dict(payload)
    values.pop("schema_version", None)
    return CaptionChunkV1(**values)


def passage_to_dict(passage: CaptionPassageV1) -> dict[str, Any]:
    payload = asdict(passage)
    payload["schema_version"] = CAPTION_SCHEMA_VERSION
    return payload


def passage_from_dict(payload: Mapping[str, Any]) -> CaptionPassageV1:
    values = dict(payload)
    values.pop("schema_version", None)
    return CaptionPassageV1(**values)


def stable_digest(payload: Any) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def text_digest(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def normalize_caption_text(value: str) -> str:
    return _normalize_text(value)


def count_timestamp_tokens(value: str) -> int:
    return len(tuple(TIMESTAMP_RE.finditer(str(value))))


def count_repaired_timestamp_tokens(value: str, *, chunk_duration_sec: float) -> int:
    duration = max(0.0, float(chunk_duration_sec))
    repairs = 0
    for match in TIMESTAMP_RE.finditer(str(value)):
        if match.group("hours") is None:
            repairs += 1
            continue
        local_sec, repaired_duplicate = _repair_duplicate_minute_hour(
            match,
            _timestamp_seconds(match),
            duration,
        )
        _, repaired_shifted = _repair_shifted_minute_second(match, local_sec, duration)
        repairs += int(repaired_duplicate or repaired_shifted)
    return repairs


def _timestamp_seconds(match: re.Match[str]) -> float:
    return round(
        int(match.group("hours") or 0) * 3600.0
        + int(match.group("minutes")) * 60.0
        + float(match.group("seconds")),
        3,
    )


def _repair_duplicate_minute_hour(
    match: re.Match[str],
    local_sec: float,
    duration_sec: float,
) -> tuple[float, bool]:
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes"))
    seconds = float(match.group("seconds"))
    candidate = round(minutes * 60.0 + seconds, 3)
    if (
        local_sec > float(duration_sec) + 0.001
        and hours > 0
        and hours == minutes
        and candidate <= float(duration_sec) + 0.001
    ):
        return candidate, True
    return local_sec, False


def _repair_shifted_minute_second(
    match: re.Match[str],
    local_sec: float,
    duration_sec: float,
) -> tuple[float, bool]:
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes"))
    candidate = round(hours * 60.0 + minutes, 3)
    if (
        local_sec > float(duration_sec) + 0.001
        and candidate <= float(duration_sec) + 0.001
    ):
        return candidate, True
    return local_sec, False


def _timestamp_label(seconds: float) -> str:
    whole_seconds = int(float(seconds))
    remainder = round(float(seconds) - whole_seconds, 3)
    hours, remainder_seconds = divmod(whole_seconds, 3600)
    minutes, secs = divmod(remainder_seconds, 60)
    if remainder:
        return f"[{hours:02d}:{minutes:02d}:{secs + remainder:06.3f}]"
    return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"


def _passage_interval(
    start_sec: float,
    end_sec: float,
    *,
    chunk_start: float,
    chunk_end: float,
) -> tuple[float, float]:
    start = max(chunk_start, min(chunk_end, float(start_sec)))
    end = max(start, min(chunk_end, float(end_sec)))
    if end <= start and chunk_end > chunk_start:
        start = max(chunk_start, min(start, chunk_end - 0.001))
        end = max(start, min(chunk_end, start + 0.001))
    return round(start, 3), round(end, 3)


def _normalize_text(value: str) -> str:
    return " ".join(str(value).split())


def _anchor(value: CaptionAnchorV1 | Mapping[str, Any]) -> CaptionAnchorV1:
    if isinstance(value, CaptionAnchorV1):
        return value
    return CaptionAnchorV1(**dict(value))
