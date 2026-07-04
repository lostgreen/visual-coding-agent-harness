from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping


@dataclass(frozen=True)
class Frame:
    frame_id: str
    time_sec: float
    path: str
    ocr_text: str = ""


@dataclass(frozen=True)
class Beat:
    beat_id: str
    chapter_id: str
    start_sec: float
    end_sec: float
    keyframe_path: str
    asr_text: str = ""
    ocr_text: tuple[str, ...] = ()
    frame_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if float(self.end_sec) < float(self.start_sec):
            raise ValueError("Beat end_sec must be greater than or equal to start_sec")
        object.__setattr__(self, "start_sec", float(self.start_sec))
        object.__setattr__(self, "end_sec", float(self.end_sec))
        object.__setattr__(self, "ocr_text", tuple(str(item) for item in self.ocr_text if str(item).strip()))
        object.__setattr__(self, "frame_paths", tuple(str(item) for item in self.frame_paths if str(item).strip()))


@dataclass(frozen=True)
class Chapter:
    chapter_id: str
    start_sec: float
    end_sec: float
    beat_ids: tuple[str, ...]
    thumb_path: str = ""

    def __post_init__(self) -> None:
        if float(self.end_sec) < float(self.start_sec):
            raise ValueError("Chapter end_sec must be greater than or equal to start_sec")
        object.__setattr__(self, "start_sec", float(self.start_sec))
        object.__setattr__(self, "end_sec", float(self.end_sec))
        object.__setattr__(self, "beat_ids", tuple(str(item) for item in self.beat_ids if str(item).strip()))


@dataclass(frozen=True)
class Hit:
    beat_id: str
    score: float
    modality: Literal["text", "visual"]


@dataclass(frozen=True)
class IndexDiagnostics:
    duration_sec: float
    chapter_count: int
    beat_count: int
    median_beat_sec: float
    max_beat_sec: float
    visual_index_dim: int
    visual_embedding_norm_mean: float
    embedding_backend: str
    index_mode: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    beat_id: str
    start_sec: float
    end_sec: float
    modality: Literal["asr", "ocr", "frame"]
    pointer: str
    verbatim: str
    claim: str = ""


@dataclass(frozen=True)
class ToolAction:
    type: str
    query: str = ""
    beat_id: str = ""
    beat_ids: tuple[str, ...] = ()
    answer: str = ""
    citations: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ToolAction":
        citations = payload.get("citations") or ()
        if isinstance(citations, str):
            citations = (citations,)
        beat_ids = payload.get("beat_ids") or ()
        if isinstance(beat_ids, str):
            beat_ids = (beat_ids,)
        return cls(
            type=str(payload.get("type") or payload.get("tool") or ""),
            query=str(payload.get("query") or ""),
            beat_id=str(payload.get("beat_id") or ""),
            beat_ids=tuple(str(item) for item in beat_ids),
            answer=str(payload.get("answer") or ""),
            citations=tuple(str(item) for item in citations),
        )


@dataclass(frozen=True)
class ToolResult:
    tool: str
    beat_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    text: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Answer:
    answer: str
    citations: tuple[str, ...]
    run_dir: Path | None = None


def to_jsonable(value: object) -> object:
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value
