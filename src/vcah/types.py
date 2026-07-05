from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any, Iterable, Literal, Mapping, Sequence


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
    asr_cues: tuple[Mapping[str, Any], ...] = ()
    ocr_cues: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if float(self.end_sec) < float(self.start_sec):
            raise ValueError("Beat end_sec must be greater than or equal to start_sec")
        object.__setattr__(self, "start_sec", float(self.start_sec))
        object.__setattr__(self, "end_sec", float(self.end_sec))
        object.__setattr__(self, "ocr_text", tuple(str(item) for item in self.ocr_text if str(item).strip()))
        object.__setattr__(self, "frame_paths", tuple(str(item) for item in self.frame_paths if str(item).strip()))
        object.__setattr__(self, "asr_cues", tuple(_cue_mapping(item) for item in self.asr_cues))
        object.__setattr__(self, "ocr_cues", tuple(_cue_mapping(item) for item in self.ocr_cues))


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
class Window:
    start_sec: float
    end_sec: float

    def __post_init__(self) -> None:
        start = float(self.start_sec)
        end = float(self.end_sec)
        if end < start:
            raise ValueError("Window end_sec must be greater than or equal to start_sec")
        object.__setattr__(self, "start_sec", start)
        object.__setattr__(self, "end_sec", end)


@dataclass(frozen=True)
class WindowCoverage:
    requested: Window
    actuals: tuple[Window, ...]
    coverage: float
    passed: bool


def window_overlap_ratio(requested: Window, actuals: Iterable[Window]) -> float:
    requested_len = requested.end_sec - requested.start_sec
    if requested_len <= 0:
        return 0.0
    intervals: list[tuple[float, float]] = []
    for actual in actuals:
        overlap_start = max(requested.start_sec, actual.start_sec)
        overlap_end = min(requested.end_sec, actual.end_sec)
        if overlap_end > overlap_start:
            intervals.append((overlap_start, overlap_end))
    intervals.sort()
    merged: list[list[float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    covered = sum(end - start for start, end in merged)
    return min(1.0, covered / requested_len)


@dataclass(frozen=True)
class ToolAction:
    type: str
    query: str = ""
    beat_id: str = ""
    beat_ids: tuple[str, ...] = ()
    windows: tuple[Window, ...] = ()
    modalities: tuple[Literal["asr", "ocr", "frames"], ...] = ()
    selected: str = ""
    evidence_table: Mapping[str, Any] = field(default_factory=dict)
    investigator_payload: Mapping[str, Any] = field(default_factory=dict)
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
        windows = _parse_windows(payload)
        modalities = payload.get("modalities") or ()
        if isinstance(modalities, str):
            modalities = (modalities,)
        return cls(
            type=str(payload.get("type") or payload.get("tool") or ""),
            query=str(payload.get("query") or ""),
            beat_id=str(payload.get("beat_id") or ""),
            beat_ids=tuple(str(item) for item in beat_ids),
            windows=windows,
            modalities=tuple(_normalize_modality(item) for item in modalities if _normalize_modality(item)),
            selected=str(payload.get("selected") or payload.get("option") or ""),
            evidence_table=_mapping(payload.get("evidence_table")),
            investigator_payload=_mapping(payload.get("investigator_payload")),
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


class InvestigatorOutputEmpty(ValueError):
    pass


class InvestigatorOutputInvalid(ValueError):
    pass


HYPOTHESIS_KEYS = {
    "hypothesis",
    "likely_answer",
    "predicted_option",
    "reasoner_guess",
    "candidate_answer",
    "initial_answer",
    "answer_hypothesis",
}
HYPOTHESIS_PATTERNS = (
    re.compile(r"\blikely answer\b", re.IGNORECASE),
    re.compile(r"\banswer hypothesis\b", re.IGNORECASE),
    re.compile(r"\bhypothesis\s*:", re.IGNORECASE),
    re.compile(r"\bverify option\s+[A-H]\b", re.IGNORECASE),
)


def investigator_input_has_hypothesis(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() in HYPOTHESIS_KEYS:
                return True
            if investigator_input_has_hypothesis(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(investigator_input_has_hypothesis(item) for item in value)
    if isinstance(value, str):
        return any(pattern.search(value) for pattern in HYPOTHESIS_PATTERNS)
    return False


def validate_investigator_input(payload: Mapping[str, Any]) -> None:
    if investigator_input_has_hypothesis(payload):
        raise InvestigatorOutputInvalid("Investigator input contains answer hypothesis")


def validate_investigator_output(output: Mapping[str, Any] | None, *, options: Sequence[str] = ("A", "B", "C", "D")) -> None:
    if not output:
        raise InvestigatorOutputEmpty("Investigator output is empty")
    valid_statuses = {"supported", "contradicted", "unknown"}
    for option in options:
        item = output.get(option)
        if not isinstance(item, Mapping):
            raise InvestigatorOutputInvalid(f"Missing evidence table entry for option {option}")
        if item.get("status") not in valid_statuses:
            raise InvestigatorOutputInvalid(f"Invalid evidence status for option {option}")
        for bucket in ("support", "contradict"):
            entries = item.get(bucket)
            if not isinstance(entries, list):
                raise InvestigatorOutputInvalid(f"Option {option} {bucket} must be a list")


def verify_final_answer(question: str, evidence_table: Mapping[str, Any], selected: str) -> dict[str, Any]:
    selected = str(selected).strip().upper()
    if not selected:
        return {"passed": False, "reason": "missing_selected_option"}
    entry = evidence_table.get(selected)
    if not isinstance(entry, Mapping):
        return {"passed": False, "reason": "missing_selected_option_evidence"}
    polarity = "negative" if _is_negative_question(question) else "positive"
    expected = "contradicted" if polarity == "negative" else "supported"
    if entry.get("status") != expected:
        return {"passed": False, "reason": f"selected_option_not_{expected}", "polarity": polarity}
    return {"passed": True, "reason": None, "polarity": polarity}


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


def _parse_windows(payload: Mapping[str, Any]) -> tuple[Window, ...]:
    raw_windows = payload.get("windows") or payload.get("inspect_windows") or ()
    windows: list[Window] = []
    if payload.get("start_sec") is not None or payload.get("end_sec") is not None:
        windows.append(Window(_parse_seconds(payload.get("start_sec")), _parse_seconds(payload.get("end_sec"))))
    if isinstance(raw_windows, Mapping):
        raw_windows = (raw_windows,)
    for item in raw_windows:
        if not isinstance(item, Mapping):
            continue
        start = item.get("start_sec", item.get("start"))
        end = item.get("end_sec", item.get("end"))
        if start is None or end is None:
            continue
        windows.append(Window(_parse_seconds(start), _parse_seconds(end)))
    return tuple(windows)


def _parse_seconds(value: Any) -> float:
    if value is None:
        raise ValueError("Window start_sec/end_sec is required")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if ":" not in text:
        return float(text)
    parts = [float(part) for part in text.split(":")]
    if len(parts) == 3:
        return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60.0 + parts[1]
    raise ValueError(f"Invalid time value: {value!r}")


def _normalize_modality(value: object) -> Literal["asr", "ocr", "frames"] | None:
    text = str(value).casefold()
    if text in {"asr", "ocr", "frames"}:
        return text  # type: ignore[return-value]
    if text == "frame":
        return "frames"
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _cue_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return {
            "start_sec": float(value.get("start_sec", value.get("start", value.get("time_sec", value.get("time", 0.0)))) or 0.0),
            "end_sec": float(value.get("end_sec", value.get("end", value.get("time_sec", value.get("time", 0.0)))) or 0.0),
            "text": str(value.get("text", "") or ""),
        }
    return _mapping(value)


def _is_negative_question(question: str) -> bool:
    return bool(re.search(r"\b(not correct|incorrect|false|not true|except)\b", question, re.IGNORECASE))
