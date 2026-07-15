from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
import zipfile


@dataclass(frozen=True)
class VideoMMEV2Question:
    video_id: str
    question_id: str
    question: str
    options: str
    answer: str
    metadata: Mapping[str, Any]


def options_mapping(options: Any) -> dict[str, str]:
    if isinstance(options, Mapping):
        return {
            str(label).strip().upper().rstrip(".):"): str(text).strip()
            for label, text in options.items()
            if str(label).strip() and str(text).strip()
        }
    text = _format_options(options)
    parsed: dict[str, str] = {}
    current_label = ""
    current_parts: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^\s*([A-H])\s*[.):]\s*(.*)$", line.strip(), flags=re.IGNORECASE)
        if match:
            if current_label:
                parsed[current_label] = " ".join(current_parts).strip()
            current_label = match.group(1).upper()
            current_parts = [match.group(2).strip()]
        elif current_label and line.strip():
            current_parts.append(line.strip())
    if current_label:
        parsed[current_label] = " ".join(current_parts).strip()
    return parsed


def load_case_group(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = tuple(dict(item) for item in payload.get("cases", ()) if isinstance(item, Mapping))
    case_ids = tuple(str(item.get("case_id", "") or "").strip() for item in cases)
    if not case_ids or any(not case_id for case_id in case_ids):
        raise ValueError(f"VideoMME v2 group {path} requires non-empty cases[].case_id values")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError(f"VideoMME v2 group {path} contains duplicate case ids")
    return {
        **payload,
        "group_id": str(payload.get("group_id", Path(path).stem) or Path(path).stem),
        "cases": cases,
        "case_ids": case_ids,
    }


def summarize_group_results(
    results: Sequence[Mapping[str, Any]],
    *,
    group: Mapping[str, Any],
) -> dict[str, Any]:
    by_case = {str(item.get("case_id", "") or ""): dict(item) for item in results}
    case_metadata = {
        str(item.get("case_id", "") or ""): dict(item)
        for item in tuple(group.get("cases", ()) or ())
    }
    ordered_ids = tuple(str(item) for item in group.get("case_ids", ()) or ())
    rows = [by_case[case_id] for case_id in ordered_ids if case_id in by_case]
    correct = sum(bool(item.get("correct")) for item in rows)
    by_video: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        case_id = str(row.get("case_id", "") or "")
        metadata = case_metadata.get(case_id, {})
        video_id = str(row.get("video_id", metadata.get("video_id", "")) or "")
        task_type = str(metadata.get("task_type", metadata.get("third_head", "unknown")) or "unknown")
        by_video[video_id].append(row)
        by_type[task_type].append(row)
    video_groups = {}
    for video_id, video_rows in sorted(by_video.items()):
        ordered = sorted(video_rows, key=lambda item: _sort_key(str(item.get("case_id", ""))))
        prefix = 0
        for item in ordered:
            if not bool(item.get("correct")):
                break
            prefix += 1
        video_groups[video_id] = {
            "case_count": len(ordered),
            "correct": sum(bool(item.get("correct")) for item in ordered),
            "all_correct": bool(ordered) and all(bool(item.get("correct")) for item in ordered),
            "correct_prefix_length": prefix,
        }
    return {
        "group_id": str(group.get("group_id", "") or ""),
        "case_count": len(rows),
        "correct": correct,
        "accuracy": correct / max(1, len(rows)),
        "all_correct_group_count": sum(bool(item["all_correct"]) for item in video_groups.values()),
        "group_count": len(video_groups),
        "by_video": video_groups,
        "by_task_type": {
            key: {
                "case_count": len(items),
                "correct": sum(bool(item.get("correct")) for item in items),
                "accuracy": sum(bool(item.get("correct")) for item in items) / max(1, len(items)),
            }
            for key, items in sorted(by_type.items())
        },
        "cases": rows,
    }


def load_questions(dataset_root: Path) -> tuple[VideoMMEV2Question, ...]:
    parquet_path = Path(dataset_root)
    if parquet_path.is_dir():
        parquet_path = parquet_path / "test.parquet"
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on runtime image
        raise RuntimeError("pandas/pyarrow is required to read VideoMME v2 test.parquet") from exc
    frame = pd.read_parquet(parquet_path)
    return tuple(_question_from_record(record) for record in frame.to_dict("records"))


def format_videomme_v2_question(question: VideoMMEV2Question) -> str:
    prompt = str(question.question or "").strip()
    options = _format_options(question.options)
    return "\n".join(part for part in (prompt, options) if part)


def questions_by_video(questions: Iterable[VideoMMEV2Question]) -> dict[str, tuple[VideoMMEV2Question, ...]]:
    grouped: dict[str, list[VideoMMEV2Question]] = defaultdict(list)
    for question in questions:
        grouped[str(question.video_id)].append(question)
    return {video_id: tuple(sorted(items, key=lambda item: _sort_key(item.question_id))) for video_id, items in grouped.items()}


def select_video_ids(
    questions: Sequence[VideoMMEV2Question],
    *,
    videos_dir: Path,
    count: int = 5,
    requested_video_ids: Sequence[str] = (),
    min_questions_per_video: int = 4,
) -> tuple[str, ...]:
    grouped = questions_by_video(questions)
    candidates = tuple(_normal_video_id(video_id) for video_id in requested_video_ids) or tuple(sorted(grouped, key=_sort_key))
    selected: list[str] = []
    for video_id in candidates:
        video_questions = grouped.get(video_id, ())
        if len(video_questions) < int(min_questions_per_video):
            continue
        if not (Path(videos_dir) / f"{video_id}.mp4").exists():
            continue
        selected.append(video_id)
        if len(selected) >= int(count):
            break
    return tuple(selected)


def load_subtitle_cues(
    subtitle_zip: Path,
    video_id: str,
    *,
    max_gap_sec: float = 0.8,
    max_words: int = 32,
    max_duration_sec: float = 12.0,
) -> tuple[dict[str, object], ...]:
    member = f"subtitle/{_normal_video_id(video_id)}.jsonl"
    cues: list[dict[str, object]] = []
    words: list[str] = []
    cue_start: float | None = None
    cue_end: float | None = None
    previous_end: float | None = None

    def flush() -> None:
        nonlocal cue_start, cue_end, words
        text = _join_words(words)
        if cue_start is not None and cue_end is not None and text:
            cues.append({"start": round(cue_start, 3), "end": round(cue_end, 3), "text": text})
        words = []
        cue_start = None
        cue_end = None

    with zipfile.ZipFile(subtitle_zip) as archive:
        if member not in archive.namelist():
            return ()
        raw_lines = archive.read(member).decode("utf-8", errors="replace").splitlines()

    for raw in raw_lines:
        if not raw.strip():
            continue
        row = json.loads(raw)
        word = str(row.get("text") or "").strip()
        if not word:
            continue
        start = float(row.get("start_time", row.get("start", 0.0)) or 0.0)
        end = float(row.get("end_time", row.get("end", start)) or start)
        if words and previous_end is not None and start - previous_end > float(max_gap_sec):
            flush()
        if not words:
            cue_start = start
        words.append(word)
        cue_end = end
        previous_end = end
        duration = 0.0 if cue_start is None else end - cue_start
        if _ends_sentence(word) or len(words) >= int(max_words) or duration >= float(max_duration_sec):
            flush()
    flush()
    return tuple(cues)


def cache_subtitle_cues(cues: Sequence[Mapping[str, object]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(cues), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def load_cached_subtitle_cues(path: Path) -> tuple[dict[str, object], ...] | None:
    path = Path(path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(dict(item) for item in payload)


def score_videomme_v2_answer(answer: str, gold: str) -> bool:
    selected = extract_option_letter(answer)
    expected = extract_option_letter(gold) or str(gold or "").strip().upper()[:1]
    return bool(selected and expected and selected == expected)


def extract_option_letter(answer: str) -> str:
    text = str(answer or "").strip()
    upper = text.upper()
    patterns = (
        r"^\s*([A-H])(?:\b|[.):])",
        r"\bANSWER\s*(?:IS|:)?\s*([A-H])\b",
        r"\bOPTION\s*([A-H])\b",
        r"\bCHOOSE\s*([A-H])\b",
    )
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)
    return ""


def _question_from_record(record: Mapping[str, Any]) -> VideoMMEV2Question:
    video_id = _normal_video_id(str(record.get("video_id") or ""))
    question_id = str(record.get("question_id") or "").strip()
    known = {"video_id", "question_id", "question", "options", "answer"}
    metadata = {key: _jsonable(value) for key, value in record.items() if key not in known}
    return VideoMMEV2Question(
        video_id=video_id,
        question_id=question_id,
        question=str(record.get("question") or "").strip(),
        options=_format_options(record.get("options") or ""),
        answer=str(record.get("answer") or "").strip().upper(),
        metadata=metadata,
    )


def _format_options(options: Any) -> str:
    if isinstance(options, str):
        return options.strip()
    if isinstance(options, Mapping):
        lines = []
        for label, text in sorted(options.items(), key=lambda item: _sort_key(str(item[0]))):
            label_text = str(label).strip().rstrip(".):")
            lines.append(f"{label_text}. {str(text).strip()}")
        return "\n".join(lines).strip()
    if isinstance(options, Sequence):
        labels = "ABCDEFGH"
        return "\n".join(f"{labels[idx]}. {str(text).strip()}" for idx, text in enumerate(options) if idx < len(labels)).strip()
    return str(options or "").strip()


def _join_words(words: Sequence[str]) -> str:
    text = ""
    no_space_before = set(".,!?;:%)]}")
    no_space_after = set("([{")
    for word in words:
        if not text:
            text = word
        elif word in no_space_before or word.startswith("'"):
            text += word
        elif text[-1] in no_space_after:
            text += word
        else:
            text += " " + word
    return re.sub(r"\s+", " ", text).strip()


def _ends_sentence(word: str) -> bool:
    return bool(str(word).strip().endswith((".", "?", "!")))


def _normal_video_id(video_id: str) -> str:
    value = str(video_id or "").strip()
    if value.isdigit():
        return f"{int(value):03d}"
    return value


def _sort_key(value: str) -> tuple[int, object]:
    text = str(value or "")
    match = re.match(r"^(\d+)(?:-(\d+))?$", text)
    if match:
        first = int(match.group(1))
        second = int(match.group(2) or 0)
        return (0, (first, second))
    return (1, text)


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return None if value is None else str(value)
