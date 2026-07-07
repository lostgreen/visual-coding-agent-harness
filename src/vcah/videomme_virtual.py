from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from vcah.video import probe_duration
from vcah.virtual_video import (
    VirtualVideoCase,
    VirtualVideoManifest,
    VirtualVideoSegment,
    VirtualVideoWorkspace,
    load_srt_as_virtual_cues,
)


DEFAULT_SMOKE_CASE_IDS = ("477-2", "548-1", "371-1")


def build_videomme_smoke_workspaces(
    dataset_root: Path,
    out_dir: Path,
    *,
    seed: int = 20260707,
    case_ids: Sequence[str] = DEFAULT_SMOKE_CASE_IDS,
    distractor_count: int = 4,
) -> tuple[VirtualVideoWorkspace, ...]:
    dataset_root = Path(dataset_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_videomme_rows(dataset_root)
    by_qid = {str(row.get("question_id")): row for row in rows}
    long_rows = [row for row in rows if str(row.get("duration", "")).casefold() == "long"]
    rng = random.Random(int(seed))
    workspaces: list[VirtualVideoWorkspace] = []
    for case_id in case_ids:
        target = by_qid[str(case_id)]
        distractors = _sample_distractors(long_rows, target, count=distractor_count, rng=rng)
        segment_specs = [_target_spec(target, dataset_root), *(_distractor_spec(row, dataset_root, rng=rng) for row in distractors)]
        rng.shuffle(segment_specs)
        segment_specs = _avoid_target_edges(segment_specs)
        segments: list[VirtualVideoSegment] = []
        cursor = 0.0
        for index, spec in enumerate(segment_specs, start=1):
            duration = float(spec["source_end_sec"]) - float(spec["source_start_sec"])
            segment = VirtualVideoSegment(
                segment_id=f"seg_{index:04d}",
                source_video_id=str(spec["video_id"]),
                source_path=str(dataset_root / "video" / f"{spec['video_id']}.mp4"),
                source_start_sec=round(float(spec["source_start_sec"]), 3),
                source_end_sec=round(float(spec["source_end_sec"]), 3),
                virtual_start_sec=round(cursor, 3),
                virtual_end_sec=round(cursor + duration, 3),
                role=str(spec["role"]),
            )
            segments.append(segment)
            cursor += duration

        target_segment = next(segment for segment in segments if segment.role == "target")
        manifest = VirtualVideoManifest(workspace_id=str(case_id), segments=tuple(segments))
        case = VirtualVideoCase(
            case_id=str(case_id),
            question=str(target.get("question", "")),
            options=_options_mapping(target.get("options")),
            gold=str(target.get("answer", "")),
            target_segment_id=target_segment.segment_id,
            target_virtual_interval=(target_segment.virtual_start_sec, target_segment.virtual_end_sec),
            metadata={
                "source_video_id": _video_id(target),
                "seed": int(seed),
                "distractor_count": int(distractor_count),
            },
        )
        workspace = VirtualVideoWorkspace.create(out_dir / str(case_id), manifest=manifest, case=case)
        cues = []
        for segment in segments:
            cues.extend(load_srt_as_virtual_cues(dataset_root / "subtitle" / f"{segment.source_video_id}.srt", segment))
        workspace.write_asr_virtual_cues(tuple(cues))
        workspaces.append(workspace)
    return tuple(workspaces)


def _load_videomme_rows(dataset_root: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas/pyarrow is required to read VideoMME parquet metadata") from exc
    parquet = dataset_root / "videomme" / "test-00000-of-00001.parquet"
    if not parquet.exists():
        parquet = dataset_root / "videomme" / "test.parquet"
    frame = pd.read_parquet(parquet)
    return [dict(row) for row in frame.to_dict("records")]


def _sample_distractors(
    long_rows: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
    *,
    count: int,
    rng: random.Random,
) -> tuple[Mapping[str, Any], ...]:
    target_video = _video_id(target)
    pool_by_video: dict[str, Mapping[str, Any]] = {}
    for row in long_rows:
        video_id = _video_id(row)
        if video_id and video_id != target_video and video_id not in pool_by_video:
            pool_by_video[video_id] = row
    pool = list(pool_by_video.values())
    if len(pool) < int(count):
        raise ValueError(f"Need at least {count} long distractor videos; found {len(pool)}")
    rng.shuffle(pool)
    return tuple(pool[: int(count)])


def _target_spec(row: Mapping[str, Any], dataset_root: Path) -> dict[str, Any]:
    duration = _duration_sec(row, dataset_root)
    return {
        "role": "target",
        "video_id": _video_id(row),
        "source_start_sec": 0.0,
        "source_end_sec": duration,
    }


def _distractor_spec(row: Mapping[str, Any], dataset_root: Path, *, rng: random.Random) -> dict[str, Any]:
    duration = _duration_sec(row, dataset_root)
    length = min(duration, rng.uniform(120.0, 360.0))
    start = 0.0 if duration <= length else rng.uniform(0.0, duration - length)
    return {
        "role": "distractor",
        "video_id": _video_id(row),
        "source_start_sec": start,
        "source_end_sec": start + length,
    }


def _avoid_target_edges(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(specs) <= 2:
        return specs
    target_idx = next((idx for idx, spec in enumerate(specs) if spec["role"] == "target"), 0)
    if target_idx == 0:
        specs[0], specs[1] = specs[1], specs[0]
    elif target_idx == len(specs) - 1:
        specs[-1], specs[-2] = specs[-2], specs[-1]
    return specs


def _video_id(row: Mapping[str, Any]) -> str:
    return str(row.get("videoID") or row.get("video_id") or "").strip()


def _duration_sec(row: Mapping[str, Any], dataset_root: Path) -> float:
    for key in ("duration_sec", "video_duration_sec", "seconds"):
        value = row.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    video_id = _video_id(row)
    path = Path(str(row.get("video_path") or dataset_root / "video" / f"{video_id}.mp4"))
    return probe_duration(str(path))


def _options_mapping(value: Any) -> dict[str, str]:
    labels = "ABCDEFGH"
    if isinstance(value, str):
        parts = [part.strip() for part in value.split("|") if part.strip()]
    else:
        try:
            parts = [str(part).strip() for part in list(value) if str(part).strip()]
        except TypeError:
            parts = []
    options: dict[str, str] = {}
    for idx, part in enumerate(parts):
        label = labels[idx]
        text = part
        if len(part) >= 2 and part[0].upper() in labels and part[1] == ".":
            label = part[0].upper()
            text = part[2:].strip()
        options[label] = text
    return options
