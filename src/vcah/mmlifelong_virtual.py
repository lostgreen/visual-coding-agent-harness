from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

from vcah.video import probe_duration
from vcah.virtual_video import (
    VirtualVideoCase,
    VirtualVideoManifest,
    VirtualVideoSegment,
    VirtualVideoWorkspace,
    virtual_to_source_windows,
)


DurationProbe = Callable[[str], float]
_SUBSET_ALIASES = {"day": "game", "game": "game"}
_SOURCE_SUBSETS = {"game": "day"}
_NATURAL_PART_RE = re.compile(r"(\d+)")
_MIN_CLUE_DURATION_SEC = 0.001


@dataclass(frozen=True)
class MMLifelongBuildResult:
    subset: str
    split: str
    asset_root: Path
    case_root: Path
    workspaces: tuple[VirtualVideoWorkspace, ...]
    validation_path: Path

    def summary(self) -> dict[str, Any]:
        validation = json.loads(self.validation_path.read_text(encoding="utf-8"))
        return {
            "subset": self.subset,
            "split": self.split,
            "asset_root": str(self.asset_root),
            "case_root": str(self.case_root),
            "segment_count": validation["segment_count"],
            "case_count": validation["case_count"],
            "duration_sec": validation["duration_sec"],
            "clue_interval_count": validation["clue_interval_count"],
            "clue_repair_count": validation["clue_repair_count"],
            "validation_status": validation["status"],
            "validation_path": str(self.validation_path),
        }


def build_mmlifelong_workspaces(
    dataset_root: Path,
    asset_root: Path,
    case_root: Path,
    *,
    subset: str = "game",
    split: str = "test",
    verify_durations: bool = True,
    verify_clues: bool = True,
    overwrite: bool = False,
    duration_probe: DurationProbe = probe_duration,
) -> MMLifelongBuildResult:
    dataset_root = Path(dataset_root)
    asset_root = Path(asset_root)
    case_root = Path(case_root)
    internal_subset = _normalize_subset(subset)
    source_subset = _SOURCE_SUBSETS[internal_subset]
    metadata_path = dataset_root / source_subset / f"{split}.json"
    video_root = dataset_root / "videos" / source_subset

    if not metadata_path.is_file():
        raise FileNotFoundError(f"MM-Lifelong metadata not found: {metadata_path}")
    if not video_root.is_dir():
        raise FileNotFoundError(f"MM-Lifelong video directory not found: {video_root}")
    _check_output_paths(asset_root, case_root, overwrite=overwrite)

    video_paths = tuple(
        sorted(
            (path for path in video_root.rglob("*") if path.is_file() and path.suffix.casefold() == ".mp4"),
            key=lambda path: _natural_key(path.relative_to(video_root)),
        )
    )
    if not video_paths:
        raise ValueError(f"No MP4 files found under {video_root}")

    segments = _build_segments(
        video_paths,
        video_root=video_root,
        source_subset=source_subset,
        duration_probe=duration_probe,
    )
    manifest = VirtualVideoManifest(
        workspace_id=f"mmlifelong-{internal_subset}",
        segments=segments,
    )
    rows = _load_rows(metadata_path)
    cases, repairs, clue_count, mapped_clue_count = _build_cases(
        rows,
        manifest=manifest,
        subset=internal_subset,
        source_subset=source_subset,
        split=split,
        verify_clues=verify_clues,
    )
    validation = _validation_payload(
        manifest,
        cases=cases,
        clue_count=clue_count,
        mapped_clue_count=mapped_clue_count,
        repairs=repairs,
        verify_durations=verify_durations,
    )

    asset_root.mkdir(parents=True, exist_ok=True)
    case_root.mkdir(parents=True, exist_ok=True)
    workspaces = tuple(
        VirtualVideoWorkspace.create(
            case_root / case.case_id,
            manifest=manifest,
            case=case,
            asset_root=asset_root,
        )
        for case in cases
    )
    timeline_payload = json.loads((asset_root / "virtual_timeline.json").read_text(encoding="utf-8"))
    _write_json(asset_root / "timeline.json", timeline_payload)
    _write_json(
        asset_root / "manifest.json",
        {
            "schema_version": 1,
            "dataset": "MM-Lifelong",
            "subset": internal_subset,
            "source_subset": source_subset,
            "source_root": str(dataset_root),
            "segment_count": len(segments),
            "duration_sec": manifest.duration_sec,
            "timeline_file": "timeline.json",
            "validation_file": "validation.json",
        },
    )
    _write_json(
        asset_root / "asset_meta.json",
        {
            "schema_version": 1,
            "dataset": "MM-Lifelong",
            "subset": internal_subset,
            "source_subset": source_subset,
            "source_video_root": str(video_root),
            "metadata_path": str(metadata_path),
            "split": split,
            "shared_asset": True,
            "merged_video_created": False,
        },
    )
    validation_path = asset_root / "validation.json"
    _write_json(validation_path, validation)
    return MMLifelongBuildResult(
        subset=internal_subset,
        split=split,
        asset_root=asset_root,
        case_root=case_root,
        workspaces=workspaces,
        validation_path=validation_path,
    )


def _normalize_subset(value: str) -> str:
    key = str(value).strip().casefold()
    try:
        return _SUBSET_ALIASES[key]
    except KeyError as exc:
        raise ValueError("This integration supports the MM-Lifelong Day subset (game) only") from exc


def _check_output_paths(asset_root: Path, case_root: Path, *, overwrite: bool) -> None:
    existing = [
        path
        for path in (asset_root / "virtual_timeline.json", asset_root / "manifest.json")
        if path.exists()
    ]
    if case_root.exists() and any(case_root.iterdir()):
        existing.append(case_root)
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"MM-Lifelong output already exists: {joined}; pass overwrite=True to replace named files"
        )


def _build_segments(
    video_paths: Sequence[Path],
    *,
    video_root: Path,
    source_subset: str,
    duration_probe: DurationProbe,
) -> tuple[VirtualVideoSegment, ...]:
    segments: list[VirtualVideoSegment] = []
    cursor = 0.0
    for index, path in enumerate(video_paths, start=1):
        try:
            duration = round(float(duration_probe(str(path))), 3)
        except Exception as exc:
            relative_path = path.relative_to(video_root)
            raise RuntimeError(f"Failed to probe MM-Lifelong video duration: {relative_path}") from exc
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError(f"Invalid video duration for {path}: {duration}")
        virtual_start = round(cursor, 3)
        virtual_end = round(virtual_start + duration, 3)
        relative_path = path.relative_to(video_root).as_posix()
        segments.append(
            VirtualVideoSegment(
                segment_id=f"seg_{index:04d}",
                source_video_id=path.stem,
                source_path=str(path.resolve()),
                source_start_sec=0.0,
                source_end_sec=duration,
                virtual_start_sec=virtual_start,
                virtual_end_sec=virtual_end,
                day_index=1,
                metadata={
                    "source_subset": source_subset,
                    "relative_source_path": relative_path,
                },
            )
        )
        cursor = virtual_end
    return tuple(segments)


def _load_rows(metadata_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"Expected a list of MM-Lifelong cases in {metadata_path}")
    return [dict(row) for row in payload]


def _build_cases(
    rows: Sequence[Mapping[str, Any]],
    *,
    manifest: VirtualVideoManifest,
    subset: str,
    source_subset: str,
    split: str,
    verify_clues: bool,
) -> tuple[tuple[VirtualVideoCase, ...], tuple[dict[str, Any], ...], int, int]:
    cases: list[VirtualVideoCase] = []
    repairs: list[dict[str, Any]] = []
    clue_count = 0
    mapped_clue_count = 0
    seen_case_ids: set[str] = set()
    errors: list[str] = []
    for position, row in enumerate(rows):
        source_index = row.get("index", position)
        case_id = _case_id(subset, split, source_index)
        if case_id in seen_case_ids:
            errors.append(f"duplicate case id {case_id}")
            continue
        seen_case_ids.add(case_id)
        raw_clues = row.get("clue_intervals", ())
        if not isinstance(raw_clues, (list, tuple)):
            errors.append(f"{case_id}: clue_intervals must be a list")
            raw_clues = ()
        normalized_clues: list[tuple[float, float]] = []
        case_repairs: list[dict[str, Any]] = []
        for interval_index, raw_interval in enumerate(raw_clues):
            clue_count += 1
            try:
                interval, repair = _normalize_clue_interval(
                    raw_interval,
                    duration_sec=manifest.duration_sec,
                )
            except (TypeError, ValueError) as exc:
                errors.append(f"{case_id} clue {interval_index}: {exc}")
                continue
            if repair is not None:
                repair = {
                    "case_id": case_id,
                    "source_index": source_index,
                    "interval_index": interval_index,
                    **repair,
                }
                repairs.append(repair)
                case_repairs.append(repair)
            windows = virtual_to_source_windows(manifest, *interval)
            if not windows:
                errors.append(f"{case_id} clue {interval_index}: interval does not map to a source segment")
            else:
                mapped_clue_count += 1
            normalized_clues.append(interval)

        first_interval = normalized_clues[0] if normalized_clues else (0.0, 0.0)
        first_windows = virtual_to_source_windows(manifest, *first_interval)
        cases.append(
            VirtualVideoCase(
                case_id=case_id,
                question=str(row.get("question", "")),
                options=_options_mapping(row.get("options")),
                gold=str(row.get("answer", "")),
                target_segment_id=first_windows[0].segment_id if first_windows else "",
                target_virtual_interval=first_interval,
                gold_clue_intervals=tuple(normalized_clues),
                subset=subset,
                split=split,
                question_type=str(row["question_type"]) if row.get("question_type") is not None else None,
                metadata={
                    "dataset": "MM-Lifelong",
                    "source_subset": source_subset,
                    "source_index": source_index,
                    "temporal_certificate": row.get("temporal_certificate"),
                    "total_intervals": row.get("total_intervals", ()),
                    "clue_repairs": case_repairs,
                },
            )
        )
    if verify_clues and errors:
        preview = "; ".join(errors[:5])
        suffix = f"; and {len(errors) - 5} more" if len(errors) > 5 else ""
        raise ValueError(f"MM-Lifelong clue validation failed: {preview}{suffix}")
    return tuple(cases), tuple(repairs), clue_count, mapped_clue_count


def _normalize_clue_interval(
    value: Any,
    *,
    duration_sec: float,
) -> tuple[tuple[float, float], dict[str, Any] | None]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"expected [start, end], got {value!r}")
    start = float(value[0])
    end = float(value[1])
    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError(f"interval must be finite, got {value!r}")
    repair_kind: str | None = None
    if end < start:
        start, end = end, start
        repair_kind = "reversed"
    elif end == start:
        if start >= duration_sec:
            start = max(0.0, duration_sec - _MIN_CLUE_DURATION_SEC)
            end = duration_sec
        else:
            end = start + _MIN_CLUE_DURATION_SEC
        repair_kind = "zero_length_expanded"
    start = round(start, 3)
    end = round(end, 3)
    if start < 0.0 or end > duration_sec or end <= start:
        raise ValueError(f"interval {value!r} is outside [0, {duration_sec}]")
    repair = None
    if repair_kind is not None:
        repair = {
            "kind": repair_kind,
            "original": [float(value[0]), float(value[1])],
            "normalized": [start, end],
        }
    return (start, end), repair


def _validation_payload(
    manifest: VirtualVideoManifest,
    *,
    cases: Sequence[VirtualVideoCase],
    clue_count: int,
    mapped_clue_count: int,
    repairs: Sequence[Mapping[str, Any]],
    verify_durations: bool,
) -> dict[str, Any]:
    segments = manifest.segments
    source_duration_sum = round(sum(segment.source_end_sec - segment.source_start_sec for segment in segments), 3)
    ordered = all(
        segment.source_end_sec > segment.source_start_sec
        and segment.virtual_end_sec > segment.virtual_start_sec
        and (index == 0 or abs(segment.virtual_start_sec - segments[index - 1].virtual_end_sec) <= 0.001)
        for index, segment in enumerate(segments)
    )
    duration_matches = abs(source_duration_sum - manifest.duration_sec) <= 0.001
    all_clues_mapped = clue_count == mapped_clue_count
    checks = {
        "source_durations_match_manifest": duration_matches,
        "segments_ordered_and_non_overlapping": ordered,
        "all_clues_map_to_source_segments": all_clues_mapped,
    }
    if verify_durations and not duration_matches:
        raise ValueError(
            f"MM-Lifelong duration validation failed: sources={source_duration_sum}, manifest={manifest.duration_sec}"
        )
    status = "passed" if ordered and all_clues_mapped and (duration_matches or not verify_durations) else "failed"
    return {
        "schema_version": 1,
        "status": status,
        "segment_count": len(segments),
        "case_count": len(cases),
        "duration_sec": manifest.duration_sec,
        "source_duration_sum_sec": source_duration_sum,
        "clue_interval_count": clue_count,
        "mapped_clue_interval_count": mapped_clue_count,
        "clue_repair_count": len(repairs),
        "clue_repairs": [dict(repair) for repair in repairs],
        "checks": checks,
    }


def _case_id(subset: str, split: str, source_index: Any) -> str:
    try:
        index_part = f"{int(source_index):04d}"
    except (TypeError, ValueError):
        index_part = re.sub(r"[^A-Za-z0-9._-]+", "_", str(source_index)).strip("_") or "unknown"
    return f"mmlifelong-{subset}-{split}-{index_part}"


def _options_mapping(value: Any) -> dict[str, str]:
    if isinstance(value, Mapping):
        return {str(key): str(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return {labels[index]: str(item) for index, item in enumerate(value) if index < len(labels)}
    return {}


def _natural_key(path: Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in _NATURAL_PART_RE.split(path.as_posix())
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
