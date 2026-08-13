from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

from benchmarks.mmlifelong.schema import BENCHMARK_ID, EvaluationRecord, RuntimeQuestion
from vcah.video import probe_duration
from vcah.virtual_video import (
    VirtualVideoCase,
    VirtualVideoManifest,
    VirtualVideoSegment,
    VirtualVideoWorkspace,
    virtual_to_source_windows,
)


DurationProbe = Callable[[str], float]
_SUBSET_ALIASES = {
    "day": "game",
    "game": "game",
    "week": "week",
    "month": "month",
}
_SOURCE_SUBSETS = {"game": "day", "week": "week", "month": "month"}
_NATURAL_PART_RE = re.compile(r"(\d+)")
_MIN_CLUE_DURATION_SEC = 0.001
_DURATION_CACHE_SCHEMA_VERSION = "MMLifelongSourceDurationCacheV1"
_DURATION_CACHE_FLUSH_INTERVAL = 128


@dataclass(frozen=True)
class MMLifelongCaseBundle:
    runtime_question: RuntimeQuestion
    evaluation_record: EvaluationRecord


@dataclass(frozen=True)
class MMLifelongBuildResult:
    subset: str
    split: str
    asset_root: Path
    case_root: Path
    workspaces: tuple[VirtualVideoWorkspace, ...]
    runtime_questions: tuple[RuntimeQuestion, ...]
    evaluation_records: tuple[EvaluationRecord, ...]
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

    asset_root.mkdir(parents=True, exist_ok=True)
    segments = _build_segments(
        video_paths,
        video_root=video_root,
        source_subset=source_subset,
        duration_probe=duration_probe,
        duration_cache_path=asset_root / "source_durations.json",
    )
    manifest = VirtualVideoManifest(
        workspace_id=f"mmlifelong-{internal_subset}",
        segments=segments,
    )
    rows = _load_rows(metadata_path)
    bundles, repairs, clue_count, mapped_clue_count = _build_case_bundles(
        rows,
        manifest=manifest,
        subset=internal_subset,
        source_subset=source_subset,
        split=split,
        verify_clues=verify_clues,
    )
    validation = _validation_payload(
        manifest,
        case_count=len(bundles),
        clue_count=clue_count,
        mapped_clue_count=mapped_clue_count,
        repairs=repairs,
        verify_durations=verify_durations,
    )

    asset_root.mkdir(parents=True, exist_ok=True)
    case_root.mkdir(parents=True, exist_ok=True)
    workspaces = tuple(
        _create_case_workspace(
            case_root,
            asset_root=asset_root,
            manifest=manifest,
            bundle=bundle,
        )
        for bundle in bundles
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
        runtime_questions=tuple(bundle.runtime_question for bundle in bundles),
        evaluation_records=tuple(bundle.evaluation_record for bundle in bundles),
        validation_path=validation_path,
    )


def _normalize_subset(value: str) -> str:
    key = str(value).strip().casefold()
    try:
        return _SUBSET_ALIASES[key]
    except KeyError as exc:
        raise ValueError("MM-Lifelong subset must be one of: day, game, week, month") from exc


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
    duration_cache_path: Path | None = None,
) -> tuple[VirtualVideoSegment, ...]:
    segments: list[VirtualVideoSegment] = []
    cursor = 0.0
    cache_path = Path(duration_cache_path) if duration_cache_path is not None else None
    cached_durations = _load_duration_cache(cache_path, video_root=video_root)
    cache_changed = False
    for index, path in enumerate(video_paths, start=1):
        relative_path = path.relative_to(video_root).as_posix()
        stat = path.stat()
        cached = cached_durations.get(relative_path)
        if (
            isinstance(cached, Mapping)
            and int(cached.get("size_bytes", -1)) == stat.st_size
            and int(cached.get("mtime_ns", -1)) == stat.st_mtime_ns
        ):
            duration = round(float(cached.get("duration_sec", 0.0) or 0.0), 3)
        else:
            try:
                duration = round(float(duration_probe(str(path))), 3)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to probe MM-Lifelong video duration: {relative_path}"
                ) from exc
            cached_durations[relative_path] = {
                "duration_sec": duration,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
            cache_changed = True
            if cache_path is not None and index % _DURATION_CACHE_FLUSH_INTERVAL == 0:
                _write_duration_cache(
                    cache_path,
                    video_root=video_root,
                    entries=cached_durations,
                )
                cache_changed = False
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError(f"Invalid video duration for {path}: {duration}")
        virtual_start = round(cursor, 3)
        virtual_end = round(virtual_start + duration, 3)
        day_index = _day_index(path.relative_to(video_root).parts)
        segments.append(
            VirtualVideoSegment(
                segment_id=f"seg_{index:04d}",
                source_video_id=path.stem,
                source_path=str(path.resolve()),
                source_start_sec=0.0,
                source_end_sec=duration,
                virtual_start_sec=virtual_start,
                virtual_end_sec=virtual_end,
                day_index=day_index,
                metadata={
                    "source_subset": source_subset,
                    "relative_source_path": relative_path,
                    "source_day_id": f"day{day_index}" if day_index is not None else None,
                },
            )
        )
        cursor = virtual_end
    if cache_path is not None and (cache_changed or not cache_path.is_file()):
        _write_duration_cache(
            cache_path,
            video_root=video_root,
            entries=cached_durations,
        )
    return tuple(segments)


def _load_duration_cache(
    path: Path | None, *, video_root: Path
) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return {}
    if payload.get("schema_version") != _DURATION_CACHE_SCHEMA_VERSION:
        return {}
    if str(payload.get("video_root", "")) != str(video_root.resolve()):
        return {}
    entries = payload.get("entries")
    if not isinstance(entries, Mapping):
        return {}
    return {
        str(key): dict(value)
        for key, value in entries.items()
        if isinstance(value, Mapping)
    }


def _write_duration_cache(
    path: Path, *, video_root: Path, entries: Mapping[str, Mapping[str, Any]]
) -> None:
    _write_json(
        path,
        {
            "schema_version": _DURATION_CACHE_SCHEMA_VERSION,
            "video_root": str(video_root.resolve()),
            "entry_count": len(entries),
            "entries": {key: dict(entries[key]) for key in sorted(entries)},
        },
    )


def _day_index(parts: Sequence[str]) -> int | None:
    for part in parts:
        match = re.fullmatch(r"day[_-]?(\d+)", str(part), flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 1


def _load_rows(metadata_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"Expected a list of MM-Lifelong cases in {metadata_path}")
    return [dict(row) for row in payload]


def load_runtime_question(path: Path) -> RuntimeQuestion:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"Runtime question must be an object: {path}")
    return RuntimeQuestion.from_mapping(payload)


def load_evaluation_record(path: Path) -> EvaluationRecord:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"Evaluation record must be an object: {path}")
    return EvaluationRecord.from_mapping(payload)


def evaluation_record_from_dataset(
    dataset_root: Path,
    *,
    case_id: str,
    subset: str | None = None,
    split: str | None = None,
    source_index: Any = None,
) -> EvaluationRecord:
    parsed_subset, parsed_split, parsed_index = _case_coordinates(case_id)
    internal_subset = _normalize_subset(subset or parsed_subset)
    source_subset = _SOURCE_SUBSETS[internal_subset]
    resolved_split = str(split or parsed_split)
    resolved_index = parsed_index if source_index is None else source_index
    metadata_path = Path(dataset_root) / source_subset / f"{resolved_split}.json"
    rows = _load_rows(metadata_path)
    row = next(
        (
            item
            for position, item in enumerate(rows)
            if _source_indices_match(item.get("index", position), resolved_index)
        ),
        None,
    )
    if row is None:
        raise KeyError(
            f"MM-Lifelong case {case_id} (source index {resolved_index}) not found in {metadata_path}"
        )
    question_type = (
        str(row["question_type"])
        if row.get("question_type") is not None
        else None
    )
    return EvaluationRecord(
        case_id=case_id,
        reference_answer=str(row.get("answer", "")),
        clue_intervals=_dataset_clue_intervals(row),
        evaluation_metadata={
            "benchmark": BENCHMARK_ID,
            "question": str(row.get("question", "")),
            "options": _options_mapping(row.get("options")),
            "question_type": question_type,
            "subset": internal_subset,
            "split": resolved_split,
            "source_subset": source_subset,
            "source_index": row.get("index", resolved_index),
            "temporal_certificate": row.get("temporal_certificate"),
            "total_intervals": row.get("total_intervals", ()),
            "dataset_record_path": str(metadata_path),
        },
    )


def runtime_question_from_case(value: Mapping[str, Any]) -> RuntimeQuestion:
    metadata = value.get("runtime_metadata", value.get("metadata", {}))
    source = dict(metadata) if isinstance(metadata, Mapping) else {}
    safe_metadata = {
        key: source[key]
        for key in ("benchmark", "dataset", "source_subset", "source_index")
        if key in source
    }
    return RuntimeQuestion(
        case_id=str(value["case_id"]),
        question=str(value["question"]),
        options=dict(value.get("options", {})),
        question_type=(
            str(value["question_type"])
            if value.get("question_type") is not None
            else None
        ),
        subset=str(value["subset"]) if value.get("subset") is not None else None,
        split=str(value["split"]) if value.get("split") is not None else None,
        runtime_metadata=safe_metadata,
    )


def _case_coordinates(case_id: str) -> tuple[str, str, Any]:
    parts = str(case_id).split("-")
    if len(parts) < 4 or parts[0].casefold() != "mmlifelong":
        raise ValueError(f"Unsupported MM-Lifelong case id: {case_id}")
    raw_index = parts[-1]
    try:
        source_index: Any = int(raw_index)
    except ValueError:
        source_index = raw_index
    return "-".join(parts[1:-2]), parts[-2], source_index


def _source_indices_match(left: Any, right: Any) -> bool:
    try:
        return int(left) == int(right)
    except (TypeError, ValueError):
        return str(left) == str(right)


def _dataset_clue_intervals(row: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
    raw = row.get("clue_intervals")
    if _is_interval_sequence(raw):
        return tuple(_coerce_clue_interval(item) for item in raw)
    total = row.get("total_intervals")
    if _is_interval_sequence(total):
        return tuple(_coerce_clue_interval(item) for item in total)
    nested = raw if isinstance(raw, (list, tuple)) else row.get("clue_interval", ())
    if not isinstance(nested, (list, tuple)):
        return ()
    return tuple(
        _coerce_clue_interval(interval)
        for group in nested
        if isinstance(group, Mapping)
        for interval in tuple(group.get("intervals", ()) or ())
    )


def _build_case_bundles(
    rows: Sequence[Mapping[str, Any]],
    *,
    manifest: VirtualVideoManifest,
    subset: str,
    source_subset: str,
    split: str,
    verify_clues: bool,
) -> tuple[tuple[MMLifelongCaseBundle, ...], tuple[dict[str, Any], ...], int, int]:
    bundles: list[MMLifelongCaseBundle] = []
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
        source_clues = row.get("clue_intervals", ())
        try:
            raw_clues = _global_clue_intervals(row)
        except (TypeError, ValueError) as exc:
            errors.append(f"{case_id}: {exc}")
            raw_clues = ()
        nested_count = _nested_clue_count(source_clues)
        if nested_count is not None and nested_count != len(raw_clues):
            errors.append(
                f"{case_id}: total_intervals count {len(raw_clues)} does not "
                f"match nested clue count {nested_count}"
            )
        official_clues: list[tuple[float, float]] = []
        normalized_clues: list[tuple[float, float]] = []
        case_repairs: list[dict[str, Any]] = []
        for interval_index, raw_interval in enumerate(raw_clues):
            clue_count += 1
            try:
                official_interval = _coerce_clue_interval(raw_interval)
                interval, repair = _normalize_clue_interval(
                    raw_interval,
                    duration_sec=manifest.duration_sec,
                )
            except (TypeError, ValueError) as exc:
                errors.append(f"{case_id} clue {interval_index}: {exc}")
                continue
            official_clues.append(official_interval)
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

        question_type = (
            str(row["question_type"])
            if row.get("question_type") is not None
            else None
        )
        options = _options_mapping(row.get("options"))
        bundles.append(
            MMLifelongCaseBundle(
                runtime_question=RuntimeQuestion(
                    case_id=case_id,
                    question=str(row.get("question", "")),
                    options=options,
                    subset=subset,
                    split=split,
                    question_type=question_type,
                    runtime_metadata={
                        "benchmark": BENCHMARK_ID,
                        "source_subset": source_subset,
                        "source_index": source_index,
                    },
                ),
                evaluation_record=EvaluationRecord(
                    case_id=case_id,
                    reference_answer=str(row.get("answer", "")),
                    clue_intervals=tuple(official_clues),
                    evaluation_metadata={
                        "benchmark": BENCHMARK_ID,
                        "question": str(row.get("question", "")),
                        "options": options,
                        "question_type": question_type,
                        "subset": subset,
                        "split": split,
                        "source_subset": source_subset,
                        "source_index": source_index,
                        "total_seconds": manifest.duration_sec,
                        "temporal_certificate": row.get("temporal_certificate"),
                        "total_intervals": row.get("total_intervals", ()),
                        "source_clue_intervals": source_clues,
                        "normalized_clue_intervals": [list(item) for item in normalized_clues],
                        "clue_repairs": case_repairs,
                    },
                ),
            )
        )
    if verify_clues and errors:
        preview = "; ".join(errors[:5])
        suffix = f"; and {len(errors) - 5} more" if len(errors) > 5 else ""
        raise ValueError(f"MM-Lifelong clue validation failed: {preview}{suffix}")
    return tuple(bundles), tuple(repairs), clue_count, mapped_clue_count


def _create_case_workspace(
    case_root: Path,
    *,
    asset_root: Path,
    manifest: VirtualVideoManifest,
    bundle: MMLifelongCaseBundle,
) -> VirtualVideoWorkspace:
    question = bundle.runtime_question
    workspace = VirtualVideoWorkspace.create(
        case_root / question.case_id,
        manifest=manifest,
        case=VirtualVideoCase(
            case_id=question.case_id,
            question=question.question,
            options=question.options,
            subset=question.subset,
            split=question.split,
            question_type=question.question_type,
            metadata=question.runtime_metadata,
        ),
        asset_root=asset_root,
    )
    legacy_payload = json.loads((workspace.root_dir / "case.json").read_text(encoding="utf-8"))
    runtime_payload = question.to_dict()
    if legacy_payload.get("asset_ref"):
        runtime_payload["asset_ref"] = legacy_payload["asset_ref"]
    _write_json(workspace.root_dir / "case.json", runtime_payload)
    _write_json(
        workspace.root_dir / "evaluation_case.json",
        bundle.evaluation_record.to_dict(),
    )
    return VirtualVideoWorkspace.load(workspace.root_dir)


def _coerce_clue_interval(value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"expected [start, end], got {value!r}")
    start = float(value[0])
    end = float(value[1])
    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError(f"interval must be finite, got {value!r}")
    return start, end


def _global_clue_intervals(row: Mapping[str, Any]) -> tuple[Any, ...]:
    raw = row.get("clue_intervals", ())
    if _is_interval_sequence(raw):
        return tuple(raw)
    total = row.get("total_intervals")
    if _is_interval_sequence(total):
        return tuple(total)
    if isinstance(raw, (list, tuple)) and any(
        isinstance(value, Mapping) for value in raw
    ):
        raise ValueError(
            "nested clue_intervals require global total_intervals for virtual-time mapping"
        )
    raise TypeError("clue_intervals must be a list of [start, end] pairs")


def _is_interval_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and all(
        isinstance(item, (list, tuple)) and len(item) == 2 for item in value
    )


def _nested_clue_count(value: Any) -> int | None:
    if not isinstance(value, (list, tuple)) or not any(
        isinstance(item, Mapping) for item in value
    ):
        return None
    return sum(
        len(tuple(item.get("intervals", ()) or ()))
        for item in value
        if isinstance(item, Mapping)
    )


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
    case_count: int,
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
        "case_count": int(case_count),
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
