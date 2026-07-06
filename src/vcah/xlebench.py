from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from vcah.index import ColdIndex, KeyframeSampler, RangeDetector, build_cold_index
from vcah.model import ModelClient
from vcah.types import Beat, Hit


@dataclass(frozen=True)
class FrameRef:
    video_uid: str
    beat_id: str
    chapter_id: str
    source_time_sec: float
    virtual_time_sec: float
    path: str


@dataclass(frozen=True)
class LifeLogSegment:
    video_uid: str
    video_path: Path
    duration_sec: float
    virtual_start_sec: float = 0.0

    @property
    def virtual_end_sec(self) -> float:
        return self.virtual_start_sec + self.duration_sec


@dataclass(frozen=True)
class LifeLogScope:
    video_uid: str | None
    source_start_sec: float
    source_end_sec: float
    virtual_start_sec: float
    virtual_end_sec: float


@dataclass(frozen=True)
class GroundTruthInterval:
    video_uid: str
    source_start_sec: float
    source_end_sec: float
    virtual_start_sec: float
    virtual_end_sec: float


@dataclass(frozen=True)
class XLEBenchCase:
    case_id: str
    question: str
    scope: LifeLogScope | None
    gt_intervals: tuple[GroundTruthInterval, ...] = ()
    raw: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class LifeLogManifest:
    segments: tuple[LifeLogSegment, ...]
    cases: tuple[XLEBenchCase, ...] = ()

    @property
    def duration_sec(self) -> float:
        return sum(segment.duration_sec for segment in self.segments)

    def segment(self, video_uid: str) -> LifeLogSegment:
        for segment in self.segments:
            if segment.video_uid == video_uid:
                return segment
        raise ValueError(f"Unknown video_uid: {video_uid}")


@dataclass(frozen=True)
class LifeLogIndexConfig:
    max_chapters: int = 40
    max_range_sec: float = 60.0
    max_beat_sec: float = 60.0
    index_mode: str = "xle-cold-mvp"


@dataclass(frozen=True)
class SegmentColdIndex:
    segment: LifeLogSegment
    index: ColdIndex
    index_dir: Path
    frame_visual_index: "FrameVisualIndex | None" = None
    build_seconds: float = 0.0
    resumed: bool = False

    @property
    def video_uid(self) -> str:
        return self.segment.video_uid

    @property
    def video_path(self) -> Path:
        return self.segment.video_path

    @property
    def duration_sec(self) -> float:
        return self.segment.duration_sec


@dataclass(frozen=True)
class LifeLogColdIndex:
    manifest: LifeLogManifest
    segments: tuple[SegmentColdIndex, ...]
    run_dir: Path

    def segment(self, video_uid: str) -> SegmentColdIndex:
        for segment in self.segments:
            if segment.segment.video_uid == video_uid:
                return segment
        raise ValueError(f"Unknown indexed video_uid: {video_uid}")

    def save(self) -> None:
        payload = {
            "schema": "vcah.lifelog_cold_index.v1",
            "duration_sec": self.manifest.duration_sec,
            "segments": [
                {
                    "video_uid": item.segment.video_uid,
                    "video_path": str(item.segment.video_path),
                    "duration_sec": item.segment.duration_sec,
                    "virtual_start_sec": item.segment.virtual_start_sec,
                    "virtual_end_sec": item.segment.virtual_end_sec,
                    "index_dir": str(item.index_dir),
                    "beat_count": len(item.index.beats),
                    "chapter_count": len(item.index.chapters),
                    "build_seconds": item.build_seconds,
                    "resumed": item.resumed,
                }
                for item in self.segments
            ],
        }
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "lifelog_index.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, run_dir: Path, *, model: ModelClient | None = None) -> "LifeLogColdIndex":
        run_dir = Path(run_dir)
        payload = json.loads((run_dir / "lifelog_index.json").read_text(encoding="utf-8"))
        segments = []
        manifest_segments = []
        for item in payload.get("segments", ()):
            segment = LifeLogSegment(
                video_uid=str(item["video_uid"]),
                video_path=Path(item["video_path"]),
                duration_sec=float(item["duration_sec"]),
                virtual_start_sec=float(item.get("virtual_start_sec", 0.0)),
            )
            index_dir = Path(item["index_dir"])
            manifest_segments.append(segment)
            segments.append(
                SegmentColdIndex(
                    segment=segment,
                    index=ColdIndex.load(index_dir, model=model),
                    index_dir=index_dir,
                    frame_visual_index=FrameVisualIndex.load(index_dir.parent / "frame_visual_index.npz", model)
                    if (index_dir.parent / "frame_visual_index.npz").exists()
                    else None,
                    build_seconds=float(item.get("build_seconds", 0.0)),
                    resumed=True,
                )
            )
        return cls(LifeLogManifest(tuple(manifest_segments)), tuple(segments), run_dir)


class FrameVisualIndex:
    def __init__(self, model: ModelClient) -> None:
        self.model = model
        self.beat_ids: tuple[str, ...] = ()
        self.frame_paths: tuple[str, ...] = ()
        self.frame_times: tuple[float, ...] = ()
        self.embeddings = np.zeros((0, int(getattr(model, "embedding_dim", 0) or 0)), dtype=np.float32)

    @property
    def frame_count(self) -> int:
        return len(self.frame_paths)

    def build(self, beats: Sequence[Beat]) -> None:
        entries = tuple(
            (beat.beat_id, path, time_sec)
            for beat in beats
            for path, time_sec in zip(beat.frame_paths, beat.frame_times)
            if str(path).strip()
        )
        self.beat_ids = tuple(beat_id for beat_id, _path, _time_sec in entries)
        self.frame_paths = tuple(str(path) for _beat_id, path, _time_sec in entries)
        self.frame_times = tuple(float(time_sec) for _beat_id, _path, time_sec in entries)
        if not self.frame_paths:
            return
        rows = np.asarray(self.model.embed_image(self.frame_paths), dtype=np.float32)
        if rows.ndim != 2 or rows.shape[0] != len(self.frame_paths):
            raise ValueError("embed_image must return an (N, D) array")
        self.embeddings = _l2_normalize(rows)

    def search(self, query: str, *, k: int = 20) -> tuple[Hit, ...]:
        if getattr(self.model, "embed_model", "") == "local-hash" and not bool(getattr(self.model, "allow_placeholder_visual", False)):
            return ()
        if not self.frame_paths or self.embeddings.size == 0 or k <= 0:
            return ()
        query_vec = np.asarray(self.model.embed_text((query,)), dtype=np.float32)
        if query_vec.ndim != 2 or query_vec.shape[0] != 1:
            raise ValueError("embed_text must return a (1, D) array")
        scores = self.embeddings @ _l2_normalize(query_vec)[0]
        best_by_beat: dict[str, float] = {}
        for idx, score in enumerate(scores):
            value = float(score)
            if value <= 0.0:
                continue
            beat_id = self.beat_ids[idx]
            best_by_beat[beat_id] = max(best_by_beat.get(beat_id, 0.0), value)
        return tuple(
            Hit(beat_id, score, "visual")
            for beat_id, score in sorted(best_by_beat.items(), key=lambda item: (-item[1], item[0]))[: max(0, int(k))]
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            beat_ids=np.asarray(self.beat_ids, dtype=str),
            frame_paths=np.asarray(self.frame_paths, dtype=str),
            frame_times=np.asarray(self.frame_times, dtype=np.float32),
            embeddings=self.embeddings,
        )

    @classmethod
    def load(cls, path: Path, model: ModelClient) -> "FrameVisualIndex":
        index = cls(model)
        payload = np.load(path, allow_pickle=False)
        index.beat_ids = tuple(str(item) for item in payload["beat_ids"].tolist())
        index.frame_paths = tuple(str(item) for item in payload["frame_paths"].tolist())
        index.frame_times = tuple(float(item) for item in payload["frame_times"].tolist())
        index.embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        return index


@dataclass(frozen=True)
class CandidateWindow:
    video_uid: str
    beat_id: str
    source_start_sec: float
    source_end_sec: float
    virtual_start_sec: float
    virtual_end_sec: float
    score: float
    modalities: tuple[str, ...]
    frame_refs: tuple[FrameRef, ...] = ()

    def overlap_ratio(self, interval: GroundTruthInterval) -> float:
        if self.video_uid != interval.video_uid:
            return 0.0
        intersection = max(
            0.0,
            min(self.source_end_sec, interval.source_end_sec) - max(self.source_start_sec, interval.source_start_sec),
        )
        width = max(1e-9, interval.source_end_sec - interval.source_start_sec)
        return intersection / width


@dataclass(frozen=True)
class RetrievalResult:
    candidates: tuple[CandidateWindow, ...]
    per_channel_hits: Mapping[str, tuple[str, ...]]
    debug: Mapping[str, Any]
    per_level_hits: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    fusion_weights: Mapping[str, float] = field(default_factory=dict)


class LifeLogColdIndexBuilder:
    def __init__(
        self,
        manifest: LifeLogManifest,
        config: LifeLogIndexConfig | None = None,
        *,
        model: ModelClient | None = None,
        range_detector: RangeDetector | None = None,
        keyframe_sampler: KeyframeSampler | None = None,
    ) -> None:
        self.manifest = manifest
        self.config = config or LifeLogIndexConfig()
        self.model = model or ModelClient()
        self.range_detector = range_detector
        self.keyframe_sampler = keyframe_sampler

    def build(self, run_dir: Path, *, resume: bool = True) -> LifeLogColdIndex:
        run_dir = Path(run_dir)
        indexed_segments = []
        for segment in self.manifest.segments:
            indexed_segments.append(self.build_segment(segment, run_dir, resume=resume))
        index = LifeLogColdIndex(self.manifest, tuple(indexed_segments), run_dir)
        index.save()
        return index

    def build_segment(self, segment: LifeLogSegment, run_dir: Path, *, resume: bool = True) -> SegmentColdIndex:
        segment_root = Path(run_dir) / "segments" / _safe_id(segment.video_uid)
        index_dir = segment_root / "cold_index"
        state_path = segment_root / "state.json"
        fingerprint_inputs = _segment_fingerprint_inputs(segment, self.config, self.model)
        fingerprint = _fingerprint(fingerprint_inputs)
        if (
            resume
            and _segment_artifacts_exist(index_dir)
            and _state_matches(state_path, fingerprint)
        ):
            return SegmentColdIndex(
                segment=segment,
                index=ColdIndex.load(index_dir, model=self.model),
                index_dir=index_dir,
                frame_visual_index=FrameVisualIndex.load(segment_root / "frame_visual_index.npz", self.model),
                resumed=True,
            )
        start = perf_counter()
        cold = build_cold_index(
            str(segment.video_path),
            duration_sec=segment.duration_sec,
            run_dir=segment_root,
            model=self.model,
            range_detector=self.range_detector,
            keyframe_sampler=self.keyframe_sampler,
            max_chapters=self.config.max_chapters,
            max_range_sec=self.config.max_range_sec,
            max_beat_sec=self.config.max_beat_sec,
            index_mode=self.config.index_mode,
        )
        frame_visual_index = FrameVisualIndex(self.model)
        frame_visual_index.build(cold.beats)
        frame_visual_index.save(segment_root / "frame_visual_index.npz")
        build_seconds = perf_counter() - start
        _write_segment_state(
            state_path,
            fingerprint=fingerprint,
            fingerprint_inputs=fingerprint_inputs,
            segment=segment,
            index=cold,
            index_dir=index_dir,
        )
        return SegmentColdIndex(
            segment=segment,
            index=cold,
            index_dir=index_dir,
            frame_visual_index=frame_visual_index,
            build_seconds=build_seconds,
            resumed=False,
        )


class LifeLogRetriever:
    def __init__(self, index: LifeLogColdIndex) -> None:
        self.index = index

    def retrieve(self, query: str, *, scope: LifeLogScope | None = None, top_k: int = 20) -> RetrievalResult:
        candidates: dict[tuple[str, str], CandidateWindow] = {}
        per_channel: dict[str, list[str]] = {"text": [], "visual": []}
        segment_scores: dict[str, float] = {}
        frame_level_hits: list[str] = []
        fusion_weights = {"text": 1.0, "visual": 1.0, "segment": 0.05}
        for segment_index in self.index.segments:
            if scope and scope.video_uid and segment_index.segment.video_uid != scope.video_uid:
                continue
            visual_hits = (
                segment_index.frame_visual_index.search(query, k=top_k)
                if segment_index.frame_visual_index is not None
                else segment_index.index.search_visual(query, k=top_k)
            )
            hits = [*segment_index.index.search_text(query), *visual_hits]
            if hits:
                segment_scores[segment_index.segment.video_uid] = sum(max(0.0, float(hit.score)) for hit in hits)
            for hit in hits:
                beat = segment_index.index.get_beat(hit.beat_id)
                if not _beat_overlaps_scope(beat, scope):
                    continue
                key = (segment_index.segment.video_uid, beat.beat_id)
                previous = candidates.get(key)
                modalities = _append_unique(previous.modalities if previous else (), hit.modality)
                segment_boost = fusion_weights["segment"] * segment_scores.get(segment_index.segment.video_uid, 0.0)
                score = float(hit.score) + (previous.score if previous else 0.0) + (segment_boost if previous is None else 0.0)
                candidates[key] = _candidate_from_beat(segment_index.segment, beat, score=score, modalities=modalities)
                per_channel.setdefault(hit.modality, []).append(f"{segment_index.segment.video_uid}:{beat.beat_id}")
                if hit.modality == "visual":
                    frame_level_hits.append(f"{segment_index.segment.video_uid}:{beat.beat_id}")
        ranked = tuple(sorted(candidates.values(), key=lambda item: (-item.score, item.video_uid, item.source_start_sec))[:top_k])
        per_level_hits = {
            "segment": tuple(
                video_uid
                for video_uid, _score in sorted(segment_scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
            ),
            "beat": tuple(f"{candidate.video_uid}:{candidate.beat_id}" for candidate in ranked),
            "frame": _dedupe(frame_level_hits)[:top_k],
        }
        return RetrievalResult(
            candidates=ranked,
            per_channel_hits={key: tuple(values) for key, values in per_channel.items()},
            debug={
                "retrieval_mode": "hierarchical-mvp",
                "segment_count": len(self.index.segments),
                "scope_video_uid": scope.video_uid if scope else None,
                "top_k": int(top_k),
            },
            per_level_hits=per_level_hits,
            fusion_weights=fusion_weights,
        )


def diagnose_cold_recall(
    index: LifeLogColdIndex,
    cases: Sequence[XLEBenchCase],
    *,
    top_ks: Sequence[int] = (5, 20),
) -> dict[str, Any]:
    top_ks = tuple(sorted({int(k) for k in top_ks if int(k) > 0}))
    max_k = max(top_ks, default=20)
    hits = {k: 0 for k in top_ks}
    channel_hits = {f"text@{k}": 0 for k in top_ks} | {f"visual@{k}": 0 for k in top_ks}
    level_hits = {f"segment@{k}": 0 for k in top_ks} | {f"beat@{k}": 0 for k in top_ks} | {f"frame@{k}": 0 for k in top_ks}
    coverage_values = []
    evaluated = 0
    retriever = LifeLogRetriever(index)
    for case in cases:
        if not case.gt_intervals:
            continue
        evaluated += 1
        result = retriever.retrieve(case.question, scope=case.scope, top_k=max_k)
        coverage = max(
            (candidate.overlap_ratio(interval) for candidate in result.candidates for interval in case.gt_intervals),
            default=0.0,
        )
        coverage_values.append(coverage)
        for k in top_ks:
            top = result.candidates[:k]
            if _any_candidate_hits(top, case.gt_intervals):
                hits[k] += 1
            for channel in ("text", "visual"):
                if _any_candidate_hits([item for item in top if channel in item.modalities], case.gt_intervals):
                    channel_hits[f"{channel}@{k}"] += 1
            if _level_hits_interval(result.per_level_hits.get("segment", ())[:k], case.gt_intervals, level="segment"):
                level_hits[f"segment@{k}"] += 1
            if _any_candidate_hits(top, case.gt_intervals):
                level_hits[f"beat@{k}"] += 1
            if _any_candidate_hits([item for item in top if "visual" in item.modalities], case.gt_intervals):
                level_hits[f"frame@{k}"] += 1
    denominator = max(1, evaluated)
    return {
        "case_count": evaluated,
        **{f"cold_recall@{k}": hits[k] / denominator for k in top_ks},
        "gt_interval_candidate_coverage": sum(coverage_values) / denominator if coverage_values else 0.0,
        "per_channel_recall": {key: value / denominator for key, value in channel_hits.items()},
        "per_level_recall": {key: value / denominator for key, value in level_hits.items()},
        "counts": {
            "segments": len(index.segments),
            "beats": sum(len(segment.index.beats) for segment in index.segments),
            "frames": sum(len(beat.frame_paths) for segment in index.segments for beat in segment.index.beats),
            "embeddings": sum(_visual_embedding_count(segment) for segment in index.segments),
        },
    }


def load_xlebench_manifest(
    root: Path,
    *,
    video_template: str | None = None,
    annotation_file: Path | None = None,
    default_duration_sec: float | None = None,
) -> LifeLogManifest:
    root = Path(root)
    records = _load_records(annotation_file or _find_annotation_file(root))
    segments_by_uid: dict[str, LifeLogSegment] = {}
    cases = []
    for record in records:
        raw_segments = _segments_from_record(record, root=root, video_template=video_template, default_duration_sec=default_duration_sec)
        for segment in raw_segments:
            if segment.video_uid not in segments_by_uid:
                virtual_start = sum(item.duration_sec for item in segments_by_uid.values())
                segments_by_uid[segment.video_uid] = LifeLogSegment(
                    segment.video_uid,
                    segment.video_path,
                    segment.duration_sec,
                    virtual_start,
                )
        segments = tuple(segments_by_uid.values())
        cases.append(_case_from_record(record, segments))
    return LifeLogManifest(tuple(segments_by_uid.values()), tuple(cases))


def write_diagnose_report(report: Mapping[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _load_records(path: Path) -> tuple[Mapping[str, Any], ...]:
    suffix = path.suffix.casefold()
    if suffix == ".jsonl":
        return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return tuple(item for item in payload if isinstance(item, Mapping))
    for key in ("cases", "data", "annotations", "questions"):
        value = payload.get(key) if isinstance(payload, Mapping) else None
        if isinstance(value, list):
            return tuple(item for item in value if isinstance(item, Mapping))
    if isinstance(payload, Mapping):
        return (payload,)
    raise ValueError(f"Unsupported X-LeBench annotation payload: {path}")


def _find_annotation_file(root: Path) -> Path:
    candidates = (
        "cases.jsonl",
        "cases.json",
        "annotations.jsonl",
        "annotations.json",
        "questions.jsonl",
        "questions.json",
        "manifest.json",
    )
    for name in candidates:
        path = root / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No X-LeBench annotation file found under {root}")


def _segments_from_record(
    record: Mapping[str, Any],
    *,
    root: Path,
    video_template: str | None,
    default_duration_sec: float | None,
) -> tuple[LifeLogSegment, ...]:
    raw_videos = record.get("videos") or record.get("segments") or record.get("video_segments")
    if isinstance(raw_videos, Sequence) and not isinstance(raw_videos, (str, bytes, bytearray)):
        segments = tuple(
            _segment_from_value(item, record, root=root, video_template=video_template, default_duration_sec=default_duration_sec)
            for item in raw_videos
        )
        if segments:
            return segments
    video_uids = record.get("video_uids") or record.get("video_ids")
    if isinstance(video_uids, Sequence) and not isinstance(video_uids, (str, bytes, bytearray)):
        return tuple(
            _segment_from_uid(str(uid), record, index=idx, root=root, video_template=video_template, default_duration_sec=default_duration_sec)
            for idx, uid in enumerate(video_uids)
        )
    video_uid = str(record.get("video_uid") or record.get("video_id") or record.get("uid") or "").strip()
    if video_uid:
        return (_segment_from_uid(video_uid, record, index=0, root=root, video_template=video_template, default_duration_sec=default_duration_sec),)
    raise ValueError("X-LeBench record does not contain video_uid/video_uids/videos")


def _segment_from_value(
    value: Any,
    record: Mapping[str, Any],
    *,
    root: Path,
    video_template: str | None,
    default_duration_sec: float | None,
) -> LifeLogSegment:
    if isinstance(value, Mapping):
        uid = str(value.get("video_uid") or value.get("video_id") or value.get("uid") or Path(str(value.get("video_path", ""))).stem)
        duration = _duration_for_uid(uid, value, default_duration_sec=default_duration_sec)
        path = _video_path_for_uid(uid, root=root, video_template=video_template, value=value)
        return LifeLogSegment(uid, path, duration)
    return _segment_from_uid(str(value), record, index=0, root=root, video_template=video_template, default_duration_sec=default_duration_sec)


def _segment_from_uid(
    uid: str,
    record: Mapping[str, Any],
    *,
    index: int,
    root: Path,
    video_template: str | None,
    default_duration_sec: float | None,
) -> LifeLogSegment:
    duration = _duration_for_uid(uid, record, index=index, default_duration_sec=default_duration_sec)
    return LifeLogSegment(uid, _video_path_for_uid(uid, root=root, video_template=video_template, value=record), duration)


def _duration_for_uid(uid: str, value: Mapping[str, Any], *, index: int = 0, default_duration_sec: float | None) -> float:
    for key in ("duration_sec", "duration", "video_duration_sec"):
        if key in value and not isinstance(value[key], Mapping):
            return float(value[key])
    durations = value.get("durations") or value.get("duration_secs")
    if isinstance(durations, Mapping) and uid in durations:
        return float(durations[uid])
    if isinstance(durations, Sequence) and not isinstance(durations, (str, bytes, bytearray)) and index < len(durations):
        return float(durations[index])
    if default_duration_sec is not None:
        return float(default_duration_sec)
    raise ValueError(f"Missing duration for X-LeBench video_uid={uid!r}")


def _video_path_for_uid(uid: str, *, root: Path, video_template: str | None, value: Mapping[str, Any]) -> Path:
    raw = value.get("video_path") or value.get("path") or value.get("video")
    if raw:
        path = Path(str(raw).format(video_uid=uid))
        return path if path.is_absolute() else root / path
    if video_template:
        return Path(video_template.format(video_uid=uid))
    return root / "videos" / f"{uid}.mp4"


def _case_from_record(record: Mapping[str, Any], segments: Sequence[LifeLogSegment]) -> XLEBenchCase:
    case_id = str(record.get("case_id") or record.get("id") or record.get("question_id") or "")
    question = str(record.get("question") or record.get("query") or record.get("prompt") or "")
    scope_uid = _range_video_uid(record.get("query_range"), record) or _only_or_default_uid(record, segments)
    scope = _scope_from_range(record.get("query_range"), scope_uid, segments)
    intervals = _gt_intervals(record, scope_uid, segments)
    return XLEBenchCase(case_id=case_id, question=question, scope=scope, gt_intervals=intervals, raw=dict(record))


def _scope_from_range(value: Any, video_uid: str | None, segments: Sequence[LifeLogSegment]) -> LifeLogScope | None:
    if video_uid is None:
        return None
    segment = _segment_by_uid(segments, video_uid)
    start, end = _parse_range(value, default=(0.0, segment.duration_sec))
    return LifeLogScope(video_uid, start, end, segment.virtual_start_sec + start, segment.virtual_start_sec + end)


def _gt_intervals(record: Mapping[str, Any], fallback_uid: str | None, segments: Sequence[LifeLogSegment]) -> tuple[GroundTruthInterval, ...]:
    value = (
        record.get("gt_intervals")
        or record.get("gt_interval")
        or record.get("answer_intervals")
        or record.get("answer_interval")
    )
    if value is None:
        return ()
    raw_items = value if _is_interval_list(value) else (value,)
    intervals = []
    for item in raw_items:
        uid = _range_video_uid(item, record) or fallback_uid
        if uid is None:
            continue
        segment = _segment_by_uid(segments, uid)
        start, end = _parse_range(item, default=(0.0, segment.duration_sec))
        intervals.append(GroundTruthInterval(uid, start, end, segment.virtual_start_sec + start, segment.virtual_start_sec + end))
    return tuple(intervals)


def _is_interval_list(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and bool(value)
        and not (len(value) == 2 and all(isinstance(item, (int, float)) for item in value))
    )


def _range_video_uid(value: Any, record: Mapping[str, Any]) -> str | None:
    if isinstance(value, Mapping):
        raw = value.get("video_uid") or value.get("video_id")
        if raw:
            return str(raw)
    raw = record.get("video_uid") or record.get("video_id")
    return str(raw) if raw else None


def _only_or_default_uid(record: Mapping[str, Any], segments: Sequence[LifeLogSegment]) -> str | None:
    raw = record.get("video_uid") or record.get("video_id")
    if raw:
        return str(raw)
    if len(segments) == 1:
        return segments[0].video_uid
    return None


def _parse_range(value: Any, *, default: tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, Mapping):
        start = value.get("start_sec", value.get("start", value.get("begin", default[0])))
        end = value.get("end_sec", value.get("end", value.get("stop", default[1])))
        return float(start), float(end)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return default


def _segment_by_uid(segments: Sequence[LifeLogSegment], video_uid: str) -> LifeLogSegment:
    for segment in segments:
        if segment.video_uid == video_uid:
            return segment
    raise ValueError(f"Unknown video_uid in X-LeBench case: {video_uid}")


def _beat_overlaps_scope(beat: Beat, scope: LifeLogScope | None) -> bool:
    if scope is None:
        return True
    return beat.end_sec >= scope.source_start_sec and beat.start_sec <= scope.source_end_sec


def _candidate_from_beat(segment: LifeLogSegment, beat: Beat, *, score: float, modalities: Sequence[str]) -> CandidateWindow:
    return CandidateWindow(
        video_uid=segment.video_uid,
        beat_id=beat.beat_id,
        source_start_sec=beat.start_sec,
        source_end_sec=beat.end_sec,
        virtual_start_sec=segment.virtual_start_sec + beat.start_sec,
        virtual_end_sec=segment.virtual_start_sec + beat.end_sec,
        score=float(score),
        modalities=tuple(modalities),
        frame_refs=_frame_refs(segment, beat),
    )


def _frame_refs(segment: LifeLogSegment, beat: Beat) -> tuple[FrameRef, ...]:
    paths = beat.frame_paths or ((beat.keyframe_path,) if beat.keyframe_path else ())
    return tuple(
        FrameRef(
            video_uid=segment.video_uid,
            beat_id=beat.beat_id,
            chapter_id=beat.chapter_id,
            source_time_sec=float(time_sec),
            virtual_time_sec=segment.virtual_start_sec + float(time_sec),
            path=str(path),
        )
        for path, time_sec in zip(paths, beat.frame_times)
        if str(path).strip()
    )


def _append_unique(values: Sequence[str], value: str) -> tuple[str, ...]:
    result = list(values)
    if value not in result:
        result.append(value)
    return tuple(result)


def _any_candidate_hits(candidates: Iterable[CandidateWindow], intervals: Sequence[GroundTruthInterval]) -> bool:
    return any(candidate.overlap_ratio(interval) > 0.0 for candidate in candidates for interval in intervals)


def _level_hits_interval(ids: Sequence[str], intervals: Sequence[GroundTruthInterval], *, level: str) -> bool:
    if level == "segment":
        return any(interval.video_uid in ids for interval in intervals)
    return False


def _segment_fingerprint_inputs(segment: LifeLogSegment, config: LifeLogIndexConfig, model: ModelClient) -> dict[str, Any]:
    return {
        "video_uid": segment.video_uid,
        "video_path": str(segment.video_path),
        "duration_sec": segment.duration_sec,
        "virtual_start_sec": segment.virtual_start_sec,
        "max_chapters": config.max_chapters,
        "max_range_sec": config.max_range_sec,
        "max_beat_sec": config.max_beat_sec,
        "index_mode": config.index_mode,
        "model_fingerprint": _model_fingerprint(model),
    }


def _model_fingerprint(model: ModelClient) -> dict[str, Any]:
    return {
        "class": model.__class__.__name__,
        "embedding_dim": int(getattr(model, "embedding_dim", 0) or 0),
        "embed_model": str(getattr(model, "embed_model", model.__class__.__name__) or "unknown"),
        "vision_model": str(getattr(model, "vision_model", "") or ""),
    }


def _fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _segment_artifacts_exist(index_dir: Path) -> bool:
    segment_root = index_dir.parent
    return (
        (index_dir / "index.json").exists()
        and (index_dir / "diagnostics.json").exists()
        and (index_dir / "visual_index.npz").exists()
        and (segment_root / "frame_visual_index.npz").exists()
    )


def _state_matches(state_path: Path, fingerprint: str) -> bool:
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return str(payload.get("schema")) == "vcah.xle.segment_state.v1" and str(payload.get("fingerprint")) == fingerprint


def _write_segment_state(
    state_path: Path,
    *,
    fingerprint: str,
    fingerprint_inputs: Mapping[str, Any],
    segment: LifeLogSegment,
    index: ColdIndex,
    index_dir: Path,
) -> None:
    payload = {
        "schema": "vcah.xle.segment_state.v1",
        "fingerprint": fingerprint,
        "fingerprint_inputs": dict(fingerprint_inputs),
        "video_uid": segment.video_uid,
        "index_dir": str(index_dir),
        "beat_count": len(index.beats),
        "chapter_count": len(index.chapters),
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _visual_embedding_count(segment: SegmentColdIndex) -> int:
    if segment.frame_visual_index is not None:
        return segment.frame_visual_index.frame_count
    return len(segment.index.visual_index.beat_ids)


def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result)


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return array / norms


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in str(value))
