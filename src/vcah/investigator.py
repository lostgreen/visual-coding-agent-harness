from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Mapping, Sequence

from vcah.caption_hybrid_search import CaptionHybridSearch
from vcah.caption_lexical_index import CaptionLexicalIndex, render_caption_hits
from vcah.caption_semantic_index import CaptionSemanticIndex
from vcah.embedding_adapter import TextEmbeddingAdapter
from vcah.types import EvidenceRecord
from vcah.virtual_index import load_virtual_beats
from vcah.virtual_video import (
    FrameSampler,
    VirtualFrameRef,
    VirtualVideoWorkspace,
    materialize_window_frames,
    virtual_to_source_windows,
)
from vcah.workspace import evidence_attempt_id


@dataclass(frozen=True)
class ObservationAttempt:
    """One immutable inspection interpretation of identifiable source material."""

    attempt_id: str
    task_id: str = ""
    requested_range: tuple[float, float] | None = None
    inspected_ranges: tuple[tuple[float, float], ...] = ()
    attached_frame_times: tuple[float, ...] = ()
    sampling_config: Mapping[str, Any] = field(default_factory=dict)
    images_requested: int = 0
    images_attached: int = 0
    images_dropped: int = 0
    parse_status: str = "unknown"
    execution_status: str = "completed"
    frame_refs: tuple[str, ...] = ()
    modality: str = "visual"
    evidence_role: str = "unclassified"
    prompt_digest: str = ""
    raw_output: str = ""
    round_id: str = ""
    source_video_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "attempt_id", str(self.attempt_id or "").strip())
        object.__setattr__(self, "task_id", str(self.task_id or "").strip())
        requested = self.requested_range
        object.__setattr__(
            self,
            "requested_range",
            _normalized_range(requested) if requested is not None and _valid_range(requested) else None,
        )
        object.__setattr__(
            self,
            "inspected_ranges",
            tuple(_normalized_range(item) for item in self.inspected_ranges if _valid_range(item)),
        )
        object.__setattr__(self, "attached_frame_times", tuple(float(item) for item in self.attached_frame_times))
        config = dict(self.sampling_config or {})
        object.__setattr__(self, "sampling_config", config)
        requested_count = max(0, int(self.images_requested or 0))
        attached_count = max(0, int(self.images_attached or 0))
        object.__setattr__(self, "images_requested", requested_count)
        object.__setattr__(self, "images_attached", min(requested_count, attached_count))
        object.__setattr__(self, "images_dropped", max(0, int(self.images_dropped or 0)))
        object.__setattr__(self, "parse_status", str(self.parse_status or "unknown").strip().casefold())
        execution_status = str(self.execution_status or "completed").strip().casefold()
        if execution_status not in {"completed", "failed"}:
            raise ValueError(f"invalid execution_status: {execution_status}")
        object.__setattr__(
            self,
            "execution_status",
            execution_status,
        )
        object.__setattr__(self, "frame_refs", tuple(str(item) for item in self.frame_refs if str(item).strip()))
        object.__setattr__(self, "modality", str(self.modality or "visual").strip().casefold())
        evidence_role = str(self.evidence_role or "unclassified").strip().casefold()
        if evidence_role not in {"unclassified", "candidate", "supporting", "negative"}:
            raise ValueError(f"invalid evidence_role: {evidence_role}")
        object.__setattr__(self, "evidence_role", evidence_role)
        object.__setattr__(self, "prompt_digest", str(self.prompt_digest or "").strip())
        object.__setattr__(self, "raw_output", str(self.raw_output or ""))
        object.__setattr__(self, "round_id", str(self.round_id or "").strip())
        object.__setattr__(
            self,
            "source_video_ids",
            tuple(dict.fromkeys(str(item).strip() for item in self.source_video_ids if str(item).strip())),
        )

    @property
    def sampling_fps(self) -> float:
        try:
            return max(0.0, float(self.sampling_config.get("fps", 0.0) or 0.0))
        except (TypeError, ValueError):
            return 0.0


@dataclass(frozen=True)
class InvestigationReport:
    query_id: str
    status: str
    evidence: tuple[EvidenceRecord, ...] = ()
    attempts: tuple[ObservationAttempt, ...] = ()
    cost: Mapping[str, Any] = field(default_factory=dict)
    failure_reason: str = ""
    coverage_delta: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", str(self.query_id or "").strip())
        status = str(self.status or "").strip().casefold()
        if status not in {"completed", "failed"}:
            raise ValueError(f"invalid investigation status: {status or 'missing'}")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "attempts", tuple(_attempt(item) for item in self.attempts))
        object.__setattr__(self, "cost", dict(self.cost or {}))
        object.__setattr__(self, "failure_reason", str(self.failure_reason or "").strip())
        object.__setattr__(
            self,
            "coverage_delta",
            tuple(_normalized_range(item) for item in self.coverage_delta if _valid_range(item)),
        )


class VirtualVideoInvestigator:
    """Mechanical virtual-video tools shared by observation-only Investigators."""

    tool_names = ("open_segment", "inspect_window", "search_asr", "search_caption")

    def __init__(
        self,
        workspace: VirtualVideoWorkspace,
        *,
        sampler: FrameSampler | None = None,
        caption_index: CaptionLexicalIndex | None = None,
        caption_embedding_adapter: TextEmbeddingAdapter | None = None,
        caption_config_digest: str | None = None,
    ) -> None:
        self.workspace = workspace
        self.sampler = sampler
        self._caption_indexes: dict[str, Any] = {}
        if caption_index is not None:
            self._caption_indexes["lexical"] = caption_index
        self._caption_embedding_adapter = caption_embedding_adapter
        self._caption_config_digest = str(caption_config_digest or "").strip() or None
        self.ledger_path = self.workspace.root_dir / "exploration_ledger.jsonl"
        self._visit_count = 0

    def run_batch(self, tasks: Sequence[Any]) -> tuple[InvestigationReport, ...]:
        return tuple(self._investigate_task(task) for task in tasks)

    def reset_run_state(self) -> None:
        self._visit_count = 0
        self.ledger_path.write_text("", encoding="utf-8")

    def open_segment(self, segment_id: str) -> Mapping[str, Any]:
        segment = self.workspace.manifest.segment(str(segment_id))
        beats = _beats_for_segment(self.workspace, segment.segment_id)
        cues = _asr_cues_in_window(self.workspace, segment.virtual_start_sec, segment.virtual_end_sec)
        return {
            "segment_id": segment.segment_id,
            "virtual_time_range": [segment.virtual_start_sec, segment.virtual_end_sec],
            "duration_sec": segment.duration_sec,
            "asr_timeline_summary": " ".join(cue["text"] for cue in cues)[:500],
            "asr_cues": cues,
            "source_lineage": _source_lineage(
                self.workspace,
                segment.virtual_start_sec,
                segment.virtual_end_sec,
            ),
            "beats": [
                {
                    "beat_id": str(beat.get("beat_id", "")),
                    "virtual_time_range": list(beat.get("virtual_time_range", ())),
                    "asr_excerpt": _beat_asr_excerpt(beat),
                    "thumbnail_grid_paths": list(
                        beat.get("thumbnail_grid_paths") or [beat.get("thumbnail_grid_path", "")]
                    ),
                }
                for beat in beats
            ],
        }

    def inspect_window(
        self,
        start_sec: float,
        end_sec: float,
        *,
        fps: float = 0.5,
        max_frames: int = 64,
        query_id: str = "manual",
        phase_offset_sec: float = 0.0,
    ) -> Mapping[str, Any]:
        capped = min(512, max(1, int(max_frames)))
        frames = materialize_window_frames(
            self.workspace,
            float(start_sec),
            float(end_sec),
            query_id=str(query_id),
            fps=float(fps),
            max_frames=capped,
            sampler=self.sampler,
            phase_offset_sec=float(phase_offset_sec),
        )
        return {
            "virtual_time_range": [float(start_sec), float(end_sec)],
            "sampling": {
                "fps": float(fps),
                "max_frames": capped,
                "actual_frames": len(frames),
                "sampling": "uniform",
                "phase_offset_sec": float(phase_offset_sec),
            },
            "frames": [_frame_payload(frame) for frame in frames],
            "asr_cues": _asr_cues_in_window(self.workspace, float(start_sec), float(end_sec)),
            "source_lineage": _source_lineage(self.workspace, float(start_sec), float(end_sec)),
        }

    def search_asr(
        self,
        terms: Sequence[str],
        *,
        segment_id: str = "",
        time_range: Sequence[float] | None = None,
        max_clusters: int = 8,
        padding_sec: float = 10.0,
    ) -> Mapping[str, Any]:
        normalized = tuple(dict.fromkeys(str(term).strip().casefold() for term in terms if str(term).strip()))
        requested_segment = str(segment_id or "").strip()
        known_segments = {segment.segment_id for segment in self.workspace.manifest.segments}
        if requested_segment and requested_segment not in known_segments:
            raise ValueError(f"unknown ASR segment_id: {requested_segment}")
        segment_scope = requested_segment
        range_scope = _normalized_range(time_range) if time_range is not None and _valid_range(time_range) else None
        grouped: dict[str, list[dict[str, Any]]] = {}
        for cue in self.workspace.read_asr_virtual_cues():
            text = str(cue.get("text", "") or "")
            matched = tuple(term for term in normalized if term in text.casefold())
            if not matched:
                continue
            start = float(cue.get("start_sec", cue.get("start", 0.0)) or 0.0)
            end = float(cue.get("end_sec", cue.get("end", start)) or start)
            cue_segment_id = str(cue.get("segment_id", "") or "") or next(
                (
                    segment.segment_id
                    for segment in self.workspace.manifest.segments
                    if segment.virtual_start_sec <= start < segment.virtual_end_sec
                ),
                "",
            )
            if segment_scope and cue_segment_id != segment_scope:
                continue
            if range_scope and (end < range_scope[0] or start > range_scope[1]):
                continue
            if cue_segment_id:
                grouped.setdefault(cue_segment_id, []).append(
                    {"start": start, "end": end, "terms": matched, "text": text}
                )
        candidates = []
        for group_segment_id, hits in grouped.items():
            segment = self.workspace.manifest.segment(group_segment_id)
            scope_start = max(
                segment.virtual_start_sec,
                range_scope[0] if range_scope else segment.virtual_start_sec,
            )
            scope_end = min(
                segment.virtual_end_sec,
                range_scope[1] if range_scope else segment.virtual_end_sec,
            )
            if scope_end <= scope_start:
                continue
            for cluster in _cluster_asr_hits(hits):
                start, end = _padded_window(
                    float(cluster[0]["start"]),
                    float(cluster[-1]["end"]),
                    segment_start=scope_start,
                    segment_end=scope_end,
                    padding_sec=padding_sec,
                )
                candidates.append(
                    {
                        "segment_id": group_segment_id,
                        "virtual_time_range": [start, end],
                        "matched_terms": [
                            term for term in normalized if any(term in hit["terms"] for hit in cluster)
                        ],
                        "excerpt": " ".join(str(hit["text"]).strip() for hit in cluster)[:800],
                        "source_lineage": [
                            dict(item) for item in _source_lineage(self.workspace, start, end)
                        ],
                        "hit_count": len(cluster),
                    }
                )
        candidates.sort(
            key=lambda row: (
                -len(row["matched_terms"]),
                -int(row["hit_count"]),
                float(row["virtual_time_range"][0]),
            )
        )
        return {
            "terms": list(normalized),
            "scope": {
                "segment_id": segment_scope,
                "time_range": list(range_scope) if range_scope else [],
            },
            "clusters": candidates[: max(1, int(max_clusters))],
        }

    def search_caption(
        self,
        queries: Sequence[str],
        *,
        time_range: tuple[float, float] | None = None,
        segment_ids: Sequence[str] = (),
        source_video_ids: Sequence[str] = (),
        top_k: int = 12,
        expand_neighbors: int = 0,
        index_mode: str = "lexical",
    ) -> Mapping[str, Any]:
        mode = str(index_mode or "lexical").strip().casefold()
        if mode not in {"lexical", "dense", "hybrid"}:
            raise ValueError(f"unsupported caption index mode: {mode}")
        segments = tuple(self.workspace.manifest.segments)
        by_segment_id = {segment.segment_id: segment for segment in segments}
        known_source_ids = {segment.source_video_id for segment in segments}
        requested_segment_ids = tuple(
            dict.fromkeys(str(item).strip() for item in segment_ids if str(item).strip())
        )
        requested_source_ids = tuple(
            dict.fromkeys(str(item).strip() for item in source_video_ids if str(item).strip())
        )
        unknown_segments = tuple(item for item in requested_segment_ids if item not in by_segment_id)
        if unknown_segments:
            raise ValueError(f"unknown Caption segment_ids: {', '.join(unknown_segments)}")
        unknown_sources = tuple(item for item in requested_source_ids if item not in known_source_ids)
        if unknown_sources:
            raise ValueError(f"unknown Caption source_video_ids: {', '.join(unknown_sources)}")
        source_scoped_segments = {
            segment.segment_id
            for segment in segments
            if segment.source_video_id in set(requested_source_ids)
        }
        if requested_segment_ids and requested_source_ids:
            scoped_segment_ids = tuple(
                item for item in requested_segment_ids if item in source_scoped_segments
            )
        elif requested_segment_ids:
            scoped_segment_ids = requested_segment_ids
        elif requested_source_ids:
            scoped_segment_ids = tuple(
                segment.segment_id
                for segment in segments
                if segment.segment_id in source_scoped_segments
            )
        else:
            scoped_segment_ids = ()
        scope_empty = bool(requested_segment_ids and requested_source_ids and not scoped_segment_ids)
        index_segment_ids = ("__no_matching_scope__",) if scope_empty else scoped_segment_ids
        index = self._caption_indexes.get(mode)
        if index is None:
            index = self._load_caption_index(mode)
            self._caption_indexes[mode] = index
        index.save_manifest(self.workspace.asset_root)
        hits = index.search(
            queries,
            top_k=top_k,
            time_range=time_range,
            segment_ids=index_segment_ids,
            expand_neighbors=expand_neighbors,
        )
        fingerprint = index.query_fingerprint(
            queries,
            top_k=top_k,
            time_range=time_range,
            segment_ids=index_segment_ids,
            expand_neighbors=expand_neighbors,
        )
        return {
            "queries": [str(query) for query in queries],
            "time_range": list(time_range) if time_range else None,
            "segment_ids": list(scoped_segment_ids),
            "source_video_ids": list(requested_source_ids),
            "scope_empty": scope_empty,
            "top_k": int(top_k),
            "expand_neighbors": int(expand_neighbors),
            "index_mode": mode,
            "index_digest": index.index_digest,
            "config_digest": index.config_digest,
            "query_fingerprint": fingerprint,
            "hits": [asdict(hit) for hit in hits],
            "rendered": render_caption_hits(hits),
        }

    def _load_caption_index(self, mode: str) -> Any:
        lexical = self._caption_indexes.get("lexical")
        if lexical is None:
            lexical = CaptionLexicalIndex.from_asset_root(
                self.workspace.asset_root,
                config_digest=self._caption_config_digest,
            )
            self._caption_indexes["lexical"] = lexical
        if mode == "lexical":
            return lexical
        adapter = self._caption_embedding_adapter
        if adapter is None:
            raise RuntimeError(
                "caption dense/hybrid search requires a real TextEmbeddingAdapter"
            )
        dense = self._caption_indexes.get("dense")
        if dense is None:
            dense = CaptionSemanticIndex.from_asset_root(
                self.workspace.asset_root,
                adapter=adapter,
                config_digest=lexical.config_digest,
            )
            self._caption_indexes["dense"] = dense
        if mode == "dense":
            return dense
        return CaptionHybridSearch(lexical, dense)

    def _investigate_task(self, task: Any) -> InvestigationReport:
        raise NotImplementedError("Use an observation-only Investigator implementation")

    def _record_visit(
        self,
        task: Any,
        evidence: EvidenceRecord,
        *,
        status: str,
    ) -> None:
        self._visit_count += 1
        row = {
            "visit_id": f"visit_{self._visit_count:04d}",
            "query_id": str(getattr(task, "query_id", "") or ""),
            "attempt_id": evidence_attempt_id(evidence),
            "virtual_time_range": [evidence.start_sec, evidence.end_sec],
            "sampling_fps": evidence.sampling_fps,
            "source_lineage": [dict(item) for item in evidence.source_lineage],
            "status": status,
        }
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _beats_for_segment(
    workspace: VirtualVideoWorkspace,
    segment_id: str,
) -> tuple[Mapping[str, Any], ...]:
    path = workspace.asset_root / "beat_index.json"
    if not path.exists():
        return ()
    return tuple(
        beat
        for beat in load_virtual_beats(path)
        if any(
            str(item.get("segment_id", "")) == segment_id
            for item in tuple(beat.get("source_lineage", ()) or ())
        )
    )


def _cluster_asr_hits(
    hits: Sequence[Mapping[str, Any]],
    *,
    max_gap_sec: float = 20.0,
) -> tuple[tuple[Mapping[str, Any], ...], ...]:
    clusters: list[list[Mapping[str, Any]]] = []
    for hit in sorted(hits, key=lambda row: (float(row["start"]), float(row["end"]))):
        if not clusters or float(hit["start"]) - float(clusters[-1][-1]["end"]) > max_gap_sec:
            clusters.append([hit])
        else:
            clusters[-1].append(hit)
    return tuple(tuple(cluster) for cluster in clusters)


def _padded_window(
    start_sec: float,
    end_sec: float,
    *,
    segment_start: float,
    segment_end: float,
    padding_sec: float,
) -> tuple[float, float]:
    return (
        round(max(segment_start, start_sec - padding_sec), 3),
        round(min(segment_end, end_sec + padding_sec), 3),
    )


def _asr_cues_in_window(
    workspace: VirtualVideoWorkspace,
    start_sec: float,
    end_sec: float,
) -> list[dict[str, Any]]:
    rows = []
    for cue in workspace.read_asr_virtual_cues():
        start = float(cue.get("start_sec", cue.get("start", 0.0)) or 0.0)
        end = float(cue.get("end_sec", cue.get("end", start)) or start)
        if min(end, end_sec) <= max(start, start_sec):
            continue
        rows.append(
            {
                key: value
                for key, value in {
                    **dict(cue),
                    "start_sec": start,
                    "end_sec": end,
                    "text": str(cue.get("text", "") or ""),
                }.items()
                if key not in {"embedding"}
            }
        )
    return rows


def _source_lineage(
    workspace: VirtualVideoWorkspace,
    start_sec: float,
    end_sec: float,
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "segment_id": window.segment_id,
            "source_video_id": window.source_video_id,
            "source_path": window.source_path,
            "source_time_range": [window.source_start_sec, window.source_end_sec],
            "virtual_time_range": [window.virtual_start_sec, window.virtual_end_sec],
        }
        for window in virtual_to_source_windows(workspace.manifest, start_sec, end_sec)
    )


def _beat_asr_excerpt(beat: Mapping[str, Any]) -> str:
    return " ".join(str(cue.get("text", "")).strip() for cue in beat.get("asr_cues", ()) or ())[:240]


def _frame_payload(frame: VirtualFrameRef) -> dict[str, Any]:
    return {
        "path": frame.path,
        "virtual_time_sec": frame.virtual_time_sec,
        "source_video_id": frame.source_video_id,
        "source_path": frame.source_path,
        "source_time_sec": frame.source_time_sec,
        "segment_id": frame.segment_id,
    }


def _attempt(value: ObservationAttempt | Mapping[str, Any]) -> ObservationAttempt:
    if isinstance(value, ObservationAttempt):
        return value
    return ObservationAttempt(**dict(value))


def _valid_range(value: Sequence[float]) -> bool:
    try:
        return len(value) == 2 and float(value[0]) != float(value[1])
    except (TypeError, ValueError):
        return False


def _normalized_range(value: Sequence[float]) -> tuple[float, float]:
    start, end = float(value[0]), float(value[1])
    return (start, end) if start < end else (end, start)
