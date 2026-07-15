from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Mapping, Sequence

from vcah.evidence_primitives import ConditionResult
from vcah.types import CoverageSegment, EvidenceRecord
from vcah.virtual_index import load_virtual_beats
from vcah.virtual_video import (
    FrameSampler,
    VirtualFrameRef,
    VirtualVideoWorkspace,
    materialize_window_frames,
    virtual_to_source_windows,
)


HIGHFPS_KEYWORDS = {
    "ocr",
    "number",
    "text",
    "read",
    "written",
    "action",
    "motion",
    "transition",
    "change",
    "overtake",
    "track",
    "throw",
    "threw",
    "spatial",
    "position",
    "above",
    "below",
    "left",
    "right",
}


@dataclass(frozen=True)
class InvestigationReport:
    query_id: str
    status: str
    evidence: tuple[EvidenceRecord, ...] = ()
    cost: Mapping[str, Any] = field(default_factory=dict)
    gap_id: str = ""
    resolution: str = ""
    resolved_conditions: tuple[str, ...] = ()
    unresolved_conditions: tuple[str, ...] = ()
    failure_reason: str = ""
    progress_flags: tuple[str, ...] = ()
    coverage_delta: tuple[tuple[float, float], ...] = ()
    condition_results: tuple[ConditionResult, ...] = ()
    goal_progress: tuple[str, ...] = ()
    coverage_progress: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", str(self.query_id or "").strip())
        object.__setattr__(self, "status", str(self.status or "").strip().casefold())
        object.__setattr__(self, "gap_id", str(self.gap_id or "").strip())
        resolution = str(self.resolution or "").strip().casefold()
        if resolution not in {"resolved", "partial", "unresolved"}:
            resolution = "resolved" if self.status == "satisfied" and self.evidence else "unresolved"
        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(
            self,
            "resolved_conditions",
            tuple(str(item).strip() for item in self.resolved_conditions if str(item).strip()),
        )
        object.__setattr__(
            self,
            "unresolved_conditions",
            tuple(str(item).strip() for item in self.unresolved_conditions if str(item).strip()),
        )
        object.__setattr__(self, "failure_reason", str(self.failure_reason or "").strip())
        object.__setattr__(
            self,
            "progress_flags",
            tuple(dict.fromkeys(str(item).strip() for item in self.progress_flags if str(item).strip())),
        )
        object.__setattr__(
            self,
            "coverage_delta",
            tuple(_normalized_range(item) for item in self.coverage_delta if _valid_range(item)),
        )
        object.__setattr__(
            self,
            "condition_results",
            tuple(_condition_result(item) for item in self.condition_results),
        )
        object.__setattr__(
            self,
            "goal_progress",
            tuple(dict.fromkeys(str(item).strip() for item in self.goal_progress if str(item).strip())),
        )
        object.__setattr__(
            self,
            "coverage_progress",
            tuple(dict.fromkeys(str(item).strip() for item in self.coverage_progress if str(item).strip())),
        )


@dataclass(frozen=True)
class _CachedObservation:
    goal_fingerprint: str
    source_lineage: tuple[Mapping[str, Any], ...]
    evidence: EvidenceRecord


class VirtualVideoInvestigator:
    tool_names = ("open_segment", "inspect_window")

    def __init__(
        self,
        workspace: VirtualVideoWorkspace,
        *,
        sampler: FrameSampler | None = None,
        highfps: float = 2.0,
        highfps_max_frames: int = 64,
    ) -> None:
        self.workspace = workspace
        self.sampler = sampler
        self.highfps = float(highfps)
        self.highfps_max_frames = min(512, int(highfps_max_frames))
        self.ledger_path = self.workspace.root_dir / "exploration_ledger.jsonl"
        self._visit_count = 0
        self._observation_cache: list[_CachedObservation] = []

    def run_batch(self, tasks: Sequence[Any]) -> tuple[InvestigationReport, ...]:
        return tuple(self._investigate_task(task) for task in tasks)

    def reset_run_state(self) -> None:
        self._visit_count = 0
        self._observation_cache.clear()
        self.ledger_path.write_text("", encoding="utf-8")

    def _find_reusable_evidence(
        self,
        task: Any,
        start_sec: float,
        end_sec: float,
        *,
        required_fps: float,
    ) -> EvidenceRecord | None:
        goal = _goal_fingerprint(task)
        lineage = _source_lineage(self.workspace, start_sec, end_sec)
        for cached in reversed(self._observation_cache):
            if cached.goal_fingerprint != goal:
                continue
            if cached.evidence.sampling_fps + 1e-6 < float(required_fps):
                continue
            if _lineage_iou(cached.source_lineage, lineage) >= 0.8:
                return cached.evidence
        return None

    def _remember_evidence(self, task: Any, evidence: EvidenceRecord) -> None:
        self._observation_cache.append(
            _CachedObservation(
                goal_fingerprint=_goal_fingerprint(task),
                source_lineage=tuple(dict(item) for item in evidence.source_lineage),
                evidence=evidence,
            )
        )

    def _record_visit(
        self,
        task: Any,
        evidence: EvidenceRecord,
        *,
        status: str,
        reused_from: str = "",
    ) -> None:
        self._visit_count += 1
        row = {
            "visit_id": f"visit_{self._visit_count:04d}",
            "query_id": str(getattr(task, "query_id", "") or ""),
            "observation_id": evidence.observation_id,
            "goal_fingerprint": _goal_fingerprint(task),
            "virtual_time_range": [evidence.start_sec, evidence.end_sec],
            "sampling_fps": evidence.sampling_fps,
            "source_lineage": [dict(item) for item in evidence.source_lineage],
            "evidence_ids": [evidence.evidence_id],
            "status": status,
            "reused_from": str(reused_from or ""),
        }
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def _reuse_report(
        self,
        task: Any,
        evidence: EvidenceRecord,
        *,
        tool_trace: Sequence[str],
        vlm_calls: int = 0,
    ) -> InvestigationReport:
        self._record_visit(task, evidence, status="reused", reused_from=evidence.evidence_id)
        conditions = tuple(getattr(task, "conditions", ()) or ())
        condition_results = tuple(
            ConditionResult(item.condition_id, "unknown", "Cached observation requires a new semantic assessment.")
            for item in conditions
        )
        return InvestigationReport(
            query_id=str(getattr(task, "query_id", "") or ""),
            status="satisfied",
            evidence=(evidence,),
            cost={
                "tool_trace": tuple(tool_trace),
                "frames": 0,
                "vlm_calls": int(vlm_calls),
                "reused": True,
            },
            gap_id=str(getattr(task, "gap_id", "") or ""),
            resolution="partial" if getattr(task, "success_conditions", ()) else "resolved",
            unresolved_conditions=tuple(getattr(task, "success_conditions", ()) or ()),
            progress_flags=("observation_reused",),
            coverage_delta=(),
            condition_results=condition_results,
        )

    def open_segment(self, segment_id: str) -> Mapping[str, Any]:
        segment = self.workspace.manifest.segment(str(segment_id))
        beats = _beats_for_segment(self.workspace, segment.segment_id)
        asr_cues = _asr_cues_in_window(self.workspace, segment.virtual_start_sec, segment.virtual_end_sec)
        return {
            "segment_id": segment.segment_id,
            "virtual_time_range": [segment.virtual_start_sec, segment.virtual_end_sec],
            "duration_sec": segment.duration_sec,
            "asr_timeline_summary": " ".join(cue["text"] for cue in asr_cues)[:500],
            "asr_cues": asr_cues,
            "source_lineage": _source_lineage(self.workspace, segment.virtual_start_sec, segment.virtual_end_sec),
            "beats": [
                {
                    "beat_id": str(beat.get("beat_id")),
                    "virtual_time_range": list(beat.get("virtual_time_range", ())),
                    "asr_excerpt": _beat_asr_excerpt(beat),
                    "thumbnail_grid_paths": list(beat.get("thumbnail_grid_paths") or [beat.get("thumbnail_grid_path", "")]),
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
        max_clusters: int = 8,
        padding_sec: float = 10.0,
    ) -> Mapping[str, Any]:
        normalized_terms = tuple(
            dict.fromkeys(str(term).strip().casefold() for term in terms if str(term).strip())
        )
        grouped_hits: dict[str, list[dict[str, Any]]] = {}
        for cue in self.workspace.read_asr_virtual_cues():
            text = str(cue.get("text", "") or "")
            folded = text.casefold()
            matched = tuple(term for term in normalized_terms if term in folded)
            if not matched:
                continue
            start = float(cue.get("start_sec", cue.get("start", 0.0)) or 0.0)
            end = float(cue.get("end_sec", cue.get("end", start)) or start)
            segment_id = str(cue.get("segment_id", "") or "")
            if not segment_id:
                segment_id = next(
                    (
                        segment.segment_id
                        for segment in self.workspace.manifest.segments
                        if segment.virtual_start_sec <= start < segment.virtual_end_sec
                    ),
                    "",
                )
            if segment_id:
                grouped_hits.setdefault(segment_id, []).append(
                    {"start": start, "end": end, "terms": matched, "text": text}
                )

        candidates = []
        for segment_id, hits in grouped_hits.items():
            segment = self.workspace.manifest.segment(segment_id)
            for cluster in _cluster_asr_hits(hits):
                start, end = _padded_window(
                    float(cluster[0]["start"]),
                    float(cluster[-1]["end"]),
                    segment_start=segment.virtual_start_sec,
                    segment_end=segment.virtual_end_sec,
                    padding_sec=float(padding_sec),
                )
                matched_terms = tuple(
                    term for term in normalized_terms if any(term in hit["terms"] for hit in cluster)
                )
                candidates.append(
                    {
                        "segment_id": segment_id,
                        "virtual_time_range": [start, end],
                        "matched_terms": list(matched_terms),
                        "excerpt": " ".join(str(hit["text"]).strip() for hit in cluster)[:800],
                        "source_lineage": [dict(item) for item in _source_lineage(self.workspace, start, end)],
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
            "terms": list(normalized_terms),
            "clusters": candidates[: max(1, int(max_clusters))],
        }

    def _investigate_task(self, task: Any) -> InvestigationReport:
        segment_id = str(getattr(task, "segment_id", "") or "")
        tool_trace: list[str] = []
        segment_packet: Mapping[str, Any] | None = None
        if segment_id:
            segment_packet = self.open_segment(segment_id)
            tool_trace.append("open_segment")
        if getattr(task, "time_range", None) is None and segment_packet is not None:
            start_sec, end_sec = _choose_window_from_segment_packet(task, segment_packet)
        else:
            start_sec, end_sec = _task_time_range(task)

        required_fps = self.highfps if _needs_highfps(task) else 0.5
        cached = self._find_reusable_evidence(task, start_sec, end_sec, required_fps=required_fps)
        if cached is not None:
            return self._reuse_report(task, cached, tool_trace=(*tool_trace, "reuse_observation"))

        window = self.inspect_window(
            start_sec,
            end_sec,
            fps=0.5,
            max_frames=self.highfps_max_frames,
            query_id=f"{getattr(task, 'query_id')}_preview",
        )
        tool_trace.append("inspect_window:0.5")
        if _needs_highfps(task):
            window = self.inspect_window(
                start_sec,
                end_sec,
                fps=self.highfps,
                max_frames=self.highfps_max_frames,
                query_id=str(getattr(task, "query_id")),
            )
            tool_trace.append(f"inspect_window:{self.highfps:.1f}")
        frame_paths = tuple(str(frame["path"]) for frame in window["frames"])
        query_id = str(getattr(task, "query_id", "") or "")
        evidence = EvidenceRecord(
            evidence_id=f"ev_{getattr(task, 'query_id')}_001",
            beat_id="",
            start_sec=float(start_sec),
            end_sec=float(end_sec),
            modality="visual",
            pointer=f"virtual://{self.workspace.workspace_id}/observations/{query_id}",
            verbatim=_summary_for_task(task, window=window),
            frame_refs=frame_paths,
            attestation_model="deterministic-window-inspector",
            temporal_scope="window",
            evidence_kind="visual_observation",
            observation_polarity="positive" if frame_paths else "unknown",
            sampling_coverage="sparse",
            request_ids=(query_id,),
            coverage_manifest=(CoverageSegment(query_id, float(start_sec), float(end_sec), "visual", 1.0),),
            task_id=query_id,
            observation_id=query_id,
            sampling_fps=float(window["sampling"]["fps"]),
            confidence=0.7 if frame_paths else 0.0,
            source_lineage=tuple(dict(item) for item in window["source_lineage"]),
        )
        if frame_paths:
            self._remember_evidence(task, evidence)
            self._record_visit(task, evidence, status="satisfied")
        condition_results = tuple(
            ConditionResult(
                item.condition_id,
                "unknown",
                "Deterministic frame sampling cannot attest semantic success.",
            )
            for item in tuple(getattr(task, "conditions", ()) or ())
        )
        return InvestigationReport(
            query_id=query_id,
            status="satisfied" if frame_paths else "empty",
            evidence=(evidence,) if frame_paths else (),
            cost={
                "tool_steps": len(tool_trace),
                "tool_trace": tuple(tool_trace),
                "frames": len(frame_paths),
                "vlm_calls": 1 if frame_paths else 0,
                "reused": False,
            },
            gap_id=str(getattr(task, "gap_id", "") or ""),
            resolution=(
                "partial"
                if frame_paths and tuple(getattr(task, "success_conditions", ()) or ())
                else "resolved"
                if frame_paths
                else "unresolved"
            ),
            unresolved_conditions=tuple(getattr(task, "success_conditions", ()) or ()) if frame_paths else (),
            failure_reason="deterministic inspector cannot attest semantic success conditions" if frame_paths and getattr(task, "success_conditions", ()) else "",
            progress_flags=(),
            coverage_delta=((float(start_sec), float(end_sec)),) if frame_paths else (),
            condition_results=condition_results,
        )


def _task_time_range(task: Any) -> tuple[float, float]:
    time_range = getattr(task, "time_range", None)
    if time_range is None:
        raise ValueError("InvestigationTask requires either time_range or segment_id")
    start, end = time_range
    return float(start), float(end)


def _beats_for_segment(workspace: VirtualVideoWorkspace, segment_id: str) -> tuple[Mapping[str, Any], ...]:
    path = workspace.root_dir / "beat_index.json"
    if not path.exists():
        return ()
    rows = []
    for beat in load_virtual_beats(path):
        lineage = tuple(beat.get("source_lineage", ()) or ())
        if any(str(item.get("segment_id")) == segment_id for item in lineage):
            rows.append(beat)
    return tuple(rows)


def _choose_window_from_segment_packet(task: Any, segment_packet: Mapping[str, Any]) -> tuple[float, float]:
    segment_start, segment_end = tuple(segment_packet.get("virtual_time_range", (0.0, 0.0)))
    segment_start = float(segment_start)
    segment_end = float(segment_end)
    terms = _task_terms(task)
    hits: list[dict[str, Any]] = []
    for cue in segment_packet.get("asr_cues", ()) or ():
        text = str(cue.get("text", "") or "").casefold()
        matched_terms = tuple(term for term in terms if term in text)
        if terms and not matched_terms:
            continue
        start = max(segment_start, float(cue.get("start_sec", segment_start)))
        end = min(segment_end, float(cue.get("end_sec", start)))
        if end > start:
            hits.append({"start": start, "end": end, "terms": matched_terms})
    if hits:
        clusters = _cluster_asr_hits(hits)
        best = max(
            clusters,
            key=lambda cluster: (
                len({term for hit in cluster for term in hit["terms"]}),
                len(cluster),
                -float(cluster[0]["start"]),
            ),
        )
        return _padded_window(
            float(best[0]["start"]),
            float(best[-1]["end"]),
            segment_start=segment_start,
            segment_end=segment_end,
            padding_sec=2.0,
        )

    beats = tuple(segment_packet.get("beats", ()) or ())
    if beats:
        scored = []
        for index, beat in enumerate(beats):
            text = str(beat.get("asr_excerpt", "") or "").casefold()
            score = len({term for term in terms if term in text})
            scored.append((score, -abs(index - (len(beats) - 1) / 2.0), index, beat))
        _, _, _, beat = max(scored)
        window = tuple(beat.get("virtual_time_range", (segment_start, segment_end)))
        if len(window) == 2 and float(window[1]) > float(window[0]):
            return max(segment_start, float(window[0])), min(segment_end, float(window[1]))
    return segment_start, segment_end


def _cluster_asr_hits(hits: Sequence[Mapping[str, Any]], *, max_gap_sec: float = 20.0) -> tuple[tuple[Mapping[str, Any], ...], ...]:
    ordered = sorted(hits, key=lambda hit: (float(hit["start"]), float(hit["end"])))
    clusters: list[list[Mapping[str, Any]]] = []
    for hit in ordered:
        if not clusters or float(hit["start"]) - float(clusters[-1][-1]["end"]) > float(max_gap_sec):
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
    start = max(float(segment_start), float(start_sec) - float(padding_sec))
    end = min(float(segment_end), float(end_sec) + float(padding_sec))
    return round(start, 3), round(end, 3)


def _task_terms(task: Any) -> tuple[str, ...]:
    text = " ".join(
        [
            str(getattr(task, "goal", "") or ""),
            str(getattr(task, "expected_evidence", "") or ""),
            " ".join(str(item) for item in getattr(task, "modality_hint", ()) or ()),
            " ".join(str(item) for item in getattr(task, "success_conditions", ()) or ()),
        ]
    ).casefold()
    stop = {"the", "and", "for", "with", "that", "this", "visual", "verify", "read"}
    return tuple(term for term in re.findall(r"[a-z0-9]+", text) if len(term) >= 3 and term not in stop)


def _asr_cues_in_window(workspace: VirtualVideoWorkspace, start_sec: float, end_sec: float) -> list[dict[str, Any]]:
    rows = []
    for cue in workspace.read_asr_virtual_cues():
        start = float(cue.get("start_sec", cue.get("start", 0.0)) or 0.0)
        end = float(cue.get("end_sec", cue.get("end", start)) or start)
        if min(end, end_sec) <= max(start, start_sec):
            continue
        row = {
            "start_sec": start,
            "end_sec": end,
            "text": str(cue.get("text", "") or ""),
        }
        for key in ("segment_id", "source_video_id", "source_start_sec", "source_end_sec", "source_start", "source_end"):
            if key in cue:
                row[key] = cue[key]
        rows.append(row)
    return rows


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


def _needs_highfps(task: Any) -> bool:
    text = " ".join(
        [
            " ".join(str(item) for item in getattr(task, "modality_hint", ()) or ()),
            str(getattr(task, "expected_evidence", "") or ""),
            str(getattr(task, "goal", "") or ""),
            " ".join(str(item) for item in getattr(task, "success_conditions", ()) or ()),
            str(getattr(task, "region_hint", "") or ""),
        ]
    ).casefold()
    return any(keyword in text for keyword in HIGHFPS_KEYWORDS)


def _summary_for_task(task: Any, *, window: Mapping[str, Any]) -> str:
    frames = tuple(window.get("frames", ()) or ())
    if not frames:
        return ""
    times = [float(frame["virtual_time_sec"]) for frame in frames]
    return (
        f"inspect_window at {window['sampling']['fps']} fps for {getattr(task, 'query_id')} covers "
        f"{min(times):.1f}-{max(times):.1f}s while pursuing: "
        f"{getattr(task, 'goal', '')}"
    )


def _goal_fingerprint(task: Any) -> str:
    text = " ".join(
        [
            str(getattr(task, "goal", "") or ""),
            str(getattr(task, "expected_evidence", "") or ""),
            " ".join(sorted(str(item) for item in getattr(task, "modality_hint", ()) or ())),
            " ".join(str(item) for item in getattr(task, "success_conditions", ()) or ()),
            str(getattr(task, "region_hint", "") or ""),
        ]
    ).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _valid_range(value: Sequence[float]) -> bool:
    try:
        return len(value) == 2 and float(value[0]) != float(value[1])
    except (TypeError, ValueError):
        return False


def _normalized_range(value: Sequence[float]) -> tuple[float, float]:
    start, end = float(value[0]), float(value[1])
    return (start, end) if start < end else (end, start)


def _condition_result(value: ConditionResult | Mapping[str, Any]) -> ConditionResult:
    if isinstance(value, ConditionResult):
        return value
    return ConditionResult(
        condition_id=str(value.get("condition_id", "") or ""),
        status=str(value.get("status", "unknown") or "unknown"),
        observation=str(value.get("observation", "") or ""),
        evidence_ids=tuple(value.get("evidence_ids", ()) or ()),
    )


def _lineage_iou(first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]]) -> float:
    first_ranges = _lineage_ranges(first)
    second_ranges = _lineage_ranges(second)
    keys = set(first_ranges) | set(second_ranges)
    intersection = 0.0
    union = 0.0
    for key in keys:
        left = first_ranges.get(key)
        right = second_ranges.get(key)
        if left is None:
            union += max(0.0, right[1] - right[0]) if right is not None else 0.0
            continue
        if right is None:
            union += max(0.0, left[1] - left[0])
            continue
        intersection += max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
        union += max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union > 0.0 else 0.0


def _lineage_ranges(lineage: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], tuple[float, float]]:
    ranges: dict[tuple[str, str], tuple[float, float]] = {}
    for item in lineage:
        values = tuple(item.get("source_time_range", ()) or ())
        if len(values) != 2:
            continue
        key = (str(item.get("segment_id", "")), str(item.get("source_video_id", "")))
        ranges[key] = float(values[0]), float(values[1])
    return ranges


def _source_lineage(workspace: VirtualVideoWorkspace, start_sec: float, end_sec: float) -> tuple[Mapping[str, Any], ...]:
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
