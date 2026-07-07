from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

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
class InvestigationEvidence:
    evidence_id: str
    summary: str
    modality: str
    sampling: Mapping[str, Any]
    virtual_time_range: tuple[float, float]
    source_lineage: tuple[Mapping[str, Any], ...]
    supporting_frames: tuple[str, ...] = ()
    confidence: float = 0.0


@dataclass(frozen=True)
class InvestigationReport:
    query_id: str
    status: str
    evidence: tuple[InvestigationEvidence, ...] = ()
    cost: Mapping[str, Any] = field(default_factory=dict)


class VirtualVideoInvestigator:
    tool_names = ("open_segment", "inspect_window")

    def __init__(
        self,
        workspace: VirtualVideoWorkspace,
        *,
        sampler: FrameSampler | None = None,
        highfps: float = 2.0,
        highfps_max_frames: int = 32,
    ) -> None:
        self.workspace = workspace
        self.sampler = sampler
        self.highfps = float(highfps)
        self.highfps_max_frames = min(64, int(highfps_max_frames))

    def run_batch(self, tasks: Sequence[Any]) -> tuple[InvestigationReport, ...]:
        return tuple(self._investigate_task(task) for task in tasks)

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
    ) -> Mapping[str, Any]:
        capped = min(64, max(1, int(max_frames)))
        frames = materialize_window_frames(
            self.workspace,
            float(start_sec),
            float(end_sec),
            query_id=str(query_id),
            fps=float(fps),
            max_frames=capped,
            sampler=self.sampler,
        )
        return {
            "virtual_time_range": [float(start_sec), float(end_sec)],
            "sampling": {
                "fps": float(fps),
                "max_frames": capped,
                "actual_frames": len(frames),
                "sampling": "uniform",
            },
            "frames": [_frame_payload(frame) for frame in frames],
            "asr_cues": _asr_cues_in_window(self.workspace, float(start_sec), float(end_sec)),
            "source_lineage": _source_lineage(self.workspace, float(start_sec), float(end_sec)),
        }

    def _investigate_task(self, task: Any) -> InvestigationReport:
        segment_id = str(getattr(task, "segment_id", "") or "")
        tool_steps = 0
        segment_packet: Mapping[str, Any] | None = None
        if segment_id:
            segment_packet = self.open_segment(segment_id)
            tool_steps += 1
        if getattr(task, "time_range", None) is None and segment_packet is not None:
            start_sec, end_sec = tuple(segment_packet["virtual_time_range"])  # type: ignore[assignment]
            start_sec = float(start_sec)
            end_sec = float(end_sec)
        else:
            start_sec, end_sec = _task_time_range(task)
        fps = self.highfps if _needs_highfps(task) else 0.5
        window = self.inspect_window(
            start_sec,
            end_sec,
            fps=fps,
            max_frames=self.highfps_max_frames,
            query_id=str(getattr(task, "query_id")),
        )
        tool_steps += 1
        frame_paths = tuple(str(frame["path"]) for frame in window["frames"])
        evidence = InvestigationEvidence(
            evidence_id=f"ev_{getattr(task, 'query_id')}_001",
            summary=_summary_for_task(task, window=window),
            modality="visual",
            sampling=dict(window["sampling"]),
            virtual_time_range=(float(start_sec), float(end_sec)),
            source_lineage=tuple(dict(item) for item in window["source_lineage"]),
            supporting_frames=frame_paths,
            confidence=0.7 if frame_paths else 0.0,
        )
        return InvestigationReport(
            query_id=str(getattr(task, "query_id")),
            status="satisfied" if frame_paths else "empty",
            evidence=(evidence,) if frame_paths else (),
            cost={
                "tool_steps": tool_steps,
                "frames": len(frame_paths),
                "vlm_calls": 1 if frame_paths else 0,
            },
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
