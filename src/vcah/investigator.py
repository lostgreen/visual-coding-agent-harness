from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from vcah.types import Frame
from vcah.virtual_index import load_virtual_beats
from vcah.virtual_video import (
    FrameSampler,
    VirtualFrameRef,
    VirtualVideoWorkspace,
    materialize_highfps_window,
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
        self.highfps_max_frames = int(highfps_max_frames)

    def run_batch(self, tasks: Sequence[Any]) -> tuple[InvestigationReport, ...]:
        return tuple(self.inspect_window_auto(task) for task in tasks)

    def open_beat_grid(self, beat_ids: Sequence[str]) -> tuple[str, ...]:
        path = self.workspace.root_dir / "beat_index.json"
        if not path.exists():
            return ()
        beats = load_virtual_beats(path)
        wanted = {str(beat_id) for beat_id in beat_ids}
        return tuple(str(beat["thumbnail_grid_path"]) for beat in beats if str(beat.get("beat_id")) in wanted)

    def inspect_window_lowfps(self, start_sec: float, end_sec: float, *, max_frames: int = 12) -> tuple[VirtualFrameRef, ...]:
        frames = tuple(
            frame
            for frame in self.workspace.read_frame_manifest()
            if start_sec <= frame.virtual_time_sec <= end_sec and frame.fps_level == "low"
        )
        return frames[: max(0, int(max_frames))]

    def inspect_window_highfps(self, task: Any) -> tuple[VirtualFrameRef, ...]:
        start_sec, end_sec = _task_time_range(task)
        return materialize_highfps_window(
            self.workspace,
            start_sec,
            end_sec,
            query_id=str(getattr(task, "query_id")),
            fps=self.highfps,
            max_frames=self.highfps_max_frames,
            sampler=self.sampler,
        )

    def inspect_window_auto(self, task: Any) -> InvestigationReport:
        start_sec, end_sec = _task_time_range(task)
        low = self.inspect_window_lowfps(start_sec, end_sec)
        needs_high = _needs_highfps(task)
        high: tuple[VirtualFrameRef, ...] = ()
        if needs_high:
            high = self.inspect_window_highfps(task)
        frames = high or low
        level = "highfps" if high else "lowfps"
        evidence = InvestigationEvidence(
            evidence_id=f"ev_{getattr(task, 'query_id')}_001",
            summary=_summary_for_task(task, level=level, frames=frames),
            modality="visual",
            sampling={
                "level": level,
                "fps": self.highfps if high else 0.0,
                "frame_count": len(frames),
            },
            virtual_time_range=(float(start_sec), float(end_sec)),
            source_lineage=_source_lineage(self.workspace, start_sec, end_sec),
            supporting_frames=tuple(frame.path for frame in frames),
            confidence=0.7 if frames else 0.0,
        )
        return InvestigationReport(
            query_id=str(getattr(task, "query_id")),
            status="satisfied" if frames else "empty",
            evidence=(evidence,) if frames else (),
            cost={
                "lowfps_frames": len(low),
                "highfps_frames": len(high),
                "vlm_calls": 1 if frames else 0,
            },
        )


def _task_time_range(task: Any) -> tuple[float, float]:
    start, end = getattr(task, "time_range")
    return float(start), float(end)


def _needs_highfps(task: Any) -> bool:
    text = " ".join(
        [
            " ".join(str(item) for item in getattr(task, "modality_hint", ()) or ()),
            str(getattr(task, "expected_evidence", "") or ""),
            str(getattr(task, "goal", "") or ""),
        ]
    ).casefold()
    return any(keyword in text for keyword in HIGHFPS_KEYWORDS)


def _summary_for_task(task: Any, *, level: str, frames: Sequence[VirtualFrameRef]) -> str:
    if not frames:
        return ""
    return (
        f"{level} frames for {getattr(task, 'query_id')} cover "
        f"{frames[0].virtual_time_sec:.1f}-{frames[-1].virtual_time_sec:.1f}s while pursuing: "
        f"{getattr(task, 'goal', '')}"
    )


def _source_lineage(workspace: VirtualVideoWorkspace, start_sec: float, end_sec: float) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "segment_id": window.segment_id,
            "source_video_id": window.source_video_id,
            "source_path": window.source_path,
            "source_time_range": (window.source_start_sec, window.source_end_sec),
            "virtual_time_range": (window.virtual_start_sec, window.virtual_end_sec),
        }
        for window in virtual_to_source_windows(workspace.manifest, start_sec, end_sec)
    )
