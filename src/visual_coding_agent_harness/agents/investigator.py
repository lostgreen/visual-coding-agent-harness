"""Investigator execution for one multi_v3 ScopedQuery."""

from __future__ import annotations

from typing import Callable, Sequence

from visual_coding_agent_harness.backends.base import VisionLanguageBackend
from visual_coding_agent_harness.contracts.query import ScopedQuery
from visual_coding_agent_harness.contracts.report import CandidateShot, Finding, InvestigationReport, VerifyRequest
from visual_coding_agent_harness.tools.explore import explore
from visual_coding_agent_harness.tools.verify import verify_window
from visual_coding_agent_harness.video.index import Shot, VideoIndex
from visual_coding_agent_harness.workspace.investigator_ws import InvestigatorWorkspace


ExploreFn = Callable[..., Sequence[CandidateShot]]
VerifyFn = Callable[..., Sequence[Finding]]
FrameSamplerFn = Callable[[Shot, int], Sequence[str]]


class Investigator:
    def __init__(
        self,
        *,
        index: VideoIndex,
        workspace: InvestigatorWorkspace,
        backend: VisionLanguageBackend,
        explore_fn: ExploreFn = explore,
        verify_fn: VerifyFn = verify_window,
        frame_sampler: FrameSamplerFn | None = None,
    ) -> None:
        self.index = index
        self.workspace = workspace
        self.backend = backend
        self.explore_fn = explore_fn
        self.verify_fn = verify_fn
        self.frame_sampler = frame_sampler or _default_frame_sampler

    def run(self, query: ScopedQuery) -> InvestigationReport:
        self.workspace.record_request(query)
        explore_result = self.explore_fn(query=query, index=self.index, backend=self.backend)
        candidates = tuple(explore_result)
        explore_calls = int(getattr(explore_result, "batch_count", 1 if candidates else 0) or 0)
        self.workspace.record_explore(query.query_id, candidates)
        findings: list[Finding] = []
        verified_shots: list[str] = []
        frames_read = 0
        for candidate in candidates:
            shot = self.index.get_shot(candidate.shot_id)
            frame_paths = tuple(self.frame_sampler(shot, query.budget.max_frames))
            frames_read += len(frame_paths)
            request = VerifyRequest(
                shot_id=shot.shot_id,
                time_range=(shot.start_sec, shot.end_sec),
                focus_claim=query.expected_evidence,
                sampling={"fps": 2, "max_frames": query.budget.max_frames, "resolution": "high"},
                checks=({"target_id": query.goal_id, "claim": query.expected_evidence, "polarity": "presence"},),
            )
            shot_findings = tuple(
                self.verify_fn(query_id=query.query_id, request=request, frame_paths=frame_paths, backend=self.backend)
            )
            self.workspace.record_verify(query.query_id, shot.shot_id, shot_findings)
            if shot_findings:
                verified_shots.append(shot.shot_id)
                findings.extend(shot_findings)
        report = InvestigationReport(
            query_id=query.query_id,
            status="satisfied" if findings else ("empty" if not candidates else "partial"),
            findings=tuple(findings),
            explored_shots=tuple(candidate.shot_id for candidate in candidates),
            verified_shots=tuple(verified_shots),
            unresolved=() if findings else (query.expected_evidence,),
            cost={"explore_calls": explore_calls, "verify_calls": len(candidates), "frames_read": frames_read},
        )
        self.workspace.record_report(report)
        return report


def _default_frame_sampler(shot: Shot, max_frames: int) -> tuple[str, ...]:
    paths = tuple(frame.thumb_path for frame in shot.frames if frame.thumb_path)
    return paths[:max_frames] if max_frames else ()
