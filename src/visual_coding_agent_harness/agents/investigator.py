"""Investigator execution for one multi_v3 ScopedQuery."""

from __future__ import annotations

from typing import Callable, Sequence

from visual_coding_agent_harness.agents.playbook_programs import PROGRAMS, PlaybookProgram
from visual_coding_agent_harness.backends.base import VisionLanguageBackend
from visual_coding_agent_harness.contracts.query import ScopedQuery
from visual_coding_agent_harness.contracts.report import CandidateShot, Finding, InvestigationReport, VerifyRequest
from visual_coding_agent_harness.tools.vlm_tools import explore
from visual_coding_agent_harness.tools.vlm_tools import explore_via_search
from visual_coding_agent_harness.tools.vlm_tools import verify_window
from visual_coding_agent_harness.workspace.investigator_ws import InvestigatorWorkspace
from visual_coding_agent_harness.workspace.video_workspace import Beat
from visual_coding_agent_harness.workspace.video_workspace import VideoWorkspace


ExploreFn = Callable[..., Sequence[CandidateShot]]
VerifyFn = Callable[..., Sequence[Finding]]
FrameSamplerFn = Callable[[Beat, int], Sequence[str]]


class Investigator:
    def __init__(
        self,
        *,
        index=None,
        workspace: InvestigatorWorkspace,
        backend: VisionLanguageBackend,
        explore_fn: ExploreFn = explore,
        verify_fn: VerifyFn = verify_window,
        frame_sampler: FrameSamplerFn | None = None,
        video_workspace: VideoWorkspace | None = None,
        use_search: bool = False,
        search_explore_fn: ExploreFn = explore_via_search,
        programs: dict | None = None,
    ) -> None:
        self.index = index
        self.workspace = workspace
        self.backend = backend
        self.explore_fn = explore_fn
        self.verify_fn = verify_fn
        self.frame_sampler = frame_sampler or _default_frame_sampler
        self.video_workspace = video_workspace
        self.use_search = bool(use_search)
        self.search_explore_fn = search_explore_fn
        self.programs = PROGRAMS if programs is None else programs

    def run(self, query: ScopedQuery) -> InvestigationReport:
        self.workspace.record_request(query)
        if self.video_workspace is not None and query.playbook in self.programs:
            program = self.programs[query.playbook]
            explore_result = program.execute(
                query=query,
                workspace=self.video_workspace,
                backend=self.backend,
                frame_sampler=lambda beat, max_frames: (beat.keyframe_path,)[:max_frames] if beat.keyframe_path else (),
                verify_fn=self.verify_fn,
            )
            self.workspace.record_report(explore_result)
            return explore_result
        if self.use_search and self.video_workspace is not None:
            explore_result = self.search_explore_fn(workspace=self.video_workspace, query=query, backend=self.backend)
        else:
            if self.video_workspace is None:
                raise ValueError("video_workspace is required for Beat-based investigation")
            explore_result = self.explore_fn(query=query, workspace=self.video_workspace, backend=self.backend)
        candidates = tuple(explore_result)
        explore_calls = int(getattr(explore_result, "batch_count", 1 if candidates else 0) or 0)
        self.workspace.record_explore(query.query_id, candidates)
        findings: list[Finding] = []
        verified_shots: list[str] = []
        frames_read = 0
        for candidate in candidates:
            beat = self._get_beat_for_candidate(candidate.shot_id)
            frame_paths = tuple(self.frame_sampler(beat, query.budget.max_frames))
            frames_read += len(frame_paths)
            request = VerifyRequest(
                shot_id=candidate.shot_id,
                time_range=(beat.start_sec, beat.end_sec),
                focus_claim=query.expected_evidence,
                sampling={"fps": 2, "max_frames": query.budget.max_frames, "resolution": "high"},
                checks=({"target_id": query.goal_id, "claim": query.expected_evidence, "polarity": "presence"},),
            )
            shot_findings = tuple(
                self.verify_fn(query_id=query.query_id, request=request, frame_paths=frame_paths, backend=self.backend)
            )
            self.workspace.record_verify(query.query_id, candidate.shot_id, shot_findings)
            if shot_findings:
                verified_shots.append(candidate.shot_id)
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

    def _get_beat_for_candidate(self, shot_id: str) -> Beat:
        if self.video_workspace is None:
            raise ValueError("video_workspace is required for Beat-based investigation")
        for beat in self.video_workspace.beats:
            if shot_id == beat.beat_id or shot_id in beat.shot_ids:
                return beat
        raise ValueError(f"Unknown shot_id: {shot_id}")


def _default_frame_sampler(beat: Beat, max_frames: int) -> tuple[str, ...]:
    if max_frames <= 0 or not beat.keyframe_path:
        return ()
    return (beat.keyframe_path,)
