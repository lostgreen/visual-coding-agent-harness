from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from vcah.investigator import InvestigationReport, ObservationAttempt
from vcah.multiround import (
    InvestigationTask,
    ReasonerDecision,
    VirtualVideoMultiRoundDriver,
    _decision_preflight,
)
from vcah.virtual_video import (
    VirtualVideoCase,
    VirtualVideoManifest,
    VirtualVideoSegment,
    VirtualVideoWorkspace,
)
from vcah.workspace import stable_attempt_id


def _workspace(tmp_path: Path) -> VirtualVideoWorkspace:
    segment = VirtualVideoSegment(
        "seg_0001",
        "video-a",
        "video-a.mp4",
        0.0,
        20.0,
        0.0,
        20.0,
    )
    return VirtualVideoWorkspace.create(
        tmp_path,
        manifest=VirtualVideoManifest("closure", (segment,)),
        case=VirtualVideoCase(
            "closure",
            "What is visible?",
            {"A": "A book", "B": "A cup"},
        ),
    )


class ScriptedReasoner:
    def __init__(self, decisions: Sequence[ReasonerDecision]) -> None:
        self.decisions = list(decisions)
        self.calls: list[dict[str, Any]] = []

    def decide(self, **kwargs: Any) -> ReasonerDecision:
        self.calls.append(dict(kwargs))
        return self.decisions.pop(0)


class FakeInvestigator:
    def reset_run_state(self) -> None:
        return None

    def run_batch(self, tasks: Sequence[InvestigationTask]) -> tuple[InvestigationReport, ...]:
        task = tasks[0]
        attempt_id = stable_attempt_id(
            source_video_ids=("video-a",),
            frame_times=(5.0,),
            sampling_fps=1.0,
            modality="visual",
        )
        return (
            InvestigationReport(
                query_id=task.query_id,
                status="completed",
                attempts=(
                    ObservationAttempt(
                        attempt_id=attempt_id,
                        task_id=task.query_id,
                        requested_range=(5.0, 5.0),
                        inspected_ranges=((5.0, 5.0),),
                        attached_frame_times=(5.0,),
                        sampling_config={"fps": 1.0, "mode": "window"},
                        images_requested=1,
                        images_attached=1,
                        frame_refs=("frame-5.jpg",),
                        raw_output='{"summary":"A cup is visible."}',
                        source_video_ids=("video-a",),
                    ),
                ),
                cost={"frames": 1, "vlm_calls": 1, "consumes_budget": True},
            ),
        )

    def mechanical_status(self) -> dict[str, object]:
        return {}


def test_final_answer_gets_exactly_one_bounded_closure_repair(tmp_path: Path) -> None:
    task = InvestigationTask(
        query_id="inspect",
        goal="Inspect the visible object.",
        segment_id="seg_0001",
        time_range=(5.0, 6.0),
        sampling_floor_fps=1.0,
    )
    attempt_id = stable_attempt_id(
        source_video_ids=("video-a",),
        frame_times=(5.0,),
        sampling_fps=1.0,
        modality="visual",
    )
    reasoner = ScriptedReasoner(
        (
            ReasonerDecision(action="investigate", tasks=(task,)),
            ReasonerDecision(
                action="answer",
                answer="B. A cup",
                supporting_claim_ids=("claim_missing",),
            ),
            ReasonerDecision(
                action="answer",
                answer="B. A cup",
                workspace_ops=(
                    {
                        "op": "add_claim",
                        "claim_id": "claim_cup",
                        "text": "A cup is visible.",
                        "source": "observation",
                        "cites": [attempt_id],
                        "confidence": "high",
                    },
                ),
                supporting_claim_ids=("claim_cup",),
            ),
        )
    )
    result = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=FakeInvestigator(),
        max_rounds=1,
        max_investigations=1,
        closure_repair_budget=1,
    ).run(_workspace(tmp_path))

    assert result.reference_valid
    assert reasoner.calls[2]["closure_repair"] is True
    repair_rows = [row for row in result.trace if row.get("type") == "closure_repair"]
    assert len(repair_rows) == 1


def test_closure_repair_preflight_forbids_global_search_and_unbound_scan() -> None:
    global_search = ReasonerDecision(
        action="investigate",
        tasks=(
            InvestigationTask(
                query_id="search",
                goal="Search globally.",
                inspection_mode="search_caption",
                caption_queries=("target",),
            ),
        ),
    )
    unbound_window = ReasonerDecision(
        action="investigate",
        tasks=(
            InvestigationTask(
                query_id="scan",
                goal="Scan a new range.",
                segment_id="seg_0001",
                time_range=(0.0, 20.0),
            ),
        ),
    )

    assert _decision_preflight(global_search, closure_repair=True)[0]["code"] == (
        "closure_repair_global_search_forbidden"
    )
    assert _decision_preflight(unbound_window, closure_repair=True)[0]["code"] == (
        "closure_repair_unbound_window_forbidden"
    )
