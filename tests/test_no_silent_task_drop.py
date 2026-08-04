from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from vcah.investigator import InvestigationReport
from vcah.multiround import (
    InvestigationTask,
    ReasonerDecision,
    VirtualVideoMultiRoundDriver,
)
from vcah.runtime_metrics import agent_run_metrics
from vcah.virtual_video import (
    VirtualVideoCase,
    VirtualVideoManifest,
    VirtualVideoSegment,
    VirtualVideoWorkspace,
)


class ScriptedReasoner:
    def __init__(self, decisions: Sequence[ReasonerDecision]) -> None:
        self.decisions = list(decisions)

    def decide(self, **_: Any) -> ReasonerDecision:
        return self.decisions.pop(0)


class RecordingInvestigator:
    def __init__(self) -> None:
        self.tasks: list[InvestigationTask] = []

    def reset_run_state(self) -> None:
        self.tasks.clear()

    def mechanical_status(self) -> dict[str, Any]:
        return {}

    def run_batch(
        self,
        tasks: Sequence[InvestigationTask],
    ) -> tuple[InvestigationReport, ...]:
        self.tasks.extend(tasks)
        return tuple(
            InvestigationReport(
                query_id=task.query_id,
                status="failed",
                cost={"consumes_budget": True},
                failure_reason="test fixture has no media",
            )
            for task in tasks
        )


def _workspace(tmp_path: Path) -> VirtualVideoWorkspace:
    segment = VirtualVideoSegment(
        segment_id="seg_0001",
        source_video_id="video-a",
        source_path="video-a.mp4",
        source_start_sec=0.0,
        source_end_sec=20.0,
        virtual_start_sec=0.0,
        virtual_end_sec=20.0,
    )
    return VirtualVideoWorkspace.create(
        tmp_path,
        manifest=VirtualVideoManifest("task-ledger", (segment,)),
        case=VirtualVideoCase(case_id="task-ledger", question="What happens?"),
    )


def test_every_requested_acquisition_has_terminal_outcome(tmp_path: Path) -> None:
    tasks = (
        InvestigationTask(
            query_id="executed",
            goal="Inspect the first window.",
            segment_id="seg_0001",
            time_range=(1.0, 2.0),
        ),
        InvestigationTask(
            query_id="invalid",
            goal="",
            segment_id="seg_0001",
            time_range=(3.0, 4.0),
        ),
        InvestigationTask(
            query_id="over_limit",
            goal="Inspect the third window.",
            segment_id="seg_0001",
            time_range=(5.0, 6.0),
        ),
    )
    reasoner = ScriptedReasoner(
        (
            ReasonerDecision(action="investigate", tasks=tasks),
            ReasonerDecision(action="answer"),
            ReasonerDecision(action="answer"),
        )
    )
    investigator = RecordingInvestigator()

    result = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=investigator,
        max_rounds=1,
        max_tasks_per_round=1,
    ).run(_workspace(tmp_path))
    requests = [row for row in result.trace if row.get("type") == "task_request"]
    outcomes = [row for row in result.trace if row.get("type") == "task_outcome"]
    metrics = agent_run_metrics(
        result.trace,
        (),
        answer_present=result.answer_present,
        reference_valid=result.reference_valid,
    )

    assert [task.query_id for task in investigator.tasks] == ["executed"]
    assert len(requests) == 3
    assert len(outcomes) == 3
    assert {row["ledger_id"] for row in requests} == {
        row["ledger_id"] for row in outcomes
    }
    assert sum(row["status"] == "executed" for row in outcomes) == 1
    assert sum(row["status"] == "explicit_resolution_error" for row in outcomes) == 2
    assert {
        error["code"]
        for row in outcomes
        for error in row.get("errors", ())
    } == {"goal_missing", "per_round_task_limit_exceeded"}
    assert metrics["requested_acquisition_count"] == 3
    assert metrics["executed_acquisition_count"] == 1
    assert metrics["task_resolution_error_count"] == 2
    assert metrics["silently_dropped_acquisition_count"] == 0
