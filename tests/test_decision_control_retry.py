from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from vcah.interactive_agents import WorkspaceReasoner
from vcah.investigator import InvestigationReport
from vcah.multiround import (
    InvestigationTask,
    ReasonerDecision,
    VirtualVideoMultiRoundDriver,
    _control_retry_feedback,
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
        self.calls: list[dict[str, Any]] = []

    def decide(self, **kwargs: Any) -> ReasonerDecision:
        self.calls.append(dict(kwargs))
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


class OccurrenceInvestigator(RecordingInvestigator):
    def mechanical_status(self) -> dict[str, Any]:
        return {
            "caption_occurrence_sets": [
                {
                    "candidates": [
                        {"occurrence_id": "occ_1"},
                        {"occurrence_id": "occ_2"},
                    ]
                }
            ]
        }


class MetadataScriptedReasoner(ScriptedReasoner):
    def __init__(
        self,
        decisions: Sequence[ReasonerDecision],
        metadata: Sequence[dict[str, Any]],
    ) -> None:
        super().__init__(decisions)
        self.metadata = list(metadata)
        self.last_metadata: dict[str, Any] = {}

    def decide(self, **kwargs: Any) -> ReasonerDecision:
        decision = super().decide(**kwargs)
        self.last_metadata = self.metadata.pop(0) if self.metadata else {}
        return decision

    def consume_decision_metadata(self) -> dict[str, Any]:
        value = self.last_metadata
        self.last_metadata = {}
        return value


class FakeAPI:
    model = "fake-model"

    def __init__(self, responses: Sequence[str]) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []
        self.last_response_metadata: dict[str, Any] = {}

    def chat(self, prompt: str, **_: Any) -> str:
        self.calls.append(prompt)
        self.last_response_metadata = {
            "finish_reason": "stop",
            "completion_tokens": 20,
            "completion_tokens_details": {"reasoning_tokens": 5},
        }
        return self.responses.pop(0)


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
        manifest=VirtualVideoManifest("control-retry", (segment,)),
        case=VirtualVideoCase(case_id="control-retry", question="What happens?"),
    )


def test_action_like_workspace_op_repairs_within_same_semantic_round(tmp_path: Path) -> None:
    task = InvestigationTask(
        query_id="inspect_valid_task",
        goal="Inspect the visible action.",
        segment_id="seg_0001",
        time_range=(4.0, 6.0),
    )
    invalid_placement = ReasonerDecision(
        action="investigate",
        workspace_ops=(
            {
                "op": "investigate",
                "tasks": [
                    {
                        "query_id": task.query_id,
                        "goal": task.goal,
                        "segment_id": task.segment_id,
                        "time_range": list(task.time_range or ()),
                    }
                ],
            },
        ),
    )
    reasoner = ScriptedReasoner(
        (
            invalid_placement,
            ReasonerDecision(action="investigate", tasks=(task,)),
            ReasonerDecision(action="answer"),
            ReasonerDecision(action="answer"),
        )
    )
    investigator = RecordingInvestigator()

    result = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=investigator,
        max_rounds=1,
        control_retry_budget=2,
    ).run(_workspace(tmp_path))

    schema_errors = [
        row for row in result.trace if row.get("type") == "decision_schema_error"
    ]
    metrics = agent_run_metrics(
        result.trace,
        (),
        answer_present=result.answer_present,
        reference_valid=result.reference_valid,
    )
    assert schema_errors[0]["code"] == "action_like_op_inside_workspace_ops"
    assert reasoner.calls[0]["semantic_round"] == 1
    assert reasoner.calls[0]["control_attempt"] == 0
    assert reasoner.calls[1]["semantic_round"] == 1
    assert reasoner.calls[1]["control_attempt"] == 1
    assert reasoner.calls[1]["control_retry"] is True
    assert [task.query_id for task in investigator.tasks] == ["inspect_valid_task"]
    assert sum(
        int(row.get("count", 0))
        for row in result.trace
        if row.get("type") == "control_retry"
    ) == 1
    assert metrics["requested_acquisition_count"] == 2
    assert metrics["executed_acquisition_count"] == 1
    assert metrics["task_resolution_error_count"] == 1
    assert metrics["silently_dropped_acquisition_count"] == 0


def test_arbitration_task_has_deliberate_interpretation_purpose() -> None:
    task = InvestigationTask(
        query_id="reread",
        goal="Arbitrate two readings of the same pixels.",
        inspection_mode="arbitrate_observation",
        arbitration_attempt_id="attempt_existing",
        interpretation_purpose="primary",
    )

    assert task.interpretation_purpose == "deliberate_arbitration"


def test_workspace_retry_feedback_names_mechanical_repairs() -> None:
    feedback = _control_retry_feedback(
        (
            {
                "code": "workspace_transaction_rejected",
                "detail": "op[1]: obligation_already_exists:req_1",
            },
            {
                "code": "workspace_transaction_rejected",
                "detail": "satisfied_obligation_requires_attempt:req_2",
            },
            {
                "code": "workspace_transaction_rejected",
                "detail": "obligation_dependency_lineage_missing:req_2:req_1",
            },
        ),
        revision=3,
        previous_feedback={},
    )

    instruction = feedback["instruction"]
    assert "Omit add operations for IDs that already exist" in instruction
    assert "supporting_attempt_ids" in instruction
    assert "supporting claim derives from the dependency" in instruction


def test_a2_recovers_from_workspace_exhaustion_before_selection(
    tmp_path: Path,
) -> None:
    rejected = ReasonerDecision(
        action="update_workspace",
        workspace_ops=({"op": "unsupported_test_operation"},),
    )
    reasoner = ScriptedReasoner(
        (
            rejected,
            rejected,
            ReasonerDecision(
                action="update_workspace",
                occurrence_ops=(
                    {"op": "select", "occurrence_id": "occ_2"},
                ),
            ),
            ReasonerDecision(action="answer", answer="resolved"),
        )
    )

    result = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=OccurrenceInvestigator(),
        max_rounds=4,
        control_retry_budget=1,
        controller_mode="frozen_baseline",
        evidence_control_mode="shadow",
        evidence_state_mode="llm_authored",
        occurrence_method_arm="a2",
    ).run(_workspace(tmp_path))

    decisions = [
        row for row in result.trace if row.get("type") == "reasoner_decision"
    ]
    selection_index = next(
        index
        for index, row in enumerate(decisions)
        if row.get("occurrence_selection_committed")
    )
    answer_index = next(
        index
        for index, row in enumerate(decisions)
        if row.get("action") == "answer"
    )
    assert selection_index < answer_index
    assert any(
        row.get("type") == "occurrence_lifecycle_repair_scheduled"
        and row.get("stage") == "workspace_transaction"
        for row in result.trace
    )
    assert result.answer_present is True


def test_a2_has_dedicated_repair_after_json_retry_consumes_control_budget(
    tmp_path: Path,
) -> None:
    reasoner = MetadataScriptedReasoner(
        (
            ReasonerDecision(action="answer", answer="premature"),
            ReasonerDecision(
                action="update_workspace",
                occurrence_ops=(
                    {"op": "select", "occurrence_id": "occ_1"},
                ),
            ),
            ReasonerDecision(action="answer", answer="resolved"),
        ),
        (
            {"internal_control_retry_count": 1, "format_repaired": True},
            {},
            {},
        ),
    )

    result = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=OccurrenceInvestigator(),
        max_rounds=4,
        control_retry_budget=1,
        controller_mode="frozen_baseline",
        evidence_control_mode="shadow",
        evidence_state_mode="llm_authored",
        occurrence_method_arm="a2",
    ).run(_workspace(tmp_path))

    decisions = [
        row for row in result.trace if row.get("type") == "reasoner_decision"
    ]
    assert [row.get("action") for row in decisions] == [
        "update_workspace",
        "answer",
    ]
    assert decisions[0]["occurrence_selection_committed"] is True
    assert any(
        row.get("type") == "occurrence_lifecycle_repair_scheduled"
        and row.get("stage") == "decision_preflight"
        for row in result.trace
    )
    assert not any(
        row.get("type") == "decision_control_exhausted"
        and any(
            str(error.get("code", "")).startswith("occurrence_")
            for error in row.get("errors", ())
            if isinstance(error, dict)
        )
        for row in result.trace
    )
    assert result.answer_present is True


def test_a2_rejects_non_answer_immediately_after_selection(
    tmp_path: Path,
) -> None:
    reasoner = ScriptedReasoner(
        (
            ReasonerDecision(
                action="update_workspace",
                occurrence_ops=(
                    {"op": "select", "occurrence_id": "occ_1"},
                ),
            ),
            ReasonerDecision(action="investigate"),
            ReasonerDecision(action="answer", answer="resolved"),
        )
    )

    result = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=OccurrenceInvestigator(),
        max_rounds=4,
        control_retry_budget=1,
        controller_mode="frozen_baseline",
        evidence_control_mode="shadow",
        evidence_state_mode="llm_authored",
        occurrence_method_arm="a2",
    ).run(_workspace(tmp_path))

    decisions = [
        row for row in result.trace if row.get("type") == "reasoner_decision"
    ]
    assert [row.get("action") for row in decisions] == [
        "update_workspace",
        "answer",
    ]
    assert any(
        row.get("type") == "decision_schema_error"
        and row.get("code") == "occurrence_answer_required_after_selection"
        for row in result.trace
    )
    assert reasoner.calls[1]["force_finalize"] is True
    assert reasoner.calls[2]["control_retry"] is True
    assert result.answer_present is True


def test_json_repair_uses_control_budget_without_advancing_semantic_round(
    tmp_path: Path,
) -> None:
    multi_block = '{"analysis":"first"}\n{"action":"update_workspace"}'
    api = FakeAPI(
        (
            multi_block,
            json.dumps({"action": "update_workspace"}),
            json.dumps({"action": "answer"}),
            json.dumps({"action": "answer"}),
        )
    )
    reasoner = WorkspaceReasoner(api, trace_path=tmp_path / "interactions.jsonl")

    result = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=RecordingInvestigator(),
        max_rounds=1,
        control_retry_budget=2,
    ).run(_workspace(tmp_path / "workspace"))
    metrics = agent_run_metrics(
        result.trace,
        (),
        answer_present=result.answer_present,
        reference_valid=result.reference_valid,
    )

    assert len(api.calls) == 4
    assert reasoner.calls == 3
    assert [
        row["semantic_round"]
        for row in result.trace
        if row.get("type") == "reasoner_decision"
        and row.get("semantic_committed")
    ] == [1, 2, 3]
    assert metrics["control_retry_count"] == 1
    assert metrics["decision_repair_count"] == 1
    assert metrics["rounds"] == 3
