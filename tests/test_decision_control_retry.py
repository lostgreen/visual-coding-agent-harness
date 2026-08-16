from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from vcah.interactive_agents import WorkspaceReasoner
from vcah.investigator import InvestigationReport, ObservationAttempt
from vcah.occurrence_agent import OccurrenceResolutionStateV2
from vcah.multiround import (
    InvestigationTask,
    ReasonerDecision,
    VirtualVideoMultiRoundDriver,
    _append_contradictory_gate_state,
    _control_retry_feedback,
    _record_occurrence_locator_outcome,
)
from vcah.runtime_metrics import agent_run_metrics
from vcah.virtual_video import (
    VirtualVideoCase,
    VirtualVideoManifest,
    VirtualVideoSegment,
    VirtualVideoWorkspace,
)
from vcah.workspace import stable_attempt_id


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


class ScopedOccurrenceInvestigator(RecordingInvestigator):
    locator_attempt_id = stable_attempt_id(
        source_video_ids=(),
        frame_refs=(),
        frame_times=(),
        inspected_ranges=(),
        sampling_fps=0.0,
        modality="caption_search",
    )
    occurrence_set = {
        "status": "competing_candidates",
        "occurrence_ambiguous": True,
        "candidates": [
            {
                "occurrence_id": "occ_1",
                "time_range": [2.0, 4.0],
                "source_video_ids": ["video-a"],
                "segment_ids": ["seg_0001"],
            },
            {
                "occurrence_id": "occ_2",
                "time_range": [8.0, 10.0],
                "source_video_ids": ["video-a"],
                "segment_ids": ["seg_0001"],
            },
        ],
    }

    def run_batch(
        self,
        tasks: Sequence[InvestigationTask],
    ) -> tuple[InvestigationReport, ...]:
        self.tasks.extend(tasks)
        reports = []
        for task in tasks:
            if task.inspection_mode == "search_caption":
                attempt = ObservationAttempt(
                    attempt_id=self.locator_attempt_id,
                    task_id=task.query_id,
                    sampling_config={
                        "mode": "search_caption",
                        "queries": ["target event"],
                        "occurrence_set": self.occurrence_set,
                        "hits": [],
                    },
                    modality="caption_search",
                    evidence_role="candidate",
                    parse_status="deterministic",
                )
            else:
                attempt = ObservationAttempt(
                    attempt_id="",
                    task_id=task.query_id,
                    requested_range=task.time_range,
                    inspected_ranges=(task.time_range,) if task.time_range else (),
                    sampling_config={
                        "mode": "window",
                        "candidate_binding": {
                            "locator_attempt_id": task.locator_attempt_id,
                            "occurrence_id": task.occurrence_id,
                        },
                    },
                    modality="visual",
                    parse_status="parsed",
                )
            reports.append(
                InvestigationReport(
                    query_id=task.query_id,
                    status="completed",
                    attempts=(attempt,),
                    cost={"consumes_budget": True},
                )
            )
        return tuple(reports)


class SingleScopedOccurrenceInvestigator(ScopedOccurrenceInvestigator):
    occurrence_set = {
        "status": "candidate",
        "occurrence_ambiguous": False,
        "candidates": [
            {
                "occurrence_id": "occ_single",
                "time_range": [8.0, 10.0],
                "source_video_ids": ["video-a"],
                "segment_ids": ["seg_0001"],
            }
        ],
    }


class SufficiencyScopedOccurrenceInvestigator(ScopedOccurrenceInvestigator):
    occurrence_set = {
        **ScopedOccurrenceInvestigator.occurrence_set,
        "candidates": [
            {
                **candidate,
                "passage_ids": [f"p{index}"],
            }
            for index, candidate in enumerate(
                ScopedOccurrenceInvestigator.occurrence_set["candidates"], start=1
            )
        ],
    }


class RotatingScopedOccurrenceInvestigator(ScopedOccurrenceInvestigator):
    locator_attempt_ids = tuple(
        stable_attempt_id(
            source_video_ids=(source_video_id,),
            frame_refs=(),
            frame_times=(),
            inspected_ranges=(),
            sampling_fps=0.0,
            modality="caption_search",
        )
        for source_video_id in ("video-a", "video-b")
    )

    def reset_run_state(self) -> None:
        super().reset_run_state()
        self.search_count = 0

    def run_batch(
        self,
        tasks: Sequence[InvestigationTask],
    ) -> tuple[InvestigationReport, ...]:
        self.tasks.extend(tasks)
        reports = []
        for task in tasks:
            if task.inspection_mode == "search_caption":
                index = min(self.search_count, 1)
                self.search_count += 1
                occurrence_id = f"occ_{index + 1}"
                source_video_id = ("video-a", "video-b")[index]
                attempt = ObservationAttempt(
                    attempt_id=self.locator_attempt_ids[index],
                    task_id=task.query_id,
                    sampling_config={
                        "mode": "search_caption",
                        "queries": ["target event"],
                        "occurrence_set": {
                            "status": "candidate",
                            "occurrence_ambiguous": False,
                            "candidates": [
                                {
                                    "occurrence_id": occurrence_id,
                                    "time_range": [2.0 + 6.0 * index, 4.0 + 6.0 * index],
                                    "source_video_ids": [source_video_id],
                                    "segment_ids": ["seg_0001"],
                                }
                            ],
                        },
                        "hits": [],
                    },
                    modality="caption_search",
                    evidence_role="candidate",
                    parse_status="deterministic",
                    source_video_ids=(source_video_id,),
                )
            else:
                attempt = ObservationAttempt(
                    attempt_id="",
                    task_id=task.query_id,
                    requested_range=task.time_range,
                    inspected_ranges=(task.time_range,) if task.time_range else (),
                    sampling_config={
                        "mode": "window",
                        "candidate_binding": {
                            "locator_attempt_id": task.locator_attempt_id,
                            "occurrence_id": task.occurrence_id,
                        },
                    },
                    modality="visual",
                    parse_status="parsed",
                )
            reports.append(
                InvestigationReport(
                    query_id=task.query_id,
                    status="completed",
                    attempts=(attempt,),
                    cost={"consumes_budget": True},
                )
            )
        return tuple(reports)


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


def test_occurrence_retry_feedback_rejects_contradictory_gate_state() -> None:
    feedback = _control_retry_feedback(
        (
            {"code": "occurrence_answer_required_after_resolution"},
            {"code": "occurrence_locator_inspection_required"},
        ),
        revision=4,
        previous_feedback={},
    )
    trace: list[dict[str, Any]] = []
    _append_contradictory_gate_state(trace, feedback, round_id=5)

    assert feedback["type"] == "contradictory_gate_state"
    assert "Return action=answer" not in feedback["instruction"]
    assert "Return action=investigate" not in feedback["instruction"]
    assert trace == [
        {
            "type": "contradictory_gate_state",
            "round": 5,
            "must_answer_codes": [
                "occurrence_answer_required_after_resolution"
            ],
            "must_not_answer_codes": [
                "occurrence_locator_inspection_required"
            ],
        }
    ]


def test_a4_retry_feedback_uses_persisted_insufficient_verdict() -> None:
    state = OccurrenceResolutionStateV2(sufficiency_enabled=True)
    state.sync_sets(
        (
            {
                "attempt_id": "attempt_sufficiency",
                "candidates": [
                    {"occurrence_id": "occ_1", "passage_ids": ["p1"]},
                ],
            },
        )
    )
    assert state.apply_ops(
        (
            {
                "op": "assess_sufficiency",
                "set_id": "attempt_sufficiency",
                "verdict": "insufficient",
                "constraints_checked": [
                    {
                        "constraint_id": "identity",
                        "constraint_type": "identity",
                        "description": "target identity",
                        "support": [
                            {
                                "occurrence_id": "occ_1",
                                "status": "unknown",
                                "evidence_passage_ids": [],
                            }
                        ],
                    }
                ],
            },
        )
    )["accepted"] is True

    feedback = _control_retry_feedback(
        ({"code": "occurrence_sufficiency_already_assessed"},),
        revision=2,
        previous_feedback={},
        occurrence_state=state,
        force_finalize=True,
    )

    assert "Do not answer, reassess, select, or defer" in feedback["instruction"]
    assert "exactly one no_match" in feedback["instruction"]


def test_a4_finalization_retry_forbids_defer() -> None:
    feedback = _control_retry_feedback(
        ({"code": "occurrence_sufficiency_resolution_required"},),
        revision=2,
        previous_feedback={},
        force_finalize=True,
    )

    assert "select or no_match" in feedback["instruction"]
    assert "Do not use defer during finalization" in feedback["instruction"]


def test_a4_incomplete_support_retry_requests_exact_missing_rows() -> None:
    feedback = _control_retry_feedback(
        (
            {
                "code": "occurrence_sufficiency_support_incomplete",
                "constraint_id": "target_identity",
                "missing_occurrence_ids": ["occ_2", "occ_3"],
            },
        ),
        revision=2,
        previous_feedback={},
        force_finalize=True,
    )

    instruction = feedback["instruction"]
    assert "incomplete serialization" in instruction
    assert "explicit status=unknown" in instruction
    assert "occ_2" in instruction
    assert "occ_3" in instruction
    assert "select or no_match" in instruction


def test_occurrence_locator_terminal_outcomes_are_explicit_and_exclusive() -> None:
    trace: list[dict[str, Any]] = []
    outcomes: dict[tuple[str, str], str] = {}
    expected = (
        ("inspected", ""),
        ("released_at_budget_exhaustion", "budget_exhausted_at_finalize"),
        ("released_on_set_retirement", "set_retired"),
        ("released_by_revision", "resolution_revision:no_match"),
    )
    for index, (outcome, reason) in enumerate(expected, start=1):
        _record_occurrence_locator_outcome(
            trace,
            outcomes,
            round_id=index,
            locator={
                "locator_attempt_id": f"set_{index}",
                "occurrence_id": f"occ_{index}",
            },
            outcome=outcome,
            reason=reason,
            revision_op=("no_match" if outcome == "released_by_revision" else ""),
        )

    assert set(outcomes.values()) == {value[0] for value in expected}
    assert [row["type"] for row in trace] == [
        "occurrence_locator_inspected",
        "occurrence_locator_released_unexecuted",
        "occurrence_locator_released_unexecuted",
        "occurrence_locator_released_unexecuted",
    ]
    _record_occurrence_locator_outcome(
        trace,
        outcomes,
        round_id=6,
        locator={"locator_attempt_id": "set_1", "occurrence_id": "occ_1"},
        outcome="released_by_revision",
        reason="resolution_revision:reopen",
        revision_op="reopen",
    )
    assert trace[-1]["type"] == "occurrence_locator_accounting_conflict"


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


def test_a2_commits_valid_selection_when_workspace_transaction_fails(
    tmp_path: Path,
) -> None:
    reasoner = ScriptedReasoner(
        (
            ReasonerDecision(
                action="update_workspace",
                workspace_ops=({"op": "unsupported_test_operation"},),
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
    assert [row.get("action") for row in decisions] == [
        "update_workspace",
        "answer",
    ]
    assert decisions[0]["workspace_ops_accepted"] is False
    assert decisions[0]["occurrence_ops_accepted"] is True
    assert decisions[0]["occurrence_selection_committed"] is True
    assert decisions[0]["selected_occurrence_id"] == "occ_2"
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


def test_a2_clean_is_identical_before_ambiguous_set_exposure(
    tmp_path: Path,
) -> None:
    reasoner = ScriptedReasoner(
        (
            ReasonerDecision(
                action="investigate",
                tasks=(
                    InvestigationTask(
                        query_id="locate",
                        goal="locate target event",
                        inspection_mode="search_caption",
                        caption_queries=("target event",),
                    ),
                ),
            ),
            ReasonerDecision(
                action="update_workspace",
                occurrence_ops=(
                    {
                        "op": "select",
                        "set_id": ScopedOccurrenceInvestigator.locator_attempt_id,
                        "occurrence_id": "occ_2",
                    },
                ),
            ),
            ReasonerDecision(action="answer", answer="resolved"),
        )
    )

    result = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=ScopedOccurrenceInvestigator(),
        max_rounds=4,
        control_retry_budget=1,
        controller_mode="frozen_baseline",
        evidence_control_mode="shadow",
        evidence_state_mode="llm_authored",
        occurrence_method_arm="a2-clean",
    ).run(_workspace(tmp_path))

    assert "occurrence_resolution_state" not in reasoner.calls[0][
        "mechanical_status"
    ]
    assert reasoner.calls[1]["mechanical_status"][
        "occurrence_resolution_state"
    ]["schema_version"] == "OccurrenceResolutionStateV2"
    activation = [
        row
        for row in result.trace
        if row.get("type") == "occurrence_arbitration_activated"
    ]
    assert len(activation) == 1
    decisions = [
        row for row in result.trace if row.get("type") == "reasoner_decision"
    ]
    assert decisions[1]["selected_occurrence_ids"] == ["occ_2"]
    assert result.answer_present is True


@pytest.mark.parametrize(
    ("terminal_op", "rewrite_op"),
    (
        (
            {
                "op": "select",
                "set_id": ScopedOccurrenceInvestigator.locator_attempt_id,
                "occurrence_id": "occ_2",
            },
            {
                "op": "no_match",
                "set_id": ScopedOccurrenceInvestigator.locator_attempt_id,
            },
        ),
        (
            {
                "op": "no_match",
                "set_id": ScopedOccurrenceInvestigator.locator_attempt_id,
            },
            {
                "op": "select",
                "set_id": ScopedOccurrenceInvestigator.locator_attempt_id,
                "occurrence_id": "occ_1",
            },
        ),
    ),
)
def test_scoped_terminal_resolution_requires_immediate_answer_and_rejects_rewrite(
    tmp_path: Path,
    terminal_op: dict[str, Any],
    rewrite_op: dict[str, Any],
) -> None:
    late_search = InvestigationTask(
        query_id="late_search",
        goal="replace committed resolution",
        inspection_mode="search_caption",
        caption_queries=("different event",),
    )
    reasoner = ScriptedReasoner(
        (
            ReasonerDecision(
                action="investigate",
                tasks=(
                    InvestigationTask(
                        query_id="locate",
                        goal="locate target event",
                        inspection_mode="search_caption",
                        caption_queries=("target event",),
                    ),
                ),
            ),
            ReasonerDecision(
                action="update_workspace",
                occurrence_ops=(terminal_op,),
            ),
            ReasonerDecision(
                action="investigate",
                tasks=(late_search,),
                occurrence_ops=(rewrite_op,),
            ),
            ReasonerDecision(action="answer", answer="resolved"),
        )
    )
    investigator = ScopedOccurrenceInvestigator()

    result = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=investigator,
        max_rounds=4,
        control_retry_budget=1,
        controller_mode="frozen_baseline",
        evidence_control_mode="shadow",
        evidence_state_mode="llm_authored",
        occurrence_method_arm="a2-clean",
    ).run(_workspace(tmp_path))

    errors = [
        error
        for row in result.trace
        if row.get("type") == "decision_schema_error"
        for error in row.get("errors", ())
    ]
    assert {error["code"] for error in errors} >= {
        "occurrence_resolution_already_committed",
        "occurrence_answer_required_after_resolution",
    }
    assert reasoner.calls[2]["force_finalize"] is False
    assert reasoner.calls[3]["control_retry"] is True
    assert [task.query_id for task in investigator.tasks] == ["locate"]
    decisions = [
        row for row in result.trace if row.get("type") == "reasoner_decision"
    ]
    assert [row.get("action") for row in decisions] == [
        "investigate",
        "update_workspace",
        "answer",
    ]
    assert result.answer_present is True


@pytest.mark.parametrize("occurrence_method_arm", ("a2-clean", "a3"))
def test_scoped_occurrence_answer_recovery_uses_dedicated_call_budget(
    tmp_path: Path,
    occurrence_method_arm: str,
) -> None:
    ignored_answer_gate = ReasonerDecision(action="investigate")
    reasoner = ScriptedReasoner(
        (
            ReasonerDecision(
                action="investigate",
                tasks=(
                    InvestigationTask(
                        query_id="locate",
                        goal="locate target event",
                        inspection_mode="search_caption",
                        caption_queries=("target event",),
                    ),
                ),
            ),
            ReasonerDecision(
                action="update_workspace",
                occurrence_ops=(
                    {
                        "op": "select",
                        "set_id": ScopedOccurrenceInvestigator.locator_attempt_id,
                        "occurrence_id": "occ_2",
                    },
                ),
            ),
            ignored_answer_gate,
            ignored_answer_gate,
            ReasonerDecision(action="answer", answer="resolved"),
        )
    )

    result = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=ScopedOccurrenceInvestigator(),
        max_rounds=1,
        control_retry_budget=1,
        controller_mode="frozen_baseline",
        evidence_control_mode="shadow",
        evidence_state_mode="llm_authored",
        occurrence_method_arm=occurrence_method_arm,
    ).run(_workspace(tmp_path))

    recovery_events = [
        row
        for row in result.trace
        if row.get("type") == "occurrence_recovery_round_granted"
    ]
    assert [row.get("recovery_index") for row in recovery_events] == [1, 2]
    assert not any(
        row.get("type") == "decision_control_exhausted"
        for row in result.trace
    )
    assert result.answer_present is True


def test_single_candidate_requires_resolution_without_arbitration(
    tmp_path: Path,
) -> None:
    reasoner = ScriptedReasoner(
        (
            ReasonerDecision(
                action="investigate",
                tasks=(
                    InvestigationTask(
                        query_id="locate_single",
                        goal="locate target event",
                        inspection_mode="search_caption",
                        caption_queries=("target event",),
                    ),
                ),
            ),
            ReasonerDecision(action="answer", answer="premature"),
            ReasonerDecision(
                action="update_workspace",
                occurrence_ops=(
                    {
                        "op": "select",
                        "set_id": SingleScopedOccurrenceInvestigator.locator_attempt_id,
                        "occurrence_id": "occ_single",
                    },
                ),
            ),
            ReasonerDecision(action="answer", answer="resolved"),
        )
    )

    result = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=SingleScopedOccurrenceInvestigator(),
        max_rounds=4,
        control_retry_budget=1,
        controller_mode="frozen_baseline",
        evidence_control_mode="shadow",
        evidence_state_mode="llm_authored",
        occurrence_method_arm="a2-clean",
    ).run(_workspace(tmp_path))

    resolution_events = [
        row
        for row in result.trace
        if row.get("type") == "occurrence_resolution_activated"
    ]
    assert len(resolution_events) == 1
    assert resolution_events[0]["candidate_count"] == 1
    assert resolution_events[0]["arbitration_required"] is False
    assert not any(
        row.get("type") == "occurrence_arbitration_activated"
        for row in result.trace
    )
    assert any(
        row.get("type") == "decision_schema_error"
        and row.get("code") == "occurrence_resolution_required"
        for row in result.trace
    )
    assert result.answer_present is True


def test_a3_rejects_answer_until_selected_locator_is_inspected(
    tmp_path: Path,
) -> None:
    reasoner = ScriptedReasoner(
        (
            ReasonerDecision(
                action="investigate",
                tasks=(
                    InvestigationTask(
                        query_id="locate",
                        goal="locate target event",
                        inspection_mode="search_caption",
                        caption_queries=("target event",),
                    ),
                ),
            ),
            ReasonerDecision(
                action="update_workspace",
                occurrence_ops=(
                    {
                        "op": "select",
                        "set_id": ScopedOccurrenceInvestigator.locator_attempt_id,
                        "occurrence_id": "occ_2",
                    },
                ),
            ),
            ReasonerDecision(action="answer", answer="too early"),
            ReasonerDecision(
                action="investigate",
                tasks=(
                    InvestigationTask(
                        query_id="inspect_selected",
                        goal="inspect selected target",
                        occurrence_id="occ_2",
                        locator_attempt_id=ScopedOccurrenceInvestigator.locator_attempt_id,
                    ),
                ),
            ),
            ReasonerDecision(action="answer", answer="resolved"),
        )
    )
    investigator = ScopedOccurrenceInvestigator()

    result = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=investigator,
        max_rounds=4,
        control_retry_budget=1,
        controller_mode="frozen_baseline",
        evidence_control_mode="shadow",
        evidence_state_mode="llm_authored",
        occurrence_method_arm="a3",
    ).run(_workspace(tmp_path))

    pre_selection_status = reasoner.calls[1]["mechanical_status"]
    assert "selected_occurrence_locators" not in pre_selection_status
    assert "active_occurrence_locators" not in pre_selection_status
    post_selection_status = reasoner.calls[2]["mechanical_status"]
    assert post_selection_status["selected_occurrence_locators"]
    assert post_selection_status["active_occurrence_locators"]
    assert any(
        row.get("type") == "decision_schema_error"
        and row.get("code") == "occurrence_locator_inspection_required"
        for row in result.trace
    )
    bound = [
        task
        for task in investigator.tasks
        if task.locator_attempt_id and task.occurrence_id
    ]
    assert len(bound) == 1
    assert (
        bound[0].locator_attempt_id
        == ScopedOccurrenceInvestigator.locator_attempt_id
    )
    assert bound[0].occurrence_id == "occ_2"
    assert result.answer_present is True


def test_a4_orders_sufficiency_selection_locator_and_answer(
    tmp_path: Path,
) -> None:
    set_id = SufficiencyScopedOccurrenceInvestigator.locator_attempt_id
    reasoner = ScriptedReasoner(
        (
            ReasonerDecision(
                action="investigate",
                tasks=(
                    InvestigationTask(
                        query_id="locate",
                        goal="locate target event",
                        inspection_mode="search_caption",
                        caption_queries=("target event",),
                    ),
                ),
            ),
            ReasonerDecision(
                action="update_workspace",
                occurrence_ops=(
                    {
                        "op": "assess_sufficiency",
                        "set_id": set_id,
                        "verdict": "sufficient",
                        "constraints_checked": [
                            {
                                "constraint_id": "target_identity",
                                "constraint_type": "identity",
                                "description": "candidate depicts the target",
                                "support": [
                                    {
                                        "occurrence_id": "occ_1",
                                        "status": "unknown",
                                        "evidence_passage_ids": [],
                                    },
                                    {
                                        "occurrence_id": "occ_2",
                                        "status": "supported",
                                        "evidence_passage_ids": ["p2"],
                                    },
                                ],
                            }
                        ],
                    },
                    {
                        "op": "select",
                        "set_id": set_id,
                        "occurrence_id": "occ_2",
                    },
                ),
            ),
            ReasonerDecision(
                action="investigate",
                tasks=(
                    InvestigationTask(
                        query_id="inspect_selected",
                        goal="inspect selected target",
                        occurrence_id="occ_2",
                        locator_attempt_id=set_id,
                    ),
                ),
            ),
            ReasonerDecision(action="answer", answer="resolved"),
        )
    )

    result = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=SufficiencyScopedOccurrenceInvestigator(),
        max_rounds=4,
        control_retry_budget=1,
        controller_mode="frozen_baseline",
        evidence_control_mode="shadow",
        evidence_state_mode="llm_authored",
        occurrence_method_arm="a4",
    ).run(_workspace(tmp_path))

    decisions = [
        row for row in result.trace if row.get("type") == "reasoner_decision"
    ]
    sufficiency = [
        row
        for row in result.trace
        if row.get("type") == "occurrence_sufficiency_decision"
    ]
    assert len(sufficiency) == 1
    assert sufficiency[0]["verdict"] == "sufficient"
    assert sufficiency[0]["sufficient_occurrence_ids"] == ["occ_2"]
    assert decisions[1]["occurrence_selection_committed"] is True
    assert decisions[2]["action"] == "investigate"
    assert decisions[3]["action"] == "answer"
    assert result.answer_present is True


def test_a3_rejects_resolution_revision_after_locator_inspection(
    tmp_path: Path,
) -> None:
    set_id = ScopedOccurrenceInvestigator.locator_attempt_id
    reasoner = ScriptedReasoner(
        (
            ReasonerDecision(
                action="update_workspace",
                occurrence_ops=(
                    {"op": "select", "set_id": set_id, "occurrence_id": "occ_2"},
                ),
            ),
            ReasonerDecision(
                action="investigate",
                tasks=(
                    InvestigationTask(
                        query_id="inspect_selected",
                        goal="inspect selected occurrence",
                        locator_attempt_id=set_id,
                        occurrence_id="occ_2",
                    ),
                ),
            ),
            ReasonerDecision(
                action="update_workspace",
                occurrence_ops=({"op": "defer", "set_id": set_id},),
            ),
            ReasonerDecision(action="answer", answer="resolved"),
        )
    )

    result = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=ScopedOccurrenceInvestigator(),
        max_rounds=4,
        max_investigations=4,
        control_retry_budget=1,
        controller_mode="frozen_baseline",
        evidence_control_mode="shadow",
        evidence_state_mode="llm_authored",
        occurrence_method_arm="a3",
        bootstrap_tasks=(
            InvestigationTask(
                query_id="locate",
                goal="locate target event",
                inspection_mode="search_caption",
                caption_queries=("target event",),
            ),
        ),
    ).run(_workspace(tmp_path))

    assert any(
        error.get("code") == "occurrence_resolution_already_committed"
        for row in result.trace
        if row.get("type") == "decision_schema_error"
        for error in row.get("errors", ())
    )
    assert not any(
        row.get("type") == "occurrence_locator_released_unexecuted"
        and row.get("outcome") == "released_by_revision"
        for row in result.trace
    )
    assert result.answer_present is True


def test_a3_allows_deferred_set_retirement_before_terminal_resolution(
    tmp_path: Path,
) -> None:
    first_set, second_set = RotatingScopedOccurrenceInvestigator.locator_attempt_ids
    reasoner = ScriptedReasoner(
        (
            ReasonerDecision(
                action="update_workspace",
                occurrence_ops=({"op": "defer", "set_id": first_set},),
            ),
            ReasonerDecision(
                action="investigate",
                tasks=(
                    InvestigationTask(
                        query_id="search_more",
                        goal="search for another occurrence set",
                        inspection_mode="search_caption",
                        caption_queries=("target event",),
                    ),
                ),
            ),
            ReasonerDecision(
                action="update_workspace",
                occurrence_ops=({"op": "no_match", "set_id": second_set},),
            ),
            ReasonerDecision(action="answer", answer="resolved"),
        )
    )

    result = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=RotatingScopedOccurrenceInvestigator(),
        max_rounds=4,
        max_investigations=4,
        control_retry_budget=0,
        controller_mode="frozen_baseline",
        evidence_control_mode="shadow",
        evidence_state_mode="llm_authored",
        occurrence_method_arm="a3",
        bootstrap_tasks=(
            InvestigationTask(
                query_id="locate",
                goal="locate target event",
                inspection_mode="search_caption",
                caption_queries=("target event",),
            ),
        ),
    ).run(_workspace(tmp_path))

    decisions = [
        row for row in result.trace if row.get("type") == "reasoner_decision"
    ]
    assert decisions[-2]["active_occurrence_set_id"] == second_set
    assert not any(
        error.get("code") == "occurrence_resolution_already_committed"
        for row in result.trace
        if row.get("type") == "decision_schema_error"
        for error in row.get("errors", ())
    )
    assert result.answer_present is True


def test_force_finalize_is_arm_symmetric(tmp_path: Path) -> None:
    calls_by_arm: dict[str, list[dict[str, Any]]] = {}
    metrics_by_arm: dict[str, dict[str, float | int | None]] = {}
    traces_by_arm: dict[str, tuple[dict[str, Any], ...]] = {}

    for arm in ("a2-clean", "a3"):
        reasoner = ScriptedReasoner(
            (
                ReasonerDecision(action="update_workspace"),
                ReasonerDecision(
                    action="update_workspace",
                    occurrence_ops=(
                        {
                            "op": "select",
                            "set_id": ScopedOccurrenceInvestigator.locator_attempt_id,
                            "occurrence_id": "occ_2",
                        },
                    ),
                ),
                ReasonerDecision(action="answer", answer="resolved"),
                ReasonerDecision(action="answer", answer="resolved"),
                ReasonerDecision(action="answer", answer="resolved"),
            )
        )
        result = VirtualVideoMultiRoundDriver(
            reasoner=reasoner,
            investigator=ScopedOccurrenceInvestigator(),
            max_rounds=1,
            max_investigations=4,
            control_retry_budget=0,
            controller_mode="frozen_baseline",
            evidence_control_mode="shadow",
            evidence_state_mode="llm_authored",
            occurrence_method_arm=arm,
            bootstrap_tasks=(
                InvestigationTask(
                    query_id="locate",
                    goal="locate target event",
                    inspection_mode="search_caption",
                    caption_queries=("target event",),
                ),
            ),
        ).run(_workspace(tmp_path / arm))
        calls_by_arm[arm] = reasoner.calls
        metrics_by_arm[arm] = agent_run_metrics(
            result.trace,
            (),
            answer_present=result.answer_present,
            reference_valid=result.reference_valid,
        )
        traces_by_arm[arm] = tuple(dict(row) for row in result.trace)

    assert calls_by_arm["a2-clean"][1]["force_finalize"] is True
    assert calls_by_arm["a3"][1]["force_finalize"] is True
    assert metrics_by_arm["a2-clean"]["forced_finalize_round"] == 2
    assert metrics_by_arm["a3"]["forced_finalize_round"] == 2
    assert metrics_by_arm["a2-clean"]["semantic_rounds_used"] == 3
    assert metrics_by_arm["a3"]["semantic_rounds_used"] == 3
    assert metrics_by_arm["a2-clean"]["extra_rounds_granted"] == 1
    assert metrics_by_arm["a3"]["extra_rounds_granted"] == 1
    assert any(
        row.get("type") == "occurrence_locator_budget_exhausted_at_finalize"
        and row.get("method_arm") == "a3"
        and row.get("pending_locator_count") == 1
        for row in traces_by_arm["a3"]
    )
    assert any(
        row.get("type") == "occurrence_locator_released_unexecuted"
        and row.get("outcome") == "released_at_budget_exhaustion"
        and row.get("reason") == "budget_exhausted_at_finalize"
        for row in traces_by_arm["a3"]
    )
    assert not any(
        row.get("type") == "decision_schema_error"
        and row.get("code")
        in {
            "occurrence_locator_inspection_required",
            "occurrence_answer_required_after_resolution",
        }
        for row in traces_by_arm["a3"]
        if int(row.get("round", 0) or 0) >= 2
    )


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
