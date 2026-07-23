from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from vcah.investigator import InvestigationReport, ObservationAttempt
from vcah.multiround import (
    InvestigationTask,
    ReasonerDecision,
    VirtualVideoMultiRoundDriver,
    _mechanical_status,
)
from vcah.types import CoverageSegment, EvidenceRecord
from vcah.virtual_video import (
    VirtualVideoCase,
    VirtualVideoManifest,
    VirtualVideoSegment,
    VirtualVideoWorkspace,
)
from vcah.workspace import stable_attempt_id


def _workspace(tmp_path: Path) -> VirtualVideoWorkspace:
    segment = VirtualVideoSegment(
        segment_id="seg_0001",
        source_video_id="video-a",
        source_path="video-a.mp4",
        source_start_sec=0.0,
        source_end_sec=20.0,
        virtual_start_sec=0.0,
        virtual_end_sec=20.0,
        role="target",
    )
    return VirtualVideoWorkspace.create(
        tmp_path,
        manifest=VirtualVideoManifest(workspace_id="case-1", segments=(segment,)),
        case=VirtualVideoCase(
            case_id="case-1",
            question="What does the person raise?",
            options={"A": "A book", "B": "A cup"},
            gold="B",
            target_segment_id=segment.segment_id,
            target_virtual_interval=(0.0, 20.0),
        ),
    )


def _report(task: InvestigationTask) -> InvestigationReport:
    attempt_id = stable_attempt_id(
        source_video_ids=("video-a",),
        frame_times=(5.0, 6.0),
        sampling_fps=1.0,
        modality="visual",
    )
    evidence = EvidenceRecord(
        evidence_id=f"ev_{task.query_id}",
        beat_id="",
        start_sec=5.0,
        end_sec=6.0,
        modality="visual",
        pointer=f"virtual://case-1/observations/{attempt_id}",
        verbatim="The person raises a cup.",
        frame_refs=("f5.jpg", "f6.jpg"),
        attestation_model="fake-investigator",
        temporal_scope="window",
        evidence_kind="visual_observation",
        request_ids=(task.query_id,),
        coverage_manifest=(CoverageSegment(task.query_id, 5.0, 6.0, "visual", 1.0),),
        task_id=task.query_id,
        observation_id=attempt_id,
        sampling_fps=1.0,
        source_lineage=({"source_video_id": "video-a"},),
    )
    attempt = ObservationAttempt(
        attempt_id=attempt_id,
        task_id=task.query_id,
        requested_range=(5.0, 6.0),
        inspected_ranges=((5.0, 6.0),),
        attached_frame_times=(5.0, 6.0),
        sampling_config={"fps": 1.0, "modality": "visual"},
        images_requested=2,
        images_attached=2,
        parse_status="parsed",
        execution_status="completed",
        frame_refs=("f5.jpg", "f6.jpg"),
        prompt_digest="prompt",
        raw_output='{"summary":"The person raises a cup."}',
        source_video_ids=("video-a",),
    )
    return InvestigationReport(
        query_id=task.query_id,
        status="completed",
        evidence=(evidence,),
        attempts=(attempt,),
        cost={"consumes_budget": True},
        coverage_delta=((5.0, 6.0),),
    )


class FakeInvestigator:
    def __init__(self) -> None:
        self.tasks: list[InvestigationTask] = []

    def reset_run_state(self) -> None:
        self.tasks.clear()

    def run_batch(self, tasks: Sequence[InvestigationTask]) -> tuple[InvestigationReport, ...]:
        self.tasks.extend(tasks)
        return tuple(_report(task) for task in tasks)


def _caption_report(task: InvestigationTask) -> InvestigationReport:
    pointer = "caption-search://case-1/index/query"
    attempt_id = stable_attempt_id(
        source_video_ids=("video-a",),
        frame_refs=(pointer,),
        modality="caption_search",
    )
    attempt = ObservationAttempt(
        attempt_id=attempt_id,
        task_id=task.query_id,
        sampling_config={
            "mode": "search_caption",
            "modality": "caption_search",
            "hits": [{"passage_id": "p1", "range": [5.0, 10.0], "score": 0.9}],
        },
        execution_status="completed",
        parse_status="deterministic",
        frame_refs=(pointer,),
        modality="caption_search",
        evidence_role="candidate",
        source_video_ids=("video-a",),
    )
    return InvestigationReport(
        query_id=task.query_id,
        status="completed",
        attempts=(attempt,),
        cost={"consumes_budget": True},
    )


class CaptionThenVisualInvestigator(FakeInvestigator):
    def run_batch(self, tasks: Sequence[InvestigationTask]) -> tuple[InvestigationReport, ...]:
        self.tasks.extend(tasks)
        return tuple(
            _caption_report(task) if task.inspection_mode == "search_caption" else _report(task)
            for task in tasks
        )


class ScriptedReasoner:
    def __init__(self, decisions: Sequence[ReasonerDecision]) -> None:
        self.decisions = list(decisions)
        self.calls: list[dict[str, Any]] = []

    def decide(self, **kwargs: Any) -> ReasonerDecision:
        self.calls.append(dict(kwargs))
        return self.decisions.pop(0)


def test_driver_requires_both_semantic_roles() -> None:
    with pytest.raises(ValueError, match="Reasoner"):
        VirtualVideoMultiRoundDriver(reasoner=None, investigator=FakeInvestigator())
    with pytest.raises(ValueError, match="Investigator"):
        VirtualVideoMultiRoundDriver(reasoner=ScriptedReasoner(()), investigator=None)


def _investigate() -> ReasonerDecision:
    return ReasonerDecision(
        action="investigate",
        tasks=(
            InvestigationTask(
                query_id="inspect_cup",
                goal="Describe what the person raises.",
                segment_id="seg_0001",
                time_range=(5.0, 6.0),
                sampling_floor_fps=1.0,
            ),
        ),
    )


def test_driver_keeps_semantic_authority_with_reasoner(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    attempt_id = _report(_investigate().tasks[0]).attempts[0].attempt_id
    reasoner = ScriptedReasoner(
        (
            _investigate(),
            ReasonerDecision(
                action="answer",
                answer="B. A cup",
                workspace_ops=(
                    {
                        "op": "add_claim",
                        "claim_id": "claim_cup",
                        "text": "The person raises a cup.",
                        "source": "observation",
                        "cites": (attempt_id,),
                        "time_anchor": (5.0, 6.0),
                    },
                ),
                supporting_claim_ids=("claim_cup",),
            ),
        )
    )

    result = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=FakeInvestigator(),
        max_rounds=3,
        max_investigations=4,
    ).run(workspace)

    assert result.answer == "B. A cup"
    assert result.selected_option == "B"
    assert result.reference_valid
    assert result.correct
    assert result.investigation_count == 1
    assert attempt_id in reasoner.calls[1]["working_document_view"]
    assert "The person raises a cup" in reasoner.calls[1]["working_document_view"]
    log_rows = [
        json.loads(line)
        for line in (tmp_path / "observation_log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert log_rows[0]["raw_output"] == '{"summary":"The person raises a cup."}'
    assert json.loads((tmp_path / "working_document.json").read_text(encoding="utf-8"))["claims"]["claim_cup"]


def test_invalid_workspace_reference_is_returned_to_reasoner_without_answer_override(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    attempt_id = _report(_investigate().tasks[0]).attempts[0].attempt_id
    invalid = ReasonerDecision(
        action="answer",
        answer="A. A book",
        workspace_ops=(
            {
                "op": "add_claim",
                "claim_id": "forged",
                "text": "Unsupported observation.",
                "source": "observation",
                "cites": ("attempt_missing",),
            },
        ),
        supporting_claim_ids=("forged",),
    )
    valid = ReasonerDecision(
        action="answer",
        answer="B. A cup",
        workspace_ops=(
            {
                "op": "add_claim",
                "claim_id": "observed",
                "text": "The person raises a cup.",
                "source": "observation",
                "cites": (attempt_id,),
            },
        ),
        supporting_claim_ids=("observed",),
    )
    reasoner = ScriptedReasoner((_investigate(), invalid, valid))

    result = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=FakeInvestigator(),
        max_rounds=3,
        max_investigations=4,
    ).run(workspace)

    assert result.answer == "B. A cup"
    assert "workspace_ops_rejected" in reasoner.calls[2]["working_document_view"]
    assert "attempt_missing" in reasoner.calls[2]["working_document_view"]
    assert not any(row.get("framework_answer_mutation") for row in result.trace)


def test_final_claim_commit_is_available_to_reference_repair(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    attempt_id = _report(_investigate().tasks[0]).attempts[0].attempt_id
    reasoner = ScriptedReasoner(
        (
            _investigate(),
            ReasonerDecision(
                action="answer",
                answer="B. A cup",
                workspace_ops=(
                    {
                        "op": "add_claim",
                        "claim_id": "claim_cup",
                        "text": "The person raises a cup.",
                        "source": "observation",
                        "cites": (attempt_id,),
                        "confidence": "high",
                    },
                ),
            ),
            ReasonerDecision(
                action="answer",
                answer="B. A cup",
                supporting_claim_ids=("claim_cup",),
            ),
        )
    )

    result = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=FakeInvestigator(),
        max_rounds=1,
        max_investigations=1,
    ).run(workspace)

    assert result.answer == "B. A cup"
    assert result.reference_valid
    assert result.investigation_count == 1
    assert len(reasoner.calls) == 3
    assert reasoner.calls[1]["final_attempt"] == 1
    assert reasoner.calls[2]["final_attempt"] == 2
    assert "claim_cup" in reasoner.calls[2]["working_document_view"]
    assert "answer_reference_rejected" in reasoner.calls[2]["working_document_view"]
    assert reasoner.calls[2]["mechanical_status"]["supported_observation_claim_count"] == 1
    assert reasoner.calls[2]["mechanical_status"]["unresolved_observation_count"] == 0
    assert sum(bool(row.get("answer_workspace_commit")) for row in result.trace) == 1


def test_reference_invalid_forced_answer_is_not_returned(tmp_path: Path) -> None:
    invalid = ReasonerDecision(
        action="answer",
        answer="A. A book",
        supporting_claim_ids=("claim_missing",),
    )
    reasoner = ScriptedReasoner((_investigate(), invalid, invalid))

    result = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=FakeInvestigator(),
        max_rounds=1,
        max_investigations=1,
    ).run(_workspace(tmp_path))

    assert result.answer == "No valid answer was returned."
    assert not result.selected_option
    assert not result.answer_present
    assert not result.reference_valid
    assert not result.correct
    assert result.reference_reason == "answer_missing"


def test_benchmark_best_effort_preserves_last_candidate_and_invalid_reference(tmp_path: Path) -> None:
    invalid = ReasonerDecision(
        action="answer",
        answer="A. A book",
        supporting_claim_ids=("claim_missing",),
    )
    reasoner = ScriptedReasoner((_investigate(), invalid, invalid))

    result = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=FakeInvestigator(),
        max_rounds=1,
        max_investigations=1,
        answer_policy="benchmark_best_effort",
    ).run(_workspace(tmp_path))

    assert result.answer == "A. A book"
    assert result.answer_present
    assert result.answer_policy == "benchmark_best_effort"
    assert not result.reference_valid
    assert result.reference_reason == "supporting_claim_missing"
    assert reasoner.calls[-1]["final_attempt"] == 2
    summary = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["answer_present"] is True
    assert summary["answer_policy"] == "benchmark_best_effort"


def test_answer_and_claims_validate_in_one_transaction(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    attempt_id = _report(_investigate().tasks[0]).attempts[0].attempt_id
    late_answer = ReasonerDecision(
        action="answer",
        answer="B. A cup",
        workspace_ops=(
            {
                "op": "add_claim",
                "claim_id": "claim_cup",
                "text": "The person raises a cup.",
                "source": "observation",
                "cites": (attempt_id,),
                "confidence": "high",
            },
        ),
        supporting_claim_ids=("claim_cup",),
    )
    reasoner = ScriptedReasoner(
        (
            _investigate(),
            late_answer,
        )
    )

    result = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=FakeInvestigator(),
        max_rounds=1,
        max_investigations=1,
    ).run(workspace)
    document = json.loads((tmp_path / "working_document.json").read_text(encoding="utf-8"))

    assert result.reference_valid
    assert result.answer == "B. A cup"
    assert "claim_cup" in document["claims"]
    assert len(reasoner.calls) == 2
    assert sum(bool(row.get("answer_workspace_commit")) for row in result.trace) == 1


def test_answer_with_residual_uncertainty_requires_repair(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    attempt_id = _report(_investigate().tasks[0]).attempts[0].attempt_id
    claim = {
        "op": "add_claim",
        "claim_id": "claim_cup",
        "text": "The person raises a cup.",
        "source": "observation",
        "cites": (attempt_id,),
        "confidence": "high",
    }
    reasoner = ScriptedReasoner(
        (
            _investigate(),
            ReasonerDecision(
                action="answer",
                answer="B. A cup",
                workspace_ops=(claim,),
                supporting_claim_ids=("claim_cup",),
                residual_uncertainty="The evidence does not confirm the option's extra detail.",
            ),
            ReasonerDecision(
                action="update_workspace",
            ),
            ReasonerDecision(
                action="answer",
                answer="B. A cup",
                supporting_claim_ids=("claim_cup",),
            ),
        )
    )

    result = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=FakeInvestigator(),
        max_rounds=2,
        max_investigations=4,
    ).run(workspace)

    assert result.reference_valid
    assert result.answer == "B. A cup"
    assert "answer_support_uncertain" in reasoner.calls[2]["working_document_view"]
    assert "B. A cup" in reasoner.calls[2]["working_document_view"]
    assert "does not confirm the option's extra detail" in reasoner.calls[2]["working_document_view"]


def test_task_schema_rejects_old_inspection_modes() -> None:
    with pytest.raises(ValueError, match="unsupported inspection_mode"):
        InvestigationTask(
            query_id="old_mode",
            goal="Inspect a visible action.",
            segment_id="seg_0001",
            inspection_mode="verify_claim",
        )

    assert not hasattr(ReasonerDecision(action="answer"), "option_verdicts")


def test_free_form_answer_keeps_reference_gate_but_allows_uncertainty(tmp_path: Path) -> None:
    multiple_choice = _workspace(tmp_path)
    workspace = VirtualVideoWorkspace.create(
        tmp_path / "free-form",
        manifest=multiple_choice.manifest,
        case=VirtualVideoCase(
            case_id="free-form-case",
            question="What does the person raise?",
            options={},
            gold="The person raises a cup.",
            target_segment_id="seg_0001",
            target_virtual_interval=(0.0, 20.0),
        ),
    )
    attempt_id = _report(_investigate().tasks[0]).attempts[0].attempt_id
    reasoner = ScriptedReasoner(
        (
            _investigate(),
            ReasonerDecision(
                action="answer",
                answer="A cup.",
                workspace_ops=(
                    {
                        "op": "add_claim",
                        "claim_id": "claim_cup",
                        "text": "The person raises a cup.",
                        "source": "observation",
                        "cites": (attempt_id,),
                        "time_anchor": (5.0, 6.0),
                    },
                ),
                supporting_claim_ids=("claim_cup",),
                residual_uncertainty="The cup's material is unclear.",
            ),
        )
    )

    result = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=FakeInvestigator(),
        max_rounds=3,
        max_investigations=4,
    ).run(workspace)

    assert result.answer == "A cup."
    assert result.selected_option == ""
    assert result.reference_valid
    assert result.reference_reason == "reference_integrity_verified"
    assert not result.correct


def test_mechanical_status_exposes_unconfirmed_caption_candidate(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    log_path = tmp_path / "candidate-observations.jsonl"
    from vcah.workspace import ObservationLog, WorkingDocument

    log = ObservationLog(log_path)
    pointer = "caption-search://case-1/index/query"
    attempt_id = stable_attempt_id(
        source_video_ids=("video-a",),
        frame_refs=(pointer,),
        modality="caption_search",
    )
    log.append_attempt(
        ObservationAttempt(
            attempt_id=attempt_id,
            sampling_config={
                "mode": "search_caption",
                "modality": "caption_search",
                "hits": [
                    {"passage_id": "p1", "range": [5.0, 10.0], "score": 0.9},
                    {"passage_id": "p2", "range": [12.0, 15.0], "score": 0.8},
                ],
            },
            frame_refs=(pointer,),
            modality="caption_search",
            evidence_role="candidate",
            source_video_ids=("video-a",),
        ),
        round_id=1,
    )

    status = _mechanical_status(
        workspace,
        WorkingDocument.with_question_premise(workspace.case.question),
        log,
    )

    assert status["pending_caption_candidate_count"] == 2
    assert status["pending_caption_candidates"][0]["time_range"] == [5.0, 10.0]
    assert any("locator candidates" in hint for hint in status["prompt_hints"])

    visual_attempt_id = stable_attempt_id(
        source_video_ids=("video-a",),
        frame_times=(6.0,),
        sampling_fps=1.0,
        modality="visual",
    )
    log.append_attempt(
        ObservationAttempt(
            attempt_id=visual_attempt_id,
            inspected_ranges=((5.0, 10.0),),
            attached_frame_times=(6.0,),
            sampling_config={"mode": "window", "modality": "visual", "fps": 1.0},
            frame_refs=("frame-6.jpg",),
            modality="visual",
            source_video_ids=("video-a",),
        ),
        round_id=2,
    )
    confirmed = _mechanical_status(
        workspace,
        WorkingDocument.with_question_premise(workspace.case.question),
        log,
    )

    assert confirmed["pending_caption_candidate_count"] == 1
    assert confirmed["pending_caption_candidates"][0]["time_range"] == [12.0, 15.0]


def test_reasoner_can_inspect_pending_caption_candidate_within_round_budget(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    visual_attempt_id = _report(_investigate().tasks[0]).attempts[0].attempt_id
    search = ReasonerDecision(
        action="investigate",
        tasks=(
            InvestigationTask(
                query_id="caption-search",
                goal="Locate the raised object.",
                inspection_mode="search_caption",
                caption_queries=("person raises a cup",),
            ),
        ),
    )
    reasoner = ScriptedReasoner(
        (
            search,
            ReasonerDecision(
                action="investigate",
                tasks=(
                    InvestigationTask(
                        query_id="selected-caption-window",
                        goal="Inspect the relevant Caption candidate.",
                        inspection_mode="window",
                        time_range=(5.0, 10.0),
                        sampling_floor_fps=1.0,
                    ),
                ),
            ),
            ReasonerDecision(
                action="answer",
                answer="B. A cup",
                workspace_ops=(
                    {
                        "op": "add_claim",
                        "claim_id": "claim_cup",
                        "text": "The person raises a cup.",
                        "source": "observation",
                        "cites": (visual_attempt_id,),
                        "confidence": "high",
                    },
                ),
                supporting_claim_ids=("claim_cup",),
            ),
            ReasonerDecision(
                action="answer",
                answer="B. A cup",
                supporting_claim_ids=("claim_cup",),
            ),
        )
    )
    investigator = CaptionThenVisualInvestigator()

    result = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=investigator,
        max_rounds=2,
        max_investigations=4,
    ).run(workspace)

    assert result.reference_valid
    assert result.investigation_count == 2
    assert [task.inspection_mode for task in investigator.tasks] == ["search_caption", "window"]
    assert len(reasoner.calls) == 3
    assert reasoner.calls[1]["force_finalize"] is False
    assert any(row.get("answer_workspace_commit") for row in result.trace)


def test_pending_caption_candidate_is_not_automatically_selected(tmp_path: Path) -> None:
    search = ReasonerDecision(
        action="investigate",
        tasks=(
            InvestigationTask(
                query_id="caption-search",
                goal="Locate the raised object.",
                inspection_mode="search_caption",
                caption_queries=("person raises a cup",),
            ),
        ),
    )
    reasoner = ScriptedReasoner(
        (
            search,
            ReasonerDecision(action="update_workspace"),
                ReasonerDecision(
                    action="answer",
                    answer="A. A book",
                    supporting_claim_ids=("claim_missing",),
                ),
            )
        )
    investigator = CaptionThenVisualInvestigator()

    result = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=investigator,
        max_rounds=1,
        max_investigations=4,
        answer_policy="benchmark_best_effort",
    ).run(_workspace(tmp_path))

    assert result.answer_present
    assert [task.inspection_mode for task in investigator.tasks] == ["search_caption"]
    assert not any(row.get("decision_source") == "reasoner_caption_followup" for row in result.trace)
