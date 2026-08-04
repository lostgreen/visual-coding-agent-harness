from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from vcah.evidence_runtime import (
    EvidencePlan,
    RuntimeEvidenceCatalog,
    compile_evidence_plan,
)
from vcah.evidence_state import InterpretationItem
from vcah.investigator import InvestigationReport, ObservationAttempt
from vcah.multiround import (
    InvestigationTask,
    ReasonerDecision,
    VirtualVideoMultiRoundDriver,
    _resolve_runtime_decision,
    _resolve_tasks,
)
from vcah.runtime_metrics import agent_run_metrics
from vcah.types import CoverageSegment, EvidenceRecord
from vcah.virtual_video import (
    VirtualVideoCase,
    VirtualVideoManifest,
    VirtualVideoSegment,
    VirtualVideoWorkspace,
)
from vcah.workspace import ObservationLog, WorkingDocument, stable_attempt_id


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
        manifest=VirtualVideoManifest(workspace_id="case-runtime", segments=(segment,)),
        case=VirtualVideoCase(
            case_id="case-runtime",
            question="After the stated event, what does the person raise?",
            options={"A": "A book", "B": "A cup"},
            gold="B",
            target_segment_id=segment.segment_id,
            target_virtual_interval=(0.0, 20.0),
        ),
    )


def _point_attempt(task: InvestigationTask) -> ObservationAttempt:
    attempt_id = stable_attempt_id(
        source_video_ids=("video-a",),
        frame_times=(5.0, 6.0),
        sampling_fps=1.0,
        modality="visual",
    )
    return ObservationAttempt(
        attempt_id=attempt_id,
        task_id=task.query_id,
        requested_range=(5.0, 6.0),
        inspected_ranges=((5.0, 6.0),),
        attached_frame_times=(5.0, 6.0),
        sampling_config={
            "fps": 1.0,
            "modality": "visual",
            "evidence_kind": task.evidence_kind,
        },
        images_requested=2,
        images_attached=2,
        parse_status="parsed",
        execution_status="completed",
        frame_refs=("f5.jpg", "f6.jpg"),
        prompt_digest="prompt",
        raw_output='{"summary":"The person raises a cup."}',
        source_video_ids=("video-a",),
        evidence_role="supporting",
        interpretation_purpose=task.interpretation_purpose,
        interpretation_items=(
            InterpretationItem(
                item_id=f"item_{task.query_id}",
                time_anchor=(5.0, 5.0),
                text="The person visibly raises a cup.",
            ),
        ),
    )


class RuntimeReasoner:
    def __init__(self, decisions: Sequence[ReasonerDecision]) -> None:
        self.decisions = list(decisions)
        self.calls: list[dict[str, Any]] = []
        self.plan_calls = 0

    def plan_evidence(self, **kwargs: Any) -> dict[str, Any]:
        self.plan_calls += 1
        return {
            "requirements": [
                {
                    "name": "raised_object",
                    "goal": "Read the exact raised object from the visible scene.",
                    "kind": "text_exact",
                    "role": "answer_bearing",
                }
            ]
        }

    def decide(self, **kwargs: Any) -> ReasonerDecision:
        self.calls.append(dict(kwargs))
        return self.decisions.pop(0)


class ItemInvestigator:
    def __init__(self) -> None:
        self.tasks: list[InvestigationTask] = []

    def reset_run_state(self) -> None:
        self.tasks.clear()

    def run_batch(self, tasks: Sequence[InvestigationTask]) -> tuple[InvestigationReport, ...]:
        self.tasks.extend(tasks)
        reports = []
        for task in tasks:
            attempt = _point_attempt(task)
            evidence = EvidenceRecord(
                evidence_id=f"ev_{task.query_id}",
                beat_id="",
                start_sec=5.0,
                end_sec=6.0,
                modality="visual",
                pointer=f"virtual://case-runtime/observations/{attempt.attempt_id}",
                verbatim="The person raises a cup.",
                frame_refs=("f5.jpg", "f6.jpg"),
                attestation_model="fake-investigator",
                temporal_scope="window",
                evidence_kind="visual_observation",
                request_ids=(task.query_id,),
                coverage_manifest=(
                    CoverageSegment(task.query_id, 5.0, 6.0, "visual", 1.0),
                ),
                task_id=task.query_id,
                observation_id=attempt.attempt_id,
                sampling_fps=1.0,
                source_lineage=({"source_video_id": "video-a"},),
            )
            reports.append(
                InvestigationReport(
                    query_id=task.query_id,
                    status="completed",
                    evidence=(evidence,),
                    attempts=(attempt,),
                    cost={"consumes_budget": True},
                    coverage_delta=((5.0, 6.0),),
                )
            )
        return tuple(reports)


def test_plan_compiler_separates_premise_from_answer_requirement(tmp_path: Path) -> None:
    plan = EvidencePlan.from_mapping(
        {
            "requirements": [
                {
                    "name": "poison_event",
                    "goal": "Use the stated poisoning event as location context.",
                    "role": "premise",
                },
                {
                    "name": "body_change",
                    "goal": "Observe the visible body change after that event.",
                    "kind": "persistent_state",
                    "role": "answer_bearing",
                    "depends_on": ["poison_event"],
                    "dependency_type": "locator",
                },
            ]
        },
        question="After the player is poisoned, what changes?",
    )
    document = WorkingDocument.with_question_premise("question")

    compilation = compile_evidence_plan(document, plan, question="question")
    catalog = RuntimeEvidenceCatalog.build(
        document,
        ObservationLog(tmp_path / "empty_observations.jsonl"),
    )

    assert compilation["compiled"] is True
    assert [row[0] for row in catalog.requirements] == ["R1", "R2"]
    premise_id = catalog.resolve_requirement("R1")
    answer_id = catalog.resolve_requirement("R2")
    assert document.obligations[premise_id].role == "premise"
    assert not document.obligations[premise_id].answer_bearing
    assert document.obligations[answer_id].answer_bearing
    assert document.obligations[answer_id].dependency_type == "locator"
    assert document.obligations[answer_id].depends_on == (premise_id,)


def test_plan_compiler_canonicalizes_duplicate_names_without_overwrite() -> None:
    plan = EvidencePlan.from_mapping(
        {
            "requirements": [
                {"name": "event", "goal": "Locate the first event.", "role": "locator"},
                {
                    "name": "event",
                    "goal": "Observe what follows the event.",
                    "role": "answer_bearing",
                    "depends_on": ["event"],
                },
            ]
        },
        question="What follows?",
    )
    document = WorkingDocument.with_question_premise("question")

    compilation = compile_evidence_plan(document, plan, question="question")

    assert compilation["requirement_count"] == 2
    assert [spec.name for spec in plan.requirements] == ["event", "event_2"]
    obligations = tuple(document.obligations.values())
    assert obligations[1].depends_on == (obligations[0].requirement_id,)


def test_catalog_exposes_short_handles_and_caps_refinable_items(tmp_path: Path) -> None:
    document = WorkingDocument.with_question_premise("question")
    compile_evidence_plan(document, EvidencePlan.fallback("question"), question="question")
    observation_log = ObservationLog(tmp_path / "observations.jsonl")
    frame_times = tuple(float(index) for index in range(1, 8))
    attempt = ObservationAttempt(
        attempt_id=stable_attempt_id(
            source_video_ids=("video-a",),
            frame_times=frame_times,
            sampling_fps=1.0,
            modality="visual",
        ),
        task_id="inspect",
        requested_range=(1.0, 7.0),
        inspected_ranges=((1.0, 7.0),),
        attached_frame_times=frame_times,
        sampling_config={
            "fps": 1.0,
            "modality": "visual",
            "evidence_kind": "transient_event",
        },
        frame_refs=tuple(f"f{index}.jpg" for index in range(1, 8)),
        source_video_ids=("video-a",),
        interpretation_items=tuple(
            InterpretationItem(
                item_id=f"item_{index}",
                time_anchor=(float(index), float(index)),
                text=f"Visible item {index}",
                item_kind="event",
            )
            for index in range(1, 8)
        ),
    )
    observation_log.append_attempt(attempt, round_id=1)

    catalog = RuntimeEvidenceCatalog.build(document, observation_log)
    rendered = catalog.render(document)

    assert catalog.resolve_requirement("R1").startswith("requirement_")
    assert catalog.resolve_item("E1") is not None
    assert sum(item.advertise_refinement for item in catalog.items) == 6
    assert all(item.refinable for item in catalog.items)
    assert "R1" in rendered and "E1" in rendered and "M1" in rendered
    assert "requirement_" not in rendered
    assert "attempt_" not in rendered


def test_runtime_derived_driver_closes_text_requirement_from_item_support(tmp_path: Path) -> None:
    reasoner = RuntimeReasoner(
        (
            ReasonerDecision(
                action="investigate",
                tasks=(
                    InvestigationTask(
                        query_id="inspect_text",
                        goal="Inspect the raised object.",
                        time_range=(5.0, 6.0),
                        requirement_id="R1",
                    ),
                ),
            ),
            ReasonerDecision(
                action="answer",
                answer="B. A cup",
                supporting_item_ids=("E1",),
                supports_requirement_ids=("R1",),
            ),
        )
    )
    investigator = ItemInvestigator()

    result = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=investigator,
        max_rounds=2,
        max_investigations=4,
        require_obligation_coverage=True,
        require_item_provenance=True,
        require_evidence_kind_requirements=True,
        answer_policy="benchmark_best_effort",
        evidence_control_mode="strict",
        evidence_state_mode="runtime_derived",
    ).run(_workspace(tmp_path))

    assert reasoner.plan_calls == 1
    assert result.answer == "B. A cup"
    assert result.answer_present and result.reference_valid
    assert result.verification_status == "verified"
    assert len(result.supporting_item_ids) == 1
    assert result.supporting_item_ids[0].startswith("item_")
    assert len(result.supporting_attempt_ids) == 1
    assert [task.interpretation_purpose for task in investigator.tasks] == [
        "primary",
        "manual_reread",
    ]
    assert investigator.tasks[0].evidence_kind == "text_exact"
    assert "E1" in reasoner.calls[1]["working_document_view"]
    assert "attempt_" not in reasoner.calls[1]["working_document_view"]
    document = json.loads((tmp_path / "working_document.json").read_text(encoding="utf-8"))
    states = tuple(document["obligation_states"].values())
    assert len(states) == 1 and states[0]["status"] == "supported"
    rows = tuple(
        json.loads(line)
        for line in (tmp_path / "observation_log.jsonl").read_text(encoding="utf-8").splitlines()
    )
    metrics = agent_run_metrics(
        result.trace,
        rows,
        answer_present=result.answer_present,
        reference_valid=result.reference_valid,
        supporting_intervals=result.supporting_intervals,
    )
    assert metrics["state_mutation_op_count"] == 0
    assert metrics["decision_repair_count"] == 0


def test_refine_item_resolves_directly_to_child_refinement(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    document = WorkingDocument.with_question_premise("question")
    compile_evidence_plan(document, EvidencePlan.fallback("question"), question="question")
    observation_log = ObservationLog(tmp_path / "catalog.jsonl")
    observation_log.append_attempt(
        _point_attempt(
            InvestigationTask(
                query_id="parent",
                goal="Inspect a point.",
                time_range=(5.0, 6.0),
            )
        ),
        round_id=1,
    )
    catalog = RuntimeEvidenceCatalog.build(document, observation_log)
    decision, errors, _ = _resolve_runtime_decision(
        ReasonerDecision(
            action="investigate",
            tasks=(
                InvestigationTask(
                    query_id="refine",
                    goal="Refine the visible transition.",
                    refine_item_id="E1",
                    requirement_id="R1",
                ),
            ),
        ),
        catalog,
        document,
    )

    resolved = _resolve_tasks(
        workspace,
        decision.tasks,
        limit=1,
        observation_rows=observation_log.rows,
        cue_states={},
    )

    assert errors == []
    assert len(resolved) == 1
    assert resolved[0].cue_stage == "child_refinement"
    assert resolved[0].interpretation_purpose == "manual_reread"
    assert resolved[0].time_range == (0.0, 10.0)
