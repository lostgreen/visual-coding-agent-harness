from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from PIL import Image

import vcah.multiround as multiround
from vcah.evidence_primitives import ConditionResult
from vcah.multiround import EvidenceGap, InvestigationTask, ReasonerDecision, VirtualVideoMultiRoundDriver
from vcah.investigator import InvestigationReport, VirtualVideoInvestigator
from vcah.types import CoverageSegment, EvidenceRecord, Frame
from vcah.virtual_index import build_virtual_beat_index
from vcah.virtual_video import (
    VirtualVideoCase,
    VirtualVideoManifest,
    VirtualVideoSegment,
    VirtualVideoWorkspace,
    materialize_lowfps_frame_cache,
)


def _sampler(video_path: str, start_sec: float, end_sec: float, n_frames: int, out_dir: Path) -> tuple[Frame, ...]:
    del end_sec
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(max(1, int(n_frames))):
        time_sec = round(float(start_sec) + index * 0.5, 3)
        path = out_dir / f"{Path(video_path).stem}_{time_sec:.3f}_{index:03d}.jpg"
        Image.new("RGB", (32, 18), color=(20, 40, 230)).save(path)
        frames.append(Frame(frame_id=f"fr{index:03d}", time_sec=time_sec, path=str(path)))
    return tuple(frames)


class TinyModel:
    embedding_dim = 1
    embed_model = "tiny"
    allow_placeholder_visual = True

    def embed_image(self, paths: Sequence[str]):
        import numpy as np

        return np.ones((len(paths), 1), dtype=np.float32)

    def embed_text(self, queries: Sequence[str]):
        import numpy as np

        return np.ones((len(queries), 1), dtype=np.float32)


def test_primary_gap_is_bound_to_parallel_investigation_tasks() -> None:
    decision = ReasonerDecision(
        action="investigate",
        primary_gap=EvidenceGap(
            gap_id="gap_transition",
            description="The visible state transition boundary.",
            success_conditions=("observe the before state", "observe the after state"),
        ),
        tasks=(
            InvestigationTask("q_before", "Inspect before the transition."),
            InvestigationTask("q_after", "Inspect after the transition."),
        ),
    )

    bound = multiround._bind_gap_to_tasks(decision)

    assert {task.gap_id for task in bound.tasks} == {"gap_transition"}
    assert all(task.success_conditions == decision.primary_gap.success_conditions for task in bound.tasks)
    assert all(task.conditions == decision.primary_gap.conditions for task in bound.tasks)


def test_query_compiler_distinguishes_scalar_quantity_from_entity_count() -> None:
    scalar = multiround.compile_query_contract(
        "How many calories had he consumed by the time he met his teammate?"
    )
    distance = multiround.compile_query_contract("How many light-years wide is the observable universe?")
    option_unit_distance = multiround.compile_query_contract(
        "What total diameter does the video state?",
        {"A": "About 100 trillion lightyears", "C": "Over 25 trillion lightyears"},
    )
    distance_with_duration = multiround.compile_query_contract(
        "How many meters did they complete in 25 minutes?"
    )
    entities = multiround.compile_query_contract(
        "Throughout the video, how many scholars comment on Napoleon?"
    )
    events = multiround.compile_query_contract("How many times does the title card appear?")
    timed_events = multiround.compile_query_contract(
        "How many times does the title card appear in the first 5 minutes?"
    )
    timed_entities = multiround.compile_query_contract(
        "How many people appear in the first 5 minutes?"
    )

    assert scalar.quantifier == "scalar_quantity"
    assert scalar.aggregation == "accumulate"
    assert scalar.measurement_unit == "calorie"
    assert scalar.boundary_hint == "by the time he met his teammate"
    assert distance.quantifier == "scalar_quantity"
    assert distance.measurement_unit == "light_year"
    assert option_unit_distance.quantifier == "scalar_quantity"
    assert option_unit_distance.measurement_unit == "light_year"
    assert distance_with_duration.measurement_unit == "meter"
    assert entities.quantifier == "distinct_count"
    assert events.quantifier == "total_count"
    assert timed_events.quantifier == "total_count"
    assert timed_entities.quantifier == "distinct_count"


def test_new_coverage_does_not_count_as_goal_progress() -> None:
    condition = ConditionResult("gap_clock_c1", "unknown", "The scoreboard is too small.")
    report = InvestigationReport(
        query_id="q_clock",
        status="satisfied",
        gap_id="gap_clock",
        resolution="unresolved",
        condition_results=(condition,),
        coverage_delta=((10.0, 20.0),),
    )

    annotated = multiround._annotate_batch_progress((report,), ())

    assert annotated[0].goal_progress == ()
    assert annotated[0].coverage_progress == ("new_frontier_coverage",)


def test_investigation_task_validation_requires_executable_locator() -> None:
    valid_window = InvestigationTask("q_window", "Inspect candidate.", segment_id="seg_1")
    valid_search = InvestigationTask(
        "q_search",
        "Search transcript.",
        inspection_mode="search_asr",
        search_terms=("professor",),
    )
    meta_task = InvestigationTask("q_meta", "Deduplicate and count all people.")

    assert multiround._task_is_executable(valid_window) is True
    assert multiround._task_is_executable(valid_search) is True
    assert multiround._task_is_executable(meta_task) is False


def test_entity_candidate_repair_uses_witness_virtual_timestamp(tmp_path: Path) -> None:
    workspace = _two_chunk_workspace(tmp_path)
    candidate_id = "broad:person_1"
    candidate = EvidenceRecord(
        evidence_id="ev_candidate",
        beat_id="",
        start_sec=10.0,
        end_sec=15.0,
        modality="visual",
        pointer="virtual://candidate",
        verbatim="A white-haired woman appears in a coarse scan.",
        frame_refs=("candidate.jpg",),
        attestation_model="test-vlm",
        source_lineage=(
            {"segment_id": "seg_target_b", "source_video_id": "target", "virtual_time_range": [10.0, 15.0]},
        ),
        operation_metadata={
            "structured_parse_status": "parsed",
            "entities": [
                {
                    "entity_observation_id": candidate_id,
                    "description": "older woman with white hair",
                    "visual_signature": "short white hair and blue top",
                    "witness_virtual_times_sec": [12.0],
                    "candidate_only": True,
                    "countable": False,
                }
            ],
        },
    )

    tasks = multiround._entity_candidate_repair_tasks(
        workspace,
        (candidate,),
        (candidate_id,),
        round_id=2,
        limit=4,
    )

    assert len(tasks) == 1
    assert tasks[0].segment_id == "seg_target_b"
    assert tasks[0].time_range == (10.0, 15.0)
    assert tasks[0].source_candidate_ids == (candidate_id,)
    assert tasks[0].inspection_intent == "entity_candidate_verification"


def test_distinct_count_readiness_tracks_unresolved_and_resolved_candidates(tmp_path: Path) -> None:
    workspace = _two_chunk_workspace(tmp_path)
    contract = multiround.compile_query_contract(workspace.case.question, workspace.case.options)
    candidate_id = "broad:person_1"
    candidate = EvidenceRecord(
        evidence_id="ev_candidate",
        beat_id="",
        start_sec=0.0,
        end_sec=180.0,
        modality="visual",
        pointer="virtual://candidate",
        verbatim="A candidate person appears in a coarse scan.",
        frame_refs=("candidate.jpg",),
        attestation_model="test-vlm",
        operation_metadata={
            "structured_parse_status": "parsed",
            "entities": [
                {
                    "entity_observation_id": candidate_id,
                    "candidate_only": True,
                    "countable": False,
                    "witness_virtual_times_sec": [12.0],
                }
            ],
        },
    )
    verified = EvidenceRecord(
        evidence_id="ev_verified",
        beat_id="",
        start_sec=10.0,
        end_sec=15.0,
        modality="visual",
        pointer="virtual://verified",
        verbatim="The candidate is verified in a narrow window.",
        frame_refs=("verified.jpg",),
        attestation_model="test-vlm",
        operation_metadata={
            "structured_parse_status": "parsed",
            "source_candidate_ids": [candidate_id],
            "entities": [],
        },
    )

    unresolved = multiround._apply_entity_completion({}, contract, (candidate,))
    resolved = multiround._apply_entity_completion({}, contract, (candidate, verified))

    assert unresolved["unresolved_candidate_entity_observation_ids"] == [candidate_id]
    assert unresolved["ready_for_answer"] is False
    assert resolved["unresolved_candidate_entity_observation_ids"] == []
    assert multiround._entity_census_coverage_evidence((candidate, verified)) == (verified,)


def test_evidence_digest_serializes_structured_condition_results() -> None:
    evidence = EvidenceRecord(
        evidence_id="ev_condition",
        beat_id="",
        start_sec=1.0,
        end_sec=2.0,
        modality="visual",
        pointer="virtual://condition",
        verbatim="The target remains unreadable.",
        frame_refs=("frame.jpg",),
        attestation_model="test-vlm",
        operation_metadata={
            "investigation": {
                "resolution": "unresolved",
                "condition_results": (
                    ConditionResult("gap_text_c1", "unknown", "Text remains unreadable."),
                ),
            }
        },
    )

    payload = multiround._evidence_digest((evidence,))

    json.dumps(payload)
    assert payload[0]["investigation"]["condition_results"][0]["condition_id"] == "gap_text_c1"


def test_scalar_quantity_gate_uses_boundary_aware_measurement_derivation(tmp_path: Path) -> None:
    manifest = VirtualVideoManifest(
        workspace_id="scalar",
        segments=(VirtualVideoSegment("seg_1", "source", "source.mp4", 0.0, 60.0, 0.0, 60.0, "content"),),
    )
    case = VirtualVideoCase(
        case_id="scalar",
        question="How many calories had he consumed by the time he met his teammate?",
        options={"A": "500 calories", "B": "700 calories"},
        gold="B",
        target_segment_id="seg_1",
        target_virtual_interval=(0.0, 60.0),
    )
    workspace = VirtualVideoWorkspace.create(tmp_path / "scalar", manifest=manifest, case=case)
    evidence = tuple(
        EvidenceRecord(
            evidence_id=f"ev_measure_{index}",
            beat_id="",
            start_sec=float(index * 10),
            end_sec=float(index * 10 + 5),
            modality="visual",
            pointer=f"virtual://measure/{index}",
            verbatim=summary,
            frame_refs=(f"frame_{index}.jpg",),
            attestation_model="test-vlm",
            coverage_manifest=(CoverageSegment(f"q_{index}", float(index * 10), float(index * 10 + 5), "visual", 1.0),),
            operation_metadata={
                "measurements": [
                    {
                        "value": value,
                        "unit": "calorie",
                        "measurement_semantics": "delta",
                        "subject_id": subject,
                        "source_time_sec": float(index * 10),
                        "boundary_relation": boundary,
                    }
                ]
            },
        )
        for index, (value, subject, boundary, summary) in enumerate(
            (
                (300, "meal_1", "before", "The first consumed item is shown."),
                (400, "meal_2", "at", "A second consumed item is shown at the meeting boundary."),
                (100, "meal_3", "after", "Another item appears only after the meeting."),
            ),
            start=1,
        )
    )
    contract = multiround.compile_query_contract(case.question)
    digest = multiround._evidence_digest(evidence[:1])
    assert digest[0]["measurements"][0]["value"] == 300

    gate = multiround._answer_completion_gate(
        workspace,
        contract,
        "B. 700 calories",
        tuple(record.evidence_id for record in evidence),
        (),
        evidence,
    )

    assert gate["passed"] is True
    assert gate["reason"] == "scalar_quantity_grounded"
    assert gate["derivation"]["operator"] == "sum_delta"
    assert gate["derivation"]["result"] == 700
    assert gate["derivation"]["evidence_ids"] == ["ev_measure_1", "ev_measure_2"]
    wrong = multiround._answer_completion_gate(
        workspace,
        contract,
        "A. 500 calories",
        tuple(record.evidence_id for record in evidence),
        (),
        evidence,
    )
    assert wrong["passed"] is False
    assert wrong["reason"] == "scalar_quantity_answer_mismatch"

    aggregate = multiround._derived_answer_evidence(
        workspace,
        answer="B. 700 calories",
        citations=tuple(record.evidence_id for record in evidence[:2]),
        entity_clusters=(),
        evidence=evidence,
        derivation=gate["derivation"],
    )
    assert aggregate.operation_metadata["derivation"]["result"] == 700


def test_repeated_partial_attempts_emit_soft_stagnation_warning() -> None:
    evidence = EvidenceRecord(
        evidence_id="ev_repeat",
        beat_id="",
        start_sec=10.0,
        end_sec=20.0,
        modality="visual",
        pointer="virtual://repeat",
        verbatim="The requested small text remains unreadable.",
        frame_refs=("frame.jpg",),
        attestation_model="test",
    )
    reports = tuple(
        InvestigationReport(
            query_id=f"q_{index}",
            status="satisfied",
            evidence=(evidence,),
            gap_id="gap_text",
            resolution="partial",
            unresolved_conditions=("read the final displayed value",),
        )
        for index in range(2)
    )

    status = multiround._stagnation_status(reports)

    assert status["stagnant"] is True
    assert status["gap_id"] == "gap_text"
    assert "change range" in status["required_shift"]


class ScriptedReasoner:
    def __init__(self) -> None:
        self.calls = 0
        self.kwargs: list[dict[str, object]] = []

    def decide(self, **kwargs: object) -> ReasonerDecision:
        self.calls += 1
        self.kwargs.append(dict(kwargs))
        if self.calls == 1:
            return ReasonerDecision(
                action="investigate",
                tasks=tuple(
                    InvestigationTask(
                        query_id=f"q{i}",
                        goal="Read the number on the jersey.",
                        segment_id="seg_target",
                        time_range=None,
                        modality_hint=("visual", "ocr"),
                        expected_evidence="number written on jersey",
                    )
                    for i in range(6)
                ),
            )
        evidence_digest = tuple(kwargs.get("evidence_digest", ()) or ())
        return ReasonerDecision(
            action="answer",
            answer="B. 11",
            citations=(str(evidence_digest[0]["evidence_id"]),),
        )


class CoverageReasoner:
    def __init__(self) -> None:
        self.calls = 0
        self.completion_statuses: list[dict[str, object]] = []

    def decide(self, **kwargs: object) -> ReasonerDecision:
        self.calls += 1
        self.completion_statuses.append(dict(kwargs.get("completion_status", {}) or {}))
        if self.calls == 1:
            return ReasonerDecision(
                action="investigate",
                tasks=(
                    InvestigationTask(
                        query_id="q_chunk_a",
                        goal="Identify scholars commenting on Napoleon.",
                        segment_id="seg_target_a",
                        modality_hint=("visual",),
                        expected_evidence="distinct scholars discussing Napoleon",
                    ),
                ),
            )
        if self.calls == 2:
            return ReasonerDecision(action="answer", answer="B. Three", citations=("ev_q_chunk_a_001",))
        evidence_digest = tuple(kwargs.get("evidence_digest", ()) or ())
        citations = tuple(
            str(item["evidence_id"])
            for item in evidence_digest
            if str(item.get("modality")) == "visual"
        )
        return ReasonerDecision(
            action="answer",
            answer="B. Three",
            citations=citations,
            entity_clusters=(
                {"entity_id": "scholar_1", "description": "bald man with glasses", "evidence_ids": ("ev_q_chunk_a_001",)},
                {
                    "entity_id": "scholar_2",
                    "description": "older woman with white hair",
                    "evidence_ids": citations,
                },
                {"entity_id": "scholar_3", "description": "brown-haired man", "evidence_ids": (citations[-1],)},
            ),
        )


class MissingEntityClustersReasoner:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, **kwargs: object) -> ReasonerDecision:
        self.calls += 1
        if self.calls == 1:
            return ReasonerDecision(
                action="investigate",
                tasks=(
                    InvestigationTask("q_a", "Inspect first target chunk.", "seg_target_a", modality_hint=("visual",)),
                    InvestigationTask("q_b", "Inspect second target chunk.", "seg_target_b", modality_hint=("visual",)),
                ),
            )
        return ReasonerDecision(
            action="answer",
            answer="B. Three",
            citations=("ev_q_a_001", "ev_q_b_001"),
        )


class EmptyAnswerReasoner:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, **kwargs: object) -> ReasonerDecision:
        self.calls += 1
        if self.calls == 1:
            return ReasonerDecision(
                action="investigate",
                tasks=(InvestigationTask("q1", "Inspect the relevant scene.", "seg_target"),),
            )
        return ReasonerDecision(action="answer", answer="", citations=("ev_q1_001",))


class FinalizationReasoner:
    def __init__(self) -> None:
        self.calls = 0
        self.force_flags: list[bool] = []

    def decide(self, **kwargs: object) -> ReasonerDecision:
        self.calls += 1
        self.force_flags.append(bool(kwargs.get("force_finalize")))
        if self.calls == 1:
            return ReasonerDecision(
                action="investigate",
                tasks=(InvestigationTask("q1", "Inspect the relevant scene.", "seg_target"),),
            )
        evidence_digest = tuple(kwargs.get("evidence_digest", ()) or ())
        return ReasonerDecision(
            action="answer",
            answer="B. 11",
            citations=(str(evidence_digest[-1]["evidence_id"]),),
        )


class UnsupportedFinalizationReasoner(FinalizationReasoner):
    def decide(self, **kwargs: object) -> ReasonerDecision:
        decision = super().decide(**kwargs)
        if decision.action != "answer":
            return decision
        return ReasonerDecision(
            action="answer",
            answer=decision.answer,
            citations=decision.citations,
            support_status="insufficient",
            support_reason="The observation is related but does not establish the proposed causal claim.",
        )


class RankedCandidateReasoner:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, **kwargs: object) -> ReasonerDecision:
        del kwargs
        self.calls += 1
        if self.calls == 1:
            return ReasonerDecision(
                action="investigate",
                tasks=(InvestigationTask("q1", "Inspect the relevant scene.", "seg_target"),),
            )
        if self.calls == 2:
            return ReasonerDecision(
                action="answer",
                answer="B. 11",
                citations=(),
                support_status="supported",
                support_reason="The observed jersey directly shows 11.",
            )
        return ReasonerDecision(
            action="answer",
            answer="A. 7",
            citations=(),
            support_status="contradicted",
            support_reason="The later candidate conflicts with the observation.",
        )


class AlwaysEmptyReasoner:
    def decide(self, **kwargs: object) -> ReasonerDecision:
        del kwargs
        return ReasonerDecision(action="answer", answer="", citations=())


class NegativeAnchorInvestigator:
    def __init__(self, workspace: VirtualVideoWorkspace) -> None:
        self.workspace = workspace
        self.tasks: list[InvestigationTask] = []

    def reset_run_state(self) -> None:
        self.tasks.clear()

    def run_batch(self, tasks: Sequence[InvestigationTask]) -> tuple[InvestigationReport, ...]:
        reports = []
        by_id = {segment.segment_id: segment for segment in self.workspace.manifest.segments}
        for task in tasks:
            self.tasks.append(task)
            segment = by_id[task.segment_id]
            evidence = EvidenceRecord(
                evidence_id=f"ev_{task.query_id}_001",
                beat_id="",
                start_sec=segment.virtual_start_sec,
                end_sec=segment.virtual_end_sec,
                modality="visual",
                pointer=f"virtual://negative/{task.segment_id}",
                verbatim="No matching person is visible in this segment.",
                frame_refs=(f"{task.segment_id}.jpg",),
                attestation_model="negative-anchor-test",
                evidence_kind="visual_observation",
                coverage_manifest=(
                    CoverageSegment(
                        task.query_id,
                        segment.virtual_start_sec,
                        segment.virtual_end_sec,
                        "visual",
                        1.0,
                    ),
                ),
                source_lineage=(
                    {
                        "segment_id": segment.segment_id,
                        "source_video_id": segment.source_video_id,
                        "source_time_range": [segment.source_start_sec, segment.source_end_sec],
                        "virtual_time_range": [segment.virtual_start_sec, segment.virtual_end_sec],
                    },
                ),
                operation_metadata={"supports_identity_anchor": False},
            )
            reports.append(
                InvestigationReport(
                    query_id=task.query_id,
                    status="satisfied",
                    evidence=(evidence,),
                    cost={},
                )
            )
        return tuple(reports)


def _workspace(tmp_path: Path) -> VirtualVideoWorkspace:
    manifest = VirtualVideoManifest(
        workspace_id="case-1",
        segments=(VirtualVideoSegment("seg_target", "target", "target.mp4", 10.0, 15.0, 0.0, 5.0, "target"),),
    )
    case = VirtualVideoCase(
        case_id="case-1",
        question="What number is written on the jersey?",
        options={"A": "7", "B": "11"},
        gold="B",
        target_segment_id="seg_target",
        target_virtual_interval=(0.0, 2.0),
    )
    workspace = VirtualVideoWorkspace.create(tmp_path / "case-1", manifest=manifest, case=case)
    frames = materialize_lowfps_frame_cache(workspace, fps=1.0, sampler=_sampler)
    workspace.write_asr_virtual_cues(({"start": 0.5, "end": 1.5, "text": "number written on jersey", "segment_id": "seg_target"},))
    build_virtual_beat_index(workspace, frames, model=TinyModel(), beat_sec=3.0)
    return VirtualVideoWorkspace.load(workspace.root_dir)


def _two_chunk_workspace(tmp_path: Path) -> VirtualVideoWorkspace:
    manifest = VirtualVideoManifest(
        workspace_id="full-count",
        segments=(
            VirtualVideoSegment("seg_target_a", "target", "target.mp4", 0.0, 5.0, 0.0, 5.0, "content"),
            VirtualVideoSegment("seg_noise", "noise", "noise.mp4", 0.0, 5.0, 5.0, 10.0, "content"),
            VirtualVideoSegment("seg_target_b", "target", "target.mp4", 5.0, 10.0, 10.0, 15.0, "content"),
        ),
    )
    case = VirtualVideoCase(
        case_id="full-count",
        question="Throughout the video, how many scholars in total show up and comment on Napoleon?",
        options={"A": "Two", "B": "Three"},
        gold="B",
        target_segment_id="seg_target_a",
        target_virtual_interval=(0.0, 15.0),
    )
    workspace = VirtualVideoWorkspace.create(tmp_path / "full-count", manifest=manifest, case=case)
    frames = materialize_lowfps_frame_cache(workspace, fps=1.0, sampler=_sampler)
    workspace.write_asr_virtual_cues(
        (
            {"start": 0.5, "end": 1.5, "text": "a scholar comments on Napoleon", "segment_id": "seg_target_a"},
            {"start": 10.5, "end": 11.5, "text": "another scholar discusses Napoleon", "segment_id": "seg_target_b"},
        )
    )
    build_virtual_beat_index(workspace, frames, model=TinyModel(), beat_sec=3.0)
    return VirtualVideoWorkspace.load(workspace.root_dir)


def _identity_repair_workspace(tmp_path: Path) -> VirtualVideoWorkspace:
    segments = tuple(
        VirtualVideoSegment(
            f"seg_{index}",
            f"source_{index}",
            f"source_{index}.mp4",
            0.0,
            5.0,
            float(index * 5),
            float((index + 1) * 5),
            "content",
        )
        for index in range(3)
    )
    manifest = VirtualVideoManifest(workspace_id="identity-repair", segments=segments)
    case = VirtualVideoCase(
        case_id="identity-repair",
        question="Why did the guest, who was carrying a folder and wearing glasses, leave?",
        options={"A": "A meeting ended", "B": "An alarm sounded"},
        gold="B",
        target_segment_id="seg_2",
        target_virtual_interval=(10.0, 15.0),
    )
    workspace = VirtualVideoWorkspace.create(tmp_path / "identity-repair", manifest=manifest, case=case)
    frames = materialize_lowfps_frame_cache(workspace, fps=1.0, sampler=_sampler)
    build_virtual_beat_index(workspace, frames, model=TinyModel(), beat_sec=3.0)
    return VirtualVideoWorkspace.load(workspace.root_dir)


def test_investigator_exposes_only_open_segment_and_inspect_window_tools(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    investigator = VirtualVideoInvestigator(workspace, sampler=_sampler)

    assert investigator.tool_names == ("open_segment", "inspect_window")
    assert not hasattr(investigator, "open_beat_page")
    assert not hasattr(investigator, "inspect_window_auto")


def test_open_segment_returns_navigation_packet_and_inspect_window_returns_frames_asr_lineage(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    investigator = VirtualVideoInvestigator(workspace, sampler=_sampler)

    packet = investigator.open_segment("seg_target")
    window = investigator.inspect_window(0.0, 2.0, fps=2.0, max_frames=64, query_id="q1")

    assert packet["segment_id"] == "seg_target"
    assert packet["virtual_time_range"] == [0.0, 5.0]
    assert packet["asr_cues"][0]["text"] == "number written on jersey"
    assert packet["beats"][0]["thumbnail_grid_paths"]
    assert window["virtual_time_range"] == [0.0, 2.0]
    assert window["sampling"]["fps"] == 2.0
    assert window["sampling"]["actual_frames"] > 0
    assert window["frames"][0]["source_video_id"] == "target"
    assert window["asr_cues"][0]["text"] == "number written on jersey"
    assert window["source_lineage"][0]["source_time_range"] == [10.0, 12.0]
    assert (workspace.root_dir / "observations" / "window_frame_manifest.jsonl").exists()


def test_investigator_run_batch_uses_segment_task_and_reports_lineage(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    investigator = VirtualVideoInvestigator(workspace, sampler=_sampler)
    task = InvestigationTask(
        query_id="q1",
        goal="Read the number on the jersey.",
        segment_id="seg_target",
        time_range=None,
        modality_hint=("visual", "ocr"),
        expected_evidence="number written on jersey",
    )

    report = investigator.run_batch((task,))[0]

    assert report.status == "satisfied"
    assert report.evidence
    assert isinstance(report.evidence[0], EvidenceRecord)
    assert report.evidence[0].evidence_id == "ev_q1_001"
    assert report.evidence[0].sampling_fps == 2.0
    assert report.evidence[0].task_id == "q1"
    assert report.evidence[0].source_lineage[0]["source_video_id"] == "target"
    assert report.cost["tool_trace"] == ("open_segment", "inspect_window:0.5", "inspect_window:2.0")
    assert (workspace.root_dir / "observations" / "window_frame_manifest.jsonl").exists()


def test_multiround_driver_caps_tasks_and_requires_cited_visual_evidence(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    reasoner = ScriptedReasoner()
    investigator = VirtualVideoInvestigator(workspace, sampler=_sampler)
    driver = VirtualVideoMultiRoundDriver(reasoner=reasoner, investigator=investigator, max_rounds=4, max_investigations=4)

    result = driver.run(workspace)

    assert result.answer == "B. 11"
    assert result.correct is True
    assert result.accepted_investigations == 4
    assert result.rounds == 2
    assert result.citations == ("ev_q0_001",)
    assert result.evidence[0].source_lineage[0]["source_time_range"] == [10.0, 13.5]
    evidence_rows = [
        json.loads(line)
        for line in (workspace.root_dir / "evidence.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert evidence_rows[0]["evidence_id"] == "ev_q0_001"
    assert evidence_rows[0]["source_lineage"][0]["source_video_id"] == "target"


def test_reasoner_initial_context_uses_segment_overview_not_cold_candidates(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    reasoner = ScriptedReasoner()
    investigator = VirtualVideoInvestigator(workspace, sampler=_sampler)
    driver = VirtualVideoMultiRoundDriver(reasoner=reasoner, investigator=investigator, max_rounds=1, max_investigations=4)

    driver.run(workspace)

    first_call = reasoner.kwargs[0]
    assert "workspace_overview" in first_call
    assert "cold_candidates" not in first_call
    overview = first_call["workspace_overview"]
    assert isinstance(overview, dict)
    assert overview["thumbnail_count"] == 1
    assert overview["segment_overviews"][0]["segment_id"] == "seg_target"
    assert "target" not in overview["segment_overviews"][0]
    assert first_call["available_tools"] == ("open_segment", "inspect_window")
    assert first_call["available_navigation"] == ("search_asr",)


def test_query_contract_marks_throughout_count_as_full_video_deduplication() -> None:
    contract = multiround.compile_query_contract(
        "Throughout the video, how many scholars in total show up and comment on Napoleon?"
    )

    assert contract.required_scope == "full_video"
    assert contract.quantifier == "distinct_count"
    assert contract.aggregation == "deduplicate"
    assert contract.required_observability == ("visual", "asr")


def test_query_contract_marks_how_many_times_as_full_video_event_count() -> None:
    contract = multiround.compile_query_contract("How many times do news segments appear in this video?")

    assert contract.required_scope == "full_video"
    assert contract.quantifier == "total_count"
    assert contract.observation_target == "event"
    assert contract.aggregation == "count"


def test_total_count_contract_marks_tasks_for_event_enumeration() -> None:
    contract = multiround.compile_query_contract("How many times does a title card appear in this video?")
    task = InvestigationTask(
        query_id="q_title_cards",
        goal="Inspect one source segment for title-card appearances.",
        segment_id="seg_0001",
        modality_hint=("visual",),
        expected_evidence="timestamped title-card occurrences",
    )

    compiled = multiround._task_for_contract(task, contract)

    assert compiled.inspection_mode == "enumerate_events"
    assert contract.required_observability == ("visual",)


def test_query_contract_generalizes_across_full_recording_paraphrases() -> None:
    entity_contract = multiround.compile_query_contract(
        "Across the entire recording, what is the number of different experts who speak about the subject?"
    )
    event_contract = multiround.compile_query_contract("How many times does the title card appear in the film?")

    assert entity_contract.required_scope == "full_video"
    assert entity_contract.quantifier == "distinct_count"
    assert entity_contract.required_observability == ("visual", "asr")
    assert event_contract.required_scope == "full_video"
    assert event_contract.quantifier == "total_count"


def test_source_time_hint_maps_39th_to_43rd_minute_to_candidate_segments(tmp_path: Path) -> None:
    manifest = VirtualVideoManifest(
        workspace_id="source-time",
        segments=(
            VirtualVideoSegment("seg_a", "source_a", "a.mp4", 2100.0, 2400.0, 0.0, 300.0),
            VirtualVideoSegment("seg_b", "source_a", "a.mp4", 2400.0, 2700.0, 300.0, 600.0),
            VirtualVideoSegment("seg_c", "source_b", "b.mp4", 0.0, 300.0, 600.0, 900.0),
        ),
    )
    case = VirtualVideoCase(
        case_id="source-time",
        question="During the 39th to 43rd minute of the film, why does Mike give Nathan a check?",
        options={"A": "one", "B": "two"},
        gold="B",
        target_segment_id="seg_a",
        target_virtual_interval=(0.0, 600.0),
    )
    workspace = VirtualVideoWorkspace.create(tmp_path / "source-time", manifest=manifest, case=case)

    hint = multiround.compile_source_time_hint(case.question)
    navigation = multiround.source_time_navigation(workspace, hint)

    assert hint == (2340.0, 2580.0)
    assert [item["segment_id"] for item in navigation["candidate_segments"]] == ["seg_a", "seg_b"]
    assert navigation["candidate_segments"][0]["virtual_time_range"] == [240.0, 300.0]
    assert navigation["candidate_segments"][1]["virtual_time_range"] == [300.0, 480.0]


def test_source_time_hint_supports_non_ordinal_paraphrases() -> None:
    assert multiround.compile_source_time_hint("From minute 12 to minute 15, what happened?") == (720.0, 900.0)
    assert multiround.compile_source_time_hint("Between minutes 7 and 9, who entered?") == (420.0, 540.0)


def test_identity_anchor_contract_extracts_generic_visible_attributes() -> None:
    question = "How was the woman, who was carrying a folder and wearing glasses, injured later?"
    contract = multiround.compile_query_contract(question)
    requirements = multiround.compile_query_requirements(question)

    assert contract.required_scope == "multi_window"
    assert contract.observation_target == "entity"
    assert contract.aggregation == "compare"
    assert requirements["identity_anchor_terms"] == ["folder", "glasses"]


def test_identity_anchor_contract_supports_active_relative_clause_paraphrase() -> None:
    question = "What happened to the guest who carried a suitcase and wore a red hat?"

    requirements = multiround.compile_query_requirements(question)

    assert requirements["identity_anchor_terms"] == ["suitcase", "red", "hat"]


def test_identity_completion_respects_explicit_negative_anchor_attestation(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    question = "What happened to the guest who carried a suitcase and wore a red hat?"
    contract = multiround.compile_query_contract(question)
    requirements = multiround.compile_query_requirements(question)
    negative = EvidenceRecord(
        evidence_id="ev_negative_anchor",
        beat_id="",
        start_sec=0.0,
        end_sec=5.0,
        modality="visual",
        pointer="virtual://negative-anchor",
        verbatim="No guest carrying a suitcase or wearing a red hat is visible.",
        frame_refs=("negative.jpg",),
        attestation_model="test-vlm",
        evidence_kind="visual_observation",
        coverage_manifest=(CoverageSegment("q_negative", 0.0, 5.0, "visual", 1.0),),
        source_lineage=(),
        operation_metadata={"supports_identity_anchor": False},
    )

    completion = multiround._completion_status(
        workspace,
        contract,
        (negative,),
        query_requirements=requirements,
    )

    assert completion["ready_for_answer"] is False
    assert completion["identity_anchor_evidence_ids"] == []


def test_full_video_count_answer_repairs_missing_source_chunks_before_aggregate(tmp_path: Path) -> None:
    workspace = _two_chunk_workspace(tmp_path)
    reasoner = CoverageReasoner()
    investigator = VirtualVideoInvestigator(workspace, sampler=_sampler)
    driver = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=investigator,
        max_rounds=4,
        max_investigations=8,
    )

    result = driver.run(workspace)

    assert result.answer == "B. Three"
    assert result.correct is True
    assert result.verified is True
    assert result.grounded_answer == "B. Three"
    assert result.forced_answer == "B. Three"
    assert result.selected_option == "B"
    assert result.answer_mode == "grounded"
    assert result.grounding_status == "verified"
    assert result.retrieval_status == "sufficient"
    assert result.verification_reason == "full_source_coverage_verified"
    assert reasoner.calls == 3
    assert reasoner.completion_statuses[1]["missing_segment_ids"] == ["seg_target_b"]
    assert reasoner.completion_statuses[2]["missing_segment_ids"] == []
    assert len(result.citations) == 1
    aggregate = next(item for item in result.evidence if item.evidence_id == result.citations[0])
    assert aggregate.modality == "derived"
    assert "ev_q_chunk_a_001" in aggregate.parent_evidence_ids
    assert len(aggregate.parent_evidence_ids) == 2
    assert aggregate.entity_ids == ("scholar_1", "scholar_2", "scholar_3")
    assert len(aggregate.operation_metadata["entity_clusters"]) == 3
    repair = next(row for row in result.trace if row.get("type") == "repair_override")
    assert repair["missing_segment_ids"] == ["seg_target_b"]
    gate = next(row for row in result.trace if row.get("type") == "completion_gate")
    assert gate["passed"] is True


def test_distinct_count_gate_rejects_answer_without_entity_reconciliation(tmp_path: Path) -> None:
    workspace = _two_chunk_workspace(tmp_path)
    reasoner = MissingEntityClustersReasoner()
    investigator = VirtualVideoInvestigator(workspace, sampler=_sampler)
    driver = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=investigator,
        max_rounds=2,
        max_investigations=4,
    )

    result = driver.run(workspace)

    assert result.answer == "B. Three"
    assert result.correct is True
    assert result.verified is False
    assert result.grounded_answer == ""
    assert result.forced_answer == "B. Three"
    assert result.selected_option == "B"
    assert result.answer_mode == "forced_choice"
    assert result.grounding_status == "insufficient"
    assert result.verification_reason == "entity_reconciliation_missing"
    gate = next(row for row in result.trace if row.get("type") == "completion_gate")
    assert gate["passed"] is False
    assert gate["reason"] == "entity_reconciliation_missing"


def test_distinct_count_gate_rejects_free_text_entity_without_frame_witness(tmp_path: Path) -> None:
    workspace = _two_chunk_workspace(tmp_path)
    contract = multiround.compile_query_contract(workspace.case.question, workspace.case.options)
    evidence = EvidenceRecord(
        evidence_id="ev_broad_summary",
        beat_id="",
        start_sec=0.0,
        end_sec=15.0,
        modality="visual",
        pointer="virtual://broad-summary",
        verbatim="Three different men appear in interviews.",
        frame_refs=("preview.jpg",),
        attestation_model="test-vlm",
        evidence_kind="visual_observation",
        coverage_manifest=(CoverageSegment("q_broad", 0.0, 15.0, "visual", 1.0),),
        source_lineage=(
            {"segment_id": "seg_target_a", "source_video_id": "target", "virtual_time_range": [0.0, 5.0]},
            {"segment_id": "seg_target_b", "source_video_id": "target", "virtual_time_range": [10.0, 15.0]},
        ),
        operation_metadata={
            "entities": [],
            "structured_parse_status": "fallback_extracted",
        },
    )
    clusters = tuple(
        {
            "entity_id": f"scholar_{index}",
            "description": "summary-only candidate",
            "evidence_ids": (evidence.evidence_id,),
        }
        for index in range(1, 4)
    )

    gate = multiround._answer_completion_gate(
        workspace,
        contract,
        "B. Three",
        (evidence.evidence_id,),
        clusters,
        (evidence,),
    )

    assert gate["passed"] is False
    assert gate["reason"] == "entity_cluster_witness_missing"
    assert len(gate["unsupported_entity_clusters"]) == 3


def test_distinct_count_gate_accepts_explicit_witnessed_entity_observations(tmp_path: Path) -> None:
    workspace = _two_chunk_workspace(tmp_path)
    contract = multiround.compile_query_contract(workspace.case.question, workspace.case.options)
    entities = tuple(
        {
            "local_id": f"person_{index}",
            "entity_observation_id": f"obs:person_{index}",
            "description": description,
            "visual_signature": description,
            "supports_question_relation": True,
            "witness_frame_refs": [f"person_{index}.jpg"],
            "countable": True,
        }
        for index, description in enumerate(
            ("bald man with glasses", "older woman with white hair", "brown-haired man in a light shirt"),
            start=1,
        )
    )
    evidence = EvidenceRecord(
        evidence_id="ev_witnessed_entities",
        beat_id="",
        start_sec=0.0,
        end_sec=15.0,
        modality="visual",
        pointer="virtual://witnessed",
        verbatim="Three individually witnessed scholars appear.",
        frame_refs=tuple(f"person_{index}.jpg" for index in range(1, 4)),
        attestation_model="test-vlm",
        evidence_kind="entity_observation",
        coverage_manifest=(CoverageSegment("q_witness", 0.0, 15.0, "visual", 1.0),),
        source_lineage=(
            {"segment_id": "seg_target_a", "source_video_id": "target", "virtual_time_range": [0.0, 5.0]},
            {"segment_id": "seg_target_b", "source_video_id": "target", "virtual_time_range": [10.0, 15.0]},
        ),
        operation_metadata={
            "entities": entities,
            "structured_parse_status": "parsed",
        },
    )
    clusters = tuple(
        {
            "entity_id": f"scholar_{index}",
            "description": entities[index - 1]["description"],
            "evidence_ids": (evidence.evidence_id,),
            "entity_observation_ids": (entities[index - 1]["entity_observation_id"],),
        }
        for index in range(1, 4)
    )

    gate = multiround._answer_completion_gate(
        workspace,
        contract,
        "B. Three",
        (evidence.evidence_id,),
        clusters,
        (evidence,),
    )

    assert gate["passed"] is True
    assert gate["reason"] == "full_source_coverage_verified"


def test_full_video_gate_uses_all_observations_for_coverage_but_positive_citations_for_answer(tmp_path: Path) -> None:
    workspace = _two_chunk_workspace(tmp_path)
    contract = multiround.compile_query_contract("How many times does a title card appear in this video?")
    positive = EvidenceRecord(
        evidence_id="ev_title_card",
        beat_id="",
        start_sec=0.0,
        end_sec=5.0,
        modality="visual",
        pointer="virtual://title-card",
        verbatim="A title card appears.",
        frame_refs=("title.jpg",),
        attestation_model="test-vlm",
        evidence_kind="event_observation",
        coverage_manifest=(CoverageSegment("q_a", 0.0, 5.0, "visual", 1.0),),
        source_lineage=(
            {
                "segment_id": "seg_target_a",
                "source_video_id": "target",
                "source_time_range": [0.0, 5.0],
                "virtual_time_range": [0.0, 5.0],
            },
        ),
        operation_metadata={
            "supports_answer_event": True,
            "events": [
                {
                    "local_id": "event_1",
                    "description": "The opening title card appears.",
                    "start_sec": 1.0,
                    "end_sec": 1.5,
                    "supports_question_event": True,
                },
                {
                    "local_id": "event_2",
                    "description": "A second title card appears.",
                    "start_sec": 3.0,
                    "end_sec": 3.5,
                    "supports_question_event": True,
                },
            ],
        },
    )
    negative = EvidenceRecord(
        evidence_id="ev_no_title_card",
        beat_id="",
        start_sec=10.0,
        end_sec=15.0,
        modality="visual",
        pointer="virtual://no-title-card",
        verbatim="No title card appears in this source chunk.",
        frame_refs=("negative.jpg",),
        attestation_model="test-vlm",
        evidence_kind="visual_observation",
        coverage_manifest=(CoverageSegment("q_b", 10.0, 15.0, "visual", 1.0),),
        source_lineage=(
            {
                "segment_id": "seg_target_b",
                "source_video_id": "target",
                "source_time_range": [5.0, 10.0],
                "virtual_time_range": [10.0, 15.0],
            },
        ),
        operation_metadata={"supports_answer_event": False},
    )

    gate = multiround._answer_completion_gate(
        workspace,
        contract,
        "A. Two",
        ("ev_title_card",),
        (),
        (positive, negative),
    )

    assert gate["passed"] is True
    assert gate["reason"] == "full_source_coverage_verified"
    wrong_count = multiround._answer_completion_gate(
        workspace,
        contract,
        "B. Three",
        ("ev_title_card",),
        (),
        (positive, negative),
    )
    assert wrong_count["passed"] is False
    assert wrong_count["reason"] == "event_count_answer_mismatch"
    invalid_answer = multiround._answer_completion_gate(
        workspace,
        contract,
        "The video is unclear, so no option can be selected.",
        ("ev_title_card",),
        (),
        (positive, negative),
    )
    assert invalid_answer["passed"] is False
    assert invalid_answer["reason"] == "invalid_option_answer"
    aggregate = multiround._derived_answer_evidence(
        workspace,
        answer="A. Two",
        citations=("ev_title_card",),
        entity_clusters=(),
        evidence=(positive, negative),
        coverage_source_ids=gate["source_video_ids"],
    )
    assert aggregate.parent_evidence_ids == ("ev_title_card", "ev_no_title_card")
    assert len(aggregate.operation_metadata["event_occurrences"]) == 2


def test_event_occurrences_merge_only_explicit_cross_beat_continuations() -> None:
    lineage = (
        {
            "segment_id": "seg_1",
            "source_video_id": "source",
            "source_time_range": [180.0, 300.0],
            "virtual_time_range": [180.0, 300.0],
        },
    )
    first = EvidenceRecord(
        evidence_id="ev_first",
        beat_id="",
        start_sec=180.0,
        end_sec=240.0,
        modality="visual",
        pointer="virtual://first",
        verbatim="Two different news segments appear.",
        frame_refs=("first.jpg",),
        attestation_model="test-vlm",
        evidence_kind="event_observation",
        source_lineage=lineage,
        operation_metadata={
            "events": [
                {
                    "local_id": "event_1",
                    "event_key": "meta-human summit",
                    "description": "A summit news report.",
                    "start_sec": 180.0,
                    "end_sec": 196.0,
                    "supports_question_event": True,
                    "continues_from_previous": False,
                    "continues_to_next": False,
                },
                {
                    "local_id": "event_2",
                    "event_key": "markovia royal interview",
                    "description": "A report about the Markovian royal family begins.",
                    "start_sec": 220.0,
                    "end_sec": 240.0,
                    "supports_question_event": True,
                    "continues_from_previous": False,
                    "continues_to_next": True,
                },
            ]
        },
    )
    second = EvidenceRecord(
        evidence_id="ev_second",
        beat_id="",
        start_sec=240.0,
        end_sec=300.0,
        modality="visual",
        pointer="virtual://second",
        verbatim="The Markovian report continues.",
        frame_refs=("second.jpg",),
        attestation_model="test-vlm",
        evidence_kind="event_observation",
        source_lineage=lineage,
        operation_metadata={
            "events": [
                {
                    "local_id": "event_1",
                    "event_key": "markovia royal interview",
                    "description": "The report about the Markovian royal family continues.",
                    "start_sec": 240.0,
                    "end_sec": 256.0,
                    "supports_question_event": True,
                    "continues_from_previous": True,
                    "continues_to_next": False,
                }
            ]
        },
    )

    occurrences = multiround._event_occurrences((first, second))

    assert len(occurrences) == 2
    continued = next(row for row in occurrences if row["event_key"] == "markovia royal interview")
    assert (continued["start_sec"], continued["end_sec"]) == (220.0, 256.0)
    assert continued["evidence_ids"] == ["ev_first", "ev_second"]


def test_answer_gate_rejects_empty_answer_even_with_visual_citation(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    reasoner = EmptyAnswerReasoner()
    investigator = VirtualVideoInvestigator(workspace, sampler=_sampler)
    result = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=investigator,
        max_rounds=2,
        max_investigations=2,
    ).run(workspace)

    assert result.answer == "Insufficient verified evidence."
    assert result.verified is False
    assert result.verification_reason == "answer_missing"
    gate = next(row for row in result.trace if row.get("type") == "completion_gate")
    assert gate["reason"] == "answer_missing"


def test_answer_gate_soft_rejects_ungrounded_large_numeric_option(tmp_path: Path) -> None:
    manifest = VirtualVideoManifest(
        workspace_id="numeric-gate",
        segments=(VirtualVideoSegment("seg_1", "source", "source.mp4", 0.0, 60.0, 0.0, 60.0, "content"),),
    )
    case = VirtualVideoCase(
        case_id="numeric-gate",
        question="What total diameter does the video state?",
        options={"A": "About 100 trillion lightyears", "C": "Over 25 trillion lightyears"},
        gold="C",
        target_segment_id="seg_1",
        target_virtual_interval=(0.0, 60.0),
    )
    workspace = VirtualVideoWorkspace.create(tmp_path / "numeric-gate", manifest=manifest, case=case)
    evidence = EvidenceRecord(
        evidence_id="ev_numeric",
        beat_id="",
        start_sec=10.0,
        end_sec=20.0,
        modality="visual",
        pointer="virtual://numeric",
        verbatim="The observable universe is 93 billion lightyears across and a tiny fraction of the total expanse.",
        frame_refs=("numeric.jpg",),
        attestation_model="test-vlm",
        evidence_kind="event_observation",
    )

    gate = multiround._answer_completion_gate(
        workspace,
        multiround.compile_query_contract(case.question),
        "A. About 100 trillion lightyears",
        ("ev_numeric",),
        (),
        (evidence,),
    )

    assert gate["passed"] is False
    assert gate["reason"] == "quantitative_answer_not_grounded"
    assert gate["missing_numeric_atoms"] == ["100"]


def test_answer_gate_accepts_interval_when_both_boundary_times_are_cited(tmp_path: Path) -> None:
    manifest = VirtualVideoManifest(
        workspace_id="interval-gate",
        segments=(VirtualVideoSegment("seg_1", "source", "source.mp4", 0.0, 900.0, 0.0, 900.0, "content"),),
    )
    case = VirtualVideoCase(
        case_id="interval-gate",
        question="In which period does the home team overtake the guest team?",
        options={"B": "5:58 - 2:57", "C": "8:13 - 5:58"},
        gold="B",
        target_segment_id="seg_1",
        target_virtual_interval=(0.0, 900.0),
    )
    workspace = VirtualVideoWorkspace.create(tmp_path / "interval-gate", manifest=manifest, case=case)
    before = EvidenceRecord(
        evidence_id="ev_before",
        beat_id="",
        start_sec=500.0,
        end_sec=510.0,
        modality="visual",
        pointer="virtual://before",
        verbatim="At 8:13 the home team is behind.",
        frame_refs=("before.jpg",),
        attestation_model="test-vlm",
        evidence_kind="event_observation",
    )
    after = EvidenceRecord(
        evidence_id="ev_after",
        beat_id="",
        start_sec=690.0,
        end_sec=700.0,
        modality="visual",
        pointer="virtual://after",
        verbatim="At 5:58 the home team is ahead.",
        frame_refs=("after.jpg",),
        attestation_model="test-vlm",
        evidence_kind="event_observation",
    )

    gate = multiround._answer_completion_gate(
        workspace,
        multiround.compile_query_contract(case.question),
        "C. 8:13 - 5:58",
        ("ev_before", "ev_after"),
        (),
        (before, after),
    )

    assert gate["passed"] is True
    assert gate["reason"] == "verified_window_evidence"

    mislocalized = EvidenceRecord(
        evidence_id="ev_mislocalized",
        beat_id="",
        start_sec=500.0,
        end_sec=700.0,
        modality="visual",
        pointer="virtual://mislocalized",
        verbatim="The summary claims the clock boundaries are 8:13 and 5:58.",
        frame_refs=("wrong_crop.jpg",),
        attestation_model="test-vlm",
        operation_metadata={
            "target_presence": {"target": "scoreboard", "status": "absent", "confidence": 0.95},
            "measurements": [
                {"value": 8.13, "unit": "game_clock", "raw_text": "8:13"},
                {"value": 5.58, "unit": "game_clock", "raw_text": "5:58"},
            ],
        },
    )
    blocked = multiround._answer_completion_gate(
        workspace,
        multiround.compile_query_contract(case.question),
        "C. 8:13 - 5:58",
        ("ev_mislocalized",),
        (),
        (mislocalized,),
    )
    assert blocked["passed"] is False
    assert blocked["reason"] == "quantitative_target_not_present"


def test_driver_runs_answer_only_finalization_after_last_investigation_round(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    reasoner = FinalizationReasoner()
    investigator = VirtualVideoInvestigator(workspace, sampler=_sampler)
    result = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=investigator,
        max_rounds=1,
        max_investigations=2,
    ).run(workspace)

    assert result.answer == "B. 11"
    assert result.correct is True
    assert reasoner.force_flags == [False, True]
    assert any(row.get("type") == "reasoner_finalization" for row in result.trace)


def test_answer_audit_soft_fails_verification_without_erasing_best_effort_answer(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result = VirtualVideoMultiRoundDriver(
        reasoner=UnsupportedFinalizationReasoner(),
        investigator=VirtualVideoInvestigator(workspace, sampler=_sampler),
        max_rounds=1,
        max_investigations=2,
    ).run(workspace)

    assert result.answer == "B. 11"
    assert result.correct is True
    assert result.verified is False
    assert result.verification_reason == "answer_audit_insufficient"
    gate = next(row for row in result.trace if row.get("type") == "completion_gate")
    assert gate["base_gate_passed"] is True
    assert gate["audit_reason"].startswith("The observation is related")


def test_driver_keeps_stronger_supported_candidate_over_later_contradicted_answer(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result = VirtualVideoMultiRoundDriver(
        reasoner=RankedCandidateReasoner(),
        investigator=VirtualVideoInvestigator(workspace, sampler=_sampler),
        max_rounds=3,
        max_investigations=2,
    ).run(workspace)

    assert result.answer == "B. 11"
    assert result.correct is True
    assert result.verified is False


def test_navigation_hint_does_not_satisfy_visual_completion(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    hint = EvidenceRecord(
        evidence_id="ev_asr_hint_1",
        beat_id="",
        start_sec=10.0,
        end_sec=20.0,
        modality="asr",
        pointer="virtual://case/asr-search/1",
        verbatim="Literal ASR hit for the requested terms.",
        temporal_scope="window",
        evidence_kind="navigation_hint",
        observation_polarity="positive",
        sampling_coverage="exact",
        source_lineage=(
            {
                "segment_id": "seg_target",
                "source_video_id": "target",
                "source_time_range": [10.0, 20.0],
                "virtual_time_range": [10.0, 20.0],
            },
        ),
    )

    status = multiround._completion_status(
        workspace,
        multiround.compile_query_contract(workspace.case.question),
        (hint,),
    )

    assert status["ready_for_answer"] is False
    assert status["candidate_available"] is True
    assert status["retrieval_ready"] is False
    assert status["choice_ready"] is False
    assert status["grounded_ready"] is False


def test_completion_status_uses_monotonic_condition_ledger(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    evidence = EvidenceRecord(
        evidence_id="ev_clock", beat_id="", start_sec=0.0, end_sec=2.0, modality="visual",
        pointer="virtual://clock", verbatim="The scoreboard clock reads 8:13.",
        frame_refs=("clock.jpg",), attestation_model="test-vlm",
    )
    reports = (
        InvestigationReport(
            query_id="q_clock", status="satisfied", evidence=(evidence,), gap_id="gap_clock", resolution="resolved",
            condition_results=(ConditionResult("gap_clock_c1", "satisfied", "Clock reads 8:13.", ("ev_clock",)),),
        ),
        InvestigationReport(
            query_id="q_unrelated", status="satisfied", evidence=(evidence,), gap_id="gap_clock", resolution="partial",
            condition_results=(ConditionResult("gap_clock_c1", "unknown", "Unrelated view."),),
        ),
    )
    status = multiround._completion_status(
        workspace, multiround.compile_query_contract(workspace.case.question), (evidence,),
        reports=reports, best_choice="B. 11",
    )
    assert status["retrieval_ready"] is True
    assert status["choice_ready"] is True
    assert status["grounded_ready"] is True
    assert status["condition_states"]["gap_clock_c1"]["status"] == "satisfied"


def test_navigation_repair_prioritizes_unverified_positive_hint() -> None:
    verified_hint = EvidenceRecord(
        evidence_id="ev_firework_hint",
        beat_id="",
        start_sec=100.0,
        end_sec=120.0,
        modality="asr",
        pointer="virtual://firework",
        verbatim="fireworks exploding",
        evidence_kind="navigation_hint",
        observation_polarity="positive",
        task_id="contrastive_search",
        operation_metadata={
            "search_terms": ["firework", "fire", "chasing", "dog"],
            "matched_terms": ["firework", "fire"],
            "hit_count": 2,
        },
        source_lineage=(
            {
                "segment_id": "seg_2",
                "source_video_id": "source",
                "source_time_range": [100.0, 120.0],
                "virtual_time_range": [100.0, 120.0],
            },
        ),
    )
    verified_visual = EvidenceRecord(
        evidence_id="ev_firework_visual",
        beat_id="",
        start_sec=102.0,
        end_sec=118.0,
        modality="visual",
        pointer="virtual://firework-visual",
        verbatim="A firework explodes near a man.",
        frame_refs=("firework.jpg",),
        attestation_model="test-vlm",
        evidence_kind="event_observation",
        source_lineage=verified_hint.source_lineage,
        operation_metadata={"source_candidate_ids": ["ev_firework_hint"]},
    )
    unverified_hint = EvidenceRecord(
        evidence_id="ev_dog_hint",
        beat_id="",
        start_sec=20.0,
        end_sec=40.0,
        modality="asr",
        pointer="virtual://dog",
        verbatim="dog growls",
        evidence_kind="navigation_hint",
        observation_polarity="positive",
        task_id="contrastive_search",
        operation_metadata={
            "search_terms": ["firework", "fire", "chasing", "dog"],
            "matched_terms": ["dog"],
            "hit_count": 2,
        },
        source_lineage=(
            {
                "segment_id": "seg_1",
                "source_video_id": "source",
                "source_time_range": [20.0, 40.0],
                "virtual_time_range": [20.0, 40.0],
            },
        ),
    )

    tasks = multiround._navigation_repair_tasks(
        (verified_hint, verified_visual, unverified_hint),
        round_id=3,
        limit=4,
    )

    assert len(tasks) == 1
    assert tasks[0].segment_id == "seg_1"
    assert tasks[0].time_range == (20.0, 40.0)
    assert tasks[0].modality_hint == ("visual",)
    assert "dog growls" in tasks[0].expected_evidence
    assert tasks[0].source_candidate_ids == ("ev_dog_hint",)


def test_candidate_status_requires_explicit_provenance_not_time_overlap() -> None:
    hint = EvidenceRecord(
        evidence_id="ev_candidate", beat_id="", start_sec=10.0, end_sec=30.0, modality="asr",
        pointer="virtual://candidate", verbatim="dog and father", evidence_kind="navigation_hint",
        observation_polarity="positive", operation_metadata={"matched_terms": ["dog", "father"]},
    )
    overlap_only = EvidenceRecord(
        evidence_id="ev_overlap", beat_id="", start_sec=10.0, end_sec=30.0, modality="visual",
        pointer="virtual://overlap", verbatim="A firework is visible.", frame_refs=("frame.jpg",),
        attestation_model="test-vlm",
    )
    explicit = EvidenceRecord(
        evidence_id="ev_explicit", beat_id="", start_sec=10.0, end_sec=30.0, modality="visual",
        pointer="virtual://explicit", verbatim="A dog chases the father.", frame_refs=("frame.jpg",),
        attestation_model="test-vlm", operation_metadata={"source_candidate_ids": ["ev_candidate"]},
    )
    unseen = multiround._navigation_candidates((hint, overlap_only), {"D": "A dog chases his father"})
    inspected = multiround._navigation_candidates((hint, overlap_only, explicit), {"D": "A dog chases his father"})
    assert unseen[0]["status"] == "unseen"
    assert unseen[0]["possibly_covered"] is True
    assert inspected[0]["status"] == "inspected"
    assert inspected[0]["resulting_evidence_ids"] == ["ev_explicit"]


def test_contrastive_candidate_override_uses_at_most_one_slot() -> None:
    hints = tuple(
        EvidenceRecord(
            evidence_id=f"ev_{name}", beat_id="", start_sec=start, end_sec=start + 20.0, modality="asr",
            pointer=f"virtual://{name}", verbatim=" ".join(terms), evidence_kind="navigation_hint",
            observation_polarity="positive", operation_metadata={"matched_terms": list(terms), "hit_count": 2},
            source_lineage=({"segment_id": f"seg_{name}", "source_video_id": "source"},),
        )
        for name, start, terms in (("firework", 100.0, ("firework",)), ("dog", 20.0, ("dog", "father")))
    )
    requested = (
        InvestigationTask("model_1", "Inspect another firework.", "seg_model_1"),
        InvestigationTask("model_2", "Inspect a later firework.", "seg_model_2"),
    )
    tasks = multiround._prefer_navigation_repairs(
        requested, hints, options={"A": "A firework explodes", "D": "A dog chases his father"}, round_id=2, limit=2,
    )
    assert len(tasks) == 2
    assert sum(task.query_id.startswith("navigation_repair_") for task in tasks) == 1
    assert tasks[0].source_candidate_ids == ("ev_dog",)


def test_search_round_can_dispatch_one_new_option_linked_candidate() -> None:
    hint = EvidenceRecord(
        evidence_id="ev_dog", beat_id="", start_sec=20.0, end_sec=40.0, modality="asr",
        pointer="virtual://dog", verbatim="dog and father", evidence_kind="navigation_hint",
        observation_polarity="positive", operation_metadata={"matched_terms": ["dog", "father"], "hit_count": 2},
        source_lineage=({"segment_id": "seg_dog", "source_video_id": "source"},),
    )
    search_tasks = (
        InvestigationTask("search", "Search competing causes.", inspection_mode="search_asr", search_terms=("dog", "firework")),
    )

    followup = multiround._post_search_candidate_tasks(
        search_tasks,
        (hint,),
        options={"A": "A firework explodes", "D": "A dog attacks his father"},
        round_id=6,
        remaining_round_slots=3,
        remaining_budget=4,
    )

    assert len(followup) == 1
    assert followup[0].source_candidate_ids == ("ev_dog",)


def test_search_tasks_yield_to_unresolved_navigation_windows() -> None:
    hint = EvidenceRecord(
        evidence_id="ev_dog_hint",
        beat_id="",
        start_sec=20.0,
        end_sec=40.0,
        modality="asr",
        pointer="virtual://dog",
        verbatim="dog growls",
        evidence_kind="navigation_hint",
        observation_polarity="positive",
        task_id="contrastive_search",
        source_lineage=(
            {
                "segment_id": "seg_1",
                "source_video_id": "source",
                "source_time_range": [20.0, 40.0],
                "virtual_time_range": [20.0, 40.0],
            },
        ),
        operation_metadata={
            "search_terms": ["firework", "fire", "chasing", "dog"],
            "matched_terms": ["dog"],
            "hit_count": 2,
        },
    )
    requested = (
        InvestigationTask("anchor", "Inspect the visual identity anchor.", "seg_7"),
        InvestigationTask(
            "search_again",
            "Search for more candidate causes.",
            inspection_mode="search_asr",
            search_terms=("firework", "dog", "fall"),
        ),
    )

    tasks = multiround._prefer_navigation_repairs(requested, (hint,), round_id=2, limit=4)

    assert tasks[0] == requested[0]
    assert all(task.inspection_mode != "search_asr" for task in tasks)
    assert tasks[1].segment_id == "seg_1"
    assert tasks[1].time_range == (20.0, 40.0)


def test_evidence_digest_exposes_navigation_search_state() -> None:
    hint = EvidenceRecord(
        evidence_id="ev_nav",
        beat_id="",
        start_sec=20.0,
        end_sec=40.0,
        modality="asr",
        pointer="virtual://nav",
        verbatim="dog growls",
        evidence_kind="navigation_hint",
        observation_polarity="positive",
        operation_metadata={
            "navigation_only": True,
            "search_terms": ["firework", "fire", "chasing", "dog"],
            "matched_terms": ["dog"],
            "hit_count": 2,
        },
    )

    digest = multiround._evidence_digest((hint,))

    assert digest[0]["observation_polarity"] == "positive"
    assert digest[0]["navigation"] == {
        "search_terms": ["firework", "fire", "chasing", "dog"],
        "matched_terms": ["dog"],
        "hit_count": 2,
    }


def test_answer_citations_drop_navigation_hints_when_visual_support_remains() -> None:
    hint = EvidenceRecord(
        evidence_id="ev_hint",
        beat_id="",
        start_sec=10.0,
        end_sec=20.0,
        modality="asr",
        pointer="virtual://hint",
        verbatim="Literal transcript hit.",
        temporal_scope="window",
        evidence_kind="navigation_hint",
    )
    visual = EvidenceRecord(
        evidence_id="ev_visual",
        beat_id="",
        start_sec=10.0,
        end_sec=20.0,
        modality="visual",
        pointer="virtual://visual",
        verbatim="The requested fact is directly visible.",
        frame_refs=("frame.jpg",),
        attestation_model="test-vlm",
        temporal_scope="window",
        evidence_kind="visual_observation",
    )

    citations = multiround._answer_citations(("ev_hint", "ev_visual", "ev_missing"), (hint, visual))

    assert citations == ("ev_visual",)


def test_driver_filters_navigation_hint_from_mixed_final_citations(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    class MixedCitationReasoner:
        def __init__(self) -> None:
            self.calls = 0

        def decide(self, **kwargs: object) -> ReasonerDecision:
            del kwargs
            self.calls += 1
            if self.calls == 1:
                return ReasonerDecision(
                    action="investigate",
                    tasks=(InvestigationTask("q1", "Read the jersey number.", "seg_target"),),
                )
            return ReasonerDecision(
                action="answer",
                answer="B. 11",
                citations=("ev_hint", "ev_visual"),
            )

    class MixedCitationInvestigator:
        def reset_run_state(self) -> None:
            return None

        def run_batch(self, tasks: Sequence[InvestigationTask]) -> tuple[InvestigationReport, ...]:
            hint = EvidenceRecord(
                evidence_id="ev_hint",
                beat_id="",
                start_sec=0.0,
                end_sec=2.0,
                modality="asr",
                pointer="virtual://hint",
                verbatim="number written on jersey",
                temporal_scope="window",
                evidence_kind="navigation_hint",
            )
            visual = EvidenceRecord(
                evidence_id="ev_visual",
                beat_id="",
                start_sec=0.0,
                end_sec=2.0,
                modality="visual",
                pointer="virtual://visual",
                verbatim="The jersey visibly shows number 11.",
                frame_refs=("frame.jpg",),
                attestation_model="test-vlm",
                temporal_scope="window",
                evidence_kind="visual_observation",
                source_lineage=(
                    {
                        "segment_id": "seg_target",
                        "source_video_id": "target",
                        "source_time_range": [10.0, 12.0],
                        "virtual_time_range": [0.0, 2.0],
                    },
                ),
            )
            return (
                InvestigationReport(
                    query_id=tasks[0].query_id,
                    status="satisfied",
                    evidence=(hint, visual),
                    cost={},
                ),
            )

    result = VirtualVideoMultiRoundDriver(
        reasoner=MixedCitationReasoner(),
        investigator=MixedCitationInvestigator(),
        max_rounds=2,
        max_investigations=2,
    ).run(workspace)

    assert result.verified is True
    assert result.citations == ("ev_visual",)
    citation_filter = next(row for row in result.trace if row.get("type") == "citation_filter")
    assert citation_filter["removed_citations"] == ["ev_hint"]


def test_score_answer_maps_unlabeled_numeric_text_back_to_option() -> None:
    assert multiround._score_answer(
        "Based on the race progress, the athletes completed approximately 6000 meters in 25 minutes.",
        "B",
        {"A": "5000m.", "B": "6000m.", "C": "7000m.", "D": "8000m."},
    )


def test_score_answer_does_not_treat_indefinite_article_as_option_label() -> None:
    options = {
        "A": "One of his hands was hit by a firework while he was setting it off.",
        "D": "One of his arms was dragged down by a dog lured with food by Wayne, while he was insulting Wayne's father.",
    }
    answer = options["D"]

    assert multiround._letter(answer) == ""
    assert multiround._letter("A. First option") == "A"
    assert multiround._letter("The answer is D because the dog caused the injury.") == "D"
    assert multiround._score_answer(answer, "D", options) is True


def test_driver_repairs_empty_answers_with_unvisited_identity_anchor_segments(tmp_path: Path) -> None:
    workspace = _identity_repair_workspace(tmp_path)
    investigator = NegativeAnchorInvestigator(workspace)

    result = VirtualVideoMultiRoundDriver(
        reasoner=AlwaysEmptyReasoner(),
        investigator=investigator,
        max_rounds=2,
        max_investigations=2,
        max_tasks_per_round=1,
    ).run(workspace)

    assert result.answer == "Insufficient verified evidence."
    assert result.grounded_answer == ""
    assert result.forced_answer == ""
    assert result.selected_option == ""
    assert result.answer_mode == "insufficient"
    assert result.accepted_investigations == 2
    assert [task.segment_id for task in investigator.tasks] == ["seg_0", "seg_1"]
    repairs = [row for row in result.trace if row.get("type") == "repair_override"]
    assert [row["reason"] for row in repairs] == ["identity_anchor_missing", "identity_anchor_missing"]


def test_identity_gate_requires_anchor_and_event_evidence_in_same_cluster(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    question = "How did the man, who was wearing a bandage and holding an envelope, sustain his injury?"
    contract = multiround.compile_query_contract(question)
    requirements = multiround.compile_query_requirements(question)
    lineage = (
        {
            "segment_id": "seg_target",
            "source_video_id": "target",
            "source_time_range": [10.0, 15.0],
            "virtual_time_range": [0.0, 5.0],
        },
    )
    anchor = EvidenceRecord(
        evidence_id="ev_anchor",
        beat_id="",
        start_sec=0.0,
        end_sec=2.0,
        modality="visual",
        pointer="virtual://anchor",
        verbatim="A man wearing a bandage is holding an envelope.",
        frame_refs=("anchor.jpg",),
        attestation_model="test-vlm",
        evidence_kind="entity_observation",
        coverage_manifest=(CoverageSegment("q_anchor", 0.0, 2.0, "visual", 1.0),),
        source_lineage=lineage,
    )
    cause = EvidenceRecord(
        evidence_id="ev_cause",
        beat_id="",
        start_sec=2.0,
        end_sec=5.0,
        modality="visual",
        pointer="virtual://cause",
        verbatim="The same man has his arm pulled down by a dog.",
        frame_refs=("cause.jpg",),
        attestation_model="test-vlm",
        evidence_kind="event_observation",
        coverage_manifest=(CoverageSegment("q_cause", 2.0, 5.0, "visual", 1.0),),
        source_lineage=lineage,
    )

    missing_anchor = multiround._answer_completion_gate(
        workspace,
        contract,
        "D. A dog pulled his arm",
        ("ev_cause",),
        ({"entity_id": "person_1", "description": "injured man", "evidence_ids": ("ev_cause",)},),
        (anchor, cause),
        query_requirements=requirements,
    )
    linked = multiround._answer_completion_gate(
        workspace,
        contract,
        "D. A dog pulled his arm",
        ("ev_anchor", "ev_cause"),
        (
            {
                "entity_id": "person_1",
                "description": "man with bandage and envelope",
                "evidence_ids": ("ev_anchor", "ev_cause"),
            },
        ),
        (anchor, cause),
        query_requirements=requirements,
    )

    assert missing_anchor["reason"] == "identity_anchor_not_linked"
    assert linked["passed"] is True


def test_identity_gate_accepts_one_joint_anchor_event_observation(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    question = "Why did the woman, who was carrying a folder and wearing glasses, fall?"
    contract = multiround.compile_query_contract(question)
    requirements = multiround.compile_query_requirements(question)
    joint = EvidenceRecord(
        evidence_id="ev_joint",
        beat_id="",
        start_sec=0.0,
        end_sec=5.0,
        modality="visual",
        pointer="virtual://joint",
        verbatim="The woman carrying a folder and wearing glasses slips on a wet floor.",
        frame_refs=("joint.jpg",),
        attestation_model="test-vlm",
        evidence_kind="event_observation",
        coverage_manifest=(CoverageSegment("q_joint", 0.0, 5.0, "visual", 1.0),),
        source_lineage=(
            {
                "segment_id": "seg_target",
                "source_video_id": "target",
                "source_time_range": [10.0, 15.0],
                "virtual_time_range": [0.0, 5.0],
            },
        ),
    )

    gate = multiround._answer_completion_gate(
        workspace,
        contract,
        "A. She slipped on a wet floor",
        ("ev_joint",),
        ({"entity_id": "person_1", "description": "woman with folder and glasses", "evidence_ids": ("ev_joint",)},),
        (joint,),
        query_requirements=requirements,
    )

    assert gate["passed"] is True


def test_identity_gate_rejects_anchor_only_cluster_without_event_observation(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    question = "Why did the woman, who was carrying a folder and wearing glasses, fall?"
    contract = multiround.compile_query_contract(question)
    requirements = multiround.compile_query_requirements(question)
    lineage = (
        {
            "segment_id": "seg_target",
            "source_video_id": "target",
            "source_time_range": [10.0, 15.0],
            "virtual_time_range": [0.0, 5.0],
        },
    )
    folder = EvidenceRecord(
        evidence_id="ev_folder",
        beat_id="",
        start_sec=0.0,
        end_sec=2.0,
        modality="visual",
        pointer="virtual://folder",
        verbatim="A woman is carrying a folder.",
        frame_refs=("folder.jpg",),
        attestation_model="test-vlm",
        evidence_kind="entity_observation",
        coverage_manifest=(CoverageSegment("q_folder", 0.0, 2.0, "visual", 1.0),),
        source_lineage=lineage,
    )
    glasses = EvidenceRecord(
        evidence_id="ev_glasses",
        beat_id="",
        start_sec=2.0,
        end_sec=4.0,
        modality="visual",
        pointer="virtual://glasses",
        verbatim="The same woman is wearing glasses.",
        frame_refs=("glasses.jpg",),
        attestation_model="test-vlm",
        evidence_kind="entity_observation",
        coverage_manifest=(CoverageSegment("q_glasses", 2.0, 4.0, "visual", 1.0),),
        source_lineage=lineage,
    )

    gate = multiround._answer_completion_gate(
        workspace,
        contract,
        "A. She slipped on a wet floor",
        ("ev_folder", "ev_glasses"),
        (
            {
                "entity_id": "person_1",
                "description": "woman with folder and glasses",
                "evidence_ids": ("ev_folder", "ev_glasses"),
            },
        ),
        (folder, glasses),
        query_requirements=requirements,
    )
    completion = multiround._completion_status(
        workspace,
        contract,
        (folder, glasses),
        query_requirements=requirements,
    )

    assert gate["passed"] is False
    assert gate["reason"] == "identity_anchor_not_linked"
    assert completion["ready_for_answer"] is False
    assert completion["identity_anchor_evidence_ids"] == []
