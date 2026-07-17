from __future__ import annotations

import json
import inspect
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from PIL import Image

import vcah.multiround as multiround
from vcah.evidence_primitives import ConditionResult, ConditionState
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


def test_query_compiler_emits_semantic_contracts_for_failure_families() -> None:
    events = multiround.compile_query_contract(
        "How many dance group auditions are included in this video?"
    )
    spatial = multiround.compile_query_contract(
        "Which direction is the narrator in red facing in relation to the narrator in green?"
    )
    absence = multiround.compile_query_contract(
        "Which acrobatics skill is absent from this video?"
    )
    boundary_score = multiround.compile_query_contract(
        "What was the halftime score?",
        {"A": "32 - 23", "B": "37 - 27"},
    )
    episode = multiround.compile_query_contract("In which episode do they get married?")
    temporal = multiround.compile_query_contract(
        "In the last gathering, at what time does the green model in the lower left appear?"
    )
    synopsis = multiround.compile_query_contract("What title best summarizes this video?")
    progress = multiround.compile_query_contract(
        "As one team enjoys pizza, how many tasks has the other team completed?"
    )
    causal = multiround.compile_query_contract(
        "Which stated factor was not the cause of the split?"
    )
    outcome = multiround.compile_query_contract("How did the final battle unfold?")
    agent_relation = multiround.compile_query_contract("Who protects the mermaid?")
    sequence_requirements = multiround.compile_query_requirements(
        "Which option correctly describes the sequence before and after the color changed?"
    )

    assert (events.required_scope, events.quantifier, events.observation_target) == (
        "full_video",
        "total_count",
        "event",
    )
    assert (spatial.quantifier, spatial.observation_target, spatial.aggregation) == (
        "comparison",
        "relation",
        "compare",
    )
    assert multiround.compile_query_requirements(
        "Which direction is red facing in relation to green?"
    )["spatial_relation_type"] == "relative_facing"
    assert (absence.required_scope, absence.quantifier, absence.aggregation) == (
        "full_video",
        "universal",
        "compare",
    )
    assert boundary_score.measurement_unit == "point"
    assert boundary_score.boundary_hint.casefold() == "halftime"
    assert (episode.required_scope, episode.observation_target, episode.aggregation) == (
        "multi_window",
        "event",
        "compare",
    )
    assert (temporal.required_scope, temporal.observation_target, temporal.aggregation) == (
        "multi_window",
        "event",
        "compare",
    )
    assert temporal.boundary_hint
    assert (synopsis.required_scope, synopsis.observation_target, synopsis.aggregation) == (
        "full_video",
        "event",
        "summarize",
    )
    synopsis_task = multiround._task_for_contract(
        InvestigationTask("q_summary", "Inspect this segment.", segment_id="seg_1"),
        synopsis,
    )
    assert synopsis_task.modality_hint == ()
    assert synopsis_task.expected_evidence == ""
    assert (progress.quantifier, progress.measurement_unit, progress.boundary_hint) == (
        "scalar_quantity",
        "task",
        "As one team enjoys pizza",
    )
    assert (causal.required_scope, causal.observation_target, causal.aggregation) == (
        "multi_window",
        "relation",
        "compare",
    )
    assert sequence_requirements["requires_temporal_sequence"] is True
    assert sequence_requirements["requires_state_tracking"] is True
    assert (outcome.required_scope, outcome.observation_target, outcome.aggregation) == (
        "multi_window",
        "event",
        "compare",
    )
    assert (agent_relation.quantifier, agent_relation.observation_target, agent_relation.aggregation) == (
        "comparison",
        "relation",
        "compare",
    )


def test_query_compiler_covers_errors10_global_and_cross_window_contracts() -> None:
    overtake_count = multiround.compile_query_contract(
        "How many times was the video recorder overtaken after reaching 1st place and failing to maintain it?"
    )
    food_sequence = multiround.compile_query_contract(
        "According to the video, which option correctly describes the sequence of the protagonist's eating-related events?",
        {"A": "Breakfast -> yogurt -> dinner", "B": "Yogurt -> breakfast -> dinner"},
    )
    final_position = multiround.compile_query_contract(
        "In that same last instance, what final position did the second person who overtook the video recorder ultimately finish in?"
    )
    second_overtaker = multiround.compile_query_contract(
        "In the last instance where the recorder reached 1st place, who was the second person to overtake them?"
    )
    final_decision = multiround.compile_query_contract(
        "After confronting his parents at the door, did Joe ultimately stick to his original idea?"
    )
    narrative_gap = multiround.compile_query_contract(
        "During this narrative gap between scenes, which option best represents Joe's internal monologue?"
    )
    uncertain_caller = multiround.compile_query_contract(
        "Who had called the police?",
        {"A": "Joe called.", "B": "It's uncertain, but it's not Joe's family."},
    )
    color_change = multiround.compile_query_contract(
        "What color change occurs when the red sauce is mixed with the white batter?"
    )

    assert (overtake_count.required_scope, overtake_count.quantifier, overtake_count.aggregation) == (
        "full_video",
        "total_count",
        "count",
    )
    assert (food_sequence.required_scope, food_sequence.quantifier, food_sequence.aggregation) == (
        "full_video",
        "order",
        "order",
    )
    assert (final_position.required_scope, final_position.observation_target) == ("multi_window", "entity")
    assert (second_overtaker.required_scope, second_overtaker.observation_target) == ("multi_window", "entity")
    assert (final_decision.required_scope, final_decision.aggregation) == ("multi_window", "compare")
    assert (narrative_gap.required_scope, narrative_gap.aggregation) == ("multi_window", "compare")
    assert (uncertain_caller.required_scope, uncertain_caller.aggregation) == ("multi_window", "compare")
    assert "elimination" in uncertain_caller.boundary_hint
    assert (color_change.required_scope, color_change.aggregation) == ("multi_window", "compare")


def test_cross_window_identity_and_narrative_requirements_trigger_p1_protocols() -> None:
    identity = multiround.compile_query_requirements(
        "In that same last instance, what final position did the second person who overtook the recorder finish in?"
    )
    narrative = multiround.compile_query_requirements(
        "During this narrative gap, which option best represents Joe's internal monologue?"
    )

    assert identity["requires_identity_link"] is True
    assert identity["requires_event_participant_link"] is True
    assert narrative["requires_narrative_inference"] is True


def test_query_compiler_keeps_explicitly_bounded_event_count_local() -> None:
    contract = multiround.compile_query_contract(
        "How many times does the title card appear in the first 5 minutes?"
    )

    assert contract.required_scope == "multi_window"
    assert contract.quantifier == "total_count"


def test_condition_alignment_reuses_semantic_ids_across_reasoner_rounds() -> None:
    first = ReasonerDecision(
        action="investigate",
        primary_gap=EvidenceGap(
            "gap_r1",
            "count cylinders",
            success_conditions=(
                "Count all visible cylinders",
                "Confirm exterior factory setting",
                "Confirm side-by-side layout",
            ),
        ),
    )
    second = ReasonerDecision(
        action="investigate",
        primary_gap=EvidenceGap(
            "gap_r2",
            "verify the same count",
            success_conditions=(
                "Two green cylinders are countable",
                "Factory exterior context is visible",
                "Objects are side by side",
            ),
        ),
    )

    aligned_first, registry = multiround._align_decision_conditions(first, ())
    aligned_second, registry = multiround._align_decision_conditions(second, registry)

    assert [item.condition_id for item in aligned_second.primary_gap.conditions] == [
        item.condition_id for item in aligned_first.primary_gap.conditions
    ]
    assert len(registry) == 3


def test_readiness_fails_closed_when_active_condition_has_no_result() -> None:
    status = multiround._apply_readiness_dashboard(
        {"ready_for_answer": True},
        (),
        (),
        (),
        "A",
        ("stable_condition_1",),
    )

    assert status["ready_for_answer"] is False
    assert status["unresolved_critical_condition_ids"] == ["stable_condition_1"]


def test_readiness_allows_partial_grounding_with_supported_majority_and_no_conflict() -> None:
    evidence = EvidenceRecord(
        evidence_id="ev_direct",
        beat_id="",
        start_sec=1.0,
        end_sec=2.0,
        modality="visual",
        pointer="virtual://direct",
        verbatim="Direct visual support for the discriminative claim.",
        frame_refs=("direct.jpg",),
        observation_polarity="positive",
    )
    report = InvestigationReport(
        query_id="q_direct",
        status="satisfied",
        evidence=(evidence,),
        gap_id="gap_direct",
        resolution="partial",
        condition_results=(
            ConditionResult("c1", "satisfied", "First key atom is visible.", ("ev_direct",)),
            ConditionResult("c2", "satisfied", "Second key atom is visible.", ("ev_direct",)),
            ConditionResult("c3", "unknown", "Background atom remains uncertain."),
        ),
    )

    status = multiround._apply_readiness_dashboard(
        {"ready_for_answer": True},
        (evidence,),
        (evidence,),
        (report,),
        "B. supported option",
        ("c1", "c2", "c3"),
    )

    assert status["grounded_ready"] is False
    assert status["partial_grounded_ready"] is True
    assert status["grounding_level_ready"] == "partial"
    assert status["ready_for_answer"] is True


def test_readiness_blocks_conflicted_structured_slot() -> None:
    evidence = EvidenceRecord(
        evidence_id="ev_conflict", beat_id="", start_sec=1.0, end_sec=2.0,
        modality="visual", pointer="virtual://conflict", verbatim="Conflicting observations.",
        frame_refs=("frame.jpg",), observation_polarity="positive",
        operation_metadata={"conflicted_slot_ids": ["jacket_color"]},
    )
    report = InvestigationReport(
        query_id="q_conflict", status="satisfied", evidence=(evidence,), gap_id="gap_conflict",
        condition_results=(ConditionResult("c1", "satisfied", "Target person is visible.", ("ev_conflict",)),),
    )

    status = multiround._apply_readiness_dashboard(
        {"ready_for_answer": True}, (evidence,), (evidence,), (report,), "A", ("c1",)
    )

    assert status["ready_for_answer"] is False
    assert status["conflicted_structured_slot_ids"] == ["slot:jacket_color"]


def test_global_absence_requires_resolution_qualified_negative(tmp_path: Path) -> None:
    base = _workspace(tmp_path)
    workspace = VirtualVideoWorkspace.create(
        tmp_path / "absence-case",
        manifest=base.manifest,
        case=VirtualVideoCase(
            case_id="absence-case",
            question="Which skill is absent?",
            options={"A": "backflip skill", "B": "handstand skill"},
            gold="A",
            target_segment_id="seg_target",
            target_virtual_interval=(0.0, 5.0),
        ),
    )

    def negative(evidence_id: str, qualified: bool) -> EvidenceRecord:
        qualification = {
            "status": "qualified_absence" if qualified else "unknown_due_to_coverage",
            "coverage_ratio": 1.0,
            "sampling_interval_sec": 0.5 if qualified else 2.0,
            "expected_dwell_time_sec": 2.0,
            "visibility_status": "clear",
        }
        return EvidenceRecord(
            evidence_id=evidence_id, beat_id="", start_sec=0.0, end_sec=5.0,
            modality="visual", pointer=f"virtual://{evidence_id}", verbatim="The backflip was not found.",
            frame_refs=("frame.jpg",), observation_polarity="negative",
            operation_metadata={
                "target_presence": {"target": "backflip skill", "status": "absent", "confidence": 0.9},
                    "qualified_absence": qualified,
                    "absence_status": qualification["status"],
                    "absence_qualification": qualification,
                    "absence_resolution_fps": 2.0 if qualified else 0.5,
            },
        )

    weak = negative("ev_weak", False)
    strong = negative("ev_strong", True)

    weak_gate = multiround._global_absence_gate(workspace, "A. backflip skill", (weak,), (weak,))
    strong_gate = multiround._global_absence_gate(workspace, "A. backflip skill", (strong,), (strong,))

    assert weak_gate["reason"] == "option_specific_absence_evidence_missing"
    assert strong_gate["reason"] == "global_absence_grounded"


def test_contract_task_does_not_override_reasoner_sampling_plan() -> None:
    contract = multiround.compile_query_contract(
        "Who overtook the rider second?",
        {"A": "The rider in blue", "B": "The rider in black"},
    )

    task = multiround._task_for_contract(
        InvestigationTask(
            "q_order",
            "Inspect the overtaking order.",
            segment_id="seg_1",
            sampling_floor_fps=1.0,
            temporal_resolution_rationale="The requested evidence is expected to remain visible for about two seconds.",
        ),
        contract,
    )

    assert contract.aggregation == "order"
    assert task.sampling_floor_fps == 1.0
    assert task.temporal_resolution_rationale.startswith("The requested evidence")
    source = inspect.getsource(multiround._task_for_contract)
    assert "contract." not in source


def test_sampling_plan_tracks_unspecified_floor_and_downgrades_missing_rationale() -> None:
    unspecified = InvestigationTask("q_default", "Inspect persistent evidence.")
    incomplete = InvestigationTask(
        "q_incomplete", "Inspect a brief event.", sampling_floor_fps=2.0, priority=1.0
    )

    assert unspecified.sampling_floor_fps == 0.5
    assert unspecified.sampling_floor_specified is False
    assert incomplete.sampling_floor_specified is True
    assert incomplete.priority == 0.8


def test_discriminative_audit_preserves_completed_gate_when_reason_is_missing() -> None:
    gate = {"passed": True, "reason": "verified_window_evidence"}

    missing = multiround._apply_answer_audit(
        gate,
        ReasonerDecision(action="answer", answer="A", support_status="supported"),
        required=True,
    )
    supported = multiround._apply_answer_audit(
        gate,
        ReasonerDecision(
            action="answer",
            answer="A",
            support_status="supported",
            support_reason="Witnessed before and after states distinguish A from B.",
        ),
        required=True,
    )

    assert missing["passed"] is True
    assert missing["answer_audit_status"] == "inferred_from_completion"
    assert missing["answer_audit_missing_fields"] == ["support_reason"]
    assert supported["passed"] is True


def test_total_count_option_verdicts_use_only_canonical_snapshot() -> None:
    contract = multiround.compile_query_contract(
        "How many times was the rider overtaken?",
        {"A": "Three", "B": "Seven"},
    )
    table = multiround._option_verdict_table(
        {"A": "Three", "B": "Seven"},
        contract,
        {
            "confirmed_events": [
                {"candidate_id": f"event_{index}", "evidence_ids": [f"ev_{index}"]}
                for index in range(3)
            ],
            "duplicate_suspect_events": [],
            "raw_candidate_counts": {"events": 7},
        },
        {},
    )

    assert table["best_option"] == "A"
    assert table["option_verdicts"]["A"]["status"] == "supported"
    assert table["option_verdicts"]["B"]["status"] == "contradicted"


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


def test_workspace_task_resolution_expands_global_alias_and_repairs_timed_segment(tmp_path: Path) -> None:
    workspace = _two_chunk_workspace(tmp_path)
    global_task = InvestigationTask("q_global", "Inspect the full source.", segment_id="seg_full")
    timed_task = InvestigationTask(
        "q_timed",
        "Inspect the requested virtual interval.",
        segment_id="invented_segment",
        time_range=(10.5, 11.5),
    )

    global_resolved = multiround._resolve_workspace_tasks(workspace, (global_task,), limit=2)
    timed_resolved = multiround._resolve_workspace_tasks(workspace, (timed_task,), limit=1)

    assert [task.segment_id for task in global_resolved] == ["seg_target_a", "seg_target_b"]
    assert all(task.query_id.startswith("q_global_seg_") for task in global_resolved)
    assert timed_resolved[0].segment_id == "seg_target_b"


def test_workspace_task_resolution_splits_cross_segment_time_range(tmp_path: Path) -> None:
    workspace = _two_chunk_workspace(tmp_path)
    task = InvestigationTask(
        "q_broad",
        "Inspect the full requested interval.",
        segment_id="seg_target_a",
        time_range=(0.0, 15.0),
    )

    resolved = multiround._resolve_workspace_tasks(workspace, (task,), limit=4)

    assert [item.segment_id for item in resolved] == ["seg_target_a", "seg_noise", "seg_target_b"]
    assert [item.time_range for item in resolved] == [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0)]
    assert len({item.query_id for item in resolved}) == 3


def test_workspace_task_resolution_round_robins_across_parent_ranges(tmp_path: Path) -> None:
    workspace = _two_chunk_workspace(tmp_path)
    early = InvestigationTask("q_early", "Inspect the early range.", segment_id="seg_target_a", time_range=(0.0, 10.0))
    late = InvestigationTask("q_late", "Inspect the late range.", segment_id="seg_target_b", time_range=(10.0, 15.0))

    resolved = multiround._resolve_workspace_tasks(workspace, (early, late), limit=2)

    assert [item.segment_id for item in resolved] == ["seg_target_a", "seg_target_b"]
    assert [item.time_range for item in resolved] == [(0.0, 5.0), (10.0, 15.0)]


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


def test_contrastive_progress_measurement_requires_other_subject_binding(tmp_path: Path) -> None:
    question = "As one team enjoys pizza, how many tasks has the other team completed?"
    manifest = VirtualVideoManifest(
        workspace_id="progress",
        segments=(VirtualVideoSegment("seg_1", "source", "source.mp4", 0.0, 60.0, 0.0, 60.0),),
    )
    case = VirtualVideoCase(
        case_id="progress",
        question=question,
        options={"A": "11", "B": "15"},
        gold="A",
        target_segment_id="seg_1",
        target_virtual_interval=(0.0, 60.0),
    )
    workspace = VirtualVideoWorkspace.create(tmp_path / "progress", manifest=manifest, case=case)

    def evidence(evidence_id: str, value: int, subject_id: str) -> EvidenceRecord:
        return EvidenceRecord(
            evidence_id=evidence_id,
            beat_id="",
            start_sec=20.0,
            end_sec=25.0,
            modality="ocr",
            pointer=f"virtual://{evidence_id}",
            verbatim=f"A task counter reads {value}.",
            frame_refs=(f"{evidence_id}.jpg",),
            attestation_model="test-vlm",
            operation_metadata={
                "measurements": [
                    {
                        "value": value,
                        "unit": "task",
                        "measurement_semantics": "cumulative",
                        "subject_id": subject_id,
                        "boundary_relation": "at",
                        "binding_status": "explicit" if subject_id else "contextual",
                    }
                ]
            },
        )

    contract = multiround.compile_query_contract(question, case.options)
    requirements = multiround.compile_query_requirements(question)
    unbound = evidence("ev_unbound", 15, "")
    bound = evidence("ev_bound", 11, "other_team")

    blocked = multiround._answer_completion_gate(
        workspace,
        contract,
        "B. 15",
        (unbound.evidence_id,),
        (),
        (unbound,),
        query_requirements=requirements,
    )
    passed = multiround._answer_completion_gate(
        workspace,
        contract,
        "A. 11",
        (bound.evidence_id,),
        (),
        (bound,),
        query_requirements=requirements,
    )

    assert requirements["measurement_subject_role"] == "other_team"
    assert blocked["reason"] == "scalar_measurement_subject_binding_missing"
    assert passed["reason"] == "scalar_quantity_grounded"


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


def test_task_progress_fingerprint_ignores_reworded_query_id_but_allows_strategy_shift() -> None:
    first = InvestigationTask(
        "q_first", "Inspect the same overtake.", segment_id="seg_1", time_range=(10.0, 20.0),
        inspection_mode="event_window", sampling_floor_fps=1.0,
        temporal_resolution_rationale="The event lasts about two seconds.",
    )
    renamed = replace(first, query_id="q_renamed")
    denser = replace(first, query_id="q_denser", sampling_floor_fps=2.0)

    assert multiround._task_progress_fingerprint(first) == multiround._task_progress_fingerprint(renamed)
    assert multiround._task_progress_fingerprint(first) != multiround._task_progress_fingerprint(denser)


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
                citations=("ev_q1_001",),
                support_status="insufficient",
                support_reason="The related visual evidence favors 11 but remains inconclusive.",
            )
        return ReasonerDecision(
            action="answer",
            answer="A. 7",
            citations=("ev_q1_001",),
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


def test_participant_link_repair_targets_ordinal_event_participant(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    workspace = replace(
        workspace,
        case=replace(
            workspace.case,
            question="In the last instance, what color clothes did the second person who ultimately overtook the recorder wear?",
            options={"A": "red clothes", "B": "green clothes"},
        ),
    )
    evidence = EvidenceRecord(
        evidence_id="ev_overtakes",
        beat_id="",
        start_sec=0.0,
        end_sec=4.0,
        modality="visual",
        pointer="virtual://overtakes",
        verbatim="Two racers overtake the recorder.",
        frame_refs=("frame.jpg",),
        operation_metadata={
            "events": [
                {
                    "event_key": "first racer overtakes recorder",
                    "event_class": "overtake",
                    "counting_unit": "overtake_episode",
                    "participant_ids": ["recorder", "first racer"],
                    "participants": [{"participant_id": "first racer", "role": "overtaker"}],
                    "start_sec": 1.0,
                    "end_sec": 1.5,
                    "supports_question_event": True,
                },
                {
                    "event_key": "second racer overtakes recorder",
                    "event_class": "overtake",
                    "counting_unit": "overtake_episode",
                    "participant_ids": ["recorder", "second racer"],
                    "participants": [{
                        "participant_id": "second racer",
                        "role": "overtaker",
                        "visual_signature": "green jacket; black helmet",
                    }],
                    "start_sec": 3.0,
                    "end_sec": 3.5,
                    "supports_question_event": True,
                },
            ],
        },
    )

    tasks = multiround._event_participant_association_tasks(
        workspace, (evidence,), round_id=2, limit=2,
    )

    assert tasks[0].inspection_mode == "entity_association"
    assert tasks[0].reference_entities[0]["participant_id"] == "second racer"
    assert tasks[0].reference_entities[0]["entity_hypothesis_id"]


def test_participant_link_repair_targets_ordinal_participant_in_consolidated_event(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    workspace = replace(
        workspace,
        case=replace(
            workspace.case,
            question="In the last instance, what color clothes did the second person who ultimately overtook the recorder wear?",
            options={"A": "red clothes", "B": "green clothes"},
        ),
    )
    evidence = EvidenceRecord(
        evidence_id="ev_consolidated_overtake",
        beat_id="",
        start_sec=0.0,
        end_sec=4.0,
        modality="visual",
        pointer="virtual://consolidated_overtake",
        verbatim="Two racers overtake the recorder in one continuous episode.",
        frame_refs=("frame.jpg",),
        operation_metadata={
            "events": [{
                "event_key": "two racers overtake recorder",
                "event_class": "overtake",
                "counting_unit": "overtake_episode",
                "participant_ids": ["recorder", "first racer", "second racer"],
                "participants": [
                    {"participant_id": "recorder", "role": "overtaken"},
                    {"participant_id": "first racer", "role": "overtaker", "visual_signature": "black jacket"},
                    {"participant_id": "second racer", "role": "overtaker", "visual_signature": "green jacket"},
                ],
                "start_sec": 1.0,
                "end_sec": 3.5,
                "supports_question_event": True,
            }],
        },
    )

    tasks = multiround._event_participant_association_tasks(
        workspace, (evidence,), round_id=2, limit=2,
    )

    assert tasks[0].inspection_mode == "entity_association"
    assert tasks[0].reference_entities[0]["participant_id"] == "second racer"
    assert tasks[0].reference_entities[0]["visual_signature"] == "green jacket"


def test_participant_link_repair_first_dispatches_one_anchor_event_sweep(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    tasks = multiround._event_participant_association_tasks(
        workspace, (), round_id=1, limit=2,
    )

    assert len(tasks) == 1
    assert tasks[0].query_id.startswith("participant_anchor_")
    assert tasks[0].inspection_mode == "enumerate_events"
    assert tasks[0].sampling_floor_fps == 2.0
    wide_workspace = replace(
        workspace,
        manifest=replace(
            workspace.manifest,
            segments=tuple(
                VirtualVideoSegment(
                    f"seg_{index}", "source", "source.mp4",
                    float(index * 5), float((index + 1) * 5),
                    float(index * 5), float((index + 1) * 5), "content",
                )
                for index in range(8)
            ),
        ),
    )
    wide_tasks = multiround._event_participant_association_tasks(
        wide_workspace, (), round_id=1, limit=4,
    )
    assert len(wide_tasks) == 6
    attempted = EvidenceRecord(
        evidence_id="ev_anchor_attempt",
        beat_id="",
        start_sec=0.0,
        end_sec=5.0,
        modality="visual",
        pointer="virtual://anchor_attempt",
        verbatim="No structured anchor event was recovered.",
        frame_refs=("frame.jpg",),
        task_id="participant_anchor_r1_001",
    )
    assert multiround._event_participant_association_tasks(
        workspace, (attempted,), round_id=2, limit=2,
    ) == ()


def test_canonical_option_table_uses_resolved_participant_attribute() -> None:
    contract = multiround.compile_query_contract(
        "In the last instance, what color clothes did the second person who ultimately overtook the recorder wear?",
        {"A": "red clothes", "B": "green clothes", "C": "blue clothes"},
    )
    snapshot = {
        "resolved_entities": [{
            "entity_id": "second_racer",
            "attributes": {"clothing_color": "green", "helmet_color": "black"},
            "evidence_ids": ["ev_link"],
        }],
    }

    table = multiround._option_verdict_table(
        {"A": "red clothes", "B": "green clothes", "C": "blue clothes"},
        contract,
        snapshot,
        {},
    )

    assert table["best_option"] == "B"
    assert table["option_verdicts"]["B"]["status"] == "supported"
    assert table["discriminating_predicate"] == "resolved_entity_attributes"


def test_canonical_option_table_uses_narrative_hypothesis_assessments() -> None:
    options = {"A": "Joe leaves as planned.", "B": "Joe changes his mind and stays."}
    contract = multiround.compile_query_contract(
        "During this narrative gap, which option best represents Joe's internal monologue?", options,
    )
    snapshot = {
        "inferred_facts": [{
            "fact_id": "bridge_1",
            "evidence_ids": ["ev_bridge"],
            "hypothesis_assessments": [
                {"option_id": "A", "status": "contradicted", "reason": "Joe is shown staying."},
                {"option_id": "B", "status": "supported", "reason": "The outcome reverses his plan."},
            ],
        }],
    }

    table = multiround._option_verdict_table(options, contract, snapshot, {})

    assert table["best_option"] == "B"
    assert table["option_verdicts"]["A"]["status"] == "contradicted"
    assert table["discriminating_predicate"] == "canonical_narrative_inference"


def test_canonical_forced_answer_overrides_stale_best_candidate_only_when_unique() -> None:
    evidence = EvidenceRecord(
        evidence_id="ev_bridge",
        beat_id="",
        start_sec=1.0,
        end_sec=2.0,
        modality="visual",
        pointer="virtual://bridge",
        verbatim="The complete bridge supports H.",
        frame_refs=("bridge.jpg",),
        attestation_model="test-vlm",
    )
    options = {"F": "unsupported", "H": "supported narrative"}
    completion = {
        "option_verdict_table": {
            "best_option": "H",
            "option_verdicts": {
                "F": {"status": "contradicted", "evidence_ids": ["ev_bridge"]},
                "H": {"status": "supported", "evidence_ids": ["ev_bridge"]},
            },
        },
    }

    forced = multiround._canonical_forced_answer(options, completion, (evidence,))

    assert forced == ("H. supported narrative", ("ev_bridge",))
    ambiguous = {
        "option_verdict_table": {
            "best_option": "H",
            "option_verdicts": {
                "F": {"status": "supported", "evidence_ids": ["ev_bridge"]},
                "H": {"status": "supported", "evidence_ids": ["ev_bridge"]},
            },
        },
    }
    assert multiround._canonical_forced_answer(options, ambiguous, (evidence,)) is None


def test_single_segment_narrative_repair_splits_setup_and_outcome_then_carries_context(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    first = multiround._narrative_bridge_repair_tasks(
        workspace, (), round_id=1, limit=2,
    )
    incomplete = EvidenceRecord(
        evidence_id="ev_setup",
        beat_id="",
        start_sec=0.0,
        end_sec=2.0,
        modality="visual",
        pointer="virtual://setup",
        verbatim="Only the setup is observed.",
        frame_refs=("setup.jpg",),
        task_id="narrative_bridge_r1_001",
        operation_metadata={
            "narrative_facts": [{
                "fact_id": "bridge_partial",
                "setup_state": "Joe intends to leave.",
                "outcome_state": "",
                "inference": "He may reconsider.",
            }],
        },
    )
    followup = multiround._narrative_bridge_repair_tasks(
        workspace, (incomplete,), round_id=2, limit=2,
    )

    assert len(first) == 2
    assert "setup" in first[0].goal.casefold()
    assert "outcome" in first[1].goal.casefold()
    assert followup[0].reference_facts[0]["fact_id"] == "bridge_partial"
    assert "outcome" in followup[0].goal.casefold()


def test_rejected_answer_repair_reobserves_cited_window_contrastively(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    evidence = EvidenceRecord(
        evidence_id="ev_candidate",
        beat_id="",
        start_sec=1.0,
        end_sec=2.0,
        modality="visual",
        pointer="virtual://candidate",
        verbatim="A jersey number is visible but ambiguous.",
        frame_refs=("frame.jpg",),
        attestation_model="test-vlm",
    )
    decision = ReasonerDecision(
        action="answer",
        answer="A. 7",
        citations=(evidence.evidence_id,),
        support_status="insufficient",
        support_reason="The frame does not distinguish 7 from 11.",
    )

    tasks = multiround._rejected_answer_repair_tasks(
        workspace,
        multiround.compile_query_contract(workspace.case.question, workspace.case.options),
        decision,
        (evidence,),
        {"reason": "answer_audit_insufficient"},
        round_id=2,
        limit=1,
    )

    assert len(tasks) == 1
    assert tasks[0].inspection_mode == "verify_claim"
    assert tasks[0].claim_to_verify == "7"
    assert tasks[0].alternative_answers == ("11",)
    assert tasks[0].time_range == (0.0, 5.0)


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


def test_source_coverage_requires_interval_coverage_not_only_segment_id(tmp_path: Path) -> None:
    workspace = _two_chunk_workspace(tmp_path)
    partial = EvidenceRecord(
        evidence_id="ev_partial_segment",
        beat_id="",
        start_sec=0.0,
        end_sec=2.0,
        modality="visual",
        pointer="virtual://partial",
        verbatim="Only the first two seconds were inspected.",
        frame_refs=("frame.jpg",),
        source_lineage=(
            {"source_video_id": "target", "segment_id": "seg_target_a", "virtual_time_range": [0.0, 2.0]},
        ),
    )

    coverage = multiround._source_coverage(workspace, (partial,))["target"]

    assert coverage["covered_segment_ids"] == []
    assert coverage["missing_segment_ids"] == ["seg_target_a", "seg_target_b"]
    assert coverage["segment_coverage"]["seg_target_a"]["coverage_ratio"] == 0.4


def test_event_coverage_repair_requests_entire_segment_at_high_resolution(tmp_path: Path) -> None:
    workspace = _two_chunk_workspace(tmp_path)
    contract = multiround.compile_query_contract(
        "How many times does the title card appear in the film?",
        {"A": "One", "B": "Two"},
    )

    task = multiround._coverage_repair_tasks(
        workspace, 2, ("seg_target_b",), contract, limit=1
    )[0]

    assert task.time_range == (10.0, 15.0)
    assert task.inspection_mode == "enumerate_events"
    assert task.sampling_floor_fps == 2.0


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


def test_window_sampling_supports_a_512_frame_hard_cap(tmp_path: Path) -> None:
    segment = VirtualVideoSegment("seg", "source", "/tmp/source.mp4", 0.0, 300.0, 0.0, 300.0, "content")
    manifest = VirtualVideoManifest(workspace_id="cap", segments=(segment,))
    case = VirtualVideoCase(
        case_id="cap",
        question="What happens?",
        options={"A": "One", "B": "Two"},
        gold="A",
        target_segment_id="seg",
        target_virtual_interval=(0.0, 300.0),
    )
    workspace = VirtualVideoWorkspace.create(tmp_path / "cap", manifest=manifest, case=case)
    investigator = VirtualVideoInvestigator(workspace, sampler=_sampler)

    window = investigator.inspect_window(0.0, 300.0, fps=2.0, max_frames=999, query_id="cap")

    assert window["sampling"]["max_frames"] == 512
    assert window["sampling"]["actual_frames"] == 512


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


def test_contract_task_preserves_reasoner_selected_inspection_mode() -> None:
    contract = multiround.compile_query_contract("How many times does a title card appear in this video?")
    task = InvestigationTask(
        query_id="q_title_cards",
        goal="Inspect one source segment for title-card appearances.",
        segment_id="seg_0001",
        modality_hint=("visual",),
        expected_evidence="timestamped title-card occurrences",
        inspection_mode="enumerate_events",
    )

    compiled = multiround._task_for_contract(task, contract)

    assert compiled.inspection_mode == "enumerate_events"
    assert contract.required_observability == ("visual",)


def test_event_candidate_ledger_compacts_aliases_and_exposes_generic_windows() -> None:
    lineage = (
        {
            "segment_id": "seg_1",
            "source_video_id": "source",
            "virtual_time_range": [0.0, 60.0],
            "source_time_range": [0.0, 60.0],
        },
    )

    def record(evidence_id: str, key: str, start: float, end: float) -> EvidenceRecord:
        return EvidenceRecord(
            evidence_id=evidence_id,
            beat_id="",
            start_sec=start,
            end_sec=end,
            modality="visual",
            pointer=f"virtual://{evidence_id}",
            verbatim="A question-relevant audition is visible.",
            frame_refs=(f"{evidence_id}.jpg",),
            attestation_model="test-vlm",
            source_lineage=lineage,
            operation_metadata={
                "events": [
                    {
                        "local_id": "event_1",
                        "event_key": key,
                        "description": "one occurrence",
                        "start_sec": start,
                        "end_sec": end,
                        "supports_question_event": True,
                    }
                ]
            },
        )

    evidence = (
        record("ev_1", "dance group audition: Light Balance", 10.0, 20.0),
        record("ev_2", "audition: light balance", 12.0, 22.0),
        record("ev_3", "dance group audition on America's Got Talent", 30.0, 40.0),
        record("ev_4", "audition: light balance", 50.0, 55.0),
    )
    ledger = multiround._event_candidate_ledger(evidence)
    contract = multiround.compile_query_contract("How many auditions are included in this video?")
    digest = multiround._evidence_digest(evidence, contract)

    assert ledger["confirmed_event_candidate_count"] == 2
    assert ledger["confirmed_event_candidates"][0]["signature"] == "light balance"
    assert ledger["confirmed_event_candidates"][0]["evidence_ids"] == ["ev_1", "ev_2"]
    assert ledger["confirmed_event_candidates"][1]["evidence_ids"] == ["ev_4"]
    assert ledger["unresolved_event_candidate_count"] == 1
    assert [row["evidence_kind"] for row in digest] == [
        "event_candidate",
        "event_candidate",
        "event_candidate_unresolved",
    ]


def test_event_candidate_ledger_merges_named_audition_phases_with_variant_keys() -> None:
    def record(
        evidence_id: str,
        key: str,
        start: float,
        end: float,
        *,
        from_previous: bool,
        to_next: bool,
    ) -> EvidenceRecord:
        return EvidenceRecord(
            evidence_id=evidence_id,
            beat_id="",
            start_sec=start,
            end_sec=end,
            modality="visual",
            pointer=f"virtual://{evidence_id}",
            verbatim="A phase of the Mayyas audition is visible.",
            frame_refs=(f"{evidence_id}.jpg",),
            attestation_model="test-vlm",
            source_lineage=(
                {
                    "segment_id": "seg_1",
                    "source_video_id": "source",
                    "virtual_time_range": [start, end],
                },
            ),
            operation_metadata={
                "events": [
                    {
                        "local_id": "event_1",
                        "event_key": key,
                        "description": key,
                        "start_sec": start,
                        "end_sec": end,
                        "supports_question_event": True,
                        "continues_from_previous": from_previous,
                        "continues_to_next": to_next,
                    }
                ]
            },
        )

    ledger = multiround._event_candidate_ledger(
        (
            record(
                "ev_performance",
                "dance group audition: Mayyas",
                0.0,
                60.0,
                from_previous=False,
                to_next=True,
            ),
            record(
                "ev_judging",
                "judging: Mayyas",
                60.0,
                120.0,
                from_previous=True,
                to_next=False,
            ),
        )
    )

    assert ledger["confirmed_event_candidate_count"] == 1
    assert ledger["confirmed_event_candidates"][0]["signature"] == "mayyas"
    assert ledger["confirmed_event_candidates"][0]["evidence_ids"] == ["ev_performance", "ev_judging"]


def test_event_candidate_ledger_uses_counting_unit_for_phase_merging() -> None:
    def record(
        evidence_id: str,
        *,
        key: str,
        start: float,
        end: float,
        event_class: str,
        counting_unit: str,
        participant: str,
        phase: str,
    ) -> EvidenceRecord:
        return EvidenceRecord(
            evidence_id=evidence_id,
            beat_id="",
            start_sec=start,
            end_sec=end,
            modality="visual",
            pointer=f"virtual://{evidence_id}",
            verbatim=key,
            frame_refs=(f"{evidence_id}.jpg",),
            attestation_model="test-vlm",
            source_lineage=(
                {
                    "segment_id": "seg_1",
                    "source_video_id": "source",
                    "virtual_time_range": [start, end],
                },
            ),
            operation_metadata={
                "events": [
                    {
                        "event_key": key,
                        "event_class": event_class,
                        "counting_unit": counting_unit,
                        "participant_ids": [participant],
                        "phase": phase,
                        "description": key,
                        "start_sec": start,
                        "end_sec": end,
                        "supports_question_event": True,
                    }
                ]
            },
        )

    audition = multiround._event_candidate_ledger(
        (
            record(
                "ev_intro",
                key="Mayyas introduction",
                start=0.0,
                end=30.0,
                event_class="audition",
                counting_unit="audition_group",
                participant="Mayyas",
                phase="intro",
            ),
            record(
                "ev_result",
                key="Mayyas result",
                start=90.0,
                end=120.0,
                event_class="audition",
                counting_unit="audition_group",
                participant="Mayyas dance group",
                phase="result",
            ),
        )
    )
    news = multiround._event_candidate_ledger(
        (
            record(
                "ev_news_1",
                key="Daily news",
                start=0.0,
                end=10.0,
                event_class="news_segment",
                counting_unit="news_broadcast_appearance",
                participant="Daily news",
                phase="main",
            ),
            record(
                "ev_news_2",
                key="Daily news",
                start=30.0,
                end=40.0,
                event_class="news_segment",
                counting_unit="news_broadcast_appearance",
                participant="Daily news",
                phase="main",
            ),
        )
    )

    assert audition["confirmed_event_candidate_count"] == 1
    assert audition["confirmed_event_candidates"][0]["phases"] == ["intro", "result"]
    assert news["confirmed_event_candidate_count"] == 2


def test_semantic_event_repair_targets_unresolved_window_once(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    contract = multiround.compile_query_contract("How many auditions are included in this video?")
    status = {
        "ready_for_answer": False,
        "unresolved_event_windows": [
            {
                "source_video_id": "target",
                "virtual_time_range": [1.0, 4.0],
                "evidence_ids": ["ev_generic"],
            }
        ],
    }

    reason, tasks = multiround._semantic_contract_repair_tasks(
        workspace,
        contract,
        {},
        status,
        (),
        round_id=2,
        limit=4,
    )

    assert reason == "event_candidate_unresolved"
    assert len(tasks) == 1
    assert tasks[0].inspection_mode == "enumerate_events"
    assert tasks[0].time_range == (1.0, 4.0)


def test_semantic_score_repair_scans_around_readable_non_boundary_checkpoint(tmp_path: Path) -> None:
    manifest = VirtualVideoManifest(
        workspace_id="score-context",
        segments=(VirtualVideoSegment("seg_1", "source", "source.mp4", 0.0, 600.0, 0.0, 600.0),),
    )
    case = VirtualVideoCase(
        case_id="score-context",
        question="What was the halftime score?",
        options={"A": "32 - 23", "B": "37 - 27"},
        gold="A",
        target_segment_id="seg_1",
        target_virtual_interval=(0.0, 600.0),
    )
    workspace = VirtualVideoWorkspace.create(tmp_path / "score-context", manifest=manifest, case=case)
    evidence = EvidenceRecord(
        evidence_id="ev_live_score",
        beat_id="",
        start_sec=300.0,
        end_sec=305.0,
        modality="ocr",
        pointer="virtual://score-context/live",
        verbatim="A live scoreboard reads 37-27, but the phase is unknown.",
        frame_refs=("score.jpg",),
        attestation_model="test-vlm",
        task_id="r1_score",
        source_lineage=(
            {"segment_id": "seg_1", "source_video_id": "source", "virtual_time_range": [300.0, 305.0]},
        ),
        operation_metadata={
            "measurements": [
                {
                    "value": value,
                    "unit": "point",
                    "quantity_type": "score",
                    "subject_id": subject,
                    "event_id": "live_play",
                    "boundary_relation": "unknown",
                    "binding_status": "contextual",
                }
                for value, subject in ((37, "home"), (27, "guest"))
            ]
        },
    )

    reason, tasks = multiround._semantic_contract_repair_tasks(
        workspace,
        multiround.compile_query_contract(case.question, case.options),
        {},
        {"ready_for_answer": False},
        (evidence,),
        round_id=2,
        limit=4,
    )

    assert reason == "boundary_score_context_missing"
    assert len(tasks) == 1
    assert tasks[0].query_id.startswith("semantic_score_context_")
    assert tasks[0].segment_id == "seg_1"
    assert tasks[0].time_range == (120.0, 335.0)


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


def test_identity_anchor_contract_supports_consumption_defined_measurement_subject() -> None:
    question = (
        "How many calories has the person, who consumed a $100 golden burger, already eaten when he meets his teammate?"
    )

    requirements = multiround.compile_query_requirements(question)

    assert requirements["requires_identity_link"] is True
    assert requirements["identity_anchor_terms"] == ["100", "golden", "burger"]
    assert requirements["measurement_subject_role"] == "anchored_subject"


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
    assert result.grounding_status == "verified_strict"
    assert result.grounding_level == "strict"
    assert result.retrieval_status == "sufficient"
    assert result.verification_reason == "full_source_coverage_verified"
    assert reasoner.calls == 3
    assert reasoner.completion_statuses[1]["missing_segment_ids"] == ["seg_target_a", "seg_target_b"]
    assert reasoner.completion_statuses[2]["missing_segment_ids"] == []
    assert len(result.citations) == 1
    aggregate = next(item for item in result.evidence if item.evidence_id == result.citations[0])
    assert aggregate.modality == "derived"
    assert "ev_q_chunk_a_001" in aggregate.parent_evidence_ids
    assert len(aggregate.parent_evidence_ids) == 3
    assert aggregate.entity_ids == ("scholar_1", "scholar_2", "scholar_3")
    assert len(aggregate.operation_metadata["entity_clusters"]) == 3
    repair = next(row for row in result.trace if row.get("type") == "repair_override")
    assert repair["missing_segment_ids"] == ["seg_target_a", "seg_target_b"]
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


def test_contradicted_candidate_remains_available_only_for_ungated_best_effort() -> None:
    evidence = EvidenceRecord(
        evidence_id="ev_visual",
        beat_id="",
        start_sec=1.0,
        end_sec=2.0,
        modality="visual",
        pointer="virtual://ev_visual",
        verbatim="A directly observed but disputed scene.",
        frame_refs=("frame.jpg",),
        attestation_model="test-vlm",
    )
    decision = ReasonerDecision(
        action="answer",
        answer="B. Best effort",
        citations=(evidence.evidence_id,),
        support_status="contradicted",
    )

    assert multiround._candidate_can_be_forced(decision, (evidence,)) is True
    assert multiround._answer_support_rank(decision) == 0


def test_gate_rank_does_not_let_supported_audit_override_canonical_count_mismatch() -> None:
    decision = ReasonerDecision(
        action="answer",
        answer="B. 8",
        citations=("ev_visual",),
        support_status="supported",
    )

    assert multiround._candidate_gate_rank(
        decision,
        {"passed": False, "reason": "event_count_answer_mismatch"},
    ) == 1
    assert multiround._candidate_gate_rank(
        decision,
        {"passed": True, "reason": "full_source_coverage_verified"},
    ) == 4


def test_full_video_condition_stays_unknown_until_source_coverage_is_complete() -> None:
    state = ConditionState(
        "gap_news_c1",
        status="satisfied",
        supporting_evidence_ids=("ev_news",),
        scope="full_video",
        quantifier="all_events",
        required_coverage=1.0,
    )
    partial = multiround._apply_condition_scope(
        {state.condition_id: state},
        {
            "adopted_source_video_id": "source",
            "source_coverage": {"source": {"covered_count": 1, "required_count": 3}},
        },
    )
    complete = multiround._apply_condition_scope(
        {state.condition_id: state},
        {
            "adopted_source_video_id": "source",
            "source_coverage": {"source": {"covered_count": 3, "required_count": 3}},
        },
    )

    assert partial[state.condition_id].status == "unknown"
    assert complete[state.condition_id].status == "satisfied"


def test_driver_bootstraps_an_empty_investigation_decision(tmp_path: Path) -> None:
    class NoTaskReasoner:
        def decide(self, **kwargs: object) -> ReasonerDecision:
            del kwargs
            return ReasonerDecision(action="investigate", tasks=())

    workspace = _workspace(tmp_path)
    result = VirtualVideoMultiRoundDriver(
        reasoner=NoTaskReasoner(),
        investigator=VirtualVideoInvestigator(workspace, sampler=_sampler),
        max_rounds=1,
        max_investigations=1,
    ).run(workspace)

    assert result.accepted_investigations == 1
    repair = next(row for row in result.trace if row.get("type") == "repair_override")
    assert repair["reason"] == "empty_investigation_bootstrap"


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


def test_spatial_gate_requires_same_frame_relation_and_reference_frame(tmp_path: Path) -> None:
    manifest = VirtualVideoManifest(
        workspace_id="spatial",
        segments=(VirtualVideoSegment("seg_1", "source", "source.mp4", 0.0, 5.0, 0.0, 5.0),),
    )
    case = VirtualVideoCase(
        case_id="spatial",
        question="Which direction is red facing in relation to green?",
        options={"A": "Right front", "B": "Left front"},
        gold="A",
        target_segment_id="seg_1",
        target_virtual_interval=(0.0, 5.0),
    )
    workspace = VirtualVideoWorkspace.create(tmp_path / "spatial", manifest=manifest, case=case)

    def evidence(reference_frame: str) -> EvidenceRecord:
        return EvidenceRecord(
            evidence_id="ev_spatial",
            beat_id="",
            start_sec=0.0,
            end_sec=5.0,
            modality="visual",
            pointer="virtual://spatial",
            verbatim="Red and green are visible together; red faces toward green's right-front side.",
            frame_refs=("spatial.jpg",),
            attestation_model="test-vlm",
            operation_metadata={
                "relations": [
                    {
                        "relation_type": "relative_facing",
                        "subject_id": "red",
                        "object_id": "green",
                        "value": "right_front",
                        "reference_frame": reference_frame,
                        "same_frame": True,
                        "witness_frame_indices": [0] if reference_frame else [],
                        "status": "supported",
                    }
                ]
            },
        )

    contract = multiround.compile_query_contract(case.question, case.options)
    requirements = multiround.compile_query_requirements(case.question)
    valid = evidence("object_egocentric")
    invalid = evidence("")
    ready = {"ready_for_answer": True}

    passed = multiround._answer_completion_gate(
        workspace,
        contract,
        "A. Right front",
        (valid.evidence_id,),
        (),
        (valid,),
        query_requirements=requirements,
        completion_status=ready,
    )
    blocked = multiround._answer_completion_gate(
        workspace,
        contract,
        "A. Right front",
        (invalid.evidence_id,),
        (),
        (invalid,),
        query_requirements=requirements,
        completion_status=ready,
    )

    assert passed["reason"] == "spatial_relation_grounded"
    assert blocked["reason"] == "spatial_relation_not_grounded"


def test_boundary_score_gate_requires_explicit_same_boundary_snapshot(tmp_path: Path) -> None:
    manifest = VirtualVideoManifest(
        workspace_id="score",
        segments=(VirtualVideoSegment("seg_1", "source", "source.mp4", 0.0, 5.0, 0.0, 5.0),),
    )
    case = VirtualVideoCase(
        case_id="score",
        question="What was the halftime score?",
        options={"A": "32 - 23", "B": "37 - 27"},
        gold="A",
        target_segment_id="seg_1",
        target_virtual_interval=(0.0, 5.0),
    )
    workspace = VirtualVideoWorkspace.create(tmp_path / "score", manifest=manifest, case=case)

    def evidence(event_id: str) -> EvidenceRecord:
        measurements = [
            {
                "value": value,
                "unit": "point",
                "quantity_type": "score",
                "subject_id": subject,
                "event_id": event_id,
                "boundary_relation": "at",
                "binding_status": "explicit",
            }
            for value, subject in ((32, "home"), (23, "guest"))
        ]
        return EvidenceRecord(
            evidence_id="ev_score",
            beat_id="",
            start_sec=0.0,
            end_sec=5.0,
            modality="ocr",
            pointer="virtual://score",
            verbatim="The same scoreboard frame at halftime reads 32-23.",
            frame_refs=("score.jpg",),
            attestation_model="test-vlm",
            operation_metadata={"measurements": measurements},
        )

    contract = multiround.compile_query_contract(case.question, case.options)
    valid = evidence("halftime")
    invalid = evidence("live_play")
    ready = {"ready_for_answer": True}

    passed = multiround._answer_completion_gate(
        workspace, contract, "A. 32 - 23", (valid.evidence_id,), (), (valid,), completion_status=ready
    )
    blocked = multiround._answer_completion_gate(
        workspace, contract, "A. 32 - 23", (invalid.evidence_id,), (), (invalid,), completion_status=ready
    )

    assert passed["reason"] == "boundary_score_grounded"
    assert blocked["reason"] == "boundary_score_snapshot_missing"


def test_semantic_contract_readiness_blocks_only_contracts_that_need_closure(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    evidence = EvidenceRecord(
        evidence_id="ev_visual",
        beat_id="",
        start_sec=0.0,
        end_sec=2.0,
        modality="visual",
        pointer="virtual://visual",
        verbatim="The jersey visibly reads 11.",
        frame_refs=("jersey.jpg",),
        attestation_model="test-vlm",
    )
    incomplete = {
        "ready_for_answer": False,
        "reason": "critical conditions remain unresolved",
        "unresolved_critical_condition_ids": ["gap_relation_c1"],
    }
    simple = multiround.compile_query_contract(workspace.case.question, workspace.case.options)
    spatial = multiround.compile_query_contract(
        "Which direction is red facing in relation to green?",
        {"A": "Right front", "B": "Left front"},
    )

    simple_gate = multiround._answer_completion_gate(
        workspace,
        simple,
        "B. 11",
        (evidence.evidence_id,),
        (),
        (evidence,),
        completion_status=incomplete,
    )
    spatial_gate = multiround._answer_completion_gate(
        workspace,
        spatial,
        "A. Right front",
        (evidence.evidence_id,),
        (),
        (evidence,),
        query_requirements={"spatial_reference_frame": "object_egocentric"},
        completion_status=incomplete,
    )

    assert simple_gate["passed"] is True
    assert spatial_gate["reason"] == "contract_completion_not_ready"
