from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from PIL import Image

import vcah.multiround as multiround
from vcah.multiround import InvestigationTask, ReasonerDecision, VirtualVideoMultiRoundDriver
from vcah.investigator import VirtualVideoInvestigator
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

    assert result.answer == "Insufficient verified evidence."
    gate = next(row for row in result.trace if row.get("type") == "completion_gate")
    assert gate["passed"] is False
    assert gate["reason"] == "entity_reconciliation_missing"


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
    gate = next(row for row in result.trace if row.get("type") == "completion_gate")
    assert gate["reason"] == "answer_missing"


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
