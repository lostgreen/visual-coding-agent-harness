import tempfile
from pathlib import Path

from visual_coding_agent_harness.contracts import ClaimModality, ClaimRelation, OptionSpec, TargetRegistry, TargetSpec
from visual_coding_agent_harness.agents.grounding.compiler import compile_fallback_plan, compile_grounding_plan
from visual_coding_agent_harness.interpreter import ProgramInterpreter
from visual_coding_agent_harness.registry import ToolRegistry, tool
from visual_coding_agent_harness.tools.navigation import build_video_navigation_registry
from visual_coding_agent_harness.video_map import VideoMap, VideoMapSegment
from visual_coding_agent_harness.workspace import EvidenceWorkspace


def _sequence_map(text: str = "The narration lists alpha event, then beta event, then gamma event.") -> VideoMap:
    return VideoMap(
        video_path="/videos/generic.mp4",
        duration_sec=90.0,
        segments=[
            VideoMapSegment(
                segment_id="seg_0001",
                start_sec=10.0,
                end_sec=40.0,
                asr_text=text,
            )
        ],
    )


def _sequence_registry() -> TargetRegistry:
    return TargetRegistry.from_specs(
        targets=[
            TargetSpec("T1", "alpha event", modality_hint=ClaimModality.NARRATED_FACT),
            TargetSpec("T2", "beta event", modality_hint=ClaimModality.NARRATED_FACT),
            TargetSpec("T3", "gamma event", modality_hint=ClaimModality.NARRATED_FACT),
        ],
        options=[
            OptionSpec(
                "B",
                target_sequence=("T1", "T2", "T3"),
                required_relations=("R1", "R2"),
                option_kind="sequence",
            )
        ],
        relations=[
            ClaimRelation("R1", "before", "T1", "T2"),
            ClaimRelation("R2", "before", "T2", "T3"),
        ],
    )


def test_read_segment_detail_promotes_registry_targets_without_explicit_target_refs():
    with tempfile.TemporaryDirectory() as tmp:
        workspace = EvidenceWorkspace.create(Path(tmp), run_id="promote_registry_targets")
        workspace.target_registry = _sequence_registry()
        registry = build_video_navigation_registry(_sequence_map(), workspace=workspace)

        detail = registry.execute(
            "read_segment_detail",
            {"segment_id": "seg_0001", "promote_answer_evidence": True},
        )

    assert {binding["target_id"]: binding["status"] for binding in detail["evidence_bindings"]} == {
        "T1": "supported",
        "T2": "supported",
        "T3": "supported",
    }
    assert {binding["relation_id"]: binding["status"] for binding in detail["relation_bindings"]} == {
        "R1": "supported",
        "R2": "supported",
    }
    assert detail["answer_evidence_rows"]


def test_ordered_full_sequence_produces_supported_relation_binding():
    with tempfile.TemporaryDirectory() as tmp:
        workspace = EvidenceWorkspace.create(Path(tmp), run_id="ordered_relation_binding")
        workspace.target_registry = _sequence_registry()
        registry = build_video_navigation_registry(_sequence_map(), workspace=workspace)

        detail = registry.execute(
            "read_segment_detail",
            {"segment_id": "seg_0001", "promote_answer_evidence": True},
        )

    ordered_relations = [
        relation
        for relation in detail["relation_bindings"]
        if relation.get("ordered_target_refs") == ["T1", "T2", "T3"]
    ]
    assert {relation["relation_id"] for relation in ordered_relations} == {"R1", "R2"}
    assert all(relation["status"] == "supported" for relation in ordered_relations)


def test_fallback_ordered_registry_promotes_asr_enumeration_to_supported_option():
    question = (
        "In what order does the narration list the artworks?\n"
        'A. "Aeneas", "David", "Persephone", "Apollo"\n'
        'B. "David", "Aeneas", "Persephone", "Apollo"\n'
        'C. "Aeneas", "Persephone", "David", "Apollo"\n'
        'D. "Aeneas", "David", "Apollo", "Persephone"'
    )
    options = (
        'A. "Aeneas", "David", "Persephone", "Apollo"',
        'B. "David", "Aeneas", "Persephone", "Apollo"',
        'C. "Aeneas", "Persephone", "David", "Apollo"',
        'D. "Aeneas", "David", "Apollo", "Persephone"',
    )
    plan = compile_fallback_plan(question, options, "temporal_order")
    compiled = compile_grounding_plan(
        plan,
        raw_options={
            "A": '"Aeneas", "David", "Persephone", "Apollo"',
            "B": '"David", "Aeneas", "Persephone", "Apollo"',
            "C": '"Aeneas", "Persephone", "David", "Apollo"',
            "D": '"Aeneas", "David", "Apollo", "Persephone"',
        },
        skill_ids=("visual_timeline_qa", "narration_timeline_qa", "main_idea", "grounded_factual_qa"),
    )

    with tempfile.TemporaryDirectory() as tmp:
        workspace = EvidenceWorkspace.create(Path(tmp), run_id="fallback_ordered_asr_option")
        workspace.target_registry = compiled.registry
        registry = build_video_navigation_registry(
            _sequence_map('The narration lists "Aeneas", "David", "Persephone", and "Apollo" in order.'),
            workspace=workspace,
        )

        result = ProgramInterpreter(registry=registry, workspace=workspace).run(
            [
                {
                    "tool": "read_segment_detail",
                    "args": {"segment_id": "seg_0001", "promote_answer_evidence": True},
                }
            ]
        )
        detail = workspace.get_observation(result.observation_ids[0]).raw_output
        table = workspace.read_evidence_table_v3(question=question, options=options)

    ordered_rows = [
        row
        for row in detail["answer_evidence_rows"]
        if row.get("tool") == "ordered_transcript_sequence"
    ]
    assert ordered_rows
    assert ordered_rows[0]["supported_option"] == "A"
    assert ordered_rows[0]["ordered_target_refs"] == ["T1", "T2", "T3", "T4"]
    assert table["groups"]["A"]


def test_promote_answer_evidence_does_not_bind_raw_option_text_as_target():
    with tempfile.TemporaryDirectory() as tmp:
        workspace = EvidenceWorkspace.create(Path(tmp), run_id="no_raw_option_target_binding")
        workspace.target_registry = TargetRegistry.from_specs(
            targets=[TargetSpec("T1", "canonical target", modality_hint=ClaimModality.NARRATED_FACT)],
            options=[OptionSpec("B", target_sequence=("T1",), raw_option_text="raw option text only")],
        )
        registry = build_video_navigation_registry(
            _sequence_map("The narrator says raw option text only."),
            workspace=workspace,
        )

        detail = registry.execute(
            "read_segment_detail",
            {
                "segment_id": "seg_0001",
                "promote_answer_evidence": True,
                "option_targets": {"B": ["raw option text only"]},
            },
        )

    assert all(binding["target_id"] == "T1" for binding in detail["evidence_bindings"])
    assert all(row.get("event_label") != "raw option text only" for row in detail["answer_evidence_rows"])
    assert not any(
        row.get("evidence_binding", {}).get("target_id") == "raw option text only"
        for row in detail["answer_evidence_rows"]
    )


def test_target_level_asr_rows_project_to_unique_option_coverage():
    with tempfile.TemporaryDirectory() as tmp:
        workspace = EvidenceWorkspace.create(Path(tmp), run_id="target_option_projection")
        workspace.target_registry = TargetRegistry.from_specs(
            targets=[
                TargetSpec("T1", "shared setup", modality_hint=ClaimModality.NARRATED_FACT),
                TargetSpec("T2", "distinct answer target", modality_hint=ClaimModality.NARRATED_FACT),
            ],
            options=[
                OptionSpec("A", target_sequence=("T1",)),
                OptionSpec("B", target_sequence=("T2",)),
            ],
        )
        workspace.write_evidence_row(
            {
                "tool": "transcript_evidence_binder",
                "event_label": "T2",
                "target_id": "T2",
                "claim": "Indexed ASR supports the distinct answer target.",
                "confidence": 0.88,
                "grounding_quality": "indexed_transcript",
                "confidence_signal": "asr_claim_binding_supported",
                "evidence_binding": {"target_id": "T2", "status": "supported"},
            }
        )

        table = workspace.read_evidence_table_v3(
            question="Which answer is supported?",
            options=["A. shared setup", "B. distinct answer target"],
        )
        summary = workspace.evidence_status_summary(
            question="Which answer is supported?",
            options=["A. shared setup", "B. distinct answer target"],
        )

    assert [row["event_label"] for row in table["groups"]["B"]] == ["T2"]
    assert table["groups"]["unassigned"] == []
    assert summary["option_coverage"] == "1/2"
    assert summary["option_status"]["B"]["strong_evidence_count"] == 1


def test_ordered_target_rows_project_to_matching_option_sequence():
    with tempfile.TemporaryDirectory() as tmp:
        workspace = EvidenceWorkspace.create(Path(tmp), run_id="ordered_option_projection")
        workspace.target_registry = TargetRegistry.from_specs(
            targets=[
                TargetSpec("T1", "first event", modality_hint=ClaimModality.NARRATED_FACT),
                TargetSpec("T2", "second event", modality_hint=ClaimModality.NARRATED_FACT),
            ],
            options=[
                OptionSpec("A", target_sequence=("T1", "T2")),
                OptionSpec("B", target_sequence=("T2", "T1")),
            ],
        )
        workspace.write_evidence_row(
            {
                "tool": "ordered_transcript_sequence",
                "event_label": "ordered_transcript_sequence",
                "claim": "Indexed ASR supports T2 before T1.",
                "confidence": 0.94,
                "grounding_quality": "indexed_transcript",
                "ordered_target_refs": ["T2", "T1"],
                "evidence_binding": {
                    "status": "supported",
                    "target_id": "ordered_sequence",
                    "ordered_target_refs": ["T2", "T1"],
                },
            }
        )

        table = workspace.read_evidence_table_v3(
            question="Which order is supported?",
            options=["A. first then second", "B. second then first"],
        )

    assert [row["event_label"] for row in table["groups"]["B"]] == ["ordered_transcript_sequence"]
    assert table["groups"]["A"] == []


def test_post_observation_hook_grows_answer_evidence_after_one_detail_observation(monkeypatch):
    monkeypatch.setenv("HARNESS_LEGACY_BINDER_TELEMETRY", "1")
    with tempfile.TemporaryDirectory() as tmp:
        workspace = EvidenceWorkspace.create(Path(tmp), run_id="post_observation_growth")
        workspace.target_registry = _sequence_registry()
        registry = ToolRegistry()

        @tool(name="read_segment_detail", description="Scripted detail packet.")
        def read_segment_detail(segment_id: str, promote_answer_evidence: bool = False):
            return {
                "claim": f"detail {segment_id}",
                "confidence": 1.0,
                "segment_id": segment_id,
                "start_sec": 10.0,
                "end_sec": 40.0,
                "raw_asr_excerpt": "The narration lists alpha event, then beta event, then gamma event.",
                "evidence_bindings": [],
                "relation_bindings": [],
                "answer_evidence_rows": [],
                "promote_answer_evidence": promote_answer_evidence,
            }

        registry.register(read_segment_detail)

        ProgramInterpreter(registry, workspace).run(
            [
                {
                    "tool": "read_segment_detail",
                    "args": {"segment_id": "seg_0001", "promote_answer_evidence": True},
                }
            ]
        )

        observation = workspace.read_observations(tool_name="read_segment_detail")[0]
        table = workspace.read_evidence_table_v3(question="generic question", options=["B. generic sequence"])
        row_count = workspace.evidence_table_row_count()

    assert observation.raw_output["evidence_bindings"]
    assert observation.raw_output["relation_bindings"]
    assert row_count > 0
    assert any(row["tool"] == "transcript_evidence_binder" for row in table["rows"])
