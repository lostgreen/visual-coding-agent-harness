from __future__ import annotations

from visual_coding_agent_harness.legacy.contracts_v2 import TargetRegistry, TargetSpec
from visual_coding_agent_harness.evidence.projection import ProjectionEvidence, project_option_support
from visual_coding_agent_harness.task.spec import TaskSpec, build_task_spec


def _target(target_id: str, text: str, *aliases: str) -> TargetSpec:
    return TargetSpec(target_id=target_id, canonical_text=text, aliases=aliases, source="unit_test")


def test_task_spec_compiles_option_shapes_from_registry_text() -> None:
    registry = TargetRegistry.from_specs(
        targets=[
            _target("T1", "first event"),
            _target("T2", "second event"),
            _target("T3", "third event"),
        ]
    )

    task = build_task_spec(
        task_id="generic_order",
        question="Which order is shown?",
        options=[
            "A. first event then third event then second event",
            "B. first event then second event then third event",
        ],
        route="temporal_order",
        target_registry=registry,
    )

    assert task.options[0].label == "A"
    assert task.options[0].target_sequence == ("T1", "T3", "T2")
    assert task.options[1].target_sequence == ("T1", "T2", "T3")
    assert task.options[1].required_targets == ("T1", "T2", "T3")
    assert task.answer_operator == "ordered_projection"


def test_required_target_set_projects_supported_option_without_guessing_shared_target() -> None:
    task = TaskSpec(
        task_id="required_set",
        question="Which option is true?",
        answer_format="mcq",
        route="needle_local",
        options=(
            ("A", "target one and target two", ("T1", "T2"), (), ()),
            ("B", "target one and target three", ("T1", "T3"), (), ()),
        ),
    ).normalized()

    result = project_option_support(
        task,
        evidence=[
            ProjectionEvidence("E1", "T1", confidence=0.9, segment_id="seg_1"),
            ProjectionEvidence("E2", "T3", confidence=0.9, segment_id="seg_1"),
        ],
    )

    assert result.status == "supported"
    assert result.option_label == "B"
    assert result.strategy == "required_target_set"
    assert result.supporting_evidence_ids == ("E1", "E2")


def test_ordered_sequence_projection_requires_chronological_match() -> None:
    task = TaskSpec(
        task_id="ordered",
        question="Which order is shown?",
        answer_format="mcq",
        route="temporal_order",
        options=(
            ("A", "alpha then beta", (), ("T1", "T2"), ()),
            ("D", "beta then alpha", (), ("T2", "T1"), ()),
        ),
    ).normalized()

    result = project_option_support(
        task,
        evidence=[
            ProjectionEvidence("E1", "T2", timestamp_start=5.0, confidence=0.9, segment_id="seg_1"),
            ProjectionEvidence("E2", "T1", timestamp_start=12.0, confidence=0.9, segment_id="seg_2"),
        ],
    )

    assert result.status == "supported"
    assert result.option_label == "D"
    assert result.strategy == "ordered_sequence"
    assert result.supporting_evidence_ids == ("E1", "E2")


def test_theme_coverage_prefers_broad_multi_segment_support_over_partial_local_support() -> None:
    task = TaskSpec(
        task_id="main_theme",
        question="What is the video mainly about?",
        answer_format="mcq",
        route="main_idea",
        options=(
            ("C", "local subtopic", ("T4",), (), ("T4",)),
            ("D", "full arc", ("T1", "T2", "T3"), (), ("T1", "T2", "T3")),
        ),
    ).normalized()

    result = project_option_support(
        task,
        evidence=[
            ProjectionEvidence("E1", "T1", confidence=0.8, segment_id="seg_1"),
            ProjectionEvidence("E2", "T2", confidence=0.8, segment_id="seg_4"),
            ProjectionEvidence("E3", "T3", confidence=0.8, segment_id="seg_8"),
            ProjectionEvidence("E4", "T4", confidence=0.95, segment_id="seg_4"),
        ],
    )

    assert result.status == "supported"
    assert result.option_label == "D"
    assert result.strategy == "theme_coverage"
    assert result.supporting_evidence_ids == ("E1", "E2", "E3")


def test_theme_coverage_reports_ambiguous_when_multiple_broad_options_complete() -> None:
    task = TaskSpec(
        task_id="main_theme_ambiguous",
        question="What is the video mainly about?",
        answer_format="mcq",
        route="main_idea",
        options=(
            ("A", "first broad theme", ("T1", "T2"), (), ("T1", "T2")),
            ("B", "second broad theme", ("T3", "T4"), (), ("T3", "T4")),
        ),
    ).normalized()

    result = project_option_support(
        task,
        evidence=[
            ProjectionEvidence("E1", "T1", confidence=0.9, segment_id="seg_1"),
            ProjectionEvidence("E2", "T2", confidence=0.9, segment_id="seg_2"),
            ProjectionEvidence("E3", "T3", confidence=0.9, segment_id="seg_3"),
            ProjectionEvidence("E4", "T4", confidence=0.9, segment_id="seg_3"),
        ],
    )

    assert result.status == "ambiguous"
    assert result.option_label is None
    assert result.candidate_option_labels == ("A", "B")


def test_projection_reports_ambiguous_when_options_have_equal_complete_support() -> None:
    task = TaskSpec(
        task_id="ambiguous",
        question="Which option is true?",
        answer_format="mcq",
        route="needle_local",
        options=(
            ("A", "shared target", ("T1",), (), ()),
            ("B", "same shared target", ("T1",), (), ()),
        ),
    ).normalized()

    result = project_option_support(task, evidence=[ProjectionEvidence("E1", "T1", confidence=0.9)])

    assert result.status == "ambiguous"
    assert result.option_label is None
    assert result.candidate_option_labels == ("A", "B")


def test_select_absent_projects_only_option_without_positive_support() -> None:
    task = TaskSpec(
        task_id="absent",
        question="Which object is NOT shown?",
        answer_format="mcq",
        route="needle_local",
        answer_operator="select_absent",
        options=(
            ("A", "alpha", ("T1",), (), ()),
            ("B", "beta", ("T2",), (), ()),
            ("C", "gamma", ("T3",), (), ()),
            ("D", "delta", ("T4",), (), ()),
        ),
    ).normalized()

    result = project_option_support(
        task,
        evidence=[
            ProjectionEvidence("E1", "T1", confidence=0.9),
            ProjectionEvidence("E2", "T2", confidence=0.9),
            ProjectionEvidence("E3", "T3", confidence=0.9),
        ],
    )

    assert result.status == "supported"
    assert result.option_label == "D"
    assert result.strategy == "select_absent"
    assert result.reason == "complement_resolved"


def test_select_absent_rejects_multiple_unconfirmed_options() -> None:
    task = TaskSpec(
        task_id="absent_ambiguous",
        question="Which object is not seen?",
        answer_format="mcq",
        route="needle_local",
        answer_operator="select_absent",
        options=(
            ("A", "alpha", ("T1",), (), ()),
            ("B", "beta", ("T2",), (), ()),
            ("C", "gamma", ("T3",), (), ()),
        ),
    ).normalized()

    result = project_option_support(task, evidence=[ProjectionEvidence("E1", "T1")])

    assert result.status == "ambiguous"
    assert result.option_label is None
    assert result.candidate_option_labels == ("B", "C")
    assert result.reason == "multiple_absent_candidates"


def test_causal_bind_requires_binding_source_not_topic_overlap() -> None:
    task = TaskSpec(
        task_id="causal",
        question="Why did the narrator recommend stopping?",
        answer_format="mcq",
        route="mixed_asr_visual",
        answer_operator="causal_bind",
        options=(
            ("A", "because of rain", ("T1",), (), ()),
            ("B", "because of traffic", ("T2",), (), ()),
        ),
    ).normalized()

    overlap = project_option_support(
        task,
        evidence=[
            ProjectionEvidence("E1", "T1", source="search_segments", modality="asr"),
            ProjectionEvidence("E2", "T2", source="search_segments", modality="asr"),
        ],
    )
    bound = project_option_support(
        task,
        evidence=[
            ProjectionEvidence("E1", "T1", source="search_segments", modality="asr"),
            ProjectionEvidence("E2", "T2", source="bind_asr_claim", modality="asr"),
        ],
    )

    assert overlap.status == "unsupported"
    assert overlap.reason == "topic_overlap_only"
    assert bound.status == "supported"
    assert bound.option_label == "B"
    assert bound.reason == "causal_binding_supported"


def test_universal_intersection_requires_cross_group_support() -> None:
    task = TaskSpec(
        task_id="universal",
        question="Which object appears in every case?",
        answer_format="mcq",
        route="needle_local",
        answer_operator="universal_intersection",
        options=(
            ("A", "shared object", ("T1",), (), ()),
            ("B", "local object", ("T2",), (), ()),
        ),
    ).normalized()

    result = project_option_support(
        task,
        evidence=[
            ProjectionEvidence("E1", "T1", segment_id="case_1"),
            ProjectionEvidence("E2", "T1", segment_id="case_2"),
            ProjectionEvidence("E3", "T2", segment_id="case_1"),
        ],
    )

    assert result.status == "supported"
    assert result.option_label == "A"
    assert result.strategy == "universal_intersection"
    assert result.reason == "universal_complete"


def test_ordered_projection_uses_observation_order_when_timestamps_are_missing() -> None:
    task = TaskSpec(
        task_id="ordered_without_timestamps",
        question="Which order happens successively?",
        answer_format="mcq",
        route="temporal_order",
        answer_operator="ordered_projection",
        options=(
            ("A", "T1 then T2 then T3", (), ("T1", "T2", "T3"), ()),
            ("D", "T1 then T3 then T2", (), ("T1", "T3", "T2"), ()),
        ),
    ).normalized()

    result = project_option_support(
        task,
        evidence=[
            ProjectionEvidence("E1", "T1", segment_id="seg_1"),
            ProjectionEvidence("E2", "T3", segment_id="seg_2"),
            ProjectionEvidence("E3", "T2", segment_id="seg_3"),
        ],
    )

    assert result.status == "supported"
    assert result.option_label == "D"
    assert result.strategy == "ordered_projection"
    assert result.reason == "ordered_projection_supported"
