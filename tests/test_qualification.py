from vcah.qualification import (
    QualificationRequirement,
    RequirementStatus,
    apply_observation_to_requirements,
    evaluate_requirement_graph,
    parse_option_predicates,
    qualification_status,
    qualify_event_candidates,
)


def test_unknown_custom_predicate_fails_closed() -> None:
    requirement = QualificationRequirement(
        "req_custom",
        "custom",
        {"definition": "joe concealed the event"},
    )

    evaluations = evaluate_requirement_graph((requirement,), {})

    assert evaluations[0].status is RequirementStatus.UNKNOWN
    assert qualification_status((requirement,), evaluations) == "incomplete"


def test_requirement_dependencies_block_downstream_evaluation() -> None:
    requirements = (
        QualificationRequirement("req_episode", "temporal_max", {"status": "unknown"}),
        QualificationRequirement(
            "req_participant",
            "ordinal_member",
            {"status": "supported"},
            dependency_ids=("req_episode",),
        ),
        QualificationRequirement(
            "req_attribute",
            "attribute_match",
            {"status": "supported"},
            dependency_ids=("req_participant",),
        ),
    )

    evaluations = evaluate_requirement_graph(requirements, {})

    assert [evaluation.status for evaluation in evaluations] == [
        RequirementStatus.UNKNOWN,
        RequirementStatus.BLOCKED,
        RequirementStatus.BLOCKED,
    ]
    assert evaluations[1].blocked_by == ("req_episode",)
    assert evaluations[2].blocked_by == ("req_participant",)


def test_contradicted_candidate_short_circuits_descendants_as_not_applicable() -> None:
    candidates = tuple(
        {
            "candidate_id": f"event_candidate_{index:03d}",
            "candidate_status": "observed_candidate",
            "focal_position_transition": True,
            "evidence_ids": [f"ev_{index:03d}"],
            "participant_ids": ["camera_holder", f"racer_{index:03d}"],
            "preconditions_met_observations": [False],
            "transitions": ["The racer overtakes the recorder."],
            "continues_from_previous": False,
            "continues_to_next": False,
            "provenance": [{
                "kind": "direct_witness",
                "source_evidence_ids": [f"ev_{index:03d}"],
                "witness_ranges": [[float(index), float(index) + 0.5]],
                "producer": "observation",
            }],
            "distinctness_provenance": [{
                "kind": "deterministic_derivation",
                "source_fact_ids": [f"event_candidate_{index:03d}"],
                "source_evidence_ids": [f"ev_{index:03d}"],
                "derivation": "canonical_event_reconciliation",
                "producer": "program",
                "source_kinds": ["direct_witness"],
            }],
            "qualification_provenance": [{
                "kind": "direct_witness",
                "source_evidence_ids": [f"ev_{index:03d}"],
                "witness_ranges": [[float(index), float(index) + 0.5]],
                "producer": "observation",
            }],
        }
        for index in range(14)
    )

    result = qualify_event_candidates(candidates, require_state_precondition=True)
    graph = result["requirement_graph"]

    assert len(result["unqualified_events"]) == 14
    assert graph["not_applicable"] == 42
    assert graph["blocked_unresolved"] == 0
    assert graph["unresolved_dependency_ids"] == []
    assert all(
        "terminal dependency" in evaluation["reason"]
        for event in result["unqualified_events"]
        for evaluation in event["requirement_evaluations"][3:]
    )


def test_model_declared_qualification_without_admissible_provenance_fails_closed() -> None:
    result = qualify_event_candidates(({
        "candidate_id": "event_candidate_001",
        "candidate_status": "observed_candidate",
        "qualified": True,
        "focal_position_transition": True,
        "evidence_ids": ["ev_model_only"],
        "participant_ids": ["camera_holder", "racer_1"],
        "states_before": ["ranked 1"],
        "transitions": ["The racer overtakes the recorder."],
        "qualification_observations": {
            "required_prior_state": "supported",
            "transition": "supported",
            "same_subject": "supported",
            "episode_boundary": "supported",
        },
    },), require_state_precondition=True)

    assert result["qualified_events"] == []
    assert len(result["incomplete_events"]) == 1
    assert result["requirement_graph"]["unknown"] == 1


def test_option_parser_keeps_helmet_clothing_and_brand_attributes_distinct() -> None:
    predicates = parse_option_predicates({
        "C": "The one wearing a blue helmet",
        "F": "The one wearing blue clothes",
        "G": "The one wearing a Red Bull helmet",
        "H": "The one wearing a red helmet",
    })

    assert predicates["C"][0].attribute == "helmet_color"
    assert predicates["F"][0].attribute == "clothing_color"
    assert predicates["G"][0].attribute == "helmet_brand"
    assert predicates["G"][0].value == "red_bull"
    assert predicates["H"][0].attribute == "helmet_color"
    assert predicates["H"][0].value == "red"


def test_repair_observation_without_lineage_cannot_close_requirement() -> None:
    requirement = QualificationRequirement("req_boundary", "temporal_max", {"status": "unknown"})

    without_lineage = apply_observation_to_requirements(
        {"requirement_results": {"req_boundary": "supported"}, "evidence_ids": ["ev_1"]},
        (requirement,),
    )
    with_lineage_only = apply_observation_to_requirements(
        {
            "target_requirement_ids": ["req_boundary"],
            "requirement_results": {"req_boundary": "supported"},
            "evidence_ids": ["ev_1"],
        },
        (requirement,),
    )
    with_lineage_and_provenance = apply_observation_to_requirements(
        {
            "target_requirement_ids": ["req_boundary"],
            "requirement_results": {"req_boundary": "supported"},
            "evidence_ids": ["ev_1"],
            "provenance": [{
                "kind": "direct_witness",
                "source_evidence_ids": ["ev_1"],
                "witness_ranges": [[1.0, 2.0]],
                "producer": "observation",
            }],
        },
        (requirement,),
    )

    assert without_lineage == ()
    assert with_lineage_only[0].status is RequirementStatus.UNKNOWN
    assert with_lineage_and_provenance[0].status is RequirementStatus.SUPPORTED
