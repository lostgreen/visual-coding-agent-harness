from vcah.qualification import (
    QualificationRequirement,
    RequirementStatus,
    apply_observation_to_requirements,
    evaluate_requirement_graph,
    parse_option_predicates,
    qualification_status,
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
    with_lineage = apply_observation_to_requirements(
        {
            "target_requirement_ids": ["req_boundary"],
            "requirement_results": {"req_boundary": "supported"},
            "evidence_ids": ["ev_1"],
        },
        (requirement,),
    )

    assert without_lineage == ()
    assert with_lineage[0].status is RequirementStatus.SUPPORTED
