from __future__ import annotations

from vcah.evidence_primitives import (
    ConditionResult,
    ConditionState,
    GapCondition,
    MeasurementFact,
    RelationFact,
    TargetPresenceFact,
    derive_resolution,
    extract_measurements_from_text,
    merge_condition_states,
    normalize_condition_results,
    normalize_target_presence,
)


def test_condition_state_merge_is_monotonic_and_surfaces_conflict() -> None:
    satisfied = ConditionResult("gap_c1", "satisfied", "Clock reads 8:13.", ("ev_yes",))
    unknown = ConditionResult("gap_c1", "unknown", "An unrelated frame was unreadable.")
    contradicted = ConditionResult("gap_c1", "contradicted", "Clock does not read 8:13.", ("ev_no",))
    stable = merge_condition_states((("q1", satisfied), ("q2", unknown)))
    conflicted = merge_condition_states((("q1", satisfied), ("q3", contradicted)))
    assert stable["gap_c1"] == ConditionState(
        condition_id="gap_c1", status="satisfied", supporting_evidence_ids=("ev_yes",), updated_by_task_id="q1"
    )
    assert conflicted["gap_c1"].status == "conflicted"
    assert conflicted["gap_c1"].supporting_evidence_ids == ("ev_yes",)
    assert conflicted["gap_c1"].refuting_evidence_ids == ("ev_no",)


def test_global_enumeration_condition_infers_scope_and_preserves_it_in_results() -> None:
    condition = GapCondition("gap_news_c1", "List each news segment appearance throughout the video.")
    result = normalize_condition_results(
        (
            {
                "condition_id": condition.condition_id,
                "status": "satisfied",
                "observation": "One news segment is visible in this window.",
            },
        ),
        (condition,),
        evidence_id="ev_news",
    )[0]

    assert condition.scope == "full_video"
    assert condition.quantifier == "all_events"
    assert condition.required_coverage == 1.0
    assert result.scope == "full_video"
    assert result.quantifier == "all_events"


def test_success_conditions_are_routed_to_their_own_evaluators() -> None:
    conditions = (
        GapCondition("c_observe", "Joe says that he will call the police."),
        GapCondition("c_aggregate", "Count every qualified overtake event."),
        GapCondition("c_derive", "The final decision differs from the original plan."),
        GapCondition("c_interpret", "Which monologue best fits the transition?"),
        GapCondition("c_counterfactual", "If Joe acted otherwise, what would happen instead?"),
    )

    assert tuple(condition.evaluation_type for condition in conditions) == (
        "observable",
        "aggregatable",
        "derivable",
        "interpretive",
        "counterfactual",
    )

    result = normalize_condition_results(
        ({"condition_id": "c_derive", "status": "satisfied", "observation": "Model inference."},),
        (conditions[2],),
        evidence_id="ev_inference",
    )
    assert result[0].status == "unknown"
    assert result[0].evidence_ids == ()


def test_read_condition_requires_visible_target_and_measurement() -> None:
    condition = GapCondition("gap_clock_c1", "read the displayed clock and score")
    raw = (
        {
            "condition_id": condition.condition_id,
            "status": "satisfied",
            "observation": "The crop reads 8:13 and 10-10.",
        },
    )

    without_target = normalize_condition_results(
        raw,
        (condition,),
        evidence_id="ev_clock",
        measurements=(MeasurementFact(10, "point"),),
    )
    absent_target = normalize_condition_results(
        raw,
        (condition,),
        evidence_id="ev_clock",
        target_presence=TargetPresenceFact("scoreboard", "absent", 0.9),
        measurements=(MeasurementFact(10, "point"),),
    )
    wrong_target = normalize_condition_results(
        raw,
        (condition,),
        evidence_id="ev_clock",
        target_presence=TargetPresenceFact("basket", "present", 0.9),
        measurements=(MeasurementFact(10, "point"),),
    )
    grounded = normalize_condition_results(
        raw,
        (condition,),
        evidence_id="ev_clock",
        target_presence=TargetPresenceFact("scoreboard", "present", 0.9),
        measurements=(MeasurementFact(10, "point"),),
    )

    assert without_target[0].status == "unknown"
    assert absent_target[0].status == "unknown"
    assert wrong_target[0].status == "unknown"
    assert grounded[0].status == "satisfied"
    assert grounded[0].evidence_ids == ("ev_clock",)


def test_resolution_is_derived_from_stable_condition_ids() -> None:
    conditions = (
        GapCondition("gap_transition_c1", "observe the opening state"),
        GapCondition("gap_transition_c2", "observe the closing state"),
    )
    results = normalize_condition_results(
        (
            {
                "condition_id": "gap_transition_c1",
                "status": "satisfied",
                "observation": "The scoreboard is tied 10-10.",
            },
            {
                "condition_id": "gap_transition_c2",
                "status": "unknown",
                "observation": "The later score is unreadable.",
            },
        ),
        conditions,
        evidence_id="ev_transition",
    )

    assert derive_resolution(conditions, results) == "partial"


def test_negative_observations_keep_evidence_lineage() -> None:
    target = normalize_target_presence(
        {"target": "scoreboard", "status": "absent", "confidence": 0.9},
        evidence_id="ev_absent",
    )
    result = normalize_condition_results(
        (
            {
                "condition_id": "gap_target_c1",
                "status": "contradicted",
                "observation": "No scoreboard is visible in any supplied frame.",
            },
        ),
        (GapCondition("gap_target_c1", "locate the scoreboard"),),
        evidence_id="ev_absent",
        target_presence=target,
    )[0]

    assert target.evidence_ids == ("ev_absent",)
    assert result.evidence_ids == ("ev_absent",)


def test_scaled_measurement_units_are_normalized() -> None:
    measurement = MeasurementFact(23, "trillion light years", raw_text="23 trillion light-years")

    assert measurement.value == 23_000_000_000_000
    assert measurement.unit == "light_year"


def test_explicit_measurement_fallback_normalizes_large_scale_and_clock() -> None:
    measurements = extract_measurements_from_text(
        "The stated diameter is 30 quintillion light-years. The game clock reads 8:13.",
        quantity_type="diameter",
        binding_status="explicit",
    )
    assert measurements[0].value == 30_000_000_000_000_000_000
    assert measurements[0].unit == "light_year"
    assert measurements[0].quantity_type == "diameter"
    assert measurements[0].binding_status == "explicit"
    assert measurements[0].extraction_source == "text_fallback"
    assert measurements[1].value == 493
    assert measurements[1].unit == "second"
    assert measurements[1].quantity_type == "countdown_clock"


def test_measurement_fallback_does_not_bind_unitless_count_as_diameter() -> None:
    assert extract_measurements_from_text(
        "There are 30 quintillion of them.", quantity_type="diameter", binding_status="contextual"
    ) == ()


def test_measurement_boundary_can_satisfy_temporal_condition_without_duplicate_relation() -> None:
    condition = GapCondition("gap_calories_c1", "read the calorie value before the meeting boundary")
    result = normalize_condition_results(
        (
            {
                "condition_id": condition.condition_id,
                "status": "satisfied",
                "observation": "The displayed item adds 300 calories before the meeting.",
            },
        ),
        (condition,),
        evidence_id="ev_calories",
        target_presence=TargetPresenceFact("calorie display", "present", 0.9),
        measurements=(
            MeasurementFact(300, "calorie", semantics="delta", boundary_relation="before"),
        ),
    )[0]

    assert result.status == "satisfied"


def test_typed_relation_condition_requires_exact_relation_and_participants() -> None:
    condition = GapCondition(
        "gap_identity_c1", "confirm the injured man is the envelope man", condition_type="relation",
        relation_type="same_entity", subject_role="injured_man", object_role="envelope_man",
    )
    raw = ({"condition_id": condition.condition_id, "status": "satisfied", "observation": "Compared both people."},)
    wrong_relation = normalize_condition_results(
        raw, (condition,), evidence_id="ev_wrong",
        relations=(RelationFact("causes", "injured_man", "envelope_man", "supported"),),
    )
    wrong_subject = normalize_condition_results(
        raw, (condition,), evidence_id="ev_subject",
        relations=(RelationFact("same_entity", "bystander", "envelope_man", "supported"),),
    )
    matching = normalize_condition_results(
        raw, (condition,), evidence_id="ev_match",
        relations=(RelationFact("same_entity", "injured_man", "envelope_man", "supported"),),
    )
    assert wrong_relation[0].status == "unknown"
    assert wrong_subject[0].status == "unknown"
    assert matching[0].status == "satisfied"
