from __future__ import annotations

from vcah.evidence_primitives import (
    GapCondition,
    MeasurementFact,
    TargetPresenceFact,
    derive_resolution,
    normalize_condition_results,
    normalize_target_presence,
)


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
