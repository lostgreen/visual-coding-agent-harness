from visual_coding_agent_harness.agents.followup import normalize_missing_evidence


def test_normalize_with_explicit_query():
    target = normalize_missing_evidence(
        {
            "id": "missing_001",
            "query": "red car entering the driveway",
            "event_label": "arrival",
            "reason": "need visual confirmation",
            "priority": 2,
        },
        route="needle_local",
        run_id="runA",
        seq=7,
    )

    assert target.target_id == "fu_runA_0007"
    assert target.query == "red car entering the driveway"
    assert target.event_label == "arrival"
    assert target.route == "needle_local"
    assert target.reason == "need visual confirmation"
    assert target.priority == 2
    assert target.parent_missing_evidence_id == "missing_001"


def test_normalize_with_only_description():
    target = normalize_missing_evidence(
        {
            "id": "missing_002",
            "missing_description": "  identify the item placed on the counter  ",
        },
        route="gist_global",
        run_id="runB",
        seq=1,
    )

    assert target.query == "identify the item placed on the counter"
    assert target.reason == "unspecified"
    assert target.priority == 1


def test_attempt_count_starts_zero():
    target = normalize_missing_evidence(
        {"id": "missing_003", "query": "person leaving through side door"},
        route="temporal_order",
        run_id="runC",
        seq=42,
    )

    assert target.attempt_count == 0
