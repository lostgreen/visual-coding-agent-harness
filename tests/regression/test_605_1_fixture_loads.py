from __future__ import annotations

import json
from pathlib import Path


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "case_605_1" / "fixture.json"


def test_case_605_1_fixture_records_low_confidence_or_wrong_prior_choice() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    prior = payload["prior_result"]

    assert payload["case_id"] == "605-1"
    assert payload["ground_truth"] == "D"
    assert prior["selected_option"] != payload["ground_truth"] or prior["confidence"] < 0.5


def test_case_605_1_fixture_contains_lifecycle_asr_and_growth_events() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    asr_text = " ".join(
        sentence["text"]
        for segment in payload["segments"]
        for sentence in segment.get("asr_sentences", [])
    ).lower()
    event_types = {event["type"] for event in payload["expected_trace_events"]}

    assert "rise" in asr_text
    assert "fall" in asr_text
    assert "evidence_growth" in event_types
