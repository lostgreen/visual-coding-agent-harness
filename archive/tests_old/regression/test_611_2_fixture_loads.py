from __future__ import annotations

import json
from pathlib import Path


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "case_611_2" / "fixture.json"
EXPECTED_ORDER = (
    "Aeneas",
    "David",
    "The rape of Persephone",
    "Apollo and Daphne",
)


def test_case_611_2_fixture_records_artwork_permutation_regression() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    options = payload["options"]
    option_sequences = [tuple(option["ordered_entities"]) for option in options]

    assert payload["case_id"] == "611-2"
    assert payload["ground_truth"] == "D"
    assert payload["prior_result"]["selected_option"] == "A"
    assert all(set(sequence) == set(EXPECTED_ORDER) for sequence in option_sequences)
    assert len(set(option_sequences)) == len(option_sequences)


def test_case_611_2_fixture_contains_segment_two_ordered_asr_sequence() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    seg_0002 = next(segment for segment in payload["segments"] if segment["segment_id"] == "seg_0002")
    observed = tuple(seg_0002["ordered_asr_sequence"])

    assert observed == EXPECTED_ORDER
    assert EXPECTED_ORDER == tuple(
        item.strip().rstrip(".")
        for item in seg_0002["asr_sentences"][0]["text"].split(":", 1)[1].split("->")
        if item.strip()
    )
