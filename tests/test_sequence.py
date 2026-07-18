from __future__ import annotations

from vcah.sequence import build_sequence_ledger


def test_sequence_ledger_evaluates_all_options_against_one_ordered_fact_set() -> None:
    snapshot = {
        "qualified_events": [
            {
                "candidate_id": "event_1",
                "virtual_time_range": [1.0, 2.0],
                "event_keys": ["door opens"],
                "descriptions": ["The door opens."],
                "evidence_ids": ["ev_1"],
            },
            {
                "candidate_id": "event_2",
                "virtual_time_range": [4.0, 5.0],
                "event_keys": ["joe leaves"],
                "descriptions": ["Joe leaves the room."],
                "evidence_ids": ["ev_2"],
            },
        ],
    }
    ledger = build_sequence_ledger(
        snapshot,
        {
            "A": "door opens -> Joe leaves",
            "B": "Joe leaves -> door opens",
            "C": "door opens -> siren sounds",
        },
    )

    assert ledger["option_sequence_verdicts"]["A"]["status"] == "supported"
    assert ledger["option_sequence_verdicts"]["B"]["status"] == "contradicted"
    assert ledger["option_sequence_verdicts"]["C"]["status"] == "unknown"
    assert ledger["no_option_exact_match"] is False


def test_sequence_ledger_preserves_no_exact_option_match() -> None:
    ledger = build_sequence_ledger(
        {
            "qualified_events": [{
                "candidate_id": "event_1",
                "virtual_time_range": [1.0, 2.0],
                "event_keys": ["door opens"],
                "descriptions": ["The door opens."],
            }],
        },
        {"A": "door opens -> Joe leaves", "B": "Joe leaves -> door opens"},
    )

    assert ledger["no_option_exact_match"] is True
    assert all(row["status"] == "unknown" for row in ledger["option_sequence_verdicts"].values())
