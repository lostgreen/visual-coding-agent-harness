from __future__ import annotations

import pytest

from vcah.types import Claim, ClaimVerdict, InvestigatorOutputInvalid, validate_reasoner_claims, verify_final_answer


def _ledger(*items: tuple[Claim, ClaimVerdict]) -> dict[str, tuple[Claim, ClaimVerdict]]:
    return {claim.claim_id: (claim, verdict) for claim, verdict in items}


def test_normalized_claim_scoring_requires_unique_margin() -> None:
    claim_a = Claim("cl_a", "A", "The bridge appears.")
    claim_b = Claim("cl_b", "B", "The tower appears.")

    verification = verify_final_answer(
        "Which statement is correct?",
        {},
        "A",
        claim_ledger=_ledger(
            (claim_a, ClaimVerdict("cl_a", "supported", ("ev_0001",))),
            (claim_b, ClaimVerdict("cl_b", "unknown", ())),
        ),
    )

    assert verification["passed"] is True
    assert verification["winner"] == "A"
    assert verification["scores"] == {"A": 1.0, "B": 0.0}


def test_normalized_claim_scoring_blocks_ties_and_flips_negated_claims() -> None:
    claim_a = Claim("cl_a", "A", "The bridge appears.", polarity="negate")
    claim_b = Claim("cl_b", "B", "The tower appears.")

    verification = verify_final_answer(
        "Which statement is correct?",
        {},
        "A",
        claim_ledger=_ledger(
            (claim_a, ClaimVerdict("cl_a", "supported", ("ev_0001",))),
            (claim_b, ClaimVerdict("cl_b", "unknown", ())),
        ),
    )
    assert verification["passed"] is False
    assert verification["scores"]["A"] == -1.0

    tie = verify_final_answer(
        "Which statement is correct?",
        {},
        "A",
        claim_ledger=_ledger(
            (Claim("cl_a2", "A", "The bridge appears."), ClaimVerdict("cl_a2", "supported", ("ev_0001",))),
            (Claim("cl_b2", "B", "The tower appears."), ClaimVerdict("cl_b2", "supported", ("ev_0002",))),
        ),
    )
    assert tie["passed"] is False
    assert tie["reason"] == "insufficient_verified_evidence"


def test_reasoner_claim_schema_balances_options_and_rejects_unanchored_aggregate_claims() -> None:
    with pytest.raises(InvestigatorOutputInvalid):
        validate_reasoner_claims(
            (
                Claim("cl_a1", "A", "The bridge appears."),
                Claim("cl_a2", "A", "The bridge is red."),
                Claim("cl_a3", "A", "The bridge is tall."),
                Claim("cl_b1", "B", "The tower appears."),
            ),
            options=("A", "B"),
        )

    with pytest.raises(InvestigatorOutputInvalid):
        validate_reasoner_claims((Claim("cl_count", "A", "The total number of speakers is three."),))

    validate_reasoner_claims((Claim("cl_count_ok", "A", "In beat bt00003, the total visible speakers is three."),))
