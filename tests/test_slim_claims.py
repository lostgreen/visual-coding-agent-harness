from __future__ import annotations

import pytest

from vcah.types import Claim, InvestigatorOutputInvalid, QueryClaim, ToolAction


def test_tool_action_keeps_internal_claim_separate_from_query_claim() -> None:
    action = ToolAction.from_mapping(
        {
            "type": "inspect_window",
            "claims": [
                {
                    "claim_id": "cl_R1_01",
                    "option": "C",
                    "text": "The speaker mentions the bridge.",
                    "polarity": "negate",
                }
            ],
        }
    )

    claim = action.claims[0]
    query_claim = QueryClaim.from_claim(claim)

    assert claim.option == "C"
    assert claim.polarity == "negate"
    assert query_claim.claim_id == "cl_R1_01"
    assert query_claim.text == "The speaker mentions the bridge."
    assert not hasattr(query_claim, "option")
    assert not hasattr(query_claim, "polarity")


def test_claim_text_reuses_investigator_hypothesis_guard() -> None:
    with pytest.raises(InvestigatorOutputInvalid):
        Claim("cl_bad", "C", "Option C says the bridge is correct.")
