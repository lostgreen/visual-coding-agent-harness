from __future__ import annotations

from pathlib import Path

from vcah.agent import _verify_answer_citations
from vcah.memory import EvidenceStore
from vcah.types import Claim, ClaimVerdict, EvidenceRecord, ToolAction, verify_final_answer


def test_mcq_without_claim_ledger_fails_closed_even_with_legacy_table(tmp_path: Path) -> None:
    evidence = EvidenceStore.empty(tmp_path / "evidence.jsonl")
    evidence.add(
        EvidenceRecord(
            evidence_id="ev_0001",
            beat_id="bt00001",
            start_sec=0.0,
            end_sec=4.0,
            modality="asr",
            pointer="bt00001@0.000-4.000",
            verbatim="The bridge is mentioned.",
        )
    )

    verification = _verify_answer_citations(
        evidence,
        ToolAction(
            type="answer",
            selected="A",
            answer="A",
            citations=("ev_0001",),
            evidence_table={"A": {"status": "supported", "support": [{"text": "The bridge is mentioned."}], "contradict": []}},
        ),
        "Which statement is correct?\nA. The bridge is mentioned.\nB. The tower is mentioned.",
        claim_ledger=None,
    )

    assert verification["passed"] is False
    assert verification["reason"] == "missing_claim_ledger_for_mcq"


def test_negative_mcq_claim_ledger_chooses_contradicted_option() -> None:
    claim_ledger = {
        "cl_A": (
            Claim("cl_A", "A", "The bridge is mentioned."),
            ClaimVerdict("cl_A", "supported", ("ev_0001",)),
        ),
        "cl_B": (
            Claim("cl_B", "B", "The tower is mentioned."),
            ClaimVerdict("cl_B", "contradicted", (), ("ev_0002",)),
        ),
    }

    verification = verify_final_answer(
        "Which statement is not correct?\nA. The bridge is mentioned.\nB. The tower is mentioned.",
        {},
        "B",
        claim_ledger=claim_ledger,
        threshold=0.1,
    )

    assert verification["passed"] is True
    assert verification["winner"] == "B"
    assert verification["selection_policy"] == "choose_contradicted"
