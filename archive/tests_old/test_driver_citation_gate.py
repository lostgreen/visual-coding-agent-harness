from __future__ import annotations

from visual_coding_agent_harness.agents.driver import MultiV3Driver
from visual_coding_agent_harness.contracts.evidence import EvidenceRecord
from visual_coding_agent_harness.contracts.report import Finding
from visual_coding_agent_harness.workspace.investigator_ws import InvestigatorWorkspace


class AnswerReasoner:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, **kwargs):
        del kwargs
        self.calls += 1
        return type(
            "Decision",
            (),
            {
                "action": "answer",
                "answer": "A",
                "confidence": "medium",
                "citations": ("finding_only",) if self.calls == 1 else ("er_known",),
                "rationale": "because evidence",
                "goals": (),
            },
        )()


def test_driver_rejects_answer_without_valid_citation_and_retries(tmp_path) -> None:
    workspace = InvestigatorWorkspace(tmp_path)
    workspace.ledger.append(
        Finding(
            finding_id="finding_only",
            query_id="q0",
            shot_id="sh001",
            summary="Known evidence.",
            citation_ids=("finding_only",),
        )
    )
    workspace.evidence_records.append(
        EvidenceRecord(
            evidence_id="er_known",
            claim="Known evidence.",
            stance="supports",
            modality="asr",
            time_sec=1.0,
            pointer="bt00001",
            verbatim="Known evidence.",
            query_id="q0",
            beat_id="bt00001",
        )
    )
    reasoner = AnswerReasoner()
    driver = MultiV3Driver(reasoner=reasoner, investigator=object(), workspace=workspace, max_rounds=2)

    result = driver.run(question="Q?", options={"A": "yes"}, index_context="ch01")

    assert reasoner.calls == 2
    assert result.answer == "A"
    assert result.citations == ("er_known",)
