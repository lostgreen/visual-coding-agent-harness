from __future__ import annotations

from visual_coding_agent_harness.agents.driver import MultiV3Driver
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
                "citations": () if self.calls == 1 else ("ev_known",),
                "rationale": "because evidence",
                "goals": (),
            },
        )()


def test_driver_rejects_answer_without_valid_citation_and_retries(tmp_path) -> None:
    workspace = InvestigatorWorkspace(tmp_path)
    workspace.ledger.append(
        Finding(
            finding_id="ev_known",
            query_id="q0",
            shot_id="sh001",
            summary="Known evidence.",
            citation_ids=("ev_known",),
        )
    )
    reasoner = AnswerReasoner()
    driver = MultiV3Driver(reasoner=reasoner, investigator=object(), workspace=workspace, max_rounds=2)

    result = driver.run(question="Q?", options={"A": "yes"}, index_context="ch01")

    assert reasoner.calls == 2
    assert result.answer == "A"
    assert result.citations == ("ev_known",)
