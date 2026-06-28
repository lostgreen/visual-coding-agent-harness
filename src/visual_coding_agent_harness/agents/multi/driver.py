"""Round driver for the first multi-agent video QA slice."""

from __future__ import annotations

from typing import Any, Mapping

from ..workspace_agent import WorkspaceRunResult


class MultiAgentDriver:
    """Alternate a Reasoner and Investigator over a shared workspace."""

    def __init__(self, *, reasoner: Any, investigator: Any, workspace: Any, max_rounds: int = 8) -> None:
        self.reasoner = reasoner
        self.investigator = investigator
        self.workspace = workspace
        self.max_rounds = int(max_rounds)

    def run(self, question: str, options: Mapping[str, str] | None = None) -> WorkspaceRunResult:
        option_payload = dict(options or {})
        idle_streak = 0
        last_investigator_acted = False
        for round_number in range(1, self.max_rounds + 1):
            reasoner_acted = bool(
                self.reasoner.step(round_number=round_number, question=question, options=option_payload)
            )
            answer_result = getattr(self.reasoner, "answer_result", None)
            if answer_result is not None:
                return answer_result

            investigator_acted = bool(self.investigator.step(round_number=round_number))
            last_investigator_acted = investigator_acted
            if not reasoner_acted and not investigator_acted:
                idle_streak += 1
                if idle_streak >= 2:
                    return self._forced_final(round_number, reason="double_idle")
            else:
                idle_streak = 0

        if last_investigator_acted:
            self.reasoner.step(round_number=self.max_rounds + 1, question=question, options=option_payload)
            answer_result = getattr(self.reasoner, "answer_result", None)
            if answer_result is not None:
                return answer_result

        return self._forced_final(self.max_rounds, reason="max_rounds")

    def _forced_final(self, round_number: int, *, reason: str) -> WorkspaceRunResult:
        if hasattr(self.workspace, "write_trace_event"):
            self.workspace.write_trace_event(
                "forced_final_emitted",
                {"reason": reason, "attempted_option": "", "confidence": "low"},
            )
        return WorkspaceRunResult(
            answer="need_more_evidence",
            citations=(),
            confidence="low",
            rounds=round_number,
            metadata={"strategy": "multi_agent_v0", "forced_final": True, "reason": reason},
        )
