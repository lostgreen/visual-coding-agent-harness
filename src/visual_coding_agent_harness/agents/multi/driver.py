"""Round driver for the first multi-agent video QA slice."""

from __future__ import annotations

from typing import Any, Mapping

from ..workspace_agent import WorkspaceRunResult
from .reasoner import best_effort_answer_from_workspace


class MultiAgentDriver:
    """Alternate a Reasoner and Investigator over a shared workspace."""

    def __init__(self, *, reasoner: Any, investigator: Any, workspace: Any, max_rounds: int = 8) -> None:
        self.reasoner = reasoner
        self.investigator = investigator
        self.workspace = workspace
        self.max_rounds = int(max_rounds)

    def run(self, question: str, options: Mapping[str, str] | None = None) -> WorkspaceRunResult:
        option_payload = dict(options or {})
        self._last_options = option_payload
        self._record_answer_options(option_payload)
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
        best_effort = best_effort_answer_from_workspace(self.workspace, getattr(self, "_last_options", {}))
        if best_effort is not None:
            answer, citations = best_effort
            if hasattr(self.workspace, "write_trace_event"):
                self.workspace.write_trace_event(
                    "forced_final_emitted",
                    {
                        "reason": reason,
                        "attempted_option": answer,
                        "confidence": "low",
                        "best_effort_from_visual_support": True,
                    },
                )
            return WorkspaceRunResult(
                answer=answer,
                citations=citations,
                confidence="low",
                rounds=round_number,
                metadata={
                    "strategy": "multi_agent_v0",
                    "forced_final": True,
                    "reason": reason,
                    "best_effort_from_visual_support": True,
                },
            )
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

    def _record_answer_options(self, options: Mapping[str, str]) -> None:
        if not options or not hasattr(self.workspace, "search_ledger_snapshot"):
            return
        try:
            snapshot = dict(self.workspace.search_ledger_snapshot())
            answer_options = dict(snapshot.get("answer_options", {}) or {})
            ledger_options = dict(snapshot.get("options", {}) or {})
            for option_id, text in options.items():
                key = str(option_id).strip().upper()[:1]
                if not key:
                    continue
                answer_options[key] = str(text)
                ledger_options.setdefault(key, {"option_id": key, "status": "untested"})
            snapshot["answer_options"] = answer_options
            snapshot["options"] = ledger_options
            writer = getattr(self.workspace, "_write_search_ledger_snapshot", None)
            if callable(writer):
                writer(snapshot)
        except Exception:  # noqa: BLE001 - option ledger is best-effort context only
            return
