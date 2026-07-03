"""Parallel dispatch driver for the multi_v3 loop."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Mapping, Sequence

from visual_coding_agent_harness.contracts.query import ScopedQuery
from visual_coding_agent_harness.contracts.report import DigestItem, InvestigationReport
from visual_coding_agent_harness.agents.result import WorkspaceRunResult
from visual_coding_agent_harness.workspace.digest import digest_reports
from visual_coding_agent_harness.workspace.investigator_ws import InvestigatorWorkspace


class MultiV3Driver:
    def __init__(
        self,
        *,
        reasoner,
        investigator,
        workspace: InvestigatorWorkspace,
        max_rounds: int,
        max_concurrency: int = 4,
    ) -> None:
        self.reasoner = reasoner
        self.investigator = investigator
        self.workspace = workspace
        self.max_rounds = max(1, int(max_rounds))
        self.max_concurrency = max(1, int(max_concurrency))

    def run(
        self,
        *,
        question: str,
        options: Mapping[str, str],
        index_context: str,
        overview_path: str,
    ) -> WorkspaceRunResult:
        previous_digest: tuple[DigestItem, ...] = ()
        last_reports: tuple[InvestigationReport, ...] = ()
        for round_number in range(1, self.max_rounds + 1):
            decision = self.reasoner.decide(
                question=question,
                options=options,
                index_context=index_context,
                overview_path=overview_path,
                previous_digest=previous_digest,
                round_number=round_number,
            )
            if decision.action == "answer":
                return decision.to_run_result(rounds=round_number)
            queries = tuple(decision.queries)
            if not queries:
                continue
            last_reports = self._dispatch_queries(queries)
            previous_digest = digest_reports(last_reports, query_goal_ids={query.query_id: query.goal_id for query in queries})
        return WorkspaceRunResult(
            answer="need_more_evidence",
            citations=tuple(citation for item in previous_digest for citation in item.citation_ids),
            confidence="low",
            rounds=self.max_rounds,
            metadata={
                "status": "need_more_evidence",
                "strategy": "multi_v3",
                "report_count": len(last_reports),
            },
        )

    def _dispatch_queries(self, queries: Sequence[ScopedQuery]) -> tuple[InvestigationReport, ...]:
        with ThreadPoolExecutor(max_workers=min(self.max_concurrency, len(queries))) as executor:
            return tuple(executor.map(self.investigator.run, queries))
