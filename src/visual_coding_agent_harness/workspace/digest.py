"""Digest helpers for feeding compact findings back to the Reasoner."""

from __future__ import annotations

from typing import Mapping, Sequence

from visual_coding_agent_harness.contracts.report import DigestItem, InvestigationReport


def digest_reports(
    reports: Sequence[InvestigationReport],
    *,
    query_goal_ids: Mapping[str, str],
) -> tuple[DigestItem, ...]:
    return tuple(
        DigestItem.from_report(report, goal_id=str(query_goal_ids.get(report.query_id) or ""))
        for report in reports
    )
