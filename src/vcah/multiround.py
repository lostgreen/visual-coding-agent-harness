from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from vcah.investigator import InvestigationEvidence, InvestigationReport, VirtualVideoInvestigator
from vcah.virtual_index import build_workspace_overview
from vcah.virtual_video import VirtualVideoWorkspace


@dataclass(frozen=True)
class InvestigationTask:
    query_id: str
    goal: str
    segment_id: str = ""
    time_range: tuple[float, float] | None = None
    modality_hint: tuple[str, ...] = ()
    expected_evidence: str = ""
    priority: float = 0.0

    def __post_init__(self) -> None:
        if self.time_range is not None:
            start, end = self.time_range
            object.__setattr__(self, "time_range", (float(start), float(end)))
        object.__setattr__(self, "segment_id", str(self.segment_id or ""))
        object.__setattr__(self, "modality_hint", tuple(str(item) for item in self.modality_hint))


@dataclass(frozen=True)
class ReasonerDecision:
    action: str
    tasks: tuple[InvestigationTask, ...] = ()
    answer: str = ""
    citations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tasks", tuple(_task(item) for item in self.tasks))
        object.__setattr__(self, "citations", tuple(str(item) for item in self.citations if str(item).strip()))


@dataclass(frozen=True)
class MultiRoundResult:
    case_id: str
    answer: str
    citations: tuple[str, ...]
    correct: bool
    rounds: int
    accepted_investigations: int
    evidence: tuple[InvestigationEvidence, ...]
    reports: tuple[InvestigationReport, ...]
    trace: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)


class HeuristicReasoner:
    def decide(self, **kwargs: Any) -> ReasonerDecision:
        remaining = int(kwargs.get("remaining_budget", 0))
        evidence = tuple(kwargs.get("evidence", ()) or ())
        options = dict(kwargs.get("options", {}) or {})
        if evidence:
            first_key = next(iter(options.keys()), "")
            first_answer = f"{first_key}. {options[first_key]}" if first_key else ""
            return ReasonerDecision(action="answer", answer=first_answer, citations=(evidence[0].evidence_id,))
        if remaining <= 0:
            return ReasonerDecision(action="answer", answer="", citations=())
        overview = dict(kwargs.get("workspace_overview", {}) or {})
        rows = tuple(overview.get("segment_overviews", ()) or ())
        if rows:
            segment_id = str(rows[0].get("segment_id", "") or "")
        else:
            segment_id = ""
        return ReasonerDecision(
            action="investigate",
            tasks=(
                InvestigationTask(
                    query_id="q_round1_001",
                    goal=str(kwargs.get("question", "")),
                    segment_id=segment_id,
                    time_range=None,
                    modality_hint=("visual", "ocr"),
                    expected_evidence=str(kwargs.get("question", "")),
                    priority=1.0,
                ),
            ),
        )


class VirtualVideoMultiRoundDriver:
    def __init__(
        self,
        *,
        reasoner: Any | None = None,
        investigator: VirtualVideoInvestigator | None = None,
        max_rounds: int = 4,
        max_investigations: int = 20,
        max_tasks_per_round: int = 4,
    ) -> None:
        self.reasoner = reasoner or HeuristicReasoner()
        self.investigator = investigator
        self.max_rounds = max(1, int(max_rounds))
        self.max_investigations = max(1, int(max_investigations))
        self.max_tasks_per_round = max(1, int(max_tasks_per_round))

    def run(self, workspace: VirtualVideoWorkspace) -> MultiRoundResult:
        investigator = self.investigator or VirtualVideoInvestigator(workspace)
        workspace_overview = build_workspace_overview(workspace, thumbnail_budget=40)
        evidence: list[InvestigationEvidence] = []
        reports: list[InvestigationReport] = []
        trace: list[Mapping[str, Any]] = []
        accepted = 0
        answer = ""
        citations: tuple[str, ...] = ()
        rounds_run = 0

        for round_id in range(1, self.max_rounds + 1):
            rounds_run = round_id
            remaining = self.max_investigations - accepted
            decision = _decision(
                self.reasoner.decide(
                    question=workspace.case.question,
                    options=dict(workspace.case.options),
                    workspace_id=workspace.workspace_id,
                    workspace_duration_sec=workspace.manifest.duration_sec,
                    segment_overviews=tuple(workspace_overview["segment_overviews"]),
                    workspace_overview=workspace_overview,
                    available_tools=tuple(workspace_overview["available_tools"]),
                    evidence=evidence,
                    evidence_digest=_evidence_digest(evidence),
                    remaining_budget=remaining,
                )
            )
            trace.append(
                {
                    "type": "reasoner_decision",
                    "round": round_id,
                    "action": decision.action,
                    "task_count": len(decision.tasks),
                    "remaining_budget": remaining,
                }
            )
            if decision.action == "answer":
                if _citations_are_visual(decision.citations, evidence):
                    answer = decision.answer
                    citations = decision.citations
                break
            if remaining <= 0:
                break
            tasks = decision.tasks[: min(self.max_tasks_per_round, remaining)]
            accepted += len(tasks)
            batch = investigator.run_batch(tasks)
            reports.extend(batch)
            for report in batch:
                evidence.extend(report.evidence)
            trace.append({"type": "investigator_batch", "round": round_id, "accepted_tasks": len(tasks)})
            if accepted >= self.max_investigations:
                continue

        result = MultiRoundResult(
            case_id=workspace.case.case_id,
            answer=answer or "Insufficient verified evidence.",
            citations=citations,
            correct=_score_answer(answer, workspace.case.gold),
            rounds=rounds_run,
            accepted_investigations=accepted,
            evidence=tuple(evidence),
            reports=tuple(reports),
            trace=tuple(trace),
        )
        _write_run_summary(workspace, result)
        return result


def _task(value: InvestigationTask | Mapping[str, Any]) -> InvestigationTask:
    if isinstance(value, InvestigationTask):
        return value
    return InvestigationTask(
        query_id=str(value.get("query_id", "")),
        goal=str(value.get("goal", "")),
        segment_id=str(value.get("segment_id", "") or ""),
        time_range=None if value.get("time_range") is None else tuple(value.get("time_range", (0.0, 0.0))),  # type: ignore[arg-type]
        modality_hint=tuple(value.get("modality_hint", ())),
        expected_evidence=str(value.get("expected_evidence", "")),
        priority=float(value.get("priority", 0.0) or 0.0),
    )


def _decision(value: ReasonerDecision | Mapping[str, Any]) -> ReasonerDecision:
    if isinstance(value, ReasonerDecision):
        return value
    return ReasonerDecision(
        action=str(value.get("action", "")),
        tasks=tuple(value.get("tasks", ())),
        answer=str(value.get("answer", "")),
        citations=tuple(value.get("citations", ())),
    )


def _evidence_digest(evidence: Sequence[InvestigationEvidence]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "evidence_id": item.evidence_id,
            "summary": item.summary,
            "confidence": item.confidence,
            "virtual_time_range": list(item.virtual_time_range),
        }
        for item in evidence
    )


def _citations_are_visual(citations: Sequence[str], evidence: Sequence[InvestigationEvidence]) -> bool:
    if not citations:
        return False
    by_id = {item.evidence_id: item for item in evidence}
    cited = [by_id.get(str(citation)) for citation in citations]
    return bool(cited) and all(item is not None and item.modality != "asr" for item in cited)


def _score_answer(answer: str, gold: str) -> bool:
    selected = _letter(answer)
    expected = _letter(gold) or str(gold or "").strip().upper()[:1]
    return bool(selected and expected and selected == expected)


def _letter(value: str) -> str:
    match = re.search(r"\b([A-H])\b", str(value or "").strip().upper())
    return match.group(1) if match else ""


def _write_run_summary(workspace: VirtualVideoWorkspace, result: MultiRoundResult) -> None:
    path = workspace.root_dir / "run_summary.json"
    path.write_text(
        json.dumps(
            {
                "case_id": result.case_id,
                "answer": result.answer,
                "citations": list(result.citations),
                "correct": result.correct,
                "rounds": result.rounds,
                "accepted_investigations": result.accepted_investigations,
                "evidence": [
                    {
                        "evidence_id": item.evidence_id,
                        "summary": item.summary,
                        "modality": item.modality,
                        "virtual_time_range": list(item.virtual_time_range),
                        "source_lineage": [dict(row) for row in item.source_lineage],
                    }
                    for item in result.evidence
                ],
                "trace": [dict(item) for item in result.trace],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
