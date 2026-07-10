from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from vcah.investigator import InvestigationReport, VirtualVideoInvestigator
from vcah.memory import EvidenceStore
from vcah.types import ClaimContract, EvidenceRecord, is_path_only_visual_evidence, to_jsonable
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
    evidence: tuple[EvidenceRecord, ...]
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


def compile_query_contract(question: str) -> ClaimContract:
    text = str(question or "").casefold()
    is_count = bool(re.search(r"\bhow many\b|\bnumber of\b", text))
    full_video = any(
        phrase in text
        for phrase in (
            "throughout the video",
            "in total",
            "entire video",
            "whole video",
            "across the video",
            "over the course of the video",
        )
    )
    language_action = any(term in text for term in ("comment", "say", "speak", "discuss", "mention"))
    if is_count:
        return ClaimContract(
            required_scope="full_video" if full_video else "multi_window",
            quantifier="distinct_count",
            observation_target="entity",
            aggregation="deduplicate",
            required_observability=("visual", "asr") if language_action else ("visual",),
            observability_mode="all",
        )
    return ClaimContract(
        required_scope="window",
        quantifier="existential",
        observation_target="attribute",
        aggregation="none",
        required_observability=("visual",),
        observability_mode="all",
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
        investigator.reset_run_state()
        workspace_overview = build_workspace_overview(workspace, thumbnail_budget=40)
        query_contract = compile_query_contract(workspace.case.question)
        evidence_store = EvidenceStore.empty(workspace.root_dir / "evidence.jsonl")
        reports: list[InvestigationReport] = []
        trace: list[Mapping[str, Any]] = []
        accepted = 0
        answer = ""
        citations: tuple[str, ...] = ()
        rounds_run = 0

        for round_id in range(1, self.max_rounds + 1):
            rounds_run = round_id
            remaining = self.max_investigations - accepted
            completion_status = _completion_status(workspace, query_contract, evidence_store.records)
            decision = _decision(
                self.reasoner.decide(
                    question=workspace.case.question,
                    options=dict(workspace.case.options),
                    workspace_id=workspace.workspace_id,
                    workspace_duration_sec=workspace.manifest.duration_sec,
                    segment_overviews=tuple(workspace_overview["segment_overviews"]),
                    workspace_overview=workspace_overview,
                    query_contract=to_jsonable(query_contract),
                    completion_status=completion_status,
                    available_tools=tuple(workspace_overview["available_tools"]),
                    evidence=evidence_store.records,
                    evidence_digest=_evidence_digest(evidence_store.records),
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
                    "completion_status": completion_status,
                }
            )
            if decision.action == "answer":
                gate = _answer_completion_gate(
                    workspace,
                    query_contract,
                    decision.citations,
                    evidence_store.records,
                )
                trace.append({"type": "completion_gate", "round": round_id, **gate})
                if gate["passed"]:
                    answer = decision.answer
                    if query_contract.aggregation != "none":
                        aggregate = _derived_answer_evidence(
                            workspace,
                            answer=answer,
                            citations=decision.citations,
                            evidence=evidence_store.records,
                        )
                        evidence_store.add(aggregate)
                        citations = (aggregate.evidence_id,)
                    else:
                        citations = decision.citations
                    break
                continue
            if remaining <= 0:
                break
            tasks = decision.tasks[: min(self.max_tasks_per_round, remaining)]
            accepted += len(tasks)
            batch = investigator.run_batch(tasks)
            reports.extend(batch)
            known_evidence = {record.evidence_id for record in evidence_store.records}
            for report in batch:
                for record in report.evidence:
                    if record.evidence_id not in known_evidence:
                        evidence_store.add(record)
                        known_evidence.add(record.evidence_id)
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
            evidence=tuple(evidence_store.records),
            reports=tuple(reports),
            trace=tuple(trace),
        )
        _write_run_summary(workspace, result)
        return result


def _completion_status(
    workspace: VirtualVideoWorkspace,
    contract: ClaimContract,
    evidence: Sequence[EvidenceRecord],
) -> dict[str, Any]:
    coverage = _source_coverage(workspace, evidence)
    if contract.required_scope != "full_video":
        return {
            "ready_for_answer": bool(evidence),
            "required_scope": contract.required_scope,
            "missing_segment_ids": [],
            "source_coverage": coverage,
        }
    if not coverage:
        return {
            "ready_for_answer": False,
            "required_scope": contract.required_scope,
            "reason": "source_not_identified",
            "missing_segment_ids": [],
            "source_coverage": {},
        }
    adopted_source = max(
        coverage,
        key=lambda source_id: (
            int(coverage[source_id]["covered_count"]),
            float(coverage[source_id]["confidence"]),
        ),
    )
    missing = list(coverage[adopted_source]["missing_segment_ids"])
    return {
        "ready_for_answer": not missing,
        "required_scope": contract.required_scope,
        "adopted_source_video_id": adopted_source,
        "missing_segment_ids": missing,
        "source_coverage": coverage,
    }


def _source_coverage(
    workspace: VirtualVideoWorkspace,
    evidence: Sequence[EvidenceRecord],
) -> dict[str, dict[str, Any]]:
    required: dict[str, list[str]] = {}
    for segment in workspace.manifest.segments:
        required.setdefault(segment.source_video_id, []).append(segment.segment_id)
    covered: dict[str, set[str]] = {}
    confidence: dict[str, float] = {}
    for record in evidence:
        if record.modality not in {"visual", "ocr"}:
            continue
        for lineage in record.source_lineage:
            source_id = str(lineage.get("source_video_id", "") or "")
            segment_id = str(lineage.get("segment_id", "") or "")
            if not source_id or not segment_id:
                continue
            covered.setdefault(source_id, set()).add(segment_id)
            confidence[source_id] = max(confidence.get(source_id, 0.0), record.confidence)
    result: dict[str, dict[str, Any]] = {}
    for source_id, segment_ids in covered.items():
        required_ids = tuple(required.get(source_id, ()))
        missing = [segment_id for segment_id in required_ids if segment_id not in segment_ids]
        result[source_id] = {
            "covered_segment_ids": sorted(segment_ids),
            "required_segment_ids": list(required_ids),
            "missing_segment_ids": missing,
            "covered_count": len(segment_ids),
            "required_count": len(required_ids),
            "confidence": confidence.get(source_id, 0.0),
        }
    return result


def _answer_completion_gate(
    workspace: VirtualVideoWorkspace,
    contract: ClaimContract,
    citations: Sequence[str],
    evidence: Sequence[EvidenceRecord],
) -> dict[str, Any]:
    if not _citations_are_visual(citations, evidence):
        return {"passed": False, "reason": "invalid_visual_citations", "missing_segment_ids": []}
    by_id = {record.evidence_id: record for record in evidence}
    cited = tuple(by_id[str(citation)] for citation in citations)
    if contract.required_scope != "full_video":
        return {"passed": True, "reason": "verified_window_evidence", "missing_segment_ids": []}

    cited_sources = {
        str(lineage.get("source_video_id", "") or "")
        for record in cited
        for lineage in record.source_lineage
        if str(lineage.get("source_video_id", "") or "")
    }
    cited_segments = {
        str(lineage.get("segment_id", "") or "")
        for record in cited
        for lineage in record.source_lineage
        if str(lineage.get("segment_id", "") or "")
    }
    required_segments = {
        segment.segment_id
        for segment in workspace.manifest.segments
        if segment.source_video_id in cited_sources
    }
    missing = sorted(required_segments - cited_segments)
    if not cited_sources:
        return {"passed": False, "reason": "source_not_identified", "missing_segment_ids": []}
    if missing:
        return {
            "passed": False,
            "reason": "full_source_coverage_missing",
            "source_video_ids": sorted(cited_sources),
            "missing_segment_ids": missing,
        }
    return {
        "passed": True,
        "reason": "full_source_coverage_verified",
        "source_video_ids": sorted(cited_sources),
        "missing_segment_ids": [],
    }


def _derived_answer_evidence(
    workspace: VirtualVideoWorkspace,
    *,
    answer: str,
    citations: Sequence[str],
    evidence: Sequence[EvidenceRecord],
) -> EvidenceRecord:
    by_id = {record.evidence_id: record for record in evidence}
    parents = tuple(by_id[str(citation)] for citation in citations)
    starts = [record.start_sec for record in parents if record.start_sec is not None]
    ends = [record.end_sec for record in parents if record.end_sec is not None]
    coverage = tuple(segment for record in parents for segment in record.coverage_manifest)
    lineage = tuple(dict(item) for record in parents for item in record.source_lineage)
    request_ids = tuple(dict.fromkeys(request_id for record in parents for request_id in record.request_ids))
    return EvidenceRecord(
        evidence_id="ev_final_aggregate",
        beat_id="",
        start_sec=min(starts) if starts else None,
        end_sec=max(ends) if ends else None,
        modality="derived",
        pointer=f"virtual://{workspace.workspace_id}/derived/final",
        verbatim=f"Final answer {answer!r} aggregates {len(parents)} cited observations.",
        claim=answer,
        attestation_model="reasoner",
        temporal_scope="full_video",
        evidence_kind="aggregate",
        observation_polarity="positive",
        sampling_coverage="sparse",
        parent_evidence_ids=tuple(record.evidence_id for record in parents),
        request_ids=request_ids,
        coverage_manifest=coverage,
        task_id="final_answer",
        observation_id="final_aggregate",
        confidence=min((record.confidence for record in parents), default=0.0),
        source_lineage=lineage,
    )


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


def _evidence_digest(evidence: Sequence[EvidenceRecord]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "evidence_id": item.evidence_id,
            "summary": item.verbatim,
            "confidence": item.confidence,
            "virtual_time_range": [item.start_sec, item.end_sec],
            "modality": item.modality,
            "evidence_kind": item.evidence_kind,
            "source_lineage": [dict(row) for row in item.source_lineage],
        }
        for item in evidence
    )


def _citations_are_visual(citations: Sequence[str], evidence: Sequence[EvidenceRecord]) -> bool:
    if not citations:
        return False
    by_id = {item.evidence_id: item for item in evidence}
    cited = [by_id.get(str(citation)) for citation in citations]
    return bool(cited) and all(
        item is not None and item.modality in {"visual", "ocr"} and not is_path_only_visual_evidence(item)
        for item in cited
    )


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
                        "summary": item.verbatim,
                        "modality": item.modality,
                        "evidence_kind": item.evidence_kind,
                        "virtual_time_range": [item.start_sec, item.end_sec],
                        "sampling_fps": item.sampling_fps,
                        "confidence": item.confidence,
                        "task_id": item.task_id,
                        "observation_id": item.observation_id,
                        "pointer": item.pointer,
                        "frame_refs": list(item.frame_refs),
                        "parent_evidence_ids": list(item.parent_evidence_ids),
                        "coverage_manifest": to_jsonable(item.coverage_manifest),
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
