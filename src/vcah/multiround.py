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
    entity_clusters: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tasks", tuple(_task(item) for item in self.tasks))
        object.__setattr__(self, "citations", tuple(str(item) for item in self.citations if str(item).strip()))
        object.__setattr__(self, "entity_clusters", tuple(_entity_cluster(item) for item in self.entity_clusters))


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
    is_occurrence_count = bool(re.search(r"\bhow many times\b|\bnumber of times\b", text))
    full_video = "in total" in text or bool(
        re.search(r"\b(?:throughout|across)\b.*\b(?:video|film|recording)\b", text)
        or re.search(r"\b(?:entire|whole)\s+(?:video|film|recording)\b", text)
        or re.search(r"\bin\s+(?:this|the)\s+(?:video|film|recording)\b", text)
        or re.search(r"\bover the course of\s+(?:this|the)\s+(?:video|film|recording)\b", text)
    )
    language_action = any(term in text for term in ("comment", "say", "speak", "discuss", "mention"))
    identity_anchor_terms = _identity_anchor_terms(question)
    if is_count:
        return ClaimContract(
            required_scope="full_video" if full_video else "multi_window",
            quantifier="total_count" if is_occurrence_count else "distinct_count",
            observation_target="event" if is_occurrence_count else "entity",
            aggregation="count" if is_occurrence_count else "deduplicate",
            required_observability=("visual", "asr") if language_action else ("visual",),
            observability_mode="all",
        )
    if identity_anchor_terms:
        return ClaimContract(
            required_scope="multi_window",
            quantifier="none",
            observation_target="entity",
            aggregation="compare",
            required_observability=("visual",),
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


def compile_query_requirements(question: str) -> dict[str, Any]:
    terms = _identity_anchor_terms(question)
    return {
        "requires_identity_link": bool(terms),
        "identity_anchor_terms": list(terms),
    }


def _identity_anchor_terms(question: str) -> tuple[str, ...]:
    text = str(question or "").casefold()
    match = re.search(r"\bwho\s+([^,?]+)", text)
    if not match:
        return ()
    clause = match.group(1)
    tokens = tuple(re.findall(r"[a-z0-9]+", clause))
    attribute_markers = {
        "wearing",
        "wears",
        "wore",
        "holding",
        "holds",
        "held",
        "carrying",
        "carries",
        "carried",
        "having",
        "dressed",
        "with",
    }
    if not attribute_markers.intersection(tokens):
        return ()
    stop = {
        "with",
        "was",
        "is",
        "had",
        "been",
        "wearing",
        "wears",
        "wore",
        "holding",
        "holds",
        "held",
        "carrying",
        "carries",
        "carried",
        "having",
        "dressed",
        "in",
        "and",
        "the",
        "this",
        "that",
        "his",
        "her",
        "their",
        "a",
        "an",
    }
    return tuple(
        token
        for token in tokens
        if len(token) >= 3 and token not in stop
    )


def compile_source_time_hint(question: str) -> tuple[float, float] | None:
    text = str(question or "").casefold()
    patterns = (
        r"\b(\d+)(?:st|nd|rd|th)?\s*(?:to|through|-)\s*(\d+)(?:st|nd|rd|th)?\s+minute",
        r"\bfrom\s+minute\s+(\d+)\s+(?:to|through|-)\s+minute\s+(\d+)",
        r"\bbetween\s+minutes?\s+(\d+)\s+and\s+(\d+)",
    )
    match = next((candidate for pattern in patterns if (candidate := re.search(pattern, text))), None)
    if not match:
        return None
    start_minute = int(match.group(1))
    end_minute = int(match.group(2))
    if end_minute <= start_minute:
        return None
    return float(start_minute * 60), float(end_minute * 60)


def source_time_navigation(
    workspace: VirtualVideoWorkspace,
    source_time_hint: tuple[float, float] | None,
) -> dict[str, Any]:
    if source_time_hint is None:
        return {"source_time_range": None, "candidate_segments": []}
    source_start, source_end = source_time_hint
    candidates = []
    for segment in workspace.manifest.segments:
        overlap_start = max(float(source_start), segment.source_start_sec)
        overlap_end = min(float(source_end), segment.source_end_sec)
        if overlap_end <= overlap_start:
            continue
        virtual_start = segment.virtual_start_sec + (overlap_start - segment.source_start_sec)
        virtual_end = segment.virtual_start_sec + (overlap_end - segment.source_start_sec)
        candidates.append(
            {
                "segment_id": segment.segment_id,
                "source_time_range": [round(overlap_start, 3), round(overlap_end, 3)],
                "virtual_time_range": [round(virtual_start, 3), round(virtual_end, 3)],
            }
        )
    return {
        "source_time_range": [float(source_start), float(source_end)],
        "candidate_segments": candidates,
    }


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
        query_requirements = compile_query_requirements(workspace.case.question)
        temporal_navigation = source_time_navigation(workspace, compile_source_time_hint(workspace.case.question))
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
            completion_status = _completion_status(
                workspace,
                query_contract,
                evidence_store.records,
                query_requirements=query_requirements,
            )
            decision = _decision(
                self.reasoner.decide(
                    question=workspace.case.question,
                    options=dict(workspace.case.options),
                    workspace_id=workspace.workspace_id,
                    workspace_duration_sec=workspace.manifest.duration_sec,
                    segment_overviews=tuple(workspace_overview["segment_overviews"]),
                    workspace_overview=workspace_overview,
                    query_contract=to_jsonable(query_contract),
                    query_requirements=query_requirements,
                    completion_status=completion_status,
                    temporal_navigation=temporal_navigation,
                    available_tools=tuple(workspace_overview["available_tools"]),
                    evidence=evidence_store.records,
                    evidence_digest=_evidence_digest(evidence_store.records),
                    remaining_budget=remaining,
                )
            )
            missing_segments = tuple(completion_status.get("missing_segment_ids", ()) or ())
            missing_identity_terms = tuple(completion_status.get("missing_identity_anchor_terms", ()) or ())
            if missing_segments and remaining > 0 and (decision.action != "investigate" or not decision.tasks):
                repair_tasks = _coverage_repair_tasks(
                    round_id,
                    missing_segments,
                    query_contract,
                    limit=min(self.max_tasks_per_round, remaining),
                )
                trace.append(
                    {
                        "type": "repair_override",
                        "round": round_id,
                        "reason": "premature_answer_before_coverage",
                        "missing_segment_ids": list(missing_segments),
                        "task_count": len(repair_tasks),
                    }
                )
                decision = ReasonerDecision(action="investigate", tasks=repair_tasks)
            elif missing_identity_terms and remaining > 0 and (decision.action != "investigate" or not decision.tasks):
                repair_tasks = _identity_repair_tasks(
                    workspace,
                    evidence_store.records,
                    missing_identity_terms,
                    round_id=round_id,
                    limit=min(self.max_tasks_per_round, remaining),
                )
                if repair_tasks:
                    trace.append(
                        {
                            "type": "repair_override",
                            "round": round_id,
                            "reason": "identity_anchor_missing",
                            "identity_anchor_terms": list(missing_identity_terms),
                            "task_count": len(repair_tasks),
                        }
                    )
                    decision = ReasonerDecision(action="investigate", tasks=repair_tasks)
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
                    decision.answer,
                    decision.citations,
                    decision.entity_clusters,
                    evidence_store.records,
                    query_requirements=query_requirements,
                )
                trace.append({"type": "completion_gate", "round": round_id, **gate})
                if gate["passed"]:
                    answer = decision.answer
                    if query_contract.aggregation != "none":
                        aggregate = _derived_answer_evidence(
                            workspace,
                            answer=answer,
                            citations=decision.citations,
                            entity_clusters=decision.entity_clusters,
                            evidence=evidence_store.records,
                            coverage_source_ids=gate.get("source_video_ids", ()),
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

        if not answer and evidence_store.records:
            completion_status = _completion_status(
                workspace,
                query_contract,
                evidence_store.records,
                query_requirements=query_requirements,
            )
            final_decision = _decision(
                self.reasoner.decide(
                    question=workspace.case.question,
                    options=dict(workspace.case.options),
                    workspace_id=workspace.workspace_id,
                    workspace_duration_sec=workspace.manifest.duration_sec,
                    segment_overviews=tuple(workspace_overview["segment_overviews"]),
                    workspace_overview=workspace_overview,
                    query_contract=to_jsonable(query_contract),
                    query_requirements=query_requirements,
                    completion_status=completion_status,
                    temporal_navigation=temporal_navigation,
                    available_tools=tuple(workspace_overview["available_tools"]),
                    evidence=evidence_store.records,
                    evidence_digest=_evidence_digest(evidence_store.records),
                    remaining_budget=0,
                    force_finalize=True,
                )
            )
            trace.append(
                {
                    "type": "reasoner_finalization",
                    "round": self.max_rounds + 1,
                    "action": final_decision.action,
                    "citation_count": len(final_decision.citations),
                    "completion_status": completion_status,
                }
            )
            if final_decision.action == "answer":
                gate = _answer_completion_gate(
                    workspace,
                    query_contract,
                    final_decision.answer,
                    final_decision.citations,
                    final_decision.entity_clusters,
                    evidence_store.records,
                    query_requirements=query_requirements,
                )
                trace.append(
                    {
                        "type": "completion_gate",
                        "round": self.max_rounds + 1,
                        "finalization": True,
                        **gate,
                    }
                )
                if gate["passed"]:
                    answer = final_decision.answer
                    if query_contract.aggregation != "none":
                        aggregate = _derived_answer_evidence(
                            workspace,
                            answer=answer,
                            citations=final_decision.citations,
                            entity_clusters=final_decision.entity_clusters,
                            evidence=evidence_store.records,
                            coverage_source_ids=gate.get("source_video_ids", ()),
                        )
                        evidence_store.add(aggregate)
                        citations = (aggregate.evidence_id,)
                    else:
                        citations = final_decision.citations

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


def _coverage_repair_tasks(
    round_id: int,
    segment_ids: Sequence[str],
    contract: ClaimContract,
    *,
    limit: int,
) -> tuple[InvestigationTask, ...]:
    modalities = tuple(contract.required_observability or ("visual",))
    return tuple(
        InvestigationTask(
            query_id=f"repair_r{round_id}_{index:03d}",
            goal=f"Inspect remaining source segment {segment_id} for evidence required by the full-video claim.",
            segment_id=str(segment_id),
            modality_hint=modalities,
            expected_evidence="entity observations and topic evidence needed for full-source coverage",
            priority=1.0,
        )
        for index, segment_id in enumerate(tuple(segment_ids)[: max(0, int(limit))], start=1)
    )


def _identity_repair_tasks(
    workspace: VirtualVideoWorkspace,
    evidence: Sequence[EvidenceRecord],
    identity_terms: Sequence[str],
    *,
    round_id: int,
    limit: int,
) -> tuple[InvestigationTask, ...]:
    terms = tuple(str(item).strip() for item in identity_terms if str(item).strip())
    if not terms:
        return ()
    visited = {
        str(lineage.get("segment_id", "") or "")
        for record in evidence
        for lineage in record.source_lineage
        if str(lineage.get("segment_id", "") or "")
    }
    candidates = tuple(segment for segment in workspace.manifest.segments if segment.segment_id not in visited)
    description = ", ".join(terms)
    return tuple(
        InvestigationTask(
            query_id=f"identity_repair_r{round_id}_{index:03d}",
            goal=f"Locate one visible entity jointly matching these identity attributes: {description}.",
            segment_id=segment.segment_id,
            modality_hint=("visual",),
            expected_evidence=f"one visible entity jointly matching: {description}",
            priority=1.0,
        )
        for index, segment in enumerate(candidates[: max(0, int(limit))], start=1)
    )


def _completion_status(
    workspace: VirtualVideoWorkspace,
    contract: ClaimContract,
    evidence: Sequence[EvidenceRecord],
    *,
    query_requirements: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    coverage = _source_coverage(workspace, evidence)
    if contract.required_scope != "full_video":
        return _apply_identity_completion(
            {
                "ready_for_answer": bool(evidence),
                "required_scope": contract.required_scope,
                "missing_segment_ids": [],
                "source_coverage": coverage,
            },
            evidence,
            query_requirements,
        )
    if not coverage:
        return _apply_identity_completion(
            {
                "ready_for_answer": False,
                "required_scope": contract.required_scope,
                "reason": "source_not_identified",
                "missing_segment_ids": [],
                "source_coverage": {},
            },
            evidence,
            query_requirements,
        )
    adopted_source = max(
        coverage,
        key=lambda source_id: (
            int(coverage[source_id]["covered_count"]),
            float(coverage[source_id]["confidence"]),
        ),
    )
    missing = list(coverage[adopted_source]["missing_segment_ids"])
    return _apply_identity_completion(
        {
            "ready_for_answer": not missing,
            "required_scope": contract.required_scope,
            "adopted_source_video_id": adopted_source,
            "missing_segment_ids": missing,
            "source_coverage": coverage,
        },
        evidence,
        query_requirements,
    )


def _apply_identity_completion(
    status: Mapping[str, Any],
    evidence: Sequence[EvidenceRecord],
    query_requirements: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = dict(status)
    terms = tuple(str(item) for item in (query_requirements or {}).get("identity_anchor_terms", ()) if str(item))
    if not terms:
        return result
    anchor_ids = [
        record.evidence_id
        for record in evidence
        if _record_matches_identity_anchor(record, terms)
    ]
    missing_terms = [] if anchor_ids else list(terms)
    result.update(
        {
            "identity_anchor_terms": list(terms),
            "identity_anchor_evidence_ids": anchor_ids,
            "missing_identity_anchor_terms": missing_terms,
            "identity_anchor_joint_match": bool(anchor_ids),
        }
    )
    result["ready_for_answer"] = bool(result.get("ready_for_answer")) and not missing_terms
    return result


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
    answer: str,
    citations: Sequence[str],
    entity_clusters: Sequence[Mapping[str, Any]],
    evidence: Sequence[EvidenceRecord],
    *,
    query_requirements: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not str(answer or "").strip():
        return {"passed": False, "reason": "answer_missing", "missing_segment_ids": []}
    if not _citations_are_visual(citations, evidence):
        return {"passed": False, "reason": "invalid_visual_citations", "missing_segment_ids": []}
    by_id = {record.evidence_id: record for record in evidence}
    cited = tuple(by_id[str(citation)] for citation in citations)
    identity_gate = _identity_link_gate(
        entity_clusters,
        cited,
        query_requirements=query_requirements,
    )
    if identity_gate is not None:
        return identity_gate
    if contract.required_scope != "full_video":
        return {"passed": True, "reason": "verified_window_evidence", "missing_segment_ids": []}

    cited_sources = {
        str(lineage.get("source_video_id", "") or "")
        for record in cited
        for lineage in record.source_lineage
        if str(lineage.get("source_video_id", "") or "")
    }
    if not cited_sources:
        return {"passed": False, "reason": "source_not_identified", "missing_segment_ids": []}
    source_coverage = _source_coverage(workspace, evidence)
    missing = sorted(
        {
            segment_id
            for source_id in cited_sources
            for segment_id in source_coverage.get(source_id, {}).get("missing_segment_ids", ())
        }
    )
    if missing:
        return {
            "passed": False,
            "reason": "full_source_coverage_missing",
            "source_video_ids": sorted(cited_sources),
            "missing_segment_ids": missing,
        }
    if contract.quantifier == "distinct_count":
        clusters = tuple(_entity_cluster(item) for item in entity_clusters)
        if not clusters:
            return {"passed": False, "reason": "entity_reconciliation_missing", "missing_segment_ids": []}
        entity_ids = tuple(cluster["entity_id"] for cluster in clusters)
        if any(not entity_id for entity_id in entity_ids) or len(set(entity_ids)) != len(entity_ids):
            return {"passed": False, "reason": "entity_cluster_ids_invalid", "missing_segment_ids": []}
        cited_ids = {str(citation) for citation in citations}
        if any(not cluster["evidence_ids"] or not set(cluster["evidence_ids"]).issubset(cited_ids) for cluster in clusters):
            return {"passed": False, "reason": "entity_cluster_evidence_invalid", "missing_segment_ids": []}
        expected_count = _answer_count(answer, workspace.case.options)
        if expected_count is not None and expected_count != len(clusters):
            return {
                "passed": False,
                "reason": "entity_count_answer_mismatch",
                "expected_count": expected_count,
                "cluster_count": len(clusters),
                "missing_segment_ids": [],
            }
    return {
        "passed": True,
        "reason": "full_source_coverage_verified",
        "source_video_ids": sorted(cited_sources),
        "entity_cluster_count": len(entity_clusters),
        "missing_segment_ids": [],
    }


def _identity_link_gate(
    entity_clusters: Sequence[Mapping[str, Any]],
    cited: Sequence[EvidenceRecord],
    *,
    query_requirements: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    terms = tuple(str(item) for item in (query_requirements or {}).get("identity_anchor_terms", ()) if str(item))
    if not terms:
        return None
    by_id = {record.evidence_id: record for record in cited}
    for cluster in (_entity_cluster(item) for item in entity_clusters):
        cluster_records = tuple(by_id[evidence_id] for evidence_id in cluster["evidence_ids"] if evidence_id in by_id)
        if not cluster_records:
            continue
        combined = " ".join(_evidence_search_text(record) for record in cluster_records)
        has_anchor = any(_record_matches_identity_anchor(record, terms) for record in cluster_records) or all(
            term in combined for term in terms
        )
        has_answer_event = any(_record_supports_answer_event(record) for record in cluster_records)
        if has_anchor and has_answer_event:
            return None
    return {
        "passed": False,
        "reason": "identity_anchor_not_linked",
        "identity_anchor_terms": list(terms),
        "missing_segment_ids": [],
    }


def _evidence_search_text(record: EvidenceRecord) -> str:
    entity_text = " ".join(
        str(item.get("description", "") or "")
        for item in record.operation_metadata.get("entities", ())
        if isinstance(item, Mapping)
    )
    return f"{record.verbatim} {entity_text}".casefold()


def _record_matches_identity_anchor(record: EvidenceRecord, terms: Sequence[str]) -> bool:
    if "supports_identity_anchor" in record.operation_metadata:
        return _metadata_flag(record.operation_metadata.get("supports_identity_anchor"))
    text = _evidence_search_text(record)
    return bool(terms) and all(term in text for term in terms)


def _record_supports_answer_event(record: EvidenceRecord) -> bool:
    return record.evidence_kind == "event_observation" or _metadata_flag(
        record.operation_metadata.get("supports_answer_event")
    )


def _metadata_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes"}


def _derived_answer_evidence(
    workspace: VirtualVideoWorkspace,
    *,
    answer: str,
    citations: Sequence[str],
    entity_clusters: Sequence[Mapping[str, Any]],
    evidence: Sequence[EvidenceRecord],
    coverage_source_ids: Sequence[str] = (),
) -> EvidenceRecord:
    by_id = {record.evidence_id: record for record in evidence}
    parents = [by_id[str(citation)] for citation in citations]
    known_ids = {record.evidence_id for record in parents}
    coverage_sources = {str(item) for item in coverage_source_ids if str(item)}
    for record in evidence:
        record_sources = {
            str(lineage.get("source_video_id", "") or "")
            for lineage in record.source_lineage
            if str(lineage.get("source_video_id", "") or "")
        }
        if (
            coverage_sources.intersection(record_sources)
            and record.modality in {"visual", "ocr"}
            and record.evidence_id not in known_ids
        ):
            parents.append(record)
            known_ids.add(record.evidence_id)
    parent_records = tuple(parents)
    starts = [record.start_sec for record in parent_records if record.start_sec is not None]
    ends = [record.end_sec for record in parent_records if record.end_sec is not None]
    coverage = tuple(segment for record in parent_records for segment in record.coverage_manifest)
    lineage = tuple(dict(item) for record in parent_records for item in record.source_lineage)
    request_ids = tuple(dict.fromkeys(request_id for record in parent_records for request_id in record.request_ids))
    clusters = tuple(_entity_cluster(item) for item in entity_clusters)
    return EvidenceRecord(
        evidence_id="ev_final_aggregate",
        beat_id="",
        start_sec=min(starts) if starts else None,
        end_sec=max(ends) if ends else None,
        modality="derived",
        pointer=f"virtual://{workspace.workspace_id}/derived/final",
        verbatim=f"Final answer {answer!r} aggregates {len(parent_records)} supporting observations.",
        claim=answer,
        attestation_model="reasoner",
        temporal_scope="full_video",
        evidence_kind="aggregate",
        observation_polarity="positive",
        sampling_coverage="sparse",
        parent_evidence_ids=tuple(record.evidence_id for record in parent_records),
        request_ids=request_ids,
        coverage_manifest=coverage,
        task_id="final_answer",
        observation_id="final_aggregate",
        confidence=min((record.confidence for record in parent_records), default=0.0),
        source_lineage=lineage,
        entity_ids=tuple(cluster["entity_id"] for cluster in clusters),
        operation_metadata={"algorithm": "reasoner_reconciliation", "entity_clusters": clusters},
    )


def _answer_count(answer: str, options: Mapping[str, str]) -> int | None:
    selected = _letter(answer)
    text = " ".join((str(answer or ""), str(options.get(selected, "") or ""))).casefold()
    match = re.search(r"\b(\d+)\b", text)
    if match:
        return int(match.group(1))
    words = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    for word, value in words.items():
        if re.search(rf"\b{word}\b", text):
            return value
    return None


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


def _entity_cluster(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    return {
        "entity_id": str(row.get("entity_id", "") or ""),
        "description": str(row.get("description", "") or ""),
        "evidence_ids": tuple(str(item) for item in row.get("evidence_ids", ()) if str(item).strip()),
    }


def _decision(value: ReasonerDecision | Mapping[str, Any]) -> ReasonerDecision:
    if isinstance(value, ReasonerDecision):
        return value
    return ReasonerDecision(
        action=str(value.get("action", "")),
        tasks=tuple(value.get("tasks", ())),
        answer=str(value.get("answer", "")),
        citations=tuple(value.get("citations", ())),
        entity_clusters=tuple(value.get("entity_clusters", ())),
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
            "entities": list(item.operation_metadata.get("entities", ())),
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
                        "entity_ids": list(item.entity_ids),
                        "operation_metadata": to_jsonable(item.operation_metadata),
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
