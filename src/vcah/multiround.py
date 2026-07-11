from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import re
from typing import Any, Mapping, Sequence

from vcah.evidence_primitives import GapCondition, MeasurementFact, canonical_unit, make_gap_conditions
from vcah.investigator import InvestigationReport, VirtualVideoInvestigator
from vcah.memory import EvidenceStore
from vcah.types import ClaimContract, EvidenceRecord, is_path_only_visual_evidence, to_jsonable
from vcah.virtual_index import build_workspace_overview
from vcah.virtual_video import VirtualVideoWorkspace


@dataclass(frozen=True)
class EvidenceGap:
    gap_id: str
    description: str
    success_conditions: tuple[str, ...] = ()
    falsification_conditions: tuple[str, ...] = ()
    conditions: tuple[GapCondition, ...] = ()
    importance: str = "critical"
    status: str = "open"

    def __post_init__(self) -> None:
        object.__setattr__(self, "gap_id", str(self.gap_id or "").strip())
        object.__setattr__(self, "description", str(self.description or "").strip())
        object.__setattr__(
            self,
            "success_conditions",
            tuple(str(item).strip() for item in self.success_conditions if str(item).strip()),
        )
        object.__setattr__(
            self,
            "falsification_conditions",
            tuple(str(item).strip() for item in self.falsification_conditions if str(item).strip()),
        )
        conditions = tuple(_gap_condition(item) for item in self.conditions)
        if not conditions:
            conditions = make_gap_conditions(self.gap_id, self.success_conditions)
        object.__setattr__(self, "conditions", conditions)
        importance = str(self.importance or "critical").strip().casefold()
        object.__setattr__(self, "importance", importance if importance in {"critical", "useful", "optional"} else "critical")
        status = str(self.status or "open").strip().casefold()
        object.__setattr__(self, "status", status if status in {"open", "partial", "resolved", "abandoned"} else "open")


@dataclass(frozen=True)
class InvestigationTask:
    query_id: str
    goal: str
    segment_id: str = ""
    time_range: tuple[float, float] | None = None
    modality_hint: tuple[str, ...] = ()
    expected_evidence: str = ""
    inspection_mode: str = "window"
    priority: float = 0.0
    claim_to_verify: str = ""
    claim_relation: str = ""
    alternative_answers: tuple[str, ...] = ()
    search_terms: tuple[str, ...] = ()
    gap_id: str = ""
    success_conditions: tuple[str, ...] = ()
    direction: str = ""
    preferred_ranges: tuple[tuple[float, float], ...] = ()
    excluded_ranges: tuple[tuple[float, float], ...] = ()
    region_hint: str = ""
    conditions: tuple[GapCondition, ...] = ()

    def __post_init__(self) -> None:
        if self.time_range is not None:
            start, end = self.time_range
            object.__setattr__(self, "time_range", (float(start), float(end)))
        object.__setattr__(self, "segment_id", str(self.segment_id or ""))
        object.__setattr__(self, "modality_hint", tuple(str(item) for item in self.modality_hint))
        object.__setattr__(self, "claim_to_verify", str(self.claim_to_verify or "").strip())
        object.__setattr__(self, "claim_relation", str(self.claim_relation or "").strip().casefold())
        object.__setattr__(self, "alternative_answers", tuple(str(item) for item in self.alternative_answers))
        object.__setattr__(self, "gap_id", str(self.gap_id or "").strip())
        object.__setattr__(
            self,
            "success_conditions",
            tuple(str(item).strip() for item in self.success_conditions if str(item).strip()),
        )
        object.__setattr__(self, "direction", str(self.direction or "").strip().casefold())
        object.__setattr__(self, "preferred_ranges", _time_ranges(self.preferred_ranges))
        object.__setattr__(self, "excluded_ranges", _time_ranges(self.excluded_ranges))
        object.__setattr__(self, "region_hint", str(self.region_hint or "").strip())
        conditions = tuple(_gap_condition(item) for item in self.conditions)
        if not conditions and self.success_conditions:
            conditions = make_gap_conditions(self.gap_id or self.query_id, self.success_conditions)
        object.__setattr__(self, "conditions", conditions)
        object.__setattr__(
            self,
            "search_terms",
            tuple(dict.fromkeys(str(item).strip().casefold() for item in self.search_terms if str(item).strip())),
        )
        if self.inspection_mode not in {"window", "enumerate_events", "event_window", "verify_claim", "search_asr"}:
            object.__setattr__(self, "inspection_mode", "window")


@dataclass(frozen=True)
class ReasonerDecision:
    action: str
    tasks: tuple[InvestigationTask, ...] = ()
    answer: str = ""
    citations: tuple[str, ...] = ()
    entity_clusters: tuple[Mapping[str, Any], ...] = ()
    support_status: str = ""
    support_reason: str = ""
    primary_gap: EvidenceGap | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tasks", tuple(_task(item) for item in self.tasks))
        object.__setattr__(self, "citations", tuple(str(item) for item in self.citations if str(item).strip()))
        object.__setattr__(self, "entity_clusters", tuple(_entity_cluster(item) for item in self.entity_clusters))
        object.__setattr__(self, "support_status", str(self.support_status or "").strip().casefold())
        object.__setattr__(self, "support_reason", str(self.support_reason or "").strip())
        if isinstance(self.primary_gap, Mapping):
            object.__setattr__(self, "primary_gap", _gap(self.primary_gap))
        elif self.primary_gap is not None and not isinstance(self.primary_gap, EvidenceGap):
            object.__setattr__(self, "primary_gap", None)


@dataclass(frozen=True)
class MultiRoundResult:
    case_id: str
    answer: str
    citations: tuple[str, ...]
    correct: bool
    verified: bool
    verification_reason: str
    rounds: int
    accepted_investigations: int
    evidence: tuple[EvidenceRecord, ...]
    reports: tuple[InvestigationReport, ...]
    trace: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    grounded_answer: str = ""
    forced_answer: str = ""
    selected_option: str = ""
    answer_mode: str = "insufficient"
    grounding_status: str = "insufficient"
    retrieval_status: str = "failed"


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


def compile_query_contract(
    question: str,
    options: Mapping[str, str] | None = None,
) -> ClaimContract:
    text = str(question or "").casefold()
    is_count = bool(re.search(r"\bhow many\b|\bnumber of\b", text))
    is_occurrence_count = bool(re.search(r"\bhow many times\b|\bnumber of times\b", text))
    measurement_text = " ".join((text, *(str(item).casefold() for item in (options or {}).values())))
    asked_measurement_unit = _asked_measurement_unit(text)
    measurement_unit = asked_measurement_unit or _measurement_unit(measurement_text)
    boundary_hint = _boundary_hint(question)
    full_video = "in total" in text or bool(
        re.search(r"\b(?:throughout|across)\b.*\b(?:video|film|recording)\b", text)
        or re.search(r"\b(?:entire|whole)\s+(?:video|film|recording)\b", text)
        or re.search(r"\bin\s+(?:this|the)\s+(?:video|film|recording)\b", text)
        or re.search(r"\bover the course of\s+(?:this|the)\s+(?:video|film|recording)\b", text)
    )
    language_action = any(term in text for term in ("comment", "say", "speak", "discuss", "mention"))
    identity_anchor_terms = _identity_anchor_terms(question)
    if measurement_unit and not is_occurrence_count and (
        bool(asked_measurement_unit) or "how much" in text or "what total" in text
    ):
        cumulative = bool(boundary_hint) or any(term in text for term in ("total", "consumed", "accumulated", "altogether"))
        return ClaimContract(
            required_scope="multi_window" if cumulative else "window",
            quantifier="scalar_quantity",
            observation_target="attribute",
            aggregation="accumulate" if cumulative else "compare",
            required_observability=("visual", "asr"),
            observability_mode="any",
            measurement_unit=measurement_unit,
            boundary_hint=boundary_hint,
        )
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


def _measurement_unit(text: str) -> str:
    patterns = (
        (r"\bcal(?:orie)?s?\b", "calorie"),
        (r"\b(?:usd|dollars?|pounds?|euros?)\b", "dollar"),
        (r"\bminutes?\b", "minute"),
        (r"\b(?:meters?|metres?)\b", "meter"),
        (r"\b(?:kilometers?|kilometres?)\b", "kilometer"),
        (r"\bkilograms?\b|\bkg\b", "kilogram"),
        (r"\b(?:light[ -]?years?|ly)\b", "light_year"),
        (r"\byears?\b", "year"),
        (r"\bpoints?\b", "point"),
    )
    normalized = str(text or "").casefold()
    anchor = re.search(r"\b(?:how many|how much|what total)\b", normalized)
    candidates = []
    for order, (pattern, unit) in enumerate(patterns):
        for match in re.finditer(pattern, normalized):
            if anchor is None:
                distance = match.start()
            elif match.start() >= anchor.end():
                distance = match.start() - anchor.end()
            else:
                distance = len(normalized) + anchor.start() - match.start()
            candidates.append((distance, match.start(), order, canonical_unit(unit)))
    return min(candidates)[-1] if candidates else ""


def _asked_measurement_unit(text: str) -> str:
    anchor = re.search(r"\b(?:how many|number of)\b", str(text or "").casefold())
    if anchor is None:
        return ""
    tokens = re.findall(r"[a-z]+(?:-[a-z]+)?", str(text or "").casefold()[anchor.end() :])[:3]
    return _measurement_unit(" ".join(tokens)) if tokens else ""


def _boundary_hint(question: str) -> str:
    text = " ".join(str(question or "").split())
    match = re.search(
        r"\b(?:before|when|until|by the time|at the time)\b\s+([^?.,;]+)",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(0).strip() if match else ""


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
        query_contract = compile_query_contract(workspace.case.question, workspace.case.options)
        query_requirements = compile_query_requirements(workspace.case.question)
        temporal_navigation = source_time_navigation(workspace, compile_source_time_hint(workspace.case.question))
        evidence_store = EvidenceStore.empty(workspace.root_dir / "evidence.jsonl")
        reports: list[InvestigationReport] = []
        trace: list[Mapping[str, Any]] = []
        accepted = 0
        answer = ""
        citations: tuple[str, ...] = ()
        verified = False
        verification_reason = ""
        best_answer = ""
        best_citations: tuple[str, ...] = ()
        best_verification_reason = ""
        best_support_rank = -1
        last_gate_reason = "answer_missing"
        gate_feedback: dict[str, Any] = {}
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
            stagnation_status = _stagnation_status(reports)
            decision = _bind_gap_to_tasks(_decision(
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
                    available_navigation=tuple(workspace_overview.get("available_navigation", ())),
                    evidence=evidence_store.records,
                    evidence_digest=_evidence_digest(evidence_store.records),
                    investigation_outcomes=_outcome_digest(reports),
                    stagnation_status=stagnation_status,
                    answer_gate_feedback=gate_feedback,
                    remaining_budget=remaining,
                )
            ))
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
                repair_tasks = _navigation_repair_tasks(
                    evidence_store.records,
                    round_id=round_id,
                    limit=min(self.max_tasks_per_round, remaining),
                )
                repair_reason = "unverified_navigation_hint"
                if not repair_tasks:
                    repair_tasks = _identity_repair_tasks(
                        workspace,
                        evidence_store.records,
                        missing_identity_terms,
                        round_id=round_id,
                        limit=min(self.max_tasks_per_round, remaining),
                    )
                    repair_reason = "identity_anchor_missing"
                if repair_tasks:
                    trace.append(
                        {
                            "type": "repair_override",
                            "round": round_id,
                            "reason": repair_reason,
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
                    "primary_gap": to_jsonable(decision.primary_gap) if decision.primary_gap else None,
                    "stagnation_status": stagnation_status,
                }
            )
            if decision.action == "answer":
                filtered_citations = _answer_citations(decision.citations, evidence_store.records)
                if filtered_citations != decision.citations:
                    trace.append(
                        {
                            "type": "citation_filter",
                            "round": round_id,
                            "removed_citations": [
                                citation for citation in decision.citations if citation not in filtered_citations
                            ],
                        }
                    )
                    decision = replace(decision, citations=filtered_citations)
                gate = _answer_completion_gate(
                    workspace,
                    query_contract,
                    decision.answer,
                    decision.citations,
                    decision.entity_clusters,
                    evidence_store.records,
                    query_requirements=query_requirements,
                )
                gate = _apply_answer_audit(gate, decision)
                trace.append({"type": "completion_gate", "round": round_id, **gate})
                last_gate_reason = str(gate.get("reason", "") or "verification_failed")
                gate_feedback = dict(gate)
                support_rank = _answer_support_rank(decision)
                if decision.answer.strip() and support_rank >= best_support_rank:
                    best_answer = decision.answer
                    best_citations = decision.citations
                    best_verification_reason = last_gate_reason
                    best_support_rank = support_rank
                if gate["passed"]:
                    answer = decision.answer
                    verified = True
                    verification_reason = last_gate_reason
                    if query_contract.aggregation != "none":
                        aggregate = _derived_answer_evidence(
                            workspace,
                            answer=answer,
                            citations=decision.citations,
                            entity_clusters=decision.entity_clusters,
                            evidence=evidence_store.records,
                            coverage_source_ids=gate.get("source_video_ids", ()),
                            derivation=gate.get("derivation"),
                        )
                        evidence_store.add(aggregate)
                        citations = (aggregate.evidence_id,)
                    else:
                        citations = decision.citations
                    break
                continue
            if remaining <= 0:
                break
            requested_tasks = tuple(
                _task_for_contract(task, query_contract)
                for task in decision.tasks[: min(self.max_tasks_per_round, remaining)]
            )
            tasks = _prefer_navigation_repairs(
                requested_tasks,
                evidence_store.records,
                round_id=round_id,
                limit=min(self.max_tasks_per_round, remaining),
            )
            if tasks != requested_tasks:
                trace.append(
                    {
                        "type": "navigation_drilldown_override",
                        "round": round_id,
                        "requested_search_tasks": sum(
                            task.inspection_mode == "search_asr" for task in requested_tasks
                        ),
                        "visual_repair_tasks": sum(
                            task.query_id.startswith("navigation_repair_") for task in tasks
                        ),
                    }
                )
            accepted += len(tasks)
            batch = _annotate_batch_progress(investigator.run_batch(tasks), reports)
            reports.extend(batch)
            known_evidence = {record.evidence_id for record in evidence_store.records}
            for report in batch:
                for record in report.evidence:
                    if record.evidence_id not in known_evidence:
                        evidence_store.add(record)
                        known_evidence.add(record.evidence_id)
            trace.append({"type": "investigator_batch", "round": round_id, "accepted_tasks": len(tasks)})
            trace.append(
                {
                    "type": "investigation_outcomes",
                    "round": round_id,
                    "outcomes": list(_outcome_digest(batch)),
                }
            )
            if accepted >= self.max_investigations:
                continue

        if not answer and evidence_store.records:
            completion_status = _completion_status(
                workspace,
                query_contract,
                evidence_store.records,
                query_requirements=query_requirements,
            )
            final_decision = _bind_gap_to_tasks(_decision(
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
                    available_navigation=tuple(workspace_overview.get("available_navigation", ())),
                    evidence=evidence_store.records,
                    evidence_digest=_evidence_digest(evidence_store.records),
                    investigation_outcomes=_outcome_digest(reports),
                    stagnation_status=_stagnation_status(reports),
                    answer_gate_feedback=gate_feedback,
                    remaining_budget=0,
                    force_finalize=True,
                )
            ))
            filtered_citations = _answer_citations(final_decision.citations, evidence_store.records)
            if filtered_citations != final_decision.citations:
                trace.append(
                    {
                        "type": "citation_filter",
                        "round": self.max_rounds + 1,
                        "removed_citations": [
                            citation for citation in final_decision.citations if citation not in filtered_citations
                        ],
                    }
                )
                final_decision = replace(final_decision, citations=filtered_citations)
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
                gate = _apply_answer_audit(gate, final_decision)
                trace.append(
                    {
                        "type": "completion_gate",
                        "round": self.max_rounds + 1,
                        "finalization": True,
                        **gate,
                    }
                )
                last_gate_reason = str(gate.get("reason", "") or "verification_failed")
                support_rank = _answer_support_rank(final_decision)
                if final_decision.answer.strip() and support_rank >= best_support_rank:
                    best_answer = final_decision.answer
                    best_citations = final_decision.citations
                    best_verification_reason = last_gate_reason
                    best_support_rank = support_rank
                if gate["passed"]:
                    answer = final_decision.answer
                    verified = True
                    verification_reason = last_gate_reason
                    if query_contract.aggregation != "none":
                        aggregate = _derived_answer_evidence(
                            workspace,
                            answer=answer,
                            citations=final_decision.citations,
                            entity_clusters=final_decision.entity_clusters,
                            evidence=evidence_store.records,
                            coverage_source_ids=gate.get("source_video_ids", ()),
                            derivation=gate.get("derivation"),
                        )
                        evidence_store.add(aggregate)
                        citations = (aggregate.evidence_id,)
                    else:
                        citations = final_decision.citations

        if verified:
            final_answer = answer
            final_citations = citations
            grounded_answer = answer
            forced_answer = answer
            answer_mode = "grounded"
            grounding_status = "verified"
        elif best_answer:
            final_answer = best_answer
            final_citations = best_citations
            verification_reason = best_verification_reason or last_gate_reason
            grounded_answer = ""
            forced_answer = best_answer
            answer_mode = "forced_choice"
            grounding_status = "insufficient"
        else:
            final_answer = "Insufficient verified evidence."
            final_citations = ()
            verification_reason = last_gate_reason or "answer_missing"
            grounded_answer = ""
            forced_answer = ""
            answer_mode = "insufficient"
            grounding_status = "insufficient"
        selected_option = _letter(final_answer) or _option_letter_from_answer(final_answer, workspace.case.options)
        retrieval_status = _retrieval_status(reports, verified=verified)
        trace.append(
            {
                "type": "answer_outcome",
                "answer": final_answer,
                "grounded_answer": grounded_answer,
                "forced_answer": forced_answer,
                "selected_option": selected_option,
                "answer_mode": answer_mode,
                "grounding_status": grounding_status,
                "retrieval_status": retrieval_status,
                "verified": verified,
                "verification_reason": verification_reason,
                "best_effort": bool(final_answer and not verified and best_answer),
            }
        )
        result = MultiRoundResult(
            case_id=workspace.case.case_id,
            answer=final_answer,
            citations=final_citations,
            correct=_score_answer(final_answer, workspace.case.gold, workspace.case.options),
            verified=verified,
            verification_reason=verification_reason,
            rounds=rounds_run,
            accepted_investigations=accepted,
            evidence=tuple(evidence_store.records),
            reports=tuple(reports),
            trace=tuple(trace),
            grounded_answer=grounded_answer,
            forced_answer=forced_answer,
            selected_option=selected_option,
            answer_mode=answer_mode,
            grounding_status=grounding_status,
            retrieval_status=retrieval_status,
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


def _navigation_repair_tasks(
    evidence: Sequence[EvidenceRecord],
    *,
    round_id: int,
    limit: int,
) -> tuple[InvestigationTask, ...]:
    visual = tuple(record for record in evidence if record.modality in {"visual", "ocr"})
    grouped: dict[str, list[EvidenceRecord]] = {}
    for record in evidence:
        if (
            record.evidence_kind != "navigation_hint"
            or record.observation_polarity != "positive"
            or record.start_sec is None
            or record.end_sec is None
        ):
            continue
        matched_terms = tuple(
            dict.fromkeys(str(item).strip().casefold() for item in record.operation_metadata.get("matched_terms", ()) if str(item).strip())
        )
        group_id = record.task_id or record.evidence_id
        grouped.setdefault(f"{group_id}:{'|'.join(matched_terms)}", []).append(record)

    unresolved = []
    for hints in grouped.values():
        if any(_navigation_hint_is_observed(hint, visual) for hint in hints):
            continue
        hint = hints[0]
        lineage = next(
            (
                item
                for item in hint.source_lineage
                if str(item.get("segment_id", "") or "")
            ),
            None,
        )
        if lineage is not None:
            unresolved.append((hint, str(lineage["segment_id"])))

    return tuple(
        InvestigationTask(
            query_id=f"navigation_repair_r{round_id}_{index:03d}",
            goal="Visually verify an unresolved ASR navigation clue without assuming it proves an answer option.",
            segment_id=segment_id,
            time_range=(float(hint.start_sec), float(hint.end_sec)),
            modality_hint=("visual",),
            expected_evidence=f"complete temporal context and direct visual evidence for or against: {hint.verbatim[:240]}",
            priority=1.0,
        )
        for index, (hint, segment_id) in enumerate(unresolved[: max(0, int(limit))], start=1)
    )


def _prefer_navigation_repairs(
    requested: Sequence[InvestigationTask],
    evidence: Sequence[EvidenceRecord],
    *,
    round_id: int,
    limit: int,
) -> tuple[InvestigationTask, ...]:
    bounded = tuple(requested[: max(0, int(limit))])
    if not any(task.inspection_mode == "search_asr" for task in bounded):
        return bounded
    kept = tuple(task for task in bounded if task.inspection_mode != "search_asr")
    repairs = _navigation_repair_tasks(
        evidence,
        round_id=round_id,
        limit=max(0, int(limit) - len(kept)),
    )
    if not repairs:
        return bounded
    return (*kept, *repairs)


def _navigation_hint_is_observed(hint: EvidenceRecord, visual: Sequence[EvidenceRecord]) -> bool:
    if hint.start_sec is None or hint.end_sec is None:
        return False
    hint_sources = {
        str(item.get("source_video_id", "") or "")
        for item in hint.source_lineage
        if str(item.get("source_video_id", "") or "")
    }
    for record in visual:
        if record.start_sec is None or record.end_sec is None:
            continue
        record_sources = {
            str(item.get("source_video_id", "") or "")
            for item in record.source_lineage
            if str(item.get("source_video_id", "") or "")
        }
        if hint_sources and record_sources and hint_sources.isdisjoint(record_sources):
            continue
        if min(float(hint.end_sec), float(record.end_sec)) > max(float(hint.start_sec), float(record.start_sec)):
            return True
    return False


def _task_for_contract(task: InvestigationTask, contract: ClaimContract) -> InvestigationTask:
    if (
        task.inspection_mode == "window"
        and contract.quantifier == "total_count"
        and contract.observation_target == "event"
    ):
        return replace(task, inspection_mode="enumerate_events")
    return task


def _completion_status(
    workspace: VirtualVideoWorkspace,
    contract: ClaimContract,
    evidence: Sequence[EvidenceRecord],
    *,
    query_requirements: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    answer_evidence = tuple(record for record in evidence if record.evidence_kind != "navigation_hint")
    coverage = _source_coverage(workspace, answer_evidence)
    if contract.required_scope != "full_video":
        return _apply_identity_completion(
            {
                "ready_for_answer": bool(answer_evidence),
                "required_scope": contract.required_scope,
                "missing_segment_ids": [],
                "source_coverage": coverage,
            },
            answer_evidence,
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
            answer_evidence,
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
        answer_evidence,
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
    if not (_letter(answer) or _option_letter_from_answer(answer, workspace.case.options)):
        return {"passed": False, "reason": "invalid_option_answer", "missing_segment_ids": []}
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
    quantitative_gate = _quantitative_answer_gate(workspace, contract, answer, cited)
    if quantitative_gate is not None:
        return quantitative_gate
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
    event_occurrences: tuple[dict[str, Any], ...] = ()
    if contract.quantifier == "total_count":
        event_occurrences = _event_occurrences(cited)
        if not event_occurrences:
            return {"passed": False, "reason": "event_occurrences_missing", "missing_segment_ids": []}
        expected_count = _answer_count(answer, workspace.case.options)
        if expected_count is not None and expected_count != len(event_occurrences):
            return {
                "passed": False,
                "reason": "event_count_answer_mismatch",
                "expected_count": expected_count,
                "event_occurrence_count": len(event_occurrences),
                "missing_segment_ids": [],
            }
    return {
        "passed": True,
        "reason": "full_source_coverage_verified",
        "source_video_ids": sorted(cited_sources),
        "entity_cluster_count": len(entity_clusters),
        "event_occurrence_count": len(event_occurrences),
        "missing_segment_ids": [],
    }


def _quantitative_answer_gate(
    workspace: VirtualVideoWorkspace,
    contract: ClaimContract,
    answer: str,
    cited: Sequence[EvidenceRecord],
) -> dict[str, Any] | None:
    if contract.quantifier in {"distinct_count", "total_count"}:
        return None
    selected = _letter(answer) or _option_letter_from_answer(answer, workspace.case.options)
    option_text = str(workspace.case.options.get(selected, "") or answer)
    if contract.quantifier == "scalar_quantity":
        return _scalar_quantity_gate(contract, option_text, cited)
    atoms = _quantitative_atoms(option_text)
    if not atoms:
        return None
    primitive_records = tuple(
        record
        for record in cited
        if "target_presence" in record.operation_metadata or "measurements" in record.operation_metadata
    )
    if primitive_records:
        present_records = tuple(
            record
            for record in primitive_records
            if str(
                dict(record.operation_metadata.get("target_presence", {}) or {}).get("status", "")
            ).casefold()
            == "present"
        )
        if not present_records:
            return {
                "passed": False,
                "reason": "quantitative_target_not_present",
                "missing_numeric_atoms": list(atoms),
                "missing_segment_ids": [],
            }
        evidence_text = " ".join(
            " ".join(
                (
                    str(row.get("raw_text", "") or ""),
                    str(row.get("value", "") or ""),
                    str(row.get("unit", "") or ""),
                )
            )
            for record in present_records
            for row in record.operation_metadata.get("measurements", ()) or ()
            if isinstance(row, Mapping)
        ).casefold()
    else:
        evidence_text = " ".join(
            [record.verbatim for record in cited]
            + [
                str(event.get("description", "") or "")
                for record in cited
                for event in record.operation_metadata.get("events", ())
                if isinstance(event, Mapping)
            ]
            + [
                json.dumps(record.operation_metadata.get("derivation"), ensure_ascii=False)
                for record in cited
                if record.operation_metadata.get("derivation")
            ]
        ).casefold()
    evidence_times = set(re.findall(r"\b\d{1,2}:\d{2}\b", evidence_text))
    evidence_numbers = {
        token.replace(",", "")
        for token in re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", evidence_text)
    }
    missing = [
        atom
        for atom in atoms
        if (":" in atom and atom not in evidence_times) or (":" not in atom and atom not in evidence_numbers)
    ]
    if not missing:
        return None
    return {
        "passed": False,
        "reason": "quantitative_answer_not_grounded",
        "missing_numeric_atoms": missing,
        "missing_segment_ids": [],
    }


def _scalar_quantity_gate(
    contract: ClaimContract,
    option_text: str,
    cited: Sequence[EvidenceRecord],
) -> dict[str, Any]:
    target = _quantity_value(option_text)
    if target is None:
        return {"passed": False, "reason": "scalar_option_value_missing", "missing_segment_ids": []}
    unit = canonical_unit(contract.measurement_unit)
    facts = tuple(
        fact
        for fact in _measurement_facts(cited)
        if fact.unit == unit and (not contract.boundary_hint or fact.boundary_relation in {"before", "at"})
    )
    if not facts:
        return {
            "passed": False,
            "reason": "scalar_measurement_evidence_missing",
            "measurement_unit": unit,
            "missing_segment_ids": [],
        }

    cumulative = tuple(fact for fact in facts if fact.semantics == "cumulative")
    if cumulative:
        fact = max(
            cumulative,
            key=lambda item: (
                item.boundary_relation == "at",
                item.source_time_sec if item.source_time_sec is not None else float("-inf"),
            ),
        )
        observed = fact.value
        derivation = {
            "operator": "read_cumulative",
            "operands": [fact.value],
            "result": observed,
            "unit": unit,
            "boundary_hint": contract.boundary_hint,
            "evidence_ids": list(fact.evidence_ids),
        }
    else:
        deltas = _deduplicated_delta_measurements(facts)
        if not deltas:
            return {
                "passed": False,
                "reason": "scalar_measurement_semantics_missing",
                "measurement_unit": unit,
                "missing_segment_ids": [],
            }
        observed = sum(fact.value for fact in deltas)
        derivation = {
            "operator": "sum_delta",
            "operands": [fact.value for fact in deltas],
            "result": observed,
            "unit": unit,
            "boundary_hint": contract.boundary_hint,
            "evidence_ids": list(
                dict.fromkeys(evidence_id for fact in deltas for evidence_id in fact.evidence_ids)
            ),
        }
    relation = _quantity_relation(option_text)
    if _quantity_entails(observed, target, relation):
        return {
            "passed": True,
            "reason": "scalar_quantity_grounded",
            "derivation": derivation,
            "expected_value": target,
            "expected_relation": relation,
            "missing_segment_ids": [],
        }
    return {
        "passed": False,
        "reason": "scalar_quantity_answer_mismatch",
        "derivation": derivation,
        "expected_value": target,
        "expected_relation": relation,
        "observed_value": observed,
        "missing_segment_ids": [],
    }


def _measurement_facts(cited: Sequence[EvidenceRecord]) -> tuple[MeasurementFact, ...]:
    facts = []
    for record in cited:
        for row in record.operation_metadata.get("measurements", ()) or ():
            if not isinstance(row, Mapping):
                continue
            try:
                facts.append(
                    MeasurementFact(
                        value=float(row.get("value")),
                        unit=str(row.get("unit", "") or ""),
                        relation=str(row.get("relation", "exact") or "exact"),
                        semantics=str(row.get("measurement_semantics", row.get("semantics", "unknown")) or "unknown"),
                        subject_id=str(row.get("subject_id", "") or ""),
                        source_time_sec=row.get("source_time_sec"),
                        boundary_relation=str(row.get("boundary_relation", "unknown") or "unknown"),
                        raw_text=str(row.get("raw_text", "") or ""),
                        evidence_ids=tuple(row.get("evidence_ids", ()) or ()) or (record.evidence_id,),
                    )
                )
            except (TypeError, ValueError):
                continue
    return tuple(facts)


def _deduplicated_delta_measurements(facts: Sequence[MeasurementFact]) -> tuple[MeasurementFact, ...]:
    result = []
    seen = set()
    for fact in facts:
        if fact.semantics != "delta" or fact.relation not in {"exact", "approx"}:
            continue
        key = (
            fact.subject_id or "|".join(fact.evidence_ids),
            fact.source_time_sec,
            fact.value,
            fact.unit,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(fact)
    return tuple(result)


def _quantity_value(text: str) -> float | None:
    match = re.search(
        r"(?<!:)\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*(thousand|million|billion|trillion)?\b",
        str(text or "").casefold(),
    )
    if not match:
        return None
    scale = {
        "": 1.0,
        "thousand": 1e3,
        "million": 1e6,
        "billion": 1e9,
        "trillion": 1e12,
    }[str(match.group(2) or "")]
    return float(match.group(1).replace(",", "")) * scale


def _quantity_relation(text: str) -> str:
    normalized = str(text or "").casefold()
    if any(term in normalized for term in ("more than", "greater than", "over ", "above ")):
        return "greater_than"
    if any(term in normalized for term in ("less than", "under ", "below ")):
        return "less_than"
    return "exact"


def _quantity_entails(observed: float, expected: float, relation: str) -> bool:
    if relation == "greater_than":
        return observed > expected
    if relation == "less_than":
        return observed < expected
    tolerance = max(1e-6, abs(expected) * 0.01)
    return abs(observed - expected) <= tolerance


def _quantitative_atoms(option_text: str) -> tuple[str, ...]:
    text = str(option_text or "").casefold()
    times = tuple(re.findall(r"\b\d{1,2}:\d{2}\b", text))
    remainder = re.sub(r"\b\d{1,2}:\d{2}\b", " ", text)
    numbers = tuple(
        token.replace(",", "")
        for token in re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", remainder)
    )
    has_quantity_unit = bool(
        re.search(
            r"\b(?:thousand|million|billion|trillion|calories?|light[ -]?years?|kilometers?|kilometres?|meters?|metres?|km)\b",
            text,
        )
    )
    meaningful_numbers = tuple(token for token in numbers if len(token.split(".", 1)[0]) >= 3 or has_quantity_unit)
    return tuple(dict.fromkeys(times + meaningful_numbers))


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


def _event_occurrences(records: Sequence[EvidenceRecord]) -> tuple[dict[str, Any], ...]:
    occurrences: list[dict[str, Any]] = []
    for record in records:
        for index, value in enumerate(record.operation_metadata.get("events", ()) or (), start=1):
            if not isinstance(value, Mapping) or not _metadata_flag(value.get("supports_question_event")):
                continue
            try:
                start = float(value.get("start_sec"))
                end = float(value.get("end_sec"))
            except (TypeError, ValueError):
                continue
            if end < start:
                start, end = end, start
            source_id = _event_source_id(record, start, end)
            event_key = _normalize_event_key(value.get("event_key"))
            continues_from_previous = _metadata_flag(value.get("continues_from_previous"))
            continues_to_next = _metadata_flag(value.get("continues_to_next"))
            duplicate = next(
                (
                    item
                    for item in occurrences
                    if item["source_video_id"] == source_id
                    and (
                        _event_intervals_equivalent(start, end, item["start_sec"], item["end_sec"])
                        or _events_are_explicit_continuations(
                            start,
                            end,
                            event_key=event_key,
                            continues_from_previous=continues_from_previous,
                            continues_to_next=continues_to_next,
                            existing=item,
                        )
                    )
                ),
                None,
            )
            if duplicate is not None:
                duplicate["evidence_ids"].append(record.evidence_id)
                duplicate["start_sec"] = min(float(duplicate["start_sec"]), start)
                duplicate["end_sec"] = max(float(duplicate["end_sec"]), end)
                duplicate["continues_from_previous"] = bool(
                    duplicate.get("continues_from_previous") or continues_from_previous
                )
                duplicate["continues_to_next"] = bool(duplicate.get("continues_to_next") or continues_to_next)
                continue
            occurrences.append(
                {
                    "event_id": f"{record.evidence_id}:{value.get('local_id') or f'event_{index}'}",
                    "event_key": event_key,
                    "description": str(value.get("description", "") or ""),
                    "start_sec": start,
                    "end_sec": end,
                    "source_video_id": source_id,
                    "evidence_ids": [record.evidence_id],
                    "continues_from_previous": continues_from_previous,
                    "continues_to_next": continues_to_next,
                }
            )
    return tuple(occurrences)


def _event_source_id(record: EvidenceRecord, start_sec: float, end_sec: float) -> str:
    center = (float(start_sec) + float(end_sec)) / 2.0
    for lineage in record.source_lineage:
        virtual_range = tuple(lineage.get("virtual_time_range", ()) or ())
        if len(virtual_range) == 2 and float(virtual_range[0]) <= center <= float(virtual_range[1]):
            return str(lineage.get("source_video_id", "") or "")
    if record.source_lineage:
        return str(record.source_lineage[0].get("source_video_id", "") or "")
    return ""


def _event_intervals_equivalent(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    overlap = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    shorter = min(max(0.0, end_a - start_a), max(0.0, end_b - start_b))
    if shorter > 0.0 and overlap / shorter >= 0.5:
        return True
    center_a = (start_a + end_a) / 2.0
    center_b = (start_b + end_b) / 2.0
    return abs(center_a - center_b) <= 1.0


def _events_are_explicit_continuations(
    start: float,
    end: float,
    *,
    event_key: str,
    continues_from_previous: bool,
    continues_to_next: bool,
    existing: Mapping[str, Any],
) -> bool:
    if not event_key or event_key != str(existing.get("event_key", "") or ""):
        return False
    follows_existing = (
        continues_from_previous
        and _metadata_flag(existing.get("continues_to_next"))
        and abs(float(start) - float(existing["end_sec"])) <= 2.0
    )
    precedes_existing = (
        continues_to_next
        and _metadata_flag(existing.get("continues_from_previous"))
        and abs(float(end) - float(existing["start_sec"])) <= 2.0
    )
    return follows_existing or precedes_existing


def _normalize_event_key(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


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
    derivation: Mapping[str, Any] | None = None,
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
    event_occurrences = _event_occurrences(parent_records)
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
        operation_metadata={
            "algorithm": "reasoner_reconciliation",
            "entity_clusters": clusters,
            "event_occurrences": event_occurrences,
            "derivation": dict(derivation or {}),
        },
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
        inspection_mode=str(value.get("inspection_mode", "window") or "window"),
        priority=float(value.get("priority", 0.0) or 0.0),
        claim_to_verify=str(value.get("claim_to_verify", "") or ""),
        claim_relation=str(value.get("claim_relation", "") or ""),
        alternative_answers=tuple(value.get("alternative_answers", ()) or ()),
        search_terms=tuple(value.get("search_terms", ()) or ()),
        gap_id=str(value.get("gap_id", "") or ""),
        success_conditions=tuple(value.get("success_conditions", ()) or ()),
        direction=str(value.get("direction", "") or ""),
        preferred_ranges=tuple(value.get("preferred_ranges", ()) or ()),
        excluded_ranges=tuple(value.get("excluded_ranges", ()) or ()),
        region_hint=str(value.get("region_hint", "") or ""),
        conditions=tuple(value.get("conditions", ()) or ()),
    )


def _time_ranges(value: Sequence[Sequence[float]] | None) -> tuple[tuple[float, float], ...]:
    rows = []
    for item in value or ():
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
            continue
        try:
            start, end = float(item[0]), float(item[1])
        except (TypeError, ValueError):
            continue
        if end < start:
            start, end = end, start
        if end > start:
            rows.append((start, end))
    return tuple(rows)


def _gap(value: EvidenceGap | Mapping[str, Any]) -> EvidenceGap:
    if isinstance(value, EvidenceGap):
        return value
    return EvidenceGap(
        gap_id=str(value.get("gap_id", "") or ""),
        description=str(value.get("description", "") or ""),
        success_conditions=tuple(value.get("success_conditions", ()) or ()),
        falsification_conditions=tuple(value.get("falsification_conditions", ()) or ()),
        conditions=tuple(value.get("conditions", ()) or ()),
        importance=str(value.get("importance", "critical") or "critical"),
        status=str(value.get("status", "open") or "open"),
    )


def _gap_condition(value: GapCondition | Mapping[str, Any] | str) -> GapCondition:
    if isinstance(value, GapCondition):
        return value
    if isinstance(value, Mapping):
        return GapCondition(
            condition_id=str(value.get("condition_id", "") or ""),
            description=str(value.get("description", value.get("condition", "")) or ""),
            critical=bool(value.get("critical", True)),
        )
    return GapCondition("", str(value or ""))


def _bind_gap_to_tasks(decision: ReasonerDecision) -> ReasonerDecision:
    gap = decision.primary_gap
    if decision.action != "investigate" or gap is None:
        return decision
    tasks = tuple(
        replace(
            task,
            gap_id=task.gap_id or gap.gap_id,
            success_conditions=task.success_conditions or gap.success_conditions,
            conditions=task.conditions or gap.conditions,
        )
        for task in decision.tasks
    )
    return replace(decision, tasks=tasks)


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
        support_status=str(value.get("support_status", "") or ""),
        support_reason=str(value.get("support_reason", "") or ""),
        primary_gap=_gap(value["primary_gap"]) if isinstance(value.get("primary_gap"), Mapping) else None,
    )


def _apply_answer_audit(gate: Mapping[str, Any], decision: ReasonerDecision) -> dict[str, Any]:
    result = dict(gate)
    status = str(decision.support_status or "").strip().casefold()
    if status:
        result["answer_audit_status"] = status
        result["audit_reason"] = decision.support_reason
    if status not in {"insufficient", "contradicted"}:
        return result
    result.update(
        {
            "base_gate_passed": bool(gate.get("passed")),
            "base_gate_reason": str(gate.get("reason", "") or ""),
            "passed": False,
            "reason": f"answer_audit_{status}",
        }
    )
    return result


def _answer_support_rank(decision: ReasonerDecision) -> int:
    return {
        "contradicted": 0,
        "insufficient": 1,
        "": 2,
        "unknown": 2,
        "supported": 3,
    }.get(decision.support_status, 2)


def _evidence_digest(evidence: Sequence[EvidenceRecord]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "evidence_id": item.evidence_id,
            "summary": item.verbatim,
            "confidence": item.confidence,
            "virtual_time_range": [item.start_sec, item.end_sec],
            "modality": item.modality,
            "evidence_kind": item.evidence_kind,
            "observation_polarity": item.observation_polarity,
            "source_lineage": [dict(row) for row in item.source_lineage],
            "entities": list(item.operation_metadata.get("entities", ())),
            "events": list(item.operation_metadata.get("events", ())),
            "target_presence": dict(item.operation_metadata.get("target_presence", {}) or {}),
            "measurements": list(item.operation_metadata.get("measurements", ())),
            "relations": list(item.operation_metadata.get("relations", ())),
            "derivation": dict(item.operation_metadata.get("derivation", {}) or {}),
            "claim_assessment": dict(item.operation_metadata.get("claim_assessment", {}) or {}),
            "investigation": dict(item.operation_metadata.get("investigation", {}) or {}),
            "navigation": (
                {
                    "search_terms": list(item.operation_metadata.get("search_terms", ())),
                    "matched_terms": list(item.operation_metadata.get("matched_terms", ())),
                    "hit_count": int(item.operation_metadata.get("hit_count", 0) or 0),
                }
                if item.evidence_kind == "navigation_hint"
                else {}
            ),
        }
        for item in evidence
    )


def _outcome_digest(reports: Sequence[InvestigationReport]) -> tuple[dict[str, Any], ...]:
    rows = []
    for report in tuple(reports)[-12:]:
        ranges = [
            [record.start_sec, record.end_sec]
            for record in report.evidence
            if record.start_sec is not None and record.end_sec is not None
        ]
        rows.append(
            {
                "query_id": report.query_id,
                "gap_id": report.gap_id,
                "resolution": report.resolution,
                "resolved_conditions": list(report.resolved_conditions),
                "unresolved_conditions": list(report.unresolved_conditions),
                "failure_reason": report.failure_reason,
                "progress_flags": list(report.progress_flags),
                "goal_progress": list(report.goal_progress),
                "coverage_progress": list(report.coverage_progress),
                "condition_results": to_jsonable(report.condition_results),
                "coverage_delta": [list(item) for item in report.coverage_delta],
                "observed_ranges": ranges,
                "evidence_ids": [record.evidence_id for record in report.evidence],
                "reused": bool(report.cost.get("reused")),
            }
        )
    return tuple(rows)


def _annotate_batch_progress(
    batch: Sequence[InvestigationReport],
    previous: Sequence[InvestigationReport],
) -> tuple[InvestigationReport, ...]:
    covered: dict[str, list[tuple[float, float]]] = {}
    for report in previous:
        covered.setdefault(report.gap_id, []).extend(report.coverage_delta)
    result = []
    for report in batch:
        intervals = tuple(report.coverage_delta)
        prior = covered.setdefault(report.gap_id, [])
        adds_frontier = any(_uncovered_duration(interval, prior) >= 0.5 for interval in intervals)
        goal_progress = tuple(report.goal_progress) or tuple(
            dict.fromkeys(
                f"condition_{item.status}:{item.condition_id}"
                for item in report.condition_results
                if item.status in {"satisfied", "contradicted"}
            )
        )
        coverage_progress = tuple(report.coverage_progress)
        if adds_frontier and "new_frontier_coverage" not in coverage_progress:
            coverage_progress = (*coverage_progress, "new_frontier_coverage")
        progress_flags = tuple(dict.fromkeys((*report.progress_flags, *goal_progress, *coverage_progress)))
        normalized = replace(
            report,
            goal_progress=goal_progress,
            coverage_progress=coverage_progress,
            progress_flags=progress_flags,
        )
        result.append(normalized)
        prior.extend(intervals)
    return tuple(result)


def _uncovered_duration(
    interval: tuple[float, float],
    covered: Sequence[tuple[float, float]],
) -> float:
    start, end = float(interval[0]), float(interval[1])
    fragments = [(start, end)]
    for left, right in covered:
        next_fragments = []
        for frag_start, frag_end in fragments:
            if right <= frag_start or left >= frag_end:
                next_fragments.append((frag_start, frag_end))
                continue
            if left > frag_start:
                next_fragments.append((frag_start, min(left, frag_end)))
            if right < frag_end:
                next_fragments.append((max(right, frag_start), frag_end))
        fragments = next_fragments
        if not fragments:
            return 0.0
    return sum(max(0.0, frag_end - frag_start) for frag_start, frag_end in fragments)


def _stagnation_status(reports: Sequence[InvestigationReport]) -> dict[str, Any]:
    ordered = tuple(reports)
    last_gap = ordered[-1].gap_id if ordered else ""
    trailing = []
    for report in reversed(ordered):
        if report.gap_id != last_gap:
            break
        trailing.append(report)
        if len(trailing) >= 3:
            break
    recent = tuple(trailing)
    if len(recent) < 2:
        return {"stagnant": False, "gap_id": "", "reason": ""}
    gap_ids = {report.gap_id for report in recent if report.gap_id}
    if len(gap_ids) != 1:
        return {"stagnant": False, "gap_id": "", "reason": ""}
    unresolved = all(report.resolution in {"partial", "unresolved"} for report in recent)
    no_goal_progress = not any(report.goal_progress for report in recent)
    no_coverage_progress = not any(report.coverage_progress for report in recent)
    stagnant = unresolved and no_goal_progress and no_coverage_progress
    low_yield_coverage = len(recent) >= 3 and unresolved and no_goal_progress and not no_coverage_progress
    gap_id = next(iter(gap_ids)) if stagnant or low_yield_coverage else ""
    return {
        "stagnant": stagnant,
        "gap_id": gap_id,
        "low_yield_coverage": low_yield_coverage,
        "reason": "no_goal_or_frontier_progress" if stagnant else "coverage_without_condition_progress" if low_yield_coverage else "",
        "required_shift": "change range, direction, modality, or region focus" if stagnant or low_yield_coverage else "",
    }


def _retrieval_status(reports: Sequence[InvestigationReport], *, verified: bool) -> str:
    if verified:
        return "sufficient"
    if any(report.goal_progress for report in reports):
        return "partial"
    return "failed"


def _reports_share_range(reports: Sequence[InvestigationReport]) -> bool:
    ranges = []
    for report in reports:
        record = next(
            (
                item
                for item in report.evidence
                if item.start_sec is not None and item.end_sec is not None
            ),
            None,
        )
        if record is None:
            return False
        ranges.append((float(record.start_sec), float(record.end_sec)))
    first_start, first_end = ranges[0]
    for start, end in ranges[1:]:
        intersection = max(0.0, min(first_end, end) - max(first_start, start))
        union = max(first_end, end) - min(first_start, start)
        if union <= 0.0 or intersection / union < 0.8:
            return False
    return True


def _citations_are_visual(citations: Sequence[str], evidence: Sequence[EvidenceRecord]) -> bool:
    if not citations:
        return False
    return _answer_citations(citations, evidence) == tuple(str(citation) for citation in citations)


def _answer_citations(citations: Sequence[str], evidence: Sequence[EvidenceRecord]) -> tuple[str, ...]:
    by_id = {item.evidence_id: item for item in evidence}
    return tuple(
        citation
        for citation in dict.fromkeys(str(item) for item in citations if str(item).strip())
        if (record := by_id.get(citation)) is not None
        and record.modality in {"visual", "ocr"}
        and not is_path_only_visual_evidence(record)
    )


def _score_answer(answer: str, gold: str, options: Mapping[str, str] | None = None) -> bool:
    selected = _letter(answer) or _option_letter_from_answer(answer, options or {})
    expected = _letter(gold) or str(gold or "").strip().upper()[:1]
    return bool(selected and expected and selected == expected)


def _option_letter_from_answer(answer: str, options: Mapping[str, str]) -> str:
    normalized_answer = _answer_match_text(answer)
    if not normalized_answer:
        return ""
    direct = [
        str(label).upper()
        for label, text in options.items()
        if (normalized_option := _answer_match_text(text))
        and (normalized_option in normalized_answer or normalized_answer in normalized_option)
    ]
    if len(direct) == 1:
        return direct[0]
    answer_numbers = set(re.findall(r"\d+(?:\.\d+)?", str(answer or "")))
    numeric = []
    for label, text in options.items():
        option_numbers = set(re.findall(r"\d+(?:\.\d+)?", str(text or "")))
        if option_numbers and option_numbers.issubset(answer_numbers):
            numeric.append(str(label).upper())
    return numeric[0] if len(numeric) == 1 else ""


def _answer_match_text(value: str) -> str:
    text = str(value or "").casefold()
    replacements = {
        "kilometres": "km",
        "kilometers": "km",
        "kilometre": "km",
        "kilometer": "km",
        "metres": "m",
        "meters": "m",
        "metre": "m",
        "meter": "m",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return re.sub(r"[^a-z0-9]+", "", text)


def _letter(value: str) -> str:
    text = str(value or "").strip().upper()
    leading = re.match(r"^[\(\[]?([A-H])(?:[\)\].:\-]|\s*$)", text)
    if leading:
        return leading.group(1)
    explicit = re.search(
        r"\b(?:ANSWER|OPTION|CHOICE)\s*(?:IS\s*)?[:\-]?\s*[\(\[]?([A-H])(?:[\)\].:\-]|\b)",
        text,
    )
    return explicit.group(1) if explicit else ""


def _write_run_summary(workspace: VirtualVideoWorkspace, result: MultiRoundResult) -> None:
    path = workspace.root_dir / "run_summary.json"
    path.write_text(
        json.dumps(
            {
                "case_id": result.case_id,
                "answer": result.answer,
                "grounded_answer": result.grounded_answer,
                "forced_answer": result.forced_answer,
                "selected_option": result.selected_option,
                "answer_mode": result.answer_mode,
                "grounding_status": result.grounding_status,
                "retrieval_status": result.retrieval_status,
                "citations": list(result.citations),
                "correct": result.correct,
                "verified": result.verified,
                "verification_reason": result.verification_reason,
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
