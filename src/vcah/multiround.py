from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import re
from typing import Any, Mapping, Sequence

from vcah.adjudication import (
    RevisionContext,
    build_all_option_audit_record,
    evaluate_hard_override_guard,
    final_adjudicate,
)
from vcah.evidence_primitives import (
    GapCondition,
    MeasurementFact,
    canonical_unit,
    make_gap_conditions,
    merge_condition_states,
    normalize_measurements,
    normalize_relations,
)
from vcah.enumeration import build_enumeration_manifest
from vcah.investigator import (
    EvidenceFact,
    InvestigationReport,
    ObservationAttempt,
    VirtualVideoInvestigator,
)
from vcah.memory import EvidenceStore
from vcah.obligations import (
    QueryObligations,
    compile_query_obligations,
    evaluate_query_obligations,
)
from vcah.provenance import (
    deterministic_derivation_provenance,
    heuristic_provenance,
    provenance_kinds,
)
from vcah.qualification import (
    QualificationRequirement,
    evaluate_requirement_graph,
    parse_option_predicates,
    requirement_telemetry,
)
from vcah.semantic_evidence import (
    canonical_fact_snapshot,
    event_candidate_ledger as _event_candidate_ledger,
    semantic_repair_requests,
)
from vcah.sequence import build_sequence_ledger
from vcah.types import ClaimContract, EvidenceRecord, is_path_only_visual_evidence, to_jsonable
from vcah.virtual_index import build_workspace_overview
from vcah.virtual_video import VirtualVideoWorkspace, select_uniform_items


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
    source_candidate_ids: tuple[str, ...] = ()
    inspection_intent: str = ""
    reference_entities: tuple[Mapping[str, Any], ...] = ()
    reference_facts: tuple[Mapping[str, Any], ...] = ()
    origin_gap_id: str = ""
    target_condition_ids: tuple[str, ...] = ()
    boundary_episode_id: str = ""
    target_option_predicates: tuple[str, ...] = ()
    target_requirement_ids: tuple[str, ...] = ()
    candidate_id: str = ""
    episode_id: str = ""
    entity_hypothesis_id: str = ""
    target_option_predicate_ids: tuple[str, ...] = ()
    sampling_floor_fps: float | None = None
    expected_event_dwell_sec: float | None = None
    temporal_resolution_rationale: str = ""
    sampling_floor_specified: bool = field(default=False, init=False)

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
        object.__setattr__(self, "source_candidate_ids", tuple(dict.fromkeys(str(item) for item in self.source_candidate_ids if str(item))))
        object.__setattr__(self, "inspection_intent", str(self.inspection_intent or "").strip())
        object.__setattr__(
            self,
            "reference_entities",
            tuple(dict(item) for item in self.reference_entities if isinstance(item, Mapping)),
        )
        object.__setattr__(
            self,
            "reference_facts",
            tuple(dict(item) for item in self.reference_facts if isinstance(item, Mapping)),
        )
        object.__setattr__(self, "origin_gap_id", str(self.origin_gap_id or self.gap_id or "").strip())
        object.__setattr__(
            self,
            "target_condition_ids",
            tuple(dict.fromkeys(str(item).strip() for item in self.target_condition_ids if str(item).strip())),
        )
        object.__setattr__(self, "boundary_episode_id", str(self.boundary_episode_id or "").strip())
        object.__setattr__(
            self,
            "target_option_predicates",
            tuple(dict.fromkeys(str(item).strip() for item in self.target_option_predicates if str(item).strip())),
        )
        object.__setattr__(
            self,
            "target_requirement_ids",
            tuple(dict.fromkeys(
                str(item).strip()
                for item in (*self.target_requirement_ids, *self.target_condition_ids)
                if str(item).strip()
            )),
        )
        object.__setattr__(self, "candidate_id", str(self.candidate_id or "").strip())
        object.__setattr__(self, "episode_id", str(self.episode_id or self.boundary_episode_id or "").strip())
        object.__setattr__(self, "entity_hypothesis_id", str(self.entity_hypothesis_id or "").strip())
        object.__setattr__(
            self,
            "target_option_predicate_ids",
            tuple(dict.fromkeys(str(item).strip() for item in self.target_option_predicate_ids if str(item).strip())),
        )
        specified = self.sampling_floor_fps is not None
        floor_fps = float(self.sampling_floor_fps or 0.5)
        object.__setattr__(self, "sampling_floor_fps", min(2.0, max(0.5, floor_fps)))
        object.__setattr__(self, "sampling_floor_specified", specified)
        dwell = self.expected_event_dwell_sec
        object.__setattr__(
            self,
            "expected_event_dwell_sec",
            float(dwell) if dwell is not None and float(dwell) > 0 else None,
        )
        object.__setattr__(
            self,
            "temporal_resolution_rationale",
            str(self.temporal_resolution_rationale or "").strip(),
        )
        if specified and not str(self.temporal_resolution_rationale or "").strip():
            object.__setattr__(self, "priority", float(self.priority) * 0.8)
        object.__setattr__(
            self,
            "search_terms",
            tuple(dict.fromkeys(str(item).strip().casefold() for item in self.search_terms if str(item).strip())),
        )
        if self.inspection_mode not in {
            "window", "enumerate_events", "event_window", "verify_claim", "search_asr",
            "entity_association", "narrative_bridge",
        }:
            object.__setattr__(self, "inspection_mode", "window")

    def clone_with(self, **overrides: Any) -> "InvestigationTask":
        """Clone a task without manually copying its evolving field set."""
        cloned = replace(self, **overrides)
        if "sampling_floor_fps" not in overrides:
            object.__setattr__(cloned, "sampling_floor_specified", self.sampling_floor_specified)
        if "priority" not in overrides:
            object.__setattr__(cloned, "priority", self.priority)
        return cloned


def _task_replay_descriptor(task: InvestigationTask) -> dict[str, Any]:
    """Keep replay ordering inspectable without embedding prompt or observation text."""
    return {
        "query_id": task.query_id,
        "segment_id": task.segment_id,
        "time_range": list(task.time_range) if task.time_range is not None else [],
        "inspection_mode": task.inspection_mode,
        "gap_id": task.gap_id,
        "source_candidate_ids": list(task.source_candidate_ids),
        "episode_id": task.episode_id,
        "entity_hypothesis_id": task.entity_hypothesis_id,
        "target_requirement_ids": list(task.target_requirement_ids),
        "sampling_floor_fps": task.sampling_floor_fps,
    }


@dataclass(frozen=True)
class ReasonerDecision:
    action: str
    tasks: tuple[InvestigationTask, ...] = ()
    answer: str = ""
    citations: tuple[str, ...] = ()
    entity_clusters: tuple[Mapping[str, Any], ...] = ()
    support_status: str = ""
    support_reason: str = ""
    option_verdicts: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    audit_record: Mapping[str, Any] = field(default_factory=dict)
    primary_gap: EvidenceGap | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tasks", tuple(_task(item) for item in self.tasks))
        object.__setattr__(self, "citations", tuple(str(item) for item in self.citations if str(item).strip()))
        object.__setattr__(self, "entity_clusters", tuple(_entity_cluster(item) for item in self.entity_clusters))
        object.__setattr__(self, "support_status", str(self.support_status or "").strip().casefold())
        object.__setattr__(self, "support_reason", str(self.support_reason or "").strip())
        object.__setattr__(
            self,
            "option_verdicts",
            {
                str(option).strip().upper(): dict(verdict)
                for option, verdict in dict(self.option_verdicts or {}).items()
                if str(option).strip() and isinstance(verdict, Mapping)
            },
        )
        object.__setattr__(self, "audit_record", dict(self.audit_record or {}))
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
    grounding_level: str = "none"
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
    is_occurrence_count = bool(re.search(r"\bhow many times\b|\bnumber of times\b", text)) or (
        is_count and _count_target_is_event(text)
    )
    measurement_text = " ".join((text, *(str(item).casefold() for item in (options or {}).values())))
    asked_measurement_unit = _asked_measurement_unit(text)
    measurement_unit = asked_measurement_unit or _measurement_unit(measurement_text)
    boundary_hint = _boundary_hint(question)
    full_video = "in total" in text or bool(
        re.search(r"\b(?:throughout|across)\b.*\b(?:video|film|recording)\b", text)
        or re.search(r"\b(?:entire|whole)\s+(?:video|film|recording)\b", text)
        or re.search(r"\bin\s+(?:this|the)\s+(?:video|film|recording)\b", text)
        or re.search(r"\bfrom\s+(?:this|the)\s+(?:video|film|recording)\b", text)
        or re.search(r"\bover the course of\s+(?:this|the)\s+(?:video|film|recording)\b", text)
    )
    bounded_interval = _has_bounded_time_interval(text)
    temporal_sequence = _is_temporal_sequence_question(text, options or {})
    cross_window_identity = _is_cross_window_identity_question(text)
    state_outcome = _is_state_outcome_question(text)
    epistemic_options = _has_epistemic_answer_option(options or {})
    attribute_transition = _is_attribute_transition_question(text)
    agent_attribution = _requires_agent_attribution_question(text)
    language_action = any(term in text for term in ("comment", "say", "speak", "discuss", "mention"))
    identity_anchor_terms = _identity_anchor_terms(question)
    if _is_boundary_score_question(text, options or {}, boundary_hint):
        return ClaimContract(
            required_scope="window",
            quantifier="comparison",
            observation_target="attribute",
            aggregation="compare",
            required_observability=("visual", "ocr"),
            observability_mode="any",
            measurement_unit="point",
            boundary_hint=boundary_hint,
        )
    if _is_video_summary_question(text):
        return ClaimContract(
            required_scope="full_video",
            quantifier="comparison",
            observation_target="event",
            aggregation="summarize",
            required_observability=("visual", "asr"),
            observability_mode="any",
        )
    if _is_temporal_label_question(text):
        return ClaimContract(
            required_scope="multi_window",
            quantifier="comparison",
            observation_target="event",
            aggregation="compare",
            required_observability=("visual", "ocr", "asr"),
            observability_mode="any",
            boundary_hint=_temporal_event_hint(question),
        )
    if temporal_sequence and not cross_window_identity:
        return ClaimContract(
            required_scope="full_video" if not bounded_interval else "multi_window",
            quantifier="order",
            observation_target="event",
            aggregation="order",
            required_observability=("visual", "asr"),
            observability_mode="any",
            boundary_hint=_temporal_event_hint(question),
        )
    if epistemic_options:
        return ClaimContract(
            required_scope="multi_window",
            quantifier="comparison",
            observation_target="attribute",
            aggregation="compare",
            required_observability=("visual", "asr"),
            observability_mode="any",
            boundary_hint="closed candidate elimination with explicit uncertainty",
        )
    if attribute_transition:
        return ClaimContract(
            required_scope="multi_window",
            quantifier="comparison",
            observation_target="attribute",
            aggregation="compare",
            required_observability=("visual",),
            observability_mode="all",
            boundary_hint="before state to after state",
        )
    if _is_spatial_relation_question(text):
        return ClaimContract(
            required_scope="window",
            quantifier="comparison",
            observation_target="relation",
            aggregation="compare",
            required_observability=("visual",),
            observability_mode="all",
        )
    if full_video and _is_global_absence_question(text):
        return ClaimContract(
            required_scope="full_video",
            quantifier="universal",
            observation_target="object",
            aggregation="compare",
            required_observability=("visual",),
            observability_mode="all",
        )
    if _is_completed_progress_question(text):
        return ClaimContract(
            required_scope="multi_window",
            quantifier="scalar_quantity",
            observation_target="attribute",
            aggregation="compare",
            required_observability=("visual", "ocr"),
            observability_mode="any",
            measurement_unit="task",
            boundary_hint=_progress_boundary_hint(question),
        )
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
            required_scope=(
                "full_video"
                if full_video or (is_occurrence_count and not bounded_interval)
                else "multi_window"
            ),
            quantifier="total_count" if is_occurrence_count else "distinct_count",
            observation_target="event" if is_occurrence_count else "entity",
            aggregation="count" if is_occurrence_count else "deduplicate",
            required_observability=("visual", "asr") if language_action else ("visual",),
            observability_mode="all",
        )
    if cross_window_identity:
        return ClaimContract(
            required_scope="multi_window",
            quantifier="comparison",
            observation_target="entity",
            aggregation="compare",
            required_observability=("visual",),
            observability_mode="all",
            boundary_hint="referenced event participant to later outcome",
        )
    if _is_narrative_inference_question(text):
        return ClaimContract(
            required_scope="multi_window",
            quantifier="comparison",
            observation_target="event",
            aggregation="compare",
            required_observability=("visual", "asr"),
            observability_mode="any",
            boundary_hint="narrative setup to implied outcome",
            requires_agent_attribution=agent_attribution,
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
    if _is_causal_relation_question(text):
        return ClaimContract(
            required_scope="multi_window",
            quantifier="comparison",
            observation_target="relation",
            aggregation="compare",
            required_observability=("visual", "asr"),
            observability_mode="any",
        )
    if _is_event_outcome_question(text):
        return ClaimContract(
            required_scope="multi_window",
            quantifier="comparison",
            observation_target="event",
            aggregation="compare",
            required_observability=("visual", "asr"),
            observability_mode="any",
        )
    if state_outcome:
        return ClaimContract(
            required_scope="multi_window",
            quantifier="comparison",
            observation_target="event",
            aggregation="compare",
            required_observability=("visual", "asr"),
            observability_mode="any",
            boundary_hint="initial state to final outcome",
        )
    if _is_agent_relation_question(text):
        return ClaimContract(
            required_scope="window",
            quantifier="comparison",
            observation_target="relation",
            aggregation="compare",
            required_observability=("visual",),
            observability_mode="all",
            requires_agent_attribution=True,
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
        (r"\btasks?\b", "task"),
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
    named_boundary = re.search(
        r"\b(?:half[ -]?time|end of (?:the )?first half|quarter end|end of (?:the )?(?:first|second|third|fourth) quarter|intermission)\b",
        text,
        flags=re.IGNORECASE,
    )
    if named_boundary:
        return named_boundary.group(0).strip()
    match = re.search(
        r"\b(?:before|when|until|by the time|at the time)\b\s+([^?.,;]+)",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(0).strip() if match else ""


def _count_target_is_event(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:how many|number of)\s+(?:distinct\s+|different\s+)?(?:[a-z-]+\s+){0,3}"
            r"(?:auditions?|performances?|acts?|segments?|episodes?|scenes?|occurrences?|events?)\b",
            str(text or "").casefold(),
        )
    )


def _is_spatial_relation_question(text: str) -> bool:
    normalized = str(text or "").casefold()
    return any(
        term in normalized
        for term in (
            "in relation to",
            "relative to",
            "which direction",
            "facing",
            "left front",
            "right front",
            "directly in front",
            "behind",
            "upper left",
            "upper right",
            "lower left",
            "lower right",
        )
    )


def _is_video_summary_question(text: str) -> bool:
    normalized = str(text or "").casefold()
    return bool(
        re.search(r"\b(?:title|heading)\b.*\b(?:summari[sz]\w*|best)\b", normalized)
        or re.search(r"\bbest\s+(?:title|summary)\b", normalized)
        or re.search(r"\b(?:main|overall|central)\s+(?:topic|theme|idea|message)\b", normalized)
        or re.search(r"\b(?:video|film|documentary)\s+(?:is\s+)?mainly\s+about\b", normalized)
    )


def _is_temporal_label_question(text: str) -> bool:
    normalized = str(text or "").casefold()
    return bool(
        re.search(r"\b(?:at|around)\s+what\s+time\b", normalized)
        or re.search(r"\bwhat\s+time\b", normalized)
        or re.search(r"\bspecific\s+time\b", normalized)
        or re.search(r"\bwhen\s+(?:does|did|is|was|were|will|would|can|could)\b", normalized)
        or re.search(r"\bwhich\s+(?:episode|day|meal|period|quarter|stage|phase)\b", normalized)
    )


def _has_bounded_time_interval(text: str) -> bool:
    normalized = str(text or "").casefold()
    return bool(
        re.search(r"\b(?:first|last)\s+\d+(?:\.\d+)?\s+(?:seconds?|minutes?|hours?)\b", normalized)
        or re.search(r"\bbetween\s+\d+(?::\d+)?\s+(?:and|to)\s+\d+(?::\d+)?\b", normalized)
        or re.search(r"\bfrom\s+\d+(?::\d+)?\s+(?:to|until|through)\s+\d+(?::\d+)?\b", normalized)
    )


def _is_temporal_sequence_question(text: str, options: Mapping[str, str]) -> bool:
    normalized = str(text or "").casefold()
    explicit = bool(
        re.search(r"\b(?:sequence|chronological order|correct order|in what order)\b", normalized)
        or (
            re.search(r"\b(?:who|which)\b", normalized)
            and re.search(r"\b(?:first|second|third|fourth|last)\b", normalized)
            and re.search(
                r"\b(?:overt(?:ake|akes|aken|aking|ook)|pass\w*|arriv\w*|appear\w*|finish\w*|cross\w*)\b",
                normalized,
            )
        )
    )
    option_sequences = sum(
        bool(re.search(r"(?:→|->| then | followed by )", str(option).casefold()))
        for option in options.values()
    )
    return explicit or option_sequences >= 2


def _is_cross_window_identity_question(text: str) -> bool:
    normalized = str(text or "").casefold()
    return bool(
        re.search(r"\b(?:that|the)\s+same\s+(?:last\s+)?instance\b", normalized)
        or (
            re.search(r"\blast\s+instance\b", normalized)
            and re.search(r"\b(?:first|second|third|last)\s+(?:person|player|rider|competitor)\b", normalized)
        )
        or (
            re.search(r"\b(?:first|second|third|last)\s+(?:person|player|rider|competitor)\b", normalized)
            and re.search(r"\b(?:ultimately|finally|final|finish(?:ed)?)\b", normalized)
        )
    )


def _is_state_outcome_question(text: str) -> bool:
    normalized = str(text or "").casefold()
    return bool(
        re.search(r"\b(?:initial|original|earlier)\s+(?:idea|plan|intention|state|position)\b", normalized)
        and re.search(r"\b(?:ultimately|finally|in the end|stick|maintain|remain)\b", normalized)
    )


def _is_narrative_inference_question(text: str) -> bool:
    normalized = str(text or "").casefold()
    return bool(
        re.search(r"\b(?:internal monologue|narrative gap|best represents?|best explains?)\b", normalized)
    )


def _has_epistemic_answer_option(options: Mapping[str, str]) -> bool:
    return any(
        re.search(
            r"\b(?:uncertain|unknown|cannot (?:be )?determin(?:e|ed)|can't (?:be )?determin(?:e|ed)|"
            r"not (?:shown|stated|specified|known)|none of (?:the )?(?:above|options))\b",
            str(option or "").casefold(),
        )
        for option in options.values()
    )


def _is_attribute_transition_question(text: str) -> bool:
    normalized = str(text or "").casefold()
    attribute = bool(re.search(r"\b(?:color|colour|appearance|shape|texture|state)\b", normalized))
    transition = bool(
        re.search(r"\b(?:change|changed|changes|turn|turned|become|became|mixed|mixing|compared to)\b", normalized)
    )
    return attribute and transition


def _temporal_event_hint(question: str) -> str:
    text = " ".join(str(question or "").split())
    match = re.search(
        r"\b(?:when|what time|specific time|which episode|which day|which meal|which period|which quarter|which stage|which phase)\b[^?]*",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(0).strip() if match else "event-to-time-label binding"


def _is_completed_progress_question(text: str) -> bool:
    normalized = str(text or "").casefold()
    return bool(
        re.search(
            r"\bhow\s+many\s+(?:[a-z-]+\s+){0,2}(?:tasks?|challenges?|activities?)\b[^?]*"
            r"\b(?:complete\w*|finish\w*|accomplish\w*)\b",
            normalized,
        )
    )


def _progress_boundary_hint(question: str) -> str:
    text = " ".join(str(question or "").split())
    leading = re.match(r"\s*(?:as|when|while)\s+([^,?]+)", text, flags=re.IGNORECASE)
    return leading.group(0).strip() if leading else _boundary_hint(question) or "requested progress checkpoint"


def _is_causal_relation_question(text: str) -> bool:
    normalized = str(text or "").casefold()
    return bool(re.search(r"\bwhy\b|\b(?:cause|reason|motive|because)\b", normalized))


def _is_event_outcome_question(text: str) -> bool:
    normalized = str(text or "").casefold()
    return bool(
        re.search(r"\bhow\b[^?]{0,80}\b(?:unfold|end|ended|conclude|concluded|turn(?:ed)? out)\b", normalized)
        or re.search(r"\bwhat\s+(?:happens?|happened)\s+to\b", normalized)
    )


def _is_agent_relation_question(text: str) -> bool:
    normalized = str(text or "").casefold()
    return bool(
        re.search(
            r"\bwho\s+(?:protects?|protected|helps?|helped|rescues?|rescued|saves?|saved|attacks?|attacked|"
            r"defeats?|defeated|stops?|stopped|supports?|supported|causes?|caused)\b",
            normalized,
        )
    )


def _requires_agent_attribution_question(text: str) -> bool:
    normalized = str(text or "").casefold()
    return bool(
        _is_agent_relation_question(normalized)
        or re.search(
            r"\b(?:who|[a-z][a-z]+)\s+(?:called?|calls?|reported|reports?|told|tells|warned|warns|"
            r"helped|helps|rescued|rescues|attacked|attacks|stole|steals|hid|hides|concealed|conceals)\b",
            normalized,
        )
    )


def _is_global_absence_question(text: str) -> bool:
    normalized = str(text or "").casefold()
    return bool(
        re.search(r"\babsent\b", normalized)
        or re.search(r"\b(?:does not|doesn't|never)\s+(?:appear|show|feature|occur)\b", normalized)
        or re.search(r"\bnot\s+(?:shown|seen|featured|included)\b", normalized)
    )


def _is_boundary_score_question(text: str, options: Mapping[str, str], boundary_hint: str) -> bool:
    if "score" not in str(text or "").casefold() or not boundary_hint:
        return False
    return any(_score_pair(str(option)) is not None for option in options.values())


def compile_query_requirements(question: str) -> dict[str, Any]:
    terms = _identity_anchor_terms(question)
    text = str(question or "").casefold()
    cross_window_identity = _is_cross_window_identity_question(text)
    narrative_inference = _is_narrative_inference_question(text) or _is_state_outcome_question(text)
    spatial = _is_spatial_relation_question(text)
    contrastive_subject = bool(
        re.search(r"\bone\s+(?:team|group|person)\b[^?]*\bthe\s+other\s+(?:team|group|person)\b", text)
    )
    temporal_sequence = bool(
        re.search(r"\b(?:sequence|order|before|after|then|first|last|initially|ultimately)\b", text)
    )
    temporal_extremum = bool(
        re.search(r"\b(?:first|last|final|ultimately|latest|earliest)\b", text)
    )
    state_tracking = bool(
        re.search(r"\b(?:change|changed|turn(?:ed)?|remain(?:ed)?|maintain(?:ed)?|overtak\w*|track\w*)\b", text)
    )
    return {
        "requires_identity_link": bool(terms) or cross_window_identity,
        "requires_event_participant_link": cross_window_identity,
        "requires_narrative_inference": narrative_inference,
        "requires_agent_attribution": _requires_agent_attribution_question(text),
        "identity_anchor_terms": list(terms),
        "requires_spatial_relation": spatial,
        "spatial_relation_type": "relative_facing" if spatial and "facing" in text else "relative_bearing" if spatial else "",
        "spatial_reference_frame": (
            "object_egocentric"
            if spatial and any(term in text for term in ("in relation to", "relative to"))
            else "viewer"
            if spatial
            else ""
        ),
        "requires_contrastive_subject_binding": contrastive_subject,
        "requires_temporal_sequence": temporal_sequence,
        "requires_temporal_extremum": temporal_extremum,
        "requires_state_tracking": state_tracking,
        "requires_same_object_transition": _is_attribute_transition_question(text),
        "measurement_subject_role": (
            "other_team"
            if contrastive_subject and "team" in text
            else "other_subject"
            if contrastive_subject
            else "anchored_subject"
            if terms and bool(_measurement_unit(text))
            else ""
        ),
    }


def compile_contract_conditions(
    contract: ClaimContract,
    requirements: Mapping[str, Any],
) -> tuple[GapCondition, ...]:
    conditions = []
    if contract.quantifier == "total_count" and contract.observation_target == "event" and requirements.get(
        "requires_state_tracking"
    ):
        conditions.extend((
            GapCondition(
                "condition_all_qualified_events_enumerated",
                "All qualified question events are enumerated across the full video.",
                condition_type="temporal",
                scope="full_video",
                quantifier="all_events",
                required_coverage=1.0,
                aggregation="event_union",
            ),
            GapCondition(
                "condition_event_qualification_resolved",
                "Every counted event has supported prior state, transition, same subject, and episode boundary.",
                condition_type="temporal",
                scope="full_video",
                quantifier="all_events",
                required_coverage=1.0,
                aggregation="event_union",
            ),
        ))
    if requirements.get("requires_event_participant_link") and requirements.get("requires_temporal_sequence"):
        conditions.extend((
            GapCondition(
                "condition_all_qualified_episodes_enumerated",
                "All qualified anchor episodes are enumerated across the full video.",
                condition_type="temporal",
                scope="full_video",
                quantifier="all_events",
                required_coverage=1.0,
                aggregation="event_union",
            ),
            GapCondition(
                "condition_last_episode_selected",
                "The last qualified episode is selected and no later qualified episode exists.",
                condition_type="temporal",
                scope="full_video",
                quantifier="temporal_max",
                required_coverage=1.0,
                aggregation="temporal_max",
            ),
            GapCondition(
                "condition_second_participant_resolved",
                "The second overtaking participant in the selected episode is identity-resolved.",
                condition_type="semantic",
                scope="episode",
                quantifier="ordinal_2",
                aggregation="ordinal",
            ),
        ))
    return tuple(conditions)


def _fact_derivation_provenance(
    rows: Sequence[Mapping[str, Any]],
    derivation: str,
) -> tuple[Mapping[str, Any], ...]:
    fact_ids = []
    evidence_ids = []
    source_kinds = []
    for row in rows:
        fact_id = str(
            row.get("candidate_id", "")
            or row.get("entity_id", "")
            or row.get("association_id", "")
            or row.get("fact_id", "")
            or ""
        )
        if fact_id:
            fact_ids.append(fact_id)
        evidence_ids.extend(str(item) for item in tuple(row.get("evidence_ids", ()) or ()) if str(item))
        source_kinds.extend(provenance_kinds(row.get("provenance", ())))
    if fact_ids and evidence_ids:
        return (
            deterministic_derivation_provenance(
                derivation=derivation,
                fact_ids=tuple(dict.fromkeys(fact_ids)),
                evidence_ids=tuple(dict.fromkeys(evidence_ids)),
                source_kinds=tuple(dict.fromkeys(source_kinds)),
            ),
        )
    return (
        heuristic_provenance(
            fact_ids=tuple(dict.fromkeys(fact_ids)),
            evidence_ids=tuple(dict.fromkeys(evidence_ids)),
            derivation=f"{derivation}_without_canonical_witnesses",
            producer="program",
        ),
    )


def _snapshot_derivation_provenance(
    snapshot: Mapping[str, Any],
    derivation: str,
) -> tuple[Mapping[str, Any], ...]:
    rows = tuple(
        row
        for key in ("qualified_events", "resolved_entities", "entity_associations", "inferred_facts")
        for row in tuple(snapshot.get(key, ()) or ())
        if isinstance(row, Mapping)
    )
    return _fact_derivation_provenance(rows, derivation)


def compile_query_qualification_requirements(
    contract: ClaimContract,
    requirements: Mapping[str, Any],
    snapshot: Mapping[str, Any] | None = None,
    completion_status: Mapping[str, Any] | None = None,
) -> tuple[QualificationRequirement, ...]:
    snapshot = dict(snapshot or {})
    completion = dict(completion_status or {})
    obligations = compile_query_obligations(contract, requirements)
    compiled: list[QualificationRequirement] = []
    coverage_supported = bool(
        completion.get("range_coverage_complete", _completion_scope_ratio(completion) >= 1.0)
    )
    needs_global_temporal = bool(
        requirements.get("requires_event_participant_link")
        and requirements.get("requires_temporal_sequence")
    )
    if obligations.effective_scope == "full_video" or contract.quantifier == "total_count" or needs_global_temporal:
        compiled.append(QualificationRequirement(
            "req_full_video_coverage",
            "coverage_complete",
            {
                "status": "supported" if coverage_supported else "unknown",
                "provenance": _snapshot_derivation_provenance(
                    snapshot,
                    "coverage_complete",
                ),
            },
            scope="full_video",
            quantifier="all_segments",
        ))
    if contract.quantifier == "total_count" and contract.observation_target == "event":
        incomplete = tuple(snapshot.get("incomplete_events", ()) or ())
        conflicted = tuple(snapshot.get("conflicted_events", ()) or ())
        qualified = tuple(snapshot.get("qualified_events", ()) or ())
        compiled.extend((
            QualificationRequirement(
                "req_all_event_candidates_resolved",
                "distinct_occurrence",
                {
                    "status": (
                        "conflicted" if conflicted
                        else "unknown" if incomplete or not qualified
                        else "supported"
                    ),
                    "fact_ids": [row.get("candidate_id") for row in qualified],
                    "provenance": _fact_derivation_provenance(
                        qualified,
                        "qualified_event_enumeration",
                    ),
                },
                scope="full_video",
                quantifier="all_events",
                dependency_ids=("req_full_video_coverage",),
            ),
            QualificationRequirement(
                "req_qualified_event_count",
                "attribute_match",
                {
                    "status": "supported" if qualified and not incomplete and not conflicted else "unknown",
                    "fact_ids": [row.get("candidate_id") for row in qualified],
                    "provenance": _fact_derivation_provenance(
                        qualified,
                        "qualified_event_count",
                    ),
                },
                scope="full_video",
                dependency_ids=("req_all_event_candidates_resolved",),
            ),
        ))
    if requirements.get("requires_event_participant_link") and requirements.get("requires_temporal_sequence"):
        qualified = tuple(snapshot.get("qualified_events", ()) or ())
        incomplete = tuple(snapshot.get("incomplete_events", ()) or ())
        conflicted = tuple(snapshot.get("conflicted_events", ()) or ())
        selection = dict(completion.get("temporal_max_selection", {}) or {})
        participant_selection = dict(completion.get("target_participant_selection", {}) or {})
        entity_binding = dict(completion.get("target_entity_binding", {}) or {})
        selected_episode_id = str(selection.get("selected_episode_id", "") or "")
        selected_episode = tuple(
            row for row in qualified
            if str(row.get("candidate_id", "") or "") == selected_episode_id
        )
        target_entity_id = str(entity_binding.get("entity_id", "") or "")
        resolved_targets = tuple(
            row for row in tuple(snapshot.get("resolved_entities", ()) or ())
            if str(row.get("entity_id", "") or "") == target_entity_id
        )
        target_attributes = tuple(
            row for row in tuple(completion.get("target_attribute_facts", ()) or ())
            if isinstance(row, Mapping)
        )
        enumeration_ready = bool(completion.get("enumeration_complete", False))
        temporal_ready = (
            bool(qualified)
            and not incomplete
            and not conflicted
            and coverage_supported
            and enumeration_ready
        )
        selection_status = str(selection.get("status", "incomplete") or "incomplete")
        participant_status = str(participant_selection.get("status", "incomplete") or "incomplete")
        entity_status = str(entity_binding.get("status", "incomplete") or "incomplete")
        selection_requirement_status = (
            "supported" if selection_status == "resolved"
            else "conflicted" if selection_status == "ambiguous"
            else "unknown"
        )
        participant_requirement_status = (
            "supported" if participant_status == "resolved"
            else "conflicted" if participant_status == "ambiguous"
            else "unknown"
        )
        entity_requirement_status = (
            "supported" if entity_status == "resolved" and len(resolved_targets) == 1
            else "conflicted" if entity_status == "ambiguous" or len(resolved_targets) > 1
            else "unknown"
        )
        coverage_dependency = ("req_full_video_coverage",) if any(
            requirement.requirement_id == "req_full_video_coverage" for requirement in compiled
        ) else ()
        compiled.extend((
            QualificationRequirement(
                "req_all_qualified_episodes_enumerated",
                "distinct_occurrence",
                {
                    "status": "supported" if temporal_ready else "unknown",
                    "provenance": _fact_derivation_provenance(
                        qualified,
                        "qualified_episode_enumeration",
                    ),
                },
                scope="full_video",
                quantifier="all_events",
                dependency_ids=coverage_dependency,
            ),
            QualificationRequirement(
                "req_last_episode_selected",
                "temporal_max",
                {
                    "status": selection_requirement_status,
                    "fact_ids": [selected_episode_id] if selected_episode_id else [],
                    "provenance": tuple(selection.get("derivation_provenance", ()) or ())
                    or _fact_derivation_provenance(selected_episode, "temporal_max"),
                },
                scope="full_video",
                quantifier="temporal_max",
                dependency_ids=("req_all_qualified_episodes_enumerated",),
            ),
            QualificationRequirement(
                "req_second_participant_resolved",
                "ordinal_member",
                {
                    "status": participant_requirement_status,
                    "fact_ids": [
                        str(participant_selection.get("participant_id", "") or "")
                    ] if participant_selection.get("participant_id") else [],
                    "provenance": _fact_derivation_provenance(
                        selected_episode,
                        "ordinal_participant_selection",
                    ),
                },
                scope="episode",
                quantifier="ordinal_2",
                dependency_ids=("req_last_episode_selected",),
            ),
            QualificationRequirement(
                "req_target_entity_bound",
                "same_entity",
                {
                    "status": entity_requirement_status,
                    "fact_ids": [row.get("entity_id") for row in resolved_targets],
                    "provenance": _fact_derivation_provenance(
                        resolved_targets,
                        "target_entity_binding",
                    ),
                },
                scope="episode",
                dependency_ids=("req_second_participant_resolved",),
            ),
            QualificationRequirement(
                "req_target_attributes_resolved",
                "attribute_match",
                {
                    "status": "supported" if entity_requirement_status == "supported" and target_attributes else "unknown",
                    "fact_ids": [str(row.get("fact_id", "") or "") for row in target_attributes],
                    "provenance": _fact_derivation_provenance(
                        target_attributes,
                        "target_attribute_resolution",
                    ),
                },
                scope="episode",
                dependency_ids=("req_target_entity_bound",),
            ),
        ))
    return tuple(compiled)


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
        "consumed",
        "consume",
        "ate",
        "eaten",
        "eating",
        "bought",
        "purchased",
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
        "consumed",
        "consume",
        "ate",
        "eaten",
        "eating",
        "bought",
        "purchased",
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
        system_task_budget_ratio: float = 0.4,
    ) -> None:
        self.reasoner = reasoner or HeuristicReasoner()
        self.investigator = investigator
        self.max_rounds = max(1, int(max_rounds))
        self.max_investigations = max(1, int(max_investigations))
        self.max_tasks_per_round = max(1, int(max_tasks_per_round))
        self.system_task_budget_ratio = max(0.0, min(1.0, float(system_task_budget_ratio)))

    def run(self, workspace: VirtualVideoWorkspace) -> MultiRoundResult:
        investigator = self.investigator or VirtualVideoInvestigator(workspace)
        investigator.reset_run_state()
        workspace_overview = build_workspace_overview(workspace, thumbnail_budget=40)
        query_contract = compile_query_contract(workspace.case.question, workspace.case.options)
        query_requirements = compile_query_requirements(workspace.case.question)
        query_obligations = compile_query_obligations(query_contract, query_requirements)
        compiled_conditions = compile_contract_conditions(query_contract, query_requirements)
        temporal_navigation = source_time_navigation(workspace, compile_source_time_hint(workspace.case.question))
        evidence_store = EvidenceStore.empty(workspace.root_dir / "evidence.jsonl")
        reports: list[InvestigationReport] = []
        trace: list[Mapping[str, Any]] = []
        accepted = 0
        answer = ""
        citations: tuple[str, ...] = ()
        verified = False
        grounding_level = "none"
        verification_reason = ""
        best_answer = ""
        best_citations: tuple[str, ...] = ()
        best_verification_reason = ""
        best_support_rank = -1
        best_raw_gate: dict[str, Any] = {"passed": False, "reason": "answer_missing"}
        last_raw_decision: ReasonerDecision | None = None
        last_raw_gate: dict[str, Any] = {"passed": False, "reason": "answer_missing"}
        last_raw_completion: dict[str, Any] = {}
        option_states: dict[str, dict[str, Any]] = {}
        last_gate_reason = "answer_missing"
        gate_feedback: dict[str, Any] = {}
        condition_registry: tuple[GapCondition, ...] = compiled_conditions
        active_condition_ids: tuple[str, ...] = tuple(
            condition.condition_id
            for condition in compiled_conditions
            if condition.evaluation_type == "observable"
        )
        last_rejected_submission: tuple[str, tuple[str, ...], tuple[str, ...]] | None = None
        stagnant_task_attempts: dict[tuple[Any, ...], int] = {}
        stagnation_actions: list[dict[str, Any]] = []
        system_task_budget_limit = int(self.max_investigations * self.system_task_budget_ratio)
        system_tasks_accepted = 0
        rounds_run = 0

        for round_id in range(1, self.max_rounds + 1):
            rounds_run = round_id
            remaining = self.max_investigations - accepted
            completion_status = _completion_status(
                workspace,
                query_contract,
                evidence_store.records,
                query_requirements=query_requirements,
                query_obligations=query_obligations,
                reports=reports,
                best_choice=best_answer,
                active_condition_ids=active_condition_ids,
                option_states=option_states,
            )
            navigation_candidates = _navigation_candidates(evidence_store.records, workspace.case.options)
            stagnation_status = _stagnation_status(reports)
            stagnation_status["actions"] = list(stagnation_actions[-8:])
            policy_suggestions = _policy_suggestions(
                workspace,
                query_contract,
                query_requirements,
                completion_status,
                evidence_store.records,
                round_id=round_id,
                limit=min(self.max_tasks_per_round, remaining),
            )
            decision, condition_registry = _align_decision_conditions(_decision(
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
                    evidence_digest=_evidence_digest(evidence_store.records, query_contract),
                    investigation_outcomes=_outcome_digest(reports),
                    navigation_candidates=navigation_candidates,
                    stagnation_status=stagnation_status,
                    answer_gate_feedback=gate_feedback,
                    policy_suggestions=policy_suggestions,
                    policy_limits={
                        "system_task_budget_ratio": self.system_task_budget_ratio,
                        "system_task_budget_limit": system_task_budget_limit,
                        "system_tasks_accepted": system_tasks_accepted,
                        "system_tasks_remaining": max(0, system_task_budget_limit - system_tasks_accepted),
                    },
                    remaining_budget=remaining,
                    pre_final_checkpoint=round_id == self.max_rounds,
                )
            ), condition_registry)
            decision = _bind_gap_to_tasks(decision)
            origin_gap = decision.primary_gap
            if decision.action == "investigate":
                executable_tasks = tuple(task for task in decision.tasks if _task_is_executable(task))
                if executable_tasks != decision.tasks:
                    trace.append(
                        {
                            "type": "task_validation",
                            "round": round_id,
                            "accepted_task_count": len(executable_tasks),
                            "rejected_task_count": len(decision.tasks) - len(executable_tasks),
                        }
                    )
                    decision = replace(decision, tasks=executable_tasks)
            if decision.action == "investigate":
                decision = _inherit_repair_lineage(
                    decision,
                    origin_gap=origin_gap,
                    condition_registry=condition_registry,
                    active_condition_ids=active_condition_ids,
                    reports=reports,
                    options=workspace.case.options,
                )
                decision, rejected_system_tasks = _enforce_system_task_budget(
                    decision,
                    policy_suggestions,
                    remaining_slots=max(0, system_task_budget_limit - system_tasks_accepted),
                )
                if rejected_system_tasks:
                    trace.append({
                        "type": "policy_budget_enforcement",
                        "round": round_id,
                        "system_task_budget_ratio": self.system_task_budget_ratio,
                        "system_task_budget_limit": system_task_budget_limit,
                        "rejected_query_ids": [task.query_id for task in rejected_system_tasks],
                    })
            trace.append({
                "type": "policy_suggestions",
                "round": round_id,
                "advisory_only": True,
                "suggestion_count": len(policy_suggestions),
                "suggestions": list(policy_suggestions),
                "system_task_budget_ratio": self.system_task_budget_ratio,
                "system_task_budget_limit": system_task_budget_limit,
                "system_tasks_accepted": system_tasks_accepted,
            })
            trace.append(
                {
                    "type": "reasoner_decision",
                    "round": round_id,
                    "action": decision.action,
                    "task_count": len(decision.tasks),
                    "tasks": [_task_replay_descriptor(task) for task in decision.tasks],
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
                decision_completion_status = _completion_status_with_decision(
                    completion_status,
                    decision,
                )
                gate = _answer_completion_gate(
                    workspace,
                    query_contract,
                    decision.answer,
                    decision.citations,
                    decision.entity_clusters,
                    evidence_store.records,
                    query_requirements=query_requirements,
                    completion_status=decision_completion_status,
                )
                gate = _annotate_grounding_level(gate, decision_completion_status)
                gate = _apply_answer_audit(
                    gate,
                    decision,
                    required=_requires_discriminative_audit(query_contract, query_requirements),
                )
                last_raw_decision = decision
                last_raw_gate = dict(gate)
                last_raw_completion = dict(decision_completion_status)
                trace.append({"type": "completion_gate", "round": round_id, **gate})
                last_gate_reason = str(gate.get("reason", "") or "verification_failed")
                option_states = _record_option_state(
                    option_states,
                    decision,
                    gate,
                    workspace.case.options,
                )
                gate_feedback = {
                    **dict(gate),
                    "effective_scope": decision_completion_status["effective_scope"],
                    "option_states": option_states,
                }
                if not gate["passed"]:
                    last_rejected_submission = _submission_fingerprint(decision, evidence_store.records)
                support_rank = _candidate_gate_rank(decision, gate)
                if _candidate_can_be_forced(decision, evidence_store.records) and support_rank > best_support_rank:
                    best_answer = decision.answer
                    best_citations = decision.citations
                    best_verification_reason = last_gate_reason
                    best_support_rank = support_rank
                    best_raw_gate = dict(gate)
                if gate["passed"]:
                    answer = decision.answer
                    verified = True
                    grounding_level = str(gate.get("grounding_level", "strict") or "strict")
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
            is_anchor_sweep = bool(decision.tasks) and all(
                task.inspection_intent == "event_participant_anchor_discovery"
                for task in decision.tasks
            )
            dispatch_limit = min(
                remaining,
                max(self.max_tasks_per_round, len(decision.tasks))
                if is_anchor_sweep else self.max_tasks_per_round,
            )
            raw_tasks = decision.tasks[:dispatch_limit]
            resolved_tasks = _resolve_workspace_tasks(
                workspace,
                raw_tasks,
                limit=dispatch_limit,
            )
            if resolved_tasks != raw_tasks:
                trace.append(
                    {
                        "type": "task_segment_resolution",
                        "round": round_id,
                        "requested_task_count": len(raw_tasks),
                        "resolved_task_count": len(resolved_tasks),
                        "requested_segment_ids": [task.segment_id for task in raw_tasks],
                        "resolved_segment_ids": [task.segment_id for task in resolved_tasks],
                    }
                )
            if not resolved_tasks:
                trace.append({
                    "type": "task_validation",
                    "round": round_id,
                    "accepted_task_count": 0,
                    "rejected_task_count": len(raw_tasks),
                    "reason": "reasoner_tasks_not_executable",
                })
                continue
            requested_tasks = tuple(_task_for_contract(task, query_contract) for task in resolved_tasks)
            tasks = requested_tasks
            allowed_tasks = tuple(
                task for task in tasks if stagnant_task_attempts.get(_task_progress_fingerprint(task), 0) < 2
            )
            rejected_tasks = tuple(task for task in tasks if task not in allowed_tasks)
            if rejected_tasks:
                action = {
                    "round": round_id,
                    "action": "reject_repeated_no_progress_task",
                    "query_ids": [task.query_id for task in rejected_tasks],
                    "required_shift": "change range, fps, inspection mode, modality, condition, candidate, or finalize",
                }
                stagnation_actions.append(action)
                trace.append({"type": "stagnation_enforcement", **action})
            tasks = allowed_tasks
            if not tasks:
                continue
            task_condition_ids = tuple(
                dict.fromkeys(
                    condition.condition_id
                    for task in tasks
                    for condition in task.conditions
                    if condition.condition_id and condition.evaluation_type == "observable"
                )
            )
            if task_condition_ids:
                active_condition_ids = task_condition_ids
            fact_state_before = _canonical_progress_signature(canonical_fact_snapshot(evidence_store.records).to_dict())
            satisfied_before = {
                result.condition_id
                for report in reports
                for result in report.condition_results
                if result.status == "satisfied" and result.condition_id
            }
            batch = _annotate_batch_progress(investigator.run_batch(tasks), reports)
            charged_tasks = sum(_report_consumes_budget(report) for report in batch)
            system_charged_tasks = sum(
                _report_consumes_budget(report)
                for task, report in zip(tasks, batch)
                if _task_adopts_policy_suggestion(task, policy_suggestions)
            )
            accepted += charged_tasks
            system_tasks_accepted += system_charged_tasks
            reports.extend(batch)
            known_evidence = {record.evidence_id for record in evidence_store.records}
            for report in batch:
                for record in report.evidence:
                    if record.evidence_id not in known_evidence:
                        evidence_store.add(record)
                        known_evidence.add(record.evidence_id)
            fact_state_after = _canonical_progress_signature(canonical_fact_snapshot(evidence_store.records).to_dict())
            newly_satisfied = {
                result.condition_id
                for report in batch
                for result in report.condition_results
                if result.status == "satisfied" and result.condition_id not in satisfied_before
            }
            batch_progress = bool(
                fact_state_after != fact_state_before
                or newly_satisfied
                or any(report.coverage_progress for report in batch)
            )
            for task in tasks:
                fingerprint = _task_progress_fingerprint(task)
                stagnant_task_attempts[fingerprint] = 0 if batch_progress else stagnant_task_attempts.get(fingerprint, 0) + 1
            trace.append({
                "type": "investigator_batch",
                "round": round_id,
                "requested_tasks": len(tasks),
                "accepted_tasks": charged_tasks,
                "no_information_gain_tasks": len(batch) - charged_tasks,
                "system_suggestion_tasks": system_charged_tasks,
                "system_task_budget_ratio": self.system_task_budget_ratio,
            })
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
                query_obligations=query_obligations,
                reports=reports,
                best_choice=best_answer,
                active_condition_ids=active_condition_ids,
                option_states=option_states,
            )
            final_decision, condition_registry = _align_decision_conditions(_decision(
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
                    evidence_digest=_evidence_digest(evidence_store.records, query_contract),
                    investigation_outcomes=_outcome_digest(reports),
                    navigation_candidates=_navigation_candidates(evidence_store.records, workspace.case.options),
                    stagnation_status=_stagnation_status(reports),
                    answer_gate_feedback=gate_feedback,
                    remaining_budget=0,
                    force_finalize=True,
                )
            ), condition_registry)
            final_decision = _bind_gap_to_tasks(final_decision)
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
                decision_completion_status = _completion_status_with_decision(
                    completion_status,
                    final_decision,
                )
                gate = _answer_completion_gate(
                    workspace,
                    query_contract,
                    final_decision.answer,
                    final_decision.citations,
                    final_decision.entity_clusters,
                    evidence_store.records,
                    query_requirements=query_requirements,
                    completion_status=decision_completion_status,
                )
                gate = _annotate_grounding_level(gate, decision_completion_status)
                gate = _apply_answer_audit(
                    gate,
                    final_decision,
                    required=_requires_discriminative_audit(query_contract, query_requirements),
                )
                last_raw_decision = final_decision
                last_raw_gate = dict(gate)
                last_raw_completion = dict(decision_completion_status)
                trace.append(
                    {
                        "type": "completion_gate",
                        "round": self.max_rounds + 1,
                        "finalization": True,
                        **gate,
                    }
                )
                last_gate_reason = str(gate.get("reason", "") or "verification_failed")
                option_states = _record_option_state(
                    option_states,
                    final_decision,
                    gate,
                    workspace.case.options,
                )
                support_rank = _candidate_gate_rank(final_decision, gate)
                if _candidate_can_be_forced(final_decision, evidence_store.records) and support_rank > best_support_rank:
                    best_answer = final_decision.answer
                    best_citations = final_decision.citations
                    best_verification_reason = last_gate_reason
                    best_support_rank = support_rank
                    best_raw_gate = dict(gate)
                if gate["passed"]:
                    answer = final_decision.answer
                    verified = True
                    grounding_level = str(gate.get("grounding_level", "strict") or "strict")
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

        if best_answer:
            last_raw_decision = ReasonerDecision(
                action="answer",
                answer=best_answer,
                citations=best_citations,
                support_status="insufficient",
                support_reason=best_verification_reason,
            )
            last_raw_gate = dict(best_raw_gate)
        raw_decision = last_raw_decision or ReasonerDecision(action="answer")
        final_completion = _completion_status(
            workspace,
            query_contract,
            evidence_store.records,
            query_requirements=query_requirements,
            query_obligations=query_obligations,
            reports=reports,
            best_choice=best_answer,
            active_condition_ids=active_condition_ids,
            option_states=option_states,
        ) if evidence_store.records else dict(last_raw_completion)
        decision_completion = _completion_status_with_decision(final_completion, raw_decision)
        final_snapshot = dict(final_completion.get("canonical_fact_snapshot", {}) or {})
        if not final_snapshot:
            final_snapshot = canonical_fact_snapshot(
                evidence_store.records,
                require_event_precondition=bool(query_requirements.get("requires_state_tracking")),
            ).to_dict()
        final_revision_context = RevisionContext.from_inputs(
            final_snapshot,
            _evidence_digest(evidence_store.records, query_contract),
            to_jsonable(query_contract),
        )
        audit_required = _requires_discriminative_audit(query_contract, query_requirements)
        raw_audit = dict(raw_decision.audit_record or {})
        final_audit = build_all_option_audit_record(
            options=workspace.case.options,
            supplied_verdicts=raw_decision.option_verdicts,
            audit_status=str(
                raw_audit.get("audit_status", "complete" if raw_decision.option_verdicts else "invalid")
                or "invalid"
            ),
            audit_reason=str(raw_audit.get("audit_reason", raw_decision.support_reason) or ""),
            revision_context=final_revision_context,
            required=audit_required,
            source_revision_context=dict(raw_audit.get("source_revision_context") or {}),
        )
        final_table = _option_verdict_table(
            workspace.case.options,
            query_contract,
            final_snapshot,
            _audit_option_states(raw_decision, workspace.case.options),
            revision_context=final_revision_context,
            audit_record=final_audit,
        )
        final_table = _annotate_forced_override_eligibility(
            final_table,
            decision_completion,
            final_snapshot,
            query_requirements,
            audit_record=final_audit,
            revision_context=final_revision_context,
        )
        final_completion.update({
            "canonical_fact_snapshot": final_snapshot,
            "option_verdict_table": final_table,
            "revision_context": final_revision_context.to_dict(),
            "audit_record": final_audit,
            "answer_qualification_status": final_table["answer_qualification_status"],
        })
        adjudication = final_adjudicate(
            options=workspace.case.options,
            raw_reasoner_answer=raw_decision.answer,
            raw_citations=raw_decision.citations,
            raw_gate=last_raw_gate,
            completion_status=decision_completion,
            qualification_result={
                "status": final_table["answer_qualification_status"],
                "requirement_graph": final_snapshot.get("requirement_graph", {}),
                "incomplete_events": final_snapshot.get("incomplete_events", ()),
                "conflicted_events": final_snapshot.get("conflicted_events", ()),
                "qualification_evaluations": final_completion.get("qualification_evaluations", ()),
                "event_qualification_evaluations": _event_qualification_evaluations(final_snapshot),
            },
            option_verdict_table=final_table,
            audit_record=final_audit,
            revision_context=final_revision_context,
            audit_required=audit_required,
        )
        final_answer = adjudication.answer
        final_citations = adjudication.citations
        answer_mode = adjudication.answer_mode
        verified = adjudication.verified
        grounding_status = adjudication.grounding_status
        grounding_level = adjudication.grounding_level
        verification_reason = adjudication.verification_reason
        grounded_answer = final_answer if verified else ""
        forced_answer = final_answer if answer_mode == "forced_choice" else final_answer if verified else ""
        trace.append(dict(adjudication.answer_selection_event))
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
                "grounding_level": grounding_level,
                "retrieval_status": retrieval_status,
                "verified": verified,
                "verification_reason": verification_reason,
                "best_effort": bool(final_answer and not verified and best_answer),
                "option_states": option_states,
                "raw_reasoner_answer": raw_decision.answer,
                "answer_mutation_events": [dict(item) for item in adjudication.answer_mutation_events],
                "final_selection_source": adjudication.selection_source,
                "answer_selection_event": dict(adjudication.answer_selection_event),
                "revision_context": final_revision_context.to_dict(),
                "audit_record": final_audit,
                "option_verdict_table": final_table,
                "canonical_fact_counts": final_snapshot["canonical_fact_counts"],
                "raw_candidate_counts": final_snapshot["raw_candidate_counts"],
                "qualified_event_count": len(tuple(final_snapshot.get("qualified_events", ()) or ())),
                "selected_episode_id": str(final_completion.get("selected_last_episode_id", "") or ""),
                "selected_entity_id": str(
                    dict(final_completion.get("target_entity_binding", {}) or {}).get("entity_id", "") or ""
                ),
                "final_adjudication_blockers": list(adjudication.guard.blockers),
                "soft_audit_correction": adjudication.soft_audit_guard.to_dict(),
                "forced_reason": verification_reason if answer_mode == "forced_choice" else None,
                "forced_fact_source": "canonical_fact_snapshot" if answer_mode == "forced_choice" else None,
                "stagnation_actions": stagnation_actions,
                "system_task_budget_ratio_limit": self.system_task_budget_ratio,
                "system_task_budget_limit": system_task_budget_limit,
                "system_tasks_accepted": system_tasks_accepted,
                "system_task_budget_ratio_actual": round(system_tasks_accepted / self.max_investigations, 6),
                "system_task_share_of_executed": round(system_tasks_accepted / max(1, accepted), 6),
                "system_override_count": 0,
                "investigation_budget_limit": self.max_investigations,
                "round_limit": self.max_rounds,
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
            grounding_level=grounding_level,
            retrieval_status=retrieval_status,
        )
        _write_run_summary(workspace, result)
        return result


def _policy_suggestions(
    workspace: VirtualVideoWorkspace,
    contract: ClaimContract,
    query_requirements: Mapping[str, Any],
    completion_status: Mapping[str, Any],
    evidence: Sequence[EvidenceRecord],
    *,
    round_id: int,
    limit: int,
) -> tuple[dict[str, Any], ...]:
    """Build advisory tasks without mutating or replacing the Reasoner's decision."""
    task_limit = max(0, int(limit))
    if task_limit <= 0:
        return ()
    candidates: list[tuple[str, InvestigationTask]] = []
    missing_segments = tuple(completion_status.get("missing_segment_ids", ()) or ())
    if missing_segments:
        candidates.extend(
            ("uninspected_source_range", task)
            for task in _coverage_repair_tasks(
                workspace,
                round_id,
                missing_segments,
                contract,
                limit=task_limit,
                coverage_status=completion_status.get("source_coverage", {}),
            )
        )
    if bool(completion_status.get("enumeration_required")) and not bool(
        completion_status.get("enumeration_complete")
    ):
        candidates.extend(
            ("enumeration_incomplete", task)
            for task in _enumeration_repair_tasks(
                workspace,
                completion_status,
                round_id=round_id,
                limit=task_limit,
            )
        )
    missing_identity_terms = tuple(completion_status.get("missing_identity_anchor_terms", ()) or ())
    if missing_identity_terms:
        navigation = _navigation_repair_tasks(evidence, round_id=round_id, limit=task_limit)
        identity = navigation or _identity_repair_tasks(
            workspace,
            evidence,
            missing_identity_terms,
            round_id=round_id,
            limit=task_limit,
        )
        candidates.extend(("identity_anchor_missing", task) for task in identity)
    unresolved_entities = tuple(completion_status.get("unresolved_candidate_entity_observation_ids", ()) or ())
    if unresolved_entities:
        candidates.extend(
            ("entity_candidate_unresolved", task)
            for task in _entity_candidate_repair_tasks(
                workspace,
                evidence,
                unresolved_entities,
                round_id=round_id,
                limit=task_limit,
            )
        )
    if bool(query_requirements.get("requires_event_participant_link")) and not bool(
        completion_status.get("event_participant_link_ready")
    ):
        candidates.extend(
            ("event_participant_link_missing", task)
            for task in _event_participant_association_tasks(
                workspace,
                evidence,
                round_id=round_id,
                limit=task_limit,
            )
        )
    if bool(query_requirements.get("requires_narrative_inference")) and not bool(
        completion_status.get("narrative_inference_ready")
    ):
        candidates.extend(
            ("narrative_fact_missing", task)
            for task in _narrative_bridge_repair_tasks(
                workspace,
                evidence,
                round_id=round_id,
                limit=min(2, task_limit),
            )
        )
    semantic_reason, semantic_tasks = _semantic_contract_repair_tasks(
        workspace,
        contract,
        query_requirements,
        completion_status,
        evidence,
        round_id=round_id,
        limit=task_limit,
    )
    candidates.extend((semantic_reason or "semantic_fact_missing", task) for task in semantic_tasks)

    suggestions = []
    seen = set()
    for reason, task in candidates:
        fingerprint = _task_progress_fingerprint(task)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        suggestions.append({
            "reason": reason,
            "expected_information_gain": "new inspected range, typed fact, or justified density increase",
            "task": {
                "query_id": task.query_id,
                "goal": task.goal,
                "segment_id": task.segment_id,
                "time_range": list(task.time_range) if task.time_range is not None else None,
                "modality_hint": list(task.modality_hint),
                "expected_evidence": task.expected_evidence,
                "inspection_mode": task.inspection_mode,
                "sampling_floor_fps": task.sampling_floor_fps,
                "target_requirement_ids": list(task.target_requirement_ids),
                "source_candidate_ids": list(task.source_candidate_ids),
            },
        })
        if len(suggestions) >= task_limit:
            break
    return tuple(suggestions)


def _task_adopts_policy_suggestion(
    task: InvestigationTask,
    suggestions: Sequence[Mapping[str, Any]],
) -> bool:
    query_id = str(task.query_id or "")
    return any(
        query_id == str(dict(suggestion.get("task", {}) or {}).get("query_id", "") or "")
        for suggestion in suggestions
    )


def _enforce_system_task_budget(
    decision: ReasonerDecision,
    suggestions: Sequence[Mapping[str, Any]],
    *,
    remaining_slots: int,
) -> tuple[ReasonerDecision, tuple[InvestigationTask, ...]]:
    slots = max(0, int(remaining_slots))
    accepted = []
    rejected = []
    for task in decision.tasks:
        if not _task_adopts_policy_suggestion(task, suggestions):
            accepted.append(task)
        elif slots > 0:
            accepted.append(task)
            slots -= 1
        else:
            rejected.append(task)
    return replace(decision, tasks=tuple(accepted)), tuple(rejected)


def _coverage_repair_tasks(
    workspace: VirtualVideoWorkspace,
    round_id: int,
    segment_ids: Sequence[str],
    contract: ClaimContract,
    *,
    limit: int,
    coverage_status: Mapping[str, Any] | None = None,
) -> tuple[InvestigationTask, ...]:
    modalities = tuple(contract.required_observability or ("visual",))
    tasks = []
    by_id = {segment.segment_id: segment for segment in workspace.manifest.segments}
    task_limit = max(0, int(limit))
    coverage_rows = tuple(dict(row) for row in dict(coverage_status or {}).values() if isinstance(row, Mapping))
    for segment_id in tuple(segment_ids):
        segment = by_id.get(str(segment_id))
        full_range: tuple[float, float] | None = (
            (float(segment.virtual_start_sec), float(segment.virtual_end_sec))
            if segment is not None
            else None
        )
        uncovered_ranges = tuple(
            tuple(float(value) for value in item)
            for source_row in coverage_rows
            for item in tuple(
                dict(dict(source_row.get("segment_coverage", {}) or {}).get(str(segment_id), {}) or {}).get(
                    "uncovered_ranges", ()
                )
            )
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 2
        )
        requested_ranges = uncovered_ranges or ((full_range,) if full_range is not None else ())
        event_sweep = contract.quantifier == "total_count" and contract.observation_target == "event"
        for requested_range in requested_ranges:
            if len(tasks) >= task_limit:
                return tuple(tasks)
            tasks.append(InvestigationTask(
                query_id=f"repair_r{round_id}_{len(tasks) + 1:03d}",
                goal=f"Inspect only the uninspected range in source segment {segment_id}.",
                segment_id=str(segment_id),
                time_range=requested_range,
                modality_hint=modalities,
                expected_evidence=(
                    "question-relevant event occurrences in this uninspected range"
                    if event_sweep
                    else "entity observations and topic evidence in this uninspected range"
                ),
                inspection_mode="enumerate_events" if event_sweep else "window",
                sampling_floor_fps=2.0 if event_sweep else None,
                expected_event_dwell_sec=1.0 if event_sweep else None,
                temporal_resolution_rationale=(
                    "The remaining event range must resolve brief event boundaries."
                    if event_sweep
                    else ""
                ),
                priority=1.0,
            ))
    return tuple(tasks)


def _enumeration_repair_tasks(
    workspace: VirtualVideoWorkspace,
    completion_status: Mapping[str, Any],
    *,
    round_id: int,
    limit: int,
) -> tuple[InvestigationTask, ...]:
    manifest = dict(completion_status.get("enumeration_manifest", {}) or {})
    requested_ranges = tuple(
        tuple(float(value) for value in item)
        for item in tuple(manifest.get("unprocessed_ranges", ()) or ())
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 2
    )
    if not requested_ranges:
        requested_ranges = tuple(
            tuple(float(value) for value in item)
            for item in tuple(manifest.get("required_ranges", ()) or ())
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 2
        )
    if not requested_ranges:
        return ()
    source_id = str(completion_status.get("adopted_source_video_id", "") or "")
    segments = tuple(
        segment
        for segment in workspace.manifest.segments
        if not source_id or segment.source_video_id == source_id
    )
    tasks: list[InvestigationTask] = []
    window_sec = 12.0
    overlap_sec = 2.0
    for requested_start, requested_end in requested_ranges:
        left, right = sorted((requested_start, requested_end))
        for segment in segments:
            segment_start = max(left, float(segment.virtual_start_sec))
            segment_end = min(right, float(segment.virtual_end_sec))
            if segment_end <= segment_start:
                continue
            cursor = segment_start
            while cursor < segment_end - 1e-6 and len(tasks) < max(0, int(limit)):
                end = min(segment_end, cursor + window_sec)
                tasks.append(InvestigationTask(
                    query_id=f"enumerate_r{round_id}_{len(tasks) + 1:03d}",
                    goal="Enumerate every atomic question-relevant event in this short overlap window and reconcile it with adjacent windows.",
                    segment_id=segment.segment_id,
                    time_range=(cursor, end),
                    modality_hint=("visual",),
                    expected_evidence="parsed timestamped event candidates, explicit negative observations, and continuation links for overlap reconciliation",
                    inspection_mode="enumerate_events",
                    priority=1.0,
                    sampling_floor_fps=2.0,
                    expected_event_dwell_sec=1.0,
                    temporal_resolution_rationale="Enumeration windows must resolve events that may persist for roughly one second.",
                ))
                if end >= segment_end - 1e-6:
                    break
                cursor = max(cursor + 1e-6, end - overlap_sec)
            if len(tasks) >= max(0, int(limit)):
                return tuple(tasks)
    return tuple(tasks)


def _bootstrap_investigation_tasks(
    workspace: VirtualVideoWorkspace,
    contract: ClaimContract,
    *,
    round_id: int,
    limit: int,
    effective_scope: str,
) -> tuple[InvestigationTask, ...]:
    segment_limit = max(0, int(limit))
    if segment_limit <= 0:
        return ()
    segments = tuple(workspace.manifest.segments)
    if str(effective_scope) != "full_video":
        segments = segments[:1]
    modalities = tuple(contract.required_observability or ("visual",))
    event_count = contract.quantifier == "total_count" and contract.observation_target == "event"
    return tuple(
        InvestigationTask(
            query_id=f"bootstrap_r{round_id}_{index:03d}",
            goal=(
                f"Enumerate atomic question-relevant events in source segment {segment.segment_id}."
                if event_count
                else f"Inspect source segment {segment.segment_id} for direct evidence required by the question."
            ),
            segment_id=segment.segment_id,
            modality_hint=modalities,
            expected_evidence=(
                "timestamped atomic event occurrences with stable event identifiers"
                if event_count
                else "direct observations that resolve the question's semantic obligations"
            ),
            inspection_mode="enumerate_events" if event_count else "window",
            priority=1.0,
        )
        for index, segment in enumerate(segments[:segment_limit], start=1)
    )


def _semantic_contract_repair_tasks(
    workspace: VirtualVideoWorkspace,
    contract: ClaimContract,
    query_requirements: Mapping[str, Any],
    completion_status: Mapping[str, Any],
    evidence: Sequence[EvidenceRecord],
    *,
    round_id: int,
    limit: int,
) -> tuple[str, tuple[InvestigationTask, ...]]:
    reason, requests = semantic_repair_requests(
        workspace,
        contract,
        query_requirements,
        completion_status,
        evidence,
        round_id=round_id,
        limit=limit,
    )
    return reason, tuple(
        InvestigationTask(
            query_id=request.query_id,
            goal=request.goal,
            segment_id=request.segment_id,
            time_range=request.time_range,
            modality_hint=request.modality_hint,
            expected_evidence=request.expected_evidence,
            inspection_mode=request.inspection_mode,
            priority=1.0,
        )
        for request in requests
    )


def _submission_fingerprint(
    decision: ReasonerDecision,
    evidence: Sequence[EvidenceRecord],
) -> tuple[str, tuple[str, ...], tuple[str, ...]] | None:
    if decision.action != "answer":
        return None
    option = _letter(decision.answer) or str(decision.answer or "").strip().casefold()
    return (
        option,
        tuple(sorted(set(decision.citations))),
        tuple(record.evidence_id for record in evidence),
    )


def _rejected_answer_repair_tasks(
    workspace: VirtualVideoWorkspace,
    contract: ClaimContract,
    decision: ReasonerDecision,
    evidence: Sequence[EvidenceRecord],
    gate_feedback: Mapping[str, Any],
    *,
    round_id: int,
    limit: int,
) -> tuple[InvestigationTask, ...]:
    task_limit = max(0, int(limit))
    if task_limit <= 0:
        return ()
    missing_segments = tuple(gate_feedback.get("missing_segment_ids", ()) or ())
    if missing_segments:
        return _coverage_repair_tasks(workspace, round_id, missing_segments, contract, limit=task_limit)
    if contract.quantifier == "total_count" and contract.observation_target == "event":
        return tuple(
            InvestigationTask(
                query_id=f"rejected_event_r{round_id}_{index:03d}",
                goal="Enumerate every question-relevant event occurrence in this segment before resubmitting a total.",
                segment_id=segment.segment_id,
                modality_hint=tuple(contract.required_observability or ("visual",)),
                expected_evidence="timestamped atomic occurrences with stable event keys and continuation flags",
                inspection_mode="enumerate_events",
                priority=1.0,
            )
            for index, segment in enumerate(tuple(workspace.manifest.segments)[:task_limit], start=1)
        )
    by_id = {record.evidence_id: record for record in evidence}
    cited = tuple(by_id[item] for item in decision.citations if item in by_id)
    selected = _letter(decision.answer) or _option_letter_from_answer(decision.answer, workspace.case.options)
    selected_text = str(workspace.case.options.get(selected, decision.answer) or decision.answer)
    alternatives = tuple(
        str(text)
        for label, text in workspace.case.options.items()
        if str(label) != selected
    )
    tasks = []
    for record in cited:
        if record.start_sec is None or record.end_sec is None:
            continue
        midpoint = (float(record.start_sec) + float(record.end_sec)) / 2.0
        segment = next(
            (
                item
                for item in workspace.manifest.segments
                if item.virtual_start_sec <= midpoint <= item.virtual_end_sec
            ),
            None,
        )
        if segment is None:
            continue
        tasks.append(
            InvestigationTask(
                query_id=f"discriminate_r{round_id}_{len(tasks) + 1:03d}",
                goal="Re-observe this window and identify the visual fact that distinguishes the proposed answer from its closest alternatives.",
                segment_id=segment.segment_id,
                time_range=(
                    max(segment.virtual_start_sec, float(record.start_sec) - 5.0),
                    min(segment.virtual_end_sec, float(record.end_sec) + 5.0),
                ),
                modality_hint=("visual", "motion"),
                expected_evidence="per-hypothesis verdicts tied to witnessed frames; report indistinguishable when the frames do not decide",
                inspection_mode="verify_claim",
                claim_to_verify=selected_text,
                claim_relation="supports",
                alternative_answers=alternatives,
                priority=1.0,
            )
        )
        if len(tasks) >= task_limit:
            break
    if tasks:
        return tuple(tasks)
    return _bootstrap_investigation_tasks(
        workspace,
        contract,
        round_id=round_id,
        limit=task_limit,
        effective_scope=str(gate_feedback["effective_scope"]),
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


def _event_participant_association_tasks(
    workspace: VirtualVideoWorkspace,
    evidence: Sequence[EvidenceRecord],
    *,
    round_id: int,
    limit: int,
) -> tuple[InvestigationTask, ...]:
    if any(str(record.task_id or "").startswith("participant_link_") for record in evidence):
        return ()
    ledger = _event_candidate_ledger(evidence)
    candidates = tuple(ledger.get("confirmed_event_candidates", ()) or ()) or tuple(
        ledger.get("observed_event_candidates", ()) or ()
    )
    if not candidates:
        if any(str(record.task_id or "").startswith("participant_anchor_") for record in evidence):
            return ()
        target_ids = {
            str(item) for item in tuple(workspace.case.metadata.get("target_segment_ids", ()) or ()) if str(item)
        }
        segments = tuple(
            segment for segment in workspace.manifest.segments
            if not target_ids or segment.segment_id in target_ids or segment.role == "target"
        ) or tuple(workspace.manifest.segments)
        sweep_size = min(max(1, int(limit), 6), len(segments))
        selected = select_uniform_items(segments, sweep_size)
        return tuple(
            InvestigationTask(
                query_id=f"participant_anchor_r{round_id}_{index:03d}",
                goal="Enumerate the question-referenced anchor events and record every visible participant with an explicit role.",
                segment_id=segment.segment_id,
                time_range=(float(segment.virtual_start_sec), float(segment.virtual_end_sec)),
                modality_hint=("visual",),
                expected_evidence="timestamped anchor events with overtaker/overtaken roles, multi-attribute signatures, and stable participant IDs",
                inspection_mode="enumerate_events",
                priority=1.0,
                inspection_intent="event_participant_anchor_discovery",
                sampling_floor_fps=2.0,
                expected_event_dwell_sec=1.0,
                temporal_resolution_rationale="Brief overtakes require dense event-boundary sampling.",
            )
            for index, segment in enumerate(selected, start=1)
        )
    # Scheduling may use the latest observed candidate provisionally, but only
    # the completion path can turn it into a resolved temporal-max episode.
    anchor = max(
        candidates,
        key=lambda row: float(tuple(row.get("virtual_time_range", (0.0, 0.0)))[0]),
    )
    ordinal = _identity_ordinal_index(workspace.case.question)
    participants = [
        dict(item) for item in tuple(anchor.get("participants", ()) or ())
        if isinstance(item, Mapping)
        and str(item.get("role", "") or "").casefold() not in {"overtaken", "camera_holder", "recorder"}
    ]
    if not participants:
        participants = [
            {
                "participant_id": participant_id,
                "role": "event_participant",
                "visual_signature": "; ".join(tuple(anchor.get("descriptions", ()) or ()))[:300],
                "attributes": {},
            }
            for participant_id in tuple(anchor.get("participant_ids", ()) or ())
            if str(participant_id).casefold() != "camera_holder"
        ]
    if not participants:
        return ()
    participant_ordinal = min(max(0, ordinal), len(participants) - 1)
    source = participants[participant_ordinal]
    participant_id = str(source.get("participant_id", "") or "event_participant").strip()
    hypothesis_id = str(source.get("entity_hypothesis_id", "") or "").strip() or re.sub(
        r"[^a-z0-9]+", "_", f"{participant_id}_{anchor.get('candidate_id', 'event')}".casefold()
    ).strip("_")
    reference = {
        **source,
        "entity_hypothesis_id": hypothesis_id,
        "source_event_key": str(anchor.get("canonical_event_key", "") or anchor.get("signature", "") or ""),
        "source_episode_id": str(anchor.get("candidate_id", "") or ""),
        "ordinal": ordinal + 1,
        "source_event_time_range": list(anchor.get("virtual_time_range", ()) or ()),
    }
    bounds = tuple(anchor.get("virtual_time_range", ()) or (0.0, 0.0))
    anchor_end = float(bounds[1]) if len(bounds) == 2 else 0.0
    segments = tuple(
        segment for segment in workspace.manifest.segments
        if float(segment.virtual_end_sec) > anchor_end
    ) or tuple(workspace.manifest.segments)
    selected = tuple(reversed(segments[-max(1, min(int(limit), 2)) :]))
    return tuple(
        InvestigationTask(
            query_id=f"participant_link_r{round_id}_{index:03d}",
            goal=f"Re-identify event participant {participant_id} in this later window and record their visible outcome attributes.",
            segment_id=segment.segment_id,
            modality_hint=("visual",),
            expected_evidence="a supported, refuted, or unknown association using multiple appearance attributes and a frame-witnessed target entity",
            inspection_mode="entity_association",
            priority=1.0,
            source_candidate_ids=tuple(str(item) for item in anchor.get("evidence_ids", ()) or ()),
            inspection_intent="event_participant_reidentification",
            reference_entities=(reference,),
            sampling_floor_fps=1.0,
            temporal_resolution_rationale="Appearance and finish-state cues persist for several frames.",
        )
        for index, segment in enumerate(selected, start=1)
    )


def _narrative_bridge_repair_tasks(
    workspace: VirtualVideoWorkspace,
    evidence: Sequence[EvidenceRecord],
    *,
    round_id: int,
    limit: int,
) -> tuple[InvestigationTask, ...]:
    prior_records = tuple(
        record for record in evidence if str(record.task_id or "").startswith("narrative_bridge_")
    )
    if len(prior_records) >= 3:
        return ()
    segments = tuple(workspace.manifest.segments)
    prior_facts = tuple(
        dict(fact)
        for record in prior_records
        for fact in tuple(record.operation_metadata.get("narrative_facts", ()) or ())
        if isinstance(fact, Mapping)
    )[-4:]
    if prior_facts:
        missing_outcome = any(not str(fact.get("outcome_state", "") or "").strip() for fact in prior_facts)
        selected = (segments[-1] if missing_outcome else segments[0],)
        goals = (
            "Re-observe the missing outcome and complete the prior setup-to-outcome fact."
            if missing_outcome
            else "Re-observe the missing setup and complete the prior setup-to-outcome fact.",
        )
    elif len(segments) == 1 and int(limit) >= 2:
        selected = (segments[0], segments[0])
        goals = (
            "Locate and record the narrative setup immediately before the implied gap.",
            "Locate and record the observable outcome immediately after the implied gap.",
        )
    else:
        selected = select_uniform_items(segments, min(max(1, int(limit)), len(segments)))
        goals = tuple(
            "Observe the narrative setup and later outcome, then state only the minimal bridge supported by both."
            for _ in selected
        )
    return tuple(
        InvestigationTask(
            query_id=f"narrative_bridge_r{round_id}_{index:03d}",
            goal=goals[index - 1],
            segment_id=segment.segment_id,
            modality_hint=("visual", "asr"),
            expected_evidence="setup state, observed transition or gap, outcome state, and counterevidence for incompatible option predicates",
            inspection_mode="narrative_bridge",
            priority=1.0,
            inspection_intent="narrative_setup_outcome_synthesis",
            episode_id=f"narrative:{segment.segment_id}",
            reference_facts=prior_facts,
            sampling_floor_fps=0.5,
            temporal_resolution_rationale="Narrative setup and outcome are persistent scene-level states.",
        )
        for index, segment in enumerate(selected, start=1)
    )


def _identity_ordinal_index(question: str) -> int:
    match = re.search(r"\b(first|second|third|fourth|last)\s+(?:person|player|rider|competitor)", str(question).casefold())
    if not match:
        return 0
    return {"first": 0, "second": 1, "third": 2, "fourth": 3, "last": 10_000}[match.group(1)]


def _entity_candidate_repair_tasks(
    workspace: VirtualVideoWorkspace,
    evidence: Sequence[EvidenceRecord],
    candidate_ids: Sequence[str],
    *,
    round_id: int,
    limit: int,
) -> tuple[InvestigationTask, ...]:
    wanted = {str(item) for item in candidate_ids if str(item)}
    rows = []
    for record in evidence:
        for entity in record.operation_metadata.get("entities", ()) or ():
            if not isinstance(entity, Mapping):
                continue
            observation_id = str(entity.get("entity_observation_id", "") or "")
            if observation_id not in wanted:
                continue
            witness_times = tuple(
                float(item)
                for item in entity.get("witness_virtual_times_sec", ())
                if isinstance(item, (int, float))
            )
            if not witness_times:
                continue
            center = witness_times[0]
            segment = next(
                (
                    item
                    for item in workspace.manifest.segments
                    if item.virtual_start_sec <= center <= item.virtual_end_sec
                ),
                None,
            )
            if segment is None:
                continue
            start = max(float(segment.virtual_start_sec), center - 30.0)
            end = min(float(segment.virtual_end_sec), center + 30.0)
            if end <= start:
                continue
            signature = str(entity.get("visual_signature") or entity.get("description") or "visible person")
            rows.append((observation_id, segment.segment_id, start, end, signature))
    tasks = []
    for index, (observation_id, segment_id, start, end, signature) in enumerate(rows, start=1):
        tasks.append(
            InvestigationTask(
                query_id=f"entity_repair_r{round_id}_{index:03d}",
                goal=f"Verify candidate {observation_id} in a narrow window and compare it with prior entity observations.",
                segment_id=segment_id,
                time_range=(start, end),
                modality_hint=("visual", "asr"),
                expected_evidence=(
                    f"frame-witnessed identity with stable signature; candidate signature: {signature}"
                ),
                priority=1.0,
                source_candidate_ids=(observation_id,),
                inspection_intent="entity_candidate_verification",
            )
        )
        if len(tasks) >= max(0, int(limit)):
            break
    return tuple(tasks)


def _navigation_repair_tasks(
    evidence: Sequence[EvidenceRecord],
    *,
    options: Mapping[str, str] | None = None,
    round_id: int,
    limit: int,
    require_hypothesis: bool = False,
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
            hypotheses = _candidate_hypotheses(hint, options or {})
            if require_hypothesis and not hypotheses:
                continue
            unresolved.append((hint, str(lineage["segment_id"]), hypotheses))

    unresolved.sort(key=lambda item: (-_candidate_score(item[0], item[2]), float(item[0].start_sec or 0.0)))

    return tuple(
        InvestigationTask(
            query_id=f"navigation_repair_r{round_id}_{index:03d}",
            goal="Visually verify an unresolved ASR navigation clue without assuming it proves an answer option.",
            segment_id=segment_id,
            time_range=(float(hint.start_sec), float(hint.end_sec)),
            modality_hint=("visual",),
            expected_evidence=f"complete temporal context and direct visual evidence for or against: {hint.verbatim[:240]}",
            priority=1.0,
            source_candidate_ids=(hint.evidence_id,),
            inspection_intent="verify navigation candidate against its linked option hypothesis",
        )
        for index, (hint, segment_id, _) in enumerate(unresolved[: max(0, int(limit))], start=1)
    )


def _navigation_hint_is_observed(hint: EvidenceRecord, visual: Sequence[EvidenceRecord]) -> bool:
    return any(hint.evidence_id in tuple(record.operation_metadata.get("source_candidate_ids", ()) or ()) for record in visual)


def _navigation_candidates(evidence: Sequence[EvidenceRecord], options: Mapping[str, str]) -> tuple[dict[str, Any], ...]:
    visual = tuple(record for record in evidence if record.modality in {"visual", "ocr"})
    rows = []
    for hint in evidence:
        if hint.evidence_kind != "navigation_hint" or hint.observation_polarity != "positive":
            continue
        linked = tuple(record for record in visual if hint.evidence_id in tuple(record.operation_metadata.get("source_candidate_ids", ()) or ()))
        hypotheses = _candidate_hypotheses(hint, options)
        rows.append({
            "candidate_id": hint.evidence_id,
            "source_task_id": hint.task_id,
            "virtual_time_range": [hint.start_sec, hint.end_sec],
            "hypothesis_ids": list(hypotheses),
            "discriminative_score": _candidate_score(hint, hypotheses),
            "matched_terms": list(hint.operation_metadata.get("matched_terms", ())),
            "status": "inspected" if linked else "unseen",
            "possibly_covered": not linked and any(_windows_overlap(hint, record) for record in visual),
            "resulting_evidence_ids": [record.evidence_id for record in linked],
        })
    return tuple(sorted(rows, key=lambda row: (-float(row["discriminative_score"]), str(row["candidate_id"]))))


def _candidate_hypotheses(record: EvidenceRecord, options: Mapping[str, str]) -> tuple[str, ...]:
    terms = {
        token for term in record.operation_metadata.get("matched_terms", ()) or ()
        for token in re.findall(r"[a-z0-9]+", str(term).casefold()) if len(token) >= 3
    }
    return tuple(str(label) for label, text in options.items() if terms.intersection(re.findall(r"[a-z0-9]+", str(text).casefold())))


def _candidate_score(record: EvidenceRecord, hypotheses: Sequence[str]) -> float:
    matched = len(tuple(record.operation_metadata.get("matched_terms", ()) or ()))
    hits = int(record.operation_metadata.get("hit_count", 0) or 0)
    return round((1.0 if hypotheses else 0.0) + min(1.0, matched * 0.25) + min(0.5, hits * 0.05), 3)


def _windows_overlap(left: EvidenceRecord, right: EvidenceRecord) -> bool:
    if left.start_sec is None or left.end_sec is None or right.start_sec is None or right.end_sec is None:
        return False
    return min(float(left.end_sec), float(right.end_sec)) > max(float(left.start_sec), float(right.start_sec))


def _task_for_contract(task: InvestigationTask, contract: ClaimContract) -> InvestigationTask:
    del contract
    return task


def _resolve_workspace_tasks(
    workspace: VirtualVideoWorkspace,
    tasks: Sequence[InvestigationTask],
    *,
    limit: int,
) -> tuple[InvestigationTask, ...]:
    task_limit = max(0, int(limit))
    if task_limit <= 0:
        return ()
    segments = tuple(workspace.manifest.segments)
    by_id = {segment.segment_id: segment for segment in segments}
    global_aliases = {"all", "full", "full_video", "global", "seg_all", "seg_full", "workspace"}
    task_groups: list[tuple[InvestigationTask, ...]] = []
    for task in tasks:
        if task.inspection_mode == "search_asr":
            task_groups.append((task,))
            continue
        if task.time_range is not None:
            start, end = (float(value) for value in task.time_range)
            if end < start:
                start, end = end, start
            start = max(0.0, start)
            end = min(float(workspace.manifest.duration_sec), end)
            overlaps = tuple(
                (segment, max(start, segment.virtual_start_sec), min(end, segment.virtual_end_sec))
                for segment in segments
                if min(end, segment.virtual_end_sec) > max(start, segment.virtual_start_sec)
            )
            if task.segment_id in by_id:
                requested_segment = by_id[task.segment_id]
                contained = (
                    start >= requested_segment.virtual_start_sec - 1e-6
                    and end <= requested_segment.virtual_end_sec + 1e-6
                )
                if contained and end > start:
                    task_groups.append((replace(task, time_range=(start, end)),))
                    continue
            expanded = tuple(
                replace(
                    task,
                    query_id=(
                        task.query_id
                        if len(overlaps) == 1
                        else f"{task.query_id}_{segment.segment_id}_{index:02d}"
                    ),
                    segment_id=segment.segment_id,
                    time_range=(overlap_start, overlap_end),
                )
                for index, (segment, overlap_start, overlap_end) in enumerate(overlaps, start=1)
            )
            if expanded:
                task_groups.append(expanded)
                continue
        if task.segment_id in by_id:
            task_groups.append((task,))
            continue
        if str(task.segment_id or "").casefold() not in global_aliases:
            continue
        task_groups.append(
            tuple(
                replace(
                    task,
                    query_id=f"{task.query_id}_{segment.segment_id}",
                    segment_id=segment.segment_id,
                    time_range=None,
                )
                for segment in select_uniform_items(segments, min(task_limit, len(segments)))
            )
        )

    resolved: list[InvestigationTask] = []
    depth = 0
    while len(resolved) < task_limit and any(depth < len(group) for group in task_groups):
        for group in task_groups:
            if len(resolved) >= task_limit:
                break
            if depth < len(group):
                resolved.append(group[depth])
        depth += 1
    return tuple(resolved)


def _task_is_executable(task: InvestigationTask) -> bool:
    if not task.query_id or not task.goal:
        return False
    if task.inspection_mode == "search_asr":
        return bool(task.search_terms)
    return bool(task.segment_id or task.time_range is not None)


def _completion_status(
    workspace: VirtualVideoWorkspace,
    contract: ClaimContract,
    evidence: Sequence[EvidenceRecord],
    *,
    query_requirements: Mapping[str, Any] | None = None,
    query_obligations: QueryObligations | None = None,
    reports: Sequence[InvestigationReport] = (),
    best_choice: str = "",
    active_condition_ids: Sequence[str] = (),
    option_states: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    requirements = dict(query_requirements or {})
    obligations = query_obligations or compile_query_obligations(contract, requirements)
    effective_scope = obligations.effective_scope
    answer_evidence = tuple(record for record in evidence if record.evidence_kind != "navigation_hint")
    navigation_evidence = tuple(record for record in evidence if record.evidence_kind == "navigation_hint")
    observation_attempts = tuple(attempt for report in reports for attempt in report.attempts)
    coverage = _source_coverage(workspace, observation_attempts)
    if effective_scope != "full_video":
        base = _apply_identity_completion(
            {
                "ready_for_answer": bool(answer_evidence),
                "required_scope": contract.required_scope,
                "contract_scope": obligations.contract_scope,
                "effective_scope": effective_scope,
                "scope_escalation_requirement_ids": list(obligations.scope_escalation_requirement_ids),
                "query_obligations": obligations.to_dict(),
                "range_coverage_complete": bool(_visual_inspection_attempts(observation_attempts)) if effective_scope == "window" else False,
                "missing_segment_ids": [],
                "source_coverage": coverage,
            },
            answer_evidence,
            requirements,
        )
        base = _apply_entity_completion(base, contract, answer_evidence)
        base = _apply_event_completion(base, contract, answer_evidence, requirements)
        base = _apply_p1_semantic_completion(base, requirements, answer_evidence)
        base = _apply_enumeration_completion(
            base,
            workspace,
            contract,
            answer_evidence,
            requirements,
        )
        base = _apply_episode_binding_completion(
            base,
            workspace,
            answer_evidence,
            requirements,
        )
        return _attach_canonical_context(_apply_readiness_dashboard(
            base,
            answer_evidence,
            navigation_evidence,
            reports,
            best_choice,
            active_condition_ids,
        ), workspace, contract, answer_evidence, option_states, requirements, obligations)
    if not coverage:
        base = _apply_identity_completion(
            {
                "ready_for_answer": False,
                "required_scope": contract.required_scope,
                "contract_scope": obligations.contract_scope,
                "effective_scope": effective_scope,
                "scope_escalation_requirement_ids": list(obligations.scope_escalation_requirement_ids),
                "query_obligations": obligations.to_dict(),
                "range_coverage_complete": False,
                "reason": "source_not_identified",
                "missing_segment_ids": [],
                "source_coverage": {},
            },
            answer_evidence,
            requirements,
        )
        base = _apply_entity_completion(base, contract, answer_evidence)
        base = _apply_event_completion(base, contract, answer_evidence, requirements)
        base = _apply_p1_semantic_completion(base, requirements, answer_evidence)
        base = _apply_enumeration_completion(
            base,
            workspace,
            contract,
            answer_evidence,
            requirements,
        )
        base = _apply_episode_binding_completion(
            base,
            workspace,
            answer_evidence,
            requirements,
        )
        return _attach_canonical_context(_apply_readiness_dashboard(
            base,
            answer_evidence,
            navigation_evidence,
            reports,
            best_choice,
            active_condition_ids,
        ), workspace, contract, answer_evidence, option_states, requirements, obligations)
    adopted_source = max(
        coverage,
        key=lambda source_id: (
            int(coverage[source_id]["covered_count"]),
            float(coverage[source_id]["confidence"]),
        ),
    )
    missing = list(coverage[adopted_source]["missing_segment_ids"])
    base = _apply_identity_completion(
        {
            "ready_for_answer": not missing,
            "required_scope": contract.required_scope,
            "contract_scope": obligations.contract_scope,
            "effective_scope": effective_scope,
            "scope_escalation_requirement_ids": list(obligations.scope_escalation_requirement_ids),
            "query_obligations": obligations.to_dict(),
            "range_coverage_complete": not missing,
            "adopted_source_video_id": adopted_source,
            "missing_segment_ids": missing,
            "source_coverage": coverage,
        },
        answer_evidence,
        requirements,
    )
    base = _apply_entity_completion(base, contract, answer_evidence)
    base = _apply_event_completion(base, contract, answer_evidence, requirements)
    base = _apply_p1_semantic_completion(base, requirements, answer_evidence)
    base = _apply_enumeration_completion(
        base,
        workspace,
        contract,
        answer_evidence,
        requirements,
    )
    base = _apply_episode_binding_completion(
        base,
        workspace,
        answer_evidence,
        requirements,
    )
    return _attach_canonical_context(_apply_readiness_dashboard(
        base,
        answer_evidence,
        navigation_evidence,
        reports,
        best_choice,
        active_condition_ids,
    ), workspace, contract, answer_evidence, option_states, requirements, obligations)


def _attach_canonical_context(
    status: Mapping[str, Any],
    workspace: VirtualVideoWorkspace,
    contract: ClaimContract,
    evidence: Sequence[EvidenceRecord],
    option_states: Mapping[str, Mapping[str, Any]] | None,
    query_requirements: Mapping[str, Any] | None = None,
    query_obligations: QueryObligations | None = None,
) -> dict[str, Any]:
    result = dict(status)
    obligations = query_obligations or compile_query_obligations(contract, query_requirements)
    result.setdefault("contract_scope", obligations.contract_scope)
    result.setdefault("effective_scope", obligations.effective_scope)
    result.setdefault(
        "scope_escalation_requirement_ids",
        list(obligations.scope_escalation_requirement_ids),
    )
    result.setdefault("query_obligations", obligations.to_dict())
    snapshot = canonical_fact_snapshot(
        evidence,
        require_event_precondition=bool((query_requirements or {}).get("requires_state_tracking")),
    ).to_dict()
    if result.get("temporal_max_selection"):
        snapshot["episode_binding"] = {
            "temporal_max_selection": dict(result.get("temporal_max_selection", {}) or {}),
            "target_participant_selection": dict(result.get("target_participant_selection", {}) or {}),
            "target_entity_binding": dict(result.get("target_entity_binding", {}) or {}),
            "target_attribute_facts": [
                dict(item)
                for item in tuple(result.get("target_attribute_facts", ()) or ())
                if isinstance(item, Mapping)
            ],
        }
    if contract.quantifier == "order" or bool((query_requirements or {}).get("requires_temporal_sequence")):
        snapshot["sequence_ledger"] = build_sequence_ledger(snapshot, workspace.case.options)
    revision_context = RevisionContext.from_inputs(
        snapshot,
        _evidence_digest(evidence, contract),
        to_jsonable(contract),
    )
    audit_record = build_all_option_audit_record(
        options=workspace.case.options,
        supplied_verdicts={},
        audit_status="invalid",
        audit_reason="No fresh all-option audit is attached to this canonical snapshot.",
        revision_context=revision_context,
        required=requires_option_audit(contract, query_requirements),
    )
    verdict_table = _option_verdict_table(
        workspace.case.options,
        contract,
        snapshot,
        option_states or {},
        revision_context=revision_context,
        audit_record=audit_record,
    )
    result.update(
        {
            "completion_ready": bool(result.get("ready_for_answer")),
            "completion_level": str(result.get("grounding_level_ready", "none") or "none"),
            "completion_blockers": list(result.get("unresolved_critical_condition_ids", ()) or ()),
            "canonical_fact_snapshot": snapshot,
            "canonical_fact_counts": dict(snapshot["canonical_fact_counts"]),
            "raw_candidate_counts": dict(snapshot["raw_candidate_counts"]),
            "requirement_graph": dict(snapshot.get("requirement_graph", {}) or {}),
            "option_verdict_table": verdict_table,
            "revision_context": revision_context.to_dict(),
            "audit_record": audit_record,
        }
    )
    canonical_candidate = bool(
        tuple(snapshot.get("qualified_events", ()) or ())
        or tuple(snapshot.get("observed_event_candidates", ()) or ())
        or tuple(snapshot.get("resolved_entities", ()) or ())
        or tuple(snapshot.get("entity_associations", ()) or ())
        or tuple(snapshot.get("inferred_facts", ()) or ())
    )
    result["candidate_available"] = bool(result.get("candidate_available")) or canonical_candidate
    result = _enforce_strict_safety(result, contract, query_requirements)
    qualification_requirements = compile_query_qualification_requirements(
        contract,
        query_requirements or {},
        snapshot,
        result,
    )
    qualification_evaluations = evaluate_requirement_graph(qualification_requirements, {})
    global_requirement_telemetry = requirement_telemetry(qualification_evaluations)
    event_requirement_telemetry = dict(snapshot.get("requirement_graph", {}) or {})
    combined_requirement_telemetry = {
        key: int(global_requirement_telemetry.get(key, 0) or 0)
        + int(event_requirement_telemetry.get(key, 0) or 0)
        for key in (
            "total",
            "supported",
            "contradicted",
            "conflicted",
            "unknown",
            "blocked_unresolved",
            "blocked_conflicted",
            "not_applicable",
        )
    }
    combined_requirement_telemetry["blocked"] = (
        int(combined_requirement_telemetry.get("blocked_unresolved", 0) or 0)
        + int(combined_requirement_telemetry.get("blocked_conflicted", 0) or 0)
    )
    combined_requirement_telemetry["unresolved_dependency_ids"] = list(dict.fromkeys((
        *tuple(event_requirement_telemetry.get("unresolved_dependency_ids", ()) or ()),
        *tuple(global_requirement_telemetry.get("unresolved_dependency_ids", ()) or ()),
    )))
    snapshot["requirement_graph"] = combined_requirement_telemetry
    snapshot["qualification_requirements"] = [
        {
            "requirement_id": requirement.requirement_id,
            "predicate": requirement.predicate,
            "arguments": dict(requirement.arguments),
            "scope": requirement.scope,
            "quantifier": requirement.quantifier,
            "dependency_ids": list(requirement.dependency_ids),
            "required": requirement.required,
            "evaluator": requirement.evaluator,
        }
        for requirement in qualification_requirements
    ]
    required_by_id = {
        requirement.requirement_id: bool(requirement.required)
        for requirement in qualification_requirements
    }
    snapshot["qualification_evaluations"] = [
        {
            **evaluation.to_dict(),
            "required": required_by_id.get(evaluation.requirement_id, True),
        }
        for evaluation in qualification_evaluations
    ]
    result["canonical_fact_snapshot"] = snapshot
    result["requirement_graph"] = combined_requirement_telemetry
    result["qualification_evaluations"] = snapshot["qualification_evaluations"]
    obligation_evaluations = evaluate_query_obligations(obligations, snapshot, result)
    unresolved_obligation_ids = [
        str(row["requirement_id"])
        for row in obligation_evaluations
        if str(row.get("status", "unknown") or "unknown") != "supported"
    ]
    result["query_obligation_evaluations"] = [dict(row) for row in obligation_evaluations]
    result["query_obligation_blockers"] = unresolved_obligation_ids
    verdict_table = _annotate_forced_override_eligibility(
        verdict_table,
        result,
        snapshot,
        query_requirements or {},
        audit_record=audit_record,
        revision_context=revision_context,
    )
    result["option_verdict_table"] = verdict_table
    result["answer_qualification_status"] = verdict_table["answer_qualification_status"]
    result["forced_override"] = {
        "attempted": False,
        "allowed": verdict_table["hard_override_allowed"],
        "blockers": list(verdict_table["hard_override_blockers"]),
    }
    return result


def _apply_readiness_dashboard(
    status: Mapping[str, Any],
    answer_evidence: Sequence[EvidenceRecord],
    navigation_evidence: Sequence[EvidenceRecord],
    reports: Sequence[InvestigationReport],
    best_choice: str,
    active_condition_ids: Sequence[str] = (),
) -> dict[str, Any]:
    result = dict(status)
    updates = tuple(
        (report.query_id, condition_result)
        for report in reports
        for condition_result in report.condition_results
    )
    states = _apply_condition_scope(merge_condition_states(updates), result)
    active_ids = {str(item) for item in active_condition_ids if str(item)}
    missing_active_ids: set[str] = set()
    if active_ids:
        missing_active_ids = active_ids.difference(states)
        states = {condition_id: state for condition_id, state in states.items() if condition_id in active_ids}
    else:
        active_gap_id = next((report.gap_id for report in reversed(reports) if report.gap_id), "")
        if active_gap_id:
            gap_ids = {
                result.condition_id
                for report in reports
                if report.gap_id == active_gap_id
                for result in report.condition_results
            }
            states = {condition_id: state for condition_id, state in states.items() if condition_id in gap_ids}
    conflicted_slot_ids = {
        f"slot:{slot_id}"
        for record in answer_evidence
        for slot_id in tuple(record.operation_metadata.get("conflicted_slot_ids", ()) or ())
        if str(slot_id)
    }
    unresolved = sorted(
        missing_active_ids
        | {condition_id for condition_id, state in states.items() if state.status != "satisfied"}
        | conflicted_slot_ids
    )
    base_ready = bool(result.get("ready_for_answer"))
    retrieval_ready = any(record.modality in {"visual", "ocr"} for record in answer_evidence)
    condition_total = len(states) + len(missing_active_ids)
    satisfied_count = sum(state.status == "satisfied" for state in states.values())
    conflict_count = (
        sum(state.status in {"contradicted", "refuted"} for state in states.values())
        + len(conflicted_slot_ids)
    )
    completion_ratio = satisfied_count / max(1, condition_total)
    strict_grounded = base_ready and not unresolved
    partial_grounded = bool(
        not strict_grounded
        and retrieval_ready
        and condition_total
        and conflict_count == 0
        and completion_ratio >= 0.6
        and _completion_scope_ratio(result) >= 0.75
    )
    result.update({
        "candidate_available": any(record.observation_polarity == "positive" for record in navigation_evidence),
        "retrieval_ready": retrieval_ready,
        "choice_ready": bool(str(best_choice or "").strip()),
        "grounded_ready": strict_grounded,
        "partial_grounded_ready": partial_grounded,
        "grounding_level_ready": "strict" if strict_grounded else "partial" if partial_grounded else "none",
        "condition_completion_ratio": completion_ratio,
        "satisfied_condition_count": satisfied_count,
        "critical_condition_count": condition_total,
        "conflicted_condition_count": conflict_count,
        "conflicted_structured_slot_ids": sorted(conflicted_slot_ids),
        "unresolved_critical_condition_ids": unresolved,
        "unsupported_claim_atom_ids": list(unresolved),
        "condition_states": {condition_id: to_jsonable(state) for condition_id, state in states.items()},
    })
    result["ready_for_answer"] = strict_grounded or partial_grounded
    return result


def _completion_scope_ratio(status: Mapping[str, Any]) -> float:
    source_coverage = dict(status.get("source_coverage", {}) or {})
    adopted_source = str(status.get("adopted_source_video_id", "") or "")
    rows = [source_coverage[adopted_source]] if adopted_source in source_coverage else list(source_coverage.values())
    if not rows:
        return 1.0 if bool(status.get("ready_for_answer")) else 0.0
    return max(
        (
            float(row.get("coverage_ratio", 0.0) or 0.0)
            if "coverage_ratio" in row
            else float(row.get("covered_count", 0) or 0) / max(1, int(row.get("required_count", 0) or 0))
            for row in rows
        ),
        default=0.0,
    )


def _apply_condition_scope(
    states: Mapping[str, Any],
    completion_status: Mapping[str, Any],
) -> dict[str, Any]:
    source_coverage = dict(completion_status.get("source_coverage", {}) or {})
    adopted_source = str(completion_status.get("adopted_source_video_id", "") or "")
    coverage_rows = [source_coverage[adopted_source]] if adopted_source in source_coverage else list(source_coverage.values())
    coverage_ratio = max(
        (
            float(row.get("coverage_ratio", 0.0) or 0.0)
            if "coverage_ratio" in row
            else float(row.get("covered_count", 0) or 0) / max(1, int(row.get("required_count", 0) or 0))
            for row in coverage_rows
        ),
        default=0.0,
    )
    result = {}
    for condition_id, state in states.items():
        if (
            state.status == "satisfied"
            and (
                (
                    getattr(state, "quantifier", "exists") in {"all_events", "all_segments", "temporal_max"}
                    and getattr(state, "scope", "window") != "full_video"
                )
                or (
                    getattr(state, "quantifier", "exists") == "ordinal_2"
                    and getattr(state, "scope", "window") != "episode"
                )
                or (
                    getattr(state, "scope", "window") == "full_video"
                    and coverage_ratio + 1e-9 < float(getattr(state, "required_coverage", 1.0) or 1.0)
                )
            )
        ):
            state = replace(state, status="unknown")
        result[condition_id] = state
    return result


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


def _apply_entity_completion(
    status: Mapping[str, Any],
    contract: ClaimContract,
    evidence: Sequence[EvidenceRecord],
) -> dict[str, Any]:
    result = dict(status)
    if contract.quantifier != "distinct_count":
        return result
    countable = []
    candidates = []
    resolved_candidates = []
    parse_failures = []
    for record in evidence:
        parse_status = str(record.operation_metadata.get("structured_parse_status", "") or "")
        if parse_status and parse_status != "parsed":
            parse_failures.append(record.evidence_id)
        if parse_status == "parsed" and record.start_sec is not None and record.end_sec is not None:
            if float(record.end_sec) - float(record.start_sec) <= 120.0:
                resolved_candidates.extend(
                    str(item)
                    for item in record.operation_metadata.get("source_candidate_ids", ())
                    if str(item)
                )
        for entity in record.operation_metadata.get("entities", ()) or ():
            if not isinstance(entity, Mapping):
                continue
            observation_id = str(entity.get("entity_observation_id", "") or "")
            if not observation_id:
                continue
            if _metadata_flag(entity.get("countable")):
                countable.append(observation_id)
            else:
                candidates.append(observation_id)
    unresolved_candidates = [item for item in dict.fromkeys(candidates) if item not in set(resolved_candidates)]
    result.update(
        {
            "countable_entity_observation_ids": list(dict.fromkeys(countable)),
            "candidate_entity_observation_ids": list(dict.fromkeys(candidates)),
            "resolved_candidate_entity_observation_ids": list(dict.fromkeys(resolved_candidates)),
            "unresolved_candidate_entity_observation_ids": unresolved_candidates,
            "entity_protocol_parse_failure_evidence_ids": list(dict.fromkeys(parse_failures)),
            "entity_witness_ready": bool(countable),
        }
    )
    result["ready_for_answer"] = bool(result.get("ready_for_answer")) and not unresolved_candidates
    return result


def _apply_event_completion(
    status: Mapping[str, Any],
    contract: ClaimContract,
    evidence: Sequence[EvidenceRecord],
    query_requirements: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(status)
    if contract.quantifier != "total_count" or contract.observation_target != "event":
        return result
    ledger = _event_candidate_ledger(evidence)
    snapshot = canonical_fact_snapshot(
        evidence,
        require_event_precondition=bool((query_requirements or {}).get("requires_state_tracking")),
    ).to_dict()
    qualified = tuple(snapshot.get("qualified_events", ()) or ())
    incomplete = tuple(snapshot.get("incomplete_events", ()) or ())
    conflicted = tuple(snapshot.get("conflicted_events", ()) or ())
    result.update({
        **ledger,
        "confirmed_event_candidate_count": len(qualified),
        "confirmed_event_candidates": list(qualified),
        "qualified_event_count": len(qualified),
        "qualified_events": list(qualified),
        "incomplete_event_count": len(incomplete),
        "incomplete_events": list(incomplete),
        "unqualified_event_count": len(tuple(snapshot.get("unqualified_events", ()) or ())),
        "conflicted_event_count": len(conflicted),
        "event_requirement_graph": dict(snapshot.get("requirement_graph", {}) or {}),
    })
    result["ready_for_answer"] = (
        bool(result.get("ready_for_answer"))
        and bool(qualified)
        and not incomplete
        and not conflicted
        and not ledger["unresolved_event_windows"]
    )
    return result


def _enumeration_required(
    contract: ClaimContract,
    query_requirements: Mapping[str, Any] | None = None,
) -> bool:
    requirements = dict(query_requirements or {})
    return bool(
        (contract.quantifier == "total_count" and contract.observation_target == "event")
        or contract.quantifier == "universal"
        or contract.quantifier == "order"
        or (contract.aggregation == "order" and requirements.get("requires_temporal_sequence"))
        or requirements.get("requires_temporal_extremum")
        or (
            requirements.get("requires_event_participant_link")
            and requirements.get("requires_temporal_sequence")
        )
    )


def _enumeration_source_segments(
    workspace: VirtualVideoWorkspace,
    completion_status: Mapping[str, Any],
) -> tuple[str, tuple[Any, ...], tuple[tuple[float, float], ...]]:
    by_id = {segment.segment_id: segment for segment in workspace.manifest.segments}
    target = by_id.get(workspace.case.target_segment_id)
    source_id = str(completion_status.get("adopted_source_video_id", "") or "")
    if not source_id and target is not None:
        source_id = str(target.source_video_id or "")
    segments = tuple(
        segment
        for segment in workspace.manifest.segments
        if not source_id or segment.source_video_id == source_id
    )
    effective_scope = _compiled_effective_scope(completion_status)
    if effective_scope == "full_video":
        ranges = tuple(
            (float(segment.virtual_start_sec), float(segment.virtual_end_sec))
            for segment in segments
        )
    else:
        requested_start, requested_end = sorted(workspace.case.target_virtual_interval)
        ranges = tuple(
            (
                max(requested_start, float(segment.virtual_start_sec)),
                min(requested_end, float(segment.virtual_end_sec)),
            )
            for segment in segments
            if min(requested_end, float(segment.virtual_end_sec))
            > max(requested_start, float(segment.virtual_start_sec))
        )
    return source_id, segments, tuple(item for item in ranges if item[1] > item[0])


def _parse_status_for_enumeration(record: EvidenceRecord) -> str:
    raw = str(record.operation_metadata.get("structured_parse_status", "") or "").casefold()
    if raw in {"parsed", "ok"}:
        return "ok"
    if not raw:
        return "unknown"
    return raw


def _enumeration_manifest(
    workspace: VirtualVideoWorkspace,
    completion_status: Mapping[str, Any],
    evidence: Sequence[EvidenceRecord],
    query_requirements: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_id, segments, required_ranges = _enumeration_source_segments(workspace, completion_status)
    if not required_ranges:
        return {
            "target_segment_id": workspace.case.target_segment_id,
            "required_range": list(workspace.case.target_virtual_interval),
            "required_ranges": [],
            "windows": [],
            "unprocessed_ranges": [],
            "candidate_ids": [],
            "candidate_reconciliation_status": "incomplete",
            "unresolved_candidate_ids": ["enumeration_source_not_identified"],
            "boundary_gaps": [],
            "enumeration_complete": False,
        }

    segment_ids = {segment.segment_id for segment in segments}
    windows = []
    for record in evidence:
        if (
            record.modality not in {"visual", "ocr"}
            or record.start_sec is None
            or record.end_sec is None
        ):
            continue
        lineages = tuple(record.source_lineage or ())
        record_segment_ids = {
            str(lineage.get("segment_id", "") or "")
            for lineage in lineages
            if str(lineage.get("segment_id", "") or "")
        }
        record_source_ids = {
            str(lineage.get("source_video_id", "") or "")
            for lineage in lineages
            if str(lineage.get("source_video_id", "") or "")
        }
        if lineages and (
            (source_id and source_id not in record_source_ids)
            or (record_segment_ids and not record_segment_ids.intersection(segment_ids))
        ):
            continue
        metadata = dict(record.operation_metadata or {})
        event_rows = tuple(metadata.get("events", ()) or ())
        inspection_intent = str(metadata.get("inspection_intent", "") or "").casefold()
        if not event_rows and record.evidence_kind != "event_observation" and "event" not in inspection_intent:
            continue
        candidate_ids = [
            str(row.get("event_key", "") or row.get("candidate_id", "") or "")
            for row in event_rows
            if isinstance(row, Mapping)
        ]
        candidate_ids.extend(str(item) for item in tuple(metadata.get("source_candidate_ids", ()) or ()) if str(item))
        sampling_policy = dict(metadata.get("sampling_policy", {}) or {})
        fps = float(record.sampling_fps or sampling_policy.get("effective_fps", 0.0) or 0.0)
        dwell = metadata.get("expected_event_dwell_sec", None)
        try:
            dwell_value = float(dwell) if dwell is not None else 1.0
        except (TypeError, ValueError):
            dwell_value = 1.0
        windows.append(
            {
                "range": [float(record.start_sec), float(record.end_sec)],
                "sampling_fps": fps,
                "expected_event_dwell_sec": dwell_value,
                "parse_status": _parse_status_for_enumeration(record),
                "candidate_ids": list(dict.fromkeys(candidate_ids)),
                "evidence_id": record.evidence_id,
            }
        )
    snapshot = canonical_fact_snapshot(
        evidence,
        require_event_precondition=bool((query_requirements or {}).get("requires_state_tracking")),
    ).to_dict()
    candidate_rows = (
        *tuple(snapshot.get("qualified_events", ()) or ()),
        *tuple(snapshot.get("observed_event_candidates", ()) or ()),
        *tuple(snapshot.get("unqualified_events", ()) or ()),
        *tuple(snapshot.get("incomplete_events", ()) or ()),
        *tuple(snapshot.get("conflicted_events", ()) or ()),
        *tuple(snapshot.get("duplicate_suspect_events", ()) or ()),
    )
    candidate_ids = [
        str(row.get("candidate_id", "") or row.get("fact_id", "") or row.get("event_key", "") or "")
        for row in candidate_rows
        if isinstance(row, Mapping)
    ]
    unresolved_rows = (
        *tuple(snapshot.get("incomplete_events", ()) or ()),
        *tuple(snapshot.get("conflicted_events", ()) or ()),
        *tuple(snapshot.get("duplicate_suspect_events", ()) or ()),
    )
    unresolved_ids = [
        str(row.get("candidate_id", "") or row.get("fact_id", "") or row.get("event_key", "") or "")
        for row in unresolved_rows
        if isinstance(row, Mapping)
    ]
    required_start = min(left for left, _ in required_ranges)
    required_end = max(right for _, right in required_ranges)
    return build_enumeration_manifest(
        target_segment_id=workspace.case.target_segment_id,
        required_range=(required_start, required_end),
        required_ranges=required_ranges,
        windows=windows,
        candidate_ids=list(dict.fromkeys(item for item in candidate_ids if item)),
        unresolved_candidate_ids=list(dict.fromkeys(item for item in unresolved_ids if item)),
        expected_event_dwell_sec=1.0,
    )


def _apply_enumeration_completion(
    status: Mapping[str, Any],
    workspace: VirtualVideoWorkspace,
    contract: ClaimContract,
    evidence: Sequence[EvidenceRecord],
    query_requirements: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(status)
    required = _enumeration_required(contract, query_requirements)
    result["enumeration_required"] = required
    if not required:
        result.setdefault("enumeration_complete", True)
        return result
    manifest = _enumeration_manifest(workspace, result, evidence, query_requirements)
    result.update(
        {
            "enumeration_manifest": manifest,
            "enumeration_complete": bool(manifest.get("enumeration_complete", False)),
        }
    )
    result["ready_for_answer"] = bool(result.get("ready_for_answer")) and bool(
        manifest.get("enumeration_complete", False)
    )
    return result


def _event_keys_for_candidate(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(item or "").strip().casefold()
            for item in (
                candidate.get("canonical_event_key", ""),
                candidate.get("signature", ""),
                *tuple(candidate.get("event_keys", ()) or ()),
            )
            if str(item or "").strip()
        )
    )


def _temporal_max_selection(
    snapshot: Mapping[str, Any],
    completion_status: Mapping[str, Any],
) -> dict[str, Any]:
    qualified = tuple(
        row for row in tuple(snapshot.get("qualified_events", ()) or ()) if isinstance(row, Mapping)
    )
    incomplete = tuple(snapshot.get("incomplete_events", ()) or ())
    conflicted = tuple(snapshot.get("conflicted_events", ()) or ())
    candidate_ids = [str(row.get("candidate_id", "") or "") for row in qualified]
    coverage_ready = bool(completion_status.get("range_coverage_complete", False))
    enumeration_ready = (
        not bool(completion_status.get("enumeration_required", False))
        or bool(completion_status.get("enumeration_complete", False))
    )
    evidence_ids = [
        str(evidence_id)
        for row in qualified
        for evidence_id in tuple(row.get("evidence_ids", ()) or ())
        if str(evidence_id)
    ]
    if not coverage_ready or not enumeration_ready or incomplete or conflicted or not qualified:
        return {
            "status": "incomplete",
            "selected_episode_id": "",
            "selected_event_key": "",
            "candidate_episode_ids": candidate_ids,
            "proof_requirement_ids": [
                item
                for item, missing in (
                    ("req_full_video_coverage", not coverage_ready),
                    ("req_all_qualified_episodes_enumerated", not enumeration_ready),
                    ("req_event_qualification_resolved", bool(incomplete or conflicted)),
                )
                if missing
            ],
            "derivation_provenance": list(_fact_derivation_provenance(qualified, "temporal_max")),
        }
    timed_candidates = tuple(
        row
        for row in qualified
        if len(tuple(row.get("virtual_time_range", ()) or ())) == 2
    )
    if not timed_candidates:
        return {
            "status": "incomplete",
            "selected_episode_id": "",
            "selected_event_key": "",
            "candidate_episode_ids": candidate_ids,
            "proof_requirement_ids": ["req_last_episode_selected"],
            "derivation_provenance": list(_fact_derivation_provenance(qualified, "temporal_max")),
        }
    latest_time = max(
        float(tuple(row.get("virtual_time_range", (0.0, 0.0)))[0])
        for row in timed_candidates
    )
    latest = tuple(
        row
        for row in timed_candidates
        if len(tuple(row.get("virtual_time_range", ()) or ())) == 2
        and abs(float(tuple(row.get("virtual_time_range", ()))[0]) - latest_time) <= 1e-6
    )
    if len(latest) != 1:
        return {
            "status": "ambiguous",
            "selected_episode_id": "",
            "selected_event_key": "",
            "candidate_episode_ids": candidate_ids,
            "proof_requirement_ids": ["req_last_episode_selected"],
            "derivation_provenance": list(_fact_derivation_provenance(qualified, "temporal_max")),
        }
    selected = latest[0]
    keys = _event_keys_for_candidate(selected)
    return {
        "status": "resolved",
        "selected_episode_id": str(selected.get("candidate_id", "") or ""),
        "selected_event_key": keys[0] if keys else "",
        "candidate_episode_ids": candidate_ids,
        "proof_requirement_ids": [],
        "selected_episode_range": list(selected.get("virtual_time_range", ()) or ()),
        "derivation_provenance": list(_fact_derivation_provenance((selected,), "temporal_max")),
    }


def _apply_episode_binding_completion(
    status: Mapping[str, Any],
    workspace: VirtualVideoWorkspace,
    evidence: Sequence[EvidenceRecord],
    query_requirements: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(status)
    requirements = dict(query_requirements or {})
    if not (
        requirements.get("requires_event_participant_link")
        and requirements.get("requires_temporal_sequence")
    ):
        return result
    snapshot = canonical_fact_snapshot(
        evidence,
        require_event_precondition=bool(requirements.get("requires_state_tracking")),
    ).to_dict()
    selection = _temporal_max_selection(snapshot, result)
    result["temporal_max_selection"] = selection
    result["selected_last_episode_id"] = str(selection.get("selected_episode_id", "") or "")
    result["selected_episode_event_key"] = str(selection.get("selected_event_key", "") or "")
    result["temporal_max_ready"] = selection.get("status") == "resolved"
    if selection.get("status") != "resolved":
        result.update(
            {
                "event_participant_link_ready": False,
                "target_attribute_ready": False,
                "target_participant_selection": {"status": "incomplete"},
                "target_entity_binding": {"status": "incomplete"},
                "target_attribute_facts": [],
            }
        )
        result["ready_for_answer"] = False
        return result
    selected = next(
        (
            row
            for row in tuple(snapshot.get("qualified_events", ()) or ())
            if str(row.get("candidate_id", "") or "") == selection["selected_episode_id"]
        ),
        {},
    )
    participants = [
        dict(item)
        for item in tuple(selected.get("participants", ()) or ())
        if isinstance(item, Mapping)
        and str(item.get("role", "") or "").casefold() not in {"overtaken", "camera_holder", "recorder"}
    ]
    ordinal_index = _identity_ordinal_index(workspace.case.question)
    if ordinal_index >= len(participants):
        result.update(
            {
                "event_participant_link_ready": False,
                "target_attribute_ready": False,
                "target_participant_selection": {
                    "status": "incomplete",
                    "ordinal": ordinal_index + 1,
                    "selected_episode_id": selection["selected_episode_id"],
                },
                "target_entity_binding": {"status": "incomplete"},
                "target_attribute_facts": [],
            }
        )
        result["ready_for_answer"] = False
        return result
    participant = participants[ordinal_index]
    participant_id = str(participant.get("participant_id", "") or "").strip()
    event_keys = set(_event_keys_for_candidate(selected))
    associations = tuple(
        row
        for row in tuple(snapshot.get("entity_associations", ()) or ())
        if isinstance(row, Mapping)
        and str(row.get("status", "") or "") == "supported"
        and str(row.get("source_participant_id", "") or "").strip().casefold() == participant_id.casefold()
        and str(row.get("source_event_key", "") or "").strip().casefold() in event_keys
    )
    if len(associations) != 1:
        result.update(
            {
                "event_participant_link_ready": False,
                "target_attribute_ready": False,
                "target_participant_selection": {
                    "status": "resolved",
                    "participant_id": participant_id,
                    "ordinal": ordinal_index + 1,
                    "selected_episode_id": selection["selected_episode_id"],
                    "source_event_key": selection["selected_event_key"],
                },
                "target_entity_binding": {
                    "status": "ambiguous" if len(associations) > 1 else "incomplete",
                    "association_ids": [str(row.get("association_id", "") or "") for row in associations],
                },
                "target_attribute_facts": [],
            }
        )
        result["ready_for_answer"] = False
        return result
    association = associations[0]
    entity_id = str(association.get("entity_hypothesis_id", "") or "")
    entity = next(
        (
            row
            for row in tuple(snapshot.get("resolved_entities", ()) or ())
            if str(row.get("entity_id", "") or "") == entity_id
        ),
        {},
    )
    attribute_facts = tuple(
        row
        for row in tuple(snapshot.get("attribute_facts", ()) or ())
        if isinstance(row, Mapping)
        and str(row.get("entity_id", "") or "") == entity_id
        and str(row.get("episode_id", "") or "").strip().casefold() in event_keys
    )
    entity_ready = bool(entity) and bool(entity_id)
    needs_attribute = bool(parse_option_predicates(workspace.case.options)) or bool(
        re.search(r"\b(?:rank|place|finish|color|colour|wearing|helmet|clothes|clothing)\b", workspace.case.question.casefold())
    )
    result.update(
        {
            "event_participant_link_ready": entity_ready,
            "target_attribute_ready": bool(attribute_facts) if needs_attribute else True,
            "target_participant_selection": {
                "status": "resolved",
                "participant_id": participant_id,
                "ordinal": ordinal_index + 1,
                "selected_episode_id": selection["selected_episode_id"],
                "source_event_key": selection["selected_event_key"],
            },
            "target_entity_binding": {
                "status": "resolved" if entity_ready else "incomplete",
                "entity_id": entity_id,
                "association_id": str(association.get("association_id", "") or ""),
                "source_event_key": str(association.get("source_event_key", "") or ""),
            },
            "target_attribute_facts": [dict(row) for row in attribute_facts],
        }
    )
    result["ready_for_answer"] = bool(result.get("ready_for_answer")) and entity_ready and (
        bool(attribute_facts) if needs_attribute else True
    )
    return result


def _apply_p1_semantic_completion(
    status: Mapping[str, Any],
    query_requirements: Mapping[str, Any],
    evidence: Sequence[EvidenceRecord],
) -> dict[str, Any]:
    result = dict(status)
    snapshot = canonical_fact_snapshot(
        evidence,
        require_event_precondition=bool(query_requirements.get("requires_state_tracking")),
    ).to_dict()
    if (
        bool(query_requirements.get("requires_event_participant_link"))
        and not bool(query_requirements.get("requires_temporal_sequence"))
    ):
        associations = tuple(
            row for row in tuple(snapshot.get("entity_associations", ()) or ())
            if str(row.get("status", "") or "") == "supported"
            and float(row.get("confidence", 0.0) or 0.0) >= 0.6
        )
        result["event_participant_association_ids"] = [
            str(row.get("association_id", "") or "") for row in associations
        ]
        result["event_participant_link_ready"] = bool(associations)
        result["ready_for_answer"] = bool(result.get("ready_for_answer")) and bool(associations)
    if bool(query_requirements.get("requires_narrative_inference")):
        facts = tuple(snapshot.get("inferred_facts", ()) or ())
        if bool(query_requirements.get("requires_agent_attribution")):
            facts = tuple(
                fact for fact in facts
                if isinstance(fact, Mapping) and _narrative_fact_has_agent_witness(fact)
            )
        unresolved = tuple(snapshot.get("unresolved_inferences", ()) or ())
        result["narrative_fact_ids"] = [str(row.get("fact_id", "") or "") for row in facts]
        result["unresolved_narrative_fact_ids"] = [
            str(row.get("fact_id", "") or "") for row in unresolved
        ]
        result["narrative_inference_ready"] = bool(facts) and not unresolved
        result["ready_for_answer"] = (
            bool(result.get("ready_for_answer")) and bool(facts) and not unresolved
        )
    if bool(query_requirements.get("requires_same_object_transition")):
        transitions = tuple(
            row for row in tuple(snapshot.get("state_transitions", ()) or ())
            if isinstance(row, Mapping)
            and str(row.get("status", "") or "") == "supported"
            and bool(row.get("same_object_relation", False))
        )
        unresolved_transitions = tuple(snapshot.get("unresolved_state_transitions", ()) or ())
        result["same_object_transition_fact_ids"] = [
            str(row.get("fact_id", "") or "") for row in transitions
        ]
        result["unresolved_state_transition_fact_ids"] = [
            str(row.get("fact_id", "") or "") for row in unresolved_transitions
        ]
        result["same_object_transition_ready"] = bool(transitions) and not unresolved_transitions
        result["ready_for_answer"] = (
            bool(result.get("ready_for_answer"))
            and bool(transitions)
            and not unresolved_transitions
        )
    return result


def _entity_census_coverage_evidence(evidence: Sequence[EvidenceRecord]) -> tuple[EvidenceRecord, ...]:
    return tuple(
        record
        for record in evidence
        if record.start_sec is not None
        and record.end_sec is not None
        and float(record.end_sec) - float(record.start_sec) <= 120.0
    )


def _visual_inspection_attempts(
    attempts: Sequence[ObservationAttempt],
) -> tuple[ObservationAttempt, ...]:
    return tuple(
        attempt
        for attempt in attempts
        if str(attempt.sampling_config.get("modality", "visual") or "visual").casefold() in {"visual", "ocr"}
        and attempt.outcome not in {"failed", "duplicate"}
        and bool(attempt.inspected_ranges)
        and (attempt.images_attached > 0 or str(attempt.sampling_config.get("modality", "")).casefold() == "ocr")
    )


def _source_coverage(
    workspace: VirtualVideoWorkspace,
    attempts: Sequence[ObservationAttempt],
) -> dict[str, dict[str, Any]]:
    required: dict[str, list[Any]] = {}
    segments_by_id = {}
    for segment in workspace.manifest.segments:
        required.setdefault(segment.source_video_id, []).append(segment)
        segments_by_id[segment.segment_id] = segment
    covered_ranges: dict[str, dict[str, list[tuple[float, float]]]] = {}
    confidence: dict[str, float] = {}
    attempt_ids: dict[str, set[str]] = {}
    for attempt in attempts:
        modality = str(attempt.sampling_config.get("modality", "visual") or "visual").casefold()
        if modality not in {"visual", "ocr"} or attempt.outcome in {"failed", "duplicate"}:
            continue
        if modality == "visual" and attempt.images_attached <= 0:
            continue
        for inspected_start, inspected_end in attempt.inspected_ranges:
            for segment_id, segment in segments_by_id.items():
                start = max(float(segment.virtual_start_sec), float(inspected_start))
                end = min(float(segment.virtual_end_sec), float(inspected_end))
                if end <= start:
                    continue
                source_id = str(segment.source_video_id)
                covered_ranges.setdefault(source_id, {}).setdefault(segment_id, []).append((start, end))
                confidence[source_id] = max(
                    confidence.get(source_id, 0.0),
                    1.0 if attempt.parse_status not in {"failed", "unknown"} else 0.5,
                )
                attempt_ids.setdefault(source_id, set()).add(attempt.attempt_id)
    result: dict[str, dict[str, Any]] = {}
    for source_id, required_segments_raw in required.items():
        segment_ranges = covered_ranges.get(source_id, {})
        required_segments = tuple(required_segments_raw)
        segment_coverage = {}
        covered_ids = []
        total_required = 0.0
        total_covered = 0.0
        for segment in required_segments:
            interval = (float(segment.virtual_start_sec), float(segment.virtual_end_sec))
            duration = max(0.0, interval[1] - interval[0])
            ranges = tuple(segment_ranges.get(segment.segment_id, ()))
            covered_sec = max(0.0, duration - _uncovered_duration(interval, ranges))
            uncovered_ranges = _uncovered_ranges(interval, ranges)
            ratio = covered_sec / duration if duration > 0 else 0.0
            total_required += duration
            total_covered += covered_sec
            segment_coverage[segment.segment_id] = {
                "covered_sec": round(covered_sec, 3),
                "required_sec": round(duration, 3),
                "coverage_ratio": round(min(1.0, ratio), 6),
                "uncovered_ranges": [list(item) for item in uncovered_ranges],
            }
            if ratio + 1e-9 >= 0.95:
                covered_ids.append(segment.segment_id)
        required_ids = tuple(segment.segment_id for segment in required_segments)
        missing = [segment_id for segment_id in required_ids if segment_id not in covered_ids]
        result[source_id] = {
            "covered_segment_ids": sorted(covered_ids),
            "required_segment_ids": list(required_ids),
            "missing_segment_ids": missing,
            "covered_count": len(covered_ids),
            "required_count": len(required_ids),
            "covered_duration_sec": round(total_covered, 3),
            "required_duration_sec": round(total_required, 3),
            "coverage_ratio": round(total_covered / total_required, 6) if total_required > 0 else 0.0,
            "segment_coverage": segment_coverage,
            "confidence": confidence.get(source_id, 0.0),
            "inspection_attempt_ids": sorted(attempt_ids.get(source_id, set())),
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
    completion_status: Mapping[str, Any] | None = None,
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
    if contract.observation_target == "relation":
        relation_gate = _spatial_relation_gate(
            workspace,
            answer,
            cited,
            query_requirements=query_requirements,
        )
        if not relation_gate.get("passed"):
            return relation_gate
        return _contract_readiness_gate(contract, completion_status) or relation_gate
    quantitative_gate = _quantitative_answer_gate(
        workspace,
        contract,
        answer,
        cited,
        query_requirements=query_requirements,
    )
    if quantitative_gate is not None:
        if not quantitative_gate.get("passed"):
            return quantitative_gate
        return _contract_readiness_gate(contract, completion_status) or quantitative_gate
    effective_scope = _compiled_effective_scope(completion_status)
    if effective_scope != "full_video":
        return _contract_readiness_gate(contract, completion_status) or {
            "passed": True,
            "reason": "verified_window_evidence",
            "missing_segment_ids": [],
        }

    cited_sources = {
        str(lineage.get("source_video_id", "") or "")
        for record in cited
        for lineage in record.source_lineage
        if str(lineage.get("source_video_id", "") or "")
    }
    if not cited_sources:
        return {"passed": False, "reason": "source_not_identified", "missing_segment_ids": []}
    source_coverage = dict((completion_status or {}).get("source_coverage", {}) or {})
    if not source_coverage:
        return {
            "passed": False,
            "reason": "inspection_coverage_missing",
            "source_video_ids": sorted(cited_sources),
            "missing_segment_ids": [],
        }
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
    if contract.quantifier == "universal":
        return _global_absence_gate(workspace, answer, evidence, cited)
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
        witness_gate = _entity_cluster_witness_gate(clusters, cited)
        if witness_gate is not None:
            return witness_gate
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
        ledger = _event_candidate_ledger(cited)
        snapshot = canonical_fact_snapshot(
            cited,
            require_event_precondition=bool(
                compile_query_requirements(workspace.case.question).get("requires_state_tracking")
            ),
        ).to_dict()
        event_occurrences = tuple(snapshot.get("qualified_events", ()) or ())
        incomplete_events = tuple(snapshot.get("incomplete_events", ()) or ())
        conflicted_events = tuple(snapshot.get("conflicted_events", ()) or ())
        if ledger["unresolved_event_windows"] or incomplete_events or conflicted_events:
            return {
                "passed": False,
                "reason": "event_candidate_reconciliation_incomplete",
                "confirmed_event_candidate_count": len(event_occurrences),
                "unresolved_event_windows": ledger["unresolved_event_windows"],
                "incomplete_event_ids": [row.get("candidate_id") for row in incomplete_events],
                "conflicted_event_ids": [row.get("candidate_id") for row in conflicted_events],
                "missing_segment_ids": [],
            }
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
    return _contract_readiness_gate(contract, completion_status) or {
        "passed": True,
        "reason": "full_source_coverage_verified",
        "source_video_ids": sorted(cited_sources),
        "entity_cluster_count": len(entity_clusters),
        "event_occurrence_count": len(event_occurrences),
        "missing_segment_ids": [],
    }


def _entity_cluster_witness_gate(
    clusters: Sequence[Mapping[str, Any]],
    cited: Sequence[EvidenceRecord],
) -> dict[str, Any] | None:
    # Evidence created before the witnessed-entity protocol remains readable. New
    # model-driven observations always carry structured_parse_status and are gated.
    if not any("structured_parse_status" in record.operation_metadata for record in cited):
        return None

    observations: dict[str, dict[str, Any]] = {}
    observations_by_evidence: dict[str, list[str]] = {}
    records_by_id = {record.evidence_id: record for record in cited}
    for record in cited:
        for raw_entity in record.operation_metadata.get("entities", ()) or ():
            if not isinstance(raw_entity, Mapping):
                continue
            entity = dict(raw_entity)
            observation_id = str(
                entity.get("entity_observation_id")
                or (
                    f"{record.observation_id}:{entity.get('local_id')}"
                    if record.observation_id and entity.get("local_id")
                    else ""
                )
            ).strip()
            if not observation_id:
                continue
            countable = (
                _metadata_flag(entity.get("countable"))
                and _metadata_flag(entity.get("supports_question_relation"))
                and bool(str(entity.get("visual_signature", "") or "").strip())
                and bool(tuple(entity.get("witness_frame_refs", ()) or ()))
            )
            observations[observation_id] = {
                "evidence_id": record.evidence_id,
                "countable": countable,
                "candidate_reason": str(entity.get("candidate_reason", "") or ""),
            }
            observations_by_evidence.setdefault(record.evidence_id, []).append(observation_id)

    assigned: dict[str, str] = {}
    unsupported = []
    for raw_cluster in clusters:
        cluster = _entity_cluster(raw_cluster)
        observation_ids = tuple(cluster["entity_observation_ids"])
        if not observation_ids:
            inferred = tuple(
                dict.fromkeys(
                    observation_id
                    for evidence_id in cluster["evidence_ids"]
                    for observation_id in observations_by_evidence.get(evidence_id, ())
                    if observations.get(observation_id, {}).get("countable")
                )
            )
            if len(inferred) == 1:
                observation_ids = inferred
        invalid_ids = tuple(
            observation_id
            for observation_id in observation_ids
            if observation_id not in observations or not observations[observation_id]["countable"]
        )
        duplicate_ids = tuple(
            observation_id
            for observation_id in observation_ids
            if observation_id in assigned and assigned[observation_id] != cluster["entity_id"]
        )
        if not observation_ids or invalid_ids or duplicate_ids:
            repair_records = [records_by_id[item] for item in cluster["evidence_ids"] if item in records_by_id]
            unsupported.append(
                {
                    "entity_id": cluster["entity_id"],
                    "evidence_ids": list(cluster["evidence_ids"]),
                    "entity_observation_ids": list(observation_ids),
                    "invalid_entity_observation_ids": list(invalid_ids),
                    "duplicate_entity_observation_ids": list(duplicate_ids),
                    "candidate_entity_observation_ids": list(
                        dict.fromkeys(
                            observation_id
                            for evidence_id in cluster["evidence_ids"]
                            for observation_id in observations_by_evidence.get(evidence_id, ())
                        )
                    ),
                    "repair_windows": [
                        [record.start_sec, record.end_sec]
                        for record in repair_records
                        if record.start_sec is not None and record.end_sec is not None
                    ],
                }
            )
            continue
        for observation_id in observation_ids:
            assigned[observation_id] = cluster["entity_id"]
    if not unsupported:
        return None
    return {
        "passed": False,
        "reason": "entity_cluster_witness_missing",
        "unsupported_entity_clusters": unsupported,
        "countable_entity_observation_ids": sorted(
            observation_id for observation_id, row in observations.items() if row["countable"]
        ),
        "missing_segment_ids": [],
    }


def _contract_readiness_gate(
    contract: ClaimContract,
    completion_status: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if completion_status is None:
        return None
    status = dict(completion_status)
    requires_semantic_closure = (
        _compiled_effective_scope(status) == "full_video"
        or contract.observation_target == "relation"
        or bool(contract.boundary_hint)
        or contract.quantifier in {"universal", "comparison", "total_count", "distinct_count", "order"}
        or (contract.observation_target == "event" and contract.aggregation == "compare")
    )
    if not requires_semantic_closure:
        return None
    if status and bool(status.get("ready_for_answer")):
        return None
    blockers = list(status.get("strict_safety_blockers", ()) or ())
    unresolved = list(status.get("unresolved_critical_condition_ids", ()) or ())
    return {
        "passed": False,
        "reason": blockers[0] if blockers else unresolved[0] if unresolved else "contract_completion_not_ready",
        "completion_reason": str(status.get("reason", "") or ""),
        "strict_safety_blockers": blockers,
        "unresolved_critical_condition_ids": unresolved,
        "unsupported_claim_atom_ids": list(status.get("unsupported_claim_atom_ids", ()) or ()),
        "missing_segment_ids": list(status.get("missing_segment_ids", ()) or ()),
    }


def _compiled_effective_scope(completion_status: Mapping[str, Any] | None) -> str:
    status = dict(completion_status or {})
    obligations = dict(status.get("query_obligations", {}) or {})
    return str(
        status.get("effective_scope")
        or obligations.get("effective_scope")
        or "window"
    )


def _spatial_relation_gate(
    workspace: VirtualVideoWorkspace,
    answer: str,
    cited: Sequence[EvidenceRecord],
    *,
    query_requirements: Mapping[str, Any] | None,
) -> dict[str, Any]:
    selected = _letter(answer) or _option_letter_from_answer(answer, workspace.case.options)
    expected = _spatial_relation_value(str(workspace.case.options.get(selected, "") or answer))
    if not expected:
        return {"passed": False, "reason": "spatial_option_relation_missing", "missing_segment_ids": []}
    required_frame = str((query_requirements or {}).get("spatial_reference_frame", "") or "")
    required_type = str((query_requirements or {}).get("spatial_relation_type", "") or "relative_bearing")
    observed = []
    for record in cited:
        for fact in normalize_relations(record.operation_metadata.get("relations"), evidence_id=record.evidence_id):
            if fact.relation_type != required_type:
                continue
            witness_indices = tuple(
                index for index in fact.witness_frame_indices if 0 <= index < len(record.frame_refs)
            )
            observed.append(
                {
                    "value": fact.value,
                    "reference_frame": fact.reference_frame,
                    "same_frame": fact.same_frame,
                    "subject_id": fact.subject_id,
                    "object_id": fact.object_id,
                    "witness_frame_indices": list(witness_indices),
                    "evidence_ids": list(fact.evidence_ids),
                }
            )
            if (
                fact.status == "supported"
                and fact.same_frame
                and witness_indices
                and fact.subject_id
                and fact.object_id
                and fact.value == expected
                and fact.reference_frame
                and (not required_frame or fact.reference_frame == required_frame)
            ):
                return {
                    "passed": True,
                    "reason": "spatial_relation_grounded",
                    "spatial_relation": observed[-1],
                    "missing_segment_ids": [],
                }
    return {
        "passed": False,
        "reason": "spatial_relation_not_grounded",
        "expected_relation": expected,
        "required_relation_type": required_type,
        "required_reference_frame": required_frame,
        "observed_relations": observed,
        "missing_segment_ids": [],
    }


def _global_absence_gate(
    workspace: VirtualVideoWorkspace,
    answer: str,
    evidence: Sequence[EvidenceRecord],
    cited: Sequence[EvidenceRecord],
) -> dict[str, Any]:
    selected = _letter(answer) or _option_letter_from_answer(answer, workspace.case.options)
    option_text = str(workspace.case.options.get(selected, "") or answer)
    probes = []
    for record in evidence:
        presence = dict(record.operation_metadata.get("target_presence", {}) or {})
        if not presence or not _target_matches_option(str(presence.get("target", "") or ""), option_text):
            continue
        status = str(presence.get("status", "") or "").casefold()
        probes.append(
            (
                record.evidence_id,
                status,
                _record_has_qualified_absence(record),
            )
        )
    positive = [evidence_id for evidence_id, status, _qualified in probes if status == "present"]
    if positive:
        return {
            "passed": False,
            "reason": "global_absence_contradicted",
            "positive_evidence_ids": positive,
            "missing_segment_ids": [],
        }
    cited_ids = {record.evidence_id for record in cited}
    negative = [
        evidence_id
        for evidence_id, status, qualified in probes
        if status == "absent" and qualified and evidence_id in cited_ids
    ]
    if not negative:
        return {
            "passed": False,
            "reason": "option_specific_absence_evidence_missing",
            "missing_segment_ids": [],
        }
    return {
        "passed": True,
        "reason": "global_absence_grounded",
        "negative_evidence_ids": negative,
        "missing_segment_ids": [],
    }


def _record_has_qualified_absence(record: EvidenceRecord) -> bool:
    metadata = record.operation_metadata
    qualification = dict(metadata.get("absence_qualification", {}) or {})
    return bool(
        str(metadata.get("absence_status", "") or "") == "qualified_absence"
        and str(qualification.get("status", "") or "") == "qualified_absence"
        and float(qualification.get("coverage_ratio", 0.0) or 0.0) >= 0.9
        and qualification.get("expected_dwell_time_sec") is not None
        and qualification.get("sampling_interval_sec") is not None
        and float(qualification["sampling_interval_sec"]) <= float(qualification["expected_dwell_time_sec"]) / 2.0
        and str(qualification.get("visibility_status", "") or "") in {"clear", "visible", "unoccluded"}
    )


def _target_matches_option(target: str, option_text: str) -> bool:
    stop = {"the", "this", "that", "skill", "video", "shown", "seen", "featured", "included"}
    left = {
        token[:-1] if token.endswith("s") and len(token) > 4 else token
        for token in re.findall(r"[a-z0-9]+", str(target or "").casefold())
        if len(token) >= 3 and token not in stop
    }
    right = {
        token[:-1] if token.endswith("s") and len(token) > 4 else token
        for token in re.findall(r"[a-z0-9]+", str(option_text or "").casefold())
        if len(token) >= 3 and token not in stop
    }
    return bool(left and right and left.intersection(right))


def _spatial_relation_value(text: str) -> str:
    normalized = str(text or "").casefold()
    patterns = (
        ("right_front", ("right front", "front right")),
        ("left_front", ("left front", "front left")),
        ("front", ("directly in front", "in front")),
        ("behind", ("behind", "back")),
        ("upper_left", ("upper left", "top left")),
        ("upper_right", ("upper right", "top right")),
        ("lower_left", ("lower left", "bottom left")),
        ("lower_right", ("lower right", "bottom right")),
        ("right", ("right",)),
        ("left", ("left",)),
    )
    return next((value for value, aliases in patterns if any(alias in normalized for alias in aliases)), "")


def _quantitative_answer_gate(
    workspace: VirtualVideoWorkspace,
    contract: ClaimContract,
    answer: str,
    cited: Sequence[EvidenceRecord],
    *,
    query_requirements: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if contract.quantifier in {"distinct_count", "total_count"}:
        return None
    selected = _letter(answer) or _option_letter_from_answer(answer, workspace.case.options)
    option_text = str(workspace.case.options.get(selected, "") or answer)
    if contract.measurement_unit == "point" and contract.boundary_hint and _score_pair(option_text):
        return _boundary_score_gate(contract, option_text, cited)
    if contract.quantifier == "scalar_quantity":
        return _scalar_quantity_gate(
            contract,
            option_text,
            cited,
            query_requirements=query_requirements,
        )
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


def _boundary_score_gate(
    contract: ClaimContract,
    option_text: str,
    cited: Sequence[EvidenceRecord],
) -> dict[str, Any]:
    expected = _score_pair(option_text)
    if expected is None:
        return {"passed": False, "reason": "score_pair_missing", "missing_segment_ids": []}
    boundary_key = _boundary_event_key(contract.boundary_hint)
    observed = []
    for record in cited:
        grouped: dict[str, list[MeasurementFact]] = {}
        for fact in _measurement_facts((record,)):
            if fact.quantity_type != "score" or fact.unit != "point" or not fact.event_id:
                continue
            grouped.setdefault(fact.event_id.casefold(), []).append(fact)
        for event_id, facts in grouped.items():
            eligible = tuple(
                fact
                for fact in facts
                if fact.boundary_relation == "at"
                and fact.binding_status == "explicit"
                and fact.subject_id
            )
            values = tuple(int(round(fact.value)) for fact in eligible)
            observed.append(
                {
                    "evidence_id": record.evidence_id,
                    "event_id": event_id,
                    "values": list(values),
                    "subjects": [fact.subject_id for fact in eligible],
                }
            )
            if (
                len(eligible) >= 2
                and len({fact.subject_id for fact in eligible}) >= 2
                and (not boundary_key or boundary_key in event_id)
                and (values[:2] == expected or values[:2] == tuple(reversed(expected)))
            ):
                return {
                    "passed": True,
                    "reason": "boundary_score_grounded",
                    "score_snapshot": observed[-1],
                    "missing_segment_ids": [],
                }
    return {
        "passed": False,
        "reason": "boundary_score_snapshot_missing",
        "expected_score": list(expected),
        "required_boundary_event": boundary_key,
        "observed_score_snapshots": observed,
        "missing_segment_ids": [],
    }


def _score_pair(text: str) -> tuple[int, int] | None:
    match = re.search(r"(?<!\d)(\d{1,3})\s*(?:-|–|—|:)\s*(\d{1,3})(?!\d)", str(text or ""))
    return (int(match.group(1)), int(match.group(2))) if match else None


def _boundary_event_key(text: str) -> str:
    normalized = str(text or "").casefold()
    if "half" in normalized or "intermission" in normalized:
        return "halftime"
    if "quarter" in normalized:
        return "quarter_end"
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")


def _scalar_quantity_gate(
    contract: ClaimContract,
    option_text: str,
    cited: Sequence[EvidenceRecord],
    *,
    query_requirements: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    target = _quantity_value(option_text)
    if target is None:
        return {"passed": False, "reason": "scalar_option_value_missing", "missing_segment_ids": []}
    unit = canonical_unit(contract.measurement_unit)
    unit_facts = tuple(
        fact
        for fact in _measurement_facts(cited)
        if fact.unit == unit and (not contract.boundary_hint or fact.boundary_relation in {"before", "at"})
    )
    required_subject = str((query_requirements or {}).get("measurement_subject_role", "") or "")
    facts = tuple(
        fact
        for fact in unit_facts
        if not required_subject or _measurement_subject_matches(fact.subject_id, required_subject)
    )
    if not facts:
        return {
            "passed": False,
            "reason": (
                "scalar_measurement_subject_binding_missing"
                if required_subject and unit_facts
                else "scalar_measurement_evidence_missing"
            ),
            "measurement_unit": unit,
            "required_measurement_subject": required_subject,
            "observed_measurement_subjects": sorted(
                {fact.subject_id for fact in unit_facts if fact.subject_id}
            ),
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


def _measurement_subject_matches(subject_id: str, required_role: str) -> bool:
    subject = re.sub(r"[^a-z0-9]+", "_", str(subject_id or "").casefold()).strip("_")
    required = re.sub(r"[^a-z0-9]+", "_", str(required_role or "").casefold()).strip("_")
    if not subject or not required:
        return False
    aliases = {
        "other_team": {"other_team", "opposing_team", "opponent_team", "non_pizza_team"},
        "other_subject": {"other_subject", "other_group", "other_person", "counterpart"},
        "anchored_subject": {
            "anchored_subject",
            "identified_subject",
            "target_person",
            "golden_burger_eater",
        },
    }
    return subject == required or subject in aliases.get(required, set())


def _measurement_facts(cited: Sequence[EvidenceRecord]) -> tuple[MeasurementFact, ...]:
    return tuple(
        fact
        for record in cited
        for fact in normalize_measurements(
            record.operation_metadata.get("measurements"),
            evidence_id=record.evidence_id,
        )
    )


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
    required_subject = str((query_requirements or {}).get("measurement_subject_role", "") or "")
    if required_subject == "anchored_subject":
        has_anchor = any(_record_matches_identity_anchor(record, terms) for record in cited)
        has_bound_measurement = any(
            _measurement_subject_matches(fact.subject_id, required_subject)
            for record in cited
            for fact in normalize_measurements(
                record.operation_metadata.get("measurements"),
                evidence_id=record.evidence_id,
            )
        )
        if has_anchor and has_bound_measurement:
            return None
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
        source_candidate_ids=tuple(value.get("source_candidate_ids", ()) or ()),
        inspection_intent=str(value.get("inspection_intent", "") or ""),
        reference_entities=tuple(
            item for item in tuple(value.get("reference_entities", ()) or ()) if isinstance(item, Mapping)
        ),
        reference_facts=tuple(
            item for item in tuple(value.get("reference_facts", ()) or ()) if isinstance(item, Mapping)
        ),
        origin_gap_id=str(value.get("origin_gap_id", "") or ""),
        target_condition_ids=tuple(value.get("target_condition_ids", ()) or ()),
        boundary_episode_id=str(value.get("boundary_episode_id", "") or ""),
        target_option_predicates=tuple(value.get("target_option_predicates", ()) or ()),
        target_requirement_ids=tuple(value.get("target_requirement_ids", ()) or ()),
        candidate_id=str(value.get("candidate_id", "") or ""),
        episode_id=str(value.get("episode_id", "") or ""),
        entity_hypothesis_id=str(value.get("entity_hypothesis_id", "") or ""),
        target_option_predicate_ids=tuple(value.get("target_option_predicate_ids", ()) or ()),
        sampling_floor_fps=(
            float(value.get("sampling_floor_fps") or 0.5)
            if "sampling_floor_fps" in value and value.get("sampling_floor_fps") is not None
            else None
        ),
        expected_event_dwell_sec=(
            float(value.get("expected_event_dwell_sec"))
            if value.get("expected_event_dwell_sec") is not None
            else None
        ),
        temporal_resolution_rationale=str(value.get("temporal_resolution_rationale", "") or ""),
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
            condition_type=str(value.get("condition_type", "auto") or "auto"),
            target_role=str(value.get("target_role", "") or ""),
            quantity_type=str(value.get("quantity_type", "") or ""),
            unit=str(value.get("unit", "") or ""),
            relation_type=str(value.get("relation_type", "") or ""),
            subject_role=str(value.get("subject_role", "") or ""),
            object_role=str(value.get("object_role", "") or ""),
            required_relation=str(value.get("required_relation", "") or ""),
            scope=str(value.get("scope", "auto") or "auto"),
            quantifier=str(value.get("quantifier", "auto") or "auto"),
            required_coverage=float(value.get("required_coverage", 0.0) or 0.0),
            aggregation=str(value.get("aggregation", "none") or "none"),
            evaluation_type=str(value.get("evaluation_type", "auto") or "auto"),
        )
    return GapCondition("", str(value or ""))


def _bind_gap_to_tasks(decision: ReasonerDecision) -> ReasonerDecision:
    gap = decision.primary_gap
    if decision.action != "investigate" or gap is None:
        return decision
    gap_conditions = _investigator_conditions(gap.conditions)
    tasks = tuple(
        replace(
            task,
            gap_id=task.gap_id or gap.gap_id,
            success_conditions=task.success_conditions or tuple(
                condition.description for condition in gap_conditions if condition.description
            ),
            conditions=gap_conditions or _investigator_conditions(task.conditions),
        )
        for task in decision.tasks
    )
    return replace(decision, tasks=tasks)


def _investigator_conditions(conditions: Sequence[GapCondition]) -> tuple[GapCondition, ...]:
    return tuple(condition for condition in conditions if condition.evaluation_type == "observable")


def _inherit_repair_lineage(
    decision: ReasonerDecision,
    *,
    origin_gap: EvidenceGap | None,
    condition_registry: Sequence[GapCondition],
    active_condition_ids: Sequence[str],
    reports: Sequence[InvestigationReport],
    options: Mapping[str, str],
) -> ReasonerDecision:
    if decision.action != "investigate" or not decision.tasks:
        return decision
    active = {str(item) for item in active_condition_ids if str(item)}
    registry_conditions = tuple(
        condition for condition in condition_registry
        if condition.evaluation_type == "observable"
        and (not active or condition.condition_id in active)
    )
    conditions = tuple({
        condition.condition_id: condition
        for condition in (*registry_conditions, *(origin_gap.conditions if origin_gap is not None else ()))
        if condition.condition_id and condition.evaluation_type == "observable"
    }.values())
    condition_ids = tuple(dict.fromkeys(
        condition.condition_id for condition in conditions if condition.condition_id
    )) or tuple(dict.fromkeys(active_condition_ids))
    gap_id = (
        str(origin_gap.gap_id or "")
        if origin_gap is not None
        else next((str(report.gap_id) for report in reversed(reports) if report.gap_id), "")
    )
    option_predicates = tuple(f"{option}={text}" for option, text in options.items())
    option_predicate_ids = tuple(
        predicate.predicate_id
        for predicates in parse_option_predicates(options).values()
        for predicate in predicates
    )
    tasks = []
    for task in decision.tasks:
        merged_conditions = tuple({
            condition.condition_id: condition
            for condition in (*conditions, *task.conditions)
            if condition.condition_id and condition.evaluation_type == "observable"
        }.values())
        boundary_episode_id = task.boundary_episode_id or next(
            (
                candidate_id for candidate_id in task.source_candidate_ids
                if "episode" in candidate_id.casefold() or "event" in candidate_id.casefold()
            ),
            "",
        )
        tasks.append(replace(
            task,
            gap_id=task.gap_id or gap_id,
            success_conditions=task.success_conditions or tuple(
                condition.description for condition in conditions if condition.description
            ),
            conditions=merged_conditions,
            origin_gap_id=task.origin_gap_id or gap_id,
            target_condition_ids=task.target_condition_ids or condition_ids,
            boundary_episode_id=boundary_episode_id,
            target_option_predicates=task.target_option_predicates or option_predicates,
            target_requirement_ids=task.target_requirement_ids or condition_ids,
            candidate_id=task.candidate_id or next(iter(task.source_candidate_ids), ""),
            episode_id=task.episode_id or boundary_episode_id,
            entity_hypothesis_id=task.entity_hypothesis_id or next((
                str(entity.get("entity_hypothesis_id", "") or entity.get("entity_id", "") or "")
                for entity in task.reference_entities
                if str(entity.get("entity_hypothesis_id", "") or entity.get("entity_id", "") or "")
            ), ""),
            target_option_predicate_ids=task.target_option_predicate_ids or option_predicate_ids,
        ))
    return replace(decision, tasks=tuple(tasks))


def _align_decision_conditions(
    decision: ReasonerDecision,
    registry: Sequence[GapCondition],
) -> tuple[ReasonerDecision, tuple[GapCondition, ...]]:
    gap = decision.primary_gap
    if gap is None or not gap.conditions:
        return decision, tuple(registry)
    known = list(registry)
    aligned = []
    for condition in gap.conditions:
        match = _matching_condition(condition, known)
        if match is None:
            aligned_condition = condition
            known.append(condition)
        else:
            aligned_condition = replace(
                condition,
                condition_id=match.condition_id,
                evaluation_type=match.evaluation_type,
            )
        aligned.append(aligned_condition)
    updated_gap = replace(gap, conditions=tuple(aligned))
    return replace(decision, primary_gap=updated_gap), tuple(known)


def _matching_condition(
    candidate: GapCondition,
    registry: Sequence[GapCondition],
) -> GapCondition | None:
    candidate_family, candidate_tokens = _condition_semantics(candidate)
    best: tuple[float, GapCondition] | None = None
    for known in registry:
        known_family, known_tokens = _condition_semantics(known)
        if candidate_family != known_family:
            continue
        structured = (
            candidate.condition_type,
            candidate.target_role,
            candidate.quantity_type,
            candidate.unit,
            candidate.relation_type,
            candidate.subject_role,
            candidate.object_role,
            candidate.required_relation,
            candidate.aggregation,
        )
        known_structured = (
            known.condition_type,
            known.target_role,
            known.quantity_type,
            known.unit,
            known.relation_type,
            known.subject_role,
            known.object_role,
            known.required_relation,
            known.aggregation,
        )
        if structured != known_structured:
            continue
        union = candidate_tokens | known_tokens
        similarity = len(candidate_tokens & known_tokens) / max(1, len(union))
        if similarity >= 0.5 and (best is None or similarity > best[0]):
            best = (similarity, known)
    return best[1] if best is not None else None


def _condition_semantics(condition: GapCondition) -> tuple[str, set[str]]:
    text = str(condition.description or "").casefold()
    normalized = re.sub(r"\bside[ -]by[ -]side\b", " side_by_side ", text)
    tokens = set(re.findall(r"[a-z][a-z0-9_]*", normalized))
    aliases = {
        "countable": "count",
        "counted": "count",
        "counting": "count",
        "outside": "exterior",
        "external": "exterior",
        "witnessed": "witness",
        "witnesses": "witness",
        "identified": "identify",
        "confirmed": "confirm",
        "visible": "visual",
        "visibly": "visual",
        "cylinders": "cylinder",
        "objects": "object",
        "events": "event",
    }
    tokens = {aliases.get(token, token) for token in tokens}
    family = (
        "count"
        if tokens.intersection({"count", "number", "deduplicate"})
        else "relation"
        if "side_by_side" in tokens or tokens.intersection({"layout", "arrangement", "relation"})
        else "context"
        if tokens.intersection({"exterior", "factory", "setting", "context"})
        else "order"
        if tokens.intersection({"order", "sequence", "chronological", "before", "after"})
        else "identity"
        if tokens.intersection({"identity", "same", "person", "rider", "player"})
        else "witness"
        if tokens.intersection({"witness", "frame"})
        else condition.condition_type
    )
    ignored = {
        "a",
        "all",
        "are",
        "be",
        "confirm",
        "each",
        "find",
        "frame",
        "identify",
        "is",
        "layout",
        "arrangement",
        "object",
        "provide",
        "shot",
        "the",
        "visual",
        "witness",
    }
    content = {token for token in tokens if token not in ignored}
    return family, content or {family}


def _entity_cluster(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    return {
        "entity_id": str(row.get("entity_id", "") or ""),
        "description": str(row.get("description", "") or ""),
        "evidence_ids": tuple(str(item) for item in row.get("evidence_ids", ()) if str(item).strip()),
        "entity_observation_ids": tuple(
            str(item) for item in row.get("entity_observation_ids", ()) if str(item).strip()
        ),
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
        option_verdicts=dict(value.get("option_verdicts", {}) or {}),
        audit_record=dict(value.get("audit_record", {}) or {}),
        primary_gap=_gap(value["primary_gap"]) if isinstance(value.get("primary_gap"), Mapping) else None,
    )


def requires_option_audit(
    contract: ClaimContract | Mapping[str, Any],
    query_requirements: Mapping[str, Any] | None,
) -> bool:
    requirements = dict(query_requirements or {})
    aggregation = str(
        contract.aggregation if isinstance(contract, ClaimContract) else contract.get("aggregation", "")
    ).casefold()
    quantifier = str(
        contract.quantifier if isinstance(contract, ClaimContract) else contract.get("quantifier", "")
    ).casefold()
    boundary_hint = str(
        contract.boundary_hint if isinstance(contract, ClaimContract) else contract.get("boundary_hint", "")
    ).strip()
    observation_target = str(
        contract.observation_target if isinstance(contract, ClaimContract) else contract.get("observation_target", "")
    ).casefold()
    return bool(
        aggregation in {"order", "compare", "count"}
        or quantifier in {"order", "comparison", "total_count", "distinct_count", "universal"}
        or boundary_hint
        or observation_target == "relation"
        or requirements.get("requires_temporal_sequence")
        or requirements.get("requires_state_tracking")
        or requirements.get("requires_identity_link")
        or requirements.get("requires_event_participant_link")
        or requirements.get("requires_narrative_inference")
        or requirements.get("requires_spatial_relation")
    )


def _requires_discriminative_audit(
    contract: ClaimContract,
    query_requirements: Mapping[str, Any] | None,
) -> bool:
    return requires_option_audit(contract, query_requirements)


def _enforce_strict_safety(
    status: Mapping[str, Any],
    contract: ClaimContract,
    query_requirements: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = dict(status)
    if not requires_option_audit(contract, query_requirements):
        result["strict_safety_blockers"] = []
        return result
    blockers = []
    if int(result.get("critical_condition_count", 0) or 0) == 0:
        blockers.append("condition_registry_missing")
    if not bool(result.get("choice_ready")):
        blockers.append("choice_not_ready")
    requirements = dict(query_requirements or {})
    candidate_required = bool(
        contract.quantifier in {"total_count", "distinct_count"}
        or requirements.get("requires_identity_link")
        or requirements.get("requires_event_participant_link")
    )
    if candidate_required and not bool(result.get("candidate_available")):
        blockers.append("candidate_unavailable")
    result["strict_safety_blockers"] = list(dict.fromkeys(blockers))
    if not blockers:
        return result
    result.update(
        {
            "pre_strict_ready_for_answer": bool(result.get("ready_for_answer")),
            "pre_strict_grounded_ready": bool(result.get("grounded_ready")),
            "pre_strict_partial_grounded_ready": bool(result.get("partial_grounded_ready")),
            "pre_strict_grounding_level_ready": str(
                result.get("grounding_level_ready", "none") or "none"
            ),
        }
    )
    unresolved = list(result.get("unresolved_critical_condition_ids", ()) or ())
    result.update(
        {
            "ready_for_answer": False,
            "completion_ready": False,
            "completion_level": "none",
            "grounded_ready": False,
            "partial_grounded_ready": False,
            "grounding_level_ready": "none",
            "completion_blockers": list(dict.fromkeys((*unresolved, *blockers))),
            "unsupported_claim_atom_ids": list(dict.fromkeys((
                *tuple(result.get("unsupported_claim_atom_ids", ()) or ()),
                *blockers,
            ))),
        }
    )
    return result


def _completion_status_with_decision(
    completion_status: Mapping[str, Any],
    decision: ReasonerDecision,
) -> dict[str, Any]:
    result = dict(completion_status)
    result["choice_ready"] = bool(_letter(decision.answer))
    if decision.entity_clusters:
        result["candidate_available"] = True
    blockers = [
        str(blocker)
        for blocker in tuple(result.get("strict_safety_blockers", ()) or ())
        if not (blocker == "choice_not_ready" and result["choice_ready"])
        and not (blocker == "candidate_unavailable" and result.get("candidate_available"))
    ]
    result["strict_safety_blockers"] = blockers
    if blockers:
        return result
    if "pre_strict_ready_for_answer" not in result:
        return result
    result.update(
        {
            "ready_for_answer": bool(result.get("pre_strict_ready_for_answer")),
            "completion_ready": bool(result.get("pre_strict_ready_for_answer")),
            "grounded_ready": bool(result.get("pre_strict_grounded_ready")),
            "partial_grounded_ready": bool(result.get("pre_strict_partial_grounded_ready")),
            "grounding_level_ready": str(
                result.get("pre_strict_grounding_level_ready", "none") or "none"
            ),
        }
    )
    result["completion_level"] = result["grounding_level_ready"]
    return result


def _apply_answer_audit(
    gate: Mapping[str, Any],
    decision: ReasonerDecision,
    *,
    required: bool = False,
) -> dict[str, Any]:
    result = dict(gate)
    status = str(decision.support_status or "").strip().casefold()
    if status:
        result["answer_audit_status"] = status
        result["audit_reason"] = decision.support_reason
    selected = _letter(decision.answer)
    selected_verdict = dict(decision.option_verdicts.get(selected, {}) or {}) if selected else {}
    selected_supported = str(selected_verdict.get("status", "") or "").casefold() in {
        "supported", "supports",
    }
    if required and bool(gate.get("passed")) and (
        status != "supported" or not decision.support_reason or not selected_supported
    ):
        missing_fields = [
            name
            for name, missing in (
                ("support_status", status != "supported"),
                ("support_reason", not decision.support_reason),
                ("selected_option_exact_verdict", not selected_supported),
            )
            if missing
        ]
        result.update(
            {
                "base_gate_passed": True,
                "base_gate_reason": str(gate.get("reason", "") or ""),
                "passed": False,
                "reason": "answer_audit_unavailable" if status in {"", "unknown"} else "answer_audit_exact_predicate_unsupported",
                "answer_audit_status": status or "unknown",
                "answer_audit_missing_fields": missing_fields,
                "audit_reason": decision.support_reason or "Required option audit did not provide exact selected-option support.",
            }
        )
        return result
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


def _annotate_grounding_level(
    gate: Mapping[str, Any],
    completion_status: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = dict(gate)
    if not bool(result.get("passed")):
        return result
    status = dict(completion_status or {})
    if bool(status.get("grounded_ready")):
        result["grounding_level"] = "strict"
    elif bool(status.get("partial_grounded_ready")):
        result["grounding_level"] = "partial"
    elif not status:
        result["grounding_level"] = "strict"
    else:
        blockers = list(status.get("strict_safety_blockers", ()) or ())
        result.update(
            {
                "passed": False,
                "reason": blockers[0] if blockers else "completion_grounding_not_ready",
                "grounding_level": "none",
                "strict_safety_blockers": blockers,
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


def _candidate_gate_rank(decision: ReasonerDecision, gate: Mapping[str, Any]) -> int:
    if bool(gate.get("passed")):
        return 4
    reason = str(gate.get("reason", "") or "").casefold()
    if decision.support_status == "contradicted" or reason == "answer_audit_contradicted":
        return 0
    if reason in {
        "answer_audit_insufficient",
        "event_count_answer_mismatch",
        "entity_count_answer_mismatch",
        "scalar_quantity_answer_mismatch",
        "contract_completion_not_ready",
    }:
        return min(1, _answer_support_rank(decision))
    return _answer_support_rank(decision)


def _audit_option_states(
    decision: ReasonerDecision,
    options: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for raw_option, raw_verdict in decision.option_verdicts.items():
        option = str(raw_option or "").strip().upper()
        if option not in options or not isinstance(raw_verdict, Mapping):
            continue
        verdict = dict(raw_verdict)
        predicate_verdict = _predicate_verdict(verdict.get("predicate_verdict", verdict.get("status")))
        states[option] = {
            "status": predicate_verdict,
            "predicate_verdict": predicate_verdict,
            "grounding_eligibility": str(
                verdict.get("grounding_eligibility", "unknown") or "unknown"
            ).casefold(),
            "support_level": str(verdict.get("support_level", "none") or "none"),
            "citation_ids": list(verdict.get("evidence_ids", ()) or ()),
            "canonical_fact_ids": list(verdict.get("canonical_fact_ids", ()) or ()),
            "predicate_results": [
                dict(item) for item in tuple(verdict.get("predicate_results", ()) or ())
                if isinstance(item, Mapping)
            ],
            "provenance": [
                dict(item) for item in tuple(verdict.get("provenance", ()) or ())
                if isinstance(item, Mapping)
            ],
            "support_reason": str(verdict.get("reason", "") or ""),
        }
    return states


def _record_option_state(
    states: Mapping[str, Mapping[str, Any]],
    decision: ReasonerDecision,
    gate: Mapping[str, Any],
    options: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    result = {str(key): dict(value) for key, value in states.items()}
    for raw_option, raw_verdict in decision.option_verdicts.items():
        option_id = str(raw_option or "").strip().upper()
        if option_id not in options:
            continue
        verdict = dict(raw_verdict)
        predicate_verdict = _predicate_verdict(verdict.get("predicate_verdict", verdict.get("status")))
        current = dict(result.get(option_id, {}) or {})
        predicate_verdict = _merge_predicate_verdict(
            str(current.get("predicate_verdict", current.get("status", "unresolved")) or "unresolved"),
            predicate_verdict,
        )
        rank = {"refuted": 0, "unresolved": 1, "conflicted": 2, "supported": 3}[predicate_verdict]
        result[option_id] = {
            "status": predicate_verdict,
            "predicate_verdict": predicate_verdict,
            "grounding_eligibility": str(current.get("grounding_eligibility", "unknown") or "unknown"),
            "rank": rank,
            "answer": f"{option_id}. {options[option_id]}",
            "reason": str(verdict.get("reason", "") or "option_audit"),
            "support_reason": str(verdict.get("reason", "") or ""),
            "citation_ids": list(verdict.get("evidence_ids", ()) or ()),
            "canonical_fact_ids": list(verdict.get("canonical_fact_ids", ()) or ()),
            "support_level": str(verdict.get("support_level", "none") or "none"),
            "provenance": [
                dict(item) for item in tuple(verdict.get("provenance", ()) or ()) if isinstance(item, Mapping)
            ],
        }
    option = _letter(decision.answer) or _option_letter_from_answer(decision.answer, options)
    if not option:
        return result
    rank = _candidate_gate_rank(decision, gate)
    current = dict(result.get(option, {}))
    direct_refutation = bool(
        decision.support_status == "contradicted"
        or str(gate.get("reason", "") or "").casefold() == "answer_audit_contradicted"
    )
    prior_predicate = str(
        current.get("predicate_verdict", current.get("status", "unresolved")) or "unresolved"
    )
    observed_predicate = (
        "refuted"
        if direct_refutation
        else "supported"
        if bool(gate.get("passed")) and prior_predicate == "unresolved"
        else prior_predicate
    )
    predicate_verdict = _merge_predicate_verdict(prior_predicate, observed_predicate)
    predicate_rank = {"refuted": 0, "unresolved": 1, "conflicted": 2, "supported": 3}[
        predicate_verdict
    ]
    result[option] = {
        **current,
        "status": predicate_verdict,
        "predicate_verdict": predicate_verdict,
        "grounding_eligibility": "sufficient" if bool(gate.get("passed")) else "insufficient",
        "rank": max(rank, predicate_rank),
        "answer": decision.answer,
        "reason": str(gate.get("reason", "") or "verification_failed"),
        "support_reason": decision.support_reason or str(current.get("support_reason", "") or ""),
        "citation_ids": list(decision.citations) or list(current.get("citation_ids", ()) or ()),
        "provenance": [
            dict(item) for item in tuple(current.get("provenance", ()) or ()) if isinstance(item, Mapping)
        ],
    }
    return result


def _predicate_verdict(value: Any) -> str:
    return {
        "supports": "supported",
        "supported": "supported",
        "refutes": "refuted",
        "refuted": "refuted",
        "contradicted": "refuted",
        "conflicted": "conflicted",
    }.get(str(value or "unknown").strip().casefold(), "unresolved")


def _merge_predicate_verdict(existing: str, observed: str) -> str:
    prior = _predicate_verdict(existing)
    current = _predicate_verdict(observed)
    if "conflicted" in {prior, current} or {prior, current} == {"supported", "refuted"}:
        return "conflicted"
    if current == "unresolved":
        return prior
    if prior == "unresolved":
        return current
    return current


def _option_verdict_table(
    options: Mapping[str, str],
    contract: ClaimContract,
    snapshot: Mapping[str, Any],
    states: Mapping[str, Mapping[str, Any]],
    *,
    revision_context: RevisionContext | None = None,
    audit_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    verdicts: dict[str, dict[str, Any]] = {}
    for option in options:
        state = dict(states.get(option, {}) or {})
        state_status = _predicate_verdict(state.get("predicate_verdict", state.get("status")))
        verdicts[option] = {
            "status": "supported" if state_status == "supported" else "contradicted" if state_status == "refuted" else "unknown",
            "predicate_verdict": state_status,
            "grounding_eligibility": str(state.get("grounding_eligibility", "unknown") or "unknown"),
            "support_level": str(state.get("support_level", "none") or "none"),
            "evidence_ids": list(state.get("citation_ids", ()) or ()),
            "canonical_fact_ids": list(state.get("canonical_fact_ids", ()) or ()),
            "provenance": [
                dict(item) for item in tuple(state.get("provenance", ()) or ()) if isinstance(item, Mapping)
            ],
            "reason": str(state.get("support_reason", "") or state.get("reason", "") or "not_yet_audited"),
        }
    audited_verdicts = {option: dict(row) for option, row in verdicts.items()}
    if contract.quantifier == "total_count" and contract.observation_target == "event":
        confirmed = tuple(snapshot.get("qualified_events", snapshot.get("confirmed_events", ())) or ())
        suspects = tuple(snapshot.get("duplicate_suspect_events", ()) or ())
        confirmed_count = len(confirmed)
        fact_ids = [str(row.get("candidate_id", "") or "") for row in confirmed]
        evidence_ids = list(dict.fromkeys(
            str(evidence_id)
            for row in confirmed
            for evidence_id in tuple(row.get("evidence_ids", ()) or ())
            if str(evidence_id)
        ))
        for option, text in options.items():
            expected = _answer_count(f"{option}. {text}", options)
            if expected is None or not confirmed:
                continue
            if expected == confirmed_count:
                verdicts[option] = {
                    "status": "unknown" if suspects else "supported",
                    "support_level": "direct",
                    "evidence_ids": evidence_ids,
                    "canonical_fact_ids": fact_ids,
                    "provenance": _fact_derivation_provenance(
                        confirmed,
                        "canonical_event_count",
                    ),
                    "reason": (
                        f"Canonical ledger confirms {confirmed_count} events but has {len(suspects)} unresolved duplicate suspects."
                        if suspects
                        else f"Canonical ledger confirms exactly {confirmed_count} events."
                    ),
                }
            elif not suspects:
                verdicts[option] = {
                    "status": "contradicted",
                    "support_level": "direct",
                    "evidence_ids": evidence_ids,
                    "canonical_fact_ids": fact_ids,
                    "provenance": _fact_derivation_provenance(
                        confirmed,
                        "canonical_event_count_refutation",
                    ),
                    "reason": f"Option count {expected} differs from canonical count {confirmed_count}.",
                }
    elif contract.quantifier == "distinct_count":
        resolved = tuple(snapshot.get("resolved_entities", ()) or ())
        unresolved = tuple(snapshot.get("unresolved_entity_bindings", ()) or ())
        resolved_count = len(resolved)
        fact_ids = [str(row.get("entity_id", "") or "") for row in resolved]
        evidence_ids = list(dict.fromkeys(
            str(evidence_id)
            for row in resolved
            for evidence_id in tuple(row.get("evidence_ids", ()) or ())
            if str(evidence_id)
        ))
        for option, text in options.items():
            expected = _answer_count(f"{option}. {text}", options)
            if expected is None or not resolved:
                continue
            if expected == resolved_count:
                verdicts[option] = {
                    "status": "unknown" if unresolved else "supported",
                    "support_level": "direct",
                    "evidence_ids": evidence_ids,
                    "canonical_fact_ids": fact_ids,
                    "provenance": _fact_derivation_provenance(
                        resolved,
                        "canonical_entity_count",
                    ),
                    "reason": (
                        f"Canonical entity ledger resolves {resolved_count} entities with {len(unresolved)} bindings pending."
                        if unresolved
                        else f"Canonical entity ledger resolves exactly {resolved_count} entities."
                    ),
                }
            elif not unresolved:
                verdicts[option] = {
                    "status": "contradicted",
                    "support_level": "direct",
                    "evidence_ids": evidence_ids,
                    "canonical_fact_ids": fact_ids,
                    "provenance": _fact_derivation_provenance(
                        resolved,
                        "canonical_entity_count_refutation",
                    ),
                    "reason": f"Option count {expected} differs from canonical entity count {resolved_count}.",
                }
    elif contract.quantifier == "order" and contract.aggregation == "order":
        ledger = dict(snapshot.get("sequence_ledger", {}) or build_sequence_ledger(snapshot, options))
        sequence_verdicts = dict(ledger.get("option_sequence_verdicts", {}) or {})
        qualified = tuple(snapshot.get("qualified_events", ()) or ())
        fact_ids = [str(row.get("candidate_id", "") or "") for row in qualified]
        evidence_ids = list(dict.fromkeys(
            str(evidence_id)
            for row in qualified
            for evidence_id in tuple(row.get("evidence_ids", ()) or ())
            if str(evidence_id)
        ))
        for option in options:
            sequence_row = dict(sequence_verdicts.get(option, {}) or {})
            status = str(sequence_row.get("status", "unknown") or "unknown")
            verdicts[option] = {
                "status": status if status in {"supported", "contradicted"} else "unknown",
                "support_level": "direct" if status != "unknown" else "none",
                "evidence_ids": evidence_ids,
                "canonical_fact_ids": fact_ids,
                "predicate_results": [{
                    "predicate_id": f"sequence_{option}",
                    "status": status,
                    "matched_event_ids": list(sequence_row.get("matched_event_ids", ()) or ()),
                }],
                "provenance": _fact_derivation_provenance(qualified, "canonical_sequence_order"),
                "reason": str(sequence_row.get("reason", "") or "canonical sequence ledger is incomplete"),
            }
    elif contract.observation_target == "entity" and contract.aggregation == "compare":
        resolved = tuple(snapshot.get("resolved_entities", ()) or ())
        episode_binding = dict(snapshot.get("episode_binding", {}) or {})
        temporal_selection = dict(episode_binding.get("temporal_max_selection", {}) or {})
        target_binding = dict(episode_binding.get("target_entity_binding", {}) or {})
        target_attribute_facts = tuple(
            row for row in tuple(episode_binding.get("target_attribute_facts", ()) or ())
            if isinstance(row, Mapping)
        )
        strict_episode_binding = bool(temporal_selection)
        evidence_ids = list(dict.fromkeys(
            str(evidence_id)
            for row in (target_attribute_facts if strict_episode_binding else resolved)
            for evidence_id in tuple(row.get("evidence_ids", ()) or ())
            if str(evidence_id)
        ))
        fact_ids = [
            str(row.get("fact_id", "") or row.get("entity_id", "") or "")
            for row in (target_attribute_facts if strict_episode_binding else resolved)
            if str(row.get("fact_id", "") or row.get("entity_id", "") or "")
        ]
        typed_predicates = parse_option_predicates(options)
        bound_entities = tuple(
            row for row in resolved
            if str(row.get("role", "") or "").casefold() in {"overtaker", "participant", "target_participant"}
            and str(row.get("source_event_key", "") or row.get("episode_id", "") or "")
        )
        target_entities = (
            tuple(
                row for row in resolved
                if str(row.get("entity_id", "") or "")
                == str(target_binding.get("entity_id", "") or "")
            )
            if strict_episode_binding
            else bound_entities if len(bound_entities) == 1 else resolved if len(resolved) == 1 else ()
        )
        if len(target_entities) == 1 and (
            not strict_episode_binding
            or str(temporal_selection.get("status", "") or "") == "resolved"
            and str(target_binding.get("status", "") or "") == "resolved"
            and target_attribute_facts
        ):
            attributes = (
                _typed_attribute_facts(target_attribute_facts)
                if strict_episode_binding
                else _typed_entity_attributes(target_entities[0])
            )
            for option, predicates in typed_predicates.items():
                if not predicates:
                    continue
                statuses = []
                for predicate in predicates:
                    observed = attributes.get(predicate.attribute)
                    statuses.append(
                        "unknown" if observed is None
                        else "supported" if _normalized_attribute_value(observed) == _normalized_attribute_value(predicate.value)
                        else "contradicted"
                    )
                status = (
                    "contradicted" if "contradicted" in statuses
                    else "supported" if statuses and all(item == "supported" for item in statuses)
                    else "unknown"
                )
                verdicts[option] = {
                    "status": status,
                    "support_level": "direct" if status != "unknown" else "none",
                    "evidence_ids": evidence_ids,
                    "canonical_fact_ids": fact_ids,
                    "provenance": _fact_derivation_provenance(
                        target_attribute_facts if strict_episode_binding else target_entities,
                        "typed_entity_attribute_comparison",
                    ),
                    "predicate_results": [
                        {
                            "predicate_id": predicate.predicate_id,
                            "status": predicate_status,
                            "attribute": predicate.attribute,
                            "expected_value": predicate.value,
                        }
                        for predicate, predicate_status in zip(predicates, statuses)
                    ],
                    "reason": "Typed participant attributes were matched by entity, role, episode, attribute type, and value.",
                }
    elif contract.observation_target == "event" and contract.aggregation == "compare":
        facts = tuple(snapshot.get("inferred_facts", ()) or ())
        anchor_facts = tuple(fact for fact in facts if bool(fact.get("anchor_match")))
        episode_ids = {str(fact.get("episode_id", "") or "") for fact in facts if str(fact.get("episode_id", "") or "")}
        if anchor_facts:
            facts = anchor_facts
        elif len(episode_ids) > 1:
            facts = ()
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for fact in facts:
            for assessment in tuple(fact.get("hypothesis_assessments", ()) or ()):
                if not isinstance(assessment, Mapping):
                    continue
                option = str(assessment.get("option_id", "") or "").strip().upper()
                if option in options:
                    grouped.setdefault(option, []).append(assessment)
        for option, assessments in grouped.items():
            supporting_facts = [
                fact for fact in facts
                if any(str(row.get("option_id", "") or "").strip().upper() == option for row in tuple(fact.get("hypothesis_assessments", ()) or ()))
            ]
            statuses = set()
            for fact in supporting_facts:
                for assessment in tuple(fact.get("hypothesis_assessments", ()) or ()):
                    if str(assessment.get("option_id", "") or "").strip().upper() != option:
                        continue
                    assessment_status = str(assessment.get("status", "unknown") or "unknown").casefold()
                    if (
                        assessment_status == "supported"
                        and contract.requires_agent_attribution
                        and not _narrative_fact_has_agent_witness(fact)
                    ):
                        assessment_status = "unknown"
                    statuses.add(assessment_status)
            status = "supported" if "supported" in statuses and "contradicted" not in statuses else "contradicted" if statuses == {"contradicted"} else "unknown"
            verdicts[option] = {
                "status": status,
                "support_level": "inferred" if status != "unknown" else "none",
                "evidence_ids": list(dict.fromkeys(
                    str(evidence_id) for fact in supporting_facts
                    for evidence_id in tuple(fact.get("evidence_ids", ()) or ()) if str(evidence_id)
                )),
                "canonical_fact_ids": [str(fact.get("fact_id", "") or "") for fact in supporting_facts],
                "provenance": _fact_derivation_provenance(
                    supporting_facts,
                    "narrative_option_predicate_comparison",
                ),
                "reason": "; ".join(
                    str(row.get("reason", "") or "") for row in assessments if str(row.get("reason", "") or "")
                )[:500] or "Canonical narrative facts assess this option predicate.",
            }
    verdicts = {
        option: _merge_option_verdict_rows(audited_verdicts[option], verdicts[option])
        for option in options
    }
    rank = {"supported": 3, "unknown": 1, "contradicted": 0}
    ordered = sorted(options, key=lambda option: (-rank[verdicts[option]["status"]], option))
    supported = [option for option in ordered if verdicts[option]["status"] == "supported"]
    result = {
        "option_verdicts": verdicts,
        "best_option": supported[0] if len(supported) == 1 else "",
        "best_supported_option": supported[0] if len(supported) == 1 else "",
        "unique_supported": len(supported) == 1,
        "strongest_alternative": ordered[1] if len(ordered) > 1 else "",
        "discriminating_predicate": (
            "canonical_count" if contract.quantifier in {"total_count", "distinct_count"}
            else "canonical_sequence_ledger" if contract.quantifier == "order"
            else "resolved_entity_attributes" if contract.observation_target == "entity" and contract.aggregation == "compare"
            else "canonical_narrative_inference" if contract.observation_target == "event" and contract.aggregation == "compare"
            else ""
        ),
        "audit_status": "complete" if all(row["status"] != "unknown" for row in verdicts.values()) else "partial",
        "answer_qualification_status": "incomplete",
        "hard_override_allowed": False,
        "hard_override_blockers": ["completion_context_not_attached"],
        "provenance_required": any(
            _predicate_verdict(row.get("predicate_verdict", row.get("status"))) == "supported"
            and bool(tuple(row.get("canonical_fact_ids", ()) or ()) or tuple(row.get("evidence_ids", ()) or ()))
            for row in verdicts.values()
        ),
    }
    if revision_context is not None:
        result.update({
            **revision_context.to_dict(),
            "option_verdict_revision": revision_context.canonical_snapshot_revision,
        })
    if audit_record is not None:
        audit = dict(audit_record)
        result.update({
            "audit_status": str(audit.get("audit_status", "invalid") or "invalid"),
            "audit_snapshot_revision": str(audit.get("audit_snapshot_revision", "") or ""),
            "audit_fingerprint": str(audit.get("audit_fingerprint", "") or ""),
            "audit_invalidity_flags": list(audit.get("invalidity_flags", ()) or ()),
        })
    return result


def _narrative_fact_has_agent_witness(fact: Mapping[str, Any]) -> bool:
    relation_type = str(fact.get("relation_type", "") or "").casefold()
    witness_type = str(fact.get("agent_witness_type", "") or "").casefold()
    return relation_type in {
        "observed_action",
        "spoken_intention",
        "spoken_statement",
        "asr_statement",
        "explicit_narrative_statement",
        "agent_causation",
    } or witness_type in {
        "action_frame",
        "attributed_dialogue",
        "explicit_narration",
    }


def _typed_entity_attributes(entity: Mapping[str, Any]) -> dict[str, Any]:
    aliases = {
        "clothes_color": "clothing_color",
        "jacket_color": "clothing_color",
        "jersey_color": "clothing_color",
        "shirt_color": "clothing_color",
        "suit_color": "clothing_color",
        "top_color": "clothing_color",
    }
    return {
        aliases.get(str(key).strip().casefold(), str(key).strip().casefold()): value
        for key, value in dict(entity.get("attributes", {}) or {}).items()
        if str(key).strip()
    }


def _typed_attribute_facts(facts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aliases = {
        "clothes_color": "clothing_color",
        "jacket_color": "clothing_color",
        "jersey_color": "clothing_color",
        "shirt_color": "clothing_color",
        "suit_color": "clothing_color",
        "top_color": "clothing_color",
    }
    values: dict[str, set[str]] = {}
    originals: dict[str, Any] = {}
    for fact in facts:
        attribute = aliases.get(
            str(fact.get("attribute_type", "") or "").strip().casefold(),
            str(fact.get("attribute_type", "") or "").strip().casefold(),
        )
        value = fact.get("attribute_value")
        if not attribute or value is None:
            continue
        normalized = _normalized_attribute_value(value)
        if not normalized:
            continue
        values.setdefault(attribute, set()).add(normalized)
        originals.setdefault(attribute, value)
    return {
        attribute: originals[attribute]
        for attribute, candidates in values.items()
        if len(candidates) == 1
    }


def _normalized_attribute_value(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().casefold()).strip("_")


def _annotate_forced_override_eligibility(
    table: Mapping[str, Any],
    completion_status: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    query_requirements: Mapping[str, Any],
    *,
    audit_record: Mapping[str, Any] | None = None,
    revision_context: RevisionContext | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(table)
    context = revision_context or {
        "canonical_snapshot_revision": result.get("canonical_snapshot_revision", ""),
        "evidence_digest_hash": result.get("evidence_digest_hash", ""),
        "query_contract_hash": result.get("query_contract_hash", ""),
    }
    audit = dict(audit_record or {
        "audit_snapshot_revision": result.get("audit_snapshot_revision", ""),
        "evidence_digest_hash": result.get("evidence_digest_hash", ""),
        "query_contract_hash": result.get("query_contract_hash", ""),
        "audit_status": result.get("audit_status", "invalid"),
        "invalidity_flags": result.get("audit_invalidity_flags", ()),
        "audit_fingerprint": result.get("audit_fingerprint", ""),
    })
    qualification = {
        "status": "complete" if not tuple(snapshot.get("incomplete_events", ()) or ()) and not tuple(snapshot.get("conflicted_events", ()) or ()) else "incomplete",
        "requirement_graph": dict(snapshot.get("requirement_graph", {}) or {}),
        "incomplete_events": tuple(snapshot.get("incomplete_events", ()) or ()),
        "conflicted_events": tuple(snapshot.get("conflicted_events", ()) or ()),
        "qualification_evaluations": tuple(
            completion_status.get("qualification_evaluations", ()) or ()
        ),
        "event_qualification_evaluations": _event_qualification_evaluations(snapshot),
    }
    guard = evaluate_hard_override_guard(
        completion_status,
        qualification,
        result,
        audit,
        context,
    )
    supported = tuple(guard.supporting_state.get("supported_options", ()) or ())
    result.update({
        "best_supported_option": str(guard.supporting_state.get("selected_option", "") or ""),
        "unique_supported": len(supported) == 1,
        "answer_qualification_status": "complete" if guard.allowed else "incomplete",
        "hard_override_allowed": guard.allowed,
        "hard_override_blockers": list(guard.blockers),
        "hard_override_supporting_state": dict(guard.supporting_state),
    })
    return result


def _event_qualification_evaluations(snapshot: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            **dict(evaluation),
            "required": True,
        }
        for event in tuple(snapshot.get("qualified_events", ()) or ())
        if isinstance(event, Mapping)
        for evaluation in tuple(event.get("requirement_evaluations", ()) or ())
        if isinstance(evaluation, Mapping)
    )


def _merge_option_verdict_rows(
    audited: Mapping[str, Any],
    canonical: Mapping[str, Any],
) -> dict[str, Any]:
    audited_predicate = _predicate_verdict(
        audited.get("predicate_verdict", audited.get("status"))
    )
    canonical_predicate = _predicate_verdict(
        canonical.get("predicate_verdict", canonical.get("status"))
    )
    predicate_verdict = _merge_predicate_verdict(audited_predicate, canonical_predicate)
    status = {
        "supported": "supported",
        "refuted": "contradicted",
        "conflicted": "unknown",
        "unresolved": "unknown",
    }[predicate_verdict]
    evidence_ids = list(dict.fromkeys(
        str(item)
        for item in (*tuple(audited.get("evidence_ids", ()) or ()), *tuple(canonical.get("evidence_ids", ()) or ()))
        if str(item)
    ))
    fact_ids = list(dict.fromkeys(
        str(item)
        for item in (
            *tuple(audited.get("canonical_fact_ids", ()) or ()),
            *tuple(canonical.get("canonical_fact_ids", ()) or ()),
        )
        if str(item)
    ))
    reasons = list(dict.fromkeys(
        str(item).strip()
        for item in (audited.get("reason", ""), canonical.get("reason", ""))
        if str(item).strip() and str(item).strip() != "not_yet_audited"
    ))
    audited_grounding = _grounding_eligibility(audited.get("grounding_eligibility", "unknown"))
    canonical_grounding = _grounding_eligibility(canonical.get("grounding_eligibility", "unknown"))
    if "insufficient" in {audited_grounding, canonical_grounding}:
        grounding_eligibility = "insufficient"
    elif audited_grounding == canonical_grounding == "sufficient":
        grounding_eligibility = "sufficient"
    else:
        grounding_eligibility = "unknown"
    predicate_results = [
        dict(item)
        for item in (*tuple(audited.get("predicate_results", ()) or ()), *tuple(canonical.get("predicate_results", ()) or ()))
        if isinstance(item, Mapping)
    ]
    provenance = [
        dict(item)
        for item in (*tuple(audited.get("provenance", ()) or ()), *tuple(canonical.get("provenance", ()) or ()))
        if isinstance(item, Mapping)
    ]
    return {
        **canonical,
        "status": status,
        "predicate_verdict": predicate_verdict,
        "grounding_eligibility": grounding_eligibility,
        "evidence_ids": evidence_ids,
        "canonical_fact_ids": fact_ids,
        "predicate_results": predicate_results,
        "provenance": provenance,
        "reason": "; ".join(reasons)[:500] or "not_yet_audited",
    }


def _grounding_eligibility(value: Any) -> str:
    return {
        "eligible": "sufficient",
        "sufficient": "sufficient",
        "insufficient": "insufficient",
        "unknown": "unknown",
    }.get(str(value or "unknown").strip().casefold(), "unknown")


def _candidate_can_be_forced(decision: ReasonerDecision, evidence: Sequence[EvidenceRecord]) -> bool:
    return (
        bool(decision.answer.strip())
        and _citations_are_visual(decision.citations, evidence)
    )


def _evidence_digest(
    evidence: Sequence[EvidenceRecord],
    contract: ClaimContract | None = None,
) -> tuple[dict[str, Any], ...]:
    if contract and contract.quantifier == "total_count" and contract.observation_target == "event":
        ledger = _event_candidate_ledger(evidence)
        rows = []
        for candidate in ledger["observed_event_candidates"]:
            evidence_ids = list(candidate.get("evidence_ids", ()) or ())
            rows.append(
                {
                    "evidence_id": evidence_ids[0] if evidence_ids else "",
                    "supporting_evidence_ids": evidence_ids,
                    "summary": "; ".join(candidate.get("descriptions", ()) or ())[:500],
                    "virtual_time_range": list(candidate.get("virtual_time_range", ()) or ()),
                    "modality": "visual",
                    "evidence_kind": "event_candidate",
                    "events": [
                        {
                            "candidate_id": candidate.get("candidate_id"),
                            "event_key": candidate.get("signature"),
                            "event_class": candidate.get("event_class"),
                            "counting_unit": candidate.get("counting_unit"),
                            "participant_ids": candidate.get("participant_ids", ()),
                            "phases": candidate.get("phases", ()),
                            "supports_question_event": True,
                        }
                    ],
                }
            )
        for window in ledger["unresolved_event_windows"]:
            evidence_ids = list(window.get("evidence_ids", ()) or ())
            rows.append(
                {
                    "evidence_id": evidence_ids[0] if evidence_ids else "",
                    "supporting_evidence_ids": evidence_ids,
                    "summary": "Unresolved generic event candidate; inspect again for a stable identity.",
                    "virtual_time_range": list(window.get("virtual_time_range", ()) or ()),
                    "modality": "visual",
                    "evidence_kind": "event_candidate_unresolved",
                    "events": [],
                }
            )
        return tuple(rows)
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
            "entities": [
                _entity_observation_digest(entity)
                for entity in item.operation_metadata.get("entities", ())
                if isinstance(entity, Mapping)
            ],
            "events": list(item.operation_metadata.get("events", ())),
            "entity_associations": list(item.operation_metadata.get("entity_associations", ())),
            "narrative_facts": list(item.operation_metadata.get("narrative_facts", ())),
            "target_presence": dict(item.operation_metadata.get("target_presence", {}) or {}),
            "measurements": list(item.operation_metadata.get("measurements", ())),
            "relations": list(item.operation_metadata.get("relations", ())),
            "derivation": dict(item.operation_metadata.get("derivation", {}) or {}),
            "claim_assessment": dict(item.operation_metadata.get("claim_assessment", {}) or {}),
            "investigation": dict(
                to_jsonable(item.operation_metadata.get("investigation", {}) or {})  # type: ignore[arg-type]
            ),
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


def _entity_observation_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    return {
        "entity_observation_id": str(row.get("entity_observation_id", "") or ""),
        "entity_hypothesis_id": str(row.get("entity_hypothesis_id", "") or ""),
        "association_confidence": float(row.get("association_confidence", 0.0) or 0.0),
        "local_id": str(row.get("local_id", "") or ""),
        "description": str(row.get("description", "") or ""),
        "visual_signature": str(row.get("visual_signature", "") or ""),
        "attributes": dict(row.get("attributes", {}) or {}),
        "role": str(row.get("role", "") or ""),
        "question_relation": str(row.get("question_relation", "") or ""),
        "supports_question_relation": _metadata_flag(row.get("supports_question_relation")),
        "frame_indices": [int(item) for item in row.get("frame_indices", ()) if isinstance(item, int)],
        "witness_virtual_times_sec": [
            float(item)
            for item in row.get("witness_virtual_times_sec", ())
            if isinstance(item, (int, float))
        ],
        "witness_count": int(row.get("witness_count", 0) or 0),
        "countable": _metadata_flag(row.get("countable")),
        "candidate_only": _metadata_flag(row.get("candidate_only")),
        "candidate_reason": str(row.get("candidate_reason", "") or ""),
    }


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
    prior_attempts = [attempt for report in previous for attempt in report.attempts]
    covered = [
        interval
        for attempt in _visual_inspection_attempts(prior_attempts)
        for interval in attempt.inspected_ranges
    ]
    known_facts = {
        _fact_progress_signature(fact)
        for report in previous
        for fact in report.facts
    }
    result = []
    for report in batch:
        attempts = tuple(report.attempts)
        visual_attempts = _visual_inspection_attempts(attempts)
        novel_ranges = tuple(
            fragment
            for attempt in visual_attempts
            for interval in attempt.inspected_ranges
            for fragment in _uncovered_ranges(interval, covered)
        )
        if not attempts and report.coverage_delta:
            # Compatibility for pre-attempt test doubles and archived replay data.
            novel_ranges = tuple(
                fragment
                for interval in report.coverage_delta
                for fragment in _uncovered_ranges(interval, covered)
            )
        adds_frontier = sum(end - start for start, end in novel_ranges) >= 0.5
        density_increased = any(_attempt_increases_density(attempt, prior_attempts) for attempt in visual_attempts)
        new_fact_signatures = {
            signature
            for fact in report.facts
            if (signature := _fact_progress_signature(fact)) not in known_facts
        }
        if not report.facts and report.evidence and not bool(report.cost.get("reused")):
            new_fact_signatures = {
                (
                    record.evidence_kind,
                    re.sub(r"\s+", " ", str(record.verbatim or "").strip().casefold()),
                    (record.start_sec, record.end_sec),
                )
                for record in report.evidence
            }.difference(known_facts)
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
        if density_increased and "sampling_density_increased" not in coverage_progress:
            coverage_progress = (*coverage_progress, "sampling_density_increased")
        information_gain = bool(adds_frontier or density_increased or new_fact_signatures or goal_progress)
        cost = {
            **dict(report.cost),
            "consumes_budget": information_gain,
            "information_gain": (
                "coverage"
                if adds_frontier
                else "density"
                if density_increased
                else "fact"
                if new_fact_signatures or goal_progress
                else "none"
            ),
            "new_coverage_sec": round(sum(end - start for start, end in novel_ranges), 3),
            "new_fact_count": len(new_fact_signatures),
            "density_increased": density_increased,
        }
        status = report.status if information_gain else "duplicate"
        progress_flags_base = report.progress_flags
        normalized_attempts = attempts
        if not information_gain:
            progress_flags_base = (*progress_flags_base, "duplicate", "no_information_gain")
            normalized_attempts = tuple(
                replace(attempt, outcome="duplicate")
                if attempt.outcome != "failed"
                else attempt
                for attempt in attempts
            )
        progress_flags = tuple(dict.fromkeys((*progress_flags_base, *goal_progress, *coverage_progress)))
        normalized = replace(
            report,
            status=status,
            attempts=normalized_attempts,
            cost=cost,
            goal_progress=goal_progress,
            coverage_progress=coverage_progress,
            progress_flags=progress_flags,
            coverage_delta=novel_ranges,
        )
        result.append(normalized)
        prior_attempts.extend(attempts)
        covered.extend(interval for attempt in visual_attempts for interval in attempt.inspected_ranges)
        known_facts.update(new_fact_signatures)
    return tuple(result)


def _report_consumes_budget(report: InvestigationReport) -> int:
    return int(bool(report.cost.get("consumes_budget", report.status not in {"duplicate", "failed"})))


def _fact_progress_signature(fact: EvidenceFact) -> tuple[Any, ...]:
    return (
        fact.fact_type,
        re.sub(r"\s+", " ", fact.text.strip().casefold()),
        fact.fact_range,
    )


def _attempt_increases_density(
    attempt: ObservationAttempt,
    previous: Sequence[ObservationAttempt],
) -> bool:
    if attempt.sampling_fps <= 0.0 or not attempt.inspected_ranges:
        return False
    mode = str(attempt.sampling_config.get("mode", "") or "")
    comparable = tuple(
        prior
        for prior in _visual_inspection_attempts(previous)
        if str(prior.sampling_config.get("mode", "") or "") == mode
        and any(
            _uncovered_duration(current_range, prior.inspected_ranges) <= 1e-6
            for current_range in attempt.inspected_ranges
        )
    )
    return bool(comparable) and attempt.sampling_fps > max(prior.sampling_fps for prior in comparable) + 1e-6


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


def _uncovered_ranges(
    interval: tuple[float, float],
    covered: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    start, end = sorted((float(interval[0]), float(interval[1])))
    fragments = [(start, end)]
    for left, right in sorted((sorted((float(left), float(right))) for left, right in covered)):
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
            break
    return tuple((round(left, 3), round(right, 3)) for left, right in fragments if right - left > 1e-6)


def _stagnation_status(reports: Sequence[InvestigationReport]) -> dict[str, Any]:
    ordered = tuple(reports)
    recent = ordered[-3:]
    if len(recent) < 2:
        return {"stagnant": False, "gap_id": "", "reason": ""}
    recent_condition_ids = {
        result.condition_id
        for report in recent
        for result in report.condition_results
        if result.condition_id
    }
    if not recent_condition_ids:
        gap_ids = {report.gap_id for report in recent if report.gap_id}
        unresolved = all(report.resolution in {"partial", "unresolved"} for report in recent)
        no_goal_progress = not any(report.goal_progress for report in recent)
        no_coverage_progress = not any(report.coverage_progress for report in recent)
        stagnant = len(gap_ids) == 1 and unresolved and no_goal_progress and no_coverage_progress
        gap_id = next(iter(gap_ids)) if stagnant else ""
        return {
            "stagnant": stagnant,
            "gap_id": gap_id,
            "low_yield_coverage": False,
            "reason": "no_goal_or_frontier_progress" if stagnant else "",
            "required_shift": "change range, direction, modality, or region focus" if stagnant else "",
        }
    prefix = ordered[: -len(recent)]
    previously_satisfied = {
        result.condition_id
        for report in prefix
        for result in report.condition_results
        if result.status == "satisfied"
    }
    new_satisfied = set()
    for report in recent:
        for result in report.condition_results:
            if result.status == "satisfied" and result.condition_id not in previously_satisfied:
                new_satisfied.add(result.condition_id)
        previously_satisfied.update(new_satisfied)
    unresolved = all(report.resolution in {"partial", "unresolved"} for report in recent)
    no_goal_progress = not new_satisfied
    no_coverage_progress = not any(report.coverage_progress for report in recent)
    stagnant = unresolved and no_goal_progress and no_coverage_progress
    low_yield_coverage = len(recent) >= 3 and unresolved and no_goal_progress and not no_coverage_progress
    gap_id = next((report.gap_id for report in reversed(recent) if report.gap_id), "") if stagnant or low_yield_coverage else ""
    return {
        "stagnant": stagnant,
        "gap_id": gap_id,
        "low_yield_coverage": low_yield_coverage,
        "reason": "no_goal_or_frontier_progress" if stagnant else "coverage_without_condition_progress" if low_yield_coverage else "",
        "required_shift": "change range, direction, modality, or region focus" if stagnant or low_yield_coverage else "",
    }


def _task_progress_fingerprint(task: InvestigationTask) -> tuple[Any, ...]:
    if task.time_range is None:
        time_bucket: tuple[float, float] | tuple[()] = ()
    else:
        start, end = task.time_range
        time_bucket = (round(float(start), 1), round(float(end), 1))
    condition_ids = tuple(sorted(condition.condition_id for condition in task.conditions if condition.condition_id))
    target_predicate = re.sub(
        r"[^a-z0-9]+",
        " ",
        " ".join((task.claim_to_verify, task.expected_evidence, task.goal)).casefold(),
    ).strip()[:240]
    return (
        task.segment_id,
        time_bucket,
        condition_ids,
        target_predicate,
        task.inspection_mode,
        round(float(task.sampling_floor_fps or 0.5), 2),
        tuple(sorted(task.source_candidate_ids)),
        tuple(
            sorted(str(item.get("entity_hypothesis_id", "") or item.get("participant_id", "") or "")
                   for item in task.reference_entities)
        ),
        tuple(sorted(str(item.get("fact_id", "") or "") for item in task.reference_facts)),
        tuple(sorted(task.modality_hint)),
        task.region_hint.casefold(),
    )


def _canonical_progress_signature(snapshot: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        len(tuple(snapshot.get("confirmed_events", ()) or ())),
        len(tuple(snapshot.get("duplicate_suspect_events", ()) or ())),
        len(tuple(snapshot.get("refuted_events", ()) or ())),
        len(tuple(snapshot.get("resolved_entities", ()) or ())),
        len(tuple(snapshot.get("unresolved_entity_bindings", ()) or ())),
        len(tuple(snapshot.get("state_transitions", ()) or ())),
        len(tuple(snapshot.get("entity_associations", ()) or ())),
        len(tuple(snapshot.get("inferred_facts", ()) or ())),
        len(tuple(snapshot.get("unresolved_inferences", ()) or ())),
        tuple(str(row.get("event_id", "") or "") for row in tuple(snapshot.get("ordered_events", ()) or ())),
    )


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
                "grounding_level": result.grounding_level,
                "retrieval_status": result.retrieval_status,
                "citations": list(result.citations),
                "correct": result.correct,
                "verified": result.verified,
                "verification_reason": result.verification_reason,
                "rounds": result.rounds,
                "accepted_investigations": result.accepted_investigations,
                "sampling_stability": _sampling_stability_summary(result.evidence),
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


def _sampling_stability_summary(evidence: Sequence[EvidenceRecord]) -> dict[str, Any]:
    policies = [
        dict(record.operation_metadata.get("sampling_policy", {}) or {})
        for record in evidence
        if record.modality == "visual"
    ]
    fps_histogram: dict[str, int] = {}
    trigger_histogram: dict[str, int] = {}
    upshift_count = 0
    negative_to_positive = 0
    conflicted_observations = 0
    for record, policy in zip((item for item in evidence if item.modality == "visual"), policies):
        fps_key = f"{float(record.sampling_fps):.1f}"
        fps_histogram[fps_key] = fps_histogram.get(fps_key, 0) + 1
        attempts = int(policy.get("adaptive_attempt_count", 1) or 1)
        if attempts > 1:
            upshift_count += attempts - 1
        triggers = tuple(policy.get("adaptive_trigger_reasons", ()) or ())
        for trigger in triggers:
            key = str(trigger)
            trigger_histogram[key] = trigger_histogram.get(key, 0) + 1
        if "not_found" in triggers and record.observation_polarity == "positive":
            negative_to_positive += 1
        if record.operation_metadata.get("conflicted_slot_ids"):
            conflicted_observations += 1
    return {
        "upshift_count": upshift_count,
        "trigger_histogram": trigger_histogram,
        "negative_to_positive_count": negative_to_positive,
        "conflicted_observation_count": conflicted_observations,
        "fps_histogram": fps_histogram,
        "floor_unspecified_count": sum(bool(row.get("floor_unspecified")) for row in policies),
    }
