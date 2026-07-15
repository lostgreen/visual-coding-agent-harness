from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import re
from typing import Any, Mapping, Sequence

from vcah.evidence_primitives import (
    GapCondition,
    MeasurementFact,
    canonical_unit,
    make_gap_conditions,
    merge_condition_states,
    normalize_measurements,
    normalize_relations,
)
from vcah.investigator import InvestigationReport, VirtualVideoInvestigator
from vcah.memory import EvidenceStore
from vcah.semantic_evidence import event_candidate_ledger as _event_candidate_ledger, semantic_repair_requests
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
    sampling_floor_fps: float = 0.5

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
        floor_fps = float(self.sampling_floor_fps or 0.5)
        object.__setattr__(self, "sampling_floor_fps", min(2.0, max(0.5, floor_fps)))
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
    spatial = _is_spatial_relation_question(text)
    contrastive_subject = bool(
        re.search(r"\bone\s+(?:team|group|person)\b[^?]*\bthe\s+other\s+(?:team|group|person)\b", text)
    )
    temporal_sequence = bool(
        re.search(r"\b(?:sequence|order|before|after|then|first|last|initially|ultimately)\b", text)
    )
    state_tracking = bool(
        re.search(r"\b(?:change|changed|turn(?:ed)?|remain(?:ed)?|maintain(?:ed)?|overtak\w*|track\w*)\b", text)
    )
    return {
        "requires_identity_link": bool(terms),
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
        "requires_state_tracking": state_tracking,
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
        grounding_level = "none"
        verification_reason = ""
        best_answer = ""
        best_citations: tuple[str, ...] = ()
        best_verification_reason = ""
        best_support_rank = -1
        option_states: dict[str, dict[str, Any]] = {}
        last_gate_reason = "answer_missing"
        gate_feedback: dict[str, Any] = {}
        condition_registry: tuple[GapCondition, ...] = ()
        active_condition_ids: tuple[str, ...] = ()
        last_rejected_submission: tuple[str, tuple[str, ...], tuple[str, ...]] | None = None
        rounds_run = 0

        for round_id in range(1, self.max_rounds + 1):
            rounds_run = round_id
            remaining = self.max_investigations - accepted
            completion_status = _completion_status(
                workspace,
                query_contract,
                evidence_store.records,
                query_requirements=query_requirements,
                reports=reports,
                best_choice=best_answer,
                active_condition_ids=active_condition_ids,
            )
            navigation_candidates = _navigation_candidates(evidence_store.records, workspace.case.options)
            stagnation_status = _stagnation_status(reports)
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
                    remaining_budget=remaining,
                    pre_final_checkpoint=round_id == self.max_rounds,
                )
            ), condition_registry)
            decision = _bind_gap_to_tasks(decision)
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
            missing_segments = tuple(completion_status.get("missing_segment_ids", ()) or ())
            missing_identity_terms = tuple(completion_status.get("missing_identity_anchor_terms", ()) or ())
            unresolved_entity_candidates = tuple(
                completion_status.get("unresolved_candidate_entity_observation_ids", ()) or ()
            )
            if missing_segments and remaining > 0:
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
                        "reason": "mandatory_full_source_coverage",
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
            elif unresolved_entity_candidates and remaining > 0 and (
                decision.action != "investigate" or not decision.tasks
            ):
                repair_tasks = _entity_candidate_repair_tasks(
                    workspace,
                    evidence_store.records,
                    unresolved_entity_candidates,
                    round_id=round_id,
                    limit=min(self.max_tasks_per_round, remaining),
                )
                if repair_tasks:
                    trace.append(
                        {
                            "type": "repair_override",
                            "round": round_id,
                            "reason": "entity_candidate_unresolved",
                            "entity_observation_ids": list(unresolved_entity_candidates),
                            "task_count": len(repair_tasks),
                        }
                    )
                    decision = ReasonerDecision(action="investigate", tasks=repair_tasks)
            elif remaining > 0:
                repair_reason, repair_tasks = _semantic_contract_repair_tasks(
                    workspace,
                    query_contract,
                    query_requirements,
                    completion_status,
                    evidence_store.records,
                    round_id=round_id,
                    limit=min(self.max_tasks_per_round, remaining),
                )
                if repair_tasks:
                    trace.append(
                        {
                            "type": "repair_override",
                            "round": round_id,
                            "reason": repair_reason,
                            "task_count": len(repair_tasks),
                        }
                    )
                    decision = ReasonerDecision(action="investigate", tasks=repair_tasks)
            if decision.action == "answer" and remaining > 0:
                candidate_repairs = _navigation_repair_tasks(
                    evidence_store.records,
                    options=workspace.case.options,
                    round_id=round_id,
                    limit=1,
                    require_hypothesis=True,
                )
                if candidate_repairs:
                    trace.append({
                        "type": "repair_override",
                        "round": round_id,
                        "reason": "high_value_candidate_uninspected",
                        "source_candidate_ids": list(candidate_repairs[0].source_candidate_ids),
                        "task_count": 1,
                    })
                    decision = ReasonerDecision(action="investigate", tasks=candidate_repairs)
            if decision.action == "investigate" and not decision.tasks and remaining > 0:
                bootstrap_tasks = _bootstrap_investigation_tasks(
                    workspace,
                    query_contract,
                    round_id=round_id,
                    limit=min(self.max_tasks_per_round, remaining),
                )
                if bootstrap_tasks:
                    trace.append(
                        {
                            "type": "repair_override",
                            "round": round_id,
                            "reason": "empty_investigation_bootstrap",
                            "task_count": len(bootstrap_tasks),
                        }
                    )
                    decision = ReasonerDecision(action="investigate", tasks=bootstrap_tasks)
            submission_fingerprint = _submission_fingerprint(decision, evidence_store.records)
            if (
                decision.action == "answer"
                and remaining > 0
                and last_rejected_submission is not None
                and submission_fingerprint == last_rejected_submission
            ):
                repair_tasks = _rejected_answer_repair_tasks(
                    workspace,
                    query_contract,
                    decision,
                    evidence_store.records,
                    gate_feedback,
                    round_id=round_id,
                    limit=min(self.max_tasks_per_round, remaining),
                )
                if repair_tasks:
                    trace.append(
                        {
                            "type": "repair_override",
                            "round": round_id,
                            "reason": "repeated_rejected_submission",
                            "previous_gate_reason": str(gate_feedback.get("reason", "") or ""),
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
                    completion_status=completion_status,
                )
                gate = _annotate_grounding_level(gate, completion_status)
                gate = _apply_answer_audit(
                    gate,
                    decision,
                    required=_requires_discriminative_audit(query_contract, query_requirements),
                )
                trace.append({"type": "completion_gate", "round": round_id, **gate})
                last_gate_reason = str(gate.get("reason", "") or "verification_failed")
                option_states = _record_option_state(
                    option_states,
                    decision,
                    gate,
                    workspace.case.options,
                )
                gate_feedback = {**dict(gate), "option_states": option_states}
                if not gate["passed"]:
                    last_rejected_submission = _submission_fingerprint(decision, evidence_store.records)
                support_rank = _candidate_gate_rank(decision, gate)
                if _candidate_can_be_forced(decision, evidence_store.records) and support_rank > best_support_rank:
                    best_answer = decision.answer
                    best_citations = decision.citations
                    best_verification_reason = last_gate_reason
                    best_support_rank = support_rank
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
            raw_tasks = decision.tasks[: min(self.max_tasks_per_round, remaining)]
            resolved_tasks = _resolve_workspace_tasks(
                workspace,
                raw_tasks,
                limit=min(self.max_tasks_per_round, remaining),
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
                resolved_tasks = _bootstrap_investigation_tasks(
                    workspace,
                    query_contract,
                    round_id=round_id,
                    limit=min(self.max_tasks_per_round, remaining),
                )
            requested_tasks = tuple(_task_for_contract(task, query_contract) for task in resolved_tasks)
            tasks = _prefer_navigation_repairs(
                requested_tasks,
                evidence_store.records,
                options=workspace.case.options,
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
            task_condition_ids = tuple(
                dict.fromkeys(
                    condition.condition_id
                    for task in tasks
                    for condition in task.conditions
                    if condition.condition_id
                )
            )
            if task_condition_ids:
                active_condition_ids = task_condition_ids
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
            followup_tasks = _post_search_candidate_tasks(
                tasks,
                evidence_store.records,
                options=workspace.case.options,
                round_id=round_id,
                remaining_round_slots=max(0, self.max_tasks_per_round - len(tasks)),
                remaining_budget=max(0, self.max_investigations - accepted),
            )
            if followup_tasks:
                accepted += len(followup_tasks)
                followup_batch = _annotate_batch_progress(investigator.run_batch(followup_tasks), reports)
                reports.extend(followup_batch)
                known_evidence = {record.evidence_id for record in evidence_store.records}
                for report in followup_batch:
                    for record in report.evidence:
                        if record.evidence_id not in known_evidence:
                            evidence_store.add(record)
                            known_evidence.add(record.evidence_id)
                trace.append({
                    "type": "candidate_dispatch_conversion",
                    "round": round_id,
                    "candidate_ids": [candidate_id for task in followup_tasks for candidate_id in task.source_candidate_ids],
                    "accepted_tasks": len(followup_tasks),
                    "outcomes": list(_outcome_digest(followup_batch)),
                })
            if accepted >= self.max_investigations:
                continue

        if not answer and evidence_store.records:
            completion_status = _completion_status(
                workspace,
                query_contract,
                evidence_store.records,
                query_requirements=query_requirements,
                reports=reports,
                best_choice=best_answer,
                active_condition_ids=active_condition_ids,
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
                gate = _answer_completion_gate(
                    workspace,
                    query_contract,
                    final_decision.answer,
                    final_decision.citations,
                    final_decision.entity_clusters,
                    evidence_store.records,
                    query_requirements=query_requirements,
                    completion_status=completion_status,
                )
                gate = _annotate_grounding_level(gate, completion_status)
                gate = _apply_answer_audit(
                    gate,
                    final_decision,
                    required=_requires_discriminative_audit(query_contract, query_requirements),
                )
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

        if verified:
            final_answer = answer
            final_citations = citations
            grounded_answer = answer
            forced_answer = answer
            answer_mode = "grounded"
            grounding_status = f"verified_{grounding_level}"
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
                "grounding_level": grounding_level,
                "retrieval_status": retrieval_status,
                "verified": verified,
                "verification_reason": verification_reason,
                "best_effort": bool(final_answer and not verified and best_answer),
                "option_states": option_states,
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


def _bootstrap_investigation_tasks(
    workspace: VirtualVideoWorkspace,
    contract: ClaimContract,
    *,
    round_id: int,
    limit: int,
) -> tuple[InvestigationTask, ...]:
    segment_limit = max(0, int(limit))
    if segment_limit <= 0:
        return ()
    segments = tuple(workspace.manifest.segments)
    if contract.required_scope != "full_video":
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
        return _coverage_repair_tasks(round_id, missing_segments, contract, limit=task_limit)
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
    return _bootstrap_investigation_tasks(workspace, contract, round_id=round_id, limit=task_limit)


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


def _prefer_navigation_repairs(
    requested: Sequence[InvestigationTask],
    evidence: Sequence[EvidenceRecord],
    *,
    options: Mapping[str, str] | None = None,
    round_id: int,
    limit: int,
) -> tuple[InvestigationTask, ...]:
    bounded = tuple(requested[: max(0, int(limit))])
    has_search = any(task.inspection_mode == "search_asr" for task in bounded)
    repairs = _navigation_repair_tasks(
        evidence,
        options=options,
        round_id=round_id,
        limit=1,
        require_hypothesis=not has_search,
    )
    if not repairs:
        return bounded
    kept = tuple(task for task in bounded if task.inspection_mode != "search_asr")
    if has_search:
        return (*kept[: max(0, int(limit) - 1)], *repairs)
    return (*repairs, *kept[: max(0, int(limit) - 1)])


def _post_search_candidate_tasks(
    completed_tasks: Sequence[InvestigationTask],
    evidence: Sequence[EvidenceRecord],
    *,
    options: Mapping[str, str],
    round_id: int,
    remaining_round_slots: int,
    remaining_budget: int,
) -> tuple[InvestigationTask, ...]:
    if not any(task.inspection_mode == "search_asr" for task in completed_tasks):
        return ()
    if min(int(remaining_round_slots), int(remaining_budget)) <= 0:
        return ()
    return _navigation_repair_tasks(
        evidence,
        options=options,
        round_id=round_id,
        limit=1,
        require_hypothesis=True,
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
    sampling_floor = max(
        float(task.sampling_floor_fps),
        2.0
        if contract.aggregation == "order"
        or (contract.observation_target == "entity" and contract.aggregation == "compare")
        else 1.0
        if contract.quantifier in {"distinct_count", "total_count"}
        or contract.aggregation == "compare"
        else 0.5,
    )
    task = replace(task, sampling_floor_fps=sampling_floor)
    if contract.aggregation == "summarize" and task.inspection_mode != "search_asr":
        expected = str(task.expected_evidence or "").strip()
        narrative = (
            "segment narrative thesis and role (setup, example, process detail, comparison, or conclusion), "
            "grounded in local ASR and kept separate from visually salient examples"
        )
        return replace(
            task,
            modality_hint=("visual", "asr"),
            expected_evidence=f"{expected}; {narrative}" if expected else narrative,
        )
    if (
        task.inspection_mode == "window"
        and contract.quantifier == "total_count"
        and contract.observation_target == "event"
    ):
        return replace(task, inspection_mode="enumerate_events")
    if contract.required_scope == "multi_window" and contract.aggregation == "compare":
        expected = str(task.expected_evidence or "").strip()
        transition = (
            "continuous before-transition-after evidence with ordered timestamps and stable subject/event identity"
        )
        return replace(
            task,
            modality_hint=tuple(dict.fromkeys((*task.modality_hint, "visual", "motion"))),
            expected_evidence=f"{expected}; {transition}" if expected else transition,
        )
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
    reports: Sequence[InvestigationReport] = (),
    best_choice: str = "",
    active_condition_ids: Sequence[str] = (),
) -> dict[str, Any]:
    answer_evidence = tuple(record for record in evidence if record.evidence_kind != "navigation_hint")
    navigation_evidence = tuple(record for record in evidence if record.evidence_kind == "navigation_hint")
    coverage_evidence = (
        _entity_census_coverage_evidence(answer_evidence)
        if contract.quantifier == "distinct_count"
        else answer_evidence
    )
    coverage = _source_coverage(workspace, coverage_evidence)
    if contract.required_scope != "full_video":
        base = _apply_identity_completion(
            {
                "ready_for_answer": bool(answer_evidence),
                "required_scope": contract.required_scope,
                "missing_segment_ids": [],
                "source_coverage": coverage,
            },
            answer_evidence,
            query_requirements,
        )
        base = _apply_entity_completion(base, contract, answer_evidence)
        base = _apply_event_completion(base, contract, answer_evidence)
        return _apply_readiness_dashboard(
            base,
            answer_evidence,
            navigation_evidence,
            reports,
            best_choice,
            active_condition_ids,
        )
    if not coverage:
        base = _apply_identity_completion(
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
        base = _apply_entity_completion(base, contract, answer_evidence)
        base = _apply_event_completion(base, contract, answer_evidence)
        return _apply_readiness_dashboard(
            base,
            answer_evidence,
            navigation_evidence,
            reports,
            best_choice,
            active_condition_ids,
        )
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
            "adopted_source_video_id": adopted_source,
            "missing_segment_ids": missing,
            "source_coverage": coverage,
        },
        answer_evidence,
        query_requirements,
    )
    base = _apply_entity_completion(base, contract, answer_evidence)
    base = _apply_event_completion(base, contract, answer_evidence)
    return _apply_readiness_dashboard(
        base,
        answer_evidence,
        navigation_evidence,
        reports,
        best_choice,
        active_condition_ids,
    )


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
    unresolved = sorted(
        missing_active_ids
        | {condition_id for condition_id, state in states.items() if state.status != "satisfied"}
    )
    base_ready = bool(result.get("ready_for_answer"))
    retrieval_ready = any(record.modality in {"visual", "ocr"} for record in answer_evidence)
    condition_total = len(states) + len(missing_active_ids)
    satisfied_count = sum(state.status == "satisfied" for state in states.values())
    conflict_count = sum(state.status in {"contradicted", "refuted"} for state in states.values())
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
            float(row.get("covered_count", 0) or 0) / max(1, int(row.get("required_count", 0) or 0))
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
            float(row.get("covered_count", 0) or 0) / max(1, int(row.get("required_count", 0) or 0))
            for row in coverage_rows
        ),
        default=0.0,
    )
    result = {}
    for condition_id, state in states.items():
        if (
            getattr(state, "scope", "window") == "full_video"
            and state.status == "satisfied"
            and coverage_ratio + 1e-9 < float(getattr(state, "required_coverage", 1.0) or 1.0)
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
) -> dict[str, Any]:
    result = dict(status)
    if contract.quantifier != "total_count" or contract.observation_target != "event":
        return result
    ledger = _event_candidate_ledger(evidence)
    result.update(ledger)
    result["ready_for_answer"] = (
        bool(result.get("ready_for_answer"))
        and bool(ledger["confirmed_event_candidates"])
        and not ledger["unresolved_event_windows"]
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
    completion_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not str(answer or "").strip():
        return {"passed": False, "reason": "answer_missing", "missing_segment_ids": []}
    if not (_letter(answer) or _option_letter_from_answer(answer, workspace.case.options)):
        return {"passed": False, "reason": "invalid_option_answer", "missing_segment_ids": []}
    if not _citations_are_visual(citations, evidence):
        return {"passed": False, "reason": "invalid_visual_citations", "missing_segment_ids": []}
    readiness_gate = _contract_readiness_gate(contract, completion_status)
    if readiness_gate is not None:
        return readiness_gate
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
        return _spatial_relation_gate(
            workspace,
            answer,
            cited,
            query_requirements=query_requirements,
        )
    quantitative_gate = _quantitative_answer_gate(
        workspace,
        contract,
        answer,
        cited,
        query_requirements=query_requirements,
    )
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
    coverage_evidence = _entity_census_coverage_evidence(evidence) if contract.quantifier == "distinct_count" else evidence
    source_coverage = _source_coverage(workspace, coverage_evidence)
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
        event_occurrences = tuple(ledger["confirmed_event_candidates"])
        if ledger["unresolved_event_windows"]:
            return {
                "passed": False,
                "reason": "event_candidate_reconciliation_incomplete",
                "confirmed_event_candidate_count": len(event_occurrences),
                "unresolved_event_windows": ledger["unresolved_event_windows"],
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
    return {
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
    status = dict(completion_status or {})
    if not status or bool(status.get("ready_for_answer")):
        return None
    requires_semantic_closure = (
        contract.required_scope == "full_video"
        or contract.observation_target == "relation"
        or bool(contract.boundary_hint)
        or contract.quantifier in {"universal", "comparison"}
        or (contract.observation_target == "event" and contract.aggregation == "compare")
    )
    if not requires_semantic_closure:
        return None
    return {
        "passed": False,
        "reason": "contract_completion_not_ready",
        "completion_reason": str(status.get("reason", "") or ""),
        "unresolved_critical_condition_ids": list(
            status.get("unresolved_critical_condition_ids", ()) or ()
        ),
        "unsupported_claim_atom_ids": list(status.get("unsupported_claim_atom_ids", ()) or ()),
        "missing_segment_ids": list(status.get("missing_segment_ids", ()) or ()),
    }


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
        probes.append((record.evidence_id, status))
    positive = [evidence_id for evidence_id, status in probes if status == "present"]
    if positive:
        return {
            "passed": False,
            "reason": "global_absence_contradicted",
            "positive_evidence_ids": positive,
            "missing_segment_ids": [],
        }
    cited_ids = {record.evidence_id for record in cited}
    negative = [evidence_id for evidence_id, status in probes if status == "absent" and evidence_id in cited_ids]
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
        sampling_floor_fps=float(value.get("sampling_floor_fps", 0.5) or 0.5),
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
            conditions=gap.conditions or task.conditions,
        )
        for task in decision.tasks
    )
    return replace(decision, tasks=tasks)


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
            aligned_condition = replace(condition, condition_id=match.condition_id)
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
        primary_gap=_gap(value["primary_gap"]) if isinstance(value.get("primary_gap"), Mapping) else None,
    )


def _requires_discriminative_audit(
    contract: ClaimContract,
    query_requirements: Mapping[str, Any] | None,
) -> bool:
    requirements = dict(query_requirements or {})
    return bool(
        contract.aggregation in {"order", "compare"}
        or contract.quantifier in {"order", "comparison"}
        or requirements.get("requires_temporal_sequence")
        or requirements.get("requires_state_tracking")
        or requirements.get("requires_identity_link")
    )


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
    if required and bool(gate.get("passed")) and (status != "supported" or not decision.support_reason):
        result.update(
            {
                "base_gate_passed": True,
                "base_gate_reason": str(gate.get("reason", "") or ""),
                "passed": False,
                "reason": "answer_audit_missing",
                "answer_audit_status": status or "missing",
                "audit_reason": decision.support_reason,
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
    else:
        result["grounding_level"] = "strict"
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


def _record_option_state(
    states: Mapping[str, Mapping[str, Any]],
    decision: ReasonerDecision,
    gate: Mapping[str, Any],
    options: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    result = {str(key): dict(value) for key, value in states.items()}
    option = _letter(decision.answer) or _option_letter_from_answer(decision.answer, options)
    if not option:
        return result
    rank = _candidate_gate_rank(decision, gate)
    current = dict(result.get(option, {}))
    current_rank = int(current.get("rank", -1) or -1)
    direct_refutation = bool(
        decision.support_status == "contradicted"
        or str(gate.get("reason", "") or "").casefold() == "answer_audit_contradicted"
    )
    prior_status = str(current.get("status", "") or "")
    status = "supported" if bool(gate.get("passed")) else "refuted" if direct_refutation else "unresolved"
    if {prior_status, status} == {"supported", "refuted"}:
        status = "conflicted"
    if rank >= current_rank or status in {"refuted", "conflicted"}:
        result[option] = {
            "status": status,
            "rank": max(rank, current_rank) if status == "conflicted" else rank,
            "answer": decision.answer,
            "reason": str(gate.get("reason", "") or "verification_failed"),
            "support_reason": decision.support_reason,
            "citation_ids": list(decision.citations),
        }
    return result


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
        for candidate in ledger["confirmed_event_candidates"]:
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
        "local_id": str(row.get("local_id", "") or ""),
        "description": str(row.get("description", "") or ""),
        "visual_signature": str(row.get("visual_signature", "") or ""),
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
