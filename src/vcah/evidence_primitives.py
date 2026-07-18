from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class GapCondition:
    condition_id: str
    description: str
    critical: bool = True
    condition_type: str = "auto"
    target_role: str = ""
    quantity_type: str = ""
    unit: str = ""
    relation_type: str = ""
    subject_role: str = ""
    object_role: str = ""
    required_relation: str = ""
    scope: str = "auto"
    quantifier: str = "auto"
    required_coverage: float = 0.0
    aggregation: str = "none"

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition_id", str(self.condition_id or "").strip())
        object.__setattr__(self, "description", str(self.description or "").strip())
        object.__setattr__(self, "critical", bool(self.critical))
        condition_type = str(self.condition_type or "auto").strip().casefold()
        allowed = {"auto", "semantic", "lexical_navigation", "presence", "measurement", "relation", "temporal"}
        object.__setattr__(self, "condition_type", condition_type if condition_type in allowed else "auto")
        for name in ("target_role", "quantity_type", "relation_type", "subject_role", "object_role", "required_relation"):
            object.__setattr__(self, name, str(getattr(self, name) or "").strip().casefold())
        object.__setattr__(self, "unit", canonical_unit(self.unit))
        description = self.description.casefold()
        scope = str(self.scope or "auto").strip().casefold()
        if scope == "auto":
            scope = (
                "full_video"
                if _requires_global_scope(description) or _requires_temporal_max(description)
                else "episode"
                if _requires_ordinal_participant(description)
                else "window"
            )
        object.__setattr__(self, "scope", scope if scope in {"window", "segment", "episode", "full_video"} else "window")
        quantifier = str(self.quantifier or "auto").strip().casefold()
        if quantifier == "auto":
            quantifier = (
                "temporal_max"
                if _requires_temporal_max(description)
                else "ordinal_2"
                if _requires_ordinal_participant(description)
                else "all_events"
                if _requires_event_union(description)
                else "all_segments"
                if scope == "full_video"
                else "exists"
            )
        object.__setattr__(
            self,
            "quantifier",
            quantifier if quantifier in {"exists", "all_segments", "all_events", "temporal_max", "ordinal_2"} else "exists",
        )
        required_coverage = max(0.0, min(1.0, float(self.required_coverage or 0.0)))
        if scope == "full_video" and required_coverage == 0.0:
            required_coverage = 1.0
        object.__setattr__(self, "required_coverage", required_coverage)
        aggregation = str(self.aggregation or "none").strip().casefold()
        if aggregation == "none" and quantifier == "all_events":
            aggregation = "event_union"
        elif aggregation == "none" and quantifier == "temporal_max":
            aggregation = "temporal_max"
        elif aggregation == "none" and quantifier == "ordinal_2":
            aggregation = "ordinal"
        object.__setattr__(self, "aggregation", aggregation)


@dataclass(frozen=True)
class ConditionResult:
    condition_id: str
    status: str
    observation: str = ""
    evidence_ids: tuple[str, ...] = ()
    scope: str = "window"
    quantifier: str = "exists"
    required_coverage: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition_id", str(self.condition_id or "").strip())
        status = str(self.status or "unknown").strip().casefold()
        aliases = {"resolved": "satisfied", "partial": "unknown", "unresolved": "unknown", "refuted": "contradicted"}
        status = aliases.get(status, status)
        object.__setattr__(self, "status", status if status in {"satisfied", "unknown", "contradicted"} else "unknown")
        object.__setattr__(self, "observation", str(self.observation or "").strip())
        object.__setattr__(
            self,
            "evidence_ids",
            tuple(dict.fromkeys(str(item).strip() for item in self.evidence_ids if str(item).strip())),
        )
        scope = str(self.scope or "window").strip().casefold()
        object.__setattr__(self, "scope", scope if scope in {"window", "segment", "episode", "full_video"} else "window")
        quantifier = str(self.quantifier or "exists").strip().casefold()
        object.__setattr__(
            self,
            "quantifier",
            quantifier if quantifier in {"exists", "all_segments", "all_events", "temporal_max", "ordinal_2"} else "exists",
        )
        object.__setattr__(self, "required_coverage", max(0.0, min(1.0, float(self.required_coverage or 0.0))))


@dataclass(frozen=True)
class ConditionState:
    condition_id: str
    status: str = "unknown"
    supporting_evidence_ids: tuple[str, ...] = ()
    refuting_evidence_ids: tuple[str, ...] = ()
    updated_by_task_id: str | None = None
    scope: str = "window"
    quantifier: str = "exists"
    required_coverage: float = 0.0

    def __post_init__(self) -> None:
        status = str(self.status or "unknown").strip().casefold()
        object.__setattr__(self, "condition_id", str(self.condition_id or "").strip())
        object.__setattr__(self, "status", status if status in {"unknown", "satisfied", "refuted", "conflicted"} else "unknown")
        object.__setattr__(self, "supporting_evidence_ids", _unique_strings(self.supporting_evidence_ids))
        object.__setattr__(self, "refuting_evidence_ids", _unique_strings(self.refuting_evidence_ids))
        task_id = str(self.updated_by_task_id or "").strip()
        object.__setattr__(self, "updated_by_task_id", task_id or None)
        scope = str(self.scope or "window").strip().casefold()
        object.__setattr__(self, "scope", scope if scope in {"window", "segment", "episode", "full_video"} else "window")
        quantifier = str(self.quantifier or "exists").strip().casefold()
        object.__setattr__(
            self,
            "quantifier",
            quantifier if quantifier in {"exists", "all_segments", "all_events", "temporal_max", "ordinal_2"} else "exists",
        )
        object.__setattr__(self, "required_coverage", max(0.0, min(1.0, float(self.required_coverage or 0.0))))


def merge_condition_states(updates: Sequence[tuple[str, ConditionResult]]) -> dict[str, ConditionState]:
    states: dict[str, ConditionState] = {}
    for task_id, result in updates:
        current = states.get(result.condition_id, ConditionState(result.condition_id))
        supporting = list(current.supporting_evidence_ids)
        refuting = list(current.refuting_evidence_ids)
        updated_by = current.updated_by_task_id
        if result.status == "satisfied":
            supporting.extend(result.evidence_ids)
            updated_by = str(task_id or "") or updated_by
        elif result.status == "contradicted":
            refuting.extend(result.evidence_ids)
            updated_by = str(task_id or "") or updated_by
        supporting_ids = _unique_strings(supporting)
        refuting_ids = _unique_strings(refuting)
        status = "conflicted" if supporting_ids and refuting_ids else "satisfied" if supporting_ids else "refuted" if refuting_ids else "unknown"
        states[result.condition_id] = ConditionState(
            result.condition_id,
            status,
            supporting_ids,
            refuting_ids,
            updated_by,
            "full_video" if "full_video" in {current.scope, result.scope} else result.scope or current.scope,
            result.quantifier if result.quantifier != "exists" else current.quantifier,
            max(current.required_coverage, result.required_coverage),
        )
    return states


def _unique_strings(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


@dataclass(frozen=True)
class TargetPresenceFact:
    target: str
    status: str = "uncertain"
    confidence: float = 0.0
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", str(self.target or "").strip())
        status = str(self.status or "uncertain").strip().casefold()
        object.__setattr__(self, "status", status if status in {"present", "absent", "uncertain"} else "uncertain")
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence or 0.0))))
        object.__setattr__(
            self,
            "evidence_ids",
            tuple(dict.fromkeys(str(item).strip() for item in self.evidence_ids if str(item).strip())),
        )


@dataclass(frozen=True)
class MeasurementFact:
    value: float
    unit: str
    relation: str = "exact"
    semantics: str = "unknown"
    subject_id: str = ""
    source_time_sec: float | None = None
    boundary_relation: str = "unknown"
    raw_text: str = ""
    evidence_ids: tuple[str, ...] = ()
    quantity_type: str = ""
    predicate: str = ""
    object_id: str = ""
    event_id: str = ""
    extraction_source: str = "vlm_structured"
    binding_status: str = "unbound"

    def __post_init__(self) -> None:
        value, unit = _normalized_measurement_value_unit(self.value, self.unit)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "unit", unit)
        relation = str(self.relation or "exact").strip().casefold()
        object.__setattr__(self, "relation", relation if relation in {"exact", "approx", "greater_than", "less_than"} else "exact")
        semantics = str(self.semantics or "unknown").strip().casefold()
        object.__setattr__(self, "semantics", semantics if semantics in {"delta", "cumulative", "unknown"} else "unknown")
        object.__setattr__(self, "subject_id", str(self.subject_id or "").strip())
        object.__setattr__(self, "source_time_sec", None if self.source_time_sec is None else float(self.source_time_sec))
        boundary = str(self.boundary_relation or "unknown").strip().casefold()
        object.__setattr__(self, "boundary_relation", boundary if boundary in {"before", "at", "after", "unknown"} else "unknown")
        object.__setattr__(self, "raw_text", str(self.raw_text or "").strip())
        object.__setattr__(self, "quantity_type", str(self.quantity_type or "").strip().casefold())
        object.__setattr__(self, "predicate", str(self.predicate or "").strip().casefold())
        object.__setattr__(self, "object_id", str(self.object_id or "").strip())
        object.__setattr__(self, "event_id", str(self.event_id or "").strip())
        object.__setattr__(self, "extraction_source", str(self.extraction_source or "vlm_structured").strip().casefold())
        binding = str(self.binding_status or "unbound").strip().casefold()
        object.__setattr__(self, "binding_status", binding if binding in {"explicit", "contextual", "ambiguous", "unbound"} else "unbound")
        object.__setattr__(
            self,
            "evidence_ids",
            tuple(dict.fromkeys(str(item).strip() for item in self.evidence_ids if str(item).strip())),
        )


@dataclass(frozen=True)
class RelationFact:
    relation_type: str
    subject_id: str
    object_id: str = ""
    status: str = "unknown"
    description: str = ""
    evidence_ids: tuple[str, ...] = ()
    value: str = ""
    reference_frame: str = ""
    same_frame: bool = False
    witness_frame_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation_type", str(self.relation_type or "").strip().casefold())
        object.__setattr__(self, "subject_id", str(self.subject_id or "").strip())
        object.__setattr__(self, "object_id", str(self.object_id or "").strip())
        object.__setattr__(self, "value", _normalized_role(self.value))
        object.__setattr__(self, "reference_frame", _normalized_role(self.reference_frame))
        object.__setattr__(self, "same_frame", bool(self.same_frame))
        object.__setattr__(
            self,
            "witness_frame_indices",
            tuple(dict.fromkeys(int(item) for item in self.witness_frame_indices if isinstance(item, int) and item >= 0)),
        )
        status = str(self.status or "unknown").strip().casefold()
        object.__setattr__(self, "status", status if status in {"supported", "contradicted", "unknown"} else "unknown")
        object.__setattr__(self, "description", str(self.description or "").strip())
        object.__setattr__(
            self,
            "evidence_ids",
            tuple(dict.fromkeys(str(item).strip() for item in self.evidence_ids if str(item).strip())),
        )


def make_gap_conditions(gap_id: str, descriptions: Sequence[str]) -> tuple[GapCondition, ...]:
    prefix = re.sub(r"[^a-z0-9]+", "_", str(gap_id or "gap").casefold()).strip("_") or "gap"
    return tuple(
        GapCondition(f"{prefix}_c{index}", str(description).strip())
        for index, description in enumerate(descriptions, start=1)
        if str(description).strip()
    )


def normalize_target_presence(value: Any, *, evidence_id: str = "") -> TargetPresenceFact:
    payload = value if isinstance(value, Mapping) else {}
    return TargetPresenceFact(
        target=str(payload.get("target", "") or ""),
        status=str(payload.get("status", "uncertain") or "uncertain"),
        confidence=float(payload.get("confidence", 0.0) or 0.0),
        evidence_ids=(evidence_id,)
        if evidence_id and str(payload.get("status", "")).casefold() in {"present", "absent"}
        else (),
    )


def normalize_measurements(value: Any, *, evidence_id: str = "") -> tuple[MeasurementFact, ...]:
    rows = value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()
    result = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        try:
            fact = MeasurementFact(
                value=float(row.get("value")),
                unit=str(row.get("unit", "") or ""),
                relation=str(row.get("relation", "exact") or "exact"),
                semantics=str(row.get("measurement_semantics", row.get("semantics", "unknown")) or "unknown"),
                subject_id=str(row.get("subject_id", "") or ""),
                source_time_sec=row.get("source_time_sec"),
                boundary_relation=str(row.get("boundary_relation", "unknown") or "unknown"),
                raw_text=str(row.get("raw_text", "") or ""),
                evidence_ids=(evidence_id,) if evidence_id else (),
                quantity_type=str(row.get("quantity_type", "") or ""),
                predicate=str(row.get("predicate", "") or ""),
                object_id=str(row.get("object_id", "") or ""),
                event_id=str(row.get("event_id", "") or ""),
                extraction_source=str(row.get("extraction_source", "vlm_structured") or "vlm_structured"),
                binding_status=str(row.get("binding_status", "unbound") or "unbound"),
            )
        except (TypeError, ValueError):
            continue
        if fact.unit:
            result.append(fact)
    return tuple(result)


def normalize_relations(value: Any, *, evidence_id: str = "") -> tuple[RelationFact, ...]:
    rows = value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()
    result = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        fact = RelationFact(
            relation_type=str(row.get("relation_type", "") or ""),
            subject_id=str(row.get("subject_id", "") or ""),
            object_id=str(row.get("object_id", "") or ""),
            value=str(row.get("value", "") or ""),
            reference_frame=str(row.get("reference_frame", "") or ""),
            same_frame=_as_bool(row.get("same_frame")),
            witness_frame_indices=tuple(row.get("witness_frame_indices", ()) or ()),
            status=str(row.get("status", "unknown") or "unknown"),
            description=str(row.get("description", "") or ""),
            evidence_ids=(evidence_id,)
            if evidence_id and str(row.get("status", "")).casefold() in {"supported", "contradicted"}
            else (),
        )
        if fact.relation_type:
            result.append(fact)
    return tuple(result)


def normalize_condition_results(
    value: Any,
    conditions: Sequence[GapCondition],
    *,
    evidence_id: str,
    target_presence: TargetPresenceFact | None = None,
    measurements: Sequence[MeasurementFact] = (),
    relations: Sequence[RelationFact] = (),
) -> tuple[ConditionResult, ...]:
    raw_rows = value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()
    by_id: dict[str, Mapping[str, Any]] = {}
    by_description: dict[str, Mapping[str, Any]] = {}
    for row in raw_rows:
        if not isinstance(row, Mapping):
            continue
        condition_id = str(row.get("condition_id", "") or "").strip()
        description = str(row.get("condition", row.get("description", "")) or "").strip().casefold()
        if condition_id:
            by_id[condition_id] = row
        if description:
            by_description[description] = row

    results = []
    for condition in conditions:
        row = by_id.get(condition.condition_id) or by_description.get(condition.description.casefold()) or {}
        status = str(row.get("status", "unknown") or "unknown").strip().casefold()
        observation = str(row.get("observation", "") or "").strip()
        if status in {"satisfied", "resolved"} and not observation:
            status = "unknown"
        requires_target = condition.condition_type == "presence" or (
            condition.condition_type == "auto" and _requires_visible_target(condition.description)
        )
        requires_measurement = condition.condition_type == "measurement" or (
            condition.condition_type == "auto" and _requires_measurement(condition.description)
        )
        requires_relation = condition.condition_type in {"relation", "temporal"} or (
            condition.condition_type == "auto" and _requires_relation(condition.description)
        )
        if status in {"satisfied", "resolved"} and requires_target:
            if (
                target_presence is None
                or target_presence.status != "present"
                or not _target_matches_condition(target_presence.target, condition.target_role or condition.description)
            ):
                status = "unknown"
        if status in {"satisfied", "resolved"} and requires_measurement:
            matching = tuple(
                fact for fact in measurements
                if (not condition.quantity_type or fact.quantity_type == condition.quantity_type)
                and (not condition.unit or fact.unit == condition.unit)
            )
            if not matching:
                status = "unknown"
        if status in {"satisfied", "resolved"} and requires_relation:
            temporal_match = condition.condition_type in {"auto", "temporal"} and _measurement_carries_temporal_relation(
                condition.description,
                measurements,
            )
            if not temporal_match and not any(_relation_matches_condition(fact, condition) for fact in relations):
                status = "unknown"
        normalized = ConditionResult(
            condition_id=condition.condition_id,
            status=status,
            observation=observation,
            evidence_ids=(evidence_id,)
            if evidence_id and status in {"satisfied", "resolved", "contradicted"}
            else (),
            scope=condition.scope,
            quantifier=condition.quantifier,
            required_coverage=condition.required_coverage,
        )
        results.append(normalized)
    return tuple(results)


def derive_resolution(
    conditions: Sequence[GapCondition],
    results: Sequence[ConditionResult],
) -> str:
    if not conditions:
        return "unresolved"
    by_id = {result.condition_id: result for result in results}
    critical = tuple(condition for condition in conditions if condition.critical)
    critical_results = tuple(by_id.get(condition.condition_id, ConditionResult(condition.condition_id, "unknown")) for condition in critical)
    if critical_results and all(result.status == "satisfied" for result in critical_results):
        return "resolved"
    if any(result.status in {"satisfied", "contradicted"} for result in results):
        return "partial"
    return "unresolved"


def _requires_global_scope(description: str) -> bool:
    text = str(description or "").casefold()
    return bool(
        re.search(r"\b(?:throughout|entire|whole|full)\s+(?:source|video|film|workspace|race)\b", text)
        or re.search(
            r"\b(?:list|enumerate|catalog|count|find|verify|inspect)\b[^.]{0,80}"
            r"\b(?:all|every|each|total)\b",
            text,
        )
        or re.search(r"\b(?:all|every|each)\b[^.]{0,80}\b(?:appearance|event|occurrence|segment|audition|overtake|episode)s?\b", text)
        or _requires_temporal_max(text)
    )


def _requires_event_union(description: str) -> bool:
    text = str(description or "").casefold()
    return bool(
        re.search(r"\b(?:list|enumerate|catalog|count)\b[^.]{0,100}\b(?:event|occurrence|appearance|audition|segment)s?\b", text)
        or re.search(r"\b(?:all|every|each)\b[^.]{0,80}\b(?:event|occurrence|appearance|audition|overtake|episode)s?\b", text)
    )


def _requires_temporal_max(description: str) -> bool:
    text = str(description or "").casefold()
    return bool(
        re.search(r"\b(?:no\s+later|last|latest|final)\b[^.]{0,80}\b(?:event|occurrence|episode|overtake|lead[ -]?loss|instance|timestamp)\b", text)
        or re.search(r"\b(?:event|occurrence|episode|overtake|lead[ -]?loss|instance)\b[^.]{0,80}\b(?:last|latest|final)\b", text)
    )


def _requires_ordinal_participant(description: str) -> bool:
    text = str(description or "").casefold()
    return bool(
        re.search(r"\b(?:second|2nd)\b[^.]{0,60}\b(?:person|participant|overtaker|racer|rider)\b", text)
        or re.search(r"\b(?:person|participant|overtaker|racer|rider)\b[^.]{0,60}\b(?:second|2nd)\b", text)
    )


def canonical_unit(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold()).strip("_")
    aliases = {
        "calories": "calorie",
        "cal": "calorie",
        "cals": "calorie",
        "lightyears": "light_year",
        "lightyear": "light_year",
        "light_years": "light_year",
        "ly": "light_year",
        "dollars": "dollar",
        "usd": "dollar",
        "minutes": "minute",
        "meters": "meter",
        "metres": "meter",
        "kilometers": "kilometer",
        "kilometres": "kilometer",
        "kilograms": "kilogram",
        "years": "year",
        "points": "point",
    }
    return aliases.get(text, text[:-1] if text.endswith("s") and len(text) > 3 else text)


def _normalized_measurement_value_unit(value: float, unit: str) -> tuple[float, str]:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(unit or "").casefold()).strip("_")
    scales = {
        "thousand": 1e3,
        "million": 1e6,
        "billion": 1e9,
        "trillion": 1e12,
        "quadrillion": 1e15,
        "quintillion": 1e18,
    }
    for prefix, factor in scales.items():
        marker = f"{prefix}_"
        if normalized.startswith(marker):
            return float(value) * factor, canonical_unit(normalized[len(marker) :])
    return float(value), canonical_unit(normalized)


def extract_measurements_from_text(
    text: str,
    *,
    quantity_type: str = "",
    binding_status: str = "unbound",
    evidence_id: str = "",
) -> tuple[MeasurementFact, ...]:
    source = str(text or "")
    matches: list[tuple[int, MeasurementFact]] = []
    pattern = re.compile(
        r"(?P<relation>over|more\s+than|greater\s+than|under|less\s+than|about|approximately|around)?\s*"
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<scale>thousand|million|billion|trillion|quadrillion|quintillion)?\s*"
        r"(?P<unit>light[\s-]?years?|calories?|points?|meters?|metres?|kilometers?|kilometres?|minutes?|seconds?|dollars?|usd)\b",
        re.IGNORECASE,
    )
    for match in pattern.finditer(source):
        relation_text = " ".join(str(match.group("relation") or "").casefold().split())
        relation = "greater_than" if relation_text in {"over", "more than", "greater than"} else "less_than" if relation_text in {"under", "less than"} else "approx" if relation_text else "exact"
        unit = " ".join(part for part in (str(match.group("scale") or ""), str(match.group("unit") or "")) if part)
        matches.append((match.start(), MeasurementFact(
            float(match.group("value")), unit, relation=relation, raw_text=match.group(0).strip(),
            evidence_ids=(evidence_id,) if evidence_id else (), quantity_type=quantity_type,
            extraction_source="text_fallback", binding_status=binding_status,
        )))
    for match in re.finditer(r"\b(?P<minutes>\d{1,2}):(?P<seconds>[0-5]\d)\b", source):
        matches.append((match.start(), MeasurementFact(
            int(match.group("minutes")) * 60 + int(match.group("seconds")), "second", raw_text=match.group(0),
            evidence_ids=(evidence_id,) if evidence_id else (), quantity_type="countdown_clock",
            extraction_source="text_fallback", binding_status="explicit",
        )))
    return tuple(fact for _, fact in sorted(matches, key=lambda item: item[0]))


def _relation_matches_condition(fact: RelationFact, condition: GapCondition) -> bool:
    if fact.status != "supported":
        return False
    required_type = condition.relation_type or condition.required_relation
    if required_type and fact.relation_type != required_type:
        return False
    if condition.subject_role and _normalized_role(fact.subject_id) != _normalized_role(condition.subject_role):
        return False
    if condition.object_role and _normalized_role(fact.object_id) != _normalized_role(condition.object_role):
        return False
    return True


def _normalized_role(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold()).strip("_")


def _requires_visible_target(description: str) -> bool:
    text = str(description or "").casefold()
    return any(term in text for term in ("read", "visible", "display", "scoreboard", "clock", "written", "number", "text"))


def _target_matches_condition(target: str, description: str) -> bool:
    stop = {
        "read",
        "visible",
        "display",
        "displayed",
        "written",
        "observe",
        "final",
        "value",
        "number",
        "text",
        "unit",
    }

    def tokens(value: str) -> tuple[str, ...]:
        return tuple(
            token[:-1] if token.endswith("s") and len(token) > 4 else token
            for token in re.findall(r"[a-z0-9]+", str(value or "").casefold())
            if len(token) >= 4 and token not in stop
        )

    target_tokens = tokens(target)
    condition_tokens = tokens(description)
    if not target_tokens:
        return False
    if not condition_tokens:
        return True
    return any(
        first == second or first in second or second in first
        for first in target_tokens
        for second in condition_tokens
    )


def _requires_measurement(description: str) -> bool:
    text = str(description or "").casefold()
    return any(term in text for term in ("value", "unit", "score", "clock", "calorie", "amount", "measurement", "number"))


def _requires_relation(description: str) -> bool:
    text = str(description or "").casefold()
    return any(
        term in text
        for term in (
            "same person",
            "same entity",
            "identity",
            "before",
            "after",
            "transition",
            "overtake",
            "causal",
            "cause",
            "left",
            "right",
            "front",
            "behind",
            "facing",
            "direction",
            "relative to",
        )
    )


def _measurement_carries_temporal_relation(
    description: str,
    measurements: Sequence[MeasurementFact],
) -> bool:
    text = str(description or "").casefold()
    expected = set()
    if "before" in text:
        expected.add("before")
    if "after" in text:
        expected.add("after")
    if any(
        term in text
        for term in (
            " at ",
            "when",
            "boundary",
            "halftime",
            "half-time",
            "first half",
            "quarter end",
            "intermission",
        )
    ):
        expected.add("at")
    return bool(expected) and any(fact.boundary_relation in expected for fact in measurements)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes"}
