from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class GapCondition:
    condition_id: str
    description: str
    critical: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition_id", str(self.condition_id or "").strip())
        object.__setattr__(self, "description", str(self.description or "").strip())
        object.__setattr__(self, "critical", bool(self.critical))


@dataclass(frozen=True)
class ConditionResult:
    condition_id: str
    status: str
    observation: str = ""
    evidence_ids: tuple[str, ...] = ()

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

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation_type", str(self.relation_type or "").strip().casefold())
        object.__setattr__(self, "subject_id", str(self.subject_id or "").strip())
        object.__setattr__(self, "object_id", str(self.object_id or "").strip())
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
        if status in {"satisfied", "resolved"} and _requires_visible_target(condition.description):
            if (
                target_presence is None
                or target_presence.status != "present"
                or not _target_matches_condition(target_presence.target, condition.description)
            ):
                status = "unknown"
        if status in {"satisfied", "resolved"} and _requires_measurement(condition.description) and not measurements:
            status = "unknown"
        if status in {"satisfied", "resolved"} and _requires_relation(condition.description):
            if not _measurement_carries_temporal_relation(condition.description, measurements) and not any(
                fact.status == "supported" for fact in relations
            ):
                status = "unknown"
        normalized = ConditionResult(
            condition_id=condition.condition_id,
            status=status,
            observation=observation,
            evidence_ids=(evidence_id,)
            if evidence_id and status in {"satisfied", "resolved", "contradicted"}
            else (),
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
    }
    for prefix, factor in scales.items():
        marker = f"{prefix}_"
        if normalized.startswith(marker):
            return float(value) * factor, canonical_unit(normalized[len(marker) :])
    return float(value), canonical_unit(normalized)


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
    return any(term in text for term in ("same person", "same entity", "identity", "before", "after", "transition", "overtake", "causal", "cause"))


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
    if any(term in text for term in (" at ", "when", "boundary")):
        expected.add("at")
    return bool(expected) and any(fact.boundary_relation in expected for fact in measurements)
