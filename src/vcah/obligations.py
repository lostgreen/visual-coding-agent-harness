from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


_SCOPE_RANK = {"window": 0, "multi_window": 1, "full_video": 2}


def _scope(value: Any) -> str:
    normalized = str(value or "window").strip().casefold()
    return normalized if normalized in _SCOPE_RANK else "window"


def _contract_value(contract: Any, key: str, default: Any = "") -> Any:
    if isinstance(contract, Mapping):
        return contract.get(key, default)
    return getattr(contract, key, default)


@dataclass(frozen=True)
class QueryObligation:
    requirement_id: str
    kind: str
    scope: str
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "kind": self.kind,
            "scope": self.scope,
            "required": self.required,
        }


@dataclass(frozen=True)
class QueryObligations:
    contract_scope: str
    effective_scope: str
    obligations: tuple[QueryObligation, ...]
    scope_escalation_requirement_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_scope": self.contract_scope,
            "effective_scope": self.effective_scope,
            "scope_escalation_requirement_ids": list(self.scope_escalation_requirement_ids),
            "obligations": [obligation.to_dict() for obligation in self.obligations],
        }


def compile_query_obligations(
    query_contract: Any,
    query_requirements: Mapping[str, Any] | None = None,
) -> QueryObligations:
    requirements = dict(query_requirements or {})
    contract_scope = _scope(_contract_value(query_contract, "required_scope"))
    quantifier = str(_contract_value(query_contract, "quantifier", "") or "").casefold()
    target = str(_contract_value(query_contract, "observation_target", "") or "").casefold()
    aggregation = str(_contract_value(query_contract, "aggregation", "") or "").casefold()
    obligations: list[QueryObligation] = [
        QueryObligation("req_contract_scope", "contract_scope", contract_scope),
    ]
    if contract_scope == "full_video":
        obligations.append(QueryObligation("req_full_video_coverage", "range_coverage", "full_video"))
    if quantifier == "total_count" and target == "event":
        obligations.extend((
            QueryObligation("req_full_video_coverage", "range_coverage", "full_video"),
            QueryObligation("req_all_event_candidates_enumerated", "event_enumeration", "full_video"),
        ))
    if quantifier == "universal":
        obligations.extend((
            QueryObligation("req_full_video_coverage", "range_coverage", "full_video"),
            QueryObligation("req_qualified_absence_enumerated", "absence_enumeration", "full_video"),
        ))
    if bool(requirements.get("requires_temporal_extremum")):
        obligations.extend((
            QueryObligation("req_temporal_extremum_coverage", "range_coverage", "full_video"),
            QueryObligation("req_temporal_max_episode", "temporal_extremum", "full_video"),
        ))
    if bool(requirements.get("requires_temporal_sequence")) and aggregation == "order":
        obligations.extend((
            QueryObligation("req_sequence_coverage", "range_coverage", "full_video"),
            QueryObligation("req_sequence_enumeration", "event_enumeration", "full_video"),
        ))
    if bool(requirements.get("requires_event_participant_link")):
        obligations.append(QueryObligation("req_cross_window_entity_binding", "entity_binding", "multi_window"))
    if bool(requirements.get("requires_same_object_transition")):
        obligations.append(QueryObligation("req_state_transition", "state_transition", "multi_window"))
    if bool(requirements.get("requires_narrative_inference")):
        obligations.append(QueryObligation("req_narrative_trajectory", "narrative_trajectory", "multi_window"))
    # Preserve first occurrence while avoiding a second parallel obligation with the same stable ID.
    unique: list[QueryObligation] = []
    seen: set[str] = set()
    for raw_obligation in obligations:
        obligation = QueryObligation(
            raw_obligation.requirement_id,
            raw_obligation.kind,
            _scope(raw_obligation.scope),
            raw_obligation.required,
        )
        if obligation.requirement_id in seen:
            continue
        seen.add(obligation.requirement_id)
        unique.append(obligation)
    effective_scope = max(
        (item.scope for item in unique if item.required),
        key=lambda item: _SCOPE_RANK[item],
        default=contract_scope,
    )
    escalations = tuple(
        item.requirement_id
        for item in unique
        if item.required and _SCOPE_RANK[item.scope] > _SCOPE_RANK[contract_scope]
    )
    return QueryObligations(contract_scope, effective_scope, tuple(unique), escalations)


def evaluate_query_obligations(
    obligations: QueryObligations,
    snapshot: Mapping[str, Any],
    completion_status: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    completion = dict(completion_status or {})
    snapshot_rows = dict(snapshot or {})
    coverage_complete = bool(
        completion.get("range_coverage_complete", _coverage_ratio(completion) >= 1.0)
    )
    enumeration_complete = bool(completion.get("enumeration_complete", False))
    qualified = tuple(snapshot_rows.get("qualified_events", ()) or ())
    incomplete = tuple(snapshot_rows.get("incomplete_events", ()) or ())
    conflicted = tuple(snapshot_rows.get("conflicted_events", ()) or ())
    evaluations = []
    for obligation in obligations.obligations:
        if obligation.kind == "range_coverage":
            status = "supported" if coverage_complete else "unknown"
        elif obligation.kind in {"event_enumeration", "absence_enumeration"}:
            status = "supported" if enumeration_complete else "unknown"
        elif obligation.kind == "temporal_extremum":
            status = "supported" if coverage_complete and qualified and not incomplete and not conflicted else "unknown"
        elif obligation.kind == "entity_binding":
            status = "supported" if bool(completion.get("event_participant_link_ready")) else "unknown"
        elif obligation.kind == "state_transition":
            transitions = tuple(snapshot_rows.get("state_transitions", ()) or ())
            status = "supported" if any(
                str(dict(row).get("status", "") or "") == "supported"
                and bool(dict(row).get("same_object_relation", False))
                for row in transitions
                if isinstance(row, Mapping)
            ) else "unknown"
        elif obligation.kind == "narrative_trajectory":
            status = "supported" if bool(completion.get("narrative_inference_ready")) else "unknown"
        else:
            status = "supported"
        evaluations.append({
            "requirement_id": obligation.requirement_id,
            "kind": obligation.kind,
            "scope": obligation.scope,
            "status": status,
        })
    return tuple(evaluations)


def _coverage_ratio(completion_status: Mapping[str, Any]) -> float:
    coverage = dict(completion_status.get("source_coverage", {}) or {})
    adopted = str(completion_status.get("adopted_source_video_id", "") or "")
    row = dict(coverage.get(adopted, {}) or {}) if adopted else {}
    if not row and coverage:
        row = max(
            (dict(value) for value in coverage.values() if isinstance(value, Mapping)),
            key=lambda value: float(value.get("coverage_ratio", 0.0) or 0.0),
            default={},
        )
    return float(row.get("coverage_ratio", 0.0) or 0.0)
