from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping, Sequence

from vcah.evidence_primitives import normalize_measurements, normalize_relations
from vcah.provenance import (
    deterministic_derivation_provenance,
    direct_witness_provenance,
    heuristic_provenance,
    normalize_provenance,
    provenance_kinds,
)
from vcah.qualification import qualify_event_candidates
from vcah.types import ClaimContract, EvidenceRecord
from vcah.virtual_video import VirtualVideoWorkspace


@dataclass(frozen=True)
class SemanticRepairRequest:
    query_id: str
    goal: str
    segment_id: str
    time_range: tuple[float, float] | None
    modality_hint: tuple[str, ...]
    expected_evidence: str
    inspection_mode: str = "window"


@dataclass(frozen=True)
class AbsenceQualification:
    status: str
    coverage_ratio: float
    sampling_interval_sec: float | None
    expected_dwell_time_sec: float | None
    visibility_status: str
    targeted_inspection_count: int
    reason: str


@dataclass(frozen=True)
class CanonicalFactSnapshot:
    qualified_events: tuple[Mapping[str, Any], ...] = ()
    observed_event_candidates: tuple[Mapping[str, Any], ...] = ()
    unqualified_events: tuple[Mapping[str, Any], ...] = ()
    incomplete_events: tuple[Mapping[str, Any], ...] = ()
    conflicted_events: tuple[Mapping[str, Any], ...] = ()
    duplicate_suspect_events: tuple[Mapping[str, Any], ...] = ()
    refuted_events: tuple[Mapping[str, Any], ...] = ()
    resolved_entities: tuple[Mapping[str, Any], ...] = ()
    unresolved_entity_bindings: tuple[Mapping[str, Any], ...] = ()
    state_transitions: tuple[Mapping[str, Any], ...] = ()
    unresolved_state_transitions: tuple[Mapping[str, Any], ...] = ()
    entity_associations: tuple[Mapping[str, Any], ...] = ()
    attribute_facts: tuple[Mapping[str, Any], ...] = ()
    inferred_facts: tuple[Mapping[str, Any], ...] = ()
    unresolved_inferences: tuple[Mapping[str, Any], ...] = ()
    ordered_events: tuple[Mapping[str, Any], ...] = ()
    raw_candidate_counts: Mapping[str, int] = field(default_factory=dict)
    requirement_graph: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_candidate_counts", dict(self.raw_candidate_counts or {}))
        object.__setattr__(self, "requirement_graph", dict(self.requirement_graph or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "qualified_events": [dict(row) for row in self.qualified_events],
            "confirmed_events": [dict(row) for row in self.qualified_events],
            "observed_event_candidates": [dict(row) for row in self.observed_event_candidates],
            "unqualified_events": [dict(row) for row in self.unqualified_events],
            "incomplete_events": [dict(row) for row in self.incomplete_events],
            "conflicted_events": [dict(row) for row in self.conflicted_events],
            "duplicate_suspect_events": [dict(row) for row in self.duplicate_suspect_events],
            "refuted_events": [dict(row) for row in self.refuted_events],
            "resolved_entities": [dict(row) for row in self.resolved_entities],
            "unresolved_entity_bindings": [dict(row) for row in self.unresolved_entity_bindings],
            "state_transitions": [dict(row) for row in self.state_transitions],
            "unresolved_state_transitions": [dict(row) for row in self.unresolved_state_transitions],
            "entity_associations": [dict(row) for row in self.entity_associations],
            "attribute_facts": [dict(row) for row in self.attribute_facts],
            "inferred_facts": [dict(row) for row in self.inferred_facts],
            "unresolved_inferences": [dict(row) for row in self.unresolved_inferences],
            "ordered_events": [dict(row) for row in self.ordered_events],
            "requirement_graph": dict(self.requirement_graph),
            "canonical_fact_counts": {
                "events": len(self.qualified_events),
                "entities": len(self.resolved_entities),
                "state_transitions": len(self.state_transitions),
                "entity_associations": len(self.entity_associations),
                "attribute_facts": len(self.attribute_facts),
                "inferred_facts": len(self.inferred_facts),
            },
            "raw_candidate_counts": dict(self.raw_candidate_counts),
        }


def qualify_absence(
    interval: tuple[float, float],
    inspected_ranges: Sequence[tuple[float, float]],
    sampling_interval: float | None,
    expected_dwell_time: float | None,
    visibility_status: str,
    *,
    targeted_inspection_count: int = 1,
    coverage_threshold: float = 0.9,
) -> AbsenceQualification:
    start, end = sorted((float(interval[0]), float(interval[1])))
    duration = max(0.0, end - start)
    clipped = sorted(
        (max(start, min(float(left), float(right))), min(end, max(float(left), float(right))))
        for left, right in inspected_ranges
        if min(end, max(float(left), float(right))) > max(start, min(float(left), float(right)))
    )
    covered = 0.0
    cursor = start
    for left, right in clipped:
        if right <= cursor:
            continue
        covered += max(0.0, right - max(cursor, left))
        cursor = max(cursor, right)
    coverage_ratio = min(1.0, covered / duration) if duration > 0 else 0.0
    visibility = str(visibility_status or "unknown").strip().casefold()
    sample_interval = float(sampling_interval) if sampling_interval and sampling_interval > 0 else None
    dwell = float(expected_dwell_time) if expected_dwell_time and expected_dwell_time > 0 else None
    common = {
        "coverage_ratio": coverage_ratio,
        "sampling_interval_sec": sample_interval,
        "expected_dwell_time_sec": dwell,
        "visibility_status": visibility,
        "targeted_inspection_count": max(0, int(targeted_inspection_count)),
    }
    if visibility not in {"clear", "visible", "unoccluded"}:
        return AbsenceQualification(status="unknown_due_to_visibility", reason="visibility_not_established", **common)
    if coverage_ratio + 1e-9 < float(coverage_threshold):
        return AbsenceQualification(status="unknown_due_to_coverage", reason="target_interval_coverage_insufficient", **common)
    if dwell is None:
        return AbsenceQualification(status="unknown_due_to_coverage", reason="expected_event_dwell_time_unknown", **common)
    if sample_interval is None or sample_interval > dwell / 2.0:
        return AbsenceQualification(status="unknown_due_to_coverage", reason="sampling_interval_exceeds_half_dwell_time", **common)
    if int(targeted_inspection_count) < 1:
        return AbsenceQualification(status="not_observed", reason="no_targeted_inspection", **common)
    return AbsenceQualification(status="qualified_absence", reason="coverage_sampling_and_visibility_sufficient", **common)


def semantic_repair_requests(
    workspace: VirtualVideoWorkspace,
    contract: ClaimContract,
    query_requirements: Mapping[str, Any],
    completion_status: Mapping[str, Any],
    evidence: Sequence[EvidenceRecord],
    *,
    round_id: int,
    limit: int,
) -> tuple[str, tuple[SemanticRepairRequest, ...]]:
    task_limit = max(0, int(limit))
    if task_limit <= 0 or bool(completion_status.get("ready_for_answer")):
        return "", ()
    if contract.quantifier == "total_count" and contract.observation_target == "event":
        requests = _event_repairs(
            workspace,
            evidence,
            tuple(completion_status.get("unresolved_event_windows", ()) or ()),
            round_id,
            task_limit,
        )
        if requests:
            return "event_candidate_unresolved", requests
    if contract.measurement_unit == "point" and contract.boundary_hint:
        requests = _boundary_score_repairs(workspace, evidence, round_id, task_limit)
        if requests:
            reason = (
                "boundary_score_context_missing"
                if requests[0].query_id.startswith("semantic_score_context_")
                else "boundary_score_unvisited_segments"
            )
            return reason, requests
    if contract.observation_target == "relation" and query_requirements.get("requires_spatial_relation"):
        requests = _spatial_repairs(
            workspace,
            evidence,
            str(query_requirements.get("spatial_relation_type", "") or "relative_bearing"),
            str(query_requirements.get("spatial_reference_frame", "") or ""),
            round_id,
            task_limit,
        )
        if requests:
            return "spatial_reference_frame_missing", requests
    return "", ()


def _event_repairs(
    workspace: VirtualVideoWorkspace,
    evidence: Sequence[EvidenceRecord],
    windows: Sequence[Mapping[str, Any]],
    round_id: int,
    limit: int,
) -> tuple[SemanticRepairRequest, ...]:
    if any(str(record.task_id or "").startswith("semantic_event_") for record in evidence):
        return ()
    requests = []
    for row in windows:
        bounds = tuple(row.get("virtual_time_range", ()) or ())
        if len(bounds) != 2:
            continue
        start, end = float(bounds[0]), float(bounds[1])
        segment = _segment_at(workspace, (start + end) / 2.0)
        if segment is None:
            continue
        requests.append(
            SemanticRepairRequest(
                query_id=f"semantic_event_r{round_id}_{len(requests) + 1:03d}",
                goal="Resolve this ambiguous event window into stable, question-relevant atomic occurrences.",
                segment_id=segment.segment_id,
                time_range=(start, end),
                modality_hint=("visual", "asr"),
                expected_evidence=(
                    "one row per occurrence with a distinctive event_key, precise onset/end, and continuation flags; "
                    "generic event-class keys remain unresolved"
                ),
                inspection_mode="enumerate_events",
            )
        )
        if len(requests) >= limit:
            break
    return tuple(requests)


def _boundary_score_repairs(
    workspace: VirtualVideoWorkspace,
    evidence: Sequence[EvidenceRecord],
    round_id: int,
    limit: int,
) -> tuple[SemanticRepairRequest, ...]:
    visited = {
        str(lineage.get("segment_id", "") or "")
        for record in evidence
        for lineage in record.source_lineage
        if str(lineage.get("segment_id", "") or "")
    }
    option_text = ", ".join(
        f"{label}={text}" for label, text in workspace.case.options.items() if _score_pair(str(text))
    )
    if not any(str(record.task_id or "").startswith("semantic_score_context_") for record in evidence):
        context_requests = []
        context_windows = []
        for record in reversed(tuple(evidence)):
            facts = normalize_measurements(
                record.operation_metadata.get("measurements"),
                evidence_id=record.evidence_id,
            )
            if not any(fact.unit == "point" and fact.quantity_type == "score" for fact in facts):
                continue
            if record.start_sec is None or record.end_sec is None:
                continue
            segment = _segment_at(workspace, (float(record.start_sec) + float(record.end_sec)) / 2.0)
            if segment is None:
                continue
            start = max(segment.virtual_start_sec, float(record.start_sec) - 180.0)
            end = min(segment.virtual_end_sec, float(record.end_sec) + 30.0)
            if end - start < 2.0 or any(min(end, right) > max(start, left) for left, right in context_windows):
                continue
            context_windows.append((start, end))
            context_requests.append(
                SemanticRepairRequest(
                    query_id=f"semantic_score_context_r{round_id}_{len(context_requests) + 1:03d}",
                    goal="Scan around this readable non-boundary scoreboard, especially backward, for the requested phase boundary.",
                    segment_id=segment.segment_id,
                    time_range=(start, end),
                    modality_hint=("visual", "ocr"),
                    expected_evidence=(
                        f"same-frame boundary label or clock plus one score option from {option_text}; "
                        "use the readable checkpoint only to navigate, not as the boundary answer"
                    ),
                )
            )
            if len(context_requests) >= limit:
                break
        if context_requests:
            return tuple(context_requests)
    candidates = tuple(segment for segment in workspace.manifest.segments if segment.segment_id not in visited)
    return tuple(
        SemanticRepairRequest(
            query_id=f"semantic_score_r{round_id}_{index:03d}",
            goal="Inspect this unvisited segment for the requested boundary scoreboard; compare every score option.",
            segment_id=segment.segment_id,
            time_range=None,
            modality_hint=("visual", "ocr"),
            expected_evidence=(
                f"same-frame team scores plus period/phase and clock binding for {option_text}; "
                "do not adopt a score without the requested boundary"
            ),
        )
        for index, segment in enumerate(candidates[:limit], start=1)
    )


def _spatial_repairs(
    workspace: VirtualVideoWorkspace,
    evidence: Sequence[EvidenceRecord],
    relation_type: str,
    reference_frame: str,
    round_id: int,
    limit: int,
) -> tuple[SemanticRepairRequest, ...]:
    if any(str(record.task_id or "").startswith("semantic_spatial_") for record in evidence):
        return ()
    candidates = []
    seen = set()
    for record in evidence:
        facts = normalize_relations(record.operation_metadata.get("relations"), evidence_id=record.evidence_id)
        if not any(fact.status == "supported" and fact.same_frame for fact in facts):
            continue
        if record.start_sec is None or record.end_sec is None:
            continue
        segment = _segment_at(workspace, (float(record.start_sec) + float(record.end_sec)) / 2.0)
        if segment is None:
            continue
        start = max(segment.virtual_start_sec, float(record.start_sec) - 4.0)
        end = min(segment.virtual_end_sec, float(record.end_sec) + 4.0)
        key = (segment.segment_id, round(start, 1), round(end, 1))
        if key in seen:
            continue
        seen.add(key)
        candidates.append((segment.segment_id, start, end))
    return tuple(
        SemanticRepairRequest(
            query_id=f"semantic_spatial_r{round_id}_{index:03d}",
            goal="Re-observe both named subjects together and resolve their exact relative bearing.",
            segment_id=segment_id,
            time_range=(start, end),
            modality_hint=("visual",),
            expected_evidence=(
                f"supported {relation_type} with both subject/object IDs, same_frame=true, and "
                f"reference_frame={reference_frame or 'explicit'}"
            ),
        )
        for index, (segment_id, start, end) in enumerate(candidates[:limit], start=1)
    )


_GENERIC_EVENT_TOKENS = {
    "a", "an", "and", "america", "american", "appears", "audition", "auditions", "by", "card",
    "dance", "dancing", "event", "events", "got", "graphic", "group", "groups", "introduction",
    "introduced", "occurrence", "occurrences", "on", "one", "performance", "recap", "receives",
    "show", "stage", "start", "talent", "the", "title", "to", "world", "golden", "buzzer", "acrobatic",
    "crew", "gymnastics", "celebrate", "celebrates", "celebration", "conclude", "concludes",
    "feedback", "judge", "judges", "judging", "ovation", "reaction", "reactions", "vote", "votes",
}


def event_candidate_ledger(evidence: Sequence[EvidenceRecord]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for record in evidence:
        for event in record.operation_metadata.get("events", ()) or ():
            if not isinstance(event, Mapping):
                continue
            try:
                start, end = float(event.get("start_sec")), float(event.get("end_sec"))
            except (TypeError, ValueError):
                continue
            if end < start:
                start, end = end, start
            source_id = _source_id(record, start, end)
            signature = _event_signature(event)
            if not signature:
                unresolved.append(
                    {
                        "source_video_id": source_id,
                        "virtual_time_range": [start, end],
                        "evidence_ids": [record.evidence_id],
                        "descriptions": [str(event.get("description", "") or "")],
                    }
                )
                continue
            event_key = _normalize_key(event.get("event_key"))
            event_class = _normalize_key(event.get("event_class"))
            counting_unit = _normalize_key(event.get("counting_unit"))
            participant_ids = tuple(
                dict.fromkeys(
                    _normalize_participant_id(value)
                    for value in tuple(event.get("participant_ids", ()) or ())
                    if _normalize_participant_id(value)
                )
            )
            participant_roles = tuple(
                dict(item)
                for item in tuple(event.get("participants", ()) or ())
                if isinstance(item, Mapping)
            )
            phase = _normalize_key(event.get("phase")) or "unknown"
            from_previous = _flag(event.get("continues_from_previous"))
            to_next = _flag(event.get("continues_to_next"))
            row = next(
                (
                    item for item in candidates
                    if item["source_video_id"] == source_id
                    and (
                        item["signature"] == signature
                        or _same_focal_transition_episode(
                            start,
                            end,
                            event_class=event_class,
                            counting_unit=counting_unit,
                            participant_ids=participant_ids,
                            transition=_normalize_key(event.get("transition")),
                            existing=item,
                        )
                    )
                    and _same_counted_occurrence(
                        start,
                        end,
                        event_key=event_key,
                        event_class=event_class,
                        counting_unit=counting_unit,
                        participant_ids=participant_ids,
                        from_previous=from_previous,
                        to_next=to_next,
                        existing=item,
                    )
                ),
                None,
            )
            if row is None:
                row = {
                    "signature": signature,
                    "source_video_id": source_id,
                    "virtual_time_range": [start, end],
                    "evidence_ids": [],
                    "event_keys": [],
                    "descriptions": [],
                    "canonical_event_key": event_key,
                    "event_class": event_class,
                    "counting_unit": counting_unit,
                    "participant_ids": list(participant_ids),
                    "participants": [],
                    "states_before": [],
                    "transitions": [],
                    "states_after": [],
                    "merge_history": [],
                    "phases": [],
                    "continues_from_previous": from_previous,
                    "continues_to_next": to_next,
                    "candidate_statuses": [],
                    "candidate_conflict_reasons": [],
                    "participant_binding_conflicts": [],
                    "qualification_observations": {},
                    "preconditions_met": [],
                    "supports_question_event": [],
                }
                candidates.append(row)
            existing_actors = set(row["participant_ids"]).difference({
                "camera_holder", "video recorder", "recorder",
            })
            incoming_actors = set(participant_ids).difference({
                "camera_holder", "video recorder", "recorder",
            })
            if (
                existing_actors
                and incoming_actors
                and not existing_actors.intersection(incoming_actors)
                and _is_focal_position_transition(event)
            ):
                row["participant_binding_conflicts"].append({
                    "existing_actor_ids": sorted(existing_actors),
                    "observed_actor_ids": sorted(incoming_actors),
                    "reason": "same_occurrence_has_ambiguous_actor_attributes",
                })
            row["virtual_time_range"] = [min(row["virtual_time_range"][0], start), max(row["virtual_time_range"][1], end)]
            row["evidence_ids"].append(record.evidence_id)
            row["event_keys"].append(str(event.get("event_key", "") or ""))
            row["merge_history"].append(
                {
                    "evidence_id": record.evidence_id,
                    "local_id": str(event.get("local_id", "") or ""),
                    "event_key": str(event.get("event_key", "") or ""),
                    "virtual_time_range": [start, end],
                }
            )
            row["descriptions"].append(str(event.get("description", "") or ""))
            row["states_before"].append(str(event.get("state_before", "") or "").strip())
            row["transitions"].append(_normalize_key(event.get("transition")))
            row["states_after"].append(str(event.get("state_after", "") or "").strip())
            row["participant_ids"] = list(dict.fromkeys((*row["participant_ids"], *participant_ids)))[:12]
            known_participants = {
                (
                    str(item.get("entity_hypothesis_id", "") or ""),
                    str(item.get("participant_id", "") or ""),
                    str(item.get("role", "") or ""),
                )
                for item in row["participants"]
            }
            for participant in participant_roles:
                key = (
                    str(participant.get("entity_hypothesis_id", "") or ""),
                    str(participant.get("participant_id", "") or ""),
                    str(participant.get("role", "") or ""),
                )
                if key not in known_participants:
                    row["participants"].append(participant)
                    known_participants.add(key)
            row["phases"].append(phase)
            row["continues_from_previous"] = bool(row["continues_from_previous"] or from_previous)
            row["continues_to_next"] = bool(row["continues_to_next"] or to_next)
            explicit_status = str(event.get("qualification_status", "") or "").strip().casefold()
            row["candidate_statuses"].append(
                "conflicted_candidate" if explicit_status == "conflicted" else "observed_candidate"
            )
            if "preconditions_met" in event:
                row["preconditions_met"].append(event.get("preconditions_met"))
            row["supports_question_event"].append(event.get("supports_question_event"))
            if isinstance(event.get("qualification"), Mapping):
                row["qualification_observations"] = {
                    **dict(row.get("qualification_observations", {}) or {}),
                    **dict(event.get("qualification", {}) or {}),
                }
    public = []
    for index, row in enumerate(sorted(candidates, key=lambda item: item["virtual_time_range"][0]), start=1):
        candidate_status = (
            "conflicted_candidate"
            if "conflicted_candidate" in set(row["candidate_statuses"])
            else "observed_candidate"
        )
        public.append(
            {
                "candidate_id": f"event_candidate_{index:03d}",
                "signature": row["signature"],
                "source_video_id": row["source_video_id"],
                "virtual_time_range": row["virtual_time_range"],
                "evidence_ids": list(dict.fromkeys(row["evidence_ids"]))[:12],
                "event_keys": list(dict.fromkeys(row["event_keys"]))[:4],
                "descriptions": list(dict.fromkeys(row["descriptions"]))[:3],
                "event_class": row["event_class"],
                "counting_unit": row["counting_unit"],
                "participant_ids": list(row["participant_ids"]),
                "participants": [dict(item) for item in row["participants"]],
                "states_before": [item for item in dict.fromkeys(row["states_before"]) if item],
                "transitions": [item for item in dict.fromkeys(row["transitions"]) if item],
                "states_after": [item for item in dict.fromkeys(row["states_after"]) if item],
                "merge_history": list(row["merge_history"])[:12],
                "phases": list(dict.fromkeys(row["phases"])),
                "candidate_status": candidate_status,
                "candidate_conflict_reasons": list(dict.fromkeys(row["candidate_conflict_reasons"])),
                "participant_binding_status": (
                    "ambiguous" if row["participant_binding_conflicts"] else "resolved"
                ),
                "participant_binding_conflicts": list(row["participant_binding_conflicts"])[:8],
                "qualification_observations": dict(row.get("qualification_observations", {}) or {}),
                "preconditions_met_observations": list(row.get("preconditions_met", ()) or ()),
                "supports_question_event_observations": list(row.get("supports_question_event", ()) or ()),
                "continues_from_previous": bool(row.get("continues_from_previous")),
                "continues_to_next": bool(row.get("continues_to_next")),
                "focal_position_transition": _candidate_is_focal_position_transition(row),
            }
        )
    observed = [row for row in public if row["candidate_status"] == "observed_candidate"]
    conflicted = [row for row in public if row["candidate_status"] == "conflicted_candidate"]
    unresolved_rows = _merge_unresolved(unresolved)
    return {
        "confirmed_event_candidate_count": 0,
        "confirmed_event_candidates": [],
        "observed_event_candidate_count": len(observed),
        "observed_event_candidates": observed[:40],
        "unqualified_event_candidate_count": 0,
        "unqualified_event_candidates": [],
        "conflicted_event_candidate_count": len(conflicted),
        "conflicted_event_candidates": conflicted[:40],
        "unresolved_event_candidate_count": len(unresolved_rows),
        "unresolved_event_windows": unresolved_rows[:16],
    }


def _record_fact_provenance(record: EvidenceRecord, *, fact_id: str) -> tuple[dict[str, Any], ...]:
    parse_status = str(record.operation_metadata.get("structured_parse_status", "") or "").casefold()
    has_witness = bool(record.frame_refs) and record.start_sec is not None and record.end_sec is not None
    if has_witness and parse_status in {"", "parsed", "ok"}:
        return normalize_provenance((
            direct_witness_provenance(
                fact_ids=(fact_id,),
                evidence_ids=(record.evidence_id,),
                witness_ranges=((float(record.start_sec), float(record.end_sec)),),
            ),
        ))
    return normalize_provenance((
        heuristic_provenance(
            fact_ids=(fact_id,),
            evidence_ids=(record.evidence_id,),
            derivation="observation_missing_direct_witness_requirements",
        ),
    ))


def _candidate_with_provenance(
    candidate: Mapping[str, Any],
    evidence_by_id: Mapping[str, EvidenceRecord],
) -> dict[str, Any]:
    row = dict(candidate)
    candidate_id = str(row.get("candidate_id", "") or row.get("signature", "") or "event_candidate")
    provenance = []
    for evidence_id in tuple(row.get("evidence_ids", ()) or ()):
        record = evidence_by_id.get(str(evidence_id))
        if record is not None:
            provenance.extend(_record_fact_provenance(record, fact_id=candidate_id))
    if not provenance:
        provenance.append(heuristic_provenance(
            fact_ids=(candidate_id,),
            evidence_ids=tuple(row.get("evidence_ids", ()) or ()),
            derivation="candidate_without_observation_witness",
        ))
    row["provenance"] = normalize_provenance(provenance)
    return row


def _append_attribute_facts(
    destination: list[dict[str, Any]],
    *,
    entity_id: str,
    episode_id: str,
    event_role: str,
    attributes: Mapping[str, Any],
    record: EvidenceRecord,
    provenance: Sequence[Mapping[str, Any]],
) -> None:
    if not entity_id or not attributes or record.start_sec is None or record.end_sec is None:
        return
    for attribute_type, attribute_value in dict(attributes).items():
        normalized_type = str(attribute_type or "").strip().casefold()
        if not normalized_type or attribute_value is None or not str(attribute_value).strip():
            continue
        fact_id = f"attribute:{entity_id}:{episode_id or 'unbound'}:{normalized_type}:{record.evidence_id}"
        destination.append(
            {
                "fact_id": fact_id,
                "entity_id": entity_id,
                "episode_id": episode_id,
                "event_role": event_role,
                "attribute_type": normalized_type,
                "attribute_value": attribute_value,
                "witness_range": [float(record.start_sec), float(record.end_sec)],
                "observation_id": str(record.observation_id or record.evidence_id),
                "evidence_ids": [record.evidence_id],
                "provenance": [dict(item) for item in provenance],
            }
        )


def _typed_state_transition_fact(
    raw: Mapping[str, Any],
    *,
    record: EvidenceRecord,
    fact_id: str,
) -> dict[str, Any]:
    object_id = str(raw.get("object_hypothesis_id", raw.get("object_id", "")) or "").strip()
    attribute_type = str(raw.get("attribute_type", "") or "").strip().casefold()
    before = raw.get("raw_value_before", raw.get("value_before"))
    after = raw.get("raw_value_after", raw.get("value_after"))
    same_object = str(raw.get("same_object_relation", "unknown") or "unknown").casefold()
    before_witness = raw.get("before_witness", raw.get("before_witness_range"))
    after_witness = raw.get("after_witness", raw.get("after_witness_range"))
    occlusion = str(raw.get("coverage_occlusion_status", raw.get("occlusion_status", "unknown")) or "unknown").casefold()
    witness_ranges = _transition_witness_ranges(before_witness, after_witness)
    supported = bool(
        object_id
        and attribute_type
        and before is not None
        and after is not None
        and str(before).strip()
        and str(after).strip()
        and same_object == "supported"
        and len(witness_ranges) == 2
        and occlusion not in {"occluded", "unknown"}
    )
    provenance = _record_fact_provenance(record, fact_id=fact_id) if supported else normalize_provenance((
        heuristic_provenance(
            fact_ids=(fact_id,),
            evidence_ids=(record.evidence_id,),
            derivation="state_transition_missing_same_object_or_before_after_witness",
        ),
    ))
    return {
        "fact_id": fact_id,
        "object_hypothesis_id": object_id,
        "attribute_type": attribute_type,
        "raw_value_before": before,
        "raw_value_after": after,
        "before_witness": before_witness,
        "after_witness": after_witness,
        "same_object_relation": same_object == "supported",
        "coverage_occlusion_status": occlusion,
        "witness_ranges": witness_ranges,
        "evidence_ids": [record.evidence_id],
        "status": "supported" if supported else "unknown",
        "provenance": [dict(item) for item in provenance],
    }


def _transition_witness_ranges(*values: Any) -> list[list[float]]:
    ranges = []
    for value in values:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
            continue
        try:
            start, end = sorted((float(value[0]), float(value[1])))
        except (TypeError, ValueError):
            continue
        ranges.append([start, end])
    return ranges


def canonical_fact_snapshot(
    evidence: Sequence[EvidenceRecord],
    *,
    require_event_precondition: bool = False,
) -> CanonicalFactSnapshot:
    ledger = event_candidate_ledger(evidence)
    evidence_by_id = {record.evidence_id: record for record in evidence}
    observed_candidates = tuple(
        _candidate_with_provenance(candidate, evidence_by_id)
        for candidate in tuple(ledger.get("observed_event_candidates", ()) or ())
    )
    conflicted_candidates = tuple(
        _candidate_with_provenance(candidate, evidence_by_id)
        for candidate in tuple(ledger.get("conflicted_event_candidates", ()) or ())
    )
    event_qualification = qualify_event_candidates(
        (
            *observed_candidates,
            *conflicted_candidates,
        ),
        require_state_precondition=require_event_precondition,
    )
    raw_event_count = 0
    refuted_events: list[dict[str, Any]] = []
    entities: dict[str, dict[str, Any]] = {}
    unresolved_entities: dict[str, dict[str, Any]] = {}
    transitions: list[dict[str, Any]] = []
    unresolved_transitions: list[dict[str, Any]] = []
    associations: list[dict[str, Any]] = []
    attribute_facts: list[dict[str, Any]] = []
    inferred_facts: list[dict[str, Any]] = []
    unresolved_inferences: list[dict[str, Any]] = []
    for record in evidence:
        for event_index, event in enumerate(record.operation_metadata.get("events", ()) or ()):
            if not isinstance(event, Mapping):
                continue
            raw_event_count += 1
            status = str(event.get("status", "") or "").strip().casefold()
            if status in {"refuted", "contradicted"} or event.get("supports_question_event") is False:
                refuted_events.append(
                    {
                        "fact_id": f"refuted_event_{len(refuted_events) + 1:03d}",
                        "event_key": str(event.get("event_key", "") or ""),
                        "evidence_ids": [record.evidence_id],
                        "raw_candidate_index": event_index,
                    }
                )
            for participant in tuple(event.get("participants", ()) or ()):
                if not isinstance(participant, Mapping):
                    continue
                hypothesis_id = str(participant.get("entity_hypothesis_id", "") or "").strip()
                participant_id = str(participant.get("participant_id", "") or hypothesis_id).strip()
                if not participant_id:
                    continue
                try:
                    confidence = float(participant.get("association_confidence", 0.0) or 0.0)
                except (TypeError, ValueError):
                    confidence = 0.0
                entity_row = {
                    "entity_id": hypothesis_id or participant_id,
                    "entity_observation_id": f"{record.evidence_id}:{participant_id}",
                    "association_confidence": max(0.0, min(1.0, confidence)),
                    "role": str(participant.get("role", "") or ""),
                    "visual_signature": str(participant.get("visual_signature", "") or ""),
                    "attributes": dict(participant.get("attributes", {}) or {}),
                    "source_event_key": str(event.get("event_key", "") or ""),
                    "episode_id": str(event.get("event_key", "") or ""),
                    "evidence_ids": [record.evidence_id],
                    "provenance": _record_fact_provenance(
                        record,
                        fact_id=f"entity:{hypothesis_id or participant_id}",
                    ),
                }
                if hypothesis_id and confidence >= 0.6:
                    existing = entities.setdefault(hypothesis_id, entity_row)
                    existing["evidence_ids"] = list(
                        dict.fromkeys((*existing["evidence_ids"], record.evidence_id))
                    )
                    existing["association_confidence"] = max(
                        float(existing.get("association_confidence", 0.0) or 0.0),
                        entity_row["association_confidence"],
                    )
                    existing["attributes"] = {**dict(existing.get("attributes", {}) or {}), **entity_row["attributes"]}
                else:
                    unresolved_entities.setdefault(entity_row["entity_id"], entity_row)
                _append_attribute_facts(
                    attribute_facts,
                    entity_id=entity_row["entity_id"],
                    episode_id=entity_row["episode_id"],
                    event_role=entity_row["role"],
                    attributes=entity_row["attributes"],
                    record=record,
                    provenance=entity_row["provenance"],
                )
        for entity in record.operation_metadata.get("entities", ()) or ():
            if not isinstance(entity, Mapping):
                continue
            observation_id = str(entity.get("entity_observation_id", "") or "").strip()
            if not observation_id:
                continue
            hypothesis_id = str(entity.get("entity_hypothesis_id", "") or "").strip()
            try:
                association_confidence = float(entity.get("association_confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                association_confidence = 0.0
            row = {
                "entity_id": hypothesis_id or observation_id,
                "entity_observation_id": observation_id,
                "association_confidence": max(0.0, min(1.0, association_confidence)),
                "role": str(entity.get("role", "") or ""),
                "visual_signature": str(entity.get("visual_signature", "") or ""),
                "attributes": dict(entity.get("attributes", {}) or {}),
                "episode_id": str(entity.get("episode_id", "") or ""),
                "event_role": str(entity.get("role", "") or ""),
                "evidence_ids": [record.evidence_id],
                "provenance": _record_fact_provenance(
                    record,
                    fact_id=f"entity:{hypothesis_id or observation_id}",
                ),
            }
            if _flag(entity.get("countable")) and hypothesis_id and association_confidence >= 0.6:
                existing = entities.setdefault(row["entity_id"], row)
                existing["evidence_ids"] = list(dict.fromkeys((*existing["evidence_ids"], record.evidence_id)))
                existing["association_confidence"] = max(
                    float(existing.get("association_confidence", 0.0) or 0.0),
                    row["association_confidence"],
                )
                existing["attributes"] = {**dict(existing.get("attributes", {}) or {}), **row["attributes"]}
            else:
                unresolved_entities.setdefault(row["entity_id"], row)
            _append_attribute_facts(
                attribute_facts,
                entity_id=row["entity_id"],
                episode_id=row["episode_id"],
                event_role=row["event_role"],
                attributes=row["attributes"],
                record=record,
                provenance=row["provenance"],
            )
        for relation in normalize_relations(record.operation_metadata.get("relations"), evidence_id=record.evidence_id):
            if relation.relation_type != "transition" or relation.status != "supported":
                continue
            transitions.append(
                {
                    "fact_id": f"state_transition_{len(transitions) + 1:03d}",
                    "object_hypothesis_id": relation.subject_id,
                    "related_object_id": relation.object_id,
                    "transition": relation.value or relation.description,
                    "evidence_ids": list(relation.evidence_ids),
                    "provenance": [
                        heuristic_provenance(
                            fact_ids=(f"state_transition_{len(transitions) + 1:03d}",),
                            evidence_ids=relation.evidence_ids,
                            derivation="transition_relation_without_before_after_witness_chain",
                        )
                    ],
                }
            )
        for raw_transition in tuple(record.operation_metadata.get("state_transitions", ()) or ()):
            if not isinstance(raw_transition, Mapping):
                continue
            row = _typed_state_transition_fact(
                raw_transition,
                record=record,
                fact_id=f"state_transition_typed_{len(transitions) + len(unresolved_transitions) + 1:03d}",
            )
            if row["status"] == "supported":
                transitions.append(row)
            else:
                unresolved_transitions.append(row)
        for raw_association in record.operation_metadata.get("entity_associations", ()) or ():
            if not isinstance(raw_association, Mapping):
                continue
            status = str(raw_association.get("status", "unknown") or "unknown").strip().casefold()
            try:
                confidence = float(raw_association.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            source_participant_id = str(raw_association.get("source_participant_id", "") or "").strip()
            hypothesis_id = str(raw_association.get("entity_hypothesis_id", "") or "").strip()
            target_observation_id = str(raw_association.get("target_entity_observation_id", "") or "").strip()
            if not source_participant_id or not hypothesis_id or not target_observation_id:
                continue
            row = {
                "association_id": str(raw_association.get("association_id", "") or f"association_{len(associations) + 1:03d}"),
                "source_participant_id": source_participant_id,
                "source_event_key": str(raw_association.get("source_event_key", "") or ""),
                "source_episode_id": str(raw_association.get("source_episode_id", "") or raw_association.get("source_event_key", "") or ""),
                "source_event_role": str(raw_association.get("source_event_role", "") or ""),
                "ordinal": raw_association.get("ordinal", None),
                "target_entity_observation_id": target_observation_id,
                "entity_hypothesis_id": hypothesis_id,
                "status": status if status in {"supported", "refuted", "unknown"} else "unknown",
                "confidence": max(0.0, min(1.0, confidence)),
                "shared_attributes": dict(raw_association.get("shared_attributes", {}) or {}),
                "distinguishing_attributes": dict(raw_association.get("distinguishing_attributes", {}) or {}),
                "evidence_ids": [record.evidence_id],
                "provenance": normalize_provenance(raw_association.get("provenance", ())) or (
                    heuristic_provenance(
                        fact_ids=(str(raw_association.get("association_id", "") or ""),),
                        evidence_ids=(record.evidence_id,),
                        derivation="entity_association_model_assertion",
                    ),
                ),
            }
            associations.append(row)
            if row["status"] != "supported" or row["confidence"] < 0.6:
                continue
            target = next(
                (
                    entity for entity in record.operation_metadata.get("entities", ()) or ()
                    if isinstance(entity, Mapping)
                    and str(entity.get("entity_observation_id", "") or "") == target_observation_id
                ),
                {},
            )
            entity_row = {
                "entity_id": hypothesis_id,
                "entity_observation_id": target_observation_id,
                "association_confidence": row["confidence"],
                "role": str(target.get("role", "") or ""),
                "visual_signature": str(target.get("visual_signature", "") or ""),
                "attributes": dict(target.get("attributes", {}) or {}),
                "source_participant_id": source_participant_id,
                "source_event_key": row["source_event_key"],
                "episode_id": row["source_episode_id"],
                "event_role": row["source_event_role"],
                "evidence_ids": [record.evidence_id],
                "provenance": row["provenance"],
            }
            existing = entities.setdefault(hypothesis_id, entity_row)
            existing["evidence_ids"] = list(dict.fromkeys((*existing["evidence_ids"], record.evidence_id)))
            existing["association_confidence"] = max(
                float(existing.get("association_confidence", 0.0) or 0.0),
                entity_row["association_confidence"],
            )
            existing["attributes"] = {
                **dict(existing.get("attributes", {}) or {}),
                **entity_row["attributes"],
                **dict(row["distinguishing_attributes"]),
            }
            if entity_row["visual_signature"]:
                existing["visual_signature"] = entity_row["visual_signature"]
            _append_attribute_facts(
                attribute_facts,
                entity_id=hypothesis_id,
                episode_id=entity_row["episode_id"],
                event_role=entity_row["event_role"],
                attributes=entity_row["attributes"],
                record=record,
                provenance=entity_row["provenance"],
            )
        for raw_fact in record.operation_metadata.get("narrative_facts", ()) or ():
            if not isinstance(raw_fact, Mapping):
                continue
            try:
                confidence = float(raw_fact.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            row = {
                "fact_id": str(raw_fact.get("fact_id", "") or f"inferred_fact_{len(inferred_facts) + len(unresolved_inferences) + 1:03d}"),
                "episode_id": str(raw_fact.get("episode_id", "") or "").strip(),
                "timeline_phase": str(raw_fact.get("timeline_phase", "unknown") or "unknown").strip().casefold(),
                "temporal_role": str(raw_fact.get("temporal_role", raw_fact.get("timeline_phase", "unknown")) or "unknown").strip().casefold(),
                "anchor_event_id": str(raw_fact.get("anchor_event_id", "") or "").strip(),
                "anchor_match": _flag(raw_fact.get("anchor_match")),
                "subject_id": str(raw_fact.get("subject_id", "") or ""),
                "relation_type": str(raw_fact.get("relation_type", "") or "").strip().casefold(),
                "predicate": str(raw_fact.get("predicate", "") or "").strip().casefold(),
                "setup_state": str(raw_fact.get("setup_state", "") or ""),
                "observed_bridge": str(raw_fact.get("observed_bridge", "") or ""),
                "outcome_state": str(raw_fact.get("outcome_state", "") or ""),
                "inference": str(raw_fact.get("inference", "") or ""),
                "inference_basis": str(raw_fact.get("inference_basis", "") or ""),
                "confidence": max(0.0, min(1.0, confidence)),
                "hypothesis_assessments": [
                    dict(item) for item in tuple(raw_fact.get("hypothesis_assessments", ()) or ())
                    if isinstance(item, Mapping)
                ],
                "alternative_counterevidence": [
                    dict(item) for item in tuple(raw_fact.get("alternative_counterevidence", ()) or ())
                    if isinstance(item, Mapping)
                ],
                "evidence_ids": [record.evidence_id],
                "supporting_observation_ids": list(dict.fromkeys((
                    str(record.observation_id or record.evidence_id),
                    *(str(item) for item in tuple(raw_fact.get("supporting_observation_ids", ()) or ()) if str(item)),
                ))),
                "competing_explanation_evidence": [
                    dict(item) for item in tuple(raw_fact.get("alternative_counterevidence", ()) or ())
                    if isinstance(item, Mapping)
                ],
                "agent_witness_type": str(raw_fact.get("agent_witness_type", "") or "").strip().casefold(),
                "provenance": normalize_provenance(raw_fact.get("provenance", ())) or (
                    heuristic_provenance(
                        fact_ids=(str(raw_fact.get("fact_id", "") or ""),),
                        evidence_ids=(record.evidence_id,),
                        derivation="narrative_model_inference",
                    ),
                ),
            }
            if row["relation_type"] == "temporal_cooccurrence":
                row["hypothesis_assessments"] = [
                    {
                        **assessment,
                        "status": "unknown" if str(assessment.get("status", "")) == "supported" else assessment.get("status", "unknown"),
                        "reason": "Temporal co-occurrence cannot establish agent causation.",
                    }
                    for assessment in row["hypothesis_assessments"]
                ]
            complete = bool(
                row["episode_id"]
                and row["setup_state"]
                and row["outcome_state"]
                and row["inference"]
                and row["relation_type"]
            )
            if complete and row["confidence"] >= 0.6:
                inferred_facts.append(row)
            else:
                unresolved_inferences.append(row)
    resolved_source_participants = {
        str(row["source_participant_id"]).casefold()
        for row in associations
        if row["status"] == "supported" and float(row["confidence"]) >= 0.6
    }
    unresolved_entities = {
        key: row for key, row in unresolved_entities.items()
        if str(row.get("entity_id", "") or "").casefold() not in resolved_source_participants
    }
    if inferred_facts:
        completed_episodes = {str(row.get("episode_id", "") or "") for row in inferred_facts}
        unresolved_inferences = [
            row for row in unresolved_inferences
            if str(row.get("episode_id", "") or "") not in completed_episodes
        ]
    qualified = tuple(dict(row) for row in event_qualification["qualified_events"])
    suspects = tuple(
        {
            "fact_id": f"duplicate_suspect_{index:03d}",
            "status": "unresolved",
            **dict(row),
        }
        for index, row in enumerate(ledger["unresolved_event_windows"], start=1)
    )
    ordered = tuple(
        {
            "event_id": str(row.get("candidate_id", "") or ""),
            "time": float(tuple(row.get("virtual_time_range", (0.0, 0.0)))[0]),
            "evidence_ids": list(row.get("evidence_ids", ()) or ()),
            "provenance": [
                deterministic_derivation_provenance(
                    derivation="canonical_event_order",
                    fact_ids=(str(row.get("candidate_id", "") or ""),),
                    evidence_ids=tuple(row.get("evidence_ids", ()) or ()),
                    source_kinds=provenance_kinds(row.get("provenance", ())),
                )
            ],
        }
        for row in qualified
    )
    return CanonicalFactSnapshot(
        qualified_events=qualified,
        observed_event_candidates=observed_candidates,
        unqualified_events=tuple(event_qualification["unqualified_events"]),
        incomplete_events=tuple(event_qualification["incomplete_events"]),
        conflicted_events=tuple(event_qualification["conflicted_events"]),
        duplicate_suspect_events=suspects,
        refuted_events=tuple(refuted_events),
        resolved_entities=tuple(entities.values()),
        unresolved_entity_bindings=tuple(unresolved_entities.values()),
        state_transitions=tuple(transitions),
        unresolved_state_transitions=tuple(unresolved_transitions),
        entity_associations=tuple(associations),
        attribute_facts=tuple(attribute_facts),
        inferred_facts=tuple(inferred_facts),
        unresolved_inferences=tuple(unresolved_inferences),
        ordered_events=ordered,
        raw_candidate_counts={"events": raw_event_count},
        requirement_graph=dict(event_qualification["requirement_graph"]),
    )


def _event_qualification_status(event: Mapping[str, Any]) -> tuple[str, str]:
    explicit = str(event.get("qualification_status", "") or "").strip().casefold()
    aliases = {
        "confirmed": "qualified",
        "supported": "qualified",
        "candidate": "observed_candidate",
        "unqualified": "unqualified_precondition",
        "contradicted": "conflicted",
    }
    explicit = aliases.get(explicit, explicit)
    if explicit in {
        "qualified",
        "observed_candidate",
        "unqualified_precondition",
        "conflicted",
        "refuted",
    }:
        return explicit, f"explicit_{explicit}"
    if event.get("preconditions_met") is False:
        return "unqualified_precondition", "required_precondition_not_met"

    qualification = dict(event.get("qualification", {}) or {}) if isinstance(
        event.get("qualification"), Mapping
    ) else {}
    if _is_focal_position_transition(event):
        required = {
            "required_prior_state": (
                qualification.get("required_prior_state")
                or qualification.get("recorder_was_first_before")
            ),
            "transition": qualification.get("transition") or qualification.get("overtake_transition"),
            "same_subject": qualification.get("same_subject") or qualification.get("same_recorder"),
            "episode_boundary": qualification.get("episode_boundary"),
        }
        statuses = {key: _qualification_value(value) for key, value in required.items()}
        if any(status == "conflicted" for status in statuses.values()):
            return "conflicted", "position_event_qualification_conflicted"
        if any(status == "refuted" for status in statuses.values()):
            return "unqualified_precondition", "position_event_precondition_refuted"
        if all(status == "supported" for status in statuses.values()):
            return "qualified", "all_position_event_qualifiers_supported"
        return "observed_candidate", "position_event_qualification_incomplete"
    if _flag(event.get("supports_question_event")):
        return "qualified", "legacy_nonconditional_event_support"
    return "refuted", "question_event_not_supported"


def _qualification_value(value: Any) -> str:
    if value is True:
        return "supported"
    if value is False:
        return "refuted"
    return {
        "supported": "supported",
        "verified": "supported",
        "satisfied": "supported",
        "refuted": "refuted",
        "contradicted": "refuted",
        "conflicted": "conflicted",
    }.get(str(value or "unknown").strip().casefold(), "unknown")


def _is_focal_position_transition(event: Mapping[str, Any]) -> bool:
    participants = {
        _normalize_participant_id(value)
        for value in tuple(event.get("participant_ids", ()) or ())
        if _normalize_participant_id(value)
    }
    if "camera_holder" not in participants:
        return False
    text = " ".join(
        str(event.get(name, "") or "").casefold()
        for name in (
            "event_key",
            "event_class",
            "counting_unit",
            "state_before",
            "transition",
            "state_after",
        )
    )
    return bool(re.search(r"\b(?:overtak\w*|pass(?:es|ed|ing)?|rank|first place|lead|position loss)\b", text))


def _candidate_is_focal_position_transition(candidate: Mapping[str, Any]) -> bool:
    participants = set(candidate.get("participant_ids", ()) or ())
    if "camera_holder" not in participants:
        return False
    text = " ".join((
        str(candidate.get("event_class", "") or ""),
        str(candidate.get("counting_unit", "") or ""),
        *tuple(str(item) for item in tuple(candidate.get("event_keys", ()) or ())),
        *tuple(str(item) for item in tuple(candidate.get("transitions", ()) or ())),
        *tuple(str(item) for item in tuple(candidate.get("descriptions", ()) or ())),
    )).casefold()
    return bool(re.search(r"\b(?:overtak\w*|pass(?:es|ed|ing)?|rank|first place|lead|position loss)\b", text))


def _combined_qualification_status(row: Mapping[str, Any]) -> str:
    statuses = set(row.get("qualification_statuses", ()) or ())
    if "conflicted" in statuses:
        return "conflicted"
    if "qualified" in statuses:
        return "qualified"
    if "unqualified_precondition" in statuses:
        return "unqualified_precondition"
    return "observed_candidate"


def _event_signature(event: Mapping[str, Any]) -> str:
    participants = tuple(
        _normalize_participant_id(value)
        for value in tuple(event.get("participant_ids", ()) or ())
        if _normalize_participant_id(value)
    )
    if participants:
        return " ".join(dict.fromkeys(participants))[:120]
    event_key = str(event.get("event_key", "") or "").strip()
    description = str(event.get("description", "") or "").strip()
    text = (event_key or description).casefold()
    tokens = [token for token in re.findall(r"[a-z0-9]+", text) if len(token) > 1 and token not in _GENERIC_EVENT_TOKENS]
    if not tokens and description and description.casefold() != text:
        tokens = [
            token for token in re.findall(r"[a-z0-9]+", description.casefold())
            if len(token) > 1 and token not in _GENERIC_EVENT_TOKENS
        ]
    return " ".join(dict.fromkeys(tokens))[:120] if tokens else ""


def _same_counted_occurrence(
    start: float,
    end: float,
    *,
    event_key: str,
    event_class: str,
    counting_unit: str,
    participant_ids: Sequence[str],
    from_previous: bool,
    to_next: bool,
    existing: Mapping[str, Any],
) -> bool:
    if _equivalent_interval(start, end, *existing["virtual_time_range"]):
        return True
    if _continuation(start, end, event_key, from_previous, to_next, existing):
        return True
    existing_participants = tuple(existing.get("participant_ids", ()) or ())
    same_participants = bool(participant_ids and existing_participants and set(participant_ids) == set(existing_participants))
    unit = counting_unit or str(existing.get("counting_unit", "") or "")
    event_type = event_class or str(existing.get("event_class", "") or "")
    gap = max(
        0.0,
        start - float(existing["virtual_time_range"][1]),
        float(existing["virtual_time_range"][0]) - end,
    )
    if unit == "audition_group" or event_type == "audition":
        return bool(same_participants and gap <= 300.0)
    if unit == "news_broadcast_appearance" or event_type == "news_segment":
        return bool(same_participants and gap <= 5.0)
    return False


def _same_focal_transition_episode(
    start: float,
    end: float,
    *,
    event_class: str,
    counting_unit: str,
    participant_ids: Sequence[str],
    transition: str,
    existing: Mapping[str, Any],
) -> bool:
    if not _equivalent_interval(start, end, *existing["virtual_time_range"]):
        return False
    existing_participants = set(existing.get("participant_ids", ()) or ())
    shared = existing_participants.intersection(participant_ids)
    if not shared.intersection({"camera_holder", "video recorder", "recorder"}):
        return False
    existing_class = str(existing.get("event_class", "") or "")
    if event_class and existing_class and event_class != existing_class:
        return False
    units = {item for item in (counting_unit, str(existing.get("counting_unit", "") or "")) if item}
    if any("person" in unit or "racer" in unit for unit in units):
        return False
    existing_transitions = {str(item) for item in existing.get("transitions", ()) or () if str(item)}
    if transition and existing_transitions and not any(
        _transition_family(transition) == _transition_family(item) for item in existing_transitions
    ):
        return False
    return True


def _transition_family(value: str) -> str:
    text = _normalize_key(value)
    if any(token in text for token in ("overtak", "pass", "rank", "place", "drops position")):
        return "position_loss"
    return text


def _normalize_participant_id(value: Any) -> str:
    generic = {"dance", "group", "crew", "team", "performer", "performers", "act"}
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").casefold())
        if token not in generic
    ]
    normalized = " ".join(tokens)
    if normalized in {
        "video recorder", "recorder", "camera wearer", "camera holder", "camera operator",
        "camera operator wearer", "pov skier", "pov rider",
    }:
        return "camera_holder"
    return normalized


def _merge_unresolved(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (str(item.get("source_video_id", "")), item["virtual_time_range"][0])):
        start, end = (float(value) for value in row["virtual_time_range"])
        existing = next(
            (
                item for item in merged
                if item["source_video_id"] == row.get("source_video_id")
                and min(item["virtual_time_range"][1], end) >= max(item["virtual_time_range"][0], start)
            ),
            None,
        )
        if existing is None:
            merged.append(
                {
                    "source_video_id": str(row.get("source_video_id", "") or ""),
                    "virtual_time_range": [start, end],
                    "evidence_ids": list(row.get("evidence_ids", ()) or ()),
                    "descriptions": list(row.get("descriptions", ()) or ()),
                }
            )
            continue
        existing["virtual_time_range"] = [min(existing["virtual_time_range"][0], start), max(existing["virtual_time_range"][1], end)]
        existing["evidence_ids"] = list(dict.fromkeys((*existing["evidence_ids"], *row.get("evidence_ids", ()))))[:12]
        existing["descriptions"] = list(dict.fromkeys((*existing["descriptions"], *row.get("descriptions", ()))))[:3]
    return merged


def _segment_at(workspace: VirtualVideoWorkspace, time_sec: float):
    return next(
        (segment for segment in workspace.manifest.segments if segment.virtual_start_sec <= time_sec <= segment.virtual_end_sec),
        None,
    )


def _source_id(record: EvidenceRecord, start: float, end: float) -> str:
    center = (start + end) / 2.0
    for lineage in record.source_lineage:
        bounds = tuple(lineage.get("virtual_time_range", ()) or ())
        if len(bounds) == 2 and float(bounds[0]) <= center <= float(bounds[1]):
            return str(lineage.get("source_video_id", "") or "")
    return str(record.source_lineage[0].get("source_video_id", "") or "") if record.source_lineage else ""


def _equivalent_interval(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    overlap = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    shorter = min(max(0.0, end_a - start_a), max(0.0, end_b - start_b))
    return bool((shorter > 0.0 and overlap / shorter >= 0.5) or abs((start_a + end_a - start_b - end_b) / 2.0) <= 1.0)


def _continuation(
    start: float,
    end: float,
    event_key: str,
    from_previous: bool,
    to_next: bool,
    existing: Mapping[str, Any],
) -> bool:
    if not event_key or not existing.get("canonical_event_key"):
        return False
    return bool(
        from_previous and existing.get("continues_to_next") and abs(start - existing["virtual_time_range"][1]) <= 2.0
        or to_next and existing.get("continues_from_previous") and abs(end - existing["virtual_time_range"][0]) <= 2.0
    )


def _normalize_key(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _flag(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value or "").strip().casefold() in {"1", "true", "yes"}


def _score_pair(text: str) -> tuple[int, int] | None:
    match = re.search(r"(?<!\d)(\d{1,3})\s*(?:-|–|—|:)\s*(\d{1,3})(?!\d)", str(text or ""))
    return (int(match.group(1)), int(match.group(2))) if match else None
