from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from vcah.evidence_primitives import normalize_relations
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
            return "boundary_score_unvisited_segments", requests
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
    "crew", "gymnastics",
}


def event_candidate_ledger(evidence: Sequence[EvidenceRecord]) -> dict[str, Any]:
    confirmed: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for record in evidence:
        for event in record.operation_metadata.get("events", ()) or ():
            if not isinstance(event, Mapping) or not _flag(event.get("supports_question_event")):
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
            from_previous = _flag(event.get("continues_from_previous"))
            to_next = _flag(event.get("continues_to_next"))
            row = next(
                (
                    item for item in confirmed
                    if item["source_video_id"] == source_id
                    and item["signature"] == signature
                    and (
                        _equivalent_interval(start, end, *item["virtual_time_range"])
                        or _continuation(start, end, event_key, from_previous, to_next, item)
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
                    "continues_from_previous": from_previous,
                    "continues_to_next": to_next,
                }
                confirmed.append(row)
            row["virtual_time_range"] = [min(row["virtual_time_range"][0], start), max(row["virtual_time_range"][1], end)]
            row["evidence_ids"].append(record.evidence_id)
            row["event_keys"].append(str(event.get("event_key", "") or ""))
            row["descriptions"].append(str(event.get("description", "") or ""))
            row["continues_from_previous"] = bool(row["continues_from_previous"] or from_previous)
            row["continues_to_next"] = bool(row["continues_to_next"] or to_next)
    public = []
    for index, row in enumerate(sorted(confirmed, key=lambda item: item["virtual_time_range"][0]), start=1):
        public.append(
            {
                "candidate_id": f"event_candidate_{index:03d}",
                "signature": row["signature"],
                "source_video_id": row["source_video_id"],
                "virtual_time_range": row["virtual_time_range"],
                "evidence_ids": list(dict.fromkeys(row["evidence_ids"]))[:12],
                "event_keys": list(dict.fromkeys(row["event_keys"]))[:4],
                "descriptions": list(dict.fromkeys(row["descriptions"]))[:3],
            }
        )
    unresolved_rows = _merge_unresolved(unresolved)
    return {
        "confirmed_event_candidate_count": len(public),
        "confirmed_event_candidates": public[:40],
        "unresolved_event_candidate_count": len(unresolved_rows),
        "unresolved_event_windows": unresolved_rows[:16],
    }


def _event_signature(event: Mapping[str, Any]) -> str:
    event_key = str(event.get("event_key", "") or "").strip()
    text = (event_key or str(event.get("description", "") or "")).casefold()
    tokens = [token for token in re.findall(r"[a-z0-9]+", text) if len(token) > 1 and token not in _GENERIC_EVENT_TOKENS]
    return " ".join(dict.fromkeys(tokens))[:120] if tokens else ""


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
    if not event_key or event_key != existing.get("canonical_event_key"):
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
