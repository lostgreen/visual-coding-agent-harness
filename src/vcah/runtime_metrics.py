from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from vcah.caption_schema import CaptionHitV1
from vcah.workspace import Claim, WorkingDocument


def merge_intervals(
    intervals: Sequence[Sequence[float]],
    *,
    adjacency_tolerance_sec: float = 1e-9,
) -> tuple[tuple[float, float], ...]:
    normalized = sorted(
        (float(item[0]), float(item[1]))
        for item in intervals
        if len(item) == 2 and float(item[1]) > float(item[0])
    )
    merged: list[list[float]] = []
    tolerance = max(0.0, float(adjacency_tolerance_sec))
    for start, end in normalized:
        if not merged or start > merged[-1][1] + tolerance:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return tuple((start, end) for start, end in merged)


def export_supporting_intervals(
    document: WorkingDocument,
    supporting_claim_ids: Sequence[str],
    observation_rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[float, float], ...]:
    rows_by_attempt: dict[str, list[Mapping[str, Any]]] = {}
    for row in observation_rows:
        attempt_id = str(row.get("attempt_id", "") or "")
        if attempt_id:
            rows_by_attempt.setdefault(attempt_id, []).append(row)

    intervals: list[Sequence[float]] = []
    for claim_id in dict.fromkeys(str(item) for item in supporting_claim_ids if str(item)):
        claim = document.claims.get(claim_id)
        if claim is None or claim.status != "active":
            continue
        claim_anchors = _claim_anchor_intervals(document, claim)
        if claim_anchors:
            intervals.extend(claim_anchors)
            continue
        for attempt_id in _claim_attempt_ids(document, claim):
            for row in rows_by_attempt.get(attempt_id, ()):
                if not _attempt_is_supporting(row):
                    continue
                intervals.extend(tuple(row.get("inspected_ranges", ()) or ()))
    return merge_intervals(intervals)


def export_item_supporting_intervals(
    supporting_item_ids: Sequence[str],
    observation_rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[float, float], ...]:
    requested = {str(item) for item in supporting_item_ids if str(item)}
    intervals: list[tuple[float, float]] = []
    for row in observation_rows:
        fps = max(1e-6, float(row.get("sampling_fps", 0.0) or 0.0))
        inspected = tuple(row.get("inspected_ranges", ()) or ())
        for item in tuple(row.get("interpretation_items", ()) or ()):
            if not isinstance(item, Mapping) or str(item.get("item_id", "") or "") not in requested:
                continue
            anchor = tuple(item.get("time_anchor", ()) or ())
            if len(anchor) != 2:
                continue
            start, end = sorted((float(anchor[0]), float(anchor[1])))
            if end <= start:
                half_width = 0.5 / fps
                containing = next(
                    (
                        tuple(sorted((float(value[0]), float(value[1]))))
                        for value in inspected
                        if len(value) == 2
                        and float(value[0]) - 1e-6 <= start <= float(value[1]) + 1e-6
                    ),
                    None,
                )
                lower = containing[0] if containing else start - half_width
                upper = containing[1] if containing else start + half_width
                start = max(lower, start - half_width)
                end = min(upper, start + 2 * half_width)
            if end > start:
                intervals.append((start, end))
    return merge_intervals(intervals)


def retrieval_dedup_rate(
    hits: Sequence[CaptionHitV1 | Mapping[str, Any]],
) -> float:
    """Measure repeated Caption passages across one or more retrieval attempts."""
    if not hits:
        return 0.0
    identities = {_caption_hit_identity(hit) for hit in hits}
    return 1.0 - len(identities) / len(hits)


def anchor_consistency(
    document: WorkingDocument,
    supporting_claim_ids: Sequence[str],
    observation_rows: Sequence[Mapping[str, Any]],
) -> float:
    """Measure whether checkable claim anchors fall inside their cited attempts."""
    rows_by_attempt: dict[str, list[Mapping[str, Any]]] = {}
    for row in observation_rows:
        attempt_id = str(row.get("attempt_id", "") or "")
        if attempt_id:
            rows_by_attempt.setdefault(attempt_id, []).append(row)

    checkable = 0
    consistent = 0
    for claim_id in dict.fromkeys(str(item) for item in supporting_claim_ids if str(item)):
        claim = document.claims.get(claim_id)
        if claim is None or claim.status != "active":
            continue
        anchors = _claim_anchor_intervals(document, claim)
        cited_rows = tuple(
            row
            for attempt_id in _claim_attempt_ids(document, claim)
            for row in rows_by_attempt.get(attempt_id, ())
            if _attempt_is_supporting(row)
        )
        if not anchors or not cited_rows:
            continue
        checkable += 1
        if all(
            any(_anchor_overlaps_attempt(anchor, row) for row in cited_rows)
            for anchor in anchors
        ):
            consistent += 1
    return consistent / checkable if checkable else 1.0


def caption_hits_from_observation_rows(
    observation_rows: Sequence[Mapping[str, Any]],
) -> tuple[CaptionHitV1, ...]:
    hits: list[CaptionHitV1] = []
    seen: set[str] = set()
    for row in observation_rows:
        config = row.get("sampling_config", {})
        if not isinstance(config, Mapping) or config.get("mode") != "search_caption":
            continue
        for raw_hit in tuple(config.get("hits", ()) or ()):
            if not isinstance(raw_hit, Mapping):
                continue
            passage_id = str(raw_hit.get("passage_id", "") or "")
            interval = tuple(raw_hit.get("range", ()) or ())
            if not passage_id or passage_id in seen or len(interval) != 2:
                continue
            seen.add(passage_id)
            score = float(raw_hit.get("score", 0.0) or 0.0)
            hits.append(
                CaptionHitV1(
                    passage_id=passage_id,
                    caption_id=str(raw_hit.get("caption_id", "") or "unknown"),
                    rank=len(hits) + 1,
                    lexical_score=None,
                    dense_score=None,
                    fused_score=score,
                    virtual_start_sec=float(interval[0]),
                    virtual_end_sec=float(interval[1]),
                    wall_clock_begin=None,
                    wall_clock_end=None,
                    text="",
                    interval_precision=str(raw_hit.get("interval_precision", "unknown") or "unknown"),
                    source_pointer=str(raw_hit.get("source_pointer", "") or f"caption://unknown/{passage_id}"),
                    metadata={"observation_attempt_id": row.get("attempt_id", "")},
                )
            )
    return tuple(hits)


def caption_occurrence_candidates_from_observation_rows(
    observation_rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    candidates: dict[str, Mapping[str, Any]] = {}
    for row in observation_rows:
        config = row.get("sampling_config", {})
        if not isinstance(config, Mapping) or config.get("mode") != "search_caption":
            continue
        occurrence_set = config.get("occurrence_set")
        if not isinstance(occurrence_set, Mapping):
            continue
        for raw_candidate in tuple(occurrence_set.get("candidates", ()) or ()):
            if not isinstance(raw_candidate, Mapping):
                continue
            occurrence_id = str(raw_candidate.get("occurrence_id", "") or "")
            key = occurrence_id or json.dumps(
                [
                    list(raw_candidate.get("source_video_ids", ()) or ()),
                    list(raw_candidate.get("time_range", ()) or ()),
                    list(raw_candidate.get("passage_ids", ()) or ()),
                ],
                sort_keys=True,
                separators=(",", ":"),
            )
            candidates.setdefault(key, dict(raw_candidate))
    return tuple(candidates.values())


def agent_run_metrics(
    trace: Sequence[Mapping[str, Any]],
    observation_rows: Sequence[Mapping[str, Any]],
    *,
    answer_present: bool,
    reference_valid: bool,
    supporting_intervals: Sequence[Sequence[float]] = (),
) -> dict[str, float | int | None]:
    decisions = tuple(row for row in trace if row.get("type") == "reasoner_decision")
    decision_attempts = tuple(
        row for row in trace if row.get("type") == "reasoner_decision_attempt"
    )
    committed_decisions = tuple(
        row for row in decisions if row.get("semantic_committed", True)
    )
    batches = tuple(row for row in trace if row.get("type") == "investigator_batch")
    task_requests = tuple(row for row in trace if row.get("type") == "task_request")
    task_outcomes = tuple(row for row in trace if row.get("type") == "task_outcome")
    task_outcome_ids = {
        str(row.get("ledger_id", "") or "")
        for row in task_outcomes
        if row.get("status") in {"executed", "explicit_resolution_error"}
    }
    control_retries = tuple(row for row in trace if row.get("type") == "control_retry")
    closure_repairs = tuple(row for row in trace if row.get("type") == "closure_repair")
    occurrence_recovery_grants = tuple(
        row
        for row in trace
        if row.get("type") == "occurrence_recovery_round_granted"
    )
    answer_outcomes = tuple(row for row in trace if row.get("type") == "answer_outcome")
    ignored_state_ops = tuple(
        row for row in trace if row.get("type") == "runtime_state_ops_ignored"
    )
    evidence_plans = tuple(
        row for row in trace if row.get("type") == "runtime_evidence_plan"
    )
    obligation_summary = dict(
        answer_outcomes[-1].get("obligation_summary", {})
        if answer_outcomes
        and isinstance(answer_outcomes[-1].get("obligation_summary"), Mapping)
        else {}
    )
    temporal_scope_summary = dict(
        answer_outcomes[-1].get("temporal_scope_summary", {})
        if answer_outcomes
        and isinstance(answer_outcomes[-1].get("temporal_scope_summary"), Mapping)
        else {}
    )
    provenance_summary = dict(
        answer_outcomes[-1].get("provenance_summary", {})
        if answer_outcomes
        and isinstance(answer_outcomes[-1].get("provenance_summary"), Mapping)
        else {}
    )
    cue_summary = dict(
        answer_outcomes[-1].get("cue_summary", {})
        if answer_outcomes
        and isinstance(answer_outcomes[-1].get("cue_summary"), Mapping)
        else {}
    )
    closure_validation = dict(
        answer_outcomes[-1].get("closure_validation", {})
        if answer_outcomes
        and isinstance(answer_outcomes[-1].get("closure_validation"), Mapping)
        else {}
    )
    executed_ledger_ids = {
        str(row.get("ledger_id", "") or "")
        for row in task_outcomes
        if row.get("status") == "executed"
    }
    occurrence_request_ids = {
        str(row.get("ledger_id", "") or "")
        for row in task_requests
        if isinstance(row.get("task"), Mapping)
        and str(row["task"].get("occurrence_id", "") or "")
    }
    source_rows: dict[str, Mapping[str, Any]] = {}
    for row in observation_rows:
        source_rows.setdefault(str(row.get("attempt_id", "")), row)
    caption_rows = tuple(
        row
        for row in observation_rows
        if isinstance(row.get("sampling_config"), Mapping)
        and row["sampling_config"].get("mode") == "search_caption"
    )
    caption_material_attempt_count = len(
        {
            str(row.get("attempt_id", "") or "")
            for row in caption_rows
            if str(row.get("attempt_id", "") or "")
        }
    )
    candidates = caption_hits_from_observation_rows(observation_rows)
    raw_caption_hits = tuple(
        hit
        for row in caption_rows
        for hit in tuple(row.get("sampling_config", {}).get("hits", ()) or ())
        if isinstance(hit, Mapping)
    )
    caption_dedup_rate = retrieval_dedup_rate(raw_caption_hits)
    occurrence_candidates = caption_occurrence_candidates_from_observation_rows(
        observation_rows
    )
    occurrence_ambiguous_searches = sum(
        bool(occurrence_set.get("occurrence_ambiguous", False))
        for row in caption_rows
        if isinstance(
            (occurrence_set := row.get("sampling_config", {}).get("occurrence_set")),
            Mapping,
        )
    )
    visual_interpretation_rows = tuple(
        row
        for row in observation_rows
        if str(row.get("modality", "") or "").casefold() in {"visual", "ocr"}
    )
    visual_material_attempt_ids = {
        str(row.get("attempt_id", "") or "")
        for row in visual_interpretation_rows
        if str(row.get("attempt_id", "") or "")
    }
    bound_visual_attempt_ids = {
        str(row.get("attempt_id", "") or "")
        for row in visual_interpretation_rows
        if isinstance(row.get("sampling_config"), Mapping)
        and isinstance(row["sampling_config"].get("candidate_binding"), Mapping)
        and str(row.get("attempt_id", "") or "")
    }
    executed_occurrence_requests = occurrence_request_ids.intersection(executed_ledger_ids)
    seen_visual_attempts: set[str] = set()
    deliberate_reinterpretations = 0
    accidental_reinterpretations = 0
    for row in visual_interpretation_rows:
        attempt_id = str(row.get("attempt_id", "") or "")
        if not attempt_id or attempt_id not in seen_visual_attempts:
            if attempt_id:
                seen_visual_attempts.add(attempt_id)
            continue
        purpose = str(row.get("interpretation_purpose", "primary") or "primary").casefold()
        if purpose in {"deliberate_arbitration", "cue_verification", "manual_reread"}:
            deliberate_reinterpretations += 1
        else:
            accidental_reinterpretations += 1
    purpose_counts = {
        purpose: sum(
            str(row.get("interpretation_purpose", "primary") or "primary").casefold()
            == purpose
            for row in visual_interpretation_rows
        )
        for purpose in (
            "primary",
            "deliberate_arbitration",
            "cue_verification",
            "control_retry",
            "manual_reread",
        )
    }
    refinement_rows = tuple(
        row
        for row in visual_interpretation_rows
        if isinstance(row.get("sampling_config"), Mapping)
        and isinstance(row["sampling_config"].get("refinement_binding"), Mapping)
    )
    child_refinement_rows = tuple(
        row
        for row in refinement_rows
        if str(row["sampling_config"]["refinement_binding"].get("stage", "") or "")
        == "child_refinement"
    )
    refined_cue_ids = {
        str(row["sampling_config"]["refinement_binding"].get("cue_id", "") or "")
        for row in child_refinement_rows
        if str(row["sampling_config"]["refinement_binding"].get("cue_id", "") or "")
    }
    false_refinements = sum(
        str(row["sampling_config"]["refinement_binding"].get("cue_status", "") or "")
        != "verified"
        for row in child_refinement_rows
    )
    converted = sum(
        any(_overlap(_hit_interval(hit), interval) for interval in supporting_intervals)
        for hit in candidates
    )
    report_outcomes = tuple(
        outcome
        for batch in batches
        for outcome in tuple(batch.get("outcomes", ()) or ())
        if isinstance(outcome, Mapping)
    )
    visual_frame_count = sum(
        len(tuple(row.get("frame_times", ()) or ()))
        for row in source_rows.values()
        if str(row.get("modality", "") or "").casefold() in {"visual", "ocr"}
    )
    observed_case = bool(visual_interpretation_rows)
    malformed_decision_count = sum(
        not bool(row.get("schema_valid", False)) for row in decision_attempts
    )
    return {
        "answer_rate": float(bool(answer_present)),
        "reference_valid_rate": float(bool(reference_valid)),
        "observed_case_rate": float(observed_case),
        "conditional_visual_frames": visual_frame_count if observed_case else None,
        "reasoner_decision_attempt_count": len(decision_attempts),
        "malformed_decision_count": malformed_decision_count,
        "malformed_decision_rate": (
            malformed_decision_count / len(decision_attempts)
            if decision_attempts
            else 0.0
        ),
        "rounds": len(
            {
                int(row.get("semantic_round", row.get("round", index)) or index)
                for index, row in enumerate(committed_decisions, start=1)
            }
        ),
        "semantic_rounds_used": len(
            {
                int(row.get("semantic_round", row.get("round", index)) or index)
                for index, row in enumerate(
                    decision_attempts or decisions, start=1
                )
            }
        ),
        "forced_finalize_round": min(
            (
                int(row.get("semantic_round", row.get("round", 0)) or 0)
                for row in decision_attempts or decisions
                if row.get("force_finalize")
            ),
            default=None,
        ),
        "extra_rounds_granted": sum(
            max(0, int(row.get("count", 1) or 0))
            for row in occurrence_recovery_grants
        ),
        "control_retry_count": sum(
            max(0, int(row.get("count", 1) or 0)) for row in control_retries
        ),
        "decision_repair_count": sum(
            max(0, int(row.get("count", 1) or 0)) for row in control_retries
        ),
        "state_mutation_op_count": sum(
            int(row.get("state_mutation_op_count", 0) or 0) for row in decisions
        )
        + sum(len(tuple(row.get("operations", ()) or ())) for row in ignored_state_ops),
        "prompt_schema_token_cost": sum(
            int(row.get("prompt_schema_token_cost", 0) or 0) for row in decisions
        )
        + sum(
            int(row.get("prompt_schema_token_cost", 0) or 0)
            for row in evidence_plans
        ),
        "closure_repair_count": sum(
            max(0, int(row.get("count", 1) or 0)) for row in closure_repairs
        ),
        "reference_integrity_ok": float(
            bool(closure_validation.get("reference_integrity_ok", False))
        ),
        "material_support_ok": float(
            bool(closure_validation.get("material_support_ok", False))
        ),
        "provenance_binding_ok": float(
            bool(closure_validation.get("provenance_binding_ok", False))
        ),
        "temporal_consistency_ok": float(
            bool(closure_validation.get("temporal_consistency_ok", False))
        ),
        "occurrence_binding_ok": float(
            bool(closure_validation.get("occurrence_binding_ok", False))
        ),
        "obligation_coverage_ok": float(
            bool(closure_validation.get("obligation_coverage_ok", False))
        ),
        "observation_consumption_ok": float(
            bool(closure_validation.get("observation_consumption_ok", False))
        ),
        "evidence_kind_requirements_ok": float(
            bool(closure_validation.get("evidence_kind_requirements_ok", False))
        ),
        "requested_acquisition_count": len(task_requests),
        "executed_acquisition_count": sum(
            row.get("status") == "executed" for row in task_outcomes
        ),
        "task_resolution_error_count": sum(
            row.get("status") == "explicit_resolution_error" for row in task_outcomes
        ),
        "silently_dropped_acquisition_count": sum(
            str(row.get("ledger_id", "") or "") not in task_outcome_ids
            for row in task_requests
        ),
        "answer_bearing_obligation_count": int(
            obligation_summary.get("answer_bearing_obligation_count", 0) or 0
        ),
        "satisfied_obligation_count": int(
            obligation_summary.get("satisfied_obligation_count", 0) or 0
        ),
        "open_obligation_count_at_answer": int(
            obligation_summary.get("open_obligation_count_at_answer", 0) or 0
        ),
        "unresolved_obligation_count_at_answer": int(
            obligation_summary.get("unresolved_obligation_count_at_answer", 0) or 0
        ),
        "obligation_coverage_rate": float(
            obligation_summary.get("obligation_coverage_rate", 0.0) or 0.0
        ),
        "observation_claim_count": int(
            provenance_summary.get("observation_claim_count", 0) or 0
        ),
        "item_bound_observation_claim_count": int(
            provenance_summary.get("item_bound_observation_claim_count", 0) or 0
        ),
        "observation_claim_item_binding_rate": float(
            provenance_summary.get("observation_claim_item_binding_rate", 0.0) or 0.0
        ),
        "dangling_interpretation_item_count": int(
            provenance_summary.get("dangling_interpretation_item_count", 0) or 0
        ),
        "dangling_interpretation_item_rate": float(
            provenance_summary.get("dangling_interpretation_item_rate", 0.0) or 0.0
        ),
        "observation_cue_count": int(cue_summary.get("observation_cue_count", 0) or 0),
        "verified_cue_count": int(cue_summary.get("verified_cue_count", 0) or 0),
        "rejected_cue_count": int(cue_summary.get("rejected_cue_count", 0) or 0),
        "unverified_cue_count": int(cue_summary.get("unverified_cue_count", 0) or 0),
        "cue_verification_rate": float(cue_summary.get("cue_verification_rate", 0.0) or 0.0),
        "cue_rejection_rate": float(cue_summary.get("cue_rejection_rate", 0.0) or 0.0),
        "candidate_to_refined_material_rate": (
            len(refined_cue_ids)
            / int(cue_summary.get("observation_cue_count", 0) or 0)
            if int(cue_summary.get("observation_cue_count", 0) or 0)
            else 0.0
        ),
        "false_refinement_rate": (
            false_refinements / len(child_refinement_rows)
            if child_refinement_rows
            else 0.0
        ),
        "occurrence_bound_material_count": len(bound_visual_attempt_ids),
        "occurrence_binding_rate": (
            min(1.0, len(bound_visual_attempt_ids) / len(executed_occurrence_requests))
            if executed_occurrence_requests
            else 0.0
        ),
        "temporal_scope_count": int(
            temporal_scope_summary.get("temporal_scope_count", 0) or 0
        ),
        "resolved_temporal_scope_count": int(
            temporal_scope_summary.get("resolved_temporal_scope_count", 0) or 0
        ),
        "temporal_scope_resolved_rate": float(
            temporal_scope_summary.get("temporal_scope_resolved_rate", 0.0) or 0.0
        ),
        "dedicated_read_rounds": sum(
            row.get("action") == "read_observations" for row in committed_decisions
        ),
        "caption_searches": len(caption_rows),
        "caption_material_attempts": caption_material_attempt_count,
        "empty_search_count": sum(
            not tuple(row.get("sampling_config", {}).get("hits", ()) or ())
            for row in caption_rows
        ),
        "duplicate_search_count": sum(bool(row.get("reused")) for row in report_outcomes),
        "caption_result_set_reuse_count": sum(
            row.get("failure_reason") == "caption_result_set_has_no_new_material"
            for row in report_outcomes
        ),
        "caption_result_novelty_rate": 1.0 - caption_dedup_rate if raw_caption_hits else 0.0,
        "caption_result_dedup_rate": caption_dedup_rate,
        "caption_occurrence_candidate_count": len(occurrence_candidates),
        "caption_occurrence_ambiguous_search_count": occurrence_ambiguous_searches,
        "unique_visual_material_attempts": len(visual_material_attempt_ids),
        "visual_interpretation_count": len(visual_interpretation_rows),
        "visual_reinterpretation_count": max(
            0,
            len(visual_interpretation_rows) - len(visual_material_attempt_ids),
        ),
        "deliberate_reinterpretation_count": deliberate_reinterpretations,
        "accidental_reinterpretation_count": accidental_reinterpretations,
        "primary_interpretation_count": purpose_counts["primary"],
        "deliberate_arbitration_interpretation_count": purpose_counts[
            "deliberate_arbitration"
        ],
        "cue_verification_interpretation_count": purpose_counts["cue_verification"],
        "control_retry_interpretation_count": purpose_counts["control_retry"],
        "manual_reread_interpretation_count": purpose_counts["manual_reread"],
        "visual_confirmations": sum(
            str(row.get("modality", "") or "").casefold() in {"visual", "ocr"}
            for row in source_rows.values()
        ),
        "visual_frames_inspected": visual_frame_count,
        "candidate_to_support_conversion": converted / len(candidates) if candidates else 0.0,
    }


def _claim_attempt_ids(document: WorkingDocument, claim: Claim) -> tuple[str, ...]:
    attempt_ids: list[str] = list(claim.cites)
    visited: set[str] = set()

    def visit(claim_id: str) -> None:
        if claim_id in visited:
            return
        visited.add(claim_id)
        parent = document.claims.get(claim_id)
        if parent is None:
            return
        attempt_ids.extend(parent.cites)
        for ancestor_id in parent.derived_from:
            visit(ancestor_id)

    for parent_id in claim.derived_from:
        visit(parent_id)
    return tuple(dict.fromkeys(attempt_ids))


def _claim_anchor_intervals(
    document: WorkingDocument,
    claim: Claim,
) -> tuple[tuple[float, float], ...]:
    anchors: list[tuple[float, float]] = []
    visited: set[str] = set()

    def visit(current: Claim) -> None:
        if current.claim_id in visited or current.status != "active":
            return
        visited.add(current.claim_id)
        if current.time_anchor is not None:
            anchors.append(current.time_anchor)
            return
        for parent_id in current.derived_from:
            parent = document.claims.get(parent_id)
            if parent is not None:
                visit(parent)

    visit(claim)
    return tuple(anchors)


def _attempt_is_supporting(row: Mapping[str, Any]) -> bool:
    modality = str(row.get("modality", "") or "").casefold()
    role = str(row.get("evidence_role", row.get("interval_role", "supporting")) or "supporting").casefold()
    return modality != "caption_search" and role not in {"candidate", "negative"}


def _caption_hit_identity(
    hit: CaptionHitV1 | Mapping[str, Any],
) -> tuple[str, float, float, str]:
    if isinstance(hit, CaptionHitV1):
        return (
            str(hit.passage_id),
            float(hit.virtual_start_sec),
            float(hit.virtual_end_sec),
            str(hit.text or ""),
        )
    raw_range = tuple(hit.get("range", ()) or ())
    if len(raw_range) == 2:
        start_sec, end_sec = float(raw_range[0]), float(raw_range[1])
    else:
        start_sec = float(hit.get("virtual_start_sec", 0.0) or 0.0)
        end_sec = float(hit.get("virtual_end_sec", start_sec) or start_sec)
    return (
        str(hit.get("passage_id", "") or ""),
        start_sec,
        end_sec,
        str(hit.get("text", hit.get("caption_excerpt", "")) or ""),
    )


def _anchor_overlaps_attempt(
    anchor: Sequence[float],
    row: Mapping[str, Any],
) -> bool:
    config = row.get("sampling_config", {})
    manifest = config.get("sampling_manifest", {}) if isinstance(config, Mapping) else {}
    effective_fps = (
        float(manifest.get("effective_fps", 0.0) or 0.0)
        if isinstance(manifest, Mapping)
        else 0.0
    )
    fallback_fps = float(row.get("sampling_fps", 0.0) or 0.0)
    observed_fps = effective_fps or fallback_fps
    tolerance_sec = 1.0 / observed_fps if observed_fps > 0.0 else 0.0
    anchor_start, anchor_end = float(anchor[0]), float(anchor[1])
    return any(
        min(anchor_end, float(item[1]) + tolerance_sec)
        >= max(anchor_start, float(item[0]) - tolerance_sec)
        for item in tuple(row.get("inspected_ranges", ()) or ())
        if len(item) == 2
    )


def _hit_interval(hit: CaptionHitV1) -> tuple[float, float]:
    return float(hit.virtual_start_sec), float(hit.virtual_end_sec)


def _overlap(left: Sequence[float], right: Sequence[float]) -> bool:
    return min(float(left[1]), float(right[1])) > max(float(left[0]), float(right[0]))
