from __future__ import annotations

from statistics import mean
from typing import Any, Mapping, Sequence

from vcah.caption_schema import CaptionHitV1
from vcah.runtime_metrics import (
    anchor_consistency,
    caption_occurrence_candidates_from_observation_rows,
    merge_intervals,
    retrieval_dedup_rate,
)
from vcah.virtual_video import sampling_fidelity
from vcah.workspace import WorkingDocument


DEFAULT_REF_BUCKETS = (60, 300, 600)
DEFAULT_RETRIEVAL_KS = (1, 5, 10, 20)


def bins_for_interval(
    start: float,
    end: float,
    bucket_size: int,
    *,
    total_seconds: float,
) -> set[int]:
    bucket = int(bucket_size)
    if bucket <= 0:
        raise ValueError("bucket_size must be positive")
    duration = float(total_seconds)
    if duration < 0.0:
        raise ValueError("total_seconds must be non-negative")
    start_value = max(0.0, float(start))
    end_value = min(duration, float(end))
    if start_value >= end_value:
        return set()
    first = int(start_value // bucket)
    last = int((end_value - 1e-9) // bucket)
    return set(range(first, last + 1))


def ref_score(
    predicted_intervals: Sequence[Sequence[float]],
    reference_intervals: Sequence[Sequence[float]],
    *,
    total_seconds: float,
    bucket_size: int,
) -> float:
    """Wrap the vendored eval_ref.Ref_N semantics without modifying upstream."""
    predicted_bins = _interval_bins(
        predicted_intervals,
        bucket_size,
        total_seconds=total_seconds,
    )
    reference_bins = _interval_bins(
        reference_intervals,
        bucket_size,
        total_seconds=total_seconds,
    )
    union = predicted_bins | reference_bins
    if not union:
        return 0.0
    return len(predicted_bins & reference_bins) / len(union)


def ref_scores(
    predicted_intervals: Sequence[Sequence[float]],
    reference_intervals: Sequence[Sequence[float]],
    *,
    total_seconds: float,
    bucket_sizes: Sequence[int] = DEFAULT_REF_BUCKETS,
) -> dict[str, float]:
    return {
        f"ref_{int(bucket)}": ref_score(
            predicted_intervals,
            reference_intervals,
            total_seconds=total_seconds,
            bucket_size=int(bucket),
        )
        for bucket in bucket_sizes
    }


def retrieval_metrics(
    hits: Sequence[CaptionHitV1],
    reference_intervals: Sequence[Sequence[float]],
    *,
    ks: Sequence[int] = DEFAULT_RETRIEVAL_KS,
) -> dict[str, float | int | None]:
    clues = merge_intervals(reference_intervals)
    first_rank = next(
        (
            rank
            for rank, hit in enumerate(hits, start=1)
            if any(_overlap(_hit_interval(hit), clue) for clue in clues)
        ),
        None,
    )
    metrics: dict[str, float | int | None] = {"FirstClueRank": first_rank}
    for raw_k in ks:
        k = max(1, int(raw_k))
        predicted = merge_intervals(_hit_interval(hit) for hit in hits[:k])
        covered = tuple(
            any(_overlap(candidate, clue) for candidate in predicted)
            for clue in clues
        )
        metrics[f"ClueRecall@{k}"] = float(any(covered))
        metrics[f"AllCluesRecall@{k}"] = float(bool(clues) and all(covered))
        metrics[f"ClueIoU@{k}"] = interval_iou(predicted, clues)
    return metrics


def interval_iou(
    predicted_intervals: Sequence[Sequence[float]],
    reference_intervals: Sequence[Sequence[float]],
) -> float:
    predicted = merge_intervals(predicted_intervals)
    reference = merge_intervals(reference_intervals)
    intersection = sum(
        max(0.0, min(pred_end, ref_end) - max(pred_start, ref_start))
        for pred_start, pred_end in predicted
        for ref_start, ref_end in reference
    )
    predicted_duration = sum(end - start for start, end in predicted)
    reference_duration = sum(end - start for start, end in reference)
    union = predicted_duration + reference_duration - intersection
    return intersection / union if union > 0.0 else 0.0


def clue_frame_coverage(
    frame_times: Sequence[float],
    reference_intervals: Sequence[Sequence[float]],
) -> float:
    clues = tuple(
        (float(item[0]), float(item[1]))
        for item in reference_intervals
        if len(item) == 2 and float(item[1]) >= float(item[0])
    )
    if not clues:
        return 0.0
    times = tuple(sorted({float(value) for value in frame_times}))
    covered = sum(
        any(start_sec <= time_sec <= end_sec for time_sec in times)
        for start_sec, end_sec in clues
    )
    return covered / len(clues)


def candidate_clue_recall(
    observation_rows: Sequence[Mapping[str, Any]],
    reference_intervals: Sequence[Sequence[float]],
    *,
    tolerance_sec: float = 5.0,
) -> float:
    candidate_intervals = tuple(
        interval
        for row in observation_rows
        if isinstance((config := row.get("sampling_config")), Mapping)
        and config.get("mode") == "search_caption"
        for hit in tuple(config.get("hits", ()) or ())
        if isinstance(hit, Mapping)
        and (interval := _hit_interval(hit)) is not None
    )
    return _interval_recall(
        candidate_intervals,
        reference_intervals,
        tolerance_sec=tolerance_sec,
    )


def occurrence_candidate_recall(
    observation_rows: Sequence[Mapping[str, Any]],
    reference_intervals: Sequence[Sequence[float]],
    *,
    tolerance_sec: float = 5.0,
) -> float:
    candidate_intervals = tuple(
        interval
        for candidate in caption_occurrence_candidates_from_observation_rows(
            observation_rows
        )
        if (interval := _interval(candidate.get("time_range", ()))) is not None
    )
    return _interval_recall(
        candidate_intervals,
        reference_intervals,
        tolerance_sec=tolerance_sec,
    )


def recorded_case_diagnostics(
    evaluation: Mapping[str, Any],
    document: WorkingDocument,
    observation_rows: Sequence[Mapping[str, Any]],
    *,
    runtime_summary: Mapping[str, Any] | None = None,
    supporting_claim_ids: Sequence[str],
    reference_intervals: Sequence[Sequence[float]],
) -> dict[str, float | int]:
    source_rows: dict[str, Mapping[str, Any]] = {}
    for row in observation_rows:
        attempt_id = str(row.get("attempt_id", "") or "")
        if attempt_id:
            source_rows.setdefault(attempt_id, row)
    visual_rows = tuple(
        row
        for row in source_rows.values()
        if str(row.get("modality", "") or "").casefold() in {"visual", "ocr"}
    )
    frame_times = tuple(
        float(value)
        for row in visual_rows
        for value in tuple(row.get("frame_times", ()) or ())
    )
    caption_hits = tuple(
        hit
        for row in observation_rows
        if isinstance((config := row.get("sampling_config")), Mapping)
        and config.get("mode") == "search_caption"
        for hit in tuple(config.get("hits", ()) or ())
        if isinstance(hit, Mapping)
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
    occurrence_candidates = caption_occurrence_candidates_from_observation_rows(
        observation_rows
    )
    occurrence_ambiguous_searches = sum(
        bool(occurrence_set.get("occurrence_ambiguous", False))
        for row in observation_rows
        if isinstance((config := row.get("sampling_config")), Mapping)
        and config.get("mode") == "search_caption"
        and isinstance((occurrence_set := config.get("occurrence_set")), Mapping)
    )
    fidelities = tuple(
        sampling_fidelity(
            float(row.get("sampling_fps", 0.0) or 0.0),
            tuple(row.get("frame_times", ()) or ()),
            tuple(row.get("requested_range", ()) or ()),
        )
        for row in visual_rows
        if float(row.get("sampling_fps", 0.0) or 0.0) > 0.0
    )
    answer = evaluation.get("answer", {})
    answer_metrics = answer if isinstance(answer, Mapping) else {}
    runtime_payload = runtime_summary or {}
    runtime = runtime_payload.get(
        "runtime_metrics",
        runtime_payload.get(
            "agent",
            evaluation.get("runtime_metrics", evaluation.get("agent", {})),
        ),
    )
    runtime_metrics = runtime if isinstance(runtime, Mapping) else {}
    reference_valid = runtime_payload.get(
        "reference_valid",
        evaluation.get("reference_valid"),
    )
    caption_dedup_rate = retrieval_dedup_rate(caption_hits)
    return {
        "official_accuracy": float(answer_metrics.get("score") or 0.0),
        "reference_valid": int(bool(reference_valid)),
        "visual_frames_inspected": int(
            runtime_metrics.get("visual_frames_inspected", len(frame_times)) or 0
        ),
        "clue_frame_coverage": clue_frame_coverage(frame_times, reference_intervals),
        "candidate_clue_recall": candidate_clue_recall(
            observation_rows,
            reference_intervals,
        ),
        "occurrence_candidate_recall": occurrence_candidate_recall(
            observation_rows,
            reference_intervals,
        ),
        "retrieval_dedup_rate": caption_dedup_rate,
        "caption_result_novelty_rate": 1.0 - caption_dedup_rate if caption_hits else 0.0,
        "caption_occurrence_candidate_count": len(occurrence_candidates),
        "caption_occurrence_ambiguous_search_count": occurrence_ambiguous_searches,
        "unique_visual_material_attempts": len(visual_material_attempt_ids),
        "visual_interpretation_count": len(visual_interpretation_rows),
        "visual_reinterpretation_count": max(
            0,
            len(visual_interpretation_rows) - len(visual_material_attempt_ids),
        ),
        "sampling_fidelity_mean": mean(fidelities) if fidelities else 0.0,
        "sampling_fidelity_min": min(fidelities) if fidelities else 0.0,
        "anchor_consistency": anchor_consistency(
            document,
            supporting_claim_ids,
            tuple(source_rows.values()),
        ),
    }


def _interval_bins(
    intervals: Sequence[Sequence[float]],
    bucket_size: int,
    *,
    total_seconds: float,
) -> set[int]:
    bins: set[int] = set()
    for item in intervals:
        if len(item) == 2:
            bins.update(
                bins_for_interval(
                    float(item[0]),
                    float(item[1]),
                    bucket_size,
                    total_seconds=total_seconds,
                )
            )
    return bins


def _interval_recall(
    candidate_intervals: Sequence[Sequence[float]],
    reference_intervals: Sequence[Sequence[float]],
    *,
    tolerance_sec: float,
) -> float:
    candidates = tuple(
        interval
        for value in candidate_intervals
        if (interval := _interval(value)) is not None
    )
    references = tuple(
        interval
        for value in reference_intervals
        if (interval := _interval(value)) is not None
    )
    if not references:
        return 0.0
    tolerance = max(0.0, float(tolerance_sec))
    recalled = sum(
        any(
            min(candidate[1] + tolerance, reference[1])
            >= max(candidate[0] - tolerance, reference[0])
            for candidate in candidates
        )
        for reference in references
    )
    return recalled / len(references)


def _interval(value: object) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return None
    try:
        start, end = sorted((float(value[0]), float(value[1])))
    except (TypeError, ValueError):
        return None
    return (start, end) if end >= start else None


def _hit_interval(hit: CaptionHitV1 | Mapping[str, Any]) -> tuple[float, float]:
    if isinstance(hit, Mapping):
        raw_range = tuple(hit.get("range", ()) or ())
        if len(raw_range) == 2:
            start, end = float(raw_range[0]), float(raw_range[1])
        else:
            start = float(hit.get("virtual_start_sec", 0.0) or 0.0)
            end = float(hit.get("virtual_end_sec", start) or start)
        return min(start, end), max(start, end)
    return float(hit.virtual_start_sec), float(hit.virtual_end_sec)


def _overlap(left: Sequence[float], right: Sequence[float]) -> bool:
    return min(float(left[1]), float(right[1])) > max(float(left[0]), float(right[0]))
