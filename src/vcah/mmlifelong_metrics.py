from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from statistics import mean
from typing import Any, Callable, Mapping, Sequence

from vcah.caption_schema import CaptionHitV1
from vcah.virtual_video import sampling_fidelity
from vcah.workspace import Claim, WorkingDocument


DEFAULT_REF_BUCKETS = (60, 300, 600)
DEFAULT_RETRIEVAL_KS = (1, 5, 10, 20)


def smooth_score(score: int) -> float:
    value = int(score)
    if value < 0 or value > 5:
        raise ValueError("judge score must be between 0 and 5")
    if value in (4, 5):
        return 1.0
    if value == 3:
        return 0.5
    return 0.0


def bins_for_interval(start: float, end: float, bucket_size: int) -> set[int]:
    bucket = int(bucket_size)
    if bucket <= 0:
        raise ValueError("bucket_size must be positive")
    start_value = float(start)
    end_value = float(end)
    if end_value <= start_value:
        return set()
    first = int(start_value // bucket)
    last = int((end_value - 1e-9) // bucket)
    return set(range(first, last + 1))


def ref_score(
    predicted_intervals: Sequence[Sequence[float]],
    gold_intervals: Sequence[Sequence[float]],
    *,
    bucket_size: int,
) -> float:
    predicted_bins = _interval_bins(predicted_intervals, bucket_size)
    gold_bins = _interval_bins(gold_intervals, bucket_size)
    union = predicted_bins | gold_bins
    if not union:
        return 0.0
    return 100.0 * len(predicted_bins & gold_bins) / len(union)


def ref_scores(
    predicted_intervals: Sequence[Sequence[float]],
    gold_intervals: Sequence[Sequence[float]],
    *,
    bucket_sizes: Sequence[int] = DEFAULT_REF_BUCKETS,
) -> dict[str, float]:
    return {
        f"Ref@{int(bucket)}": ref_score(
            predicted_intervals,
            gold_intervals,
            bucket_size=int(bucket),
        )
        for bucket in bucket_sizes
    }


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


def retrieval_metrics(
    hits: Sequence[CaptionHitV1],
    gold_clue_intervals: Sequence[Sequence[float]],
    *,
    ks: Sequence[int] = DEFAULT_RETRIEVAL_KS,
) -> dict[str, float | int | None]:
    clues = merge_intervals(gold_clue_intervals)
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
        covered = tuple(any(_overlap(candidate, clue) for candidate in predicted) for clue in clues)
        metrics[f"ClueRecall@{k}"] = float(any(covered))
        metrics[f"AllCluesRecall@{k}"] = float(bool(clues) and all(covered))
        metrics[f"ClueIoU@{k}"] = interval_iou(predicted, clues)
    return metrics


def interval_iou(
    predicted_intervals: Sequence[Sequence[float]],
    gold_intervals: Sequence[Sequence[float]],
) -> float:
    predicted = merge_intervals(predicted_intervals)
    gold = merge_intervals(gold_intervals)
    intersection = sum(
        max(0.0, min(pred_end, gold_end) - max(pred_start, gold_start))
        for pred_start, pred_end in predicted
        for gold_start, gold_end in gold
    )
    predicted_duration = sum(end - start for start, end in predicted)
    gold_duration = sum(end - start for start, end in gold)
    union = predicted_duration + gold_duration - intersection
    return intersection / union if union > 0.0 else 0.0


def clue_frame_coverage(
    frame_times: Sequence[float],
    gold_intervals: Sequence[Sequence[float]],
) -> float:
    """Return the fraction of official clue intervals touched by an actual frame."""
    clues = tuple(
        (float(item[0]), float(item[1]))
        for item in gold_intervals
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


def recorded_case_diagnostics(
    evaluation: Mapping[str, Any],
    document: WorkingDocument,
    observation_rows: Sequence[Mapping[str, Any]],
    *,
    supporting_claim_ids: Sequence[str],
    gold_intervals: Sequence[Sequence[float]],
) -> dict[str, float | int]:
    """Recompute deterministic diagnostics from one recorded case directory."""
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
        for row in source_rows.values()
        if isinstance((config := row.get("sampling_config")), Mapping)
        and config.get("mode") == "search_caption"
        for hit in tuple(config.get("hits", ()) or ())
        if isinstance(hit, Mapping)
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
    agent = evaluation.get("agent", {})
    agent_metrics = agent if isinstance(agent, Mapping) else {}
    return {
        "accuracy_score": float(evaluation.get("accuracy_score") or 0.0),
        "reference_valid": int(bool(evaluation.get("reference_valid"))),
        "visual_frames_inspected": int(
            agent_metrics.get("visual_frames_inspected", len(frame_times)) or 0
        ),
        "clue_frame_coverage": clue_frame_coverage(frame_times, gold_intervals),
        "retrieval_dedup_rate": retrieval_dedup_rate(caption_hits),
        "sampling_fidelity_mean": mean(fidelities) if fidelities else 0.0,
        "sampling_fidelity_min": min(fidelities) if fidelities else 0.0,
        "anchor_consistency": anchor_consistency(
            document,
            supporting_claim_ids,
            tuple(source_rows.values()),
        ),
    }


@dataclass(frozen=True)
class AnswerJudgeResult:
    raw_score: int | None
    smoothed_score: float
    rationale: str
    judge_model: str
    prompt_digest: str
    retry_count: int
    parse_status: str
    raw_response: str = ""
    response_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "response_metadata", dict(self.response_metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_answer_judge_response(
    response: str,
    *,
    judge_model: str,
    prompt: str,
    retry_count: int = 0,
    response_metadata: Mapping[str, Any] | None = None,
) -> AnswerJudgeResult:
    raw = str(response or "")
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:20]
    try:
        payload = _json_object(raw)
        score = int(payload["score"])
        rationale = str(payload.get("rationale", "") or "").strip()
        return AnswerJudgeResult(
            raw_score=score,
            smoothed_score=smooth_score(score),
            rationale=rationale,
            judge_model=str(judge_model),
            prompt_digest=digest,
            retry_count=max(0, int(retry_count)),
            parse_status="parsed",
            raw_response=raw,
            response_metadata=dict(response_metadata or {}),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return AnswerJudgeResult(
            raw_score=None,
            smoothed_score=0.0,
            rationale="",
            judge_model=str(judge_model),
            prompt_digest=digest,
            retry_count=max(0, int(retry_count)),
            parse_status="failed",
            raw_response=raw,
            response_metadata=dict(response_metadata or {}),
        )


def judge_free_form_answer(
    generate: Callable[[str], str],
    *,
    question: str,
    reference_answer: str,
    predicted_answer: str,
    judge_model: str,
    max_retries: int = 2,
    response_metadata: Callable[[], Mapping[str, Any]] | None = None,
) -> AnswerJudgeResult:
    prompt = answer_judge_prompt(
        question=question,
        reference_answer=reference_answer,
        predicted_answer=predicted_answer,
    )
    last = parse_answer_judge_response(
        "",
        judge_model=judge_model,
        prompt=prompt,
        response_metadata={},
    )
    for attempt in range(max(0, int(max_retries)) + 1):
        try:
            raw = generate(prompt)
            metadata = dict(response_metadata() if response_metadata else {})
        except Exception as exc:
            raw = ""
            metadata = {"error_type": type(exc).__name__}
        last = parse_answer_judge_response(
            raw,
            judge_model=judge_model,
            prompt=prompt,
            retry_count=attempt,
            response_metadata=metadata,
        )
        if last.parse_status == "parsed":
            return last
    return last


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


def agent_run_metrics(
    trace: Sequence[Mapping[str, Any]],
    observation_rows: Sequence[Mapping[str, Any]],
    *,
    answer_present: bool,
    reference_valid: bool,
    supporting_intervals: Sequence[Sequence[float]] = (),
) -> dict[str, float | int]:
    decisions = tuple(row for row in trace if row.get("type") == "reasoner_decision")
    batches = tuple(row for row in trace if row.get("type") == "investigator_batch")
    source_rows: dict[str, Mapping[str, Any]] = {}
    for row in observation_rows:
        source_rows.setdefault(str(row.get("attempt_id", "")), row)
    caption_rows = tuple(
        row
        for row in source_rows.values()
        if isinstance(row.get("sampling_config"), Mapping)
        and row["sampling_config"].get("mode") == "search_caption"
    )
    candidates = caption_hits_from_observation_rows(tuple(source_rows.values()))
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
    return {
        "answer_rate": float(bool(answer_present)),
        "reference_valid_rate": float(bool(reference_valid)),
        "rounds": len(decisions),
        "dedicated_read_rounds": sum(row.get("action") == "read_observations" for row in decisions),
        "caption_searches": len(caption_rows),
        "empty_search_count": sum(
            not tuple(row.get("sampling_config", {}).get("hits", ()) or ())
            for row in caption_rows
        ),
        "duplicate_search_count": sum(bool(row.get("reused")) for row in report_outcomes),
        "visual_confirmations": sum(
            str(row.get("modality", "") or "").casefold() in {"visual", "ocr"}
            for row in source_rows.values()
        ),
        "visual_frames_inspected": sum(
            len(tuple(row.get("frame_times", ()) or ()))
            for row in source_rows.values()
            if str(row.get("modality", "") or "").casefold() in {"visual", "ocr"}
        ),
        "candidate_to_support_conversion": converted / len(candidates) if candidates else 0.0,
    }


def answer_judge_prompt(*, question: str, reference_answer: str, predicted_answer: str) -> str:
    return (
        "Score the predicted answer against the reference for semantic correctness on a 0-5 scale. "
        "Return JSON only as {\"score\": 0, \"rationale\": \"brief reason\"}. "
        "A score of 5 is fully correct, 4 is correct with a minor omission, 3 is partially correct, and 0-2 is incorrect.\n"
        f"Question: {question}\nReference answer: {reference_answer}\nPredicted answer: {predicted_answer}"
    )


def _interval_bins(intervals: Sequence[Sequence[float]], bucket_size: int) -> set[int]:
    bins: set[int] = set()
    for item in intervals:
        if len(item) == 2:
            bins.update(bins_for_interval(float(item[0]), float(item[1]), bucket_size))
    return bins


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


def _json_object(value: str) -> Mapping[str, Any]:
    stripped = value.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        stripped = stripped.rsplit("```", 1)[0].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise json.JSONDecodeError("missing JSON object", stripped, 0)
    payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, Mapping):
        raise TypeError("judge response must be an object")
    return payload
