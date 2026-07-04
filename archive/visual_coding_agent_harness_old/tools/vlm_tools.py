"""VLM-backed explore and verify tools for multi_v3 investigators."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Sequence

from visual_coding_agent_harness.backends.base import BackendRequest, VisionLanguageBackend
from visual_coding_agent_harness.contracts.query import ScopedQuery
from visual_coding_agent_harness.contracts.report import CandidateShot, Finding, VerifyRequest
from visual_coding_agent_harness.video.pipeline import is_image_path
from visual_coding_agent_harness.workspace.video_workspace import Beat, VideoWorkspace
from visual_coding_agent_harness.workspace.visual_index import BeatHit


class ExploreResult(tuple):
    """Candidate shots plus execution metadata for cost accounting."""

    batch_count: int
    degraded: bool

    def __new__(
        cls,
        items: Iterable[CandidateShot] = (),
        *,
        batch_count: int = 0,
        degraded: bool = False,
    ) -> "ExploreResult":
        obj = tuple.__new__(cls, tuple(items))
        obj.batch_count = max(0, int(batch_count))
        obj.degraded = bool(degraded)
        return obj


def explore(
    *,
    query: ScopedQuery,
    workspace: VideoWorkspace,
    backend: VisionLanguageBackend,
    batch_size: int = 16,
) -> ExploreResult:
    beats = workspace.beats_in_chapters(query.scope.chapter_ids)
    return _explore_beats(query=query, beats=beats, backend=backend, batch_size=batch_size)


def _explore_beats(
    *,
    query: ScopedQuery,
    beats: Sequence[Beat],
    backend: VisionLanguageBackend,
    batch_size: int,
) -> ExploreResult:
    candidates: list[CandidateShot] = []
    batch_count = 0
    degraded = False
    beat_by_id = {beat.beat_id: beat for beat in beats}
    for batch in _chunks(beats, max(1, int(batch_size))):
        batch_count += 1
        frames = [beat.keyframe_path for beat in batch if is_image_path(beat.keyframe_path, must_exist=True)]
        batch_degraded = len(frames) < len(batch)
        degraded = degraded or batch_degraded
        response = backend.generate(
            BackendRequest(
                task="multi_v3_explore",
                prompt=_explore_prompt(query=query, beats=batch),
                frames=frames,
                media_type="image" if frames else None,
                max_new_tokens=512,
                metadata={
                    "query_id": query.query_id,
                    "chapter_ids": list(query.scope.chapter_ids),
                    "degraded": batch_degraded,
                },
            )
        )
        for candidate in _parse_candidates(response.text):
            beat = beat_by_id.get(candidate.shot_id)
            if beat is not None:
                shot_id = beat.shot_ids[0] if beat.shot_ids else beat.beat_id
                candidates.append(CandidateShot(shot_id=shot_id, score=candidate.score, reason=candidate.reason))
            else:
                candidates.append(candidate)
    candidates.sort(key=lambda item: item.score, reverse=True)
    limit = query.budget.max_beats_to_verify
    return ExploreResult(candidates[:limit] if limit else (), batch_count=batch_count, degraded=degraded)


def explore_via_search(
    *,
    workspace: VideoWorkspace,
    query: ScopedQuery,
    backend: VisionLanguageBackend,
    top_k: int = 20,
    batch_size: int = 16,
) -> ExploreResult:
    text_hits = workspace.search_text(query.natural_query)
    visual_hits = workspace.search_visual(query.natural_query, k=top_k)
    candidate_beats = _rank_search_candidates(workspace, query=query, hit_lists=(text_hits, visual_hits), top_k=top_k)
    if not candidate_beats:
        return ExploreResult((), batch_count=0, degraded=False)
    return _explore_beats(query=query, beats=candidate_beats, backend=backend, batch_size=batch_size)


def _explore_prompt(*, query: ScopedQuery, beats: Sequence[Beat]) -> str:
    lines = [
        "Select the 1-3 beat ids that best answer the query. Return JSON only:",
        '{"picks":[{"shot_id":"...","score":0.0,"reason":"..."}]}',
        f"Query: {query.natural_query}",
        "BeatMeta:",
    ]
    for beat in beats:
        asr = " ".join(beat.asr_verbatim.split())[:180]
        ocr = " ".join(beat.ocr_verbatim)[:120]
        lines.append(f"{beat.beat_id} | {beat.start_sec:.1f}-{beat.end_sec:.1f}s | {asr} | {ocr}")
    return "\n".join(lines)


def _parse_candidates(text: str) -> list[CandidateShot]:
    payload = _json_payload(text)
    picks = payload.get("picks") if isinstance(payload, dict) else []
    candidates = []
    for item in picks if isinstance(picks, list) else []:
        if not isinstance(item, dict):
            continue
        shot_id = str(item.get("shot_id") or "").strip()
        if not shot_id:
            continue
        candidates.append(
            CandidateShot(
                shot_id=shot_id,
                score=float(item.get("score", 0.0) or 0.0),
                reason=str(item.get("reason") or ""),
            )
        )
    return candidates


def verify_window(
    *,
    query_id: str,
    request: VerifyRequest,
    frame_paths: Sequence[str],
    backend: VisionLanguageBackend,
) -> tuple[Finding, ...]:
    image_frames = [path for path in frame_paths if is_image_path(path, must_exist=True)]
    response = backend.generate(
        BackendRequest(
            task="multi_v3_verify_window",
            prompt=_verify_prompt(request),
            frames=image_frames,
            media_type="image" if image_frames else None,
            max_new_tokens=512,
            metadata={
                "query_id": query_id,
                "shot_id": request.shot_id,
                "time_range": list(request.time_range),
                "sampling": dict(request.sampling),
            },
        )
    )
    return tuple(_parse_findings(response.text, query_id=query_id, shot_id=request.shot_id))


def _verify_prompt(request: VerifyRequest) -> str:
    return "\n".join(
        [
            "Verify the focused claim against the provided high-resolution frames. Return JSON only:",
            '{"findings":[{"summary":"...","supports_options":[],"refutes_options":[],"citation_ids":[],"confidence":0.0}]}',
            f"ShotId: {request.shot_id}",
            f"TimeRange: {request.time_range[0]:.1f}-{request.time_range[1]:.1f}s",
            f"FocusClaim: {request.focus_claim}",
            f"Checks: {json.dumps(list(request.checks), ensure_ascii=False)}",
        ]
    )


def _parse_findings(text: str, *, query_id: str, shot_id: str) -> list[Finding]:
    payload = _json_payload(text)
    findings = payload.get("findings") if isinstance(payload, dict) else []
    parsed = []
    for index, item in enumerate(findings if isinstance(findings, list) else [], start=1):
        if not isinstance(item, dict):
            continue
        citation_ids = _text_tuple(item.get("citation_ids") or ())
        finding_id = str(item.get("finding_id") or "").strip() or (citation_ids[0] if citation_ids else f"{query_id}_{shot_id}_{index}")
        parsed.append(
            Finding(
                finding_id=finding_id,
                query_id=query_id,
                shot_id=shot_id,
                summary=str(item.get("summary") or ""),
                supports_options=_text_tuple(item.get("supports_options") or ()),
                refutes_options=_text_tuple(item.get("refutes_options") or ()),
                citation_ids=citation_ids,
                confidence=float(item["confidence"]) if item.get("confidence") is not None else None,
            )
        )
    return parsed


def _json_payload(text: str) -> Any:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return json.loads(match.group(0)) if match else {}


def _chunks(items: Sequence[Beat], size: int) -> tuple[tuple[Beat, ...], ...]:
    return tuple(tuple(items[index : index + size]) for index in range(0, len(items), size))


def _text_tuple(value: object) -> tuple[str, ...]:
    if value is None or isinstance(value, (str, bytes)):
        values = () if value is None else (value,)
    else:
        try:
            values = tuple(value)  # type: ignore[arg-type]
        except TypeError:
            values = (value,)
    return tuple(text for item in values if (text := str(item).strip()))


def _rank_search_candidates(
    workspace: VideoWorkspace,
    *,
    query: ScopedQuery,
    hit_lists: Sequence[Sequence[BeatHit]],
    top_k: int,
) -> tuple[Beat, ...]:
    chapter_ids = {chapter.chapter_id for chapter in workspace.chapters}
    scoped_chapters = set(query.scope.chapter_ids) & chapter_ids
    beat_by_id = {beat.beat_id: beat for beat in workspace.beats}
    scores: dict[str, float] = {}
    for hits in hit_lists:
        for rank, hit in enumerate(tuple(hits)[: max(0, int(top_k))], start=1):
            beat = beat_by_id.get(hit.beat_id)
            if beat is None:
                continue
            if scoped_chapters and beat.chapter_id not in scoped_chapters:
                continue
            scores[beat.beat_id] = scores.get(beat.beat_id, 0.0) + 1.0 / float(60 + rank)
    ordered = sorted(scores, key=lambda beat_id: (-scores[beat_id], beat_id))
    return tuple(beat_by_id[beat_id] for beat_id in ordered[: max(0, int(top_k))])
