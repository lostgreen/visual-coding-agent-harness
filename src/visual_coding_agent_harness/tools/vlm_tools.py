"""VLM-backed explore and verify tools for multi_v3 investigators."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Sequence

from visual_coding_agent_harness.backends.base import BackendRequest, VisionLanguageBackend
from visual_coding_agent_harness.contracts.query import ScopedQuery
from visual_coding_agent_harness.contracts.report import CandidateShot, Finding, VerifyRequest
from visual_coding_agent_harness.video.pipeline import is_image_path
from visual_coding_agent_harness.video.index import Shot, VideoIndex


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
    index: VideoIndex,
    backend: VisionLanguageBackend,
    batch_size: int = 16,
) -> ExploreResult:
    shots = tuple(shot for scene_id in query.scope.scene_ids for shot in index.get_scene(scene_id).shots)
    candidates: list[CandidateShot] = []
    batch_count = 0
    degraded = False
    for batch in _chunks(shots, max(1, int(batch_size))):
        batch_count += 1
        frames = [shot.lowres_grid_path for shot in batch if is_image_path(shot.lowres_grid_path, must_exist=True)]
        batch_degraded = len(frames) < len(batch)
        degraded = degraded or batch_degraded
        response = backend.generate(
            BackendRequest(
                task="multi_v3_explore",
                prompt=_explore_prompt(query=query, shots=batch),
                frames=frames,
                media_type="image" if frames else None,
                max_new_tokens=512,
                metadata={
                    "query_id": query.query_id,
                    "scene_ids": list(query.scope.scene_ids),
                    "degraded": batch_degraded,
                },
            )
        )
        candidates.extend(_parse_candidates(response.text))
    candidates.sort(key=lambda item: item.score, reverse=True)
    limit = query.budget.max_shots_to_verify
    return ExploreResult(candidates[:limit] if limit else (), batch_count=batch_count, degraded=degraded)


def _explore_prompt(*, query: ScopedQuery, shots: Sequence[Shot]) -> str:
    lines = [
        "Select the 1-3 shot ids that best answer the query. Return JSON only:",
        '{"picks":[{"shot_id":"...","score":0.0,"reason":"..."}]}',
        f"Query: {query.natural_query}",
        "ShotMeta:",
    ]
    for shot in shots:
        asr = " ".join(shot.asr_text.split())[:180]
        entities = ", ".join(shot.entities)
        lines.append(f"{shot.shot_id} | {shot.start_sec:.1f}-{shot.end_sec:.1f}s | {asr} | {entities}")
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


def _chunks(items: Sequence[Shot], size: int) -> tuple[tuple[Shot, ...], ...]:
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
