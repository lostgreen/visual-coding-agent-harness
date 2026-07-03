"""Low-resolution visual explore tool for multi_v3 investigators."""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from visual_coding_agent_harness.backends.base import BackendRequest, VisionLanguageBackend
from visual_coding_agent_harness.contracts.query import ScopedQuery
from visual_coding_agent_harness.contracts.report import CandidateShot
from visual_coding_agent_harness.video.index import Shot, VideoIndex


def explore(
    *,
    query: ScopedQuery,
    index: VideoIndex,
    backend: VisionLanguageBackend,
    batch_size: int = 16,
) -> tuple[CandidateShot, ...]:
    shots = tuple(shot for scene_id in query.scope.scene_ids for shot in index.get_scene(scene_id).shots)
    candidates: list[CandidateShot] = []
    for batch in _chunks(shots, max(1, int(batch_size))):
        response = backend.generate(
            BackendRequest(
                task="multi_v3_explore",
                prompt=_explore_prompt(query=query, shots=batch),
                frames=[shot.lowres_grid_path for shot in batch if shot.lowres_grid_path],
                media_type="image",
                max_new_tokens=512,
                metadata={"query_id": query.query_id, "scene_ids": list(query.scope.scene_ids)},
            )
        )
        candidates.extend(_parse_candidates(response.text))
    candidates.sort(key=lambda item: item.score, reverse=True)
    limit = query.budget.max_shots_to_verify
    return tuple(candidates[:limit] if limit else ())


def _explore_prompt(*, query: ScopedQuery, shots: Sequence[Shot]) -> str:
    lines = [
        "Select the 1-3 shot ids that best answer the query. Return JSON only:",
        '{"picks":[{"shot_id":"...","score":0.0,"reason":"..."}]}',
        f"Query: {query.natural_query}",
        f"ExpectedEvidence: {query.expected_evidence}",
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
