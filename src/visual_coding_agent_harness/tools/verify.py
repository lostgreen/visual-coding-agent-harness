"""High-resolution verify adapter for multi_v3 investigators."""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from visual_coding_agent_harness.backends.base import BackendRequest, VisionLanguageBackend
from visual_coding_agent_harness.contracts.report import Finding, VerifyRequest
from visual_coding_agent_harness.video.artifacts import is_image_path


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


def _text_tuple(value: object) -> tuple[str, ...]:
    if value is None or isinstance(value, (str, bytes)):
        values = () if value is None else (value,)
    else:
        try:
            values = tuple(value)  # type: ignore[arg-type]
        except TypeError:
            values = (value,)
    return tuple(text for item in values if (text := str(item).strip()))
