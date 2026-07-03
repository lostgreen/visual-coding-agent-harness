"""ASR-to-claim binding tools."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from visual_coding_agent_harness.backends.base import BackendRequest, VisionLanguageBackend
from visual_coding_agent_harness.core.registry import ToolRegistry, tool
from visual_coding_agent_harness.video._map import VideoMapSegment, VideoMapStore
from visual_coding_agent_harness.legacy.workspace_v2 import EvidenceWorkspace


_SUPPORTED_VERDICTS = {"supports", "supported"}
_KNOWN_VERDICTS = _SUPPORTED_VERDICTS | {"contradicts", "insufficient"}


def build_asr_binding_registry(
    *,
    video_map_store: VideoMapStore,
    backend: VisionLanguageBackend,
    workspace: EvidenceWorkspace | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()

    @tool(
        name="bind_asr_claim",
        description="Bind indexed ASR cue ids in one segment to registered target_refs.",
    )
    def bind_asr_claim(segment_id: str, target_refs: Sequence[str]) -> Mapping[str, object]:
        segment_id_text = str(segment_id or "").strip()
        target_ref_list = _string_list(target_refs)
        current = video_map_store.current
        try:
            segment = current.get(segment_id_text)
        except ValueError:
            return _limited_result(
                segment_id=segment_id_text,
                target_refs=target_ref_list,
                limitations=f"Unknown segment_id for bind_asr_claim: {segment_id_text}",
            )

        registry_obj = getattr(workspace, "target_registry", None) if workspace is not None else None
        if workspace is None or registry_obj is None:
            return _limited_result(
                segment_id=segment_id_text,
                target_refs=target_ref_list,
                limitations="bind_asr_claim requires a workspace TargetRegistry.",
            )

        cues = _asr_cues(segment)
        if not cues:
            return _limited_result(
                segment_id=segment_id_text,
                target_refs=target_ref_list,
                limitations="bind_asr_claim requires indexed asr_sentences on the current segment.",
            )

        targets, target_limitations = _resolve_targets(registry_obj=registry_obj, target_refs=target_ref_list)
        if target_limitations or not targets:
            return _limited_result(
                segment_id=segment_id_text,
                target_refs=target_ref_list,
                limitations="; ".join(target_limitations) or "bind_asr_claim requires at least one target_ref.",
            )

        cues_by_id = {str(cue["cue_id"]): cue for cue in cues}
        prompt = _binding_prompt(segment=segment, cues=cues, targets=targets)
        response = backend.generate(
            BackendRequest(
                task="asr_claim_binding",
                prompt=prompt,
                max_new_tokens=800,
                temperature=0.0,
            )
        )
        try:
            payload = _parse_json_object(response.text)
        except ValueError as exc:
            return _limited_result(
                segment_id=segment_id_text,
                target_refs=target_ref_list,
                limitations=f"Could not parse asr_claim_binding JSON: {exc}",
            )

        evidence_bindings, answer_rows, limitations, illegal_retry_feedback = _build_binding_rows(
            segment=segment,
            segment_id_text=segment_id_text,
            targets=targets,
            payload=payload,
            cues_by_id=cues_by_id,
        )
        if illegal_retry_feedback and not answer_rows:
            retry_response = backend.generate(
                BackendRequest(
                    task="asr_claim_binding",
                    prompt=_binding_prompt(
                        segment=segment,
                        cues=cues,
                        targets=targets,
                        retry_feedback=illegal_retry_feedback,
                    ),
                    max_new_tokens=800,
                    temperature=0.0,
                )
            )
            try:
                retry_payload = _parse_json_object(retry_response.text)
            except ValueError as exc:
                limitations.append(f"Retry could not parse asr_claim_binding JSON: {exc}")
            else:
                evidence_bindings, answer_rows, retry_limitations, _retry_illegal_feedback = _build_binding_rows(
                    segment=segment,
                    segment_id_text=segment_id_text,
                    targets=targets,
                    payload=retry_payload,
                    cues_by_id=cues_by_id,
                )
                limitations = [*limitations, *retry_limitations]

        claim = (
            f"ASR claim binding checked {len(targets)} target_ref(s) in {segment_id_text}; "
            f"{len(answer_rows)} supported binding row(s) produced."
        )
        return {
            "claim": claim,
            "confidence": 1.0 if answer_rows else 0.0,
            "segment_id": segment_id_text,
            "target_refs": [str(target["target_id"]) for target in targets],
            "bindings": evidence_bindings,
            "evidence_bindings": evidence_bindings,
            "answer_evidence_rows": answer_rows,
            "regions": [segment.to_dict()],
            "limitations": "; ".join(limitations),
        }

    registry.register(bind_asr_claim)
    return registry


def _asr_cues(segment: VideoMapSegment) -> list[dict[str, object]]:
    cues: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, sentence in enumerate(getattr(segment, "asr_sentences", ()) or (), start=1):
        if not isinstance(sentence, Mapping):
            continue
        text = str(sentence.get("text") or "").strip()
        if not text:
            continue
        cue_id = str(
            sentence.get("cue_id")
            or sentence.get("cueId")
            or sentence.get("id")
            or sentence.get("sentence_id")
            or f"cue_{index:04d}"
        ).strip()
        if not cue_id or cue_id in seen:
            cue_id = f"cue_{index:04d}"
        seen.add(cue_id)
        cues.append(
            {
                "cue_id": cue_id,
                "start_sec": _optional_float(sentence.get("start_sec")),
                "end_sec": _optional_float(sentence.get("end_sec")),
                "text": text,
            }
        )
    return cues


def _resolve_targets(*, registry_obj: Any, target_refs: Sequence[str]) -> tuple[list[dict[str, str]], list[str]]:
    targets: list[dict[str, str]] = []
    limitations: list[str] = []
    seen: set[str] = set()
    for raw_ref in target_refs:
        ref = str(raw_ref or "").strip()
        if not ref or ref in seen:
            continue
        seen.add(ref)
        try:
            target = registry_obj.resolve_target_ref(ref)
        except KeyError:
            limitations.append(f"Unknown target_ref: {ref}")
            continue
        target_id = str(getattr(target, "target_id", ref)).strip() or ref
        claim = _target_canonical_claim(target)
        if not claim:
            limitations.append(f"target_ref has no canonical claim: {target_id}")
            continue
        targets.append({"target_id": target_id, "canonical_claim": claim})
    return targets, limitations


def _target_canonical_claim(target: Any) -> str:
    return str(getattr(target, "canonical_claim", "") or getattr(target, "canonical_text", "") or "").strip()


def _binding_prompt(
    *,
    segment: VideoMapSegment,
    cues: Sequence[Mapping[str, object]],
    targets: Sequence[Mapping[str, str]],
    retry_feedback: str = "",
) -> str:
    cue_lines = []
    for cue in cues:
        window = _cue_window(cue)
        cue_lines.append(f"- {cue['cue_id']} {window}: {cue['text']}")
    target_lines = [f"- {target['target_id']}: {target['canonical_claim']}" for target in targets]
    example_target = str(targets[0]["target_id"]) if targets else "T1"
    example_cue_ids = [str(cue["cue_id"]) for cue in cues[:2] if str(cue.get("cue_id", "")).strip()]
    if not example_cue_ids:
        example_cue_ids = ["cue_id_from_list"]
    example_cue_json = json.dumps(example_cue_ids)
    return "\n".join(
        [
            "Bind registered target claims to indexed ASR cues from one video segment.",
            "Return only JSON with exactly this shape:",
            f'{{"{example_target}": {{"verdict": "supports|supported|contradicts|insufficient", "cue_ids": {example_cue_json}, "quote": "short exact ASR quote"}}}}',
            "Use supports/supported only when the ASR cue directly says the target claim.",
            "cue_ids must be copied exactly from the ASR cues list below; do not add prefixes or invent cue ids.",
            *([f"Previous response used illegal cue_ids: {retry_feedback}. Use only exact ids from the list below."] if retry_feedback else []),
            f"Segment: {segment.segment_id} [{float(segment.start_sec):.3f}-{float(segment.end_sec):.3f}s]",
            "Targets:",
            *target_lines,
            "ASR cues:",
            *cue_lines,
        ]
    )


def _cue_window(cue: Mapping[str, object]) -> str:
    start = cue.get("start_sec")
    end = cue.get("end_sec")
    if start is None or end is None:
        return ""
    return f"[{float(start):.3f}-{float(end):.3f}s]"


def _build_binding_rows(
    *,
    segment: VideoMapSegment,
    segment_id_text: str,
    targets: Sequence[Mapping[str, str]],
    payload: Mapping[str, Any],
    cues_by_id: Mapping[str, Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str], str]:
    evidence_bindings: list[dict[str, object]] = []
    answer_rows: list[dict[str, object]] = []
    limitations: list[str] = []
    illegal_feedback: list[str] = []
    for target in targets:
        target_id = str(target["target_id"])
        entry = payload.get(target_id)
        if not isinstance(entry, Mapping):
            limitations.append(f"missing binding for {target_id}")
            continue

        verdict = str(entry.get("verdict", "")).strip().lower()
        if verdict not in _KNOWN_VERDICTS:
            limitations.append(f"unknown verdict for {target_id}: {verdict or '(empty)'}")
            continue

        raw_cue_ids = _string_list(entry.get("cue_ids"))
        cue_ids, normalized = _normalize_cue_ids(raw_cue_ids, legal_cue_ids=cues_by_id.keys())
        if normalized:
            limitations.append(f"normalized_cue_ids for {target_id}: {', '.join(normalized)}")
        quote = str(entry.get("quote") or "").strip()
        binding = {
            "evidence_id": _evidence_id(segment_id=segment_id_text, target_id=target_id),
            "status": "supported" if verdict in _SUPPORTED_VERDICTS else verdict,
            "target_id": target_id,
            "segment_id": segment_id_text,
            "source": "indexed_transcript",
            "cue_ids": cue_ids,
            "snippet": quote,
            "canonical_claim": str(target["canonical_claim"]),
        }
        evidence_bindings.append(binding)

        if verdict not in _SUPPORTED_VERDICTS:
            continue
        if not cue_ids:
            limitations.append(f"supported verdict for {target_id} has no cue_ids")
            continue
        illegal = [cue_id for cue_id in cue_ids if cue_id not in cues_by_id]
        if illegal:
            legal_preview = _cue_id_preview(cues_by_id.keys())
            message = (
                f"illegal cue_ids for {target_id}: {', '.join(illegal)}; "
                f"use bare ids from {legal_preview}; do not repeat this exact call"
            )
            limitations.append(message)
            illegal_feedback.append(message)
            continue

        selected_cues = [cues_by_id[cue_id] for cue_id in cue_ids]
        answer_rows.append(
            _answer_evidence_row(
                segment=segment,
                target=target,
                cue_ids=cue_ids,
                cues=selected_cues,
                quote=quote,
                evidence_binding=binding,
            )
        )
    return evidence_bindings, answer_rows, limitations, "; ".join(illegal_feedback)


def _normalize_cue_ids(raw_cue_ids: Sequence[str], *, legal_cue_ids: Sequence[str]) -> tuple[list[str], list[str]]:
    legal = {str(cue_id): str(cue_id) for cue_id in legal_cue_ids}
    normalized: list[str] = []
    notes: list[str] = []
    for raw_cue_id in raw_cue_ids:
        cue_id = str(raw_cue_id or "").strip()
        if not cue_id:
            continue
        if cue_id in legal:
            normalized.append(legal[cue_id])
            continue
        candidate = _strip_cue_prefix(cue_id)
        if candidate in legal:
            normalized.append(legal[candidate])
            notes.append(f"{cue_id}->{candidate}")
            continue
        normalized.append(cue_id)
    return normalized, notes


def _strip_cue_prefix(cue_id: str) -> str:
    match = re.fullmatch(r"(?i)cue[_-]?(.+)", cue_id.strip())
    return match.group(1).strip() if match else cue_id


def _cue_id_preview(cue_ids: Sequence[str], *, limit: int = 8) -> str:
    values = [str(cue_id) for cue_id in cue_ids if str(cue_id)]
    if len(values) <= limit:
        return "{" + ", ".join(values) + "}"
    return "{" + ", ".join(values[:limit]) + f", ... +{len(values) - limit}" + "}"


def _answer_evidence_row(
    *,
    segment: VideoMapSegment,
    target: Mapping[str, str],
    cue_ids: Sequence[str],
    cues: Sequence[Mapping[str, object]],
    quote: str,
    evidence_binding: Mapping[str, object],
) -> dict[str, object]:
    snippet = quote or " ".join(str(cue.get("text") or "").strip() for cue in cues if str(cue.get("text") or "").strip())
    start_sec = _min_present_float(cue.get("start_sec") for cue in cues)
    end_sec = _max_present_float(cue.get("end_sec") for cue in cues)
    time_range = [start_sec, end_sec] if start_sec is not None and end_sec is not None else None
    target_id = str(target["target_id"])
    canonical_claim = str(target["canonical_claim"])
    binding_payload = dict(evidence_binding)
    binding_payload["snippet"] = snippet
    return {
        "evidence_id": str(evidence_binding.get("evidence_id") or _evidence_id(segment_id=segment.segment_id, target_id=target_id)),
        "tool": "transcript_evidence_binder",
        "segment_id": segment.segment_id,
        "time_range": time_range,
        "t_start": start_sec,
        "t_end": end_sec,
        "target_id": target_id,
        "event_label": target_id,
        "snippet": snippet,
        "claim": f"Indexed ASR in {segment.segment_id} supports {target_id}: {canonical_claim}. Snippet: {snippet}",
        "confidence": 0.88,
        "grounding_quality": "indexed_transcript",
        "confidence_signal": "asr_claim_binding_supported",
        "limitations": "Model-assisted binding over indexed ASR; cue ids were checked against the current segment.",
        "source_segment_id": segment.segment_id,
        "raw_asr_ref": {"cue_ids": list(cue_ids)},
        "evidence_binding": binding_payload,
    }


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    if not stripped:
        raise ValueError("empty response")
    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no JSON object found") from None
        try:
            loaded = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(str(exc)) from exc
    if not isinstance(loaded, dict):
        raise ValueError("top-level response is not an object")
    return loaded


def _limited_result(*, segment_id: str, target_refs: Sequence[str], limitations: str) -> dict[str, object]:
    return {
        "claim": f"ASR claim binding could not produce supported evidence for {segment_id}.",
        "confidence": 0.0,
        "segment_id": segment_id,
        "target_refs": list(target_refs),
        "bindings": [],
        "answer_evidence_rows": [],
        "regions": [],
        "limitations": limitations,
    }


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)):
        values = [value]
    elif isinstance(value, Sequence):
        values = list(value)
    else:
        values = []
    return [str(item).strip() for item in values if str(item or "").strip()]


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _min_present_float(values: Any) -> float | None:
    floats = [_optional_float(value) for value in values]
    present = [value for value in floats if value is not None]
    return min(present) if present else None


def _max_present_float(values: Any) -> float | None:
    floats = [_optional_float(value) for value in values]
    present = [value for value in floats if value is not None]
    return max(present) if present else None


def _evidence_id(*, segment_id: str, target_id: str) -> str:
    segment = re.sub(r"[^A-Za-z0-9_]+", "_", str(segment_id or "segment")).strip("_") or "segment"
    target = re.sub(r"[^A-Za-z0-9_]+", "_", str(target_id or "target")).strip("_") or "target"
    return f"asr_bind_{segment}_{target}"
