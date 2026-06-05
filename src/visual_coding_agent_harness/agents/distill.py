"""Distill raw tool observations into staged evidence records."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from ..workspace import EvidenceRecord, EvidenceWorkspace, Observation
from .output_quality import confidence_signal_from_text


def distill(observation: Observation, workspace: EvidenceWorkspace) -> list[EvidenceRecord]:
    if observation.tool in {"vision_read", "inspect_segment"}:
        records = _fact_distiller(observation, workspace)
        if records:
            return records
    return [_default_distiller(observation, workspace)]


def _default_distiller(observation: Observation, workspace: EvidenceWorkspace) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=workspace.next_evidence_id("distilled"),
        stage="distilled",
        parent_id=None,
        tool=observation.tool,
        observation_id=observation.observation_id,
        frame_set_id=observation.frame_set_id,
        content={
            "claim": observation.claim,
            "regions": list(observation.regions),
            "limitations": observation.limitations,
            "confidence_signal": _confidence_signal(observation),
        },
        grounding_quality=_grounding_quality(observation),
        confidence=observation.confidence,
        created_at=time.time(),
    )


def _fact_distiller(observation: Observation, workspace: EvidenceWorkspace) -> list[EvidenceRecord]:
    facts = _facts_from_raw_output(observation.raw_output)
    records = []
    for offset, fact in enumerate(facts):
        records.append(
            EvidenceRecord(
                evidence_id=workspace.next_evidence_id("distilled", sequence_offset=offset),
                stage="distilled",
                parent_id=None,
                tool=observation.tool,
                observation_id=observation.observation_id,
                frame_set_id=observation.frame_set_id,
                content={
                    "claim": str(fact.get("claim", "")),
                    "regions": list(observation.regions),
                    "limitations": observation.limitations,
                    "confidence_signal": _confidence_signal(observation, fact=fact),
                },
                grounding_quality=_fact_grounding_quality(observation, fact),  # type: ignore[arg-type]
                confidence=float(fact.get("confidence", observation.confidence) or 0.0),
                created_at=time.time(),
            )
        )
    return records


def _facts_from_raw_output(raw_output: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_facts = raw_output.get("facts") or raw_output.get("vision_facts") or []
    if not isinstance(raw_facts, Sequence) or isinstance(raw_facts, (str, bytes)):
        return []
    facts = []
    for item in raw_facts:
        if isinstance(item, Mapping):
            claim = str(item.get("fact") or item.get("claim") or "").strip()
            if claim:
                payload = dict(item)
                payload["claim"] = claim
                facts.append(payload)
            continue
        claim = str(item).strip()
        if claim:
            facts.append({"claim": claim})
    return facts


def _grounding_quality(observation: Observation) -> str:
    signal = _confidence_signal(observation)
    if signal == "unsupported":
        return "inferred"
    if signal == "degenerate":
        return "weak"
    explicit = str(observation.raw_output.get("grounding_quality", "")).strip()
    if explicit:
        return explicit
    if observation.tool in {"vision_read", "inspect_segment", "caption_segment", "qa_segment"}:
        return "visually_confirmed"
    if observation.tool == "query_context":
        return "query_global_context"
    if observation.tool == "global_gist":
        return "global_sparse"
    return "navigation_only"


def _fact_grounding_quality(observation: Observation, fact: Mapping[str, Any]) -> str:
    signal = _confidence_signal(observation, fact=fact)
    if signal == "unsupported":
        return "inferred"
    if signal == "degenerate":
        return "weak"
    explicit = str(fact.get("grounding_quality", "")).strip()
    return explicit or _grounding_quality(observation)


def _confidence_signal(observation: Observation, *, fact: Mapping[str, Any] | None = None) -> str:
    for source in [
        fact or {},
        observation.raw_output,
        {"confidence_signal": observation.confidence_signal},
    ]:
        signal = str(source.get("confidence_signal", "")).strip().lower()
        if signal:
            return signal
    text = str((fact or {}).get("claim", "") or observation.claim)
    return confidence_signal_from_text(text)
