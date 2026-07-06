from __future__ import annotations

import re
from typing import Sequence

from vcah.types import Claim, ClaimContract, ClaimVerdict, EvidenceRecord, QueryClaim, is_path_only_visual_evidence


def verify_claim(claim: Claim, evidence: Sequence[EvidenceRecord]) -> ClaimVerdict:
    contract = claim.contract or ClaimContract()
    capability_failures = _capability_failures(contract, evidence)
    if capability_failures:
        return ClaimVerdict(
            claim.claim_id,
            "unknown",
            capability_checks=tuple(capability_failures),
            reason=capability_failures[0],
        )

    claim_tokens = set(_tokens(claim.text))
    best: EvidenceRecord | None = None
    best_overlap = 0
    best_kind = "none"
    for record in evidence:
        if is_path_only_visual_evidence(record):
            continue
        local_failure = _record_claim_incompatibility(claim, contract, record)
        if local_failure:
            best_kind = local_failure
            continue
        overlap = len(claim_tokens & set(_tokens(record.verbatim)))
        if overlap > best_overlap:
            best = record
            best_overlap = overlap
            best_kind = _entailment_kind(claim, record)

    if best is None or best_overlap < max(1, int(len(claim_tokens) * 0.35)):
        return ClaimVerdict(
            claim.claim_id,
            "unknown",
            entailment_kind="proxy" if best_kind == "predicate_mismatch" else ("absence" if best_kind == "local_absence_overreach" else "none"),
            capability_checks=(best_kind,) if best_kind not in {"none", ""} else (),
            reason="proxy_evidence_cannot_support_claim" if best_kind == "predicate_mismatch" else (best_kind if best_kind not in {"none", ""} else "insufficient_direct_evidence"),
        )
    if best_kind == "proxy":
        return ClaimVerdict(
            claim.claim_id,
            "unknown",
            entailment_kind="proxy",
            capability_checks=("predicate_mismatch",),
            reason="proxy_evidence_cannot_support_claim",
        )
    if best.observation_polarity == "negative":
        return ClaimVerdict(
            claim.claim_id,
            "contradicted",
            contradict_evidence_ids=(best.evidence_id,),
            entailment_kind="absence",
            capability_checks=("scope_compatible", "observability_compatible"),
        )
    return ClaimVerdict(
        claim.claim_id,
        "supported",
        support_evidence_ids=(best.evidence_id,),
        entailment_kind="derived" if best.modality == "derived" else "direct",
        capability_checks=("scope_compatible", "observability_compatible"),
    )


def verify_query_claim(claim: QueryClaim, evidence: Sequence[EvidenceRecord]) -> ClaimVerdict:
    return verify_claim(Claim(claim.claim_id, "", claim.text), evidence)


def verify_claims(claims: Sequence[Claim], evidence: Sequence[EvidenceRecord]) -> tuple[ClaimVerdict, ...]:
    return tuple(verify_claim(claim, evidence) for claim in claims)


def _capability_failures(contract: ClaimContract, evidence: Sequence[EvidenceRecord]) -> list[str]:
    usable = tuple(record for record in evidence if not is_path_only_visual_evidence(record))
    if not usable:
        return ["missing_evidence"]
    if contract.required_observability:
        modalities = {record.modality for record in usable}
        modalities.update(segment.modality for record in usable for segment in record.coverage_manifest)
        if not set(contract.required_observability) & modalities:
            return ["observability_mismatch"]
    if contract.required_scope == "full_video" and not any(record.temporal_scope == "full_video" for record in usable):
        if contract.quantifier in {"distinct_count", "total_count", "universal"}:
            return ["aggregation_or_coverage_missing"]
    if contract.required_scope == "multi_window" and not any(record.temporal_scope in {"multi_window", "full_video"} for record in usable):
        if len({rid for record in usable for rid in record.request_ids}) < 2:
            return ["insufficient_scope"]
    if contract.aggregation != "none" or contract.quantifier in {"distinct_count", "total_count", "order", "comparison"}:
        if not any(record.modality == "derived" and record.parent_evidence_ids and record.coverage_manifest for record in usable):
            return ["aggregation_or_coverage_missing"]
    return []


def _record_claim_incompatibility(claim: Claim, contract: ClaimContract, record: EvidenceRecord) -> str:
    del claim
    if record.observation_polarity == "negative":
        if record.modality == "visual" and record.temporal_scope in {"local_frame", "window"} and record.sampling_coverage in {"sparse", "unknown"}:
            if contract.required_scope in {"multi_window", "full_video"} or contract.observation_target in {"relation", "event"}:
                return "local_absence_overreach"
    if _looks_like_proxy_claim(contract, record):
        return "predicate_mismatch"
    return ""


def _looks_like_proxy_claim(contract: ClaimContract, record: EvidenceRecord) -> bool:
    text = str(record.verbatim or "").casefold()
    if contract.observation_target == "relation" and contract.required_scope in {"multi_window", "full_video"}:
        relation_terms = {"because", "reason", "obstacle", "difficult", "difficulty", "important", "main", "major"}
        if not any(term in text for term in relation_terms):
            return True
    return False


def _entailment_kind(claim: Claim, record: EvidenceRecord) -> str:
    if record.modality == "derived":
        return "derived"
    claim_text = str(claim.text or "").casefold()
    evidence_text = str(record.verbatim or "").casefold()
    strong_terms = {"most important", "main reason", "major obstacle", "because", "difficulty"}
    if any(term in claim_text for term in strong_terms) and not any(term in evidence_text for term in strong_terms):
        return "proxy"
    return "direct"


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0) for match in re.finditer(r"\w+", str(text or "").casefold()))
