from __future__ import annotations

import re
from typing import Sequence

from vcah.types import Claim, ClaimContract, ClaimVerdict, EvidenceRecord, QueryClaim, is_path_only_visual_evidence


def verify_claim(claim: Claim, evidence: Sequence[EvidenceRecord]) -> ClaimVerdict:
    contract = claim.contract or ClaimContract()
    semantic = _semantic_verify_claim(claim, evidence)
    return apply_capability_gate(claim, semantic, evidence, contract=contract)


def _semantic_verify_claim(claim: Claim, evidence: Sequence[EvidenceRecord]) -> ClaimVerdict:
    contract = claim.contract or ClaimContract()
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
    )


def verify_query_claim(claim: QueryClaim, evidence: Sequence[EvidenceRecord]) -> ClaimVerdict:
    return verify_claim(Claim(claim.claim_id, "", claim.text), evidence)


def verify_claims(claims: Sequence[Claim], evidence: Sequence[EvidenceRecord]) -> tuple[ClaimVerdict, ...]:
    return tuple(verify_claim(claim, evidence) for claim in claims)


def apply_capability_gate(
    claim: Claim,
    semantic_verdict: ClaimVerdict | None,
    evidence: Sequence[EvidenceRecord],
    *,
    contract: ClaimContract | None = None,
) -> ClaimVerdict:
    contract = contract or claim.contract or ClaimContract()
    semantic = semantic_verdict or ClaimVerdict(claim.claim_id, "unknown", reason="missing_semantic_verdict")
    cited_evidence = _cited_records(semantic, evidence)
    gate_input = cited_evidence if semantic.status in {"supported", "contradicted"} else evidence
    failures = _capability_failures(contract, gate_input)
    if failures:
        return ClaimVerdict(
            claim.claim_id,
            "unknown",
            capability_checks=tuple(failures),
            reason=failures[0],
            source=semantic.source,
        )
    if semantic.status == "unknown":
        return semantic
    return ClaimVerdict(
        claim.claim_id,
        semantic.status,
        support_evidence_ids=semantic.support_evidence_ids if semantic.status == "supported" else (),
        contradict_evidence_ids=semantic.contradict_evidence_ids if semantic.status == "contradicted" else (),
        entailment_kind=semantic.entailment_kind,
        capability_checks=tuple(dict.fromkeys((*semantic.capability_checks, "scope_compatible", "observability_compatible"))),
        reason=semantic.reason,
        source=semantic.source,
    )


def _cited_records(verdict: ClaimVerdict, evidence: Sequence[EvidenceRecord]) -> tuple[EvidenceRecord, ...]:
    citations = set(verdict.citations)
    if not citations:
        return ()
    return tuple(record for record in evidence if record.evidence_id in citations)


def _capability_failures(contract: ClaimContract, evidence: Sequence[EvidenceRecord]) -> list[str]:
    usable = tuple(record for record in evidence if not is_path_only_visual_evidence(record))
    if not usable:
        return ["missing_evidence"]
    if contract.required_observability:
        modalities = {record.modality for record in usable}
        modalities.update(segment.modality for record in usable for segment in record.coverage_manifest)
        required = set(contract.required_observability)
        if contract.observability_mode == "any":
            compatible = bool(required & modalities)
        else:
            compatible = required.issubset(modalities)
        if not compatible:
            return ["observability_mismatch"]
    if not _has_required_scope(contract.required_scope, usable):
        return ["insufficient_scope" if contract.required_scope != "full_video" else "aggregation_or_coverage_missing"]
    if contract.aggregation != "none" or contract.quantifier in {"distinct_count", "total_count", "order", "comparison"}:
        if not any(record.modality == "derived" and record.parent_evidence_ids and record.coverage_manifest for record in usable):
            return ["aggregation_or_coverage_missing"]
    return []


def _has_required_scope(required_scope: str, evidence: Sequence[EvidenceRecord]) -> bool:
    scope_rank = {"local": 0, "local_frame": 0, "window": 1, "multi_window": 2, "full_video": 3}
    required = scope_rank.get(required_scope, 1)
    if any(scope_rank.get(record.temporal_scope, 1) >= required for record in evidence):
        return True
    if required_scope == "multi_window":
        return len({rid for record in evidence for rid in record.request_ids}) >= 2
    return False


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
