"""Evidence verification tools for answer-facing ledger checks."""

from __future__ import annotations

import re
from typing import Mapping, Optional

from ..registry import ToolRegistry, tool
from ..workspace import EvidenceWorkspace


def build_verification_registry(*, workspace: Optional[EvidenceWorkspace] = None) -> ToolRegistry:
    registry = ToolRegistry()

    @tool(name="verify_ledger_answer", description="Check whether the current evidence ledger lexically supports an answer.")
    def verify_ledger_answer(answer: str, ledger_text: str = "", min_score: float = 0.6) -> Mapping[str, object]:
        resolved_ledger = ledger_text or _read_workspace_ledger(workspace)
        answer_terms = _tokens(answer)
        ledger_terms = _tokens(resolved_ledger)
        supported_terms = sorted(answer_terms.intersection(ledger_terms))
        missing_terms = sorted(answer_terms.difference(ledger_terms))
        score = len(supported_terms) / max(1, len(answer_terms))
        verdict = "supported" if score >= min_score else "insufficient"
        cited_observations = _observation_ids(resolved_ledger)
        return {
            "claim": f"Ledger support is {verdict} with lexical score {score:.2f}.",
            "confidence": round(score, 3),
            "input_artifacts": [str(workspace.root / "ledger.md")] if workspace is not None and not ledger_text else [],
            "regions": [
                {
                    "answer": answer,
                    "support_score": round(score, 3),
                    "supported_terms": supported_terms,
                    "missing_terms": missing_terms,
                    "cited_observations": cited_observations,
                    "verdict": verdict,
                }
            ],
            "limitations": "Lexical verifier; use model-based entailment and temporal checks for stronger validation.",
        }

    @tool(name="summarize_ledger_evidence", description="Extract compact claims from evidence ledger text.")
    def summarize_ledger_evidence(ledger_text: str = "", max_claims: int = 5) -> Mapping[str, object]:
        resolved_ledger = ledger_text or _read_workspace_ledger(workspace)
        claims = _ledger_claims(resolved_ledger, max_claims=max_claims)
        count = len(claims)
        return {
            "claim": f"Extracted {count} ledger claim{'s' if count != 1 else ''}.",
            "confidence": 1.0 if count else 0.0,
            "input_artifacts": [str(workspace.root / "ledger.md")] if workspace is not None and not ledger_text else [],
            "regions": [{"claims": claims}],
            "limitations": "Rule-based ledger summarizer; preserves only compact claim text.",
        }

    registry.register(verify_ledger_answer)
    registry.register(summarize_ledger_evidence)
    return registry


def _read_workspace_ledger(workspace: Optional[EvidenceWorkspace]) -> str:
    if workspace is None:
        return ""
    ledger_path = workspace.root / "ledger.md"
    if not ledger_path.exists():
        return ""
    return ledger_path.read_text(encoding="utf-8")


def _tokens(text: str) -> set[str]:
    stopwords = {"the", "and", "with", "from", "that", "this", "onto", "into", "beside"}
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+", text)
        if len(token) > 2 and token.lower() not in stopwords
    }


def _observation_ids(ledger_text: str) -> list[str]:
    return re.findall(r"`(obs_[0-9]{4})`", ledger_text)


def _ledger_claims(ledger_text: str, *, max_claims: int) -> list[Mapping[str, str]]:
    claims = []
    for line in ledger_text.splitlines():
        obs_match = re.search(r"`(obs_[0-9]{4})`", line)
        claim_match = re.search(r"claim:\s*(.*?)\s*\|\s*limitations:", line)
        if not claim_match:
            claim_match = re.search(r"claim:\s*(.*)$", line)
        if obs_match and claim_match:
            claims.append({"observation_id": obs_match.group(1), "claim": claim_match.group(1).strip()})
        if len(claims) >= max_claims:
            break
    return claims
