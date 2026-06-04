"""Evidence verification tools for answer-facing ledger checks."""

from __future__ import annotations

import re
from typing import Mapping, Optional, Sequence

from ..registry import ToolRegistry, tool
from ..workspace import EvidenceWorkspace


_VISUAL_EVIDENCE_TOOLS = {"caption_segment", "qa_segment", "inspect_segment"}


def build_verification_registry(*, workspace: Optional[EvidenceWorkspace] = None) -> ToolRegistry:
    registry = ToolRegistry()

    @tool(name="verify_ledger_answer", description="Check whether grounded ledger evidence supports an answer.")
    def verify_ledger_answer(
        answer: str,
        ledger_text: str = "",
        min_score: float = 0.6,
        required_citations: Sequence[str] = (),
        requires_visual_evidence: bool = True,
    ) -> Mapping[str, object]:
        resolved_ledger = ledger_text or _read_workspace_ledger(workspace)
        answer_terms = _tokens(answer)
        ledger_terms = _tokens(resolved_ledger)
        supported_terms = sorted(answer_terms.intersection(ledger_terms))
        missing_terms = sorted(answer_terms.difference(ledger_terms))
        score = len(supported_terms) / max(1, len(answer_terms))
        cited_observations = _observation_ids(resolved_ledger)
        observation_tools = _observation_tools(resolved_ledger)
        visual_observation_ids = [
            observation_id
            for observation_id in cited_observations
            if observation_tools.get(observation_id) in _VISUAL_EVIDENCE_TOOLS
        ]
        missing_citations = [citation for citation in required_citations if citation not in cited_observations]
        gate_reasons = []
        if missing_citations:
            gate_reasons.append("missing required citations")
        if requires_visual_evidence and not visual_observation_ids:
            gate_reasons.append("no non-navigation visual evidence")
        if score < min_score:
            gate_reasons.append("low lexical support")
        verdict = "supported" if not gate_reasons else "insufficient"
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
                    "observation_tools": observation_tools,
                    "evidence_gate": {
                        "required_citations": list(required_citations),
                        "missing_citations": missing_citations,
                        "requires_visual_evidence": requires_visual_evidence,
                        "visual_observation_ids": visual_observation_ids,
                        "reasons": gate_reasons,
                    },
                    "verdict": verdict,
                }
            ],
            "limitations": (
                "Rule-based verifier with citation and visual-evidence gates; "
                "use model-based entailment and temporal checks for stronger validation."
            ),
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


def _observation_tools(ledger_text: str) -> Mapping[str, str]:
    tools = {}
    for line in ledger_text.splitlines():
        obs_match = re.search(r"`(obs_[0-9]{4})`", line)
        tool_match = re.search(r"tool:\s*`?([A-Za-z0-9_]+)`?", line)
        if obs_match and tool_match:
            tools[obs_match.group(1)] = tool_match.group(1)
    return tools


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
