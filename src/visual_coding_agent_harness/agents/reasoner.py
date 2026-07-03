"""Reasoner for the multi_v3 long-video loop."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from visual_coding_agent_harness.backends.base import BackendRequest, VisionLanguageBackend
from visual_coding_agent_harness.contracts.query import ScopedQuery, VerifiableGoal
from visual_coding_agent_harness.contracts.report import DigestItem
from visual_coding_agent_harness.agents.result import WorkspaceRunResult


@dataclass(frozen=True)
class ReasonerDecision:
    action: str
    goals: Sequence[VerifiableGoal] = field(default_factory=tuple)
    queries: Sequence[ScopedQuery] = field(default_factory=tuple)
    rationale: str = ""
    answer: str = ""
    confidence: str = ""
    citations: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "goals", tuple(self.goals))
        object.__setattr__(self, "queries", tuple(self.queries))
        object.__setattr__(self, "citations", tuple(str(item) for item in self.citations if str(item)))

    def to_run_result(self, *, rounds: int) -> WorkspaceRunResult:
        return WorkspaceRunResult(
            answer=self.answer,
            citations=tuple(self.citations),
            confidence=self.confidence,
            rounds=rounds,
            metadata={"status": "final" if self.answer else "need_more_evidence", "strategy": "multi_v3"},
        )


class Reasoner:
    def __init__(self, *, backend: VisionLanguageBackend) -> None:
        self.backend = backend

    def decide(
        self,
        *,
        question: str,
        options: Mapping[str, str],
        index_context: str,
        overview_path: str,
        previous_digest: Sequence[DigestItem],
        round_number: int,
    ) -> ReasonerDecision:
        response = self.backend.generate(
            BackendRequest(
                task="multi_v3_reasoner",
                system_prompt=(
                    "You are Reasoner. Only plan ScopedQuery batches or return a final answer. "
                    "Do not call tools or inspect raw observations."
                ),
                prompt=_reasoner_prompt(
                    question=question,
                    options=options,
                    index_context=index_context,
                    previous_digest=previous_digest,
                    round_number=round_number,
                ),
                media_path=overview_path,
                media_type="image" if overview_path else None,
                max_new_tokens=1024,
                metadata={"round_number": round_number, "strategy": "multi_v3"},
            )
        )
        return _parse_decision(response.text)


def _reasoner_prompt(
    *,
    question: str,
    options: Mapping[str, str],
    index_context: str,
    previous_digest: Sequence[DigestItem],
    round_number: int,
) -> str:
    return "\n".join(
        [
            f"Round: {round_number}",
            f"Question: {question}",
            "Options:",
            json.dumps(dict(options), ensure_ascii=False, sort_keys=True),
            "SceneIndex:",
            index_context,
            "PreviousDigest:",
            json.dumps([item.to_dict() for item in previous_digest], ensure_ascii=False),
            'Return JSON: {"action":"plan","goals":[],"queries":[],"rationale":"..."} or {"action":"answer","answer":"A","confidence":"medium","citations":[]}',
        ]
    )


def _parse_decision(text: str) -> ReasonerDecision:
    payload = _json_payload(text)
    action = str(payload.get("action") or "").strip() if isinstance(payload, Mapping) else ""
    if action == "answer":
        return ReasonerDecision(
            action="answer",
            answer=str(payload.get("answer") or ""),
            confidence=str(payload.get("confidence") or ""),
            citations=tuple(str(item) for item in _sequence(payload.get("citations"))),
        )
    return ReasonerDecision(
        action="plan",
        goals=tuple(VerifiableGoal.from_dict(item) for item in _sequence(payload.get("goals")) if isinstance(item, Mapping)),
        queries=tuple(ScopedQuery.from_dict(item) for item in _sequence(payload.get("queries")) if isinstance(item, Mapping)),
        rationale=str(payload.get("rationale") or ""),
    )


def _json_payload(text: str) -> Mapping[str, Any]:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        payload = json.loads(match.group(0)) if match else {}
    return payload if isinstance(payload, Mapping) else {}


def _sequence(value: object) -> tuple[object, ...]:
    if value is None or isinstance(value, (str, bytes)):
        return ()
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return ()
