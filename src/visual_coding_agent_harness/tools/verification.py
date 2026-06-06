"""Evidence verification tools for answer-facing ledger checks."""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Sequence

from ..registry import ToolRegistry, tool
from ..workspace import EvidenceWorkspace
from ..agents.skills.predicates import (
    direct_floor_holds,
    grounding_quality_floor,
    no_decisive_weak_grounding,
    no_unaddressed_conflict,
    selected_option_has_structured_support,
    temporal_order_consistent,
)


_VISUAL_EVIDENCE_TOOLS = {"caption_segment", "qa_segment", "inspect_segment", "vision_read"}
_GROUNDING_WEIGHTS = {
    "global_sparse": 0.35,
    "visually_confirmed": 1.0,
    "indexed_transcript": 0.85,
    "inferred": 0.35,
    "weak": 0.2,
    "external_knowledge": 0.1,
}


def build_verification_registry(*, workspace: Optional[EvidenceWorkspace] = None) -> ToolRegistry:
    registry = ToolRegistry()

    @tool(name="verify_ledger_answer", description="Check whether grounded ledger evidence supports an answer.")
    def verify_ledger_answer(
        answer: str,
        ledger_text: str = "",
        question: str = "",
        candidate_options: Sequence[str] = (),
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
        option_gate = _option_conflict_gate(
            workspace=workspace,
            answer=answer,
            question=question,
            candidate_options=candidate_options,
            required_citations=required_citations,
        )
        gate_reasons.extend(option_gate["reasons"])
        temporal_gate = _temporal_order_gate(
            workspace=workspace,
            answer=answer,
            question=question,
            candidate_options=candidate_options,
        )
        gate_reasons.extend(temporal_gate["reasons"])
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
                        "option_relations": option_gate["option_relations"],
                        "top_conflicting_observation": option_gate["top_conflicting_observation"],
                        "top_conflict_relation": option_gate["top_conflict_relation"],
                        "temporal_order_verdict": temporal_gate["temporal_order_verdict"],
                        "temporal_order": temporal_gate["temporal_order"],
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


def _option_conflict_gate(
    *,
    workspace: Optional[EvidenceWorkspace],
    answer: str,
    question: str,
    candidate_options: Sequence[str],
    required_citations: Sequence[str],
) -> Mapping[str, Any]:
    default = {
        "reasons": [],
        "option_relations": {},
        "top_conflicting_observation": "",
        "top_conflict_relation": "",
    }
    if workspace is None:
        return default
    answer_option = _answer_option(answer)
    if not answer_option:
        return default

    table = workspace.evidence_table_v2(question=question, options=candidate_options)
    rows = table.get("rows", []) if isinstance(table.get("rows", []), Sequence) else []
    cited = set(str(item) for item in required_citations)
    option_relations = {}
    selected_support = 0.0
    conflicts = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        observation_id = str(row.get("obs_id", ""))
        relation = _option_relation(answer_option=answer_option, supported_option=row.get("supported_option"))
        if observation_id:
            option_relations[observation_id] = relation
        row_score = _weighted_row_score(row)
        if relation == "Support" and observation_id in cited:
            selected_support = max(selected_support, row_score)
        if relation == "Contradict" and observation_id not in cited and _is_well_grounded(row):
            conflicts.append((row_score, observation_id, relation))

    conflicts.sort(key=lambda item: (-item[0], item[1]))
    reasons = []
    top_conflicting_observation = ""
    top_conflict_relation = ""
    if conflicts and conflicts[0][0] > selected_support:
        reasons.append("uncited stronger conflicting option support")
        top_conflicting_observation = conflicts[0][1]
        top_conflict_relation = conflicts[0][2]

    predicate_results = [
        selected_option_has_structured_support(table, selected_option=answer_option),
        no_decisive_weak_grounding(table, selected_option=answer_option),
        no_unaddressed_conflict(table, selected_option=answer_option, cited_obs_ids=required_citations),
        direct_floor_holds(table, selected_option=answer_option),
    ]
    for result in predicate_results:
        if result.passed:
            continue
        for reason in result.reasons:
            if reason not in reasons:
                reasons.append(str(reason))
    mapped_records = workspace.mapped_evidence_records(
        observation_ids=required_citations,
        selected_option=answer_option,
    )
    if mapped_records:
        grounding_reason = grounding_quality_floor(
            mapped_records,
            workspace=workspace,
            require_visual=True,
        )
        if grounding_reason and grounding_reason not in reasons:
            reasons.append(grounding_reason)

    return {
        "reasons": reasons,
        "option_relations": option_relations,
        "top_conflicting_observation": top_conflicting_observation,
        "top_conflict_relation": top_conflict_relation,
    }


def _temporal_order_gate(
    *,
    workspace: Optional[EvidenceWorkspace],
    answer: str,
    question: str,
    candidate_options: Sequence[str],
) -> Mapping[str, Any]:
    default = {
        "reasons": [],
        "temporal_order_verdict": "Neutral",
        "temporal_order": {},
    }
    if workspace is None or not _looks_temporal_ordering(question, candidate_options, answer):
        return default

    selected_text = _selected_option_text(answer=answer, candidate_options=candidate_options)
    expected_events = _option_event_sequence(selected_text)
    if len(expected_events) < 2:
        return default

    table = workspace.evidence_table_v2(question=question, options=candidate_options)
    timeline_rows = workspace.read_timeline_sorted()
    if timeline_rows:
        observed_events = _observed_timeline_events(timeline_rows)
        predicate_table: Mapping[str, Any] = {**table, "timeline": timeline_rows}
    else:
        rows = table.get("rows", []) if isinstance(table.get("rows", []), Sequence) else []
        observed_events = _observed_events(rows)
        predicate_table = table
    if len(observed_events) < 2:
        return default

    matched = []
    for event in expected_events:
        observed = _match_observed_event(event, observed_events)
        if observed:
            matched.append({"expected": event, "observed": observed["event"], "start_sec": observed["start_sec"]})
    if len(matched) < 2:
        return default

    matched_times = [float(item["start_sec"]) for item in matched]
    verdict = "Support" if matched_times == sorted(matched_times) else "Contradict"
    reasons = ["temporal order contradicts evidence"] if verdict == "Contradict" else []
    predicate_result = temporal_order_consistent(
        predicate_table,
        selected_option=_answer_option(answer),
        expected_events=expected_events,
    )
    if not predicate_result.passed:
        for reason in predicate_result.reasons:
            if reason not in reasons:
                reasons.append(str(reason))
    return {
        "reasons": reasons,
        "temporal_order_verdict": verdict,
        "temporal_order": {
            "expected_events": expected_events,
            "observed_events": observed_events,
            "matched_events": matched,
        },
    }


def _answer_option(answer: str) -> str:
    match = re.search(r"^\s*([A-H])(?:[\).:-]|\s*$|\s+(?:option|choice|answer)\b)", answer.strip(), flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b(?:answer|choice|option)\s*(?:is|:)?\s*([A-H])\b", answer, flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""


def _selected_option_text(*, answer: str, candidate_options: Sequence[str]) -> str:
    answer_option = _answer_option(answer)
    option_map = _candidate_option_text_map(candidate_options)
    if answer_option and answer_option in option_map:
        return option_map[answer_option]
    return answer


def _candidate_option_text_map(candidate_options: Sequence[str]) -> dict[str, str]:
    mapping = {}
    for index, option in enumerate(candidate_options):
        text = str(option).strip()
        match = re.match(r"^([A-Za-z])(?:[\.)]\s*|\s+|$)", text)
        letter = match.group(1).upper() if match else chr(ord("A") + index)
        mapping[letter] = text
    return mapping


def _option_relation(*, answer_option: str, supported_option: Any) -> str:
    if not supported_option:
        return "Neutral"
    supported = str(supported_option).strip().upper()[:1]
    if not supported or not re.match(r"[A-H]", supported):
        return "Neutral"
    return "Support" if supported == answer_option else "Contradict"


def _weighted_row_score(row: Mapping[str, Any]) -> float:
    return float(row.get("confidence", 0.0) or 0.0) * _GROUNDING_WEIGHTS.get(
        str(row.get("grounding_quality", "weak")),
        0.2,
    )


def _is_well_grounded(row: Mapping[str, Any]) -> bool:
    return str(row.get("grounding_quality", "")) == "visually_confirmed" and float(row.get("confidence", 0.0) or 0.0) >= 0.75


def _looks_temporal_ordering(question: str, candidate_options: Sequence[str], answer: str) -> bool:
    text = " ".join([question, answer, " ".join(str(option) for option in candidate_options)]).lower()
    return any(marker in text for marker in ["before", "after", " then ", "order", "sequence", "first"])


def _option_event_sequence(option_text: str) -> list[str]:
    text = _strip_option_prefix(option_text).lower()
    text = text.replace("->", " then ").replace("→", " then ").replace(">", " then ")
    if re.search(r"\bbefore\b", text):
        parts = re.split(r"\bbefore\b", text, maxsplit=1)
        return _clean_event_sequence(parts)
    if re.search(r"\bafter\b", text):
        parts = re.split(r"\bafter\b", text, maxsplit=1)
        return list(reversed(_clean_event_sequence(parts)))
    return _clean_event_sequence(re.split(r"\bthen\b|[,;/]+", text))


def _strip_option_prefix(text: str) -> str:
    return re.sub(r"^\s*[A-H](?:[\.)]\s*|\s+)", "", text, flags=re.IGNORECASE).strip()


def _clean_event_sequence(parts: Sequence[str]) -> list[str]:
    events = []
    for part in parts:
        cleaned = re.sub(r"\s+", " ", str(part)).strip(" .:-")
        if _event_tokens(cleaned):
            events.append(cleaned)
    return events


def _observed_events(rows: Sequence[Any]) -> list[Mapping[str, Any]]:
    observed = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        event = str(row.get("event_label", "")).strip()
        time_range = row.get("time_range")
        if not event or not isinstance(time_range, Sequence) or isinstance(time_range, (str, bytes)) or len(time_range) < 2:
            continue
        try:
            start_sec = float(time_range[0])
        except (TypeError, ValueError):
            continue
        observed.append(
            {
                "event": event,
                "start_sec": start_sec,
                "obs_id": str(row.get("obs_id", "")),
            }
        )
    return sorted(observed, key=lambda item: (float(item["start_sec"]), str(item.get("obs_id", ""))))


def _observed_timeline_events(rows: Sequence[Any]) -> list[Mapping[str, Any]]:
    observed = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("confidence_signal", "")).strip().lower() != "confirmed":
            continue
        event = str(row.get("entity") or row.get("event_label") or "").strip()
        observed_at_sec = row.get("observed_at_sec")
        if not event or observed_at_sec is None:
            continue
        try:
            start_sec = float(observed_at_sec)
        except (TypeError, ValueError):
            continue
        observed.append({"event": event, "start_sec": start_sec, "obs_id": str(row.get("obs_id", ""))})
    return sorted(observed, key=lambda item: (float(item["start_sec"]), str(item.get("obs_id", ""))))


def _match_observed_event(expected_event: str, observed_events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for observed in observed_events:
        if _events_match(expected_event, str(observed.get("event", ""))):
            return observed
    return None


def _events_match(left: str, right: str) -> bool:
    left_tokens = _event_tokens(left)
    right_tokens = _event_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    if left_tokens.issubset(right_tokens) or right_tokens.issubset(left_tokens):
        return True
    overlap = len(left_tokens.intersection(right_tokens))
    union = len(left_tokens.union(right_tokens))
    return bool(union) and overlap / union >= 0.5


def _event_tokens(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "the",
        "and",
        "then",
        "before",
        "after",
        "first",
        "second",
        "third",
        "fourth",
        "later",
        "appears",
        "appear",
        "shown",
        "shows",
        "object",
        "event",
        "option",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in stopwords
    }
