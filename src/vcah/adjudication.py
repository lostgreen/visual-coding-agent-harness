from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from vcah.provenance import provenance_is_admissible


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _option_id(answer: str, options: Mapping[str, str]) -> str:
    match = re.match(r"\s*([A-Z])(?:\.|\)|:|\s|$)", str(answer or "").upper())
    if match and match.group(1) in options:
        return match.group(1)
    normalized = str(answer or "").strip().casefold()
    return next(
        (
            option
            for option, text in options.items()
            if normalized == str(text or "").strip().casefold()
        ),
        "",
    )


def _predicate(value: Any) -> str:
    return {
        "supported": "supported",
        "supports": "supported",
        "contradicted": "contradicted",
        "refuted": "contradicted",
        "refutes": "contradicted",
        "conflicted": "conflicted",
        "unknown": "unknown",
        "unresolved": "unknown",
    }.get(str(value or "unknown").strip().casefold(), "unknown")


@dataclass(frozen=True)
class RevisionContext:
    canonical_snapshot_revision: str
    evidence_digest_hash: str
    query_contract_hash: str

    @classmethod
    def from_inputs(
        cls,
        snapshot: Mapping[str, Any],
        evidence_digest: Sequence[Mapping[str, Any]],
        query_contract: Mapping[str, Any],
    ) -> "RevisionContext":
        return cls(
            canonical_snapshot_revision=_stable_hash(snapshot),
            evidence_digest_hash=_stable_hash(tuple(evidence_digest)),
            query_contract_hash=_stable_hash(query_contract),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "canonical_snapshot_revision": self.canonical_snapshot_revision,
            "evidence_digest_hash": self.evidence_digest_hash,
            "query_contract_hash": self.query_contract_hash,
        }


@dataclass(frozen=True)
class GuardDecision:
    allowed: bool
    blockers: tuple[str, ...]
    supporting_state: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "blockers": list(self.blockers),
            "supporting_state": dict(self.supporting_state),
        }


@dataclass(frozen=True)
class FinalAdjudication:
    answer: str
    citations: tuple[str, ...]
    answer_mode: str
    verified: bool
    grounding_status: str
    grounding_level: str
    verification_reason: str
    selection_source: str
    answer_mutation_events: tuple[Mapping[str, Any], ...]
    answer_selection_event: Mapping[str, Any]
    guard: GuardDecision
    soft_audit_guard: GuardDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": list(self.citations),
            "answer_mode": self.answer_mode,
            "verified": self.verified,
            "grounding_status": self.grounding_status,
            "grounding_level": self.grounding_level,
            "verification_reason": self.verification_reason,
            "final_selection_source": self.selection_source,
            "answer_mutation_events": [dict(item) for item in self.answer_mutation_events],
            "answer_selection_event": dict(self.answer_selection_event),
            "hard_override": self.guard.to_dict(),
            "soft_audit_correction": self.soft_audit_guard.to_dict(),
        }


def build_all_option_audit_record(
    *,
    options: Mapping[str, str],
    supplied_verdicts: Mapping[str, Mapping[str, Any]] | None,
    audit_status: str,
    audit_reason: str,
    revision_context: RevisionContext,
    required: bool,
    source_revision_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    supplied = {
        str(option).strip().upper(): dict(row)
        for option, row in dict(supplied_verdicts or {}).items()
        if str(option).strip().upper() in options and isinstance(row, Mapping)
    }
    invalidity_flags: list[str] = []
    if required and not supplied:
        invalidity_flags.append("all_option_verdicts_missing")
    missing = [option for option in options if option not in supplied]
    if required and missing:
        invalidity_flags.append("all_option_verdicts_incomplete")
    normalized: dict[str, dict[str, Any]] = {}
    for option in options:
        row = dict(supplied.get(option, {}) or {})
        verdict = _predicate(row.get("predicate_verdict", row.get("status")))
        grounding = str(row.get("grounding_eligibility", "unknown") or "unknown").casefold()
        if grounding not in {"sufficient", "insufficient", "unknown"}:
            grounding = "unknown"
        normalized[option] = {
            "predicate_verdict": verdict,
            "grounding_eligibility": grounding,
            "canonical_fact_ids": [str(item) for item in tuple(row.get("canonical_fact_ids", ()) or ()) if str(item)],
            "evidence_ids": [str(item) for item in tuple(row.get("evidence_ids", ()) or ()) if str(item)],
            "predicate_results": [
                dict(item) for item in tuple(row.get("predicate_results", ()) or ()) if isinstance(item, Mapping)
            ],
            "reason": str(row.get("reason", "") or ""),
            "provenance": [
                dict(item) for item in tuple(row.get("provenance", ()) or ()) if isinstance(item, Mapping)
            ],
        }
    normalized_status = str(audit_status or "unknown").strip().casefold()
    if normalized_status not in {"complete", "partial", "invalid"}:
        normalized_status = "partial" if supplied else "invalid"
    if invalidity_flags:
        normalized_status = "invalid"
    source = dict(source_revision_context or revision_context.to_dict())
    source_snapshot_revision = str(
        source.get("canonical_snapshot_revision", source.get("snapshot_revision", ""))
        or revision_context.canonical_snapshot_revision
    )
    source_evidence_digest = str(
        source.get("evidence_digest_hash", "") or revision_context.evidence_digest_hash
    )
    source_contract_hash = str(
        source.get("query_contract_hash", "") or revision_context.query_contract_hash
    )
    fingerprint = _stable_hash({
        "snapshot_revision": source_snapshot_revision,
        "evidence_digest_hash": source_evidence_digest,
        "query_contract_hash": source_contract_hash,
        "option_verdicts": normalized,
        "audit_status": normalized_status,
        "invalidity_flags": invalidity_flags,
    })
    return {
        "snapshot_revision": source_snapshot_revision,
        "canonical_snapshot_revision": source_snapshot_revision,
        "audit_snapshot_revision": source_snapshot_revision,
        "evidence_digest_hash": source_evidence_digest,
        "query_contract_hash": source_contract_hash,
        "audit_status": normalized_status,
        "option_verdicts": normalized,
        "invalidity_flags": invalidity_flags,
        "reason": str(audit_reason or ""),
        "audit_fingerprint": fingerprint,
    }


def evaluate_hard_override_guard(
    completion_status: Mapping[str, Any],
    qualification_result: Mapping[str, Any],
    option_verdict_table: Mapping[str, Any],
    audit_record: Mapping[str, Any],
    revision_context: RevisionContext | Mapping[str, Any],
) -> GuardDecision:
    context = (
        revision_context.to_dict()
        if isinstance(revision_context, RevisionContext)
        else dict(revision_context or {})
    )
    table = dict(option_verdict_table or {})
    audit = dict(audit_record or {})
    completion = dict(completion_status or {})
    qualification = dict(qualification_result or {})
    blockers: list[str] = []
    canonical_revision = str(context.get("canonical_snapshot_revision", "") or "")
    if str(table.get("canonical_snapshot_revision", table.get("snapshot_revision", "")) or "") != canonical_revision:
        blockers.append("verdict_table_stale")
    if str(audit.get("audit_snapshot_revision", audit.get("snapshot_revision", "")) or "") != canonical_revision:
        blockers.append("audit_stale")
    for key in ("evidence_digest_hash", "query_contract_hash"):
        expected = str(context.get(key, "") or "")
        if str(table.get(key, "") or "") != expected:
            blockers.append("verdict_table_stale")
        if str(audit.get(key, "") or "") != expected:
            blockers.append("audit_stale")
    if not bool(completion.get("completion_ready", completion.get("ready_for_answer", False))):
        blockers.append("completion_not_ready")
    blockers.extend(
        str(item)
        for item in tuple(completion.get("unresolved_critical_condition_ids", ()) or ())
        if str(item)
    )
    blockers.extend(
        str(item)
        for item in tuple(completion.get("strict_safety_blockers", ()) or ())
        if str(item)
    )
    blockers.extend(
        str(item)
        for item in tuple(completion.get("query_obligation_blockers", ()) or ())
        if str(item)
    )
    if int(completion.get("conflicted_condition_count", 0) or 0):
        blockers.append("critical_condition_conflicted")
    if str(qualification.get("status", completion.get("answer_qualification_status", "incomplete")) or "incomplete") != "complete":
        blockers.append("qualification_incomplete")
    if str(audit.get("audit_status", "invalid") or "invalid") != "complete":
        blockers.append("answer_audit_incomplete")
    if tuple(audit.get("invalidity_flags", ()) or ()):
        blockers.append("answer_audit_invalid")
    verdicts = {
        str(option).strip().upper(): dict(row)
        for option, row in dict(table.get("option_verdicts", {}) or {}).items()
    }
    supported = [
        option for option, row in verdicts.items()
        if _predicate(row.get("predicate_verdict", row.get("status"))) == "supported"
    ]
    if len(supported) != 1:
        blockers.append("supported_option_not_unique")
    if any(_predicate(row.get("predicate_verdict", row.get("status"))) == "conflicted" for row in verdicts.values()):
        blockers.append("option_predicate_conflicted")
    selected = supported[0] if len(supported) == 1 else ""
    selected_row = dict(verdicts.get(selected, {}) or {})
    grounding = str(selected_row.get("grounding_eligibility", "unknown") or "unknown").casefold()
    if grounding == "unknown":
        blockers.append("selected_option_grounding_unknown")
    elif grounding != "sufficient":
        blockers.append("selected_option_grounding_insufficient")
    predicate_results = tuple(selected_row.get("predicate_results", ()) or ())
    if any(_predicate(dict(row).get("status")) != "supported" for row in predicate_results if isinstance(row, Mapping)):
        blockers.append("selected_option_predicates_incomplete")
    if bool(table.get("provenance_required", False)) and not provenance_is_admissible(
        selected_row.get("provenance", ())
    ):
        blockers.append("provenance_insufficient")
    provenance_evaluations = (
        *tuple(qualification.get("qualification_evaluations", completion.get("qualification_evaluations", ())) or ()),
        *tuple(qualification.get("event_qualification_evaluations", ()) or ()),
    )
    for evaluation in provenance_evaluations:
        if not isinstance(evaluation, Mapping):
            continue
        if not bool(evaluation.get("required", True)):
            continue
        if _predicate(evaluation.get("status")) != "supported":
            continue
        if not provenance_is_admissible(evaluation.get("provenance", ())):
            blockers.append("provenance_insufficient")
            break
    graph = dict(qualification.get("requirement_graph", completion.get("requirement_graph", {})) or {})
    if any(
        int(graph.get(key, 0) or 0)
        for key in ("unknown", "blocked", "blocked_unresolved", "blocked_conflicted", "conflicted")
    ):
        blockers.append("requirement_graph_incomplete")
    if tuple(graph.get("unresolved_dependency_ids", ()) or ()):
        blockers.append("requirement_graph_incomplete")
    if tuple(qualification.get("incomplete_events", ()) or ()):
        blockers.append("event_qualification_incomplete")
    if tuple(qualification.get("conflicted_events", ()) or ()):
        blockers.append("event_qualification_conflicted")
    blockers = list(dict.fromkeys(blockers))
    return GuardDecision(
        allowed=not blockers,
        blockers=tuple(blockers),
        supporting_state={
            "selected_option": selected,
            "selected_grounding_eligibility": grounding,
            "supported_options": supported,
            "canonical_snapshot_revision": canonical_revision,
            "audit_fingerprint": str(audit.get("audit_fingerprint", "") or ""),
        },
    )


def evaluate_soft_audit_correction_guard(
    *,
    options: Mapping[str, str],
    raw_reasoner_answer: str,
    completion_status: Mapping[str, Any],
    qualification_result: Mapping[str, Any],
    option_verdict_table: Mapping[str, Any],
    audit_record: Mapping[str, Any],
    revision_context: RevisionContext | Mapping[str, Any],
) -> GuardDecision:
    """Permit a forced-only correction only from fresh, admissible audit facts.

    This is intentionally a guard, not a second answer selector. The caller remains
    `final_adjudicate`, which emits the only answer-selection event.
    """
    context = revision_context.to_dict() if isinstance(revision_context, RevisionContext) else dict(revision_context or {})
    completion = dict(completion_status or {})
    qualification = dict(qualification_result or {})
    table = dict(option_verdict_table or {})
    audit = dict(audit_record or {})
    blockers: list[str] = []
    raw_option = _option_id(raw_reasoner_answer, options)
    if not raw_option:
        blockers.append("raw_option_invalid")
    canonical_revision = str(context.get("canonical_snapshot_revision", "") or "")
    expected_evidence = str(context.get("evidence_digest_hash", "") or "")
    expected_contract = str(context.get("query_contract_hash", "") or "")
    if str(audit.get("audit_status", "invalid") or "invalid") != "complete":
        blockers.append("answer_audit_incomplete")
    if tuple(audit.get("invalidity_flags", ()) or ()):
        blockers.append("answer_audit_invalid")
    if str(audit.get("audit_snapshot_revision", audit.get("snapshot_revision", "")) or "") != canonical_revision:
        blockers.append("audit_stale")
    if str(audit.get("evidence_digest_hash", "") or "") != expected_evidence:
        blockers.append("audit_stale")
    if str(audit.get("query_contract_hash", "") or "") != expected_contract:
        blockers.append("audit_stale")
    if str(table.get("canonical_snapshot_revision", table.get("snapshot_revision", "")) or "") != canonical_revision:
        blockers.append("verdict_table_stale")
    if str(table.get("evidence_digest_hash", "") or "") != expected_evidence:
        blockers.append("verdict_table_stale")
    if str(table.get("query_contract_hash", "") or "") != expected_contract:
        blockers.append("verdict_table_stale")

    audit_verdicts = {
        str(option).strip().upper(): dict(row)
        for option, row in dict(audit.get("option_verdicts", {}) or {}).items()
        if str(option).strip().upper() in options and isinstance(row, Mapping)
    }
    if set(audit_verdicts) != set(options):
        blockers.append("all_option_verdicts_incomplete")
    if raw_option and _predicate(dict(audit_verdicts.get(raw_option, {}) or {}).get("predicate_verdict")) != "contradicted":
        blockers.append("raw_option_not_explicitly_contradicted")
    alternatives = [
        option for option, row in audit_verdicts.items()
        if option != raw_option and _predicate(row.get("predicate_verdict")) == "supported"
    ]
    if len(alternatives) != 1:
        blockers.append("audit_supported_alternative_not_unique")
    selected = alternatives[0] if len(alternatives) == 1 else ""
    selected_row = dict(dict(table.get("option_verdicts", {}) or {}).get(selected, {}) or {})
    if selected and _predicate(selected_row.get("predicate_verdict", selected_row.get("status"))) != "supported":
        blockers.append("alternative_predicate_not_supported")
    if selected and not tuple(selected_row.get("evidence_ids", ()) or ()):
        blockers.append("alternative_predicate_citations_missing")
    if selected and not provenance_is_admissible(selected_row.get("provenance", ())):
        blockers.append("alternative_provenance_insufficient")

    temporal_selection = dict(completion.get("temporal_max_selection", {}) or {})
    target_binding = dict(completion.get("target_entity_binding", {}) or {})
    participant_selection = dict(completion.get("target_participant_selection", {}) or {})
    if temporal_selection and str(temporal_selection.get("status", "") or "") != "resolved":
        blockers.append("episode_binding_incomplete")
    if participant_selection and str(participant_selection.get("status", "") or "") != "resolved":
        blockers.append("episode_binding_incomplete")
    if target_binding and str(target_binding.get("status", "") or "") != "resolved":
        blockers.append("episode_binding_incomplete")
    if "event_participant_link_ready" in completion and not bool(completion.get("event_participant_link_ready")):
        blockers.append("episode_binding_incomplete")
    if "target_attribute_ready" in completion and not bool(completion.get("target_attribute_ready")):
        blockers.append("episode_binding_incomplete")
    if tuple(qualification.get("conflicted_events", ()) or ()):
        blockers.append("event_qualification_conflicted")

    blockers = list(dict.fromkeys(blockers))
    return GuardDecision(
        allowed=not blockers,
        blockers=tuple(blockers),
        supporting_state={
            "raw_option": raw_option,
            "selected_option": selected,
            "audit_supported_alternatives": alternatives,
            "canonical_snapshot_revision": canonical_revision,
            "audit_fingerprint": str(audit.get("audit_fingerprint", "") or ""),
        },
    )


def final_adjudicate(
    *,
    options: Mapping[str, str],
    raw_reasoner_answer: str,
    raw_citations: Sequence[str],
    raw_gate: Mapping[str, Any],
    completion_status: Mapping[str, Any],
    qualification_result: Mapping[str, Any],
    option_verdict_table: Mapping[str, Any],
    audit_record: Mapping[str, Any],
    revision_context: RevisionContext | Mapping[str, Any],
    audit_required: bool = False,
) -> FinalAdjudication:
    guard = evaluate_hard_override_guard(
        completion_status,
        qualification_result,
        option_verdict_table,
        audit_record,
        revision_context,
    )
    soft_audit_guard = evaluate_soft_audit_correction_guard(
        options=options,
        raw_reasoner_answer=raw_reasoner_answer,
        completion_status=completion_status,
        qualification_result=qualification_result,
        option_verdict_table=option_verdict_table,
        audit_record=audit_record,
        revision_context=revision_context,
    )
    raw_option = _option_id(raw_reasoner_answer, options)
    raw_is_valid = bool(raw_option)
    gate = dict(raw_gate or {})
    table = dict(option_verdict_table or {})
    audit = dict(audit_record or {})
    context = revision_context.to_dict() if isinstance(revision_context, RevisionContext) else dict(revision_context or {})
    audit_fresh = bool(
        str(audit.get("audit_status", "invalid") or "invalid") == "complete"
        and not tuple(audit.get("invalidity_flags", ()) or ())
        and str(audit.get("audit_snapshot_revision", audit.get("snapshot_revision", "")) or "")
        == str(context.get("canonical_snapshot_revision", "") or "")
        and str(audit.get("evidence_digest_hash", "") or "")
        == str(context.get("evidence_digest_hash", "") or "")
        and str(audit.get("query_contract_hash", "") or "")
        == str(context.get("query_contract_hash", "") or "")
    )
    verdicts = dict(table.get("option_verdicts", {}) or {})
    selected = str(guard.supporting_state.get("selected_option", "") or "")
    mutations: list[Mapping[str, Any]] = []
    if bool(gate.get("passed")) and raw_is_valid and (not audit_required or audit_fresh):
        answer = raw_reasoner_answer
        citations = tuple(str(item) for item in raw_citations if str(item))
        mode = "grounded"
        verified = True
        grounding_level = str(gate.get("grounding_level", "strict") or "strict")
        status = f"verified_{grounding_level}"
        reason = str(gate.get("reason", "verified") or "verified")
        source = "raw_reasoner"
    elif guard.allowed and selected in options:
        answer = f"{selected}. {options[selected]}"
        citations = tuple(
            str(item) for item in tuple(dict(verdicts.get(selected, {}) or {}).get("evidence_ids", ()) or ()) if str(item)
        )
        mode = "grounded"
        verified = True
        grounding_level = "strict"
        status = "verified_strict"
        reason = "final_adjudicator_hard_override"
        source = "final_adjudicator_hard_override"
        if raw_is_valid and _option_id(answer, options) != raw_option:
            mutations.append({
                "source": source,
                "from_answer": raw_reasoner_answer,
                "to_answer": answer,
                "reason": reason,
            })
    elif soft_audit_guard.allowed and str(soft_audit_guard.supporting_state.get("selected_option", "") or "") in options:
        selected = str(soft_audit_guard.supporting_state["selected_option"])
        answer = f"{selected}. {options[selected]}"
        citations = tuple(
            str(item) for item in tuple(dict(verdicts.get(selected, {}) or {}).get("evidence_ids", ()) or ()) if str(item)
        )
        mode = "forced_choice"
        verified = False
        grounding_level = "none"
        status = "insufficient"
        reason = "soft_audit_correction"
        source = "soft_audit_correction"
        if raw_is_valid and _option_id(answer, options) != raw_option:
            mutations.append({
                "source": source,
                "from_answer": raw_reasoner_answer,
                "to_answer": answer,
                "reason": reason,
            })
    elif raw_is_valid:
        answer = raw_reasoner_answer
        citations = tuple(str(item) for item in raw_citations if str(item))
        mode = "forced_choice"
        verified = False
        grounding_level = "none"
        status = "insufficient"
        reason = (
            "answer_audit_incomplete"
            if bool(gate.get("passed")) and audit_required and not audit_fresh
            else str(gate.get("reason", "final_adjudication_incomplete") or "final_adjudication_incomplete")
        )
        source = "raw_reasoner"
    else:
        answer = "Insufficient verified evidence."
        citations = ()
        mode = "insufficient"
        verified = False
        grounding_level = "none"
        status = "insufficient"
        reason = str(gate.get("reason", "answer_missing") or "answer_missing")
        source = "final_adjudicator_insufficient"
    event = {
        "type": "answer_selection_event",
        "raw_reasoner_answer": raw_reasoner_answer,
        "final_answer": answer,
        "final_selection_source": source,
        "answer_mutation_event_count": len(mutations),
    }
    return FinalAdjudication(
        answer=answer,
        citations=citations,
        answer_mode=mode,
        verified=verified,
        grounding_status=status,
        grounding_level=grounding_level,
        verification_reason=reason,
        selection_source=source,
        answer_mutation_events=tuple(mutations),
        answer_selection_event=event,
        guard=guard,
        soft_audit_guard=soft_audit_guard,
    )
