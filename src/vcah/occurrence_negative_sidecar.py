from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from vcah.occurrence_agent import (
    assert_no_oracle_packet,
    build_occurrence_candidate_cards,
)
from vcah.occurrence_sufficiency import (
    MAX_EVIDENCE_PASSAGES,
    QUESTION_CRITICAL_CONSTRAINT_TYPES,
)


NEGATIVE_SIDECAR_CONTRACT = "oob_negative_only_occurrence_evidence_v1"
_TOP_LEVEL_FIELDS = frozenset({"contradictions"})
_ROW_FIELDS = frozenset({"constraint_id", "occurrence_id", "evidence_passage_ids"})
_SAFE_RESPONSE_METADATA_FIELDS = (
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "content_chars",
    "requested_completion_tokens",
    "retry_count",
    "truncated_then_retried",
    "truncation_retry_count",
    "provider_seed_supported",
    "provider_reported_seed_support",
    "response_format_type",
)


@dataclass(frozen=True)
class NegativeSidecarSnapshot:
    case_id: str
    question: str
    options: Any
    set_id: str
    constraints: tuple[dict[str, str], ...]
    candidates: tuple[dict[str, Any], ...]
    source_case_sha256: str
    source_runtime_sha256: str
    replay_fixture_sha256: str

    def model_payload(self) -> dict[str, Any]:
        payload = {
            "question": self.question,
            "options": _json_copy(self.options),
            "constraints": [dict(row) for row in self.constraints],
            "candidate_evidence": [_json_copy(row) for row in self.candidates],
        }
        assert_no_oracle_packet(payload, surface="negative_occurrence_sidecar")
        return payload

    @property
    def digest(self) -> str:
        return stable_digest(
            {
                "contract": NEGATIVE_SIDECAR_CONTRACT,
                "case_id": self.case_id,
                "set_id": self.set_id,
                "model_payload": self.model_payload(),
                "source_case_sha256": self.source_case_sha256,
                "source_runtime_sha256": self.source_runtime_sha256,
                "replay_fixture_sha256": self.replay_fixture_sha256,
            }
        )


def load_negative_sidecar_snapshot(
    positive_run_dir: Path,
    *,
    replay_fixture_path: Path,
) -> NegativeSidecarSnapshot:
    run_dir = Path(positive_run_dir)
    case_path = run_dir / "case.json"
    runtime_path = run_dir / "runtime_summary.json"
    fixture_path = Path(replay_fixture_path)
    case = _read_json(case_path)
    runtime = _read_json(runtime_path)
    fixture = _read_json(fixture_path)
    case_id = str(case.get("case_id", run_dir.name) or run_dir.name)
    if str(fixture.get("case_id", "") or "") != case_id:
        raise ValueError(f"{case_id}: replay fixture case mismatch")

    trace = tuple(
        row for row in tuple(runtime.get("trace", ()) or ()) if isinstance(row, Mapping)
    )
    declarations = tuple(
        row for row in trace if row.get("type") == "occurrence_evidence_declaration"
    )
    if not declarations:
        raise ValueError(f"{case_id}: no frozen evidence declaration")
    decisions = tuple(
        row for row in trace if row.get("type") == "occurrence_sufficiency_decision"
    )
    target_set_id = str(
        (decisions[-1] if decisions else declarations[-1]).get("set_id", "") or ""
    )
    declaration = next(
        (
            row
            for row in reversed(declarations)
            if str(row.get("set_id", "") or "") == target_set_id
        ),
        None,
    )
    if declaration is None:
        raise ValueError(f"{case_id}: no declaration for final set")
    scope_ids = tuple(
        dict.fromkeys(
            str(value)
            for value in tuple(declaration.get("scope_occurrence_ids", ()) or ())
            if str(value)
        )
    )
    if not scope_ids:
        raise ValueError(f"{case_id}: empty frozen occurrence scope")

    constraints = _frozen_constraints(declaration, case_id=case_id)
    candidates = _candidate_cards_for_scope(
        fixture,
        set_id=target_set_id,
        scope_ids=scope_ids,
        case_id=case_id,
    )
    question = str(case.get("question", "") or "").strip()
    if not question:
        raise ValueError(f"{case_id}: missing question")
    return NegativeSidecarSnapshot(
        case_id=case_id,
        question=question,
        options=_json_copy(case.get("options", {})),
        set_id=target_set_id,
        constraints=constraints,
        candidates=candidates,
        source_case_sha256=file_sha256(case_path),
        source_runtime_sha256=file_sha256(runtime_path),
        replay_fixture_sha256=file_sha256(fixture_path),
    )


def negative_sidecar_prompt(snapshot: NegativeSidecarSnapshot) -> str:
    payload = json.dumps(
        snapshot.model_payload(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        "Act as an independent negative-evidence auditor. Inspect only the supplied "
        "candidate passages against the fixed question constraints. Report a row "
        "only when a visible passage directly contradicts that candidate on that "
        "constraint. Omit uncertainty and absence of evidence. Do not rank candidates, "
        "propose an answer, or emit positive support. Return exactly one JSON object "
        "with key contradictions. Each contradiction must contain only constraint_id, "
        "occurrence_id, and one to three evidence_passage_ids copied from that "
        "candidate. An empty list is valid.\nINPUT="
        + payload
        + '\nOUTPUT_SCHEMA={"contradictions":[{"constraint_id":"...",'
        '"occurrence_id":"...","evidence_passage_ids":["..."]}]}'
    )


def parse_negative_sidecar_response(raw: str) -> Mapping[str, Any] | None:
    text = str(raw or "").strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return dict(value) if isinstance(value, Mapping) else None


def validate_negative_sidecar_output(
    payload: Mapping[str, Any] | None,
    *,
    snapshot: NegativeSidecarSnapshot,
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    if payload is None:
        return (), ("negative_sidecar_invalid_json",)
    errors: list[str] = []
    if set(payload) != _TOP_LEVEL_FIELDS:
        errors.append("negative_sidecar_top_level_field_invalid")
    raw_rows = payload.get("contradictions")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        return (), tuple(dict.fromkeys((*errors, "negative_sidecar_rows_invalid")))

    constraints = {str(row["constraint_id"]): row for row in snapshot.constraints}
    passages_by_candidate = {
        str(candidate.get("occurrence_id", "") or ""): {
            str(passage.get("passage_id", "") or "")
            for passage in tuple(candidate.get("representative_passages", ()) or ())
            if isinstance(passage, Mapping) and str(passage.get("passage_id", "") or "")
        }
        for candidate in snapshot.candidates
    }
    normalized: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for raw_row in raw_rows:
        if not isinstance(raw_row, Mapping):
            errors.append("negative_sidecar_row_must_be_object")
            continue
        if set(raw_row) != _ROW_FIELDS:
            errors.append("negative_sidecar_row_field_invalid")
            continue
        constraint_id = str(raw_row.get("constraint_id", "") or "").strip()
        occurrence_id = str(raw_row.get("occurrence_id", "") or "").strip()
        if constraint_id not in constraints:
            errors.append("negative_sidecar_constraint_invalid")
            continue
        if occurrence_id not in passages_by_candidate:
            errors.append("negative_sidecar_candidate_invalid")
            continue
        pair = (constraint_id, occurrence_id)
        if pair in seen_pairs:
            errors.append("negative_sidecar_duplicate_pair")
            continue
        seen_pairs.add(pair)
        raw_passages = raw_row.get("evidence_passage_ids")
        if not isinstance(raw_passages, Sequence) or isinstance(
            raw_passages, (str, bytes)
        ):
            errors.append("negative_sidecar_passages_invalid")
            continue
        passage_ids = tuple(
            dict.fromkeys(str(value) for value in raw_passages if str(value))
        )
        if not 1 <= len(passage_ids) <= MAX_EVIDENCE_PASSAGES:
            errors.append("negative_sidecar_passages_invalid")
            continue
        if any(
            passage_id not in passages_by_candidate[occurrence_id]
            for passage_id in passage_ids
        ):
            errors.append("negative_sidecar_passage_not_visible")
            continue
        normalized.append(
            {
                "constraint_id": constraint_id,
                "constraint_type": str(constraints[constraint_id]["constraint_type"]),
                "occurrence_id": occurrence_id,
                "evidence_passage_ids": list(passage_ids),
            }
        )
    if errors:
        return (), tuple(dict.fromkeys(errors))
    return tuple(normalized), ()


def safe_response_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    safe = {
        key: _json_copy(metadata.get(key))
        for key in _SAFE_RESPONSE_METADATA_FIELDS
        if key in metadata
    }
    request_id = str(metadata.get("provider_request_id", "") or "")
    if request_id:
        safe["provider_request_id_digest"] = stable_digest(request_id)
    return safe


def stable_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_constraints(
    declaration: Mapping[str, Any], *, case_id: str
) -> tuple[dict[str, str], ...]:
    constraints: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in tuple(declaration.get("constraints", ()) or ()):
        if not isinstance(raw, Mapping):
            continue
        constraint_id = str(raw.get("constraint_id", "") or "").strip()
        constraint_type = str(raw.get("constraint_type", "") or "").strip().casefold()
        description = str(raw.get("description", "") or "").strip()
        if (
            not constraint_id
            or constraint_id in seen
            or constraint_type not in QUESTION_CRITICAL_CONSTRAINT_TYPES
            or not description
        ):
            raise ValueError(f"{case_id}: invalid frozen constraint surface")
        seen.add(constraint_id)
        constraints.append(
            {
                "constraint_id": constraint_id,
                "constraint_type": constraint_type,
                "description": description,
            }
        )
    if not constraints:
        raise ValueError(f"{case_id}: no frozen constraints")
    return tuple(constraints)


def _candidate_cards_for_scope(
    fixture: Mapping[str, Any],
    *,
    set_id: str,
    scope_ids: Sequence[str],
    case_id: str,
) -> tuple[dict[str, Any], ...]:
    scope = set(scope_ids)
    exact: list[tuple[dict[str, Any], ...]] = []
    fallback: list[tuple[dict[str, Any], ...]] = []
    for raw_packet in tuple(fixture.get("packets", ()) or ()):
        if not isinstance(raw_packet, Mapping):
            continue
        packet = raw_packet.get("packet", raw_packet)
        if not isinstance(packet, Mapping):
            continue
        occurrence_set = packet.get("occurrence_set")
        if not isinstance(occurrence_set, Mapping):
            continue
        cards = tuple(build_occurrence_candidate_cards(packet))
        card_ids = {str(card.get("occurrence_id", "") or "") for card in cards}
        if not scope <= card_ids:
            continue
        filtered = tuple(
            _model_candidate_card(card)
            for card in cards
            if str(card.get("occurrence_id", "") or "") in scope
        )
        packet_set_id = str(
            occurrence_set.get("attempt_id", occurrence_set.get("set_id", "")) or ""
        )
        (exact if packet_set_id == set_id else fallback).append(filtered)
    matches = exact or fallback
    if not matches:
        raise ValueError(f"{case_id}: no replay packet covers frozen scope")
    selected = matches[-1]
    if {row["occurrence_id"] for row in selected} != scope:
        raise ValueError(f"{case_id}: candidate scope mismatch")
    if any(not row["representative_passages"] for row in selected):
        raise ValueError(f"{case_id}: candidate has no visible passage")
    return selected


def _model_candidate_card(card: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "occurrence_id": str(card.get("occurrence_id", "") or ""),
        "rank": int(card.get("rank", 0) or 0),
        "time_range": list(card.get("time_range", ()) or ()),
        "matched_queries": [
            str(value)
            for value in tuple(card.get("matched_queries", ()) or ())
            if str(value)
        ],
        "representative_passages": [
            {
                "passage_id": str(passage.get("passage_id", "") or ""),
                "time_range": list(passage.get("time_range", ()) or ()),
                "caption_excerpt": str(passage.get("caption_excerpt", "") or ""),
                "query_matches": [
                    str(value)
                    for value in tuple(passage.get("query_matches", ()) or ())
                    if str(value)
                ],
            }
            for passage in tuple(card.get("representative_passages", ()) or ())
            if isinstance(passage, Mapping)
        ],
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))
