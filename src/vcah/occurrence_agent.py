from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from vcah.caption_schema import stable_digest


OCCURRENCE_METHOD_ARMS = ("none", "a0", "a1", "a1-flat", "a2")
FORBIDDEN_AGENT_VISIBLE_KEYS = frozenset(
    {
        "clue_intervals",
        "evaluation_case",
        "gold",
        "gold_answer",
        "gold_clue_intervals",
        "normalized_clue_intervals",
        "oracle_guidance",
        "oracle_intervention",
        "reference_answer",
    }
)
DEFAULT_CARD_CANDIDATE_LIMIT = 8
DEFAULT_CARD_EXCERPT_LIMIT = 3
DEFAULT_CARD_EXCERPT_CHARS = 240
DEFAULT_CARD_QUERY_LIMIT = 4
DEFAULT_CARD_QUERY_CHARS = 160
OCCURRENCE_RESOLUTION_OPS = frozenset({"keep", "eliminate", "select", "reopen"})


@dataclass
class OccurrenceResolutionStateV1:
    """Runtime-owned state; the Reasoner remains responsible for every semantic choice."""

    states: dict[str, str] = field(default_factory=dict)
    current_visible_ids: tuple[str, ...] = ()
    selected_occurrence_id: str = ""
    revision: int = 0

    def sync_visible(self, occurrence_ids: Sequence[str]) -> bool:
        visible = tuple(
            dict.fromkeys(
                str(value or "").strip()
                for value in occurrence_ids
                if str(value or "").strip()
            )
        )
        changed = visible != self.current_visible_ids
        self.current_visible_ids = visible
        for occurrence_id in visible:
            self.states.setdefault(occurrence_id, "active")
        if self.selected_occurrence_id not in set(visible):
            if self.states.get(self.selected_occurrence_id) == "selected":
                self.states[self.selected_occurrence_id] = "active"
            self.selected_occurrence_id = ""
        if changed:
            self.revision += 1
        return changed

    def validate_ops(
        self, operations: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        errors: list[dict[str, Any]] = []
        visible = set(self.current_visible_ids)
        shadow_states = dict(self.states)
        shadow_selected = self.selected_occurrence_id
        for index, operation in enumerate(operations):
            if not isinstance(operation, Mapping):
                errors.append(
                    {
                        "code": "occurrence_op_must_be_object",
                        "occurrence_op_index": index,
                    }
                )
                continue
            op = str(operation.get("op", operation.get("type", "")) or "").casefold()
            occurrence_id = str(operation.get("occurrence_id", "") or "").strip()
            if op not in OCCURRENCE_RESOLUTION_OPS:
                errors.append(
                    {
                        "code": "unsupported_occurrence_op",
                        "occurrence_op_index": index,
                        "op": op,
                    }
                )
                continue
            if not occurrence_id or occurrence_id not in visible:
                errors.append(
                    {
                        "code": "occurrence_id_not_currently_visible",
                        "occurrence_op_index": index,
                        "occurrence_id": occurrence_id,
                    }
                )
                continue
            if op == "select" and shadow_states.get(occurrence_id) == "eliminated":
                errors.append(
                    {
                        "code": "eliminated_occurrence_requires_reopen",
                        "occurrence_op_index": index,
                        "occurrence_id": occurrence_id,
                    }
                )
                continue
            if op == "keep" and shadow_states.get(occurrence_id) == "eliminated":
                errors.append(
                    {
                        "code": "eliminated_occurrence_requires_reopen",
                        "occurrence_op_index": index,
                        "occurrence_id": occurrence_id,
                    }
                )
                continue
            shadow_selected = _apply_occurrence_op(
                shadow_states,
                selected_occurrence_id=shadow_selected,
                op=op,
                occurrence_id=occurrence_id,
            )
        return errors

    def apply_ops(
        self, operations: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        normalized = tuple(dict(item) for item in operations if isinstance(item, Mapping))
        errors = self.validate_ops(normalized)
        if errors:
            return {"accepted": False, "errors": errors, "applied": []}
        applied: list[dict[str, str]] = []
        for operation in normalized:
            op = str(operation.get("op", operation.get("type", "")) or "").casefold()
            occurrence_id = str(operation.get("occurrence_id", "") or "").strip()
            self.selected_occurrence_id = _apply_occurrence_op(
                self.states,
                selected_occurrence_id=self.selected_occurrence_id,
                op=op,
                occurrence_id=occurrence_id,
            )
            applied.append({"op": op, "occurrence_id": occurrence_id})
        if applied:
            self.revision += 1
        return {"accepted": True, "errors": [], "applied": applied}

    @property
    def viable_occurrence_ids(self) -> tuple[str, ...]:
        return tuple(
            occurrence_id
            for occurrence_id in self.current_visible_ids
            if self.states.get(occurrence_id, "active") != "eliminated"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "OccurrenceResolutionStateV1",
            "revision": self.revision,
            "current_visible_occurrence_ids": list(self.current_visible_ids),
            "candidates": [
                {
                    "occurrence_id": occurrence_id,
                    "status": self.states.get(occurrence_id, "active"),
                }
                for occurrence_id in self.current_visible_ids
            ],
            "viable_occurrence_ids": list(self.viable_occurrence_ids),
            "selected_occurrence_id": self.selected_occurrence_id or None,
        }

    def save(self, path: Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)


def _apply_occurrence_op(
    states: dict[str, str],
    *,
    selected_occurrence_id: str,
    op: str,
    occurrence_id: str,
) -> str:
    selected = selected_occurrence_id
    if op == "select":
        if selected and selected != occurrence_id and states.get(selected) == "selected":
            states[selected] = "active"
        states[occurrence_id] = "selected"
        return occurrence_id
    if op == "eliminate":
        states[occurrence_id] = "eliminated"
        return "" if selected == occurrence_id else selected
    if op == "reopen":
        states[occurrence_id] = "active"
        return "" if selected == occurrence_id else selected
    if states.get(occurrence_id) != "selected":
        states[occurrence_id] = "active"
    return selected


def validate_occurrence_method_configuration(
    *,
    method_arm: str,
    oracle_arm: str,
    oracle_intervention: str | Path | None,
) -> str:
    arm = str(method_arm or "none").strip().casefold()
    if arm not in OCCURRENCE_METHOD_ARMS:
        raise ValueError(f"unsupported occurrence method arm: {arm}")
    if arm != "none" and (
        str(oracle_arm or "o0").strip().casefold() != "o0"
        or bool(str(oracle_intervention or "").strip())
    ):
        raise ValueError(
            f"occurrence method arm {arm} requires O0 and no oracle intervention"
        )
    return arm


def assert_no_oracle_packet(packet: Mapping[str, Any], *, surface: str) -> None:
    forbidden_paths = tuple(sorted(_forbidden_key_paths(packet)))
    if forbidden_paths:
        raise ValueError(
            f"no-oracle runtime gate failed for {surface}: "
            + ", ".join(forbidden_paths)
        )


class OccurrencePacketTransform:
    """Audit or reshape occurrence evidence without changing retrieval."""

    def __init__(
        self,
        *,
        arm: str,
        audit_path: Path,
        candidate_limit: int = DEFAULT_CARD_CANDIDATE_LIMIT,
        excerpt_limit: int = DEFAULT_CARD_EXCERPT_LIMIT,
        excerpt_chars: int = DEFAULT_CARD_EXCERPT_CHARS,
        query_limit: int = DEFAULT_CARD_QUERY_LIMIT,
        query_chars: int = DEFAULT_CARD_QUERY_CHARS,
    ) -> None:
        normalized = validate_occurrence_method_configuration(
            method_arm=arm,
            oracle_arm="o0",
            oracle_intervention=None,
        )
        if normalized == "none":
            raise ValueError("OccurrencePacketTransform requires a method arm")
        self.arm = normalized
        self.audit_path = Path(audit_path)
        self.candidate_limit = max(1, int(candidate_limit))
        self.excerpt_limit = max(1, int(excerpt_limit))
        self.excerpt_chars = max(1, int(excerpt_chars))
        self.query_limit = max(1, int(query_limit))
        self.query_chars = max(1, int(query_chars))
        self._surface_checks: list[str] = []
        self._call_count = 0
        self._retrieval_parity_passed = True
        self._card_counts: list[int] = []
        self._representations: list[str] = []
        self._visible_excerpt_digests: list[str] = []
        self._retrieval_identity_digests: list[str] = []
        self._text_budget_parity_checks: list[dict[str, Any]] = []
        self._write_audit()

    @property
    def audit(self) -> Mapping[str, Any]:
        return self._audit_payload()

    def validate_surface(self, packet: Mapping[str, Any], *, surface: str) -> None:
        assert_no_oracle_packet(packet, surface=surface)
        self._surface_checks.append(str(surface))
        self._write_audit()

    def __call__(self, packet: Mapping[str, Any]) -> Mapping[str, Any]:
        assert_no_oracle_packet(packet, surface="caption_packet_before_transform")
        retrieval_before = _retrieval_identity(packet)
        representation = "identity"
        visible_excerpt_digest = stable_digest([])
        if self.arm == "a0":
            transformed: Mapping[str, Any] = packet
            card_count = 0
        else:
            transformed = dict(packet)
            occurrence_set = dict(transformed.get("occurrence_set") or {})
            cards = build_occurrence_candidate_cards(
                transformed,
                candidate_limit=self.candidate_limit,
                excerpt_limit=self.excerpt_limit,
                excerpt_chars=self.excerpt_chars,
                query_limit=self.query_limit,
                query_chars=self.query_chars,
            )
            flat_passages = flatten_occurrence_candidate_cards(cards)
            flat_queries = flatten_occurrence_candidate_queries(cards)
            grouped_text_digest = occurrence_visible_text_digest(cards=cards)
            flat_text_digest = occurrence_visible_text_digest(
                flat_passages=flat_passages,
                flat_queries=flat_queries,
            )
            grouped_text_chars = occurrence_visible_text_chars(cards=cards)
            flat_text_chars = occurrence_visible_text_chars(
                flat_passages=flat_passages,
                flat_queries=flat_queries,
            )
            text_budget_parity = {
                "grouped_text_digest": grouped_text_digest,
                "flat_text_digest": flat_text_digest,
                "grouped_text_chars": grouped_text_chars,
                "flat_text_chars": flat_text_chars,
                "passed": (
                    grouped_text_digest == flat_text_digest
                    and grouped_text_chars == flat_text_chars
                ),
            }
            if not text_budget_parity["passed"]:
                raise ValueError(
                    "grouped and flat occurrence text budgets differ"
                )
            occurrence_set["method_arm"] = self.arm
            if self.arm == "a1-flat":
                representation = "flat"
                occurrence_set["flat_candidate_passages"] = flat_passages
                occurrence_set["flat_candidate_queries"] = flat_queries
            else:
                representation = "grouped"
                occurrence_set["candidate_cards"] = cards
            occurrence_set["candidate_card_budget"] = self._card_budget()
            transformed["occurrence_set"] = occurrence_set
            card_count = len(cards)
            visible_excerpt_digest = candidate_card_excerpt_digest(cards)
            self._text_budget_parity_checks.append(text_budget_parity)
        retrieval_after = _retrieval_identity(transformed)
        parity_passed = retrieval_before == retrieval_after
        self._retrieval_parity_passed &= parity_passed
        if not parity_passed:
            raise ValueError("occurrence method transform changed retrieval identity")
        assert_no_oracle_packet(
            transformed, surface="caption_packet_after_transform"
        )
        self._call_count += 1
        self._card_counts.append(card_count)
        self._representations.append(representation)
        self._visible_excerpt_digests.append(visible_excerpt_digest)
        self._retrieval_identity_digests.append(retrieval_before)
        self._write_audit()
        return transformed

    def _card_budget(self) -> dict[str, int]:
        return {
            "candidate_limit": self.candidate_limit,
            "excerpt_limit_per_candidate": self.excerpt_limit,
            "excerpt_char_limit": self.excerpt_chars,
            "query_limit_per_candidate": self.query_limit,
            "query_char_limit": self.query_chars,
        }

    def _audit_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "MMLifelongNoOracleRuntimeAuditV2",
            "method_arm": self.arm,
            "no_oracle_runtime_gate_passed": True,
            "forbidden_agent_visible_keys": sorted(FORBIDDEN_AGENT_VISIBLE_KEYS),
            "forbidden_key_paths_seen": [],
            "surface_checks": list(self._surface_checks),
            "caption_packet_call_count": self._call_count,
            "retrieval_parity_passed": self._retrieval_parity_passed,
            "candidate_card_counts": list(self._card_counts),
            "candidate_card_budget": self._card_budget(),
            "representations": list(self._representations),
            "visible_excerpt_digests": list(self._visible_excerpt_digests),
            "retrieval_identity_digests": list(
                self._retrieval_identity_digests
            ),
            "text_budget_parity_checks": list(
                self._text_budget_parity_checks
            ),
            "text_budget_parity_passed": all(
                bool(row.get("passed"))
                for row in self._text_budget_parity_checks
            ),
        }

    def _write_audit(self) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.audit_path.with_name(f".{self.audit_path.name}.tmp")
        temporary.write_text(
            json.dumps(self._audit_payload(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.audit_path)


def build_occurrence_candidate_cards(
    packet: Mapping[str, Any],
    *,
    candidate_limit: int = DEFAULT_CARD_CANDIDATE_LIMIT,
    excerpt_limit: int = DEFAULT_CARD_EXCERPT_LIMIT,
    excerpt_chars: int = DEFAULT_CARD_EXCERPT_CHARS,
    query_limit: int = DEFAULT_CARD_QUERY_LIMIT,
    query_chars: int = DEFAULT_CARD_QUERY_CHARS,
) -> list[dict[str, Any]]:
    occurrence_set = packet.get("occurrence_set")
    if not isinstance(occurrence_set, Mapping):
        return []
    hits_by_passage: dict[str, dict[str, Any]] = {}
    for retrieval_rank, raw_hit in enumerate(
        tuple(packet.get("hits", ()) or ()), start=1
    ):
        if not isinstance(raw_hit, Mapping):
            continue
        hit = dict(raw_hit)
        passage_id = str(hit.get("passage_id", "") or "")
        if not passage_id:
            continue
        hit["_retrieval_rank"] = retrieval_rank
        hits_by_passage.setdefault(passage_id, hit)

    cards: list[dict[str, Any]] = []
    raw_candidates = tuple(occurrence_set.get("candidates", ()) or ())
    for raw_candidate in raw_candidates[: max(1, int(candidate_limit))]:
        if not isinstance(raw_candidate, Mapping):
            continue
        candidate = dict(raw_candidate)
        passage_ids = tuple(str(value) for value in candidate.get("passage_ids", ()))
        candidate_hits = sorted(
            (hits_by_passage[value] for value in passage_ids if value in hits_by_passage),
            key=lambda hit: int(hit.get("_retrieval_rank", 0) or 0),
        )
        representative_passages = [
            _representative_passage(
                hit,
                excerpt_chars=max(1, int(excerpt_chars)),
                query_limit=max(1, int(query_limit)),
                query_chars=max(1, int(query_chars)),
            )
            for hit in candidate_hits[: max(1, int(excerpt_limit))]
        ]
        cards.append(
            {
                "occurrence_id": str(candidate.get("occurrence_id", "") or ""),
                "rank": int(candidate.get("rank", len(cards) + 1) or len(cards) + 1),
                "time_range": list(candidate.get("time_range", ()) or ()),
                "source_video_ids": list(candidate.get("source_video_ids", ()) or ()),
                "matched_queries": _candidate_queries(
                    candidate,
                    candidate_hits,
                    limit=max(1, int(query_limit)),
                    char_limit=max(1, int(query_chars)),
                ),
                "representative_passages": representative_passages,
                "evidence_role": "locator_only",
                "answer_support": False,
            }
        )
    return cards


def candidate_cards_by_occurrence(
    occurrence_set: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        str(card.get("occurrence_id", "") or ""): dict(card)
        for card in tuple(occurrence_set.get("candidate_cards", ()) or ())
        if isinstance(card, Mapping) and str(card.get("occurrence_id", "") or "")
    }


def flatten_occurrence_candidate_cards(
    cards: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expose A1's excerpt budget without mechanically binding text to an occurrence."""
    flattened: list[dict[str, Any]] = []
    for card in cards:
        for passage in tuple(card.get("representative_passages", ()) or ()):
            if not isinstance(passage, Mapping):
                continue
            flattened.append(
                {
                    "passage_id": str(passage.get("passage_id", "") or ""),
                    "time_range": list(passage.get("time_range", ()) or ()),
                    "caption_excerpt": str(
                        passage.get("caption_excerpt", "") or ""
                    ),
                    "query_matches": list(passage.get("query_matches", ()) or ()),
                    "evidence_role": "locator_only",
                    "answer_support": False,
                }
            )
    return flattened


def flatten_occurrence_candidate_queries(
    cards: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for card in cards:
        for query in tuple(card.get("matched_queries", ()) or ()):
            text = str(query or "").strip()
            if not text:
                continue
            flattened.append(
                {
                    "query": text,
                    "evidence_role": "locator_only",
                    "answer_support": False,
                }
            )
    return flattened


def candidate_card_excerpt_digest(cards: Sequence[Mapping[str, Any]]) -> str:
    excerpts = [
            {
                "caption_excerpt": str(
                    passage.get("caption_excerpt", "") or ""
                ),
                "query_matches": list(passage.get("query_matches", ()) or ()),
            }
            for card in cards
            for passage in tuple(card.get("representative_passages", ()) or ())
            if isinstance(passage, Mapping)
        ]
    return occurrence_excerpt_digest(excerpts)


def occurrence_excerpt_digest(excerpts: Sequence[Mapping[str, Any]]) -> str:
    normalized = [
        {
            "caption_excerpt": str(item.get("caption_excerpt", "") or ""),
            "query_matches": list(item.get("query_matches", ()) or ()),
        }
        for item in excerpts
        if isinstance(item, Mapping)
    ]
    return stable_digest(sorted(normalized, key=stable_digest))


def occurrence_visible_text_digest(
    *,
    cards: Sequence[Mapping[str, Any]] = (),
    flat_passages: Sequence[Mapping[str, Any]] = (),
    flat_queries: Sequence[Mapping[str, Any]] = (),
) -> str:
    inventory = _occurrence_visible_text_inventory(
        cards=cards,
        flat_passages=flat_passages,
        flat_queries=flat_queries,
    )
    return stable_digest(sorted(inventory, key=stable_digest))


def _occurrence_visible_text_inventory(
    *,
    cards: Sequence[Mapping[str, Any]] = (),
    flat_passages: Sequence[Mapping[str, Any]] = (),
    flat_queries: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, str]]:
    inventory: list[dict[str, Any]] = []
    for card in cards:
        for query in tuple(card.get("matched_queries", ()) or ()):
            inventory.append({"kind": "candidate_query", "text": str(query or "")})
        for passage in tuple(card.get("representative_passages", ()) or ()):
            if not isinstance(passage, Mapping):
                continue
            inventory.append(
                {
                    "kind": "caption_excerpt",
                    "text": str(passage.get("caption_excerpt", "") or ""),
                }
            )
            inventory.extend(
                {"kind": "passage_query", "text": str(query or "")}
                for query in tuple(passage.get("query_matches", ()) or ())
            )
    for passage in flat_passages:
        if not isinstance(passage, Mapping):
            continue
        inventory.append(
            {
                "kind": "caption_excerpt",
                "text": str(passage.get("caption_excerpt", "") or ""),
            }
        )
        inventory.extend(
            {"kind": "passage_query", "text": str(query or "")}
            for query in tuple(passage.get("query_matches", ()) or ())
        )
    inventory.extend(
        {
            "kind": "candidate_query",
            "text": str(query.get("query", "") or ""),
        }
        for query in flat_queries
        if isinstance(query, Mapping)
    )
    return inventory


def occurrence_visible_text_chars(
    *,
    cards: Sequence[Mapping[str, Any]] = (),
    flat_passages: Sequence[Mapping[str, Any]] = (),
    flat_queries: Sequence[Mapping[str, Any]] = (),
) -> int:
    inventory = _occurrence_visible_text_inventory(
        cards=cards,
        flat_passages=flat_passages,
        flat_queries=flat_queries,
    )
    return sum(len(str(row["text"])) for row in inventory)


def _representative_passage(
    hit: Mapping[str, Any], *, excerpt_chars: int, query_limit: int, query_chars: int
) -> dict[str, Any]:
    metadata = hit.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    return {
        "passage_id": str(hit.get("passage_id", "") or ""),
        "time_range": [
            float(hit.get("virtual_start_sec", 0.0) or 0.0),
            float(hit.get("virtual_end_sec", 0.0) or 0.0),
        ],
        "caption_excerpt": str(hit.get("text", "") or "")[:excerpt_chars],
        "query_matches": _query_strings(
            metadata.get("query_matches") or metadata.get("matched_queries", ()),
            limit=query_limit,
            char_limit=query_chars,
        ),
    }


def _candidate_queries(
    candidate: Mapping[str, Any],
    hits: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    char_limit: int,
) -> list[str]:
    values: list[Any] = list(candidate.get("query_matches", ()) or ())
    for hit in hits:
        metadata = hit.get("metadata")
        if isinstance(metadata, Mapping):
            values.extend(
                tuple(
                    metadata.get("query_matches")
                    or metadata.get("matched_queries", ())
                    or ()
                )
            )
    return _query_strings(values, limit=limit, char_limit=char_limit)


def _query_strings(value: Any, *, limit: int, char_limit: int) -> list[str]:
    raw_values = (value,) if isinstance(value, (str, Mapping)) else tuple(value or ())
    queries: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        query = (
            str(raw.get("query", "") or "").strip()
            if isinstance(raw, Mapping)
            else str(raw or "").strip()
        )
        if query and query not in seen:
            seen.add(query)
            queries.append(query[:char_limit])
            if len(queries) >= limit:
                break
    return queries


def _retrieval_identity(packet: Mapping[str, Any]) -> str:
    occurrence_set = packet.get("occurrence_set")
    candidates = (
        occurrence_set.get("candidates")
        if isinstance(occurrence_set, Mapping)
        else None
    )
    return stable_digest(
        {
            "queries": packet.get("queries"),
            "time_range": packet.get("time_range"),
            "segment_ids": packet.get("segment_ids"),
            "source_video_ids": packet.get("source_video_ids"),
            "top_k": packet.get("top_k"),
            "index_digest": packet.get("index_digest"),
            "query_fingerprint": packet.get("query_fingerprint"),
            "hits": packet.get("hits"),
            "occurrence_candidates": candidates,
            "rendered": packet.get("rendered"),
        }
    )


def _forbidden_key_paths(value: Any, *, prefix: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}"
            if key.casefold() in FORBIDDEN_AGENT_VISIBLE_KEYS:
                paths.append(path)
            paths.extend(_forbidden_key_paths(child, prefix=path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            paths.extend(_forbidden_key_paths(child, prefix=f"{prefix}[{index}]"))
    return paths
