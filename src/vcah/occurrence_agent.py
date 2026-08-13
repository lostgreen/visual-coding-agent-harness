from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from vcah.caption_schema import stable_digest


OCCURRENCE_METHOD_ARMS = ("none", "a0", "a1")
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
    """Audit A0 packets or enrich A1 packets without changing retrieval."""

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
            raise ValueError("OccurrencePacketTransform requires a0 or a1")
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
            occurrence_set["method_arm"] = self.arm
            occurrence_set["candidate_cards"] = cards
            occurrence_set["candidate_card_budget"] = self._card_budget()
            transformed["occurrence_set"] = occurrence_set
            card_count = len(cards)
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
            "schema_version": "MMLifelongNoOracleRuntimeAuditV1",
            "method_arm": self.arm,
            "no_oracle_runtime_gate_passed": True,
            "forbidden_agent_visible_keys": sorted(FORBIDDEN_AGENT_VISIBLE_KEYS),
            "forbidden_key_paths_seen": [],
            "surface_checks": list(self._surface_checks),
            "caption_packet_call_count": self._call_count,
            "retrieval_parity_passed": self._retrieval_parity_passed,
            "candidate_card_counts": list(self._card_counts),
            "candidate_card_budget": self._card_budget(),
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
