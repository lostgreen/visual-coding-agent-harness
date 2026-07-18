from __future__ import annotations

from typing import Any, Mapping, Sequence


ADMISSIBLE_KINDS = frozenset({"direct_witness", "deterministic_derivation"})
PROVENANCE_KINDS = ADMISSIBLE_KINDS | frozenset({"heuristic"})


def normalize_provenance(value: Any) -> tuple[dict[str, Any], ...]:
    rows = value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else (value,)
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        kind = str(row.get("kind", "heuristic") or "heuristic").strip().casefold()
        if kind not in PROVENANCE_KINDS:
            kind = "heuristic"
        normalized.append(
            {
                "kind": kind,
                "source_fact_ids": _strings(row.get("source_fact_ids", ())),
                "source_evidence_ids": _strings(row.get("source_evidence_ids", row.get("evidence_ids", ()))),
                "derivation": str(row.get("derivation", "") or ""),
                "witness_ranges": _ranges(row.get("witness_ranges", ())),
                "producer": str(row.get("producer", "") or ""),
                "producer_revision": str(row.get("producer_revision", "") or ""),
                "source_kinds": _strings(row.get("source_kinds", ())),
            }
        )
    return tuple(normalized)


def direct_witness_provenance(
    *,
    fact_ids: Sequence[str],
    evidence_ids: Sequence[str],
    witness_ranges: Sequence[Sequence[float]],
    producer: str = "observation",
) -> dict[str, Any]:
    return {
        "kind": "direct_witness",
        "source_fact_ids": [str(item) for item in fact_ids if str(item)],
        "source_evidence_ids": [str(item) for item in evidence_ids if str(item)],
        "witness_ranges": [list(item) for item in witness_ranges if len(item) == 2],
        "producer": producer,
    }


def deterministic_derivation_provenance(
    *,
    derivation: str,
    fact_ids: Sequence[str],
    evidence_ids: Sequence[str],
    source_kinds: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "kind": "deterministic_derivation",
        "source_fact_ids": [str(item) for item in fact_ids if str(item)],
        "source_evidence_ids": [str(item) for item in evidence_ids if str(item)],
        "derivation": str(derivation or ""),
        "producer": "program",
        "source_kinds": [str(item) for item in source_kinds if str(item)],
    }


def heuristic_provenance(
    *,
    fact_ids: Sequence[str] = (),
    evidence_ids: Sequence[str] = (),
    derivation: str = "",
    producer: str = "model",
) -> dict[str, Any]:
    return {
        "kind": "heuristic",
        "source_fact_ids": [str(item) for item in fact_ids if str(item)],
        "source_evidence_ids": [str(item) for item in evidence_ids if str(item)],
        "derivation": str(derivation or ""),
        "producer": producer,
    }


def provenance_is_admissible(value: Any) -> bool:
    rows = normalize_provenance(value)
    if not rows:
        return False
    for row in rows:
        kind = row["kind"]
        if kind == "direct_witness":
            if row["source_evidence_ids"] and row["witness_ranges"]:
                return True
        elif kind == "deterministic_derivation":
            source_kinds = set(row["source_kinds"])
            if (
                row["derivation"]
                and row["source_fact_ids"]
                and row["source_evidence_ids"]
                and source_kinds
                and not source_kinds.difference(ADMISSIBLE_KINDS)
            ):
                return True
    return False


def provenance_kinds(value: Any) -> tuple[str, ...]:
    return tuple(dict.fromkeys(row["kind"] for row in normalize_provenance(value)))


def _strings(value: Any) -> list[str]:
    values = (value,) if isinstance(value, str) else tuple(value or ())
    return list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def _ranges(value: Any) -> list[list[float]]:
    rows = value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()
    normalized = []
    for item in rows:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
            continue
        try:
            start, end = sorted((float(item[0]), float(item[1])))
        except (TypeError, ValueError):
            continue
        if end >= start:
            normalized.append([start, end])
    return normalized
