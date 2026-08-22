from __future__ import annotations

from typing import Any, Mapping, Sequence

from vcah.caption_context import CAPTION_CONTEXT_CONTRACT
from vcah.caption_occurrence import build_caption_occurrence_set
from vcah.caption_schema import CaptionHitV1, stable_digest


CAPTION_EVIDENCE_BUNDLE_SCHEMA_VERSION = "CaptionEvidenceBundleSetV1"


def build_caption_evidence_bundle_set(
    hits: Sequence[CaptionHitV1 | Mapping[str, Any]],
) -> dict[str, Any]:
    """Group retrieval seeds with related context while preserving member events."""

    rows = tuple(_hit_row(hit) for hit in hits)
    seed_rows = tuple(
        row
        for row in rows
        if row["metadata"].get("context_expansion_contract")
        != CAPTION_CONTEXT_CONTRACT
    )
    seed_occurrences = build_caption_occurrence_set(seed_rows)
    bundles: list[dict[str, Any]] = []
    for candidate in tuple(seed_occurrences.get("candidates", ()) or ()):
        seed_ids = tuple(str(value) for value in candidate.get("passage_ids", ()))
        seed_id_set = set(seed_ids)
        context_rows = tuple(
            row
            for row in rows
            if row["passage_id"] not in seed_id_set
            and seed_id_set
            & set(row["metadata"].get("context_seed_passage_ids", ()) or ())
        )
        members = tuple(
            sorted(
                (
                    *(row for row in seed_rows if row["passage_id"] in seed_id_set),
                    *context_rows,
                ),
                key=lambda row: (
                    row["virtual_start_sec"],
                    row["virtual_end_sec"],
                    row["passage_id"],
                ),
            )
        )
        member_ids = tuple(row["passage_id"] for row in members)
        bundle_digest = stable_digest(
            {
                "seed_occurrence_id": candidate.get("occurrence_id"),
                "seed_passage_ids": sorted(seed_ids),
                "member_passage_ids": sorted(member_ids),
            }
        )
        bundles.append(
            {
                "bundle_id": f"bundle_{bundle_digest[:20]}",
                "rank": int(candidate.get("rank", len(bundles) + 1) or 1),
                "seed_occurrence_id": str(candidate.get("occurrence_id", "") or ""),
                "seed_passage_ids": list(seed_ids),
                "context_passage_ids": [
                    row["passage_id"] for row in context_rows
                ],
                "member_passage_ids": list(member_ids),
                "member_passages": [
                    {
                        "passage_id": row["passage_id"],
                        "caption_id": row["caption_id"],
                        "time_range": [
                            row["virtual_start_sec"],
                            row["virtual_end_sec"],
                        ],
                        "role": (
                            "seed"
                            if row["passage_id"] in seed_id_set
                            else "context"
                        ),
                        "cross_caption": bool(
                            row["metadata"].get("cross_caption", False)
                        ),
                    }
                    for row in members
                ],
                "time_range": [
                    min(
                        (row["virtual_start_sec"] for row in members),
                        default=float(candidate.get("time_range", [0.0, 0.0])[0]),
                    ),
                    max(
                        (row["virtual_end_sec"] for row in members),
                        default=float(candidate.get("time_range", [0.0, 0.0])[-1]),
                    ),
                ],
                "source_video_ids": sorted(
                    {
                        str(value)
                        for row in members
                        for value in tuple(
                            row["metadata"].get("source_video_ids", ()) or ()
                        )
                        if str(value)
                    }
                ),
                "context_count": len(context_rows),
                "cross_caption_context_count": sum(
                    bool(row["metadata"].get("cross_caption", False))
                    for row in context_rows
                ),
                "event_boundaries_preserved": True,
                "semantic_claim": "temporally_related_evidence_not_single_event",
            }
        )
    bundles.sort(key=lambda row: (int(row["rank"]), str(row["bundle_id"])))
    return {
        "schema_version": CAPTION_EVIDENCE_BUNDLE_SCHEMA_VERSION,
        "context_contract": CAPTION_CONTEXT_CONTRACT,
        "candidate_count": len(bundles),
        "status": "empty" if not bundles else "ready",
        "bundles": bundles,
    }


def _hit_row(hit: CaptionHitV1 | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(hit, CaptionHitV1):
        return {
            "passage_id": hit.passage_id,
            "caption_id": hit.caption_id,
            "rank": hit.rank,
            "fused_score": hit.fused_score,
            "virtual_start_sec": hit.virtual_start_sec,
            "virtual_end_sec": hit.virtual_end_sec,
            "metadata": dict(hit.metadata),
        }
    raw_range = tuple(hit.get("range", ()) or ())
    start = float(
        raw_range[0]
        if len(raw_range) == 2
        else hit.get("virtual_start_sec", 0.0) or 0.0
    )
    end = float(
        raw_range[1]
        if len(raw_range) == 2
        else hit.get("virtual_end_sec", start) or start
    )
    return {
        "passage_id": str(hit.get("passage_id", "") or ""),
        "caption_id": str(hit.get("caption_id", "") or ""),
        "rank": max(1, int(hit.get("rank", 1) or 1)),
        "fused_score": float(hit.get("fused_score", 0.0) or 0.0),
        "virtual_start_sec": min(start, end),
        "virtual_end_sec": max(start, end),
        "metadata": dict(hit.get("metadata") or {}),
    }
