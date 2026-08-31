#!/usr/bin/env python3
"""Freeze the WP17 slot-construction development endpoint analysis protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from vcah.occurrence_negative_sidecar import file_sha256


ANALYSIS_PROTOCOL_CONTRACT = "WP17-slot-construction-endpoint-analysis-v1"


def build_protocol(
    field_spec: Mapping[str, Any],
    timeline_spec: Mapping[str, Any],
    posthoc_spec: Mapping[str, Any],
    *,
    field_spec_sha256: str,
    timeline_spec_sha256: str,
    posthoc_spec_sha256: str,
    construction_protocol_sha256: str,
    source_commit: str,
) -> dict[str, Any]:
    field_cases = dict(field_spec.get("cases", {}) or {})
    timeline_cases = dict(timeline_spec.get("cases", {}) or {})
    posthoc_cases = dict(posthoc_spec.get("cases", {}) or {})
    if len(field_cases) != 10 or len(posthoc_cases) != 1 or len(timeline_cases) != 11:
        raise ValueError("WP17 endpoint analysis expects 10 frozen plus 1 post-hoc case")

    cases = {}
    for case_id, raw in sorted(field_cases.items()):
        row = dict(raw)
        if case_id not in timeline_cases:
            raise ValueError("WP17 field annotation is outside the local timeline")
        cases[case_id] = {
            "anchor_intervals": list(timeline_cases[case_id]["anchor_intervals"]),
            "entity_terms": list(row["entity"]["document_terms"]),
            "event_terms": list(row["event"]["document_terms"]),
            "state_terms": list(row["state"]["document_terms"]),
            "occurrence_terms": [],
            "ordinal_terms": [],
            "annotation_timing": "frozen_before_construction_outcomes",
        }
    for case_id, raw in posthoc_cases.items():
        if case_id not in timeline_cases or case_id in cases:
            raise ValueError("WP17 post-hoc annotation case is invalid")
        row = dict(raw)
        cases[case_id] = {
            "anchor_intervals": list(timeline_cases[case_id]["anchor_intervals"]),
            "entity_terms": list(row["entity_terms"]),
            "event_terms": list(row["event_terms"]),
            "state_terms": list(row["state_terms"]),
            "occurrence_terms": list(row.get("occurrence_terms", ())),
            "ordinal_terms": list(row.get("ordinal_terms", ())),
            "annotation_timing": "post_hoc_before_merged_completion_and_endpoint_read",
        }

    return {
        "schema_version": "MMLifelongWP17SlotEndpointAnalysisProtocolV1",
        "contract": ANALYSIS_PROTOCOL_CONTRACT,
        "source_commit": str(source_commit),
        "construction_protocol_sha256": str(construction_protocol_sha256),
        "annotation_sources": {
            "frozen10_field_spec_sha256": str(field_spec_sha256),
            "local_timeline_sha256": str(timeline_spec_sha256),
            "posthoc_annotation_sha256": str(posthoc_spec_sha256),
        },
        "annotation_timing": {
            "frozen10": "frozen before construction outcomes",
            "case_0115": (
                "post-hoc after the parent run produced partial outputs, but before "
                "merged completion and before endpoint values were inspected"
            ),
            "unbiased_publication_claim_allowed": False,
            "development_diagnostic_only": True,
        },
        "unbiased_publication_claim_allowed": False,
        "matching_policy": {
            "artifact": "trusted structured_event_record only",
            "transaction_abstain": "score_zero_not_excluded",
            "normalization": "unicode_nfkc_casefold_punctuation_and_underscore_to_space",
            "term_match": "normalized_contiguous_substring",
            "canonical_entity_coverage": "any entity term",
            "event_coverage": "any event term",
            "relation_state_coverage": "any state term",
            "anchor_representation_coverage": "entity AND event",
            "occurrence_coverage": "any occurrence term on annotated cases",
            "ordinal_accuracy": "any ordinal term on annotated cases",
        },
        "endpoints": [
            "anchor_representation_coverage",
            "canonical_entity_coverage",
            "relation_state_coverage",
            "occurrence_coverage",
            "ordinal_accuracy",
            "mechanical_provenance_reference_validity",
            "unsupported_state_write_reference_rate",
            "history_context_token_distribution",
            "slot_churn_rate",
            "transaction_abstain_rate",
            "model_and_storage_cost",
        ],
        "effects": {
            "primary": "e1c2-e1c1",
            "secondary": "e1c1-e1c0",
            "bootstrap_samples": 10000,
            "bootstrap_seed": 20260831,
            "bootstrap_unit": "case",
            "report": ["paired_delta", "bootstrap_ci95", "wins_ties_losses"],
        },
        "decision_rule": {
            "go": (
                "e1c2-e1c1 is positive on at least two representation/state "
                "endpoints, mechanical provenance does not degrade, unsupported "
                "state-write references remain zero, and all context stays <=600 tokens"
            ),
            "no_go": (
                "e1c2-e1c1 is not positive on at least two endpoints or provenance, "
                "unsupported writes, or the context budget degrades"
            ),
            "visual_provenance_required_for_final_claim": True,
            "post_hoc_0115_occurrence_ordinal_can_drive_decision": False,
        },
        "cases": cases,
        "case_count": len(cases),
        "endpoint_values_evaluated": False,
        "day_test140_accessed": False,
        "week_accessed": False,
        "model_calls": 0,
    }


def run(args: argparse.Namespace) -> Path:
    field_path = Path(args.field_spec)
    timeline_path = Path(args.timeline_spec)
    posthoc_path = Path(args.posthoc_annotation)
    construction_protocol_path = Path(args.construction_protocol)
    if file_sha256(construction_protocol_path) != str(
        args.expected_construction_protocol_sha256
    ):
        raise ValueError("WP17 analysis construction protocol SHA mismatch")
    protocol = build_protocol(
        _read_json(field_path),
        _read_json(timeline_path),
        _read_json(posthoc_path),
        field_spec_sha256=file_sha256(field_path),
        timeline_spec_sha256=file_sha256(timeline_path),
        posthoc_spec_sha256=file_sha256(posthoc_path),
        construction_protocol_sha256=file_sha256(construction_protocol_path),
        source_commit=str(args.source_commit),
    )
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("WP17 endpoint analysis protocol already exists")
    _write_json_atomic(out, protocol)
    print(
        "WP17_SLOT_ENDPOINT_PROTOCOL_FROZEN "
        "cases=11 frozen10=10 posthoc0115=1 endpoints=false model_calls=0",
        flush=True,
    )
    return out


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--field-spec", required=True)
    parser.add_argument("--timeline-spec", required=True)
    parser.add_argument("--posthoc-annotation", required=True)
    parser.add_argument("--construction-protocol", required=True)
    parser.add_argument("--expected-construction-protocol-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
