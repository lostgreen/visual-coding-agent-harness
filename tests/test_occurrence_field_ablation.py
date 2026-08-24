from __future__ import annotations

from vcah.occurrence_field_ablation import build_field_ablation_report


VARIANTS = (
    "caption_only_hybrid",
    "caption_plus_entity_hybrid",
    "caption_plus_event_hybrid",
    "caption_plus_state_hybrid",
    "caption_plus_all_hybrid",
    "field_rrf_hybrid",
)


def _case(case_id: str, ranks: dict[str, int | None]) -> dict[str, object]:
    return {
        "case_id": case_id,
        "anchor_description": "anchor",
        "oracle_passage_id": f"passage-{case_id}",
        "field_names": ["entity", "event", "state"],
        "fields_frozen_before_ablation_outcomes": True,
        "oracle_gold_occurrence_only": True,
        "target_evidence_and_answer_excluded": True,
        "ranks": ranks,
    }


def test_report_distinguishes_concat_from_field_separation() -> None:
    cases = (
        _case(
            "mmlifelong-game-test-0001",
            {
                "caption_only_hybrid": None,
                "caption_plus_entity_hybrid": 3,
                "caption_plus_event_hybrid": None,
                "caption_plus_state_hybrid": None,
                "caption_plus_all_hybrid": 6,
                "field_rrf_hybrid": 1,
            },
        ),
        _case(
            "mmlifelong-game-test-0002",
            {
                "caption_only_hybrid": None,
                "caption_plus_entity_hybrid": None,
                "caption_plus_event_hybrid": 4,
                "caption_plus_state_hybrid": None,
                "caption_plus_all_hybrid": 7,
                "field_rrf_hybrid": 1,
            },
        ),
    )
    report = build_field_ablation_report(
        cases,
        expected_cases=2,
        variant_order=VARIANTS,
    )
    assert report["structural_gate_passed"] is True
    assert report["decision"] == "ORACLE_FIELD_SEPARATION_REQUIRED"
    assert report["diagnostics"]["single_field_hybrid_at5"] == {
        "entity": 1,
        "event": 1,
        "state": 0,
    }
    comparison = report["comparisons"]["field_rrf_hybrid_vs_caption_only_hybrid_at5"]
    assert comparison["net_recovery"] == 2


def test_invalid_rank_fails_gate_without_crashing() -> None:
    ranks = {variant: None for variant in VARIANTS}
    ranks["caption_only_hybrid"] = 0
    report = build_field_ablation_report(
        (_case("mmlifelong-game-test-0001", ranks),),
        expected_cases=1,
        variant_order=VARIANTS,
    )
    assert report["structural_gate_passed"] is False
    assert report["structural_checks"]["rank_domain_valid"] is False
