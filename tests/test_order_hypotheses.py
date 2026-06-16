from __future__ import annotations

from visual_coding_agent_harness.agents.grounding.compiler import compile_fallback_plan, compile_grounding_plan


OPTIONS = [
    'A. "The rape of Persephone", "Apollo and Daphne", "David", "Aeneas"',
    'B. "David", "Aeneas", "Apollo and Daphne", "The rape of Persephone"',
    'C. "Apollo and Daphne", "Aeneas", "David", "The rape of Persephone"',
    'D. "Aeneas", "David", "The rape of Persephone", "Apollo and Daphne"',
]


def test_timeline_same_entities_compile_to_ordered_set() -> None:
    plan = compile_fallback_plan("Which order are the artworks presented in?", OPTIONS, route_hint="temporal_order")
    compiled = compile_grounding_plan(plan, raw_options={option[0]: option[3:] for option in OPTIONS})

    ordered_set = compiled.ordered_sets[0]
    names_by_id = {entity.entity_id: entity.canonical_name for entity in ordered_set.entities}
    d_hypothesis = next(hypothesis for hypothesis in ordered_set.hypotheses if hypothesis.option_id == "D")

    assert names_by_id == {
        "E1": "Aeneas",
        "E2": "David",
        "E3": "The rape of Persephone",
        "E4": "Apollo and Daphne",
    }
    assert d_hypothesis.ordered_entity_ids == ("E1", "E2", "E3", "E4")


def test_timeline_same_entities_do_not_compile_option_sentence_targets() -> None:
    plan = compile_fallback_plan("Which order are the artworks presented in?", OPTIONS, route_hint="temporal_order")

    canonical_claims = [target.canonical_claim.lower() for target in plan.targets]

    assert plan.ordered_sets
    assert not any("presented first" in claim or "presented second" in claim for claim in canonical_claims)
