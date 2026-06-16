from __future__ import annotations

from visual_coding_agent_harness.agents.grounding.compiler import compile_fallback_plan
from visual_coding_agent_harness.agents.grounding.discriminators import derive_discriminators_lexical


def test_fallback_discriminators_include_rise_fall_variants() -> None:
    options = {
        "A": "A cooking contest",
        "B": "A city traffic update",
        "C": "A sports highlight reel",
        "D": "The rise and fall of an ancient empire",
    }

    derived = derive_discriminators_lexical(options)

    assert "rise" in derived["D"]
    assert "fall" in derived["D"]


def test_compile_fallback_plan_attaches_option_discriminators() -> None:
    plan = compile_fallback_plan(
        "Question: What is the video mainly about?",
        [
            "A. A cooking contest",
            "B. A city traffic update",
            "C. A sports highlight reel",
            "D. The rise and fall of an ancient empire",
        ],
        route_hint="gist_global",
    )

    target_by_option = {
        option.option_id: next(target for target in plan.targets if target.target_key in option.required_target_keys)
        for option in plan.options
    }

    assert "rise" in target_by_option["D"].discriminators
    assert "fall" in target_by_option["D"].discriminators
