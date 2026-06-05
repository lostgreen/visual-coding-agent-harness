from visual_coding_agent_harness.agents.skills.specs import builtin_skill_registry, select_skill
from visual_coding_agent_harness.tools.inspector import _mutex_read_prompt


def test_main_idea_demands_agreement_between_two_gists():
    skill = select_skill("What is the video mainly about?")

    assert skill.name == "main_idea"
    assert skill.version == 1
    assert [step.op for step in skill.procedure].count("global_gist") == 2
    assert "two_global_gists_agree" in skill.sufficiency
    assert "option_coverage_margin_gt_0_15" in skill.sufficiency


def test_mutex_fact_skill_one_call_per_window():
    skill = builtin_skill_registry().get("mutex_fact_qa")
    read_steps = [step for step in skill.procedure if step.op == "vision_read"]

    assert len(read_steps) == 1
    assert read_steps[0].foreach == "mutex_windows"
    assert "{option_x_text}" in str(read_steps[0].args)
    assert "{option_y_text}" in str(read_steps[0].args)

    prompt = _mutex_read_prompt(
        segment_id="seg_0001",
        start_sec=0.0,
        end_sec=5.0,
        option_x="A",
        option_x_text="the statue is humble",
        option_y="B",
        option_y_text="the statue is upper-class",
    )

    assert "option A (`the statue is humble`) true" in prompt
    assert "option B (`the statue is upper-class`) true" in prompt
    assert "OR NEITHER true" in prompt
    assert "Cite only visible frames" in prompt
