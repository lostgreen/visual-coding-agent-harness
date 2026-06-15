from visual_coding_agent_harness.agents.skills.specs import builtin_skill_registry, select_skill, skill_catalog_prompt
from visual_coding_agent_harness.tools.inspector import _mutex_read_prompt


def test_main_idea_uses_global_gist_as_topic_hint_not_option_floor():
    skill = select_skill("What is the video mainly about?")

    assert skill.name == "main_idea"
    assert skill.version == 1
    assert [step.op for step in skill.procedure].count("global_gist") == 1
    assert "whole_video_coverage_evidence" in skill.sufficiency
    assert "localized_or_indexed_fact_support" in skill.sufficiency
    assert "global_gist is not an option vote" in skill.self_check


def test_skill_catalog_redacts_exhausted_one_shot_tool():
    rendered = skill_catalog_prompt(exhausted_tools=frozenset({"global_gist"}))
    main_idea_line = next(line for line in rendered.splitlines() if line.startswith("- main_idea@"))
    suggested_actions = main_idea_line.split("suggested_actions=", 1)[1].split(";", 1)[0]

    assert "global_gist" not in suggested_actions.split("(", 1)[0]
    assert "(global_gist=exhausted)" in main_idea_line


def test_skill_catalog_keeps_one_shot_tool_when_available():
    rendered = skill_catalog_prompt()
    main_idea_line = next(line for line in rendered.splitlines() if line.startswith("- main_idea@"))

    assert "global_gist" in main_idea_line


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
