import pytest

from visual_coding_agent_harness.agents.skills.specs import (
    EvidenceFollowupKind,
    ExplorationProfile,
    FinalGateProfile,
    FinalizationStrategy,
    OptionEvaluationKind,
    PrefinalRepairKind,
    RouteRepairPolicyKind,
    SchedulerKind,
    SkillBehaviors,
    SkillRegistry,
    SkillSpec,
    SkillTrigger,
    builtin_skill_registry,
    select_skill,
    skill_catalog_prompt,
)
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
    suggested_actions = _catalog_field(rendered, "main_idea", "suggested_actions")

    assert "global_gist" not in suggested_actions.split("(", 1)[0]
    assert "(global_gist=exhausted)" in suggested_actions


def test_skill_catalog_keeps_one_shot_tool_when_available():
    rendered = skill_catalog_prompt()

    assert "global_gist" in _catalog_field(rendered, "main_idea", "suggested_actions")


def test_skill_registry_init_rejects_duplicate_names():
    skill = SkillSpec(
        name="duplicate",
        version=1,
        trigger=SkillTrigger(route="needle_local"),
        input_slots=(),
        procedure=(),
        sufficiency=(),
        verifier_checks=(),
    )

    with pytest.raises(ValueError, match="Skill already registered: duplicate"):
        SkillRegistry([skill, skill])


def test_builtin_skill_registry_has_expected_unique_skill_names():
    assert set(builtin_skill_registry().names()) == {
        "general_exploration",
        "grounded_factual_qa",
        "main_idea",
        "complement_absence_qa",
        "causal_asr_qa",
        "universal_set_qa",
        "mutex_fact_qa",
        "timeline_ordering",
        "narration_timeline_qa",
        "visual_timeline_qa",
        "mixed_asr_visual_qa",
    }


def test_grounded_factual_playbook_retains_merged_final_check():
    skill = builtin_skill_registry().get("grounded_factual_qa")

    assert skill.recovery["insufficient"]["target"] == "distinguishing fact window"
    assert set(skill.self_check) == {
        "final citations are mem_ ids",
        "cited memory entries have real anchors",
        "use visual anchors when the selected claim is visually observable; use ASR/OCR/caption anchors when the selected claim is narrated or textual",
    }


def test_skill_spec_decomposition():
    for skill in builtin_skill_registry().list():
        assert skill.guide.name == skill.name
        assert skill.guide.version == skill.version
        assert skill.guide.description
        assert skill.guide.when_to_use
        assert skill.policy.allowed_modalities
        assert skill.policy.verifier_checks == tuple(skill.verifier_checks)


def test_builtin_skills_declare_typed_behaviors() -> None:
    expected = {
        "grounded_factual_qa": SkillBehaviors(
            exploration_profile=ExplorationProfile.GROUNDED_FACTUAL,
            option_evaluation=OptionEvaluationKind.MUTEX_OR_GROUNDED,
        ),
        "main_idea": SkillBehaviors(
            finalization=FinalizationStrategy.WHOLE_VIDEO_COVERAGE,
            exploration_profile=ExplorationProfile.MAIN_IDEA,
            route_repair=RouteRepairPolicyKind.GIST_FAMILY,
        ),
        "narration_timeline_qa": SkillBehaviors(
            evidence_followup=EvidenceFollowupKind.SEGMENT_DETAIL_AND_ASR,
            exploration_profile=ExplorationProfile.TIMELINE_FAMILY,
            final_gate=FinalGateProfile.NARRATION_EXTRA_HINTS,
            prefinal_repair=PrefinalRepairKind.NARRATION_TIMELINE,
        ),
        "visual_timeline_qa": SkillBehaviors(
            exploration_profile=ExplorationProfile.TIMELINE_FAMILY,
            final_gate=FinalGateProfile.TIMELINE_FAMILY_HINTS,
        ),
        "timeline_ordering": SkillBehaviors(
            scheduler=SchedulerKind.SUBEVENT_TIMELINE,
            exploration_profile=ExplorationProfile.TIMELINE_FAMILY,
            final_gate=FinalGateProfile.TIMELINE_FAMILY_HINTS,
        ),
        "mutex_fact_qa": SkillBehaviors(
            scheduler=SchedulerKind.FOLLOWUP_QUEUE,
            exploration_profile=ExplorationProfile.GROUNDED_FACTUAL,
            option_evaluation=OptionEvaluationKind.MUTEX_OR_GROUNDED,
        ),
    }
    default_names = {
        "mixed_asr_visual_qa",
        "causal_asr_qa",
        "complement_absence_qa",
        "universal_set_qa",
        "general_exploration",
    }
    registry = builtin_skill_registry()

    for name, behaviors in expected.items():
        assert registry.get(name).behaviors == behaviors
    for name in default_names:
        assert registry.get(name).behaviors == SkillBehaviors()


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


def _catalog_field(rendered: str, skill_name: str, field_name: str) -> str:
    lines = rendered.splitlines()
    start = lines.index(next(line for line in lines if line.startswith(f"- {skill_name}@")))
    for line in lines[start + 1 :]:
        if line.startswith("- "):
            break
        prefix = f"  {field_name}: "
        if line.startswith(prefix):
            return line[len(prefix) :]
    raise AssertionError(f"{field_name} not found for {skill_name}")
