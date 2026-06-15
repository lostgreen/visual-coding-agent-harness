from visual_coding_agent_harness.agents.skills.playbook import (
    Playbook,
    StopDiagnostic,
    render_playbook_block,
)
from visual_coding_agent_harness.contracts import OptionSpec, TargetRegistry, TargetSpec


def test_render_playbook_block_includes_operator_targets_and_diagnostics():
    registry = TargetRegistry.from_specs(
        targets=[
            TargetSpec(target_id="T1", canonical_text="first option theme"),
            TargetSpec(target_id="T2", canonical_text="second option theme"),
        ],
        options=[
            OptionSpec(option_id="A", raw_option_text="A. first", target_sequence=("T1",)),
            OptionSpec(option_id="B", raw_option_text="B. second", target_sequence=("T2",)),
        ],
    )
    playbook = Playbook(
        skill_name="main_idea",
        answer_operator="main_arc",
        decomposition="Use global_gist only as a seed, then compare option-level breadth.",
        evidence_shape_target=("selected option has whole-video coverage",),
        investigation_hints=("global_gist is not an option vote",),
        unsafe_final_conditions=("global_gist-only answer",),
        stop_diagnostics=(
            StopDiagnostic(
                reason_code="arc_not_dominant",
                unmet_shape="top option does not dominate runner-up",
                repair_hint="compare top-2 option coverage",
            ),
        ),
        suggested_actions=("global_gist", "target_coverage"),
    )

    rendered = render_playbook_block(
        playbook,
        registry=registry,
        option_labels=("A. first", "B. second"),
        central_subjects=("overall arc",),
    )

    assert "## Skill Playbook: main_idea" in rendered
    assert "Operator: main_arc" in rendered
    assert "Suggested actions" in rendered
    assert "T1: first option theme" in rendered
    assert "A -> T1" in rendered
    assert "global_gist is not an option vote" in rendered
    assert "arc_not_dominant -> compare top-2 option coverage" in rendered


def test_render_playbook_block_respects_character_cap():
    playbook = Playbook(
        skill_name="grounded_factual_qa",
        answer_operator="select_present",
        decomposition="x" * 2000,
        evidence_shape_target=tuple(f"shape {index}" for index in range(20)),
        investigation_hints=tuple(f"hint {index}" for index in range(20)),
        unsafe_final_conditions=tuple(f"unsafe {index}" for index in range(20)),
        stop_diagnostics=tuple(
            StopDiagnostic(reason_code=f"r{index}", unmet_shape="missing", repair_hint="repair")
            for index in range(20)
        ),
    )

    rendered = render_playbook_block(playbook, max_chars=900)

    assert len(rendered) <= 900
    assert rendered.endswith("...")
