from __future__ import annotations

from visual_coding_agent_harness.agents.context_budget import default_context_budget_allocator
from visual_coding_agent_harness.agents.iterative_agent import AgentBudget
from visual_coding_agent_harness.agents.prompt_stack import compose_planner_prompts
from visual_coding_agent_harness.video_index import fixed_window_scene_index


def test_system_prompt_byte_stable_across_rounds() -> None:
    scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=10.0, window_sec=5.0)
    budget = AgentBudget(max_rounds=5, prompt_role_split_enabled=True)
    allocator = default_context_budget_allocator(total_budget_tokens=12000)

    prompts = [
        compose_planner_prompts(
            prompt_role_split_enabled=True,
            question="What is visible?",
            scene_index=scene_index,
            ledger_text=f"# Compact Evidence Context\n- obs_{round_number:04d}",
            round_number=round_number,
            budget=budget,
            allocator=allocator,
            active_skill="visual_timeline_qa@v1",
            route="needle_local",
        )
        for round_number in range(1, 6)
    ]

    assert len({prompt.system_prompt for prompt in prompts}) == 1
    assert len({prompt.user_prompt for prompt in prompts}) == 5
