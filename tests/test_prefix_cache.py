from __future__ import annotations

from pathlib import Path

from visual_coding_agent_harness.agents.context_budget import default_context_budget_allocator
from visual_coding_agent_harness.agents.iterative_agent import AgentBudget, IterativeVisualAgent
from visual_coding_agent_harness.agents.prompt_frames import PromptFrameLedger
from visual_coding_agent_harness.agents.prompt_stack import compose_planner_prompts
from visual_coding_agent_harness.agents.runtime_capabilities import PromptRetentionMode
from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse
from visual_coding_agent_harness.registry import ToolRegistry
from visual_coding_agent_harness.video_index import fixed_window_scene_index
from visual_coding_agent_harness.workspace import EvidenceWorkspace


class PromptRecordingBackend:
    def __init__(self, *, prefix_cache: bool, persistent_conversation: bool) -> None:
        self.capabilities = {
            "prefix_cache": prefix_cache,
            "persistent_conversation": persistent_conversation,
        }
        self.requests: list[BackendRequest] = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        if request.task == "replan":
            return BackendResponse(text='{"status":"continue","program":[]}')
        return BackendResponse(
            text='{"answer":"need_more_evidence","citations":[],"missing_evidence":["fixture"],"confidence":0.0}'
        )


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


def test_prefix_cached_frame_retention_keeps_full_active_skill_body_each_round() -> None:
    scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=10.0, window_sec=5.0)
    budget = AgentBudget(max_rounds=3, prompt_role_split_enabled=True)
    allocator = default_context_budget_allocator(total_budget_tokens=12000)
    ledger = PromptFrameLedger(mode=PromptRetentionMode.PREFIX_CACHED_FULL)

    prompts = [
        compose_planner_prompts(
            prompt_role_split_enabled=True,
            question="What is the main idea of the video?",
            scene_index=scene_index,
            ledger_text=f"# Compact Evidence Context\n- obs_{round_number:04d}",
            round_number=round_number,
            budget=budget,
            allocator=allocator,
            active_skill="main_idea@v1",
            route="gist_global",
            prompt_frame_ledger=ledger,
        )
        for round_number in (1, 2)
    ]

    assert "global_gist is not an option vote" in prompts[0].system_prompt
    assert "global_gist is not an option vote" in prompts[1].system_prompt
    assert "[loaded" not in prompts[1].system_prompt


def test_sticky_frame_retention_may_reference_active_skill_after_initial_body() -> None:
    scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=10.0, window_sec=5.0)
    budget = AgentBudget(max_rounds=3, prompt_role_split_enabled=True)
    allocator = default_context_budget_allocator(total_budget_tokens=12000)
    ledger = PromptFrameLedger(mode=PromptRetentionMode.STICKY_REFERENCE)

    first = compose_planner_prompts(
        prompt_role_split_enabled=True,
        question="What is the main idea of the video?",
        scene_index=scene_index,
        ledger_text="# Compact Evidence Context\n- obs_0001",
        round_number=1,
        budget=budget,
        allocator=allocator,
        active_skill="main_idea@v1",
        route="gist_global",
        prompt_frame_ledger=ledger,
    )
    second = compose_planner_prompts(
        prompt_role_split_enabled=True,
        question="What is the main idea of the video?",
        scene_index=scene_index,
        ledger_text="# Compact Evidence Context\n- obs_0002",
        round_number=2,
        budget=budget,
        allocator=allocator,
        active_skill="main_idea@v1",
        route="gist_global",
        prompt_frame_ledger=ledger,
    )

    assert "global_gist is not an option vote" in first.system_prompt
    assert "global_gist is not an option vote" not in second.system_prompt
    assert "# Skill Catalog [loaded" in second.system_prompt


def test_iterative_agent_applies_prompt_frame_retention_from_backend_capabilities(tmp_path: Path) -> None:
    scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=10.0, window_sec=5.0)

    cases = [
        (False, False, False),
        (True, False, False),
        (True, True, True),
    ]
    for prefix_cache, persistent_conversation, expect_reference in cases:
        backend = PromptRecordingBackend(
            prefix_cache=prefix_cache,
            persistent_conversation=persistent_conversation,
        )
        workspace = EvidenceWorkspace.create(
            tmp_path,
            run_id=f"prompt_retention_{prefix_cache}_{persistent_conversation}",
        )
        agent = IterativeVisualAgent(
            backend=backend,
            registry=ToolRegistry(),
            workspace=workspace,
            scene_index=scene_index,
            budget=AgentBudget(max_rounds=2, reserve_final_round=False, prompt_role_split_enabled=True),
        )

        agent.run(question="What is visible?", video_path="/videos/demo.mp4")

        replan_requests = [request for request in backend.requests if request.task == "replan"]
        assert len(replan_requests) == 2
        assert "Available skills:" in replan_requests[0].system_prompt
        if expect_reference:
            assert "# Skill Catalog [loaded" in replan_requests[1].system_prompt
            assert "Available skills:" not in replan_requests[1].system_prompt
        else:
            assert "Available skills:" in replan_requests[1].system_prompt
            assert "# Skill Catalog [loaded" not in replan_requests[1].system_prompt
