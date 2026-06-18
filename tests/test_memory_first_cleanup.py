from pathlib import Path
from unittest.mock import patch

from visual_coding_agent_harness.agents.context_budget import default_context_budget_allocator
from visual_coding_agent_harness.agents.iterative_agent import AgentBudget
from visual_coding_agent_harness.agents.prompt_stack import build_replanning_prompt
from visual_coding_agent_harness.memory import SourceAnchor
from visual_coding_agent_harness.video_index import fixed_window_scene_index
from visual_coding_agent_harness.workspace import EvidenceWorkspace


ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _default_replanning_prompt(**kwargs: object) -> str:
    prompt, _report = build_replanning_prompt(
        question="Which option is supported?\nA. red shield\nB. blue door",
        scene_index=fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=30.0),
        ledger_text="# Compact Evidence Context\n(none)",
        round_number=1,
        budget=AgentBudget(max_rounds=3),
        allocator=default_context_budget_allocator(total_budget_tokens=12000),
        **kwargs,
    )
    return prompt


def test_minimal_final_gate_module_is_memory_only() -> None:
    source = _source("src/visual_coding_agent_harness/agents/final_gate.py")

    forbidden = [
        "evaluate_final_candidate",
        "FinalGateDecision",
        "OptionEvaluation",
        "SkillPolicy",
        "TRANSCRIPT_MODALITIES",
        "VISUAL_MODALITIES",
        "get_skill_policy",
        "policy_constants",
        "evidence_table",
        "TranscriptEvidenceBinder",
    ]
    for symbol in forbidden:
        assert symbol not in source


def test_iterative_agent_uses_legacy_final_gate_only_from_legacy_module() -> None:
    source = _source("src/visual_coding_agent_harness/agents/iterative_agent.py")

    assert "from .final_gate import evaluate_final_integrity" in source
    assert "from .final_gate_legacy import evaluate_final_candidate" in source


def test_iterative_progress_no_longer_uses_evidence_table_growth_names() -> None:
    source = _source("src/visual_coding_agent_harness/agents/iterative_agent.py")

    assert "last_evidence_table_row_count" not in source
    assert "current_evidence_table_row_count" not in source
    assert "evidence_table_no_growth" not in source


def test_default_policy_prompts_use_memory_first_language() -> None:
    prompt_facing_sources = [
        _source("src/visual_coding_agent_harness/agents/question_policy.py"),
        _source("src/visual_coding_agent_harness/agents/skills/playbook.py"),
    ]

    for source in prompt_facing_sources:
        assert "answer-grade" not in source
        assert "evidence_binding.status=supported" not in source


def test_default_prompt_contract_requires_memory_citations() -> None:
    prompt = _default_replanning_prompt()

    assert '"citations": [memory_id]' in prompt
    assert '"citations": [observation_id]' not in prompt
    assert "Final answers must cite mem_" in prompt
    assert "first write_memory" in prompt
    assert "answer-grade" not in prompt


def test_default_prompt_does_not_expose_legacy_evidence_tools() -> None:
    prompt = _default_replanning_prompt(
        target_ref_descriptions=["T1: red shield appears"],
        active_skill="causal_asr_qa@v1",
    )

    assert "bind_asr_claim" not in prompt
    assert "verify_ledger_answer" not in prompt
    assert "query_evidence_table" not in prompt
    assert "promote indexed ASR cue_ids into supported evidence" not in prompt
    assert "supported evidence" not in prompt
    assert "answer-grade" not in prompt


def test_legacy_prompt_tools_are_env_gated() -> None:
    with patch.dict("os.environ", {"HARNESS_LEGACY_PROMPT_TOOLS": "1"}):
        prompt = _default_replanning_prompt(target_ref_descriptions=["T1: red shield appears"])

    assert "bind_asr_claim(segment_id: str, target_refs: list)" in prompt
    assert "verify_ledger_answer(" in prompt
    assert "query_evidence_table(filter: dict)" in prompt


def test_workspace_exposes_memory_first_aliases_and_anchor_helpers(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "memory_first_aliases")
    workspace.write_produced_anchors(
        [
            SourceAnchor(
                anchor_id="anch_obs_0001_caption",
                observation_id="obs_0001",
                source_kind="caption_fact",
                segment_id="seg_0001",
                field_path="caption",
                excerpt="A red shield appears.",
            )
        ]
    )
    entry = workspace.write_memory_entry(
        kind="note",
        claim="A red shield appears.",
        anchors=[{"anchor_id": "anch_obs_0001_caption", "excerpt": "red shield"}],
    )

    assert workspace.read_memory_by_id(entry.entry_id) == entry
    assert [anchor.anchor_id for anchor in workspace.observation_anchors("obs_0001")] == ["anch_obs_0001_caption"]
    assert [item["observation_id"] for item in workspace.uncommitted_observations()] == []


def test_observation_anchor_registration_uses_memory_first_trace(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "anchor_registration_trace")

    workspace.write_observation(
        tool_name="caption_segment",
        claim="A red shield appears.",
        confidence=0.8,
        raw_output={
            "produced_anchors": [
                {
                    "anchor_id": "anch_caption_001",
                    "observation_id": "__pending__",
                    "source_kind": "caption_fact",
                    "segment_id": "seg_0001",
                    "field_path": "caption",
                    "excerpt": "A red shield appears.",
                }
            ]
        },
    )

    trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
    assert "observation_anchors_registered" in trace
    assert "post_observation_textual_evidence_promoted" not in trace
