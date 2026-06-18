from pathlib import Path

from visual_coding_agent_harness.memory import SourceAnchor
from visual_coding_agent_harness.workspace import EvidenceWorkspace


ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


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
