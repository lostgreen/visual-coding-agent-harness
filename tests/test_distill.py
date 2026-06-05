from pathlib import Path

from visual_coding_agent_harness.agents.distill import distill
from visual_coding_agent_harness.interpreter import ProgramInterpreter
from visual_coding_agent_harness.registry import ToolRegistry, tool
from visual_coding_agent_harness.workspace import EvidenceWorkspace


def test_default_distiller_preserves_grounding_quality(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "distill_run")
    observation = workspace.write_observation(
        tool_name="global_gist",
        claim="The video is mainly about sculpture.",
        confidence=0.7,
        raw_output={"grounding_quality": "global_sparse"},
    )

    records = distill(observation, workspace)

    assert len(records) == 1
    assert records[0].grounding_quality == "global_sparse"


def test_vision_read_distiller_splits_facts(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "distill_run")
    observation = workspace.write_observation(
        tool_name="vision_read",
        claim="Three facts.",
        confidence=0.9,
        raw_output={"facts": ["red car appears", {"fact": "blue car appears", "confidence": 0.8}]},
    )

    records = distill(observation, workspace)

    assert [record.content["claim"] for record in records] == ["red car appears", "blue car appears"]
    assert all(record.observation_id == observation.observation_id for record in records)


def test_interpreter_writes_distilled_record_links_to_observation(tmp_path: Path):
    registry = ToolRegistry()

    @tool(name="vision_read", description="Read a fact.")
    def vision_read():
        return {
            "claim": "The red car appears.",
            "confidence": 0.9,
            "grounding_quality": "visually_confirmed",
        }

    registry.register(vision_read)
    workspace = EvidenceWorkspace.create(tmp_path, "distill_run")

    ProgramInterpreter(registry=registry, workspace=workspace).run([{"tool": "vision_read"}])

    records = [workspace.load_evidence("ev_distilled_distill_run_00001")]
    assert records[0] is not None
    assert records[0].observation_id == "obs_0001"
