from pathlib import Path

from visual_coding_agent_harness.workspace.distill import distill
from visual_coding_agent_harness.legacy.interpreter import ProgramInterpreter
from visual_coding_agent_harness.core.registry import ToolRegistry, tool
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


def test_unsupported_claim_downgrades_grounding(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "distill_run")
    observation = workspace.write_observation(
        tool_name="vision_read",
        claim="Limitation: no direct evidence is visible for this fact.",
        confidence=0.9,
        raw_output={
            "confidence_signal": "unsupported",
            "grounding_quality": "visually_confirmed",
            "facts": [
                {
                    "fact": "Limitation: no direct evidence is visible for this fact.",
                    "grounding_quality": "visually_confirmed",
                    "confidence": 0.9,
                }
            ],
        },
    )

    records = distill(observation, workspace)

    assert len(records) == 1
    assert records[0].grounding_quality == "inferred"
    assert records[0].content["confidence_signal"] == "unsupported"


def test_repetition_loop_observation_is_weak(tmp_path: Path):
    registry = ToolRegistry()
    repeated = "The Ecstasy of Saint Teresa " * 8

    @tool(name="vision_read", description="Read a repeated fact.")
    def vision_read():
        return {
            "claim": repeated,
            "confidence": 0.9,
            "grounding_quality": "visually_confirmed",
            "supported_option": "D",
            "candidate_option_relations": [
                {"option": "D", "relation": "support", "strength": 0.9},
            ],
        }

    registry.register(vision_read)
    workspace = EvidenceWorkspace.create(tmp_path, "distill_run")

    ProgramInterpreter(registry=registry, workspace=workspace).run([{"tool": "vision_read"}])

    table = workspace.evidence_table_v2(
        question="Which artwork is shown?\nA. David\nD. The Ecstasy of Saint Teresa",
        options=["A. David", "D. The Ecstasy of Saint Teresa"],
    )
    row = table["groups"]["D"][0]
    trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")

    assert row["grounding_quality"] == "weak"
    assert "tool_output_degenerate" in trace


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
