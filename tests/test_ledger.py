import json
from pathlib import Path

from visual_coding_agent_harness.legacy.interpreter import ProgramInterpreter
from visual_coding_agent_harness.core.registry import ToolRegistry, tool
from visual_coding_agent_harness.legacy.workspace_v2 import EvidenceWorkspace


def test_ledger_line_has_evidence_ref(tmp_path: Path):
    registry = ToolRegistry()

    @tool(name="caption_image", description="Caption an image.")
    def caption_image():
        return {
            "claim": "The image shows a red door.",
            "confidence": 0.84,
            "grounding_quality": "visually_confirmed",
        }

    registry.register(caption_image)
    workspace = EvidenceWorkspace.create(tmp_path, "ledger_run")

    ProgramInterpreter(registry=registry, workspace=workspace).run([{"tool": "caption_image"}])

    ledger_text = (workspace.root / "ledger.md").read_text(encoding="utf-8")
    assert "ev_ledger_" in ledger_text


def test_ledger_record_parent_is_distilled(tmp_path: Path):
    registry = ToolRegistry()

    @tool(name="caption_image", description="Caption an image.")
    def caption_image():
        return {
            "claim": "The image shows a red door.",
            "confidence": 0.84,
            "grounding_quality": "visually_confirmed",
        }

    registry.register(caption_image)
    workspace = EvidenceWorkspace.create(tmp_path, "ledger_run")

    ProgramInterpreter(registry=registry, workspace=workspace).run([{"tool": "caption_image"}])

    distilled = workspace.load_evidence("ev_distilled_ledger_run_00001")
    ledger = workspace.load_evidence("ev_ledger_ledger_run_00002")
    assert distilled is not None
    assert ledger is not None
    assert ledger.parent_id == distilled.evidence_id
    assert [record.stage for record in workspace.evidence_chain(ledger.evidence_id)] == ["distilled", "ledger"]


def test_split_fact_ledger_records_match_ledger_rows(tmp_path: Path):
    registry = ToolRegistry()

    @tool(name="vision_read", description="Read visible facts.")
    def vision_read():
        return {
            "claim": "Two facts are visible.",
            "confidence": 0.9,
            "facts": ["a red door appears", "a blue sign appears"],
            "grounding_quality": "visually_confirmed",
        }

    registry.register(vision_read)
    workspace = EvidenceWorkspace.create(tmp_path, "ledger_run")

    ProgramInterpreter(registry=registry, workspace=workspace).run([{"tool": "vision_read"}])

    evidence_rows = [
        json.loads(line)
        for line in (workspace.root / "evidence.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ledger_records = [row for row in evidence_rows if row.get("stage") == "ledger"]
    ledger_text = (workspace.root / "ledger.md").read_text(encoding="utf-8")

    assert len(ledger_records) == 2
    assert ledger_text.count("ev_ledger_") == 2
    assert {row["content"]["claim"] for row in ledger_records} == {
        "a red door appears",
        "a blue sign appears",
    }


def test_mapped_evidence_persisted_with_ledger_parent(tmp_path: Path):
    registry = ToolRegistry()

    @tool(name="caption_image", description="Caption an image.")
    def caption_image():
        return {
            "claim": "The image shows a red door.",
            "confidence": 0.84,
            "grounding_quality": "visually_confirmed",
        }

    registry.register(caption_image)
    workspace = EvidenceWorkspace.create(tmp_path, "mapped_run")
    ProgramInterpreter(registry=registry, workspace=workspace).run([{"tool": "caption_image"}])

    changed = workspace.annotate_candidate_option_relations(
        observation_ids=["obs_0001"],
        relations=[
            {
                "option": "B",
                "relation": "support",
                "strength": 0.84,
                "observation_id": "obs_0001",
                "rationale": "red door supports option B",
            }
        ],
    )

    ledger = workspace.load_evidence("ev_ledger_mapped_run_00002")
    mapped = workspace.load_evidence("ev_mapped_mapped_run_00003")
    observation = workspace.get_observation("obs_0001")
    assert changed == 1
    assert ledger is not None
    assert mapped is not None
    assert mapped.parent_id == ledger.evidence_id
    assert mapped.content["candidate_option_relation"]["parent_evidence_id"] == ledger.evidence_id
    assert mapped.content["candidate_option_relation"]["option"] == "B"
    assert observation is not None
    assert observation.raw_output["candidate_option_relations"][0]["parent_evidence_id"] == ledger.evidence_id
    assert [record.stage for record in workspace.evidence_chain(mapped.evidence_id)] == [
        "distilled",
        "ledger",
        "mapped",
    ]


def test_mapped_evidence_must_have_parent(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "mapped_run")
    observation = workspace.write_observation(
        tool_name="caption_image",
        claim="The image shows a red door.",
        confidence=0.84,
        raw_output={"grounding_quality": "visually_confirmed"},
    )

    changed = workspace.annotate_candidate_option_relations(
        observation_ids=[observation.observation_id],
        relations=[
            {
                "option": "B",
                "relation": "support",
                "strength": 0.84,
                "observation_id": observation.observation_id,
                "parent_evidence_id": "ev_ledger_missing",
            }
        ],
    )

    evidence_rows = [
        json.loads(line)
        for line in (workspace.root / "evidence.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reloaded = workspace.get_observation(observation.observation_id)
    trace_text = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")

    assert changed == 0
    assert [row for row in evidence_rows if row.get("stage") == "mapped"] == []
    assert reloaded is not None
    assert "candidate_option_relations" not in reloaded.raw_output
    assert "mapped_evidence_orphan_count" in trace_text
