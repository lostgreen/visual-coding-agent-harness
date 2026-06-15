from pathlib import Path

from visual_coding_agent_harness.tools.workspace_primitives import build_workspace_primitives_registry
from visual_coding_agent_harness.workspace import EvidenceWorkspace


def test_workspace_primitives_return_deterministic_results(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_primitives")
    workspace.write_observation(
        tool_name="vision_read",
        claim="The red door opens.",
        confidence=0.9,
        raw_output={"grounding_quality": "visually_confirmed"},
    )
    workspace.write_evidence_row(
        {
            "obs_id": "obs_0001",
            "tool": "vision_read",
            "claim": "The red door opens.",
            "grounding_quality": "visually_confirmed",
            "confidence": 0.9,
            "confidence_signal": "confirmed",
            "supported_option": "B",
        }
    )
    workspace.append_to_timeline(
        obs_id="obs_0001",
        entity="red door opens",
        observed_at_sec=12.0,
        confidence_signal="confirmed",
    )
    workspace.write_hypothesis({"slot_door": {"status": "empty", "evidence_obs_id": ""}})

    registry = build_workspace_primitives_registry(workspace=workspace)

    view = registry.execute("view_observation", {"obs_id": "obs_0001"})
    detail = registry.execute("read_observation_detail", {"obs_id": "obs_0001"})
    grep = registry.execute("grep_evidence", {"pattern": "red door"})
    query = registry.execute("query_evidence_table", {"filter": {"confidence_signal": "confirmed"}})
    timeline = registry.execute("read_timeline_sorted", {})
    hypothesis_before = registry.execute("read_hypothesis", {})
    updated = registry.execute(
        "update_hypothesis_slot",
        {"slot_name": "slot_door", "status": "satisfied", "evidence_obs_id": "obs_0001"},
    )
    hypothesis_after = registry.execute("read_hypothesis", {})

    assert view["regions"][0]["claim"] == "The red door opens."
    assert detail["regions"][0]["claim"] == "The red door opens."
    assert grep["regions"][0]["obs_ids"] == ["obs_0001"]
    assert query["regions"][0]["rows"][0]["obs_id"] == "obs_0001"
    assert timeline["regions"][0]["entries"][0]["obs_id"] == "obs_0001"
    assert hypothesis_before["regions"][0]["slots"]["slot_door"]["status"] == "empty"
    assert updated["regions"][0]["slot"]["status"] == "satisfied"
    assert hypothesis_after["regions"][0]["slots"]["slot_door"]["status"] == "satisfied"


def test_recent_tool_outputs_returns_latest_three_with_raw_payload(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "recent_tool_outputs")
    for index in range(4):
        workspace.write_observation(
            tool_name="vision_read",
            claim=f"Observation {index}.",
            confidence=0.7 + index / 10,
            raw_output={
                "visual_caption": f"caption {index}",
                "anchors_for_vlm": [{"segment_id": f"seg_{index:04d}"}],
                "long_field": "x" * 1200,
            },
        )
    workspace.write_evidence_row(
        {
            "obs_id": "obs_0004",
            "tool": "vision_read",
            "claim": "Observation 3.",
            "segment_id": "seg_0003",
            "grounding_quality": "visually_confirmed",
        }
    )

    outputs = workspace.recent_tool_outputs(limit=3)

    assert [item["observation_id"] for item in outputs] == ["obs_0002", "obs_0003", "obs_0004"]
    assert outputs[-1]["in_evidence_table"] is True
    assert outputs[-1]["segment_id"] == "seg_0003"
    assert outputs[-1]["modality"] == "visually_confirmed"
    assert outputs[-1]["raw_output"]["visual_caption"] == "caption 3"
    assert outputs[-1]["raw_output"]["anchors_for_vlm"] == [{"segment_id": "seg_0003"}]
    assert len(outputs[-1]["raw_output"]["long_field"]) < 1200
