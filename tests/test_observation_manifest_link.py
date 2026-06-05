from pathlib import Path

from visual_coding_agent_harness.workspace import EvidenceWorkspace


def test_link_manifest_updates_observation(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "link_run")
    observation = workspace.write_observation(tool_name="vision_read", claim="A visible fact.", confidence=0.8)

    workspace.link_manifest(observation.observation_id, "fs_link_run_00001")

    loaded = workspace.get_observation(observation.observation_id)
    assert loaded is not None
    assert loaded.frame_set_id == "fs_link_run_00001"


def test_obs_without_manifest_has_none(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "link_run")
    observation = workspace.write_observation(tool_name="vision_read", claim="A visible fact.", confidence=0.8)

    loaded = workspace.get_observation(observation.observation_id)

    assert loaded is not None
    assert loaded.frame_set_id is None
