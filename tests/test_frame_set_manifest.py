from pathlib import Path

from visual_coding_agent_harness.core.contracts import CONTRACT_VERSION
from visual_coding_agent_harness.workspace import EvidenceWorkspace, FrameSetManifest


def _create_manifest(workspace: EvidenceWorkspace, *, approximate: bool = False) -> FrameSetManifest:
    return workspace.create_manifest(
        video_path="/videos/example.mp4",
        segment_id="seg_0001",
        start_sec=1.0,
        end_sec=9.0,
        target_nframes=128,
        nframes=128,
        sampling_policy="uniform",
        frame_times_sec=[1.0, 5.0, 9.0],
        frame_times_approximate=approximate,
        created_by_tool="vision_read",
        observation_id="obs_0001",
        budget_reason="default_contract",
    )


def test_create_manifest_assigns_unique_ids(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "manifest_run")

    ids = [_create_manifest(workspace).frame_set_id for _ in range(3)]

    assert ids == ["fs_manifest_run_00001", "fs_manifest_run_00002", "fs_manifest_run_00003"]


def test_manifest_persisted_to_jsonl(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "manifest_run")

    _create_manifest(workspace)

    path = workspace.root / "frame_sets" / "manifests.jsonl"
    assert path.exists()
    assert len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]) == 1


def test_get_manifest_roundtrip(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "manifest_run")
    manifest = _create_manifest(workspace)

    assert workspace.get_manifest(manifest.frame_set_id) == manifest


def test_frame_times_approximate_flag_preserved(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "manifest_run")

    manifest = _create_manifest(workspace, approximate=True)

    assert manifest.frame_times_approximate is True
    assert manifest.contract_version == CONTRACT_VERSION
