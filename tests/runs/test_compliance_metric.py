from pathlib import Path

from runs.eval_runner import compute_nframes_metrics
from visual_coding_agent_harness.legacy.workspace_v2 import EvidenceWorkspace


def _manifest(workspace: EvidenceWorkspace, *, tool: str, nframes: int, target: int = 128) -> None:
    workspace.create_manifest(
        video_path="/videos/demo.mp4",
        segment_id="seg_0001",
        start_sec=0.0,
        end_sec=12.0,
        target_nframes=target,
        nframes=nframes,
        sampling_policy="uniform",
        frame_times_sec=[0.0],
        frame_times_approximate=True,
        created_by_tool=tool,
        observation_id="obs_0001",
        budget_reason="default_contract",
    )


def test_compliance_100_when_all_hit(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "nframes_run")
    _manifest(workspace, tool="global_gist", nframes=128)
    _manifest(workspace, tool="vision_read", nframes=128)

    compliance, _ = compute_nframes_metrics(workspace)

    assert compliance == 1.0


def test_compliance_partial(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "nframes_run")
    _manifest(workspace, tool="global_gist", nframes=128)
    _manifest(workspace, tool="vision_read", nframes=64)
    _manifest(workspace, tool="vision_read", nframes=128)
    _manifest(workspace, tool="inspect_segment", nframes=100)

    compliance, _ = compute_nframes_metrics(workspace)

    assert compliance == 0.5


def test_histogram_grouped_by_tool(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "nframes_run")
    _manifest(workspace, tool="vision_read", nframes=128)
    _manifest(workspace, tool="vision_read", nframes=64)
    _manifest(workspace, tool="global_gist", nframes=128)

    _, histogram = compute_nframes_metrics(workspace)

    assert histogram == {"global_gist": {128: 1}, "vision_read": {64: 1, 128: 1}}
