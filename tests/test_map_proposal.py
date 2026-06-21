from pathlib import Path

from visual_coding_agent_harness.tools.navigation import build_video_navigation_registry
from visual_coding_agent_harness.video.map import VideoMap, VideoMapSegment, VideoMapStore
from visual_coding_agent_harness.workspace import EvidenceWorkspace, MapUpdateProposal


def test_proposal_persisted(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "map_run")
    proposal = MapUpdateProposal(
        proposal_id=workspace.next_proposal_id(),
        target_segment_id="seg_0001",
        update_type="entity_add",
        payload={"entities": ["Apollo"]},
        source_evidence_id="ev_distilled_map_run_00001",
        source_frame_set_id="fs_map_run_00001",
        confidence=0.75,
        proposed_at=1.0,
    )

    workspace.write_proposal(proposal)

    assert workspace.load_pending_proposals() == [proposal]


def test_committed_at_default_none():
    proposal = MapUpdateProposal(
        proposal_id="mp_map_run_00001",
        target_segment_id="seg_0001",
        update_type="entity_add",
        payload={"entities": ["Apollo"]},
        source_evidence_id="ev_distilled_map_run_00001",
        source_frame_set_id="fs_map_run_00001",
        confidence=0.75,
        proposed_at=1.0,
    )

    assert proposal.committed_at is None


def test_commit_map_proposals_updates_video_map_and_marks_committed(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "map_commit")
    store = VideoMapStore(
        VideoMap(
            video_path="/videos/demo.mp4",
            duration_sec=60.0,
            segments=[
                VideoMapSegment(segment_id="seg_0001", start_sec=0.0, end_sec=30.0, low_fps_caption="opening"),
            ],
        )
    )
    workspace.write_proposal(
        MapUpdateProposal(
            proposal_id=workspace.next_proposal_id(),
            target_segment_id="seg_0001",
            update_type="context_update",
            payload={"low_fps_caption": "Aircraft runway sequence.", "entities": ["aircraft", "runway"]},
            source_evidence_id="ev_distilled_map_commit_00001",
            source_frame_set_id="fs_map_commit_00001",
            confidence=0.8,
            proposed_at=1.0,
        )
    )
    registry = build_video_navigation_registry(store, workspace=workspace)

    result = registry.execute("commit_map_proposals", {"limit": 4})

    assert result["applied"][0]["segment_id"] == "seg_0001"
    assert store.current.get("seg_0001").low_fps_caption == "Aircraft runway sequence."
    assert store.current.get("seg_0001").entities == ["aircraft", "runway"]
    assert workspace.load_pending_proposals() == []
