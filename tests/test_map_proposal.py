from pathlib import Path

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
