import json
from pathlib import Path

from visual_coding_agent_harness.legacy.workspace_v2 import EvidenceRecord, EvidenceWorkspace


def _record(workspace: EvidenceWorkspace, *, stage: str, parent_id: str | None = None) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=workspace.next_evidence_id(stage),
        stage=stage,
        parent_id=parent_id,
        tool="vision_read",
        observation_id="obs_0001",
        frame_set_id="fs_ev_run_00001",
        content={"claim": f"{stage} claim"},
        grounding_quality="visually_confirmed",
        confidence=0.8,
        created_at=1.0,
    )


def test_chain_walks_to_root(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "ev_run")
    root = _record(workspace, stage="distilled")
    ledger = _record(workspace, stage="ledger", parent_id=root.evidence_id)
    mapped = _record(workspace, stage="mapped", parent_id=ledger.evidence_id)
    final = _record(workspace, stage="final_support", parent_id=mapped.evidence_id)
    for record in [root, ledger, mapped, final]:
        workspace.write_evidence(record)

    assert len(workspace.evidence_chain(final.evidence_id)) == 4


def test_chain_returns_root_to_leaf_order(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "ev_run")
    root = _record(workspace, stage="distilled")
    child = _record(workspace, stage="ledger", parent_id=root.evidence_id)
    workspace.write_evidence(root)
    workspace.write_evidence(child)

    assert [record.evidence_id for record in workspace.evidence_chain(child.evidence_id)] == [
        root.evidence_id,
        child.evidence_id,
    ]


def test_missing_evidence_returns_none(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "ev_run")

    assert workspace.load_evidence("ev_missing") is None


def test_exports_compact_evidence_chains(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "ev_run")
    root = _record(workspace, stage="distilled")
    ledger = _record(workspace, stage="ledger", parent_id=root.evidence_id)
    mapped = EvidenceRecord(
        evidence_id=workspace.next_evidence_id("mapped"),
        stage="mapped",
        parent_id=ledger.evidence_id,
        tool="vision_read",
        observation_id="obs_0001",
        frame_set_id="fs_ev_run_00001",
        content={
            "candidate_option_relation": {
                "option": "B",
                "relation": "support",
                "strength": 0.8,
                "parent_evidence_id": ledger.evidence_id,
            },
            "raw_output": "should not be exported",
        },
        grounding_quality="visually_confirmed",
        confidence=0.8,
        created_at=1.0,
    )
    for record in [root, ledger, mapped]:
        workspace.write_evidence(record)

    payload = workspace.export_evidence_chains()

    export_path = workspace.root / "artifacts" / "evidence_chains" / "evidence_chains.json"
    disk_payload = json.loads(export_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "EvidenceChainsV1"
    assert disk_payload["chain_count"] == 1
    assert disk_payload["chains"][0]["stages"] == ["distilled", "ledger", "mapped"]
    assert disk_payload["chains"][0]["records"][1]["parent_id"] == root.evidence_id
    assert "raw_output" not in json.dumps(disk_payload)


def test_exports_workspace_memory_evidence_chains_when_legacy_records_are_absent(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "multi_v3_chain")
    observation = workspace.write_observation(
        tool_name="read_segment",
        claim="The video shows Austria-Hungary rising and falling.",
        confidence=0.8,
        raw_output={
            "produced_anchors": [
                {
                    "anchor_id": "clip_anch_seg_0001",
                    "source_kind": "visual_fact",
                    "modality": "visual",
                    "excerpt": "Austria-Hungary rising and falling.",
                }
            ]
        },
    )
    workspace.commit_observation(
        observation.observation_id,
        writes={
            "pinned_anchors": [
                {
                    "anchor_id": "clip_anch_seg_0001",
                    "source_kind": "visual_fact",
                    "modality": "visual",
                    "excerpt": "Austria-Hungary rising and falling.",
                }
            ],
            "memory": [
                {
                    "kind": "answer_support",
                    "claim": "The video shows Austria-Hungary rising and falling.",
                    "supports_option": "D",
                    "anchor_ids": ["clip_anch_seg_0001"],
                    "confidence": "high",
                }
            ],
        },
    )

    payload = workspace.export_evidence_chains()

    assert payload["chain_count"] == 1
    chain = payload["chains"][0]
    assert chain["leaf_evidence_id"] == "mem_0001"
    assert chain["stages"] == ["observation", "anchor", "memory"]
    assert chain["memory_id"] == "mem_0001"
    assert chain["observation_id"] == observation.observation_id
    assert chain["records"][-1]["kind"] == "answer_support"
    assert chain["records"][-1]["supports_option"] == "D"
