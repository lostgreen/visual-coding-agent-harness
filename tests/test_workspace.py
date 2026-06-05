from pathlib import Path

from visual_coding_agent_harness.workspace import EvidenceWorkspace


def test_evidence_table_jsonl_roundtrip(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "table_roundtrip")

    row = workspace.write_evidence_row(
        {
            "obs_id": "obs_0002",
            "tool": "vision_read",
            "segment_id": "seg_0001",
            "t_start": 10.0,
            "t_end": 12.5,
            "entity": "humble background",
            "claim": "The clip directly shows a humble background.",
            "grounding_quality": "visually_confirmed",
            "confidence_signal": "confirmed",
            "confidence": 0.84,
            "supported_option": "B",
            "mutex_group_id": "q_612_1_opt_AB",
            "candidate_option_relations": [
                {"option": "B", "relation": "support", "strength": 0.84}
            ],
        }
    )

    table = workspace.read_evidence_table_v3(
        question="Which background is shown?",
        options=["A. wealthy background", "B. humble background"],
    )

    assert (workspace.root / "evidence_table.jsonl").exists()
    assert row["evidence_id"] == "ev_table_table_roundtrip_00001"
    assert table["schema_version"] == "EvidenceTableV3"
    assert table["groups"]["B"][0]["evidence_id"] == row["evidence_id"]
    assert table["groups"]["B"][0]["segment_id"] == "seg_0001"
    assert table["groups"]["B"][0]["time_range"] == [10.0, 12.5]


def test_answer_evidence_table_prefers_jsonl_artifact(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "table_file_source")
    workspace.write_evidence_row(
        {
            "obs_id": "obs_file_only",
            "tool": "vision_read",
            "claim": "File-backed row supports option D.",
            "grounding_quality": "visually_confirmed",
            "confidence": 0.91,
            "supported_option": "D",
        }
    )

    table = workspace.evidence_table_v2(
        question="Which option is supported?",
        options=["A. first", "D. fourth"],
    )

    assert table["schema_version"] == "EvidenceTableV2"
    assert table["source_artifact"] == "evidence_table.jsonl"
    assert table["groups"]["D"][0]["obs_id"] == "obs_file_only"
