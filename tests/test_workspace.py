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


def test_evidence_table_preserves_source_provenance(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "table_provenance")

    workspace.write_evidence_row(
        {
            "obs_id": "scene_order_seg_0001",
            "tool": "timeline_asr_summary",
            "segment_id": "seg_0001",
            "source_segment_id": "dual_seg_0001",
            "raw_asr_ref": {"cue_ids": ["cue-1", "cue-2"]},
            "visual_caption_source": "caption_scene_segment:vl-mini",
            "citation_provenance": {"asr": "subtitle", "visual": "video"},
            "claim": "Indexed evidence supports option D.",
            "grounding_quality": "indexed_transcript",
            "confidence": 0.86,
            "supported_option": "D",
        }
    )

    table = workspace.read_evidence_table_v3(
        question="Which sequence is shown?",
        options=["A. first", "D. fourth"],
    )
    row = table["groups"]["D"][0]

    assert row["source_segment_id"] == "dual_seg_0001"
    assert row["raw_asr_ref"] == {"cue_ids": ["cue-1", "cue-2"]}
    assert row["visual_caption_source"] == "caption_scene_segment:vl-mini"
    assert row["citation_provenance"] == {"asr": "subtitle", "visual": "video"}


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


def test_evidence_status_summary_reports_coverage_and_duplicates(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "status_summary")
    first = workspace.write_observation(
        tool_name="vision_read",
        claim="The clip shows a red aircraft.",
        confidence=0.82,
        regions=[{"segment_id": "seg_0001", "start_sec": 1.0, "end_sec": 2.0}],
        raw_output={
            "grounding_quality": "visually_confirmed",
            "candidate_option_relations": [{"option": "B", "relation": "support", "strength": 0.82}],
        },
    )
    workspace.write_observation(
        tool_name="vision_read",
        claim="The clip shows a red aircraft.",
        confidence=0.74,
        regions=[{"segment_id": "seg_0002", "start_sec": 3.0, "end_sec": 4.0}],
        raw_output={
            "grounding_quality": "visually_confirmed",
            "candidate_option_relations": [{"option": "B", "relation": "support", "strength": 0.74}],
        },
    )
    workspace.write_hypothesis({"entered upper class": {"status": "missing", "evidence_obs_id": ""}})
    workspace.update_hypothesis_slot(slot_name="humble background", status="satisfied", evidence_obs_id=first.observation_id)

    summary = workspace.evidence_status_summary(
        question="Which option is visible?\nA. blue car\nB. red aircraft",
        options=["A. blue car", "B. red aircraft"],
    )

    assert summary["option_coverage"] == "1/2"
    assert summary["coverage_pct"] == 0.5
    assert summary["duplicate_observations"] == 1
    assert summary["total_evidence_rows"] == 2
    assert summary["option_status"]["B"]["strong_evidence_count"] == 2
    assert summary["option_status"]["B"]["has_visual_citation"] is True
    assert summary["option_status"]["A"]["strong_evidence_count"] == 0
    assert summary["hypothesis_gaps"] == ["entered upper class"]
