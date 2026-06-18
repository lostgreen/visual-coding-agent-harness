from pathlib import Path

import pytest

from visual_coding_agent_harness.contracts import OptionSpec, TargetRegistry, TargetSpec
from visual_coding_agent_harness.memory import SourceAnchor
from visual_coding_agent_harness.workspace import EvidenceWorkspace


def _anchor(anchor_id: str = "anch_seg_0001_caption_001", **overrides: object) -> SourceAnchor:
    payload = {
        "anchor_id": anchor_id,
        "observation_id": "obs_0001",
        "source_kind": "caption_fact",
        "segment_id": "seg_0001",
        "field_path": "caption",
        "excerpt": "A red shield with a white cross appears over Central Europe.",
    }
    payload.update(overrides)
    return SourceAnchor(**payload)


def test_write_memory_rejects_unknown_anchor(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "memory_unknown_anchor")

    with pytest.raises(ValueError, match="unknown anchor_id=anch_missing"):
        workspace.write_memory(
            kind="support",
            claim="The caption mentions the shield.",
            anchors=[{"anchor_id": "anch_missing"}],
            supports_option="D",
        )


def test_write_memory_rejects_fake_excerpt(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "memory_fake_excerpt")
    workspace.write_produced_anchors([_anchor()])

    with pytest.raises(ValueError, match="excerpt not found"):
        workspace.write_memory(
            kind="support",
            claim="The caption mentions the shield.",
            anchors=[{"anchor_id": "anch_seg_0001_caption_001", "excerpt": "fake excerpt"}],
            supports_option="D",
        )


def test_write_memory_accepts_caption_anchor(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "memory_caption_anchor")
    workspace.write_produced_anchors([_anchor()])

    entry = workspace.write_memory(
        kind="support",
        claim="A red shield with a white cross appears over Central Europe.",
        anchors=[{"anchor_id": "anch_seg_0001_caption_001", "excerpt": "white cross appears"}],
        supports_option="D",
        confidence="high",
    )

    assert entry.entry_id == "mem_0001"
    assert entry.anchors[0].source_kind == "caption_fact"
    assert entry.supports_option == "D"
    assert (workspace.root / "memory.jsonl").exists()


def test_write_memory_persists_extension_fields(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "memory_extension_fields")
    workspace.write_produced_anchors([_anchor()])

    entry = workspace.write_memory(
        kind="support",
        claim="A red shield with a white cross appears over Central Europe.",
        anchors=[{"anchor_id": "anch_seg_0001_caption_001", "excerpt": "white cross appears"}],
        supports_option="D",
        confidence="high",
        role="episodic",
        layer="visual",
        embedding_refs=["clip://seg_0001/frame_0003"],
        metadata={"frame_count": 3},
    )

    loaded = workspace.get_memory(entry.entry_id)
    assert loaded is not None
    assert loaded.role == "episodic"
    assert loaded.layer == "visual"
    assert loaded.embedding_refs == ("clip://seg_0001/frame_0003",)
    assert loaded.metadata == {"frame_count": 3}


def test_committed_memory_anchor_ids_counts_unique_anchor_ids(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "memory_unique_anchor_progress")
    workspace.write_produced_anchors([_anchor()])

    workspace.write_memory(
        kind="note",
        claim="First note about the shield.",
        anchors=[{"anchor_id": "anch_seg_0001_caption_001"}],
    )
    workspace.write_memory(
        kind="reject",
        claim="Duplicate note still references the same anchor.",
        anchors=[{"anchor_id": "anch_seg_0001_caption_001"}],
    )

    assert workspace.committed_memory_anchor_ids() == {"anch_seg_0001_caption_001"}


def test_write_memory_accepts_retrieval_anchor(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "memory_retrieval_anchor")
    workspace.write_produced_anchors(
        [
            _anchor(
                "anch_cov_seg_0005_t5",
                source_kind="coverage_hit",
                field_path="matches",
                excerpt="matched_terms=['buffer', 'Russia', 'Western Europe']",
            )
        ]
    )

    entry = workspace.write_memory(
        kind="note",
        claim="The coverage result points at the buffer/Russia/Western Europe segment.",
        anchors=[{"anchor_id": "anch_cov_seg_0005_t5", "excerpt": "buffer"}],
        tags=["retrieval"],
    )

    assert entry.anchors[0].source_kind == "coverage_hit"
    assert entry.supports_option is None


def test_write_observation_persists_pending_anchors_with_observation_id(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "memory_observation_anchor")

    observation = workspace.write_observation(
        tool_name="caption_segment",
        claim="A captioned shield appears.",
        confidence=0.8,
        raw_output={
            "produced_anchors": [
                {
                    "anchor_id": "anch_seg_0001_caption_001",
                    "observation_id": "__pending__",
                    "source_kind": "caption_fact",
                    "segment_id": "seg_0001",
                    "field_path": "caption",
                    "excerpt": "A captioned shield appears.",
                }
            ]
        },
    )

    anchor = workspace.read_produced_anchors_by_id()["anch_seg_0001_caption_001"]
    assert anchor.observation_id == observation.observation_id
    assert observation.raw_output["produced_anchors"][0]["observation_id"] == observation.observation_id


def test_legacy_textual_binder_does_not_promote_rows_by_default_in_minimal_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HARNESS_FINAL_GATE_MODE", raising=False)
    monkeypatch.delenv("HARNESS_LEGACY_BINDER_TELEMETRY", raising=False)
    workspace = EvidenceWorkspace.create(tmp_path, "legacy_binder_minimal_off")
    workspace.target_registry = TargetRegistry.from_specs(
        targets=[TargetSpec("T1", "red shield")],
        options=[OptionSpec("A", target_sequence=("T1",))],
    )

    observation = workspace.write_observation(
        tool_name="read_segment",
        claim="The transcript mentions a red shield.",
        confidence=0.8,
        raw_output={
            "segment_id": "seg_0001",
            "asr_text": "A red shield is visible on the wall.",
        },
    )

    assert "answer_evidence_rows" not in observation.raw_output
    assert workspace.evidence_table_row_count() == 0


def test_legacy_textual_binder_promotes_rows_when_explicitly_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARNESS_LEGACY_BINDER_TELEMETRY", "1")
    workspace = EvidenceWorkspace.create(tmp_path, "legacy_binder_enabled")
    workspace.target_registry = TargetRegistry.from_specs(
        targets=[TargetSpec("T1", "red shield")],
        options=[OptionSpec("A", target_sequence=("T1",))],
    )

    observation = workspace.write_observation(
        tool_name="read_segment",
        claim="The transcript mentions a red shield.",
        confidence=0.8,
        raw_output={
            "segment_id": "seg_0001",
            "asr_text": "A red shield is visible on the wall.",
        },
    )

    assert observation.raw_output["answer_evidence_rows"]
    assert workspace.evidence_table_row_count() > 0
