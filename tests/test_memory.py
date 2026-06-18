from pathlib import Path

import pytest

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
