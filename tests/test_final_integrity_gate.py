from pathlib import Path
from types import SimpleNamespace

import pytest

from visual_coding_agent_harness.agents.final_gate import evaluate_final_integrity
from visual_coding_agent_harness.memory import SourceAnchor
from visual_coding_agent_harness.workspace import EvidenceWorkspace


def _workspace_with_memory(tmp_path: Path) -> EvidenceWorkspace:
    workspace = EvidenceWorkspace.create(tmp_path, "minimal_final_gate")
    workspace.target_registry = SimpleNamespace(options_by_id={"A": object(), "D": object()})
    workspace.write_produced_anchors(
        [
            SourceAnchor(
                anchor_id="anch_seg_0005_asr_206",
                observation_id="obs_0017",
                source_kind="asr_cue",
                segment_id="seg_0005",
                cue_id="206",
                field_path="asr_sentences[cue_id=206].text",
                excerpt="Austria-Hungary was therefore seen as a good buffer between Russia and Western Europe.",
            )
        ]
    )
    workspace.write_memory(
        kind="support",
        claim="The narration says Austria-Hungary was a buffer between Russia and Western Europe.",
        anchors=[{"anchor_id": "anch_seg_0005_asr_206", "excerpt": "buffer between Russia and Western Europe"}],
        supports_option="D",
        confidence="high",
    )
    return workspace


def test_final_accepts_memory_citation_with_valid_anchor(tmp_path: Path) -> None:
    workspace = _workspace_with_memory(tmp_path)

    decision = evaluate_final_integrity(
        selected_option="D",
        citations=["mem_0001"],
        workspace=workspace,
    )

    assert decision.accepted
    assert decision.selected_option == "D"
    assert decision.cited_memory_ids == ("mem_0001",)
    assert decision.cited_observation_ids == ()


def test_final_rejects_raw_observation_citation_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HARNESS_ALLOW_RAW_OBS_FINAL_CITATION", raising=False)
    workspace = EvidenceWorkspace.create(tmp_path, "minimal_final_raw_obs")
    observation = workspace.write_observation(
        tool_name="vision_read",
        claim="The door opens.",
        confidence=0.9,
        raw_output={"grounding_quality": "visually_confirmed"},
    )

    decision = evaluate_final_integrity(
        selected_option="A",
        citations=[observation.observation_id],
        workspace=workspace,
    )

    assert not decision.accepted
    assert decision.rejection_reason == "raw_observation_citation_without_memory"


def test_final_allows_raw_observation_citation_with_explicit_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARNESS_ALLOW_RAW_OBS_FINAL_CITATION", "1")
    workspace = EvidenceWorkspace.create(tmp_path, "minimal_final_raw_obs_allowed")
    observation = workspace.write_observation(
        tool_name="vision_read",
        claim="The door opens.",
        confidence=0.9,
        raw_output={"grounding_quality": "visually_confirmed"},
    )

    decision = evaluate_final_integrity(
        selected_option="A",
        citations=[observation.observation_id],
        workspace=workspace,
    )

    assert decision.accepted
    assert decision.cited_observation_ids == (observation.observation_id,)
    assert "raw_observation_citation_without_memory" in decision.warnings


def test_final_rejects_no_citation(tmp_path: Path) -> None:
    workspace = _workspace_with_memory(tmp_path)

    decision = evaluate_final_integrity(
        selected_option="D",
        citations=[],
        workspace=workspace,
    )

    assert not decision.accepted
    assert decision.rejection_reason == "missing_citation"


def test_final_rejects_dangling_memory_id(tmp_path: Path) -> None:
    workspace = _workspace_with_memory(tmp_path)

    decision = evaluate_final_integrity(
        selected_option="D",
        citations=["mem_missing"],
        workspace=workspace,
    )

    assert not decision.accepted
    assert decision.rejection_reason == "dangling_memory_id"


def test_final_rejects_dangling_anchor_id(tmp_path: Path) -> None:
    workspace = _workspace_with_memory(tmp_path)
    rows = workspace._read_jsonl_dicts("memory.jsonl")
    rows[0]["anchors"][0]["anchor_id"] = "anch_missing"
    workspace._write_jsonl("memory.jsonl", rows)

    decision = evaluate_final_integrity(
        selected_option="D",
        citations=["mem_0001"],
        workspace=workspace,
    )

    assert not decision.accepted
    assert decision.rejection_reason == "dangling_anchor_id"


def test_final_does_not_check_phrase_match(tmp_path: Path) -> None:
    workspace = _workspace_with_memory(tmp_path)

    decision = evaluate_final_integrity(
        selected_option="D",
        citations=["mem_0001"],
        workspace=workspace,
        reason="This deliberately does not repeat the anchor wording.",
    )

    assert decision.accepted
