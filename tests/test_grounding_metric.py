from pathlib import Path

from visual_coding_agent_harness.workspace import EvidenceWorkspace


def _workspace_with_memory(tmp_path: Path, *, kind: str) -> tuple[EvidenceWorkspace, str]:
    workspace = EvidenceWorkspace.create(tmp_path, f"grounding_metric_{kind}")
    observation = workspace.write_observation(
        tool_name="verify_window",
        claim="The local window supports the visual claim.",
        confidence=0.8,
        raw_output={
            "mode": "verify_window",
            "produced_anchors": [
                {
                    "anchor_id": "anch_verify_obs_0001_001",
                    "observation_id": "__pending__",
                    "source_kind": "visual_fact",
                    "segment_id": "seg_0001",
                    "start_sec": 0.0,
                    "end_sec": 10.0,
                    "field_path": "verification_results[0]",
                    "excerpt": "The local window supports the visual claim.",
                    "modality": "visual",
                }
            ],
        },
    )
    anchor = observation.raw_output["produced_anchors"][0]
    entry = workspace.write_memory(
        kind=kind,
        claim="The local window supports the visual claim.",
        anchors=[anchor],
        confidence="medium",
    )
    return workspace, entry.entry_id


def test_grounded_citation_recognizes_mem_visual_support(tmp_path: Path) -> None:
    workspace, entry_id = _workspace_with_memory(tmp_path, kind="visual_support")

    assert workspace.has_non_navigation_visual_citation([entry_id]) is True


def test_grounded_citation_recognizes_mem_synthesized_support(tmp_path: Path) -> None:
    workspace, entry_id = _workspace_with_memory(tmp_path, kind="synthesized_support")

    assert workspace.has_non_navigation_visual_citation([entry_id]) is True


def test_grounded_citation_rejects_mem_retrieval_candidate(tmp_path: Path) -> None:
    workspace, entry_id = _workspace_with_memory(tmp_path, kind="retrieval_candidate")

    assert workspace.has_non_navigation_visual_citation([entry_id]) is False


def test_grounded_citation_recognizes_verify_window_observation_fallback(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "grounding_metric_verify_window_obs")
    observation = workspace.write_observation(
        tool_name="verify_window",
        claim="The local window supports the visual claim.",
        confidence=0.8,
        raw_output={"mode": "verify_window"},
    )

    assert workspace.has_non_navigation_visual_citation([observation.observation_id]) is True
