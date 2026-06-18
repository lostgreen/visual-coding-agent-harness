from pathlib import Path

import pytest

from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse, VisionLanguageBackend
from visual_coding_agent_harness.tools.workspace_v2 import build_workspace_v2_registry
from visual_coding_agent_harness.video_map import VideoMap, VideoMapSegment
from visual_coding_agent_harness.workspace import EvidenceWorkspace


class RecordingBackend(VisionLanguageBackend):
    def __init__(self, text: str = "A shield icon appears over Central Europe.") -> None:
        self.text = text
        self.requests: list[BackendRequest] = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        return BackendResponse(text=self.text, raw={"supported_option": "D"})


def _video_map() -> VideoMap:
    return VideoMap(
        video_path="/videos/demo.mp4",
        duration_sec=120.0,
        segments=[
            VideoMapSegment(
                segment_id="seg_0001",
                start_sec=0.0,
                end_sec=60.0,
                low_fps_caption="A map shows Central Europe with a shield icon.",
                asr_text="Austria-Hungary was seen as a buffer between Russia and Western Europe.",
                entities=["Austria-Hungary", "Russia", "Western Europe", "shield"],
            ),
            VideoMapSegment(
                segment_id="seg_0002",
                start_sec=60.0,
                end_sec=120.0,
                low_fps_caption="Closing map animation.",
                asr_text="The story moves to another topic.",
            ),
        ],
    )


def test_workspace_v2_search_returns_candidate_only_hits(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_search")
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)

    result = registry.execute("search", {"query": "buffer Russia", "modality": "asr"})

    assert result["results"][0]["segment_id"] == "seg_0001"
    assert result["results"][0]["support_status"] == "candidate_only"
    assert result["produced_anchors"][0]["observation_id"] == "__pending__"
    assert "candidate_only" in result["limitations"]


def test_workspace_v2_list_reads_segments_and_workspace_sections(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_list")
    workspace.note_entity(kind="concept", name="Austria-Hungary")
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)

    segments = registry.execute("list", {"kind": "segments", "filter": {"segment_id": "seg_0001"}})
    entities = registry.execute("list", {"kind": "entities"})

    assert segments["items"][0]["segment_id"] == "seg_0001"
    assert entities["items"][0]["name"] == "Austria-Hungary"


def test_workspace_v2_registry_excludes_legacy_exploration_tools(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_no_legacy_tools")
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)

    tool_names = {spec.tool_spec.name for spec in registry.list_runtime_specs()}

    assert {
        "read_clip",
        "search",
        "list",
        "read_workspace",
        "verify",
        "synthesize_memory",
        "answer",
        "commit_observation",
        "reject_observation",
        "defer_observation",
        "no_commit_needed",
    }.issubset(tool_names)
    assert tool_names.isdisjoint(
        {
            "global_gist",
            "video_ls",
            "search_segments",
            "read_segment",
            "read_segment_detail",
            "inspect_segment",
            "verify_ledger_answer",
        }
    )


def test_workspace_v2_read_clip_normalizes_facts_only(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_read_clip")
    backend = RecordingBackend("The shield icon remains over Central Europe.")
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)

    result = registry.execute(
        "read_clip",
        {
            "scope": {"segment_id": "seg_0001"},
            "focus": ["shield icon meaning"],
            "sampling": {"nframes": 4},
        },
    )

    assert result["facts"][0]["source_kind"] == "visual_fact"
    assert result["facts"][0]["text"] == "The shield icon remains over Central Europe."
    assert "supported_option" not in result
    assert backend.requests[0].task == "vision_read"


def test_workspace_v2_verify_is_provenance_gate(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_verify")
    observation = workspace.write_observation(
        tool_name="read_clip",
        claim="Austria-Hungary was seen as a buffer between Russia and Western Europe.",
        confidence=0.9,
    )
    workspace.commit_observation(
        observation.observation_id,
        writes={
            "pinned_anchors": [
                {
                    "anchor_id": "anch_asr_206",
                    "kind": "asr",
                    "source_kind": "audio_fact",
                    "excerpt": "buffer between Russia and Western Europe",
                }
            ],
            "memory": [
                {
                    "kind": "answer_support",
                    "claim": "Narration says Austria-Hungary was a buffer.",
                    "anchor_ids": ["anch_asr_206"],
                    "confidence": "high",
                }
            ],
        },
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)

    accepted = registry.execute("verify", {"claim": "D", "against": {"citations": ["mem_0001"]}})
    rejected = registry.execute("verify", {"claim": "D", "against": {"citations": ["obs_missing"]}})

    assert accepted["accepted"] is True
    assert accepted["phase"] == "provenance_gate"
    assert rejected["accepted"] is False
    assert "unknown citation" in rejected["reason"]


def test_workspace_v2_verify_final_requires_memory_citation(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_verify_memory_required")
    workspace.write_observation(tool_name="read_clip", claim="Raw observation only.", confidence=0.7)
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)

    with pytest.raises(ValueError, match="planner-authored memory"):
        registry.execute("answer", {"text": "D", "citations": ["obs_0001"], "confidence": "high"})


def test_workspace_v2_synthesize_memory_derives_from_committed_memory(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_synthesize_memory")
    observation = workspace.write_observation(
        tool_name="read_clip",
        claim="Austria-Hungary was seen as a buffer between Russia and Western Europe.",
        confidence=0.9,
    )
    workspace.commit_observation(
        observation.observation_id,
        writes={
            "pinned_anchors": [
                {
                    "anchor_id": "anch_asr_206",
                    "kind": "asr",
                    "source_kind": "audio_fact",
                    "excerpt": "buffer between Russia and Western Europe",
                }
            ],
            "memory": [
                {
                    "kind": "answer_support",
                    "claim": "Narration says Austria-Hungary was a buffer.",
                    "anchor_ids": ["anch_asr_206"],
                    "confidence": "high",
                }
            ],
        },
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)

    result = registry.execute(
        "synthesize_memory",
        {
            "claim": "The narration supports option D's buffer description.",
            "supports": ["mem_0001"],
            "derived_from": ["mem_0001"],
            "evidence_obs_ids": [observation.observation_id],
            "confidence": "high",
            "tags": ["final_support"],
        },
    )

    entry = workspace.get_memory("mem_0002")
    assert result["memory_id"] == "mem_0002"
    assert entry is not None
    assert entry.kind == "synthesized"
    assert entry.previous_memory_refs == ("mem_0001",)
    assert [anchor.anchor_id for anchor in entry.anchors] == ["anch_asr_206"]
    assert entry.metadata["supports"] == ["mem_0001"]
    assert entry.metadata["evidence_obs_ids"] == [observation.observation_id]
