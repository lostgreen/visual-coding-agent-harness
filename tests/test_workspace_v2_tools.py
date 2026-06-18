from pathlib import Path

import pytest

from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse, VisionLanguageBackend
from visual_coding_agent_harness.registry import ToolError
from visual_coding_agent_harness.tools.workspace_v2 import build_workspace_v2_registry
from visual_coding_agent_harness.video_map import VideoMap, VideoMapSegment
from visual_coding_agent_harness.workspace import EvidenceWorkspace


class RecordingBackend(VisionLanguageBackend):
    def __init__(self, text: str = "A shield icon appears over Central Europe.", raw: dict[str, object] | None = None) -> None:
        self.text = text
        self.raw = raw or {"supported_option": "D"}
        self.requests: list[BackendRequest] = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        return BackendResponse(text=self.text, raw=self.raw)


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


def test_workspace_v2_search_commit_required_predicate_depends_on_evidence_shape(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_search_predicate")
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)
    predicate = registry.get_runtime_spec("search").commit_required_predicate

    assert predicate is not None
    assert predicate({"results": [{"segment_id": "seg_0001", "excerpt": "Austria-Hungary was a buffer."}]}) is True
    assert predicate({"results": []}) is False
    assert predicate({"results": [{"segment_id": "seg_0001", "excerpt": "seg_0001"}]}) is False


def test_workspace_v2_verify_commit_required_predicate_tracks_rejections(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_verify_predicate")
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)
    predicate = registry.get_runtime_spec("verify").commit_required_predicate

    assert predicate is not None
    assert predicate({"accepted": True, "reason": ""}) is False
    assert predicate({"accepted": False, "reason": "unknown citation: mem_x"}) is True


def test_workspace_v2_list_reads_segments_and_workspace_sections(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_list")
    workspace.note_entity(kind="concept", name="Austria-Hungary")
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)

    segments = registry.execute("list", {"kind": "segments", "filter": {"segment_id": "seg_0001"}})
    entities = registry.execute("list", {"kind": "entities"})

    assert segments["items"][0]["segment_id"] == "seg_0001"
    assert entities["items"][0]["name"] == "Austria-Hungary"


def test_workspace_v2_plan_phase_tool_surface_is_exact(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_tool_surface")
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)

    tool_names = {spec.tool_spec.name for spec in registry.list_runtime_specs()}

    assert tool_names == {
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
    }


def test_workspace_v2_write_memory_backdoor_is_not_callable(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_no_write_memory_backdoor")
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)

    with pytest.raises(ToolError, match="Unknown tool: write_memory"):
        registry.execute(
            "write_memory",
            {
                "kind": "answer_support",
                "claim": "Bypass attempt.",
                "anchors": [{"anchor_id": "anch_search_seg_0001_001"}],
            },
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


def test_workspace_v2_read_clip_uses_backend_source_kind(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_read_clip_source_kind")
    backend = RecordingBackend("The label reads buffer zone.", raw={"source_kind": "ocr_fact"})
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)

    result = registry.execute("read_clip", {"scope": {"segment_id": "seg_0001"}, "focus": ["shield icon"]})

    assert result["facts"][0]["source_kind"] == "ocr_fact"
    assert result["produced_anchors"][0]["modality"] == "ocr"


def test_workspace_v2_read_clip_splits_sentence_facts(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_read_clip_sentence_facts")
    backend = RecordingBackend("Austria-Hungary is shown near Russia. A shield marks the buffer zone.")
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)

    result = registry.execute("read_clip", {"scope": {"segment_id": "seg_0001"}, "focus": ["audio narration"]})

    assert [fact["text"] for fact in result["facts"]] == [
        "Austria-Hungary is shown near Russia.",
        "A shield marks the buffer zone.",
    ]
    assert [fact["source_kind"] for fact in result["facts"]] == ["audio_fact", "audio_fact"]
    assert len(result["candidate_anchor_ids"]) == 2
    assert result["produced_anchors"][1]["field_path"] == "facts[1].text"


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


@pytest.mark.parametrize("kind", ["locator", "unverified_capture"])
def test_workspace_v2_answer_rejects_non_answer_supporting_memory_kinds(tmp_path: Path, kind: str) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, f"workspace_v2_answer_rejects_{kind}")
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
                    "kind": kind,
                    "claim": "A non-final memory entry exists.",
                    "anchor_ids": ["anch_asr_206"],
                    "confidence": "low",
                }
            ],
        },
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)

    with pytest.raises(ValueError, match="answer_support"):
        registry.execute("answer", {"text": "D", "citations": ["mem_0001"], "confidence": "high"})


@pytest.mark.parametrize("kind", ["answer_support", "synthesized_support", "answer_conflict_resolved"])
def test_workspace_v2_answer_accepts_answer_supporting_memory_kinds(tmp_path: Path, kind: str) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, f"workspace_v2_answer_accepts_{kind}")
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
                    "kind": kind,
                    "claim": "A final-supporting memory entry exists.",
                    "anchor_ids": ["anch_asr_206"],
                    "confidence": "high",
                }
            ],
        },
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)

    result = registry.execute("answer", {"text": "D", "citations": ["mem_0001"], "confidence": "high"})

    assert result["accepted"] is True


def test_workspace_v2_answer_rejects_mixed_final_citations_with_unsupported_memory(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_answer_rejects_mixed_memory")
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
                    "claim": "Answer support exists.",
                    "anchor_ids": ["anch_asr_206"],
                    "confidence": "high",
                },
                {
                    "kind": "unverified_capture",
                    "claim": "Unverified capture must not be final-cited.",
                    "anchor_ids": ["anch_asr_206"],
                    "confidence": "low",
                },
            ],
        },
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)

    with pytest.raises(ValueError, match="unsupported final citation"):
        registry.execute("answer", {"text": "D", "citations": ["mem_0001", "mem_0002"], "confidence": "high"})


def test_workspace_v2_answer_rejects_mixed_final_citations_with_raw_observation(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_answer_rejects_raw_obs")
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
                    "claim": "Answer support exists.",
                    "anchor_ids": ["anch_asr_206"],
                    "confidence": "high",
                }
            ],
        },
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)

    with pytest.raises(ValueError, match="final answer citations must be memory ids"):
        registry.execute("answer", {"text": "D", "citations": ["mem_0001", "obs_0001"], "confidence": "high"})


def test_workspace_v2_synthesize_memory_rejects_unverified_support_laundering(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_synthesize_rejects_unverified")
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
                    "kind": "unverified_capture",
                    "claim": "Unverified capture exists.",
                    "anchor_ids": ["anch_asr_206"],
                    "confidence": "low",
                }
            ],
        },
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)

    with pytest.raises(ValueError, match="supporting memory kind"):
        registry.execute(
            "synthesize_memory",
            {"claim": "laundered", "supports": ["mem_0001"], "confidence": "high"},
        )


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
            "evidence_obs_ids": ["obs_0099"],
            "confidence": "high",
            "tags": ["final_support"],
        },
    )

    entry = workspace.get_memory("mem_0002")
    assert result["memory_id"] == "mem_0002"
    assert entry is not None
    assert entry.kind == "synthesized_support"
    assert entry.previous_memory_refs == ("mem_0001",)
    assert [anchor.anchor_id for anchor in entry.anchors] == ["anch_asr_206"]
    assert entry.metadata["supports"] == ["mem_0001"]
    assert entry.metadata["evidence_obs_ids"] == [observation.observation_id]
