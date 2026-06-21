import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse, VisionLanguageBackend
from visual_coding_agent_harness.core.protocol import ToolRequest
from visual_coding_agent_harness.core.registry import ToolError, ToolRegistry
from visual_coding_agent_harness.tools.workspace_v2 import build_workspace_v2_registry
from visual_coding_agent_harness.video.index import TimelineBeat
from visual_coding_agent_harness.video.map import IndexRefiner, VideoMap, VideoMapSegment, VideoMapStore
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
                timeline_beats=(
                    TimelineBeat(
                        beat_id="seg_0001_b01",
                        start_sec=0.0,
                        end_sec=30.0,
                        summary="A shield icon appears over Central Europe.",
                        modality_hints=("visual",),
                    ),
                ),
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


class RefinementBackend(RecordingBackend):
    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        return BackendResponse(
            text=(
                '{"children":[{"segment_id":"backend_supplied_child","start_sec":10,"end_sec":25,'
                '"summary":"Fresh local view shows the shield over Central Europe.",'
                '"beats":[{"start_sec":10,"end_sec":25,"summary":"Shield remains over the map.",'
                '"modality_hints":["visual"]}],'
                '"entity_hints":["shield","Central Europe"],"modality_hints":["visual"]}]}'
            ),
            raw={"source_kind": "visual_fact"},
        )


class InvalidRefinementBackend(RecordingBackend):
    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        if request.task == "refine_segment_index":
            return BackendResponse(text='{"summary":"local refined index"}')
        return super().generate(request)


@dataclass
class _ToolSpecContext:
    workspace: EvidenceWorkspace
    registry: ToolRegistry
    scene_index: object | None = None
    budget: object | None = None


def _tool_spec_context(workspace: EvidenceWorkspace, registry: ToolRegistry) -> _ToolSpecContext:
    return _ToolSpecContext(workspace=workspace, registry=registry)


def test_workspace_v2_search_returns_candidate_only_hits(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_search")
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)

    result = registry.execute("search", {"query": "buffer Russia", "modality": "asr"})

    assert result["results"][0]["segment_id"] == "seg_0001"
    assert result["results"][0]["support_status"] == "candidate_only"
    assert result["results"][0]["needs_local_read"] is True
    assert result["results"][0]["recommended_next_tool"] == "read_segment"
    assert result["results"][0]["recommended_mode"] == "verify"
    assert result["results"][0]["recommended_scope"] == {
        "segment_id": "seg_0001",
        "sub_window": {"start_sec": 0.0, "end_sec": 60.0},
    }
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


def test_workspace_v2_registry_installs_tool_normalizers(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_tool_specs")
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)

    for tool_name in ("read_segment", "read_clip", "search", "list", "verify", "synthesize_memory", "answer"):
        assert registry.get_runtime_spec(tool_name).argument_normalizer is not None, tool_name


def test_workspace_v2_verify_normalizer_accepts_legacy_citation_args(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_verify_legacy_args")
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)
    normalizer = registry.get_runtime_spec("verify").argument_normalizer
    assert normalizer is not None

    normalized = normalizer(
        _tool_spec_context(workspace, registry),
        ToolRequest(tool="verify", arguments={"answer": "D", "citations": ["mem_0001"], "final": True}),
    )

    assert normalized == {
        "claim": "D",
        "against": {"citations": ["mem_0001"], "final": True},
    }


def test_read_segment_verify_semantic_key_preserves_focus(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_read_segment_focus_key")
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)
    key_builder = registry.get_runtime_spec("read_segment").semantic_key_builder
    assert key_builder is not None
    ctx = _tool_spec_context(workspace, registry)

    first = key_builder(
        ctx,
        ToolRequest(
            tool="read_segment",
            arguments={
                "segment_id": "seg_0001",
                "mode": "verify",
                "sub_window": {"start_sec": 0.0, "end_sec": 60.0},
                "evidence_mode": "visual",
                "focus": ["shield"],
            },
        ),
    )
    second = key_builder(
        ctx,
        ToolRequest(
            tool="read_segment",
            arguments={
                "segment_id": "seg_0001",
                "mode": "verify",
                "sub_window": {"start_sec": 0.0, "end_sec": 60.0},
                "evidence_mode": "visual",
                "focus": ["map label"],
            },
        ),
    )

    assert first != second


def test_workspace_v2_plan_phase_tool_surface_is_exact(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_tool_surface")
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)

    tool_names = {spec.tool_spec.name for spec in registry.list_runtime_specs()}

    assert tool_names == {
        "read_segment",
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


def test_video_map_store_apply_refinement_rejects_child_recursion_and_parent_caption_copy() -> None:
    store = VideoMapStore(_video_map())
    child = VideoMapSegment(
        segment_id="seg_0001_r_medium_0010000_0025000",
        start_sec=10.0,
        end_sec=25.0,
        index_level="refined",
        parent_segment_id="seg_0001",
        root_segment_id="seg_0001",
        low_fps_caption="A map shows Central Europe with a shield icon.",
        refinement_state="refined",
    )

    with pytest.raises(ValueError, match="must not copy parent caption"):
        store.apply_refinement(
            parent_segment_id="seg_0001",
            requested_start_sec=10.0,
            requested_end_sec=25.0,
            resolution="medium",
            children=[child],
            provenance={"backend": "mock"},
        )

    fresh_child = VideoMapSegment(
        segment_id="seg_0001_r_medium_0010000_0025000",
        start_sec=10.0,
        end_sec=25.0,
        index_level="refined",
        parent_segment_id="seg_0001",
        root_segment_id="seg_0001",
        low_fps_caption="Fresh local view shows the shield.",
        refinement_state="refined",
    )
    patch = store.apply_refinement(
        parent_segment_id="seg_0001",
        requested_start_sec=10.0,
        requested_end_sec=25.0,
        resolution="medium",
        children=[fresh_child],
        provenance={"backend": "mock"},
    )

    assert patch.cache_hit is False
    assert store.current.get("seg_0001").index_level == "root"
    assert store.current.get(fresh_child.segment_id).index_level == "refined"
    cached = store.apply_refinement(
        parent_segment_id="seg_0001",
        requested_start_sec=10.0,
        requested_end_sec=25.0,
        resolution="medium",
        children=[fresh_child],
        provenance={"backend": "mock"},
    )
    assert cached.cache_hit is True
    with pytest.raises(ValueError, match="root parent"):
        store.apply_refinement(
            parent_segment_id=fresh_child.segment_id,
            requested_start_sec=10.0,
            requested_end_sec=25.0,
            resolution="medium",
            children=[fresh_child],
            provenance={"backend": "mock"},
        )


def test_video_map_store_apply_refinement_rejects_invalid_child_timing() -> None:
    base_kwargs = {
        "index_level": "refined",
        "parent_segment_id": "seg_0001",
        "root_segment_id": "seg_0001",
        "low_fps_caption": "Fresh local view.",
        "refinement_state": "refined",
    }
    invalid_sets = [
        [
            VideoMapSegment(
                segment_id="bad_empty",
                start_sec=10.0,
                end_sec=10.0,
                **base_kwargs,
            )
        ],
        [
            VideoMapSegment(segment_id="bad_2", start_sec=20.0, end_sec=25.0, **base_kwargs),
            VideoMapSegment(segment_id="bad_1", start_sec=15.0, end_sec=18.0, **base_kwargs),
        ],
        [
            VideoMapSegment(segment_id="bad_overlap_1", start_sec=10.0, end_sec=20.0, **base_kwargs),
            VideoMapSegment(segment_id="bad_overlap_2", start_sec=18.0, end_sec=25.0, **base_kwargs),
        ],
    ]

    for children in invalid_sets:
        store = VideoMapStore(_video_map())
        with pytest.raises(ValueError, match="duration|chronological"):
            store.apply_refinement(
                parent_segment_id="seg_0001",
                requested_start_sec=10.0,
                requested_end_sec=25.0,
                resolution="medium",
                children=children,
                provenance={"backend": "mock"},
            )


def test_index_refiner_uses_frame_cache_only_and_creates_fresh_children() -> None:
    store = VideoMapStore(_video_map())
    backend = RefinementBackend()
    sampled = []

    def fake_frame_sampler(video_path: str, start_sec: float, end_sec: float, max_frames: int) -> list[str]:
        sampled.append((video_path, start_sec, end_sec, max_frames))
        return ["/frames/demo/00010.jpg", "/frames/demo/00020.jpg"]

    refiner = IndexRefiner(backend=backend, frame_sampler=fake_frame_sampler)

    patch = refiner.refine(
        store,
        parent_segment_id="seg_0001",
        requested_start_sec=10.0,
        requested_end_sec=25.0,
        resolution="medium",
        focus=["shield"],
    )

    assert patch.cache_hit is False
    assert sampled == [("/videos/demo.mp4", 10.0, 25.0, 15)]
    assert backend.requests[0].task == "refine_segment_index"
    child = patch.children[0]
    assert child.segment_id == "seg_0001_r_medium_0010000_0025000"
    assert child.low_fps_caption != store.current.get("seg_0001").low_fps_caption
    assert child.timeline_beats[0].summary == "Shield remains over the map."


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


def test_workspace_v2_read_clip_uses_sampled_frames_when_available(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_read_clip_frames")
    backend = RecordingBackend("The shield icon remains over Central Europe.")
    sampled = []

    def fake_frame_sampler(video_path: str, start_sec: float, end_sec: float, max_frames: int) -> list[str]:
        sampled.append((video_path, start_sec, end_sec, max_frames))
        return ["/frames/demo/00001.jpg", "/frames/demo/00002.jpg"]

    registry = build_workspace_v2_registry(
        video_map=_video_map(),
        backend=backend,
        workspace=workspace,
        frame_sampler=fake_frame_sampler,
    )

    registry.execute(
        "read_clip",
        {
            "scope": {"segment_id": "seg_0001"},
            "focus": ["shield icon meaning"],
            "sampling": {"nframes": 2},
        },
    )

    request = backend.requests[0]
    assert sampled == [("/videos/demo.mp4", 0.0, 60.0, 2)]
    assert request.media_path is None
    assert request.media_type == "video"
    assert request.frames == ("/frames/demo/00001.jpg", "/frames/demo/00002.jpg")
    assert request.metadata["nframes"] == 2


def test_workspace_v2_read_segment_index_and_refine_are_navigation_only(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_read_segment_navigation")
    backend = RefinementBackend()
    sampled = []

    def fake_frame_sampler(video_path: str, start_sec: float, end_sec: float, max_frames: int) -> list[str]:
        sampled.append((video_path, start_sec, end_sec, max_frames))
        return ["/frames/demo/00010.jpg"]

    registry = build_workspace_v2_registry(
        video_map=_video_map(),
        backend=backend,
        workspace=workspace,
        index_refiner=IndexRefiner(backend=backend, frame_sampler=fake_frame_sampler),
    )

    indexed = registry.execute("read_segment", {"segment_id": "seg_0001", "mode": "index"})
    refined = registry.execute(
        "read_segment",
        {
            "segment_id": "seg_0001",
            "mode": "refine",
            "sub_window": {"start_sec": 10.0, "end_sec": 25.0},
            "resolution": "medium",
            "focus": ["shield"],
        },
    )

    assert indexed["timeline_beats"][0]["summary"] == "A shield icon appears over Central Europe."
    assert indexed["limitations"][0] == "navigation only; not answer evidence"
    assert refined["patch"]["cache_hit"] is False
    assert refined["commit_required"] is False
    assert sampled == [("/videos/demo.mp4", 10.0, 25.0, 15)]
    assert registry.get_runtime_spec("read_segment").commit_required is False


def test_workspace_v2_read_segment_requires_index_before_refine_or_verify(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_read_segment_requires_index")
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)

    with pytest.raises(ValueError, match="requires_index_read"):
        registry.execute(
            "read_segment",
            {
                "segment_id": "seg_0001",
                "mode": "refine",
                "sub_window": {"start_sec": 10.0, "end_sec": 25.0},
            },
        )

    with pytest.raises(ValueError, match="requires_index_read"):
        registry.execute(
            "read_segment",
            {
                "segment_id": "seg_0001",
                "mode": "verify",
                "sub_window": {"start_sec": 10.0, "end_sec": 25.0},
            },
        )

    indexed = registry.execute("read_segment", {"segment_id": "seg_0001", "mode": "index"})
    assert indexed["mode"] == "index"
    verified = registry.execute(
        "read_segment",
        {
            "segment_id": "seg_0001",
            "mode": "verify",
            "sub_window": {"start_sec": 10.0, "end_sec": 25.0},
        },
    )
    assert verified["mode"] == "verify"


def test_workspace_v2_refine_invalid_backend_output_is_rejected_with_artifacts(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_refine_invalid")
    backend = InvalidRefinementBackend()
    artifact_root = tmp_path / "artifacts" / "index_refinement"
    registry = build_workspace_v2_registry(
        video_map=_video_map(),
        backend=backend,
        workspace=workspace,
        index_refiner=IndexRefiner(backend=backend, artifact_root=artifact_root),
    )

    registry.execute("read_segment", {"segment_id": "seg_0001", "mode": "index"})
    with pytest.raises(ValueError, match="refinement_output_invalid"):
        registry.execute(
            "read_segment",
            {
                "segment_id": "seg_0001",
                "mode": "refine",
                "sub_window": {"start_sec": 10.0, "end_sec": 25.0},
            },
        )

    prefix = artifact_root / "seg_0001_0010000_0025000_medium"
    assert (prefix.with_name(prefix.name + "_request.json")).exists()
    assert (prefix.with_name(prefix.name + "_response.txt")).read_text() == '{"summary":"local refined index"}'
    validation = json.loads((prefix.with_name(prefix.name + "_validation.json")).read_text())
    assert validation["valid"] is False
    assert validation["error"].startswith("refinement_output_invalid")


def test_workspace_v2_read_segment_verify_reuses_read_clip_commit_contract(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_read_segment_verify")
    backend = RecordingBackend("The shield icon remains over Central Europe.")
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)

    registry.execute("read_segment", {"segment_id": "seg_0001", "mode": "index"})
    result = registry.execute(
        "read_segment",
        {
            "segment_id": "seg_0001",
            "mode": "verify",
            "sub_window": {"start_sec": 10.0, "end_sec": 25.0},
            "evidence_mode": "visual",
            "focus": ["shield"],
        },
    )
    spec = registry.get_runtime_spec("read_segment")

    assert result["facts"][0]["text"] == "The shield icon remains over Central Europe."
    assert result["regions"][0]["segment_id"] == "seg_0001"
    assert backend.requests[0].task == "vision_read"
    assert spec.commit_required_predicate is not None
    assert spec.commit_required_predicate(result) is True


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


@pytest.mark.parametrize("kind", ["locator", "unverified_capture", "retrieval_candidate"])
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
