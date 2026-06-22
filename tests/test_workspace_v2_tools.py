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


class ExploreReasoningBackend(RecordingBackend):
    def __init__(self, reasoning: dict[str, object]) -> None:
        super().__init__()
        self.reasoning = reasoning

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        if request.task == "explore_caption_reasoning":
            return BackendResponse(text=json.dumps(self.reasoning), raw=self.reasoning)
        return super().generate(request)


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


def _gypsy_video_map() -> VideoMap:
    return VideoMap(
        video_path="/videos/gypsy.mp4",
        duration_sec=600.0,
        segments=[
            VideoMapSegment(
                segment_id="seg_0001",
                start_sec=0.0,
                end_sec=300.0,
                low_fps_caption="The video introduces the migration of Gypsies into Europe.",
                asr_text="When Gypsies migrated to Europe, they fought with Selic or Seljuk Turks.",
            ),
            VideoMapSegment(
                segment_id="seg_0002",
                start_sec=300.0,
                end_sec=600.0,
                low_fps_caption="A later section describes slavery in the Balkans under Ottoman expansion.",
                asr_text="Later, many became enslaved in the Balkans as the Ottomans expanded territory.",
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


def _commit_verified_memory(workspace: EvidenceWorkspace, *, kind: str = "answer_support") -> None:
    backend = RecordingBackend(
        raw={
            "facts": [{"text": "Austria-Hungary was a buffer.", "source_kind": "audio_fact", "confidence": 0.9}],
            "verification_results": [
                {
                    "target_id": "buffer",
                    "claim": "Austria-Hungary was a buffer.",
                    "verdict": "supported",
                    "confidence": 0.9,
                    "rationale": "Narration states the buffer relation.",
                }
            ],
        }
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)
    verified = registry.execute(
        "verify_window",
        {
            "segment_id": "seg_0001",
            "time_range": [0.0, 30.0],
            "checks": [{"target_id": "buffer", "claim": "Austria-Hungary was a buffer.", "polarity": "presence"}],
        },
    )
    observation = workspace.write_observation(
        tool_name="verify_window",
        claim=str(verified["claim"]),
        confidence=float(verified["confidence"]),
        regions=verified["regions"],
        limitations=str(verified["limitations"]),
        raw_output=verified,
    )
    anchor_id = verified["produced_anchors"][0]["anchor_id"]
    workspace.commit_observation(
        observation.observation_id,
        writes={
            "pinned_anchors": [
                {
                    "anchor_id": anchor_id,
                    "kind": "asr",
                    "source_kind": "audio_fact",
                    "excerpt": "Austria-Hungary was a buffer",
                }
            ],
            "memory": [
                {
                    "kind": kind,
                    "claim": "Austria-Hungary was a buffer.",
                    "anchor_ids": [anchor_id],
                    "confidence": "high",
                    "supports_option": "D",
                }
            ],
        },
    )


def test_workspace_v2_explore_returns_candidate_only_windows(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_explore")
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)

    result = registry.execute("explore", {"query": "buffer Russia", "modalities": ["asr"], "top_k": 3})

    assert result["mode"] == "candidate_discovery"
    assert result["support_status"] == "candidate_only"
    assert result["cannot_final_cite"] is True
    assert result["candidate_windows"][0]["candidate_id"] == "cand_0001"
    assert result["candidate_windows"][0]["candidate_key"] == "obs_0001:cand_0001"
    assert result["candidate_windows"][0]["source_observation_id"] == "obs_0001"
    assert result["candidate_windows"][0]["segment_id"] == "seg_0001"
    assert result["candidate_windows"][0]["status"] == "pending_verification"
    assert result["produced_anchors"][0]["observation_id"] == "__pending__"
    assert "navigation only" in result["limitations"]


def test_workspace_v2_explore_can_return_caption_fact_and_final_citable_memory(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_explore_caption_fact")
    backend = ExploreReasoningBackend(
        {
            "mode": "caption_fact",
            "support_status": "caption_supported",
            "claim": "When the Gypsies migrated to Europe, they fought with Selic/Seljuk Turks.",
            "confidence": 0.82,
            "facts": [
                {
                    "claim": "When the Gypsies migrated to Europe, they fought with Selic/Seljuk Turks.",
                    "confidence": 0.82,
                    "source_kind": "dense_caption",
                    "segment_id": "seg_0001",
                    "time_range": [0.0, 300.0],
                    "excerpt": "When Gypsies migrated to Europe, they fought with Selic or Seljuk Turks.",
                    "supports_option": "C",
                    "opposes_options": ["B"],
                }
            ],
            "anchors": [
                {
                    "source_kind": "dense_caption",
                    "segment_id": "seg_0001",
                    "time_range": [0.0, 300.0],
                    "excerpt": "When Gypsies migrated to Europe, they fought with Selic or Seljuk Turks.",
                    "reliability": "medium",
                }
            ],
            "candidate_windows": [],
            "answer_mapping": {
                "supports_option": "C",
                "opposes_options": ["B"],
                "reason": "The slavery caption is a later section and does not answer the migration event.",
            },
            "needs_visual_verify": False,
            "limitations": "Caption-level narration directly addresses the question condition.",
        }
    )
    registry = build_workspace_v2_registry(video_map=_gypsy_video_map(), backend=backend, workspace=workspace)

    result = registry.execute(
        "explore",
        {
            "query": "what happened when the Gypsies migrated to Europe",
            "targets": [{"target_id": "migration_event", "claim": "Event after Gypsies migrated to Europe"}],
            "modalities": ["caption", "asr"],
            "top_k": 4,
        },
    )

    assert result["mode"] == "caption_fact"
    assert result["support_status"] == "caption_supported"
    assert result["cannot_final_cite"] is False
    assert result["needs_visual_verify"] is False
    assert result["facts"][0]["supports_option"] == "C"
    assert result["answer_mapping"]["opposes_options"] == ["B"]
    assert result["produced_anchors"][0]["anchor_id"] == "anch_caption_obs_0001_001"
    assert backend.requests[0].task == "explore_caption_reasoning"

    observation = workspace.write_observation(
        tool_name="explore",
        claim=str(result["claim"]),
        confidence=float(result["confidence"]),
        regions=result["regions"],
        limitations=str(result["limitations"]),
        raw_output=result,
    )
    anchor_id = result["produced_anchors"][0]["anchor_id"]
    workspace.commit_observation(
        observation.observation_id,
        writes={
            "pinned_anchors": [
                {
                    "anchor_id": anchor_id,
                    "kind": "dense_caption",
                    "source_kind": "dense_caption",
                    "excerpt": "When Gypsies migrated to Europe, they fought with Selic or Seljuk Turks.",
                    "segment_id": "seg_0001",
                    "start_sec": 0.0,
                    "end_sec": 300.0,
                }
            ],
            "memory": [
                {
                    "kind": "caption_support",
                    "claim": "Gypsies fought with Selic/Seljuk Turks after migrating to Europe.",
                    "anchor_ids": [anchor_id],
                    "supports_option": "C",
                    "confidence": "medium",
                    "metadata": {"visual_verified": False},
                }
            ],
        },
    )

    memory = workspace.get_memory("mem_0001")
    assert memory is not None
    assert memory.kind == "caption_support"
    assert memory.metadata["source_tool"] == "explore"
    accepted = registry.execute("answer", {"text": "C", "citations": ["mem_0001"], "confidence": "medium"})
    assert accepted["accepted"] is True


def test_workspace_v2_answer_rejects_caption_support_when_visual_verify_required(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_caption_requires_visual")
    observation = workspace.write_observation(
        tool_name="explore",
        claim="Caption mentions two timeouts. The caption mentions two possible HUN timeouts.",
        confidence=0.8,
        raw_output={"mode": "caption_fact", "support_status": "caption_supported"},
    )
    workspace.commit_observation(
        observation.observation_id,
        writes={
            "pinned_anchors": [
                {
                    "anchor_id": "anch_caption_timeout",
                    "kind": "dense_caption",
                    "source_kind": "dense_caption",
                    "excerpt": "The caption mentions two possible HUN timeouts.",
                }
            ],
            "memory": [
                {
                    "kind": "caption_support",
                    "claim": "Caption-only timeout count requires visual verification.",
                    "anchor_ids": ["anch_caption_timeout"],
                    "confidence": "medium",
                    "metadata": {"requires_visual_verify": True},
                }
            ],
        },
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)

    with pytest.raises(ValueError, match="requires visual verification"):
        registry.execute("answer", {"text": "D", "citations": ["mem_0001"], "confidence": "medium"})


def test_workspace_v2_explore_supports_scoped_segment_query(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_explore_scoped")
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)

    result = registry.execute(
        "explore",
        {
            "query": "closing topic",
            "scope": {"segment_ids": ["seg_0002"]},
            "modalities": ["index"],
            "top_k": 2,
        },
    )

    assert result["candidate_windows"]
    assert {candidate["segment_id"] for candidate in result["candidate_windows"]} == {"seg_0002"}


def test_workspace_v2_explore_commit_required_predicate_depends_on_candidates(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_explore_predicate")
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)
    predicate = registry.get_runtime_spec("explore").commit_required_predicate

    assert predicate is not None
    assert predicate({"candidate_windows": [{"candidate_id": "cand_0001"}]}) is True
    assert predicate({"mode": "caption_fact", "facts": [{"claim": "caption fact"}], "candidate_windows": []}) is True
    assert predicate({"candidate_windows": []}) is False


def test_workspace_v2_read_workspace_reads_sections(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_read_workspace")
    workspace.note_entity(kind="concept", name="Austria-Hungary")
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)

    entities = registry.execute("read_workspace", {"section": "entities"})

    assert entities["items"][0]["name"] == "Austria-Hungary"


def test_workspace_v2_registry_installs_tool_normalizers(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_tool_specs")
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)

    for tool_name in ("explore", "verify_window", "read_workspace", "synthesize_memory", "answer"):
        assert registry.get_runtime_spec(tool_name).argument_normalizer is not None, tool_name


def test_workspace_v2_verify_window_normalizer_accepts_aliases(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_verify_window_aliases")
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)
    normalizer = registry.get_runtime_spec("verify_window").argument_normalizer
    assert normalizer is not None

    normalized = normalizer(
        _tool_spec_context(workspace, registry),
        ToolRequest(
            tool="verify_window",
            arguments={
                "candidate": "obs_0003:cand_0001",
                "facts": [{"name": "target_1", "question": "The Big Bang appears.", "kind": "presence"}],
                "modalities": ["ocr", "asr"],
                "max_frames": 64,
            },
        ),
    )

    assert normalized == {
        "candidate_key": "obs_0003:cand_0001",
        "candidate_id": "",
        "source_observation_id": "",
        "segment_id": "",
        "time_range": None,
        "evidence_mode": "ocr,asr",
        "focus": (),
        "checks": [{"target_id": "target_1", "claim": "The Big Bang appears.", "polarity": "presence", "expected_evidence": []}],
        "sampling": {"max_frames": 64},
    }


def test_verify_window_semantic_key_preserves_checks(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_verify_window_key")
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)
    key_builder = registry.get_runtime_spec("verify_window").semantic_key_builder
    assert key_builder is not None
    ctx = _tool_spec_context(workspace, registry)

    first = key_builder(
        ctx,
        ToolRequest(
            tool="verify_window",
            arguments={
                "candidate_key": "obs_0001:cand_0001",
                "checks": [{"target_id": "target_1", "claim": "shield", "polarity": "presence"}],
            },
        ),
    )
    second = key_builder(
        ctx,
        ToolRequest(
            tool="verify_window",
            arguments={
                "candidate_key": "obs_0001:cand_0001",
                "checks": [{"target_id": "target_2", "claim": "map label", "polarity": "presence"}],
            },
        ),
    )

    assert first != second


def test_workspace_v2_plan_phase_tool_surface_is_exact(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_tool_surface")
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)

    tool_names = {spec.tool_spec.name for spec in registry.list_runtime_specs()}

    assert tool_names == {
        "explore",
        "verify_window",
        "read_workspace",
        "synthesize_memory",
        "answer",
        "commit_observation",
        "reject_observation",
        "defer_observation",
        "no_commit_needed",
    }
    for removed in ("scan_segment", "read_clip", "read_segment", "verify", "search", "list"):
        with pytest.raises(ToolError, match="Unknown tool"):
            registry.get_runtime_spec(removed)


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


def test_workspace_v2_verify_window_normalizes_facts_only(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_verify_window_facts")
    backend = RecordingBackend("The shield icon remains over Central Europe.")
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)

    result = registry.execute(
        "verify_window",
        {
            "segment_id": "seg_0001",
            "time_range": [0.0, 60.0],
            "focus": ["shield icon meaning"],
            "sampling": {"nframes": 4},
        },
    )

    assert result["facts"][0]["source_kind"] == "visual_fact"
    assert result["facts"][0]["text"] == "The shield icon remains over Central Europe."
    assert "supported_option" not in result
    assert backend.requests[0].task == "vision_read"


def test_workspace_v2_verify_window_uses_sampled_frames_when_available(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_verify_window_frames")
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
        "verify_window",
        {
            "segment_id": "seg_0001",
            "time_range": [0.0, 60.0],
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


def test_workspace_v2_explore_and_verify_window_delegate_local_workers(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_explore_verify_workers")
    backend = RecordingBackend()
    sampled = []

    def fake_frame_sampler(video_path: str, start_sec: float, end_sec: float, max_frames: int) -> list[str]:
        sampled.append((video_path, start_sec, end_sec, max_frames))
        return ["/frames/demo/00010.jpg", "/frames/demo/00020.jpg"]

    registry = build_workspace_v2_registry(
        video_map=_video_map(),
        backend=backend,
        workspace=workspace,
        frame_sampler=fake_frame_sampler,
    )

    explored = registry.execute(
        "explore",
        {
            "query": "shield buffer",
            "scope": {"segment_ids": ["seg_0001"]},
            "modalities": ["visual", "asr"],
            "top_k": 1,
        },
    )
    workspace.write_observation(
        tool_name="explore",
        claim=str(explored["claim"]),
        confidence=float(explored["confidence"]),
        regions=explored["regions"],
        limitations=str(explored["limitations"]),
        raw_output=explored,
    )
    backend.text = "The local window shows a shield icon over Central Europe while narration describes Austria-Hungary as a buffer."

    verified = registry.execute(
        "verify_window",
        {
            "candidate_key": explored["candidate_windows"][0]["candidate_key"],
            "sampling": {"fps": 2, "max_frames": 32},
            "focus": ["shield meaning"],
        },
    )

    assert explored["candidate_windows"][0]["candidate_key"] == "obs_0001:cand_0001"
    assert verified["worker"] == "EvidenceVerifier"
    assert verified["candidate_key"] == "obs_0001:cand_0001"
    assert verified["mode"] == "verify_window"
    assert sampled == [("/videos/demo.mp4", 0.0, 30.0, 32)]
    assert backend.requests[-1].metadata["tool"] == "verify_window"
    plan_view = workspace.render_plan_view(question="Why?", video_map=_video_map())
    assert "## Pending Candidate Windows" in plan_view
    assert "obs_0001:cand_0001 [0.0-30.0] segment=seg_0001 status=pending_verification" in plan_view


def test_workspace_v2_verify_window_accepts_multiple_verification_targets(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_verify_multi_targets")
    backend = RecordingBackend(
        text=(
            "Target hubble: not found in this local window. "
            "Target telescope: a telescope graphic appears near the narration."
        )
    )
    registry = build_workspace_v2_registry(
        video_map=_video_map(),
        backend=backend,
        workspace=workspace,
        frame_sampler=lambda _video_path, _start_sec, _end_sec, max_frames: [f"/frames/{idx:03d}.jpg" for idx in range(max_frames)],
    )

    explore = registry.execute(
        "explore",
        {
            "query": "telescope",
            "scope": {"segment_ids": ["seg_0001"]},
            "modalities": ["visual", "asr"],
            "top_k": 1,
        },
    )
    workspace.write_observation(
        tool_name="explore",
        claim=str(explore["claim"]),
        confidence=float(explore["confidence"]),
        regions=explore["regions"],
        limitations=str(explore["limitations"]),
        raw_output=explore,
    )
    candidate_key = str(explore["candidate_windows"][0]["candidate_key"])

    verified = registry.execute(
        "verify_window",
        {
            "candidate_key": candidate_key,
            "evidence_mode": "multimodal",
            "sampling": {"fps": 2, "max_frames": 128},
            "focus": ["telescope identity"],
            "checks": [
                {"target_id": "hubble", "claim": "Hubble Telescope is mentioned or shown", "polarity": "presence"},
                {"target_id": "generic", "claim": "A telescope appears in the local window", "polarity": "presence"},
            ],
        },
    )

    assert verified["verification_targets"] == [
        {"target_id": "hubble", "claim": "Hubble Telescope is mentioned or shown", "polarity": "presence", "expected_evidence": []},
        {"target_id": "generic", "claim": "A telescope appears in the local window", "polarity": "presence", "expected_evidence": []},
    ]
    assert [result["target_id"] for result in verified["verification_results"]] == ["hubble", "generic"]
    assert {result["verdict"] for result in verified["verification_results"]} == {"uncertain"}
    request = backend.requests[-1]
    assert "Verification targets" in request.prompt
    assert "Hubble Telescope is mentioned or shown" in request.prompt
    assert "A telescope appears in the local window" in request.prompt
    assert request.metadata["verification_targets"] == verified["verification_targets"]
    assert verified["raw"]["verification_targets"] == verified["verification_targets"]


def test_workspace_v2_verify_window_resolves_candidate_key_and_rejects_ambiguous_bare_id(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_candidate_identity")
    backend = RecordingBackend(raw={"facts": [{"text": "The shield is visible.", "source_kind": "visual_fact"}]})
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)

    first = registry.execute("explore", {"query": "shield", "scope": {"segment_ids": ["seg_0001"]}, "top_k": 1})
    workspace.write_observation(
        tool_name="explore",
        claim=str(first["claim"]),
        confidence=float(first["confidence"]),
        regions=first["regions"],
        limitations=str(first["limitations"]),
        raw_output=first,
    )
    second = registry.execute("explore", {"query": "closing", "scope": {"segment_ids": ["seg_0002"]}, "top_k": 1})
    workspace.write_observation(
        tool_name="explore",
        claim=str(second["claim"]),
        confidence=float(second["confidence"]),
        regions=second["regions"],
        limitations=str(second["limitations"]),
        raw_output=second,
    )

    assert first["candidate_windows"][0]["candidate_id"] == second["candidate_windows"][0]["candidate_id"] == "cand_0001"
    assert first["candidate_windows"][0]["candidate_key"] != second["candidate_windows"][0]["candidate_key"]

    with pytest.raises(ValueError, match="ambiguous"):
        registry.execute(
            "verify_window",
            {
                "candidate_id": "cand_0001",
                "checks": [{"target_id": "shield", "claim": "The shield is visible.", "polarity": "presence"}],
            },
        )

    verified = registry.execute(
        "verify_window",
        {
            "candidate_key": first["candidate_windows"][0]["candidate_key"],
            "checks": [{"target_id": "shield", "claim": "The shield is visible.", "polarity": "presence"}],
        },
    )

    assert verified["candidate_key"] == first["candidate_windows"][0]["candidate_key"]
    assert verified["segment_id"] == "seg_0001"


def test_workspace_v2_verify_window_uses_backend_source_kind(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_verify_window_source_kind")
    backend = RecordingBackend("The label reads buffer zone.", raw={"source_kind": "ocr_fact"})
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)

    result = registry.execute(
        "verify_window",
        {"segment_id": "seg_0001", "time_range": [0.0, 60.0], "focus": ["shield icon"]},
    )

    assert result["facts"][0]["source_kind"] == "ocr_fact"
    assert result["produced_anchors"][0]["modality"] == "ocr"


def test_workspace_v2_verify_window_splits_sentence_facts(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_verify_window_sentence_facts")
    backend = RecordingBackend("Austria-Hungary is shown near Russia. A shield marks the buffer zone.")
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)

    result = registry.execute(
        "verify_window",
        {"segment_id": "seg_0001", "time_range": [0.0, 60.0], "focus": ["audio narration"]},
    )

    assert [fact["text"] for fact in result["facts"]] == [
        "Austria-Hungary is shown near Russia.",
        "A shield marks the buffer zone.",
    ]
    assert [fact["source_kind"] for fact in result["facts"]] == ["audio_fact", "audio_fact"]
    assert len(result["candidate_anchor_ids"]) == 2
    assert result["produced_anchors"][1]["field_path"] == "facts[1].text"


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


def test_workspace_v2_local_negative_memory_requires_scope_and_cannot_final_cite(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_local_negative_scope")
    observation = workspace.write_observation(
        tool_name="verify_window",
        claim="The Hubble Telescope is not found in this local window.",
        confidence=0.8,
        regions=[{"segment_id": "seg_0001", "time_range": [10.0, 20.0]}],
    )

    with pytest.raises(ValueError, match="local_negative.*scope"):
        workspace.commit_observation(
            observation.observation_id,
            writes={
                "pinned_anchors": [
                    {
                        "anchor_id": "anch_hubble_absent",
                        "kind": "asr",
                        "source_kind": "audio_fact",
                        "excerpt": "Hubble Telescope is not found",
                    }
                ],
                "memory": [
                    {
                        "kind": "local_negative",
                        "claim": "Hubble Telescope was not found in this local window.",
                        "anchor_ids": ["anch_hubble_absent"],
                        "confidence": "medium",
                    }
                ],
            },
        )

    workspace.commit_observation(
        observation.observation_id,
        writes={
            "pinned_anchors": [
                {
                    "anchor_id": "anch_hubble_absent",
                    "kind": "asr",
                    "source_kind": "audio_fact",
                    "excerpt": "Hubble Telescope is not found",
                }
            ],
            "memory": [
                {
                    "kind": "local_negative",
                    "claim": "Hubble Telescope was not found in this local window.",
                    "anchor_ids": ["anch_hubble_absent"],
                    "confidence": "medium",
                    "metadata": {
                        "scope": {"segment_id": "seg_0001", "time_range": [10.0, 20.0]},
                        "global_negation_allowed": False,
                    },
                }
            ],
        },
    )
    entry = workspace.get_memory("mem_0001")
    assert entry is not None
    assert entry.kind == "local_negative"
    assert entry.metadata["scope"] == {"segment_id": "seg_0001", "time_range": [10.0, 20.0]}
    assert entry.metadata["global_negation_allowed"] is False

    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)
    with pytest.raises(ValueError, match="answer_support"):
        registry.execute("answer", {"text": "A", "citations": ["mem_0001"], "confidence": "low"})


def test_workspace_v2_commit_rejects_explore_answer_support(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_explore_commit_gate")
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)
    explore = registry.execute("explore", {"query": "buffer", "top_k": 1})
    observation = workspace.write_observation(
        tool_name="explore",
        claim=str(explore["claim"]),
        confidence=float(explore["confidence"]),
        regions=explore["regions"],
        limitations=str(explore["limitations"]),
        raw_output=explore,
    )
    anchor = explore["produced_anchors"][0]

    with pytest.raises(ValueError, match="candidate-only explore observations cannot become answer support"):
        workspace.commit_observation(
            observation.observation_id,
            writes={
                "pinned_anchors": [
                    {
                        "anchor_id": anchor["anchor_id"],
                        "kind": anchor["source_kind"],
                        "source_kind": anchor["source_kind"],
                        "excerpt": anchor["excerpt"],
                    }
                ],
                "memory": [
                    {
                        "kind": "answer_support",
                        "claim": "Candidate-only evidence supports an answer.",
                        "anchor_ids": [anchor["anchor_id"]],
                        "confidence": "medium",
                    }
                ],
            },
        )


def test_workspace_v2_commit_accepts_supported_verify_window_answer_support(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_verify_commit_gate")
    backend = RecordingBackend(
        raw={
            "facts": [{"text": "The shield is visible in the inspected window.", "source_kind": "visual_fact", "confidence": 0.91}],
            "verification_results": [
                {
                    "target_id": "shield",
                    "claim": "The shield is visible.",
                    "verdict": "supported",
                    "confidence": 0.91,
                    "rationale": "The shield appears in the frame.",
                }
            ],
        }
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)
    verified = registry.execute(
        "verify_window",
        {
            "segment_id": "seg_0001",
            "time_range": [0.0, 30.0],
            "checks": [{"target_id": "shield", "claim": "The shield is visible.", "polarity": "presence"}],
        },
    )
    observation = workspace.write_observation(
        tool_name="verify_window",
        claim=str(verified["claim"]),
        confidence=float(verified["confidence"]),
        regions=verified["regions"],
        limitations=str(verified["limitations"]),
        raw_output=verified,
    )
    anchor_id = verified["produced_anchors"][0]["anchor_id"]

    workspace.commit_observation(
        observation.observation_id,
        writes={
            "pinned_anchors": [
                {
                    "anchor_id": anchor_id,
                    "kind": "visual",
                    "source_kind": "visual_fact",
                    "excerpt": "The shield is visible",
                }
            ],
            "memory": [
                {
                    "kind": "answer_support",
                    "claim": "The shield is visible.",
                    "anchor_ids": [anchor_id],
                    "confidence": "high",
                    "supports_option": "D",
                }
            ],
        },
    )

    entry = workspace.get_memory("mem_0001")
    assert entry is not None
    assert entry.metadata["source_tool"] == "verify_window"
    assert entry.metadata["source_observation_id"] == observation.observation_id
    assert entry.metadata["target_id"] == "shield"
    assert entry.metadata["verdict"] == "supported"
    result = registry.execute("answer", {"text": "D", "citations": ["mem_0001"], "confidence": "high"})
    assert result["accepted"] is True


def test_workspace_v2_commit_rejects_not_found_as_answer_support(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_not_found_commit_gate")
    backend = RecordingBackend(
        raw={
            "facts": [{"text": "Hubble is not found in this local window.", "source_kind": "visual_fact", "confidence": 0.8}],
            "verification_results": [
                {
                    "target_id": "hubble",
                    "claim": "Hubble appears.",
                    "verdict": "not_found_in_window",
                    "confidence": 0.8,
                    "rationale": "No Hubble reference is visible locally.",
                }
            ],
        }
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=backend, workspace=workspace)
    verified = registry.execute(
        "verify_window",
        {
            "segment_id": "seg_0001",
            "time_range": [0.0, 30.0],
            "checks": [{"target_id": "hubble", "claim": "Hubble appears.", "polarity": "presence"}],
        },
    )
    observation = workspace.write_observation(
        tool_name="verify_window",
        claim=str(verified["claim"]),
        confidence=float(verified["confidence"]),
        regions=verified["regions"],
        limitations=str(verified["limitations"]),
        raw_output=verified,
    )
    anchor_id = verified["produced_anchors"][0]["anchor_id"]

    with pytest.raises(ValueError, match="not_found_in_window.*answer_support"):
        workspace.commit_observation(
            observation.observation_id,
            writes={
                "pinned_anchors": [
                    {
                        "anchor_id": anchor_id,
                        "kind": "visual",
                        "source_kind": "visual_fact",
                        "excerpt": "Hubble is not found",
                    }
                ],
                "memory": [
                    {
                        "kind": "answer_support",
                        "claim": "Hubble is absent from the video.",
                        "anchor_ids": [anchor_id],
                        "confidence": "medium",
                    }
                ],
            },
        )


def test_workspace_v2_answer_rejects_legacy_tool_answer_support(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_legacy_answer_gate")
    observation = workspace.write_observation(
        tool_name="read_clip",
        claim="Legacy read_clip observation.",
        confidence=0.9,
    )
    workspace.commit_observation(
        observation.observation_id,
        writes={
            "pinned_anchors": [
                {
                    "anchor_id": "anch_legacy",
                    "kind": "asr",
                    "source_kind": "audio_fact",
                    "excerpt": "Legacy read_clip observation",
                }
            ],
            "memory": [
                {
                    "kind": "answer_support",
                    "claim": "Legacy read_clip memory should not be final-cited.",
                    "anchor_ids": ["anch_legacy"],
                    "confidence": "high",
                }
            ],
        },
    )
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)

    with pytest.raises(ValueError, match="source provenance|removed legacy tool"):
        registry.execute("answer", {"text": "D", "citations": ["mem_0001"], "confidence": "high"})


@pytest.mark.parametrize("kind", ["answer_support", "answer_conflict_resolved"])
def test_workspace_v2_answer_accepts_answer_supporting_memory_kinds(tmp_path: Path, kind: str) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, f"workspace_v2_answer_accepts_{kind}")
    _commit_verified_memory(workspace, kind=kind)
    registry = build_workspace_v2_registry(video_map=_video_map(), backend=RecordingBackend(), workspace=workspace)

    result = registry.execute("answer", {"text": "D", "citations": ["mem_0001"], "confidence": "high"})

    assert result["accepted"] is True


def test_workspace_v2_answer_rejects_mixed_final_citations_with_unsupported_memory(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_v2_answer_rejects_mixed_memory")
    _commit_verified_memory(workspace)
    observation = workspace.write_observation(tool_name="verify_window", claim="Extra observation.", confidence=0.5)
    workspace.commit_observation(
        observation.observation_id,
        writes={
            "pinned_anchors": [
                {
                    "anchor_id": "anch_extra",
                    "kind": "asr",
                    "source_kind": "audio_fact",
                    "excerpt": "Extra observation",
                }
            ],
            "memory": [
                {
                    "kind": "unverified_capture",
                    "claim": "Unverified capture must not be final-cited.",
                    "anchor_ids": ["anch_extra"],
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
    _commit_verified_memory(workspace)
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
    _commit_verified_memory(workspace)
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
    assert [anchor.anchor_id for anchor in entry.anchors] == ["clip_anch_seg_0001_00000000_00030000"]
    assert entry.metadata["supports"] == ["mem_0001"]
    assert entry.metadata["evidence_obs_ids"] == ["obs_0001"]
    answered = registry.execute("answer", {"text": "D", "citations": ["mem_0002"], "confidence": "high"})
    assert answered["accepted"] is True
