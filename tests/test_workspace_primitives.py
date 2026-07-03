from pathlib import Path

import pytest

from visual_coding_agent_harness.memory import SourceAnchor
from visual_coding_agent_harness.legacy.workspace_v2.tools import build_workspace_primitives_registry
from visual_coding_agent_harness.video.index import TimelineBeat
from visual_coding_agent_harness.video.map import VideoMap, VideoMapSegment
from visual_coding_agent_harness.workspace import EvidenceWorkspace


def test_workspace_primitives_return_deterministic_results(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_primitives")
    workspace.write_observation(
        tool_name="vision_read",
        claim="The red door opens.",
        confidence=0.9,
        raw_output={"grounding_quality": "visually_confirmed"},
    )
    workspace.write_evidence_row(
        {
            "obs_id": "obs_0001",
            "tool": "vision_read",
            "claim": "The red door opens.",
            "grounding_quality": "visually_confirmed",
            "confidence": 0.9,
            "confidence_signal": "confirmed",
            "supported_option": "B",
        }
    )
    workspace.append_to_timeline(
        obs_id="obs_0001",
        entity="red door opens",
        observed_at_sec=12.0,
        confidence_signal="confirmed",
    )
    workspace.write_hypothesis({"slot_door": {"status": "empty", "evidence_obs_id": ""}})

    registry = build_workspace_primitives_registry(workspace=workspace, include=("all",))

    view = registry.execute("view_observation", {"obs_id": "obs_0001"})
    detail = registry.execute("read_observation_detail", {"obs_id": "obs_0001"})
    grep = registry.execute("grep_evidence", {"pattern": "red door"})
    query = registry.execute("query_evidence_table", {"filter": {"confidence_signal": "confirmed"}})
    timeline = registry.execute("read_timeline_sorted", {})
    hypothesis_before = registry.execute("read_hypothesis", {})
    updated = registry.execute(
        "update_hypothesis_slot",
        {"slot_name": "slot_door", "status": "satisfied", "evidence_obs_id": "obs_0001"},
    )
    hypothesis_after = registry.execute("read_hypothesis", {})

    assert view["regions"][0]["claim"] == "The red door opens."
    assert detail["regions"][0]["claim"] == "The red door opens."
    assert grep["regions"][0]["obs_ids"] == ["obs_0001"]
    assert query["regions"][0]["rows"][0]["obs_id"] == "obs_0001"
    assert timeline["regions"][0]["entries"][0]["obs_id"] == "obs_0001"
    assert hypothesis_before["regions"][0]["slots"]["slot_door"]["status"] == "empty"
    assert updated["regions"][0]["slot"]["status"] == "satisfied"
    assert hypothesis_after["regions"][0]["slots"]["slot_door"]["status"] == "satisfied"


def test_recent_tool_outputs_returns_latest_three_with_raw_payload(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "recent_tool_outputs")
    for index in range(4):
        workspace.write_observation(
            tool_name="vision_read",
            claim=f"Observation {index}.",
            confidence=0.7 + index / 10,
            raw_output={
                "visual_caption": f"caption {index}",
                "anchors_for_vlm": [{"segment_id": f"seg_{index:04d}"}],
                "long_field": "x" * 1200,
            },
        )
    workspace.write_evidence_row(
        {
            "obs_id": "obs_0004",
            "tool": "vision_read",
            "claim": "Observation 3.",
            "segment_id": "seg_0003",
            "grounding_quality": "visually_confirmed",
        }
    )

    outputs = workspace.recent_tool_outputs(limit=3)

    assert [item["observation_id"] for item in outputs] == ["obs_0002", "obs_0003", "obs_0004"]
    assert outputs[-1]["in_evidence_table"] is True
    assert outputs[-1]["segment_id"] == "seg_0003"
    assert outputs[-1]["modality"] == "visually_confirmed"
    assert outputs[-1]["raw_output"]["visual_caption"] == "caption 3"
    assert outputs[-1]["raw_output"]["anchors_for_vlm"] == [{"segment_id": "seg_0003"}]
    assert len(outputs[-1]["raw_output"]["long_field"]) < 1200


def test_write_memory_tool_persists_anchor_backed_memory(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_write_memory_tool")
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
    registry = build_workspace_primitives_registry(workspace=workspace, include=("internal",))

    result = registry.execute(
        "write_memory",
        {
            "kind": "support",
            "claim": "The narration says Austria-Hungary was a buffer.",
            "anchors": [{"anchor_id": "anch_seg_0005_asr_206", "excerpt": "buffer between Russia and Western Europe"}],
            "supports_option": "D",
            "confidence": "high",
            "role": "episodic",
            "layer": "visual",
            "embedding_refs": ["clip://seg_0005/frame_0012"],
            "metadata": {"source": "planner"},
        },
    )

    assert result["entry_id"] == "mem_0001"
    assert result["claim"] == "Memory mem_0001 written."
    entry = workspace.memory_entries()[0]
    assert entry.supports_option == "D"
    assert entry.role == "episodic"
    assert entry.layer == "visual"
    assert entry.embedding_refs == ("clip://seg_0005/frame_0012",)
    assert entry.metadata == {"source": "planner"}


def test_multi_v3_layout_creates_workspace_files(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "multi_v3_layout")

    expected_paths = [
        "index/coarse_segments.jsonl",
        "index/asr.jsonl",
        "index/shots.jsonl",
        "index/captions.jsonl",
        "entities/entities.jsonl",
        "events/events.jsonl",
        "relations/relations.jsonl",
        "attributes/attributes.jsonl",
        "memory/memory.jsonl",
        "notes/plan.md",
        "notes/open_questions.md",
        "observations/observations.jsonl",
        "observations/disposition.jsonl",
        "pinned/pinned_anchors.jsonl",
    ]

    for relative_path in expected_paths:
        assert (workspace.root / relative_path).exists(), relative_path
    assert (workspace.root / "memory.jsonl").exists()
    assert (workspace.root / "observations.jsonl").exists()


def test_commit_observation_writes_pinned_state_and_views(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_commit_observation")
    observation = workspace.write_observation(
        tool_name="read_clip",
        claim="The narration says Austria-Hungary was therefore seen as a good buffer between Russia and Western Europe.",
        confidence=0.88,
        raw_output={
            "facts": [
                {
                    "text": "Austria-Hungary was therefore seen as a good buffer between Russia and Western Europe.",
                    "source_kind": "audio_fact",
                    "time_range": [1234.5, 1240.0],
                }
            ]
        },
    )

    disposition = workspace.commit_observation(
        observation.observation_id,
        writes={
            "pinned_anchors": [
                {
                    "anchor_id": "anch_asr_206",
                    "kind": "asr",
                    "source_kind": "audio_fact",
                    "time_range": [1234.5, 1240.0],
                    "excerpt": "Austria-Hungary was therefore seen as a good buffer between Russia and Western Europe.",
                }
            ],
            "entities": [
                {"kind": "concept", "name": "Austria-Hungary"},
                {"kind": "location", "name": "Russia"},
                {"kind": "location", "name": "Western Europe"},
            ],
            "relations": [
                {
                    "subject": "Austria-Hungary",
                    "predicate": "acts_as_buffer_between",
                    "objects": ["Russia", "Western Europe"],
                    "time_range": [1234.5, 1240.0],
                    "evidence_obs_ids": [observation.observation_id],
                }
            ],
            "memory": [
                {
                    "kind": "answer_support",
                    "claim": "The narration describes Austria-Hungary as a buffer between Russia and Western Europe.",
                    "supports_option": "D",
                    "anchor_ids": ["anch_asr_206"],
                    "evidence_obs_ids": [observation.observation_id],
                    "confidence": "high",
                }
            ],
            "plan_update": "Need visual confirmation that the shield symbol is the same buffer concept.",
            "open_questions_add": ["Does the shield visual appear near the same explanatory span?"],
        },
    )

    assert disposition["disposition"] == "committed"
    assert workspace.observation_status(observation.observation_id) == "committed"
    assert workspace.read_workspace_section("entities")[0]["name"] == "Austria-Hungary"
    assert workspace.read_workspace_section("relations")[0]["predicate"] == "acts_as_buffer_between"
    assert workspace.memory_entries()[0].entry_id == "mem_0001"
    assert workspace.read_workspace_section("pinned_anchors")[0]["anchor_id"] == "anch_asr_206"

    plan_view = workspace.render_plan_view(
        question="Why was Austria-Hungary shown between Russia and Western Europe?",
    )
    assert "# Workspace" in plan_view
    assert "## Committed Memory" in plan_view
    assert "mem_0001" in plan_view
    assert "anch_asr_206" in plan_view


def test_commit_observation_rejects_excerpt_not_present_in_observation(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_commit_fake_excerpt")
    observation = workspace.write_observation(
        tool_name="read_clip",
        claim="The map shows a shield icon.",
        confidence=0.7,
    )

    with pytest.raises(ValueError, match="excerpt must appear"):
        workspace.commit_observation(
            observation.observation_id,
            writes={
                "pinned_anchors": [
                    {
                        "anchor_id": "anch_fake",
                        "kind": "asr",
                        "source_kind": "audio_fact",
                        "excerpt": "Austria-Hungary was a buffer.",
                    }
                ]
            },
        )


def test_commit_view_includes_facts_candidate_anchors_and_scope(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_commit_view_details")
    facts = [
        {
            "text": f"fact {index}: Austria-Hungary buffer detail.",
            "source_kind": "audio_fact",
            "confidence": 0.8,
            "time_range": [10.0 + index, 11.0 + index],
        }
        for index in range(9)
    ]
    observation = workspace.write_observation(
        tool_name="read_clip",
        claim="The clip discusses Austria-Hungary as a buffer.",
        confidence=0.8,
        regions=[{"segment_id": "seg_0001", "start_sec": 10.0, "end_sec": 20.0}],
        raw_output={
            "facts": facts,
            "candidate_anchor_ids": ["anch_clip_seg_0001_001"],
            "produced_anchors": [
                {
                    "anchor_id": "anch_clip_seg_0001_001",
                    "observation_id": "__pending__",
                    "source_kind": "audio_fact",
                    "segment_id": "seg_0001",
                    "start_sec": 10.0,
                    "end_sec": 20.0,
                    "field_path": "facts[0].text",
                    "excerpt": "fact 0: Austria-Hungary buffer detail.",
                    "modality": "asr",
                }
            ],
        },
    )

    view = workspace.render_commit_view(question="Why?", observation_id=observation.observation_id)

    assert "## Scope" in view
    assert "segment=seg_0001 time=[10.0-20.0]" in view
    assert "## Facts" in view
    assert "fact 0: Austria-Hungary buffer detail." in view
    assert "fact 8: Austria-Hungary buffer detail." not in view
    assert "... more facts hidden: 1" in view
    assert "candidate_anchor_ids: anch_clip_seg_0001_001" in view
    assert "## Candidate Anchors (verbatim excerpts you may pin)" in view
    assert "anch_clip_seg_0001_001 [asr]: fact 0: Austria-Hungary buffer detail." in view


def test_commit_view_includes_search_hits_without_produced_anchors(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_commit_view_search_hits")
    observation = workspace.write_observation(
        tool_name="search",
        claim="Search for 'buffer' returned one hit.",
        confidence=0.8,
        raw_output={
            "results": [
                {
                    "hit_id": "hit_001",
                    "modality": "asr",
                    "segment_id": "seg_0001",
                    "time_range": [10.0, 20.0],
                    "excerpt": "Austria-Hungary was seen as a buffer.",
                }
            ]
        },
    )

    view = workspace.render_commit_view(question="Why?", observation_id=observation.observation_id)

    assert "## Search Hits" in view
    assert "hit_001 [asr] Austria-Hungary was seen as a buffer." in view


def test_commit_view_includes_search_hits_when_candidate_anchors_exist(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_commit_view_search_hits_with_anchors")
    observation = workspace.write_observation(
        tool_name="search",
        claim="Search for 'buffer' returned one hit.",
        confidence=0.8,
        raw_output={
            "results": [
                {
                    "hit_id": "hit_001",
                    "modality": "asr",
                    "segment_id": "seg_0001",
                    "time_range": [10.0, 20.0],
                    "excerpt": "Austria-Hungary was seen as a buffer.",
                }
            ],
            "produced_anchors": [
                {
                    "anchor_id": "anch_search_seg_0001_001",
                    "observation_id": "__pending__",
                    "source_kind": "retrieval_hit",
                    "segment_id": "seg_0001",
                    "start_sec": 10.0,
                    "end_sec": 20.0,
                    "field_path": "results",
                    "excerpt": "Austria-Hungary was seen as a buffer.",
                    "modality": "asr",
                }
            ],
        },
    )

    view = workspace.render_commit_view(question="Why?", observation_id=observation.observation_id)

    assert "## Candidate Anchors (verbatim excerpts you may pin)" in view
    assert "## Search Hits" in view
    assert "hit_001 [asr] Austria-Hungary was seen as a buffer." in view


def test_failed_commit_observation_does_not_leave_partial_workspace_writes(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_commit_atomic_failure")
    observation = workspace.write_observation(
        tool_name="read_clip",
        claim="The map shows a shield icon.",
        confidence=0.7,
    )

    with pytest.raises(ValueError, match="attribute_validation_failed"):
        workspace.commit_observation(
            observation.observation_id,
            writes={
                "pinned_anchors": [
                    {
                        "anchor_id": "anch_shield",
                        "kind": "visual_fact",
                        "source_kind": "visual_fact",
                        "excerpt": "shield icon",
                    }
                ],
                "entities": [{"kind": "concept", "name": "shield icon"}],
                "attributes": [{"target": "shield icon", "name": "", "value": "visible"}],
            },
        )

    assert workspace.observation_status(observation.observation_id) == "uncommitted"
    assert workspace.read_workspace_section("pinned_anchors") == []
    assert workspace.read_workspace_section("entities") == []
    assert workspace.observation_dispositions() == []


def test_pure_read_tool_observations_default_to_auto_acknowledged(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_auto_ack")
    observation = workspace.write_observation(tool_name="list", claim="Listed segments.", confidence=1.0)

    assert workspace.observation_status(observation.observation_id) == "auto_acknowledged"


def test_search_observations_require_explicit_disposition(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_search_disposition")
    observation = workspace.write_observation(tool_name="search", claim="One candidate ASR hit.", confidence=1.0)

    assert workspace.observation_status(observation.observation_id) == "uncommitted"


def test_deferred_observations_are_prioritized_in_plan_view(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_deferred_view")
    observation = workspace.write_observation(tool_name="read_clip", claim="Needs another anchor.", confidence=0.5)
    workspace.defer_observation(
        observation.observation_id,
        until="after_event_anchor_resolved",
        reason="Need the event anchor first.",
    )

    plan_view = workspace.render_plan_view(question="What does the shield mean?")

    assert "## Deferred Observations" in plan_view
    assert observation.observation_id in plan_view
    assert "after_event_anchor_resolved" in plan_view


def test_plan_view_folds_large_sections_and_shows_read_workspace_hint(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_plan_view_folding")
    workspace.write_produced_anchors(
        [
            SourceAnchor(
                anchor_id="anch_shared",
                observation_id="obs_seed",
                source_kind="audio_fact",
                excerpt="shared evidence",
            )
        ]
    )
    for index in range(30):
        workspace.write_memory(
            kind="answer_support",
            claim=f"memory claim {index:02d}",
            anchors=[{"anchor_id": "anch_shared"}],
            confidence="high",
        )
    observation = workspace.write_observation(
        tool_name="search",
        claim="Search found a candidate buffer cue.",
        confidence=0.8,
    )
    workspace.no_commit_needed(observation.observation_id, reason="No durable evidence.")

    plan_view = workspace.render_plan_view(question="What does the buffer cue mean?")

    assert "memory claim 09" in plan_view
    assert "memory claim 10" not in plan_view
    assert "... shown 10/30; use read_workspace(section=\"memory\") to see more" in plan_view
    assert "obs_0001 (search -> acknowledged): Search found a candidate buffer cue." in plan_view
    assert "## Budget" in plan_view
    assert "workspace tokens ~" in plan_view


def test_plan_view_renders_compact_root_ledger_and_index_evidence_coverage(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_plan_view_index_coverage")
    video_map = VideoMap(
        video_path="/videos/demo.mp4",
        duration_sec=90.0,
        segments=[
            VideoMapSegment(
                segment_id="seg_0001",
                start_sec=0.0,
                end_sec=60.0,
                low_fps_caption="Root overview of a Central Europe map.",
                timeline_beats=(
                    TimelineBeat(
                        beat_id="seg_0001_b01",
                        start_sec=0.0,
                        end_sec=20.0,
                        summary="This beat should only appear through read_segment(index).",
                    ),
                ),
            ),
            VideoMapSegment(
                segment_id="seg_0001_r_medium_0010000_0020000",
                start_sec=10.0,
                end_sec=20.0,
                low_fps_caption="Fresh refined local map view.",
                index_level="refined",
                parent_segment_id="seg_0001",
                root_segment_id="seg_0001",
                refinement_state="refined",
            ),
        ],
    )
    observation = workspace.write_observation(
        tool_name="read_segment",
        claim="The shield remains visible.",
        confidence=0.8,
    )
    workspace.commit_observation(
        observation.observation_id,
        writes={
            "pinned_anchors": [
                {
                    "anchor_id": "clip_anch_seg_0001",
                    "kind": "visual",
                    "source_kind": "visual_fact",
                    "excerpt": "shield remains visible",
                    "segment_id": "seg_0001",
                    "start_sec": 10.0,
                    "end_sec": 20.0,
                }
            ],
            "memory": [
                {
                    "kind": "answer_support",
                    "claim": "The shield remains visible.",
                    "anchor_ids": ["clip_anch_seg_0001"],
                    "confidence": "high",
                }
            ],
        },
    )

    plan_view = workspace.render_plan_view(question="Why?", video_map=video_map)

    assert "## Segment Cards" in plan_view
    assert "seg_0001 [0.0-60.0s] navigation_only=true index_status=available" in plan_view
    assert "scan_segment" not in plan_view
    assert "read_segment(index)" not in plan_view
    assert "verify_window" in plan_view
    assert "explicit time_range" in plan_view
    assert "Summary: Root overview" in plan_view
    assert "dense_video_caption:" not in plan_view
    assert "This beat should only appear through read_segment(index)." not in plan_view
    assert "## Index Coverage" in plan_view
    assert "root indexed: 0.0-60.0s (1 roots)" in plan_view
    assert "refined: 10.0-20.0s" in plan_view
    assert "Index coverage != evidence coverage" in plan_view
    assert "## Evidence Coverage" in plan_view
    assert "answer_support_memories: 1" in plan_view


def test_plan_view_includes_segment_time_coverage_and_gaps(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_time_coverage")
    video_map = VideoMap(
        video_path="/videos/demo.mp4",
        duration_sec=100.0,
        segments=[
            VideoMapSegment(
                segment_id="seg_0001",
                start_sec=0.0,
                end_sec=100.0,
                low_fps_caption="Root overview",
                index_level="root",
            )
        ],
    )
    for start_sec, end_sec in [(0.0, 10.0), (20.0, 30.0)]:
        workspace.write_observation(
            tool_name="verify_window",
            claim="Checked local window.",
            confidence=0.6,
            raw_output={"mode": "verify_window", "segment_id": "seg_0001", "time_range": [start_sec, end_sec]},
        )

    plan_view = workspace.render_plan_view(question="Which item is absent?", video_map=video_map)

    assert "## Segment Time Coverage" in plan_view
    assert "seg_0001 [0.0-100.0]: verified 20.0%" in plan_view
    assert "covered: [0.0-10.0], [20.0-30.0]" in plan_view
    assert "uncovered: [10.0-20.0] (10.0s), [30.0-100.0] (70.0s)" in plan_view


def test_plan_view_omits_legacy_sweep_recommendation(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_time_coverage_sweep")
    video_map = VideoMap(
        video_path="/videos/demo.mp4",
        duration_sec=100.0,
        segments=[
            VideoMapSegment(
                segment_id="seg_0001",
                start_sec=0.0,
                end_sec=100.0,
                low_fps_caption="Root overview",
                index_level="root",
            )
        ],
    )
    workspace.write_observation(
        tool_name="verify_window",
        claim="Checked first window.",
        confidence=0.6,
        raw_output={"mode": "verify_window", "segment_id": "seg_0001", "time_range": [0.0, 10.0]},
    )

    plan_view = workspace.render_plan_view(question="Which option is not discussed?", video_map=video_map)

    assert "verified 10.0%" in plan_view
    assert "MUST verify_window" not in plan_view
    assert "sweep largest uncovered region" not in plan_view


def test_plan_view_does_not_truncate_root_index_summaries(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_plan_view_full_root_index")
    long_summary = (
        "The video explains the collapse of the Austro-Hungarian Empire, highlighting that its multi-ethnic "
        "nature, war losses, diplomatic pressure, and nationalist movements all contributed to the sequence "
        "of political changes shown across the root segment."
    )
    video_map = VideoMap(
        video_path="/videos/demo.mp4",
        duration_sec=600.0,
        segments=[
            VideoMapSegment(
                segment_id="seg_0001",
                start_sec=0.0,
                end_sec=300.0,
                low_fps_caption=long_summary,
                timeline_beats=(
                    TimelineBeat(
                        beat_id="seg_0001_b01",
                        start_sec=0.0,
                        end_sec=120.0,
                        summary="Opening maps introduce the empire's multi-ethnic structure and early pressure points.",
                        entity_hints=("Austro-Hungarian Empire",),
                        modality_hints=("visual", "asr"),
                    ),
                    TimelineBeat(
                        beat_id="seg_0001_b02",
                        start_sec=120.0,
                        end_sec=300.0,
                        summary="Later narration connects war losses and nationalist movements to the collapse.",
                        entity_hints=("nationalist movements",),
                        modality_hints=("asr",),
                    ),
                ),
            ),
            VideoMapSegment(
                segment_id="seg_0002",
                start_sec=300.0,
                end_sec=600.0,
                low_fps_caption="The next root segment remains visible even when other sections are budget-limited.",
                timeline_beats=(
                    TimelineBeat(
                        beat_id="seg_0002_b01",
                        start_sec=300.0,
                        end_sec=600.0,
                        summary="A follow-up section keeps the causal timeline in view.",
                    ),
                ),
            ),
        ],
    )

    plan_view = workspace.render_plan_view(question="What is the main arc?", video_map=video_map, max_per_section=1)

    assert long_summary in plan_view
    root_index = plan_view.split("## Segment Cards", 1)[1].split("## Index Coverage", 1)[0]
    assert "seg_0002 [300.0-600.0s]" in root_index
    assert "dense_video_caption:" not in root_index
    assert "Opening maps introduce the empire's multi-ethnic structure" not in root_index
    assert "Later narration connects war losses and nationalist movements" not in root_index
    assert "A follow-up section keeps the causal timeline in view." not in root_index
    assert "Entities: Austro-Hungarian Empire, nationalist movements" in root_index
    assert "Modalities: visual, asr" in root_index
    assert "shown 1/2" not in root_index
    assert "..." not in root_index


def test_workspace_disposition_tools_and_read_workspace(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_disposition_tools")
    reject_obs = workspace.write_observation(tool_name="read_clip", claim="Contradicted fact.", confidence=0.2)
    defer_obs = workspace.write_observation(tool_name="read_clip", claim="Needs another anchor.", confidence=0.5)
    ack_obs = workspace.write_observation(tool_name="list", claim="Listed segments only.", confidence=1.0)
    registry = build_workspace_primitives_registry(workspace=workspace)

    rejected = registry.execute(
        "reject_observation",
        {"observation_id": reject_obs.observation_id, "reason": "Contradicts committed ASR."},
    )
    deferred = registry.execute(
        "defer_observation",
        {
            "observation_id": defer_obs.observation_id,
            "until": "after_event_anchor_resolved",
            "reason": "Need the event anchor first.",
        },
    )
    acknowledged = registry.execute(
        "no_commit_needed",
        {"observation_id": ack_obs.observation_id, "reason": "No new evidence."},
    )
    workspace_read = registry.execute(
        "read_workspace",
        {"section": "observations_by_id", "filter": {"observation_id": reject_obs.observation_id}},
    )

    assert rejected["regions"][0]["disposition"] == "rejected"
    assert deferred["regions"][0]["defer_count"] == 1
    assert acknowledged["regions"][0]["disposition"] == "acknowledged"
    assert workspace.observation_status(reject_obs.observation_id) == "rejected"
    assert workspace.observation_status(defer_obs.observation_id) == "deferred"
    assert workspace.observation_status(ack_obs.observation_id) == "acknowledged"
    assert workspace_read["regions"][0]["observations"][0]["observation_id"] == reject_obs.observation_id


def test_commit_observation_rejects_pin_outside_observation_produced_anchors(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "workspace_commit_strict_anchor")
    observation = workspace.write_observation(
        tool_name="read_clip",
        claim="The clip discusses Austria-Hungary as a buffer.",
        confidence=0.8,
        raw_output={
            "facts": [{"text": "Austria-Hungary was seen as a buffer."}],
            "produced_anchors": [
                {
                    "anchor_id": "anch_real",
                    "observation_id": "__pending__",
                    "source_kind": "audio_fact",
                    "field_path": "facts[0].text",
                    "excerpt": "Austria-Hungary was seen as a buffer.",
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="not in observation produced_anchors"):
        workspace.commit_observation(
            observation.observation_id,
            writes={
                "pinned_anchors": [
                    {
                        "anchor_id": "anch_fake",
                        "kind": "asr",
                        "source_kind": "audio_fact",
                        "excerpt": "Austria-Hungary was seen as a buffer.",
                    }
                ]
            },
        )
