from pathlib import Path

from visual_coding_agent_harness.legacy.interpreter import ProgramInterpreter
from visual_coding_agent_harness.evidence.predicates import temporal_order_consistent
from visual_coding_agent_harness.core.registry import ToolRegistry, tool
from visual_coding_agent_harness.tools.verification import build_verification_registry
from visual_coding_agent_harness.workspace import EvidenceWorkspace


def test_timeline_auto_appended_from_vision_read(tmp_path: Path):
    registry = ToolRegistry()

    @tool(name="vision_read", description="Read a localized visual fact.")
    def vision_read():
        return {
            "claim": "The door opens at 12.3 seconds.",
            "confidence": 0.93,
            "event_label": "door opens",
            "observed_at_sec": 12.3,
            "start_sec": 10.0,
            "end_sec": 15.0,
            "grounding_quality": "visually_confirmed",
        }

    registry.register(vision_read)
    workspace = EvidenceWorkspace.create(tmp_path, "timeline_auto")

    ProgramInterpreter(registry=registry, workspace=workspace).run([{"tool": "vision_read"}])

    rows = workspace.read_timeline_sorted()
    assert (workspace.root / "timeline.md").exists()
    assert rows == [
        {
            "obs_id": "obs_0001",
            "entity": "door opens",
            "observed_at_sec": 12.3,
            "window": [10.0, 15.0],
            "confidence_signal": "visually_confirmed",
            "claim": "The door opens at 12.3 seconds.",
        }
    ]


def test_timeline_window_only_marked_when_no_explicit_timestamp(tmp_path: Path):
    registry = ToolRegistry()

    @tool(name="inspect_segment", description="Inspect a localized visual fact.")
    def inspect_segment():
        return {
            "claim": "The door is visible somewhere in this window.",
            "confidence": 0.81,
            "event_label": "door visible",
            "start_sec": 20.0,
            "end_sec": 30.0,
            "grounding_quality": "visually_confirmed",
        }

    registry.register(inspect_segment)
    workspace = EvidenceWorkspace.create(tmp_path, "timeline_window")

    ProgramInterpreter(registry=registry, workspace=workspace).run([{"tool": "inspect_segment"}])

    rows = workspace.read_timeline_sorted()
    assert rows[0]["obs_id"] == "obs_0001"
    assert rows[0]["entity"] == "door visible"
    assert rows[0]["observed_at_sec"] is None
    assert rows[0]["window"] == [20.0, 30.0]
    assert rows[0]["confidence_signal"] == "window_only"


def test_timeline_auto_appended_from_positive_caption_segment(tmp_path: Path):
    registry = ToolRegistry()

    @tool(name="caption_segment", description="Caption a localized temporal window.")
    def caption_segment():
        return {
            "claim": "The author presents Apollo and Daphne after discussing earlier sculptures.",
            "confidence": 0.72,
            "regions": [{"segment_id": "seg_0007", "start_sec": 1800.0, "end_sec": 1804.957}],
            "grounding_quality": "visually_confirmed",
        }

    registry.register(caption_segment)
    workspace = EvidenceWorkspace.create(tmp_path, "timeline_caption")

    ProgramInterpreter(registry=registry, workspace=workspace).run([{"tool": "caption_segment"}])

    rows = workspace.read_timeline_sorted()
    assert len(rows) == 1
    assert rows[0]["obs_id"] == "obs_0001"
    assert rows[0]["entity"].startswith("The author presents Apollo and Daphne")
    assert rows[0]["observed_at_sec"] is None
    assert rows[0]["window"] == [1800.0, 1804.957]
    assert rows[0]["confidence_signal"] == "window_only"


def test_timeline_skips_unsupported_caption_segment(tmp_path: Path):
    registry = ToolRegistry()

    @tool(name="caption_segment", description="Caption a localized temporal window.")
    def caption_segment():
        return {
            "claim": "The video does not depict Bernini's four masterpieces in this segment.",
            "confidence": 0.72,
            "regions": [{"segment_id": "seg_0002", "start_sec": 300.0, "end_sec": 600.0}],
        }

    registry.register(caption_segment)
    workspace = EvidenceWorkspace.create(tmp_path, "timeline_caption_negative")

    ProgramInterpreter(registry=registry, workspace=workspace).run([{"tool": "caption_segment"}])

    assert workspace.read_timeline_sorted() == []


def test_temporal_predicate_uses_only_confirmed_timeline_entries():
    table = {
        "timeline": [
            {
                "obs_id": "obs_door",
                "entity": "door opens",
                "observed_at_sec": 20.0,
                "confidence_signal": "visually_confirmed",
            },
            {
                "obs_id": "obs_light",
                "entity": "light turns on",
                "observed_at_sec": 5.0,
                "confidence_signal": "visually_confirmed",
            },
            {
                "obs_id": "obs_noise",
                "entity": "curtain moves",
                "observed_at_sec": 1.0,
                "confidence_signal": "window_only",
            },
        ],
        "rows": [],
    }

    support = temporal_order_consistent(
        table,
        expected_events=["light turns on", "door opens"],
    )
    blocked = temporal_order_consistent(
        table,
        expected_events=["curtain moves", "light turns on"],
    )

    assert support.passed
    assert not blocked.passed
    assert blocked.details["observed_events"] == [
        {"event": "light turns on", "start_sec": 5.0, "obs_id": "obs_light"},
        {"event": "door opens", "start_sec": 20.0, "obs_id": "obs_door"},
    ]


def test_verify_ledger_answer_temporal_gate_reads_timeline(tmp_path: Path):
    workspace = EvidenceWorkspace.create(tmp_path, "timeline_verify")
    workspace.append_to_timeline(
        obs_id="obs_light",
        entity="light turns on",
        observed_at_sec=5.0,
        confidence_signal="visually_confirmed",
    )
    workspace.append_to_timeline(
        obs_id="obs_door",
        entity="door opens",
        observed_at_sec=20.0,
        confidence_signal="visually_confirmed",
    )
    registry = build_verification_registry(workspace=workspace)

    result = registry.execute(
        "verify_ledger_answer",
        {
            "answer": "A. door opens then light turns on",
            "question": "Which order is shown?",
            "candidate_options": [
                "A. door opens then light turns on",
                "B. light turns on then door opens",
            ],
            "min_score": 0.0,
            "requires_visual_evidence": False,
        },
    )

    gate = result["regions"][0]["evidence_gate"]
    assert result["regions"][0]["verdict"] == "insufficient"
    assert "temporal order contradicts evidence" in gate["reasons"]
    assert gate["temporal_order"]["observed_events"] == [
        {"event": "light turns on", "start_sec": 5.0, "obs_id": "obs_light"},
        {"event": "door opens", "start_sec": 20.0, "obs_id": "obs_door"},
    ]
