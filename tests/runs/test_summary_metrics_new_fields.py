from pathlib import Path

from runs import eval_runner, report_metrics
from runs.summary_schema import RunSummary
from visual_coding_agent_harness.legacy.workspace_v2 import EvidenceWorkspace


def test_phase_d_metrics_default_to_zero():
    summary = RunSummary.with_defaults("run_001", ["case_001"])

    assert summary.unsupported_citation_rate == 0.0
    assert summary.mutex_conflict_detection_count == 0
    assert summary.timeline_completeness == 0.0
    assert summary.degenerate_observation_rate == 0.0
    assert summary.normalization_notes_per_round == 0.0


def test_summary_payload_aggregates_phase_d_trace_metrics(tmp_path: Path):
    unsupported = EvidenceWorkspace.create(tmp_path, "unsupported_final")
    unsupported_obs = unsupported.write_observation(
        tool_name="vision_read",
        claim="The model says it cannot confirm the cited fact.",
        confidence=0.2,
        confidence_signal="unsupported",
    )
    unsupported.write_trace_event("iterative_final", {"citations": [unsupported_obs.observation_id]})

    supported = EvidenceWorkspace.create(tmp_path, "supported_final")
    supported_obs = supported.write_observation(
        tool_name="vision_read",
        claim="The door opens at 10 seconds.",
        confidence=0.9,
        confidence_signal="confirmed",
    )
    supported.write_trace_event("iterative_final", {"citations": [supported_obs.observation_id]})
    supported.write_trace_event(
        "iterative_timeline_temporal_decision",
        {"matched_events": [{"event": "light turns on"}, {"event": "door opens"}]},
    )

    diagnostic = EvidenceWorkspace.create(tmp_path, "diagnostic_need_more")
    diagnostic.write_observation(tool_name="vision_read", claim="valid", confidence=0.8)
    diagnostic.write_observation(tool_name="vision_read", claim="degenerate", confidence=0.1)
    diagnostic.write_trace_event("tool_output_degenerate", {"observation_id": "obs_0002"})
    diagnostic.write_trace_event(
        "iterative_answer_agent",
        {
            "status": "need_more_evidence",
            "missing_evidence": ["mutex_conflict: obs_0001 vs obs_0002"],
        },
    )
    diagnostic.write_trace_event(
        "timeline_ordering_missing_entity",
        {
            "target_facts": ["light turns on", "door opens"],
            "missing_entities": ["door opens"],
        },
    )
    diagnostic.write_trace_event(
        "iterative_normalization_empty",
        {"round": 1, "notes": [{"reason": "route_violation"}, {"reason": "tool_budget_exhausted"}]},
    )
    diagnostic.write_trace_event(
        "iterative_normalization_empty",
        {"round": 2, "notes": [{"reason": "unresolved_media_segment"}]},
    )

    summary = eval_runner._summary_payload(
        run_id="eval",
        case_ids=["case_001", "case_002", "case_003"],
        config_payload={},
        results=[
            {
                "question_id": "case_001",
                "strategies": {"multi_v3": {"status": "final", "correct": False, "citations": ["obs_0001"]}},
                "raw_artifacts": {"workspaces": {"multi_v3": str(unsupported.root)}},
            },
            {
                "question_id": "case_002",
                "strategies": {"multi_v3": {"status": "final", "correct": True, "citations": ["obs_0001"]}},
                "raw_artifacts": {"workspaces": {"multi_v3": str(supported.root)}},
            },
            {
                "question_id": "case_003",
                "strategies": {"multi_v3": {"status": "need_more_evidence", "correct": False}},
                "raw_artifacts": {"workspaces": {"multi_v3": str(diagnostic.root)}},
            },
        ],
    )

    assert summary["unsupported_citation_rate"] == 0.5
    assert summary["mutex_conflict_detection_count"] == 1
    assert summary["timeline_completeness"] == 0.75
    assert summary["degenerate_observation_rate"] == 0.25
    assert summary["normalization_notes_per_round"] == 1.5


def test_report_metrics_exposes_phase_d_strategy_metrics(tmp_path: Path):
    run_root = tmp_path / "run"
    workspace_root = run_root / "workspaces"
    unsupported = EvidenceWorkspace.create(workspace_root, "case_001_multi_v3")
    unsupported_obs = unsupported.write_observation(
        tool_name="vision_read",
        claim="Unsupported citation.",
        confidence=0.2,
        confidence_signal="unsupported",
    )
    unsupported.write_trace_event("iterative_final", {"citations": [unsupported_obs.observation_id]})

    diagnostic = EvidenceWorkspace.create(workspace_root, "case_002_multi_v3")
    diagnostic.write_observation(tool_name="vision_read", claim="valid", confidence=0.8)
    diagnostic.write_observation(tool_name="vision_read", claim="degenerate", confidence=0.1)
    diagnostic.write_trace_event("tool_output_degenerate", {"observation_id": "obs_0002"})
    diagnostic.write_trace_event(
        "iterative_answer_agent",
        {"status": "need_more_evidence", "missing_evidence": ["mutex_conflict: A vs B"]},
    )
    diagnostic.write_trace_event(
        "timeline_ordering_missing_entity",
        {"target_facts": ["first", "second"], "missing_entities": ["second"]},
    )
    diagnostic.write_trace_event(
        "iterative_normalization_empty",
        {"round": 1, "notes": [{"reason": "route_violation"}, {"reason": "route_violation"}]},
    )

    summary_path = run_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        """
{
  "cases": [
    {
      "question_id": "case_001",
      "gt": "D",
      "strategies": {
        "multi_v3": {
          "choice": "B",
          "correct": false,
          "status": "final",
          "citations": ["obs_0001"]
        }
      },
      "raw_artifacts": {"workspaces": {"multi_v3": "workspaces/runs/case_001_multi_v3"}}
    },
    {
      "question_id": "case_002",
      "gt": "B",
      "strategies": {
        "multi_v3": {
          "choice": "",
          "correct": false,
          "status": "need_more_evidence"
        }
      },
      "raw_artifacts": {"workspaces": {"multi_v3": "workspaces/runs/case_002_multi_v3"}}
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    report = report_metrics.build_report(summary_path)
    strategy = report["strategies"]["multi_v3"]

    assert strategy["unsupported_citation_rate"] == 1.0
    assert strategy["mutex_conflict_detection_count"] == 1
    assert strategy["timeline_completeness"] == 0.5
    assert strategy["degenerate_observation_rate"] == 1 / 3
    assert strategy["normalization_notes_per_round"] == 2.0

    rendered = report_metrics.render_markdown(report)
    assert "Unsupported Citations" in rendered
    assert "Timeline Completeness" in rendered
