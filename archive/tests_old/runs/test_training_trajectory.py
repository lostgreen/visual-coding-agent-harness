import json
from pathlib import Path

from runs.training_trajectory import TrainingTrajectory
from visual_coding_agent_harness.legacy.workspace_v2 import EvidenceRecord, EvidenceWorkspace


def test_from_workspace_collects_all_chains_and_trace_context(tmp_path):
    workspace = _workspace_with_chain(tmp_path)
    workspace.write_trace_event("tool_use", {"step": 1, "tool": "vision_read", "arguments": {"segment_id": "seg_0001"}})
    prompt_meta = workspace.write_text_artifact(
        "artifacts/planner_io/round_0001_prompt.txt",
        "## Evidence\nobs_0001 | red car\n## Feedback\nnone",
    )
    response_meta = workspace.write_text_artifact(
        "artifacts/planner_io/round_0001_response.txt",
        '{"status": "continue"}',
    )
    workspace.write_trace_event(
        "planner_io",
        {
            "round": 1,
            "planner_input_mode": "text-only",
            "prompt": prompt_meta,
            "response": response_meta,
            "response_excerpt": '{"status": "continue"}',
        },
    )
    workspace.write_trace_event("tool_result", {"step": 1, "tool": "vision_read", "observation_id": "obs_0001"})
    workspace.write_trace_event(
        "context_budget_report",
        {"used_tokens_per_slot": {"task": 10, "navigation": 5, "evidence": 8, "feedback": 1}, "overflow": False},
    )
    workspace.write_trace_event("hard_skill_followup_handoff", {"rounds": 1})

    trajectory = TrainingTrajectory.from_workspace(
        workspace,
        case_id="case_001",
        question="Which object is visible?",
        options=["A. blue car", "B. red car"],
        ground_truth="B",
        final_decision="final",
        selected_option="B",
        is_correct=True,
        output_path="artifacts/training/case_001.json",
    )

    assert trajectory.schema_version == "TrainingTrajectoryV1"
    assert trajectory.case_id == "case_001"
    assert trajectory.evidence_chain_ids == [[record.evidence_id for record in _chain_records(workspace)]]
    assert trajectory.tool_calls[0]["tool"] == "vision_read"
    assert trajectory.tool_results[0]["observation_id"] == "obs_0001"
    assert trajectory.tool_results[0]["claim"] == "The localized window shows a red car."
    assert trajectory.tool_results[0]["visible_in_planner_rounds"] == [1]
    assert trajectory.planner_turns[0]["round"] == 1
    assert trajectory.planner_turns[0]["evidence_observation_ids"] == ["obs_0001"]
    assert trajectory.planner_turns[0]["prompt_artifact"]["sha256"]
    assert trajectory.context_budget_reports[0]["overflow"] is False
    assert trajectory.followup_history[0]["type"] == "hard_skill_followup_handoff"
    assert trajectory.framework_fallback_used is False
    assert trajectory.no_model_final is False
    assert trajectory.format_repair_count == 0
    assert trajectory.final_diagnostics == {}
    assert Path(trajectory.trajectory_path).exists()


def test_training_trajectory_export_does_not_inline_raw_output(tmp_path):
    workspace = _workspace_with_chain(tmp_path)

    trajectory = TrainingTrajectory.from_workspace(
        workspace,
        case_id="case_001",
        question="Which object is visible?",
        final_decision="final",
        selected_option="B",
        output_path="artifacts/training/case_001.json",
    )

    payload = json.loads(Path(trajectory.trajectory_path).read_text(encoding="utf-8"))
    assert "raw_output" not in json.dumps(payload)


def test_training_trajectory_flags_disabled_framework_final_events(tmp_path):
    workspace = _workspace_with_chain(tmp_path)
    workspace.write_trace_event("mcq_forced_fallback", {"answer": "A"})
    workspace.write_trace_event("iterative_final", {"answer": "A", "final_decision_owner": "framework"})

    trajectory = TrainingTrajectory.from_workspace(
        workspace,
        case_id="case_001",
        question="Which object is visible?",
        final_decision="final",
        selected_option="A",
    )

    assert trajectory.final_decision_owner == "framework"
    assert trajectory.framework_fallback_used is True
    assert "active trace emitted disabled mcq_forced_fallback" in trajectory.diagnostic_errors
    assert "active trace emitted framework-owned final decision" in trajectory.diagnostic_errors


def test_training_trajectory_exports_final_control_status_fields(tmp_path):
    workspace = _workspace_with_chain(tmp_path)
    workspace.write_trace_event("model_final_format_repair_result", {"status": "final"})
    workspace.write_trace_event(
        "structured_final_diagnostics",
        {
            "gate_status": "accepted",
            "reason_code": "",
            "advisory_only": True,
        },
    )
    workspace.write_trace_event("no_model_final", {"reason": "budget_exhausted", "final_decision_owner": "none"})

    trajectory = TrainingTrajectory.from_workspace(
        workspace,
        case_id="case_001",
        question="Which object is visible?",
        final_decision="no_model_final",
    )

    assert trajectory.no_model_final is True
    assert trajectory.format_repair_count == 1
    assert trajectory.final_diagnostics["gate_status"] == "accepted"


def test_training_trajectory_redacts_reasoning_leaks_from_public_fields(tmp_path):
    workspace = _workspace_with_chain(
        tmp_path,
        claim=(
            "The user is asking me to inspect the clip. "
            "The localized window shows a red car near the curb."
        ),
    )
    response_meta = workspace.write_text_artifact(
        "artifacts/planner_io/round_0001_response.txt",
        (
            "<think>The user wants hidden chain-of-thought.</think>"
            '{"status":"continue","rationale":"The user wants me to inspect the car",'
            '"program":[{"tool":"vision_read","args":{"segment_id":"seg_0001"}}]}'
        ),
    )
    prompt_meta = workspace.write_text_artifact(
        "artifacts/planner_io/round_0001_prompt.txt",
        "## Evidence\n(none)\n## Feedback\nnone",
    )
    workspace.write_trace_event(
        "planner_io",
        {
            "round": 1,
            "prompt": prompt_meta,
            "response": response_meta,
            "response_excerpt": (
                "<think>The user wants hidden chain-of-thought.</think>"
                '{"status":"continue","rationale":"The user wants me to inspect the car",'
                '"program":[{"tool":"vision_read","args":{"segment_id":"seg_0001"}}]}'
            ),
        },
    )
    workspace.write_trace_event(
        "iterative_plan",
        {
            "round": 1,
            "rationale": "The user wants me to inspect the car before answering.",
            "program": [{"tool": "vision_read", "args": {"segment_id": "seg_0001"}}],
        },
    )
    workspace.write_trace_event("tool_result", {"step": 1, "tool": "vision_read", "observation_id": "obs_0001"})

    trajectory = TrainingTrajectory.from_workspace(
        workspace,
        case_id="case_001",
        question="Which object is visible?",
        output_path="artifacts/training/case_001.json",
    )
    payload = json.loads(Path(trajectory.trajectory_path).read_text(encoding="utf-8"))
    encoded = json.dumps(payload)

    assert "hidden chain-of-thought" not in encoded
    assert "The user wants" not in encoded
    assert '"status":"continue"' in trajectory.planner_turns[0]["response_excerpt"]
    assert "rationale" not in trajectory.planner_turns[0]["response_excerpt"]
    assert trajectory.planner_plans[0]["rationale"] == ""
    assert trajectory.planner_plans[0]["rationale_redacted"] is True
    assert trajectory.tool_results[0]["claim"] == "The localized window shows a red car near the curb."
    assert trajectory.tool_results[0]["claim_redacted"] is True


def test_planner_turns_show_context_growth_without_inlining_prompts(tmp_path):
    workspace = _workspace_with_chain(tmp_path)
    first_prompt = workspace.write_text_artifact(
        "artifacts/planner_io/round_0001_prompt.txt",
        "## Evidence\n(none)\n## Feedback\nnone",
    )
    second_prompt = workspace.write_text_artifact(
        "artifacts/planner_io/round_0002_prompt.txt",
        "## Evidence\nobs_0001 | red car\n## Feedback\nnone",
    )
    first_response = workspace.write_text_artifact(
        "artifacts/planner_io/round_0001_response.txt",
        '{"status":"continue"}',
    )
    second_response = workspace.write_text_artifact(
        "artifacts/planner_io/round_0002_response.txt",
        '{"status":"final"}',
    )
    workspace.write_trace_event(
        "planner_io",
        {
            "round": 1,
            "planner_input_mode": "text-only",
            "prompt": first_prompt,
            "response": first_response,
            "response_excerpt": '{"status":"continue"}',
        },
    )
    workspace.write_trace_event(
        "route_tool_repaired",
        {
            "skill": "main_idea",
            "requested_tool": "vision_read",
            "resolved_tool": "global_gist",
            "reason": "repair_main_idea_vision_read_to_global_gist",
        },
    )
    workspace.write_trace_event(
        "iterative_plan",
        {
            "round": 1,
            "rationale": "inspect local evidence",
            "program": [{"tool": "vision_read", "args": {"segment_id": "seg_0001"}}],
        },
    )
    workspace.write_trace_event(
        "planner_io",
        {
            "round": 2,
            "planner_input_mode": "text-only",
            "prompt": second_prompt,
            "response": second_response,
            "response_excerpt": '{"status":"final"}',
        },
    )
    workspace.write_trace_event("tool_use", {"step": 1, "tool": "vision_read", "arguments": {"segment_id": "seg_0001"}})
    workspace.write_trace_event("tool_result", {"step": 1, "tool": "vision_read", "observation_id": "obs_0001"})

    trajectory = TrainingTrajectory.from_workspace(
        workspace,
        case_id="case_001",
        question="Which object is visible?",
        output_path="artifacts/training/case_001.json",
    )
    payload = json.loads(Path(trajectory.trajectory_path).read_text(encoding="utf-8"))

    assert [turn["evidence_observation_ids"] for turn in trajectory.planner_turns] == [[], ["obs_0001"]]
    assert trajectory.planner_turns[0]["prompt_artifact"]["sha256"] != trajectory.planner_turns[1]["prompt_artifact"]["sha256"]
    assert trajectory.planner_plans[0]["round"] == 1
    assert trajectory.route_repairs[0]["round"] == 1
    assert trajectory.tool_calls[0]["source_round"] == 1
    assert trajectory.tool_results[0]["source_round"] == 1
    assert trajectory.tool_results[0]["visible_in_planner_rounds"] == [2]
    assert "## Evidence" not in json.dumps(payload)


def test_training_trajectory_supports_multi_v3_model_io_events(tmp_path):
    workspace = EvidenceWorkspace.create(tmp_path / "workspace", run_id="multi_v3_case")
    observation = workspace.write_observation(
        tool_name="read_segment",
        claim="The selected window shows a doctor explaining visceral fat research with chart evidence.",
        confidence=0.82,
        regions=[{"segment_id": "seg_0003", "start_sec": 870.0, "end_sec": 900.0}],
        raw_output={
            "mode": "verify",
            "evidence_mode": "verify",
            "time_range": {"start_sec": 870.0, "end_sec": 900.0},
            "facts": [
                {
                    "time_range": {"start_sec": 872.0, "end_sec": 884.0},
                    "claim": "The speaker discusses visceral fat and cites chart evidence.",
                }
            ],
        },
    )
    log_root = tmp_path / "workspace_logs"
    prompt_1 = _write_log(log_root / "round_001_plan_prompt.txt", "## Evidence\n(none)\n## Feedback\nnone")
    response_1 = _write_log(
        log_root / "round_001_plan_response.txt",
        '{"tool":"read_segment","args":{"segment_id":"seg_0003","mode":"verify"}}',
    )
    prompt_2 = _write_log(
        log_root / "round_002_plan_prompt.txt",
        "## Evidence\nobs_0001 | claim: doctor explains visceral fat\n## Feedback\nnone",
    )
    response_2 = _write_log(
        log_root / "round_002_plan_response.txt",
        '{"tool":"answer","args":{"text":"C","citations":["mem_0001"],"confidence":"high"}}',
    )
    workspace.write_trace_event(
        "workspace_plan_model_io",
        {
            "round": 1,
            "prompt_path": prompt_1.as_posix(),
            "response_path": response_1.as_posix(),
            "prompt_chars": prompt_1.stat().st_size,
            "response_chars": response_1.stat().st_size,
            "response": response_1.read_text(encoding="utf-8"),
        },
    )
    workspace.write_trace_event("tool_use", {"step": 1, "tool": "read_segment", "arguments": {"segment_id": "seg_0003"}})
    workspace.write_trace_event(
        "tool_result",
        {"step": 1, "tool": "read_segment", "observation_id": observation.observation_id},
    )
    workspace.write_trace_event(
        "workspace_plan_model_io",
        {
            "round": 2,
            "prompt_path": prompt_2.as_posix(),
            "response_path": response_2.as_posix(),
            "prompt_chars": prompt_2.stat().st_size,
            "response_chars": response_2.stat().st_size,
            "response": response_2.read_text(encoding="utf-8"),
        },
    )

    trajectory = TrainingTrajectory.from_workspace(
        workspace,
        case_id="case_001",
        question="What evidence is shown?",
        output_path="artifacts/training/case_001.json",
    )

    assert [turn["round"] for turn in trajectory.planner_turns] == [1, 2]
    assert trajectory.planner_turns[0]["response_excerpt"] == (
        '{"args":{"mode":"verify","segment_id":"seg_0003"},"tool":"read_segment"}'
    )
    assert trajectory.planner_plans[0]["round"] == 1
    assert trajectory.planner_plans[0]["program"][0]["tool"] == "read_segment"
    assert trajectory.tool_calls[0]["source_round"] == 1
    assert trajectory.tool_results[0]["source_round"] == 1
    assert trajectory.tool_results[0]["visible_in_planner_rounds"] == [2]
    assert trajectory.tool_results[0]["time_range"] == {"start_sec": 870.0, "end_sec": 900.0}
    assert trajectory.tool_results[0]["facts"][0]["claim"] == "The speaker discusses visceral fat and cites chart evidence."


def _workspace_with_chain(
    tmp_path,
    *,
    claim: str = "The localized window shows a red car.",
) -> EvidenceWorkspace:
    workspace = EvidenceWorkspace.create(tmp_path, run_id="trajectory_case")
    observation = workspace.write_observation(
        tool_name="vision_read",
        claim=claim,
        confidence=0.9,
        regions=[{"segment_id": "seg_0001", "start_sec": 0.0, "end_sec": 12.0}],
        raw_output={"secret_raw_output": "should not be exported"},
    )
    distilled, ledger, mapped = _chain_records(workspace, observation_id=observation.observation_id)
    for record in [distilled, ledger, mapped]:
        workspace.write_evidence(record)
    return workspace


def _chain_records(workspace: EvidenceWorkspace, observation_id: str = "obs_0001") -> list[EvidenceRecord]:
    distilled = EvidenceRecord(
        evidence_id=f"ev_distilled_{workspace.run_id}_00001",
        stage="distilled",
        parent_id=None,
        tool="vision_read",
        observation_id=observation_id,
        frame_set_id=None,
        content={"claim": "red car"},
        grounding_quality="visually_confirmed",
        confidence=0.9,
        created_at=1.0,
    )
    ledger = EvidenceRecord(
        evidence_id=f"ev_ledger_{workspace.run_id}_00002",
        stage="ledger",
        parent_id=distilled.evidence_id,
        tool="vision_read",
        observation_id=observation_id,
        frame_set_id=None,
        content={"claim": "red car"},
        grounding_quality="visually_confirmed",
        confidence=0.9,
        created_at=1.0,
    )
    mapped = EvidenceRecord(
        evidence_id=f"ev_mapped_{workspace.run_id}_00003",
        stage="mapped",
        parent_id=ledger.evidence_id,
        tool="vision_read",
        observation_id=observation_id,
        frame_set_id=None,
        content={"candidate_option_relation": {"option": "B", "relation": "support", "strength": 0.9}},
        grounding_quality="visually_confirmed",
        confidence=0.9,
        created_at=1.0,
    )
    return [distilled, ledger, mapped]


def _write_log(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
