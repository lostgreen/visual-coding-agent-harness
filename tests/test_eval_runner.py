import os
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from visual_coding_agent_harness.agents.iterative_agent import AgentBudget, IterativeRound, IterativeRunResult
from visual_coding_agent_harness.video_index import SceneIndex, VideoSegment
from visual_coding_agent_harness.workspace import EvidenceRecord, EvidenceWorkspace


class EvalRunnerTest(unittest.TestCase):
    def test_eval_runner_script_entrypoint_imports_summary_schema(self):
        repo_root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"

        completed = subprocess.run(
            [sys.executable, "runs/eval_runner.py", "--help"],
            cwd=repo_root,
            env=env,
            check=False,
            text=True,
            capture_output=True,
            timeout=20,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr[:500])
        self.assertIn("Run reproducible VideoMME", completed.stdout)

    def test_run_eval_cases_writes_summary_for_requested_strategy_and_budget(self):
        from runs import eval_runner

        captured = {}

        def fake_run_loop(backend, **kwargs):
            captured["budget"] = kwargs["budget"]
            captured["scene_index"] = kwargs["scene_index"]
            captured["run_id"] = kwargs["run_id"]
            return {
                "answer": "B. The visual evidence supports option B.",
                "choice": "B",
                "status": "final",
                "confidence": 0.8,
                "citations": ["obs_0001"],
                "rounds": 2,
                "tools": ["video_ls", "inspect_segment"],
                "segments": ["seg_0001"],
                "seconds": 12.5,
            }

        rows_by_id = {
            "605-1": {
                "question_id": "605-1",
                "video_id": "vid605",
                "videoID": "video605",
                "task_type": "Information Synopsis",
                "question": "What is shown?",
                "options": ["A. one", "B. two", "C. three", "D. four"],
                "answer": "B",
            }
        }

        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "eval"
            config = eval_runner.EvalConfig(
                run_root=run_root,
                workspace_root=run_root / "workspaces",
                model_path="/model",
                data_root=Path("/dataset"),
                parquet_path=Path("/dataset/videomme/test.parquet"),
                video_dir=Path("/dataset/video"),
                subtitle_dir=Path("/dataset/subtitle"),
                cases=("605-1",),
                strategies=("empty_index_loop",),
                window_sec=300.0,
                budget=AgentBudget(max_rounds=8, max_tool_calls_per_round=2, default_nframes=12),
            )

            with patch.object(eval_runner, "run_loop", side_effect=fake_run_loop):
                summary = eval_runner.run_eval_cases(
                    backend=object(),
                    rows_by_id=rows_by_id,
                    config=config,
                    duration_fn=lambda path: 1896.0,
                )

            summary_path = run_root / "summary.json"
            evidence_chains_path = run_root / "evidence_chains.jsonl"
            self.assertTrue(summary_path.exists())
            self.assertTrue(evidence_chains_path.exists())
            self.assertEqual(summary["evidence_chains_path"], str(evidence_chains_path))
            self.assertEqual(summary["run_id"], "eval")
            self.assertEqual(summary["case_ids"], ["605-1"])
            self.assertEqual(summary["accuracy"], 1.0)
            self.assertEqual(summary["final_rate"], 1.0)
            self.assertEqual(summary["unsupported_final_rate"], 0.0)
            self.assertEqual(summary["legacy_worker_vote_rows"], 0)
            self.assertEqual(summary["route_violations"], 0)
            self.assertEqual(summary["per_case"], summary["cases"])
            case = summary["cases"][0]
            self.assertEqual(case["question_id"], "605-1")
            self.assertEqual(case["strategies"]["empty_index_loop"]["choice"], "B")
            self.assertTrue(case["strategies"]["empty_index_loop"]["correct"])
            self.assertEqual(case["raw_artifacts"]["workspaces"]["empty_index_loop"], str(config.workspace_root / "runs" / captured["run_id"]))
            self.assertEqual(captured["budget"].max_rounds, 8)
            self.assertEqual(captured["budget"].max_tool_calls_per_round, 2)
            self.assertEqual(captured["budget"].default_nframes, 12)
            self.assertIsInstance(captured["scene_index"], SceneIndex)
            self.assertEqual(captured["scene_index"].segments[0].source, "fixed_window_empty")

    def test_summary_payload_aggregates_route_violations_from_workspace_traces(self):
        from runs import eval_runner

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="route_trace")
            workspace.write_trace_event("route_violation", {"reason": "blocked tool"})
            workspace.write_trace_event("route_violation", {"reason": "blocked final"})

            summary = eval_runner._summary_payload(
                run_id="eval",
                case_ids=["case_001"],
                config_payload={},
                results=[
                    {
                        "question_id": "case_001",
                        "strategies": {"agent_v2": {"status": "need_more_evidence", "correct": False}},
                        "raw_artifacts": {"workspaces": {"agent_v2": str(workspace.root)}},
                    }
                ],
            )

            self.assertEqual(summary["route_violations"], 2)

    def test_summary_payload_aggregates_hard_skill_followup_metrics(self):
        from runs import eval_runner

        with tempfile.TemporaryDirectory() as tmp:
            success_workspace = EvidenceWorkspace.create(Path(tmp), run_id="followup_success")
            success_workspace.write_trace_event("hard_skill_runtime", {"skill": "temporal_ordering@v1"})
            success_workspace.write_trace_event("tool_use", {"tool": "ground_question", "arguments": {"query": "first gap"}})
            success_workspace.write_trace_event("tool_use", {"tool": "ground_question", "arguments": {"query": "second gap"}})
            success_workspace.write_trace_event("iterative_final", {"source": "hard_skill_runtime"})

            handoff_workspace = EvidenceWorkspace.create(Path(tmp), run_id="followup_handoff")
            handoff_workspace.write_trace_event("hard_skill_runtime", {"skill": "temporal_ordering@v1"})
            handoff_workspace.write_trace_event("tool_use", {"tool": "ground_question", "arguments": {"query": "remaining gap"}})
            handoff_workspace.write_trace_event("hard_skill_followup_handoff", {"rounds": 1})

            summary = eval_runner._summary_payload(
                run_id="eval",
                case_ids=["case_001", "case_002"],
                config_payload={},
                results=[
                    {
                        "question_id": "case_001",
                        "strategies": {"agent_v2": {"status": "final", "correct": True}},
                        "raw_artifacts": {"workspaces": {"agent_v2": str(success_workspace.root)}},
                    },
                    {
                        "question_id": "case_002",
                        "strategies": {"agent_v2": {"status": "max_rounds_reached", "correct": False}},
                        "raw_artifacts": {"workspaces": {"agent_v2": str(handoff_workspace.root)}},
                    },
                ],
            )

            self.assertEqual(summary["avg_followups_per_case"], 1.5)
            self.assertEqual(summary["followup_success_rate"], 0.5)

    def test_summary_payload_prefers_explicit_followup_attempt_metrics(self):
        from runs import eval_runner

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="followup_explicit")
            workspace.write_trace_event("followup_attempt", {"target_id": "fu_1"})
            workspace.write_trace_event("followup_attempt", {"target_id": "fu_2"})
            workspace.write_trace_event("iterative_answer_agent", {"status": "low_confidence_final"})

            summary = eval_runner._summary_payload(
                run_id="eval",
                case_ids=["case_001"],
                config_payload={},
                results=[
                    {
                        "question_id": "case_001",
                        "strategies": {"agent_v2": {"status": "low_confidence_final", "correct": False}},
                        "raw_artifacts": {"workspaces": {"agent_v2": str(workspace.root)}},
                    }
                ],
            )

            self.assertEqual(summary["avg_followups_per_case"], 2.0)
            self.assertEqual(summary["followup_success_rate"], 1.0)
            self.assertEqual(summary["low_confidence_final_rate"], 1.0)

    def test_summary_payload_aggregates_context_budget_metrics(self):
        from runs import eval_runner

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="context_budget")
            workspace.write_trace_event(
                "context_budget_report",
                {
                    "used_tokens_per_slot": {"task": 10, "navigation": 20, "evidence": 30, "feedback": 5},
                    "overflow": False,
                    "turn_index": 0,
                },
            )
            workspace.write_trace_event(
                "context_budget_report",
                {
                    "used_tokens_per_slot": {"task": 12, "navigation": 20, "evidence": 40, "feedback": 8},
                    "overflow": True,
                    "turn_index": 1,
                },
            )

            summary = eval_runner._summary_payload(
                run_id="eval",
                case_ids=["case_001"],
                config_payload={},
                results=[
                    {
                        "question_id": "case_001",
                        "strategies": {"agent_v2": {"status": "need_more_evidence", "correct": False}},
                        "raw_artifacts": {"workspaces": {"agent_v2": str(workspace.root)}},
                    }
                ],
            )

            self.assertEqual(summary["context_budget_overflow_count"], 1)
            self.assertEqual(summary["avg_tokens_per_turn"], 72)

    def test_summary_payload_computes_evidence_provenance_completeness(self):
        from runs import eval_runner

        with tempfile.TemporaryDirectory() as tmp:
            complete_workspace = EvidenceWorkspace.create(Path(tmp), run_id="complete_chain")
            root = EvidenceRecord(
                evidence_id=complete_workspace.next_evidence_id("distilled"),
                stage="distilled",
                parent_id=None,
                tool="vision_read",
                observation_id="obs_0001",
                frame_set_id=None,
                content={"claim": "red car"},
                grounding_quality="visually_confirmed",
                confidence=0.9,
                created_at=1.0,
            )
            ledger = EvidenceRecord(
                evidence_id=complete_workspace.next_evidence_id("ledger"),
                stage="ledger",
                parent_id=root.evidence_id,
                tool="vision_read",
                observation_id="obs_0001",
                frame_set_id=None,
                content={"claim": "red car"},
                grounding_quality="visually_confirmed",
                confidence=0.9,
                created_at=1.0,
            )
            mapped = EvidenceRecord(
                evidence_id=complete_workspace.next_evidence_id("mapped"),
                stage="mapped",
                parent_id=ledger.evidence_id,
                tool="vision_read",
                observation_id="obs_0001",
                frame_set_id=None,
                content={"candidate_option_relation": {"option": "B", "relation": "support"}},
                grounding_quality="visually_confirmed",
                confidence=0.9,
                created_at=1.0,
            )
            for record in [root, ledger, mapped]:
                complete_workspace.write_evidence(record)
            empty_workspace = EvidenceWorkspace.create(Path(tmp), run_id="empty_chain")

            summary = eval_runner._summary_payload(
                run_id="eval",
                case_ids=["case_001", "case_002"],
                config_payload={},
                results=[
                    {
                        "question_id": "case_001",
                        "strategies": {"agent_v2": {"status": "final", "correct": True}},
                        "raw_artifacts": {"workspaces": {"agent_v2": str(complete_workspace.root)}},
                    },
                    {
                        "question_id": "case_002",
                        "strategies": {"agent_v2": {"status": "need_more_evidence", "correct": False}},
                        "raw_artifacts": {"workspaces": {"agent_v2": str(empty_workspace.root)}},
                    },
                ],
            )

            self.assertEqual(summary["evidence_provenance_completeness"], 0.5)

    def test_agent_v2_uses_subtitle_index(self):
        from runs import eval_runner

        captured = {}

        def fake_run_loop(backend, **kwargs):
            captured["scene_index"] = kwargs["scene_index"]
            return {
                "answer": "D. Based on subtitles and visual inspection.",
                "choice": "D",
                "status": "final",
                "confidence": 0.7,
                "citations": ["obs_0001"],
                "rounds": 1,
                "tools": ["inspect_segment"],
                "segments": ["seg_0001"],
                "seconds": 7.0,
            }

        rows_by_id = {
            "611-2": {
                "question_id": "611-2",
                "video_id": "vid611",
                "videoID": "video611",
                "task_type": "Temporal Reasoning",
                "question": "What happens last?",
                "options": ["A. one", "B. two", "C. three", "D. four"],
                "answer": "D",
            }
        }

        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "eval"
            subtitle_dir = Path(tmp) / "subtitle"
            subtitle_dir.mkdir()
            (subtitle_dir / "video611.srt").write_text(
                "1\n00:00:02,000 --> 00:00:03,000\nopening clue\n",
                encoding="utf-8",
            )
            config = eval_runner.EvalConfig(
                run_root=run_root,
                workspace_root=run_root / "workspaces",
                model_path="/model",
                data_root=Path("/dataset"),
                parquet_path=Path("/dataset/videomme/test.parquet"),
                video_dir=Path("/dataset/video"),
                subtitle_dir=subtitle_dir,
                cases=("611-2",),
                strategies=("agent_v2",),
                window_sec=300.0,
                budget=AgentBudget(),
            )

            with patch.object(eval_runner, "run_loop", side_effect=fake_run_loop):
                eval_runner.run_eval_cases(
                    backend=object(),
                    rows_by_id=rows_by_id,
                    config=config,
                    duration_fn=lambda path: 1805.0,
                )

            self.assertEqual(captured["scene_index"].segments[0].source, "fixed_window_subtitle")
            self.assertIn("ASR/subtitle excerpt: opening clue", captured["scene_index"].segments[0].low_fps_caption)

    def test_free_explore_cli_builds_budgetless_agent_config(self):
        from runs import eval_runner

        parser = eval_runner.build_arg_parser()
        args = parser.parse_args(
            [
                "--strategy",
                "agent_v2",
                "--cases",
                "611-2",
                "--run-root",
                "/tmp/vcah-free",
                "--free-explore",
                "--hard-skill-runtime",
                "--free-max-rounds",
                "24",
                "--free-max-tool-calls-per-round",
                "4",
            ]
        )

        config = eval_runner.config_from_args(args)

        self.assertEqual(config.strategies, ("agent_v2",))
        self.assertEqual(config.cases, ("611-2",))
        self.assertTrue(config.budget.free_exploration)
        self.assertEqual(config.budget.max_rounds, 24)
        self.assertEqual(config.budget.max_tool_calls_per_round, 4)
        self.assertFalse(config.budget.reserve_final_round)
        self.assertTrue(config.budget.hard_skill_runtime)

    def test_context_budget_cli_flags_build_agent_config(self):
        from runs import eval_runner

        parser = eval_runner.build_arg_parser()
        args = parser.parse_args(
            [
                "--strategy",
                "agent_v2",
                "--cases",
                "611-2",
                "--run-root",
                "/tmp/vcah-context",
                "--context-budget-tokens",
                "9000",
                "--budget-ratios",
                "task:0.2,navigation:0.2,evidence:0.4,feedback:0.2",
            ]
        )

        config = eval_runner.config_from_args(args)

        self.assertEqual(config.budget.context_budget_tokens, 9000)
        self.assertEqual(
            config.budget.context_budget_ratios,
            {"task": 0.2, "navigation": 0.2, "evidence": 0.4, "feedback": 0.2},
        )

    def test_context_budget_cli_rejects_bad_ratio_sum(self):
        from runs import eval_runner

        parser = eval_runner.build_arg_parser()
        args = parser.parse_args(
            [
                "--run-root",
                "/tmp/vcah-context",
                "--budget-ratios",
                "task:0.1,navigation:0.1,evidence:0.1,feedback:0.1",
            ]
        )

        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            eval_runner.config_from_args(args)

    def test_run_loop_exports_longvideoagent_trajectory(self):
        from runs import eval_runner

        def fake_run_iterative_smoke(**kwargs):
            workspace = EvidenceWorkspace.create(base_dir=kwargs["base_dir"], run_id=kwargs["run_id"])
            workspace.write_trace_event(
                "tool_use",
                {"step": 1, "tool": "vision_read", "arguments": {"segment_id": "seg_0001"}},
            )
            observation = workspace.write_observation(
                tool_name="vision_read",
                claim="The localized window shows a red car.",
                confidence=0.9,
                regions=[{"segment_id": "seg_0001", "start_sec": 0.0, "end_sec": 12.0}],
                raw_output={
                    "grounding_quality": "visually_confirmed",
                    "candidate_option_relations": [{"option": "B", "relation": "support", "strength": 0.9}],
                },
            )
            distilled = EvidenceRecord(
                evidence_id=workspace.next_evidence_id("distilled"),
                stage="distilled",
                parent_id=None,
                tool="vision_read",
                observation_id=observation.observation_id,
                frame_set_id=None,
                content={"claim": observation.claim},
                grounding_quality="visually_confirmed",
                confidence=0.9,
                created_at=1.0,
            )
            workspace.write_evidence(distilled)
            workspace.write_ledger_entry(observation, parent_records=[distilled])
            workspace.write_trace_event(
                "tool_result",
                {"step": 1, "tool": "vision_read", "observation_id": observation.observation_id},
            )
            return IterativeRunResult(
                question=kwargs["question"],
                video_path=kwargs["media_path"],
                answer="B. red car",
                status="final",
                citations=[observation.observation_id],
                confidence=0.9,
                rounds=[
                    IterativeRound(
                        round_number=1,
                        status="final",
                        planner_text="",
                        program=[{"tool": "vision_read", "args": {"segment_id": "seg_0001"}}],
                        observation_ids=[observation.observation_id],
                    )
                ],
            )

        with tempfile.TemporaryDirectory() as tmp:
            workspace_root = Path(tmp) / "workspaces"
            scene_index = SceneIndex(
                video_path="/videos/demo.mp4",
                duration_sec=12.0,
                segments=[VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=12.0)],
            )
            with patch.object(eval_runner, "run_iterative_smoke", side_effect=fake_run_iterative_smoke):
                raw = eval_runner.run_loop(
                    backend=object(),
                    video_path="/videos/demo.mp4",
                    question="Which object is visible?\nA. blue car\nB. red car",
                    duration_sec=12.0,
                    run_id="case_agent_v2",
                    scene_index=scene_index,
                    workspace_root=workspace_root,
                    budget=AgentBudget(),
                    extract_clips=False,
                )

            trajectory_path = Path(raw["trajectory_path"])
            self.assertTrue(trajectory_path.exists())
            evidence_chains_path = Path(raw["evidence_chains_path"])
            self.assertTrue(evidence_chains_path.exists())
            self.assertEqual(raw["evidence_chain_count"], 1)
            self.assertEqual(raw["planner_prompt_count"], 0)
            self.assertIn("non_navigation_visual_citation", raw["reward_tags"])
            self.assertIn("final", raw["reward_tags"])

    def test_run_eval_cases_exports_training_trajectory_when_enabled(self):
        from runs import eval_runner

        def fake_run_loop(backend, **kwargs):
            workspace = _make_training_workspace(kwargs["workspace_root"], kwargs["run_id"])
            return {
                "answer": "B. red car",
                "choice": "B",
                "status": "final",
                "confidence": 0.9,
                "citations": ["obs_0001"],
                "rounds": 1,
                "tools": ["vision_read"],
                "segments": ["seg_0001"],
                "seconds": 1.0,
                "evidence_chain_count": len(workspace.evidence_chain_summaries()),
            }

        rows_by_id = {
            "605-1": {
                "question_id": "605-1",
                "video_id": "vid605",
                "videoID": "video605",
                "task_type": "Information Synopsis",
                "question": "What is shown?",
                "options": ["A. one", "B. two", "C. three", "D. four"],
                "answer": "B",
            }
        }

        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "eval"
            config = eval_runner.EvalConfig(
                run_root=run_root,
                workspace_root=run_root / "workspaces",
                model_path="/model",
                data_root=Path("/dataset"),
                parquet_path=Path("/dataset/videomme/test.parquet"),
                video_dir=Path("/dataset/video"),
                subtitle_dir=Path("/dataset/subtitle"),
                cases=("605-1",),
                strategies=("agent_v2",),
                window_sec=300.0,
                budget=AgentBudget(),
                export_training=True,
            )

            with patch.object(eval_runner, "run_loop", side_effect=fake_run_loop):
                summary = eval_runner.run_eval_cases(
                    backend=object(),
                    rows_by_id=rows_by_id,
                    config=config,
                    duration_fn=lambda path: 30.0,
                )

            case = summary["cases"][0]
            trajectory_path = Path(case["raw_artifacts"]["training_trajectories"]["agent_v2"])
            self.assertTrue(trajectory_path.exists())
            self.assertTrue(summary["training_trajectory_exported"])
            payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "TrainingTrajectoryV1")
            self.assertEqual(payload["ground_truth"], "B")


def _make_training_workspace(base_dir: Path, run_id: str) -> EvidenceWorkspace:
    workspace = EvidenceWorkspace.create(base_dir, run_id=run_id)
    workspace.write_trace_event("tool_use", {"step": 1, "tool": "vision_read", "arguments": {"segment_id": "seg_0001"}})
    observation = workspace.write_observation(
        tool_name="vision_read",
        claim="The localized window shows a red car.",
        confidence=0.9,
        regions=[{"segment_id": "seg_0001", "start_sec": 0.0, "end_sec": 12.0}],
    )
    distilled = EvidenceRecord(
        evidence_id=workspace.next_evidence_id("distilled"),
        stage="distilled",
        parent_id=None,
        tool="vision_read",
        observation_id=observation.observation_id,
        frame_set_id=None,
        content={"claim": observation.claim},
        grounding_quality="visually_confirmed",
        confidence=0.9,
        created_at=1.0,
    )
    ledger = EvidenceRecord(
        evidence_id=workspace.next_evidence_id("ledger", sequence_offset=1),
        stage="ledger",
        parent_id=distilled.evidence_id,
        tool="vision_read",
        observation_id=observation.observation_id,
        frame_set_id=None,
        content={"claim": observation.claim},
        grounding_quality="visually_confirmed",
        confidence=0.9,
        created_at=1.0,
    )
    mapped = EvidenceRecord(
        evidence_id=workspace.next_evidence_id("mapped", sequence_offset=2),
        stage="mapped",
        parent_id=ledger.evidence_id,
        tool="vision_read",
        observation_id=observation.observation_id,
        frame_set_id=None,
        content={"candidate_option_relation": {"option": "B", "relation": "support", "strength": 0.9}},
        grounding_quality="visually_confirmed",
        confidence=0.9,
        created_at=1.0,
    )
    for record in [distilled, ledger, mapped]:
        workspace.write_evidence(record)
    return workspace


if __name__ == "__main__":
    unittest.main()
