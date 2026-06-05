import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from visual_coding_agent_harness.agents.iterative_agent import AgentBudget, IterativeRound, IterativeRunResult
from visual_coding_agent_harness.video_index import SceneIndex, VideoSegment
from visual_coding_agent_harness.workspace import EvidenceWorkspace


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
            self.assertTrue(summary_path.exists())
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
            workspace.write_ledger_entry(observation)
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
            self.assertEqual(raw["planner_prompt_count"], 0)
            self.assertIn("non_navigation_visual_citation", raw["reward_tags"])
            self.assertIn("final", raw["reward_tags"])


if __name__ == "__main__":
    unittest.main()
