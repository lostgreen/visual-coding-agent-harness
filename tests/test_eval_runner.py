import os
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from visual_coding_agent_harness.core.budget import AgentBudget
from visual_coding_agent_harness.video.index import SceneIndex, VideoSegment
from visual_coding_agent_harness.workspace import EvidenceRecord, EvidenceWorkspace


class FakeFrameCache:
    fps = 2.0
    frame_dir = Path("/tmp/frame-cache/fake_2fps")

    def sample_paths(self, video_path, start_sec, end_sec, max_frames):
        return ()


class FakeSceneIndexBuilder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def build(self, **kwargs):
        return SceneIndex(
            video_path=kwargs["video_path"],
            duration_sec=kwargs["duration_sec"],
            segments=[
                VideoSegment(
                    segment_id="seg_0001",
                    start_sec=0.0,
                    end_sec=300.0,
                    source="dual_source_scene_index",
                )
            ],
        )


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

    def test_videomme_runner_accepts_workspace_v2_strategy(self):
        from runs import eval_runner

        self.assertEqual(eval_runner.parse_strategies(None), ("workspace_v2",))
        self.assertEqual(eval_runner.parse_strategies(("workspace_v2",)), ("workspace_v2",))
        with self.assertRaisesRegex(ValueError, "Unknown strategy: direct_full_video"):
            eval_runner.parse_strategies(("direct_full_video",))

    def test_make_question_keeps_agent_input_to_raw_question_and_options(self):
        from runs import eval_runner

        question = eval_runner.make_question(
            {
                "question": "What's the main idea of the video?",
                "options": [
                    "A. What did the French gain from World War One.",
                    "B. Why the Austro-Hungarian Empire was divided.",
                    "C. The process of World War One.",
                    "D. How the Austro-Hungarian Empire rises and falls.",
                ],
            }
        )

        self.assertEqual(
            question,
            "Question: What's the main idea of the video?\n"
            "Options:\n"
            "A. What did the French gain from World War One.\n"
            "B. Why the Austro-Hungarian Empire was divided.\n"
            "C. The process of World War One.\n"
            "D. How the Austro-Hungarian Empire rises and falls.",
        )
        self.assertNotIn("VideoMME", question)
        self.assertNotIn("outside knowledge", question)
        self.assertNotIn("exactly one option letter", question)

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
                strategies=("workspace_v2",),
                window_sec=300.0,
                budget=AgentBudget(max_rounds=8, max_tool_calls_per_round=2, default_nframes=12),
            )

            with patch.object(eval_runner, "build_frame_cache_for_video", return_value=FakeFrameCache()):
                with patch.object(eval_runner, "SceneIndexBuilder", FakeSceneIndexBuilder):
                    with patch.object(eval_runner, "run_loop", side_effect=fake_run_loop):
                        summary = eval_runner.run_eval_cases(
                            backend=object(),
                            rows_by_id=rows_by_id,
                            config=config,
                            duration_fn=lambda path: 1896.0,
                        )

            summary_path = run_root / "summary.json"
            run_config_path = run_root / "run_config.json"
            evidence_chains_path = run_root / "evidence_chains.jsonl"
            self.assertTrue(summary_path.exists())
            self.assertTrue(run_config_path.exists())
            self.assertTrue(evidence_chains_path.exists())
            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            self.assertEqual(run_config["budget"]["default_nframes"], 12)
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
            self.assertEqual(case["strategies"]["workspace_v2"]["choice"], "B")
            self.assertTrue(case["strategies"]["workspace_v2"]["correct"])
            self.assertEqual(case["raw_artifacts"]["workspaces"]["workspace_v2"], str(config.workspace_root / "runs" / captured["run_id"]))
            self.assertEqual(captured["budget"].max_rounds, 8)
            self.assertEqual(captured["budget"].max_tool_calls_per_round, 2)
            self.assertEqual(captured["budget"].default_nframes, 12)
            self.assertIsInstance(captured["scene_index"], SceneIndex)
            self.assertEqual(captured["scene_index"].segments[0].source, "dual_source_scene_index")

    def test_run_eval_cases_precomputes_two_fps_frame_cache_for_agent_loop(self):
        from runs import eval_runner

        captured = {}
        build_calls = []

        class FakeFrameCache:
            fps = 2.0
            frame_dir = Path("/tmp/frame-cache/video605_2fps")

            def sample_paths(self, video_path, start_sec, end_sec, max_frames):
                return [f"/frames/{Path(video_path).stem}_{start_sec:.0f}_{end_sec:.0f}_{max_frames}.jpg"]

        def fake_build_frame_cache_for_video(*, video_path, frame_dir, fps, duration_sec):
            build_calls.append((video_path, frame_dir, fps, duration_sec))
            return FakeFrameCache()

        def fake_run_loop(backend, **kwargs):
            captured["frame_sampler"] = kwargs["frame_sampler"]
            return {
                "answer": "B. The visual evidence supports option B.",
                "choice": "B",
                "status": "final",
                "confidence": 0.8,
                "citations": ["obs_0001"],
                "rounds": 1,
                "tools": ["caption_segment"],
                "segments": ["seg_0001"],
                "seconds": 1.0,
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
                strategies=("workspace_v2",),
                window_sec=300.0,
                budget=AgentBudget(max_rounds=8),
            )

            with patch.object(eval_runner, "build_frame_cache_for_video", side_effect=fake_build_frame_cache_for_video):
                with patch.object(eval_runner, "SceneIndexBuilder", FakeSceneIndexBuilder):
                    with patch.object(eval_runner, "run_loop", side_effect=fake_run_loop):
                        summary = eval_runner.run_eval_cases(
                            backend=object(),
                            rows_by_id=rows_by_id,
                            config=config,
                            duration_fn=lambda path: 1896.0,
                        )

        self.assertEqual(
            build_calls,
            [
                (
                    Path("/dataset/video/video605.mp4"),
                    run_root / "frame_cache" / "video605_2fps",
                    2.0,
                    1896.0,
                )
            ],
        )
        self.assertEqual(
            captured["frame_sampler"]("/dataset/video/video605.mp4", 10.0, 20.0, 4),
            ["/frames/video605_10_20_4.jpg"],
        )
        self.assertEqual(summary["cases"][0]["raw_artifacts"]["frame_cache"], "/tmp/frame-cache/video605_2fps")

    def test_prewarm_scene_indexes_deduplicates_videos_without_running_agent_loop(self):
        from runs import eval_runner

        build_calls = []
        frame_cache_calls = []

        class FakeBuilder:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def build(self, **kwargs):
                build_calls.append(kwargs)
                return SceneIndex(
                    video_path=kwargs["video_path"],
                    duration_sec=kwargs["duration_sec"],
                    segments=[
                        VideoSegment(
                            segment_id="seg_0001",
                            start_sec=0.0,
                            end_sec=300.0,
                            source="dual_source_scene_index",
                        )
                    ],
                )

        def fake_build_frame_cache_for_video(*, video_path, frame_dir, fps, duration_sec):
            frame_cache_calls.append((video_path, frame_dir, fps, duration_sec))
            return FakeFrameCache()

        rows_by_id = {
            "605-1": {"question_id": "605-1", "videoID": "video605", "video_id": "vid605"},
            "605-2": {"question_id": "605-2", "videoID": "video605", "video_id": "vid605"},
        }

        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "caption"
            config = eval_runner.EvalConfig(
                run_root=run_root,
                workspace_root=run_root / "workspaces",
                model_path="/model",
                data_root=Path("/dataset"),
                parquet_path=Path("/dataset/videomme/test.parquet"),
                video_dir=Path("/dataset/video"),
                subtitle_dir=Path("/dataset/subtitle"),
                cases=("605-1", "605-2"),
                strategies=("workspace_v2",),
                window_sec=300.0,
                scene_index_cache_dir=run_root / "scene_index_cache",
                budget=AgentBudget(),
            )

            with patch.object(eval_runner, "build_frame_cache_for_video", side_effect=fake_build_frame_cache_for_video):
                with patch.object(eval_runner, "SceneIndexBuilder", FakeBuilder):
                    with patch.object(eval_runner, "run_loop", side_effect=AssertionError("agent loop must not run")):
                        summary = eval_runner.prewarm_scene_indexes(
                            backend=object(),
                            rows_by_id=rows_by_id,
                            config=config,
                            duration_fn=lambda path: 1896.0,
                        )
            summary_exists = (run_root / "scene_index_prewarm_summary.json").exists()

        self.assertEqual([call["video_id"] for call in build_calls], ["video605"])
        self.assertEqual(len(frame_cache_calls), 1)
        self.assertEqual(summary["videos_total"], 1)
        self.assertEqual(summary["videos_done"], 1)
        self.assertEqual(summary["videos"][0]["videoID"], "video605")
        self.assertTrue(summary_exists)

    def test_prewarm_scene_indexes_can_run_videos_concurrently(self):
        from runs import eval_runner

        lock = threading.Lock()
        barrier = threading.Barrier(2, timeout=2)
        active = 0
        max_active = 0

        class FakeBuilder:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def build(self, **kwargs):
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                try:
                    barrier.wait()
                    return SceneIndex(
                        video_path=kwargs["video_path"],
                        duration_sec=kwargs["duration_sec"],
                        segments=[
                            VideoSegment(
                                segment_id="seg_0001",
                                start_sec=0.0,
                                end_sec=300.0,
                                source="dual_source_scene_index",
                            )
                        ],
                    )
                finally:
                    with lock:
                        active -= 1

        rows_by_id = {
            "605-1": {"question_id": "605-1", "videoID": "video605", "video_id": "vid605"},
            "611-1": {"question_id": "611-1", "videoID": "video611", "video_id": "vid611"},
        }

        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "caption"
            config = eval_runner.EvalConfig(
                run_root=run_root,
                workspace_root=run_root / "workspaces",
                model_path="/model",
                data_root=Path("/dataset"),
                parquet_path=Path("/dataset/videomme/test.parquet"),
                video_dir=Path("/dataset/video"),
                subtitle_dir=Path("/dataset/subtitle"),
                cases=("605-1", "611-1"),
                strategies=("workspace_v2",),
                window_sec=300.0,
                scene_index_cache_dir=run_root / "scene_index_cache",
                scene_index_video_concurrency=2,
                budget=AgentBudget(),
            )

            with patch.object(eval_runner, "build_frame_cache_for_video", return_value=FakeFrameCache()):
                with patch.object(eval_runner, "SceneIndexBuilder", FakeBuilder):
                    with patch.object(eval_runner, "run_loop", side_effect=AssertionError("agent loop must not run")):
                        summary = eval_runner.prewarm_scene_indexes(
                            backend=object(),
                            rows_by_id=rows_by_id,
                            config=config,
                            duration_fn=lambda path: 30.0,
                        )

        self.assertGreaterEqual(max_active, 2)
        self.assertEqual(summary["videos_done"], 2)
        self.assertEqual([item["videoID"] for item in summary["videos"]], ["video605", "video611"])

    def test_run_eval_cases_can_run_cases_concurrently(self):
        from runs import eval_runner

        lock = threading.Lock()
        barrier = threading.Barrier(2, timeout=2)
        active = 0
        max_active = 0

        def fake_run_loop(backend, **kwargs):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                barrier.wait()
                return {
                    "answer": "B. The visual evidence supports option B.",
                    "choice": "B",
                    "status": "final",
                    "confidence": 0.8,
                    "citations": ["obs_0001"],
                    "rounds": 1,
                    "tools": ["caption_segment"],
                    "segments": ["seg_0001"],
                    "seconds": 1.0,
                }
            finally:
                with lock:
                    active -= 1

        rows_by_id = {
            "605-1": {
                "question_id": "605-1",
                "video_id": "vid605",
                "videoID": "video605",
                "task_type": "Information Synopsis",
                "question": "What is shown?",
                "options": ["A. one", "B. two", "C. three", "D. four"],
                "answer": "B",
            },
            "611-1": {
                "question_id": "611-1",
                "video_id": "vid611",
                "videoID": "video611",
                "task_type": "Information Synopsis",
                "question": "What is shown next?",
                "options": ["A. one", "B. two", "C. three", "D. four"],
                "answer": "B",
            },
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
                cases=("605-1", "611-1"),
                strategies=("workspace_v2",),
                window_sec=300.0,
                scene_index_cache_dir=run_root / "scene_index_cache",
                eval_case_concurrency=2,
                budget=AgentBudget(),
            )

            with patch.object(eval_runner, "build_frame_cache_for_video", return_value=FakeFrameCache()):
                with patch.object(eval_runner, "SceneIndexBuilder", FakeSceneIndexBuilder):
                    with patch.object(eval_runner, "run_loop", side_effect=fake_run_loop):
                        summary = eval_runner.run_eval_cases(
                            backend=object(),
                            rows_by_id=rows_by_id,
                            config=config,
                            duration_fn=lambda path: 30.0,
                        )

        self.assertGreaterEqual(max_active, 2)
        self.assertEqual([case["question_id"] for case in summary["cases"]], ["605-1", "611-1"])
        self.assertEqual(summary["accuracy"], 1.0)

    def test_scene_index_only_cli_dispatches_without_eval_cases(self):
        from runs import eval_runner

        config = eval_runner.EvalConfig(
            run_root=Path("/tmp/caption-only"),
            workspace_root=Path("/tmp/caption-only/workspaces"),
            model_path="/model",
            data_root=Path("/dataset"),
            parquet_path=Path("/dataset/videomme/test.parquet"),
            video_dir=Path("/dataset/video"),
            subtitle_dir=Path("/dataset/subtitle"),
            cases=("605-1",),
            strategies=("workspace_v2",),
        )
        calls = []

        def fake_prewarm_scene_indexes(**kwargs):
            calls.append(kwargs)
            return {"videos_done": 1}

        with patch.object(eval_runner, "validate_python"):
            with patch.object(eval_runner, "config_from_args", return_value=config):
                with patch.object(eval_runner, "load_rows_by_id", return_value={"605-1": {"videoID": "video605"}}):
                    with patch.object(eval_runner, "build_backend", return_value=object()):
                        with patch.object(eval_runner, "prewarm_scene_indexes", side_effect=fake_prewarm_scene_indexes):
                            with patch.object(eval_runner, "run_eval_cases", side_effect=AssertionError("eval must not run")):
                                eval_runner.main(["--scene-index-only", "--allow-any-python"])

        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["config"], config)

    def test_config_file_values_are_loaded_cli_overrides_and_resolved_config_is_written(self):
        from runs import eval_runner

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

        def fake_run_loop(backend, **kwargs):
            return {
                "answer": "B. The visual evidence supports option B.",
                "choice": "B",
                "status": "final",
                "confidence": 0.8,
                "citations": ["obs_0001"],
                "rounds": 1,
                "tools": ["target_coverage"],
                "segments": ["seg_0001"],
                "seconds": 1.0,
            }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "experiment.yaml"
            run_root = tmp_path / "configured-run"
            config_path.write_text(
                "\n".join(
                    [
                        "cases: [605-1, 611-2]",
                        "strategies: [workspace_v2]",
                        f"run_root: {run_root}",
                        "model_path: /models/vl",
                        "planner_model_path: /models/planner",
                        "data_root: /dataset",
                        "parquet_path: /dataset/videomme/test.parquet",
                        "video_dir: /dataset/video",
                        "subtitle_dir: /dataset/subtitle",
                        "frame_cache_fps: 1.5",
                        "nframes: 6",
                        "max_rounds: 5",
                        "max_tool_calls_per_round: 3",
                        "hard_skill_runtime: true",
                        "export_training: true",
                        "planner_owned_grounding: true",
                    ]
                ),
                encoding="utf-8",
            )
            args = eval_runner.build_arg_parser().parse_args(
                [
                    "--config",
                    str(config_path),
                    "--cases",
                    "605-1",
                    "--max-rounds",
                    "7",
                    "--allow-any-python",
                ]
            )

            config = eval_runner.config_from_args(args)

            self.assertEqual(config.cases, ("605-1",))
            self.assertEqual(config.strategies, ("workspace_v2",))
            self.assertEqual(config.run_root, run_root)
            self.assertEqual(config.model_path, "/models/vl")
            self.assertEqual(config.planner_model_path, "/models/planner")
            self.assertEqual(config.frame_cache_fps, 1.5)
            self.assertEqual(config.scene_index_frame_fps, 0.5)
            self.assertEqual(config.budget.default_nframes, 6)
            self.assertEqual(config.budget.max_rounds, 7)
            self.assertEqual(config.budget.max_tool_calls_per_round, 3)
            self.assertTrue(config.budget.hard_skill_runtime)
            self.assertTrue(config.budget.planner_owned_grounding)
            self.assertTrue(config.export_training)

            with patch.object(eval_runner, "build_frame_cache_for_video", return_value=FakeFrameCache()):
                with patch.object(eval_runner, "SceneIndexBuilder", FakeSceneIndexBuilder):
                    with patch.object(eval_runner, "run_loop", side_effect=fake_run_loop):
                        eval_runner.run_eval_cases(
                            backend=object(),
                            rows_by_id=rows_by_id,
                            config=config,
                            duration_fn=lambda path: 60.0,
                        )

            resolved_path = run_root / "resolved_config.json"
            self.assertTrue(resolved_path.exists())
            resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
            self.assertEqual(resolved["cases"], ["605-1"])
            self.assertEqual(resolved["strategies"], ["workspace_v2"])
            self.assertEqual(resolved["budget"]["default_nframes"], 6)
            self.assertEqual(resolved["budget"]["max_rounds"], 7)
            self.assertEqual(resolved["frame_cache_fps"], 1.5)
            self.assertEqual(resolved["scene_index_frame_fps"], 0.5)
            self.assertEqual(resolved["source_config_path"], str(config_path))

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
                        "strategies": {"workspace_v2": {"status": "need_more_evidence", "correct": False}},
                        "raw_artifacts": {"workspaces": {"workspace_v2": str(workspace.root)}},
                    }
                ],
            )

            self.assertEqual(summary["route_violations"], 2)

    def test_summary_payload_aggregates_hard_skill_followup_metrics(self):
        from runs import eval_runner

        with tempfile.TemporaryDirectory() as tmp:
            success_workspace = EvidenceWorkspace.create(Path(tmp), run_id="followup_success")
            success_workspace.write_trace_event("hard_skill_runtime", {"skill": "timeline_ordering@v1"})
            success_workspace.write_trace_event("tool_use", {"tool": "caption_segment", "arguments": {"question": "first gap"}})
            success_workspace.write_trace_event("tool_use", {"tool": "vision_read", "arguments": {"ask_for": "second gap"}})
            success_workspace.write_trace_event("iterative_final", {"source": "hard_skill_runtime"})

            handoff_workspace = EvidenceWorkspace.create(Path(tmp), run_id="followup_handoff")
            handoff_workspace.write_trace_event("hard_skill_runtime", {"skill": "timeline_ordering@v1"})
            handoff_workspace.write_trace_event("tool_use", {"tool": "vision_read", "arguments": {"ask_for": "remaining gap"}})
            handoff_workspace.write_trace_event("hard_skill_followup_handoff", {"rounds": 1})

            summary = eval_runner._summary_payload(
                run_id="eval",
                case_ids=["case_001", "case_002"],
                config_payload={},
                results=[
                    {
                        "question_id": "case_001",
                        "strategies": {"workspace_v2": {"status": "final", "correct": True}},
                        "raw_artifacts": {"workspaces": {"workspace_v2": str(success_workspace.root)}},
                    },
                    {
                        "question_id": "case_002",
                        "strategies": {"workspace_v2": {"status": "max_rounds_reached", "correct": False}},
                        "raw_artifacts": {"workspaces": {"workspace_v2": str(handoff_workspace.root)}},
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
                        "strategies": {"workspace_v2": {"status": "low_confidence_final", "correct": False}},
                        "raw_artifacts": {"workspaces": {"workspace_v2": str(workspace.root)}},
                    }
                ],
            )

            self.assertEqual(summary["avg_followups_per_case"], 2.0)
            self.assertEqual(summary["followup_success_rate"], 1.0)
            self.assertEqual(summary["low_confidence_final_rate"], 1.0)
            self.assertEqual(summary["unvalidated_guess_rate"], 0.0)

    def test_summary_payload_reports_unvalidated_guess_rate(self):
        from runs import eval_runner

        summary = eval_runner._summary_payload(
            run_id="eval",
            case_ids=["case_001", "case_002"],
            config_payload={},
            results=[
                {
                    "question_id": "case_001",
                    "strategies": {"workspace_v2": {"status": "unvalidated_guess", "correct": True}},
                    "raw_artifacts": {},
                },
                {
                    "question_id": "case_002",
                    "strategies": {"workspace_v2": {"status": "final", "correct": True, "citation_count": 1}},
                    "raw_artifacts": {},
                },
            ],
        )

        self.assertEqual(summary["unvalidated_guess_rate"], 0.5)
        self.assertEqual(summary["low_confidence_final_rate"], 0.0)

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
                        "strategies": {"workspace_v2": {"status": "need_more_evidence", "correct": False}},
                        "raw_artifacts": {"workspaces": {"workspace_v2": str(workspace.root)}},
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
                        "strategies": {"workspace_v2": {"status": "final", "correct": True}},
                        "raw_artifacts": {"workspaces": {"workspace_v2": str(complete_workspace.root)}},
                    },
                    {
                        "question_id": "case_002",
                        "strategies": {"workspace_v2": {"status": "need_more_evidence", "correct": False}},
                        "raw_artifacts": {"workspaces": {"workspace_v2": str(empty_workspace.root)}},
                    },
                ],
            )

            self.assertEqual(summary["evidence_provenance_completeness"], 0.5)

    def test_workspace_v2_dual_source_scene_index_uses_builder_and_cache_root(self):
        from runs import eval_runner

        captured = {}
        builder_inits = []

        class FakeBuilder:
            def __init__(self, **kwargs):
                builder_inits.append(kwargs)

            def build(self, **kwargs):
                captured["build_kwargs"] = kwargs
                return SceneIndex(
                    video_path=kwargs["video_path"],
                    duration_sec=kwargs["duration_sec"],
                    segments=[
                        VideoSegment(
                            segment_id="seg_0001",
                            start_sec=0.0,
                            end_sec=300.0,
                            source="dual_source_scene_index",
                            asr_summary="opening clue",
                            visual_caption="wide shot",
                        )
                    ],
                )

        def fake_run_loop(backend, **kwargs):
            captured["scene_index"] = kwargs["scene_index"]
            return {
                "answer": "D. Based on dual source index.",
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
            cache_dir = Path(tmp) / "scene_cache"
            subtitle_dir.mkdir()
            (subtitle_dir / "video611.srt").write_text(
                "1\n00:00:02,000 --> 00:00:03,000\nopening clue\n",
                encoding="utf-8",
            )
            config = eval_runner.EvalConfig(
                run_root=run_root,
                workspace_root=run_root / "workspaces",
                model_path="/models/vl",
                planner_model_path="/models/text",
                data_root=Path("/dataset"),
                parquet_path=Path("/dataset/videomme/test.parquet"),
                video_dir=Path("/dataset/video"),
                subtitle_dir=subtitle_dir,
                cases=("611-2",),
                strategies=("workspace_v2",),
                window_sec=300.0,
                scene_index_cache_dir=cache_dir,
                scene_index_concurrency=8,
                scene_index_frame_fps=0.5,
                scene_index_max_new_tokens=7777,
                frame_cache_fps=2.0,
                budget=AgentBudget(),
            )

            with patch.object(eval_runner, "build_frame_cache_for_video", return_value=FakeFrameCache()):
                with patch.object(eval_runner, "SceneIndexBuilder", FakeBuilder):
                    with patch.object(eval_runner, "run_loop", side_effect=fake_run_loop):
                        eval_runner.run_eval_cases(
                            backend=object(),
                            rows_by_id=rows_by_id,
                            config=config,
                            duration_fn=lambda path: 1805.0,
                        )

            self.assertEqual(captured["scene_index"].segments[0].source, "dual_source_scene_index")
            self.assertEqual(captured["build_kwargs"]["video_id"], "video611")
            self.assertEqual(builder_inits[0]["text_model_id"], "/models/text")
            self.assertEqual(builder_inits[0]["vl_model_id"], "/models/vl")
            self.assertEqual(builder_inits[0]["cache"].cache_dir, cache_dir)
            self.assertEqual(builder_inits[0]["root_concurrency"], 8)
            self.assertFalse(hasattr(builder_inits[0]["root_policy"], "max_beats_per_root"))
            self.assertEqual(builder_inits[0]["root_policy"].max_new_tokens, 7777)
            self.assertEqual(builder_inits[0]["root_policy"].frame_cache_fps, 0.5)

    def test_run_eval_cases_labels_tool_api_scene_index_with_api_model(self):
        from runs import eval_runner

        builder_inits = []

        class FakeBuilder(FakeSceneIndexBuilder):
            def __init__(self, **kwargs):
                builder_inits.append(kwargs)
                super().__init__(**kwargs)

        def fake_run_loop(backend, **kwargs):
            return {
                "answer": "B.",
                "choice": "B",
                "status": "final",
                "confidence": 0.8,
                "citations": ["obs_0001"],
                "rounds": 1,
                "tools": ["caption_segment"],
                "segments": ["seg_0001"],
                "seconds": 1.0,
            }

        rows_by_id = {
            "611-2": {
                "question_id": "611-2",
                "video_id": "vid611",
                "videoID": "video611",
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
                model_path="/models/local-vl",
                planner_api_model="pa/gmn-2.5-pr",
                planner_api_use_for_tools=True,
                data_root=Path("/dataset"),
                parquet_path=Path("/dataset/videomme/test.parquet"),
                video_dir=Path("/dataset/video"),
                subtitle_dir=Path("/dataset/subtitle"),
                cases=("611-2",),
                strategies=("workspace_v2",),
                scene_index_cache_dir=run_root / "scene_index_cache",
                budget=AgentBudget(),
            )

            with patch.object(eval_runner, "build_frame_cache_for_video", return_value=FakeFrameCache()):
                with patch.object(eval_runner, "SceneIndexBuilder", FakeBuilder):
                    with patch.object(eval_runner, "run_loop", side_effect=fake_run_loop):
                        eval_runner.run_eval_cases(
                            backend=object(),
                            rows_by_id=rows_by_id,
                            config=config,
                            duration_fn=lambda path: 1805.0,
                        )

        self.assertEqual(builder_inits[0]["vl_model_id"], "pa/gmn-2.5-pr")

    def test_free_explore_cli_builds_budgetless_agent_config(self):
        from runs import eval_runner

        parser = eval_runner.build_arg_parser()
        args = parser.parse_args(
            [
                "--strategy",
                "workspace_v2",
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

        self.assertEqual(config.strategies, ("workspace_v2",))
        self.assertEqual(config.cases, ("611-2",))
        self.assertEqual(config.budget.max_rounds, 24)
        self.assertEqual(config.budget.max_tool_calls_per_round, 4)
        self.assertFalse(config.budget.reserve_final_round)
        self.assertTrue(config.budget.hard_skill_runtime)

    def test_workspace_v2_does_not_enable_legacy_skill_runtime_by_default(self):
        from runs import eval_runner

        parser = eval_runner.build_arg_parser()
        args = parser.parse_args(["--strategy", "workspace_v2", "--cases", "611-2", "--run-root", "/tmp/vcah-default"])

        config = eval_runner.config_from_args(args)

        self.assertFalse(config.budget.hard_skill_runtime)

    def test_videomme_runner_disables_global_mcq_rewrite_by_default(self):
        from runs import eval_runner

        parser = eval_runner.build_arg_parser()
        args = parser.parse_args([])

        config = eval_runner.config_from_args(args)

        self.assertFalse(config.budget.rewrite_mcq_for_exploration)
        self.assertFalse(config.ablation_flags["enable_mcq_rewrite"])

    def test_videomme_runner_can_enable_legacy_global_mcq_rewrite(self):
        from runs import eval_runner

        parser = eval_runner.build_arg_parser()
        args = parser.parse_args(["--use-global-question-rewrite"])

        config = eval_runner.config_from_args(args)

        self.assertTrue(config.budget.rewrite_mcq_for_exploration)
        self.assertTrue(config.ablation_flags["enable_mcq_rewrite"])

    def test_videomme_runner_can_disable_mcq_rewrite(self):
        from runs import eval_runner

        parser = eval_runner.build_arg_parser()
        args = parser.parse_args(["--disable-mcq-rewrite"])

        config = eval_runner.config_from_args(args)

        self.assertFalse(config.budget.rewrite_mcq_for_exploration)
        self.assertFalse(config.ablation_flags["enable_mcq_rewrite"])

    def test_max_rounds_raises_repeated_program_guard_for_segment_sweep(self):
        from runs import eval_runner

        parser = eval_runner.build_arg_parser()
        args = parser.parse_args(["--max-rounds", "20"])

        config = eval_runner.config_from_args(args)

        self.assertEqual(config.budget.max_rounds, 20)
        self.assertEqual(config.budget.max_repeated_programs, 20)

    def test_workspace_v2_does_not_enable_planner_owned_grounding_by_default(self):
        from runs import eval_runner

        parser = eval_runner.build_arg_parser()
        args = parser.parse_args(["--strategy", "workspace_v2"])

        config = eval_runner.config_from_args(args)

        self.assertFalse(config.budget.planner_owned_grounding)
        self.assertFalse(config.ablation_flags["planner_owned_grounding"])

    def test_planner_owned_grounding_can_be_disabled(self):
        from runs import eval_runner

        parser = eval_runner.build_arg_parser()
        args = parser.parse_args(["--strategy", "workspace_v2", "--disable-planner-owned-grounding"])

        config = eval_runner.config_from_args(args)

        self.assertFalse(config.budget.planner_owned_grounding)
        self.assertFalse(config.ablation_flags["planner_owned_grounding"])

    def test_legacy_expensive_tool_budget_flag_is_accepted_but_ignored(self):
        from runs import eval_runner

        parser = eval_runner.build_arg_parser()
        args = parser.parse_args(["--max-rounds", "20", "--expensive-tool-budget", "6"])

        config = eval_runner.config_from_args(args)

        self.assertEqual(config.budget.max_rounds, 20)
        self.assertFalse(hasattr(config.budget, "expensive_tool_budget"))

    def test_eval_case_concurrency_cli_is_capped_at_sixteen(self):
        from runs import eval_runner

        parser = eval_runner.build_arg_parser()

        default_config = eval_runner.config_from_args(parser.parse_args([]))
        capped_config = eval_runner.config_from_args(parser.parse_args(["--eval-case-concurrency", "99"]))

        self.assertEqual(default_config.eval_case_concurrency, 1)
        self.assertEqual(capped_config.eval_case_concurrency, 16)

    def test_scene_index_video_concurrency_cli_is_capped_at_sixteen(self):
        from runs import eval_runner

        parser = eval_runner.build_arg_parser()

        default_config = eval_runner.config_from_args(parser.parse_args([]))
        capped_config = eval_runner.config_from_args(parser.parse_args(["--scene-index-video-concurrency", "99"]))

        self.assertEqual(default_config.scene_index_video_concurrency, 1)
        self.assertEqual(capped_config.scene_index_video_concurrency, 16)

    def test_context_budget_cli_flags_build_agent_config(self):
        from runs import eval_runner

        parser = eval_runner.build_arg_parser()
        args = parser.parse_args(
            [
                "--strategy",
                "workspace_v2",
                "--cases",
                "611-2",
                "--run-root",
                "/tmp/vcah-context",
                "--context-budget-tokens",
                "9000",
                "--budget-ratios",
                "task:0.08,trajectory:0.07,hypothesis:0.12,evidence:0.28,scene_index:0.22,feedback:0.10,budget:0.05,tooling:0.08",
            ]
        )

        config = eval_runner.config_from_args(args)

        self.assertEqual(config.budget.context_budget_tokens, 9000)
        self.assertEqual(
            config.budget.context_budget_ratios,
            {
                "task": 0.08,
                "trajectory": 0.07,
                "hypothesis": 0.12,
                "evidence": 0.28,
                "scene_index": 0.22,
                "feedback": 0.10,
                "budget": 0.05,
                "tooling": 0.08,
            },
        )

    def test_context_budget_cli_rejects_bad_ratio_sum(self):
        from runs import eval_runner

        parser = eval_runner.build_arg_parser()
        args = parser.parse_args(
            [
                "--run-root",
                "/tmp/vcah-context",
                "--budget-ratios",
                "task:0.1,trajectory:0.1,hypothesis:0.1,evidence:0.1,scene_index:0.1,feedback:0.1,budget:0.1,tooling:0.1",
            ]
        )

        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            eval_runner.config_from_args(args)

    def test_dual_model_and_scene_cache_cli_flags_build_agent_config(self):
        from runs import eval_runner

        parser = eval_runner.build_arg_parser()
        args = parser.parse_args(
            [
                "--model-path",
                "/home/xuboshen/models/Qwen3-VL-4B-Instruct",
                "--planner-model-path",
                "/home/xuboshen/models/Qwen3-4B-Instruct-2507",
                "--scene-index-cache-dir",
                "/m2v_intern/xuboshen/zgw/visual-coding-agent-harness/scene_index_cache",
                "--scene-index-concurrency",
                "8",
            ]
        )

        config = eval_runner.config_from_args(args)

        self.assertEqual(config.model_path, "/home/xuboshen/models/Qwen3-VL-4B-Instruct")
        self.assertEqual(config.planner_model_path, "/home/xuboshen/models/Qwen3-4B-Instruct-2507")
        self.assertEqual(
            config.scene_index_cache_dir,
            Path("/m2v_intern/xuboshen/zgw/visual-coding-agent-harness/scene_index_cache"),
        )
        self.assertEqual(config.scene_index_concurrency, 8)

    def test_scene_index_yaml_ignores_retired_caption_frame_and_beat_limit_keys(self):
        from runs import eval_runner

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "gemini.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scene_index_concurrency: 8",
                        "scene_caption_nframes: 6",
                        "scene_index_max_beats_per_root: 10",
                        "dense_video_caption:",
                        "  fps: 0.5",
                        "  max_new_tokens: 8192",
                    ]
                ),
                encoding="utf-8",
            )

            parser = eval_runner.build_arg_parser()
            args = parser.parse_args(["--config", str(config_path)])
            config = eval_runner.config_from_args(args)

        self.assertEqual(config.scene_index_concurrency, 8)
        self.assertFalse(hasattr(config, "scene_caption_nframes"))
        self.assertEqual(config.scene_index_frame_fps, 0.5)
        self.assertFalse(hasattr(config, "scene_index_max_beats_per_root"))
        self.assertEqual(config.scene_index_max_new_tokens, 8192)

    def test_retired_scene_caption_and_beat_limit_cli_flags_are_rejected(self):
        from runs import eval_runner

        parser = eval_runner.build_arg_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(["--scene-caption-nframes", "6"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--scene-index-max-beats-per-root", "10"])

    def test_frame_cache_fps_caps_at_two_and_root_dvc_fps_cannot_exceed_cache(self):
        from runs import eval_runner

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "fps.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "frame_cache_fps: 3.0",
                        "dense_video_caption:",
                        "  fps: 2.5",
                    ]
                ),
                encoding="utf-8",
            )

            parser = eval_runner.build_arg_parser()
            args = parser.parse_args(["--config", str(config_path)])
            config = eval_runner.config_from_args(args)

        self.assertEqual(config.frame_cache_fps, 2.0)
        self.assertEqual(config.scene_index_frame_fps, 2.0)

    def test_default_run_and_scene_cache_roots_are_under_m2v_management_root(self):
        from runs import eval_runner

        parser = eval_runner.build_arg_parser()
        args = parser.parse_args([])

        config = eval_runner.config_from_args(args)

        self.assertEqual(
            config.run_root,
            Path("/m2v_intern/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_eval"),
        )
        self.assertEqual(
            config.scene_index_cache_dir,
            Path("/m2v_intern/xuboshen/zgw/visual-coding-agent-harness/scene_index_cache"),
        )

    def test_build_backend_uses_routed_backend_for_dual_model_config(self):
        from runs import eval_runner

        config = eval_runner.EvalConfig(
            run_root=Path("/tmp/run"),
            workspace_root=Path("/tmp/run/workspaces"),
            model_path="/models/vl",
            planner_model_path="/models/text",
            data_root=Path("/dataset"),
            parquet_path=Path("/dataset/test.parquet"),
            video_dir=Path("/dataset/video"),
            subtitle_dir=Path("/dataset/subtitle"),
            cases=("605-1",),
            strategies=("workspace_v2",),
        )

        with patch("visual_coding_agent_harness.backends.qwen_vl.QwenVLBackend.from_pretrained", return_value="vl") as vl_load:
            with patch(
                "visual_coding_agent_harness.backends.qwen_text.QwenTextBackend.from_pretrained",
                return_value="text",
            ) as text_load:
                backend = eval_runner.build_backend(config)

        self.assertEqual(backend.vl_backend, "vl")
        self.assertEqual(backend.text_backend, "text")
        vl_load.assert_called_once_with("/models/vl")
        text_load.assert_called_once_with("/models/text")

    def test_planner_api_cli_flags_build_agent_config(self):
        from runs import eval_runner

        parser = eval_runner.build_arg_parser()
        args = parser.parse_args(
            [
                "--model-path",
                "/models/vl",
                "--planner-model-path",
                "/models/local-fallback",
                "--planner-api-base",
                "http://planner-host:8000/v1",
                "--planner-api-model",
                "Qwen3.5-9B",
                "--planner-api-key",
                "EMPTY",
                "--planner-api-timeout",
                "75",
                "--planner-thinking-token-budget",
                "512",
                "--planner-enable-thinking",
            ]
        )

        config = eval_runner.config_from_args(args)

        self.assertEqual(config.planner_api_base, "http://planner-host:8000/v1")
        self.assertEqual(config.planner_api_model, "Qwen3.5-9B")
        self.assertEqual(config.planner_api_key, "EMPTY")
        self.assertEqual(config.planner_api_timeout, 75.0)
        self.assertEqual(config.planner_thinking_token_budget, 512)
        self.assertTrue(config.planner_enable_thinking)

    def test_planner_api_azure_config_uses_environment_names_without_secret_values(self):
        from runs import eval_runner

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "azure.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "planner_api:",
                        "  type: azure",
                        "  endpoint_env: GPT_ENDPOINT",
                        "  deployment_env: GPT_DEPLOYMENT",
                        "  api_key_env: GPT_API_KEY",
                        "  api_version_env: GPT_API_VERSION",
                        "  use_for_tools: true",
                        "  timeout: 45",
                    ]
                ),
                encoding="utf-8",
            )

            parser = eval_runner.build_arg_parser()
            args = parser.parse_args(["--config", str(config_path), "--model-path", "/models/vl"])
            config = eval_runner.config_from_args(args)

        self.assertEqual(config.planner_api_type, "azure")
        self.assertEqual(config.planner_api_base, "")
        self.assertEqual(config.planner_api_model, "")
        self.assertEqual(config.planner_api_key, "EMPTY")
        self.assertEqual(config.planner_api_base_env, "GPT_ENDPOINT")
        self.assertEqual(config.planner_api_model_env, "GPT_DEPLOYMENT")
        self.assertEqual(config.planner_api_key_env, "GPT_API_KEY")
        self.assertEqual(config.planner_api_version_env, "GPT_API_VERSION")
        self.assertTrue(config.planner_api_use_for_tools)
        self.assertEqual(config.planner_api_timeout, 45.0)

    def test_planner_api_gemini_gateway_config_uses_environment_names(self):
        from runs import eval_runner

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "gemini.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "planner_api:",
                        "  type: gemini_gateway",
                        "  base_env: GEMINI_API_BASE",
                        "  model_env: GEMINI_MODEL",
                        "  api_key_env: GEMINI_API_KEY",
                        "  user_key_env: GEMINI_USER_KEY",
                        "  biz_scene_env: GEMINI_BIZ_SCENE",
                        "  use_for_tools: true",
                    ]
                ),
                encoding="utf-8",
            )

            parser = eval_runner.build_arg_parser()
            args = parser.parse_args(["--config", str(config_path), "--model-path", "/models/vl"])
            config = eval_runner.config_from_args(args)

        self.assertEqual(config.planner_api_type, "gemini_gateway")
        self.assertEqual(config.planner_api_base_env, "GEMINI_API_BASE")
        self.assertEqual(config.planner_api_model_env, "GEMINI_MODEL")
        self.assertEqual(config.planner_api_key_env, "GEMINI_API_KEY")
        self.assertEqual(config.planner_api_user_key_env, "GEMINI_USER_KEY")
        self.assertEqual(config.planner_api_biz_scene_env, "GEMINI_BIZ_SCENE")
        self.assertTrue(config.planner_api_use_for_tools)

    def test_local_gemini25_pro_profile_uses_ppio_openai_compatible_api(self):
        from runs import eval_runner

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "local_gemini25_pro_ppio.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "planner_api:",
                        "  type: openai_compatible",
                        "  base: https://api.novita.ai/openai",
                        "  model: pa/gmn-2.5-pr",
                        "  api_key_env: PPIO_API_KEY",
                        "  use_for_tools: true",
                        "  timeout: 300",
                        "scene_index_concurrency: 8",
                        "dense_video_caption:",
                        "  fps: 0.5",
                        "  max_new_tokens: 6144",
                        "frame_cache_fps: 2.0",
                        "scene_index_frame_fps: 0.5",
                    ]
                ),
                encoding="utf-8",
            )

            parser = eval_runner.build_arg_parser()
            args = parser.parse_args(["--config", str(config_path)])
            config = eval_runner.config_from_args(args)

        self.assertEqual(config.planner_api_type, "openai_compatible")
        self.assertEqual(config.planner_api_base, "https://api.novita.ai/openai")
        self.assertEqual(config.planner_api_model, "pa/gmn-2.5-pr")
        self.assertEqual(config.planner_api_key, "EMPTY")
        self.assertEqual(config.planner_api_key_env, "PPIO_API_KEY")
        self.assertEqual(config.planner_api_user_key, "")
        self.assertEqual(config.planner_api_biz_scene, "")
        self.assertTrue(config.planner_api_use_for_tools)
        self.assertEqual(config.planner_api_timeout, 300.0)
        self.assertEqual(config.scene_index_concurrency, 8)
        self.assertEqual(config.scene_index_frame_fps, 0.5)
        self.assertEqual(config.frame_cache_fps, 2.0)

    def test_planner_api_gemini_gateway_config_accepts_private_yaml_values_without_serializing_secrets(self):
        from runs import eval_runner

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "local.gemini.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "planner_api:",
                        "  type: gemini_gateway",
                        "  base: http://gateway.internal/openai/team/v1/chat/completions",
                        "  model: pa/gemini-2.5-flash",
                        "  api_key: private-api-key",
                        "  user_key: team-user-key",
                        "  biz_scene: offline",
                        "  use_for_tools: true",
                        "  proxy_env:",
                        "    http_proxy: http://oversea-squid1.jp.txyun:11080",
                        "    https_proxy: http://oversea-squid1.jp.txyun:11080",
                        "    no_proxy: localhost,127.0.0.1",
                    ]
                ),
                encoding="utf-8",
            )

            parser = eval_runner.build_arg_parser()
            args = parser.parse_args(
                [
                    "--config",
                    str(config_path),
                    "--model-path",
                    "/models/vl",
                    "--run-root",
                    str(Path(tmp) / "run"),
                    "--workspace-root",
                    str(Path(tmp) / "run/workspaces"),
                    "--cases",
                    "605-1",
                ]
            )
            config = eval_runner.config_from_args(args)
            config.run_root.mkdir(parents=True, exist_ok=True)
            config.workspace_root.mkdir(parents=True, exist_ok=True)
            rows = {
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

            with patch.object(eval_runner, "build_frame_cache_for_video", return_value=FakeFrameCache()):
                with patch.object(eval_runner, "SceneIndexBuilder", FakeSceneIndexBuilder):
                    with patch.object(
                        eval_runner,
                        "run_loop",
                        return_value={
                            "answer": "B. The visual evidence supports option B.",
                            "choice": "B",
                            "status": "final",
                            "confidence": 0.8,
                            "citations": ["obs_0001"],
                            "rounds": 1,
                            "tools": ["read_clip"],
                            "segments": ["seg_0001"],
                            "seconds": 1.0,
                        },
                    ):
                        eval_runner.run_eval_cases(
                            backend=object(),
                            rows_by_id=rows,
                            config=config,
                            duration_fn=lambda path: 120.0,
                        )

            run_config = json.loads((config.run_root / "run_config.json").read_text(encoding="utf-8"))

        self.assertEqual(config.planner_api_key, "private-api-key")
        self.assertEqual(config.planner_api_user_key, "team-user-key")
        self.assertEqual(config.planner_api_biz_scene, "offline")
        self.assertTrue(config.planner_api_use_for_tools)
        self.assertEqual(config.planner_api_proxy_env["http_proxy"], "http://oversea-squid1.jp.txyun:11080")
        serialized = json.dumps(run_config, ensure_ascii=False)
        self.assertNotIn("private-api-key", serialized)
        self.assertNotIn("team-user-key", serialized)
        self.assertNotIn("oversea-squid1", serialized)
        self.assertEqual(run_config["planner_api_key_set"], True)
        self.assertEqual(run_config["planner_api_user_key_set"], True)
        self.assertEqual(run_config["planner_api_biz_scene_set"], True)
        self.assertEqual(
            run_config["planner_api_proxy_env_keys"],
            ["http_proxy", "https_proxy", "no_proxy"],
        )

    def test_build_backend_uses_openai_chat_text_backend_for_planner_api(self):
        from runs import eval_runner

        config = eval_runner.EvalConfig(
            run_root=Path("/tmp/run"),
            workspace_root=Path("/tmp/run/workspaces"),
            model_path="/models/vl",
            planner_model_path="/models/local-should-not-load",
            planner_api_base="http://planner-host:8000/v1",
            planner_api_model="Qwen3.5-9B",
            planner_api_key="EMPTY",
            planner_api_timeout=90.0,
            planner_thinking_token_budget=512,
            planner_enable_thinking=True,
            data_root=Path("/dataset"),
            parquet_path=Path("/dataset/test.parquet"),
            video_dir=Path("/dataset/video"),
            subtitle_dir=Path("/dataset/subtitle"),
            cases=("605-1",),
            strategies=("workspace_v2",),
        )

        with patch("visual_coding_agent_harness.backends.qwen_vl.QwenVLBackend.from_pretrained", return_value="vl") as vl_load:
            with patch(
                "visual_coding_agent_harness.backends.qwen_text.QwenTextBackend.from_pretrained",
                side_effect=AssertionError("local planner should not load"),
            ):
                backend = eval_runner.build_backend(config)

        self.assertEqual(backend.vl_backend, "vl")
        self.assertEqual(backend.text_backend.api_base, "http://planner-host:8000/v1")
        self.assertEqual(backend.text_backend.model, "Qwen3.5-9B")
        self.assertEqual(backend.text_backend.api_key, "EMPTY")
        self.assertEqual(backend.text_backend.timeout, 90.0)
        self.assertEqual(backend.text_backend.thinking_token_budget, 512)
        self.assertTrue(backend.text_backend.enable_thinking)
        vl_load.assert_called_once_with("/models/vl")

    def test_build_backend_uses_azure_openai_text_backend_from_env(self):
        from runs import eval_runner

        config = eval_runner.EvalConfig(
            run_root=Path("/tmp/run"),
            workspace_root=Path("/tmp/run/workspaces"),
            model_path="/models/vl",
            planner_api_type="azure",
            planner_api_base_env="GPT_ENDPOINT",
            planner_api_model_env="GPT_DEPLOYMENT",
            planner_api_key_env="GPT_API_KEY",
            planner_api_version_env="GPT_API_VERSION",
            planner_api_timeout=90.0,
            data_root=Path("/dataset"),
            parquet_path=Path("/dataset/test.parquet"),
            video_dir=Path("/dataset/video"),
            subtitle_dir=Path("/dataset/subtitle"),
            cases=("605-1",),
            strategies=("workspace_v2",),
        )

        with patch.dict(
            os.environ,
            {
                "GPT_ENDPOINT": "https://example-resource.openai.azure.com",
                "GPT_DEPLOYMENT": "gpt-prod-deployment",
                "GPT_API_KEY": "secret-from-env",
                "GPT_API_VERSION": "2025-01-01-preview",
            },
            clear=False,
        ):
            with patch("visual_coding_agent_harness.backends.qwen_vl.QwenVLBackend.from_pretrained", return_value="vl"):
                backend = eval_runner.build_backend(config)

        self.assertEqual(backend.vl_backend, "vl")
        self.assertEqual(backend.text_backend.api_type, "azure_openai")
        self.assertEqual(backend.text_backend.api_base, "https://example-resource.openai.azure.com")
        self.assertEqual(backend.text_backend.model, "gpt-prod-deployment")
        self.assertEqual(backend.text_backend.api_key, "secret-from-env")
        self.assertEqual(backend.text_backend.api_version, "2025-01-01-preview")
        self.assertEqual(backend.text_backend.timeout, 90.0)

    def test_build_backend_can_route_planner_and_tools_to_azure_openai(self):
        from runs import eval_runner

        config = eval_runner.EvalConfig(
            run_root=Path("/tmp/run"),
            workspace_root=Path("/tmp/run/workspaces"),
            model_path="/models/vl-should-not-load",
            planner_api_type="azure",
            planner_api_use_for_tools=True,
            planner_api_base_env="GPT_ENDPOINT",
            planner_api_model_env="GPT_DEPLOYMENT",
            planner_api_key_env="GPT_API_KEY",
            data_root=Path("/dataset"),
            parquet_path=Path("/dataset/test.parquet"),
            video_dir=Path("/dataset/video"),
            subtitle_dir=Path("/dataset/subtitle"),
            cases=("605-1",),
            strategies=("workspace_v2",),
        )

        with patch.dict(
            os.environ,
            {
                "GPT_ENDPOINT": "https://example-resource.openai.azure.com",
                "GPT_DEPLOYMENT": "gpt-prod-deployment",
                "GPT_API_KEY": "secret-from-env",
            },
            clear=False,
        ):
            with patch(
                "visual_coding_agent_harness.backends.qwen_vl.QwenVLBackend.from_pretrained",
                side_effect=AssertionError("local vl should not load"),
            ):
                backend = eval_runner.build_backend(config)

        self.assertIs(backend.text_backend, backend.vl_backend)
        self.assertTrue(backend.text_backend.allow_media)
        self.assertEqual(backend.text_backend.api_type, "azure_openai")

    def test_build_backend_can_route_planner_and_tools_to_gemini_gateway(self):
        from runs import eval_runner

        config = eval_runner.EvalConfig(
            run_root=Path("/tmp/run"),
            workspace_root=Path("/tmp/run/workspaces"),
            model_path="/models/vl-should-not-load",
            planner_api_type="gemini_gateway",
            planner_api_use_for_tools=True,
            planner_api_base_env="GEMINI_API_BASE",
            planner_api_model_env="GEMINI_MODEL",
            planner_api_key_env="GEMINI_API_KEY",
            planner_api_user_key_env="GEMINI_USER_KEY",
            planner_api_biz_scene_env="GEMINI_BIZ_SCENE",
            data_root=Path("/dataset"),
            parquet_path=Path("/dataset/test.parquet"),
            video_dir=Path("/dataset/video"),
            subtitle_dir=Path("/dataset/subtitle"),
            cases=("605-1",),
            strategies=("workspace_v2",),
        )

        with patch.dict(
            os.environ,
            {
                "GEMINI_API_BASE": "http://gateway.internal/openai/team/v1/chat/completions",
                "GEMINI_MODEL": "pa/gemini-2.5-flash",
                "GEMINI_API_KEY": "secret-from-env",
                "GEMINI_USER_KEY": "team-user-key",
                "GEMINI_BIZ_SCENE": "offline",
            },
            clear=False,
        ):
            with patch(
                "visual_coding_agent_harness.backends.qwen_vl.QwenVLBackend.from_pretrained",
                side_effect=AssertionError("local vl should not load"),
            ):
                backend = eval_runner.build_backend(config)

        self.assertIs(backend.text_backend, backend.vl_backend)
        self.assertTrue(backend.text_backend.allow_media)
        self.assertEqual(backend.text_backend.api_type, "gemini_gateway")
        self.assertEqual(backend.text_backend.api_base, "http://gateway.internal/openai/team/v1/chat/completions")
        self.assertEqual(backend.text_backend.model, "pa/gemini-2.5-flash")
        self.assertEqual(backend.text_backend.user_key, "team-user-key")
        self.assertEqual(backend.text_backend.biz_scene, "offline")

    def test_build_backend_passes_private_gemini_gateway_yaml_values(self):
        from runs import eval_runner

        config = eval_runner.EvalConfig(
            run_root=Path("/tmp/run"),
            workspace_root=Path("/tmp/run/workspaces"),
            model_path="/models/vl-should-not-load",
            planner_api_type="gemini_gateway",
            planner_api_base="http://gateway.internal/openai/team/v1/chat/completions",
            planner_api_model="pa/gemini-2.5-flash",
            planner_api_key="private-api-key",
            planner_api_user_key="team-user-key",
            planner_api_biz_scene="offline",
            planner_api_use_for_tools=True,
            planner_api_proxy_env={"http_proxy": "http://oversea-squid1.jp.txyun:11080"},
            data_root=Path("/dataset"),
            parquet_path=Path("/dataset/test.parquet"),
            video_dir=Path("/dataset/video"),
            subtitle_dir=Path("/dataset/subtitle"),
            cases=("605-1",),
            strategies=("workspace_v2",),
        )

        with patch(
            "visual_coding_agent_harness.backends.qwen_vl.QwenVLBackend.from_pretrained",
            side_effect=AssertionError("local vl should not load"),
        ):
            backend = eval_runner.build_backend(config)

        self.assertIs(backend.text_backend, backend.vl_backend)
        self.assertEqual(backend.text_backend.api_base, "http://gateway.internal/openai/team/v1/chat/completions")
        self.assertEqual(backend.text_backend.model, "pa/gemini-2.5-flash")
        self.assertEqual(backend.text_backend.api_key, "private-api-key")
        self.assertEqual(backend.text_backend.user_key, "team-user-key")
        self.assertEqual(backend.text_backend.biz_scene, "offline")
        self.assertEqual(backend.text_backend.proxy_env["http_proxy"], "http://oversea-squid1.jp.txyun:11080")

    def test_ablation_cli_flags_serialized_to_config(self):
        from runs import eval_runner

        parser = eval_runner.build_arg_parser()
        args = parser.parse_args(
            [
                "--strategy",
                "workspace_v2",
                "--cases",
                "611-2",
                "--run-root",
                "/tmp/vcah-ablation",
                "--enable-query-context",
                "--disable-followup",
                "--disable-context-budget",
                "--enable-map-reflux",
                "--disable-evidence-staging",
                "--contract-nframes",
                "128",
                "--followup-budget",
                "3",
            ]
        )

        config = eval_runner.config_from_args(args)

        self.assertTrue(config.ablation_flags["enable_query_context"])
        self.assertFalse(config.ablation_flags["enable_followup"])
        self.assertFalse(config.ablation_flags["enable_context_budget"])
        self.assertTrue(config.ablation_flags["enable_map_reflux"])
        self.assertFalse(config.ablation_flags["enable_evidence_staging"])
        self.assertEqual(config.ablation_flags["contract_nframes"], 128)
        self.assertEqual(config.ablation_flags["followup_budget"], 3)
        self.assertEqual(config.budget.default_nframes, 128)
        self.assertFalse(hasattr(config.budget, "cheap_tool_budget"))
        self.assertFalse(config.budget.hard_skill_runtime)

    def test_run_loop_workspace_v2_uses_workspace_agent_and_registry(self):
        from runs import eval_runner
        from visual_coding_agent_harness.agents.workspace_agent import WorkspaceRunResult
        from visual_coding_agent_harness.video.map import VideoMapStore

        class FakeWorkspaceVisualAgent:
            def __init__(self, *, backend, registry, workspace, max_rounds, video_path, video_map=None, log_root):
                self.backend = backend
                self.registry = registry
                self.workspace = workspace
                self.max_rounds = max_rounds
                self.video_path = video_path
                self.video_map = video_map
                self.log_root = log_root
                agents.append(self)

            def run(self, question):
                self.workspace.write_trace_event("workspace_plan_model_io", {"round": 1, "question": question})
                self.workspace.write_trace_event(
                    "tool_use",
                    {"step": 1, "tool": "read_segment", "arguments": {"segment_id": "seg_0001", "mode": "index"}},
                )
                return WorkspaceRunResult(
                    answer="B. workspace answer",
                    citations=(),
                    confidence="0.8",
                    rounds=2,
                    metadata={"status": "final"},
                )

        registry_calls = []
        agents = []

        def fake_build_registry(**kwargs):
            registry_calls.append(kwargs)
            return object()

        with tempfile.TemporaryDirectory() as tmp:
            workspace_root = Path(tmp) / "workspaces"
            scene_index = SceneIndex(
                video_path="/videos/demo.mp4",
                duration_sec=12.0,
                segments=[VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=12.0)],
            )
            with patch.object(eval_runner, "build_workspace_v2_registry", side_effect=fake_build_registry, create=True):
                with patch.object(eval_runner, "WorkspaceVisualAgent", FakeWorkspaceVisualAgent, create=True):
                    raw = eval_runner.run_loop(
                        backend=object(),
                        video_path="/videos/demo.mp4",
                        question="Which object is visible?\nA. blue car\nB. red car",
                        duration_sec=12.0,
                        run_id="case_workspace_v2",
                        scene_index=scene_index,
                        workspace_root=workspace_root,
                        budget=AgentBudget(max_rounds=5),
                        extract_clips=False,
                        strategy="workspace_v2",
                    )

            self.assertEqual(len(registry_calls), 1)
            self.assertIn("workspace", registry_calls[0])
            self.assertIsInstance(registry_calls[0]["video_map"], VideoMapStore)
            self.assertEqual(len(agents), 1)
            self.assertIs(agents[0].video_map, registry_calls[0]["video_map"])
            self.assertEqual(agents[0].max_rounds, 5)
            self.assertEqual(agents[0].video_path, "/videos/demo.mp4")
            self.assertEqual(agents[0].log_root, workspace_root.parent / "workspace_logs" / "case_workspace_v2")
            self.assertEqual(raw["answer"], "B. workspace answer")
            self.assertEqual(raw["choice"], "B")
            self.assertEqual(raw["status"], "final")
            self.assertEqual(raw["rounds"], 2)
            self.assertEqual(raw["tools"], ["read_segment"])
            self.assertEqual(raw["segments"], ["seg_0001"])
            self.assertEqual(raw["workspace_log_dir"], str(workspace_root.parent / "workspace_logs" / "case_workspace_v2"))
            self.assertEqual(raw["planner_io_dir"], str(workspace_root.parent / "workspace_logs" / "case_workspace_v2"))

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
                strategies=("workspace_v2",),
                window_sec=300.0,
                budget=AgentBudget(),
                export_training=True,
            )

            with patch.object(eval_runner, "build_frame_cache_for_video", return_value=FakeFrameCache()):
                with patch.object(eval_runner, "SceneIndexBuilder", FakeSceneIndexBuilder):
                    with patch.object(eval_runner, "run_loop", side_effect=fake_run_loop):
                        summary = eval_runner.run_eval_cases(
                            backend=object(),
                            rows_by_id=rows_by_id,
                            config=config,
                            duration_fn=lambda path: 30.0,
                        )

            case = summary["cases"][0]
            trajectory_path = Path(case["raw_artifacts"]["training_trajectories"]["workspace_v2"])
            markdown_path = Path(case["raw_artifacts"]["training_trajectory_markdown"]["workspace_v2"])
            self.assertTrue(trajectory_path.exists())
            self.assertTrue(markdown_path.exists())
            self.assertEqual(markdown_path, trajectory_path.with_suffix(".md"))
            self.assertEqual(case["strategies"]["workspace_v2"]["training_trajectory_markdown_path"], str(markdown_path))
            self.assertTrue(summary["training_trajectory_exported"])
            payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "TrainingTrajectoryV1")
            self.assertEqual(payload["ground_truth"], "B")
            self.assertIn("# Trajectory 605-1", markdown_path.read_text(encoding="utf-8"))

    def test_training_trajectory_export_path_exists_for_relative_run_root(self):
        from runs import eval_runner

        with tempfile.TemporaryDirectory() as tmp:
            workspace_root = Path(tmp) / "workspaces"
            workspace = _make_training_workspace(workspace_root, "case_workspace_v2")
            old_cwd = Path.cwd()
            try:
                os.chdir(tmp)
                trajectory_path = eval_runner._export_training_trajectory(
                    workspace_path=workspace.root,
                    run_root=Path("eval"),
                    case_id="605-1",
                    strategy="workspace_v2",
                    question="What is shown?",
                    options=["A. one", "B. two"],
                    gt="B",
                    strategy_summary={"choice": "B", "status": "final", "correct": True},
                )
            finally:
                os.chdir(old_cwd)

            self.assertIsNotNone(trajectory_path)
            assert trajectory_path is not None
            self.assertTrue(trajectory_path.is_absolute())
            self.assertTrue(trajectory_path.exists())
            self.assertTrue(trajectory_path.with_suffix(".md").exists())


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
