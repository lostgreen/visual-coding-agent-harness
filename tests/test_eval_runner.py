import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from visual_coding_agent_harness.agents.result import WorkspaceRunResult
from visual_coding_agent_harness.core.budget import AgentBudget
from visual_coding_agent_harness.video.index import SceneIndex, VideoSegment


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

    def test_videomme_runner_accepts_only_multi_v3_strategy(self):
        from runs import eval_runner

        self.assertEqual(eval_runner.parse_strategies(None), ("multi_v3",))
        self.assertEqual(eval_runner.parse_strategies(("multi_v3",)), ("multi_v3",))
        with self.assertRaisesRegex(ValueError, "Unknown strategy: workspace_v2"):
            eval_runner.parse_strategies(("workspace_v2",))
        with self.assertRaisesRegex(ValueError, "Unknown strategy: multi_agent_v0"):
            eval_runner.parse_strategies(("multi_agent_v0",))

    def test_run_loop_multi_v3_uses_reasoner_investigator_driver_and_sidecar_workspace(self):
        from runs import eval_runner

        created = {"indexes": [], "overviews": [], "workspaces": [], "reasoners": [], "investigators": [], "drivers": []}

        class FakeOverview:
            grid_image_path = "/tmp/overview/scene_timeline_grid.jpg"
            grid_path = grid_image_path
            manifest_path = "/tmp/overview/scene_timeline_grid.json"

        class FakeReasoner:
            def __init__(self, *, backend):
                self.backend = backend
                created["reasoners"].append(self)

        class FakeInvestigator:
            def __init__(self, *, index, workspace, backend):
                self.index = index
                self.workspace = workspace
                self.backend = backend
                created["investigators"].append(self)

        class FakeDriver:
            def __init__(self, *, reasoner, investigator, workspace, max_rounds, max_concurrency=4, valid_scene_ids=()):
                self.reasoner = reasoner
                self.investigator = investigator
                self.workspace = workspace
                self.max_rounds = max_rounds
                self.max_concurrency = max_concurrency
                self.valid_scene_ids = tuple(valid_scene_ids)
                created["drivers"].append(self)

            def run(self, *, question, options, index_context, overview_path, overview_image_path=""):
                self.workspace.root.mkdir(parents=True, exist_ok=True)
                query_dir = self.workspace.root / "queries" / "q1"
                query_dir.mkdir(parents=True, exist_ok=True)
                (query_dir / "request.json").write_text(
                    '{"query_id":"q1","scope":{"scene_ids":["sc01"]}}\n',
                    encoding="utf-8",
                )
                (query_dir / "explore.json").write_text(
                    '{"query_id":"q1","candidates":[{"shot_id":"sc01_sh001","score":0.9,"reason":"visible"}]}\n',
                    encoding="utf-8",
                )
                (query_dir / "verify_sc01_sh001.json").write_text(
                    '{"query_id":"q1","shot_id":"sc01_sh001","findings":[{"finding_id":"ev_0001","query_id":"q1","shot_id":"sc01_sh001","summary":"A red car is visible.","supports_options":["A"],"citation_ids":["ev_0001"],"confidence":0.9}]}\n',
                    encoding="utf-8",
                )
                (query_dir / "report.json").write_text(
                    '{"query_id":"q1","status":"satisfied","findings":[{"finding_id":"ev_0001","query_id":"q1","shot_id":"sc01_sh001","summary":"A red car is visible.","supports_options":["A"],"citation_ids":["ev_0001"],"confidence":0.9}],"explored_shots":["sc01_sh001"],"verified_shots":["sc01_sh001"],"unresolved":[],"cost":{"explore_calls":1,"verify_calls":1,"frames_read":2}}\n',
                    encoding="utf-8",
                )
                (self.workspace.root / "evidence_ledger.jsonl").write_text(
                    '{"finding_id":"ev_0001","query_id":"q1","shot_id":"sc01_sh001","summary":"A red car is visible.","supports_options":["A"],"refutes_options":[],"citation_ids":["ev_0001"],"confidence":0.9}\n',
                    encoding="utf-8",
                )
                (self.workspace.root / "coverage.json").write_text(
                    '{"explored_shots":["sc01_sh001"],"verified_shots":["sc01_sh001"]}\n',
                    encoding="utf-8",
                )
                (self.workspace.root / "trace.jsonl").write_text('{"event":"driver_run"}\n', encoding="utf-8")
                return WorkspaceRunResult(
                    answer="A. v3 answer",
                    citations=("ev_0001",),
                    confidence="medium",
                    rounds=2,
                    metadata={"status": "final", "strategy": "multi_v3"},
                )

        class FakeInvestigatorWorkspace:
            def __init__(self, root):
                self.root = Path(root)
                created["workspaces"].append(self)

        class FakeVideoIndex:
            scenes = (type("Scene", (), {"scene_id": "sc01"})(),)

            def summary(self):
                return "sc01 [0-12]"

        def fake_build_video_index(scene_index, **kwargs):
            created["indexes"].append((scene_index, kwargs))
            return FakeVideoIndex()

        def fake_build_overview(index, *, output_dir, cols=8):
            created["overviews"].append((index, output_dir, cols))
            return FakeOverview()

        with tempfile.TemporaryDirectory() as tmp:
            workspace_root = Path(tmp) / "workspaces"
            scene_index = SceneIndex(
                video_path="/videos/demo.mp4",
                duration_sec=12.0,
                segments=[VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=12.0)],
            )
            with patch.object(eval_runner, "build_video_index_from_scene_index", side_effect=fake_build_video_index, create=True):
                with patch.object(eval_runner, "build_scene_timeline_overview", side_effect=fake_build_overview, create=True):
                    with patch.object(eval_runner, "InvestigatorWorkspaceV3", FakeInvestigatorWorkspace, create=True):
                        with patch.object(eval_runner, "ReasonerV3", FakeReasoner, create=True):
                            with patch.object(eval_runner, "InvestigatorV3", FakeInvestigator, create=True):
                                with patch.object(eval_runner, "MultiV3Driver", FakeDriver, create=True):
                                    with patch.object(
                                        eval_runner.EvidenceWorkspace,
                                        "create",
                                        side_effect=AssertionError("legacy workspace create should not run"),
                                    ):
                                        raw = eval_runner.run_loop(
                                            backend=object(),
                                            video_path="/videos/demo.mp4",
                                            question="Question: Which object is visible?\nOptions:\nA. red car\nB. blue car",
                                            duration_sec=12.0,
                                            run_id="case_multi_v3",
                                            scene_index=scene_index,
                                            workspace_root=workspace_root,
                                            budget=AgentBudget(max_rounds=4),
                                            extract_clips=False,
                                            strategy="multi_v3",
                                        )

            self.assertEqual(created["indexes"][0][0], scene_index)
            self.assertIsInstance(created["overviews"][0][0], FakeVideoIndex)
            self.assertEqual(created["drivers"][0].max_rounds, 4)
            self.assertEqual(created["drivers"][0].valid_scene_ids, ("sc01",))
            self.assertEqual(raw["answer"], "A. v3 answer")
            self.assertEqual(raw["choice"], "A")
            self.assertEqual(raw["status"], "final")
            self.assertEqual(raw["strategy"], "multi_v3")
            self.assertGreater(raw["trajectory_action_count"], 0)
            self.assertTrue(Path(raw["trajectory_path"]).exists())
            self.assertFalse((workspace_root / "runs" / "case_multi_v3" / "observations.jsonl").exists())
            self.assertFalse((workspace_root / "runs" / "case_multi_v3" / "memory.jsonl").exists())
            self.assertEqual(raw["workspace_log_dir"], str(workspace_root.parent / "workspace_logs" / "case_multi_v3"))

    def test_default_multi_v3_frame_sampler_uses_sample_shot_frames(self):
        from runs import eval_runner

        calls = []

        def fake_sample_shot_frames(video_path, start_sec, end_sec, *, n_frames, out_dir):
            calls.append((video_path, start_sec, end_sec, n_frames, Path(out_dir).name))
            return (
                eval_runner.Frame(
                    frame_id="fr001",
                    time_sec=float(start_sec),
                    thumb_path=str(Path(out_dir) / "frame_001.jpg"),
                ),
            )

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(eval_runner, "sample_shot_frames", side_effect=fake_sample_shot_frames):
                sampler = eval_runner._default_multi_v3_frame_sampler(artifact_dir=Path(tmp))
                paths = sampler("/videos/demo.mp4", 1.25, 3.75, 5)

        self.assertEqual(paths, (str(Path(tmp) / "1.250_3.750_5" / "frame_001.jpg"),))
        self.assertEqual(calls, [("/videos/demo.mp4", 1.25, 3.75, 5, "1.250_3.750_5")])


if __name__ == "__main__":
    unittest.main()
