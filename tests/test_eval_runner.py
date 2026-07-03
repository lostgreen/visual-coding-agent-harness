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
            grid_path = "/tmp/overview/scene_timeline_grid.json"
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
            def __init__(self, *, reasoner, investigator, workspace, max_rounds, max_concurrency=4):
                self.reasoner = reasoner
                self.investigator = investigator
                self.workspace = workspace
                self.max_rounds = max_rounds
                self.max_concurrency = max_concurrency
                created["drivers"].append(self)

            def run(self, *, question, options, index_context, overview_path):
                self.workspace.root.mkdir(parents=True, exist_ok=True)
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

        def fake_build_video_index(scene_index, **kwargs):
            created["indexes"].append((scene_index, kwargs))
            return "video-index-v3"

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
            self.assertEqual(created["overviews"][0][0], "video-index-v3")
            self.assertEqual(created["drivers"][0].max_rounds, 4)
            self.assertEqual(raw["answer"], "A. v3 answer")
            self.assertEqual(raw["choice"], "A")
            self.assertEqual(raw["status"], "final")
            self.assertEqual(raw["strategy"], "multi_v3")
            self.assertEqual(raw["workspace_log_dir"], str(workspace_root.parent / "workspace_logs" / "case_multi_v3"))


if __name__ == "__main__":
    unittest.main()
