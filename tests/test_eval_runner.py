import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from visual_coding_agent_harness.agents.driver import WorkspaceRunResult
from visual_coding_agent_harness.core.budget import AgentBudget
from visual_coding_agent_harness.evals.videomme.dvc_compat import SceneIndex, VideoSegment
from visual_coding_agent_harness.video.index import Frame
from visual_coding_agent_harness.workspace.text_index import InvertedIndex
from visual_coding_agent_harness.workspace.video_workspace import Beat, Chapter, VideoWorkspace
from visual_coding_agent_harness.workspace.visual_index import VisualIndex


class EmptyEmbeddingBackend:
    embedding_dim = 1

    def encode_images(self, paths):
        raise AssertionError("not used")

    def encode_text(self, queries):
        raise AssertionError("not used")


def _fake_workspace() -> VideoWorkspace:
    beat = Beat("bt00001", "ch01", 0.0, 12.0, "", "red car", (), ("sc01_sh001",))
    return VideoWorkspace(
        video_path="/videos/demo.mp4",
        duration_sec=12.0,
        chapters=(Chapter("ch01", 0.0, 12.0, ("bt00001",), ""),),
        beats=(beat,),
        text_index=InvertedIndex(),
        visual_index=VisualIndex(EmptyEmbeddingBackend()),
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
            def __init__(self, *, video_workspace, workspace, backend, memo_store=None):
                self.video_workspace = video_workspace
                self.workspace = workspace
                self.backend = backend
                self.memo_store = memo_store
                created["investigators"].append(self)

        class FakeDriver:
            def __init__(
                self,
                *,
                reasoner,
                investigator,
                workspace,
                max_rounds,
                max_concurrency=4,
                valid_scene_ids=(),
                video_workspace=None,
            ):
                self.reasoner = reasoner
                self.investigator = investigator
                self.workspace = workspace
                self.max_rounds = max_rounds
                self.max_concurrency = max_concurrency
                self.valid_scene_ids = tuple(valid_scene_ids)
                self.video_workspace = video_workspace
                created["drivers"].append(self)

            def run(self, *, question, options, index_context, overview_image_path=""):
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

        def fake_build_workspace(*args, **kwargs):
            created["indexes"].append((args, kwargs))
            return _fake_workspace()

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
            with patch.object(eval_runner, "build_video_workspace", side_effect=fake_build_workspace, create=True):
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

            self.assertEqual(created["indexes"][0][0][0], "/videos/demo.mp4")
            self.assertIsInstance(created["overviews"][0][0], VideoWorkspace)
            self.assertEqual(created["drivers"][0].max_rounds, 4)
            self.assertEqual(created["drivers"][0].valid_scene_ids, ("ch01",))
            self.assertEqual(
                created["investigators"][0].memo_store.path,
                workspace_root / "runs" / "case_multi_v3" / "multi_v3" / "observation_memos.jsonl",
            )
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

        def fake_sample_shot_frames(video_path, start_sec, end_sec, *, n_frames, out_dir, size=None):
            calls.append((video_path, start_sec, end_sec, n_frames, Path(out_dir).name, size))
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
        self.assertEqual(calls, [("/videos/demo.mp4", 1.25, 3.75, 5, "1.250_3.750_5", None)])

    def test_run_loop_keeps_default_verify_sampler_separate_from_index_sampler(self):
        from runs import eval_runner

        captured = {"index_kwargs": None, "sample_calls": []}
        beat = Beat("bt00001", "ch01", 1.0, 3.0, "/index/thumb.jpg", "", (), ("sc01_sh001",))
        video_index = VideoWorkspace(
            video_path="/videos/demo.mp4",
            duration_sec=4.0,
            chapters=(Chapter("ch01", 0.0, 4.0, ("bt00001",), "/index/thumb.jpg"),),
            beats=(beat,),
            text_index=InvertedIndex(),
            visual_index=VisualIndex(EmptyEmbeddingBackend()),
        )

        class FakeOverview:
            grid_image_path = ""
            grid_path = ""
            manifest_path = ""

        class FakeInvestigatorWorkspace:
            def __init__(self, root):
                self.root = Path(root)

        class FakeReasoner:
            def __init__(self, *, backend):
                self.backend = backend

        class FakeInvestigator:
            def __init__(self, *, video_workspace, workspace, backend, frame_sampler=None, memo_store=None):
                self.video_workspace = video_workspace
                self.workspace = workspace
                self.backend = backend
                self.frame_sampler = frame_sampler
                self.memo_store = memo_store

        class FakeDriver:
            def __init__(
                self,
                *,
                reasoner,
                investigator,
                workspace,
                max_rounds,
                max_concurrency=4,
                valid_scene_ids=(),
                video_workspace=None,
            ):
                self.investigator = investigator
                self.workspace = workspace
                self.video_workspace = video_workspace

            def run(self, **kwargs):
                del kwargs
                self.workspace.root.mkdir(parents=True, exist_ok=True)
                assert self.investigator.frame_sampler is not None
                paths = self.investigator.frame_sampler(beat, 1)
                assert "multi_v3_verify_frames" in paths[0]
                assert paths[0] != beat.keyframe_path
                return WorkspaceRunResult(answer="A", citations=(), confidence="medium", rounds=1, metadata={"status": "final"})

        def fake_build_workspace(*args, **kwargs):
            captured["index_kwargs"] = kwargs
            return video_index

        def fake_sample_shot_frames(video_path, start_sec, end_sec, *, n_frames, out_dir, size=None):
            captured["sample_calls"].append((video_path, start_sec, end_sec, n_frames, str(out_dir), size))
            return (Frame(frame_id="verify", time_sec=start_sec, thumb_path=str(Path(out_dir) / "verify.jpg")),)

        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "demo.mp4"
            video_path.write_bytes(b"not a real mp4; build is patched")
            workspace_root = Path(tmp) / "workspaces"
            scene_index = SceneIndex(
                video_path=str(video_path),
                duration_sec=4.0,
                segments=(VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=4.0),),
            )
            with patch.object(eval_runner, "build_video_workspace", side_effect=fake_build_workspace):
                with patch.object(eval_runner, "sample_shot_frames", side_effect=fake_sample_shot_frames):
                    with patch.object(eval_runner, "build_scene_timeline_overview", return_value=FakeOverview()):
                        with patch.object(eval_runner, "InvestigatorWorkspaceV3", FakeInvestigatorWorkspace, create=True):
                            with patch.object(eval_runner, "ReasonerV3", FakeReasoner, create=True):
                                with patch.object(eval_runner, "InvestigatorV3", FakeInvestigator, create=True):
                                    with patch.object(eval_runner, "MultiV3Driver", FakeDriver, create=True):
                                        eval_runner.run_loop(
                                            backend=object(),
                                            video_path=str(video_path),
                                            question="Question: Which object?\nOptions:\nA. demo",
                                            duration_sec=4.0,
                                            run_id="case_default_verify",
                                            scene_index=scene_index,
                                            workspace_root=workspace_root,
                                            budget=AgentBudget(max_rounds=1),
                                            strategy="multi_v3",
                                        )

        self.assertIs(captured["index_kwargs"]["keyframe_sampler"], eval_runner._placeholder_keyframe_sampler)
        self.assertEqual(captured["sample_calls"][0][-1], None)


if __name__ == "__main__":
    unittest.main()
