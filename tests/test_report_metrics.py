import json
import os
import tempfile
import unittest
from pathlib import Path


class ReportMetricsTest(unittest.TestCase):
    def test_build_report_counts_agent_exhausted_runs_as_incomplete_even_with_choice(self):
        from runs import report_metrics

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspaces" / "runs" / "case_workspace_v2"
            workspace.mkdir(parents=True)
            (workspace / "trace.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"type": "tool_use", "payload": {"tool": "video_ls", "arguments": {}}}),
                        json.dumps(
                            {
                                "type": "tool_use",
                                "payload": {"tool": "inspect_segment", "arguments": {"segment_id": "seg_0002"}},
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            summary_path = root / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "question_id": "605-1",
                                "gt": "D",
                                "strategies": {
                                    "workspace_v2": {
                                        "choice": "D",
                                        "correct": True,
                                        "seconds": 40.0,
                                        "status": "max_rounds_reached",
                                        "citation_count": 2,
                                    },
                                },
                                "raw_artifacts": {
                                    "workspaces": {"workspace_v2": str(workspace)}
                                },
                            },
                            {
                                "question_id": "611-2",
                                "gt": "B",
                                "strategies": {
                                    "workspace_v2": {
                                        "choice": "A",
                                        "correct": False,
                                        "seconds": 20.0,
                                        "status": "final",
                                        "tools": ["video_ls", "qa_segment"],
                                        "segments": ["seg_0001"],
                                        "citation_count": 1,
                                    }
                                },
                            },
                        ]
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            report = report_metrics.build_report(summary_path)

            agent = report["strategies"]["workspace_v2"]
            self.assertEqual(agent["accuracy"], "1/2")
            self.assertEqual(agent["raw_choice_accuracy"], "1/2")
            self.assertEqual(agent["grounded_choice_accuracy"], "0/2")
            self.assertEqual(agent["cited_answer_rate"], 0.5)
            self.assertEqual(agent["final_rate"], 0.5)
            self.assertEqual(agent["incomplete_rate"], 0.5)
            self.assertEqual(agent["avg_seconds"], 30.0)
            self.assertNotIn("direct_regressions", agent)
            self.assertNotIn("avg_walltime_vs_direct", agent)

            detail = report["cases"][0]["strategies"]["workspace_v2"]
            self.assertTrue(detail["incomplete"])
            self.assertEqual(detail["tool_sequence"], ["video_ls", "inspect_segment"])
            self.assertEqual(detail["unique_inspected_segments"], ["seg_0002"])
            self.assertEqual(detail["citation_count"], 2)
            self.assertFalse(detail["grounded_correct"])

            rendered = report_metrics.render_markdown(report)
            self.assertIn("workspace_v2", rendered)
            self.assertNotIn("Direct Regressions", rendered)
            self.assertNotIn("Avg vs Direct", rendered)
            self.assertIn("50.0%", rendered)
            self.assertIn("video_ls -> inspect_segment", rendered)

    def test_build_report_flags_conflicting_unsupported_final(self):
        from runs import report_metrics
        from visual_coding_agent_harness.workspace import EvidenceWorkspace

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = EvidenceWorkspace.create(root / "workspaces", run_id="case_611_workspace_v2")
            d_observation = workspace.write_observation(
                tool_name="verify_local_claim",
                input_artifacts=["clip_d.mp4"],
                claim="The visible order supports option D.",
                confidence=0.82,
                regions=[{"start_sec": 300.0, "end_sec": 600.0}],
                limitations="Directly visible in the sampled segment.",
                raw_output={"supported_option": "D", "grounding_quality": "visually_confirmed"},
            )
            a_observation = workspace.write_observation(
                tool_name="verify_local_claim",
                input_artifacts=["clip_a.mp4"],
                claim="The caption guesses option A.",
                confidence=0.91,
                regions=[{"start_sec": 1500.0, "end_sec": 1800.0}],
                limitations="Inferred from context; lacks explicit visual confirmation.",
                raw_output={"supported_option": "A"},
            )
            workspace.write_ledger_entry(d_observation)
            workspace.write_ledger_entry(a_observation)

            summary_path = root / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "question_id": "611-2",
                                "gt": "D",
                                "question": "Which sculpture order is correct?",
                                "options": [
                                    "A. first order",
                                    "B. second order",
                                    "C. third order",
                                    "D. fourth order",
                                ],
                                "strategies": {
                                    "workspace_v2": {
                                        "choice": "A",
                                        "correct": False,
                                        "status": "final",
                                        "seconds": 271.0,
                                        "citations": ["obs_0002"],
                                        "citation_count": 1,
                                    }
                                },
                                "raw_artifacts": {
                                    "workspaces": {"workspace_v2": str(workspace.root)}
                                },
                            }
                        ]
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            report = report_metrics.build_report(summary_path)

            detail = report["cases"][0]["strategies"]["workspace_v2"]
            self.assertTrue(detail["has_conflict"])
            self.assertTrue(detail["final_with_conflict"])
            self.assertTrue(detail["unsupported_final"])
            self.assertFalse(detail["option_support_consistency"])
            self.assertEqual(detail["top_supported_option"], "D")

            metrics = report["strategies"]["workspace_v2"]
            self.assertEqual(metrics["conflict_rate"], 1.0)
            self.assertEqual(metrics["final_with_conflict_rate"], 1.0)
            self.assertEqual(metrics["unsupported_final_rate"], 1.0)
            self.assertEqual(metrics["option_support_consistency_rate"], 0.0)

    def test_build_report_counts_legacy_worker_vote_rows(self):
        from runs import report_metrics
        from visual_coding_agent_harness.workspace import EvidenceWorkspace

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = EvidenceWorkspace.create(root / "workspaces", run_id="case_legacy_votes")
            legacy = workspace.write_observation(
                tool_name="inspect_segment",
                input_artifacts=["clip.mp4"],
                claim="The local worker says Supported option: B.",
                confidence=0.82,
                limitations="Local observation only.",
                raw_output={"supported_option": "B", "grounding_quality": "visually_confirmed"},
            )
            global_floor = workspace.write_observation(
                tool_name="global_gist",
                input_artifacts=["video.mp4"],
                claim="Supported option: D. Whole-video synopsis.",
                confidence=0.76,
                limitations="Sparse full-video sampling.",
                raw_output={"supported_option": "D", "grounding_quality": "global_sparse"},
            )
            workspace.write_ledger_entry(legacy)
            workspace.write_ledger_entry(global_floor)

            summary_path = root / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "question_id": "605-1",
                                "gt": "D",
                                "question": "What is the video mainly about?",
                                "options": ["A. one", "B. local guess", "C. three", "D. synopsis"],
                                "strategies": {
                                    "workspace_v2": {
                                        "choice": "D",
                                        "correct": True,
                                        "status": "final",
                                        "citations": ["obs_0002"],
                                    }
                                },
                                "raw_artifacts": {"workspaces": {"workspace_v2": str(workspace.root)}},
                            }
                        ]
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            report = report_metrics.build_report(summary_path)

            detail = report["cases"][0]["strategies"]["workspace_v2"]
            self.assertEqual(detail["legacy_worker_vote_rows"], 1)
            self.assertNotIn("D", detail["option_support"])
            self.assertNotIn("B", detail["option_support"])
            self.assertEqual(report["strategies"]["workspace_v2"]["legacy_worker_vote_rows"], 1)

    def test_build_report_resolves_repo_relative_workspace_paths(self):
        from runs import report_metrics
        from visual_coding_agent_harness.workspace import EvidenceWorkspace

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "runs" / "round3"
            workspace = EvidenceWorkspace.create(run_root / "workspaces", run_id="case_workspace_v2")
            observation = workspace.write_observation(
                tool_name="global_gist",
                claim="Supported option: D. The whole video is about the empire rising and falling.",
                confidence=0.76,
                regions=[{"start_sec": 0.0, "end_sec": 1896.0}],
                limitations="Sparse full-video sampling.",
                raw_output={"supported_option": "D", "grounding_quality": "global_sparse"},
            )
            workspace.write_ledger_entry(observation)
            workspace.write_trace_event(
                "iterative_final",
                {"answer": "D. whole-video synopsis", "citations": [observation.observation_id]},
            )
            summary_path = run_root / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "question_id": "605-1",
                                "gt": "D",
                                "question": "Question: What's the main idea of the video?",
                                "options": ["A. one", "B. two", "C. three", "D. whole-video synopsis"],
                                "strategies": {
                                    "workspace_v2": {
                                        "choice": "D",
                                        "correct": True,
                                        "status": "final",
                                        "citation_count": 1,
                                    },
                                },
                                "raw_artifacts": {
                                    "workspaces": {
                                        "workspace_v2": "runs/round3/workspaces/runs/case_workspace_v2"
                                    }
                                },
                            }
                        ]
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            old_cwd = os.getcwd()
            try:
                os.chdir(root)
                report = report_metrics.build_report(summary_path)
            finally:
                os.chdir(old_cwd)

            detail = report["cases"][0]["strategies"]["workspace_v2"]
            self.assertEqual(detail["workspace"], "runs/round3/workspaces/runs/case_workspace_v2")
            self.assertEqual(detail["citations"], ["obs_0001"])
            self.assertNotIn("D", detail["option_support"])
            self.assertEqual(detail["top_supported_option"], "")


if __name__ == "__main__":
    unittest.main()
