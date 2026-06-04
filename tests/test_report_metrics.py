import json
import tempfile
import unittest
from pathlib import Path


class ReportMetricsTest(unittest.TestCase):
    def test_build_report_counts_exhausted_runs_as_incomplete_even_with_choice(self):
        from runs import report_metrics

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspaces" / "runs" / "case_empty"
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
                                    "direct_full_video": {
                                        "choice": "D",
                                        "correct": True,
                                        "seconds": 5.0,
                                        "status": "ok",
                                    },
                                    "empty_index_loop": {
                                        "choice": "D",
                                        "correct": True,
                                        "seconds": 40.0,
                                        "status": "max_rounds_reached",
                                        "citation_count": 2,
                                    },
                                },
                                "raw_artifacts": {
                                    "workspaces": {"empty_index_loop": str(workspace)}
                                },
                            },
                            {
                                "question_id": "611-2",
                                "gt": "B",
                                "strategies": {
                                    "direct_full_video": {
                                        "choice": "A",
                                        "correct": False,
                                        "seconds": 5.0,
                                        "status": "ok",
                                    },
                                    "empty_index_loop": {
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

            direct = report["strategies"]["direct_full_video"]
            self.assertEqual(direct["accuracy"], "1/2")
            self.assertEqual(direct["final_rate"], 1.0)
            self.assertEqual(direct["incomplete_rate"], 0.0)

            empty = report["strategies"]["empty_index_loop"]
            self.assertEqual(empty["accuracy"], "1/2")
            self.assertEqual(empty["final_rate"], 0.5)
            self.assertEqual(empty["incomplete_rate"], 0.5)
            self.assertEqual(empty["avg_seconds"], 30.0)
            self.assertEqual(empty["avg_walltime_vs_direct"], 6.0)

            detail = report["cases"][0]["strategies"]["empty_index_loop"]
            self.assertTrue(detail["incomplete"])
            self.assertEqual(detail["tool_sequence"], ["video_ls", "inspect_segment"])
            self.assertEqual(detail["unique_inspected_segments"], ["seg_0002"])
            self.assertEqual(detail["citation_count"], 2)

            rendered = report_metrics.render_markdown(report)
            self.assertIn("empty_index_loop", rendered)
            self.assertIn("50.0%", rendered)
            self.assertIn("video_ls -> inspect_segment", rendered)


if __name__ == "__main__":
    unittest.main()
