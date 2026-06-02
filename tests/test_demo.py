import tempfile
import unittest
from pathlib import Path

from visual_coding_agent_harness.demo import run_demo


class DemoTest(unittest.TestCase):
    def test_run_demo_creates_trace_observations_and_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_demo(base_dir=Path(tmp), run_id="demo_case")
            run_root = Path(tmp) / "runs" / "demo_case"

            self.assertEqual(result.observation_ids, ["obs_0001", "obs_0002"])
            self.assertTrue((run_root / "observations.jsonl").exists())
            self.assertTrue((run_root / "trace.jsonl").exists())
            self.assertIn("red cup", (run_root / "ledger.md").read_text())
            self.assertIn("EXIT", (run_root / "ledger.md").read_text())


if __name__ == "__main__":
    unittest.main()
