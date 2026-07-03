import os
import subprocess
import sys
import ast
from pathlib import Path


def test_videomme_eval_code_lives_in_package_with_legacy_runs_compatibility():
    from visual_coding_agent_harness.evals.videomme import metrics, runner, summary_schema, training_trajectory, trajectory_markdown
    from runs import eval_runner, report_metrics
    from runs.summary_schema import RunSummary
    from runs.training_trajectory import TrainingTrajectory

    assert eval_runner.EvalConfig is runner.EvalConfig
    assert report_metrics.build_report is metrics.build_report
    assert RunSummary is summary_schema.RunSummary
    assert TrainingTrajectory is training_trajectory.TrainingTrajectory
    assert callable(trajectory_markdown.render_trajectory_markdown)


def test_ablation_code_lives_in_package_with_legacy_scripts_compatibility():
    from visual_coding_agent_harness.evals.ablation import audit, matrix, report
    from scripts import audit_trajectory, generate_ablation_report, run_ablation

    assert audit_trajectory.audit_trajectory is audit.audit_trajectory
    assert run_ablation.load_matrix is matrix.load_matrix
    assert generate_ablation_report.build_report is report.build_report


def test_new_package_cli_entrypoints_have_help():
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = "src:."

    for module in (
        "visual_coding_agent_harness.cli.eval_videomme",
        "visual_coding_agent_harness.cli.run_ablation",
        "visual_coding_agent_harness.cli.generate_ablation_report",
        "visual_coding_agent_harness.cli.audit_trajectory",
        "visual_coding_agent_harness.cli.trajectory_markdown",
    ):
        completed = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            cwd=repo_root,
            env=env,
            check=False,
            text=True,
            capture_output=True,
            timeout=20,
        )

        assert completed.returncode == 0, completed.stderr[:500]
        assert "usage:" in completed.stdout.lower()


def test_pyproject_exposes_packaged_eval_console_scripts():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = pyproject_path.read_text(encoding="utf-8")

    assert "[project.scripts]" in pyproject
    assert 'vh-eval-videomme = "visual_coding_agent_harness.cli.eval_videomme:main"' in pyproject
    assert 'vh-run-ablation = "visual_coding_agent_harness.cli.run_ablation:main"' in pyproject
    assert 'vh-ablation-report = "visual_coding_agent_harness.cli.generate_ablation_report:main"' in pyproject
    assert 'vh-audit-trajectory = "visual_coding_agent_harness.cli.audit_trajectory:main"' in pyproject
    assert 'vh-trajectory-markdown = "visual_coding_agent_harness.cli.trajectory_markdown:main"' in pyproject


def test_videomme_eval_import_dependencies_postpone_annotations_for_python38():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "src/visual_coding_agent_harness/legacy/workspace_v2/context_budget.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))

    assert any(
        isinstance(node, ast.ImportFrom) and node.module == "__future__" and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )
