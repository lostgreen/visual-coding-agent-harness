from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


def test_videomme_runner_is_multi_v3_only() -> None:
    from runs import eval_runner

    assert eval_runner.parse_strategies(None) == ("multi_v3",)
    assert eval_runner.parse_strategies(("multi_v3",)) == ("multi_v3",)
    for legacy_strategy in ("workspace_v2", "multi_agent_v0"):
        with pytest.raises(ValueError, match=f"Unknown strategy: {legacy_strategy}"):
            eval_runner.parse_strategies((legacy_strategy,))


def test_legacy_agent_tool_and_ledger_modules_are_removed() -> None:
    removed_modules = (
        "visual_coding_agent_harness.agents.workspace_agent",
        "visual_coding_agent_harness.agents.multi",
        "visual_coding_agent_harness.agents.result",
        "visual_coding_agent_harness.tools.workspace_primitives",
        "visual_coding_agent_harness.tools.asr_binding",
        "visual_coding_agent_harness.tools.dummy",
        "visual_coding_agent_harness.tools.image_atomic",
        "visual_coding_agent_harness.tools.query_context",
        "visual_coding_agent_harness.tools.segments",
        "visual_coding_agent_harness.tools.traditional",
        "visual_coding_agent_harness.tools.video_atomic",
        "visual_coding_agent_harness.tools.vlm",
        "visual_coding_agent_harness.tools.workspace_v2",
        "visual_coding_agent_harness.tools.navigation",
        "visual_coding_agent_harness.tools.inspector",
        "visual_coding_agent_harness.tools.enrichment",
        "visual_coding_agent_harness.tools.timeline",
        "visual_coding_agent_harness.tools.global_view",
        "visual_coding_agent_harness.tools.exploration",
        "visual_coding_agent_harness.tools.verification",
        "visual_coding_agent_harness.workspace.search_ledger",
        "visual_coding_agent_harness.workspace.views",
        "visual_coding_agent_harness.workspace.context_budget",
        "visual_coding_agent_harness.workspace.distill",
        "visual_coding_agent_harness.workspace.evidence_ledger",
        "visual_coding_agent_harness.workspace.open_questions",
        "visual_coding_agent_harness.workspace.output_quality",
        "visual_coding_agent_harness.workspace.state",
        "visual_coding_agent_harness.workspace.transcript_binder",
        "visual_coding_agent_harness.workspace.workspace_state",
        "visual_coding_agent_harness.contracts.claim_modality",
        "visual_coding_agent_harness.contracts.evidence_binding",
        "visual_coding_agent_harness.contracts.ordered_sequence",
        "visual_coding_agent_harness.contracts.target_registry",
        "visual_coding_agent_harness.contracts.targets",
        "visual_coding_agent_harness.video.artifacts",
        "visual_coding_agent_harness.video.keyframes",
        "visual_coding_agent_harness.video.map",
        "visual_coding_agent_harness.video.scene_aggregate",
        "visual_coding_agent_harness.video.shot_detect",
        "visual_coding_agent_harness.video.text_norm",
        "visual_coding_agent_harness.evidence.need",
        "visual_coding_agent_harness.evidence.ledger",
        "visual_coding_agent_harness.evidence.order_extraction",
        "visual_coding_agent_harness.evidence.order_hypotheses",
        "visual_coding_agent_harness.evidence.predicates",
    )

    assert {name: importlib.util.find_spec(name) for name in removed_modules} == {
        name: None for name in removed_modules
    }


def test_active_surface_file_inventory_is_small() -> None:
    repo = Path(__file__).resolve().parents[1]

    assert _py_files(repo / "src/visual_coding_agent_harness/agents") <= {
        "__init__.py",
        "debug_hooks.py",
        "driver.py",
        "investigator.py",
        "reasoner.py",
    }
    assert _py_files(repo / "src/visual_coding_agent_harness/workspace") <= {
        "__init__.py",
        "digest.py",
        "evidence.py",
        "investigator_ws.py",
    }
    assert _py_files(repo / "src/visual_coding_agent_harness/tools") <= {
        "__init__.py",
        "explore.py",
        "frame_cache.py",
        "verify.py",
    }
    assert _py_files(repo / "src/visual_coding_agent_harness/contracts") <= {
        "__init__.py",
        "evidence.py",
        "query.py",
        "report.py",
    }
    assert _public_py_files(repo / "src/visual_coding_agent_harness/video") <= {
        "__init__.py",
        "build.py",
        "index.py",
        "overview.py",
    }


def test_active_multi_v3_modules_do_not_import_legacy_workspace_surface() -> None:
    repo = Path(__file__).resolve().parents[1]
    module_paths = (
        "src/visual_coding_agent_harness/agents/driver.py",
        "src/visual_coding_agent_harness/agents/investigator.py",
        "src/visual_coding_agent_harness/agents/reasoner.py",
        "src/visual_coding_agent_harness/tools/explore.py",
        "src/visual_coding_agent_harness/tools/verify.py",
        "src/visual_coding_agent_harness/workspace/__init__.py",
        "src/visual_coding_agent_harness/workspace/investigator_ws.py",
        "src/visual_coding_agent_harness/workspace/evidence.py",
        "src/visual_coding_agent_harness/workspace/digest.py",
        "src/visual_coding_agent_harness/contracts/__init__.py",
        "src/visual_coding_agent_harness/contracts/evidence.py",
        "src/visual_coding_agent_harness/contracts/query.py",
        "src/visual_coding_agent_harness/contracts/report.py",
        "src/visual_coding_agent_harness/video/__init__.py",
        "src/visual_coding_agent_harness/video/index.py",
        "src/visual_coding_agent_harness/video/build.py",
        "src/visual_coding_agent_harness/video/overview.py",
    )
    forbidden = (
        "visual_coding_agent_harness.agents.result",
        "visual_coding_agent_harness.tools.asr_binding",
        "visual_coding_agent_harness.tools.dummy",
        "visual_coding_agent_harness.tools.image_atomic",
        "visual_coding_agent_harness.tools.query_context",
        "visual_coding_agent_harness.tools.segments",
        "visual_coding_agent_harness.tools.traditional",
        "visual_coding_agent_harness.tools.video_atomic",
        "visual_coding_agent_harness.tools.vlm",
        "visual_coding_agent_harness.tools.workspace_primitives",
        "visual_coding_agent_harness.contracts.claim_modality",
        "visual_coding_agent_harness.contracts.evidence_binding",
        "visual_coding_agent_harness.contracts.ordered_sequence",
        "visual_coding_agent_harness.contracts.target_registry",
        "visual_coding_agent_harness.contracts.targets",
        "visual_coding_agent_harness.video.artifacts",
        "visual_coding_agent_harness.video.keyframes",
        "visual_coding_agent_harness.video.map",
        "visual_coding_agent_harness.video.scene_aggregate",
        "visual_coding_agent_harness.video.shot_detect",
        "visual_coding_agent_harness.video.text_norm",
        "visual_coding_agent_harness.workspace.context_budget",
        "visual_coding_agent_harness.workspace.distill",
        "visual_coding_agent_harness.workspace.evidence_ledger",
        "visual_coding_agent_harness.workspace.open_questions",
        "visual_coding_agent_harness.workspace.output_quality",
        "visual_coding_agent_harness.workspace.state",
        "visual_coding_agent_harness.workspace.transcript_binder",
        "visual_coding_agent_harness.workspace.workspace_state",
    )

    offenders: list[str] = []
    for relative_path in module_paths:
        path = repo / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imports = [_absolute_import(node.module or "", level=node.level, package=_module_name(relative_path))]
            else:
                continue
            for imported in imports:
                if imported in forbidden:
                    offenders.append(f"{relative_path}: {imported}")

    assert offenders == []


def _py_files(path: Path) -> set[str]:
    return {item.name for item in path.glob("*.py")}


def _public_py_files(path: Path) -> set[str]:
    return {name for name in _py_files(path) if name == "__init__.py" or not name.startswith("_")}


def _module_name(relative_path: str) -> str:
    path = relative_path.removeprefix("src/").removesuffix(".py")
    return path.replace("/", ".")


def _absolute_import(module: str, *, level: int, package: str) -> str:
    if level <= 0:
        return module
    parts = package.split(".")[:-level]
    return ".".join(parts + ([module] if module else []))
