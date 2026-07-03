"""Runnable P0 demo for the visual coding-agent harness."""

from __future__ import annotations

from pathlib import Path

from visual_coding_agent_harness.legacy.interpreter import ProgramInterpreter, ProgramResult
from visual_coding_agent_harness.legacy.tools.dummy import build_dummy_registry
from visual_coding_agent_harness.legacy.workspace_v2 import EvidenceWorkspace


def run_demo(base_dir: Path, run_id: str = "demo") -> ProgramResult:
    workspace = EvidenceWorkspace.create(base_dir=base_dir, run_id=run_id)
    interpreter = ProgramInterpreter(
        registry=build_dummy_registry(),
        workspace=workspace,
    )
    return interpreter.run(
        [
            {
                "tool": "caption_image",
                "args": {"image_path": "input/frame_001.jpg"},
                "assign": "global_caption",
            },
            {
                "tool": "ocr_region",
                "args": {"image_path": "artifacts/crops/sign_crop.jpg"},
                "assign": "sign_text",
            },
        ]
    )
