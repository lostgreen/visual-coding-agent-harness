"""Smoke runner for a VLM main-agent plus VLM-backed tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .agents.vlm_agent import AgentRunResult, VisualAgent
from .backends.base import VisionLanguageBackend
from .workspace import EvidenceWorkspace


@dataclass(frozen=True)
class SmokeConfig:
    model_path: str
    media_path: str
    question: str
    media_type: str = "video"
    run_id: str = "qwen_vl_smoke"


def run_vlm_smoke(
    *,
    base_dir: Path,
    backend: VisionLanguageBackend,
    media_path: str,
    question: str,
    run_id: str = "vlm_smoke",
    media_type: str = "video",
) -> AgentRunResult:
    workspace = EvidenceWorkspace.create(base_dir=base_dir, run_id=run_id)
    agent = VisualAgent.with_vlm_tools(backend=backend, workspace=workspace)
    return agent.run(question=question, media_path=media_path, media_type=media_type)


def run_qwen_vl_smoke(config: SmokeConfig, *, base_dir: Optional[Path] = None) -> AgentRunResult:
    from .backends.qwen_vl import QwenVLBackend

    backend = QwenVLBackend.from_pretrained(config.model_path)
    return run_vlm_smoke(
        base_dir=base_dir or Path("."),
        backend=backend,
        media_path=config.media_path,
        question=config.question,
        run_id=config.run_id,
        media_type=config.media_type,
    )
