"""Smoke runner for iterative long-video exploration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .agents.iterative_agent import AgentBudget, IterativeRunResult, IterativeVisualAgent
from .backends.base import VisionLanguageBackend
from .tools.frame_cache import FrameSampler
from .tools.exploration import build_video_exploration_registry
from .video_index import SceneIndex, fixed_window_scene_index
from .video_map import VideoMap
from .workspace import EvidenceWorkspace


@dataclass(frozen=True)
class IterativeSmokeConfig:
    model_path: str
    media_path: str
    question: str
    duration_sec: float
    window_sec: float = 30.0
    run_id: str = "qwen_vl_iterative_smoke"
    max_rounds: int = 4
    extract_clips: bool = False


def run_iterative_smoke(
    *,
    base_dir: Path,
    backend: VisionLanguageBackend,
    media_path: str,
    question: str,
    duration_sec: float,
    window_sec: float = 30.0,
    run_id: str = "iterative_smoke",
    scene_index: Optional[SceneIndex] = None,
    budget: Optional[AgentBudget] = None,
    extract_clips: bool = False,
    frame_sampler: Optional[FrameSampler] = None,
) -> IterativeRunResult:
    workspace = EvidenceWorkspace.create(base_dir=base_dir, run_id=run_id)
    resolved_index = scene_index or fixed_window_scene_index(
        video_path=media_path,
        duration_sec=duration_sec,
        window_sec=window_sec,
    )
    agent = IterativeVisualAgent(
        backend=backend,
        registry=build_video_exploration_registry(
            video_map=VideoMap.from_scene_index(resolved_index),
            backend=backend,
            workspace=workspace,
            extract_clips=extract_clips,
            frame_sampler=frame_sampler,
        ),
        workspace=workspace,
        scene_index=resolved_index,
        budget=budget,
    )
    return agent.run(question=question, video_path=media_path)


def run_qwen_iterative_smoke(
    config: IterativeSmokeConfig,
    *,
    base_dir: Optional[Path] = None,
) -> IterativeRunResult:
    from .backends.qwen_vl import QwenVLBackend

    backend = QwenVLBackend.from_pretrained(config.model_path)
    return run_iterative_smoke(
        base_dir=base_dir or Path("."),
        backend=backend,
        media_path=config.media_path,
        question=config.question,
        duration_sec=config.duration_sec,
        window_sec=config.window_sec,
        run_id=config.run_id,
        budget=AgentBudget(max_rounds=config.max_rounds),
        extract_clips=config.extract_clips,
    )
