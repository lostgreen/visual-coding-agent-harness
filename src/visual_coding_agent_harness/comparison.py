"""Ablation runners for direct video prompting versus map-first exploration."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from .agents.iterative_agent import AgentBudget, IterativeRunResult
from .backends.base import BackendRequest, VisionLanguageBackend
from .iterative_smoke import run_iterative_smoke
from .video_index import SceneIndex
from .workspace import EvidenceWorkspace


@dataclass(frozen=True)
class DescriptionComparisonConfig:
    model_path: str
    media_path: str
    question: str
    duration_sec: float
    window_sec: float = 30.0
    run_id: str = "qwen_description_comparison"
    max_rounds: int = 4
    direct_nframes: int = 64
    max_pixels: int = 151200
    extract_clips: bool = False


@dataclass(frozen=True)
class DescriptionComparisonResult:
    question: str
    video_path: str
    direct_answer: str
    direct_seconds: float
    exploration_result: IterativeRunResult
    exploration_seconds: float
    report_path: str

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "question": self.question,
            "video_path": self.video_path,
            "strategies": [
                {
                    "name": "direct_full_video",
                    "answer": self.direct_answer,
                    "seconds": round(self.direct_seconds, 3),
                },
                {
                    "name": "map_first_explore",
                    "answer": self.exploration_result.answer,
                    "status": self.exploration_result.status,
                    "citations": list(self.exploration_result.citations),
                    "confidence": self.exploration_result.confidence,
                    "rounds": len(self.exploration_result.rounds),
                    "seconds": round(self.exploration_seconds, 3),
                },
            ],
            "artifacts": {
                "report_path": self.report_path,
            },
        }


def run_description_comparison(
    *,
    base_dir: Path,
    backend: VisionLanguageBackend,
    media_path: str,
    duration_sec: float,
    question: str = "Describe the video.",
    window_sec: float = 30.0,
    run_id: str = "description_comparison",
    scene_index: Optional[SceneIndex] = None,
    budget: Optional[AgentBudget] = None,
    direct_nframes: int = 64,
    max_pixels: int = 151200,
    extract_clips: bool = False,
) -> DescriptionComparisonResult:
    """Run a direct description baseline and a map-first exploration strategy."""

    workspace = EvidenceWorkspace.create(base_dir=base_dir, run_id=run_id)
    direct_start = time.perf_counter()
    direct_response = backend.generate(
        BackendRequest(
            task="direct_description",
            prompt=_direct_description_prompt(question=question, duration_sec=duration_sec),
            media_path=media_path,
            media_type="video",
            max_new_tokens=768,
            metadata={
                "strategy": "direct_full_video",
                "duration_sec": duration_sec,
                "nframes": int(direct_nframes),
                "max_pixels": int(max_pixels),
            },
        )
    )
    direct_seconds = time.perf_counter() - direct_start
    direct_answer = direct_response.text.strip()
    workspace.write_trace_event(
        "comparison_direct_description",
        {"seconds": round(direct_seconds, 3), "answer_chars": len(direct_answer)},
    )

    explore_start = time.perf_counter()
    exploration_result = run_iterative_smoke(
        base_dir=base_dir,
        backend=backend,
        media_path=media_path,
        question=question,
        duration_sec=duration_sec,
        window_sec=window_sec,
        run_id=f"{run_id}_explore",
        scene_index=scene_index,
        budget=budget,
        extract_clips=extract_clips,
    )
    exploration_seconds = time.perf_counter() - explore_start

    result = DescriptionComparisonResult(
        question=question,
        video_path=media_path,
        direct_answer=direct_answer,
        direct_seconds=direct_seconds,
        exploration_result=exploration_result,
        exploration_seconds=exploration_seconds,
        report_path=str(workspace.root / "comparison.json"),
    )
    report = result.to_dict()
    (workspace.root / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    workspace.write_trace_event(
        "comparison_finished",
        {
            "direct_seconds": round(direct_seconds, 3),
            "exploration_seconds": round(exploration_seconds, 3),
            "exploration_status": exploration_result.status,
        },
    )
    return result


def run_qwen_description_comparison(
    config: DescriptionComparisonConfig,
    *,
    base_dir: Optional[Path] = None,
    scene_index: Optional[SceneIndex] = None,
) -> DescriptionComparisonResult:
    from .backends.qwen_vl import QwenVLBackend

    backend = QwenVLBackend.from_pretrained(config.model_path)
    return run_description_comparison(
        base_dir=base_dir or Path("."),
        backend=backend,
        media_path=config.media_path,
        duration_sec=config.duration_sec,
        question=config.question,
        window_sec=config.window_sec,
        run_id=config.run_id,
        scene_index=scene_index,
        budget=AgentBudget(max_rounds=config.max_rounds),
        direct_nframes=config.direct_nframes,
        max_pixels=config.max_pixels,
        extract_clips=config.extract_clips,
    )


def _direct_description_prompt(*, question: str, duration_sec: float) -> str:
    return (
        "Describe the input video directly from the provided video context.\n"
        f"Video duration: {duration_sec:.1f} seconds.\n"
        "Do not use external tools. Produce a concise answer and mention uncertainty when the video is too long.\n"
        f"Task: {question}"
    )
