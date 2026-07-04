from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from vcah.index import build_cold_index
from vcah.memory import AgentMemory, EvidenceStore, TraceStore
from vcah.model import ModelClient
from vcah.tools import AgentTools
from vcah.types import Answer, ToolAction
from vcah.video import probe_duration


class VideoAgent:
    def __init__(self, *, model: ModelClient | None = None, max_steps: int = 8) -> None:
        self.model = model or ModelClient()
        self.max_steps = max(1, int(max_steps))

    def ask(
        self,
        video_path: str,
        question: str,
        *,
        run_dir: Path,
        duration_sec: float | None = None,
        asr_cues: Sequence[Any] = (),
        ocr_lines: Sequence[Any] = (),
        range_detector: Any = None,
        keyframe_sampler: Any = None,
        index_mode: str = "fast",
    ) -> Answer:
        run_dir = Path(run_dir)
        run_artifacts = run_dir / "run"
        run_artifacts.mkdir(parents=True, exist_ok=True)
        duration = float(duration_sec) if duration_sec is not None else probe_duration(video_path)
        index = build_cold_index(
            video_path,
            duration_sec=duration,
            run_dir=run_dir,
            model=self.model,
            asr_cues=asr_cues,
            ocr_lines=ocr_lines,
            range_detector=range_detector,
            keyframe_sampler=keyframe_sampler,
            index_mode=index_mode,
        )
        memory = AgentMemory.empty(run_artifacts / "memory.json")
        evidence = EvidenceStore.empty(run_artifacts / "evidence.jsonl")
        trace = TraceStore(run_artifacts / "trace.jsonl")
        tools = AgentTools(index, evidence, run_artifacts)

        for _step in range(self.max_steps):
            action = self.model.controller(question, index.timeline_digest(), memory.digest(), evidence.digest())
            if not isinstance(action, ToolAction):
                action = ToolAction.from_mapping(action)
            result = tools.run(action)
            memory.record_result(result)
            trace.append(action, result)
            memory.save()
            if action.type == "answer":
                if evidence.valid(action.citations):
                    return self._write_answer(run_artifacts, Answer(action.answer, action.citations, run_dir))
                return self._write_answer(run_artifacts, Answer("Insufficient verified evidence.", (), run_dir))

        return self._write_answer(run_artifacts, Answer("Insufficient verified evidence.", (), run_dir))

    def _write_answer(self, run_artifacts: Path, answer: Answer) -> Answer:
        (run_artifacts / "answer.json").write_text(
            json.dumps({"answer": answer.answer, "citations": list(answer.citations)}, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return answer
