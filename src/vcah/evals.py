from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vcah.agent import VideoAgent


@dataclass(frozen=True)
class VideoMMECase:
    case_id: str
    video_path: Path
    question: str
    answer: str = ""


def load_videomme_case(root: Path, case_id: str) -> VideoMMECase:
    root = Path(root)
    cases_path = root / "cases.json"
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = payload if isinstance(payload, list) else payload.get("cases", [])
    for item in cases:
        if str(item.get("case_id") or item.get("id")) == str(case_id):
            return VideoMMECase(
                case_id=str(case_id),
                video_path=root / str(item.get("video") or item.get("video_path") or ""),
                question=str(item.get("question") or ""),
                answer=str(item.get("answer") or ""),
            )
    raise ValueError(f"Unknown VideoMME case: {case_id}")


def run_videomme_case(root: Path, case_id: str, *, run_dir: Path, agent: VideoAgent | None = None, duration_sec: float | None = None) -> dict[str, Any]:
    case = load_videomme_case(root, case_id)
    agent = agent or VideoAgent()
    answer = agent.ask(str(case.video_path), case.question, run_dir=run_dir, duration_sec=duration_sec)
    return {"case_id": case.case_id, "answer": answer.answer, "citations": list(answer.citations), "gold": case.answer}
