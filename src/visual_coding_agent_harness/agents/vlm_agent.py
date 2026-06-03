"""A minimal visual-model main agent with tool-use execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from ..backends.base import BackendRequest, VisionLanguageBackend
from ..interpreter import ProgramInterpreter, ProgramResult
from ..registry import ToolRegistry
from ..tools.vlm import build_vlm_registry
from ..workspace import EvidenceWorkspace


@dataclass(frozen=True)
class AgentRunResult:
    answer: str
    program: Sequence[Mapping[str, Any]]
    program_result: ProgramResult
    planner_text: str


class VisualAgent:
    """Plan tool calls with a VLM and execute them through the harness."""

    def __init__(
        self,
        *,
        backend: VisionLanguageBackend,
        registry: ToolRegistry,
        workspace: EvidenceWorkspace,
    ) -> None:
        self.backend = backend
        self.registry = registry
        self.workspace = workspace

    @classmethod
    def with_vlm_tools(
        cls,
        *,
        backend: VisionLanguageBackend,
        workspace: EvidenceWorkspace,
        registry: Optional[ToolRegistry] = None,
    ) -> "VisualAgent":
        return cls(
            backend=backend,
            registry=registry or build_vlm_registry(backend),
            workspace=workspace,
        )

    def run(self, *, question: str, media_path: str, media_type: str) -> AgentRunResult:
        planner_response = self.backend.generate(
            BackendRequest(
                task="plan",
                prompt=_planning_prompt(question=question, media_type=media_type),
                media_path=media_path,
                media_type=media_type,
                max_new_tokens=512,
            )
        )
        plan = _parse_plan(
            planner_response.text,
            question=question,
            media_path=media_path,
            media_type=media_type,
        )
        self.workspace.write_trace_event(
            "agent_plan",
            {"question": question, "media_path": media_path, "media_type": media_type, "plan": plan},
        )
        program_result = ProgramInterpreter(registry=self.registry, workspace=self.workspace).run(plan["program"])
        self.workspace.write_trace_event(
            "agent_answer",
            {"answer": plan["answer"], "observation_ids": list(program_result.observation_ids)},
        )
        return AgentRunResult(
            answer=str(plan["answer"]),
            program=plan["program"],
            program_result=program_result,
            planner_text=planner_response.text,
        )


def _planning_prompt(*, question: str, media_type: str) -> str:
    return (
        "You are a visual agent that may call tools before answering. "
        "Return only JSON with keys answer and program. "
        "program is a list of tool calls. For video, prefer caption_video then qa_video. "
        "For image, prefer caption_image then qa_image. "
        "Use args with media path fields filled by the caller. "
        f"Media type: {media_type}. Question: {question}"
    )


def _parse_plan(text: str, *, question: str, media_path: str, media_type: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(_extract_json_object(text))
    except (json.JSONDecodeError, ValueError):
        return _fallback_plan(question=question, media_path=media_path, media_type=media_type)

    program = payload.get("program")
    if not isinstance(program, list):
        return _fallback_plan(question=question, media_path=media_path, media_type=media_type)

    normalized = []
    for step in program:
        if not isinstance(step, Mapping) or "tool" not in step:
            continue
        tool_name = str(step["tool"])
        args = dict(step.get("args", {}))
        _fill_media_argument(args, tool_name=tool_name, media_path=media_path)
        normalized.append(
            {
                "tool": tool_name,
                "args": args,
                **({"assign": str(step["assign"])} if "assign" in step else {}),
            }
        )

    if not normalized:
        return _fallback_plan(question=question, media_path=media_path, media_type=media_type)
    return {"answer": str(payload.get("answer", "")), "program": normalized}


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    return stripped[start : end + 1]


def _fallback_plan(*, question: str, media_path: str, media_type: str) -> Mapping[str, Any]:
    if media_type == "video":
        return {
            "answer": "",
            "program": [
                {"tool": "caption_video", "args": {"video_path": media_path, "question": question}, "assign": "caption"},
                {"tool": "qa_video", "args": {"video_path": media_path, "question": question}, "assign": "qa"},
            ],
        }
    return {
        "answer": "",
        "program": [
            {"tool": "caption_image", "args": {"image_path": media_path, "question": question}, "assign": "caption"},
            {"tool": "qa_image", "args": {"image_path": media_path, "question": question}, "assign": "qa"},
        ],
    }


def _fill_media_argument(args: dict[str, Any], *, tool_name: str, media_path: str) -> None:
    generic_media_path = args.pop("media_path", None)
    resolved_media_path = str(generic_media_path or media_path)
    if tool_name.endswith("_video") and not args.get("video_path"):
        args["video_path"] = resolved_media_path
    if tool_name.endswith("_image") and not args.get("image_path"):
        args["image_path"] = resolved_media_path
