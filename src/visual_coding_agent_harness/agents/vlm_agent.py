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
class AgentInput:
    question: str
    media_path: str
    media_type: str
    tool_policy: str = "required"

    def to_dict(self) -> Mapping[str, str]:
        return {
            "question": self.question,
            "media_path": self.media_path,
            "media_type": self.media_type,
            "tool_policy": self.tool_policy,
        }


@dataclass(frozen=True)
class AgentRunResult:
    agent_input: AgentInput
    answer: str
    program: Sequence[Mapping[str, Any]]
    program_result: ProgramResult
    planner_text: str

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "input": self.agent_input.to_dict(),
            "output": {
                "answer": self.answer,
                "program": list(self.program),
                "observation_ids": list(self.program_result.observation_ids),
                "assignments": dict(self.program_result.assignments),
            },
            "debug": {
                "planner_text": self.planner_text,
            },
        }


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

    def run(
        self,
        *,
        question: str,
        media_path: str,
        media_type: str,
        tool_policy: str = "required",
    ) -> AgentRunResult:
        agent_input = AgentInput(
            question=question,
            media_path=media_path,
            media_type=media_type,
            tool_policy=tool_policy,
        )
        planner_response = self.backend.generate(
            BackendRequest(
                task="plan",
                prompt=_planning_prompt(agent_input=agent_input),
                media_path=media_path,
                media_type=media_type,
                max_new_tokens=512,
                metadata={"tool_policy": tool_policy},
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
            agent_input=agent_input,
            answer=str(plan["answer"]),
            program=plan["program"],
            program_result=program_result,
            planner_text=planner_response.text,
        )


def _planning_prompt(*, agent_input: AgentInput) -> str:
    return (
        "You are a visual agent that must plan tool calls before answering.\n"
        "Use only these tools; do not invent tool names or media paths.\n"
        "Available tools:\n"
        "- caption_video(video_path: str, question: str = 'Describe the video.', nframes: int = 8, max_pixels: int = 151200)\n"
        "- qa_video(video_path: str, question: str, nframes: int = 8, max_pixels: int = 151200)\n"
        "- caption_image(image_path: str, question: str = 'Describe the image.')\n"
        "- qa_image(image_path: str, question: str)\n"
        "Return only JSON in this schema:\n"
        '{"answer": string, "program": [{"tool": string, "args": object, "assign": string}]}\n'
        "Rules:\n"
        "- For video and tool_policy=required, call at least caption_video; add qa_video when the question asks for a specific answer.\n"
        "- For image and tool_policy=required, call at least caption_image; add qa_image when the question asks for a specific answer.\n"
        "- The caller will bind the real media path; use the media path fields shown in the tool signatures.\n"
        f"Input:\n"
        f"- media_type: {agent_input.media_type}\n"
        f"- media_path: {agent_input.media_path}\n"
        f"- tool_policy: {agent_input.tool_policy}\n"
        f"- question: {agent_input.question}"
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
    args.pop("media_path", None)
    resolved_media_path = str(media_path)
    if tool_name.endswith("_video"):
        args["video_path"] = resolved_media_path
    if tool_name.endswith("_image"):
        args["image_path"] = resolved_media_path
