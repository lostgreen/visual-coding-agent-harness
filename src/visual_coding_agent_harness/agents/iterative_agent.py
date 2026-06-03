"""Iterative visual agent for coarse-to-fine video exploration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from ..backends.base import BackendRequest, VisionLanguageBackend
from ..interpreter import ProgramInterpreter
from ..registry import ToolRegistry
from ..video_index import SceneIndex
from ..workspace import EvidenceWorkspace


_SEGMENT_MEDIA_TOOLS = {"caption_segment", "qa_segment"}


@dataclass(frozen=True)
class AgentBudget:
    max_rounds: int = 4
    max_tool_calls_per_round: int = 1
    default_nframes: int = 8
    high_fps_nframes: int = 32
    planner_receives_media: bool = False


@dataclass(frozen=True)
class IterativeRound:
    round_number: int
    status: str
    planner_text: str
    rationale: str = ""
    program: Sequence[Mapping[str, Any]] = field(default_factory=list)
    observation_ids: Sequence[str] = field(default_factory=list)

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "round_number": self.round_number,
            "status": self.status,
            "rationale": self.rationale,
            "program": list(self.program),
            "observation_ids": list(self.observation_ids),
            "planner_text": self.planner_text,
        }


@dataclass(frozen=True)
class IterativeRunResult:
    question: str
    video_path: str
    answer: str
    status: str
    citations: Sequence[str] = field(default_factory=list)
    confidence: float = 0.0
    rounds: Sequence[IterativeRound] = field(default_factory=list)

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "input": {"question": self.question, "video_path": self.video_path},
            "output": {
                "answer": self.answer,
                "status": self.status,
                "citations": list(self.citations),
                "confidence": self.confidence,
            },
            "rounds": [round_result.to_dict() for round_result in self.rounds],
        }


class IterativeVisualAgent:
    """Let a VLM repeatedly plan tools, inspect evidence, and decide when to stop."""

    def __init__(
        self,
        *,
        backend: VisionLanguageBackend,
        registry: ToolRegistry,
        workspace: EvidenceWorkspace,
        scene_index: SceneIndex,
        budget: Optional[AgentBudget] = None,
    ) -> None:
        self.backend = backend
        self.registry = registry
        self.workspace = workspace
        self.scene_index = scene_index
        self.budget = budget or AgentBudget()

    def run(self, *, question: str, video_path: str) -> IterativeRunResult:
        rounds: list[IterativeRound] = []
        citations: list[str] = []
        inspected_segment_ids: set[str] = set()

        for round_number in range(1, self.budget.max_rounds + 1):
            ledger_text = self._read_ledger()
            self.workspace.write_trace_event(
                "iterative_round_start",
                {"round": round_number, "question": question, "evidence_count": len(citations)},
            )
            planner_response = self.backend.generate(
                BackendRequest(
                    task="replan",
                    prompt=_replanning_prompt(
                        question=question,
                        scene_index=self.scene_index,
                        ledger_text=ledger_text,
                        round_number=round_number,
                        budget=self.budget,
                        inspected_segment_ids=sorted(inspected_segment_ids),
                    ),
                    media_path=video_path if self.budget.planner_receives_media else None,
                    media_type="video" if self.budget.planner_receives_media else None,
                    max_new_tokens=768,
                    metadata={
                        "round": round_number,
                        "segment_count": len(self.scene_index.segments),
                        "planner_input_mode": "video" if self.budget.planner_receives_media else "text-only",
                    },
                )
            )
            action = _parse_replan_action(planner_response.text)
            status = str(action.get("status", "continue"))
            rationale = str(action.get("rationale", ""))

            if status == "final":
                final_citations = [str(item) for item in action.get("citations", [])]
                result_round = IterativeRound(
                    round_number=round_number,
                    status="final",
                    planner_text=planner_response.text,
                    rationale=rationale,
                )
                rounds.append(result_round)
                self.workspace.write_trace_event(
                    "iterative_final",
                    {
                        "round": round_number,
                        "answer": str(action.get("answer", "")),
                        "citations": final_citations,
                    },
                )
                return IterativeRunResult(
                    question=question,
                    video_path=video_path,
                    answer=str(action.get("answer", "")),
                    status="final",
                    citations=final_citations,
                    confidence=float(action.get("confidence", 0.0)),
                    rounds=rounds,
                )

            program = self._normalize_program(
                action.get("program", []),
                question=question,
                video_path=video_path,
                inspected_segment_ids=inspected_segment_ids,
            )
            self.workspace.write_trace_event(
                "iterative_plan",
                {"round": round_number, "rationale": rationale, "program": program},
            )
            program_result = ProgramInterpreter(registry=self.registry, workspace=self.workspace).run(program)
            observation_ids = [str(observation_id) for observation_id in program_result.observation_ids]
            citations.extend(observation_ids)
            inspected_segment_ids.update(_segment_ids_from_program(program))
            rounds.append(
                IterativeRound(
                    round_number=round_number,
                    status="continue",
                    planner_text=planner_response.text,
                    rationale=rationale,
                    program=program,
                    observation_ids=observation_ids,
                )
            )

        self.workspace.write_trace_event(
            "iterative_budget_exhausted",
            {"max_rounds": self.budget.max_rounds, "citations": citations},
        )
        plural = "round" if self.budget.max_rounds == 1 else "rounds"
        return IterativeRunResult(
            question=question,
            video_path=video_path,
            answer=f"Stopped after {self.budget.max_rounds} exploration {plural} with partial evidence.",
            status="max_rounds_reached",
            citations=citations,
            rounds=rounds,
        )

    def _normalize_program(
        self,
        program: Any,
        *,
        question: str,
        video_path: str,
        inspected_segment_ids: set[str],
    ) -> Sequence[Mapping[str, Any]]:
        if not isinstance(program, list):
            raise ValueError("Planner action status=continue requires a list program")

        normalized = []
        reserved_segment_ids = set(inspected_segment_ids)
        for step in program:
            if not isinstance(step, Mapping):
                raise ValueError("Planner program steps must be objects")
            if "tool" not in step:
                raise ValueError("Planner program step is missing required 'tool'")
            if len(normalized) >= self.budget.max_tool_calls_per_round:
                break

            tool_name = str(step["tool"])
            args = dict(step.get("args", {}))
            segment_id = args.get("segment_id")
            if segment_id:
                segment = self.scene_index.get(str(segment_id))
                if tool_name == "read_segment" and not _segment_has_index_text(segment):
                    self.workspace.write_trace_event(
                        "exploration_policy_adjustment",
                        {
                            "reason": "upgrade_empty_read_segment_to_caption",
                            "requested_tool": "read_segment",
                            "resolved_tool": "caption_segment",
                            "segment_id": segment.segment_id,
                        },
                    )
                    tool_name = "caption_segment"

            if segment_id and tool_name in _SEGMENT_MEDIA_TOOLS:
                resolved_segment_id = self._resolve_next_segment_id(str(segment_id), reserved_segment_ids)
                if resolved_segment_id is None:
                    continue
                if resolved_segment_id != str(segment_id):
                    self.workspace.write_trace_event(
                        "exploration_policy_adjustment",
                        {
                            "reason": "avoid_repeated_segment",
                            "requested_segment_id": str(segment_id),
                            "resolved_segment_id": resolved_segment_id,
                        },
                    )
                segment = self.scene_index.get(resolved_segment_id)
                args["segment_id"] = segment.segment_id
                args["video_path"] = video_path
                args["start_sec"] = segment.start_sec
                args["end_sec"] = segment.end_sec
                args.setdefault("question", question)
                args.setdefault("nframes", self.budget.default_nframes)
                reserved_segment_ids.add(segment.segment_id)
            elif segment_id:
                self.scene_index.get(str(segment_id))

            normalized_step: dict[str, Any] = {"tool": tool_name, "args": args}
            if "assign" in step:
                normalized_step["assign"] = str(step["assign"])
            normalized.append(normalized_step)

        if not normalized:
            fallback_segment_id = self._resolve_next_segment_id("", reserved_segment_ids)
            if fallback_segment_id is not None:
                segment = self.scene_index.get(fallback_segment_id)
                normalized.append(
                    {
                        "tool": "caption_segment",
                        "args": {
                            "video_path": video_path,
                            "segment_id": segment.segment_id,
                            "start_sec": segment.start_sec,
                            "end_sec": segment.end_sec,
                            "question": question,
                            "nframes": self.budget.default_nframes,
                        },
                        "assign": f"auto_{segment.segment_id}",
                    }
                )
        return normalized

    def _resolve_next_segment_id(self, requested_segment_id: str, reserved_segment_ids: set[str]) -> Optional[str]:
        if requested_segment_id and requested_segment_id not in reserved_segment_ids:
            self.scene_index.get(requested_segment_id)
            return requested_segment_id
        for segment in self.scene_index.segments:
            if segment.segment_id not in reserved_segment_ids:
                return segment.segment_id
        return None

    def _read_ledger(self) -> str:
        ledger_path = self.workspace.root / "ledger.md"
        if not ledger_path.exists():
            return ""
        return ledger_path.read_text(encoding="utf-8")


def _replanning_prompt(
    *,
    question: str,
    scene_index: SceneIndex,
    ledger_text: str,
    round_number: int,
    budget: AgentBudget,
    inspected_segment_ids: Sequence[str] = (),
) -> str:
    inspected_line = ", ".join(inspected_segment_ids) if inspected_segment_ids else "(none)"
    uninspected_line = _uninspected_segment_summary(scene_index=scene_index, inspected_segment_ids=inspected_segment_ids)
    return (
        "You are an autonomous visual agent exploring a long video with tools.\n"
        "Planner input mode: text-only. Use the scene index and evidence ledger; tools inspect pixels/video.\n"
        "Use coarse-to-fine search: inspect promising segments, read the evidence ledger, then either continue or answer.\n"
        "Available tools:\n"
        "- video_ls(query: str = '', max_segments: int = 16, top_k: int = 5)\n"
        "- search_segments(query: str, top_k: int = 5, modalities: list = [])\n"
        "- read_segment(segment_id: str)\n"
        "- expand_window(segment_id: str, before_sec: float = 30.0, after_sec: float = 30.0)\n"
        "- caption_segment(video_path: str, segment_id: str, start_sec: float, end_sec: float, question: str, nframes: int = 8)\n"
        "- qa_segment(video_path: str, segment_id: str, start_sec: float, end_sec: float, question: str, nframes: int = 8)\n"
        "Return only JSON with one of these schemas:\n"
        '{"status": "continue", "rationale": string, "program": [{"tool": string, "args": object, "assign": string}]}\n'
        '{"status": "final", "answer": string, "citations": [observation_id], "confidence": number}\n'
        "Rules:\n"
        "- Use video_ls first for open-ended description tasks or when the relevant segment is unclear.\n"
        "- Use navigation output as a map, then call caption_segment or qa_segment on one candidate segment.\n"
        "- Prefer segment_id references; the harness binds video_path/start_sec/end_sec.\n"
        f"- Request at most {budget.max_tool_calls_per_round} new tool call(s) this round.\n"
        "- Do not repeat already inspected segments unless the ledger says the prior observation was unusable.\n"
        "- Continue when evidence is missing, ambiguous, or too coarse.\n"
        "- Final answers must cite observation ids from the ledger.\n"
        f"Round: {round_number}/{budget.max_rounds}\n"
        f"Question: {question}\n"
        f"Already inspected segments: {inspected_line}\n"
        f"Uninspected segment candidates: {uninspected_line}\n"
        "Scene index:\n"
        f"{scene_index.summary(max_segments=64)}\n"
        "Evidence ledger:\n"
        f"{ledger_text}"
    )


def _parse_replan_action(text: str) -> Mapping[str, Any]:
    payload = json.loads(_extract_json_object(text))
    if not isinstance(payload, Mapping):
        raise ValueError("Planner response must be a JSON object")
    if "status" not in payload:
        return {"status": "continue", "program": payload.get("program", []), "rationale": payload.get("rationale", "")}
    return payload


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    return stripped[start : end + 1]


def _segment_ids_from_program(program: Sequence[Mapping[str, Any]]) -> Sequence[str]:
    segment_ids = []
    for step in program:
        if step.get("tool") not in _SEGMENT_MEDIA_TOOLS:
            continue
        args = step.get("args", {})
        if isinstance(args, Mapping) and args.get("segment_id"):
            segment_ids.append(str(args["segment_id"]))
    return segment_ids


def _segment_has_index_text(segment: Any) -> bool:
    return bool(getattr(segment, "low_fps_caption", ""))


def _uninspected_segment_summary(*, scene_index: SceneIndex, inspected_segment_ids: Sequence[str], limit: int = 12) -> str:
    inspected = set(inspected_segment_ids)
    candidates = [segment.segment_id for segment in scene_index.segments if segment.segment_id not in inspected]
    if not candidates:
        return "(none)"
    visible = candidates[:limit]
    suffix = f", ... {len(candidates) - limit} more" if len(candidates) > limit else ""
    return ", ".join(visible) + suffix
