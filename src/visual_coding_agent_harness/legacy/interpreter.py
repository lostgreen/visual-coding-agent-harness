"""Visual program interpreter.

The interpreter intentionally mirrors the useful part of VisProg: a program is
an ordered list of module calls. The difference is that every call also writes a
coding-agent-style trace and evidence ledger entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from string import Formatter
from typing import Any, Callable, Dict, Mapping, Sequence

from ..core.contracts import resolve_nframes
from ..core.registry import ToolRegistry
from ..workspace import EvidenceRecord, EvidenceWorkspace, MapUpdateProposal, Observation
from ..workspace.distill import distill
from ..workspace.output_quality import DEGENERATE_CONFIDENCE_SIGNAL, is_degenerate


@dataclass(frozen=True)
class ProgramResult:
    observation_ids: Sequence[str]
    assignments: Mapping[str, str] = field(default_factory=dict)
    stopped_by_sufficiency: bool = False
    rejections: Sequence[Mapping[str, Any]] = field(default_factory=tuple)


class ProgramInterpreter:
    """Run a simple visual program against a tool registry and workspace."""

    def __init__(
        self,
        registry: ToolRegistry,
        workspace: EvidenceWorkspace,
        *,
        lifecycle_context: Any | None = None,
        pre_tool_hooks: Sequence[Any] = (),
        post_tool_hooks: Sequence[Any] = (),
    ) -> None:
        self.registry = registry
        self.workspace = workspace
        self.lifecycle_context = lifecycle_context
        self.pre_tool_hooks = tuple(pre_tool_hooks)
        self.post_tool_hooks = tuple(post_tool_hooks)

    def run(
        self,
        program: Sequence[Mapping[str, Any]],
        *,
        slots: Mapping[str, Any] | None = None,
        sufficiency_predicate: Callable[[EvidenceWorkspace, Mapping[str, str]], bool] | None = None,
    ) -> ProgramResult:
        observation_ids = []
        rejections = []
        assignments: Dict[str, str] = {}
        slot_values = dict(slots or {})
        stopped_by_sufficiency = False

        for index, step in enumerate(program, start=1):
            expanded_steps = _expand_foreach_step(step, slot_values)
            for expanded_step in expanded_steps:
                step_observation_ids = self._run_step(
                    step_index=index,
                    step=expanded_step,
                    assignments=assignments,
                    rejections=rejections,
                )
                if not step_observation_ids:
                    continue
                for observation_id in step_observation_ids:
                    observation_ids.append(observation_id)
                    slot_updates = _dynamic_slot_updates_from_observation(self.workspace.get_observation(observation_id))
                    if slot_updates:
                        slot_values.update(slot_updates)
                        self.workspace.write_trace_event(
                            "foreach_slot_update",
                            {
                                "step": index,
                                "observation_id": observation_id,
                                "slots": {key: len(value) for key, value in slot_updates.items()},
                            },
                        )
                    if sufficiency_predicate is None:
                        continue
                    if sufficiency_predicate(self.workspace, assignments):
                        stopped_by_sufficiency = True
                        self.workspace.write_trace_event(
                            "sufficiency_stop",
                            {
                                "step": index,
                                "observation_id": observation_id,
                                "assignments": dict(assignments),
                            },
                        )
                        return ProgramResult(
                            observation_ids=observation_ids,
                            assignments=assignments,
                            stopped_by_sufficiency=stopped_by_sufficiency,
                            rejections=tuple(rejections),
                        )

        return ProgramResult(
            observation_ids=observation_ids,
            assignments=assignments,
            stopped_by_sufficiency=stopped_by_sufficiency,
            rejections=tuple(rejections),
        )

    def _run_step(
        self,
        *,
        step_index: int,
        step: Mapping[str, Any],
        assignments: Dict[str, str],
        rejections: list[Mapping[str, Any]],
    ) -> Sequence[str]:
        if "tool" not in step and "op" not in step:
            raise ValueError(f"Program step {step_index} is missing required 'tool'")
        tool_name = str(step.get("tool") or step.get("op"))
        arguments = dict(step.get("args", {}))
        self.workspace.write_trace_event(
            "tool_use",
            {"step": step_index, "tool": tool_name, "arguments": arguments},
        )
        raw_output = dict(self.registry.execute(tool_name, arguments))
        is_bad_output, fingerprint = is_degenerate(str(raw_output.get("claim", "")))
        if is_bad_output:
            raw_output["confidence_signal"] = DEGENERATE_CONFIDENCE_SIGNAL
            raw_output["grounding_quality"] = "weak"
            self.workspace.write_trace_event(
                "tool_output_degenerate",
                {
                    "step": step_index,
                    "tool": tool_name,
                    "fingerprint": fingerprint,
                },
            )
        return self._write_tool_observation(
            step=step,
            step_index=step_index,
            tool_name=tool_name,
            arguments=arguments,
            raw_output=raw_output,
            assignments=assignments,
        )

    def _write_tool_observation(
        self,
        *,
        step: Mapping[str, Any],
        step_index: int,
        tool_name: str,
        arguments: Mapping[str, Any],
        raw_output: Mapping[str, Any],
        assignments: Dict[str, str],
    ) -> Sequence[str]:
        observation = self.workspace.write_observation(
            tool_name=tool_name,
            input_artifacts=raw_output.get("input_artifacts", []),
            claim=str(raw_output.get("claim", "")),
            confidence=float(raw_output.get("confidence", 0.0)),
            regions=raw_output.get("regions", []),
            limitations=str(raw_output.get("limitations", "")),
            confidence_signal=str(raw_output.get("confidence_signal", "")),
            raw_output=raw_output,
        )
        observation = self._attach_visual_manifest(
            tool_name=tool_name,
            arguments=arguments,
            raw_output=raw_output,
            observation=observation,
        )
        self.workspace.write_trace_event(
            "tool_result",
            {
                "step": step_index,
                "tool": tool_name,
                "observation_id": observation.observation_id,
            },
        )
        effects = ()
        self._write_answer_evidence_rows(observation=observation, raw_output=raw_output)
        distilled_records = distill(observation, self.workspace)
        for evidence_record in distilled_records:
            self.workspace.write_evidence(evidence_record)
        self._write_map_proposals(observation=observation, parent_records=distilled_records)
        self.workspace.write_ledger_entry(observation, parent_records=distilled_records)
        self.workspace.append_timeline_from_observation(observation)

        if "assign" in step:
            assignments[str(step["assign"])] = observation.observation_id
        observation_ids = [observation.observation_id]
        for effect in effects:
            for observation_id in effect.observation_ids:
                observation_id_text = str(observation_id).strip()
                if observation_id_text and observation_id_text not in observation_ids:
                    observation_ids.append(observation_id_text)
        return observation_ids

    def _write_answer_evidence_rows(self, *, observation: Observation, raw_output: Mapping[str, Any]) -> None:
        rows = raw_output.get("answer_evidence_rows", [])
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            return
        written = 0
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, Mapping):
                continue
            if row.get("_workspace_promoted"):
                continue
            payload = dict(row)
            payload.setdefault("obs_id", observation.observation_id)
            payload.setdefault("observation_id", observation.observation_id)
            payload.setdefault("source_observation_id", observation.observation_id)
            payload.setdefault("evidence_id", f"ev_answer_{observation.observation_id}_{index:02d}")
            relations = payload.get("candidate_option_relations", [])
            if isinstance(relations, Sequence) and not isinstance(relations, (str, bytes)):
                resolved_relations = []
                for relation in relations:
                    if not isinstance(relation, Mapping):
                        continue
                    resolved = dict(relation)
                    resolved.setdefault("observation_id", observation.observation_id)
                    resolved_relations.append(resolved)
                payload["candidate_option_relations"] = resolved_relations
            self.workspace.write_evidence_row(payload)
            written += 1
        if written:
            self.workspace.write_trace_event(
                "answer_evidence_rows_promoted",
                {
                    "tool": observation.tool,
                    "observation_id": observation.observation_id,
                    "row_count": written,
                },
            )

    def _attach_visual_manifest(
        self,
        *,
        tool_name: str,
        arguments: Mapping[str, Any],
        raw_output: Mapping[str, Any],
        observation: Observation,
    ) -> Observation:
        if tool_name not in EvidenceWorkspace.VISUAL_EVIDENCE_TOOLS:
            return observation
        manifest_args = _visual_manifest_args(
            tool_name=tool_name,
            arguments=arguments,
            raw_output=raw_output,
            observation=observation,
        )
        if manifest_args is None:
            return observation
        manifest = self.workspace.create_manifest(**manifest_args)
        self.workspace.link_manifest(observation.observation_id, manifest.frame_set_id)
        return self.workspace.get_observation(observation.observation_id) or observation

    def _write_map_proposals(self, *, observation: Observation, parent_records: Sequence[EvidenceRecord]) -> None:
        if observation.tool not in {"query_context", "vision_read", "inspect_segment", "caption_segment", "qa_segment"}:
            return
        source = parent_records[0] if parent_records else None
        if source is None or not source.frame_set_id:
            return
        for region in _proposal_regions(observation):
            segment_id = _optional_str(region.get("segment_id"))
            if not segment_id:
                continue
            payload = _proposal_payload(observation=observation, region=region)
            if not payload:
                continue
            proposal = MapUpdateProposal(
                proposal_id=self.workspace.next_proposal_id(),
                target_segment_id=segment_id,
                update_type="context_update",
                payload=payload,
                source_evidence_id=source.evidence_id,
                source_frame_set_id=source.frame_set_id,
                confidence=float(observation.confidence),
                proposed_at=_now_seconds(),
            )
            self.workspace.write_proposal(proposal)
            self.workspace.write_trace_event(
                "map_proposal_created",
                {
                    "proposal_id": proposal.proposal_id,
                    "target_segment_id": proposal.target_segment_id,
                    "source_evidence_id": proposal.source_evidence_id,
                    "source_frame_set_id": proposal.source_frame_set_id,
                    "tool": observation.tool,
                },
            )


def _expand_foreach_step(step: Mapping[str, Any], slots: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    foreach = step.get("foreach")
    if foreach is None:
        return [dict(step)]
    slot_name = str(foreach)
    values = slots.get(slot_name, [])
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"foreach slot '{slot_name}' must be a sequence")
    loop_name = str(step.get("as") or _singular_slot_name(slot_name))
    expanded = []
    for item in values:
        context = dict(slots)
        context[loop_name] = item
        context[f"{slot_name}_item"] = item
        expanded_step = {
            key: _format_value(value, context)
            for key, value in step.items()
            if key not in {"foreach", "as"}
        }
        expanded.append(expanded_step)
    return expanded


def _format_value(value: Any, context: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        return _format_template(value, context)
    if isinstance(value, Mapping):
        return {key: _format_value(child, context) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_format_value(child, context) for child in value]
    return value


def _visual_manifest_args(
    *,
    tool_name: str,
    arguments: Mapping[str, Any],
    raw_output: Mapping[str, Any],
    observation: Observation,
) -> dict[str, Any] | None:
    video_path = str(arguments.get("video_path") or raw_output.get("video_path") or "")
    if not video_path:
        return None

    region = _first_region(raw_output)
    time_range = _time_range(raw_output)
    segment_id = _optional_str(arguments.get("segment_id") or region.get("segment_id"))
    if tool_name in {"global_gist", "query_context"}:
        segment_id = None
        start_sec = 0.0
        end_sec = _float_or_none(arguments.get("duration_sec"))
        if end_sec is None and time_range is not None:
            end_sec = time_range[1]
    else:
        start_sec = _float_or_none(arguments.get("start_sec"))
        end_sec = _float_or_none(arguments.get("end_sec"))
        if start_sec is None:
            start_sec = _float_or_none(region.get("start_sec"))
        if end_sec is None:
            end_sec = _float_or_none(region.get("end_sec"))
        if (start_sec is None or end_sec is None) and time_range is not None:
            start_sec = time_range[0] if start_sec is None else start_sec
            end_sec = time_range[1] if end_sec is None else end_sec
    if start_sec is None or end_sec is None:
        return None

    requested = arguments.get("nframes") if "nframes" in arguments else None
    requested_nframes = None if requested is None or requested == "" else int(requested)
    target_nframes, budget_reason = resolve_nframes(requested_nframes)
    actual_nframes = _first_int(
        raw_output.get("nframes"),
        region.get("nframes"),
        _nested_raw_output(raw_output).get("nframes"),
        default=target_nframes,
    )
    sampling_policy = "fps" if _float_or_none(arguments.get("fps") or region.get("fps")) else "uniform"
    return {
        "video_path": video_path,
        "segment_id": segment_id,
        "start_sec": float(start_sec),
        "end_sec": float(end_sec),
        "target_nframes": int(target_nframes),
        "nframes": int(actual_nframes),
        "sampling_policy": sampling_policy,
        "frame_times_sec": _uniform_frame_times(float(start_sec), float(end_sec), int(actual_nframes)),
        "frame_times_approximate": True,
        "created_by_tool": tool_name,
        "observation_id": observation.observation_id,
        "budget_reason": budget_reason,
        "materialized_paths": _materialized_paths(observation.input_artifacts),
    }


def _proposal_regions(observation: Observation) -> list[Mapping[str, Any]]:
    return [region for region in observation.regions if isinstance(region, Mapping)]


def _proposal_payload(*, observation: Observation, region: Mapping[str, Any]) -> dict[str, Any]:
    claim = " ".join(str(observation.claim).split())
    if not claim:
        return {}
    payload: dict[str, Any] = {
        "low_fps_caption": claim,
        "source_tool": observation.tool,
    }
    start_sec = _float_or_none(region.get("start_sec"))
    end_sec = _float_or_none(region.get("end_sec"))
    if start_sec is not None and end_sec is not None:
        payload["time_range"] = [start_sec, end_sec]
    return payload


def _now_seconds() -> float:
    import time

    return time.time()


def _first_region(raw_output: Mapping[str, Any]) -> Mapping[str, Any]:
    regions = raw_output.get("regions")
    if isinstance(regions, Sequence) and not isinstance(regions, (str, bytes)):
        for region in regions:
            if isinstance(region, Mapping):
                return region
    return {}


def _time_range(raw_output: Mapping[str, Any]) -> tuple[float, float] | None:
    for value in (raw_output.get("time_range"), _nested_raw_output(raw_output).get("time_range")):
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
            start = _float_or_none(value[0])
            end = _float_or_none(value[1])
            if start is not None and end is not None:
                return (start, end)
    return None


def _nested_raw_output(raw_output: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = raw_output.get("raw_output")
    return nested if isinstance(nested, Mapping) else {}


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_int(*values: Any, default: int) -> int:
    for value in values:
        if value is None or value == "":
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return int(default)


def _uniform_frame_times(start_sec: float, end_sec: float, nframes: int) -> list[float]:
    if nframes <= 0:
        return []
    if nframes == 1:
        return [float(start_sec)]
    step = (float(end_sec) - float(start_sec)) / float(nframes - 1)
    return [float(start_sec) + step * index for index in range(nframes)]


def _materialized_paths(input_artifacts: Sequence[str]) -> list[str]:
    return [str(artifact) for artifact in input_artifacts if artifact and "#t=" not in str(artifact)]


def _dynamic_slot_updates_from_observation(observation: Observation | None) -> dict[str, list[Any]]:
    if observation is None:
        return {}
    raw_output = observation.raw_output
    if not isinstance(raw_output, Mapping):
        return {}

    updates: dict[str, list[Any]] = {}
    candidates = _collection_slot_values(raw_output.get("candidates"))
    if not candidates:
        candidates = _collection_slot_values(raw_output.get("regions"))
    if candidates:
        updates["candidates"] = candidates

    segments = _collection_slot_values(raw_output.get("segments"))
    if segments:
        updates["segments"] = segments
    return updates


def _collection_slot_values(value: Any) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    values = []
    for item in value:
        if isinstance(item, Mapping):
            values.append(dict(item))
        elif item is not None:
            values.append(item)
    return values


def _format_template(template: str, context: Mapping[str, Any]) -> Any:
    needed = [field_name for _, field_name, _, _ in Formatter().parse(template) if field_name]
    if not needed:
        return template
    if template == "{" + needed[0] + "}" and len(needed) == 1:
        return context.get(needed[0], template)
    values = {name: context.get(name, "{" + name + "}") for name in needed}
    return template.format(**values)


def _singular_slot_name(slot_name: str) -> str:
    if slot_name.endswith("ies") and len(slot_name) > 3:
        return f"{slot_name[:-3]}y"
    if slot_name.endswith("s") and len(slot_name) > 1:
        return slot_name[:-1]
    return "item"
