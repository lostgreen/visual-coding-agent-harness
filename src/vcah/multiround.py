from __future__ import annotations

from dataclasses import dataclass, replace
import json
import re
from typing import Any, Mapping, Sequence

from vcah.investigator import InvestigationReport, ObservationAttempt, VirtualVideoInvestigator
from vcah.memory import EvidenceStore
from vcah.types import EvidenceRecord, to_jsonable
from vcah.virtual_index import build_workspace_overview
from vcah.virtual_video import VirtualVideoWorkspace, select_uniform_items
from vcah.workspace import (
    AnswerValidation,
    ObservationLog,
    WorkingDocument,
    append_workspace_history,
    evidence_attempt_id,
    render_working_view,
)


_INSPECTION_MODES = {"window", "search_asr", "arbitrate_observation"}
RUN_ARTIFACT_NAMES = (
    "evidence.jsonl",
    "observation_log.jsonl",
    "working_document.json",
    "workspace_ops.jsonl",
    "exploration_ledger.jsonl",
    "run_summary.json",
)


@dataclass(frozen=True)
class InvestigationTask:
    query_id: str
    goal: str
    segment_id: str = ""
    time_range: tuple[float, float] | None = None
    expected_evidence: str = ""
    inspection_mode: str = "window"
    search_terms: tuple[str, ...] = ()
    sampling_floor_fps: float | None = None
    arbitration_attempt_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", str(self.query_id or "").strip())
        object.__setattr__(self, "goal", str(self.goal or "").strip())
        object.__setattr__(self, "segment_id", str(self.segment_id or "").strip())
        object.__setattr__(self, "time_range", _time_range(self.time_range))
        object.__setattr__(self, "expected_evidence", str(self.expected_evidence or "").strip())
        mode = str(self.inspection_mode or "window").strip().casefold()
        if mode not in _INSPECTION_MODES:
            raise ValueError(f"unsupported inspection_mode: {mode}")
        object.__setattr__(self, "inspection_mode", mode)
        object.__setattr__(
            self,
            "search_terms",
            tuple(dict.fromkeys(str(item).strip().casefold() for item in self.search_terms if str(item).strip())),
        )
        object.__setattr__(
            self,
            "sampling_floor_fps",
            min(2.0, max(0.5, float(self.sampling_floor_fps or 0.5))),
        )
        object.__setattr__(self, "arbitration_attempt_id", str(self.arbitration_attempt_id or "").strip())


@dataclass(frozen=True)
class ReasonerDecision:
    action: str
    tasks: tuple[InvestigationTask, ...] = ()
    answer: str = ""
    citations: tuple[str, ...] = ()
    workspace_ops: tuple[Mapping[str, Any], ...] = ()
    supporting_claim_ids: tuple[str, ...] = ()
    residual_uncertainty: str = ""
    observation_requests: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        action = str(self.action or "").strip().casefold()
        if action not in {"investigate", "read_observations", "update_workspace", "answer"}:
            raise ValueError(f"unsupported reasoner action: {action or 'missing'}")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "tasks", tuple(_task(item) for item in self.tasks))
        object.__setattr__(self, "answer", str(self.answer or "").strip())
        object.__setattr__(
            self,
            "citations",
            tuple(dict.fromkeys(str(item).strip() for item in self.citations if str(item).strip())),
        )
        object.__setattr__(
            self,
            "workspace_ops",
            tuple(dict(item) for item in self.workspace_ops if isinstance(item, Mapping)),
        )
        object.__setattr__(
            self,
            "supporting_claim_ids",
            tuple(dict.fromkeys(str(item).strip() for item in self.supporting_claim_ids if str(item).strip())),
        )
        object.__setattr__(self, "residual_uncertainty", str(self.residual_uncertainty or "").strip())
        object.__setattr__(
            self,
            "observation_requests",
            tuple(dict(item) for item in self.observation_requests if isinstance(item, Mapping)),
        )


@dataclass(frozen=True)
class MultiRoundResult:
    case_id: str
    answer: str
    selected_option: str
    citations: tuple[str, ...]
    correct: bool
    reference_valid: bool
    reference_reason: str
    rounds: int
    investigation_count: int
    evidence: tuple[EvidenceRecord, ...]
    reports: tuple[InvestigationReport, ...]
    trace: tuple[Mapping[str, Any], ...] = ()


class VirtualVideoMultiRoundDriver:
    def __init__(
        self,
        *,
        reasoner: Any,
        investigator: VirtualVideoInvestigator | None = None,
        max_rounds: int = 4,
        max_investigations: int = 20,
        max_tasks_per_round: int = 4,
    ) -> None:
        if reasoner is None:
            raise ValueError("VirtualVideoMultiRoundDriver requires a Reasoner")
        if investigator is None:
            raise ValueError("VirtualVideoMultiRoundDriver requires an Investigator")
        self.reasoner = reasoner
        self.investigator = investigator
        self.max_rounds = max(1, int(max_rounds))
        self.max_investigations = max(1, int(max_investigations))
        self.max_tasks_per_round = max(1, int(max_tasks_per_round))

    def run(self, workspace: VirtualVideoWorkspace) -> MultiRoundResult:
        existing = tuple(name for name in RUN_ARTIFACT_NAMES if (workspace.root_dir / name).exists())
        if existing:
            raise FileExistsError(f"workspace already contains run artifacts: {', '.join(existing)}")
        investigator = self.investigator
        investigator.reset_run_state()
        overview = build_workspace_overview(workspace, thumbnail_budget=40)
        evidence_store = EvidenceStore.empty(workspace.root_dir / "evidence.jsonl")
        observation_log = ObservationLog(workspace.root_dir / "observation_log.jsonl")
        document = WorkingDocument.with_question_premise(workspace.case.question)
        document_path = workspace.root_dir / "working_document.json"
        history_path = workspace.root_dir / "workspace_ops.jsonl"
        history_path.touch(exist_ok=False)
        document.save(document_path)

        reports: list[InvestigationReport] = []
        trace: list[Mapping[str, Any]] = []
        completed_investigations = 0
        rounds_run = 0
        requested_observations: tuple[Mapping[str, Any], ...] = ()
        feedback: dict[str, Any] = {}
        last_answer: ReasonerDecision | None = None
        final_answer: ReasonerDecision | None = None

        for round_id in range(1, self.max_rounds + 2):
            rounds_run = round_id
            remaining = max(0, self.max_investigations - completed_investigations)
            force_finalize = round_id > self.max_rounds or remaining <= 0
            status = _mechanical_status(workspace, document, observation_log)
            decision = _decision(
                self.reasoner.decide(
                    question=workspace.case.question,
                    options=dict(workspace.case.options),
                    workspace_overview=overview,
                    working_document_view=render_working_view(
                        document,
                        observation_log,
                        requested_observations=requested_observations,
                        feedback=feedback,
                    ),
                    mechanical_status=status,
                    remaining_budget=remaining,
                    force_finalize=force_finalize,
                )
            )

            apply_result = document.apply_ops(
                decision.workspace_ops,
                observation_ids=observation_log.attempt_ids,
            )
            if decision.workspace_ops:
                append_workspace_history(
                    history_path,
                    round_id=round_id,
                    operations=decision.workspace_ops,
                    result=apply_result,
                )
                if apply_result.accepted:
                    document.save(document_path)
            requested_rows = _read_observations(observation_log, decision.observation_requests)
            trace.append(
                {
                    "type": "reasoner_decision",
                    "round": round_id,
                    "action": decision.action,
                    "tasks": [_task_descriptor(task) for task in decision.tasks],
                    "workspace_revision": document.revision,
                    "workspace_ops_accepted": apply_result.accepted,
                    "workspace_errors": list(apply_result.errors),
                    "supporting_claim_ids": list(decision.supporting_claim_ids),
                    "remaining_budget": remaining,
                    "force_finalize": force_finalize,
                }
            )

            if not apply_result.accepted:
                feedback = {
                    "type": "workspace_ops_rejected",
                    "errors": list(apply_result.errors),
                    "revision": document.revision,
                }
                requested_observations = requested_rows
                if decision.action == "answer":
                    last_answer = decision
                if force_finalize:
                    break
                continue

            if decision.action == "answer":
                candidate = replace(
                    decision,
                    citations=_answer_citations(decision, document, evidence_store.records),
                )
                last_answer = candidate
                validation = _validate_answer(candidate, document, observation_log.attempt_ids, workspace.case.options)
                trace.append({"type": "reference_integrity_check", "round": round_id, **validation.to_dict()})
                if validation.passed:
                    final_answer = candidate
                    break
                feedback = {
                    "type": "answer_reference_rejected",
                    "reason": validation.reason,
                    "errors": list(validation.errors),
                    "supporting_claim_ids": list(candidate.supporting_claim_ids),
                }
                requested_observations = requested_rows
                if force_finalize:
                    break
                continue

            if decision.action in {"read_observations", "update_workspace"}:
                requested_observations = requested_rows
                feedback = {
                    "type": "workspace_action_applied",
                    "action": decision.action,
                    "returned_observation_count": len(requested_rows),
                    "revision": document.revision,
                }
                if force_finalize:
                    break
                continue

            if force_finalize:
                break

            tasks = _resolve_tasks(
                workspace,
                tuple(task for task in decision.tasks if _task_is_executable(task)),
                limit=min(self.max_tasks_per_round, remaining),
            )
            if decision.action != "investigate" or not tasks:
                feedback = {
                    "type": "task_validation",
                    "reason": "reasoner_tasks_not_executable",
                    "requested_task_count": len(decision.tasks),
                }
                requested_observations = requested_rows
                continue

            batch = investigator.run_batch(tasks)
            completed = sum(_report_completed(report) for report in batch)
            completed_investigations += completed
            reports.extend(batch)
            known_evidence_ids = {record.evidence_id for record in evidence_store.records}
            new_rows: list[Mapping[str, Any]] = []
            for report in batch:
                for record in report.evidence:
                    if record.evidence_id not in known_evidence_ids:
                        evidence_store.add(record)
                        known_evidence_ids.add(record.evidence_id)
                for attempt in report.attempts:
                    new_rows.append(
                        observation_log.append_attempt(
                            attempt,
                            round_id=round_id,
                            source_lineage=_attempt_lineage(attempt, report.evidence),
                        )
                    )
            requested_observations = tuple(new_rows[-12:]) or requested_rows
            feedback = {
                "type": "investigation_completed",
                "requested_tasks": len(tasks),
                "completed_tasks": completed,
                "new_observation_interpretations": len(new_rows),
                "outcomes": list(_outcome_digest(batch)),
            }
            trace.append(
                {
                    "type": "investigator_batch",
                    "round": round_id,
                    "requested_tasks": len(tasks),
                    "completed_tasks": completed,
                    "attempt_ids": [str(row["attempt_id"]) for row in new_rows],
                }
            )

        selected = final_answer or last_answer or ReasonerDecision(action="answer")
        selected_option = _letter(selected.answer) or _option_letter_from_answer(
            selected.answer,
            workspace.case.options,
        )
        if selected_option:
            answer = selected.answer
            validation = _validate_answer(selected, document, observation_log.attempt_ids, workspace.case.options)
            citations = _answer_citations(selected, document, evidence_store.records)
            reference_valid = validation.passed
            reference_reason = validation.reason
        else:
            answer = "No valid answer was returned."
            citations = ()
            reference_valid = False
            reference_reason = "answer_missing" if not selected.answer else "invalid_option_answer"

        trace.append(
            {
                "type": "answer_outcome",
                "answer": answer,
                "raw_reasoner_answer": selected.answer,
                "selected_option": selected_option,
                "reference_valid": reference_valid,
                "reference_reason": reference_reason,
                "supporting_claim_ids": list(selected.supporting_claim_ids),
                "residual_uncertainty": selected.residual_uncertainty,
                "answer_owner": "reasoner",
                "framework_answer_mutation": False,
                "working_document_path": str(document_path),
                "observation_log_path": str(observation_log.path),
                "workspace_history_path": str(history_path),
            }
        )
        result = MultiRoundResult(
            case_id=workspace.case.case_id,
            answer=answer,
            selected_option=selected_option,
            citations=citations,
            correct=_score_answer(answer, workspace.case.gold, workspace.case.options),
            reference_valid=reference_valid,
            reference_reason=reference_reason,
            rounds=rounds_run,
            investigation_count=completed_investigations,
            evidence=tuple(evidence_store.records),
            reports=tuple(reports),
            trace=tuple(trace),
        )
        _write_run_summary(workspace, result)
        return result


def _mechanical_status(
    workspace: VirtualVideoWorkspace,
    document: WorkingDocument,
    observations: ObservationLog,
) -> dict[str, Any]:
    errors = document.validate(observation_ids=observations.attempt_ids)
    coverage = _source_coverage(workspace, observations)
    return {
        "schema_version": "MechanicalCompletionStatusV1",
        "working_document_revision": document.revision,
        "workspace_valid": not errors,
        "workspace_errors": list(errors),
        "active_claim_count": sum(
            claim.status in {"active", "contested"}
            for claim in document.claims.values()
        ),
        "active_claim_limit": document.active_claim_limit,
        "observation_attempt_count": len(observations.attempt_ids),
        "observation_interpretation_count": len(observations.rows),
        "source_coverage": coverage,
        "missing_segment_ids": [
            segment_id
            for source in coverage.values()
            for segment_id in source["missing_segment_ids"]
        ],
        "answer_owner": "reasoner",
    }


def _source_coverage(
    workspace: VirtualVideoWorkspace,
    observations: ObservationLog,
) -> dict[str, Any]:
    inspected = tuple(
        normalized
        for row in observations.catalog_source_rows()
        if str(row.get("execution_status", "") or "") != "failed"
        and str(row.get("modality", "") or "") in {"visual", "ocr"}
        for raw in tuple(row.get("inspected_ranges", ()) or ())
        if (normalized := _time_range(raw)) is not None
    )
    by_source: dict[str, dict[str, Any]] = {}
    for segment in workspace.manifest.segments:
        source_id = str(segment.source_video_id)
        start = float(segment.virtual_start_sec)
        end = float(segment.virtual_end_sec)
        intersections = [
            (max(start, interval[0]), min(end, interval[1]))
            for interval in inspected
            if min(end, interval[1]) > max(start, interval[0])
        ]
        covered = sum(right - left for left, right in _merge_intervals(intersections))
        duration = max(0.0, end - start)
        ratio = covered / duration if duration else 1.0
        complete = ratio >= 0.98
        source = by_source.setdefault(
            source_id,
            {
                "duration_sec": 0.0,
                "covered_sec": 0.0,
                "coverage_ratio": 0.0,
                "covered_segment_ids": [],
                "missing_segment_ids": [],
                "segment_coverage": {},
            },
        )
        source["duration_sec"] += duration
        source["covered_sec"] += covered
        source["segment_coverage"][segment.segment_id] = {
            "coverage_ratio": round(ratio, 4),
            "covered_sec": round(covered, 3),
            "duration_sec": round(duration, 3),
        }
        source["covered_segment_ids" if complete else "missing_segment_ids"].append(segment.segment_id)
    for source in by_source.values():
        duration = float(source["duration_sec"] or 0.0)
        source["duration_sec"] = round(duration, 3)
        source["covered_sec"] = round(float(source["covered_sec"]), 3)
        source["coverage_ratio"] = round(float(source["covered_sec"]) / duration, 4) if duration else 1.0
    return by_source


def _resolve_tasks(
    workspace: VirtualVideoWorkspace,
    tasks: Sequence[InvestigationTask],
    *,
    limit: int,
) -> tuple[InvestigationTask, ...]:
    limit = max(0, int(limit))
    if not limit:
        return ()
    segments = tuple(workspace.manifest.segments)
    by_id = {segment.segment_id: segment for segment in segments}
    global_aliases = {"all", "full", "full_video", "global", "workspace"}
    groups: list[tuple[InvestigationTask, ...]] = []
    for task in tasks:
        if task.inspection_mode in {"search_asr", "arbitrate_observation"}:
            groups.append((task,))
            continue
        if task.time_range is not None:
            start, end = task.time_range
            overlaps = tuple(
                (segment, max(start, segment.virtual_start_sec), min(end, segment.virtual_end_sec))
                for segment in segments
                if min(end, segment.virtual_end_sec) > max(start, segment.virtual_start_sec)
            )
            if task.segment_id in by_id:
                segment = by_id[task.segment_id]
                if start >= segment.virtual_start_sec and end <= segment.virtual_end_sec:
                    groups.append((task,))
                    continue
            if overlaps:
                groups.append(
                    tuple(
                        replace(
                            task,
                            query_id=task.query_id if len(overlaps) == 1 else f"{task.query_id}_{index:02d}",
                            segment_id=segment.segment_id,
                            time_range=(overlap_start, overlap_end),
                        )
                        for index, (segment, overlap_start, overlap_end) in enumerate(overlaps, start=1)
                    )
                )
                continue
        if task.segment_id in by_id:
            groups.append((task,))
            continue
        if task.segment_id.casefold() not in global_aliases:
            continue
        selected = select_uniform_items(segments, min(limit, len(segments)))
        groups.append(
            tuple(
                replace(task, query_id=f"{task.query_id}_{segment.segment_id}", segment_id=segment.segment_id)
                for segment in selected
            )
        )
    resolved: list[InvestigationTask] = []
    depth = 0
    while len(resolved) < limit and any(depth < len(group) for group in groups):
        for group in groups:
            if len(resolved) >= limit:
                break
            if depth < len(group):
                resolved.append(group[depth])
        depth += 1
    return tuple(resolved)


def _task_is_executable(task: InvestigationTask) -> bool:
    if not task.query_id or not task.goal:
        return False
    if task.inspection_mode == "search_asr":
        return bool(task.search_terms)
    if task.inspection_mode == "arbitrate_observation":
        return bool(task.arbitration_attempt_id)
    return bool(task.segment_id or task.time_range)


def _validate_answer(
    decision: ReasonerDecision,
    document: WorkingDocument,
    observation_ids: Sequence[str],
    options: Mapping[str, str],
) -> AnswerValidation:
    validation = document.validate_answer(
        decision.supporting_claim_ids,
        observation_ids=observation_ids,
    )
    if _letter(decision.answer) or _option_letter_from_answer(decision.answer, options):
        return validation
    reason = "answer_missing" if not decision.answer else "invalid_option_answer"
    return AnswerValidation(
        False,
        reason,
        validation.supporting_claim_ids,
        validation.cited_attempt_ids,
        ("answer_must_select_exactly_one_option",),
    )


def _answer_citations(
    decision: ReasonerDecision,
    document: WorkingDocument,
    evidence: Sequence[EvidenceRecord],
) -> tuple[str, ...]:
    by_id = {record.evidence_id: record for record in evidence}
    by_attempt: dict[str, list[EvidenceRecord]] = {}
    for record in evidence:
        by_attempt.setdefault(evidence_attempt_id(record), []).append(record)
    citations = [citation for citation in decision.citations if citation in by_id]
    validation = document.validate_answer(
        decision.supporting_claim_ids,
        observation_ids=tuple(by_attempt),
    )
    citations.extend(
        record.evidence_id
        for attempt_id in validation.cited_attempt_ids
        for record in by_attempt.get(attempt_id, ())
    )
    return tuple(dict.fromkeys(citations))


def _read_observations(
    observations: ObservationLog,
    requests: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    for request in requests:
        attempt_ids = request.get("attempt_ids", ()) or ()
        if isinstance(attempt_ids, str):
            attempt_ids = (attempt_ids,)
        single = str(request.get("attempt_id", "") or "").strip()
        if single:
            attempt_ids = (*tuple(attempt_ids), single)
        raw_range = request.get("time_range")
        time_range = (
            raw_range
            if isinstance(raw_range, Sequence) and not isinstance(raw_range, (str, bytes))
            else None
        )
        rows.extend(
            observations.read(
                attempt_ids=tuple(str(item) for item in attempt_ids),
                time_range=time_range,
                max_entries=12,
            )
        )
    unique = {str(row.get("interpretation_id", "") or index): row for index, row in enumerate(rows)}
    return tuple(unique.values())[-12:]


def _attempt_lineage(
    attempt: ObservationAttempt,
    evidence: Sequence[EvidenceRecord],
) -> tuple[Mapping[str, Any], ...]:
    matching = tuple(record for record in evidence if evidence_attempt_id(record) == attempt.attempt_id)
    if not matching and evidence:
        matching = (evidence[0],)
    return tuple(dict(item) for record in matching for item in record.source_lineage)


def _outcome_digest(reports: Sequence[InvestigationReport]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "query_id": report.query_id,
            "status": report.status,
            "failure_reason": report.failure_reason,
            "attempt_ids": [attempt.attempt_id for attempt in report.attempts],
            "evidence_ids": [record.evidence_id for record in report.evidence],
        }
        for report in tuple(reports)[-12:]
    )


def _report_completed(report: InvestigationReport) -> int:
    if report.status == "failed" or report.cost.get("consumes_budget") is False:
        return 0
    return int(bool(report.evidence or report.attempts))


def _task_descriptor(task: InvestigationTask) -> dict[str, Any]:
    return {
        "query_id": task.query_id,
        "segment_id": task.segment_id,
        "time_range": list(task.time_range) if task.time_range else [],
        "inspection_mode": task.inspection_mode,
        "sampling_floor_fps": task.sampling_floor_fps,
        "arbitration_attempt_id": task.arbitration_attempt_id,
    }


def _decision(value: ReasonerDecision | Mapping[str, Any]) -> ReasonerDecision:
    if isinstance(value, ReasonerDecision):
        return value
    return ReasonerDecision(
        action=str(value.get("action", "") or ""),
        tasks=tuple(value.get("tasks", ()) or ()),
        answer=str(value.get("answer", "") or ""),
        citations=tuple(value.get("citations", ()) or ()),
        workspace_ops=tuple(value.get("workspace_ops", value.get("ops", ())) or ()),
        supporting_claim_ids=tuple(value.get("supporting_claim_ids", ()) or ()),
        residual_uncertainty=str(value.get("residual_uncertainty", "") or ""),
        observation_requests=tuple(value.get("observation_requests", ()) or ()),
    )


def _task(value: InvestigationTask | Mapping[str, Any]) -> InvestigationTask:
    if isinstance(value, InvestigationTask):
        return value
    return InvestigationTask(
        query_id=str(value.get("query_id", value.get("id", "")) or ""),
        goal=str(value.get("goal", value.get("task", "")) or ""),
        segment_id=str(value.get("segment_id", "") or ""),
        time_range=value.get("time_range"),
        expected_evidence=str(value.get("expected_evidence", "") or ""),
        inspection_mode=str(value.get("inspection_mode", "window") or "window"),
        search_terms=tuple(value.get("search_terms", ()) or ()),
        sampling_floor_fps=value.get("sampling_floor_fps"),
        arbitration_attempt_id=str(value.get("arbitration_attempt_id", "") or ""),
    )


def _time_range(value: Sequence[float] | None) -> tuple[float, float] | None:
    if value is None or len(value) != 2:
        return None
    start, end = sorted((float(value[0]), float(value[1])))
    return start, end


def _merge_intervals(intervals: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return tuple((start, end) for start, end in merged)


def _score_answer(answer: str, gold: str, options: Mapping[str, str] | None = None) -> bool:
    selected = _letter(answer) or _option_letter_from_answer(answer, options or {})
    expected = _letter(gold) or str(gold or "").strip().upper()[:1]
    return bool(selected and expected and selected == expected)


def _option_letter_from_answer(answer: str, options: Mapping[str, str]) -> str:
    normalized_answer = _answer_match_text(answer)
    if not normalized_answer:
        return ""
    direct = [
        str(label).upper()
        for label, text in options.items()
        if (normalized_option := _answer_match_text(text))
        and (normalized_option in normalized_answer or normalized_answer in normalized_option)
    ]
    if len(direct) == 1:
        return direct[0]
    answer_numbers = set(re.findall(r"\d+(?:\.\d+)?", str(answer or "")))
    numeric = [
        str(label).upper()
        for label, text in options.items()
        if (numbers := set(re.findall(r"\d+(?:\.\d+)?", str(text or ""))))
        and numbers.issubset(answer_numbers)
    ]
    return numeric[0] if len(numeric) == 1 else ""


def _answer_match_text(value: str) -> str:
    text = str(value or "").casefold()
    for source, target in {
        "kilometres": "km",
        "kilometers": "km",
        "kilometre": "km",
        "kilometer": "km",
        "metres": "m",
        "meters": "m",
        "metre": "m",
        "meter": "m",
    }.items():
        text = text.replace(source, target)
    return re.sub(r"[^a-z0-9]+", "", text)


def _letter(value: str) -> str:
    text = str(value or "").strip().upper()
    leading = re.match(r"^[\(\[]?([A-H])(?:[\)\].:\-]|\s*$)", text)
    if leading:
        return leading.group(1)
    explicit = re.search(
        r"\b(?:ANSWER|OPTION|CHOICE)\s*(?:IS\s*)?[:\-]?\s*[\(\[]?([A-H])(?:[\)\].:\-]|\b)",
        text,
    )
    return explicit.group(1) if explicit else ""


def _write_run_summary(workspace: VirtualVideoWorkspace, result: MultiRoundResult) -> None:
    payload = {
        "case_id": result.case_id,
        "answer": result.answer,
        "selected_option": result.selected_option,
        "citations": list(result.citations),
        "correct": result.correct,
        "reference_valid": result.reference_valid,
        "reference_reason": result.reference_reason,
        "rounds": result.rounds,
        "investigation_count": result.investigation_count,
        "evidence": [
            {
                "evidence_id": record.evidence_id,
                "attempt_id": evidence_attempt_id(record),
                "summary": record.verbatim,
                "modality": record.modality,
                "virtual_time_range": [record.start_sec, record.end_sec],
                "sampling_fps": record.sampling_fps,
                "pointer": record.pointer,
                "frame_refs": list(record.frame_refs),
                "coverage_manifest": to_jsonable(record.coverage_manifest),
                "source_lineage": [dict(item) for item in record.source_lineage],
            }
            for record in result.evidence
        ],
        "trace": [dict(item) for item in result.trace],
    }
    (workspace.root_dir / "run_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
