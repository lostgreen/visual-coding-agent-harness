from __future__ import annotations

from dataclasses import dataclass, replace
import json
import re
from typing import Any, Mapping, Sequence

from vcah.investigator import (
    INTERPRETATION_PURPOSES,
    InvestigationReport,
    ObservationAttempt,
    VirtualVideoInvestigator,
)
from vcah.memory import EvidenceStore
from vcah.runtime_metrics import export_supporting_intervals
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


_INSPECTION_MODES = {"window", "search_asr", "search_caption", "arbitrate_observation"}
_TIME_BOUNDARY_TOLERANCE_SEC = 1.0
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
    coordinate_space: str = "virtual"
    source_video_ids: tuple[str, ...] = ()
    conversion_trace: tuple[Mapping[str, Any], ...] = ()
    expected_evidence: str = ""
    inspection_mode: str = "window"
    search_terms: tuple[str, ...] = ()
    caption_queries: tuple[str, ...] = ()
    top_k: int = 12
    index_mode: str = "lexical"
    expand_neighbors: int = 0
    sampling_floor_fps: float | None = None
    arbitration_attempt_id: str = ""
    force_reinspect: bool = False
    interpretation_purpose: str = "primary"

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", str(self.query_id or "").strip())
        object.__setattr__(self, "goal", str(self.goal or "").strip())
        object.__setattr__(self, "segment_id", str(self.segment_id or "").strip())
        object.__setattr__(self, "time_range", _time_range(self.time_range))
        coordinate_space = str(self.coordinate_space or "virtual").strip().casefold()
        if coordinate_space not in {"virtual", "segment_local"}:
            raise ValueError(f"unsupported coordinate_space: {coordinate_space}")
        object.__setattr__(self, "coordinate_space", coordinate_space)
        object.__setattr__(
            self,
            "source_video_ids",
            tuple(
                dict.fromkeys(
                    str(item).strip()
                    for item in self.source_video_ids
                    if str(item).strip()
                )
            ),
        )
        object.__setattr__(
            self,
            "conversion_trace",
            tuple(dict(item) for item in self.conversion_trace if isinstance(item, Mapping)),
        )
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
            "caption_queries",
            tuple(dict.fromkeys(str(item).strip() for item in self.caption_queries if str(item).strip()))[:5],
        )
        object.__setattr__(self, "top_k", min(50, max(1, int(self.top_k))))
        index_mode = str(self.index_mode or "lexical").strip().casefold()
        if index_mode not in {"lexical", "dense", "hybrid"}:
            raise ValueError(f"unsupported caption index_mode: {index_mode}")
        object.__setattr__(self, "index_mode", index_mode)
        object.__setattr__(self, "expand_neighbors", min(3, max(0, int(self.expand_neighbors))))
        object.__setattr__(
            self,
            "sampling_floor_fps",
            min(2.0, max(0.5, float(self.sampling_floor_fps or 0.5))),
        )
        object.__setattr__(self, "arbitration_attempt_id", str(self.arbitration_attempt_id or "").strip())
        object.__setattr__(self, "force_reinspect", bool(self.force_reinspect))
        purpose = str(self.interpretation_purpose or "primary").strip().casefold()
        if mode == "arbitrate_observation":
            purpose = "deliberate_arbitration"
        if purpose not in INTERPRETATION_PURPOSES:
            raise ValueError(f"unsupported interpretation_purpose: {purpose}")
        object.__setattr__(self, "interpretation_purpose", purpose)


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
    correct: bool | None
    reference_valid: bool
    reference_reason: str
    rounds: int
    investigation_count: int
    evidence: tuple[EvidenceRecord, ...]
    reports: tuple[InvestigationReport, ...]
    trace: tuple[Mapping[str, Any], ...] = ()
    answer_policy: str = "strict"
    answer_present: bool = False
    candidate_answer: str = ""
    verified_answer: str = ""
    verification_status: str = "missing"
    blocking_reasons: tuple[str, ...] = ()
    supporting_claim_ids: tuple[str, ...] = ()
    supporting_intervals: tuple[tuple[float, float], ...] = ()
    residual_uncertainty: str = ""


class VirtualVideoMultiRoundDriver:
    def __init__(
        self,
        *,
        reasoner: Any,
        investigator: VirtualVideoInvestigator | None = None,
        max_rounds: int = 4,
        max_investigations: int = 20,
        max_tasks_per_round: int = 4,
        semantic_round_budget: int | None = None,
        control_retry_budget: int = 2,
        require_obligation_coverage: bool = False,
        answer_policy: str = "strict",
    ) -> None:
        if reasoner is None:
            raise ValueError("VirtualVideoMultiRoundDriver requires a Reasoner")
        if investigator is None:
            raise ValueError("VirtualVideoMultiRoundDriver requires an Investigator")
        self.reasoner = reasoner
        self.investigator = investigator
        semantic_budget = max_rounds if semantic_round_budget is None else semantic_round_budget
        self.semantic_round_budget = max(1, int(semantic_budget))
        self.max_rounds = self.semantic_round_budget
        self.control_retry_budget = max(0, int(control_retry_budget))
        self.require_obligation_coverage = bool(require_obligation_coverage)
        self.max_investigations = max(1, int(max_investigations))
        self.max_tasks_per_round = max(1, int(max_tasks_per_round))
        policy = str(answer_policy or "strict").strip().casefold()
        if policy not in {"strict", "benchmark_best_effort"}:
            raise ValueError(f"unsupported answer_policy: {policy}")
        self.answer_policy = policy

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
        final_answer: ReasonerDecision | None = None
        latest_answer_candidate: ReasonerDecision | None = None
        forced_decision_calls = 0
        protocol_exhausted = False

        for round_id in range(1, self.semantic_round_budget + 3):
            remaining = max(0, self.max_investigations - completed_investigations)
            runtime_status = (
                dict(investigator.mechanical_status())
                if callable(getattr(investigator, "mechanical_status", None))
                else {}
            )
            status = _mechanical_status(
                workspace,
                document,
                observation_log,
                runtime_status=runtime_status,
            )
            force_finalize = round_id > self.semantic_round_budget or remaining <= 0
            if force_finalize:
                forced_decision_calls += 1
            final_retry_available = force_finalize and forced_decision_calls < 2
            control_attempt = 0
            control_retries_used = 0
            decision_had_control_retry = False
            decision: ReasonerDecision | None = None
            requested_rows: tuple[Mapping[str, Any], ...] = ()
            while True:
                raw_decision = self.reasoner.decide(
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
                    final_attempt=forced_decision_calls if force_finalize else 0,
                    answer_policy=self.answer_policy,
                    semantic_round=round_id,
                    control_attempt=control_attempt,
                    control_retry=control_attempt > 0,
                    control_retries_remaining=max(
                        0,
                        self.control_retry_budget - control_retries_used,
                    ),
                )
                decision_metadata = _consume_decision_metadata(self.reasoner)
                internal_retries = max(
                    0,
                    int(decision_metadata.get("internal_control_retry_count", 0) or 0),
                )
                if internal_retries:
                    control_retries_used += internal_retries
                    decision_had_control_retry = True
                    trace.append(
                        {
                            "type": "control_retry",
                            "round": round_id,
                            "control_attempt": control_attempt,
                            "source": "reasoner_json_repair",
                            "count": internal_retries,
                            "succeeded": bool(decision_metadata.get("format_repaired")),
                        }
                    )
                _append_normalization_task_outcomes(
                    trace,
                    round_id=round_id,
                    control_attempt=control_attempt,
                    errors=tuple(decision_metadata.get("task_resolution_errors", ()) or ()),
                )
                schema_errors = _schema_error_rows(
                    decision_metadata.get("decision_schema_errors", ())
                )
                try:
                    parsed_decision = _decision(raw_decision)
                except (TypeError, ValueError) as exc:
                    parsed_decision = None
                    schema_errors.append(
                        {"code": "invalid_decision_payload", "detail": str(exc)}
                    )
                if parsed_decision is not None:
                    schema_errors.extend(_decision_preflight(parsed_decision))
                    if parsed_decision.action == "answer" and parsed_decision.answer:
                        latest_answer_candidate = parsed_decision
                if schema_errors:
                    _append_preflight_task_outcomes(
                        trace,
                        schema_errors,
                        round_id=round_id,
                        control_attempt=control_attempt,
                    )
                    trace.append(
                        {
                            "type": "decision_schema_error",
                            "round": round_id,
                            "control_attempt": control_attempt,
                            "code": schema_errors[0]["code"],
                            "errors": schema_errors,
                        }
                    )
                    if control_retries_used >= self.control_retry_budget:
                        trace.append(
                            {
                                "type": "decision_control_exhausted",
                                "round": round_id,
                                "control_retry_budget": self.control_retry_budget,
                                "errors": schema_errors,
                            }
                        )
                        protocol_exhausted = True
                        break
                    control_retries_used += 1
                    control_attempt += 1
                    decision_had_control_retry = True
                    trace.append(
                        {
                            "type": "control_retry",
                            "round": round_id,
                            "control_attempt": control_attempt,
                            "source": "decision_preflight",
                            "count": 1,
                            "succeeded": None,
                        }
                    )
                    feedback = _control_retry_feedback(
                        schema_errors,
                        revision=document.revision,
                        previous_feedback=feedback,
                    )
                    continue

                decision = parsed_decision
                assert decision is not None
                requested_rows = _read_observations(
                    observation_log,
                    decision.observation_requests,
                )
                answer_workspace_commit = bool(
                    force_finalize
                    and decision.action == "answer"
                    and decision.answer
                    and decision.workspace_ops
                )
                apply_result = document.apply_ops(
                    decision.workspace_ops,
                    observation_ids=observation_log.attempt_ids,
                )
                if decision.workspace_ops:
                    append_workspace_history(
                        history_path,
                        round_id=f"{round_id}.{control_attempt}",
                        operations=decision.workspace_ops,
                        result=apply_result,
                    )
                    if apply_result.accepted:
                        document.save(document_path)
                trace.append(
                    {
                        "type": "reasoner_decision",
                        "round": round_id,
                        "semantic_round": round_id,
                        "control_attempt": control_attempt,
                        "control_retry_count": control_retries_used,
                        "semantic_committed": apply_result.accepted,
                        "action": decision.action,
                        "tasks": [_task_descriptor(task) for task in decision.tasks],
                        "workspace_revision": document.revision,
                        "workspace_ops_accepted": apply_result.accepted,
                        "workspace_errors": list(apply_result.errors),
                        "supporting_claim_ids": list(decision.supporting_claim_ids),
                        "remaining_budget": remaining,
                        "force_finalize": force_finalize,
                        "final_attempt": forced_decision_calls if force_finalize else 0,
                        "answer_workspace_commit": answer_workspace_commit,
                    }
                )
                if apply_result.accepted:
                    rounds_run = round_id
                    break

                workspace_errors = [
                    {
                        "code": "workspace_transaction_rejected",
                        "detail": error,
                    }
                    for error in apply_result.errors
                ] or [{"code": "workspace_transaction_rejected"}]
                trace.append(
                    {
                        "type": "decision_schema_error",
                        "round": round_id,
                        "control_attempt": control_attempt,
                        "code": "workspace_transaction_rejected",
                        "errors": workspace_errors,
                    }
                )
                requested_observations = requested_rows
                if control_retries_used >= self.control_retry_budget:
                    trace.append(
                        {
                            "type": "decision_control_exhausted",
                            "round": round_id,
                            "control_retry_budget": self.control_retry_budget,
                            "errors": workspace_errors,
                        }
                    )
                    protocol_exhausted = True
                    break
                control_retries_used += 1
                control_attempt += 1
                decision_had_control_retry = True
                trace.append(
                    {
                        "type": "control_retry",
                        "round": round_id,
                        "control_attempt": control_attempt,
                        "source": "workspace_transaction_repair",
                        "count": 1,
                        "succeeded": None,
                    }
                )
                feedback = _control_retry_feedback(
                    workspace_errors,
                    revision=document.revision,
                    previous_feedback=feedback,
                )

            if protocol_exhausted:
                break
            if decision is None:
                break

            if decision.action == "answer":
                candidate = replace(
                    decision,
                    citations=_answer_citations(decision, document, evidence_store.records),
                )
                validation = _validate_answer(
                    candidate,
                    document,
                    observation_log.attempt_ids,
                    workspace.case.options,
                    supporting_observation_ids=_supporting_observation_ids(observation_log),
                    require_obligation_coverage=self.require_obligation_coverage,
                )
                trace.append({"type": "reference_integrity_check", "round": round_id, **validation.to_dict()})
                if validation.passed:
                    final_answer = candidate
                    break
                feedback = {
                    "type": "answer_reference_rejected",
                    "reason": validation.reason,
                    "errors": list(validation.errors),
                    "candidate_answer": candidate.answer,
                    "supporting_claim_ids": list(candidate.supporting_claim_ids),
                    "residual_uncertainty": candidate.residual_uncertainty,
                }
                requested_observations = requested_rows
                if force_finalize and not final_retry_available:
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
                if force_finalize and not final_retry_available:
                    break
                continue

            if force_finalize:
                task_requests = _append_task_requests(
                    trace,
                    decision.tasks,
                    round_id=round_id,
                    control_attempt=control_attempt,
                )
                _append_closed_task_outcomes(
                    trace,
                    task_requests,
                    round_id=round_id,
                    code="investigation_closed",
                )
                feedback = {
                    "type": "finalization_repair_required",
                    "reason": "investigation_closed",
                    "requested_task_count": len(decision.tasks),
                    "revision": document.revision,
                }
                requested_observations = requested_rows
                if final_retry_available:
                    continue
                break

            task_requests = _append_task_requests(
                trace,
                decision.tasks,
                round_id=round_id,
                control_attempt=control_attempt,
            )
            resolution_errors: list[dict[str, Any]] = []
            task_resolutions: list[dict[str, Any]] = []
            tasks = _resolve_tasks(
                workspace,
                decision.tasks,
                limit=min(self.max_tasks_per_round, remaining),
                errors=resolution_errors,
                resolutions=task_resolutions,
            )
            if resolution_errors:
                trace.append(
                    {
                        "type": "task_resolution",
                        "round": round_id,
                        "resolved_task_count": len(tasks),
                        "resolutions": task_resolutions,
                        "errors": resolution_errors,
                    }
                )
            if not tasks:
                _append_task_outcomes(
                    trace,
                    task_requests,
                    task_resolutions,
                    (),
                    round_id=round_id,
                )
                feedback = {
                    "type": "task_validation",
                    "reason": "reasoner_tasks_not_executable",
                    "requested_task_count": len(decision.tasks),
                    "errors": resolution_errors,
                }
                requested_observations = requested_rows
                continue

            if decision_had_control_retry:
                tasks = tuple(
                    replace(task, interpretation_purpose="control_retry")
                    if task.interpretation_purpose == "primary"
                    else task
                    for task in tasks
                )
            batch = investigator.run_batch(tasks)
            batch = _stamp_interpretation_purposes(batch, tasks)
            _append_task_outcomes(
                trace,
                task_requests,
                task_resolutions,
                batch,
                round_id=round_id,
            )
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
                    "outcomes": list(_outcome_digest(batch)),
                }
            )

        empty_answer = ReasonerDecision(action="answer")
        candidate_decision = final_answer or latest_answer_candidate or empty_answer
        selected = final_answer or (
            latest_answer_candidate
            if self.answer_policy == "benchmark_best_effort" and latest_answer_candidate is not None
            else empty_answer
        )
        selected_option = _letter(selected.answer) or _option_letter_from_answer(
            selected.answer,
            workspace.case.options,
        )
        schema_answer_present = bool(selected.answer) if not workspace.case.options else bool(selected_option)
        candidate_present = bool(selected.answer)
        preserve_candidate = self.answer_policy == "benchmark_best_effort" and candidate_present
        validation = _validate_answer(
            selected,
            document,
            observation_log.attempt_ids,
            workspace.case.options,
            supporting_observation_ids=_supporting_observation_ids(observation_log),
            require_obligation_coverage=self.require_obligation_coverage,
        )
        candidate_validation = (
            validation
            if candidate_decision is selected
            else _validate_answer(
                candidate_decision,
                document,
                observation_log.attempt_ids,
                workspace.case.options,
                supporting_observation_ids=_supporting_observation_ids(observation_log),
                require_obligation_coverage=self.require_obligation_coverage,
            )
        )
        if schema_answer_present or preserve_candidate:
            answer = selected.answer
            returned_answer_present = True
            citations = _answer_citations(selected, document, evidence_store.records)
            reference_valid = validation.passed
            reference_reason = validation.reason
        else:
            answer = "No valid answer was returned."
            returned_answer_present = False
            citations = ()
            reference_valid = False
            reference_reason = "answer_missing" if not selected.answer else "invalid_option_answer"
        supporting_intervals = export_supporting_intervals(
            document,
            selected.supporting_claim_ids,
            observation_log.rows,
        )
        candidate_answer = candidate_decision.answer
        verified_answer = final_answer.answer if final_answer is not None else ""
        verification_status = (
            "verified"
            if verified_answer
            else "candidate_only"
            if candidate_answer
            else "missing"
        )
        blocking_reasons = (
            ()
            if verification_status == "verified"
            else tuple(candidate_validation.errors)
            or (candidate_validation.reason,)
        )

        trace.append(
            {
                "type": "answer_outcome",
                "answer": answer,
                "candidate_answer": candidate_answer,
                "verified_answer": verified_answer,
                "verification_status": verification_status,
                "blocking_reasons": list(blocking_reasons),
                "raw_reasoner_answer": selected.answer,
                "selected_option": selected_option,
                "reference_valid": reference_valid,
                "reference_reason": reference_reason,
                "supporting_claim_ids": list(selected.supporting_claim_ids),
                "residual_uncertainty": selected.residual_uncertainty,
                "answer_owner": "reasoner",
                "framework_answer_mutation": False,
                "answer_policy": self.answer_policy,
                "answer_present": returned_answer_present,
                "supporting_intervals": [list(item) for item in supporting_intervals],
                "obligation_summary": document.obligation_summary(),
                "working_document_path": str(document_path),
                "observation_log_path": str(observation_log.path),
                "workspace_history_path": str(history_path),
            }
        )
        task_ledger = _task_ledger_validation(trace)
        trace.append({"type": "task_ledger_validation", **task_ledger})
        if task_ledger["silently_dropped_acquisition_count"]:
            raise RuntimeError(
                "task request ledger contains acquisitions without terminal outcomes: "
                + ", ".join(task_ledger["missing_ledger_ids"])
            )
        result = MultiRoundResult(
            case_id=workspace.case.case_id,
            answer=answer,
            selected_option=selected_option,
            citations=citations,
            correct=None,
            reference_valid=reference_valid,
            reference_reason=reference_reason,
            rounds=rounds_run,
            investigation_count=completed_investigations,
            evidence=tuple(evidence_store.records),
            reports=tuple(reports),
            trace=tuple(trace),
            answer_policy=self.answer_policy,
            answer_present=returned_answer_present,
            candidate_answer=candidate_answer,
            verified_answer=verified_answer,
            verification_status=verification_status,
            blocking_reasons=blocking_reasons,
            supporting_claim_ids=selected.supporting_claim_ids,
            supporting_intervals=supporting_intervals,
            residual_uncertainty=selected.residual_uncertainty,
        )
        _write_run_summary(workspace, result)
        return result


def _consume_decision_metadata(reasoner: Any) -> dict[str, Any]:
    consume = getattr(reasoner, "consume_decision_metadata", None)
    if not callable(consume):
        return {}
    value = consume()
    return dict(value) if isinstance(value, Mapping) else {}


def _schema_error_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        values: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = value
    elif value:
        values = (value,)
    else:
        values = ()
    rows = []
    for item in values:
        row = dict(item) if isinstance(item, Mapping) else {"detail": str(item)}
        row["code"] = str(row.get("code", "decision_schema_invalid") or "decision_schema_invalid")
        rows.append(row)
    return rows


def _decision_preflight(decision: ReasonerDecision) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    action_names = {"investigate", "read_observations", "update_workspace", "answer"}
    for index, operation in enumerate(decision.workspace_ops):
        op_type = str(operation.get("op", operation.get("type", "")) or "").strip().casefold()
        nested_action = str(operation.get("action", "") or "").strip().casefold()
        if op_type in action_names or nested_action in action_names:
            nested_tasks = operation.get("tasks", ())
            if not isinstance(nested_tasks, Sequence) or isinstance(
                nested_tasks,
                (str, bytes),
            ):
                nested_tasks = ()
            errors.append(
                {
                    "code": "action_like_op_inside_workspace_ops",
                    "workspace_op_index": index,
                    "op": op_type or nested_action,
                    "requested_task_ids": [
                        str(task.get("query_id", task.get("id", "")) or "").strip()
                        or f"workspace_op_{index}_task_{task_index}"
                        for task_index, task in enumerate(nested_tasks, start=1)
                        if isinstance(task, Mapping)
                    ],
                }
            )
    if decision.tasks and decision.action != "investigate":
        errors.append(
            {
                "code": "tasks_outside_investigate_action",
                "action": decision.action,
                "requested_task_ids": [task.query_id for task in decision.tasks],
            }
        )
    if decision.action == "investigate" and not decision.tasks:
        errors.append({"code": "investigate_action_requires_tasks"})
    return errors


def _control_retry_feedback(
    errors: Sequence[Mapping[str, Any]],
    *,
    revision: int,
    previous_feedback: Mapping[str, Any],
) -> dict[str, Any]:
    codes = [str(error.get("code", "decision_schema_invalid")) for error in errors]
    return {
        "type": "decision_control_retry",
        "cause": (
            "workspace_ops_rejected"
            if "workspace_transaction_rejected" in codes
            else "decision_schema_error"
        ),
        "errors": [dict(error) for error in errors],
        "revision": revision,
        "previous_feedback_type": str(previous_feedback.get("type", "") or ""),
        "instruction": "Preserve the semantic intent and return one corrected Decision JSON object.",
    }


def _append_normalization_task_outcomes(
    trace: list[Mapping[str, Any]],
    *,
    round_id: int,
    control_attempt: int,
    errors: Sequence[Any],
) -> None:
    for index, raw_error in enumerate(errors, start=1):
        error = (
            dict(raw_error)
            if isinstance(raw_error, Mapping)
            else {"code": "task_schema_invalid", "detail": str(raw_error)}
        )
        requested_task_id = str(
            error.get("requested_task_id", f"normalized_task_{index}")
            or f"normalized_task_{index}"
        )
        ledger_id = (
            f"semantic_{round_id}:control_{control_attempt}:"
            f"normalized_{index}:{requested_task_id}"
        )
        trace.append(
            {
                "type": "task_request",
                "round": round_id,
                "control_attempt": control_attempt,
                "ledger_id": ledger_id,
                "requested_task_id": requested_task_id,
                "origin": "reasoner_normalization",
            }
        )
        trace.append(
            {
                "type": "task_outcome",
                "round": round_id,
                "ledger_id": ledger_id,
                "requested_task_id": requested_task_id,
                "status": "explicit_resolution_error",
                "errors": [error],
            }
        )


def _append_preflight_task_outcomes(
    trace: list[Mapping[str, Any]],
    errors: Sequence[Mapping[str, Any]],
    *,
    round_id: int,
    control_attempt: int,
) -> None:
    outcome_index = 0
    for error in errors:
        for requested_task_id in tuple(error.get("requested_task_ids", ()) or ()):
            outcome_index += 1
            task_id = str(requested_task_id or f"preflight_task_{outcome_index}")
            ledger_id = (
                f"semantic_{round_id}:control_{control_attempt}:"
                f"preflight_{outcome_index}:{task_id}"
            )
            trace.append(
                {
                    "type": "task_request",
                    "round": round_id,
                    "control_attempt": control_attempt,
                    "ledger_id": ledger_id,
                    "requested_task_id": task_id,
                    "origin": "decision_preflight",
                }
            )
            trace.append(
                {
                    "type": "task_outcome",
                    "round": round_id,
                    "ledger_id": ledger_id,
                    "requested_task_id": task_id,
                    "status": "explicit_resolution_error",
                    "errors": [
                        {
                            "requested_task_id": task_id,
                            "code": str(error.get("code", "decision_schema_invalid")),
                        }
                    ],
                }
            )


def _append_task_requests(
    trace: list[Mapping[str, Any]],
    tasks: Sequence[InvestigationTask],
    *,
    round_id: int,
    control_attempt: int,
) -> tuple[dict[str, Any], ...]:
    rows = []
    for index, task in enumerate(tasks, start=1):
        requested_task_id = task.query_id or f"task_{index}"
        row = {
            "round": round_id,
            "control_attempt": control_attempt,
            "ledger_id": (
                f"semantic_{round_id}:control_{control_attempt}:"
                f"task_{index}:{requested_task_id}"
            ),
            "requested_task_id": requested_task_id,
            "task": _task_descriptor(task),
            "origin": "reasoner_decision",
        }
        rows.append(row)
        trace.append({"type": "task_request", **row})
    return tuple(rows)


def _append_closed_task_outcomes(
    trace: list[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    *,
    round_id: int,
    code: str,
) -> None:
    for request in requests:
        trace.append(
            {
                "type": "task_outcome",
                "round": round_id,
                "ledger_id": request["ledger_id"],
                "requested_task_id": request["requested_task_id"],
                "status": "explicit_resolution_error",
                "errors": [
                    {
                        "requested_task_id": request["requested_task_id"],
                        "code": code,
                    }
                ],
            }
        )


def _append_task_outcomes(
    trace: list[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    resolutions: Sequence[Mapping[str, Any]],
    reports: Sequence[InvestigationReport],
    *,
    round_id: int,
) -> None:
    reports_by_query: dict[str, list[InvestigationReport]] = {}
    for report in reports:
        reports_by_query.setdefault(report.query_id, []).append(report)
    for index, request in enumerate(requests):
        resolution = (
            dict(resolutions[index])
            if index < len(resolutions)
            else {
                "requested_task_id": request["requested_task_id"],
                "status": "explicit_resolution_error",
                "resolved_task_ids": [],
                "errors": [
                    {
                        "requested_task_id": request["requested_task_id"],
                        "code": "internal_resolution_outcome_missing",
                    }
                ],
            }
        )
        resolved_task_ids = [
            str(item) for item in tuple(resolution.get("resolved_task_ids", ()) or ())
        ]
        missing_reports = [
            task_id for task_id in resolved_task_ids if task_id not in reports_by_query
        ]
        resolution_errors = [
            dict(error)
            for error in tuple(resolution.get("errors", ()) or ())
            if isinstance(error, Mapping)
        ]
        if resolution.get("status") != "resolved" or missing_reports:
            if missing_reports:
                resolution_errors.append(
                    {
                        "requested_task_id": request["requested_task_id"],
                        "code": "investigator_outcome_missing",
                        "resolved_task_ids": missing_reports,
                    }
                )
            trace.append(
                {
                    "type": "task_outcome",
                    "round": round_id,
                    "ledger_id": request["ledger_id"],
                    "requested_task_id": request["requested_task_id"],
                    "status": "explicit_resolution_error",
                    "resolved_task_ids": resolved_task_ids,
                    "errors": resolution_errors,
                }
            )
            continue
        matched_reports = [
            report
            for task_id in resolved_task_ids
            for report in reports_by_query.get(task_id, ())
        ]
        trace.append(
            {
                "type": "task_outcome",
                "round": round_id,
                "ledger_id": request["ledger_id"],
                "requested_task_id": request["requested_task_id"],
                "status": "executed",
                "resolved_task_ids": resolved_task_ids,
                "report_outcomes": list(_outcome_digest(matched_reports)),
            }
        )


def _stamp_interpretation_purposes(
    reports: Sequence[InvestigationReport],
    tasks: Sequence[InvestigationTask],
) -> tuple[InvestigationReport, ...]:
    purpose_by_query = {task.query_id: task.interpretation_purpose for task in tasks}
    stamped = []
    for report in reports:
        purpose = purpose_by_query.get(report.query_id, "primary")
        attempts = tuple(
            replace(attempt, interpretation_purpose=purpose)
            if attempt.interpretation_purpose == "primary" and purpose != "primary"
            else attempt
            for attempt in report.attempts
        )
        stamped.append(replace(report, attempts=attempts))
    return tuple(stamped)


def _task_ledger_validation(trace: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    requests = tuple(row for row in trace if row.get("type") == "task_request")
    outcomes = tuple(row for row in trace if row.get("type") == "task_outcome")
    terminal_ids = {
        str(row.get("ledger_id", "") or "")
        for row in outcomes
        if row.get("status") in {"executed", "explicit_resolution_error"}
    }
    missing = tuple(
        str(row.get("ledger_id", "") or "")
        for row in requests
        if str(row.get("ledger_id", "") or "") not in terminal_ids
    )
    return {
        "requested_acquisition_count": len(requests),
        "executed_acquisition_count": sum(
            row.get("status") == "executed" for row in outcomes
        ),
        "task_resolution_error_count": sum(
            row.get("status") == "explicit_resolution_error" for row in outcomes
        ),
        "silently_dropped_acquisition_count": len(missing),
        "missing_ledger_ids": list(missing),
    }


def _mechanical_status(
    workspace: VirtualVideoWorkspace,
    document: WorkingDocument,
    observations: ObservationLog,
    *,
    runtime_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    errors = document.validate(observation_ids=observations.attempt_ids)
    coverage = _source_coverage(workspace, observations)
    known_attempts = set(observations.attempt_ids)
    supporting_attempts = set(_supporting_observation_ids(observations))
    active_claims = tuple(
        claim
        for claim in document.claims.values()
        if claim.status in {"active", "contested"}
    )
    supported_observation_claims = tuple(
        claim
        for claim in active_claims
        if claim.source == "observation"
        and claim.status == "active"
        and claim.confidence != "low"
        and bool(claim.cites)
        and set(claim.cites).issubset(supporting_attempts)
    )
    resolved_attempts = {
        cite
        for claim in supported_observation_claims
        for cite in claim.cites
    }
    source_rows = observations.catalog_source_rows()
    modality_by_attempt = {
        str(row.get("attempt_id", "")): str(row.get("modality", "") or "").casefold()
        for row in source_rows
    }
    asr_search_count = sum(row.get("sampling_config", {}).get("mode") == "search_asr" for row in source_rows)
    caption_search_count = sum(
        row.get("sampling_config", {}).get("mode") == "search_caption" for row in source_rows
    )
    visual_window_attempt_count = sum(
        str(row.get("modality", "") or "").casefold() in {"visual", "ocr"}
        for row in source_rows
    )
    visual_ranges = tuple(
        interval
        for row in source_rows
        if str(row.get("modality", "") or "").casefold() in {"visual", "ocr"}
        and str(row.get("evidence_role", "unclassified") or "unclassified").casefold()
        not in {"candidate", "negative"}
        for raw in tuple(row.get("inspected_ranges", ()) or ())
        if (interval := _time_range(raw)) is not None
    )
    visual_sampling_attempts = tuple(
        status
        for row in source_rows
        if (status := _visual_sampling_status(row)) is not None
    )
    unrefined_visual_attempts = tuple(
        status for status in visual_sampling_attempts if status["requires_refinement"]
    )
    low_fidelity_visual_attempts = tuple(
        status
        for status in visual_sampling_attempts
        if status["requested_fps"] > 0.0 and status["sampling_fidelity"] < 0.8
    )
    candidates_by_key: dict[tuple[str, float, float], dict[str, Any]] = {}
    caption_occurrences_by_key: dict[str, dict[str, Any]] = {}
    caption_occurrence_sets: list[dict[str, Any]] = []
    temporal_locators: list[dict[str, Any]] = []
    for row in observations.rows:
        config = row.get("sampling_config")
        if not isinstance(config, Mapping) or config.get("mode") != "search_caption":
            continue
        temporal_locator = config.get("temporal_locator")
        if isinstance(temporal_locator, Mapping):
            temporal_locators.append(dict(temporal_locator))
        occurrence_set = config.get("occurrence_set")
        if isinstance(occurrence_set, Mapping):
            compact_candidates: list[dict[str, Any]] = []
            for raw_candidate in tuple(occurrence_set.get("candidates", ()) or ()):
                if not isinstance(raw_candidate, Mapping):
                    continue
                interval = _time_range(raw_candidate.get("time_range"))
                if interval is None:
                    continue
                candidate = {
                    "attempt_id": str(row.get("attempt_id", "")),
                    "occurrence_id": str(raw_candidate.get("occurrence_id", "") or ""),
                    "time_range": list(interval),
                    "source_video_ids": list(raw_candidate.get("source_video_ids", ()) or ()),
                    "segment_ids": list(raw_candidate.get("segment_ids", ()) or ()),
                    "passage_ids": list(raw_candidate.get("passage_ids", ()) or ()),
                    "max_score": float(raw_candidate.get("max_score", 0.0) or 0.0),
                    "hit_count": int(raw_candidate.get("hit_count", 0) or 0),
                }
                compact_candidates.append(candidate)
                key = candidate["occurrence_id"] or json.dumps(
                    [candidate["source_video_ids"], candidate["time_range"]],
                    separators=(",", ":"),
                )
                existing = caption_occurrences_by_key.get(key)
                if existing is None or candidate["max_score"] > existing["max_score"]:
                    caption_occurrences_by_key[key] = candidate
            caption_occurrence_sets.append(
                {
                    "attempt_id": str(row.get("attempt_id", "")),
                    "status": str(occurrence_set.get("status", "") or ""),
                    "occurrence_ambiguous": bool(
                        occurrence_set.get("occurrence_ambiguous", False)
                    ),
                    "candidate_count": len(compact_candidates),
                    "candidates": compact_candidates,
                }
            )
        for hit in tuple(config.get("hits", ()) or ()):
            if not isinstance(hit, Mapping):
                continue
            interval = _time_range(hit.get("range"))
            if interval is None:
                continue
            candidate = {
                "attempt_id": str(row.get("attempt_id", "")),
                "passage_id": str(hit.get("passage_id", "")),
                "time_range": list(interval),
                "score": float(hit.get("score", 0.0) or 0.0),
                "query_matches": list(hit.get("query_matches", ()) or ()),
                "caption_excerpt": str(hit.get("caption_excerpt", "") or "")[:240],
            }
            key = (candidate["passage_id"], interval[0], interval[1])
            existing = candidates_by_key.get(key)
            if existing is None or candidate["score"] > existing["score"]:
                candidates_by_key[key] = candidate
    caption_candidates = tuple(
        sorted(
            candidates_by_key.values(),
            key=lambda candidate: (-float(candidate["score"]), float(candidate["time_range"][0])),
        )
    )
    pending_caption_candidates = tuple(
        candidate
        for candidate in caption_candidates
        if not any(
            _ranges_overlap(tuple(candidate["time_range"]), interval)
            for interval in visual_ranges
        )
    )
    caption_occurrence_candidates = tuple(
        sorted(
            caption_occurrences_by_key.values(),
            key=lambda candidate: (
                -float(candidate["max_score"]),
                float(candidate["time_range"][0]),
            ),
        )
    )
    pending_caption_occurrences = tuple(
        candidate
        for candidate in caption_occurrence_candidates
        if not any(
            _ranges_overlap(tuple(candidate["time_range"]), interval)
            for interval in visual_ranges
        )
    )
    pending_occurrence_ids = {
        str(candidate.get("occurrence_id", "") or "")
        for candidate in pending_caption_occurrences
    }
    unresolved_competing_occurrence_sets = tuple(
        occurrence_set
        for occurrence_set in caption_occurrence_sets
        if occurrence_set["occurrence_ambiguous"]
        and any(
            str(candidate.get("occurrence_id", "") or "") in pending_occurrence_ids
            for candidate in occurrence_set["candidates"]
        )
    )
    temporal_status: dict[str, Any] = {}
    if temporal_locators:
        latest_locator = temporal_locators[-1]
        candidate_groups = tuple(latest_locator.get("candidate_groups", ()) or ())[:4]
        recommended = latest_locator.get("recommended")
        if isinstance(recommended, Mapping):
            inspection_range = _time_range(recommended.get("inspection_range"))
            if inspection_range is not None and not any(
                _ranges_overlap(inspection_range, interval) for interval in visual_ranges
            ):
                temporal_status["recommended_temporal_candidate"] = dict(recommended)
        temporal_status["temporal_candidate_groups"] = [
            dict(item) for item in candidate_groups if isinstance(item, Mapping)
        ]
    caption_cited_claim_count = sum(
        any(modality_by_attempt.get(cite) == "caption_search" for cite in claim.cites)
        for claim in active_claims
    )
    visual_confirmed_claim_count = sum(
        any(modality_by_attempt.get(cite) in {"visual", "ocr"} for cite in claim.cites)
        for claim in active_claims
    )
    runtime = dict(runtime_status or {})
    hints: list[str] = []
    empty_streak = int(runtime.get("empty_search_streak", 0) or 0)
    zero_queries = tuple(runtime.get("previous_zero_hit_queries", ()) or ())
    if empty_streak >= 2 and zero_queries:
        last_modality = str(zero_queries[-1].get("modality", "") or "")
        if last_modality == "asr":
            hints.append("ASR has returned no hits twice; consider caption search or visual inspection.")
        elif last_modality == "caption":
            hints.append(
                "Caption retrieval may have missed a brief event; consider broader synonyms, a wider time filter, "
                "or hierarchical visual inspection."
            )
    raw_question_tags = workspace.case.metadata.get("question_type_tags", ()) or ()
    if isinstance(raw_question_tags, str):
        raw_question_tags = (raw_question_tags,)
    question_tags = {
        str(value).strip().casefold()
        for value in (
            workspace.case.question_type,
            *tuple(raw_question_tags),
        )
        if str(value or "").strip()
    }
    requires_visual = bool(
        question_tags
        & {
            "visual",
            "color",
            "clothing",
            "appearance",
            "object_appearance",
            "identity",
            "event_order",
            "event tracking",
            "event_tracking",
        }
    ) or bool(workspace.case.metadata.get("requires_visual_confirmation", False))
    if requires_visual and visual_window_attempt_count == 0:
        hints.append("Modality debt: this annotated question has no visual confirmation yet.")
    if pending_caption_candidates:
        hints.append(
            "Caption hits are locator candidates only. Inspect a top pending caption time_range with inspection_mode=window "
            "before using it as answer support."
        )
    if unresolved_competing_occurrence_sets:
        hints.append(
            "Caption retrieval spans multiple source/time occurrence clusters. Treat them as competing locator "
            "candidates and compare identity cues before promoting any interval to answer support."
        )
    if temporal_status.get("recommended_temporal_candidate"):
        hints.append(
            "An explicit after/before/first contract produced a scoped temporal locator. "
            "Inspect recommended_temporal_candidate.inspection_range before unrelated Caption hits."
        )
    if unrefined_visual_attempts:
        hints.append(
            "Wide visual scans are locator candidates only; refine a relevant neighborhood to <=120 seconds before "
            "using it as answer support."
        )
    if low_fidelity_visual_attempts:
        hints.append(
            "Observed sampling density fell below 80% of the requested fps. Do not assume the requested temporal "
            "resolution; narrow the relevant time_range before judging a brief transition or exact moment."
        )
    obligation_summary = document.obligation_summary()
    if not obligation_summary["answer_bearing_obligation_count"]:
        hints.append(
            "No answer-bearing evidence obligations exist. Decompose the observable requirements before finalizing."
        )
    elif obligation_summary["open_obligation_count_at_answer"]:
        hints.append(
            "Answer-bearing obligations remain open. Satisfy them with claim/material lineage or mark them unresolved "
            "with explicit residual uncertainty before finalizing."
        )
    return {
        "schema_version": "MechanicalCompletionStatusV1",
        "working_document_revision": document.revision,
        "workspace_valid": not errors,
        "workspace_errors": list(errors),
        "active_claim_count": len(active_claims),
        "non_premise_claim_count": sum(claim.source != "premise" for claim in active_claims),
        "supported_observation_claim_count": len(supported_observation_claims),
        "unresolved_observation_count": len(known_attempts - resolved_attempts),
        "active_claim_limit": document.active_claim_limit,
        "observation_attempt_count": len(observations.attempt_ids),
        "observation_interpretation_count": len(observations.rows),
        "asr_search_count": asr_search_count,
        "caption_search_count": caption_search_count,
        "visual_window_attempt_count": visual_window_attempt_count,
        "unrefined_visual_attempt_count": len(unrefined_visual_attempts),
        "unrefined_visual_attempts": list(unrefined_visual_attempts[-6:]),
        "low_fidelity_visual_attempt_count": len(low_fidelity_visual_attempts),
        "low_fidelity_visual_attempts": list(low_fidelity_visual_attempts[-6:]),
        "caption_cited_claim_count": caption_cited_claim_count,
        "visual_confirmed_claim_count": visual_confirmed_claim_count,
        "pending_caption_candidate_count": len(pending_caption_candidates),
        "pending_caption_candidates": list(
            pending_caption_candidates[
                : 12
                if str(runtime.get("caption_query_strategy", "") or "") == "rema"
                else 8
            ]
        ),
        "caption_occurrence_candidate_count": len(caption_occurrence_candidates),
        "pending_caption_occurrence_count": len(pending_caption_occurrences),
        "pending_caption_occurrences": list(pending_caption_occurrences[:8]),
        "caption_occurrence_ambiguous": bool(unresolved_competing_occurrence_sets),
        "caption_occurrence_sets": list(caption_occurrence_sets[-4:]),
        **temporal_status,
        "entity_count": len(document.entities),
        "candidate_interval_count": sum(note.role == "candidate" for note in document.timeline),
        "supporting_interval_count": sum(note.role == "supporting" for note in document.timeline),
        "negative_interval_count": sum(note.role == "negative" for note in document.timeline),
        "confirmed_occurrence_count": sum(
            note.role == "supporting" and str(note.metadata.get("status", "confirmed")) == "confirmed"
            for note in document.timeline
            if note.metadata.get("event_key") or note.label == "counted_event"
        ),
        "candidate_occurrence_count": sum(
            note.role == "candidate"
            for note in document.timeline
            if note.metadata.get("event_key") or note.label == "counted_event"
        ),
        "prompt_hints": hints,
        "source_coverage": coverage,
        "missing_segment_ids": [
            segment_id
            for source in coverage.values()
            for segment_id in source["missing_segment_ids"]
        ],
        "answer_owner": "reasoner",
        "obligations": [
            {
                **obligation.to_dict(),
                "state": document.obligation_states[requirement_id].to_dict(),
            }
            for requirement_id, obligation in document.obligations.items()
        ],
        **obligation_summary,
        **runtime,
    }


def _visual_sampling_status(row: Mapping[str, Any]) -> dict[str, Any] | None:
    if str(row.get("modality", "") or "").casefold() not in {"visual", "ocr"}:
        return None
    config = row.get("sampling_config")
    if not isinstance(config, Mapping):
        return None
    manifest = config.get("sampling_manifest")
    if not isinstance(manifest, Mapping):
        return None
    requested_fps = float(
        manifest.get("requested_fps", row.get("sampling_fps", 0.0)) or 0.0
    )
    effective_fps = float(manifest.get("effective_fps", 0.0) or 0.0)
    fidelity = float(
        manifest.get(
            "sampling_fidelity",
            effective_fps / requested_fps if requested_fps > 0.0 else 0.0,
        )
        or 0.0
    )
    return {
        "attempt_id": str(row.get("attempt_id", "")),
        "requested_range": list(
            manifest.get("requested_range", row.get("requested_range", ())) or ()
        ),
        "requested_fps": requested_fps,
        "effective_fps": effective_fps,
        "sampling_fidelity": fidelity,
        "max_gap": float(manifest.get("max_gap", 0.0) or 0.0),
        "coverage_ratio": float(manifest.get("coverage_ratio", 0.0) or 0.0),
        "requires_refinement": bool(manifest.get("requires_refinement")),
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
    errors: list[dict[str, Any]] | None = None,
    resolutions: list[dict[str, Any]] | None = None,
) -> tuple[InvestigationTask, ...]:
    limit = max(0, int(limit))
    segments = tuple(workspace.manifest.segments)
    by_id = {segment.segment_id: segment for segment in segments}
    global_aliases = {"all", "full", "full_video", "global", "workspace"}
    resolution_errors = errors if errors is not None else []
    resolution_rows = resolutions if resolutions is not None else []
    groups: list[dict[str, Any]] = []
    for requested_task in tasks:
        error_start = len(resolution_errors)
        task_error = _task_executability_error(requested_task)
        if task_error:
            resolution_errors.append(_task_resolution_error(requested_task, task_error))
            groups.append(
                {
                    "requested_task": requested_task,
                    "tasks": (),
                    "error_start": error_start,
                }
            )
            continue
        task = _resolve_task_coordinates(
            requested_task,
            by_id,
            global_aliases=global_aliases,
            errors=resolution_errors,
        )
        if task is None:
            groups.append(
                {
                    "requested_task": requested_task,
                    "tasks": (),
                    "error_start": error_start,
                }
            )
            continue
        if task.inspection_mode == "arbitrate_observation":
            groups.append(
                {"requested_task": requested_task, "tasks": (task,), "error_start": error_start}
            )
            continue
        if task.inspection_mode in {"search_asr", "search_caption"}:
            if task.segment_id.casefold() in global_aliases:
                task = replace(task, segment_id="")
            groups.append(
                {"requested_task": requested_task, "tasks": (task,), "error_start": error_start}
            )
            continue
        if task.time_range is not None:
            start, end = task.time_range
            if task.segment_id in by_id:
                groups.append(
                    {"requested_task": requested_task, "tasks": (task,), "error_start": error_start}
                )
                continue
            overlaps = tuple(
                (segment, max(start, segment.virtual_start_sec), min(end, segment.virtual_end_sec))
                for segment in segments
                if min(end, segment.virtual_end_sec) > max(start, segment.virtual_start_sec)
            )
            if overlaps:
                groups.append(
                    {
                        "requested_task": requested_task,
                        "error_start": error_start,
                        "tasks": tuple(
                            replace(
                                task,
                                query_id=(
                                    task.query_id
                                    if len(overlaps) == 1
                                    else f"{task.query_id}_{index:02d}"
                                ),
                                segment_id=segment.segment_id,
                                time_range=(overlap_start, overlap_end),
                            )
                            for index, (segment, overlap_start, overlap_end) in enumerate(
                                overlaps,
                                start=1,
                            )
                        ),
                    }
                )
                continue
            resolution_errors.append(
                _task_resolution_error(
                    task,
                    "range_outside_workspace",
                    requested_range=[start, end],
                    workspace_range=[0.0, workspace.manifest.duration_sec],
                )
            )
            groups.append(
                {
                    "requested_task": requested_task,
                    "tasks": (),
                    "error_start": error_start,
                }
            )
            continue
        if task.segment_id in by_id:
            groups.append(
                {"requested_task": requested_task, "tasks": (task,), "error_start": error_start}
            )
            continue
        if task.segment_id.casefold() not in global_aliases:
            resolution_errors.append(_task_resolution_error(task, "target_missing"))
            groups.append(
                {
                    "requested_task": requested_task,
                    "tasks": (),
                    "error_start": error_start,
                }
            )
            continue
        selected = select_uniform_items(segments, min(limit, len(segments)))
        groups.append(
            {
                "requested_task": requested_task,
                "error_start": error_start,
                "tasks": tuple(
                    replace(task, query_id=f"{task.query_id}_{segment.segment_id}", segment_id=segment.segment_id)
                    for segment in selected
                ),
            }
        )
    resolved: list[InvestigationTask] = []
    resolved_by_group: dict[int, list[InvestigationTask]] = {}
    depth = 0
    while len(resolved) < limit and any(depth < len(group["tasks"]) for group in groups):
        for group_index, group in enumerate(groups):
            if len(resolved) >= limit:
                break
            if depth < len(group["tasks"]):
                selected_task = group["tasks"][depth]
                resolved.append(selected_task)
                resolved_by_group.setdefault(group_index, []).append(selected_task)
        depth += 1
    resolution_error_count = len(resolution_errors)
    for group_index, group in enumerate(groups):
        selected_tasks = tuple(resolved_by_group.get(group_index, ()))
        error_end = (
            int(groups[group_index + 1]["error_start"])
            if group_index + 1 < len(groups)
            else resolution_error_count
        )
        group_errors = list(
            resolution_errors[int(group["error_start"]):error_end]
        )
        if not selected_tasks and not group_errors:
            requested_task = group["requested_task"]
            code = "investigation_budget_exhausted" if limit == 0 else "per_round_task_limit_exceeded"
            error = _task_resolution_error(requested_task, code, limit=limit)
            resolution_errors.append(error)
            group_errors = [error]
        resolution_rows.append(
            {
                "requested_task_id": group["requested_task"].query_id,
                "status": "resolved" if selected_tasks else "explicit_resolution_error",
                "resolved_task_ids": [task.query_id for task in selected_tasks],
                "errors": [dict(error) for error in group_errors],
            }
        )
    return tuple(resolved)


def _resolve_task_coordinates(
    task: InvestigationTask,
    segments_by_id: Mapping[str, Any],
    *,
    global_aliases: set[str],
    errors: list[dict[str, Any]],
) -> InvestigationTask | None:
    segment_id = task.segment_id
    segment = segments_by_id.get(segment_id)
    is_global = segment_id.casefold() in global_aliases
    if segment_id and segment is None and not is_global:
        errors.append(
            _task_resolution_error(
                task,
                "unknown_segment",
                known_segment_ids=sorted(segments_by_id),
            )
        )
        return None

    if task.coordinate_space == "segment_local":
        if segment is None:
            errors.append(
                _task_resolution_error(
                    task,
                    "segment_local_requires_known_segment",
                    known_segment_ids=sorted(segments_by_id),
                )
            )
            return None
        if task.time_range is None:
            errors.append(_task_resolution_error(task, "segment_local_requires_time_range"))
            return None
        local_start, local_end = task.time_range
        duration = max(0.0, float(segment.virtual_end_sec) - float(segment.virtual_start_sec))
        if local_start < 0.0 or local_end > duration + 1e-6:
            errors.append(
                _task_resolution_error(
                    task,
                    "range_outside_segment",
                    requested_range=[local_start, local_end],
                    valid_segment_local_range=[0.0, duration],
                    valid_virtual_range=[segment.virtual_start_sec, segment.virtual_end_sec],
                )
            )
            return None
        virtual_range = (
            float(segment.virtual_start_sec) + local_start,
            float(segment.virtual_start_sec) + local_end,
        )
        return replace(
            task,
            time_range=virtual_range,
            coordinate_space="virtual",
            conversion_trace=(
                *task.conversion_trace,
                {
                    "operation": "segment_local_to_virtual",
                    "segment_id": segment_id,
                    "input_range": [local_start, local_end],
                    "output_range": list(virtual_range),
                },
            ),
        )

    if task.time_range is not None and segment is not None:
        start, end = task.time_range
        segment_start = float(segment.virtual_start_sec)
        segment_end = float(segment.virtual_end_sec)
        start_overrun = max(0.0, segment_start - start)
        end_overrun = max(0.0, end - segment_end)
        if start_overrun > 1e-6 or end_overrun > 1e-6:
            clamped_range = (max(start, segment_start), min(end, segment_end))
            can_clamp_boundary_rounding = (
                start_overrun <= _TIME_BOUNDARY_TOLERANCE_SEC
                and end_overrun <= _TIME_BOUNDARY_TOLERANCE_SEC
                and clamped_range[1] > clamped_range[0]
            )
            if can_clamp_boundary_rounding:
                return replace(
                    task,
                    time_range=clamped_range,
                    conversion_trace=(
                        *task.conversion_trace,
                        {
                            "operation": "virtual_boundary_clamp",
                            "segment_id": segment_id,
                            "input_range": [start, end],
                            "output_range": list(clamped_range),
                            "tolerance_sec": _TIME_BOUNDARY_TOLERANCE_SEC,
                        },
                    ),
                )
            errors.append(
                _task_resolution_error(
                    task,
                    "range_outside_segment",
                    requested_range=[start, end],
                    coordinate_space="virtual",
                    valid_virtual_range=[segment.virtual_start_sec, segment.virtual_end_sec],
                    boundary_tolerance_sec=_TIME_BOUNDARY_TOLERANCE_SEC,
                    segment_local_hint=[
                        max(0.0, start - segment_start),
                        max(0.0, end - segment_start),
                    ],
                )
            )
            return None
    return task


def _task_resolution_error(
    task: InvestigationTask,
    code: str,
    **details: Any,
) -> dict[str, Any]:
    return {
        "requested_task_id": task.query_id,
        "query_id": task.query_id,
        "code": str(code),
        "segment_id": task.segment_id,
        "coordinate_space": task.coordinate_space,
        **details,
    }


def _task_executability_error(task: InvestigationTask) -> str:
    if not task.query_id:
        return "query_id_missing"
    if not task.goal:
        return "goal_missing"
    if task.inspection_mode == "search_asr" and not task.search_terms:
        return "search_terms_missing"
    if task.inspection_mode == "search_caption" and not task.caption_queries:
        return "caption_queries_missing"
    if task.inspection_mode == "arbitrate_observation" and not task.arbitration_attempt_id:
        return "arbitration_attempt_id_missing"
    if task.inspection_mode == "window" and not (task.segment_id or task.time_range):
        return "target_missing"
    return ""


def _ranges_overlap(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return min(left[1], right[1]) > max(left[0], right[0])


def _task_is_executable(task: InvestigationTask) -> bool:
    return not _task_executability_error(task)


def _validate_answer(
    decision: ReasonerDecision,
    document: WorkingDocument,
    observation_ids: Sequence[str],
    options: Mapping[str, str],
    *,
    supporting_observation_ids: Sequence[str] | None = None,
    require_obligation_coverage: bool = False,
) -> AnswerValidation:
    validation = document.validate_answer(
        decision.supporting_claim_ids,
        observation_ids=observation_ids,
        supporting_observation_ids=supporting_observation_ids,
        require_obligation_coverage=require_obligation_coverage,
    )
    if not decision.answer:
        return AnswerValidation(
            False,
            "answer_missing",
            validation.supporting_claim_ids,
            validation.cited_attempt_ids,
            ("answer_is_required",),
        )
    if options and not (_letter(decision.answer) or _option_letter_from_answer(decision.answer, options)):
        reason = "answer_missing" if not decision.answer else "invalid_option_answer"
        return AnswerValidation(
            False,
            reason,
            validation.supporting_claim_ids,
            validation.cited_attempt_ids,
            ("answer_must_select_exactly_one_option",),
        )
    if not validation.passed:
        return validation
    if options and _material_uncertainty(decision.residual_uncertainty):
        return _answer_rejected(validation, "answer_support_uncertain")
    return validation


def _supporting_observation_ids(observations: ObservationLog) -> tuple[str, ...]:
    return tuple(
        str(row.get("attempt_id", ""))
        for row in observations.catalog_source_rows()
        if str(row.get("evidence_role", "unclassified") or "unclassified").casefold()
        not in {"candidate", "negative"}
        and str(row.get("modality", "") or "").casefold() != "caption_search"
    )


def _answer_rejected(validation: AnswerValidation, reason: str) -> AnswerValidation:
    return AnswerValidation(
        False,
        reason,
        validation.supporting_claim_ids,
        validation.cited_attempt_ids,
        (reason,),
    )


def _material_uncertainty(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()
    return normalized not in {"", "none", "no", "n a", "not applicable", "no material uncertainty"}


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
            "consumes_budget": report.cost.get("consumes_budget") is not False,
            "reused": bool(report.cost.get("reused")),
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
        "coordinate_space": task.coordinate_space,
        "source_video_ids": list(task.source_video_ids),
        "conversion_trace": [dict(item) for item in task.conversion_trace],
        "inspection_mode": task.inspection_mode,
        "caption_queries": list(task.caption_queries),
        "top_k": task.top_k,
        "index_mode": task.index_mode,
        "expand_neighbors": task.expand_neighbors,
        "sampling_floor_fps": task.sampling_floor_fps,
        "arbitration_attempt_id": task.arbitration_attempt_id,
        "force_reinspect": task.force_reinspect,
        "interpretation_purpose": task.interpretation_purpose,
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
        coordinate_space=str(value.get("coordinate_space", "virtual") or "virtual"),
        source_video_ids=tuple(value.get("source_video_ids", ()) or ()),
        conversion_trace=tuple(value.get("conversion_trace", ()) or ()),
        expected_evidence=str(value.get("expected_evidence", "") or ""),
        inspection_mode=str(value.get("inspection_mode", "window") or "window"),
        search_terms=tuple(value.get("search_terms", ()) or ()),
        caption_queries=tuple(value.get("caption_queries", value.get("queries", ())) or ()),
        top_k=int(value.get("top_k", 12) or 12),
        index_mode=str(value.get("index_mode", "lexical") or "lexical"),
        expand_neighbors=int(value.get("expand_neighbors", 0) or 0),
        sampling_floor_fps=value.get("sampling_floor_fps"),
        arbitration_attempt_id=str(value.get("arbitration_attempt_id", "") or ""),
        force_reinspect=bool(value.get("force_reinspect", False)),
        interpretation_purpose=str(value.get("interpretation_purpose", "primary") or "primary"),
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
        "answer_present": result.answer_present,
        "answer_policy": result.answer_policy,
        "candidate_answer": result.candidate_answer,
        "verified_answer": result.verified_answer,
        "verification_status": result.verification_status,
        "blocking_reasons": list(result.blocking_reasons),
        "selected_option": result.selected_option,
        "citations": list(result.citations),
        "correct": result.correct,
        "correctness_source": "external_evaluator",
        "reference_valid": result.reference_valid,
        "reference_reason": result.reference_reason,
        "supporting_claim_ids": list(result.supporting_claim_ids),
        "supporting_intervals": [list(item) for item in result.supporting_intervals],
        "residual_uncertainty": result.residual_uncertainty,
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
