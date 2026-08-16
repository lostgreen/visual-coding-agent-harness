from __future__ import annotations

from dataclasses import dataclass, replace
import json
import re
from typing import Any, Mapping, Sequence

from vcah.evidence_state import EVIDENCE_KINDS
from vcah.evidence_runtime import (
    EvidencePlan,
    RuntimeEvidenceCatalog,
    advance_requirement_state,
    compile_evidence_plan,
)
from vcah.investigator import (
    INTERPRETATION_PURPOSES,
    InvestigationReport,
    ObservationAttempt,
    VirtualVideoInvestigator,
)
from vcah.memory import EvidenceStore
from vcah.occurrence_agent import (
    OCCURRENCE_METHOD_ARMS,
    OccurrenceResolutionStateV1,
    OccurrenceResolutionStateV2,
    candidate_cards_by_occurrence,
    occurrence_excerpt_digest,
    occurrence_visible_text_digest,
)
from vcah.phase5 import inspection_mode_policy_errors
from vcah.runtime_metrics import (
    export_item_supporting_intervals,
    export_supporting_intervals,
)
from vcah.temporal_scope import resolve_temporal_scope
from vcah.types import EvidenceRecord, to_jsonable
from vcah.virtual_index import build_workspace_overview
from vcah.virtual_video import VirtualVideoWorkspace, select_uniform_items
from vcah.workspace import (
    AnswerValidation,
    ObservationLog,
    WorkingDocument,
    append_workspace_history,
    evidence_attempt_id,
    prompt_digest,
    render_frozen_working_view,
    render_working_view,
)


_INSPECTION_MODES = {"window", "search_asr", "search_caption", "arbitrate_observation"}
_TIME_BOUNDARY_TOLERANCE_SEC = 1.0
_OCCURRENCE_SELECTION_FINAL_CALL_BUDGET = 3
_OCCURRENCE_ANSWER_FINAL_CALL_BUDGET = 3
_MUST_ANSWER_OCCURRENCE_CODES = frozenset(
    {
        "occurrence_answer_required_after_selection",
        "occurrence_answer_required_after_resolution",
    }
)
_MUST_NOT_ANSWER_OCCURRENCE_CODES = frozenset(
    {
        "occurrence_selection_required",
        "occurrence_resolution_required",
        "occurrence_search_required",
        "occurrence_no_match_required_at_finalization",
        "occurrence_sufficiency_resolution_required",
        "occurrence_sufficiency_assessment_required",
        "occurrence_sufficiency_requires_insufficient",
        "occurrence_sufficiency_forbids_selection",
        "occurrence_sufficiency_candidate_not_supported",
        "occurrence_locator_inspection_required",
        "occurrence_locator_binding_required",
        "occurrence_locator_unbound_window_forbidden",
    }
)
RUN_ARTIFACT_NAMES = (
    "evidence.jsonl",
    "observation_log.jsonl",
    "working_document.json",
    "workspace_ops.jsonl",
    "exploration_ledger.jsonl",
    "occurrence_resolution_state.json",
    "run_summary.json",
)
_FROZEN_MECHANICAL_STATUS_KEYS = frozenset(
    {
        "schema_version",
        "working_document_revision",
        "workspace_valid",
        "workspace_errors",
        "active_claim_count",
        "non_premise_claim_count",
        "supported_observation_claim_count",
        "unresolved_observation_count",
        "active_claim_limit",
        "observation_attempt_count",
        "observation_interpretation_count",
        "asr_search_count",
        "caption_search_count",
        "visual_window_attempt_count",
        "unrefined_visual_attempt_count",
        "unrefined_visual_attempts",
        "low_fidelity_visual_attempt_count",
        "low_fidelity_visual_attempts",
        "caption_cited_claim_count",
        "visual_confirmed_claim_count",
        "pending_caption_candidate_count",
        "pending_caption_candidates",
        "caption_occurrence_candidate_count",
        "pending_caption_occurrence_count",
        "pending_caption_occurrences",
        "caption_occurrence_ambiguous",
        "caption_occurrence_sets",
        "flat_occurrence_passages",
        "flat_occurrence_queries",
        "recommended_temporal_candidate",
        "temporal_candidate_groups",
        "oracle_guidance",
        "entity_count",
        "candidate_interval_count",
        "supporting_interval_count",
        "negative_interval_count",
        "confirmed_occurrence_count",
        "candidate_occurrence_count",
        "prompt_hints",
        "source_coverage",
        "missing_segment_ids",
        "answer_owner",
    }
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
    locator_attempt_id: str = ""
    occurrence_id: str = ""
    temporal_scope_id: str = ""
    evidence_kind: str = "generic"
    requirement_id: str = ""
    refine_item_id: str = ""
    refine_interpretation_id: str = ""
    parent_attempt_id: str = ""
    cue_id: str = ""
    window_radius_sec: float = 5.0
    cue_stage: str = ""
    cue_virtual_time: float | None = None
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
        object.__setattr__(self, "locator_attempt_id", str(self.locator_attempt_id or "").strip())
        object.__setattr__(self, "occurrence_id", str(self.occurrence_id or "").strip())
        object.__setattr__(self, "temporal_scope_id", str(self.temporal_scope_id or "").strip())
        evidence_kind = str(self.evidence_kind or "generic").strip().casefold()
        if evidence_kind not in EVIDENCE_KINDS:
            raise ValueError(f"unsupported evidence_kind: {evidence_kind}")
        object.__setattr__(self, "evidence_kind", evidence_kind)
        object.__setattr__(self, "requirement_id", str(self.requirement_id or "").strip())
        object.__setattr__(self, "refine_item_id", str(self.refine_item_id or "").strip())
        object.__setattr__(
            self,
            "refine_interpretation_id",
            str(self.refine_interpretation_id or "").strip(),
        )
        object.__setattr__(self, "parent_attempt_id", str(self.parent_attempt_id or "").strip())
        object.__setattr__(self, "cue_id", str(self.cue_id or "").strip())
        object.__setattr__(
            self,
            "window_radius_sec",
            min(30.0, max(0.5, float(self.window_radius_sec or 5.0))),
        )
        cue_stage = str(self.cue_stage or "").strip().casefold()
        if cue_stage not in {"", "cue_verification", "child_refinement"}:
            raise ValueError(f"unsupported cue_stage: {cue_stage}")
        object.__setattr__(self, "cue_stage", cue_stage)
        object.__setattr__(
            self,
            "cue_virtual_time",
            float(self.cue_virtual_time) if self.cue_virtual_time is not None else None,
        )
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
    occurrence_ops: tuple[Mapping[str, Any], ...] = ()
    supporting_claim_ids: tuple[str, ...] = ()
    supporting_item_ids: tuple[str, ...] = ()
    supports_requirement_ids: tuple[str, ...] = ()
    unresolved_requirement_ids: tuple[str, ...] = ()
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
            "occurrence_ops",
            tuple(dict(item) for item in self.occurrence_ops if isinstance(item, Mapping)),
        )
        object.__setattr__(
            self,
            "supporting_claim_ids",
            tuple(dict.fromkeys(str(item).strip() for item in self.supporting_claim_ids if str(item).strip())),
        )
        object.__setattr__(
            self,
            "supporting_item_ids",
            tuple(dict.fromkeys(str(item).strip() for item in self.supporting_item_ids if str(item).strip())),
        )
        object.__setattr__(
            self,
            "supports_requirement_ids",
            tuple(dict.fromkeys(str(item).strip() for item in self.supports_requirement_ids if str(item).strip())),
        )
        object.__setattr__(
            self,
            "unresolved_requirement_ids",
            tuple(dict.fromkeys(str(item).strip() for item in self.unresolved_requirement_ids if str(item).strip())),
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
    evidence_control_mode: str = "strict"
    evidence_state_mode: str = "llm_authored"
    answer_present: bool = False
    candidate_answer: str = ""
    verified_answer: str = ""
    verification_status: str = "missing"
    blocking_reasons: tuple[str, ...] = ()
    supporting_claim_ids: tuple[str, ...] = ()
    supporting_item_ids: tuple[str, ...] = ()
    supporting_attempt_ids: tuple[str, ...] = ()
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
        require_item_provenance: bool = False,
        require_evidence_kind_requirements: bool = False,
        closure_repair_budget: int = 0,
        answer_policy: str = "strict",
        evidence_control_mode: str = "strict",
        evidence_state_mode: str = "llm_authored",
        allowed_inspection_modes: frozenset[str] | None = None,
        controller_mode: str = "mger",
        bootstrap_tasks: Sequence[InvestigationTask | Mapping[str, Any]] = (),
        occurrence_method_arm: str = "none",
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
        self.require_item_provenance = bool(require_item_provenance)
        self.require_evidence_kind_requirements = bool(
            require_evidence_kind_requirements
        )
        self.closure_repair_budget = min(1, max(0, int(closure_repair_budget)))
        self.max_investigations = max(1, int(max_investigations))
        self.max_tasks_per_round = max(1, int(max_tasks_per_round))
        policy = str(answer_policy or "strict").strip().casefold()
        if policy not in {"strict", "benchmark_best_effort"}:
            raise ValueError(f"unsupported answer_policy: {policy}")
        self.answer_policy = policy
        control_mode = str(evidence_control_mode or "strict").strip().casefold()
        if control_mode not in {"shadow", "strict"}:
            raise ValueError(f"unsupported evidence_control_mode: {control_mode}")
        self.evidence_control_mode = control_mode
        state_mode = str(evidence_state_mode or "llm_authored").strip().casefold()
        if state_mode not in {"llm_authored", "runtime_derived"}:
            raise ValueError(f"unsupported evidence_state_mode: {state_mode}")
        self.evidence_state_mode = state_mode
        controller = str(controller_mode or "mger").strip().casefold()
        if controller not in {"frozen_baseline", "minimal_tool", "mger"}:
            raise ValueError(f"unsupported controller_mode: {controller}")
        self.controller_mode = controller
        method_arm = str(occurrence_method_arm or "none").strip().casefold()
        if method_arm not in OCCURRENCE_METHOD_ARMS:
            raise ValueError(f"unsupported occurrence method arm: {method_arm}")
        self.occurrence_method_arm = method_arm
        self.bootstrap_tasks = tuple(_task(item) for item in bootstrap_tasks)
        self.allowed_inspection_modes = (
            None
            if allowed_inspection_modes is None
            else frozenset(
                str(item).strip().casefold()
                for item in allowed_inspection_modes
                if str(item).strip()
            )
        )

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
        occurrence_selection_final_calls = 0
        occurrence_answer_final_calls = 0
        occurrence_recovery_pending = False
        occurrence_recovery_rounds_granted = 0
        locator_budget_exhaustion_recorded = False
        locator_terminal_outcomes: dict[tuple[str, str], str] = {}
        protocol_exhausted = False
        surfaced_observation_ids: set[str] = set()
        closure_repair_pending = False
        closure_repair_count = 0
        runtime_catalog = RuntimeEvidenceCatalog.build(document, observation_log)
        occurrence_state: OccurrenceResolutionStateV1 | OccurrenceResolutionStateV2 | None
        if self.occurrence_method_arm == "a2":
            occurrence_state = OccurrenceResolutionStateV1()
        elif self.occurrence_method_arm in {"a2-clean", "a3", "a4"}:
            occurrence_state = OccurrenceResolutionStateV2(
                sufficiency_enabled=self.occurrence_method_arm == "a4"
            )
        else:
            occurrence_state = None

        def grant_occurrence_recovery(*, round_id: int, stage: str) -> bool:
            nonlocal occurrence_recovery_pending
            nonlocal occurrence_recovery_rounds_granted
            if occurrence_recovery_pending:
                return True
            if isinstance(occurrence_state, OccurrenceResolutionStateV2):
                if _occurrence_resolution_complete(occurrence_state):
                    if (
                        occurrence_answer_final_calls
                        >= _OCCURRENCE_ANSWER_FINAL_CALL_BUDGET
                    ):
                        return False
                elif (
                    occurrence_selection_final_calls
                    >= _OCCURRENCE_SELECTION_FINAL_CALL_BUDGET
                ):
                    return False
            occurrence_recovery_pending = True
            occurrence_recovery_rounds_granted += 1
            trace.append(
                {
                    "type": "occurrence_recovery_round_granted",
                    "round": round_id,
                    "method_arm": self.occurrence_method_arm,
                    "stage": stage,
                    "recovery_index": occurrence_recovery_rounds_granted,
                    "count": 1,
                }
            )
            return True

        occurrence_state_path = workspace.root_dir / "occurrence_resolution_state.json"
        treatment_eligible_recorded = False
        treatment_exposed_recorded = False
        resolution_activated_recorded = False
        arbitration_activated_recorded = False
        sufficiency_activated_recorded = False
        if isinstance(occurrence_state, OccurrenceResolutionStateV1):
            occurrence_state.save(occurrence_state_path)

        if self.evidence_state_mode == "runtime_derived":
            raw_plan = (
                self.reasoner.plan_evidence(
                    question=workspace.case.question,
                    options=dict(workspace.case.options),
                )
                if callable(getattr(self.reasoner, "plan_evidence", None))
                else None
            )
            plan = EvidencePlan.from_mapping(raw_plan, question=workspace.case.question)
            consume_plan_metadata = getattr(self.reasoner, "consume_plan_metadata", None)
            plan_metadata = (
                dict(consume_plan_metadata())
                if callable(consume_plan_metadata)
                else {}
            )
            compilation = compile_evidence_plan(
                document,
                plan,
                question=workspace.case.question,
            )
            document.save(document_path)
            trace.append(
                {
                    "type": "runtime_evidence_plan",
                    "plan": plan.to_dict(),
                    "compilation": compilation,
                    "prompt_char_count": int(
                        plan_metadata.get("prompt_char_count", 0) or 0
                    ),
                    "prompt_digest": str(
                        plan_metadata.get("prompt_digest", "") or ""
                    ),
                    "prompt_schema_token_cost": int(
                        plan_metadata.get("prompt_schema_token_cost", 0) or 0
                    ),
                }
            )

        if self.bootstrap_tasks:
            bootstrap_batch = investigator.run_batch(self.bootstrap_tasks)
            bootstrap_batch = _stamp_interpretation_purposes(
                bootstrap_batch,
                self.bootstrap_tasks,
            )
            reports.extend(bootstrap_batch)
            known_evidence_ids = {record.evidence_id for record in evidence_store.records}
            bootstrap_rows: list[Mapping[str, Any]] = []
            for report in bootstrap_batch:
                for record in report.evidence:
                    if record.evidence_id not in known_evidence_ids:
                        evidence_store.add(record)
                        known_evidence_ids.add(record.evidence_id)
                for attempt in report.attempts:
                    bootstrap_rows.append(
                        observation_log.append_attempt(
                            attempt,
                            round_id="bootstrap",
                            source_lineage=_attempt_lineage(attempt, report.evidence),
                        )
                    )
            requested_observations = tuple(bootstrap_rows[-12:])
            feedback = {
                "type": "bootstrap_observations_ready",
                "requested_tasks": len(self.bootstrap_tasks),
                "completed_tasks": sum(
                    _report_completed(report) for report in bootstrap_batch
                ),
                "new_observation_interpretations": len(bootstrap_rows),
                "outcomes": list(_outcome_digest(bootstrap_batch)),
                "consumes_investigation_budget": False,
            }
            trace.append(
                {
                    "type": "bootstrap_observation_batch",
                    "requested_tasks": len(self.bootstrap_tasks),
                    "completed_tasks": feedback["completed_tasks"],
                    "attempt_ids": [
                        str(row["attempt_id"]) for row in bootstrap_rows
                    ],
                    "outcomes": list(_outcome_digest(bootstrap_batch)),
                    "consumes_investigation_budget": False,
                }
            )
            replay_prime_query_ids = {
                task.query_id
                for task in self.bootstrap_tasks
                if task.query_id == "occurrence_replay_prime"
            }
            if replay_prime_query_ids:
                replay_prime_reports = tuple(
                    report
                    for report in bootstrap_batch
                    if report.query_id in replay_prime_query_ids
                )
                trace.append(
                    {
                        "type": "occurrence_replay_primed",
                        "round": 0,
                        "requested_tasks": len(replay_prime_query_ids),
                        "completed_tasks": sum(
                            _report_succeeded(report)
                            for report in replay_prime_reports
                        ),
                        "completed": bool(replay_prime_reports)
                        and len(replay_prime_reports) == len(replay_prime_query_ids)
                        and all(
                            _report_succeeded(report)
                            for report in replay_prime_reports
                        ),
                        "consumes_investigation_budget": False,
                    }
                )

        for round_id in range(
            1,
            self.semantic_round_budget
            + self.closure_repair_budget
            + 3
            + (
                _OCCURRENCE_SELECTION_FINAL_CALL_BUDGET
                + _OCCURRENCE_ANSWER_FINAL_CALL_BUDGET
                if occurrence_state is not None
                else 0
            ),
        ):
            remaining = max(0, self.max_investigations - completed_investigations)
            occurrence_recovery_active = occurrence_recovery_pending
            occurrence_recovery_pending = False
            closure_repair_active = bool(
                closure_repair_pending
                and closure_repair_count < self.closure_repair_budget
            )
            closure_repair_pending = False
            if closure_repair_active:
                closure_repair_count += 1
                trace.append(
                    {
                        "type": "closure_repair",
                        "round": round_id,
                        "count": 1,
                        "repair_index": closure_repair_count,
                    }
                )
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
                require_item_provenance=self.require_item_provenance,
                surfaced_observation_ids=surfaced_observation_ids,
            )
            runtime_catalog = RuntimeEvidenceCatalog.build(document, observation_log)
            if self.controller_mode == "frozen_baseline":
                status = _frozen_mechanical_status(status, runtime_status)
            else:
                status["evidence_state_mode"] = self.evidence_state_mode
                status["evidence_control_mode"] = self.evidence_control_mode
            visible_occurrence_ids = _visible_occurrence_ids(status)
            if visible_occurrence_ids and not treatment_eligible_recorded:
                trace.append(
                    {
                        "type": "occurrence_treatment_eligible",
                        "round": round_id,
                        "method_arm": self.occurrence_method_arm,
                        "visible_occurrence_count": len(visible_occurrence_ids),
                    }
                )
                treatment_eligible_recorded = True
            treatment_surface = _occurrence_treatment_surface(status)
            if treatment_surface is not None and not treatment_exposed_recorded:
                trace.append(
                    {
                        "type": "occurrence_treatment_exposed",
                        "round": round_id,
                        "method_arm": self.occurrence_method_arm,
                        **treatment_surface,
                    }
                )
                treatment_exposed_recorded = True
            if isinstance(occurrence_state, OccurrenceResolutionStateV1):
                occurrence_state.sync_visible(visible_occurrence_ids)
                occurrence_state.save(occurrence_state_path)
                status["occurrence_resolution_state"] = occurrence_state.to_dict()
            elif isinstance(occurrence_state, OccurrenceResolutionStateV2):
                previous_locator_rows = _occurrence_locator_statuses(
                    occurrence_state,
                    observation_log.rows,
                )
                _record_inspected_occurrence_locators(
                    trace,
                    locator_terminal_outcomes,
                    round_id=round_id,
                    locator_rows=previous_locator_rows,
                )
                occurrence_state.sync_sets(_scoped_occurrence_sets(status))
                current_locator_pairs = {
                    _occurrence_locator_pair(locator)
                    for locator in occurrence_state.active_locators()
                }
                for locator in previous_locator_rows:
                    pair = _occurrence_locator_pair(locator)
                    if pair in current_locator_pairs or pair in locator_terminal_outcomes:
                        continue
                    previous_set = occurrence_state.sets.get(
                        str(locator.get("set_id", "") or "")
                    )
                    retired = bool(
                        previous_set is not None
                        and previous_set.lifecycle == "retired"
                    )
                    _record_occurrence_locator_outcome(
                        trace,
                        locator_terminal_outcomes,
                        round_id=round_id,
                        locator=locator,
                        outcome=(
                            "released_on_set_retirement"
                            if retired
                            else "released_by_revision"
                        ),
                        reason=(
                            "set_retired"
                            if retired
                            else "candidate_set_revision"
                        ),
                        revision_op=("" if retired else "sync_visible_set"),
                    )
                if occurrence_state.activated:
                    if not resolution_activated_recorded:
                        trace.append(
                            {
                                "type": "occurrence_resolution_activated",
                                "round": round_id,
                                "method_arm": self.occurrence_method_arm,
                                "active_set_id": occurrence_state.active_set_id,
                                "candidate_count": occurrence_state.candidate_count,
                                "arbitration_required": (
                                    occurrence_state.arbitration_required
                                ),
                            }
                        )
                        resolution_activated_recorded = True
                    if (
                        occurrence_state.arbitration_required
                        and not arbitration_activated_recorded
                    ):
                        active = occurrence_state.active_set
                        trace.append(
                            {
                                "type": "occurrence_arbitration_activated",
                                "round": round_id,
                                "method_arm": self.occurrence_method_arm,
                                "active_set_id": occurrence_state.active_set_id,
                                "candidate_count": (
                                    len(active.candidates) if active is not None else 0
                                ),
                            }
                        )
                        arbitration_activated_recorded = True
                    if (
                        occurrence_state.sufficiency_enabled
                        and not sufficiency_activated_recorded
                    ):
                        trace.append(
                            {
                                "type": "occurrence_sufficiency_activated",
                                "round": round_id,
                                "method_arm": self.occurrence_method_arm,
                                "active_set_id": occurrence_state.active_set_id,
                                "candidate_count": occurrence_state.candidate_count,
                            }
                        )
                        sufficiency_activated_recorded = True
                    occurrence_state.save(occurrence_state_path)
                    status["occurrence_resolution_state"] = occurrence_state.to_dict()
                    locator_statuses = _occurrence_locator_statuses(
                        occurrence_state,
                        observation_log.rows,
                    )
                    _record_inspected_occurrence_locators(
                        trace,
                        locator_terminal_outcomes,
                        round_id=round_id,
                        locator_rows=locator_statuses,
                    )
                    if self.occurrence_method_arm in {"a3", "a4"} and locator_statuses:
                        locator_statuses = tuple(
                            {
                                **row,
                                **(
                                    {
                                        "inspection_status": (
                                            "released_unexecuted"
                                        ),
                                        "accounting_outcome": (
                                            locator_terminal_outcomes.get(
                                                _occurrence_locator_pair(row)
                                            )
                                        ),
                                    }
                                    if locator_terminal_outcomes.get(
                                        _occurrence_locator_pair(row), ""
                                    ).startswith("released_")
                                    else {}
                                ),
                            }
                            for row in locator_statuses
                        )
                        status["selected_occurrence_locators"] = list(
                            locator_statuses
                        )
                        status["active_occurrence_locators"] = [
                            row
                            for row in locator_statuses
                            if not row["inspected"]
                            and not str(
                                row.get("accounting_outcome", "") or ""
                            ).startswith("released_")
                        ]
            pending_locator_rows = tuple(
                row
                for row in tuple(status.get("active_occurrence_locators", ()) or ())
                if isinstance(row, Mapping)
            )
            budget_exhausted = (
                round_id > self.semantic_round_budget or remaining <= 0
            )
            force_finalize = (
                budget_exhausted or occurrence_recovery_active
            ) and not closure_repair_active
            pending_pairs = sorted(
                {
                    _occurrence_locator_pair(row)
                    for row in pending_locator_rows
                    if all(_occurrence_locator_pair(row))
                }
            )
            if (
                force_finalize
                and budget_exhausted
                and pending_pairs
                and not locator_budget_exhaustion_recorded
            ):
                trace.append(
                    {
                        "type": "occurrence_locator_budget_exhausted_at_finalize",
                        "round": round_id,
                        "method_arm": self.occurrence_method_arm,
                        "pending_locator_count": len(pending_pairs),
                        "pending_locator_pairs": [
                            list(pair) for pair in pending_pairs
                        ],
                    }
                )
                locator_budget_exhaustion_recorded = True
            if force_finalize and pending_locator_rows:
                for locator in pending_locator_rows:
                    _record_occurrence_locator_outcome(
                        trace,
                        locator_terminal_outcomes,
                        round_id=round_id,
                        locator=locator,
                        outcome="released_at_budget_exhaustion",
                        reason="budget_exhausted_at_finalize",
                    )
                released_pairs = set(pending_pairs)
                status["selected_occurrence_locators"] = [
                    {
                        **row,
                        **(
                            {
                                "inspection_status": "released_unexecuted",
                                "accounting_outcome": (
                                    "released_at_budget_exhaustion"
                                ),
                                "release_reason": (
                                    "budget_exhausted_at_finalize"
                                ),
                            }
                            if _occurrence_locator_pair(row) in released_pairs
                            else {}
                        ),
                    }
                    for row in tuple(
                        status.get("selected_occurrence_locators", ()) or ()
                    )
                    if isinstance(row, Mapping)
                ]
                status["active_occurrence_locators"] = []
                pending_locator_rows = ()
            if self.controller_mode != "frozen_baseline":
                status["closure_repair_active"] = closure_repair_active
                status["closure_repair_count"] = closure_repair_count
            if force_finalize:
                forced_decision_calls += 1
                if occurrence_state is not None and _occurrence_lifecycle_active(
                    occurrence_state
                ):
                    if _occurrence_resolution_complete(occurrence_state):
                        occurrence_answer_final_calls += 1
                    else:
                        occurrence_selection_final_calls += 1
            if occurrence_state is not None and _occurrence_lifecycle_active(
                occurrence_state
            ):
                if _occurrence_resolution_complete(occurrence_state):
                    final_retry_available = (
                        force_finalize
                        and occurrence_answer_final_calls
                        < _OCCURRENCE_ANSWER_FINAL_CALL_BUDGET
                    )
                else:
                    final_retry_available = (
                        force_finalize
                        and occurrence_selection_final_calls
                        < _OCCURRENCE_SELECTION_FINAL_CALL_BUDGET
                    )
            else:
                final_retry_available = force_finalize and forced_decision_calls < 2
            control_attempt = 0
            control_retries_used = 0
            decision_had_control_retry = False
            retry_occurrence_next_round = False
            decision: ReasonerDecision | None = None
            requested_rows: tuple[Mapping[str, Any], ...] = ()
            while True:
                working_view = (
                    runtime_catalog.render(document, feedback=feedback)
                    if self.evidence_state_mode == "runtime_derived"
                    else render_frozen_working_view(
                        document,
                        observation_log,
                        requested_observations=requested_observations,
                        feedback=feedback,
                    )
                    if self.controller_mode == "frozen_baseline"
                    else render_working_view(
                        document,
                        observation_log,
                        requested_observations=requested_observations,
                        feedback=feedback,
                    )
                )
                surfaced_observation_ids.update(observation_log.attempt_ids)
                raw_decision = self.reasoner.decide(
                    question=workspace.case.question,
                    options=dict(workspace.case.options),
                    workspace_overview=overview,
                    working_document_view=working_view,
                    mechanical_status=status,
                    remaining_budget=remaining,
                    force_finalize=force_finalize,
                    final_attempt=forced_decision_calls if force_finalize else 0,
                    answer_policy=self.answer_policy,
                    evidence_control_mode=self.evidence_control_mode,
                    evidence_state_mode=self.evidence_state_mode,
                    semantic_round=round_id,
                    control_attempt=control_attempt,
                    control_retry=control_attempt > 0,
                    closure_repair=closure_repair_active,
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
                raw_schema_errors = _schema_error_rows(
                    decision_metadata.get("decision_schema_errors", ())
                )
                schema_errors = (
                    [
                        error
                        for error in raw_schema_errors
                        if str(error.get("code", "")).startswith("occurrence_")
                    ]
                    if self.controller_mode == "frozen_baseline"
                    else list(raw_schema_errors)
                )
                try:
                    parsed_decision = _decision(raw_decision)
                except (TypeError, ValueError) as exc:
                    parsed_decision = None
                    schema_errors.append(
                        {"code": "invalid_decision_payload", "detail": str(exc)}
                    )
                if parsed_decision is not None:
                    if parsed_decision.occurrence_ops:
                        if occurrence_state is None:
                            schema_errors.append(
                                {"code": "occurrence_ops_not_enabled"}
                            )
                        else:
                            schema_errors.extend(
                                occurrence_state.validate_ops(
                                    parsed_decision.occurrence_ops
                                )
                            )
                    if occurrence_state is not None:
                        require_terminal_answer = bool(
                            isinstance(occurrence_state, OccurrenceResolutionStateV2)
                            and _occurrence_resolution_complete(occurrence_state)
                            and not pending_locator_rows
                        )
                        schema_errors.extend(
                            _occurrence_answer_errors(
                                parsed_decision,
                                occurrence_state,
                                require_selection=force_finalize,
                                require_answer=(
                                    force_finalize or require_terminal_answer
                                ),
                            )
                        )
                    if (
                        self.occurrence_method_arm in {"a3", "a4"}
                        and isinstance(occurrence_state, OccurrenceResolutionStateV2)
                    ):
                        schema_errors.extend(
                            _actionable_locator_errors(
                                parsed_decision,
                                tuple(
                                    row
                                    for row in tuple(
                                        status.get("active_occurrence_locators", ())
                                        or ()
                                    )
                                    if isinstance(row, Mapping)
                                ),
                                investigation_budget_remaining=remaining,
                                force_finalize=force_finalize,
                            )
                        )
                    if self.evidence_state_mode == "runtime_derived":
                        parsed_decision, handle_errors, ignored_state_ops = _resolve_runtime_decision(
                            parsed_decision,
                            runtime_catalog,
                            document,
                        )
                        schema_errors.extend(handle_errors)
                        if ignored_state_ops:
                            trace.append(
                                {
                                    "type": "runtime_state_ops_ignored",
                                    "round": round_id,
                                    "operations": list(ignored_state_ops),
                                }
                            )
                    if self.controller_mode != "frozen_baseline":
                        schema_errors.extend(
                            _decision_preflight(
                                parsed_decision,
                                closure_repair=closure_repair_active,
                                runtime_derived=self.evidence_state_mode
                                == "runtime_derived",
                            )
                        )
                    schema_errors.extend(
                        inspection_mode_policy_errors(
                            parsed_decision.tasks,
                            allowed_modes=self.allowed_inspection_modes,
                        )
                    )
                    if (
                        parsed_decision.action == "answer"
                        and parsed_decision.answer
                        and not any(
                            str(error.get("code", "")).startswith(
                                "occurrence_"
                            )
                            for error in schema_errors
                        )
                    ):
                        latest_answer_candidate = parsed_decision
                trace.append(
                    {
                        "type": "reasoner_decision_attempt",
                        "round": round_id,
                        "semantic_round": round_id,
                        "control_attempt": control_attempt,
                        "force_finalize": force_finalize,
                        "action": (
                            parsed_decision.action
                            if parsed_decision is not None
                            else ""
                        ),
                        "schema_valid": not raw_schema_errors and not schema_errors,
                        "errors": list(raw_schema_errors or schema_errors),
                    }
                )
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
                    if (
                        self.evidence_control_mode == "shadow"
                        and parsed_decision is not None
                        and parsed_decision.action == "answer"
                        and parsed_decision.answer
                        and not any(
                            str(error.get("code", "")).startswith(
                                "occurrence_"
                            )
                            for error in schema_errors
                        )
                    ):
                        trace.append(
                            {
                                "type": "shadow_prediction_preserved",
                                "round": round_id,
                                "stage": "decision_preflight",
                                "grounding_errors": schema_errors,
                            }
                        )
                        decision = parsed_decision
                        rounds_run = round_id
                        break
                    if control_retries_used >= self.control_retry_budget:
                        if (
                            occurrence_state is not None
                            and _occurrence_repair_available(
                                occurrence_state,
                                selection_final_calls=occurrence_selection_final_calls,
                                answer_final_calls=occurrence_answer_final_calls,
                            )
                            and grant_occurrence_recovery(
                                round_id=round_id,
                                stage="decision_preflight",
                            )
                        ):
                            trace.append(
                                {
                                    "type": "occurrence_lifecycle_repair_scheduled",
                                    "round": round_id,
                                    "stage": "decision_preflight",
                                    "errors": schema_errors,
                                }
                            )
                            feedback = _control_retry_feedback(
                                schema_errors,
                                revision=document.revision,
                                previous_feedback=feedback,
                            )
                            _append_contradictory_gate_state(
                                trace,
                                feedback,
                                round_id=round_id,
                            )
                            retry_occurrence_next_round = True
                            break
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
                    _append_contradictory_gate_state(
                        trace,
                        feedback,
                        round_id=round_id,
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
                    observation_rows=observation_log.rows,
                    require_item_provenance=self.require_item_provenance,
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
                occurrence_apply_result: dict[str, Any] = {
                    "accepted": not bool(decision.occurrence_ops),
                    "errors": [],
                    "applied": [],
                }
                previous_occurrence_locators = (
                    occurrence_state.active_locators()
                    if isinstance(occurrence_state, OccurrenceResolutionStateV2)
                    else ()
                )
                # Occurrence state is a separate runtime-owned transaction. A valid
                # selection must survive an unrelated working-document rejection so
                # the following forced call can complete the answer-only phase.
                if occurrence_state is not None:
                    occurrence_apply_result = occurrence_state.apply_ops(
                        decision.occurrence_ops
                    )
                    occurrence_state.save(occurrence_state_path)
                for operation_index, operation in enumerate(
                    tuple(occurrence_apply_result.get("applied", ()) or ())
                ):
                    if not isinstance(operation, Mapping) or str(
                        operation.get("op", "") or ""
                    ) != "assess_sufficiency":
                        continue
                    constraints = tuple(
                        row
                        for row in tuple(
                            operation.get("constraints_checked", ()) or ()
                        )
                        if isinstance(row, Mapping)
                    )
                    trace.append(
                        {
                            "type": "occurrence_sufficiency_decision",
                            "round": round_id,
                            "occurrence_op_index": operation_index,
                            "set_id": str(operation.get("set_id", "") or ""),
                            "candidate_count": (
                                occurrence_state.candidate_count
                                if isinstance(
                                    occurrence_state, OccurrenceResolutionStateV2
                                )
                                else 0
                            ),
                            "verdict": str(operation.get("verdict", "") or ""),
                            "constraints_checked": [
                                str(row.get("constraint_id", "") or "")
                                for row in constraints
                            ],
                            "constraint_types": [
                                str(row.get("constraint_type", "") or "")
                                for row in constraints
                            ],
                            "sufficient_occurrence_ids": list(
                                operation.get("sufficient_occurrence_ids", ()) or ()
                            ),
                        }
                    )
                if (
                    isinstance(occurrence_state, OccurrenceResolutionStateV2)
                    and occurrence_apply_result["accepted"]
                ):
                    current_occurrence_locator_pairs = {
                        _occurrence_locator_pair(locator)
                        for locator in occurrence_state.active_locators()
                    }
                    for locator in previous_occurrence_locators:
                        pair = _occurrence_locator_pair(locator)
                        if (
                            pair in current_occurrence_locator_pairs
                            or pair in locator_terminal_outcomes
                        ):
                            continue
                        revision_op = _revision_release_op(
                            locator,
                            tuple(
                                operation
                                for operation in decision.occurrence_ops
                                if isinstance(operation, Mapping)
                            ),
                        )
                        _record_occurrence_locator_outcome(
                            trace,
                            locator_terminal_outcomes,
                            round_id=round_id,
                            locator=locator,
                            outcome="released_by_revision",
                            reason=f"resolution_revision:{revision_op}",
                            revision_op=revision_op,
                        )
                occurrence_selection_committed = bool(
                    occurrence_apply_result["accepted"]
                    and any(
                        str(
                            operation.get("op", operation.get("type", ""))
                            or ""
                        ).casefold()
                        == "select"
                        for operation in decision.occurrence_ops
                        if isinstance(operation, Mapping)
                    )
                )
                occurrence_resolution_committed = bool(
                    occurrence_apply_result["accepted"]
                    and any(
                        str(
                            operation.get("op", operation.get("type", ""))
                            or ""
                        ).casefold()
                        in {"select", "no_match"}
                        for operation in decision.occurrence_ops
                        if isinstance(operation, Mapping)
                    )
                )
                if occurrence_selection_committed:
                    if isinstance(occurrence_state, OccurrenceResolutionStateV1):
                        grant_occurrence_recovery(
                            round_id=round_id,
                            stage="occurrence_selection_committed",
                        )
                    elif (
                        isinstance(occurrence_state, OccurrenceResolutionStateV2)
                        and force_finalize
                    ):
                        grant_occurrence_recovery(
                            round_id=round_id,
                            stage="occurrence_selection_committed",
                        )
                elif (
                    occurrence_resolution_committed
                    and isinstance(occurrence_state, OccurrenceResolutionStateV2)
                    and force_finalize
                ):
                    grant_occurrence_recovery(
                        round_id=round_id,
                        stage="occurrence_resolution_committed",
                    )
                premature_occurrence_commit = bool(
                    occurrence_state is not None
                    and decision.action == "answer"
                    and decision.answer
                    and _occurrence_answer_is_premature(occurrence_state)
                )
                trace.append(
                    {
                        "type": "reasoner_decision",
                        "round": round_id,
                        "semantic_round": round_id,
                        "control_attempt": control_attempt,
                        "control_retry_count": control_retries_used,
                        "semantic_committed": (
                            apply_result.accepted
                            or self.evidence_state_mode == "runtime_derived"
                        ),
                        "action": decision.action,
                        "tasks": [_task_descriptor(task) for task in decision.tasks],
                        "workspace_revision": document.revision,
                        "workspace_ops_accepted": apply_result.accepted,
                        "workspace_errors": list(apply_result.errors),
                        "occurrence_ops": [
                            dict(operation) for operation in decision.occurrence_ops
                        ],
                        "occurrence_ops_accepted": occurrence_apply_result[
                            "accepted"
                        ],
                        "occurrence_selection_committed": (
                            occurrence_selection_committed
                        ),
                        "occurrence_resolution_committed": (
                            occurrence_resolution_committed
                        ),
                        "occurrence_state_revision": (
                            occurrence_state.revision
                            if occurrence_state is not None
                            else None
                        ),
                        "selected_occurrence_id": _selected_occurrence_id(
                            occurrence_state
                        ),
                        "selected_occurrence_ids": list(
                            _selected_occurrence_ids(occurrence_state)
                        ),
                        "active_occurrence_set_id": (
                            occurrence_state.active_set_id
                            if isinstance(
                                occurrence_state, OccurrenceResolutionStateV2
                            )
                            else ""
                        ),
                        "active_occurrence_locator_count": len(
                            tuple(status.get("active_occurrence_locators", ()) or ())
                        ),
                        "occurrence_resolution_state_exposed": bool(
                            "occurrence_resolution_state" in status
                        ),
                        "premature_occurrence_commit": premature_occurrence_commit,
                        "supporting_claim_ids": list(decision.supporting_claim_ids),
                        "supporting_item_ids": list(decision.supporting_item_ids),
                        "supports_requirement_ids": list(decision.supports_requirement_ids),
                        "remaining_budget": remaining,
                        "force_finalize": force_finalize,
                        "final_attempt": forced_decision_calls if force_finalize else 0,
                        "answer_workspace_commit": answer_workspace_commit,
                        "closure_repair": closure_repair_active,
                        "state_mutation_op_count": sum(
                            str(operation.get("op", operation.get("type", "")) or "").casefold()
                            in {"add_obligation", "set_obligation_status", "add_temporal_scope", "set_cue_status"}
                            for operation in decision.workspace_ops
                        ),
                        "prompt_char_count": int(
                            decision_metadata.get("prompt_char_count", 0) or 0
                        ),
                        "prompt_digest": str(
                            decision_metadata.get("prompt_digest", "") or ""
                        ),
                        "mechanical_status_digest": prompt_digest(
                            json.dumps(
                                to_jsonable(status),
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                        ),
                        "prompt_schema_token_cost": int(
                            decision_metadata.get("prompt_schema_token_cost", 0) or 0
                        ),
                    }
                )
                if apply_result.accepted:
                    rounds_run = round_id
                    break

                if self.evidence_state_mode == "runtime_derived":
                    trace.append(
                        {
                            "type": "optional_reasoning_memory_rejected",
                            "round": round_id,
                            "errors": list(apply_result.errors),
                        }
                    )
                    rounds_run = round_id
                    break

                if (
                    self.evidence_control_mode == "shadow"
                    and decision.action == "answer"
                    and decision.answer
                ):
                    trace.append(
                        {
                            "type": "shadow_prediction_preserved",
                            "round": round_id,
                            "stage": "workspace_transaction",
                            "grounding_errors": list(apply_result.errors),
                        }
                    )
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
                    if (
                        occurrence_state is not None
                        and _occurrence_repair_available(
                            occurrence_state,
                            selection_final_calls=occurrence_selection_final_calls,
                            answer_final_calls=occurrence_answer_final_calls,
                        )
                        and grant_occurrence_recovery(
                            round_id=round_id,
                            stage="workspace_transaction",
                        )
                    ):
                        trace.append(
                            {
                                "type": "occurrence_lifecycle_repair_scheduled",
                                "round": round_id,
                                "stage": "workspace_transaction",
                                "errors": workspace_errors,
                            }
                        )
                        feedback = _control_retry_feedback(
                            workspace_errors,
                            revision=document.revision,
                            previous_feedback=feedback,
                        )
                        _append_contradictory_gate_state(
                            trace,
                            feedback,
                            round_id=round_id,
                        )
                        retry_occurrence_next_round = True
                        decision = None
                        break
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
                _append_contradictory_gate_state(
                    trace,
                    feedback,
                    round_id=round_id,
                )

            if retry_occurrence_next_round:
                continue
            if protocol_exhausted:
                break
            if decision is None:
                break

            if decision.action == "answer":
                candidate = replace(
                    decision,
                    citations=_answer_citations(
                        decision,
                        document,
                        evidence_store.records,
                        observation_rows=observation_log.rows,
                    ),
                )
                if self.evidence_state_mode == "runtime_derived":
                    _apply_runtime_answer_state(document, candidate, runtime_catalog)
                    document.save(document_path)
                validation = _validate_answer(
                    candidate,
                    document,
                    observation_log.attempt_ids,
                    workspace.case.options,
                    supporting_observation_ids=_supporting_observation_ids(observation_log),
                    require_obligation_coverage=self.require_obligation_coverage,
                    observation_rows=observation_log.rows,
                    require_item_provenance=self.require_item_provenance,
                    temporal_scope_resolutions=_temporal_scope_resolution_map(
                        document,
                        observation_log,
                    ),
                    unconsumed_observation_ids=tuple(
                        set(observation_log.attempt_ids) - surfaced_observation_ids
                    ),
                    require_evidence_kind_requirements=self.require_evidence_kind_requirements,
                )
                trace.append({"type": "reference_integrity_check", "round": round_id, **validation.to_dict()})
                if validation.passed:
                    final_answer = candidate
                    break
                if self.evidence_control_mode == "shadow":
                    trace.append(
                        {
                            "type": "shadow_prediction_preserved",
                            "round": round_id,
                            "stage": "closure_validation",
                            "grounding_errors": list(validation.errors),
                        }
                    )
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
                if (
                    force_finalize
                    and closure_repair_count < self.closure_repair_budget
                    and _closure_repairable(validation)
                ):
                    closure_repair_pending = True
                if force_finalize and not final_retry_available:
                    if closure_repair_pending:
                        continue
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
                if (
                    force_finalize
                    and not final_retry_available
                    and not occurrence_resolution_committed
                ):
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
                observation_rows=observation_log.rows,
                temporal_scope_ids=tuple(document.temporal_scopes),
                temporal_scope_resolutions={
                    str(row.get("scope_id", "") or ""): row
                    for row in tuple(status.get("temporal_scope_statuses", ()) or ())
                    if isinstance(row, Mapping)
                    and str(row.get("scope_id", "") or "")
                },
                cue_states={
                    cue_id: state.to_dict()
                    for cue_id, state in document.cue_states.items()
                },
            )
            if self.evidence_state_mode == "runtime_derived":
                tasks = _expand_runtime_tasks(tasks)
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
            if self.evidence_state_mode == "runtime_derived":
                _advance_runtime_task_states(
                    document,
                    tasks,
                    batch,
                    observation_log,
                )
                document.save(document_path)
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
        preserve_raw_prediction = (
            self.evidence_control_mode == "shadow"
            or self.answer_policy == "benchmark_best_effort"
        )
        selected = final_answer or (
            latest_answer_candidate
            if preserve_raw_prediction and latest_answer_candidate is not None
            else empty_answer
        )
        selected_option = _letter(selected.answer) or _option_letter_from_answer(
            selected.answer,
            workspace.case.options,
        )
        schema_answer_present = bool(selected.answer) if not workspace.case.options else bool(selected_option)
        candidate_present = bool(selected.answer)
        preserve_candidate = preserve_raw_prediction and candidate_present
        validation = _validate_answer(
            selected,
            document,
            observation_log.attempt_ids,
            workspace.case.options,
            supporting_observation_ids=_supporting_observation_ids(observation_log),
            require_obligation_coverage=self.require_obligation_coverage,
            observation_rows=observation_log.rows,
            require_item_provenance=self.require_item_provenance,
            temporal_scope_resolutions=_temporal_scope_resolution_map(
                document,
                observation_log,
            ),
            unconsumed_observation_ids=tuple(
                set(observation_log.attempt_ids) - surfaced_observation_ids
            ),
            require_evidence_kind_requirements=self.require_evidence_kind_requirements,
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
                observation_rows=observation_log.rows,
                require_item_provenance=self.require_item_provenance,
                temporal_scope_resolutions=_temporal_scope_resolution_map(
                    document,
                    observation_log,
                ),
                unconsumed_observation_ids=tuple(
                    set(observation_log.attempt_ids) - surfaced_observation_ids
                ),
                require_evidence_kind_requirements=self.require_evidence_kind_requirements,
            )
        )
        if schema_answer_present or preserve_candidate:
            answer = selected.answer
            returned_answer_present = True
            citations = _answer_citations(
                selected,
                document,
                evidence_store.records,
                observation_rows=observation_log.rows,
            )
            reference_valid = validation.passed
            reference_reason = validation.reason
        else:
            answer = "No valid answer was returned."
            returned_answer_present = False
            citations = ()
            reference_valid = False
            reference_reason = "answer_missing" if not selected.answer else "invalid_option_answer"
        supporting_intervals = _merge_intervals(
            (
                *export_supporting_intervals(
                    document,
                    selected.supporting_claim_ids,
                    observation_log.rows,
                ),
                *export_item_supporting_intervals(
                    selected.supporting_item_ids,
                    observation_log.rows,
                ),
            )
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
                "supporting_item_ids": list(selected.supporting_item_ids),
                "supporting_attempt_ids": list(candidate_validation.cited_attempt_ids),
                "residual_uncertainty": selected.residual_uncertainty,
                "answer_owner": "reasoner",
                "framework_answer_mutation": False,
                "answer_policy": self.answer_policy,
                "evidence_control_mode": self.evidence_control_mode,
                "answer_present": returned_answer_present,
                "supporting_intervals": [list(item) for item in supporting_intervals],
                "obligation_summary": document.obligation_summary(),
                "temporal_scope_summary": _temporal_scope_summary(document, observation_log),
                "provenance_summary": document.provenance_summary(observation_log.rows),
                "cue_summary": document.cue_summary(observation_log.rows),
                "closure_validation": validation.to_dict(),
                "closure_repair_count": closure_repair_count,
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
            evidence_control_mode=self.evidence_control_mode,
            evidence_state_mode=self.evidence_state_mode,
            answer_present=returned_answer_present,
            candidate_answer=candidate_answer,
            verified_answer=verified_answer,
            verification_status=verification_status,
            blocking_reasons=blocking_reasons,
            supporting_claim_ids=selected.supporting_claim_ids,
            supporting_item_ids=selected.supporting_item_ids,
            supporting_attempt_ids=candidate_validation.cited_attempt_ids,
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


def _frozen_mechanical_status(
    status: Mapping[str, Any],
    runtime_status: Mapping[str, Any],
) -> dict[str, Any]:
    runtime_keys = {str(key) for key in runtime_status}
    return {
        str(key): value
        for key, value in status.items()
        if str(key) in _FROZEN_MECHANICAL_STATUS_KEYS
        or str(key) in runtime_keys
    }


def _visible_occurrence_ids(status: Mapping[str, Any]) -> tuple[str, ...]:
    occurrence_sets = tuple(status.get("caption_occurrence_sets", ()) or ())
    return tuple(
        dict.fromkeys(
            str(candidate.get("occurrence_id", "") or "")
            for occurrence_set in occurrence_sets
            if isinstance(occurrence_set, Mapping)
            for candidate in tuple(occurrence_set.get("candidates", ()) or ())
            if isinstance(candidate, Mapping)
            and str(candidate.get("occurrence_id", "") or "")
        )
    )


def _scoped_occurrence_sets(
    status: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    values: list[dict[str, Any]] = []
    for raw_set in tuple(status.get("caption_occurrence_sets", ()) or ()):
        if not isinstance(raw_set, Mapping):
            continue
        candidates = tuple(
            candidate
            for candidate in tuple(raw_set.get("candidates", ()) or ())
            if isinstance(candidate, Mapping)
            and str(candidate.get("occurrence_id", "") or "")
        )
        if len(candidates) < 1:
            continue
        values.append(
            {
                **dict(raw_set),
                "semantic_target": list(raw_set.get("semantic_target", ()) or ()),
                "candidates": [dict(candidate) for candidate in candidates],
            }
        )
    return tuple(values)


def _occurrence_lifecycle_active(
    state: OccurrenceResolutionStateV1 | OccurrenceResolutionStateV2,
) -> bool:
    if isinstance(state, OccurrenceResolutionStateV1):
        return state.selection_required
    return state.activated


def _occurrence_selection_required(
    state: OccurrenceResolutionStateV1 | OccurrenceResolutionStateV2,
) -> bool:
    return state.selection_required or (
        isinstance(state, OccurrenceResolutionStateV2) and state.search_required
    )


def _occurrence_resolution_complete(
    state: OccurrenceResolutionStateV1 | OccurrenceResolutionStateV2,
) -> bool:
    if isinstance(state, OccurrenceResolutionStateV1):
        return state.selected_occurrence_id in set(state.viable_occurrence_ids)
    active = state.active_set
    return bool(active is not None and active.resolution in {"selected", "no_match"})


def _selected_occurrence_ids(
    state: OccurrenceResolutionStateV1 | OccurrenceResolutionStateV2 | None,
) -> tuple[str, ...]:
    if isinstance(state, OccurrenceResolutionStateV1):
        return (state.selected_occurrence_id,) if state.selected_occurrence_id else ()
    if isinstance(state, OccurrenceResolutionStateV2):
        return state.selected_occurrence_ids
    return ()


def _selected_occurrence_id(
    state: OccurrenceResolutionStateV1 | OccurrenceResolutionStateV2 | None,
) -> str:
    selected = _selected_occurrence_ids(state)
    return selected[-1] if selected else ""


def _occurrence_answer_is_premature(
    state: OccurrenceResolutionStateV1 | OccurrenceResolutionStateV2,
) -> bool:
    if isinstance(state, OccurrenceResolutionStateV1):
        return bool(
            len(state.viable_occurrence_ids) > 1
            and not state.selected_occurrence_id
        )
    return state.selection_required or state.search_required


def _occurrence_locator_pair(
    locator: Mapping[str, Any],
) -> tuple[str, str]:
    return (
        str(
            locator.get("locator_attempt_id", locator.get("set_id", ""))
            or ""
        ),
        str(locator.get("occurrence_id", "") or ""),
    )


def _record_occurrence_locator_outcome(
    trace: list[Mapping[str, Any]],
    outcomes: dict[tuple[str, str], str],
    *,
    round_id: int,
    locator: Mapping[str, Any],
    outcome: str,
    reason: str = "",
    revision_op: str = "",
) -> None:
    pair = _occurrence_locator_pair(locator)
    if not all(pair):
        return
    previous = outcomes.get(pair)
    if previous == outcome:
        return
    if previous:
        trace.append(
            {
                "type": "occurrence_locator_accounting_conflict",
                "round": round_id,
                "locator_attempt_id": pair[0],
                "occurrence_id": pair[1],
                "previous_outcome": previous,
                "new_outcome": outcome,
            }
        )
        return
    outcomes[pair] = outcome
    if outcome == "inspected":
        trace.append(
            {
                "type": "occurrence_locator_inspected",
                "round": round_id,
                "locator_attempt_id": pair[0],
                "occurrence_id": pair[1],
                "outcome": outcome,
            }
        )
        return
    event: dict[str, Any] = {
        "type": "occurrence_locator_released_unexecuted",
        "round": round_id,
        "locator_attempt_id": pair[0],
        "occurrence_id": pair[1],
        "outcome": outcome,
        "reason": reason,
    }
    if revision_op:
        event["revision_op"] = revision_op
    trace.append(event)


def _record_inspected_occurrence_locators(
    trace: list[Mapping[str, Any]],
    outcomes: dict[tuple[str, str], str],
    *,
    round_id: int,
    locator_rows: Sequence[Mapping[str, Any]],
) -> None:
    for locator in locator_rows:
        if locator.get("inspected") is True:
            _record_occurrence_locator_outcome(
                trace,
                outcomes,
                round_id=round_id,
                locator=locator,
                outcome="inspected",
            )


def _revision_release_op(
    locator: Mapping[str, Any],
    operations: Sequence[Mapping[str, Any]],
) -> str:
    set_id, occurrence_id = _occurrence_locator_pair(locator)
    for operation in operations:
        if not isinstance(operation, Mapping):
            continue
        operation_set_id = str(
            operation.get("set_id", operation.get("locator_attempt_id", ""))
            or ""
        )
        if operation_set_id != set_id:
            continue
        op = str(
            operation.get("op", operation.get("type", "")) or ""
        ).casefold()
        operation_occurrence_id = str(operation.get("occurrence_id", "") or "")
        if op in {"defer", "no_match"} or (
            op in {"eliminate", "reopen"}
            and operation_occurrence_id == occurrence_id
        ):
            return op
    return "state_revision"


def _occurrence_locator_statuses(
    state: OccurrenceResolutionStateV2,
    observation_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    inspected = {
        (
            str(binding.get("locator_attempt_id", "") or ""),
            str(binding.get("occurrence_id", "") or ""),
        )
        for row in observation_rows
        if isinstance(row, Mapping)
        for config in (row.get("sampling_config"),)
        if isinstance(config, Mapping)
        for binding in (config.get("candidate_binding"),)
        if isinstance(binding, Mapping)
    }
    locators = state.active_locators()
    if any(
        str(locator.get("set_id", "") or "") != state.active_set_id
        for locator in locators
    ):
        raise RuntimeError(
            "occurrence answer and locator gates must share the active set"
        )
    return tuple(
        {
            **locator,
            "inspected": (
                str(locator["locator_attempt_id"]),
                str(locator["occurrence_id"]),
            )
            in inspected,
            "inspection_status": (
                "inspected"
                if (
                    str(locator["locator_attempt_id"]),
                    str(locator["occurrence_id"]),
                )
                in inspected
                else "pending_inspection"
            ),
        }
        for locator in locators
    )


def _actionable_locator_errors(
    decision: ReasonerDecision,
    pending_locators: Sequence[Mapping[str, Any]],
    *,
    investigation_budget_remaining: int,
    force_finalize: bool = False,
) -> list[dict[str, Any]]:
    if force_finalize or not pending_locators:
        return []
    pending = {
        (
            str(locator.get("locator_attempt_id", "") or ""),
            str(locator.get("occurrence_id", "") or ""),
        )
        for locator in pending_locators
    }
    revises_resolution = any(
        str(operation.get("op", operation.get("type", "")) or "").casefold()
        in {"eliminate", "reopen", "defer", "no_match"}
        for operation in decision.occurrence_ops
        if isinstance(operation, Mapping)
    )
    if revises_resolution:
        return []
    if investigation_budget_remaining <= 0:
        raise RuntimeError(
            "pending actionable locators must enter finalization when the "
            "investigation budget is exhausted"
        )
    if decision.action != "investigate":
        return [
            {
                "code": "occurrence_locator_inspection_required",
                "pending_locator_count": len(pending),
            }
        ]
    task_bindings = {
        (task.locator_attempt_id, task.occurrence_id)
        for task in decision.tasks
        if task.inspection_mode == "window"
        and task.locator_attempt_id
        and task.occurrence_id
    }
    if not (task_bindings & pending):
        return [
            {
                "code": "occurrence_locator_binding_required",
                "pending_locator_count": len(pending),
            }
        ]
    unbound_windows = [
        task.query_id
        for task in decision.tasks
        if task.inspection_mode == "window"
        and not (task.locator_attempt_id and task.occurrence_id)
    ]
    if unbound_windows:
        return [
            {
                "code": "occurrence_locator_unbound_window_forbidden",
                "requested_task_ids": unbound_windows,
            }
        ]
    return []


def _occurrence_answer_errors(
    decision: ReasonerDecision,
    state: OccurrenceResolutionStateV1 | OccurrenceResolutionStateV2,
    *,
    require_selection: bool = False,
    require_answer: bool = False,
) -> list[dict[str, Any]]:
    if isinstance(state, OccurrenceResolutionStateV2):
        return _scoped_occurrence_answer_errors(
            decision,
            state,
            require_resolution=require_selection,
            require_answer=require_answer,
        )
    viable = set(state.viable_occurrence_ids)
    submits_selection = any(
        str(operation.get("op", operation.get("type", "")) or "").casefold()
        == "select"
        for operation in decision.occurrence_ops
        if isinstance(operation, Mapping)
    )
    if (
        require_selection
        and state.selection_required
        and state.selected_occurrence_id not in viable
        and not submits_selection
    ):
        return [
            {
                "code": "occurrence_selection_required",
                "viable_occurrence_count": len(viable),
                "selection_must_precede_answer": True,
            }
        ]
    if (
        require_answer
        and state.selected_occurrence_id in viable
        and decision.action != "answer"
    ):
        return [
            {
                "code": "occurrence_answer_required_after_selection",
                "selected_occurrence_id": state.selected_occurrence_id,
            }
        ]
    if decision.action != "answer" or not decision.answer:
        return []
    if not state.selection_required or state.selected_occurrence_id in viable:
        return []
    return [
        {
            "code": "occurrence_selection_required",
            "viable_occurrence_count": len(viable),
            "selection_must_precede_answer": True,
        }
    ]


def _scoped_occurrence_answer_errors(
    decision: ReasonerDecision,
    state: OccurrenceResolutionStateV2,
    *,
    require_resolution: bool,
    require_answer: bool,
) -> list[dict[str, Any]]:
    active = state.active_set
    if active is None:
        return []
    operation_names = {
        str(operation.get("op", operation.get("type", "")) or "").casefold()
        for operation in decision.occurrence_ops
        if isinstance(operation, Mapping)
    }
    submits_resolution = bool(operation_names & {"select", "no_match"})
    if active.resolution == "deferred":
        if decision.action == "answer" and decision.answer:
            return [
                {
                    "code": "occurrence_search_required",
                    "set_id": active.set_id,
                    "no_match_allowed": True,
                }
            ]
        if require_resolution and "no_match" not in operation_names:
            return [
                {
                    "code": "occurrence_no_match_required_at_finalization",
                    "set_id": active.set_id,
                }
            ]
        return []
    if active.resolution == "unresolved":
        if (
            require_resolution
            and not submits_resolution
        ) or (decision.action == "answer" and decision.answer):
            return [
                {
                    "code": (
                        "occurrence_sufficiency_resolution_required"
                        if state.sufficiency_enabled
                        else "occurrence_resolution_required"
                    ),
                    "set_id": active.set_id,
                    "viable_occurrence_count": len(active.viable_occurrence_ids),
                    "selection_must_precede_answer": True,
                    "no_match_allowed": True,
                    "sufficiency_required": state.sufficiency_required,
                }
            ]
        return []
    if require_answer and decision.action != "answer":
        return [
            {
                "code": "occurrence_answer_required_after_resolution",
                "set_id": active.set_id,
                "resolution": active.resolution,
            }
        ]
    return []


def _occurrence_repair_available(
    state: OccurrenceResolutionStateV1 | OccurrenceResolutionStateV2,
    *,
    selection_final_calls: int,
    answer_final_calls: int,
) -> bool:
    if not _occurrence_lifecycle_active(state):
        return False
    if _occurrence_resolution_complete(state):
        return answer_final_calls < _OCCURRENCE_ANSWER_FINAL_CALL_BUDGET
    return selection_final_calls < _OCCURRENCE_SELECTION_FINAL_CALL_BUDGET


def _occurrence_treatment_surface(
    status: Mapping[str, Any],
) -> dict[str, Any] | None:
    grouped_cards: list[Mapping[str, Any]] = []
    occurrence_sets = tuple(status.get("caption_occurrence_sets", ()) or ())
    if occurrence_sets and isinstance(occurrence_sets[-1], Mapping):
        grouped_cards = [
            candidate["candidate_card"]
            for candidate in tuple(occurrence_sets[-1].get("candidates", ()) or ())
            if isinstance(candidate, Mapping)
            and isinstance(candidate.get("candidate_card"), Mapping)
        ]
    if grouped_cards:
        excerpts = [
            {
                "caption_excerpt": str(
                    passage.get("caption_excerpt", "") or ""
                ),
                "query_matches": list(passage.get("query_matches", ()) or ()),
            }
            for card in grouped_cards
            for passage in tuple(card.get("representative_passages", ()) or ())
            if isinstance(passage, Mapping)
        ]
        return {
            "representation": "grouped",
            "visible_occurrence_count": len(grouped_cards),
            "visible_excerpt_count": len(excerpts),
            "visible_excerpt_chars": sum(
                len(str(row["caption_excerpt"])) for row in excerpts
            ),
            "visible_excerpt_digest": occurrence_excerpt_digest(excerpts),
            "visible_text_digest": occurrence_visible_text_digest(
                cards=grouped_cards
            ),
        }
    flat = [
        passage
        for passage in tuple(status.get("flat_occurrence_passages", ()) or ())
        if isinstance(passage, Mapping)
    ]
    if not flat:
        return None
    flat_queries = [
        query
        for query in tuple(status.get("flat_occurrence_queries", ()) or ())
        if isinstance(query, Mapping)
    ]
    excerpts = [
        {
            "caption_excerpt": str(passage.get("caption_excerpt", "") or ""),
            "query_matches": list(passage.get("query_matches", ()) or ()),
        }
        for passage in flat
    ]
    return {
        "representation": "flat",
        "visible_occurrence_count": 0,
        "visible_excerpt_count": len(excerpts),
        "visible_excerpt_chars": sum(
            len(str(row["caption_excerpt"])) for row in excerpts
        ),
        "visible_excerpt_digest": occurrence_excerpt_digest(excerpts),
        "visible_text_digest": occurrence_visible_text_digest(
            flat_passages=flat,
            flat_queries=flat_queries,
        ),
    }


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


def _resolve_runtime_decision(
    decision: ReasonerDecision,
    catalog: RuntimeEvidenceCatalog,
    document: WorkingDocument,
) -> tuple[ReasonerDecision, list[dict[str, Any]], tuple[str, ...]]:
    errors: list[dict[str, Any]] = []
    ignored_state_ops: list[str] = []
    allowed_ops = []
    state_ops = {
        "add_obligation",
        "set_obligation_status",
        "add_temporal_scope",
        "set_cue_status",
    }
    for operation in decision.workspace_ops:
        op_type = str(operation.get("op", operation.get("type", "")) or "").casefold()
        if op_type in state_ops:
            ignored_state_ops.append(op_type)
        else:
            allowed_ops.append(operation)

    fallback_requirement_id = catalog.single_open_answer_requirement(document)
    resolved_tasks = []
    for task in decision.tasks:
        if task.inspection_mode in {"search_caption", "search_asr"}:
            task = replace(
                task,
                occurrence_id="",
                locator_attempt_id="",
                temporal_scope_id="",
                refine_item_id="",
                refine_interpretation_id="",
                parent_attempt_id="",
                cue_id="",
            )
        requirement_id = ""
        if task.requirement_id:
            requirement_id = catalog.resolve_requirement(task.requirement_id)
            if not requirement_id:
                errors.append(
                    {
                        "code": "requirement_handle_unknown",
                        "requested_task_ids": [task.query_id],
                        "handle": task.requirement_id,
                    }
                )
        elif fallback_requirement_id:
            requirement_id = fallback_requirement_id
        evidence_kind = (
            document.obligations[requirement_id].evidence_kind
            if requirement_id in document.obligations
            else task.evidence_kind
        )
        occurrence_id = task.occurrence_id
        locator_attempt_id = task.locator_attempt_id
        time_range = task.time_range
        segment_id = task.segment_id
        source_video_ids = task.source_video_ids
        if occurrence_id:
            occurrence = catalog.resolve_occurrence(occurrence_id)
            if occurrence is None:
                errors.append(
                    {
                        "code": "occurrence_handle_unknown",
                        "requested_task_ids": [task.query_id],
                        "handle": occurrence_id,
                    }
                )
            else:
                occurrence_id = str(occurrence.get("occurrence_id", "") or "")
                locator_attempt_id = str(
                    occurrence.get("attempt_id", locator_attempt_id) or locator_attempt_id
                )
                raw_range = tuple(occurrence.get("time_range", ()) or ())
                if time_range is None and len(raw_range) == 2:
                    time_range = tuple(
                        sorted((float(raw_range[0]), float(raw_range[1])))
                    )
                occurrence_segments = tuple(
                    str(value)
                    for value in tuple(occurrence.get("segment_ids", ()) or ())
                    if str(value)
                )
                if not segment_id and len(occurrence_segments) == 1:
                    segment_id = occurrence_segments[0]
                if not source_video_ids:
                    source_video_ids = tuple(
                        str(value)
                        for value in tuple(
                            occurrence.get("source_video_ids", ()) or ()
                        )
                        if str(value)
                    )
        temporal_scope_id = task.temporal_scope_id
        if temporal_scope_id:
            resolved_scope = catalog.resolve_scope(temporal_scope_id)
            if not resolved_scope:
                errors.append(
                    {
                        "code": "temporal_scope_handle_unknown",
                        "requested_task_ids": [task.query_id],
                        "handle": temporal_scope_id,
                    }
                )
            temporal_scope_id = resolved_scope
        refinement = catalog.resolve_item(task.refine_item_id) if task.refine_item_id else None
        if task.refine_item_id and refinement is None:
            errors.append(
                {
                    "code": "refine_item_handle_unknown",
                    "requested_task_ids": [task.query_id],
                    "handle": task.refine_item_id,
                }
            )
        if refinement is not None and not refinement.refinable:
            errors.append(
                {
                    "code": "refine_item_not_refinable",
                    "requested_task_ids": [task.query_id],
                    "handle": task.refine_item_id,
                }
            )
        resolved_tasks.append(
            replace(
                task,
                requirement_id=requirement_id,
                evidence_kind=evidence_kind,
                occurrence_id=occurrence_id,
                locator_attempt_id=locator_attempt_id,
                time_range=time_range,
                segment_id=segment_id,
                source_video_ids=source_video_ids,
                temporal_scope_id=temporal_scope_id,
                refine_item_id=refinement.item_id if refinement else task.refine_item_id,
                refine_interpretation_id=(
                    refinement.interpretation_id if refinement else task.refine_interpretation_id
                ),
                parent_attempt_id=(
                    refinement.attempt_id if refinement else task.parent_attempt_id
                ),
                cue_id=refinement.cue_id if refinement else task.cue_id,
                inspection_mode="window" if refinement else task.inspection_mode,
            )
        )

    support_items = []
    for value in decision.supporting_item_ids:
        item = catalog.resolve_item(value)
        if item is None:
            errors.append({"code": "support_item_handle_unknown", "handle": value})
        else:
            support_items.append(item.item_id)
    support_requirements = []
    for value in decision.supports_requirement_ids:
        requirement_id = catalog.resolve_requirement(value)
        if not requirement_id:
            errors.append({"code": "support_requirement_handle_unknown", "handle": value})
        else:
            support_requirements.append(requirement_id)
    if support_items and not support_requirements and fallback_requirement_id:
        support_requirements.append(fallback_requirement_id)
    unresolved_requirements = []
    for value in decision.unresolved_requirement_ids:
        requirement_id = catalog.resolve_requirement(value)
        if not requirement_id:
            errors.append({"code": "unresolved_requirement_handle_unknown", "handle": value})
        else:
            unresolved_requirements.append(requirement_id)
    return (
        replace(
            decision,
            tasks=tuple(resolved_tasks),
            workspace_ops=tuple(allowed_ops),
            supporting_item_ids=tuple(support_items),
            supports_requirement_ids=tuple(support_requirements),
            unresolved_requirement_ids=tuple(unresolved_requirements),
        ),
        errors,
        tuple(ignored_state_ops),
    )


def _expand_runtime_tasks(
    tasks: Sequence[InvestigationTask],
) -> tuple[InvestigationTask, ...]:
    expanded: list[InvestigationTask] = []
    for task in tasks:
        expanded.append(task)
        if (
            task.evidence_kind == "text_exact"
            and task.inspection_mode == "window"
            and not task.refine_item_id
            and task.interpretation_purpose == "primary"
        ):
            expanded.append(
                replace(
                    task,
                    query_id=f"{task.query_id}_reread",
                    force_reinspect=True,
                    interpretation_purpose="manual_reread",
                )
            )
    return tuple(expanded)


def _advance_runtime_task_states(
    document: WorkingDocument,
    tasks: Sequence[InvestigationTask],
    reports: Sequence[InvestigationReport],
    observations: ObservationLog,
) -> None:
    reports_by_query = {str(report.query_id): report for report in reports}
    rows_by_task: dict[str, list[Mapping[str, Any]]] = {}
    for row in observations.rows:
        rows_by_task.setdefault(str(row.get("task_id", "") or ""), []).append(row)
    for task in tasks:
        if not task.requirement_id:
            continue
        report = reports_by_query.get(task.query_id)
        if report is None or str(report.status).casefold() != "completed":
            continue
        rows = rows_by_task.get(task.query_id, ())
        attempt_ids = tuple(
            dict.fromkeys(str(row.get("attempt_id", "") or "") for row in rows)
        )
        if task.inspection_mode in {"search_caption", "search_asr"}:
            candidate_found = any(
                tuple(row.get("sampling_config", {}).get("hits", ()) or ())
                for row in rows
                if isinstance(row.get("sampling_config"), Mapping)
            )
            if candidate_found:
                advance_requirement_state(
                    document,
                    task.requirement_id,
                    "candidate_found",
                    attempt_ids=attempt_ids,
                )
            continue
        item_ids = tuple(
            str(item.get("item_id", "") or "")
            for row in rows
            for item in tuple(row.get("interpretation_items", ()) or ())
            if isinstance(item, Mapping) and str(item.get("item_id", "") or "")
        )
        if attempt_ids and item_ids:
            advance_requirement_state(
                document,
                task.requirement_id,
                "observed",
                attempt_ids=attempt_ids,
            )


def _apply_runtime_answer_state(
    document: WorkingDocument,
    decision: ReasonerDecision,
    catalog: RuntimeEvidenceCatalog,
) -> None:
    item_refs = tuple(
        item
        for item_id in decision.supporting_item_ids
        if (item := catalog.item_by_id(item_id)) is not None
    )
    attempt_ids = tuple(dict.fromkeys(item.attempt_id for item in item_refs))
    item_ids = tuple(item.item_id for item in item_refs)
    for requirement_id in decision.supports_requirement_ids:
        advance_requirement_state(
            document,
            requirement_id,
            "supported",
            attempt_ids=attempt_ids,
            item_ids=item_ids,
        )
    uncertainty = decision.residual_uncertainty or "Reasoner marked this requirement unresolved."
    for requirement_id in decision.unresolved_requirement_ids:
        advance_requirement_state(
            document,
            requirement_id,
            "unresolved",
            residual_uncertainty=uncertainty,
        )


def _decision_preflight(
    decision: ReasonerDecision,
    *,
    closure_repair: bool = False,
    runtime_derived: bool = False,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if runtime_derived and decision.action not in {"investigate", "answer"}:
        errors.append(
            {
                "code": "runtime_action_not_allowed",
                "action": decision.action,
                "allowed_actions": ["investigate", "answer"],
            }
        )
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
    if closure_repair:
        for task in decision.tasks:
            code = _closure_repair_task_error(task)
            if code:
                errors.append(
                    {
                        "code": code,
                        "requested_task_ids": [task.query_id],
                    }
                )
    return errors


def _closure_repair_task_error(task: InvestigationTask) -> str:
    if task.inspection_mode in {"search_asr", "search_caption"}:
        return "closure_repair_global_search_forbidden"
    if task.inspection_mode == "arbitrate_observation":
        return ""
    if task.inspection_mode != "window":
        return "closure_repair_mode_forbidden"
    cue_bound = bool(task.parent_attempt_id and task.cue_id)
    occurrence_bound = bool(task.locator_attempt_id and task.occurrence_id)
    if not cue_bound and not occurrence_bound:
        return "closure_repair_unbound_window_forbidden"
    if task.time_range is not None and task.time_range[1] - task.time_range[0] > 120.0:
        return "closure_repair_wide_window_forbidden"
    return ""


def _closure_repairable(validation: AnswerValidation) -> bool:
    return bool(validation.errors)


def _control_retry_feedback(
    errors: Sequence[Mapping[str, Any]],
    *,
    revision: int,
    previous_feedback: Mapping[str, Any],
) -> dict[str, Any]:
    codes = [str(error.get("code", "decision_schema_invalid")) for error in errors]
    must_answer_codes = sorted(
        set(codes) & _MUST_ANSWER_OCCURRENCE_CODES
    )
    must_not_answer_codes = sorted(
        set(codes) & _MUST_NOT_ANSWER_OCCURRENCE_CODES
    )
    if must_answer_codes and must_not_answer_codes:
        return {
            "type": "contradictory_gate_state",
            "cause": "decision_schema_error",
            "errors": [dict(error) for error in errors],
            "revision": revision,
            "previous_feedback_type": str(
                previous_feedback.get("type", "") or ""
            ),
            "must_answer_codes": must_answer_codes,
            "must_not_answer_codes": must_not_answer_codes,
            "instruction": (
                "Runtime detected contradictory occurrence gates. Do not infer "
                "a repair action; finalization precedence must resolve the state."
            ),
        }
    details = tuple(str(error.get("detail", "") or "") for error in errors)
    repair_rules: list[str] = []
    if any("_already_exists:" in detail for detail in details):
        repair_rules.append(
            "Omit add operations for IDs that already exist at this revision; use the corresponding set/update operation instead."
        )
    if any("satisfied_obligation_requires_attempt:" in detail for detail in details):
        repair_rules.append(
            "A satisfied obligation must list existing supporting_attempt_ids from the observation claims that support it; otherwise keep it observed, contested, or open."
        )
    if any(
        marker in detail
        for detail in details
        for marker in (
            "satisfied_obligation_requires_claim:",
            "obligation_supporting_claim_missing:",
        )
    ):
        repair_rules.append(
            "Use only claim IDs already present after this transaction, adding a missing claim before referencing it."
        )
    if any(
        marker in detail
        for detail in details
        for marker in (
            "obligation_dependency_lineage_missing:",
            "obligation_dependency_unsatisfied:",
        )
    ):
        repair_rules.append(
            "Do not satisfy a dependent obligation until its dependency is satisfied and its supporting claim derives from the dependency's supporting claim lineage."
        )
    if "occurrence_selection_required" in codes:
        repair_rules.append(
            "Do not answer in this decision. Return action=update_workspace with no answer and one top-level occurrence_ops select operation using an occurrence_id from the current visible state. The persisted selection must precede a later answer decision; Runtime validates the ID but never chooses it."
        )
    if "occurrence_answer_required_after_selection" in codes:
        repair_rules.append(
            "The occurrence selection is already persisted and investigation is closed. Return action=answer now with no tasks and no occurrence_ops; do not revise the selected occurrence."
        )
    if "occurrence_resolution_required" in codes:
        repair_rules.append(
            "Do not answer. Resolve only the active scoped occurrence set: return action=update_workspace with one or more select operations, or one no_match operation when none of that set's candidates fit. Every operation must copy the active set_id and visible occurrence_id exactly."
        )
    if any(code.startswith("occurrence_sufficiency_") for code in codes):
        repair_rules.append(
            "Do not answer. Begin occurrence_ops with assess_sufficiency for the active set. Check one to six question-critical constraints across every viable candidate, bind each supported status to visible evidence_passage_ids, then select only a candidate supported on every constraint; otherwise follow the insufficient verdict with defer or no_match."
        )
    if "occurrence_search_required" in codes:
        repair_rules.append(
            "The active scoped set was deferred. Do not answer; issue a refined search_caption investigation, or persist no_match for that exact set_id when further search cannot resolve it."
        )
    if "occurrence_no_match_required_at_finalization" in codes:
        repair_rules.append(
            "Investigation is closing with a deferred set. Return action=update_workspace with a no_match occurrence operation for the active set_id, then answer in a later decision."
        )
    if "occurrence_answer_required_after_resolution" in codes:
        repair_rules.append(
            "The active scoped occurrence set is resolved. Return action=answer with no occurrence_ops; selected occurrences remain persisted."
        )
    if "occurrence_resolution_already_committed" in codes:
        repair_rules.append(
            "The active scoped occurrence resolution is immutable. Remove all occurrence_ops and follow the pending locator instruction, or return action=answer when no locator remains."
        )
    if any(
        code
        in {
            "occurrence_locator_inspection_required",
            "occurrence_locator_binding_required",
            "occurrence_locator_unbound_window_forbidden",
        }
        for code in codes
    ):
        repair_rules.append(
            "Do not answer. Return action=investigate with a window task bound to exactly one pending active_occurrence_locator, copying both locator_attempt_id and occurrence_id. Do not mix in unbound window tasks."
        )
    instruction = "Preserve the semantic intent and return one corrected Decision JSON object."
    if repair_rules:
        instruction += " Mechanical repair rules: " + " ".join(repair_rules)
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
        "instruction": instruction,
    }


def _append_contradictory_gate_state(
    trace: list[Mapping[str, Any]],
    feedback: Mapping[str, Any],
    *,
    round_id: int,
) -> None:
    if feedback.get("type") != "contradictory_gate_state":
        return
    trace.append(
        {
            "type": "contradictory_gate_state",
            "round": round_id,
            "must_answer_codes": list(
                feedback.get("must_answer_codes", ()) or ()
            ),
            "must_not_answer_codes": list(
                feedback.get("must_not_answer_codes", ()) or ()
            ),
        }
    )


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
    require_item_provenance: bool = False,
    surfaced_observation_ids: Sequence[str] = (),
) -> dict[str, Any]:
    errors = document.validate(
        observation_ids=observations.attempt_ids,
        observation_rows=observations.rows,
        require_item_provenance=require_item_provenance,
    )
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
        if status["evidence_kind"] != "persistent_state"
        and status["requested_fps"] > 0.0
        and status["sampling_fidelity"] < 0.8
    )
    candidates_by_key: dict[tuple[str, float, float], dict[str, Any]] = {}
    caption_occurrences_by_key: dict[str, dict[str, Any]] = {}
    caption_occurrence_sets: list[dict[str, Any]] = []
    latest_flat_occurrence_passages: list[dict[str, Any]] = []
    latest_flat_occurrence_queries: list[dict[str, Any]] = []
    temporal_locators: list[dict[str, Any]] = []
    oracle_guidance_packets: list[dict[str, Any]] = []
    for row in observations.rows:
        config = row.get("sampling_config")
        if not isinstance(config, Mapping) or config.get("mode") != "search_caption":
            continue
        temporal_locator = config.get("temporal_locator")
        if isinstance(temporal_locator, Mapping):
            temporal_locators.append(dict(temporal_locator))
        oracle_guidance = config.get("oracle_guidance")
        if isinstance(oracle_guidance, Mapping):
            oracle_guidance_packets.append(dict(oracle_guidance))
        occurrence_set = config.get("occurrence_set")
        if isinstance(occurrence_set, Mapping):
            occurrence_cards = candidate_cards_by_occurrence(occurrence_set)
            raw_flat_passages = tuple(
                occurrence_set.get("flat_candidate_passages", ()) or ()
            )
            if raw_flat_passages:
                latest_flat_occurrence_passages = [
                    {
                        "attempt_id": str(row.get("attempt_id", "")),
                        **dict(raw_passage),
                    }
                    for raw_passage in raw_flat_passages
                    if isinstance(raw_passage, Mapping)
                ]
            raw_flat_queries = tuple(
                occurrence_set.get("flat_candidate_queries", ()) or ()
            )
            if raw_flat_queries:
                latest_flat_occurrence_queries = [
                    dict(raw_query)
                    for raw_query in raw_flat_queries
                    if isinstance(raw_query, Mapping)
                ]
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
                occurrence_id = candidate["occurrence_id"]
                if occurrence_id in occurrence_cards:
                    candidate["candidate_card"] = occurrence_cards[occurrence_id]
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
                    "semantic_target": [
                        str(value)
                        for value in tuple(config.get("queries", ()) or ())
                        if str(value)
                    ],
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
    temporal_scope_summary = _temporal_scope_summary(
        document,
        observations,
        candidates=caption_occurrence_candidates,
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
    oracle_status: dict[str, Any] = {}
    if oracle_guidance_packets:
        oracle_status = dict(oracle_guidance_packets[-1])
        selected_candidates: list[dict[str, Any]] = []
        for raw_candidate in tuple(oracle_status.get("selected_candidates", ()) or ()):
            if not isinstance(raw_candidate, Mapping):
                continue
            candidate = dict(raw_candidate)
            candidate_range = _time_range(candidate.get("inspection_range"))
            candidate["visually_inspected"] = bool(
                candidate_range
                and any(
                    _ranges_overlap(candidate_range, interval)
                    for interval in visual_ranges
                )
            )
            selected_candidates.append(candidate)
        oracle_status["selected_candidates"] = selected_candidates
        oracle_status["all_selected_candidates_inspected"] = bool(
            selected_candidates
        ) and all(
            bool(candidate["visually_inspected"])
            for candidate in selected_candidates
        )
        point_anchors: list[dict[str, Any]] = []
        for raw_anchor in tuple(oracle_status.get("point_anchors", ()) or ()):
            if not isinstance(raw_anchor, Mapping):
                continue
            timestamp = raw_anchor.get("anchor_timestamp_sec")
            if not isinstance(timestamp, (int, float)):
                continue
            anchor = dict(raw_anchor)
            anchor["visually_inspected"] = any(
                interval[0] <= float(timestamp) <= interval[1]
                for interval in visual_ranges
            )
            point_anchors.append(anchor)
        if point_anchors:
            oracle_status["point_anchors"] = point_anchors
        anchor_status = [
            {
                "timestamp_sec": float(value),
                "visually_inspected": any(
                    interval[0] <= float(value) <= interval[1]
                    for interval in visual_ranges
                ),
            }
            for value in tuple(oracle_status.get("anchor_timestamps_sec", ()) or ())
            if isinstance(value, (int, float))
        ]
        if anchor_status:
            oracle_status["anchor_inspection_status"] = anchor_status
            oracle_status["all_point_anchors_inspected"] = all(
                bool(anchor["visually_inspected"])
                for anchor in anchor_status
            )
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
    if oracle_status and (
        not oracle_status.get("all_selected_candidates_inspected")
        or (
            oracle_status.get("anchor_inspection_status")
            and not oracle_status.get("all_point_anchors_inspected")
        )
    ):
        hints.append(
            "Answer-free oracle guidance identifies locator candidates only. Inspect every selected candidate visually; "
            "choose window width and refinement yourself, and never cite the guidance as answer support."
        )
    if temporal_scope_summary["temporal_scope_count"] and not temporal_scope_summary[
        "temporal_scope_resolved_rate"
    ]:
        hints.append(
            "A declared TemporalScope is unresolved. Establish its anchor material and bind target occurrence candidates."
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
    provenance_summary = document.provenance_summary(observations.rows)
    cue_summary = document.cue_summary(observations.rows)
    unconsumed_observation_ids = tuple(
        attempt_id
        for attempt_id in observations.attempt_ids
        if attempt_id not in set(surfaced_observation_ids)
    )
    if not obligation_summary["answer_bearing_obligation_count"]:
        hints.append(
            "No answer-bearing evidence obligations exist. Decompose the observable requirements before finalizing."
        )
    elif obligation_summary["open_obligation_count_at_answer"]:
        hints.append(
            "Answer-bearing obligations remain open. Satisfy them with claim/material lineage or mark them unresolved "
            "with explicit residual uncertainty before finalizing."
        )
    if provenance_summary["observation_claim_count"] and provenance_summary[
        "observation_claim_item_binding_rate"
    ] < 1.0:
        hints.append(
            "Observation claims must copy an exact attempt_id, interpretation_id, and item_id triple from the catalog."
        )
    if cue_summary["unverified_cue_count"]:
        hints.append(
            "Point cues remain unverified. Bind parent_attempt_id and cue_id to replay the exact sampled frame before "
            "requesting a child refinement window."
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
        "flat_occurrence_passages": list(latest_flat_occurrence_passages[:24]),
        "flat_occurrence_queries": list(latest_flat_occurrence_queries[:32]),
        **temporal_status,
        **(
            {"oracle_guidance": oracle_status}
            if oracle_status
            else {}
        ),
        **temporal_scope_summary,
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
        **provenance_summary,
        **cue_summary,
        "unconsumed_observation_count": len(unconsumed_observation_ids),
        "unconsumed_observation_ids": list(unconsumed_observation_ids),
        **runtime,
    }


def _temporal_scope_summary(
    document: WorkingDocument,
    observations: ObservationLog,
    *,
    candidates: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    source_rows = observations.catalog_source_rows()
    rows_by_attempt = {
        str(row.get("attempt_id", "") or ""): row
        for row in source_rows
        if str(row.get("attempt_id", "") or "")
    }
    occurrence_candidates = tuple(candidates or _occurrence_candidates(observations.rows))
    statuses = []
    for scope in document.temporal_scopes.values():
        anchor_state = document.obligation_states.get(scope.anchor_requirement_id)
        target_state = document.obligation_states.get(scope.target_requirement_id)
        anchor_ranges = tuple(
            interval
            for attempt_id in tuple(
                anchor_state.supporting_attempt_ids if anchor_state is not None else ()
            )
            if (row := rows_by_attempt.get(attempt_id)) is not None
            for value in tuple(row.get("inspected_ranges", ()) or ())
            if (interval := _time_range(value)) is not None
        )
        target_attempt_ids = set(
            target_state.supporting_attempt_ids if target_state is not None else ()
        )
        scoped_candidates = tuple(
            candidate
            for candidate in occurrence_candidates
            if not target_attempt_ids
            or str(candidate.get("attempt_id", "") or "") in target_attempt_ids
        )
        statuses.append(
            resolve_temporal_scope(
                scope,
                anchor_intervals=anchor_ranges,
                candidates=scoped_candidates,
            )
        )
    resolved_count = sum(bool(status.get("resolved")) for status in statuses)
    return {
        "temporal_scope_count": len(statuses),
        "resolved_temporal_scope_count": resolved_count,
        "temporal_scope_resolved_rate": (
            resolved_count / len(statuses) if statuses else 0.0
        ),
        "temporal_scope_statuses": statuses,
    }


def _temporal_scope_resolution_map(
    document: WorkingDocument,
    observations: ObservationLog,
) -> dict[str, Mapping[str, Any]]:
    return {
        str(status.get("scope_id", "") or ""): status
        for status in tuple(
            _temporal_scope_summary(document, observations).get(
                "temporal_scope_statuses",
                (),
            )
            or ()
        )
        if isinstance(status, Mapping) and str(status.get("scope_id", "") or "")
    }


def _occurrence_candidates(
    observation_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    candidates = []
    for row in observation_rows:
        config = row.get("sampling_config")
        if not isinstance(config, Mapping):
            continue
        occurrence_set = config.get("occurrence_set")
        if not isinstance(occurrence_set, Mapping):
            continue
        for value in tuple(occurrence_set.get("candidates", ()) or ()):
            if isinstance(value, Mapping):
                candidates.append(
                    {
                        **dict(value),
                        "attempt_id": str(row.get("attempt_id", "") or ""),
                    }
                )
    for row in observation_rows:
        config = row.get("sampling_config")
        if not isinstance(config, Mapping):
            continue
        binding = config.get("candidate_binding")
        if not isinstance(binding, Mapping):
            continue
        candidate_range = _time_range(binding.get("candidate_range", ()))
        occurrence_id = str(binding.get("occurrence_id", "") or "")
        attempt_id = str(row.get("attempt_id", "") or "")
        if not candidate_range or not occurrence_id or not attempt_id:
            continue
        candidates.append(
            {
                **dict(binding),
                "attempt_id": attempt_id,
                "material_attempt_id": attempt_id,
                "time_range": list(candidate_range),
            }
        )
    return tuple(candidates)


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
        "requires_refinement": bool(
            config.get("requires_refinement", manifest.get("requires_refinement"))
        ),
        "evidence_kind": str(config.get("evidence_kind", "generic") or "generic"),
        "probe_count": int(config.get("probe_count", 0) or 0),
        "probe_coverage_requirement": int(
            config.get("probe_coverage_requirement", 0) or 0
        ),
        "probe_coverage_satisfied": bool(
            config.get("probe_coverage_satisfied", True)
        ),
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
    observation_rows: Sequence[Mapping[str, Any]] = (),
    temporal_scope_ids: Sequence[str] = (),
    temporal_scope_resolutions: Mapping[str, Mapping[str, Any]] | None = None,
    cue_states: Mapping[str, Mapping[str, Any]] | None = None,
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
        cue_task, cue_error = _resolve_cue_bound_task(
            workspace,
            requested_task,
            observation_rows,
            cue_states=cue_states or {},
        )
        if cue_error is not None:
            resolution_errors.append(cue_error)
            groups.append(
                {
                    "requested_task": requested_task,
                    "tasks": (),
                    "error_start": error_start,
                }
            )
            continue
        task = cue_task or requested_task
        binding_error = validate_occurrence_material_binding(
            task,
            observation_rows,
            temporal_scope_ids=temporal_scope_ids,
            temporal_scope_resolutions=temporal_scope_resolutions,
        )
        if binding_error is not None:
            resolution_errors.append(binding_error)
            groups.append(
                {
                    "requested_task": requested_task,
                    "tasks": (),
                    "error_start": error_start,
                }
            )
            continue
        task = _materialize_occurrence_bound_task(task, observation_rows)
        task_error = _task_executability_error(task)
        if task_error:
            resolution_errors.append(_task_resolution_error(task, task_error))
            groups.append(
                {
                    "requested_task": requested_task,
                    "tasks": (),
                    "error_start": error_start,
                }
            )
            continue
        task = _resolve_task_coordinates(
            task,
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


def validate_occurrence_material_binding(
    task: InvestigationTask,
    observation_rows: Sequence[Mapping[str, Any]],
    *,
    temporal_scope_ids: Sequence[str] = (),
    temporal_scope_resolutions: Mapping[str, Mapping[str, Any]] | None = None,
    neighborhood_tolerance_sec: float = 120.0,
) -> dict[str, Any] | None:
    known_scope_ids = {str(value) for value in temporal_scope_ids if str(value)}
    if task.temporal_scope_id and task.temporal_scope_id not in known_scope_ids:
        return _task_resolution_error(
            task,
            "temporal_scope_missing",
            temporal_scope_id=task.temporal_scope_id,
        )
    has_locator = bool(task.locator_attempt_id)
    has_occurrence = bool(task.occurrence_id)
    if has_locator != has_occurrence:
        return _task_resolution_error(task, "occurrence_binding_incomplete")
    if not has_locator:
        return None
    if task.inspection_mode != "window":
        return _task_resolution_error(task, "occurrence_binding_requires_visual_window")

    locator = next(
        (
            row
            for row in observation_rows
            if str(row.get("attempt_id", "") or "") == task.locator_attempt_id
        ),
        None,
    )
    if locator is None:
        return _task_resolution_error(
            task,
            "locator_attempt_missing",
            locator_attempt_id=task.locator_attempt_id,
        )
    config = locator.get("sampling_config")
    if not isinstance(config, Mapping) or config.get("mode") != "search_caption":
        return _task_resolution_error(task, "locator_attempt_not_caption_search")
    occurrence_set = config.get("occurrence_set")
    candidates = (
        tuple(occurrence_set.get("candidates", ()) or ())
        if isinstance(occurrence_set, Mapping)
        else ()
    )
    candidate = next(
        (
            value
            for value in candidates
            if isinstance(value, Mapping)
            and str(value.get("occurrence_id", "") or "") == task.occurrence_id
        ),
        None,
    )
    if candidate is None:
        return _task_resolution_error(
            task,
            "occurrence_not_in_locator",
            locator_attempt_id=task.locator_attempt_id,
            occurrence_id=task.occurrence_id,
        )
    scope_resolution = dict(
        (temporal_scope_resolutions or {}).get(task.temporal_scope_id, {})
        if task.temporal_scope_id
        else {}
    )
    selected_occurrences = {
        str(value)
        for value in tuple(scope_resolution.get("selected_occurrence_ids", ()) or ())
        if str(value)
    }
    if (
        scope_resolution.get("resolved")
        and selected_occurrences
        and task.occurrence_id not in selected_occurrences
    ):
        return _task_resolution_error(
            task,
            "occurrence_outside_temporal_selection",
            selected_occurrence_ids=sorted(selected_occurrences),
        )
    candidate_sources = {
        str(value) for value in tuple(candidate.get("source_video_ids", ()) or ()) if str(value)
    }
    if task.source_video_ids and candidate_sources and not candidate_sources.intersection(task.source_video_ids):
        return _task_resolution_error(task, "occurrence_source_mismatch")
    candidate_segments = {
        str(value) for value in tuple(candidate.get("segment_ids", ()) or ()) if str(value)
    }
    if task.segment_id and candidate_segments and task.segment_id not in candidate_segments:
        return _task_resolution_error(task, "occurrence_segment_mismatch")
    candidate_range = _time_range(candidate.get("time_range"))
    if task.time_range is not None and candidate_range is not None:
        gap = max(
            candidate_range[0] - task.time_range[1],
            task.time_range[0] - candidate_range[1],
            0.0,
        )
        if gap > max(0.0, float(neighborhood_tolerance_sec)):
            return _task_resolution_error(
                task,
                "occurrence_range_mismatch",
                candidate_range=list(candidate_range),
                neighborhood_tolerance_sec=float(neighborhood_tolerance_sec),
            )
    return None


def _materialize_occurrence_bound_task(
    task: InvestigationTask,
    observation_rows: Sequence[Mapping[str, Any]],
) -> InvestigationTask:
    """Resolve a validated occurrence handle into an executable visual target."""
    if not task.locator_attempt_id or not task.occurrence_id:
        return task
    locator = next(
        (
            row
            for row in observation_rows
            if str(row.get("attempt_id", "") or "") == task.locator_attempt_id
        ),
        None,
    )
    config = locator.get("sampling_config") if isinstance(locator, Mapping) else None
    occurrence_set = config.get("occurrence_set") if isinstance(config, Mapping) else None
    candidate = next(
        (
            value
            for value in tuple(
                occurrence_set.get("candidates", ())
                if isinstance(occurrence_set, Mapping)
                else ()
            )
            if isinstance(value, Mapping)
            and str(value.get("occurrence_id", "") or "") == task.occurrence_id
        ),
        None,
    )
    if candidate is None:
        return task
    candidate_range = _time_range(candidate.get("time_range"))
    candidate_segments = tuple(
        str(value)
        for value in tuple(candidate.get("segment_ids", ()) or ())
        if str(value)
    )
    candidate_sources = tuple(
        str(value)
        for value in tuple(candidate.get("source_video_ids", ()) or ())
        if str(value)
    )
    return replace(
        task,
        segment_id=(
            task.segment_id
            or (candidate_segments[0] if len(candidate_segments) == 1 else "")
        ),
        time_range=task.time_range or candidate_range,
        source_video_ids=task.source_video_ids or candidate_sources,
    )


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
    if task.inspection_mode == "window" and not (
        task.segment_id
        or task.time_range
        or (task.parent_attempt_id and task.cue_id)
    ):
        return "target_missing"
    return ""


def _resolve_cue_bound_task(
    workspace: VirtualVideoWorkspace,
    task: InvestigationTask,
    observation_rows: Sequence[Mapping[str, Any]],
    *,
    cue_states: Mapping[str, Mapping[str, Any]],
) -> tuple[InvestigationTask | None, dict[str, Any] | None]:
    if not task.parent_attempt_id and not task.cue_id:
        return task, None
    if not task.parent_attempt_id or not task.cue_id:
        return None, _task_resolution_error(task, "cue_binding_incomplete")
    if task.inspection_mode != "window":
        return None, _task_resolution_error(task, "cue_binding_requires_visual_window")

    cue: Mapping[str, Any] | None = None
    for row in observation_rows:
        if str(row.get("attempt_id", "") or "") != task.parent_attempt_id:
            continue
        for raw_cue in tuple(row.get("observation_cues", ()) or ()):
            if isinstance(raw_cue, Mapping) and str(raw_cue.get("cue_id", "") or "") == task.cue_id:
                cue = raw_cue
                break
        if cue is not None:
            break
    if cue is None:
        return None, _task_resolution_error(
            task,
            "cue_not_found_on_parent_attempt",
            parent_attempt_id=task.parent_attempt_id,
            cue_id=task.cue_id,
        )

    if task.refine_item_id:
        if str(cue.get("item_id", "") or "") != task.refine_item_id:
            return None, _task_resolution_error(
                task,
                "refine_item_cue_mismatch",
                refine_item_id=task.refine_item_id,
                cue_id=task.cue_id,
            )
        cue_time = float(cue.get("virtual_time", 0.0) or 0.0)
        segment = next(
            (
                item
                for item in workspace.manifest.segments
                if item.virtual_start_sec - 1e-6 <= cue_time <= item.virtual_end_sec + 1e-6
            ),
            None,
        )
        if segment is None:
            return None, _task_resolution_error(
                task,
                "refine_item_time_outside_workspace",
                refine_item_id=task.refine_item_id,
            )
        radius = task.window_radius_sec
        return (
            replace(
                task,
                segment_id=segment.segment_id,
                time_range=(
                    max(float(segment.virtual_start_sec), cue_time - radius),
                    min(float(segment.virtual_end_sec), cue_time + radius),
                ),
                coordinate_space="virtual",
                source_video_ids=(segment.source_video_id,),
                cue_stage="child_refinement",
                cue_virtual_time=cue_time,
                sampling_floor_fps=2.0,
                interpretation_purpose="manual_reread",
                force_reinspect=True,
            ),
            None,
        )

    state = cue_states.get(task.cue_id, {})
    status = str(state.get("status", "unverified") or "unverified").casefold()
    if status == "rejected":
        return None, _task_resolution_error(task, "cue_rejected", cue_id=task.cue_id)
    if status not in {"unverified", "verified"}:
        return None, _task_resolution_error(task, "cue_status_invalid", cue_id=task.cue_id)

    cue_time = float(cue.get("virtual_time", 0.0) or 0.0)
    segment = next(
        (
            item
            for item in workspace.manifest.segments
            if item.virtual_start_sec - 1e-6 <= cue_time <= item.virtual_end_sec + 1e-6
        ),
        None,
    )
    if segment is None:
        return None, _task_resolution_error(
            task,
            "cue_time_outside_workspace",
            cue_id=task.cue_id,
            cue_virtual_time=cue_time,
        )

    if status == "unverified":
        time_range = (cue_time, cue_time)
        stage = "cue_verification"
        purpose = "cue_verification"
    else:
        radius = task.window_radius_sec
        time_range = (
            max(float(segment.virtual_start_sec), cue_time - radius),
            min(float(segment.virtual_end_sec), cue_time + radius),
        )
        stage = "child_refinement"
        purpose = "manual_reread"
    return (
        replace(
            task,
            segment_id=segment.segment_id,
            time_range=time_range,
            coordinate_space="virtual",
            source_video_ids=(segment.source_video_id,),
            cue_stage=stage,
            cue_virtual_time=cue_time,
            sampling_floor_fps=2.0,
            interpretation_purpose=purpose,
            force_reinspect=True,
        ),
        None,
    )


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
    observation_rows: Sequence[Mapping[str, Any]] = (),
    require_item_provenance: bool = False,
    temporal_scope_resolutions: Mapping[str, Mapping[str, Any]] | None = None,
    unconsumed_observation_ids: Sequence[str] = (),
    require_evidence_kind_requirements: bool = False,
) -> AnswerValidation:
    validation = document.validate_answer(
        decision.supporting_claim_ids,
        supporting_item_ids=decision.supporting_item_ids,
        observation_ids=observation_ids,
        supporting_observation_ids=supporting_observation_ids,
        require_obligation_coverage=require_obligation_coverage,
        observation_rows=observation_rows,
        require_item_provenance=require_item_provenance,
        temporal_scope_resolutions=temporal_scope_resolutions,
        unconsumed_observation_ids=unconsumed_observation_ids,
        require_evidence_kind_requirements=require_evidence_kind_requirements,
    )
    if not decision.answer:
        return replace(
            validation,
            passed=False,
            reason="answer_missing",
            errors=("answer_is_required", *validation.errors),
            reference_integrity_ok=False,
        )
    if options and not (_letter(decision.answer) or _option_letter_from_answer(decision.answer, options)):
        reason = "answer_missing" if not decision.answer else "invalid_option_answer"
        return replace(
            validation,
            passed=False,
            reason=reason,
            errors=("answer_must_select_exactly_one_option", *validation.errors),
            reference_integrity_ok=False,
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
    return replace(
        validation,
        passed=False,
        reason=reason,
        errors=(reason, *validation.errors),
        material_support_ok=False,
    )


def _material_uncertainty(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()
    return normalized not in {"", "none", "no", "n a", "not applicable", "no material uncertainty"}


def _answer_citations(
    decision: ReasonerDecision,
    document: WorkingDocument,
    evidence: Sequence[EvidenceRecord],
    *,
    observation_rows: Sequence[Mapping[str, Any]] = (),
) -> tuple[str, ...]:
    by_id = {record.evidence_id: record for record in evidence}
    by_attempt: dict[str, list[EvidenceRecord]] = {}
    for record in evidence:
        by_attempt.setdefault(evidence_attempt_id(record), []).append(record)
    citations = [citation for citation in decision.citations if citation in by_id]
    validation = document.validate_answer(
        decision.supporting_claim_ids,
        supporting_item_ids=decision.supporting_item_ids,
        observation_ids=tuple(by_attempt),
        observation_rows=observation_rows,
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


def _report_succeeded(report: InvestigationReport) -> int:
    return int(
        report.status == "completed" and bool(report.evidence or report.attempts)
    )


def _task_descriptor(task: InvestigationTask) -> dict[str, Any]:
    return {
        "query_id": task.query_id,
        "goal": task.goal,
        "segment_id": task.segment_id,
        "time_range": list(task.time_range) if task.time_range else [],
        "coordinate_space": task.coordinate_space,
        "source_video_ids": list(task.source_video_ids),
        "conversion_trace": [dict(item) for item in task.conversion_trace],
        "inspection_mode": task.inspection_mode,
        "search_terms": list(task.search_terms),
        "caption_queries": list(task.caption_queries),
        "expected_evidence": task.expected_evidence,
        "top_k": task.top_k,
        "index_mode": task.index_mode,
        "expand_neighbors": task.expand_neighbors,
        "locator_attempt_id": task.locator_attempt_id,
        "occurrence_id": task.occurrence_id,
        "temporal_scope_id": task.temporal_scope_id,
        "evidence_kind": task.evidence_kind,
        "requirement_id": task.requirement_id,
        "refine_item_id": task.refine_item_id,
        "refine_interpretation_id": task.refine_interpretation_id,
        "parent_attempt_id": task.parent_attempt_id,
        "cue_id": task.cue_id,
        "window_radius_sec": task.window_radius_sec,
        "cue_stage": task.cue_stage,
        "cue_virtual_time": task.cue_virtual_time,
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
        occurrence_ops=tuple(value.get("occurrence_ops", ()) or ()),
        supporting_claim_ids=tuple(value.get("supporting_claim_ids", ()) or ()),
        supporting_item_ids=tuple(value.get("supporting_item_ids", value.get("support_items", ())) or ()),
        supports_requirement_ids=tuple(
            value.get("supports_requirement_ids", value.get("supports_requirements", ())) or ()
        ),
        unresolved_requirement_ids=tuple(
            value.get("unresolved_requirement_ids", value.get("unresolved_requirements", ())) or ()
        ),
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
        locator_attempt_id=str(value.get("locator_attempt_id", "") or ""),
        occurrence_id=str(value.get("occurrence_id", "") or ""),
        temporal_scope_id=str(value.get("temporal_scope_id", "") or ""),
        evidence_kind=str(value.get("evidence_kind", "generic") or "generic"),
        requirement_id=str(value.get("requirement_id", value.get("requirement", "")) or ""),
        refine_item_id=str(value.get("refine_item_id", value.get("refine_item", "")) or ""),
        refine_interpretation_id=str(value.get("refine_interpretation_id", "") or ""),
        parent_attempt_id=str(value.get("parent_attempt_id", "") or ""),
        cue_id=str(value.get("cue_id", "") or ""),
        window_radius_sec=float(value.get("window_radius_sec", 5.0) or 5.0),
        cue_stage=str(value.get("cue_stage", "") or ""),
        cue_virtual_time=value.get("cue_virtual_time"),
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
        "evidence_control_mode": result.evidence_control_mode,
        "evidence_state_mode": result.evidence_state_mode,
        "prediction": {
            "answer": result.answer,
            "answer_present": result.answer_present,
        },
        "grounding": {
            "passed": result.reference_valid,
            "errors": list(result.blocking_reasons),
        },
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
        "supporting_item_ids": list(result.supporting_item_ids),
        "supporting_attempt_ids": list(result.supporting_attempt_ids),
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
