#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from benchmarks.mmlifelong.adapter import runtime_question_from_case
from benchmarks.mmlifelong.runner import prediction_artifact
from benchmarks.mmlifelong.oracle import (
    ORACLE_ARMS,
    CaptionPacketIntervention,
    bootstrap_tasks,
    load_oracle_intervention,
)
from vcah.caption_schema import stable_digest
from vcah.embedding_adapter import SentenceTransformerEmbeddingAdapter
from vcah.interactive_agents import VisionInvestigator, WorkspaceReasoner
from vcah.model_client import OpenAICompatibleClient
from vcah.multiround import VirtualVideoMultiRoundDriver
from vcah.occurrence_agent import (
    OCCURRENCE_METHOD_ARMS,
    OccurrencePacketTransform,
    validate_occurrence_method_configuration,
)
from vcah.phase5 import Phase5Protocol, blind_prior_prompt
from vcah.phase5r import (
    MechanicalReplayClient,
    RecordedDecisionReasoner,
    build_run_provenance,
    frame_cost_breakdown,
    load_fixture,
    mechanical_replay_audit,
    runtime_decision_trace,
)
from vcah.runtime_metrics import agent_run_metrics
from vcah.virtual_video import VirtualVideoWorkspace


def main() -> None:
    args = _parse_args()
    occurrence_method_arm = validate_occurrence_method_configuration(
        method_arm=args.occurrence_method_arm,
        oracle_arm=args.oracle_arm,
        oracle_intervention=args.oracle_intervention,
    )
    if args.occurrence_replay_fixture and args.occurrence_replay_record:
        raise ValueError(
            "--occurrence-replay-fixture and --occurrence-replay-record are mutually exclusive"
        )
    if (
        args.occurrence_replay_fixture or args.occurrence_replay_record
    ) and occurrence_method_arm == "none":
        raise ValueError("occurrence replay requires an occurrence method arm")
    protocol = Phase5Protocol(
        controller_mode=args.controller_mode,
        controller_evidence_visibility=args.controller_evidence_visibility,
        measurement_control=args.measurement_control,
    )
    source = VirtualVideoWorkspace.load(Path(args.case_workspace))
    run_root = Path(args.out_dir)
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"run output is not empty: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    runtime_question = runtime_question_from_case(
        {
            "case_id": source.case.case_id,
            "question": source.case.question,
            "options": source.case.options,
            "question_type": source.case.question_type,
            "subset": source.case.subset,
            "split": source.case.split,
            "runtime_metadata": source.case.metadata,
        }
    )
    runtime_case_payload = runtime_question.to_dict()
    runtime_case_payload["asset_ref"] = str(source.asset_root.resolve())
    _write_json(run_root / "case.json", runtime_case_payload)
    workspace = VirtualVideoWorkspace.load(run_root)

    oracle_intervention = None
    oracle_packet_transform = None
    occurrence_packet_transform = None
    caption_packet_transform = None
    if args.oracle_arm == "o0":
        if args.oracle_intervention:
            raise ValueError("O0 must not load an oracle intervention manifest")
    else:
        if not args.oracle_intervention:
            raise ValueError(f"{args.oracle_arm} requires --oracle-intervention")
        oracle_intervention = load_oracle_intervention(
            Path(args.oracle_intervention)
        )
        if oracle_intervention.case_id != workspace.case.case_id:
            raise ValueError("oracle intervention case_id does not match runtime case")
        if oracle_intervention.caption_config_digest != str(
            args.caption_config_digest or ""
        ):
            raise ValueError("oracle intervention Caption digest mismatch")
        oracle_packet_transform = CaptionPacketIntervention(
            arm=args.oracle_arm,
            intervention=oracle_intervention,
            workspace=workspace,
            audit_path=workspace.root_dir / "oracle_intervention_audit.json",
        )
        caption_packet_transform = oracle_packet_transform
    if occurrence_method_arm != "none":
        occurrence_packet_transform = OccurrencePacketTransform(
            arm=occurrence_method_arm,
            audit_path=workspace.root_dir / "no_oracle_runtime_audit.json",
            case_id=workspace.case.case_id,
            caption_config_digest=str(args.caption_config_digest or ""),
            replay_fixture_path=(
                Path(args.occurrence_replay_fixture)
                if args.occurrence_replay_fixture
                else None
            ),
            replay_record_path=(
                Path(args.occurrence_replay_record)
                if args.occurrence_replay_record
                else None
            ),
        )
        occurrence_packet_transform.validate_surface(
            runtime_case_payload,
            surface="runtime_case",
        )
        caption_packet_transform = occurrence_packet_transform

    recorded_fixture = (
        load_fixture(Path(args.recorded_decisions))
        if args.recorded_decisions
        else None
    )
    if recorded_fixture and (
        protocol.controller_mode != "frozen_baseline"
        or protocol.measurement_control != "none"
    ):
        raise ValueError(
            "recorded replay requires --controller-mode frozen_baseline and "
            "--measurement-control none"
        )
    if recorded_fixture and str(recorded_fixture.get("case_id", "")) != workspace.case.case_id:
        raise ValueError("recorded replay fixture case_id does not match the case workspace")
    reasoner_api = (
        None
        if recorded_fixture
        else OpenAICompatibleClient.from_yaml(
            Path(args.reasoner_config or args.config),
            section=args.reasoner_section,
        )
    )
    if protocol.measurement_control == "blind_prior":
        assert reasoner_api is not None
        _run_blind_prior(
            workspace,
            source=source,
            runtime_question=runtime_question,
            reasoner_api=reasoner_api,
            protocol=protocol,
        )
        return
    if protocol.controller_mode == "minimal_tool":
        raise RuntimeError(
            "minimal_tool is gated by Phase 5 Gate-0 and is not enabled in the measurement-control revision"
        )
    investigator_api = (
        MechanicalReplayClient()
        if recorded_fixture
        else OpenAICompatibleClient.from_yaml(
            Path(args.investigator_config or args.config),
            section=args.investigator_section,
        )
    )
    embedding_adapter = None
    if args.caption_index_mode in {"dense", "hybrid"}:
        if not args.embedding_model:
            raise ValueError("--embedding-model is required for dense or hybrid caption search")
        embedding_adapter = SentenceTransformerEmbeddingAdapter(
            args.embedding_model,
            revision=args.embedding_revision,
            device=args.embedding_device,
            normalize=True,
            batch_size=args.embedding_batch_size,
        )

    trace_path = workspace.root_dir / "interactions.jsonl"
    trace_path.touch(exist_ok=False)
    reasoner = (
        RecordedDecisionReasoner(recorded_fixture, trace_path=trace_path)
        if recorded_fixture
        else WorkspaceReasoner(
            reasoner_api,
            trace_path=trace_path,
            controller_mode=protocol.controller_mode,
            controller_evidence_visibility=protocol.controller_evidence_visibility,
            measurement_control=protocol.measurement_control,
        )
    )
    investigator = VisionInvestigator(
        workspace,
        api=investigator_api,
        trace_path=trace_path,
        caption_embedding_adapter=embedding_adapter,
        caption_index_mode=args.caption_index_mode,
        caption_config_digest=args.caption_config_digest,
        caption_query_strategy=args.caption_query_strategy,
        caption_packet_transform=caption_packet_transform,
        anchor_execution_policy=(
            "force_if_requested"
            if args.oracle_arm == "o1.75-forced"
            else "agent_controlled"
        ),
    )
    effective_control_retry_budget = (
        0
        if recorded_fixture
        else 1
        if protocol.controller_mode == "frozen_baseline"
        else args.control_retry_budget
    )
    effective_evidence_control_mode = (
        args.evidence_control_mode if protocol.controller_mode == "mger" else "shadow"
    )
    effective_evidence_state_mode = (
        args.evidence_state_mode
        if protocol.controller_mode == "mger"
        else "llm_authored"
    )
    driver = VirtualVideoMultiRoundDriver(
        reasoner=reasoner,
        investigator=investigator,
        # Recorded decisions must traverse the same semantic-budget boundary as
        # their source run. The driver's finalize path can consume decisions
        # recorded after that boundary without dispatching their tasks.
        max_rounds=args.max_rounds,
        max_investigations=args.max_investigations,
        max_tasks_per_round=args.max_tasks_per_round,
        control_retry_budget=effective_control_retry_budget,
        require_obligation_coverage=protocol.controller_mode == "mger",
        require_item_provenance=protocol.controller_mode == "mger",
        require_evidence_kind_requirements=protocol.controller_mode == "mger",
        closure_repair_budget=1 if protocol.controller_mode == "mger" else 0,
        answer_policy=args.answer_policy,
        evidence_control_mode=effective_evidence_control_mode,
        evidence_state_mode=effective_evidence_state_mode,
        allowed_inspection_modes=protocol.allowed_inspection_modes,
        controller_mode=protocol.controller_mode,
        bootstrap_tasks=bootstrap_tasks(
            arm=args.oracle_arm,
            question=workspace.case.question,
            index_mode=args.caption_index_mode,
        ),
        occurrence_method_arm=occurrence_method_arm,
    )
    result = driver.run(workspace)
    observation_rows = _read_jsonl(workspace.root_dir / "observation_log.jsonl")
    runtime_metrics = agent_run_metrics(
        result.trace,
        observation_rows,
        answer_present=result.answer_present,
        reference_valid=result.reference_valid,
        supporting_intervals=result.supporting_intervals,
    )
    frame_rows = _read_jsonl(
        workspace.root_dir / "observations" / "window_frame_manifest.jsonl"
    )
    decision_trace = runtime_decision_trace(result.trace)
    cost_breakdown = frame_cost_breakdown(result.trace, observation_rows, frame_rows)
    role_settings = {
        "reasoner": (
            {
                "model": f"recorded:{reasoner.source_revision}",
                "temperature": None,
                "top_p": None,
                "requested_seed": None,
                "provider_seed_supported": False,
                "provider_reported_seed_support": "not_applicable",
            }
            if isinstance(reasoner, RecordedDecisionReasoner)
            else reasoner_api.replay_settings
        ),
        "investigator": investigator_api.replay_settings,
    }
    repository_root = Path(__file__).resolve().parents[1]
    provenance = build_run_provenance(
        workspace,
        interactions_path=trace_path,
        role_settings=role_settings,
        caption_index_digest=args.caption_config_digest,
        repository_root=repository_root,
    )

    config = {
        "schema_version": "MMLifelongRunConfigV1",
        "case_id": workspace.case.case_id,
        **protocol.to_dict(),
        "answer_policy": args.answer_policy,
        "evidence_control_mode": effective_evidence_control_mode,
        "evidence_state_mode": effective_evidence_state_mode,
        "max_rounds": driver.max_rounds,
        "semantic_round_budget": driver.semantic_round_budget,
        "control_retry_budget": effective_control_retry_budget,
        "require_obligation_coverage": protocol.controller_mode == "mger",
        "require_item_provenance": protocol.controller_mode == "mger",
        "require_evidence_kind_requirements": protocol.controller_mode == "mger",
        "closure_repair_budget": 1 if protocol.controller_mode == "mger" else 0,
        "max_investigations": args.max_investigations,
        "max_tasks_per_round": args.max_tasks_per_round,
        "caption_index_mode": args.caption_index_mode,
        "caption_query_strategy": args.caption_query_strategy,
        "caption_query_policy": investigator.caption_query_policy,
        "effective_caption_query_strategy": investigator.caption_query_strategy,
        "caption_config_digest": args.caption_config_digest,
        "occurrence_method_arm": occurrence_method_arm,
        "occurrence_replay": (
            dict(occurrence_packet_transform.audit.get("occurrence_replay", {}))
            if occurrence_packet_transform is not None
            else None
        ),
        "no_oracle_runtime_gate": (
            {
                "schema_version": "MMLifelongNoOracleRuntimeGateV1",
                "method_arm": occurrence_method_arm,
                "passed": True,
                "audit_file": "no_oracle_runtime_audit.json",
            }
            if occurrence_packet_transform is not None
            else None
        ),
        "oracle_arm": args.oracle_arm,
        "anchor_execution_policy": investigator.anchor_execution_policy,
        "oracle_intervention": (
            {
                "schema_version": "MMLifelongOracleInterventionV1",
                "digest": oracle_intervention.digest,
                "manifest_sha256": _file_sha256(Path(args.oracle_intervention)),
            }
            if oracle_intervention is not None
            else None
        ),
        "embedding": dict(embedding_adapter.manifest) if embedding_adapter else None,
        "caption_index_digests": sorted(
            {
                str(row.get("sampling_config", {}).get("index_digest"))
                for row in observation_rows
                if isinstance(row.get("sampling_config"), Mapping)
                and row["sampling_config"].get("index_digest")
            }
        ),
        "implementation_digest": _implementation_digest(),
        "input_digest": _input_digest(source, runtime_question.to_dict()),
        "models": {
            "reasoner": role_settings["reasoner"]["model"],
            "investigator": investigator_api.model,
        },
        "api_bindings": {
            "reasoner": {
                "config_name": Path(args.reasoner_config or args.config).name,
                "section": args.reasoner_section,
            },
            "investigator": {
                "config_name": Path(args.investigator_config or args.config).name,
                "section": args.investigator_section,
            },
        },
        "phase5r_mode": "recorded_replay" if recorded_fixture else "live",
        "recorded_fixture_digest": (
            _file_sha256(Path(args.recorded_decisions)) if recorded_fixture else None
        ),
        "phase5r_provenance": provenance,
        "web_enabled": False,
        "supporting_interval_source": "explicit_support",
    }
    config["config_digest"] = stable_digest(config)
    _write_json(workspace.root_dir / "run_config.json", config)
    prediction = prediction_artifact(
        runtime_question,
        answer=result.answer,
        selected_option=result.selected_option,
        supporting_intervals=result.supporting_intervals,
        supporting_attempt_ids=result.supporting_attempt_ids,
        supporting_item_ids=result.supporting_item_ids,
        answer_present=result.answer_present,
        candidate_answer=result.candidate_answer,
        verified_answer=result.verified_answer,
        verification_status=result.verification_status,
        grounding_passed=result.reference_valid,
        grounding_errors=result.blocking_reasons,
        duration_sec=workspace.manifest.duration_sec,
    )
    _write_json(workspace.root_dir / "prediction.json", prediction)
    summary_path = workspace.root_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.pop("correct", None)
    summary.pop("correctness_source", None)
    summary["schema_version"] = "RuntimeSummaryV1"
    summary["runtime_metrics"] = runtime_metrics
    summary["config_digest"] = config["config_digest"]
    summary["decision_trace"] = decision_trace
    summary["phase5r_cost_breakdown"] = cost_breakdown
    summary["oracle_arm"] = args.oracle_arm
    summary["occurrence_method_arm"] = occurrence_method_arm
    summary["no_oracle_runtime_gate"] = (
        dict(occurrence_packet_transform.audit)
        if occurrence_packet_transform is not None
        else None
    )
    summary["oracle_intervention_audit"] = (
        dict(oracle_packet_transform.audit)
        if oracle_packet_transform is not None
        else None
    )
    if recorded_fixture:
        replay_audit = mechanical_replay_audit(
            recorded_fixture,
            workspace_root=workspace.root_dir,
            trace=result.trace,
            observation_rows=observation_rows,
        )
        summary["phase5r_replay"] = {
            "decision": replay_audit["decision"],
            "failed_checks": replay_audit["failed_checks"],
        }
        _write_json(workspace.root_dir / "phase5r_replay.json", replay_audit)
    _write_json(summary_path, summary)
    _write_json(workspace.root_dir / "runtime_summary.json", summary)

    print(
        json.dumps(
            {
                "case_id": result.case_id,
                "answer_present": result.answer_present,
                "reference_valid": result.reference_valid,
                "prediction": str(workspace.root_dir / "prediction.json"),
                "runtime_summary": str(workspace.root_dir / "runtime_summary.json"),
                "runtime_metrics": runtime_metrics,
                "phase5r_replay": summary.get("phase5r_replay"),
                "rounds": result.rounds,
                "investigations": result.investigation_count,
                "config_digest": config["config_digest"],
                "oracle_arm": args.oracle_arm,
                "occurrence_method_arm": occurrence_method_arm,
                "workspace": str(workspace.root_dir),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def _run_blind_prior(
    workspace: VirtualVideoWorkspace,
    *,
    source: VirtualVideoWorkspace,
    runtime_question: Any,
    reasoner_api: OpenAICompatibleClient,
    protocol: Phase5Protocol,
) -> None:
    prompt = blind_prior_prompt(workspace.case.question)
    raw_answer = reasoner_api.chat(prompt, max_tokens=4096)
    answer = str(raw_answer or "").strip()
    answer_present = bool(answer)
    trace_path = workspace.root_dir / "interactions.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "type": "blind_prior_answer",
                "model": reasoner_api.model,
                "input_fields": ["question"],
                "prompt": prompt,
                "raw": raw_answer,
                "api_response": reasoner_api.last_response_metadata,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for name in ("observation_log.jsonl", "workspace_ops.jsonl"):
        (workspace.root_dir / name).write_text("", encoding="utf-8")

    runtime_metrics: dict[str, float | int | None] = {
        "answer_rate": float(answer_present),
        "reference_valid_rate": 0.0,
        "observed_case_rate": 0.0,
        "conditional_visual_frames": None,
        "reasoner_decision_attempt_count": 1,
        "malformed_decision_count": 0,
        "malformed_decision_rate": 0.0,
        "caption_searches": 0,
        "unique_visual_material_attempts": 0,
        "visual_interpretation_count": 0,
        "visual_reinterpretation_count": 0,
        "visual_frames_inspected": 0,
        "requested_acquisition_count": 0,
        "executed_acquisition_count": 0,
        "silently_dropped_acquisition_count": 0,
    }
    config = {
        "schema_version": "MMLifelongRunConfigV1",
        "case_id": workspace.case.case_id,
        **protocol.to_dict(),
        "answer_policy": "benchmark_best_effort",
        "evidence_control_mode": "shadow",
        "evidence_state_mode": "llm_authored",
        "measurement_input_fields": ["question"],
        "available_tools": [],
        "max_rounds": 1,
        "semantic_round_budget": 1,
        "control_retry_budget": 0,
        "require_obligation_coverage": False,
        "require_item_provenance": False,
        "require_evidence_kind_requirements": False,
        "closure_repair_budget": 0,
        "caption_index_mode": "disabled",
        "caption_query_strategy": "disabled",
        "caption_config_digest": None,
        "embedding": None,
        "caption_index_digests": [],
        "implementation_digest": _implementation_digest(),
        "input_digest": _input_digest(source, runtime_question.to_dict()),
        "models": {"reasoner": reasoner_api.model, "investigator": None},
        "web_enabled": False,
        "supporting_interval_source": "none",
    }
    config["config_digest"] = stable_digest(config)
    _write_json(workspace.root_dir / "run_config.json", config)
    prediction = prediction_artifact(
        runtime_question,
        answer=answer if answer_present else "No answer was returned.",
        selected_option="",
        supporting_intervals=(),
        supporting_attempt_ids=(),
        answer_present=answer_present,
        candidate_answer=answer,
        verified_answer="",
        verification_status="candidate_only" if answer_present else "missing",
        grounding_passed=False,
        grounding_errors=("blind_prior_has_no_observations",),
        duration_sec=workspace.manifest.duration_sec,
    )
    _write_json(workspace.root_dir / "prediction.json", prediction)
    summary = {
        "schema_version": "RuntimeSummaryV1",
        "case_id": workspace.case.case_id,
        "answer": prediction["answer"],
        "answer_present": answer_present,
        "reference_valid": False,
        "runtime_metrics": runtime_metrics,
        "config_digest": config["config_digest"],
    }
    _write_json(workspace.root_dir / "run_summary.json", summary)
    _write_json(workspace.root_dir / "runtime_summary.json", summary)
    print(
        json.dumps(
            {
                "case_id": workspace.case.case_id,
                "phase5_arm": protocol.arm,
                "answer_present": answer_present,
                "prediction": str(workspace.root_dir / "prediction.json"),
                "runtime_summary": str(workspace.root_dir / "runtime_summary.json"),
                "runtime_metrics": runtime_metrics,
                "config_digest": config["config_digest"],
                "workspace": str(workspace.root_dir),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def _read_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    if not Path(path).is_file():
        return ()
    return tuple(
        dict(value)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and isinstance((value := json.loads(line)), Mapping)
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _implementation_digest() -> str:
    repository_root = Path(__file__).resolve().parents[1]
    relative_paths = (
        Path("tools/run_mmlifelong_interactive.py"),
        Path("benchmarks/mmlifelong/oracle.py"),
        Path("src/vcah/workspace.py"),
        Path("src/vcah/multiround.py"),
        Path("src/vcah/interactive_agents.py"),
        Path("src/vcah/investigator.py"),
        Path("src/vcah/caption_lexical_index.py"),
        Path("src/vcah/caption_semantic_index.py"),
        Path("src/vcah/caption_hybrid_search.py"),
        Path("src/vcah/caption_occurrence.py"),
        Path("src/vcah/occurrence_agent.py"),
        Path("src/vcah/embedding_adapter.py"),
        Path("src/vcah/runtime_metrics.py"),
        Path("src/vcah/phase5.py"),
        Path("src/vcah/phase5r.py"),
        Path("src/vcah/evidence_state.py"),
        Path("src/vcah/evidence_runtime.py"),
        Path("src/vcah/sampling.py"),
        Path("src/vcah/temporal_scope.py"),
        Path("benchmarks/schema.py"),
        Path("benchmarks/mmlifelong/runner.py"),
    )
    return stable_digest(
        {
            path.as_posix(): _file_sha256(repository_root / path)
            for path in relative_paths
        }
    )


def _input_digest(
    source: VirtualVideoWorkspace,
    runtime_question: Mapping[str, Any],
) -> str:
    caption_files = tuple(sorted((source.asset_root / "captions").glob("passages.*.jsonl")))
    return stable_digest(
        {
            "runtime_question": stable_digest(runtime_question),
            "timeline": _file_sha256(source.asset_root / "virtual_timeline.json"),
            "captions": {path.name: _file_sha256(path) for path in caption_files},
        }
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one MM-Lifelong Day case with the workspace agent.")
    parser.add_argument("--case-workspace", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", required=True, help="OpenAI-compatible API YAML; secrets are not copied.")
    parser.add_argument("--reasoner-config")
    parser.add_argument("--investigator-config")
    parser.add_argument("--reasoner-section", default="investigator_api")
    parser.add_argument("--investigator-section", default="investigator_api")
    parser.add_argument("--answer-policy", choices=("strict", "benchmark_best_effort"), default="benchmark_best_effort")
    parser.add_argument(
        "--controller-mode",
        choices=("frozen_baseline", "minimal_tool", "mger"),
        default="mger",
    )
    parser.add_argument(
        "--controller-evidence-visibility",
        choices=("none", "candidates_only", "full"),
        default="full",
    )
    parser.add_argument(
        "--measurement-control",
        choices=("none", "blind_prior", "caption_only"),
        default="none",
    )
    parser.add_argument("--evidence-control-mode", choices=("shadow", "strict"), default="shadow")
    parser.add_argument(
        "--evidence-state-mode",
        choices=("llm_authored", "runtime_derived"),
        default="runtime_derived",
    )
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--max-investigations", type=int, default=12)
    parser.add_argument("--max-tasks-per-round", type=int, default=4)
    parser.add_argument("--control-retry-budget", type=int, default=2)
    parser.add_argument("--caption-index-mode", choices=("lexical", "dense", "hybrid"), default="hybrid")
    parser.add_argument(
        "--caption-query-strategy",
        choices=("joint", "rema", "adaptive"),
        default="joint",
    )
    parser.add_argument("--caption-config-digest")
    parser.add_argument("--oracle-arm", choices=ORACLE_ARMS, default="o0")
    parser.add_argument("--oracle-intervention")
    parser.add_argument(
        "--occurrence-method-arm",
        choices=OCCURRENCE_METHOD_ARMS,
        default="none",
    )
    parser.add_argument("--occurrence-replay-fixture")
    parser.add_argument("--occurrence-replay-record")
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-revision")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument(
        "--recorded-decisions",
        help="Phase 5R compact case fixture; disables Reasoner and Investigator API calls.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
