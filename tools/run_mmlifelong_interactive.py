#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from benchmarks.mmlifelong.adapter import runtime_question_from_case
from benchmarks.mmlifelong.runner import prediction_artifact
from vcah.caption_schema import stable_digest
from vcah.embedding_adapter import SentenceTransformerEmbeddingAdapter
from vcah.interactive_agents import VisionInvestigator, WorkspaceReasoner
from vcah.model_client import OpenAICompatibleClient
from vcah.multiround import VirtualVideoMultiRoundDriver
from vcah.runtime_metrics import agent_run_metrics
from vcah.virtual_video import VirtualVideoWorkspace
from vcah.workspace import evidence_attempt_id


def main() -> None:
    args = _parse_args()
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

    reasoner_api = OpenAICompatibleClient.from_yaml(Path(args.config), section=args.reasoner_section)
    investigator_api = OpenAICompatibleClient.from_yaml(Path(args.config), section=args.investigator_section)
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
    investigator = VisionInvestigator(
        workspace,
        api=investigator_api,
        trace_path=trace_path,
        caption_embedding_adapter=embedding_adapter,
        caption_index_mode=args.caption_index_mode,
        caption_config_digest=args.caption_config_digest,
        caption_query_strategy=args.caption_query_strategy,
    )
    driver = VirtualVideoMultiRoundDriver(
        reasoner=WorkspaceReasoner(reasoner_api, trace_path=trace_path),
        investigator=investigator,
        max_rounds=args.max_rounds,
        max_investigations=args.max_investigations,
        max_tasks_per_round=args.max_tasks_per_round,
        control_retry_budget=args.control_retry_budget,
        require_obligation_coverage=True,
        require_item_provenance=True,
        answer_policy=args.answer_policy,
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

    config = {
        "schema_version": "MMLifelongRunConfigV1",
        "case_id": workspace.case.case_id,
        "answer_policy": args.answer_policy,
        "max_rounds": args.max_rounds,
        "semantic_round_budget": args.max_rounds,
        "control_retry_budget": args.control_retry_budget,
        "require_obligation_coverage": True,
        "require_item_provenance": True,
        "max_investigations": args.max_investigations,
        "max_tasks_per_round": args.max_tasks_per_round,
        "caption_index_mode": args.caption_index_mode,
        "caption_query_strategy": args.caption_query_strategy,
        "caption_query_policy": investigator.caption_query_policy,
        "effective_caption_query_strategy": investigator.caption_query_strategy,
        "caption_config_digest": args.caption_config_digest,
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
            "reasoner": reasoner_api.model,
            "investigator": investigator_api.model,
        },
        "web_enabled": False,
    }
    config["config_digest"] = stable_digest(config)
    _write_json(workspace.root_dir / "run_config.json", config)
    cited_evidence_ids = set(result.citations)
    supporting_attempt_ids = tuple(
        evidence_attempt_id(record)
        for record in result.evidence
        if record.evidence_id in cited_evidence_ids
    )
    prediction = prediction_artifact(
        runtime_question,
        answer=result.answer,
        selected_option=result.selected_option,
        supporting_intervals=result.supporting_intervals,
        supporting_attempt_ids=supporting_attempt_ids,
        answer_present=result.answer_present,
        candidate_answer=result.candidate_answer,
        verified_answer=result.verified_answer,
        verification_status=result.verification_status,
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
                "rounds": result.rounds,
                "investigations": result.investigation_count,
                "config_digest": config["config_digest"],
                "workspace": str(workspace.root_dir),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def _read_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        dict(value)
        for line in path.read_text(encoding="utf-8").splitlines()
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
        Path("src/vcah/workspace.py"),
        Path("src/vcah/multiround.py"),
        Path("src/vcah/interactive_agents.py"),
        Path("src/vcah/investigator.py"),
        Path("src/vcah/caption_lexical_index.py"),
        Path("src/vcah/caption_semantic_index.py"),
        Path("src/vcah/caption_hybrid_search.py"),
        Path("src/vcah/caption_occurrence.py"),
        Path("src/vcah/embedding_adapter.py"),
        Path("src/vcah/runtime_metrics.py"),
        Path("src/vcah/evidence_state.py"),
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
    parser.add_argument("--reasoner-section", default="investigator_api")
    parser.add_argument("--investigator-section", default="investigator_api")
    parser.add_argument("--answer-policy", choices=("strict", "benchmark_best_effort"), default="benchmark_best_effort")
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
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-revision")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    main()
