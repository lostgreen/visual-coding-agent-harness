#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from vcah.caption_schema import stable_digest
from vcah.embedding_adapter import SentenceTransformerEmbeddingAdapter
from vcah.interactive_agents import VisionInvestigator, WorkspaceReasoner
from vcah.mmlifelong_metrics import (
    agent_run_metrics,
    caption_hits_from_observation_rows,
    judge_free_form_answer,
    ref_scores,
    retrieval_metrics,
)
from vcah.model_client import OpenAICompatibleClient
from vcah.multiround import VirtualVideoMultiRoundDriver
from vcah.virtual_video import VirtualVideoWorkspace


def main() -> None:
    args = _parse_args()
    source = VirtualVideoWorkspace.load(Path(args.case_workspace))
    run_root = Path(args.out_dir)
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"run output is not empty: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    case_payload = json.loads((source.root_dir / "case.json").read_text(encoding="utf-8"))
    case_payload["asset_ref"] = str(source.asset_root.resolve())
    _write_json(run_root / "case.json", case_payload)
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
        answer_policy=args.answer_policy,
    )
    result = driver.run(workspace)
    observation_rows = _read_jsonl(workspace.root_dir / "observation_log.jsonl")
    caption_hits = caption_hits_from_observation_rows(observation_rows)

    evaluation: dict[str, Any] = {
        "schema_version": "MMLifelongCaseEvaluationV1",
        "case_id": workspace.case.case_id,
        "subset": workspace.case.subset,
        "split": workspace.case.split,
        "answer": result.answer,
        "answer_present": result.answer_present,
        "candidate_answer": result.candidate_answer,
        "verified_answer": result.verified_answer,
        "verification_status": result.verification_status,
        "blocking_reasons": list(result.blocking_reasons),
        "reference_valid": result.reference_valid,
        "reference_reason": result.reference_reason,
        "mcq_correct": result.correct if workspace.case.options else None,
        "accuracy_score": float(result.correct) if workspace.case.options else None,
        "correct": result.correct if workspace.case.options else None,
        "correctness_source": "mcq_exact" if workspace.case.options else "unjudged",
        "gold_answer": workspace.case.gold,
        "gold_clue_intervals": [list(item) for item in workspace.case.gold_clue_intervals],
        "supporting_intervals": [list(item) for item in result.supporting_intervals],
        "ref": ref_scores(result.supporting_intervals, workspace.case.gold_clue_intervals),
        "retrieval": retrieval_metrics(caption_hits, workspace.case.gold_clue_intervals),
        "agent": agent_run_metrics(
            result.trace,
            observation_rows,
            answer_present=result.answer_present,
            reference_valid=result.reference_valid,
            supporting_intervals=result.supporting_intervals,
        ),
        "judge": None,
    }
    judge_model = ""
    if not workspace.case.options and result.answer_present and args.judge:
        judge_api = OpenAICompatibleClient.from_yaml(Path(args.config), section=args.judge_section)
        judged = judge_free_form_answer(
            lambda prompt: judge_api.chat(prompt, max_tokens=4096),
            question=workspace.case.question,
            reference_answer=workspace.case.gold,
            predicted_answer=result.answer,
            judge_model=judge_api.model,
            max_retries=args.judge_max_retries,
            response_metadata=lambda: judge_api.last_response_metadata,
        )
        evaluation["judge"] = judged.to_dict()
        evaluation["accuracy_score"] = judged.smoothed_score
        evaluation["correct"], evaluation["correctness_source"] = _correctness_outcome(
            has_options=False,
            mcq_correct=None,
            judge=evaluation["judge"],
        )
        judge_model = judge_api.model

    config = {
        "schema_version": "MMLifelongRunConfigV1",
        "case_id": workspace.case.case_id,
        "answer_policy": args.answer_policy,
        "max_rounds": args.max_rounds,
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
        "input_digest": _input_digest(source),
        "models": {
            "reasoner": reasoner_api.model,
            "investigator": investigator_api.model,
            "judge": judge_model,
        },
        "web_enabled": False,
    }
    config["config_digest"] = stable_digest(config)
    evaluation["config_digest"] = config["config_digest"]
    _write_json(workspace.root_dir / "run_config.json", config)
    _write_json(workspace.root_dir / "mmlifelong_metrics.json", evaluation)
    summary_path = workspace.root_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["correct"] = evaluation["correct"]
    summary["correctness_source"] = evaluation["correctness_source"]
    summary["accuracy_score"] = evaluation["accuracy_score"]
    summary["evaluation"] = evaluation
    summary["config_digest"] = config["config_digest"]
    _write_json(summary_path, summary)

    print(
        json.dumps(
            {
                "case_id": result.case_id,
                "answer_present": result.answer_present,
                "reference_valid": result.reference_valid,
                "judge_smoothed": (
                    evaluation["judge"]["smoothed_score"] if evaluation["judge"] else None
                ),
                "ref": evaluation["ref"],
                "retrieval": evaluation["retrieval"],
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


def _correctness_outcome(
    *,
    has_options: bool,
    mcq_correct: bool | None,
    judge: Mapping[str, Any] | None,
) -> tuple[bool | None, str]:
    if has_options:
        return bool(mcq_correct), "mcq_exact"
    if not isinstance(judge, Mapping) or judge.get("raw_score") is None:
        return None, "unjudged"
    return int(judge["raw_score"]) in {4, 5}, "answer_judge"


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
        Path("src/vcah/mmlifelong_metrics.py"),
    )
    return stable_digest(
        {
            path.as_posix(): _file_sha256(repository_root / path)
            for path in relative_paths
        }
    )


def _input_digest(source: VirtualVideoWorkspace) -> str:
    caption_files = tuple(sorted((source.asset_root / "captions").glob("passages.*.jsonl")))
    return stable_digest(
        {
            "case": _file_sha256(source.root_dir / "case.json"),
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
    parser.add_argument("--judge-section", default="investigator_api")
    parser.add_argument("--answer-policy", choices=("strict", "benchmark_best_effort"), default="benchmark_best_effort")
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--max-investigations", type=int, default=12)
    parser.add_argument("--max-tasks-per-round", type=int, default=4)
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
    parser.add_argument("--judge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--judge-max-retries", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    main()
