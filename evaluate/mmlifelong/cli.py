from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from benchmarks.mmlifelong.adapter import (
    evaluation_record_from_dataset,
    load_evaluation_record,
)
from evaluate.common.io import read_json, sha256_file, write_json
from evaluate.common.judge_client import OpenAICompatibleJudgeClient
from evaluate.common.provenance import evaluation_provenance, evaluator_revision
from evaluate.common.schema import RuntimePrediction
from evaluate.mmlifelong.evaluator import (
    UPSTREAM_REVISION,
    evaluate_prediction,
)


ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = ROOT / "vendor" / "upstream"
PROMPT_PATH = ROOT / "prompts" / "official_answer_judge.txt"


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    prediction_path = run_dir / "prediction.json"
    prediction_payload = read_json(prediction_path)
    prediction = RuntimePrediction.from_mapping(prediction_payload)
    record = _resolve_evaluation_record(args, prediction, prediction_payload)
    total_seconds = _resolve_total_seconds(args, prediction, record.evaluation_metadata)
    generate, metadata, judge_model, endpoint_family, temperature = _judge(args)

    evaluation, judged = evaluate_prediction(
        prediction,
        record,
        total_seconds=total_seconds,
        generate=generate,
        judge_model=judge_model,
        judge_temperature=temperature,
        max_retries=args.judge_max_retries,
        max_completion_tokens=args.max_completion_tokens,
        response_metadata=metadata,
    )
    output_dir = run_dir / "evaluation"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"evaluation output is not empty: {output_dir}; pass --overwrite to replace named files"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    revision = evaluator_revision(
        (
            ROOT / "evaluator.py",
            ROOT / "metrics.py",
            ROOT / "cli.py",
            ROOT.parent / "common" / "judge_client.py",
        )
    )
    provenance = evaluation_provenance(
        benchmark=str(evaluation["benchmark"]),
        benchmark_revision=UPSTREAM_REVISION,
        evaluator_revision_value=revision,
        official_eval_file_sha256=sha256_file(VENDOR_ROOT / "eval_acc.py"),
        official_ref_file_sha256=sha256_file(VENDOR_ROOT / "eval_ref.py"),
        official_prompt_sha256=sha256_file(PROMPT_PATH),
        judge_model=judge_model,
        judge_endpoint_family=endpoint_family,
        judge_temperature=temperature,
        prediction_artifact=prediction_path,
        extra={
            "official_judge_model_match": evaluation["answer"][
                "official_judge_model_match"
            ],
            "official_judge_config_match": evaluation["answer"][
                "official_judge_config_match"
            ],
            "total_seconds": total_seconds,
        },
    )
    write_json(output_dir / "mmlifelong_eval.json", evaluation)
    write_json(
        output_dir / "judge_response.json",
        {
            "schema_version": "MMLifelongJudgeResponseV1",
            **judged.to_dict(),
        },
    )
    write_json(output_dir / "eval_provenance.json", provenance)
    print(
        json.dumps(
            {
                "case_id": prediction.case_id,
                "score": evaluation["answer"]["score"],
                "raw_score": evaluation["answer"]["raw_score"],
                "parse_status": evaluation["answer"]["parse_status"],
                "official_judge_model_match": evaluation["answer"][
                    "official_judge_model_match"
                ],
                "output": str(output_dir / "mmlifelong_eval.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if judged.parse_status != "parsed":
        raise SystemExit(2)


def _resolve_evaluation_record(
    args: argparse.Namespace,
    prediction: RuntimePrediction,
    prediction_payload: Mapping[str, Any],
):
    if args.evaluation_record:
        return load_evaluation_record(Path(args.evaluation_record))
    if not args.dataset_root:
        raise ValueError("Provide --evaluation-record or --dataset-root")
    metadata = prediction.runtime_metadata
    return evaluation_record_from_dataset(
        Path(args.dataset_root),
        case_id=prediction.case_id,
        subset=args.subset or prediction_payload.get("subset"),
        split=args.split or prediction_payload.get("split"),
        source_index=metadata.get("source_index"),
    )


def _resolve_total_seconds(
    args: argparse.Namespace,
    prediction: RuntimePrediction,
    evaluation_metadata: Mapping[str, Any],
) -> float:
    candidates = (
        args.total_seconds,
        prediction.runtime_metadata.get("duration_sec"),
        evaluation_metadata.get("total_seconds"),
    )
    for value in candidates:
        if value is not None and float(value) > 0.0:
            return float(value)
    raise ValueError(
        "MM-Lifelong Ref@N requires total duration; pass --total-seconds or include duration_sec in prediction.json"
    )


def _judge(
    args: argparse.Namespace,
) -> tuple[
    Callable[[str, str, int], str],
    Callable[[], Mapping[str, Any]],
    str,
    str,
    float | None,
]:
    if args.judge_response_file:
        raw = _offline_response(Path(args.judge_response_file))
        state: dict[str, Any] = {
            "finish_reason": "offline_reparse",
            "completion_tokens": None,
            "reasoning_tokens": None,
        }
        return (
            lambda system_prompt, user_prompt, max_tokens: raw,
            lambda: state,
            str(args.judge_model),
            "offline_response",
            float(args.judge_temperature),
        )
    if not args.config:
        raise ValueError("--config is required unless --judge-response-file is provided")
    client = OpenAICompatibleJudgeClient.from_yaml(
        Path(args.config),
        section=args.judge_section,
    )
    return (
        lambda system_prompt, user_prompt, max_tokens: client.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_completion_tokens=max_tokens,
        ),
        lambda: client.last_response_metadata,
        client.model,
        client.endpoint_family,
        client.temperature,
    )


def _offline_response(path: Path) -> str:
    raw = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not isinstance(payload, Mapping):
        return raw
    for key in ("raw_response", "raw_judge_response", "gpt_raw"):
        if payload.get(key) is not None:
            return str(payload[key])
    answer = payload.get("answer")
    if isinstance(answer, Mapping) and answer.get("raw_judge_response") is not None:
        return str(answer["raw_judge_response"])
    return raw


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an existing runtime prediction with the MM-Lifelong protocol."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--evaluation-record")
    parser.add_argument("--dataset-root")
    parser.add_argument("--subset")
    parser.add_argument("--split")
    parser.add_argument("--total-seconds", type=float)
    parser.add_argument("--config")
    parser.add_argument("--judge-section", default="judge_api")
    parser.add_argument("--judge-response-file")
    parser.add_argument("--judge-model", default="gpt-5")
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-max-retries", type=int, default=2)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
