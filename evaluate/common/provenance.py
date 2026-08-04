from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluate.common.io import sha256_file, stable_digest


def evaluator_revision(paths: Sequence[Path]) -> str:
    files = {
        Path(path).name: sha256_file(Path(path))
        for path in sorted((Path(path) for path in paths), key=lambda item: item.as_posix())
    }
    return stable_digest(files)


def evaluation_provenance(
    *,
    benchmark: str,
    benchmark_revision: str,
    evaluator_revision_value: str,
    official_eval_file_sha256: str,
    official_ref_file_sha256: str,
    official_prompt_sha256: str,
    judge_model: str,
    judge_endpoint_family: str,
    judge_temperature: float | None,
    prediction_artifact: Path,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "EvaluationProvenanceV1",
        "benchmark": str(benchmark),
        "benchmark_revision": str(benchmark_revision),
        "evaluator_revision": str(evaluator_revision_value),
        "official_eval_file_sha256": str(official_eval_file_sha256),
        "official_ref_file_sha256": str(official_ref_file_sha256),
        "official_prompt_sha256": str(official_prompt_sha256),
        "judge_model": str(judge_model),
        "judge_endpoint_family": str(judge_endpoint_family),
        "judge_temperature": judge_temperature,
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction_artifact_sha256": sha256_file(Path(prediction_artifact)),
        **dict(extra or {}),
    }
