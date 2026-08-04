# MM-Lifelong Evaluation

## Boundary

The agent run ends after writing `prediction.json`, `runtime_summary.json`, and
runtime provenance. This evaluator can be rerun against the same prediction
without rerunning video acquisition and writes only under `run_dir/evaluation/`.

## Upstream

- Repository: `https://huggingface.co/datasets/MM-Lifelong/MM-Lifelong`
- Pinned revision: `4244a9f1981ed2d3f3e0cb7f628b60f8b8b59918`
- Vendored files: `vendor/upstream/eval_acc.py` and `vendor/upstream/eval_ref.py`
- Dataset-card license at the pinned revision: `cc-by-4.0`

The earlier `CG-Bench/MM-Lifelong` namespace referenced by the coding plan is
recorded as a previous alias in `UPSTREAM.json`; the current public repository
and exact revision are the provenance authority used here.

The vendored files are byte-for-byte upstream copies and must not be edited.
Local wrappers live in `evaluator.py` and `metrics.py`.

## Official Metrics

Answer accuracy uses the exact upstream system prompt, user-prompt layout,
`Final Score` parser, and `score_mapping` behavior. The current upstream script
asks GPT-5 for a raw integer from 0 through 5, then maps `4/5 -> 1`, `3 -> 0.5`,
and `0/1/2 -> 0`. The published evaluator artifact exposes both `raw_score` and
the final three-valued `score`; only the latter is benchmark accuracy.

Reference grounding uses the exact half-open, clamped bucket IoU semantics from
upstream `eval_ref.py`, reported as `ref_60`, `ref_300`, and `ref_600` in the
native `[0, 1]` scale.

Official-comparable answer results require:

- judge model exactly `gpt-5`;
- temperature `0.0`;
- the pinned official prompt and parser;
- successful parse and complete evaluation provenance.

Other judge models remain useful development diagnostics, but are marked with
`official_judge_model_match=false` and must not be mixed into an official main
table.

## VCAH Diagnostics

Exact string match, clue-frame coverage, retrieval diagnostics, material
novelty, interpretation counts, and runtime closure are not official answer
accuracy. They remain clearly separated under diagnostics/runtime fields.

## CLI

Evaluate against a prepared evaluator-only record:

```bash
PYTHONPATH=src:. python -m evaluate.mmlifelong.cli \
  --run-dir runs/mmlifelong/game-test-0031 \
  --evaluation-record data/mmlifelong/cases/game/test/mmlifelong-game-test-0031/evaluation_case.json \
  --config /path/to/api.yaml \
  --judge-section judge_api
```

Or resolve the reference from the pinned/local dataset layout:

```bash
PYTHONPATH=src:. python -m evaluate.mmlifelong.cli \
  --run-dir runs/mmlifelong/game-test-0031 \
  --dataset-root /path/to/MM-Lifelong \
  --config /path/to/api.yaml \
  --judge-section judge_api
```

For deterministic offline re-parsing of a stored official response, replace
the API options with `--judge-response-file <path>`. Output files are:

```text
evaluation/mmlifelong_eval.json
evaluation/judge_response.json
evaluation/eval_provenance.json
```
