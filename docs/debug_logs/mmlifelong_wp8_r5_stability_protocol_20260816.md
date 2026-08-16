# MM-Lifelong WP8 R5 Stability Replicate Protocol

## Objective

Measure stochastic stability of the unchanged live R5 working point on the same
frozen 39-case cohort. This replicate does not test out-of-cohort generalization
and does not authorize A4.2, A5, WP7, signed-evidence scoring, or prompt changes.

## Frozen Inputs

- Runtime code: commit `60930dd` (`Use comparative occurrence sufficiency`).
- Cases: `/home/xuboshen/zgw/mger_runs/oracle-day-prepared-fb25-20260811/cases`.
- Cohort manifest: `/home/xuboshen/zgw/mger_runs/occurrence-agent-protocol-20260813/day_val39_frozen_complete_pre_wp3_25cbe03.json`.
- Cohort SHA256: `5119f56e3272b8eedf9e1c1226e51ef7bd6659e9fbc6f8fbbe7c4cc7902d792f`.
- Occurrence replay: `/home/xuboshen/zgw/mger_runs/occurrence-agent-day-val60-replay-fixtures-25cbe03-20260814`.
- Matched pre-treatment replay: `/home/xuboshen/zgw/mger_runs/occurrence-agent-day-val39-wp8-matched-pre-fixtures-8c50115-20260816`.
- A3 matched control: `/home/xuboshen/zgw/mger_runs/occurrence-agent-day-val39-wp8-a3-8c50115-20260816`.
- Runtime and judge models, controller, caption index, embedding revision, budgets,
  and retry policy are identical to the first live R5 run.
- Runtime arm: A4 only. The existing matched A3 control is not rerun.

## New Artifacts

- A4 runtime root: `/home/xuboshen/zgw/mger_runs/occurrence-agent-day-val39-wp8-r5-replicate2-a4-60930dd-20260816`.
- Evaluation root: `/home/xuboshen/zgw/mger_runs/occurrence-agent-day-val39-wp8-r5-replicate2-60930dd-gpt55-eval-20260816`.
- Runtime workers: 16; per-case timeout: 1800 seconds.
- Judge: fixed GPT-5.5 planner API; workers 16; timeout 600 seconds;
  retries 2; maximum completion tokens 4096.
- Final paired bootstrap: 10,000 samples, seed `20260816`.

The runtime API does not expose a deterministic model-sampling seed. This is one
new API realization under otherwise frozen inputs.

## Predeclared Stability Criteria

All structural gates must pass. Endpoint values are not structural gates.

- False-commit rate: 20% to 30% on candidate-absent cases.
- Commit recall: 60% to 75% on candidate-present cases.
- OSA given commit: at least 85%.
- Bound-visual clue recall: 12% to 22%.
- Selected-locator accounting: zero silent drops.

The replicate is called stable only if every interval criterion above is met.
Borderline integer effects and confidence intervals are reported without
post-hoc widening of the bands.

## Reporting

Report modules in this order: gate, resolver, actionability, grounding, answer
quality, and cost. Raw QA is secondary. Zero verified correctness is a grounding
health warning, not an experiment-validity gate. Compare the first live R5 run
and this replicate directly, and label any inference as same-cohort stochastic
stability rather than generalization.

Offline R5 selection on this same cohort remains exploratory and must be stated
in every result summary.
