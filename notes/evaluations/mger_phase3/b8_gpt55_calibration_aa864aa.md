# MGER Phase 3 B8: GPT-5.5 Judge Calibration

Date: 2026-08-04

## Scope

- Frozen predictions: `cases10-adaptive-74f012d-r1-20260804`
- Calibration root: `/home/xuboshen/zgw/mger_runs/cases10-calibration-gpt55-74f012d-20260804`
- Evaluator commit: `aa864aa`
- Development judge: `pa/gmn-2.5-pr`
- Requested calibration judge: `mmu-0-2-openai-swedencentral-gpt-5.5`
- Official MM-Lifelong prompt SHA-256: `239156295b8e3331e1c73e95c3bf88ca21cc94e99c1a85e5b5cef65a5ac9729b`
- Vendored upstream revision: `4244a9f1981ed2d3f3e0cb7f628b60f8b8b59918`
- Evaluator revision: `ae22859b3059bc2b63033c5fec7dfe1a09730bfd057fdcd69822ddcce4952065`

The official prompt, parser, and 0-5 score mapping were unchanged. The same ten
frozen predictions were evaluated by both judges. Gold data was used only by
the evaluator and was not available to the runtime.

## Results

| Case | Development score | GPT-5.5 score | GPT-5.5 raw score | Agreement |
| --- | ---: | ---: | ---: | --- |
| 0031 | 1.0 | 0.0 | 0 | no |
| 0038 | 0.0 | 0.0 | 0 | yes |
| 0072 | 0.0 | 0.0 | 0 | yes |
| 0097 | 0.0 | 0.0 | 0 | yes |
| 0108 | 0.0 | 0.0 | 0 | yes |
| 0117 | 1.0 | 1.0 | 5 | yes |
| 0119 | 0.0 | 0.0 | 0 | yes |
| 0146 | 1.0 | 1.0 | 5 | yes |
| 0165 | 1.0 | 0.0 | 0 | no |
| 0190 | 0.0 | 0.0 | 0 | yes |

- Judge agreement: `8/10 = 0.80`
- Development accuracy: `4/10 = 0.40`
- GPT-5.5 accuracy: `2/10 = 0.20`
- Accuracy delta, GPT-5.5 minus development: `-0.20`
- Parse success: `10/10`; every response parsed on the first attempt
- Case-level disagreements: `0031`, `0165`

Under the frozen baseline reference-valid flags, GPT-5.5 yields one
Grounded-Correct case (`0117`), four Wrong-but-Verified cases (`0031`, `0038`,
`0072`, `0097`), and one Correct-but-Ungrounded case (`0146`).

## Protocol Status

`official_protocol=true` because the vendored prompt, parser, and mapping were
used. `official_judge_model_match=false` and
`official_judge_config_match=false` are expected: the evaluator's frozen
official model identifier is `gpt-5`, while this run intentionally used the
user-requested internal GPT-5.5 model. Its gateway rejects an explicit
temperature parameter, so the request omitted temperature and recorded
`judge_temperature=null`.

## Decision

Keep the calibration. Report GPT-5.5 results separately from the development
judge and do not reinterpret the 0031/0165 disagreements as framework gains.
The judge remains evaluation-only and is not used for retrieval, evidence
state transitions, closure, or answer selection.
