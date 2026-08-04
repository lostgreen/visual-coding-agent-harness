# Phase 3 B3: query isolation and occurrence binding

- Commit: `039cbfe`
- Cases: `0038`, `0108`, `0119`, `0190`
- Runtime root: `/home/xuboshen/zgw/mger_runs/cases4-b3-039cbfe-r1-20260804`
- Development judge: `pa/gmn-2.5-pr` (diagnostic only)

## Structural checks

| Case | Answer | Reference valid | Frames | Caption searches | Whole-question query | Occurrence binding | Temporal scope |
|---|---:|---:|---:|---:|---:|---:|---|
| 0038 | yes | yes | 96 | 1 | no | 1/1 | unresolved candidates |
| 0108 | yes | no | 41 | 2 | no | 2/2 | unresolved anchor (`after+first`) |
| 0119 | yes | no | 96 | 1 | no | 1/1 | unresolved candidates (`after+next`) |
| 0190 | yes | no | 96 | 1 | no | 1/1 | none |

- Silent task drops: `0`.
- Explicit caption queries were isolated; the whole question was not prepended.
- Every executed occurrence request produced material with a persisted locator/occurrence binding.
- The framework did not guess unresolved first/next selections.

## Evaluation

All four judge responses parsed without retry and scored `0.0`.

For case `0038`, evaluated against the frozen reference interval `[19950.0, 19952.0]`:

- `CandidateClueRecall = 0.0`
- `OccurrenceCandidateRecall = 0.0`

## Decision

Keep the query-isolation, explicit temporal-scope, and material-binding contracts. They remove protocol ambiguity and false provenance, but this batch did not improve answer correctness because the decisive occurrence was never retrieved. Treat B3 as an integrity improvement, not a quality win; B4-B7 must prevent unsupported closure and add bounded refinement without weakening these contracts.
