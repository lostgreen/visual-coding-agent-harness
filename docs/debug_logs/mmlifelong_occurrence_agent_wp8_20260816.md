# MM-Lifelong Occurrence Agent WP8

## Goal

Test whether an explicit, training-free candidate-sufficiency decision before
selection reduces false commits without materially reducing strict selection
accuracy when the correct occurrence is present.

## Frozen mechanism

- New arm: `a4`.
- Pre-selection treatment: an `assess_sufficiency` operation records 1-6
  question-critical constraints and candidate-level support bound only to
  visible candidate passage IDs.
- A `sufficient` verdict permits selection only from candidates supported on
  every declared constraint.
- An `insufficient` verdict permits only `defer` or `no_match`.
- The assessment and terminal resolution may be submitted in one atomic
  occurrence transaction, but the assessment must precede the terminal op.
- After selection, A4 reuses the repaired A3 locator actionability policy.
- The runtime validates structure, visible foreign keys, ordering, and verdict
  consistency. It does not determine semantic sufficiency and does not enforce
  a favorable endpoint.

## Matched-control boundary

- Record A3 model responses before the first scoped occurrence set is exposed.
- Replay those exact responses in A4 through the same exposure boundary.
- All calls after exposure remain live.
- Unlike WP6, scoped resolution identity is not required because WP8 is
  intended to alter selection and abstention.

## Frozen development evaluation

- Cohort: the existing frozen 39-case pre-WP3 development manifest.
- Runtime and judge backbones remain unchanged.
- Do not access Day-test140 or Week.
- Run a small structural canary before the full matched cohort.

Primary endpoints, in order:

1. False-commit rate on resolved-set candidate-absent cases.
2. No-match accuracy on the same cases.
3. Strict occurrence-selection accuracy on candidate-present cases.

Secondary endpoints include false abstention, raw QA, grounding, visual cost,
VLM calls, semantic rounds, and A3 locator mechanism metrics.

## Validity gates

- Exact matched responses through the declared exposure boundary.
- No-oracle and frozen replay identity/ordering gates.
- Exactly one A4 activation per case and at least one sufficiency decision.
- Every terminal resolution follows a recorded assessment with a compatible
  verdict.
- No terminal lifecycle failures, contradictory gates, silent locator drops,
  or invalid visible bindings.

False-commit, no-match, OSA, and locator-use values are endpoints, not gates.

## Local verification

- Focused occurrence, retry, matched-cache, audit, and analyzer tests: 80
  passed.
- Full suite: 472 passed.

## Next actions

1. Commit and push the isolated WP8 implementation.
2. Pull and run the full suite on the KML development machine.
3. Run a matched A3-record/A4-replay structural canary.
4. If all structural gates pass, launch the frozen 39-case matched run and
   judge only after runtime completion.
