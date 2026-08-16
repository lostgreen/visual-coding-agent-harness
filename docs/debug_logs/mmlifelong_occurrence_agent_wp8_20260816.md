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
- The model determines every semantic support status. The runtime validates
  structure, visible foreign keys, and ordering, fills omitted candidate rows
  as explicit `unknown`, and mechanically derives the verdict from that support
  matrix. It does not enforce a favorable endpoint.

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

## Full-cohort blocker and repair

The first frozen39 A4 replay completed 39 runtime records but produced only 30
answers. Eight cases ended with unresolved occurrence transactions; one more
used `decision` as the answer-text field and lost the otherwise recoverable
answer during normalization. The failure was after matched exposure, not in
replay identity or the no-oracle boundary.

The structured failure fingerprint was:

- omitted candidate support rows rejected the whole atomic assessment;
- declared verdicts could disagree with the matrix-derived verdict;
- finalization retry guidance still allowed `defer`, although only `select` or
  `no_match` can close the active set;
- the answer parser did not accept the observed string-valued `decision` alias.

The repair keeps semantic decisions model-owned while making the transaction
mechanically total:

- allow `action` as a question-critical constraint type;
- normalize omitted candidate rows to `unknown` and record the normalization;
- derive and record the verdict from the normalized support matrix, retaining
  the declared verdict for diagnostics;
- make retry feedback state-aware and require `select` or `no_match` at
  finalization;
- accept a string-valued `decision` as an answer alias only when
  `action=answer` and `answer` is absent.

Focused tests pass 78/78 and the updated full suite passes 476/476. The invalid
39-case A4 root is structural-debug evidence only and must never be judged.

## Next actions

1. Commit and push the lifecycle repair.
2. Pull and run the full suite on the KML development machine.
3. Run a failed-category matched A3-record/A4-replay structural canary.
4. If all structural gates pass, rerun both frozen39 arms at the same repaired
   commit into new roots and judge only the new matched records.
