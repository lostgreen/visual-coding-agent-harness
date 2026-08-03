# Long-video plan absorption - 2026-08-03

## Goal

Adopt the parts of `VCAH_LongVideo_Framework_Improvement_Plan_20260802.md`
that improve observability and promotion safety without reintroducing the
previously measured ReMA and Evidence Contract regressions.

## Current evidence

- The prior 20-case comparison remains authoritative for retrieval policy:
  P0 scored 37.5%, global ReMA 15.0%, and adaptive ReMA 27.5%. Global ReMA
  also increased mean inspected frames from 67.25 to 94.5.
- Offline replay of the existing paired cases 0031, 0038, 0097, and 0117 found
  that adaptive mode improved mean accuracy from 0.25 to 0.50 and clue-frame
  coverage from 0.5625 to 0.6875.
- The same replay reduced reference validity from 0.75 to 0.50 and increased
  mean inspected frames from 67.5 to 118.0 (1.748x). It therefore fails the
  promotion gate even before the required second independent paired repeat.

## Adopted

- Deterministic offline diagnostics for clue-frame coverage, retrieval
  duplication, sampling fidelity, and claim-anchor consistency.
- A paired net-gain gate requiring accuracy improvement, no reference or clue
  coverage regression, no anchor-consistency regression, and at most 1.25x
  visual-frame cost on at least two distinct paired run roots.
- Sampling manifests now expose requested fps and measured fidelity.
- Mechanical status reports low-fidelity attempts and gives a scoped refinement
  hint. It does not automatically spend more investigation budget.

## Deferred or rejected

- Do not make ReMA retrieval the adaptive default; existing paired evidence is
  a regression.
- Do not switch embedding models without a frozen-index retrieval ablation.
- Do not add automatic observation arbitration, occurrence enumeration, or a
  hard temporal requirement executor until each has a focused paired test.
- Do not restore the detailed Evidence Contract runtime that was previously
  reverted. Keep evidence records compact and derive diagnostics offline.

## Artifacts and next checks

- Source case bundle:
  `/Users/lostgreen/Downloads/vcah_cases4_evidence_ablation_405cb66_20260802.zip`
- Local replay report:
  `/tmp/vcah-plan-eval.ArQcXn/net_gain_report.json`

## Verification

- Focused diagnostics and agent tests: 54 passed.
- Full local suite: 244 passed.
- The four-case gate intentionally exited non-zero with
  `reference_non_regression` and `frame_cost_within_limit`; this is the expected
  rejection result, not an execution failure.
- Full KML development-machine suite at code commit `dbaa9fb`: 244 passed,
  88 warnings, exit code 0. The detached checkout remained clean after testing.
