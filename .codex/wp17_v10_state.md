# WP17 v10 State

## Goal

Separate raw SER quality from committed-memory reliability, close the slot
transaction repair loop, and evaluate a declared v10 reliability-policy variant
without touching Day-test140 or Week outcomes.

## Current Evidence

- Current code: `de190cc`; construction artifacts originate from `bdd2b7e`.
- Completed merged run: 363/363, 53 E1C2 transaction abstentions (43.8%).
- Frozen development decision remains `NO_GO_UNDER_DEVELOPMENT_ENDPOINTS`.
- The 11 development outcomes are burned; all v10 rescoring on them is exploratory.
- Representative failures combine valid SER content with illegal lifecycle operations.
- Zero-model forensics found raw lexical ARC E1C0/E1C1/E1C2 = 3/4/2 of 11;
  E1C2 SERs were shorter by 26.4 normalized words and 1.12 event items per
  segment versus E1C1.
- `occurrence_counter` had only one write and one update in 121 E1C2 segments.
- The 0108 canonical labels were absent before the anchor; this is not evidence
  of failed historical inheritance.
- Pre-fix exhaustive reachability failed 6 legality checks and all 14 illegal
  repair-recoverability checks. The v10 implementation passes all 30 cells.

## Decisions

- Preserve v9 replay behavior; expose v10 lifecycle behavior as an explicit policy.
- Treat idempotent terminal operations and closed-slot sweeping as method-policy
  changes, not pure bug fixes.
- Report raw SER and committed-memory scopes separately.
- Require exhaustive state-operation repair recoverability before any model canary.
- Preserve v9 capsule/snapshot/digest bytes by emitting policy fields only for v10.
- Canary requires zero E1C2 transaction abstentions; full v10 requires <5%.
- Keep frozen lexical matching secondary and freeze an arm-blind semantic judge
  before any holdout evaluation.

## Next Actions

1. Commit the v10 reliability-policy implementation and run the remote full suite.
2. Re-audit a historical v9 artifact with zero model calls to prove compatibility.
3. Freeze a v10 protocol from the untouched construction manifest.
4. Run a fresh five-segment three-arm canary; require zero E1C2 abstentions.
5. Only after the canary gate passes, decide whether to launch the full 121x3 run.
