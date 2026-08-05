# MGER Phase5R Reproducibility Audit State (2026-08-05)

## Goal

Execute the frozen Phase5R audit on the fixed MMLifelong 10-case cohort. Gate R1
must prove exact per-task frame-count and timestamp parity before any live
historical/current controller runs are allowed.

## Frozen Inputs

- Branch: `codex/mger-phase5r-repro-audit`
- Current implementation commit: `0aeb8d7`
- Historical source revision: `74f012d`
- Case root: `/home/xuboshen/zgw/mger_runs/cases10-input-74f012d-20260804`
- Recorded fixture: `tests/fixtures/mger_phase3_cases10_74f012d`
- Caption config digest: `fb25cc6e8c5c4f855d86b724882b14502d3ff17d0670e1c0dc61e52a3bd08b96`
- Embedding revision: `sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- Historical budgets: rounds `4`, investigations `12`, tasks per round `4`
- Caption mode/query strategy: `hybrid` / `adaptive`

## Current Evidence

- Local tests before the subprocess import fix: `343 passed`; focused tests
  after the fix: `7 passed`.
- Remote clean-worktree tests at `f3ed32a`: `343 passed`.
- Focus-3 replay with incorrect `6/20/4` budgets happened to pass exact frame
  parity for 0072, 0097, and 0119, but is not a formal audit artifact.
- The first all-10 replay with incorrect `6/20/4` budgets passed 9/10. Case
  0165 executed historical round-5 task `r1_t5`, adding 96 frames. Historical
  execution stopped after round 4, so this STOP is a setup error, not runtime
  drift, and is excluded from Gate R1.
- Formal focus-3 replay with fixed `4/12/4` budgets is currently running at:
  `/home/xuboshen/zgw/mger_runs/cases3-phase5r-replay-0aeb8d7-r4-cfg4124-20260805`.

## Stale Or Excluded Runs

- `cases3-phase5r-replay-f3ed32a-r1-20260805`: subprocess import failure.
- `cases3-phase5r-replay-0aeb8d7-r2-20260805`: concurrent embedding cache failure.
- `cases3-phase5r-replay-0aeb8d7-r3-20260805`: wrong `6/20/4` budgets.
- `cases10-phase5r-replay-0aeb8d7-r1-20260805`: wrong `6/20/4` budgets.

## Next Actions

1. Gate the corrected focus-3 replay, then run and gate all 10 cases.
2. If and only if R1 passes, run interleaved historical/current live roots.
3. Build `FrozenBehaviorReferenceV2`, complete the audit report, rerun the full
   test suite, push the exact commit, and package the clean ZIP.
