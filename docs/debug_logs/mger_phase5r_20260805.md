# MGER Phase5R Reproducibility Audit State (2026-08-05)

## Goal

Execute the frozen Phase5R audit on the fixed MMLifelong 10-case cohort. Gate R1
must prove exact per-task frame-count and timestamp parity before any live
historical/current controller runs are allowed.

## Frozen Inputs

- Branch: `codex/mger-phase5r-repro-audit`
- R1 implementation commit: `8b757e4`
- Current reporting/runtime commit: `e33a74c`
- Historical source revision: `74f012d`
- Case root: `/home/xuboshen/zgw/mger_runs/cases10-input-74f012d-20260804`
- Recorded fixture: `tests/fixtures/mger_phase3_cases10_74f012d`
- Caption config digest: `fb25cc6e8c5c4f855d86b724882b14502d3ff17d0670e1c0dc61e52a3bd08b96`
- Embedding revision: `sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- Historical budgets: rounds `4`, investigations `12`, tasks per round `4`
- Caption mode/query strategy: `hybrid` / `adaptive`

## Current Evidence

- Latest local full suite after the replay budget fix: `347 passed`.
- Remote clean-worktree tests at `f3ed32a`: `343 passed`.
- The replay driver previously raised `max_rounds` to the number of recorded
  decisions. Case 0165 exposed the error by dispatching historical round-5 task
  `r1_t5`. Commit `8b757e4` preserves the configured semantic budget and lets
  the normal finalization path consume post-budget decisions without dispatch.
- Formal focus-3 R1: PASS, 119/119 frames at
  `/home/xuboshen/zgw/mger_runs/phase5r-gate-r1-focus3-8b757e4-r1-20260805.json`.
- Formal all-10 R1: PASS, 10/10 cases and 551/551 frames at
  `/home/xuboshen/zgw/mger_runs/phase5r-gate-r1-all10-8b757e4-r1-20260805.json`.
- The real historical worktree is clean at full revision
  `74f012dfc1f3a3e29541ab6d21cb261c937c702a`.
- Interleaved live Track B/C has started with historical root 1 at
  `/home/xuboshen/zgw/mger_runs/cases10-phase5r-old74f012d-live-r1-20260805`.

## Stale Or Excluded Runs

- `cases3-phase5r-replay-f3ed32a-r1-20260805`: subprocess import failure.
- `cases3-phase5r-replay-0aeb8d7-r2-20260805`: concurrent embedding cache failure.
- `cases3-phase5r-replay-0aeb8d7-r3-20260805`: wrong `6/20/4` budgets.
- `cases10-phase5r-replay-0aeb8d7-r1-20260805`: wrong `6/20/4` budgets.
- `cases3-phase5r-replay-0aeb8d7-r4-cfg4124-20260805`: replay driver silently
  raised the configured round budget.
- `cases10-phase5r-replay-0aeb8d7-r2-cfg4124-20260805`: replay driver silently
  raised the configured round budget.

## Next Actions

1. Complete the interleaved historical/current three-root live sequence.
2. Build `FrozenBehaviorReferenceV2` and conclude descriptive Gate R2.
3. Complete the audit report, rerun the full
   test suite, push the exact commit, and package the clean ZIP.
