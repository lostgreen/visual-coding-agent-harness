# Phase 3 B2 - Evidence Obligation Ledger

- Commit: `c1f39f8`
- Baseline root: `/home/xuboshen/zgw/mger_runs/cases10-adaptive-74f012d-r1-20260804`
- Candidate root: `/home/xuboshen/zgw/mger_runs/cases3-b2-c1f39f8-r1-20260804`
- Reasoner: `pa/gmn-2.5-pr`
- Investigator: `pa/gmn-2.5-pr`
- Development judge: `pa/gmn-2.5-pr`
- Caption index digest: `fb25cc6e8c5c4f855d86b724882b14502d3ff17d0670e1c0dc61e52a3bd08b96`
- Cases: `0097`, `0165`, `0190`

## Results

- Runtime success: `3/3`
- Development Acc: `0/3`
- Reference valid: `0/3`
- Grounded correct: `0/3`
- Wrong but verified: `0/3`
- Correct but ungrounded: `0/3`
- Answer rate: `1/3`
- Mean visual frames: `85.33`
- Silent drops: `0`
- Decision repairs: `6`
- Answer-bearing obligations: `7`
- Satisfied obligations: `1`
- Open obligations at answer: `6`
- Unresolved obligations at answer: `0`
- Mean obligation coverage: `0.0833`
- Occurrence binding: not introduced in B2
- Cue verification: not introduced in B2
- Provenance binding: not introduced in B2

## Acceptance

- `0097` exposes four answer-bearing obligations; only one is satisfied and
  the other three remain mechanically visible and block grounded finalization.
- `0165` and `0190` also preserve open obligations instead of promoting search
  or candidate evidence into verified support.
- Every acquisition request terminates as executed or an explicit resolution
  error; no silent task drop was observed.

## Known Regressions And Decision

- `0165` produced only one obligation, so generic multi-hop decomposition must
  be reinforced in the Reasoner protocol and checked again in the final batch.
- Focused answer rate and frame cost regress because the new closure gate now
  refuses under-covered answers. B6/B7 must recover answer production without
  weakening obligation coverage.
- Decision: **keep**, conditional on the Phase 3 full-run answer-rate and frame
  gates.
