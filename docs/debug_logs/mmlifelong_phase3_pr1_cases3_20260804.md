# MM-Lifelong Phase 3 PR1 Control Retry Check - 2026-08-04

## Goal

Implement the Phase 3 PR1 scope: freeze the prior ten-case diagnosis as a
reproducible fixture, separate semantic-round and control-retry accounting,
and prove that every requested acquisition is either executed or recorded as
an explicit resolution error.

## Code And Checks

- Fixture commit: `a40bf34`.
- Runtime commit: `ca126c4`.
- Local full suite: `270 passed`.
- KML focused suite: `6 passed`.
- The frozen ten-case fixture reproduces the prior `74f012d` aggregate:
  accuracy `4/10`, answer rate `9/10`, and reference-valid rate `5/10`.
- Phase 3 B2 and later work remains out of scope for this PR1 check.

## Focused Runtime Evidence

All three MM-Lifelong runs completed successfully with the prior hybrid,
adaptive, four-round runtime configuration and a control-retry budget of two.

| Case | Answer | Ref valid | Control retries | Requested | Executed | Explicit errors | Silent drops | Frames | Judge score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0097` | no | no | 1 | 4 | 3 | 1 | 0 | 96 | 0 |
| `0119` | yes | no | 2 | 6 | 4 | 2 | 0 | 0 | 0 |
| `0190` | yes | no | 0 | 4 | 4 | 0 | 0 | 96 | 0 |

- Ledger total: 14 requested, 11 executed, 3 explicit errors, 0 silent drops.
- Case `0119` recovered from both a decision preflight error and a rejected
  workspace transaction. Each retry retained the same semantic round and used
  the next control attempt.
- Case `0097` no longer loses a requested task silently, but it still finishes
  without an answer. This is evidence for later obligation/closure work, not a
  B1 ledger failure.
- The separated evaluator parsed all three results. Aggregate accuracy is
  `0/3`, answer rate is `2/3`, and reference-valid rate is `0/3`.
- The judge was `pa/gmn-2.5-pr`; both
  `official_judge_model_match` and `official_judge_config_match` are false.
  These are development diagnostics, not official benchmark scores.

## Current And Stale Evidence

- Current evaluator attempt: `r2`, return code 0.
- Stale evaluator attempt: `r1` expanded the remote loop variable locally and
  exited before writing an evaluation. It is infrastructure failure evidence
  only and is excluded from the results above.

## Artifacts

- Remote run:
  `/home/xuboshen/zgw/mger_runs/cases3-control-ca126c4-r1-20260804`.
- Remote compact ZIP:
  `/home/xuboshen/zgw/mger_runs/VCAH_MGER_cases3_control_ca126c4_r1_light_20260804.zip`.
- Local compact ZIP:
  `/Users/lostgreen/Downloads/VCAH_MGER_cases3_control_ca126c4_r1_light_20260804.zip`.
- ZIP size: 393,920 bytes.
- ZIP SHA-256:
  `3a6f3e4b6b1723f25eda598493e286f48d7533e53ea255da7263a5397ce36b1d`.
- Remote and local ZIP integrity checks passed. The compact archive contains
  the three case directories and no image or video entries.
