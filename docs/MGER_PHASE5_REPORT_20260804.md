# MGER Phase 5 Measurement-First Report (2026-08-04)

## Decision

Phase 5 is **STOP** at the Frozen Baseline runtime reproduction precheck. The
formal ten-case Frozen run observed all cases and produced all answers, but its
mean inspected-frame cost was 85.2 rather than the 55.1 reference. This is a
54.6% increase and exceeds the deliberately liberal 25% reproduction tolerance
(maximum 68.875).

The official Frozen-versus-Blind score gate was therefore not entered. Per the
Phase 5 decision order, Caption-only execution, Minimal Controller, paired
Gate-1, Binder, 40-case confirmation, Arm C, and substrate promotion were not
run or claimed.

## Implemented Scope

- Orthogonal `controller_mode`, `controller_evidence_visibility`, and
  `measurement_control` configuration.
- Question-only Blind Prior control with all tools disabled.
- Caption-only policy wiring with visual inspection forbidden.
- Isolated Frozen Baseline compatibility path using the pre-MGER Working View,
  status surface, preflight behavior, and historical one-shot JSON repair.
- Tier-0/Tier-1 metrics including observed-case rate, conditional frame cost,
  silent acquisition drops, and malformed-decision rate.
- Official-evaluator consistency checks and a half-credit-aware Gate-0 scorer.
- A runtime-only Frozen reproduction precheck that does not pretend an
  unevaluated judge configuration passed.

Key commits:

- `0bad1b1`: add Phase 5 controls, metrics, and Gate-0 tooling.
- `fed807c`: isolate the Frozen controller from Phase 3/4 control-plane state.
- `baa3c3a`: add the machine-readable Frozen reproduction precheck.

## Verification

- Final local tests at `baa3c3a`: 339 passed.
- Final remote KML tests on the `baa3c3a` content: 339 passed; the two changed
  files were synced through the file bridge after KML-to-GitHub HTTPS timed out.
- Focused reproduction-gate tests at `baa3c3a`: 4 passed locally and remotely.
- Ruff on the changed Python files and `git diff --check`: passed.
- Runtime Reasoner and Investigator: `pa/gmn-2.5-pr`.
- Frozen caption-index digest:
  `fb25cc6e8c5c4f855d86b724882b14502d3ff17d0670e1c0dc61e52a3bd08b96`.

## Formal Runtime Screen

| Metric | Blind Prior | Frozen Baseline | Reproduction target |
| --- | ---: | ---: | ---: |
| Completed cases | 10/10 | 10/10 | 10/10 |
| Answer rate | 1.00 | 1.00 | diagnostic |
| ObservedCaseRate | 0.00 | 1.00 | Frozen = 1.00 |
| Mean frames | 0.0 | 85.2 | Frozen <= 68.875 |
| Conditional mean frames | n/a | 85.2 | diagnostic |
| Caption searches | 0 | 11 | Blind = 0 |
| Requested acquisitions | 0 | 32 | Blind = 0 |
| Silent drops | 0 | 0 | Frozen = 0 |
| Malformed decision rate | n/a | 0.000 | diagnostic |

The Blind root satisfies its runtime isolation checks. It was not judge-scored
because Frozen failed the prerequisite reproduction check.

## Frozen Frame Drift

| Case | Historical frames | Formal Frozen frames | Delta |
| --- | ---: | ---: | ---: |
| 0031 | 72 | 96 | +24 |
| 0038 | 40 | 28 | -12 |
| 0072 | 19 | 120 | +101 |
| 0097 | 84 | 192 | +108 |
| 0108 | 32 | 18 | -14 |
| 0117 | 32 | 40 | +8 |
| 0119 | 16 | 106 | +90 |
| 0146 | 104 | 96 | -8 |
| 0165 | 56 | 60 | +4 |
| 0190 | 96 | 96 | 0 |

The aggregate drift is concentrated in cases 0072, 0097, and 0119, rather than
being a uniform batching offset. The correct next investigation is environment,
prompt, or model nondeterminism around acquisition choices, not implementation
of a new controller.

## Official Judge Status

The plan requires the official MM-Lifelong evaluator with `gpt-5,
temperature=0`. No usable GPT-5 deployment configuration was available on KML:
connectivity probes using both the literal model name and the corresponding
internal deployment-name pattern returned HTTP 404. The available GPT-5.5
configuration was not relabeled or substituted.

This is a secondary readiness blocker, not the basis of the current STOP. The
runtime frame-reproduction check already failed before judge scoring. Therefore
there are no official mean-score, Frozen-minus-Blind delta, or significance
claims in this report.

## Excluded Diagnostic Root

`cases10-phase5-frozen-0bad1b1-r1-20260804` is stale and excluded. It inherited
Phase 3/4 Working View fields, the newer preflight, and a disabled historical
JSON repair. Its 0.70 ObservedCaseRate and 60.7 mean frames are diagnostic only.
The formal result comes exclusively from the corrected `fed807c` root.

## Engineering Conclusion

The Phase 5 measurement-first ordering worked as intended: it exposed an
unstable Frozen reference before new controller work could confound the result.
No downstream method claim is warranted on this screen. Resume only after:

1. Frozen reproduces 100% observation, zero silent drops, and frame cost within
   the declared tolerance on a fresh root.
2. An actual official `gpt-5, temperature=0` evaluator configuration is
   available and passes the recorded consistency check.
3. Gate-0 then shows `mean_score(Frozen) - mean_score(Blind) >= 0.15`.

## Remote Artifacts

- Frozen reproduction precheck:
  `/home/xuboshen/zgw/mger_runs/phase5-frozen-reproduction-baa3c3a-20260804.json`
- Blind root:
  `/home/xuboshen/zgw/mger_runs/cases10-phase5-blind-fed807c-r1-20260804`
- Formal Frozen root:
  `/home/xuboshen/zgw/mger_runs/cases10-phase5-frozen-fed807c-r1-20260804`
- Historical frame reference root:
  `/home/xuboshen/zgw/mger_runs/cases10-adaptive-74f012d-r1-20260804`
- Formal Frozen runtime log:
  `/home/xuboshen/zgw/mger_runs/logs/phase5-frozen-fed807c-r1-20260804.log`
- Final remote test log:
  `/home/xuboshen/zgw/mger_runs/logs/phase5-tests-baa3c3a-r2-20260804.log`
- GPT-5 connectivity diagnostics:
  `/home/xuboshen/zgw/mger_runs/logs/phase5-gpt5-connectivity-20260804.log`
  and
  `/home/xuboshen/zgw/mger_runs/logs/phase5-gpt5-deployment-connectivity-20260804.log`
