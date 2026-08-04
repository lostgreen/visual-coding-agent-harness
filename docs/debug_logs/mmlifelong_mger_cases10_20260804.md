# MM-Lifelong MGER 10-Case Retest - 2026-08-04

## Goal

Extend the latest four-case MM-Lifelong MGER run to ten cases while retaining
the original cases, running the separated evaluator, and preserving a compact
downloadable artifact.

## Fixed Setup

- Runtime/evaluator commit: `74f012d`.
- Summary fixes: `204e7a3`, `cad2458`.
- Original cases: `0031`, `0038`, `0097`, `0117`.
- Added cases: `0072`, `0108`, `0119`, `0146`, `0165`, `0190`.
- The ten cases cover one case from each question type in the frozen 20-case
  MM-Lifelong Day/game pool.
- Caption digest: `fb25cc6e8c5c4f855d86b724882b14502d3ff17d0670e1c0dc61e52a3bd08b96`.
- Hybrid index, adaptive query strategy, four rounds, twelve investigations,
  four tasks per round, and `benchmark_best_effort` were retained.
- Reasoner, investigator, and diagnostic judge used `pa/gmn-2.5-pr`.

## Current Evidence

- KML tests: `262 passed` at runtime commit `74f012d`.
- Runtime batch: 10/10 successful.
- Evaluations: 10/10 parsed.
- Accuracy: 4/10 (`0031`, `0117`, `0146`, `0165`).
- Answer rate: 90%.
- Reference-valid rate: 50%.
- Mean inspected frames: 55.1.
- Ref@60 / Ref@300 / Ref@600: 0.1583 / 0.1643 / 0.2700.
- The answer judge used the official prompt and parser but not the exact
  official GPT-5 model. Provenance therefore records
  `official_judge_model_match=false`; this is a comparable development run,
  not an official main-table result.

## Fixes From The Run

- The summary CLI now creates missing output directories.
- Cross-case aggregation removes case-specific `case_id`, `input_digest`, and
  derived effective-strategy fields before computing a comparison group digest.
- The first evaluator loop attempt is invalid infrastructure evidence: a local
  shell expanded the remote loop variable before dispatch. It exited before
  writing any evaluation. The clean `r2` attempt evaluated all ten cases.

## Artifacts

- Remote run:
  `/home/xuboshen/zgw/mger_runs/cases10-adaptive-74f012d-r1-20260804`.
- Remote compact ZIP:
  `/home/xuboshen/zgw/mger_runs/VCAH_MGER_cases10_adaptive_74f012d_r1_light_20260804.zip`.
- Local compact ZIP:
  `/Users/lostgreen/Downloads/VCAH_MGER_cases10_adaptive_74f012d_r1_light_20260804.zip`.
- ZIP size: 1,095,975 bytes.
- ZIP SHA-256:
  `aadfa8d2dfa86242cee37d715ed0e6ba1d06ea2da023d65a1dd5eb0b7c1b639f`.
- Both remote and local ZIP integrity checks passed.
