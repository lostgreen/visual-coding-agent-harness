# Evidence Contract Ablation - 2026-08-02

## Goal

Test whether a structured Evidence Contract can improve the frozen MM-Lifelong
Day sample without reviving the earlier highly detailed testimony machinery.
The experiment was feature-gated and defaulted to `off`.

## Frozen setup

- Branch: `codex/p1-rema-retrieval`.
- Caption digest: `fb25cc6e8c5c4f855d86b724882b14502d3ff17d0670e1c0dc61e52a3bd08b96`.
- Hybrid retrieval and embedding revision were held fixed across each pair.
- Model, investigation limits, answer policy, and judge were held fixed.
- Results are single paired runs and are directional, not significance claims.

## Results

| Pair | Policy | Score | Answers | Reference valid | Mean visual frames |
| --- | --- | ---: | ---: | ---: | ---: |
| Canary 8, initial | off | 18.75% | 7/8 | 6/8 | 65.4 |
| Canary 8, initial | adaptive contract | 25.00% | 7/8 | 4/8 | 129.5 |
| Canary 6, narrowed selector | off | 33.33% | 6/6 | 4/6 | 61.8 |
| Canary 6, narrowed selector | adaptive contract | 50.00% | 6/6 | 2/6 | 131.7 |
| Language recall 2 | baseline | 50.00% | 1/2 | 0/2 | 48.0 |
| Language recall 2 | lightweight split prompt | 0.00% | 2/2 | 1/2 | 48.0 |

The contract's apparent score gains did not meet a net-gain gate: reference
validity fell in both paired runs and visual inspection cost roughly doubled.
The lightweight prompt did not reproduce the one promising mechanism.

## Failure fingerprints

- `0038`: all declared conditions were covered, but the agent localized the
  wrong occurrence and returned the wrong item. Contract coverage did not
  validate candidate identity or occurrence.
- `0097`: the contract expanded work to 298-310 frames without improving the
  answer. Locator and visual identity errors remained downstream of planning.
- `0117`: the answer stayed correct while adaptive contract mode lost a valid
  reference, showing that the stricter support path can regress clean cases.
- `0182`: an initial score gain did not repeat; the narrowed run was correct in
  both modes while adaptive mode doubled frames and lost reference validity.
- `0031`: one contract run correctly separated boss defeat from subtitle
  transcription, but an independent prompt-only replay reversed the result.
  The reasoner still combined both targets into one task, so this was not a
  reliable prompt-level mechanism.
- `0027`: Caption search missed the decisive continuation. The lightweight
  prompt increased searches from two to four without producing a visual
  confirmation.

## Decision

Do not promote the Evidence Contract or the anchored-language prompt. Revert
their runtime paths and retain only the active test that forbids benchmark case
IDs and case-specific aliases in runtime code.

Any future replacement should first demonstrate, on repeated paired runs:

1. correct occurrence selection on hard negatives such as `0038`;
2. independent anchor-to-target identity verification on `0097`;
3. no decrease in reference-valid rate; and
4. no more than 1.25x mean visual-frame cost unless accuracy gain is stable.

## Artifacts

- Initial pair: `/home/xuboshen/zgw/evidence_contract_runs/canary8-{off,adaptive}-8ab8546-r1`.
- Narrowed pair: `/home/xuboshen/zgw/evidence_contract_runs/canary6-{off,adaptive}-4b7e740-r1`.
- Lightweight pair: `/home/xuboshen/zgw/anchored_text_runs/cases2-{baseline-4b7e740,light-b30de53}-r2`.
- The corresponding `r1` lightweight directories are invalid infrastructure
  runs caused by a missing `PYTHONPATH` and must not be used for metrics.
