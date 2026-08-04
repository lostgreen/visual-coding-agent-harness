# MGER Phase 4 Completion Report (2026-08-04)

## Decision

Phase 4 is **NO-GO** at the first paired gate. The required two independent MM-Lifelong roots were completed with the GPT-5.5 judge, but neither root passed. Per the Phase 4 plan, strict grounding was not run because the shadow first gate failed.

## Implemented Scope

- B0: separate semantic prediction from grounding enforcement with `shadow` and `strict` control modes.
- B1: compact `R/O/M/E/S` handles with runtime canonical resolution.
- B2: one-time evidence planning and runtime-derived obligation progression.
- B3: final answers can cite observation items directly.
- B4: agent-facing `refine_item` with child refinement lineage.
- B5: typed `locator`, `temporal`, and `semantic` dependencies without locator claim-lineage transactions.
- B6: slim runtime prompt, optional claims, no model-authored obligation or cue state mutations.
- B7: isolated 0038 Caption locator ablation.
- B8: paired 10-case evaluation with two independent roots.

Key implementation commits:

- `6bea6ad`: separate shadow prediction from grounding.
- `8c4ad15`: add paired Phase 4 gates.
- `24a2347`: derive evidence state at runtime.
- `005da75`: add the offline locator ablation.
- `456663c`: resolve occurrence handles before task validation.
- `70effa7`: normalize external temporal plan enums.
- `57b8d9c`: separate search, window, and refinement task schemas.
- `2f44b34`: enforce the slim runtime action surface.

## Verification

- Local tests: 328 passed.
- Remote KML tests at `2f44b34`: 328 passed.
- Ruff: passed.
- GitHub branch: `codex/mger-phase4-runtime-state`.
- Runtime models: `pa/gmn-2.5-pr` for Reasoner and Investigator.
- Judge model: `mmu-0-2-openai-swedencentral-gpt-5.5`.
- Judge parse completeness: 20/20.

## B0 Diagnostic

The B0 shadow root restored answer presence but did not solve protocol overload:

| Metric | B0 shadow |
| --- | ---: |
| Answer rate | 0.90 |
| Mean frames | 59.9 |
| Decision repairs | 19 |
| Task resolution errors | 9 |
| Silent drops | 0 |
| Grounding-valid answers | 0 |
| GPT-5.5 exact score | 1/10 |

This justified proceeding from LLM-authored state to the B1-B6 runtime-derived design.

## Offline Locator Track

Case 0038 was evaluated independently from the controller changes.

| Variant | CandidateClueRecall | Candidate count |
| --- | ---: | ---: |
| current hybrid | 0.0 | 12 |
| target query union | 0.0 | 12 |
| neighbor expansion | 0.0 | 34 |
| higher candidate K | 0.0 | 86 |
| source-title metadata | 0.0 | 86 |
| multilingual embedding | 0.0 | 86 |

No variant improved recall, so no locator change was promoted into the agent runtime.

## Final Paired Evaluation

| Metric | Root 1 | Root 2 | First-gate target |
| --- | ---: | ---: | ---: |
| Answer rate | 0.90 | 0.80 | >= 0.90 |
| Mean frames | 19.9 | 36.4 | <= 68.875 |
| Decision repairs | 5 | 6 | <= 5 |
| Task resolution errors | 3 | 4 | <= 2 |
| Silent drops | 0 | 0 | 0 |
| State mutation ops | 0 | 0 | diagnostic |
| Grounded Correct | 0 | 0 | >= 1 |
| Wrong-but-Verified | 0 | 0 | <= 4 |
| Prompt schema token cost | 22,914 | 24,408 | diagnostic |

GPT-5.5 results:

- Root 1: 1 exact correct, 1 partial (`0117` scored 0.5), 8 exact wrong/missing.
- Root 2: 0 exact correct, 1 partial (`0031` scored 0.5), 9 exact wrong/missing.
- `0117` regressed in both roots under the exact no-regression check.
- All answers were grounding-invalid, so no exact answer qualified as Grounded Correct.

Root 1 failed:

- `task_resolution_errors_at_most_2`
- `case_0117_no_regression`
- `grounded_correct_at_least_1`

Root 2 failed:

- `answer_rate_at_least_0_9`
- `decision_repairs_at_most_5`
- `task_resolution_errors_at_most_2`
- `case_0117_no_regression`
- `grounded_correct_at_least_1`

## Engineering Conclusion

Runtime ownership successfully removed model-authored state mutations and reduced frames, but it did not produce a stable evidence-support loop. Root-to-root answer variation, remaining task errors, and zero grounded-correct answers show that the Working Document plus multi-action controller remains the dominant bottleneck.

The next implementation should follow the plan's NO-GO branch:

1. Reasoner: `plan -> tool calls -> final answer`.
2. Evidence state: runtime-only sidecar.
3. Working Document: optional reasoning memory, not the primary control protocol.
4. Do not expand to strict grounding, 20 cases, or full MM-Lifelong until a new shadow design passes the paired first gate.

## Remote Artifacts

- Paired gate: `/home/xuboshen/zgw/mger_runs/phase4-gate-runtime-shadow-2f44b34-20260804.json`
- Locator ablation: `/home/xuboshen/zgw/mger_runs/phase4-locator-0038-005da75-r2-20260804.json`
- Root 1: `/home/xuboshen/zgw/mger_runs/cases10-phase4-runtime-shadow-2f44b34-r1-20260804`
- Root 2: `/home/xuboshen/zgw/mger_runs/cases10-phase4-runtime-shadow-2f44b34-r2-20260804`
- Root 1 runtime log: `/home/xuboshen/zgw/mger_runs/logs/phase4-runtime-shadow-2f44b34-r1-20260804.log`
- Root 2 runtime log: `/home/xuboshen/zgw/mger_runs/logs/phase4-runtime-shadow-2f44b34-r2-20260804.log`
- Root 1 judge log: `/home/xuboshen/zgw/mger_runs/logs/phase4-runtime-shadow-2f44b34-r1-gpt55-20260804.log`
- Root 2 judge logs: `/home/xuboshen/zgw/mger_runs/logs/phase4-runtime-shadow-2f44b34-r2-gpt55-20260804.log` and `/home/xuboshen/zgw/mger_runs/logs/phase4-runtime-shadow-2f44b34-r2-gpt55-remainder-20260804.log`
