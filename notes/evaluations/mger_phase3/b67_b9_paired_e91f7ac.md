# MGER Phase 3 B6/B7/B9: Paired 10-Case Evaluation

Date: 2026-08-04

## Reproducibility

- Commit: `e91f7ac`
- Baseline root: `/home/xuboshen/zgw/mger_runs/cases10-adaptive-74f012d-r1-20260804`
- Candidate root 1: `/home/xuboshen/zgw/mger_runs/cases10-phase3-e91f7ac-r1-20260804`
- Candidate root 2: `/home/xuboshen/zgw/mger_runs/cases10-phase3-e91f7ac-r2-20260804`
- Runtime models: reasoner `pa/gmn-2.5-pr`; investigator `pa/gmn-2.5-pr`
- Judge: `mmu-0-2-openai-swedencentral-gpt-5.5`
- Judge prompt SHA-256: `239156295b8e3331e1c73e95c3bf88ca21cc94e99c1a85e5b5cef65a5ac9729b`
- Implementation digest: `1f00ec2873cb0138f131abb2d085825d8550099afb99b5e33fe5aef9a4ae5de3`
- Runtime/evaluator comparison digest: `cc92820e218e4973d3e0f8b41bea177fe671f36ddf33e3e6c0464ea5c1986764`
- Caption config digest: `fb25cc6e8c5c4f855d86b724882b14502d3ff17d0670e1c0dc61e52a3bd08b96`
- Embedding adapter digest: `c3a7d38ef06df9857325b2434c093891ff9eb42af6d3ce1593d5a1927ec842bd`
- Root 1 reasoner prompt-set digest: `3a462ba10bd481bb2e84ad49d262f3970c780a2fec576b7034cca5dbdeff8438`
- Root 2 reasoner prompt-set digest: `14a5975337cf62470c337ed7178c12f8979aa7eca4657e108052b1211c44e114`
- Gate report: `/home/xuboshen/zgw/mger_runs/phase3-e91f7ac-reports-20260804/phase3_gate.json`

Both roots used the same ten frozen MM-Lifelong cases, configuration, input
digest, model IDs, and code. They were generated independently. The judge ran
after prediction generation and was not available to the runtime.

## Aggregate Results

| Metric | Frozen baseline, GPT-5.5 | Root 1 | Root 2 | Target |
| --- | ---: | ---: | ---: | ---: |
| GPT-5.5 accuracy | 0.20 | 0.10 | 0.00 | diagnostic |
| Reference Valid | 5/10 | 0/10 | 1/10 | non-regression |
| Grounded Correct | 1/10 | 0/10 | 0/10 | >2/10 |
| Wrong-but-Verified | 4/10 | 0/10 | 1/10 | <=1/10 |
| Correct-but-Ungrounded | 1/10 | 1/10 | 0/10 | diagnostic |
| Missing Answer | 1/10 | 5/10 | 4/10 | <=1/10 |
| Answer Rate | 0.90 | 0.50 | 0.60 | >=0.90 |
| Mean Frames | 55.1 | 49.0 | 61.3 | <=68.875 |
| Silent Drops | - | 0 | 0 | 0 |
| Decision Repairs | - | 25 | 30 | diagnostic |
| Mean Obligation Coverage | - | 0.000 | 0.125 | diagnostic |
| Mean Occurrence Binding | - | 0.800 | 0.800 | diagnostic |
| Mean Temporal Scope Resolution | - | 0.100 | 0.100 | diagnostic |
| Mean Cue Verification | - | 0.000 | 0.000 | diagnostic |
| Mean Claim-Item Binding | - | 0.600 | 0.500 | diagnostic |

All 20 GPT-5.5 responses parsed on the first evaluator attempt. The requested
judge is not the evaluator's frozen `gpt-5` identifier, so
`official_judge_model_match=false` is expected and recorded.

## Gate Result

The evaluation-only `runs/mger_phase3_gate.py` result is `NO-GO`.

Root 1 failures:

- `grounded_correct_above_2`
- `answer_rate_at_least_0_9`
- `case_0117_no_regression`

Root 2 has the same three failures. Case count, judge parsing, requested judge
model, frame cost, silent-drop, and Wrong-but-Verified checks pass on both
roots. Wrong-but-Verified falls mainly because strict closure invalidates
support, not because Grounded Correctness improves.

## Focused Case Acceptance

| Case | Root 1 | Root 2 | Assessment |
| --- | --- | --- | --- |
| 0031 | 1 visual interpretation; answer missing | 1 interpretation; wrong and ungrounded | Fails the same-attempt, two-interpretation target |
| 0038 | candidate/occurrence clue recall 0 | candidate/occurrence clue recall 0 | Retrieval target not improved |
| 0072 | answer support rejected; item-bound claims 2/2 | answer missing; item-bound claim 1/1 | Fake provenance is blocked, but no valid answer is recovered |
| 0097 | 4 obligations visible, all open | 4 visible; 1 satisfied and 3 unresolved; wrong but verified | Protocol is visible, but coverage does not yield correctness |
| 0108 | temporal scope resolved; answer missing | temporal scope unresolved; answer missing | The bound-occurrence resolver works in one root but is inconsistent |
| 0117 | GPT-5.5 correct, reference invalid, 8 frames | answer missing, reference invalid, 32 frames | Regression control fails |
| 0119 | answer present | answer present | Parser/control-plane missing-answer regression is repaired |
| 0146 | 192 frames, no verified cue | 192 frames, no verified cue | Parent/cue rule blocks support, but refinement protocol is not completed |
| 0165 | answer missing | answer missing | Search-miss safety exists in code, but multi-hop completion fails |
| 0190 | 5 obligations, no item-bound observation claim | 2 obligations, no item-bound observation claim | UI/closure requirements remain incomplete |

## What Worked

- Control/schema repair is tracked separately from semantic rounds in tests.
- Requested acquisitions have terminal outcomes; silent drops are zero.
- Explicit caption queries remain isolated from the whole question.
- Obligation, temporal-scope, occurrence-binding, cue, interpretation-item,
  typed-perception, and multidimensional closure schemas are present.
- Search misses cannot satisfy observation obligations.
- Visual occurrence bindings now enter temporal-scope resolution mechanically.
- Strict closure blocks unsupported candidate/locator evidence instead of
  reporting it as verified.

## Known Regressions

- Strict closure and incomplete model-authored state transitions reduce answer
  rate from 0.90 to 0.50/0.60.
- Grounded Correctness drops from 1/10 under the calibrated baseline to 0/10.
- Cue generation occurs, but cue verification and child refinement are never
  completed in either root.
- 0038 candidate recall remains zero.
- 0117 does not retain both correctness and valid grounding.

## Decision

Conditional keep as an auditable evidence-state diagnostic framework. Do not
advance to full MM-Lifelong or claim a Phase 3 accuracy gain. The result matches
the plan's NO-GO branch: the structural evidence lifecycle is implemented, but
the runtime model does not reliably complete obligation, cue, provenance, and
closure state transitions. The next phase should not add more wide scans,
frames, global retrieval, or agent complexity without first solving that state
completion problem.
