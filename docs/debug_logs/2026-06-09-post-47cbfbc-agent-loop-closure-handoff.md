# 2026-06-09 Post-47cbfbc Agent Loop Closure Handoff

## 2026-06-09 Post-7468bef Update

Current goal: continue from the post-7468bef review plan and close the remaining evidence-route failures for VideoMME `605-1`, `611-2`, and `612-1`.

Current evidence is the KML post-closure run:

`/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_post_closure_7468bef_3demo_20260609_213500_pyenv`

Compact result:

- accuracy `0.3333333333333333`
- final_rate `0.3333333333333333`
- `605-1`: final D, correct, 10 rounds
- `611-2`: `route_repair_exhausted`, no answer, 7 rounds, 12 citations, 0 evidence chains
- `612-1`: `route_repair_exhausted`, no answer, 8 rounds, 17 citations, 0 evidence chains

Current failure fingerprints:

- `611-2`: locator found ordered-list candidates and focused vision args, but route repair still preferred `verify_segment_anchors`; focused window was not a first-class executable recovery, and normalizer could overwrite focused subwindows.
- `612-1`: life-journey options did not initialize a canonical narration registry; narration route still repaired repeated locator calls to visual anchor verification instead of transcript promotion.
- Both failed cases had evidence-like observations but no answer-grade evidence chain.

Current implemented changes after this update:

- `locate_targets_in_segment` now emits top-level `recommended_next_actions[]` with `route_kind=focused_ordered_list_vision`, `candidate_type=ordered_list`, candidate id, target refs or fallback ordered targets, focused `vision_read` args, and `nframes=128`.
- Locator writes `ordered_list_candidate_detected` and `focused_ordered_list_vision_recommended`.
- Route repair precedence now selects ordered-list focused `vision_read` before generic anchor verification, and selects narration transcript promotion before visual verifier.
- Parse-error recovery executes a unique safe pending recommended action once instead of falling back to generic inspector.
- Normalizer preserves focused ordered-list subwindows and ask_for text instead of expanding them back to the whole segment.
- Prompt/ledger/playbooks now render route-aware guidance for focused ordered-list vision and narration transcript promotion.
- Life-journey option parsing now distinguishes `humble background`, `entered upper class`, `seclusion/farmhouse`, and `born in upper class`.
- 612-style registry initializes narrated `T1..T4`, aliases, fixed before-relations (`R1=T1->T2`, `R2=T2->T3`, etc.), and emits `narration_registry_initialized`.

Current local verification:

- `PYTHONPATH=src:. pytest -q` -> `467 passed in 1.80s`
- `git diff --check` -> passed

Current files changed:

- `src/visual_coding_agent_harness/agents/iterative_agent.py`
- `src/visual_coding_agent_harness/agents/prompt_stack.py`
- `src/visual_coding_agent_harness/agents/question_policy.py`
- `src/visual_coding_agent_harness/agents/skills/playbooks/narration_timeline_qa.md`
- `src/visual_coding_agent_harness/agents/skills/playbooks/visual_timeline_qa.md`
- `src/visual_coding_agent_harness/tools/navigation.py`
- `src/visual_coding_agent_harness/workspace.py`
- `tests/test_iterative_agent.py`
- `tests/test_question_policy.py`
- `tests/test_video_navigation.py`

Current next actions:

1. KML detached three-case VideoMME run is launched from commit `3de5ae3`.
2. Do not monitor unless asked.
3. If continuing, inspect only compact `summary.json` fields and selected trace-event counts.

Launched KML run:

- run root: `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_post_review_3de5ae3_3demo_20260609_223500_pyenv`
- pid: `52193`
- pid path: `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_post_review_3de5ae3_3demo_20260609_223500_pyenv/job.pid`
- log path: `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_post_review_3de5ae3_3demo_20260609_223500_pyenv/job.log`
- summary path: `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_post_review_3de5ae3_3demo_20260609_223500_pyenv/summary.json`

Older sections below are retained as stale historical context unless explicitly comparing regressions.

## Current Goal

Continue the VideoMME three-case agent-loop framework after base commit `47cbfbc`, using the post-47cbfbc plan and the completed KML three-demo run as current evidence. Push the implementation and start a new KML run with the dedicated Python path.

## Current Evidence

Current KML run root:

`/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_skill_first_47cbfbc_3demo_20260609_203046_pyenv`

Artifacts:

- summary: `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_skill_first_47cbfbc_3demo_20260609_203046_pyenv/summary.json`
- log: `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_skill_first_47cbfbc_3demo_20260609_203046_pyenv/job.log`
- pid: `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_skill_first_47cbfbc_3demo_20260609_203046_pyenv/job.pid`

Current compact result:

- accuracy `0.3333333333333333`
- `605-1`: final D, correct, 20 rounds
- `611-2`: `route_repair_exhausted`, no answer
- `612-1`: `route_repair_exhausted`, no answer

Older failed run is stale except for launch-path diagnosis:

`/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_skill_first_47cbfbc_3demo_20260609_192926`

Dedicated remote Python path:

`/home/xuboshen/Anaconda/envs/visual-agent-harness/bin/python`

## Failure Fingerprints

- `605-1`: correct D was available early, but empty target registry plus exposed `target_refs`/coverage-local `Q<n>` labels caused invalid/no-op loops; finalization waited until round 20.
- `611-2`: quoted option parser created malformed artwork targets; coarse segment reads blocked focused subwindows; ordered ASR candidate was not propagated into focused visual verification; inconsistent `ORDERED_VISIBLE` output could become answer-grade.
- `612-1`: repeated anchor/target-ref protocol loop still terminated before relation-complete closure.

## Implemented Changes

- Dynamic target-ref prompt contract: hide `target_refs` when registry is empty; render authoritative target registry; strip exact paired coverage-local `Q<n>` labels when safe.
- Route-aware option sequence parser: quoted option spans remain intact; 611-style permutation options initialize canonical `T<n>` registry and per-option ordered refs.
- Focused inspection permissions: coarse segment inspection no longer blocks explicit focused `vision_read`/QA/anchor windows.
- Ordered-list propagation: locator now emits focused vision call args and recommends focused visual verification before anchor verification.
- Observation integrity: unsupported or internally inconsistent `ORDERED_VISIBLE` claims are downgraded to weak grounding.
- AnswerAgent/verifier closure: repeated-program and no-growth stalls can finalize from a verifier result only after the normal final gate passes; writes `iterative_finalization_ready`.
- Effective skill sync: `agent_v2` enables hard skill runtime by default; prompts show locked effective skill; legacy `timeline_ordering@v1` remains loadable but is not shown as a selectable catalog skill.
- Replay regression updated so 611 no longer relies on the old hard-skill full sweep.

## Verification

Current local checks:

- `PYTHONPATH=src:. pytest -q` -> `462 passed in 1.77s`
- `git diff --check` -> passed

## Launched KML Rerun

Code commit used for launch:

`7468bef`

Run root:

`/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_post_closure_7468bef_3demo_20260609_213500_pyenv`

Artifacts:

- pid: `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_post_closure_7468bef_3demo_20260609_213500_pyenv/job.pid`
- log: `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_post_closure_7468bef_3demo_20260609_213500_pyenv/job.log`
- summary: `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_post_closure_7468bef_3demo_20260609_213500_pyenv/summary.json`

Launch PID:

`36743`

## Files Changed

- `src/visual_coding_agent_harness/agents/iterative_agent.py`
- `src/visual_coding_agent_harness/agents/prompt_stack.py`
- `src/visual_coding_agent_harness/agents/question_policy.py`
- `src/visual_coding_agent_harness/agents/skills/specs.py`
- `src/visual_coding_agent_harness/evals/videomme/runner.py`
- `src/visual_coding_agent_harness/tools/inspector.py`
- `src/visual_coding_agent_harness/tools/navigation.py`
- focused tests and replay fixture under `tests/`

## Constraints

- Do not read raw KML logs into chat or notes.
- KML git access needs:

`export http_proxy=http://oversea-squid1.jp.txyun:11080 https_proxy=http://oversea-squid1.jp.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com`

- New KML run must use `/home/xuboshen/Anaconda/envs/visual-agent-harness/bin/python`.

## Next Actions

1. Commit and push local changes to GitHub.
2. On KML, pull with the proxy environment.
3. Start a detached three-demo VideoMME run with dedicated Python.
4. Return run root, log path, pid path, and summary path to the user; do not monitor unless explicitly requested.
