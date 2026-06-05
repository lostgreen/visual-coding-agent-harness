# 2026-06-05 Visual Ticket KML Debug Handoff

## Current Goal

Implement the first foundation slice from `visual-coding-agent-harness-ticket-plan.md`, sync it through GitHub to the KML CodeBase, run a VideoMME subset debug pass on KML, and fix the hard-skill `need_more_evidence` early-exit so it can continue through the main agent loop.

## Current Evidence

- Local branch: `codex/visual-harness-ticket-plan`.
- GitHub branch: `origin/codex/visual-harness-ticket-plan`.
- KML target: `https://kml-dtmachine-23666-prod-0.kmlhb2az1l3-2.corp.kuaishou.com/`.
- KML repo: `/home/xuboshen/zgw/visual-coding-agent-harness`.
- KML run environment: `/home/xuboshen/Anaconda/envs/visual-agent-harness/bin/python`.
- KML proxy used for GitHub/model connectivity:
  - `http_proxy=http://oversea-squid1.jp.txyun:11080`
  - `https_proxy=http://oversea-squid1.jp.txyun:11080`
  - `no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com`

## Branch Sync

- Foundation commit: `dbff182 Add visual evidence foundation contracts`.
- Direct-script import fix: `3bc95d1 Support direct eval runner script imports`.
- Follow-up loop fix: `42bd30d Continue hard-skill followups before abstaining`.
- Visual manifest producer slice: `de596ec [TASK-007-010] write visual tool frame manifests`.
- Trace summary metrics slice: `f6a122e [TASK-002-024] aggregate trace summary metrics`.
- Evidence chain linkage slice: `db1e0ec [TASK-015-016] chain ledger and mapped evidence`.
- Mapped grounding floor slice: `99be1ed [TASK-017] enforce mapped grounding floor`.
- Evidence chain export slice: `266380a [TASK-018] export evidence chains`.
- Current local branch is ahead of GitHub/KML by these newer DAG commits until the next sync.
- KML branch was synced with:
  - `git fetch origin codex/visual-harness-ticket-plan`
  - `git checkout -B codex/visual-harness-ticket-plan FETCH_HEAD`
- Important preservation note: the prior remote branch `codex/v4-skill-framework` had local ahead commits with different hashes; it was not overwritten.

## Implemented Foundation Slice

- `RunSummary` schema and eval summary integration.
- Visual evidence constants and `resolve_nframes` contract.
- `FrameSetManifest` storage and Observation-to-manifest side-index links.
- `EvidenceRecord` and `MapUpdateProposal` append-only storage.
- Ledger-stage and mapped-stage `EvidenceRecord` parent chains.
- Compact `evidence_chains` workspace artifact plus run-level `evidence_chains.jsonl`.
- Follow-up target normalization and scheduler.
- Context budget allocator.
- Initial hard-skill route validator.
- Interpreter-side distill hook from observations to evidence records.
- Initial `tool_nframes_compliance` metric.
- Trace-derived `route_violations`, `avg_followups_per_case`, and `followup_success_rate` summary metrics.

## Verification

- Local full test command: `PYTHONPATH=src python -m pytest -q`.
- Local result after DAG continuation through `TASK-018`: `190 passed`.
- Focused local regression check:
  - `PYTHONPATH=src python -m pytest tests/test_prompt_stack_and_skill_runtime.py -k 'hard_skill_runtime_continues_followup_chunks or hard_skill_need_more_hands_off or hard_grounded_skill_runtime' -q`
  - Result: `3 passed`.
- KML full test command: `PYTHONPATH=src /home/xuboshen/Anaconda/envs/visual-agent-harness/bin/python -m pytest -q`.
- KML result after follow-up loop sync: `170 passed`.
- KML direct-script entrypoint check:
  - `PYTHONPATH=src /home/xuboshen/Anaconda/envs/visual-agent-harness/bin/python runs/eval_runner.py --help`
  - Result: entrypoint imported successfully.

## VideoMME Debug Runs

### Stale failed startup run

- Run root intent: `runs/videomme_agent_ticket_foundation_hardskill_3case_20260605`.
- PID/log paths:
  - `/tmp/videomme_ticket_foundation_hardskill_3case_20260605.pid`
  - `/tmp/videomme_ticket_foundation_hardskill_3case_20260605.log`
- Status: stale failure before evaluation startup.
- Failure fingerprint: `ModuleNotFoundError: No module named 'runs'`.
- Root cause: `python runs/eval_runner.py` sets `sys.path[0]` to `runs/`, while the new code imported `runs.summary_schema`; historical launch used `PYTHONPATH=src`, so repo root was not importable.
- Fix: `eval_runner.py` now falls back from `runs.summary_schema` to same-directory `summary_schema`; regression test covers `python runs/eval_runner.py --help`.

### Foundation rerun before follow-up-loop fix

- Run root: `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_ticket_foundation_hardskill_3case_rerun_20260605`.
- Cases: `605-1`, `611-2`, `612-1`.
- Strategy: `agent_v2`.
- Mode: `--hard-skill-runtime`.
- PID/log paths:
  - `/tmp/videomme_ticket_foundation_hardskill_3case_rerun_20260605.pid`
  - `/tmp/videomme_ticket_foundation_hardskill_3case_rerun_20260605.log`
- Initial draft status: running.
- Summary path expected: `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_ticket_foundation_hardskill_3case_rerun_20260605/summary.json`.
- Final status: completed successfully, returncode 0.
- Summary path: `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_ticket_foundation_hardskill_3case_rerun_20260605/summary.json`.
- `summary_violations.json`: not present.
- Evidence status: stale for follow-up-loop behavior after commit `42bd30d`; still useful as the pre-fix baseline.

Compact metrics:

| Metric | Value |
| --- | --- |
| `accuracy` | `0.3333333333333333` |
| `final_rate` | `0.3333333333333333` |
| `need_more_evidence_rate` | `0.6666666666666666` |
| `unsupported_final_rate` | `0.0` |
| `tool_nframes_compliance` | `1.0` |
| `legacy_worker_vote_rows` | `0` |
| `route_violations` | `0` |

Compact case results:

| Case | GT | Status | Choice | Correct | Rounds | Citations | Tools | Trajectory actions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `605-1` | `D` | `final` | `D` | true | 1 | 1 | `global_gist` | 1 |
| `611-2` | `D` | `need_more_evidence` | `` | false | 1 | 0 | `ground_question`, `vision_read`, `ground_question`, `vision_read` | 4 |
| `612-1` | `B` | `need_more_evidence` | `` | false | 1 | 2 | `ground_question`, `vision_read`, `ground_question`, `vision_read` | 4 |

### Follow-up-loop fix rerun

- Run root: `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_ticket_followupfix_hardskill_3case_20260605`.
- Cases: `605-1`, `611-2`, `612-1`.
- Strategy: `agent_v2`.
- Mode: `--hard-skill-runtime`.
- PID/log paths:
  - `/tmp/videomme_ticket_followupfix_hardskill_3case_20260605.pid`
  - `/tmp/videomme_ticket_followupfix_hardskill_3case_20260605.log`
- Final status: completed successfully, returncode 0.
- Summary path: `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_ticket_followupfix_hardskill_3case_20260605/summary.json`.
- `summary_violations.json`: not present.

Compact metrics:

| Metric | Value |
| --- | --- |
| `accuracy` | `0.3333333333333333` |
| `final_rate` | `0.3333333333333333` |
| `need_more_evidence_rate` | `0.0` |
| `unsupported_final_rate` | `0.0` |
| `tool_nframes_compliance` | `1.0` |
| `legacy_worker_vote_rows` | `0` |
| `route_violations` | `0` |

Compact case results:

| Case | GT | Status | Choice | Correct | Rounds | Citations | Tools | Trajectory actions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `605-1` | `D` | `final` | `D` | true | 1 | 1 | `global_gist` | 1 |
| `611-2` | `D` | `max_rounds_reached` | `` | false | 6 | 2 | `ground_question`, `vision_read` x4, then `vision_read` x2 | 10 |
| `612-1` | `B` | `max_rounds_reached` | `` | false | 5 | 0 | `ground_question`, `vision_read` x4 | 8 |

Compact trace evidence for the fixed behavior:

| Case | `hard_skill_followup_handoff` | Main planner rounds | Planner IO events | No-progress guard |
| --- | --- | --- | --- | --- |
| `605-1` | 0 | 0 | 0 | 0 |
| `611-2` | 1 | 5 | 5 | 1 |
| `612-1` | 1 | 4 | 4 | 1 |

Current diagnosis:

- The direct-script import issue is fixed and the KML runs now reach evaluation.
- The new summary schema passes validation on the VideoMME hard-skill subset.
- Root cause of `611-2`/`612-1` pre-fix `need_more_evidence`: `_try_hard_skill_route()` consumed only the first hard-skill chunk and returned `need_more_evidence` directly instead of handing control back to the iterative planner.
- Commit `42bd30d` fixes the early-exit path for hard-skill runtime: hard-skill targets are chunked through `FollowupScheduler`; if AnswerAgent still abstains and round budget remains, control is handed to the main replanning loop via `hard_skill_followup_handoff`.
- Post-fix, `611-2` and `612-1` no longer terminate as immediate `need_more_evidence`; they perform follow-up evidence collection and main planner turns, then stop at `max_rounds_reached` because no final answer is supported within budget.
- The useful pre-DAG-continuation gate signal was strong: `unsupported_final_rate=0`, `legacy_worker_vote_rows=0`, and `route_violations=0`.
- Stale before `de596ec`: `nframes_histogram` was empty because tool-to-manifest integration had not been implemented yet. A post-sync KML smoke run is needed to validate the new manifest producers.
- Stale before `f6a122e`: `avg_followups_per_case` did not yet reflect the hard-skill follow-up path. Local code now aggregates follow-up attempts and success rate from trace events; KML needs a post-sync rerun.

## 2026-06-05 Ticket Plan Delta Audit

Source plan reviewed: `/Users/lostgreen/Downloads/visual-coding-agent-harness-ticket-plan.md`.

Status counts against the original 38 TASKs:

- Done: 19
- Partial: 9
- Not started: 10

Task-level progress:

| Task | Plan item | Current status | Delta / remaining work |
| --- | --- | --- | --- |
| `TASK-001` | `RunSummary` schema and serialization | Done | Schema exists in `runs/summary_schema.py`; `eval_runner.py` writes schema fields, keeps legacy `cases`, validates, and writes `summary_violations.json` on failure. |
| `TASK-002` | route x action validator | Partial | Hard-skill planner normalization has allowed-action gating and route violation trace events; summary now aggregates `route_violation` from traces. Remaining gap: validation is not yet a central interpreter/tool-registry contract across all routes. |
| `TASK-003` | quarantine legacy worker vote | Done | Current v4 code quarantines legacy worker option votes in evidence tables, and the KML subset reports `legacy_worker_vote_rows=0`. This was mostly existing v4 behavior, not new foundation work. |
| `TASK-004` | CI smoke runs | Partial | Local and KML manual smoke/full pytest runs are recorded; no CI workflow/smoke gate was added in this branch. |
| `TASK-005` | `agents/contracts.py` | Done | Contract constants and `resolve_nframes` are implemented with tests. |
| `TASK-006` | `FrameSetManifest` storage | Done | Manifest dataclass plus append/read helpers and directory creation are implemented. |
| `TASK-007` | `global_gist` manifest integration | Done | Interpreter now attaches `FrameSetManifest` records for `global_gist`; default nframes contract is 128. |
| `TASK-008` | `vision_read` manifest integration | Done | `vision_read` observations now attach manifests and respect `resolve_nframes`. |
| `TASK-009` | `inspect_segment` manifest integration | Done | `inspect_segment` observations now attach manifests and respect `resolve_nframes`. |
| `TASK-010` | segment caption/QA contract integration | Done | `caption_segment` and `qa_segment` now use the 128-frame contract and manifest producer path. |
| `TASK-011` | Observation `frame_set_id` | Done | Visual observations are reloaded after manifest linking so downstream distill/ledger evidence sees `frame_set_id`. |
| `TASK-012` | `tool_nframes_compliance` metric | Partial | Metric and manifest histogram are wired; acceptance still needs a post-sync KML smoke run with non-empty manifest coverage and compliance >= 0.95. |
| `TASK-013` | `EvidenceRecord` storage | Done | Dataclass, IDs, append/read, and chain lookup are implemented with tests. |
| `TASK-014` | interpreter distillation hook | Partial | Interpreter writes distilled evidence records for tool observations; per-tool distillation semantics and all observation-producing paths still need completion. |
| `TASK-015` | ledger entries with `parent_id` | Done | `ledger` stage records are written alongside `ledger.md` rows and parent to distilled evidence; split facts produce one ledger record per distilled fact. |
| `TASK-016` | AnswerAgent `mapped_evidence` | Done | Candidate-option relations persist as `mapped` EvidenceRecords parented to ledger/distilled evidence; orphan parent references are dropped and counted. |
| `TASK-017` | verifier `grounding_quality_floor` gate | Done | Hard-skill final gate and `verify_ledger_answer` enforce mapped-chain visual grounding; `gist_qa` can allow global-only support. |
| `TASK-018` | `evidence_chains.jsonl` export | Done | Workspace exports compact evidence chain artifacts, eval run writes root `evidence_chains.jsonl`, and summary computes `evidence_provenance_completeness`. |
| `TASK-019` | `FollowupTarget` / `FollowupBudget` | Done | Data structures and normalization are implemented with tests. |
| `TASK-020` | `FollowupScheduler` | Done | Queue, retry, global budget, and saturation behavior are implemented with tests. |
| `TASK-021` | main loop followup integration | Partial | Hard-skill runtime now schedules target-fact follow-up chunks and hands unresolved cases back to the iterative planner; generic follow-up integration for all `need_more_evidence` sources and metrics remains incomplete. |
| `TASK-022` | per-route followup strategies | Partial | Minimal hard-skill route mapping exists (`temporal_ordering` -> `temporal_order`, default -> `needle_local`) and executes `ground_question -> vision_read`; full temporal/needle/gist strategies across all tools/routes remain incomplete. |
| `TASK-023` | `low_confidence_final` path | Not started | No low-confidence final status path or evidence-chain integration. |
| `TASK-024` | followup metrics | Done | Summary now aggregates hard-skill `ground_question` follow-up attempts and success rate from trace events. |
| `TASK-025` | `ContextSlot` / allocator | Done | Context budget primitives and tests exist. |
| `TASK-026` | slot-based `prompt_stack` refactor | Partial | Existing prompt stack remains active, but it is not driven by the new allocator and trace lacks `context_budget_report`. |
| `TASK-027` | per-slot compact strategies | Not started | No production compact strategies are wired to prompt slots. |
| `TASK-028` | CLI flags for context budgets | Not started | Eval CLI has existing tool-budget flags, but no context-slot budget controls. |
| `TASK-029` | `tools/query_context.py` | Not started | No query-context tool implementation. |
| `TASK-030` | `MapUpdateProposal` storage | Done | Dataclass plus append/read pending proposals are implemented with tests. |
| `TASK-031` | vision tools produce proposals | Not started | No tool emits `map_proposals.jsonl`. |
| `TASK-032` | proposal commit channel | Not started | No commit/apply path for map proposals. |
| `TASK-033` | `TrainingTrajectory` schema | Not started | Plan-specific schema is not implemented. |
| `TASK-034` | trajectory exporter | Partial | Existing LongVideoAgent-style trajectory export is present, but it does not use the planned `TrainingTrajectory` schema or depend on evidence-chain export. |
| `TASK-035` | audit CLI | Not started | No trajectory audit CLI. |
| `TASK-036` | eval CLI extensions | Partial | Eval runner has existing strategy, budget, hard-skill, and free-explore flags; plan-specific flags for later phases/ablation are not complete. |
| `TASK-037` | ablation matrix runner | Not started | No matrix runner. |
| `TASK-038` | report generator | Not started | Existing report utilities remain, but no plan-specific final report generator for the ablation matrix. |

Epic-level gap summary:

- EPIC 1 is partially complete: schema, quarantine behavior, and trace metric aggregation are usable; route validation and CI gate need stronger integration.
- EPIC 2 producer code is now implemented locally; KML smoke validation is the remaining blocker for frame evidence compliance.
- EPIC 3 linkage is now implemented locally through ledger, mapped evidence, grounding floor, and compact chain export.
- EPIC 4 has scheduler/data structures, a hard-skill follow-up handoff into the main loop, and trace-derived follow-up metrics; generic all-route strategies remain incomplete.
- EPIC 5 only has allocator primitives; prompt stack refactor and compaction are not connected.
- EPIC 6 and EPIC 7 have only `MapUpdateProposal` storage from EPIC 7; query-context and proposal producers are not started.
- EPIC 8 has existing trajectory export but not the planned schema/audit pipeline.
- EPIC 9 is mostly not started beyond existing eval runner/report utilities.

## Current Constraints

- Do not read or paste raw KML logs, raw model responses, trace dumps, or full JSON artifacts into chat.
- Inspect failures by filtered error fingerprints or compact JSON aggregations only.
- Keep the three pre-existing local dirty docs untouched unless explicitly taking ownership:
  - `docs/2026-06-03-flow-context-management-design.zh.md`
  - `docs/2026-06-03-handoff-videomme-agent-loop.md`
  - `docs/report-design-proposal.zh.md`

## Next Actions

1. Sync the local DAG commits to GitHub/KML, then run KML full pytest and a compact VideoMME subset smoke to validate manifest coverage, evidence-chain completeness, and updated follow-up metrics.
2. Inspect only compact KML artifacts: `summary.json` metrics, `nframes_histogram`, `evidence_provenance_completeness`, and root `evidence_chains.jsonl` row counts.
3. Continue remaining partial/not-start DAG after KML validation: generic all-route follow-up strategies (`TASK-021`/`022`), low-confidence final (`TASK-023`), context-slot prompt integration (`TASK-026`-`028`), query-context/map proposal producers (`TASK-029`/`031`/`032`), and training/reporting tasks (`TASK-033`-`038`).
4. Keep EPIC 5+ expansion behind the three anchor cases showing real manifest coverage and evidence-chain completeness.
