# 2026-06-05 Visual Ticket KML Debug Handoff

## Current Goal

Implement the first foundation slice from `visual-coding-agent-harness-ticket-plan.md`, sync it through GitHub to the KML CodeBase, and run a VideoMME subset debug pass on KML.

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
- KML branch was synced with:
  - `git fetch origin codex/visual-harness-ticket-plan`
  - `git checkout -B codex/visual-harness-ticket-plan FETCH_HEAD`
- Important preservation note: the prior remote branch `codex/v4-skill-framework` had local ahead commits with different hashes; it was not overwritten.

## Implemented Foundation Slice

- `RunSummary` schema and eval summary integration.
- Visual evidence constants and `resolve_nframes` contract.
- `FrameSetManifest` storage and Observation-to-manifest side-index links.
- `EvidenceRecord` and `MapUpdateProposal` append-only storage.
- Follow-up target normalization and scheduler.
- Context budget allocator.
- Initial hard-skill route validator.
- Interpreter-side distill hook from observations to evidence records.
- Initial `tool_nframes_compliance` metric.

## Verification

- Local full test command: `PYTHONPATH=src python -m pytest -q`.
- Local result after import fix: `168 passed`.
- KML full test command: `PYTHONPATH=src /home/xuboshen/Anaconda/envs/visual-agent-harness/bin/python -m pytest -q`.
- KML result after sync: `168 passed`.
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

### Current rerun

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

Current diagnosis:

- The direct-script import issue is fixed and the KML run now reaches evaluation.
- The new summary schema passes validation on the VideoMME hard-skill subset.
- The foundation slice does not regress the known hard-skill behavior: `605-1` remains correct through `global_gist`; temporal cases still conservatively abstain.
- The useful gate signal is strong: `unsupported_final_rate=0`, `legacy_worker_vote_rows=0`, and `route_violations=0`.
- `nframes_histogram` is currently empty because the tool-to-manifest integration tickets are not implemented yet; the compliance metric correctly defaults to `1.0` when no manifests exist.

## 2026-06-05 Ticket Plan Delta Audit

Source plan reviewed: `/Users/lostgreen/Downloads/visual-coding-agent-harness-ticket-plan.md`.

Status counts against the original 38 TASKs:

- Done: 9
- Partial: 11
- Not started: 18

Task-level progress:

| Task | Plan item | Current status | Delta / remaining work |
| --- | --- | --- | --- |
| `TASK-001` | `RunSummary` schema and serialization | Done | Schema exists in `runs/summary_schema.py`; `eval_runner.py` writes schema fields, keeps legacy `cases`, validates, and writes `summary_violations.json` on failure. |
| `TASK-002` | route x action validator | Partial | Hard-skill planner normalization has allowed-action gating and route violation trace events, but validation is not yet a central interpreter/tool-registry contract and summary does not aggregate real route violation counts from traces. |
| `TASK-003` | quarantine legacy worker vote | Done | Current v4 code quarantines legacy worker option votes in evidence tables, and the KML subset reports `legacy_worker_vote_rows=0`. This was mostly existing v4 behavior, not new foundation work. |
| `TASK-004` | CI smoke runs | Partial | Local and KML manual smoke/full pytest runs are recorded; no CI workflow/smoke gate was added in this branch. |
| `TASK-005` | `agents/contracts.py` | Done | Contract constants and `resolve_nframes` are implemented with tests. |
| `TASK-006` | `FrameSetManifest` storage | Done | Manifest dataclass plus append/read helpers and directory creation are implemented. |
| `TASK-007` | `global_gist` manifest integration | Not started | Tool still does not create `FrameSetManifest` or attach `frame_set_id`; KML `nframes_histogram` is empty. |
| `TASK-008` | `vision_read` manifest integration | Not started | Same gap as `TASK-007`. |
| `TASK-009` | `inspect_segment` manifest integration | Not started | Same gap as `TASK-007`. |
| `TASK-010` | segment caption/QA contract integration | Not started | Segment caption/QA tools are not wired to the contract/manifest producer path. |
| `TASK-011` | Observation `frame_set_id` | Partial | Observation carries `frame_set_id` and side-index links can resolve it, but visual tools do not yet reserve/write observations with real manifest IDs. |
| `TASK-012` | `tool_nframes_compliance` metric | Partial | Metric exists and validates no-manifest runs as `1.0`; acceptance requires real smoke-run manifest compliance >= 0.95 after `TASK-007`-`010`. |
| `TASK-013` | `EvidenceRecord` storage | Done | Dataclass, IDs, append/read, and chain lookup are implemented with tests. |
| `TASK-014` | interpreter distillation hook | Partial | Interpreter writes distilled evidence records for tool observations; per-tool distillation semantics and all observation-producing paths still need completion. |
| `TASK-015` | ledger entries with `parent_id` | Not started | Ledger-stage EvidenceRecord parity and parent links are not implemented. |
| `TASK-016` | AnswerAgent `mapped_evidence` | Partial | Existing v4 can annotate candidate option relations on observations; plan-specific mapped `EvidenceRecord` lifecycle records are not implemented. |
| `TASK-017` | verifier `grounding_quality_floor` gate | Partial | KML subset has `unsupported_final_rate=0`, but this is not yet enforced by a dedicated verifier floor over mapped evidence records. |
| `TASK-018` | `evidence_chains.jsonl` export | Partial | Workspace can compute an in-memory evidence chain; no `evidence_chains.jsonl` export or completeness metric is implemented. |
| `TASK-019` | `FollowupTarget` / `FollowupBudget` | Done | Data structures and normalization are implemented with tests. |
| `TASK-020` | `FollowupScheduler` | Done | Queue, retry, global budget, and saturation behavior are implemented with tests. |
| `TASK-021` | main loop followup integration | Not started | Hard-skill loop still returns `need_more_evidence` without scheduling follow-up rounds. |
| `TASK-022` | per-route followup strategies | Not started | No temporal/needle/gist follow-up execution strategies are wired. |
| `TASK-023` | `low_confidence_final` path | Not started | No low-confidence final status path or evidence-chain integration. |
| `TASK-024` | followup metrics | Not started | Summary fields default to zero; no real follow-up metrics are populated. |
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

- EPIC 1 is partially complete: schema and quarantine behavior are usable; route validation and CI gate need stronger integration.
- EPIC 2 storage is ready, but real visual tools do not yet produce manifests; this is the biggest blocker for frame evidence compliance.
- EPIC 3 storage is ready and distillation is started; mapped evidence, verifier floor, ledger parent links, and exported evidence chains remain.
- EPIC 4 only has scheduler/data structures; it is not connected to the agent loop.
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

1. Finish EPIC 2 tool producers first: `TASK-007` through `TASK-010`, then validate `TASK-011`/`TASK-012` on a KML smoke run with non-empty `nframes_histogram`.
2. Finish EPIC 3 linkage: `TASK-015`, `TASK-016`, `TASK-017`, and `TASK-018`, so final gating uses mapped evidence chains rather than only summary defaults.
3. Wire EPIC 4 follow-up into the hard-skill loop: `TASK-021`/`TASK-022`, then rerun the same three VideoMME anchors and track `need_more_evidence_rate`.
4. Integrate EPIC 5 context budgeting into `prompt_stack` only after evidence/follow-up paths are stable.
5. Defer EPIC 6-9 expansion until the three anchor cases show real manifest coverage and evidence-chain completeness.
