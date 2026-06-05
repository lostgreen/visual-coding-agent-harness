# 2026-06-05 Visual Ticket KML Debug Handoff

## Current Goal

Continue the remaining partial/not-start DAG from `visual-coding-agent-harness-ticket-plan.md` using multi-agent review, sync through GitHub to the KML CodeBase, and keep current local/KML verification plus compact VideoMME smoke evidence in this handoff.

## Current Evidence

- Local branch: `codex/visual-harness-ticket-plan`.
- GitHub branch: `origin/codex/visual-harness-ticket-plan`.
- Latest code commit verified locally and on KML: `6c819c2 fix training trajectory export path`.
- Later docs-only handoff commits may advance branch HEAD without changing the verification evidence below.
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
- Context budget prompt-slot slice: `3fa4865 [TASK-026-028] wire context budget prompt slots`.
- Training trajectory slice: `e854468 [TASK-033-035] export training trajectories`.
- Follow-up/summary metric slice: `d392a2f [TASK-021-024] trace followup attempts and low confidence metrics`.
- Query-context/map proposal slice: `7332484 [TASK-029-032] add query context map proposals`.
- Low-confidence budget path slice: `00426f0 [TASK-023] finalize low confidence budget path`.
- Ablation/reporting slice: `bec2039 [TASK-036-038] add ablation matrix reporting`.
- Training trajectory export path fix: `6c819c2 fix training trajectory export path`.
- KML branch was synced with:
  - `git fetch origin codex/visual-harness-ticket-plan`
  - `git checkout -B codex/visual-harness-ticket-plan FETCH_HEAD`
- Latest KML code verification was run at `6c819c2`; KML branch is also synced after docs-only handoff updates.
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
- Slot-based replanning prompt with context budget reports and per-slot compaction strategies.
- `TrainingTrajectoryV1` exporter plus trajectory audit CLI.
- `query_context` context-only tool, frame manifest linkage, map proposal producers, and explicit `commit_map_proposals` channel.
- `low_confidence_final` integration for reserved-final, prefinal-budget-exhausted, generic budget-exhausted, and hard-skill-budget-exhausted paths; still requires answer-facing non-navigation visual evidence.
- Ablation CLI flags, JSON matrix runner, and JSON/Markdown ablation report generator.

## Verification

- Local full test command: `PYTHONPATH=src:. python -m pytest -q`.
- Local current result at `6c819c2`: `220 passed in 0.66s`.
- Focused local regression checks after reviewer findings:
  - `PYTHONPATH=src:. python -m pytest -q tests/test_query_context.py tests/scripts/test_run_ablation.py tests/test_iterative_agent.py tests/test_answer_agent.py`
  - Result: `54 passed`.
  - `PYTHONPATH=src:. python -m pytest -q tests/test_eval_runner.py tests/scripts/test_generate_ablation_report.py tests/runs/test_training_trajectory.py tests/runs/test_audit_trajectory.py`
  - Result after trajectory-path fix: `20 passed`.
- KML full test command: `PYTHONPATH=src:. /home/xuboshen/Anaconda/envs/visual-agent-harness/bin/python -m pytest -q`.
- KML result at `6c819c2`: `220 passed in 3.74s`.
- KML focused trajectory path regression:
  - `PYTHONPATH=src:. /home/xuboshen/Anaconda/envs/visual-agent-harness/bin/python -m pytest -q tests/test_eval_runner.py -k training_trajectory_export_path`
  - Result: `1 passed, 14 deselected`.
- KML direct-script entrypoint check:
  - `PYTHONPATH=src /home/xuboshen/Anaconda/envs/visual-agent-harness/bin/python runs/eval_runner.py --help`
  - Result: entrypoint imported successfully.
- KML CLI smoke after ablation tooling:
  - `runs/eval_runner.py --help`, `scripts/run_ablation.py --help`, and `scripts/generate_ablation_report.py --help`
  - Result: `cli_smoke_ok`.

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
- Evidence status: stale for commits after `f5bc354`; useful only as the pre-DAG-continuation behavior baseline.

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

### DAG continuation smoke at `bec2039`

- Run root: `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_ticket_dag_bec2039_hardskill_3case_20260605`.
- Cases: `605-1`, `611-2`, `612-1`.
- Strategy: `agent_v2`.
- Mode: `--hard-skill-runtime --export-training --allow-any-python`.
- PID/log paths:
  - `/tmp/videomme_ticket_dag_bec2039_hardskill_3case_20260605.pid`
  - `/tmp/videomme_ticket_dag_bec2039_hardskill_3case_20260605.log`
- Final status: completed successfully, returncode 0.
- Summary path: `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_ticket_dag_bec2039_hardskill_3case_20260605/summary.json`.
- Evidence status: current for code through `bec2039`; after this smoke, `6c819c2` fixed a trajectory export path bug only, and KML pytest plus focused trajectory-path regression passed at `6c819c2`.
- Important bug found from this smoke: `training_trajectory_exported=True` but summary trajectory paths did not exist when `run_root` was relative. Fixed by `6c819c2`.

Compact metrics:

| Metric | Value |
| --- | --- |
| `accuracy` | `0.0` |
| `final_rate` | `0.6666666666666666` |
| `need_more_evidence_rate` | `0.0` |
| `unsupported_final_rate` | `0.0` |
| `low_confidence_final_rate` | `0.0` |
| `tool_nframes_compliance` | `1.0` |
| `evidence_provenance_completeness` | `0.3333333333333333` |
| `legacy_worker_vote_rows` | `0` |
| `route_violations` | `0` |
| `avg_followups_per_case` | `2.6666666666666665` |
| `followup_success_rate` | `0.5` |
| `context_budget_overflow_count` | `6` |
| `avg_tokens_per_turn` | `5062` |
| `query_context_usage_rate` | `0.0` |
| `map_reflux_commit_count` | `0` |

Nframes histogram:

| Tool | Histogram |
| --- | --- |
| `global_gist` | `{128: 1}` |
| `vision_read` | `{64: 2, 128: 8}` |

Compact case results:

| Case | GT | Status | Choice | Correct | Rounds | Citations | Tools |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `605-1` | `D` | `final` | `B` | false | 1 | 1 | `global_gist` |
| `611-2` | `D` | `max_rounds_reached` | `` | false | 8 | 4 | `ground_question`, `vision_read`, `zoom` |
| `612-1` | `B` | `final` | `D` | false | 2 | 2 | `ground_question`, `vision_read` |

Current diagnosis:

- `need_more_evidence` is no longer an early terminal for these anchors: rate is `0.0`; 611-2 now uses the loop until `max_rounds_reached`.
- The current code improves observability/compliance (`tool_nframes_compliance=1.0`, non-empty `nframes_histogram`, trace follow-up metrics) but not accuracy on the 3-case smoke.
- `context_budget_overflow_count=6` shows EPIC 5 is wired but budgets/compaction still need tuning.
- `query_context_usage_rate=0.0` and `map_reflux_commit_count=0` mean the new EPIC 6/7 code is implemented and tested, but this smoke did not exercise it.
- `evidence_provenance_completeness=0.3333` remains a gap for reportability and trajectory quality.

## 2026-06-05 Ticket Plan Delta Audit

Source plan reviewed: `/Users/lostgreen/Downloads/visual-coding-agent-harness-ticket-plan.md`.

Status counts against the original 38 TASKs:

- Done: 28
- Partial: 10
- Not started: 0

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
| `TASK-012` | `tool_nframes_compliance` metric | Done | Metric and manifest histogram are wired; current KML smoke reports `tool_nframes_compliance=1.0` with non-empty `global_gist`/`vision_read` histogram. |
| `TASK-013` | `EvidenceRecord` storage | Done | Dataclass, IDs, append/read, and chain lookup are implemented with tests. |
| `TASK-014` | interpreter distillation hook | Partial | Interpreter writes distilled evidence records for tool observations; per-tool distillation semantics and all observation-producing paths still need completion. |
| `TASK-015` | ledger entries with `parent_id` | Done | `ledger` stage records are written alongside `ledger.md` rows and parent to distilled evidence; split facts produce one ledger record per distilled fact. |
| `TASK-016` | AnswerAgent `mapped_evidence` | Done | Candidate-option relations persist as `mapped` EvidenceRecords parented to ledger/distilled evidence; orphan parent references are dropped and counted. |
| `TASK-017` | verifier `grounding_quality_floor` gate | Done | Hard-skill final gate and `verify_ledger_answer` enforce mapped-chain visual grounding; `gist_qa` can allow global-only support. |
| `TASK-018` | `evidence_chains.jsonl` export | Done | Workspace exports compact evidence chain artifacts, eval run writes root `evidence_chains.jsonl`, and summary computes `evidence_provenance_completeness`. |
| `TASK-019` | `FollowupTarget` / `FollowupBudget` | Done | Data structures and normalization are implemented with tests. |
| `TASK-020` | `FollowupScheduler` | Done | Queue, retry, global budget, and saturation behavior are implemented with tests. |
| `TASK-021` | main loop followup integration | Partial | Hard-skill runtime schedules target-fact follow-up chunks and hands unresolved cases back to the iterative planner; AnswerAgent prefinal gaps feed planner feedback. Generic scheduler-backed follow-up for every `need_more_evidence` source remains incomplete. |
| `TASK-022` | per-route followup strategies | Partial | Minimal hard-skill route mapping exists (`temporal_ordering` -> `temporal_order`, default -> `needle_local`) and executes `ground_question -> vision_read`; full temporal/needle/gist strategies across all tools/routes remain incomplete. |
| `TASK-023` | `low_confidence_final` path | Done | AnswerAgent partial-support helpers are integrated into reserved-final, prefinal-budget-exhausted, generic budget-exhausted, and hard-skill-budget-exhausted paths; gate still requires answer-facing non-navigation visual citation. Unit tests cover positive and blocked paths. Current 3-case smoke did not trigger low-confidence. |
| `TASK-024` | followup metrics | Done | Summary now aggregates hard-skill `ground_question` follow-up attempts and success rate from trace events. |
| `TASK-025` | `ContextSlot` / allocator | Done | Context budget primitives and tests exist. |
| `TASK-026` | slot-based `prompt_stack` refactor | Done | Replanning prompt is slot-based (`task`, `navigation`, `evidence`, `feedback`) and writes `context_budget_report` each round. |
| `TASK-027` | per-slot compact strategies | Partial | Production allocator registers task/nav/evidence/feedback strategies and tests cover behavior. Current KML smoke still has `context_budget_overflow_count=6`, so acceptance `overflow_count == 0` is not met. |
| `TASK-028` | CLI flags for context budgets | Done | Eval CLI parses `--context-budget-tokens` and `--budget-ratios`; tests cover valid and invalid ratio config. |
| `TASK-029` | `tools/query_context.py` | Partial | `query_context` tool, registry integration, manifest path, context-only ledger handling, and answer-support exclusion are implemented with tests. Remaining gap: no round-0-only policy enforcement and current KML smoke did not exercise the tool. |
| `TASK-030` | `MapUpdateProposal` storage | Done | Dataclass plus append/read pending proposals are implemented with tests. |
| `TASK-031` | vision tools produce proposals | Partial | Interpreter creates `context_update` proposals from `query_context`, `vision_read`, `inspect_segment`, `caption_segment`, and `qa_segment` observations when segment/frame-set evidence exists. Remaining gap: entity/OCR/ASR diff-specific proposal types from raw tool output are not complete, and current KML smoke did not write proposals. |
| `TASK-032` | proposal commit channel | Done | `commit_map_proposals` explicit tool applies whitelisted payload fields through `VideoMapStore.update_segment()` and marks proposals committed; tests cover apply/commit behavior. |
| `TASK-033` | `TrainingTrajectory` schema | Done | `TrainingTrajectoryV1` exists with evidence chains, frame sets, tool calls, context reports, and follow-up history. |
| `TASK-034` | trajectory exporter | Done | Eval runner exports per-case training trajectories behind `--export-training`; `6c819c2` fixes relative `run_root` path export and tests cover path existence. |
| `TASK-035` | audit CLI | Done | `scripts/audit_trajectory.py` audits `TrainingTrajectoryV1`; unit tests cover valid and invalid artifacts. |
| `TASK-036` | eval CLI extensions | Partial | Plan flags are parsed and serialized to `run_config.json`; `--contract-nframes`, context-budget disable, follow-up enable/disable, and follow-up budget affect current config. Remaining gap: some feature flags are recorded but not yet behaviorally wired (`enable_query_context`, `enable_map_reflux`, `enable_evidence_staging`). |
| `TASK-037` | ablation matrix runner | Done | `scripts/run_ablation.py` builds dry-run/serial matrix entries, writes `index.json`, handles paired boolean disable flags, and has tests. Uses JSON instead of YAML to avoid a new dependency. |
| `TASK-038` | report generator | Partial | `scripts/generate_ablation_report.py` writes `ablation_report.json` and `REPORT.md` with metrics/completeness and trajectory audit summaries. Remaining gap: no per-case comparison table or graph/key-finding generation yet. |

Epic-level gap summary:

- EPIC 1 is partially complete: schema, quarantine behavior, and trace metric aggregation are usable; route validation and CI gate need stronger integration.
- EPIC 2 producer code is implemented and KML smoke validates manifest coverage/compliance for `global_gist` and `vision_read`.
- EPIC 3 linkage is now implemented locally through ledger, mapped evidence, grounding floor, and compact chain export.
- EPIC 4 has scheduler/data structures, a hard-skill follow-up handoff into the main loop, low-confidence terminal support, and trace-derived follow-up metrics; generic all-route follow-up strategies remain incomplete.
- EPIC 5 is wired into the prompt stack with strategies and CLI flags, but overflow tuning remains.
- EPIC 6 and EPIC 7 are implemented at the tool/storage/commit-channel level; policy gating and smoke coverage remain.
- EPIC 8 has the planned schema/export/audit pipeline.
- EPIC 9 has CLI flags, matrix runner, and a basic report generator; feature toggles and report richness remain partial.

## Current Constraints

- Do not read or paste raw KML logs, raw model responses, trace dumps, or full JSON artifacts into chat.
- Inspect failures by filtered error fingerprints or compact JSON aggregations only.
- Keep the three pre-existing local dirty docs untouched unless explicitly taking ownership:
  - `docs/2026-06-03-flow-context-management-design.zh.md`
  - `docs/2026-06-03-handoff-videomme-agent-loop.md`
  - `docs/report-design-proposal.zh.md`

## Multi-Agent Review Notes

- Read-only subagent review found three actionable issues:
  - `query_context` could satisfy visual citation gates through `VISUAL_EVIDENCE_TOOLS`.
  - `query_context` could still appear in compact ledger answer-facing visual text.
  - `scripts/run_ablation.py` did not map structured boolean `false` to paired `--disable-*` flags.
- All three were fixed before commit:
  - `has_non_navigation_visual_citation()` now only accepts `ANSWER_EVIDENCE_TOOLS`.
  - `compact_ledger_text()` separates `Context-Only Visual Hints (Not Answer Support)` and excludes context-only entries from the short-term working buffer.
  - Matrix boolean false values for paired flags now emit `--disable-*`; tests cover `enable_followup: false`.

## Next Actions

1. Tune EPIC 5 context compaction/budgets: current smoke has `context_budget_overflow_count=6`.
2. Improve EPIC 4 generic follow-up: implement scheduler-backed targets for non-hard-skill AnswerAgent `need_more_evidence`, and richer route strategies for temporal/needle/gist.
3. Wire behavioral ablation flags that are currently recorded but not active: `enable_query_context`, `enable_map_reflux`, and `enable_evidence_staging`.
4. Exercise EPIC 6/7 in a KML smoke or controlled fixture so `query_context_usage_rate` and `map_reflux_commit_count` are non-zero when enabled.
5. Improve answer quality on the 3 anchors: current smoke at `bec2039` has `accuracy=0.0`; 605-1/612-1 produce wrong finals and 611-2 exhausts 8 rounds.
6. Strengthen TASK-038 report generator with per-case comparison, graph/key-finding generation, and trajectory audit integration over a real matrix.
7. If rerunning VideoMME after `6c819c2`, include `--export-training` and verify summary trajectory paths exist; this path bug is already covered by local/KML tests but not by a full post-fix model smoke.
