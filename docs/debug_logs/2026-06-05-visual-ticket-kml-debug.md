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
- Current status: running at handoff draft time.
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

## Current Constraints

- Do not read or paste raw KML logs, raw model responses, trace dumps, or full JSON artifacts into chat.
- Inspect failures by filtered error fingerprints or compact JSON aggregations only.
- Keep the three pre-existing local dirty docs untouched unless explicitly taking ownership:
  - `docs/2026-06-03-flow-context-management-design.zh.md`
  - `docs/2026-06-03-handoff-videomme-agent-loop.md`
  - `docs/report-design-proposal.zh.md`

## Next Actions

1. Implement the visual tool manifest producer tickets so `frame_sets/manifests.jsonl` is populated by actual `global_gist`, `vision_read`, `zoom`, and segment-read tools.
2. Wire follow-up scheduling into the hard-skill loop so temporal/local cases can continue after structured `need_more_evidence`.
3. Add answer/evidence mapping gates for temporal relation sufficiency before expanding beyond the three anchor cases.
4. Re-run the same anchor set and then a 5-10 case temporal/local subset once tool manifests and follow-up are live.
