# Handoff: Video Harness Runtime Iteration

Date: 2026-06-06
Branch: `codex/visual-harness-ticket-plan`

## Current Goal

Build a Claude Code-style long-video visual agent harness where skills,
tools, evidence state, budget, and final answers are managed by code instead
of prompt-only planner behavior.

Current diagnostic source:

- `docs/2026-06-05-harness-diagnostic-and-roadmap.md`

Do not load raw logs by default. Use summaries, compact trace fingerprints,
selected observation ids, and this handoff as the current state.

## Current Evidence

Latest local verification:

```text
PYTHONPATH=src python -m pytest -q
Result: 254 passed in 0.83s
```

This verification was run after the evidence-summary and interpreter dataflow
changes below.

## Implemented In This Iteration

- Added `EvidenceWorkspace.evidence_status_summary()` to expose compact
  answer-facing evidence state: option coverage, strong/weak evidence counts,
  visual citation presence, duplicate claim count, row count, and hypothesis
  gaps.
- Injected that evidence status summary into the replanning prompt so the
  planner sees `what is covered`, `what is missing`, and `what looks
  duplicated` before choosing the next tool.
- Updated `IterativeVisualAgent.run()` to compute the summary from the current
  workspace every round.
- Extended `ProgramInterpreter` so a tool observation can update runtime
  foreach slots, starting with `candidates` from `raw_output.candidates` or
  `raw_output.regions`, and `segments` from `raw_output.segments`.
- Preserved object-valued exact templates such as `{candidate}`, allowing a
  compiled skill step like `ground_question -> vision_read foreach=candidates`
  to pass a candidate window dict directly to the next tool.

## Files Changed

Source:

- `src/visual_coding_agent_harness/workspace.py`
- `src/visual_coding_agent_harness/agents/prompt_stack.py`
- `src/visual_coding_agent_harness/agents/iterative_agent.py`
- `src/visual_coding_agent_harness/interpreter.py`

Tests:

- `tests/test_workspace.py`
- `tests/test_prompt_stack_and_skill_runtime.py`
- `tests/test_iterative_agent.py`
- `tests/test_v4_foundation.py`

Docs:

- `docs/2026-06-06-handoff-video-harness-runtime.md`

## Current Interpretation

The harness is no longer blocked on basic evidence table plumbing or low
confidence fallback. Several Phase E/F pieces from the 2026-06-05 roadmap are
already present in code, including global gist deterministic handling,
route-whitelist enforcement in planner normalization, and low-confidence
finalization gates.

The most important remaining gap is that hard-skill direct execution still
bypasses some runtime policy checks. Normal planner programs go through route
validation and tool budget gates, but hard runtime paths call
`ProgramInterpreter.run()` directly.

## Stale Evidence

Older top-level handoff details from 2026-06-03 and 2026-06-04 are stale for
current implementation decisions. They describe earlier failures such as
budget-only incompletion, unsupported finals, quoted-option JSON parse errors,
and pre-summary prompt stack behavior.

Remote KML run artifacts from 2026-06-05 are still useful as historical
baselines, but they predate this local 2026-06-06 runtime iteration. Rerun the
3-case smoke after pushing this commit before treating KML metrics as current.

## Next Actions

1. Add a hard-runtime execution wrapper that applies `_route_violation()` and
   `_tool_budget_available()` before direct `ProgramInterpreter.run()` calls.
2. Cover non-timeline hard runtime with a focused test for
   `ground_question -> vision_read` execution under route and budget policy.
3. Decide whether timeline hard runtime should obey per-round budget strictly
   or explicitly record a bounded `timeline_max_segments` policy bypass.
4. Add dynamic `events`/`segments` slot resolution for timeline skills after
   the hard-runtime budget wrapper is in place.
5. Rerun the VideoMME 3-case KML smoke and update this handoff with compact
   metrics only.
