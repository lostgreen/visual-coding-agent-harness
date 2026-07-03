# multi_agent_v0 Debug Breakpoints

Use launch config: VideoMME multi_agent_v0 debug: 611-2 evidence flow.
It runs one case, 611-2, and stops on entry. VS Code/code-server breakpoints are stored in UI state, so this file is the portable breakpoint checklist.

## Driver / Round Loop

- src/visual_coding_agent_harness/evals/videomme/runner.py:230
  - Why: workspace creation and strategy wiring.
  - Watch: run_id, workspace.root, strategy, budget.max_rounds.
- src/visual_coding_agent_harness/agents/multi/driver.py:27
  - Why: one full multi-agent round starts here.
  - Watch: round_number, option_payload, idle_streak.
- src/visual_coding_agent_harness/agents/multi/driver.py:51
  - Why: forced final / best-effort final path.
  - Watch: reason, best_effort, self.workspace.memory_entries().

## Reasoner

- src/visual_coding_agent_harness/agents/multi/reasoner.py:44
  - Why: Reasoner reads current shared state.
  - Watch: sub_goals, active, tested_options, untested_options.
- src/visual_coding_agent_harness/agents/multi/reasoner.py:57
  - Why: option sub-goal emission.
  - Watch: option_id, option_text, question.
- src/visual_coding_agent_harness/agents/multi/reasoner.py:80
  - Why: answer scoring from findings and memory.
  - Watch: findings, scored_options, self.workspace.memory_entries().
- src/visual_coding_agent_harness/agents/multi/reasoner.py:104
  - Why: final answer gate.
  - Watch: scored_options[0], citations, self.answer_result.

## Investigator / Scout / Verifier

- src/visual_coding_agent_harness/agents/multi/investigator.py:39
  - Why: claims the next open sub-goal.
  - Watch: sub_goal.sub_goal_id, sub_goal.constraint.option_id, sub_goal.constraint.claim.
- src/visual_coding_agent_harness/agents/multi/evidence_scout.py:34
  - Why: explore tool invocation.
  - Watch: sub_goal, self._explore_args(sub_goal).
- src/visual_coding_agent_harness/agents/multi/evidence_scout.py:72
  - Why: candidate window recorded to sidecar.
  - Watch: row, recorded, candidate_ids.
- src/visual_coding_agent_harness/agents/multi/evidence_verifier.py:30
  - Why: verify_window invocation.
  - Watch: candidate_key, self._verify_args(...), candidate.
- src/visual_coding_agent_harness/agents/multi/evidence_verifier.py:36
  - Why: verify output becomes finding status/cost.
  - Watch: outcome.raw_output, outcome.memory_ids, status, cost.

## Tool Runner / Observation Commit

- src/visual_coding_agent_harness/agents/multi/tool_runner.py:54
  - Why: tool args normalization.
  - Watch: tool_name, args, request.
- src/visual_coding_agent_harness/agents/multi/tool_runner.py:68
  - Why: actual registry tool execution.
  - Watch: request.tool, request.arguments, raw_output.keys().
- src/visual_coding_agent_harness/agents/multi/tool_runner.py:93
  - Why: observation commit begins.
  - Watch: observation.observation_id, raw_output.get("mode").
- src/visual_coding_agent_harness/agents/multi/tool_runner.py:117
  - Why: verify_window structured writes.
  - Watch: anchors, writes, memory_ids.

## Verify Cross-Match / Memory

- src/visual_coding_agent_harness/agents/workspace_agent.py:1946
  - Why: verify_window raw output is converted to memory writes.
  - Watch: results, answer_options, cross_match_parts.
- src/visual_coding_agent_harness/agents/workspace_agent.py:2028
  - Why: cross-option match picks synthesized support.
  - Watch: supports_option, matched_option, match_score, answer_options.
- src/visual_coding_agent_harness/workspace/state.py:328
  - Why: raw observation is written.
  - Watch: tool_name, observation_id, raw_payload.keys().
- src/visual_coding_agent_harness/workspace/state.py:703
  - Why: observation commits pinned anchors and memory.
  - Watch: observation_id, normalized_writes, memory_entries.
- src/visual_coding_agent_harness/workspace/state.py:420
  - Why: final memory entry persistence.
  - Watch: kind, supports_option, anchors, metadata.
- src/visual_coding_agent_harness/workspace/state.py:1569
  - Why: trace event stream.
  - Watch: event_type, payload.

## Files To Inspect After A Debug Run

- /m2v_intern/xuboshen/zgw/VideoAgent/VideoMME/runs/debug_multi_agent_v0_611/summary.json
- /m2v_intern/xuboshen/zgw/VideoAgent/VideoMME/runs/debug_multi_agent_v0_611/workspaces/runs/*_multi_agent_v0/trace.jsonl
- /m2v_intern/xuboshen/zgw/VideoAgent/VideoMME/runs/debug_multi_agent_v0_611/workspaces/runs/*_multi_agent_v0/multi_agent/sub_goals.jsonl
- /m2v_intern/xuboshen/zgw/VideoAgent/VideoMME/runs/debug_multi_agent_v0_611/workspaces/runs/*_multi_agent_v0/multi_agent/findings.jsonl
- /m2v_intern/xuboshen/zgw/VideoAgent/VideoMME/runs/debug_multi_agent_v0_611/workspaces/runs/*_multi_agent_v0/memory.jsonl
