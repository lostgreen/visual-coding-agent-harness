# 2026-06-09 Skill-First Refactor Handoff

## Current Goal

Complete the skill-first VideoMME agent-v2 refactor, push branch `codex/agent-ownership-context-redesign`, sync KML, and start the three-demo VideoMME run for cases `605-1,611-2,612-1`.

## Current Evidence

- Local focused matrix: `213 passed in 0.82s`.
- Local full suite: `441 passed in 1.45s`.
- `git diff --check`: clean.
- Plan update: `docs/superpowers/plans/2026-06-09-skill-first-refactor-v2-review-optimized.md`.

## Important Decisions

- Planner remains final-answer owner.
- AnswerAgent is a verifier and must not bypass planner final gates.
- `target_refs` is only for known `T<n>` registry ids; free text stays in legacy `targets`.
- Narration timeline finals require explicit supported `EvidenceBinding` ids in `evidence_ids`; observation ids alone remain legacy citations but do not satisfy the v2 narration gate.
- ASR mention order is not event order unless explicit relation evidence supports it.

## Files Changed

- Agent loop and final gates: `src/visual_coding_agent_harness/agents/iterative_agent.py`.
- Evidence row preservation: `src/visual_coding_agent_harness/schemas.py`, `src/visual_coding_agent_harness/workspace.py`.
- Target/evidence contracts: `src/visual_coding_agent_harness/contracts/`.
- Transcript binder and navigation promotion: `src/visual_coding_agent_harness/agents/transcript_binder.py`, `src/visual_coding_agent_harness/tools/navigation.py`, `src/visual_coding_agent_harness/tools/inspector.py`.
- Skill playbooks/classifier: `src/visual_coding_agent_harness/agents/skills/specs.py`, `src/visual_coding_agent_harness/agents/skills/playbooks/`, `src/visual_coding_agent_harness/agents/question_policy.py`.
- Tests: `tests/test_iterative_agent.py`, `tests/test_route_validator.py`, `tests/test_video_navigation.py`, `tests/test_prompt_stack_and_skill_runtime.py`, `tests/test_question_policy.py`, `tests/test_caption_qa_tools.py`, `tests/test_target_registry.py`, `tests/test_evidence_binding.py`, `tests/test_transcript_binder.py`.

## Stale Evidence

- KML run `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_v2_3demo_ded2fe3_20260609_111252` predates this refactor completion.
- Older local test counts `435 passed`, `411 passed`, and `207 passed` are stale.

## Next Actions

1. Commit all local changes.
2. Push branch `codex/agent-ownership-context-redesign` to GitHub.
3. On KML, export the required proxy and fast-forward `/home/xuboshen/zgw/visual-coding-agent-harness`.
4. Start a detached VideoMME run for cases `605-1,611-2,612-1`.
5. Return only compact KML artifacts: run root, log path, pid path, pid, and expected summary path.
