# Skill-First Refactor v2 Review-Optimized Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor VideoMME agent-v2 routing/evidence flow so narration timeline questions use structured option/relation evidence instead of repeated visual checks or ASR mention-time ordering.

**Architecture:** The planner remains the final-answer owner and AnswerAgent is only a verifier. The framework will expose immutable target/option protocol objects, strict `target_refs` normalization, explicit transcript binding, markdown skill playbooks with structured metadata, and loop recovery that proposes rather than silently repeats route repairs.

**Tech Stack:** Python dataclasses/enums, existing `ToolRegistry` and `EvidenceWorkspace`, pytest, KML VideoMME runner.

---

## Current Evidence and Decisions

Current branch baseline before this plan: `codex/agent-ownership-context-redesign` after commit `924dda4`.

Current stale-but-useful failure evidence:
- Old KML run root: `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_v2_3demo_ded2fe3_20260609_111252`
- `611-2` failed because `read_timeline_sorted` received unsupported `segment_id`.
- `612-1` was not a parser bug. Planner produced final `B`, but old AnswerAgent takeover logic blocked it, so runner received partial summary and extracted no choice.

Current fixed baseline:
- Planner owns final answer.
- AnswerAgent is verifier.
- Unsupported tool args are stripped.
- Full local test suite was passing at `924dda4`.

Review-driven design decisions for this plan:
- Do not use ASR mention timestamps as event order. Store `mention_timestamp_sec` separately from option sequence and relation evidence.
- Add `OptionSpec`, `ClaimRelation`, and `RelationBinding`. B/C-style options can share targets but differ by sequence and relations.
- `TargetRegistry` belongs under `contracts/target_registry.py`, not `workspace/target_registry.py`, because the repo currently has `workspace.py` as a module.
- `TargetSpec` is immutable and stores only `modality_hint`; effective modality is written on each `EvidenceBinding`.
- `TranscriptEvidenceBinder` is explicit and conservative. Lexical hits alone must not create `status="supported"` bindings.
- `target_refs` is for known `T<n>` ids only. Legacy `targets` is for free text only.
- Recovery programs must come from structured skill metadata, not from parsing markdown prose.
- Repair counters reset only on relevant supported bindings, not globally on unrelated evidence.

## Phase Order

1. **Phase A, Protocol Foundation:** contracts, registry, option/relation model, program signature.
2. **Phase B, Tool Arg Protocol:** strict `target_refs` normalization, registry prompt rendering, no silent unknown-id drops.
3. **Phase C, Evidence Protocol:** evidence ids, transcript binder, conservative binding status, `promote_answer_evidence`.
4. **Phase D, Skill Library:** narration/visual/mixed/factual playbooks and classifier hard negatives.
5. **Phase E, Recovery and Final Gate:** generic repeat guard, skill-specific recovery metadata, no-progress warning, evidence-id final gate.
6. **Phase F, KML Regression:** expanded VideoMME matrix, then 3-demo sanity run.

No upper phase starts before its lower dependencies are merged locally and focused tests pass.

## File Map

Create:
- `src/visual_coding_agent_harness/contracts/claim_modality.py`
- `src/visual_coding_agent_harness/contracts/evidence_binding.py`
- `src/visual_coding_agent_harness/contracts/target_registry.py`
- `src/visual_coding_agent_harness/contracts/tool_output.py`
- `src/visual_coding_agent_harness/agents/transcript_binder.py`
- `src/visual_coding_agent_harness/agents/skills/playbooks/narration_timeline_qa.md`
- `src/visual_coding_agent_harness/agents/skills/playbooks/visual_timeline_qa.md`
- `src/visual_coding_agent_harness/agents/skills/playbooks/mixed_asr_visual_qa.md`
- `src/visual_coding_agent_harness/agents/skills/playbooks/grounded_factual_qa.md`
- `tests/test_target_registry.py`
- `tests/test_evidence_binding.py`
- `tests/test_transcript_binder.py`

Modify:
- `src/visual_coding_agent_harness/workspace.py`
- `src/visual_coding_agent_harness/interpreter.py`
- `src/visual_coding_agent_harness/tools/navigation.py`
- `src/visual_coding_agent_harness/agents/question_policy.py`
- `src/visual_coding_agent_harness/agents/prompt_stack.py`
- `src/visual_coding_agent_harness/agents/skills/specs.py`
- `src/visual_coding_agent_harness/agents/iterative_agent.py`
- `tests/test_iterative_agent.py`
- `tests/test_route_validator.py`
- `tests/test_video_navigation.py`
- `tests/test_prompt_stack_and_skill_runtime.py`
- `tests/test_question_policy.py`
- `tests/test_workspace.py`

## Subagent Ownership

Subagents must not revert other work. Write scopes are intentionally disjoint.

- Worker A owns contracts and registry tests:
  - `src/visual_coding_agent_harness/contracts/*.py`
  - `tests/test_target_registry.py`
  - `tests/test_evidence_binding.py`
- Worker B owns arg normalization and prompt schema:
  - `src/visual_coding_agent_harness/agents/iterative_agent.py`
  - `src/visual_coding_agent_harness/agents/prompt_stack.py`
  - normalization tests in `tests/test_route_validator.py` or `tests/test_iterative_agent.py`
- Worker C, to dispatch after A/B land, owns transcript binder and navigation promotion:
  - `src/visual_coding_agent_harness/agents/transcript_binder.py`
  - `src/visual_coding_agent_harness/tools/navigation.py`
  - `tests/test_transcript_binder.py`
  - `tests/test_video_navigation.py`
- Worker D, to dispatch after C lands, owns skill playbooks/classifier:
  - `src/visual_coding_agent_harness/agents/skills/specs.py`
  - `src/visual_coding_agent_harness/agents/skills/playbooks/*.md`
  - `src/visual_coding_agent_harness/agents/question_policy.py`
  - `tests/test_question_policy.py`
  - `tests/test_prompt_stack_and_skill_runtime.py`

## Task A1: Protocol Contracts and Target Registry

**Files:**
- Create: `src/visual_coding_agent_harness/contracts/claim_modality.py`
- Create: `src/visual_coding_agent_harness/contracts/evidence_binding.py`
- Create: `src/visual_coding_agent_harness/contracts/target_registry.py`
- Create or modify: `src/visual_coding_agent_harness/contracts/__init__.py`
- Create: `tests/test_target_registry.py`
- Create: `tests/test_evidence_binding.py`

- [ ] **Step 1: Write failing contract tests**

Add tests that assert:
- known `T1` resolves;
- unknown `T99` raises or returns an explicit error path, never silent fallback;
- option B and C can share a target set but keep different `target_sequence`;
- duplicate canonical text resolves one-to-many or is not reverse-resolved;
- registry version is stable for a run;
- `EvidenceBinding` round-trips through `dataclasses.asdict`;
- `claim_modality` is a `ClaimModality` enum value or stable enum string.

Run:

```bash
PYTHONPATH=src:. pytest -q tests/test_target_registry.py tests/test_evidence_binding.py
```

Expected before implementation: FAIL due to missing modules.

- [ ] **Step 2: Implement minimal contracts**

Implementation sketch:

```python
class ClaimModality(str, Enum):
    NARRATED_FACT = "narrated_fact"
    VISUAL_FACT = "visual_fact"
    OCR_FACT = "ocr_fact"
    MIXED = "mixed"
    UNKNOWN = "unknown"
```

Use frozen dataclasses:
- `TargetSpec`
- `OptionSpec`
- `ClaimRelation`
- `EvidenceBinding`
- `RelationBinding`

`EvidenceBinding` must use `mention_timestamp_sec`, not `timestamp_sec`.

- [ ] **Step 3: Implement `TargetRegistry`**

Required behavior:
- constructor accepts targets and options;
- `targets_by_id`, `options_by_id`, and `target_to_options` are immutable mapping-style attributes or exposed read-only copies;
- `is_target_ref(value)` matches `^T\d+$`;
- `known_target_ref("T1")` is true only for registered ids;
- `resolve_target_ref("T1")` returns `TargetSpec`;
- unknown `T<n>` is an explicit failure;
- canonical reverse lookup is one-to-many if implemented.

- [ ] **Step 4: Run focused tests**

```bash
PYTHONPATH=src:. pytest -q tests/test_target_registry.py tests/test_evidence_binding.py
```

Expected: PASS.

## Task A2: Program Signature Ignores Generated Fields

**Files:**
- Modify: `src/visual_coding_agent_harness/agents/iterative_agent.py`
- Modify: `tests/test_iterative_agent.py`

- [ ] **Step 1: Add failing test**

Add `test_program_signature_ignores_assign_names_and_trace_ids`:
- two programs use same tools and args;
- generated fields differ: `assign`, `trace_id`, `observation_id`;
- `_program_signature` must return equal signatures.

Run:

```bash
PYTHONPATH=src:. pytest -q tests/test_iterative_agent.py::IterativeAgentTest::test_program_signature_ignores_assign_names_and_trace_ids
```

Expected before implementation: FAIL.

- [ ] **Step 2: Normalize signature input**

Before JSON serialization:
- drop `assign`, `trace_id`, `observation_id`;
- recursively sort mapping keys;
- keep `tool` and normalized `args`.

- [ ] **Step 3: Run focused test**

Expected: PASS.

## Task B1: Strict `target_refs` Normalization

**Files:**
- Modify: `src/visual_coding_agent_harness/agents/iterative_agent.py`
- Modify: `src/visual_coding_agent_harness/agents/prompt_stack.py`
- Modify: `tests/test_route_validator.py` or `tests/test_iterative_agent.py`

- [ ] **Step 1: Add failing tests**

Required tests:
- known `T1` in legacy `targets` rewrites to `target_refs=["T1"]` and records a compatibility note;
- unknown `T99` in `target_refs` rejects the whole call;
- free text in `target_refs` rejects the whole call;
- natural-language legacy `targets=["humble background"]` is preserved;
- unknown `T99` in legacy `targets` rejects the whole call;
- `read_timeline_sorted(segment_id="seg_0001")` still strips unsupported args and becomes `read_timeline_sorted()`.

Run:

```bash
PYTHONPATH=src:. pytest -q tests/test_route_validator.py tests/test_iterative_agent.py -k "target_ref or read_timeline_sorted"
```

Expected before implementation: at least the new tests fail.

- [ ] **Step 2: Implement normalization helper**

Add a helper near `_normalize_program`:
- inspect `workspace.target_registry` if present;
- accept duck-typed methods such as `known_target_ref` and `resolve_target_ref`;
- reject the step by returning `None` or a skip marker when protocol is invalid;
- append `NormalizationNote` and trace event with reason:
  - `unknown_target_ref`
  - `free_text_in_target_refs`
  - `legacy_targets_rewritten_to_target_refs`

Do not silently drop unknown ids.

- [ ] **Step 3: Update prompt schema**

Change tool schema guidance to include:

```text
target_refs: list contains only TargetRegistry ids such as T1/T2.
targets: legacy free text only.
Known T<n> in targets may be rewritten for compatibility; unknown T<n> rejects the whole call.
Never invent target_refs.
```

Acceptance wording:
- Correct: `0 times T<n> appears in legacy targets`.
- Incorrect: `0 times T<n> appears in any tool args`.

- [ ] **Step 4: Run focused tests**

Expected: PASS.

## Task B2: Registry Construction and Prompt Rendering

**Files:**
- Modify: `src/visual_coding_agent_harness/agents/iterative_agent.py`
- Modify: `src/visual_coding_agent_harness/agents/prompt_stack.py`
- Modify: `tests/test_iterative_agent.py`
- Modify: `tests/test_prompt_stack_and_skill_runtime.py`

- [ ] **Step 1: Add failing tests**

Tests:
- `_initialize_run` or equivalent run setup creates a stable `workspace.target_registry` from `extract_option_target_atom_map(raw_question, include_synonyms=False)`;
- prompt includes a compact TargetRegistry block with option sequences;
- repeated target canonical text does not overwrite another target;
- registry is not rebuilt mid-run.

- [ ] **Step 2: Build registry from option atoms**

For each option:
- preserve option letter;
- preserve atom order as `OptionSpec.target_sequence`;
- de-duplicate canonical target text into stable target ids;
- generate `ClaimRelation(kind="before")` between adjacent targets in the option sequence for timeline routes.

Do not infer event order from ASR mention timestamps.

- [ ] **Step 3: Render registry to planner prompt**

Render compactly:

```text
# TargetRegistry v1
T1: humble background
T2: upper class
T3: farmhouse
Option B sequence: T1 -> T2 -> T3
```

Cap aliases and long text.

## Task C1: Transcript Evidence Binder

**Files:**
- Create: `src/visual_coding_agent_harness/agents/transcript_binder.py`
- Create: `tests/test_transcript_binder.py`

- [ ] **Step 1: Add conservative binder tests**

Tests:
- lexical collision `"the upper-class man cruelly mimics the beggar"` does not create supported `(Goya, social_transition)`;
- negation `"He never entered the upper class."` is not supported;
- reverse temporal `"He left the farmhouse before becoming successful."` does not support `successful -> farmhouse`;
- artwork subject `"The painting depicts an isolated man."` does not support `(subject=Goya, isolated)`;
- direct snippet `"Goya was a man from a humble background who rose through the ranks..."` supports `humble background -> upper class` if target subjects/relations match.

- [ ] **Step 2: Implement binder with explicit conservative features**

`TranscriptEvidenceBinder.bind(...)` returns bindings with feature fields or compact diagnostics:
- `subject_match`
- `predicate_match`
- `negated`
- `experiencer_match`
- `artwork_context`
- `relation_direction`
- `confidence`

Status rule:
- `supported` only when subject/predicate match and not negated;
- `rejected` only for explicit contradiction;
- otherwise `ambiguous`.

No lexical hit may become supported without binder approval.

## Task C2: Evidence IDs and `promote_answer_evidence`

**Files:**
- Modify: `src/visual_coding_agent_harness/tools/navigation.py`
- Modify: `src/visual_coding_agent_harness/workspace.py`
- Modify: `src/visual_coding_agent_harness/interpreter.py`
- Modify: `tests/test_video_navigation.py`
- Modify: `tests/test_workspace.py`

- [ ] **Step 1: Add failing tests**

Tests:
- `read_segment_detail(promote_answer_evidence=True, target_refs=[...])` creates answer evidence rows only through registry and binder;
- `promote_answer_evidence=False` does not create promoted rows;
- output rows include `evidence_id` and `evidence_binding`;
- navigation summary includes a short snippet line with evidence id;
- recent tool outputs preserve `snippet`;
- `group_by_option=True` on `target_coverage` returns per-option coverage matrix.

- [ ] **Step 2: Update tool signatures**

Target signatures:

```python
target_coverage(
    targets: Sequence[str] = (),
    target_refs: Sequence[str] = (),
    top_k: int = 3,
    modalities: Sequence[str] = (),
    group_by_option: bool = False,
)

read_segment_detail(
    segment_id: str,
    targets: Sequence[str] = (),
    target_refs: Sequence[str] = (),
    promote_answer_evidence: bool = False,
)
```

Keep `option_targets` only as an internal or advanced override path; do not auto-inject it unless `promote_answer_evidence=True`.

- [ ] **Step 3: Preserve legacy compatibility**

Legacy `targets` still works for free text.
Known target refs passed through legacy `targets` should already have been normalized in Task B1.

## Task D1: Skill Playbook Loader and Structured Metadata

**Files:**
- Modify: `src/visual_coding_agent_harness/agents/skills/specs.py`
- Create: `src/visual_coding_agent_harness/agents/skills/playbooks/*.md`
- Modify: `tests/test_prompt_stack_and_skill_runtime.py`

- [ ] **Step 1: Add failing playbook tests**

Tests:
- `SkillSpec.from_playbook()` or equivalent loads markdown body and front matter;
- structured `recovery_rules` are accessible without parsing prose;
- prompt renders markdown procedure/sufficiency;
- existing built-in skill registry still lists current skills.

- [ ] **Step 2: Use front matter for framework metadata**

Example:

```yaml
---
name: narration_timeline_qa
version: 1
default_claim_modality: narrated_fact
recovery_rules:
  repeated_visual_verification:
    tool: read_segment_detail
    args:
      promote_answer_evidence: true
      target_refs: "$active_target_refs"
---
```

Markdown body is for planner guidance only.

## Task D2: Narration, Visual, Mixed, and Factual Skills

**Files:**
- Modify: `src/visual_coding_agent_harness/agents/question_policy.py`
- Modify: `src/visual_coding_agent_harness/agents/skills/specs.py`
- Create: playbook markdown files
- Modify: `tests/test_question_policy.py`

- [ ] **Step 1: Add classifier tests**

Positive narration examples:
- `How was his life journey according to the video?`
- `According to the narrator, why did he leave?`
- `What does the video tell us about her early life?`

Narration hard negatives:
- `How did the man open the door?`
- `How was the painting positioned?`
- `How did the ball move after impact?`
- `What does she pick up next?`

- [ ] **Step 2: Implement classifier**

Use:
- explicit narration markers; or
- biographical markers plus ASR availability hint.

Do not classify generic `how did/how was` visual questions as narration.

- [ ] **Step 3: Split timeline skill**

Create:
- `narration_timeline_qa@v1`: default modality `narrated_fact`, no visual verification gate.
- `visual_timeline_qa@v1`: default modality `visual_fact`, keeps visual gate.
- `mixed_asr_visual_qa@v1`: ordering from ASR, object identity from visual.
- `grounded_factual_qa@v1`: protocolized existing logic.

## Task E1: Route Repair and No-Progress Guard

**Files:**
- Modify: `src/visual_coding_agent_harness/agents/iterative_agent.py`
- Modify: `tests/test_iterative_agent.py`

- [ ] **Step 1: Add failing tests**

Tests:
- first repeated route repair records explicit note;
- second same `(reason, segment_id, normalized_target_keys)` emits recovery proposal and does not execute recommended program;
- third same key hard-stops with `status=route_repair_exhausted`;
- repair count differs for same segment with different target set;
- repair count resets only when a supported binding overlaps segment, target refs, or recovery family;
- no-progress warning appears after three rounds without new supported binding.

- [ ] **Step 2: Implement generic and skill-specific phases**

Phase E generic guard must not require skill playbooks.
Skill-specific recovery uses structured `recovery_rules` only after Task D.

## Task E2: Final Evidence Contract

**Files:**
- Modify: `src/visual_coding_agent_harness/agents/iterative_agent.py`
- Modify: trajectory/export parser if needed
- Modify: tests around final gating

- [ ] **Step 1: Add failing tests**

Tests:
- final JSON can include `evidence_ids` separately from legacy `citations`;
- final gate requires at least one supported `EvidenceBinding` evidence id for answer-grade narration timeline finals;
- observation ids in `citations` remain supported for backwards compatibility but do not satisfy the v2 evidence-id gate alone.

- [ ] **Step 2: Implement contract migration**

Preferred final shape:

```json
{
  "status": "final",
  "skill": "narration_timeline_qa",
  "answer": "B",
  "citations": ["obs_0006"],
  "evidence_ids": ["ev_obs_0006_asr_0"],
  "confidence": 0.82
}
```

Do not overload one field with both id types unless all parser/export/evaluator sites are updated.

## Phase F: Verification and KML

- [ ] **Step 1: Local focused tests per task**

Run each task's focused tests before integrating the next layer.

- [ ] **Step 2: Local broad tests**

```bash
PYTHONPATH=src:. pytest -q tests/test_iterative_agent.py tests/test_route_validator.py tests/test_video_navigation.py tests/test_question_policy.py tests/test_prompt_stack_and_skill_runtime.py tests/test_workspace.py tests/test_target_registry.py tests/test_evidence_binding.py tests/test_transcript_binder.py
```

- [ ] **Step 3: Full local suite**

```bash
PYTHONPATH=src:. pytest -q
```

- [ ] **Step 4: KML 3-demo sanity**

After pushing to GitHub and syncing KML:
- `605-1` remains correct.
- `611-2` remains correct and uses visual timeline route.
- `612-1` selected option is `B`, correct, rounds <= 6.

Expected KML evidence for `612-1`:
- at least one `read_segment_detail(... promote_answer_evidence=True ...)`;
- at least one supported `EvidenceBinding` with `claim_modality=narrated_fact`;
- zero `T<n>` values in legacy `targets`;
- unknown `T<n>` hard rejects, never silently drops;
- no visual gate requirement for `narration_timeline_qa`.

## First Implementation Batch

Start in parallel:
- Worker A: Task A1.
- Worker B: Task B1 with duck-typed registry support.
- Main agent: maintain this plan, inspect mapper output, then integrate A/B.

Do not start Task C until A/B tests pass locally.

## Implementation Update 2026-06-09

Current evidence:
- Local branch has uncommitted implementation for Phases A-D and part of Phase F local verification.
- Full local regression is current: `PYTHONPATH=src:. pytest -q` -> `435 passed in 1.42s`.
- Broad focused matrix is current: `PYTHONPATH=src:. pytest -q tests/test_iterative_agent.py tests/test_route_validator.py tests/test_video_navigation.py tests/test_question_policy.py tests/test_prompt_stack_and_skill_runtime.py tests/test_workspace.py tests/test_target_registry.py tests/test_evidence_binding.py tests/test_transcript_binder.py tests/test_caption_qa_tools.py` -> `207 passed in 0.78s`.
- `git diff --check` is clean.

Completed:
- Task A1: contracts and registry.
  - Added `ClaimModality`, `TargetSpec`, `OptionSpec`, `ClaimRelation`, `EvidenceBinding`, `RelationBinding`, and immutable `TargetRegistry`.
  - Registry enforces `T<n>` ids and does not canonical-text resolve silently.
- Task A2: repeated-program signature ignores generated fields.
  - `_program_signature` removes generated `assign`, `trace_id`, and `observation_id` before comparing programs.
- Task B1: strict `target_refs` normalization.
  - Known legacy `targets=["T1"]` rewrites to `target_refs`.
  - Unknown/free-text `target_refs` hard reject.
  - Unknown legacy `T<n>` hard rejects the whole call.
  - Natural-language `targets` remains supported.
- Task C1/C2: conservative transcript binder and navigation promotion.
  - Added `TranscriptEvidenceBinder`.
  - `read_segment_detail(..., promote_answer_evidence=True, target_refs=[...])` emits promoted rows with `evidence_id` and `evidence_binding`.
  - `promote_answer_evidence=False` does not promote target-ref rows.
  - `target_coverage(..., target_refs=[...], group_by_option=True)` returns option sequence coverage.
  - `locate_targets_in_segment(..., target_refs=[...])` and `verify_segment_anchors(..., target_refs=[...])` now resolve refs instead of losing them during unsupported-arg stripping.
- Task D1/D2: playbook loader, playbooks, and narration classifier.
  - Added markdown playbooks with structured front matter and no external YAML dependency.
  - `recovery_rules` are read from front matter, not markdown prose.
  - Narration classifier handles requested positives and visual hard negatives.
  - New skills registered while keeping legacy `timeline_ordering` compatible.

Reviewer fixes applied after subagent implementation:
- `locate_targets_in_segment` prompt/schema mismatch fixed by adding real `target_refs` support.
- `verify_segment_anchors` prompt/schema mismatch fixed by adding real `target_refs` support.
- Transcript relation binding no longer treats phrase order alone as `before`; it requires explicit transition/temporal cue and rejects reverse `only after` cases.
- Painting/artwork subject collision guard now catches `Goya painted an isolated man` style snippets as not supported for `(subject=Goya, isolated)`.

Still pending:
- Task E1 route-repair/no-progress guard.
- Task E2 final `evidence_ids` contract migration.
- Commit/push.
- KML sync and VideoMME three-demo run.

Stale evidence:
- KML rerun at commit `924dda4` does not include this skill-first refactor.
- Previous full-suite count `411 passed` is stale; current local count is `435 passed`.

## Completion Update 2026-06-09

Current evidence:
- Local branch: `codex/agent-ownership-context-redesign`.
- Focused matrix: `PYTHONPATH=src:. pytest -q tests/test_iterative_agent.py tests/test_route_validator.py tests/test_video_navigation.py tests/test_question_policy.py tests/test_prompt_stack_and_skill_runtime.py tests/test_workspace.py tests/test_target_registry.py tests/test_evidence_binding.py tests/test_transcript_binder.py tests/test_caption_qa_tools.py` -> `213 passed in 0.82s`.
- Full local suite: `PYTHONPATH=src:. pytest -q` -> `441 passed in 1.45s`.
- `git diff --check` is clean.

Completed since the previous update:
- Task E1: route repair and no-progress guard.
  - First repair of the same `(reason, segment_id, normalized_target_keys)` is applied and traced.
  - Second same repair emits `route_repair_recovery_proposed` with a concrete recovery program and does not execute the repaired tool again.
  - Third same repair hard-stops with `status=route_repair_exhausted`.
  - Repair counts are target-key specific and reset only when a supported `EvidenceBinding` overlaps the repair key.
  - Three rounds without new supported bindings emit `iterative_no_progress_warning`.
- Task E2: final `evidence_ids` contract.
  - Planner final JSON now accepts `evidence_ids` separately from legacy `citations`.
  - `IterativeRunResult.output` preserves both fields.
  - `narration_timeline_qa` planner finals require an explicit supported `EvidenceBinding` evidence id; observation ids alone no longer satisfy that v2 gate.
  - AnswerAgent auto-final is blocked after this v2 gate blocks a planner final, preserving the planner-owner / AnswerAgent-verifier contract.
  - `EvidenceRowV2` preserves `evidence_binding` through workspace normalization.

Current stale evidence:
- KML run `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_v2_3demo_ded2fe3_20260609_111252` predates this completion update.
- Previous local counts `435 passed` and `207 passed` are stale after E1/E2.

Next actions:
1. Commit and push this branch to GitHub.
2. Sync KML repo `/home/xuboshen/zgw/visual-coding-agent-harness` to the pushed commit using the required proxy.
3. Start VideoMME cases `605-1,611-2,612-1` as a detached KML job and record run root, log path, pid path, and expected summary path.
