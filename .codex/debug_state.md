# VideoMME Direct Baseline Debug State

## Goal

Establish a trustworthy Gemini 512-frame direct baseline, compare it with the same 10-case Agent run, then improve the general Agent architecture rather than individual cases.

## Current Evidence

- Current branch: `codex/m3-slim-video-agent`.
- Unlabeled 512 independent frames + full timestamped ASR: 3/10 on `videomme_long_hard_rotate_v3`.
- Labeled 512 independent frames + full timestamped ASR: 5/10.
- Fresh current-code Agent result on the same group: 5/10.
- Direct-only correct: `648-3`, `702-1`; Agent-only correct: `645-3`, `698-3`; shared correct: `672-3`, `799-3`, `851-3`.
- The direct/Agent oracle union is 7/10, so the methods are complementary rather than one strictly dominating the other.
- All direct requests used `images_512`; no contact-sheet fallback.
- Case `672-3` contains the answer visually: the cat enters around the seated chair squat, matching gold B.
- Gemini returned internally inconsistent frame/time references for `672-3`, showing image/frame-map/ASR binding failure rather than missing sampled content.
- Two malformed JSON responses used `MM:SS.mmm` as an unquoted numeric value; both model choices were still wrong, so the parser bug did not change the 3/10 score.

## Latest Change

- Commit `b0808bf` burns `Fxxxx + timestamp` into every independent frame.
- Direct evidence times are aligned back to the authoritative frame manifest while preserving mismatched model-reported times.
- Parser repairs unquoted `MM:SS.mmm` timestamps.
- Verification: local full suite `168 passed`; KML focused suite `15 passed`.

## Current Run

- Labeled direct group: `/m2v_intern/xuboshen/zgw/VideoAgent/videomme_direct_512_labeled_v3`.
- Fresh Agent baseline: `/m2v_intern/xuboshen/zgw/VideoAgent/videomme_agent_current_v3`.

## Stale Evidence

- Earlier `672-3` response variants under `videomme_direct_512_v3` were overwritten by reruns. Use only the final group result plus downloaded diagnostic sheets.

## Next Actions

1. Commit and sync bounded Investigator evidence-frame replay to Reasoner, answer audit, and forced finalization.
2. Rerun the same 10-case Agent group in a fresh output root.
3. Compare per-case changes, not only aggregate accuracy.
4. Use the result to choose between navigation improvements, lexical ASR search, or deterministic aggregation next.
