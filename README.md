# VCAH

VCAH is a long-video QA harness built around virtual timelines. It indexes source
videos without concatenating them, lets a Reasoner request targeted observations,
and keeps every observation traceable to the exact source material inspected.

The active architecture has one semantic owner: the Reasoner. The framework is a
mechanical custodian. It stores observations, validates references, tracks temporal
coverage, and applies explicit Working Document edits. It does not decide whether a
claim is true, qualify events, score answer options, audit an answer, or replace the
Reasoner's choice.

## Authority Boundary

The framework performs only mechanically decidable work:

1. Compute a prompt-independent `attempt_id` from source video identity, frame
   times or frame references, sampling density, and modality.
2. Append observation interpretations to an immutable log without filtering or
   rewriting the Investigator's raw output.
3. Enforce reference integrity between Working Document claims, observation
   attempts, and derived claims.
4. Report inspected time ranges and sampling density as a coverage ledger.
5. Create the question premise as a framework-owned source record.

All interpretation stays with the Reasoner: claim wording, confidence, conflicts,
supersession, remaining uncertainty, investigation sufficiency, and the final answer.

## Architecture

```text
VirtualVideoWorkspace
  -> segment and beat overview
  -> Reasoner reads Rendered View
       -> caption/ASR locator search / visual window / same-frame arbitration
       -> edit Working Document
       -> read raw observations
       -> answer
  -> framework appends Observation Log and validates Working Document operations
  -> repeat until the Reasoner answers or the mechanical budget ends
```

### Observation Log

`ObservationAttempt` records the inspected material and the Investigator's exact raw
response. Multiple prompts over the same frames share one `attempt_id` and produce
separate interpretations under that identity. This exposes repeated or conflicting
reads without treating them as independent visual sources.

The Investigator has a deliberately small contract. It reports a free description,
timestamped observations, optional locally named entities and events, and explicit
uncertainties. It does not emit condition results, qualification status, option
predicates, or answer verdicts.

### Working Document

The persistent document uses three general primitives:

- `Claim`: premise, observation, derived, or hypothesis; active, contested,
  superseded, or retracted.
- `Entity`: a Reasoner-maintained identity and aliases.
- `IntervalNote`: a time range linked to claims.

Unresolved questions remain explicit as hypothesis or contested claims, so every
persistent semantic statement uses the same revision and provenance rules.

The Reasoner edits it with six operations: `add_claim`, `supersede`, `set_status`,
`link_conflict`, `note_interval`, and `update_entity`. Operations are transactional.
An observation claim must cite a known `attempt_id`; a derived claim must reference
existing parent claims. The framework checks those foreign keys but never evaluates
the claim text.

### Rendered View

Each Reasoner turn contains the question and options, the compact Working Document,
the observation catalog, the coverage ledger, and bounded previews requested on
the prior turn. Full raw output remains addressable through the immutable
observation-log pointer rather than being copied into every prompt.

### Same-Frame Arbitration

When interpretations conflict, the Reasoner can request
`inspection_mode=arbitrate_observation` with an `arbitration_attempt_id`. The
framework verifies the material identity and resends exactly those stored frames with
the competing raw interpretations. The new output is another interpretation under
the same attempt, not an extra vote.

## Artifacts

Each workspace writes:

```text
case.json                         question, options, and target metadata
timeline.json                     virtual-to-source segment mapping
beat_index.json                   navigational beat index
observation_log.jsonl             immutable raw observation interpretations
working_document.json             current Reasoner-owned document
workspace_ops.jsonl               accepted/rejected edit history
exploration_ledger.jsonl          mechanical inspection visits
evidence.jsonl                    compatibility pointers to observed material
interactions.jsonl                model prompts, raw responses, and usage metadata
run_summary.json                  answer, reference status, cost, and compact trace
```

`reference_valid` means only that the selected answer names existing supporting
claims and that their claim and attempt references have no dangling IDs. It is not a
semantic correctness verdict. The framework never changes one Reasoner option into
another.

## Quick Start

Requirements:

- Python 3.9+
- `ffmpeg` and `ffprobe`
- an OpenAI-compatible text Reasoner endpoint
- an OpenAI-compatible multimodal Investigator endpoint

```bash
python -m pip install -e '.[dev,interactive]'
PYTHONPATH=src:. pytest -q
```

Build and index a Video-MME workspace:

```bash
python main.py vv-build-videomme \
  --dataset-root /path/to/Video-MME/snapshot \
  --out-dir runs/videomme-smoke \
  --seed 20260707

python main.py vv-index \
  --workspace runs/videomme-smoke/477-2 \
  --low-fps 0.1 \
  --beat-sec 60
```

Build the MM-Lifelong Day subset as one shared virtual timeline plus lightweight
free-form cases. The builder probes the source clips in filename order, records
any clue-interval normalization in `validation.json`, and never creates a merged
video:

```bash
python main.py vv-build-mmlifelong \
  --dataset-root /path/to/MM-Lifelong \
  --subset game \
  --split test \
  --asset-root data/mmlifelong/assets/game \
  --case-root data/mmlifelong/cases/game/test \
  --verify-durations \
  --verify-clues
```

Generate resumable shared captions, then build a real sentence-transformer
cosine index and deterministic lexical+dense RRF index. The model revision,
dimension, and normalization setting are bound into the index manifest. The
bundled MM-Lifelong prompt reproduces the official ReMA global-caption prompt
and image contract while keeping timestamp shifting deterministic.

```bash
python main.py vv-caption \
  --asset-root data/mmlifelong/assets/game \
  --config /path/to/multimodal-api.yaml \
  --config-section investigator_api \
  --prompt-file prompts/mmlifelong_rema_caption_official.txt \
  --chunk-sec 300 \
  --sample-fps 1 \
  --max-frames 300 \
  --frame-extraction-mode fps_batch \
  --image-width 640 \
  --image-height 360 \
  --jpeg-quality 75 \
  --no-append-timestamp-map \
  --timestamp-shift-mode deterministic_rema_v3 \
  --workers 8 \
  --no-keep-frames \
  --resume

python main.py vv-index-caption \
  --asset-root data/mmlifelong/assets/game \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --index-mode hybrid
```

Run one Day case in an isolated workspace. `benchmark_best_effort` preserves the
last parseable answer for benchmark scoring while reporting reference failures;
`strict` retains the fail-closed behavior.

```bash
PYTHONPATH=src python tools/run_mmlifelong_interactive.py \
  --case-workspace data/mmlifelong/cases/game/test/mmlifelong-game-test-0038 \
  --out-dir runs/mmlifelong/game-test-0038 \
  --config /path/to/multimodal-api.yaml \
  --answer-policy benchmark_best_effort \
  --caption-index-mode hybrid \
  --caption-query-strategy adaptive \
  --caption-config-digest CAPTION_CONFIG_DIGEST \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2

PYTHONPATH=src python tools/summarize_mmlifelong_runs.py \
  --run-root runs/mmlifelong \
  --out-json reports/mmlifelong.json \
  --out-md reports/mmlifelong.md
```

`--caption-query-strategy adaptive` preserves the original `joint` behavior except
for explicit `after A ... before the first challenge/fight B` questions where `A`
is a chapter boundary. Those cases use ReMA-style independent short-query
retrieval, preserve enough shared result depth to inspect more than the first two
hits per query, perform a bounded lookback from plausible `B` events, group them
by the preceding chapter title, and expose the earliest paired event as a
locator-only visual inspection candidate. Raw Caption excerpts remain available,
and the candidate must still be verified visually. Use `rema` to force independent
multi-query retrieval for ablations; `joint` remains the default.

Use explicit sections in a shared YAML file. A separate role-specific file may put
the same fields at its root.

```yaml
reasoner_api:
  base: https://example.invalid/v1
  model: your-reasoner-model
  api_key: your-api-key
  timeout: 300
  max_retries: 5

investigator_api:
  base: https://example.invalid/v1
  model: your-vision-model
  api_key: your-api-key
  timeout: 300
  max_retries: 5
```

Run source-only cases:

```bash
python tools/run_virtual_videomme_interactive.py \
  --dataset-root /path/to/Video-MME/snapshot \
  --out-root runs/videomme-long \
  --case-ids 742-3 606-3 \
  --construction source_only \
  --reasoner-config /path/to/reasoner.yaml \
  --investigator-config /path/to/investigator.yaml \
  --low-fps 0.1 \
  --beat-sec 60 \
  --max-rounds 6 \
  --max-investigations 20 \
  --workers 2
```

For a reproducible case group, replace `--case-ids` with `--case-group PATH`.
Runs are immutable: use a new `--run-id` instead of resuming or overwriting an old
run. Workers are clamped to 1-16.

Build a 6-7 hour interleaved virtual timeline without writing a combined MP4:

```bash
python tools/run_virtual_videomme_interactive.py \
  --dataset-root /path/to/Video-MME/snapshot \
  --out-root runs/interleaved \
  --case-ids 606-3 \
  --construction interleaved_chunks \
  --min-duration-sec 21600 \
  --max-duration-sec 25200 \
  --chunk-sec 300 \
  --reasoner-config /path/to/reasoner.yaml \
  --investigator-config /path/to/investigator.yaml
```

Build a self-contained interaction viewer:

```bash
python tools/build_virtual_trace_viewer.py \
  --run-root runs/videomme-long \
  --out-dir artifacts/videomme-long-viewer \
  --light
```

Build a presentation-oriented viewer for one completed ultra-long-video case. The
output is an offline HTML page with a per-round `Reasoner task -> Investigator
call -> observation/evidence` flow, full/focus timelines, architecture playback,
workspace claims, uncertainty records, and bundled observed frames:

```bash
PYTHONPATH=src:. python tools/build_exploration_trace_viewer.py \
  --workspace runs/mmlifelong/game-test-0117 \
  --out-dir artifacts/game-test-0117-exploration \
  --zip-path artifacts/game-test-0117-exploration.zip \
  --title "Day · LongVideo Explorer"
```

## Code Map

```text
src/vcah/workspace.py            attempt identity, immutable log, Working Document, renderer
src/vcah/multiround.py           mechanical task loop and answer-reference validation
src/vcah/investigator.py         generic segment, window, and literal-ASR tools
src/vcah/interactive_agents.py   model Reasoner and observation-only Investigator
src/vcah/model_client.py         OpenAI-compatible client and usage metadata
src/vcah/virtual_video.py        timeline mapping, ASR, and frame materialization
src/vcah/virtual_index.py        overview and beat navigation index
src/vcah/captioning.py           resumable shared caption generation
src/vcah/caption_lexical_index.py  Unicode/BM25 caption retrieval
src/vcah/caption_semantic_index.py exact cosine retrieval with matrix cache
src/vcah/caption_hybrid_search.py  lexical+dense reciprocal-rank fusion
src/vcah/mmlifelong_metrics.py   free-form judge, Ref, retrieval, and run metrics
src/vcah/replay.py               immutable runs and content-free reproducibility records
tools/run_virtual_videomme_interactive.py  Video-MME evaluation orchestration
tools/run_mmlifelong_interactive.py        MM-Lifelong Day case orchestration
tools/summarize_mmlifelong_runs.py         aggregate comparison tables
tools/build_virtual_trace_viewer.py        HTML/zip trace viewer
tools/build_exploration_trace_viewer.py    presentation-oriented exploration viewer
```

The earlier generic single-agent harness remains available in `src/vcah/agent.py`.
The active long-video path is `VirtualVideoWorkspace` plus
`VirtualVideoMultiRoundDriver`.
