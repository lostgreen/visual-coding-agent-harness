# VideoMME Gemini 512-Frame Direct Baseline

## Objective

Measure a comparable direct-answer ceiling for the same VideoMME long cases used by the agent harness. The baseline must isolate Gemini's end-to-end video QA ability from agent navigation, evidence summarization, audit, and verification errors.

The first run uses `videomme_long_hard_rotate_v3.json` (10 cases). After the protocol is stable, it can be applied to all 22 unique long questions already tested by the harness.

## Input Protocol

For each source video:

- Uniformly sample exactly 512 frames over the full duration using bin midpoints: `t_i = (i + 0.5) * duration / 512`.
- Preserve a frame manifest containing frame index, source timestamp, and path.
- Include the complete existing ASR cues with source timestamps. Do not summarize, semantically index, retrieve, or truncate the ASR in the normal path.
- Send the question, answer options, 512 frames, and timestamped ASR in one Gemini request.

The preferred request contains 512 independent images. If the API rejects the image count or request size, retry once with 32 ordered 4x4 contact sheets that preserve all 512 frames. The result records `input_mode=images_512` or `input_mode=contact_sheets_32`; the two modes are reported separately.

## Model Output

Request compact JSON:

```json
{
  "answer": "A",
  "rationale": "brief observable justification",
  "evidence": [
    {
      "frame_index": 123,
      "time_sec": 456.7,
      "asr_quote": "optional short quote"
    }
  ]
}
```

`rationale` is an explicitly returned answer explanation, not hidden chain-of-thought. The evaluation stores and analyzes only the returned answer, concise rationale, evidence references, request mode, latency, retries, and errors.

## Execution

- Reuse `OpenAICompatibleVisionClient` so model configuration, timeout, retryable HTTP statuses, exponential backoff, jitter, and `Retry-After` behavior match the agent run.
- Run cases concurrently with a configurable worker count clamped to 16. Use 10 workers for the v3 group.
- Keep each case to one successful model request, excluding transport retries and the explicit contact-sheet fallback.
- Do not expose gold answers, target intervals, or harness trajectories to the model.

## Artifacts

Each case writes:

- `frame_manifest.jsonl`
- `asr_prompt.txt`
- `request_metadata.json` without credentials or image bytes
- `response.json`
- `result.json`

The group writes `summary.json` with raw accuracy, request-mode counts, failures, mean latency, and per-case results.

## Comparison

Compare direct and agent runs on identical case IDs:

- answer accuracy
- direct-versus-agent answer agreement
- localization hit: whether agent evidence overlaps direct evidence timestamps
- conditional agent accuracy after a localization hit
- answer changes introduced by audit or claim verification
- visual input count and model-call count

This separates perception limits from navigation, evidence compression, and verifier-induced regressions.

## Acceptance Criteria

- Unit tests verify midpoint sampling, exact 512-frame manifests, answer parsing, contact-sheet fallback metadata, and ordered result aggregation.
- One KML smoke case succeeds with the configured Gemini endpoint.
- The 10-case v3 group completes with workers <= 16 and existing retry behavior.
- No hidden chain-of-thought is requested, stored, or reported.
- Agent architecture changes begin only after the direct baseline and failure attribution are available.
