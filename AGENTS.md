# Project Agent Notes

## Reasoning-model completion budgets

- For GPT-5-family reasoning models, `max_completion_tokens` includes hidden reasoning tokens and visible answer tokens.
- A successful HTTP response can still contain no visible `message.content` when `finish_reason=length` and reasoning tokens consume the entire budget.
- Do not classify that response as missing evidence or model abstention. Inspect and record `finish_reason`, `completion_tokens`, and `completion_tokens_details.reasoning_tokens` first.
- Final-answer and compact final-answer calls use at least 4096 completion tokens. Keep retries at the same or a larger budget; shortening the prompt alone does not release hidden reasoning budget.
- Preserve response usage metadata in interaction traces so evaluation failures can distinguish navigation, reasoning, serialization, and token-budget failures.
