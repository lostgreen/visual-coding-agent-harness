# Evaluation Layer

This package owns benchmark scoring. Agent/runtime code under `src/vcah` emits
predictions and evidence traces without reading reference answers. Each
benchmark evaluator then combines those immutable runtime artifacts with its
own reference data and official protocol.

Shared code under `evaluate/common` is limited to artifact IO, API transport,
schemas, and provenance. Score parsing, prompts, metric semantics, and
aggregation remain benchmark-specific.
