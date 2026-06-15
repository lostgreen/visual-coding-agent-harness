---
name: general_exploration
version: 1
description: Fallback skill when no specialized skill clearly matches the question. Conservative evidence collection across modalities.
when_to_use: Use only when the question does not match any specialized skill's when_to_use. Prefer specialized skills when applicable.
default_claim_modality: mixed
recovery_rules:
  insufficient:
    action: need_more_evidence
    target: missing answer-grade evidence
---
# General Exploration

Use this playbook only when no specialized skill clearly matches the question.

Planner playbook:
- Localize likely evidence with conservative search or coverage tools.
- Collect answer-grade visual, narrated, OCR, or QA evidence before finalizing.
- Prefer switching to a specialized skill when the question clearly matches one.
