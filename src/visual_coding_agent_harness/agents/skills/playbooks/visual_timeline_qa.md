---
name: visual_timeline_qa
version: 1
default_claim_modality: visual_fact
recovery_rules:
  missing_event:
    action: need_more_evidence
    target: missing visual event window
  conflict:
    action: need_more_evidence
    target: conflicting visual timestamps
---
# Visual Timeline QA

Use this playbook when the question asks about visible actions, object motion, spatial changes, or event ordering shown on screen. Visual timeline claims require timestamped visual evidence.

Planner playbook:
- Locate coarse candidate segments before focused reads.
- Confirm each ordered event with a timestamped visual observation.
- Do not answer temporal or motion claims from narration alone when the question asks what is visibly shown.
