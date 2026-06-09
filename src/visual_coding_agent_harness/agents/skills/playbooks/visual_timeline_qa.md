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
- If a locator observation exposes a complete ordered transcript sequence, promote that sequence; visual corroboration is optional unless the question explicitly asks for onscreen/visible order.
- If a locator observation exposes a focused ordered-list `vision_read` action, execute that focused read before anchor verification.
- Confirm each ordered event with a timestamped visual observation.
- Use anchor verification for separate individual-event anchors, not as the main route for one ordered-list scene.
- Do not answer temporal or motion claims from narration alone when the question asks what is visibly shown.
