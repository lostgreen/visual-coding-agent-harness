from __future__ import annotations

from visual_coding_agent_harness.agents.prompt_frames import PromptFrame, PromptFrameLedger
from visual_coding_agent_harness.agents.runtime_capabilities import PromptRetentionMode


def test_stateless_retention_always_renders_full_body() -> None:
    frame = PromptFrame(frame_id="skill:main", title="Active Skill", body="full playbook", version="v1")
    ledger = PromptFrameLedger(mode=PromptRetentionMode.STATELESS_FULL)

    rendered = [ledger.take(frame) for _ in range(3)]

    assert all("full playbook" in item for item in rendered)
    assert all("[loaded" not in item for item in rendered)


def test_prefix_cached_retention_records_digest_but_keeps_full_body() -> None:
    frame = PromptFrame(frame_id="skill:main", title="Active Skill", body="full playbook", version="v1")
    ledger = PromptFrameLedger(mode=PromptRetentionMode.PREFIX_CACHED_FULL)

    first = ledger.take(frame)
    second = ledger.take(frame)

    assert "full playbook" in first
    assert "full playbook" in second
    assert ledger.snapshot() == {"skill:main": frame.digest}


def test_sticky_retention_uses_reference_after_initial_full_body() -> None:
    frame = PromptFrame(frame_id="skill:main", title="Active Skill", body="full playbook", version="v1")
    ledger = PromptFrameLedger(mode=PromptRetentionMode.STICKY_REFERENCE)

    first = ledger.take(frame)
    second = ledger.take(frame)

    assert "full playbook" in first
    assert second == f"# Active Skill [loaded, v=v1, digest={frame.digest}]"


def test_digest_change_renders_replacing_full_body() -> None:
    ledger = PromptFrameLedger(mode=PromptRetentionMode.STICKY_REFERENCE)
    original = PromptFrame(frame_id="tool:schema", title="Tool Schema", body="short schema", version="v1")
    changed = PromptFrame(frame_id="tool:schema", title="Tool Schema", body="wider schema", version="v2")

    ledger.take(original)
    rendered = ledger.take(changed)

    assert "wider schema" in rendered
    assert "replacing" in rendered
    assert changed.digest in rendered


def test_invalidate_forces_full_body() -> None:
    frame = PromptFrame(frame_id="skill:main", title="Active Skill", body="full playbook", version="v1")
    ledger = PromptFrameLedger(mode=PromptRetentionMode.STICKY_REFERENCE)

    ledger.take(frame)
    ledger.invalidate(frame.frame_id)
    rendered = ledger.take(frame)

    assert "full playbook" in rendered
    assert "[loaded" not in rendered
