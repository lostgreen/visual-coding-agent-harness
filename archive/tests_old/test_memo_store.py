from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from visual_coding_agent_harness.contracts.playbook import Playbook
from visual_coding_agent_harness.workspace.memo import MemoStore, ObservationMemo


def _memo(index: int) -> ObservationMemo:
    return ObservationMemo(
        memo_id=f"memo_{index:05d}_q1",
        beat_id=f"bt{index % 3:05d}",
        observation=f"visible object {index}",
        source_query_id="q1",
        source_playbook=Playbook.IDENTIFY_VISUAL,
        created_at="2026-07-04T00:00:00+00:00",
    )


def test_memo_store_concurrent_append_and_invalidate(tmp_path: Path) -> None:
    store = MemoStore(tmp_path / "memos.jsonl")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(store.append, (_memo(index) for index in range(100))))

    assert len(store.load()) == 100
    victim = "memo_00004_q1"
    assert any(memo.memo_id == victim for memo in store.get("bt00001"))

    store.invalidate(victim, by_evidence_id="ev_bad")

    assert all(memo.memo_id != victim for memo in store.get("bt00001"))
