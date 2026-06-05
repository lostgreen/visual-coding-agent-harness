from visual_coding_agent_harness.agents.followup import (
    FollowupBudget,
    FollowupScheduler,
    FollowupTarget,
)


def _target(
    target_id: str,
    *,
    query: str,
    route: str = "needle_local",
    event_label: str | None = None,
    priority: int = 1,
    attempt_count: int = 0,
) -> FollowupTarget:
    return FollowupTarget(
        target_id=target_id,
        query=query,
        event_label=event_label,
        route=route,
        reason="missing visual support",
        priority=priority,
        attempt_count=attempt_count,
        parent_missing_evidence_id=f"missing_{target_id}",
    )


def test_enqueue_deduplicates_by_route_query_and_event_label():
    scheduler = FollowupScheduler(FollowupBudget())
    first = _target("fu_run_0001", query="red car", event_label="arrival")
    duplicate = _target("fu_run_0002", query="red car", event_label="arrival")
    distinct_event = _target("fu_run_0003", query="red car", event_label="departure")

    scheduler.enqueue([first, duplicate, distinct_event])

    assert scheduler.queue == [first, distinct_event]


def test_next_prefers_lower_priority_then_lower_attempt_count():
    scheduler = FollowupScheduler(FollowupBudget())
    high_attempt = _target("fu_run_0001", query="same priority later", priority=1, attempt_count=2)
    low_priority = _target("fu_run_0002", query="best target", priority=1, attempt_count=0)
    lower_rank = _target("fu_run_0003", query="less urgent", priority=3, attempt_count=0)
    scheduler.enqueue([lower_rank, high_attempt, low_priority])

    assert scheduler.next() is low_priority


def test_next_stops_after_global_budget():
    scheduler = FollowupScheduler(FollowupBudget(global_max_followups=1))
    target = _target("fu_run_0001", query="red car")
    scheduler.enqueue([target])
    scheduler.record_attempt(target, {"fs_001"})

    assert scheduler.next() is None


def test_next_completes_targets_at_per_gap_attempt_limit():
    scheduler = FollowupScheduler(FollowupBudget(per_gap_max_attempts=2))
    exhausted = _target("fu_run_0001", query="red car", attempt_count=2)
    available = _target("fu_run_0002", query="blue car", priority=2)
    scheduler.enqueue([exhausted, available])

    assert scheduler.next() is available
    assert scheduler.completed == [exhausted]
    assert scheduler.queue == [available]


def test_record_attempt_mutates_target_and_records_frame_sets():
    scheduler = FollowupScheduler(FollowupBudget())
    target = _target("fu_run_0001", query="red car")

    scheduler.record_attempt(target, {"fs_001", "fs_002"})

    assert target.attempt_count == 1
    assert scheduler.global_attempts == 1
    assert scheduler._frame_set_history == [{"fs_001", "fs_002"}]


def test_next_stops_when_recent_attempts_are_saturated():
    scheduler = FollowupScheduler(FollowupBudget(saturation_window=2, saturation_threshold=0.1))
    target = _target("fu_run_0001", query="red car")
    scheduler.enqueue([target])
    scheduler.record_attempt(target, set())
    scheduler.record_attempt(target, set())

    assert scheduler.next() is None
