from vcah.interactive_agents import _select_caption_queries


def test_explicit_caption_queries_exclude_whole_question() -> None:
    queries = _select_caption_queries(
        "Find the target event after the anchor.",
        ("entity A", "alias A"),
        fallback="Which item appears after entity A is defeated?",
        strategy="joint",
    )

    assert queries == ("entity A", "alias A")
    assert "Which item appears after entity A is defeated?" not in queries


def test_task_goal_precedes_whole_question_fallback() -> None:
    assert _select_caption_queries(
        "the player enters a new chapter",
        (),
        fallback="A compound benchmark question",
        strategy="rema",
    ) == ("the player enters a new chapter",)

    assert _select_caption_queries(
        "",
        (),
        fallback="A compound benchmark question",
        strategy="joint",
    ) == ("A compound benchmark question",)
