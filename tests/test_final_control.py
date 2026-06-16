from visual_coding_agent_harness.agents.final_control import (
    FinalDecisionOwner,
    parse_model_final_response,
    recover_locked_answer_from_malformed_final,
)


def test_framework_owner_is_historical_only_symbol() -> None:
    assert FinalDecisionOwner.FRAMEWORK.value == "framework"
    assert FinalDecisionOwner.MODEL.value == "model"


def test_model_final_parser_accepts_explicit_option_json() -> None:
    decision = parse_model_final_response(
        '{"status":"final","answer":"D. full rise and fall","citations":["obs_0003"],"confidence":0.82}',
        allowed_options=["A. first", "D. full rise and fall"],
    )

    assert decision.is_final
    assert decision.answer == "D"
    assert decision.owner is FinalDecisionOwner.MODEL
    assert decision.citations == ["obs_0003"]


def test_model_final_parser_rejects_continue_or_program() -> None:
    decision = parse_model_final_response(
        '{"status":"continue","program":[{"tool":"vision_read","args":{}}]}',
        allowed_options=["A. first", "B. second"],
    )

    assert not decision.is_final
    assert decision.status == "invalid"
    assert decision.reason == "model_declined_final_with_program"


def test_format_repair_requires_locked_answer_to_match() -> None:
    malformed = "{status: final, answer: 'B'}"
    locked = recover_locked_answer_from_malformed_final(malformed, allowed_options=["A. alpha", "B. beta"])

    assert locked == "B"
    changed = parse_model_final_response(
        '{"status":"final","answer":"A","citations":[]}',
        allowed_options=["A. alpha", "B. beta"],
        locked_answer=locked,
    )
    same = parse_model_final_response(
        '{"status":"final","answer":"B","citations":[]}',
        allowed_options=["A. alpha", "B. beta"],
        locked_answer=locked,
    )

    assert not changed.is_final
    assert changed.reason == "format_repair_answer_changed"
    assert same.is_final
