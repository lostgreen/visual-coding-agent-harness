from __future__ import annotations

from visual_coding_agent_harness.evidence.answer_operators import derive_answer_operator, derive_modality_route


def test_plain_factual_question_defaults_to_select_present() -> None:
    assert derive_answer_operator("Which object appears in the video?", route="needle_local", options=()) == "select_present"


def test_not_questions_are_absence_operator() -> None:
    assert derive_answer_operator("Which is NOT described by the narrator?", route="needle_local", options=()) == "select_absent"
    assert derive_answer_operator("What animals did not see the trainer?", route="needle_local", options=()) == "select_absent"


def test_negative_reason_questions_are_causal_not_absent() -> None:
    assert (
        derive_answer_operator("Why is it not recommended to enter the area?", route="mixed_asr_visual", options=())
        == "causal_bind"
    )
    assert derive_answer_operator("What is the primary reason for the warning?", route="mixed_asr_visual", options=()) == "causal_bind"


def test_universal_order_and_main_arc_detection() -> None:
    assert derive_answer_operator("Which object appears in every case study?", route="needle_local", options=()) == "universal_intersection"
    assert (
        derive_answer_operator(
            "Which order happens successively?",
            route="temporal_order",
            options=("A. T1 -> T2 -> T3", "B. T1 -> T3 -> T2"),
        )
        == "ordered_projection"
    )
    assert derive_answer_operator("What is the video mainly about?", route="gist_global", options=()) == "main_arc"


def test_modality_routing_biography_to_asr() -> None:
    operator = derive_answer_operator("How was his life journey according to the video?", route="", options=())

    assert derive_modality_route("How was his life journey according to the video?", operator=operator) == "asr_primary"


def test_modality_routing_main_arc_stays_multimodal() -> None:
    operator = derive_answer_operator("What's the main idea of the video?", route="", options=())

    assert operator == "main_arc"
    assert derive_modality_route("What's the main idea of the video?", operator=operator) == "multimodal"


def test_modality_routing_visual_and_ocr_questions() -> None:
    assert derive_modality_route("What text is written on the banner?", operator="select_present") == "ocr_primary"
    assert derive_modality_route("What color shirt is the person wearing?", operator="select_present") == "visual_primary"
