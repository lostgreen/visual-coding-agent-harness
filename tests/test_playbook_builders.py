from visual_coding_agent_harness.agents.skills.playbook import playbook_for_operator


def test_each_operator_playbook_has_structural_diagnostics():
    expected = {
        "select_present": ("grounded_factual_qa", "candidate_binding_missing"),
        "select_absent": ("complement_absence_qa", "competitor_presence_missing"),
        "causal_bind": ("causal_asr_qa", "causal_binding_missing"),
        "universal_intersection": ("universal_set_qa", "group_unvisited"),
        "ordered_projection": ("timeline_ordering", "ordered_item_missing"),
        "main_arc": ("main_idea", "arc_not_dominant"),
    }

    for operator, (skill_name, diagnostic) in expected.items():
        playbook = playbook_for_operator(
            operator,
            question="Question: Which option is supported?\nOptions:\nA. alpha\nB. beta",
            options=("A. alpha", "B. beta"),
            registry=None,
        )

        assert playbook is not None
        assert playbook.skill_name == skill_name
        assert playbook.answer_operator == operator
        assert diagnostic in {item.reason_code for item in playbook.stop_diagnostics}
        assert playbook.evidence_shape_target


def test_operator_playbooks_are_benchmark_content_free():
    forbidden_static_words = {
        "videomme",
        "french",
        "austro",
        "hungarian",
        "world war",
        "borghese",
        "bernini",
    }

    for operator in (
        "select_present",
        "select_absent",
        "causal_bind",
        "universal_intersection",
        "ordered_projection",
        "main_arc",
    ):
        playbook = playbook_for_operator(operator, question="", options=(), registry=None)
        text = " ".join(
            [
                playbook.skill_name,
                playbook.answer_operator,
                playbook.decomposition,
                " ".join(playbook.evidence_shape_target),
                " ".join(playbook.investigation_hints),
                " ".join(playbook.unsafe_final_conditions),
                " ".join(
                    f"{diagnostic.reason_code} {diagnostic.unmet_shape} {diagnostic.repair_hint}"
                    for diagnostic in playbook.stop_diagnostics
                ),
            ]
        ).lower()

        for forbidden in forbidden_static_words:
            assert forbidden not in text


def test_unknown_operator_has_no_playbook():
    assert playbook_for_operator("unsupported_operator", question="", options=(), registry=None) is None
