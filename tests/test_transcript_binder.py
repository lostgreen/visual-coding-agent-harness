from visual_coding_agent_harness.agents.transcript_binder import TranscriptEvidenceBinder
from visual_coding_agent_harness.contracts import (
    ClaimRelation,
    OptionSpec,
    TargetSpec,
    build_ordered_transcript_sequence,
    ordered_sequence_exact_option,
)


def _target(target_id: str, canonical_text: str, *, subject: str = "Goya") -> TargetSpec:
    return TargetSpec(
        target_id=target_id,
        canonical_text=canonical_text,
        aliases=(),
        subject=subject,
        relation="present",
        source="unit_test",
    )


def test_rejects_lexical_hits_that_do_not_support_target_claims():
    binder = TranscriptEvidenceBinder()

    upper = _target("T1", "upper class")
    successful = _target("T2", "successful", subject="he")
    farmhouse = _target("T3", "farmhouse", subject="he")
    isolated_goya = _target("T4", "isolated", subject="Goya")

    assert binder.bind(
        text="the upper-class man cruelly mimics the beggar",
        targets=[upper],
        obs_id="obs_1",
    ).evidence_bindings[0].status != "supported"
    assert binder.bind(
        text="He never entered the upper class.",
        targets=[upper],
        obs_id="obs_2",
    ).evidence_bindings[0].status != "supported"
    relation_result = binder.bind(
        text="He left the farmhouse before becoming successful.",
        targets=[successful, farmhouse],
        relations=[
            ClaimRelation(
                relation_id="R1",
                kind="before",
                source_target_id="T2",
                destination_target_id="T3",
            )
        ],
        obs_id="obs_3",
    )
    assert relation_result.relation_bindings[0].status != "supported"
    reverse_cue_result = binder.bind(
        text="Goya became successful only after leaving the farmhouse.",
        targets=[successful, farmhouse],
        relations=[
            ClaimRelation(
                relation_id="R2",
                kind="before",
                source_target_id="T2",
                destination_target_id="T3",
            )
        ],
        obs_id="obs_3b",
    )
    assert reverse_cue_result.relation_bindings[0].status != "supported"
    assert binder.bind(
        text="The painting depicts an isolated man.",
        targets=[isolated_goya],
        obs_id="obs_4",
    ).evidence_bindings[0].status != "supported"
    assert binder.bind(
        text="Goya painted an isolated man.",
        targets=[isolated_goya],
        obs_id="obs_5",
    ).evidence_bindings[0].status != "supported"


def test_supports_explicit_goya_life_targets_and_order_relation():
    binder = TranscriptEvidenceBinder()
    humble = _target("T1", "humble background")
    upper = TargetSpec(
        target_id="T2",
        canonical_text="upper class",
        aliases=("upper",),
        subject="Goya",
        relation="present",
        source="unit_test",
    )

    result = binder.bind(
        text="Goya was a man from a humble background who rose through the ranks to reach the upper",
        targets=[humble, upper],
        relations=[
            ClaimRelation(
                relation_id="R1",
                kind="before",
                source_target_id="T1",
                destination_target_id="T2",
            )
        ],
        obs_id="obs_positive",
        segment_id="seg_0001",
        start_sec=10.0,
    )

    statuses = {binding.target_id: binding.status for binding in result.evidence_bindings}
    assert statuses == {"T1": "supported", "T2": "supported"}
    assert result.relation_bindings[0].status == "supported"
    assert result.evidence_bindings[0].mention_timestamp_sec == 10.0


def test_grounding_plan_preserves_discriminator_source():
    binder = TranscriptEvidenceBinder()
    target = TargetSpec(
        target_id="T1",
        canonical_text="ancient empire lifecycle",
        aliases=("empire documentary",),
        discriminators=("rise of an ancient empire", "fall into ruins"),
        subject="",
        relation="present",
        source="unit_test",
    )

    result = binder.bind(
        text="The narration describes the rise of an ancient empire and its later collapse.",
        targets=[target],
        obs_id="obs_discriminator",
        segment_id="seg_0001",
    )

    assert result.evidence_bindings[0].status == "supported"
    assert result.target_text_hits[0].target_ref == "T1"
    assert result.target_text_hits[0].phrase == "rise of an ancient empire"
    assert result.target_text_hits[0].match_source == "discriminator"


def test_temporal_relation_contradiction_is_explicit():
    binder = TranscriptEvidenceBinder()
    humble = _target("T1", "humble background", subject="Goya")
    upper = TargetSpec(
        target_id="T2",
        canonical_text="upper class",
        aliases=("upper",),
        subject="Goya",
        relation="present",
        source="unit_test",
    )

    result = binder.bind(
        text="Goya reached the upper class after leaving his humble background.",
        targets=[upper, humble],
        relations=[
            ClaimRelation(
                relation_id="R1",
                kind="before",
                source_target_id="T2",
                destination_target_id="T1",
            )
        ],
        obs_id="obs_contradicted",
        segment_id="seg_0001",
    )

    assert result.relation_bindings[0].status == "contradicted"


def test_611_complete_asr_enumeration_creates_supported_sequence():
    targets = [
        _target("T1", "Aeneas, Anchises, and Ascanius fleeing Troy", subject=""),
        _target("T2", "David", subject=""),
        _target("T3", "The rape of Persephone", subject=""),
        _target("T4", "Apollo and Daphne", subject=""),
    ]

    sequence = build_ordered_transcript_sequence(
        text=(
            'The narrator presents Bernini works in this order: "Aeneas, Anchises, and Ascanius '
            'fleeing Troy", "David", "The rape of Persephone", and "Apollo and Daphne".'
        ),
        targets=targets,
        segment_id="seg_0002",
        obs_id="obs_0003",
        start_sec=182.0,
        end_sec=198.0,
    )

    assert sequence is not None
    assert sequence.status == "supported"
    assert sequence.ordered_target_refs == ("T1", "T2", "T3", "T4")


def test_order_is_based_on_text_position_not_timestamp():
    targets = [
        _target("T1", "first work", subject=""),
        _target("T2", "second work", subject=""),
        _target("T3", "third work", subject=""),
    ]

    sequence = build_ordered_transcript_sequence(
        text='"third work", "first work", and "second work" are mentioned as a list.',
        targets=targets,
        segment_id="seg_0001",
        start_sec=30.0,
        end_sec=10.0,
    )

    assert sequence is not None
    assert sequence.status == "supported"
    assert sequence.ordered_target_refs == ("T3", "T1", "T2")


def test_missing_target_is_ambiguous():
    targets = [
        _target("T1", "Aeneas", subject=""),
        _target("T2", "David", subject=""),
        _target("T3", "Persephone", subject=""),
    ]

    sequence = build_ordered_transcript_sequence(
        text='The list says "Aeneas" and "David".',
        targets=targets,
        segment_id="seg_0001",
    )

    assert sequence is not None
    assert sequence.status == "ambiguous"


def test_duplicate_target_is_ambiguous():
    targets = [
        _target("T1", "Aeneas", subject=""),
        _target("T2", "David", subject=""),
        _target("T3", "Persephone", subject=""),
    ]

    sequence = build_ordered_transcript_sequence(
        text='The list says "Aeneas", "David", "Aeneas", and "Persephone".',
        targets=targets,
        segment_id="seg_0001",
    )

    assert sequence is not None
    assert sequence.status == "ambiguous"


def test_unrelated_artwork_list_is_rejected():
    targets = [
        _target("T1", "Aeneas", subject=""),
        _target("T2", "David", subject=""),
        _target("T3", "Persephone", subject=""),
    ]

    sequence = build_ordered_transcript_sequence(
        text='The catalogue comparison gives unrelated examples: "Aeneas", "David", and "Persephone".',
        targets=targets,
        segment_id="seg_0001",
    )

    assert sequence is not None
    assert sequence.status == "rejected"


def test_complete_sequence_maps_exactly_to_option_d():
    targets = [
        _target("T1", "Aeneas", subject=""),
        _target("T2", "David", subject=""),
        _target("T3", "Persephone", subject=""),
        _target("T4", "Apollo and Daphne", subject=""),
    ]
    sequence = build_ordered_transcript_sequence(
        text='"Aeneas", "David", "Persephone", and "Apollo and Daphne".',
        targets=targets,
        segment_id="seg_0001",
    )

    option = ordered_sequence_exact_option(
        sequence,
        [
            OptionSpec("C", target_sequence=("T1", "T3", "T2", "T4")),
            OptionSpec("D", target_sequence=("T1", "T2", "T3", "T4")),
        ],
    )

    assert option == "D"
