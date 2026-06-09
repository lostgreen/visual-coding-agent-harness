from visual_coding_agent_harness.agents.transcript_binder import TranscriptEvidenceBinder
from visual_coding_agent_harness.contracts import ClaimRelation, TargetSpec


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
