import pytest

from visual_coding_agent_harness.legacy.contracts_v2 import (
    ClaimModality,
    ClaimRelation,
    OptionSpec,
    TargetRegistry,
    TargetSpec,
)


def _target(target_id, canonical_text, aliases=()):
    return TargetSpec(
        target_id=target_id,
        canonical_text=canonical_text,
        aliases=tuple(aliases),
        subject=canonical_text,
        relation="present",
        modality_hint=ClaimModality.UNKNOWN,
        source="unit_test",
    )


def test_resolves_known_target_ids_and_rejects_unknown_t_refs():
    registry = TargetRegistry.from_specs(targets=[_target("T1", "red car")])

    assert registry.known_target_ref("T1")
    assert registry.resolve_target_ref("T1").target_id == "T1"
    assert not registry.known_target_ref("T2")

    with pytest.raises(KeyError):
        registry.resolve_target_ref("T2")


def test_registry_rejects_non_numeric_target_ids():
    with pytest.raises(ValueError):
        TargetRegistry.from_specs(targets=[_target("foo", "red car")])

    with pytest.raises(ValueError):
        TargetRegistry.from_specs(targets=[_target("T_upper", "upper class")])


def test_option_membership_preserves_sequence_for_same_target_set():
    target_a = _target("T1", "red car")
    target_b = _target("T2", "blue bus")
    option_b = OptionSpec(
        option_id="B",
        target_sequence=("T1", "T2"),
        required_relations=(),
    )
    option_c = OptionSpec(
        option_id="C",
        target_sequence=("T2", "T1"),
        required_relations=(),
    )

    registry = TargetRegistry.from_specs(
        targets=[target_a, target_b],
        options=[option_b, option_c],
    )

    assert registry.option_for("B").target_sequence == ("T1", "T2")
    assert registry.option_for("C").target_sequence == ("T2", "T1")
    assert registry.options_for_target("T1") == (option_b, option_c)
    assert registry.options_for_target("T2") == (option_b, option_c)


def test_duplicate_canonical_text_is_not_resolved_silently():
    registry = TargetRegistry.from_specs(
        targets=[
            _target("T1", "same label"),
            _target("T2", "same label"),
        ]
    )

    assert not registry.known_target_ref("same label")
    with pytest.raises(KeyError):
        registry.resolve_target_ref("same label")

    assert registry.targets_for_canonical("same label") == (
        registry.targets_by_id["T1"],
        registry.targets_by_id["T2"],
    )


def test_registry_is_immutable_and_versioned():
    relation = ClaimRelation(
        relation_id="R1",
        kind="before",
        source_target_id="T1",
        destination_target_id="T2",
    )
    registry = TargetRegistry.from_specs(
        targets=[_target("T1", "red car"), _target("T2", "blue bus")],
        options=[
            OptionSpec(
                option_id="B",
                target_sequence=("T1", "T2"),
                required_relations=("R1",),
            )
        ],
        relations=[relation],
        version="review-v2",
    )

    assert registry.version == "review-v2"
    with pytest.raises(TypeError):
        registry.targets_by_id["T3"] = _target("T3", "green train")
    with pytest.raises(AttributeError):
        registry.version = "mutated"
