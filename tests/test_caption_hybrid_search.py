from __future__ import annotations

import pytest

from vcah.caption_hybrid_search import CaptionHybridSearch
from vcah.caption_lexical_index import CaptionLexicalIndex, render_caption_hits
from vcah.caption_semantic_index import CaptionSemanticIndex

from test_caption_semantic_index import FakeSemanticAdapter, passages


def test_hybrid_rrf_is_deterministic_and_keeps_component_scores() -> None:
    fixture_passages = passages()
    lexical = CaptionLexicalIndex(fixture_passages, config_digest="fixture")
    dense = CaptionSemanticIndex.build(
        fixture_passages,
        adapter=FakeSemanticAdapter(),
        config_digest="fixture",
    )
    hybrid = CaptionHybridSearch(lexical, dense, rrf_k0=60)

    first = hybrid.search(("stationary car",), top_k=3)
    second = hybrid.search(("stationary car",), top_k=3)

    assert [hit.passage_id for hit in first] == [hit.passage_id for hit in second]
    assert [hit.fused_score for hit in first] == [hit.fused_score for hit in second]
    assert first[0].passage_id == "cap:p2"
    assert first[0].lexical_score is not None
    assert first[0].dense_score is not None
    assert first[0].metadata["lexical_rank"] == 1
    assert first[0].metadata["dense_rank"] is not None
    assert first[0].metadata["source_segments"] == ["seg_0002"]


def test_hybrid_query_fingerprint_includes_rank_configuration() -> None:
    fixture_passages = passages()
    lexical = CaptionLexicalIndex(fixture_passages, config_digest="fixture")
    dense = CaptionSemanticIndex.build(
        fixture_passages,
        adapter=FakeSemanticAdapter(),
        config_digest="fixture",
    )
    default = CaptionHybridSearch(lexical, dense, rrf_k0=60)
    changed = CaptionHybridSearch(lexical, dense, rrf_k0=30)

    default_fingerprint = default.query_fingerprint(
        ("stationary car",),
        top_k=5,
        time_range=None,
        expand_neighbors=0,
    )
    changed_fingerprint = changed.query_fingerprint(
        ("stationary car",),
        top_k=5,
        time_range=None,
        expand_neighbors=0,
    )

    assert default.index_digest != changed.index_digest
    assert default_fingerprint != changed_fingerprint


def test_hybrid_segment_scope_filters_components_and_changes_fingerprint() -> None:
    fixture_passages = passages()
    lexical = CaptionLexicalIndex(fixture_passages, config_digest="fixture")
    dense = CaptionSemanticIndex.build(
        fixture_passages,
        adapter=FakeSemanticAdapter(),
        config_digest="fixture",
    )
    hybrid = CaptionHybridSearch(lexical, dense)

    scoped = hybrid.search(
        ("stationary car",),
        top_k=3,
        segment_ids=("seg_0001",),
        expand_neighbors=1,
    )
    first_fingerprint = hybrid.query_fingerprint(
        ("stationary car",),
        top_k=3,
        time_range=None,
        segment_ids=("seg_0001",),
        expand_neighbors=1,
    )
    second_fingerprint = hybrid.query_fingerprint(
        ("stationary car",),
        top_k=3,
        time_range=None,
        segment_ids=("seg_0002",),
        expand_neighbors=1,
    )

    assert [hit.passage_id for hit in scoped] == ["cap:p0"]
    assert first_fingerprint != second_fingerprint


def test_rema_query_strategy_balances_independent_query_results() -> None:
    fixture_passages = passages()
    lexical = CaptionLexicalIndex(fixture_passages, config_digest="fixture")
    dense = CaptionSemanticIndex.build(
        fixture_passages,
        adapter=FakeSemanticAdapter(),
        config_digest="fixture",
    )
    joint = CaptionHybridSearch(lexical, dense)
    rema = CaptionHybridSearch(lexical, dense, query_strategy="rema")

    hits = rema.search(("stationary car", "stage performance"), top_k=2)

    assert [hit.passage_id for hit in hits] == ["cap:p2", "cap:p1"]
    assert hits[0].metadata["query_strategy"] == "rema"
    assert hits[0].metadata["query_matches"][0] == {
        "query": "stationary car",
        "rank": 1,
    }
    assert hits[1].metadata["query_matches"][0] == {
        "query": "stage performance",
        "rank": 1,
    }
    assert len(hits) == 2
    assert rema.index_digest != joint.index_digest
    rendered = render_caption_hits(hits)
    assert "matched queries: stationary car" in rendered
    assert "matched queries: stage performance" in rendered


def test_hybrid_rejects_unknown_query_strategy() -> None:
    fixture_passages = passages()
    lexical = CaptionLexicalIndex(fixture_passages, config_digest="fixture")
    dense = CaptionSemanticIndex.build(
        fixture_passages,
        adapter=FakeSemanticAdapter(),
        config_digest="fixture",
    )

    with pytest.raises(ValueError, match="unsupported hybrid query strategy"):
        CaptionHybridSearch(lexical, dense, query_strategy="unknown")
