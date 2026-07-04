from __future__ import annotations

from visual_coding_agent_harness.workspace.text_index import InvertedIndex


def test_inverted_index_supports_terms_phrases_and_modality_filters(tmp_path) -> None:
    index = InvertedIndex()
    index.add("bt001", "My father worked at the shipyard.", modality="asr")
    index.add("bt002", "A waterfront photograph appears.", modality="ocr")
    index.add("bt003", "The shipyard closed later.", modality="ocr")

    assert [hit.beat_id for hit in index.search("shipyard")] == ["bt001", "bt003"]
    assert [hit.beat_id for hit in index.search('"my father"')] == ["bt001"]
    assert [hit.beat_id for hit in index.search("shipyard", modality=("ocr",))] == ["bt003"]

    path = tmp_path / "text_index.json"
    index.save(path)
    loaded = InvertedIndex.load(path)

    assert [hit.beat_id for hit in loaded.search("shipyard")] == ["bt001", "bt003"]
