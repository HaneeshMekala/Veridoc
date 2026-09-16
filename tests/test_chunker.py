"""
tests/test_chunker.py — Layout-aware chunking (Novelty 1) must keep its promises.

WHY THIS FILE EXISTS:
The chunker's whole claim is "a chunk is a complete semantic unit that fits
the embedding model". These tests pin that claim down: one chunk never mixes
two regions, no chunk exceeds the token limit, splits never cut a word, no
text is lost, and the original casing survives (a real bug in the first
version — decoding tokens lowercased everything).

NAIVE WAY TO TEST (don't do this):
    assert chunks[3].text == "Step 57: Metformin HCl dose is raised to ..."
    PROBLEM: that hard-codes today's tokenizer. Upgrade `transformers` and the
    split points shift by a word — the test fails though nothing is broken.
BETTER (what we do): assert INVARIANTS — properties that must hold for any
correct chunking, whatever the exact split points are.

These tests need only the MiniLM *tokenizer* (a few MB), not a model.
"""

from __future__ import annotations

import pytest

from config import CHUNK_OVERLAP_TOKENS, MAX_CHUNK_TOKENS
from src.chunker import DocumentChunk, chunk_regions, count_tokens, get_chunk_stats
from src.layout import LayoutRegion


# ~1,700 tokens: long enough for several windows. Every sentence is unique, so
# each chunk can be located in the original text with str.find().
LONG_TEXT = " ".join(f"Step {i}: Metformin HCl dose is raised to {500 + i} MG." for i in range(120))


def region(text: str, label: str = "text", page: int = 0,
           bbox: tuple = (10.0, 20.0, 300.0, 80.0)) -> LayoutRegion:
    """Build a LayoutRegion as layout.py would, without running LayoutLMv3."""
    return LayoutRegion(label=label, words=[], text=text, bbox=list(bbox), page_num=page)


def spans(original: str, chunks: list[DocumentChunk]) -> list[tuple[int, int]]:
    """
    Locate each chunk in the original text.

    Returns:
        (start, end) character span per chunk.
    """
    result = []
    for c in chunks:
        start = original.find(c.text)
        assert start >= 0, f"chunk is not an exact substring of the region: {c.text[:60]!r}"
        result.append((start, start + len(c.text)))
    return result


@pytest.fixture(scope="module")
def long_chunks() -> list[DocumentChunk]:
    """The long region chunked once, shared by the sliding-window tests."""
    return chunk_regions([region(LONG_TEXT)], doc_id="long.pdf")


# ---------------------------------------------------------------------------
# SHORT REGIONS: one region -> one chunk, untouched
# ---------------------------------------------------------------------------

def test_short_region_becomes_one_unchanged_chunk() -> None:
    """A region under the limit is stored exactly as detected."""
    text = "Metformin HCl extended-release tablets: 500 mg and 1,000 mg."
    chunks = chunk_regions([region(text)], doc_id="d.pdf")
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_metadata_is_carried_from_region_to_chunk() -> None:
    """Page, label and bbox survive — retrieval filters depend on them."""
    [c] = chunk_regions([region("Dose 500 mg", label="table", page=3, bbox=(1, 2, 3, 4))], "d.pdf")
    assert (c.doc_id, c.page_num, c.region_label, c.bbox, c.chunk_index) == \
           ("d.pdf", 3, "table", [1, 2, 3, 4], 0)
    assert c.token_count == count_tokens("Dose 500 mg")


def test_empty_and_whitespace_regions_are_skipped() -> None:
    """A picture region with no text must not become an empty, searchable chunk."""
    chunks = chunk_regions([region(""), region("   \n "), region("Real text here")], "d.pdf")
    assert [c.text for c in chunks] == ["Real text here"]


def test_chunk_ids_are_unique() -> None:
    """ChromaDB rejects duplicate ids — and identical regions on one page are common."""
    regions = [region("Page 1 of 26", label="page_footer")] * 3 + [region(LONG_TEXT)]
    ids = [c.chunk_id for c in chunk_regions(regions, "d.pdf")]
    assert len(ids) == len(set(ids))


def test_a_chunk_never_mixes_two_regions() -> None:
    """
    THE Novelty-1 invariant: every chunk comes from exactly one region.

    A fixed-size chunker would happily glue the end of a paragraph to the
    start of the table below it. Here each chunk must lie inside its own
    region's text and carry that region's label.
    """
    regions = [region("Lactic acidosis warning text.", "text"),
               region("Dose\n500 mg\n2,000 mg", "table"),
               region("CONTRAINDICATIONS", "section_header")]
    for chunk in chunk_regions(regions, "d.pdf"):
        source = next(r for r in regions if r.label == chunk.region_label)
        assert chunk.text in source.text


# ---------------------------------------------------------------------------
# LONG REGIONS: sliding window
# ---------------------------------------------------------------------------

def test_long_region_is_split(long_chunks: list[DocumentChunk]) -> None:
    """A 1,700-token region cannot be one chunk; indices count up from 0."""
    assert len(long_chunks) > 1
    assert [c.chunk_index for c in long_chunks] == list(range(len(long_chunks)))


def test_every_chunk_fits_the_embedding_model(long_chunks: list[DocumentChunk]) -> None:
    """Over the limit, MiniLM silently truncates — the tail becomes unsearchable."""
    for c in long_chunks:
        assert count_tokens(c.text) <= MAX_CHUNK_TOKENS


def test_stored_token_count_matches_the_text(long_chunks: list[DocumentChunk]) -> None:
    """token_count drives the retriever's minimum-size filter, so it must be true."""
    for c in long_chunks:
        assert c.token_count == count_tokens(c.text)


def test_sub_chunks_keep_original_casing_and_symbols(long_chunks: list[DocumentChunk]) -> None:
    """
    Regression: the first version rebuilt chunk text with tokenizer.decode(),
    which lowercased everything ("MG" -> "mg") and produced "##" fragments.
    Every chunk must be an exact slice of the original text instead.
    """
    spans(LONG_TEXT, long_chunks)                 # asserts exact-substring
    assert all("##" not in c.text for c in long_chunks)
    assert all("HCl" in c.text and "MG" in c.text for c in long_chunks)


def test_splits_never_cut_through_a_word(long_chunks: list[DocumentChunk]) -> None:
    """A split between two letters or digits would create fragments like 'Metfor' + 'min'."""
    for start, end in spans(LONG_TEXT, long_chunks):
        assert not (start > 0 and LONG_TEXT[start - 1].isalnum() and LONG_TEXT[start].isalnum())
        assert not (end < len(LONG_TEXT) and LONG_TEXT[end - 1].isalnum() and LONG_TEXT[end].isalnum())


def test_consecutive_chunks_overlap(long_chunks: list[DocumentChunk]) -> None:
    """Each window starts before the previous one ended, by at most the overlap budget."""
    s = spans(LONG_TEXT, long_chunks)
    for (_, end_a), (start_b, _) in zip(s, s[1:]):
        assert start_b < end_a, "no overlap: a sentence at the boundary could be split"
        assert 0 < count_tokens(LONG_TEXT[start_b:end_a]) <= CHUNK_OVERLAP_TOKENS


def test_no_text_is_lost(long_chunks: list[DocumentChunk]) -> None:
    """Together the windows cover the whole region, first character to last."""
    s = spans(LONG_TEXT, long_chunks)
    assert s[0][0] == 0 and s[-1][1] == len(LONG_TEXT)
    for (_, end_a), (start_b, _) in zip(s, s[1:]):
        assert start_b <= end_a                   # no gap between windows


def test_text_without_spaces_still_splits_and_terminates() -> None:
    """A flattened numeric table row has no spaces; the window must still advance."""
    text = "-".join(str(i) for i in range(1500))
    chunks = chunk_regions([region(text, label="table")], "d.pdf")
    assert len(chunks) > 1
    assert all(count_tokens(c.text) <= MAX_CHUNK_TOKENS for c in chunks)
    assert spans(text, chunks)[-1][1] == len(text)


# ---------------------------------------------------------------------------
# STATS
# ---------------------------------------------------------------------------

def test_chunk_stats() -> None:
    """Label distribution and page coverage are what the Index tab reports."""
    assert get_chunk_stats([]) == {"total_chunks": 0}
    chunks = chunk_regions([region("Intro text", "text", page=0),
                            region("Dose 500 mg", "table", page=2),
                            region("More text", "text", page=2)], "d.pdf")
    stats = get_chunk_stats(chunks)
    assert stats["total_chunks"] == 3
    assert stats["label_distribution"] == {"text": 2, "table": 1}
    assert stats["pages_covered"] == 2
