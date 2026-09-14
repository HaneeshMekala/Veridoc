"""
src/chunker.py — Layout-aware document chunking (Novelty 1 core).

WHY THIS MODULE EXISTS:
This is the heart of what makes this RAG system research-worthy.
Standard RAG pipelines split documents by fixed character/token count.
This module splits by SEMANTIC REGION — a boundary detected by LayoutLMv3.

The result: every chunk is a complete semantic unit (a full paragraph,
a complete table, a full section heading), never a fragment of one.
This directly improves retrieval quality because a retrieved chunk
always contains coherent information, not a sentence cut mid-thought.

INTERVIEW ALERT — Be ready to answer all three of these:
  Q: "What chunking strategy did you use?"
  A: "Layout-aware chunking using LayoutLMv3 region boundaries."

  Q: "Why not fixed-size chunking?"
  A: "Fixed-size splits mid-sentence, mid-table, mid-heading. A 500-char
     window has no idea it just cut a table in half. Our approach respects
     semantic boundaries, so each chunk is always a complete unit of meaning."

  Q: "How did you handle regions longer than the embedding model's limit?"
  A: "We sub-split long regions with a sliding window and token overlap,
     so context at the split boundary is preserved in both sub-chunks."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from transformers import AutoTokenizer   # AutoTokenizer: auto-selects the right
                                         # tokenizer for any HuggingFace model.
                                         # Needed to count tokens accurately —
                                         # character count is NOT reliable for this.

from config import (
    EMBEDDING_MODEL_NAME,
    MAX_CHUNK_TOKENS,
    CHUNK_OVERLAP_TOKENS,
)
from src.layout import LayoutRegion


# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    """
    A single chunk of text ready for embedding and storage in ChromaDB.

    WHY METADATA MATTERS:
    Storing metadata alongside the vector is what makes ChromaDB more
    powerful than a plain FAISS index. At retrieval time, we can filter:
      "Give me the top-5 chunks, but only from 'table' regions"
    This lets users ask: "What do the tables say about drug dosage?"
    and get back only table content — much more precise than unfiltered retrieval.

    Fields:
        text:         The chunk text that will be embedded.
        chunk_id:     Unique string ID for ChromaDB (must be unique per collection).
        doc_id:       Which document this chunk came from (filename).
        page_num:     0-indexed page number.
        region_label: LayoutLMv3 label: "text", "title", "table", "figure", "list".
        bbox:         Bounding box [x0, y0, x1, y1] of this chunk on the page.
        chunk_index:  Position of this chunk within its parent region (for sub-splits).
        token_count:  How many tokens this chunk contains.
    """
    text: str
    chunk_id: str
    doc_id: str
    page_num: int
    region_label: str
    bbox: list[float]
    chunk_index: int = 0
    token_count: int = 0


# ---------------------------------------------------------------------------
# TOKENIZER (loaded once, reused for all token counting)
# ---------------------------------------------------------------------------

# WHY THIS EXISTS:
# We count tokens using the SAME tokenizer as our embedding model.
# Counting characters would be wrong — "café" is 4 chars but might be
# 5 tokens. "COVID-19" is 8 chars but might be 4 tokens.
# Token count is what the embedding model actually sees, so that's
# what we must stay within.
#
# We load this at module level (not inside a function) so it's initialized
# once per process, not once per chunk. Tokenizer loading is fast (~0.1s)
# but wasteful if repeated thousands of times.
_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)

# This tokenizer only COUNTS and ALIGNS tokens; it never feeds the model.
# Long regions are expected here (that's why we split them), so silence the
# "sequence length is longer than the specified maximum" warning.
_tokenizer.model_max_length = 1_000_000


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def chunk_regions(
    regions: list[LayoutRegion],
    doc_id: str,
) -> list[DocumentChunk]:
    """
    Convert a list of LayoutRegions into DocumentChunks ready for embedding.

    This is the main entry point for the chunker. It processes all regions
    from all pages of a document in one call.

    NAIVE APPROACH (don't do this):
        text = " ".join(r.text for r in regions)
        chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
        Problem 1: Merges across pages, headers, tables — loses all structure.
        Problem 2: 2000 chars ≠ 512 tokens. You'll silently truncate the model.
        Problem 3: No metadata — you can't filter by region type at retrieval.

    BETTER APPROACH (what we do):
        Each region is chunked independently. Short regions → 1 chunk.
        Long regions → multiple overlapping sub-chunks. All metadata preserved.

    Args:
        regions: LayoutRegion list from layout.py (all pages of one document).
        doc_id:  A unique identifier for the source document (e.g. filename).

    Returns:
        List of DocumentChunk, ordered by page then region then sub-chunk.
    """
    all_chunks: list[DocumentChunk] = []
    chunk_counter = 0  # global counter to ensure unique chunk_ids

    for region in regions:
        if not region.text.strip():
            continue  # skip empty regions (e.g. figure regions with no caption text)

        sub_chunks = _split_region(region)

        for sub_idx, (text, token_count) in enumerate(sub_chunks):
            chunk_id = f"{doc_id}_p{region.page_num}_r{chunk_counter}_c{sub_idx}"

            all_chunks.append(DocumentChunk(
                text=text,
                chunk_id=chunk_id,
                doc_id=doc_id,
                page_num=region.page_num,
                region_label=region.label,
                bbox=region.bbox,
                chunk_index=sub_idx,
                token_count=token_count,
            ))

        chunk_counter += 1

    return all_chunks


# ---------------------------------------------------------------------------
# SPLITTING LOGIC
# ---------------------------------------------------------------------------

def _split_region(region: LayoutRegion) -> list[tuple[str, int]]:
    """
    Split one region into one or more (text, token_count) pairs.

    Rules:
      - If the region fits within MAX_CHUNK_TOKENS → return as-is (1 chunk).
      - If the region is longer → split into overlapping sub-chunks using
        a sliding window whose edges snap to whole words.

    LEARN THIS — Offset mapping (token ↔ character alignment):
    A "fast" HuggingFace tokenizer can report, for every token, the
    (start_char, end_char) span it came from in the original string, and
    which word it belongs to (word_ids). That lets us choose split points in
    TOKEN space (to respect the model's limit) but cut the ORIGINAL text in
    CHARACTER space — so the chunk keeps its exact casing, spacing and symbols.

    Args:
        region: A single LayoutRegion.

    Returns:
        List of (text, token_count) tuples. Usually length 1, longer for
        large regions like multi-page tables or long paragraphs.
    """
    encoding = _tokenizer(
        region.text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    token_count = len(encoding["input_ids"])

    if token_count <= MAX_CHUNK_TOKENS:
        # Region fits entirely — no splitting needed.
        return [(region.text, token_count)]

    # Region is too long — use sliding window with overlap.
    return _sliding_window_split(region.text, encoding["offset_mapping"], encoding.word_ids())


def _sliding_window_split(
    text: str,
    offsets: list[tuple[int, int]],
    word_ids: list[int | None],
) -> list[tuple[str, int]]:
    """
    Split a long text into overlapping sub-chunks using a sliding window.

    LEARN THIS — Why overlap matters:
    Imagine splitting: "...end of context A. [SPLIT] Start of context B..."
    Without overlap, a retrieval query about the transition between A and B
    might not match either chunk well — the relevant sentence was cut in half.

    With CHUNK_OVERLAP_TOKENS=50, chunk 2 starts 50 tokens before where
    chunk 1 ended, so the boundary sentence appears in BOTH chunks.
    Any query about that boundary will now retrieve a complete version of it.

    Visualization:
      Chunk 1: |=====MAX_CHUNK_TOKENS=====|
      Chunk 2:                    |===overlap===|=====new content=====|
      Chunk 3:                                         |===overlap===|=====new content=====|

    NAIVE APPROACH (what the first version did):
        chunk_text = tokenizer.decode(tokens[start:end])
        PROBLEM: MiniLM's tokenizer is UNCASED — decoding returns
        lowercased text ("Metformin 500 MG" → "metformin 500 mg"), and a
        window starting mid-word yields "##formin". That damaged text is
        what the LLM would read and the NLI checker would verify against.
    BETTER APPROACH (what we do):
        Pick window edges in token space, snap them to word boundaries,
        then slice the ORIGINAL text using the tokens' character offsets.

    Args:
        text:     Full region text.
        offsets:  (start_char, end_char) for every token of `text`.
        word_ids: Word index for every token (tokens of one word share an id).

    Returns:
        List of (text, token_count) tuples.
    """
    sub_chunks: list[tuple[str, int]] = []
    n = len(offsets)
    start = 0

    while start < n:
        end = _snap_end_to_word(word_ids, start, min(start + MAX_CHUNK_TOKENS, n))
        chunk_text = text[offsets[start][0]:offsets[end - 1][1]].strip()
        if chunk_text:
            sub_chunks.append((chunk_text, end - start))

        if end == n:
            break   # reached the end — stop

        # Next window starts OVERLAP tokens before this one ended
        # (always moving forward by at least one token).
        next_start = max(end - CHUNK_OVERLAP_TOKENS, start + 1)
        start = _snap_start_to_word(word_ids, next_start, end)

    return sub_chunks


def _snap_end_to_word(word_ids: list[int | None], start: int, end: int) -> int:
    """
    Move a window end backwards so it does not cut through a word.

    Args:
        word_ids: Word index per token.
        start:    Window start (the end never moves to or before it).
        end:      Proposed exclusive end index.

    Returns:
        Adjusted exclusive end. Unchanged if already at the text end, or if
        the window is one giant word that cannot be split cleanly.
    """
    if end >= len(word_ids):
        return end
    snapped = end
    while snapped > start + 1 and word_ids[snapped] == word_ids[snapped - 1]:
        snapped -= 1
    return snapped if word_ids[snapped] != word_ids[snapped - 1] else end


def _snap_start_to_word(word_ids: list[int | None], start: int, limit: int) -> int:
    """
    Move a window start forwards to the first token of a word.

    Args:
        word_ids: Word index per token.
        start:    Proposed start index.
        limit:    Do not move past this index (end of the previous window).

    Returns:
        Adjusted start index.
    """
    while 0 < start < limit and word_ids[start] == word_ids[start - 1]:
        start += 1
    return start


# ---------------------------------------------------------------------------
# TOKEN COUNTING UTILITIES
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[int]:
    """
    Tokenize text and return the list of token IDs (no special tokens).

    WHY NO SPECIAL TOKENS:
    Special tokens ([CLS], [SEP]) are added by the model's forward pass
    automatically, so we count content tokens only. The 2 special tokens are
    budgeted separately: MAX_CHUNK_TOKENS is set 2+ below the model's
    256-token limit, and embedder.py verifies that margin at startup.

    Args:
        text: Raw text string.

    Returns:
        List of integer token IDs.
    """
    return _tokenizer.encode(
        text,
        add_special_tokens=False,  # don't count [CLS] and [SEP]
        truncation=False,           # we want the REAL count, not a truncated one
    )


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text string.

    Public utility — useful for debugging and for the Gradio UI to show
    chunk sizes.

    Args:
        text: Any string.

    Returns:
        Integer token count.
    """
    return len(_tokenize(text))


def get_chunk_stats(chunks: list[DocumentChunk]) -> dict:
    """
    Compute summary statistics over a list of chunks.

    Useful for debugging chunking quality and for notebooks.

    Args:
        chunks: List of DocumentChunk from chunk_regions().

    Returns:
        Dict with total_chunks, mean_tokens, label_distribution, pages_covered.
    """
    if not chunks:
        return {"total_chunks": 0}

    token_counts = [c.token_count for c in chunks]
    label_dist: dict[str, int] = {}
    for c in chunks:
        label_dist[c.region_label] = label_dist.get(c.region_label, 0) + 1

    return {
        "total_chunks": len(chunks),
        "mean_tokens": round(sum(token_counts) / len(token_counts), 1),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "label_distribution": label_dist,
        "pages_covered": len(set(c.page_num for c in chunks)),
    }
