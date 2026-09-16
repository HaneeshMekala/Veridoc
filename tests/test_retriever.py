"""
tests/test_retriever.py — Region filtering and the relevance floor.

WHY THIS FILE EXISTS:
The retriever decides what the LLM is allowed to read. Two of its promises
matter most: region filters mean exactly what they say (Novelty 1 at query
time), and an off-topic question gets NOTHING rather than the five
least-bad chunks — which the LLM would treat as evidence.

Two layers, as described in conftest.py:
  - Unit tests: build_where_filter and the score floor, with fake embedder
    and store. No model, milliseconds.
  - Behaviour tests: real MiniLM + a throwaway ChromaDB in a temp folder.
"""

from __future__ import annotations

import numpy as np
import pytest

from config import (
    EMBEDDING_DIMENSION,
    RETRIEVAL_EXCLUDED_LABELS,
    RETRIEVAL_MIN_CHUNK_TOKENS,
    RETRIEVAL_MIN_SCORE,
    RERANK_CANDIDATES,
)
from src.chunker import DocumentChunk, count_tokens
from src.retriever import Retriever, build_where_filter
from src.vectorstore import VectorStore

SIZE_CONDITION = {"token_count": {"$gte": RETRIEVAL_MIN_CHUNK_TOKENS}}


# ---------------------------------------------------------------------------
# build_where_filter — pure function, no model
# ---------------------------------------------------------------------------

def test_default_filter_excludes_headers_and_fragments() -> None:
    """With no filter, page headers/footers and tiny fragments are left out."""
    assert build_where_filter() == {"$and": [
        SIZE_CONDITION,
        {"region_label": {"$nin": list(RETRIEVAL_EXCLUDED_LABELS)}},
    ]}


def test_region_filter_includes_explicitly_requested_excluded_labels() -> None:
    """Asking for page_header on purpose must work — exclusion is only a default."""
    where = build_where_filter(region_types=["page_header"])
    assert {"region_label": {"$in": ["page_header"]}} in where["$and"]


def test_doc_filter_is_added() -> None:
    """Restricting to documents adds a third condition."""
    where = build_where_filter(region_types=["table"], doc_ids=["a.pdf"])
    assert where["$and"][1:] == [{"region_label": {"$in": ["table"]}},
                                 {"doc_id": {"$in": ["a.pdf"]}}]


@pytest.mark.parametrize("kwargs, message", [
    ({"region_types": []}, "would match nothing"),
    ({"region_types": ["tables"]}, "Unknown region type"),     # typo of "table"
    ({"doc_ids": []}, "would match nothing"),
])
def test_filters_that_could_never_match_fail_loudly(kwargs: dict, message: str) -> None:
    """
    ChromaDB accepts {"$in": ["tables"]} and returns zero hits. The user would
    read that as "not in the document" — a silent failure. We raise instead.

    LEARN THIS — @pytest.mark.parametrize runs one test function once per
    tuple, reported separately: three bugs, three clear failure lines.
    """
    with pytest.raises(ValueError, match=message):
        build_where_filter(**kwargs)


# ---------------------------------------------------------------------------
# Retriever with fakes — the relevance floor
# ---------------------------------------------------------------------------

class FakeEmbedder:
    """Returns the same unit vector for any query."""

    def embed_query(self, query: str) -> np.ndarray:
        """Return a fixed vector of the right shape."""
        return np.full(EMBEDDING_DIMENSION, 1 / np.sqrt(EMBEDDING_DIMENSION), dtype=np.float32)


class FakeStore:
    """Returns canned hits and records how it was queried."""

    def __init__(self, hits: list) -> None:
        """Store the hits to return."""
        self.hits = hits
        self.calls: list[tuple] = []

    def query(self, vec: np.ndarray, top_k: int, where: dict | None = None) -> list:
        """Record the call; return the first top_k canned hits."""
        self.calls.append((top_k, where))
        return self.hits[:top_k]


def test_hits_below_the_relevance_floor_are_dropped(make_chunk) -> None:
    """Scores 0.8 and 0.3 clear the 0.25 floor; 0.2 and 0.1 do not."""
    hits = [make_chunk(f"t{s}", score=s) for s in (0.8, 0.3, 0.2, 0.1)]
    result = Retriever(FakeEmbedder(), FakeStore(hits), min_score=0.25).retrieve("q")
    assert [h.score for h in result] == [0.8, 0.3]


def test_nothing_relevant_returns_empty_list(make_chunk) -> None:
    """All hits under the floor → [], which makes the generator abstain."""
    hits = [make_chunk("x", score=0.1), make_chunk("y", score=0.05)]
    assert Retriever(FakeEmbedder(), FakeStore(hits)).retrieve("q") == []


def test_filter_and_top_k_reach_the_store(make_chunk) -> None:
    """Filtering happens INSIDE the vector search, not on the results afterwards."""
    store = FakeStore([])
    Retriever(FakeEmbedder(), store).retrieve("q", top_k=3, region_types=["table"])
    assert store.calls == [(3, build_where_filter(region_types=["table"]))]


class FakeReranker:
    """Reverses the order it is given — any real change of order will do."""

    def __init__(self) -> None:
        """Record which passages were sent for re-ranking."""
        self.seen: list = []

    def rerank(self, query: str, hits: list) -> list:
        """Return the hits reversed, with a rerank_score set."""
        self.seen = list(hits)
        for i, h in enumerate(reversed(hits)):
            h.rerank_score = float(-i)
        return list(reversed(hits))


def test_reranker_gets_a_wider_pool_and_top_k_is_cut_after(make_chunk) -> None:
    """Fetch RERANK_CANDIDATES, re-order them, THEN keep top_k — not the other way round."""
    hits = [make_chunk(f"t{i}", score=0.9 - i * 0.01) for i in range(RERANK_CANDIDATES)]
    store, reranker = FakeStore(hits), FakeReranker()
    result = Retriever(FakeEmbedder(), store, reranker=reranker).retrieve("q", top_k=3)
    assert store.calls[0][0] == RERANK_CANDIDATES
    assert [h.text for h in result] == ["t19", "t18", "t17"]    # best AFTER re-ranking
    assert all(h.rerank_score is not None for h in result)


def test_relevance_floor_applies_before_reranking(make_chunk) -> None:
    """
    The floor is the abstention decision and stays on the calibrated cosine
    score: a below-floor passage must never reach the re-ranker, however
    well the re-ranker might score it.
    """
    hits = [make_chunk("good", score=0.8), make_chunk("junk", score=0.1)]
    reranker = FakeReranker()
    Retriever(FakeEmbedder(), FakeStore(hits), reranker=reranker).retrieve("q")
    assert [h.text for h in reranker.seen] == ["good"]


def test_nothing_relevant_skips_the_reranker(make_chunk) -> None:
    """Off-topic question: empty list, and no re-ranking work at all."""
    reranker = FakeReranker()
    result = Retriever(FakeEmbedder(), FakeStore([make_chunk("x", score=0.05)]),
                       reranker=reranker).retrieve("q")
    assert result == [] and reranker.seen == []


# ---------------------------------------------------------------------------
# Behaviour tests — real MiniLM + real ChromaDB (auto-marked `model`)
# ---------------------------------------------------------------------------

_DOCS = {
    "dose":     ("text", "The maximum recommended daily dose of metformin extended-release "
                         "tablets is 2,000 mg, taken once daily with the evening meal."),
    "renal":    ("text", "Metformin is contraindicated in patients with severe renal "
                         "impairment, defined as an eGFR below 30 mL/min/1.73 m2."),
    "table":    ("table", "Dose\nFrequency\n500 mg\nonce daily\n2,000 mg\nonce daily maximum"),
    "header":   ("page_header", "Metformin dose prescribing information, page 3"),
    "fragment": ("text", "dose mg"),                         # < 5 tokens
    # Decoy copied from the sample FDA label: mostly the drug name, so the
    # bi-encoder ranked it ABOVE the contraindications passage (0.659 vs 0.607).
    "caption":  ("caption", "Table 4: Effect of Metformin on Coadministered Drug Systemic Exposure"),
}


@pytest.fixture(scope="module")
def indexed(embedder, tmp_path_factory) -> tuple[Retriever, VectorStore, list[DocumentChunk]]:
    """A throwaway ChromaDB holding the five chunks above."""
    chunks = [DocumentChunk(text, f"test.pdf_{key}", "test.pdf", 0, label,
                            [0.0, 0.0, 100.0, 20.0], 0, count_tokens(text))
              for key, (label, text) in _DOCS.items()]
    store = VectorStore(persist_dir=str(tmp_path_factory.mktemp("chroma")),
                        collection_name="test_chunks")
    store.add_chunks(chunks, embedder.embed_chunks(chunks))
    return Retriever(embedder, store), store, chunks


def test_relevant_chunk_ranks_first(indexed) -> None:
    """The passage that answers the question is the top hit, above the floor."""
    retriever, _, _ = indexed
    hits = retriever.retrieve("What is the maximum daily dose of metformin?")
    assert hits[0].chunk_id == "test.pdf_dose"
    assert hits[0].score >= RETRIEVAL_MIN_SCORE


def test_page_headers_are_excluded_by_default_but_reachable(indexed) -> None:
    """The header mentions 'metformin dose' — keyword bait that must not win by default."""
    retriever, _, _ = indexed
    question = "metformin dose prescribing information"
    assert all(h.region_label != "page_header" for h in retriever.retrieve(question))
    explicit = retriever.retrieve(question, region_types=["page_header"])
    assert [h.chunk_id for h in explicit] == ["test.pdf_header"]


def test_table_filter_returns_only_tables(indexed) -> None:
    """Novelty 1 at query time: 'search only tables' means only tables."""
    retriever, _, _ = indexed
    hits = retriever.retrieve("What doses are available?", region_types=["table"])
    assert hits and all(h.region_label == "table" for h in hits)


def test_fragments_are_never_returned(indexed) -> None:
    """Even a query identical to a 2-token fragment can't retrieve it."""
    retriever, _, _ = indexed
    assert all(h.chunk_id != "test.pdf_fragment" for h in retriever.retrieve("dose mg"))


def test_off_topic_question_retrieves_nothing(indexed) -> None:
    """The relevance floor, measured: unrelated questions score far below 0.25."""
    retriever, _, _ = indexed
    assert retriever.retrieve("What is the capital of France?") == []


@pytest.fixture(scope="module")
def reranker():
    """The real ms-marco cross-encoder (auto-marked `model` via the embedder it pairs with)."""
    from src.reranker import Reranker
    return Reranker()


def test_reranker_puts_the_answer_above_a_drug_name_decoy(indexed, reranker) -> None:
    """
    Regression for the failure that motivated re-ranking: in a single-drug
    document every chunk says "metformin", and a short caption that is mostly
    the drug name out-scored the passage that answers the question.
    """
    retriever, store, _ = indexed
    reranked = Retriever(retriever.embedder, store, reranker=reranker)
    hits = reranked.retrieve("In which patients is metformin contraindicated?")
    assert hits[0].chunk_id == "test.pdf_renal"
    assert hits[0].rerank_score > max(h.rerank_score for h in hits[1:])


def test_reranking_keeps_the_off_topic_abstention(indexed, reranker) -> None:
    """Adding a re-ranker must not make off-topic questions return passages."""
    retriever, store, _ = indexed
    assert Retriever(retriever.embedder, store, reranker=reranker).retrieve(
        "What is the capital of France?") == []


def test_reindexing_a_document_replaces_it(indexed, embedder) -> None:
    """Uploading the same PDF twice must not duplicate its chunks (stale evidence)."""
    _, store, chunks = indexed
    before = store.count()
    store.add_chunks(chunks, embedder.embed_chunks(chunks))
    assert store.count() == before == len(chunks)
