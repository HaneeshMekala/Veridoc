"""
src/retriever.py — Top-k retrieval with layout-aware metadata filtering.

WHY THIS MODULE EXISTS:
The vector store answers "which vectors are nearest to this vector?".
The rest of the app wants something higher-level: "given this QUESTION,
which pieces of the document should the LLM read?". This module bridges
the two: it embeds the question, translates user-facing filters (region
types, documents) into ChromaDB filter syntax, removes known noise, and
drops hits that are not relevant enough to be trusted as evidence.

This is where Novelty 1 pays off at query time: because every chunk knows
its region type, the retriever can search "only tables" or "skip page
headers" — something a fixed-size-chunk pipeline simply cannot express.

Without it, generator.py and app.py would each have to know about
embeddings, ChromaDB's filter language and score thresholds.
"""

from __future__ import annotations

from typing import Any

from config import (
    REGION_LABELS,
    RETRIEVAL_EXCLUDED_LABELS,
    RETRIEVAL_MIN_CHUNK_TOKENS,
    RETRIEVAL_MIN_SCORE,
    TOP_K_RETRIEVAL,
)
from src.embedder import Embedder
from src.vectorstore import RetrievedChunk, VectorStore


# INTERVIEW ALERT — "What happens when the answer is NOT in the documents?"
# Top-k search ALWAYS returns k results, even for "What's the capital of
# France?" asked against a diabetes leaflet — the 5 least-bad chunks. Handed
# to an LLM as "context", these invite a confident, made-up answer.
# Answer: "We apply a relevance floor (minimum cosine similarity). If no chunk
# clears it, retrieval returns nothing and the generator says the information
# isn't in the document. Retrieval-side abstention is the cheapest
# hallucination defence — it stops bad context before the LLM ever sees it."


class Retriever:
    """
    Question in → relevant, filtered chunks out.

    WHY THE EMBEDDER AND STORE ARE PASSED IN (dependency injection):
    The retriever does not create its own models. app.py builds ONE Embedder
    and ONE VectorStore and hands them to everything that needs them — so
    the 90MB model is loaded once, and tests can pass in small fakes.

    Usage:
        retriever = Retriever(embedder, store)
        hits = retriever.retrieve("What is the maximum daily dose?",
                                  region_types=["table", "text"])
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        min_score: float = RETRIEVAL_MIN_SCORE,
    ) -> None:
        """
        Store references to the shared embedder and vector store.

        Args:
            embedder:  Loaded Embedder (the SAME model used at indexing time).
            store:     Opened VectorStore holding the indexed chunks.
            min_score: Relevance floor; hits below it are discarded.
        """
        self.embedder = embedder
        self.store = store
        self.min_score = min_score

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
        region_types: list[str] | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Find the chunks most relevant to a question.

        NAIVE APPROACH (don't do this):
            hits = store.query(embed(query), top_k=5)
            PROBLEM: returns running page headers, 1-token figure fragments,
            and — for off-topic questions — 5 irrelevant chunks that the LLM
            will treat as evidence.
        BETTER APPROACH (what we do):
            Filter noise INSIDE the search (so we still get top_k good hits),
            then drop anything below the relevance floor.

        Args:
            query:        The user's question.
            top_k:        Maximum number of chunks to return.
            region_types: Only search these region labels, e.g. ["table"].
                          None = all labels except RETRIEVAL_EXCLUDED_LABELS.
            doc_ids:      Only search these documents. None = all documents.

        Returns:
            Up to top_k chunks, best first. Empty list if nothing is relevant.

        Raises:
            ValueError: If the query is blank or a region type is unknown.
        """
        where = build_where_filter(region_types, doc_ids)
        query_vec = self.embedder.embed_query(query)
        hits = self.store.query(query_vec, top_k=top_k, where=where)
        return [h for h in hits if h.score >= self.min_score]


# ---------------------------------------------------------------------------
# FILTER CONSTRUCTION
# ---------------------------------------------------------------------------

def build_where_filter(
    region_types: list[str] | None = None,
    doc_ids: list[str] | None = None,
) -> dict[str, Any]:
    """
    Translate user-facing filters into a ChromaDB `where` clause.

    LEARN THIS — Filter at query time vs at index time:
    We could simply NOT index page headers and figure fragments. But deleting
    at index time is irreversible — if a user later asks "what journal is
    this from?", the running header is exactly the answer. Filtering at query
    time keeps the data and makes exclusion a per-question decision.

    ChromaDB filter syntax used here:
        {"region_label": {"$in":  [...]}}   label is one of these
        {"region_label": {"$nin": [...]}}   label is none of these
        {"token_count":  {"$gte": 5}}       at least 5 tokens
        {"$and": [cond1, cond2, ...]}       all conditions must hold

    Args:
        region_types: Labels to include. None = all except the default
                      exclusions. Explicitly listed labels are always allowed,
                      even excluded ones (e.g. ["page_header"]).
        doc_ids:      Documents to include. None = all.

    Returns:
        A ChromaDB where-dict (always contains at least the size condition).

    Raises:
        ValueError: If region_types is empty or contains an unknown label.
    """
    conditions: list[dict[str, Any]] = [
        {"token_count": {"$gte": RETRIEVAL_MIN_CHUNK_TOKENS}},
    ]

    if region_types is not None:
        _validate_region_types(region_types)
        conditions.append({"region_label": {"$in": list(region_types)}})
    elif RETRIEVAL_EXCLUDED_LABELS:
        conditions.append({"region_label": {"$nin": list(RETRIEVAL_EXCLUDED_LABELS)}})

    if doc_ids is not None:
        if not doc_ids:
            raise ValueError("doc_ids=[] would match nothing; pass None to search all documents.")
        conditions.append({"doc_id": {"$in": list(doc_ids)}})

    # ChromaDB requires $and to have at least 2 conditions.
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def _validate_region_types(region_types: list[str]) -> None:
    """
    Reject filters that could never match.

    WHY THIS EXISTS:
    ChromaDB treats {"region_label": {"$in": ["tables"]}} as a perfectly valid
    query that matches zero chunks. The user would see "no relevant content"
    and blame the document — a silent failure. We fail loudly instead.

    Args:
        region_types: Labels requested by the caller.

    Raises:
        ValueError: If the list is empty or has labels the layout model never emits.
    """
    if not region_types:
        raise ValueError("region_types=[] would match nothing; pass None to search all regions.")

    unknown = sorted(set(region_types) - set(REGION_LABELS))
    if unknown:
        raise ValueError(
            f"Unknown region type(s) {unknown}. Valid labels: {', '.join(REGION_LABELS)}."
        )
