"""
src/vectorstore.py — ChromaDB wrapper: setup, insert, query, delete.

WHY THIS MODULE EXISTS:
The embedder turns chunks into vectors; this module STORES those vectors
on disk together with their text and metadata, and finds the nearest ones
to a query vector. It is the memory of the RAG system — without it, every
PDF would have to be re-parsed, re-chunked and re-embedded on every query.

It also keeps ChromaDB-specific details (metadata type rules, result
format, distance → similarity conversion) out of the rest of the codebase.
If we ever swap ChromaDB for Qdrant or pgvector, only this file changes.

INTERVIEW ALERT — How a vector DB finds neighbours fast (HNSW):
"How does your vector store search millions of vectors quickly?"
Answer: "ChromaDB uses HNSW — Hierarchical Navigable Small World graphs.
Each vector is a node linked to its near neighbours, in layers: the top
layers are sparse 'highways' for long jumps, the bottom layer is dense for
fine search. A query greedily walks from the top layer down, so search is
roughly O(log n) instead of comparing against every vector (O(n)). It is
APPROXIMATE nearest neighbour search: it can occasionally miss the true
best match, trading a tiny bit of recall for a huge speed-up."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import chromadb                      # ChromaDB chosen over FAISS: runs locally,
                                     # persists to disk, and stores text + metadata
                                     # alongside vectors. FAISS is only an index —
                                     # we would need a separate DB for text/metadata
                                     # and would have to hand-write filtering.

import numpy as np                   # numpy: the embedder returns numpy arrays;
                                     # used here for shape validation.

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    CHROMA_DISTANCE_METRIC,
    CHROMA_INSERT_BATCH_SIZE,
    EMBEDDING_DIMENSION,
)

if TYPE_CHECKING:                    # type hints only — see embedder.py for why
    from src.chunker import DocumentChunk


# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """
    One search hit returned from the vector store.

    Mirrors DocumentChunk's fields plus a similarity score, so downstream
    code (retriever, generator, NLI checker, UI) never touches raw ChromaDB
    result dicts.

    Fields:
        chunk_id:     ChromaDB id of the chunk.
        text:         The chunk text (evidence for the LLM and the NLI check).
        doc_id:       Source document identifier.
        page_num:     0-indexed page number — lets the UI cite "page 3".
        region_label: LayoutLMv3 region type ("table", "text", ...).
        bbox:         [x0, y0, x1, y1] of the source region on the page.
        score:        Cosine similarity to the query, in [-1, 1]; higher = closer.
    """
    chunk_id: str
    text: str
    doc_id: str
    page_num: int
    region_label: str
    bbox: list[float]
    score: float


# ---------------------------------------------------------------------------
# VECTOR STORE
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Persistent ChromaDB collection of document chunks.

    Usage:
        store = VectorStore()
        store.add_chunks(chunks, embedder.embed_chunks(chunks))
        hits = store.query(embedder.embed_query("dose?"), top_k=5,
                           where={"region_label": "table"})
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ) -> None:
        """
        Open (or create) the on-disk ChromaDB collection.

        WHY embedding_function=None:
        By default ChromaDB attaches its OWN embedding model and will silently
        embed any raw text you pass in. If that ever happened, some vectors
        would come from ChromaDB's model and some from ours — two different
        vector spaces, so similarity scores between them would be garbage.
        Setting it to None forces every vector to come from src/embedder.py;
        passing text without a vector becomes an error instead of a silent bug.

        Args:
            persist_dir:     Folder for ChromaDB files. Defaults to config.CHROMA_DIR.
            collection_name: Name of the collection (like a table in SQL).
        """
        path = str(persist_dir or CHROMA_DIR)
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": CHROMA_DISTANCE_METRIC},
            embedding_function=None,
        )
        print(f"[VectorStore] Collection '{collection_name}' at {path} "
              f"({self.collection.count()} chunks)")

    # --- writing ------------------------------------------------------------

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> int:
        """
        Store chunks and their embeddings, replacing any older copy of the doc.

        NAIVE APPROACH (don't do this):
            collection.add(ids=..., embeddings=...) every time a PDF is uploaded.
            PROBLEM: re-uploading the same PDF either crashes (duplicate ids)
            or, if the PDF changed and now has fewer chunks, leaves stale old
            chunks behind that still get retrieved — outdated evidence.
        BETTER APPROACH (what we do):
            Delete every existing chunk of this doc_id first, then insert.
            One document in → exactly its current chunks in the store.

        Args:
            chunks:     DocumentChunks from chunk_regions() (may span several docs).
            embeddings: Array of shape (len(chunks), EMBEDDING_DIMENSION),
                        row i belonging to chunks[i].

        Returns:
            Number of chunks inserted.

        Raises:
            ValueError: If chunks and embeddings do not line up.
        """
        _validate_embeddings(chunks, embeddings)
        if not chunks:
            return 0

        for doc_id in {c.doc_id for c in chunks}:
            self.delete_document(doc_id)

        for start in range(0, len(chunks), CHROMA_INSERT_BATCH_SIZE):
            end = start + CHROMA_INSERT_BATCH_SIZE
            batch = chunks[start:end]
            self.collection.add(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings[start:end].tolist(),
                documents=[c.text for c in batch],
                metadatas=[_chunk_to_metadata(c) for c in batch],
            )
        return len(chunks)

    def delete_document(self, doc_id: str) -> None:
        """
        Remove every chunk belonging to one document.

        Args:
            doc_id: The document identifier used when the chunks were added.
        """
        self.collection.delete(where={"doc_id": doc_id})

    # --- reading ------------------------------------------------------------

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Return the top_k chunks nearest to a query vector.

        LEARN THIS — Metadata filtering (this is what makes Novelty 1 usable):
        `where` restricts the search to chunks whose metadata matches, e.g.
            {"region_label": "table"}                      → tables only
            {"$and": [{"doc_id": "x.pdf"}, {"page_num": 2}]} → one page of one doc
        The filter is applied as part of the search, so you get the best
        top_k matches AMONG tables — not the top_k overall with non-tables
        thrown away afterwards (which could leave you with 0 results).

        Args:
            query_embedding: Vector of shape (EMBEDDING_DIMENSION,) from embed_query().
            top_k:           Maximum number of hits to return.
            where:           Optional ChromaDB metadata filter.

        Returns:
            Hits sorted by descending similarity. May be fewer than top_k if
            the collection (or the filtered subset) is smaller.

        Raises:
            ValueError: If the query vector has the wrong shape or top_k < 1.
        """
        if query_embedding.shape != (EMBEDDING_DIMENSION,):
            raise ValueError(
                f"query_embedding must have shape ({EMBEDDING_DIMENSION},), "
                f"got {query_embedding.shape}. Use Embedder.embed_query()."
            )
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}.")

        total = self.collection.count()
        if total == 0:
            return []

        result = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, total),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return _parse_query_result(result)

    def count(self) -> int:
        """
        Return the total number of chunks stored.

        Returns:
            Chunk count across all documents.
        """
        return self.collection.count()

    def has_document(self, doc_id: str) -> bool:
        """
        Check whether a document has already been indexed.

        Lets the UI skip the slow parse → layout → embed pipeline for a PDF
        that is already in the store.

        Args:
            doc_id: The document identifier.

        Returns:
            True if at least one chunk with this doc_id exists.
        """
        hits = self.collection.get(where={"doc_id": doc_id}, limit=1, include=[])
        return len(hits["ids"]) > 0


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def _validate_embeddings(chunks: list[DocumentChunk], embeddings: np.ndarray) -> None:
    """
    Check that embeddings line up one-to-one with chunks.

    WHY THIS EXISTS:
    If the arrays were misaligned by even one row, every chunk would be
    stored with its NEIGHBOUR's vector. Nothing would crash — retrieval
    would just return the wrong text for every query. Checking shapes
    here is cheap insurance against a bug that is very hard to spot later.

    Args:
        chunks:     The chunks being inserted.
        embeddings: Their embedding matrix.

    Raises:
        ValueError: On any shape mismatch.
    """
    expected = (len(chunks), EMBEDDING_DIMENSION)
    if embeddings.ndim != 2 or embeddings.shape != expected:
        raise ValueError(
            f"embeddings shape {embeddings.shape} does not match "
            f"{len(chunks)} chunks x {EMBEDDING_DIMENSION} dims (expected {expected})."
        )


def _chunk_to_metadata(chunk: DocumentChunk) -> dict[str, str | int | float]:
    """
    Flatten a DocumentChunk into a ChromaDB-compatible metadata dict.

    ChromaDB metadata values must be scalars (str, int, float, bool) —
    lists are rejected. So the bbox list becomes four separate float fields.
    Scalars also keep every field filterable, e.g. {"bbox_y0": {"$lt": 100}}
    for "regions near the top of the page".

    Args:
        chunk: The chunk to convert.

    Returns:
        Flat metadata dict.
    """
    x0, y0, x1, y1 = chunk.bbox
    return {
        "doc_id": chunk.doc_id,
        "page_num": chunk.page_num,
        "region_label": chunk.region_label,
        "chunk_index": chunk.chunk_index,
        "token_count": chunk.token_count,
        "bbox_x0": float(x0),
        "bbox_y0": float(y0),
        "bbox_x1": float(x1),
        "bbox_y1": float(y1),
    }


def _parse_query_result(result: dict[str, Any]) -> list[RetrievedChunk]:
    """
    Convert ChromaDB's raw query output into RetrievedChunk objects.

    ChromaDB returns parallel lists-of-lists, one inner list per query
    embedding: result["ids"][0][i], result["documents"][0][i], ...
    We sent one query, so we read index [0] of each.

    Cosine DISTANCE (what ChromaDB returns) = 1 - cosine SIMILARITY,
    so similarity = 1 - distance.

    Args:
        result: The dict returned by collection.query().

    Returns:
        List of RetrievedChunk, in ChromaDB's order (nearest first).
    """
    hits: list[RetrievedChunk] = []
    rows = zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    )
    for chunk_id, text, meta, distance in rows:
        hits.append(RetrievedChunk(
            chunk_id=chunk_id,
            text=text,
            doc_id=str(meta["doc_id"]),
            page_num=int(meta["page_num"]),
            region_label=str(meta["region_label"]),
            bbox=[float(meta[k]) for k in ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1")],
            score=1.0 - float(distance),
        ))
    return hits
