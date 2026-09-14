"""
src/embedder.py — Sentence-transformer embedding wrapper.

WHY THIS MODULE EXISTS:
Retrieval works by comparing MEANING, not keywords. To compare meaning with
math, every chunk (and every user query) must become a vector — a list of
384 numbers where texts with similar meaning land close together in space.
This module is the single place where text → vector conversion happens.

Without it, ChromaDB has nothing to index and retrieval cannot run.
Keeping it in one wrapper also guarantees that chunks and queries are
embedded by the SAME model with the SAME settings — if they ever differed,
query vectors and chunk vectors would live in incompatible spaces and every
similarity score would be meaningless.

LEARN THIS — How a sentence becomes ONE vector (mean pooling):
A transformer outputs one vector PER TOKEN (e.g. 40 tokens → 40 × 384).
To get a single vector for the whole text, sentence-transformers averages
all token vectors ("mean pooling"). You already know this idea from CV:
it is exactly Global Average Pooling over a CNN feature map — collapsing
a spatial grid of features into one descriptor for the whole image.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np                    # numpy: sentence-transformers returns numpy
                                      # arrays natively, and ChromaDB accepts them
                                      # directly — no torch tensors needed downstream.

from sentence_transformers import SentenceTransformer
                                      # sentence-transformers chosen over raw
                                      # HuggingFace AutoModel: it bundles the
                                      # tokenizer + transformer + mean pooling +
                                      # normalization in one call. With AutoModel
                                      # we would hand-write pooling and masking.

from config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    MAX_CHUNK_TOKENS,
)

# WHY TYPE_CHECKING:
# We only need DocumentChunk for type hints. Importing src.chunker for real
# would also load its tokenizer at import time — wasted work for code that
# only wants to embed a query. Under TYPE_CHECKING the import is seen by
# type checkers / IDEs but skipped at runtime.
if TYPE_CHECKING:
    from src.chunker import DocumentChunk


# The model wraps every input as [CLS] + text + [SEP]. The chunker counts
# only content tokens, so the embedder must reserve room for these two.
_SPECIAL_TOKENS = 2


# INTERVIEW ALERT — Cosine similarity and why we normalize:
# "How do you measure similarity between embeddings?"
# Answer: "Cosine similarity — the angle between two vectors, ignoring their
# length. Length often reflects things like text length rather than meaning,
# so direction is the better semantic signal. We L2-normalize every embedding
# to unit length at encoding time; for unit vectors, cosine similarity equals
# the plain dot product, which is cheaper and is what vector indexes optimize."
#
# NAIVE APPROACH (don't do this):
#   Store raw, un-normalized vectors and rank by dot product.
#   PROBLEM: long chunks produce larger-magnitude vectors and win on dot
#   product even when they are LESS relevant. Ranking gets biased by length.
# BETTER APPROACH (what we do):
#   normalize_embeddings=True → every vector has length 1 → only direction
#   (meaning) affects the score.

class Embedder:
    """
    Wraps a sentence-transformer bi-encoder for chunk and query embedding.

    WHY A CLASS: same reason as LayoutDetector — loading the model takes
    seconds, so we load once in __init__ and reuse it for every call.

    Usage:
        embedder = Embedder()
        chunk_vecs = embedder.embed_chunks(chunks)   # shape (n, 384)
        query_vec  = embedder.embed_query("What is the metformin dose?")  # (384,)
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        """
        Load the sentence-transformer model and verify it matches config.

        SentenceTransformer picks CUDA automatically when a GPU is available,
        otherwise it runs on CPU (fine for MiniLM — it is only 22M params).

        Args:
            model_name: HuggingFace hub id of a sentence-transformers model.

        Raises:
            ValueError: If the model's output dimension or input length
                limit is inconsistent with config.py.
        """
        print(f"[Embedder] Loading model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._validate_model()
        print(
            f"[Embedder] Ready on {self.model.device}. "
            f"dim={EMBEDDING_DIMENSION}, max_seq_length={self.model.max_seq_length}"
        )

    def _validate_model(self) -> None:
        """
        Fail fast if the loaded model disagrees with config.py.

        WHY THIS EXISTS:
        Both failure modes below are SILENT at runtime. A dimension mismatch
        only surfaces later as a ChromaDB insert error; a length mismatch
        never errors at all — the model just truncates chunks and part of
        every long chunk becomes unsearchable. Checking here turns silent
        data loss into a loud error at startup.

        Raises:
            ValueError: On dimension or sequence-length mismatch.
        """
        # sentence-transformers >= 6 renamed this method; support both names.
        get_dim = getattr(self.model, "get_embedding_dimension", None) \
            or self.model.get_sentence_embedding_dimension
        dim = get_dim()
        if dim != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Model '{self.model_name}' outputs {dim}-dim vectors but "
                f"config.EMBEDDING_DIMENSION={EMBEDDING_DIMENSION}. "
                f"Update config.py to match the model."
            )

        limit = self.model.max_seq_length
        if MAX_CHUNK_TOKENS + _SPECIAL_TOKENS > limit:
            raise ValueError(
                f"config.MAX_CHUNK_TOKENS={MAX_CHUNK_TOKENS} (+{_SPECIAL_TOKENS} "
                f"special tokens) exceeds max_seq_length={limit} of "
                f"'{self.model_name}'. Chunks would be silently truncated. "
                f"Lower MAX_CHUNK_TOKENS to at most {limit - _SPECIAL_TOKENS}."
            )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed a batch of texts into L2-normalized vectors.

        NAIVE APPROACH (don't do this):
            vecs = [model.encode(t) for t in texts]
            PROBLEM: one forward pass per text. The hardware sits mostly
            idle — like running a CNN on one image at a time instead of a
            batch. 10–50x slower on large documents.
        BETTER APPROACH (what we do):
            Pass the whole list; encode() batches it internally.

        Args:
            texts: Non-empty list of non-blank strings.

        Returns:
            float32 array of shape (len(texts), EMBEDDING_DIMENSION).

        Raises:
            ValueError: If the list is empty or contains blank strings.
        """
        if not texts:
            raise ValueError("embed_texts() received an empty list.")

        blank = [i for i, t in enumerate(texts) if not t or not t.strip()]
        if blank:
            raise ValueError(
                f"embed_texts() received blank text at indices {blank[:10]}. "
                f"Blank inputs embed to a meaningless vector — filter them first."
            )

        return self.model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,      # unit length → cosine == dot product
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
        )

    def embed_chunks(self, chunks: list[DocumentChunk]) -> np.ndarray:
        """
        Embed DocumentChunks from chunker.py.

        Row i of the result corresponds to chunks[i] — vectorstore.py relies
        on this ordering to pair each vector with its chunk id and metadata.

        Args:
            chunks: Chunks produced by chunk_regions().

        Returns:
            float32 array of shape (len(chunks), EMBEDDING_DIMENSION).
        """
        return self.embed_texts([c.text for c in chunks])

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single user query for retrieval.

        LEARN THIS — Symmetric vs asymmetric embedding models:
        MiniLM is SYMMETRIC: queries and documents are encoded identically,
        so we reuse embed_texts(). Some models (E5, BGE) are ASYMMETRIC and
        expect prefixes like "query: ..." vs "passage: ...". If you swap to
        one of those, this method is where the query prefix would go —
        forgetting it noticeably lowers retrieval accuracy.

        Args:
            query: The user's question.

        Returns:
            float32 array of shape (EMBEDDING_DIMENSION,).

        Raises:
            ValueError: If the query is blank.
        """
        if not query or not query.strip():
            raise ValueError("embed_query() received a blank query.")
        return self.embed_texts([query])[0]
