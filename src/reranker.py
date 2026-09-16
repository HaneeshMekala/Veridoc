"""
src/reranker.py — Cross-encoder re-ranking of retrieved passages.

WHY THIS MODULE EXISTS:
The bi-encoder (src/embedder.py) finds candidates fast, but it compresses a
whole passage into one vector BEFORE seeing the question — so it matches
topic, not answer. On the sample FDA label every chunk is "about metformin",
scores bunch in a 0.58–0.66 band, and short chunks that are mostly the drug
name outrank the passage that actually answers the question.

This module takes the bi-encoder's top candidates and re-orders them with a
cross-encoder that reads (question, passage) TOGETHER. Measured on 18 labelled
questions: MRR 0.47 → 0.79, answer in the top 5 for 17 of 18 instead of 12.
Without it, the LLM is often handed the wrong 5 passages and, correctly,
answers "not found" — or worse, answers from a near-miss.

INTERVIEW ALERT — "What is two-stage retrieval (retrieve-then-rerank)?"
Answer: "Stage 1, a bi-encoder: document vectors are precomputed, so searching
millions of chunks is one ANN lookup — fast, but coarse. Stage 2, a
cross-encoder over only the top ~20: it runs one forward pass per
(question, passage) pair, too slow for the whole corpus but far more precise,
because attention sees both texts at once. You get the recall of the first
and the precision of the second." In CV terms: a cheap region-proposal stage
followed by an expensive classifier on the proposals — Faster R-CNN's shape.
"""

from __future__ import annotations

import numpy as np                   # numpy: argsort over the score vector.

from sentence_transformers import CrossEncoder
                                     # CrossEncoder chosen here, although the NLI
                                     # checker uses raw transformers: for RANKING only
                                     # the ORDER of scores matters, so we need no label
                                     # mapping and no softmax — CrossEncoder's one-call
                                     # predict() is exactly enough. NLI needed both.

from config import RERANKER_MODEL_NAME
from src.vectorstore import RetrievedChunk


class Reranker:
    """
    Re-orders passages by cross-encoder relevance to a question.

    Usage:
        reranker = Reranker()
        best_first = reranker.rerank("When is metformin contraindicated?", hits)
    """

    def __init__(self, model_name: str = RERANKER_MODEL_NAME) -> None:
        """
        Load the cross-encoder (GPU if available, else CPU).

        Args:
            model_name: HuggingFace id of a sentence-transformers CrossEncoder.

        Raises:
            ValueError: If the model outputs more than one score per pair —
                that is a classifier (e.g. NLI), not a relevance ranker.
        """
        print(f"[Reranker] Loading model: {model_name}")
        self.model_name = model_name
        self.model = CrossEncoder(model_name)
        num_labels = self.model.model.config.num_labels
        if num_labels != 1:
            raise ValueError(
                f"'{model_name}' outputs {num_labels} scores per pair; a re-ranker must "
                f"output exactly 1 relevance score. Check config.RERANKER_MODEL_NAME."
            )
        print(f"[Reranker] Ready on {self.model.device}.")

    def rerank(self, query: str, hits: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """
        Score every passage against the query and sort best first.

        NAIVE APPROACH (don't do this):
            Re-rank the whole collection with the cross-encoder.
            PROBLEM: one forward pass per chunk per question — 300 chunks is
            fine, a 300,000-chunk corpus is minutes per question.
        BETTER (what we do): re-rank only the bi-encoder's top candidates.

        The scores are raw logits: meaningful for ORDERING passages for one
        question, not as an absolute "relevant / irrelevant" cut-off. That is
        why the relevance floor stays on the calibrated-by-measurement cosine
        score in the retriever.

        Args:
            query: The user's question.
            hits:  Candidate passages from the vector store.

        Returns:
            The same passages with rerank_score set, highest first.
        """
        if not hits:
            return []
        scores = self.model.predict([(query, h.text) for h in hits], show_progress_bar=False)
        for hit, score in zip(hits, np.atleast_1d(scores)):
            hit.rerank_score = float(score)
        return [hits[i] for i in np.argsort(-np.atleast_1d(scores), kind="stable")]
