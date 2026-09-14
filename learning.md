# learning.md — Document Intelligence RAG Project

> This file drives how Claude Code teaches while building the project, and
> doubles as my learning log. A local, git-ignored `CLAUDE.md` imports it.

## Who I am

I am Haneesh, an MSc AI student at University of Bremen. I am building this project
to genuinely learn — not just to have something on my resume. I want to understand
every decision, every library choice, and every algorithm used. Please teach me
as you build.

---

## My background

- Strong in computer vision, edge deployment (Jetson), and MLOps (Flyte, Docker)
- Familiar with PyTorch, TensorFlow, OpenCV
- Less experienced with: LLMs, RAG pipelines, NLP, vector databases
- I have built and deployed real models before, so I understand ML concepts —
  I just haven't applied them to language/document tasks yet

---

## Teaching rules — follow these always

### 1. Explain before you code

Before writing any function or module, write a short comment block (3–6 lines)
explaining:

- What this component does conceptually
- Why this approach was chosen over alternatives
- What would break if this component were missing

Example format:

```python
# WHY THIS EXISTS:
# pdfplumber extracts text while preserving layout coordinates (x, y, width, height).
# We need coordinates later for LayoutLMv3, which takes bounding boxes as input.
# PyPDF2 would also extract text but throws away position info — so we'd lose
# the spatial structure that makes layout-aware chunking possible.
```

### 2. Flag every concept I may not know

Whenever you use a concept that bridges into NLP/LLM territory — things I may
not have encountered in my CV/edge AI work — add a "LEARN THIS" comment:

```python
# LEARN THIS — Cross-encoder vs Bi-encoder:
# Bi-encoders (like sentence-transformers) encode query and document SEPARATELY,
# then compare with cosine similarity. Fast, good for retrieval over large corpora.
# Cross-encoders (like DeBERTa NLI) encode query + document TOGETHER, so they
# see the relationship between them. Slower but much more accurate — ideal for
# re-ranking or verifying a small set of retrieved chunks.
```

### 3. Never use a library without justifying it

Every new import must have a one-line comment explaining why this library
and not an alternative. Example:

```python
import chromadb  # ChromaDB chosen over FAISS: runs locally, stores metadata
                 # alongside vectors, no server setup needed for a portfolio project
```

### 4. Pause points — ask me before continuing

After completing each module, stop and ask:
"Do you want me to explain how [module] works before moving to the next one,
or shall I continue building?"

This gives me a chance to ask questions before complexity compounds.

### 5. Mark interview-critical concepts

Any concept that is commonly asked in AI/ML job interviews should be flagged:

```python
# INTERVIEW ALERT — Chunking strategy is a common interview topic.
# Recruiters often ask: "How did you decide on chunk size?"
# Your answer: "Fixed-size chunking loses semantic boundaries — a table split
# across two chunks loses its meaning. Layout-aware chunking preserves
# semantic units, which directly improves retrieval precision."
```

### 6. Show me what bad looks like

When introducing a design decision, briefly show the naive/wrong approach
and why it fails — then show the right approach. This builds intuition.

```python
# NAIVE APPROACH (don't do this):
#   text = pdf.extract_text()
#   chunks = [text[i:i+500] for i in range(0, len(text), 500)]
# PROBLEM: Splits mid-sentence, mid-table, mid-heading. A 500-char window
# has no idea that it just cut a table in half. Retrieval quality suffers.
#
# BETTER APPROACH (what we do):
#   Detect region boundaries first, then chunk within regions.
```

---

## Project overview

**Goal:** A RAG system for medical document Q&A with two research novelties:

**Novelty 1 — Layout-aware chunking**
Use LayoutLMv3 to detect document regions (tables, headers, paragraphs) and
chunk by semantic region rather than fixed character count. Store region type
as metadata in ChromaDB so retrieval can be filtered by region type.

**Novelty 2 — NLI hallucination detection**
After LLM generates an answer, split it into claims. Verify each claim against
retrieved chunks using DeBERTa NLI cross-encoder. Flag ungrounded claims in
the Gradio UI (green = grounded, red = hallucinated).

---

## Stack and why

| Component           | Library                            | Reason                                                                   |
| ------------------- | ---------------------------------- | ------------------------------------------------------------------------ |
| PDF parsing         | pdfplumber                         | Preserves bounding box coordinates needed by LayoutLMv3                  |
| Scanned PDFs        | pdf2image + PIL                    | Converts pages to images for vision model input                          |
| Layout detection    | LayoutLMv3 (HuggingFace)           | Pre-trained on document understanding, outputs region labels + bboxes    |
| Embeddings          | sentence-transformers              | Fast bi-encoder, good multilingual support, runs locally                 |
| Vector store        | ChromaDB                           | Local, metadata-aware, no server setup required                          |
| Orchestration       | LlamaIndex                         | Purpose-built for document RAG, cleaner than LangChain for this use case |
| LLM                 | Gemini Flash API                   | Generous free tier, multimodal capable for future extension              |
| Hallucination check | cross-encoder/nli-deberta-v3-small | Lightweight NLI model, runs on CPU, high accuracy for entailment         |
| UI                  | Gradio                             | Native HuggingFace Spaces support, fastest to demo                       |
| Logging             | SQLite                             | Zero-config, stores query/answer/score history for analytics angle       |

---

## Folder structure

```
doc-intelligence-rag/
├── learning.md                ← this file (teaching rules + learning log)
├── README.md                  ← project overview + demo instructions
├── requirements.txt
├── app.py                     ← Gradio UI entry point
├── config.py                  ← all constants, model names, paths
├── data/
│   └── samples/               ← sample medical PDFs for testing
├── src/
│   ├── ingestion.py           ← PDF loading, page-to-image conversion
│   ├── layout.py              ← LayoutLMv3 region detection
│   ├── chunker.py             ← layout-aware chunking logic (Novelty 1)
│   ├── embedder.py            ← sentence-transformer embedding wrapper
│   ├── vectorstore.py         ← ChromaDB setup, insert, query
│   ├── retriever.py           ← top-k retrieval with metadata filtering
│   ├── generator.py           ← LLM answer generation via LlamaIndex
│   ├── hallucination.py       ← NLI claim verification (Novelty 2)
│   └── logger.py              ← SQLite query/answer logging
├── tests/
│   ├── test_chunker.py
│   ├── test_retriever.py
│   └── test_hallucination.py
└── notebooks/
    ├── 01_layout_exploration.ipynb   ← explore LayoutLMv3 outputs visually
    ├── 02_chunking_comparison.ipynb  ← fixed vs layout-aware chunk quality
    └── 03_hallucination_eval.ipynb   ← NLI scoring on sample outputs
```

---

## Build order — follow this sequence

Build and explain one module at a time in this order:

1. `config.py` — constants and paths
2. `src/ingestion.py` — PDF loading
3. `src/layout.py` — LayoutLMv3 region detection
4. `src/chunker.py` — layout-aware chunking **(Novelty 1 core)**
5. `src/embedder.py` — embedding wrapper
6. `src/vectorstore.py` — ChromaDB wrapper
7. `src/retriever.py` — retrieval logic
8. `src/generator.py` — LLM generation
9. `src/hallucination.py` — NLI verification **(Novelty 2 core)**
10. `src/logger.py` — SQLite logging
11. `app.py` — Gradio UI connecting everything
12. `tests/` — unit tests for core modules
13. `notebooks/` — exploration notebooks
14. `README.md` — final documentation

---

## Code style rules

- Python 3.10+
- Type hints on every function signature
- Docstrings on every function (Google style)
- No function longer than 40 lines — split if needed
- No silent failures — raise descriptive exceptions with context
- Every file starts with a module-level docstring explaining what it does

---

## How to run the project

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py

# Run tests
pytest tests/
```

---

## Key concepts to understand deeply (study these)

By the end of this project, I should be able to explain all of these without notes:

- [ ] What is RAG and why it exists (vs fine-tuning)
- [ ] How vector embeddings represent semantic meaning
- [ ] Cosine similarity and why it works for retrieval
- [ ] What chunking does and why strategy matters
- [ ] How LayoutLMv3 processes documents (vision + language)
- [ ] What NLI (Natural Language Inference) is: entailment vs contradiction vs neutral
- [ ] Why cross-encoders are better than bi-encoders for verification
- [ ] What hallucination means in LLMs and why it happens
- [ ] How ChromaDB stores and indexes vectors
- [ ] What LlamaIndex does differently from raw API calls

---

## Interview prep notes (add to this as we build)

Keep a running list here of questions this project enables me to answer confidently.
Claude should add a new entry here whenever we implement something interview-relevant.

- "Walk me through your RAG pipeline architecture"
- "Why did you choose layout-aware chunking over fixed-size chunking?"
- "How did you handle hallucination in your system?"
- "What is the difference between a bi-encoder and a cross-encoder?"
- "How do you measure similarity between embeddings, and why normalize them?"
  (cosine similarity; unit vectors make cosine == dot product and remove length bias — see src/embedder.py)
- "How does a transformer turn a whole sentence into one vector?"
  (mean pooling over token embeddings ≈ Global Average Pooling in CNNs)
- "How did you make sure chunks weren't silently truncated by the embedding model?"
  (MiniLM caps input at 256 tokens incl. [CLS]/[SEP]; chunk limit set to 250 and verified at startup)
- "How does a vector database search so fast?"
  (HNSW: layered proximity graph, ~O(log n) approximate nearest-neighbour search — see src/vectorstore.py)
- "Why ChromaDB over FAISS?"
  (FAISS is only an index; ChromaDB also persists text + metadata and supports filtered search)
- "How do you handle re-ingesting an updated document?"
  (delete all chunks for that doc_id, then insert — avoids duplicates and stale evidence)
- "How do you map sub-word token predictions back to words/lines?"
  (tokenizer word_ids() alignment, then average token probabilities per line — never assume token i = word i; see src/layout.py)
- "What if the input is longer than the transformer's 512-token limit?"
  (overlapping sliding windows via return_overflowing_tokens + stride, instead of silent truncation)
- "Which layout model did you use and why not the base checkpoint?"
  (base LayoutLMv3 has no trained classification head; used a DocLayNet fine-tune, 11 region classes, F1 0.87)
- "What happens when the answer isn't in the documents?"
  (top-k always returns k chunks; a relevance floor (cosine >= 0.25) returns nothing instead, so the LLM can abstain.
   Measured: relevant 0.57–0.90, off-topic <= 0.13 — see src/retriever.py)
- "Do you filter noise at index time or query time?"
  (query time — headers/footers and <5-token fragments stay indexed but are excluded by default; keeps the option to search them)
- "How do you make the LLM stick to the retrieved context?"
  (system prompt: only-from-passages, cite [n] per sentence, fixed abstain sentence, passages tagged as data vs instructions;
   skip the LLM entirely when retrieval is empty; then verify with NLI — see src/generator.py)
- "Why didn't you set temperature to 0 for factual answers?"
  (Google recommends default 1.0 for Gemini 3 — lower can loop/degrade; grounding comes from prompt + verification. Best practices are model-specific)
- "What is 'lost in the middle'?"
  (LLMs attend best to the start/end of long contexts; passages are ordered best-first)
- "What is prompt injection and how does it apply to RAG?"
  (document text can contain instructions; passages are wrapped in tags and declared to be data in the system prompt)
- (more to be added as we build)

---

## My questions log (I will update this)

Use this section to track concepts I want to revisit or ask more about.
Claude should remind me to update this when something seems unclear.
