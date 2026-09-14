"""
config.py — Central configuration for the Document Intelligence RAG system.

WHY THIS FILE EXISTS:
Every project has "magic numbers" — model names, file paths, thresholds.
If you scatter these across 10 files, changing one thing (e.g. switching
embedding models) means hunting through all 10 files. A single config file
means you change it in one place and every module picks it up automatically.
This is called the "Single Source of Truth" principle.
"""

# WHY THIS EXISTS:
# pathlib.Path is the modern Python way to handle file paths.
# It works on Windows AND Linux/Mac without you having to worry about
# backslashes vs forward slashes — critical for a project that might
# run locally on Windows but deploy on a Linux server.
from pathlib import Path

# ---------------------------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------------------------

# WHY THIS EXISTS:
# __file__ is the path to THIS file (config.py).
# .parent gives us the folder containing it — the project root.
# Every other path is defined RELATIVE to this, so the project
# works regardless of where on your machine it lives.
ROOT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# DATA PATHS
# ---------------------------------------------------------------------------

DATA_DIR = ROOT_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"

# LEARN THIS — Why we define paths as variables instead of hardcoding:
# If you write "data/samples/report.pdf" directly in ingestion.py,
# and you later rename the folder, the code silently breaks.
# Defining paths here means one rename fixes everything.

# ---------------------------------------------------------------------------
# VECTOR STORE
# ---------------------------------------------------------------------------

# WHY THIS EXISTS:
# ChromaDB is a persistent vector database — it saves your embeddings to disk
# so you don't have to re-embed every PDF every time you restart the app.
# This folder is where ChromaDB writes its files.
CHROMA_DIR = ROOT_DIR / "data" / "chroma_store"

# The name of the collection inside ChromaDB that holds our document chunks.
# Think of a "collection" like a table in SQL — it groups related vectors.
CHROMA_COLLECTION_NAME = "veridoc_chunks"

# Distance metric for the vector index. ChromaDB's default is "l2".
# Our embeddings are unit-length, so L2 and cosine RANK results identically
# (||a - b||² = 2 - 2·cos(a, b)) — but cosine distance lies in [0, 2] and
# converts to an interpretable similarity score: similarity = 1 - distance.
CHROMA_DISTANCE_METRIC = "cosine"

# Max chunks sent to ChromaDB in one insert call. ChromaDB rejects batches
# above an internal limit (~5k); 1000 stays safely under it.
CHROMA_INSERT_BATCH_SIZE = 1000

# ---------------------------------------------------------------------------
# PDF INGESTION
# ---------------------------------------------------------------------------

# WHY THIS EXISTS:
# When we convert a PDF page to an image for LayoutLMv3, higher DPI = sharper
# image = better text/layout detection. 200 DPI is a sweet spot:
# high enough for accurate detection, low enough to not blow up RAM.
#
# NAIVE APPROACH (don't do this):
#   Use default DPI (72) — images are blurry, LayoutLMv3 misses small text.
# BETTER APPROACH:
#   Use 200 DPI — matches what LayoutLMv3 was trained on.
PDF_DPI = 200

# ---------------------------------------------------------------------------
# LAYOUT DETECTION (LayoutLMv3)
# ---------------------------------------------------------------------------

# WHY THIS EXISTS:
# LayoutLMv3 is a multimodal transformer trained by Microsoft on document
# understanding tasks. It takes BOTH the text AND the image of a document
# page and outputs region labels (e.g. "title", "text", "table", "figure").
# We use the HuggingFace hub name so the model auto-downloads on first run.
#
# LEARN THIS — What "pre-trained" means for us:
# Microsoft already spent weeks training this model on millions of documents.
# We're using it as-is (zero-shot inference), no fine-tuning needed.
# This is called "transfer learning" — borrowing learned knowledge.
#
# INTERVIEW ALERT — You may be asked: "Why LayoutLMv3 over a plain OCR tool?"
# Your answer: "OCR tools like Tesseract give you text but no structural
# understanding. LayoutLMv3 understands that a block of text IS a table,
# or IS a section header — that semantic structure is what lets us chunk
# by meaning rather than by character count."
#
# WHY THIS CHECKPOINT (and not "microsoft/layoutlmv3-base"):
# The base model has NO trained classification head — its region labels
# would be random. This checkpoint is layoutlmv3-base fine-tuned for token
# classification on DocLayNet-large (80k human-annotated pages from reports,
# manuals, papers, patents…), reported F1 = 0.87. It predicts 11 labels:
#   Caption, Footnote, Formula, List-item, Page-footer, Page-header,
#   Picture, Section-header, Table, Text, Title
# It was trained at LINE level (every token carries its text LINE's bbox),
# so layout.py groups words into lines before inference to match.
# License note: inherits layoutlmv3-base's CC BY-NC-SA 4.0 (non-commercial).
LAYOUTLM_MODEL_NAME = "Kwan0/layoutlmv3-base-finetune-DocLayNet-100k"

# LayoutLMv3 expects bounding boxes normalized to a 0–1000 scale.
# This is just the convention the model was trained with.
LAYOUTLM_BBOX_SCALE = 1000

# Max tokens per forward pass (LayoutLMv3's position-embedding limit).
LAYOUTLM_MAX_SEQ_LENGTH = 512

# A dense page can exceed 512 tokens. Instead of truncating (and losing the
# bottom of the page), we run overlapping 512-token windows. STRIDE is how
# many tokens consecutive windows share, so lines at a window edge are still
# seen with context on both sides.
LAYOUTLM_STRIDE = 128

# Confidence threshold: if LayoutLMv3 is less than 50% confident about
# a region label, we treat it as generic "text" to avoid noisy metadata.
LAYOUTLM_CONFIDENCE_THRESHOLD = 0.5

# The region label LayoutLMv3 assigns when it can't confidently classify.
LAYOUTLM_DEFAULT_LABEL = "text"

# LINE GROUPING — two words are on the same text line if their tops differ
# by <= LINE_Y_TOLERANCE points AND the horizontal gap between them is
# <= LINE_MAX_GAP_RATIO x word height. The gap rule stops two COLUMNS that
# share a baseline from being glued into one line (column gutters are much
# wider than a normal space between words).
LINE_Y_TOLERANCE = 3.0
LINE_MAX_GAP_RATIO = 1.5

# REGION GROUPING — a line joins an existing same-label region only if it
# sits directly below it (horizontal overlap) with a vertical gap of at most
# REGION_MAX_GAP_RATIO x line height. Bigger gaps = new paragraph/block.
REGION_MAX_GAP_RATIO = 1.0

# ---------------------------------------------------------------------------
# CHUNKING
# ---------------------------------------------------------------------------

# INTERVIEW ALERT — Chunking strategy is one of the most common RAG
# interview questions. Be ready to explain this clearly.
#
# WHY MAX_CHUNK_TOKENS = 250:
# all-MiniLM-L6-v2 (see EMBEDDINGS below) truncates input at 256 tokens —
# its underlying BERT supports 512, but sentence-transformers caps it at 256
# because the model was TRAINED on short texts and quality degrades beyond.
# Those 256 include 2 special tokens ([CLS] + [SEP]) that the chunker does
# not count, so content must be <= 254. 250 leaves a small safety margin.
# If a chunk is longer, the model silently truncates it — you lose content
# without any warning. embedder.py verifies this limit at startup.
#
# LEARN THIS — What is a "token"?
# Tokenizers split text into sub-word units called tokens. The word
# "unhappiness" might be 3 tokens: "un", "happi", "ness". On average,
# 1 token ≈ 4 characters in English. So 250 tokens ≈ ~1000 characters.
MAX_CHUNK_TOKENS = 250

# When a single layout region is LONGER than 512 tokens, we have to split it.
# CHUNK_OVERLAP means the last N tokens of one sub-chunk are repeated at the
# start of the next. This preserves context at the boundary.
#
# NAIVE APPROACH (no overlap):
#   "...end of sentence A. [SPLIT] Start of sentence B..."
#   The retriever might fetch only the second chunk, missing the context
#   from sentence A that makes B make sense.
# BETTER APPROACH (with overlap):
#   Chunk 1: "...end of sentence A."
#   Chunk 2: "end of sentence A. Start of sentence B..."  ← repeated context
CHUNK_OVERLAP_TOKENS = 50

# ---------------------------------------------------------------------------
# EMBEDDINGS
# ---------------------------------------------------------------------------

# WHY THIS MODEL:
# sentence-transformers/all-MiniLM-L6-v2 is a small (22M parameter) but
# accurate bi-encoder. It runs on CPU in ~10ms per chunk, which is fast
# enough for a portfolio project. Larger models like 'all-mpnet-base-v2'
# are more accurate but 5x slower.
#
# LEARN THIS — What is a "bi-encoder"?
# It encodes each text INDEPENDENTLY into a fixed-size vector (embedding).
# You pre-compute embeddings for all chunks, store them in ChromaDB.
# At query time, you embed the query and find the closest chunk vectors.
# "Closest" = most semantically similar. This is the core of RAG retrieval.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# The size of the embedding vector this model outputs.
# This is fixed by the model architecture — you can't change it.
# We store it here so ChromaDB knows how many dimensions to expect.
EMBEDDING_DIMENSION = 384

# How many chunks to push through the model in one forward pass.
# Same trade-off as batch size in CV inference: bigger = better hardware
# utilization, but more RAM/VRAM. 32 is safe on CPU and small GPUs.
EMBEDDING_BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# RETRIEVAL
# ---------------------------------------------------------------------------

# How many chunks to fetch from ChromaDB for each query.
# More chunks = more context for the LLM, but also more noise and slower NLI.
# 5 is a common default in RAG literature.
TOP_K_RETRIEVAL = 5

# Every region label the layout model can emit (DocLayNet classes, cleaned
# by layout._clean_label). The retriever validates user filters against
# this list — a typo like "tables" would otherwise silently match nothing.
REGION_LABELS = (
    "caption", "footnote", "formula", "list_item", "page_footer",
    "page_header", "picture", "section_header", "table", "text", "title",
)

# Region types skipped by default at query time. Running headers/footers
# ("Page 3 of 12", journal name) repeat on every page and match many queries
# while carrying no answer. They stay in the index — an explicit
# region_types=["page_header"] filter can still reach them.
RETRIEVAL_EXCLUDED_LABELS = ("page_header", "page_footer")

# Chunks shorter than this are skipped at query time. Layout detection
# produces tiny fragments from text inside figures ("𝑖 𝑖", ", ,"); they
# score deceptively well against short queries but contain no evidence.
RETRIEVAL_MIN_CHUNK_TOKENS = 5

# Minimum cosine similarity for a chunk to count as relevant at all.
# If nothing clears this bar, the retriever returns NO chunks — letting the
# generator answer "not found in the document" instead of guessing from
# unrelated text. For MiniLM, unrelated text typically scores < 0.2 and
# relevant passages > 0.4; 0.25 is a conservative starting point —
# calibrate it in a notebook on your own documents and questions.
RETRIEVAL_MIN_SCORE = 0.25

# ---------------------------------------------------------------------------
# LLM (Gemini Flash)
# ---------------------------------------------------------------------------

# WHY GEMINI FLASH:
# Flash models have a free tier on the Gemini API — zero cost for a
# low-traffic portfolio project — and are multimodal, useful if we later
# add image-based Q&A. gemini-3.8-flash is the newest stable (GA) Flash
# model as of Sept 2026. (The original choice, gemini-1.5-flash, is retired.)
# Check https://ai.google.dev/gemini-api/docs/models when revisiting.
LLM_MODEL_NAME = "gemini-3.8-flash"

# Maximum tokens the LLM may generate per response.
# WHY 2048 when answers are ~100 words: Gemini 3 models THINK before
# answering, and those hidden reasoning tokens are generated output too.
# A tight limit (e.g. 512) can be used up by thinking, cutting the visible
# answer short. Gemini then stops with finish_reason=MAX_TOKENS, which
# generator.py reports as an error rather than returning half an answer.
LLM_MAX_OUTPUT_TOKENS = 2048

# How much the model reasons before answering: "minimal" | "low" | "medium" | "high".
# Grounded Q&A is mostly "find it in the passages and restate it" — little
# multi-step reasoning — so "low" cuts latency and thinking-token use.
LLM_THINKING_LEVEL = "low"

# Input window of the model (~1M tokens for Gemini Flash). Only used as
# metadata by LlamaIndex; giving it explicitly avoids a network call to
# look it up when the LLM object is created.
LLM_CONTEXT_WINDOW = 1_048_576

# LEARN THIS — Temperature, and why we DON'T set it:
# Temperature rescales the next-token probabilities before sampling:
# near 0 → almost always the top token (deterministic); 1.0 → sample from the
# model's actual distribution. The classic advice for factual RAG is "set it
# low". BUT Google explicitly recommends keeping temperature at its default
# 1.0 for all Gemini 3 models — lowering it can cause looping or degraded
# output, because these models were tuned around their reasoning at 1.0.
# So grounding is enforced by the PROMPT (answer only from the passages,
# cite every sentence) and verified by the NLI checker, not by temperature.
# Lesson: "best practice" parameters are model-specific — read the model card.

# Text the model must return verbatim when the passages lack the answer.
# A fixed sentinel lets the code detect abstention reliably.
LLM_NOT_FOUND_ANSWER = "I could not find this information in the provided document."

# ---------------------------------------------------------------------------
# HALLUCINATION DETECTION (NLI)
# ---------------------------------------------------------------------------

# WHY THIS MODEL:
# cross-encoder/nli-deberta-v3-small is a cross-encoder NLI model.
# It takes (claim, evidence) as a pair and outputs probabilities for:
# ENTAILMENT (claim is supported), NEUTRAL, CONTRADICTION (claim is false).
# DeBERTa-v3 is the state-of-the-art small NLI architecture as of 2024.
#
# LEARN THIS — Cross-encoder vs Bi-encoder (INTERVIEW ALERT):
# Bi-encoder (used for retrieval): encodes query and document SEPARATELY.
#   Fast, but misses subtle relationships between the two texts.
# Cross-encoder (used for NLI): encodes query + document TOGETHER.
#   Slower, but sees the full interaction — much better for "does this
#   evidence actually support this claim?" verification.
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

# If entailment probability >= this threshold, the claim is "grounded".
# Below this = flag as potentially hallucinated.
NLI_ENTAILMENT_THRESHOLD = 0.5

# NLI models output 3 scores summing to 1.0: [contradiction, neutral, entailment]
# The index of "entailment" in the output array for this specific model.
NLI_ENTAILMENT_INDEX = 2

# ---------------------------------------------------------------------------
# LOGGING (SQLite)
# ---------------------------------------------------------------------------

# WHY SQLITE:
# SQLite is a file-based database — zero server setup, zero configuration.
# We log every query + answer + NLI scores so we can later analyze:
# "Which questions get the most hallucinated answers?"
# "Which document regions are retrieved most often?"
SQLITE_DB_PATH = ROOT_DIR / "data" / "query_log.db"

# ---------------------------------------------------------------------------
# GRADIO UI
# ---------------------------------------------------------------------------

# The port Gradio serves on. 7860 is the Gradio default.
GRADIO_PORT = 7860

# If True, Gradio creates a public tunnel URL (useful for demos/sharing).
# Set to False for local-only development.
GRADIO_SHARE = False
