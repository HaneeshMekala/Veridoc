"""
app.py — Gradio UI that connects the whole Veridoc pipeline.

WHY THIS FILE EXISTS:
Every module so far works alone. This is where they run together for the
first time: upload a PDF → layout-aware index; ask a question → retrieve →
generate a cited answer → verify each claim with NLI → log → show the answer
with every sentence coloured by its verdict. It is also the demo: the two
novelties are only convincing when you can SEE a table-only search, and a
hallucinated sentence turning amber while its neighbours stay green.

STRUCTURE (read top to bottom):
  Veridoc         the pipeline — plain Python, no Gradio. Tests can drive it.
  render_*        turn pipeline results into HTML / tables. Pure functions.
  build_ui        Gradio layout and event wiring. The only Gradio-aware code.
Keeping those three apart means the UI can be swapped (FastAPI, CLI,
notebook) without touching the pipeline, and the pipeline can be unit-tested
without starting a web server.

Run:  python app.py   →   http://127.0.0.1:7860
"""

from __future__ import annotations

import html                          # html: stdlib escaping. Every string from the
                                     # LLM or a PDF is escaped before it becomes
                                     # HTML — see render_answer_html for why.
import time
from dataclasses import dataclass, field
from pathlib import Path

import gradio as gr                  # Gradio chosen over Streamlit: Streamlit re-runs
                                     # the whole script on every click (models would
                                     # need caching tricks to avoid reloading); Gradio
                                     # calls one function per event. Also deploys to
                                     # HuggingFace Spaces as-is.

import pandas as pd                  # pandas: gr.Dataframe renders a DataFrame with
                                     # its column names as headers. Already installed
                                     # as a Gradio dependency — no new package.

from config import GRADIO_PORT, GRADIO_SHARE, REGION_LABELS, RETRIEVAL_EXCLUDED_LABELS
from src.chunker import chunk_regions, get_chunk_stats
from src.embedder import Embedder
from src.generator import GeneratedAnswer, Generator
from src.hallucination import (
    CONTRADICTED,
    GROUNDED,
    UNSUPPORTED,
    HallucinationChecker,
    VerificationReport,
    is_checkable,
    split_sentences,
    strip_citations,
)
from src.ingestion import get_page_summary, load_pdf_pages
from src.layout import LayoutDetector
from src.logger import QueryLogger, StageTimings
from src.reranker import Reranker
from src.retriever import Retriever
from src.vectorstore import RetrievedChunk, VectorStore


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """
    Everything one question produced — the pipeline's output to the UI.

    Fields:
        question:    The question asked.
        hits:        Passages retrieved (shown even when generation is off).
        answer:      Generated answer, or None if generation is unavailable.
        report:      NLI verification of the answer, or None likewise.
        timings:     Per-stage latency.
        query_id:    Row id in the query log, or None if not logged.
        notice:      Message for the user (generation disabled, log failed).
    """
    question: str
    hits: list[RetrievedChunk]
    answer: GeneratedAnswer | None = None
    report: VerificationReport | None = None
    timings: StageTimings = field(default_factory=StageTimings)
    query_id: int | None = None
    notice: str = ""


class Veridoc:
    """
    The full pipeline: owns one instance of every model and store.

    WHY ONE OBJECT OWNS EVERYTHING:
    Each model takes seconds to load and hundreds of MB of GPU memory. They
    are created once here and shared by every request — the dependency-
    injection pattern the Retriever already uses. Tests pass in fakes.

    Usage:
        app = Veridoc()
        app.index_pdf(Path("data/samples/label.pdf"))
        turn = app.ask("What is the maximum daily dose?", region_types=["table"])
    """

    def __init__(
        self,
        generator: Generator | None = None,
        logger: QueryLogger | None = None,
        store: VectorStore | None = None,
    ) -> None:
        """
        Load every component except the layout model (see the `layout` property).

        A missing Gemini API key does NOT stop the app: indexing and retrieval
        still work, and the UI states clearly that generation is disabled.

        Args:
            generator: Answer generator. None = build Gemini from config.
            logger:    Query log. None = the default SQLite file.
            store:     Vector store. None = the default ChromaDB collection.
        """
        self.embedder = Embedder()
        self.store = store or VectorStore()
        self.retriever = Retriever(self.embedder, self.store, reranker=Reranker())
        self.checker = HallucinationChecker()
        self.logger = logger or QueryLogger()
        self._layout: LayoutDetector | None = None
        self.generator, self.generator_error = generator, ""
        if generator is None:
            try:
                self.generator = Generator()
            except RuntimeError as e:
                self.generator_error = str(e)
                print(f"[Veridoc] Generation disabled: {e}")

    @property
    def layout(self) -> LayoutDetector:
        """
        The LayoutLMv3 detector, loaded on first use.

        WHY LAZY: it is needed only to INDEX a PDF, not to answer questions,
        and it is the largest model (~500 MB of GPU memory). Asking questions
        over an existing index never pays for it.
        """
        if self._layout is None:
            self._layout = LayoutDetector()
        return self._layout

    # --- indexing -----------------------------------------------------------

    def index_pdf(self, pdf_path: Path) -> dict:
        """
        Parse, detect layout, chunk, embed and store one PDF.

        The file name is the doc_id, so re-uploading a file with the same name
        REPLACES its old chunks (VectorStore.add_chunks deletes first).

        Args:
            pdf_path: Path to the PDF.

        Returns:
            Stats: doc_id, pages, scanned pages, chunks, label counts, seconds.

        Raises:
            ValueError: If the PDF yields no text (e.g. a scan — no OCR yet).
        """
        start = time.perf_counter()
        pages = load_pdf_pages(pdf_path)
        chunks = chunk_regions(self.layout.detect_document(pages), doc_id=pdf_path.name)
        if not chunks:
            raise ValueError(
                f"No text found in {pdf_path.name}. If it is a scanned PDF, it needs "
                f"OCR, which Veridoc does not implement yet."
            )
        self.store.add_chunks(chunks, self.embedder.embed_chunks(chunks))
        page_info, chunk_info = get_page_summary(pages), get_chunk_stats(chunks)
        return {
            "doc_id": pdf_path.name,
            "pages": page_info["total_pages"],
            "scanned_pages": page_info["scanned_pages"],
            "chunks": chunk_info["total_chunks"],
            "labels": chunk_info["label_distribution"],
            "seconds": round(time.perf_counter() - start, 1),
        }

    # --- asking -------------------------------------------------------------

    def ask(self, question: str, region_types: list[str] | None = None) -> Turn:
        """
        Answer a question: retrieve → generate → verify → log.

        Each stage is timed separately, because "the app is slow" is not
        actionable but "generation takes 900 ms of 1200" is.

        Args:
            question:     The user's question.
            region_types: Region labels to search. None/empty = default filter.

        Returns:
            A Turn. If generation is disabled, it holds only the passages.

        Raises:
            ValueError:   If the question is blank or a region type is unknown.
            RuntimeError: If the LLM or NLI model fails.
        """
        if not question or not question.strip():
            raise ValueError("Please type a question.")
        region_types = region_types or None
        turn = Turn(question=question.strip(), hits=[])

        t0 = time.perf_counter()
        turn.hits = self.retriever.retrieve(turn.question, region_types=region_types)
        turn.timings.retrieval_ms = _ms_since(t0)
        if self.generator is None:
            turn.notice = f"Generation is disabled — {self.generator_error}"
            return turn

        t0 = time.perf_counter()
        turn.answer = self.generator.generate(turn.question, turn.hits)
        turn.timings.generation_ms = _ms_since(t0)
        t0 = time.perf_counter()
        turn.report = self.checker.verify(turn.answer)
        turn.timings.verification_ms = _ms_since(t0)

        self._log(turn, region_types)
        return turn

    def _log(self, turn: Turn, region_types: list[str] | None) -> None:
        """
        Write the turn to the query log without losing the answer if that fails.

        WHY NOT LET THE ERROR PROPAGATE: the user has already paid for the
        answer (an LLM call); a full disk shouldn't throw it away. But the
        failure is not silent either — it is shown in the UI as a notice.

        Args:
            turn:         A turn with answer and report filled in.
            region_types: The filter used, for the log.
        """
        try:
            turn.query_id = self.logger.log(turn.answer, turn.report,
                                            region_types, turn.timings)
        except (RuntimeError, ValueError) as e:
            turn.notice = f"The answer was not saved to the query log: {e}"
            print(f"[Veridoc] WARNING: {turn.notice}")

    # --- status -------------------------------------------------------------

    def status_markdown(self) -> str:
        """One-line status of index, generation and log, for the page header."""
        gen = (f"✅ `{self.generator.model_name}`" if self.generator
               else "⚠️ disabled — no Gemini API key (see README → *Gemini API key*)")
        return (f"**Index:** {self.store.count()} chunks &nbsp;·&nbsp; "
                f"**Generation:** {gen} &nbsp;·&nbsp; "
                f"**Log:** {self.logger.count()} questions")


def _ms_since(start: float) -> float:
    """Milliseconds elapsed since a time.perf_counter() reading."""
    return round((time.perf_counter() - start) * 1000, 1)


# ---------------------------------------------------------------------------
# RENDERING
# ---------------------------------------------------------------------------

# Colours are semi-transparent so the text keeps the theme's own colour and
# stays readable in both Gradio's light and dark mode.
_STATUS_STYLE = {
    GROUNDED:     ("rgba(34,197,94,0.18)",  "#16a34a", "supported by the document"),
    UNSUPPORTED:  ("rgba(245,158,11,0.24)", "#d97706", "NOT found in the document"),
    CONTRADICTED: ("rgba(239,68,68,0.22)",  "#dc2626", "CONTRADICTED by the document"),
}


def render_answer_html(answer: GeneratedAnswer, report: VerificationReport) -> str:
    """
    Render the answer with every verified sentence highlighted by its verdict.

    INTERVIEW ALERT — XSS through an LLM:
    NAIVE: f"<div>{answer.answer}</div>". The answer is LLM output, and the
    LLM read passages from an arbitrary PDF. A PDF containing
    "<img src=x onerror=...>" can be echoed by the model straight into your
    page — prompt injection turned into script injection. BETTER: escape every
    piece of model or document text; only our own tags are real HTML.

    Sentences are matched to claims by re-running the exact splitting that
    produced the claims, in order — so every checkable sentence gets its claim.

    Args:
        answer: The generated answer.
        report: Its verification report.

    Returns:
        HTML string for a gr.HTML component.

    Raises:
        RuntimeError: If sentences and claims fall out of step (a splitting bug).
    """
    if report.abstained:
        return _box(f"<p>{html.escape(answer.answer)}</p>"
                    "<p style='opacity:.7'><em>Nothing to verify — the answer says "
                    "the documents don't contain this.</em></p>")
    claims = iter(report.claims)
    lines = [" ".join(_render_sentence(s, claims) for s in split_sentences(line))
             for line in answer.answer.splitlines()]
    if next(claims, None) is not None:
        raise RuntimeError("Answer sentences and verified claims are out of step — "
                           "the claim splitter and the renderer disagree.")
    return _box("<br>".join(lines) + _legend())


def _render_sentence(sentence: str, claims) -> str:
    """
    One sentence as HTML: highlighted if it was verified, plain otherwise.

    Args:
        sentence: Sentence text including [n] markers.
        claims:   Iterator over the report's claims, advanced in step.

    Returns:
        An HTML fragment.
    """
    if not is_checkable(strip_citations(sentence)):
        return html.escape(sentence)
    claim = next(claims)
    bg, border, meaning = _STATUS_STYLE[claim.status]
    shown = claim.counter_evidence if claim.status == CONTRADICTED else claim.evidence
    tip = (f"{meaning}  (entailment {claim.entailment:.2f})\n"
           f"Passage [{claim.best_source}]: {shown[:300]}")
    return (f"<span title='{html.escape(tip, quote=True)}' style='background:{bg};"
            f"border-bottom:2px solid {border};border-radius:3px;padding:1px 2px;"
            f"cursor:help'>{html.escape(sentence)}</span>")


def _legend() -> str:
    """Colour key shown under the answer."""
    items = "".join(
        f"<span style='background:{bg};border-bottom:2px solid {border};"
        f"border-radius:3px;padding:1px 6px;margin-right:10px'>{meaning}</span>"
        for bg, border, meaning in _STATUS_STYLE.values())
    return (f"<div style='margin-top:14px;font-size:.85em;opacity:.85'>{items}"
            f"<br><span style='opacity:.8'>Hover a sentence to see its evidence.</span></div>")


def _box(inner: str) -> str:
    """Wrap answer HTML in a padded, readable container."""
    return f"<div style='line-height:1.9;font-size:1.02em;padding:4px 2px'>{inner}</div>"


def render_verdict_markdown(turn: Turn) -> str:
    """
    Summarize the verification in one or two lines of Markdown.

    Args:
        turn: A turn with an answer and report.

    Returns:
        Markdown text.
    """
    report, t = turn.report, turn.timings
    counts = {s: sum(c.status == s for c in report.claims) for s in _STATUS_STYLE}
    wrong_cites = sum(c.grounded and c.cited and not c.citation_ok for c in report.claims)
    parts = [f"**Groundedness: {report.groundedness:.0%}**"]
    if report.claims:
        parts.append(f"{counts[GROUNDED]} of {len(report.claims)} claims supported · "
                     f"{counts[UNSUPPORTED]} not in document · {counts[CONTRADICTED]} contradicted")
    if wrong_cites:
        parts.append(f"{wrong_cites} supported claim(s) cite the wrong passage")
    if turn.answer.invalid_citations:
        parts.append(f"cites non-existent passage(s) {turn.answer.invalid_citations}")
    timing = " · ".join(f"{name} {ms:.0f} ms" for name, ms in
                        [("retrieval", t.retrieval_ms), ("generation", t.generation_ms),
                         ("verification", t.verification_ms)] if ms is not None)
    return " — ".join(parts) + f"\n\n<sub>{timing}</sub>"


def sources_frame(hits: list[RetrievedChunk], cited: list[int]) -> pd.DataFrame:
    """
    Passages as a table: what the LLM was shown, and which it cited.

    Two scores are shown because they answer different questions:
    similarity (bi-encoder cosine) is "same topic?", and decided whether the
    passage was relevant enough to keep; rerank (cross-encoder) is "does it
    answer the question?", and decided the ORDER.

    Args:
        hits:  Retrieved passages, in rank order.
        cited: 1-based passage numbers the answer cited.

    Returns:
        DataFrame with one row per passage.
    """
    rows = [{"#": i, "cited": "✓" if i in cited else "", "page": h.page_num + 1,
             "region": h.region_label, "similarity": round(h.score, 2),
             "rerank": None if h.rerank_score is None else round(h.rerank_score, 2),
             "text": h.text if len(h.text) <= 400 else h.text[:400] + " …"}
            for i, h in enumerate(hits, start=1)]
    return pd.DataFrame(rows, columns=["#", "cited", "page", "region", "similarity",
                                       "rerank", "text"])


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_EXAMPLES = [
    "What is the maximum recommended daily dose?",
    "What are the risk factors for lactic acidosis?",
    "What is the capital of France?",          # off-topic → abstains, no LLM call
]

# LEARN THIS — The GPU is a shared resource:
# Indexing (LayoutLMv3) and answering (MiniLM + DeBERTa) both use the GPU.
# Gradio runs events in a thread pool, so two users could index and ask at the
# same moment. Giving both events the same concurrency_id with a limit of 1
# puts them in ONE queue — one GPU job at a time, like a single CUDA stream on
# a Jetson. Analytics reads only touch SQLite and stay outside that queue.
_GPU_QUEUE = {"concurrency_id": "gpu", "concurrency_limit": 1}


def build_ui(app: Veridoc) -> gr.Blocks:
    """
    Assemble the three tabs and wire their events.

    Args:
        app: A loaded Veridoc pipeline.

    Returns:
        The Gradio Blocks app (not yet launched).
    """
    with gr.Blocks(title="Veridoc") as demo:
        gr.Markdown("# Veridoc\nLayout-aware answers from medical PDFs — "
                    "every sentence checked against the source.")
        status = gr.Markdown(app.status_markdown())
        with gr.Tab("Ask"):
            _build_ask_tab(app, status)
        with gr.Tab("Index documents"):
            _build_index_tab(app, status)
        with gr.Tab("Analytics"):
            _build_analytics_tab(app, demo)
    return demo


def _build_ask_tab(app: Veridoc, status: gr.Markdown) -> None:
    """
    Question box, region filter (Novelty 1), colour-coded answer (Novelty 2).

    Args:
        app:    The pipeline.
        status: Header status line, refreshed after each question.
    """
    question = gr.Textbox(label="Question", placeholder="e.g. What is the maximum daily dose?")
    regions = gr.CheckboxGroup(
        choices=list(REGION_LABELS), label="Search only these region types",
        info=f"Leave empty to search everything except {', '.join(RETRIEVAL_EXCLUDED_LABELS)}.")
    ask_btn = gr.Button("Ask", variant="primary")
    gr.Examples(_EXAMPLES, inputs=question)
    verdict = gr.Markdown()
    answer_html = gr.HTML()
    sources = gr.Dataframe(label="Passages shown to the LLM", wrap=True, interactive=False)

    def on_ask(q: str, region_types: list[str]):
        try:
            turn = app.ask(q, region_types)
        except (ValueError, RuntimeError) as e:
            raise gr.Error(str(e)) from e
        if turn.notice:
            gr.Warning(turn.notice)
        if turn.answer is None:                           # generation disabled
            return ("", _box(f"<p><em>{html.escape(turn.notice)}</em></p>"),
                    sources_frame(turn.hits, []), app.status_markdown())
        return (render_verdict_markdown(turn), render_answer_html(turn.answer, turn.report),
                sources_frame(turn.hits, turn.answer.cited), app.status_markdown())

    outputs = [verdict, answer_html, sources, status]
    ask_btn.click(on_ask, [question, regions], outputs, **_GPU_QUEUE)
    question.submit(on_ask, [question, regions], outputs, **_GPU_QUEUE)


def _build_index_tab(app: Veridoc, status: gr.Markdown) -> None:
    """
    PDF upload → layout-aware indexing, with per-file results.

    Args:
        app:    The pipeline.
        status: Header status line, refreshed after indexing.
    """
    gr.Markdown("Upload one or more PDFs. A file with the same name as one "
                "already indexed **replaces** it. The first upload also loads the "
                "layout model (~500 MB, once).")
    files = gr.File(label="PDFs", file_types=[".pdf"], file_count="multiple")
    index_btn = gr.Button("Index", variant="primary")
    result = gr.Markdown()

    def on_index(paths: list[str] | None, progress=gr.Progress()):
        if not paths:
            raise gr.Error("Choose at least one PDF first.")
        lines = []
        for path in progress.tqdm(paths, desc="Indexing"):
            try:
                s = app.index_pdf(Path(path))
                labels = ", ".join(f"{k} {v}" for k, v in sorted(s["labels"].items()))
                lines.append(f"✅ **{s['doc_id']}** — {s['pages']} pages → {s['chunks']} "
                             f"chunks in {s['seconds']} s  \n<sub>{labels}</sub>")
            except (ValueError, RuntimeError, OSError) as e:
                lines.append(f"❌ **{Path(path).name}** — {e}")
        return "\n\n".join(lines), app.status_markdown()

    index_btn.click(on_index, [files], [result, status], **_GPU_QUEUE)


def _build_analytics_tab(app: Veridoc, demo: gr.Blocks) -> None:
    """
    Headline numbers, hallucination hotspots, and per-region evidence stats.

    Args:
        app:  The pipeline.
        demo: The Blocks app, to refresh on page load.
    """
    refresh = gr.Button("Refresh")
    summary = gr.Markdown()
    gr.Markdown("### Least-grounded answers")
    worst = gr.Dataframe(interactive=False, wrap=True)
    gr.Markdown("### Region types — retrieved vs. actually used as evidence")
    regions = gr.Dataframe(interactive=False)

    def on_refresh():
        s = app.logger.summary()
        if not s["queries"]:
            return "No questions logged yet.", pd.DataFrame(), pd.DataFrame()
        pct = lambda v: "—" if v is None else f"{v:.0%}"
        text = (f"**{s['queries']}** questions · mean groundedness **{pct(s['mean_groundedness'])}** · "
                f"fully grounded **{pct(s['fully_grounded_rate'])}** · "
                f"abstained **{pct(s['abstention_rate'])}**  \nClaims: " +
                ", ".join(f"{k} {v}" for k, v in sorted(s["claims_by_status"].items())))
        return (text, pd.DataFrame(app.logger.worst_answers()),
                pd.DataFrame(app.logger.region_stats()))

    refresh.click(on_refresh, None, [summary, worst, regions])
    demo.load(on_refresh, None, [summary, worst, regions])


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Load the pipeline and serve the UI.

    server_name="127.0.0.1" keeps the app reachable from THIS machine only.
    Medical documents and questions should not be exposed on the local
    network by default; GRADIO_SHARE=True in config.py creates a public link
    deliberately, when you want one.
    """
    app = Veridoc()
    build_ui(app).queue().launch(
        server_name="127.0.0.1", server_port=GRADIO_PORT, share=GRADIO_SHARE)


if __name__ == "__main__":
    main()
