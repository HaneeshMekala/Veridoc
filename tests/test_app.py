"""
tests/test_app.py — Orchestration and rendering in app.py.

WHY THIS FILE EXISTS:
app.py is where every module's output meets the user. Two things there can
fail silently: the renderer could colour the wrong sentence (claims and
sentences out of step), and it could pass LLM output into the page
unescaped (XSS). The pipeline must also degrade visibly, not crash: no API
key, a failed log write, an off-topic question.

Everything here uses fakes — no model, no GPU, no network. The pipeline's
real models are covered by the other test files.
"""

from __future__ import annotations

import pytest

from app import Turn, Veridoc, render_answer_html, render_verdict_markdown, sources_frame
from src.generator import Generator, parse_answer
from src.hallucination import (
    CONTRADICTED,
    GROUNDED,
    UNSUPPORTED,
    VerificationReport,
    VerifiedClaim,
    is_checkable,
    split_sentences,
    strip_citations,
)
from src.logger import QueryLogger


class FakeRetriever:
    """Returns fixed passages."""

    def __init__(self, hits: list) -> None:
        """Store the passages to return."""
        self.hits = hits

    def retrieve(self, question: str, region_types: list[str] | None = None) -> list:
        """Return the fixed passages (validation is tested in test_retriever.py)."""
        return self.hits


class FakeChecker:
    """Marks every checkable sentence with a status from a fixed list."""

    def __init__(self, statuses: list[str] | None = None) -> None:
        """Store the statuses to hand out, in order."""
        self.statuses = statuses or []

    def verify(self, answer) -> VerificationReport:
        """Build a report whose claims line up with the answer's sentences."""
        texts = [strip_citations(s) for s in split_sentences(answer.answer)]
        claims = [VerifiedClaim(t, [1], 0.9, 0.05, 1, "evidence", "", status, True)
                  for t, status in zip([t for t in texts if is_checkable(t)], self.statuses)]
        return VerificationReport(answer.question, answer.answer, [] if answer.abstained else claims,
                                  answer.sources, answer.abstained, "fake-nli")


class BrokenLogger:
    """A log whose disk is full."""

    def log(self, *args, **kwargs) -> int:
        """Always fail, like a real write error."""
        raise RuntimeError("disk I/O error")


def make_app(hits: list, llm, logger, statuses: list[str] | None = None) -> Veridoc:
    """
    A Veridoc with fake components.

    WHY __new__: Veridoc.__init__ loads MiniLM and DeBERTa. __new__ builds the
    object without them; we then set only the attributes ask() uses. If
    ask() starts needing a new attribute, these tests fail loudly — good.
    """
    app = Veridoc.__new__(Veridoc)
    app.retriever, app.checker, app.logger = FakeRetriever(hits), FakeChecker(statuses), logger
    app.generator = Generator(llm=llm) if llm else None
    app.generator_error = "" if llm else "No Gemini API key found."
    app._layout = None
    return app


# ---------------------------------------------------------------------------
# Orchestration: Veridoc.ask
# ---------------------------------------------------------------------------

def test_full_turn_is_verified_timed_and_logged(make_chunk, fake_llm, tmp_path) -> None:
    """Retrieve → generate → verify → log, with every stage timed."""
    app = make_app([make_chunk("Dose is 500 mg daily.")], fake_llm("Dose is 500 mg daily [1]."),
                   QueryLogger(tmp_path / "log.db"), [GROUNDED])
    turn = app.ask("  What is the dose?  ")
    assert turn.question == "What is the dose?"
    assert [c.status for c in turn.report.claims] == [GROUNDED]
    assert turn.query_id == 1 and turn.notice == ""
    t = turn.timings
    assert None not in (t.retrieval_ms, t.generation_ms, t.verification_ms)


def test_empty_retrieval_abstains_without_calling_the_llm(fake_llm, tmp_path) -> None:
    """An off-topic question costs no LLM call and cannot hallucinate."""
    llm = fake_llm("This should never be returned [1].")
    turn = make_app([], llm, QueryLogger(tmp_path / "log.db")).ask("Capital of France?")
    assert turn.answer.abstained and llm.calls == 0


def test_no_api_key_still_returns_passages(make_chunk) -> None:
    """Retrieval-only mode: passages shown, the reason stated, nothing logged."""
    turn = make_app([make_chunk("Dose is 500 mg daily.")], None, BrokenLogger()).ask("Dose?")
    assert turn.answer is None and len(turn.hits) == 1
    assert "disabled" in turn.notice and "API key" in turn.notice


def test_log_failure_keeps_the_answer_and_says_so(make_chunk, fake_llm) -> None:
    """A full disk must not throw away an answer the user already paid for."""
    app = make_app([make_chunk("Dose is 500 mg daily.")], fake_llm("Dose is 500 mg daily [1]."),
                   BrokenLogger(), [GROUNDED])
    turn = app.ask("Dose?")
    assert turn.answer is not None and turn.query_id is None
    assert "not saved" in turn.notice and "disk I/O error" in turn.notice


def test_blank_question_is_rejected(fake_llm) -> None:
    """Whitespace is not a question."""
    with pytest.raises(ValueError, match="type a question"):
        make_app([], fake_llm("x"), BrokenLogger()).ask("   ")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def rendered(make_chunk, answer_text: str, statuses: list[str]) -> str:
    """Render an answer as the UI would, with fake verdicts."""
    answer = parse_answer("q", answer_text, [make_chunk("passage")], "fake")
    return render_answer_html(answer, FakeChecker(statuses).verify(answer))


def test_each_verified_sentence_gets_its_own_colour(make_chunk) -> None:
    """Green, amber, red in order — and the short 'See below.' is left plain."""
    html = rendered(make_chunk, "Dose is 500 mg [1]. See below. Take it with juice [1]. "
                                "Max is 4000 mg [1].", [GROUNDED, UNSUPPORTED, CONTRADICTED])
    assert html.count("<span title=") == 3
    green, amber, red = (html.index(c) for c in ("rgba(34,197,94", "rgba(245,158,11", "rgba(239,68,68"))
    assert green < amber < red


def test_bullet_lines_keep_their_line_breaks(make_chunk) -> None:
    """A bulleted answer stays bulleted."""
    html = rendered(make_chunk, "- Dose is 500 mg [1]\n- Max is 2000 mg [1]", [GROUNDED, GROUNDED])
    assert "<br>" in html and html.count("<span title=") == 2


def test_llm_output_is_escaped(make_chunk) -> None:
    """
    XSS through the LLM: a PDF can contain markup the model echoes back.
    It must reach the page as text, never as a live tag.
    """
    html = rendered(make_chunk, "<img src=x onerror=alert(1)> is the dose [1].", [GROUNDED])
    assert "<img" not in html and "&lt;img" in html


def test_evidence_tooltip_is_escaped(make_chunk) -> None:
    """Document text inside the title='...' attribute can't break out of it."""
    answer = parse_answer("q", "Dose is 500 mg [1].", [make_chunk("p")], "fake")
    claim = VerifiedClaim("Dose is 500 mg.", [1], 0.9, 0.0, 1, "x' onmouseover='alert(1)", "", GROUNDED, True)
    html = render_answer_html(answer, VerificationReport("q", answer.answer, [claim], [], False, "n"))
    assert "onmouseover='alert" not in html


def test_claims_and_sentences_out_of_step_fail_loudly(make_chunk) -> None:
    """One extra claim would shift every colour by one sentence — refuse to render."""
    answer = parse_answer("q", "Dose is 500 mg [1].", [make_chunk("p")], "fake")
    extra = [VerifiedClaim("Dose is 500 mg.", status=GROUNDED), VerifiedClaim("Ghost claim here.")]
    with pytest.raises(RuntimeError, match="out of step"):
        render_answer_html(answer, VerificationReport("q", answer.answer, extra, [], False, "n"))


def test_abstention_is_rendered_plainly(make_chunk) -> None:
    """No colours for 'not found' — there is nothing to verify."""
    answer = parse_answer("q", "I could not find this information in the provided document.",
                          [make_chunk("p")], "fake")
    html = render_answer_html(answer, VerificationReport("q", answer.answer, [], [], True, "n"))
    assert "Nothing to verify" in html and "<span title=" not in html


def test_verdict_summary(make_chunk) -> None:
    """Counts per status, the wrong-citation note, and stage timings."""
    answer = parse_answer("q", "A b c [1]. D e f [9].", [make_chunk("p")], "fake")
    claims = [VerifiedClaim("A b c.", [1], status=GROUNDED, citation_ok=False),
              VerifiedClaim("D e f.", [9], status=UNSUPPORTED)]
    turn = Turn("q", [], answer, VerificationReport("q", answer.answer, claims, [], False, "n"))
    turn.timings.generation_ms = 900.0
    md = render_verdict_markdown(turn)
    assert "Groundedness: 50%" in md and "1 of 2 claims supported" in md
    assert "cite the wrong passage" in md and "[9]" in md and "generation 900 ms" in md


def test_sources_table_marks_cited_passages(make_chunk) -> None:
    """Pages are shown 1-based; long passages are truncated."""
    hits = [make_chunk("short", page=0), make_chunk("x" * 500, label="table", page=4)]
    hits[1].rerank_score = 4.2
    df = sources_frame(hits, cited=[2])
    assert df["cited"].tolist() == ["", "✓"] and df["page"].tolist() == [1, 5]
    assert df["rerank"].tolist()[1] == 4.2 and df["similarity"].tolist() == [0.7, 0.7]
    assert df["text"][1].endswith(" …") and len(df["text"][1]) == 402
