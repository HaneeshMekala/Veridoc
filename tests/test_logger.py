"""
tests/test_logger.py — The query log records faithfully and analyses correctly.

WHY THIS FILE EXISTS:
The log is the project's evaluation data: every groundedness number in a
report comes from it. A logging bug doesn't crash the app — it quietly
produces wrong statistics. These tests pin the analytics to hand-computed
values, and check the database-level guarantees the design relies on:
atomic writes, CHECK constraints, cascading deletes, thread safety, and
SQL-injection safety.

No model needed: VerificationReports are built by hand.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from src.generator import GeneratedAnswer
from src.hallucination import CONTRADICTED, GROUNDED, UNSUPPORTED, VerificationReport, VerifiedClaim
from src.logger import QueryLogger, StageTimings


@pytest.fixture
def log(tmp_path: Path) -> QueryLogger:
    """
    A fresh log database per test.

    LEARN THIS — tmp_path is a built-in pytest fixture: a new empty directory
    for each test, deleted afterwards. Tests never share (or corrupt) state.
    """
    return QueryLogger(tmp_path / "log.db")


@pytest.fixture
def interaction(make_chunk):
    """Factory: interaction(question, statuses, ...) -> (GeneratedAnswer, VerificationReport)."""
    sources = [make_chunk("passage one", "text", 0, 0.8, "c1"),
               make_chunk("passage two", "table", 1, 0.7, "c2"),
               make_chunk("passage three", "page_header", 2, 0.3, "c3")]

    def make(question: str, statuses: list[str], best: list[int] | None = None,
             cited: list[int] = (1, 2), abstained: bool = False):
        best = best or [1] * len(statuses)
        claims = [VerifiedClaim(f"claim {i} text", [b], 0.9 if s == GROUNDED else 0.1,
                                0.8 if s == CONTRADICTED else 0.05, b, "ev", "", s, s == GROUNDED)
                  for i, (s, b) in enumerate(zip(statuses, best))]
        srcs = [] if abstained else sources
        text = f"Answer to {question}"
        answer = GeneratedAnswer(question, text, srcs, list(cited), [], abstained, "fake-llm")
        return answer, VerificationReport(question, text, claims, srcs, abstained, "fake-nli")
    return make


def db_rows(log: QueryLogger, sql: str) -> list[tuple]:
    """Run raw SQL against the log file (bypassing QueryLogger) to check what was stored."""
    with sqlite3.connect(log.db_path) as conn:
        return conn.execute(sql).fetchall()


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def test_one_interaction_writes_all_three_tables(log, interaction) -> None:
    """One question → 1 query row, 3 passage rows, 2 claim rows, with timings."""
    qid = log.log(*interaction("dose?", [GROUNDED, UNSUPPORTED]),
                  region_filter=["table"], timings=StageTimings(10.0, 900.0, 250.0))
    assert qid == 1
    assert db_rows(log, "SELECT region_filter, generation_ms, n_claims, n_flagged FROM queries") \
        == [('["table"]', 900.0, 2, 1)]
    assert db_rows(log, "SELECT COUNT(*) FROM retrieved_chunks")[0][0] == 3
    assert db_rows(log, "SELECT position, status FROM claims") == [(1, GROUNDED), (2, UNSUPPORTED)]


def test_passage_text_is_stored_not_just_ids(log, interaction) -> None:
    """Re-indexing changes chunk ids; the log must still show what the model saw."""
    log.log(*interaction("dose?", [GROUNDED]))
    assert db_rows(log, "SELECT text FROM retrieved_chunks WHERE rank = 2")[0][0] == "passage two"


def test_report_for_a_different_answer_is_rejected(log, interaction) -> None:
    """Mixing up two requests' objects would log a wrong verdict against an answer."""
    answer, _ = interaction("dose?", [GROUNDED])
    _, other_report = interaction("side effects?", [GROUNDED])
    with pytest.raises(ValueError, match="different answer"):
        log.log(answer, other_report)


def test_failed_write_rolls_back_the_whole_interaction(log, interaction) -> None:
    """
    A claim with an invalid status violates a CHECK constraint half-way through.
    The query row written before it must be rolled back too — otherwise the
    log would hold an answer with "zero hallucinations" that were never recorded.
    """
    answer, report = interaction("atomic?", [GROUNDED])
    report.claims[0].status = "grounde"                       # typo → CHECK fails
    with pytest.raises(RuntimeError, match="CHECK"):
        log.log(answer, report)
    assert log.count() == 0


def test_sql_injection_is_stored_as_plain_text(log, interaction) -> None:
    """Parameterized queries: the classic payload is data, and the table survives."""
    evil = "'); DROP TABLE queries; --"
    log.log(*interaction(evil, [GROUNDED]))
    assert log.recent(1)[0]["question"] == evil
    assert log.count() == 1


def test_deleting_a_query_cascades(log, interaction) -> None:
    """GDPR-style erasure: one DELETE removes the question, its passages and claims."""
    qid = log.log(*interaction("delete me", [GROUNDED, UNSUPPORTED]))
    with sqlite3.connect(log.db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("DELETE FROM queries WHERE id = ?", (qid,))
    assert db_rows(log, "SELECT (SELECT COUNT(*) FROM claims) + "
                        "(SELECT COUNT(*) FROM retrieved_chunks)")[0][0] == 0


def test_concurrent_writes_from_threads(log, interaction) -> None:
    """Gradio serves requests from a thread pool: 8 threads × 10 writes, no errors, no loss."""
    errors: list[Exception] = []

    def worker(n: int) -> None:
        try:
            for k in range(10):
                log.log(*interaction(f"t{n}-{k}", [GROUNDED]))
        except Exception as e:                              # collected, asserted below
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and log.count() == 80


# ---------------------------------------------------------------------------
# Analytics — checked against hand-computed values
# ---------------------------------------------------------------------------

@pytest.fixture
def filled(log, interaction) -> QueryLogger:
    """Three questions: fully grounded, 1-of-3 grounded, and an abstention."""
    log.log(*interaction("dose?", [GROUNDED, GROUNDED], best=[2, 1]))
    log.log(*interaction("side effects?", [GROUNDED, UNSUPPORTED, CONTRADICTED], best=[2, 1, 1]))
    log.log(*interaction("weather?", [], cited=[], abstained=True))
    return log


def test_summary(filled) -> None:
    """Mean groundedness averages ANSWERED questions only: (1 + 1/3) / 2."""
    s = filled.summary()
    assert (s["queries"], s["abstained"]) == (3, 1)
    assert s["mean_groundedness"] == pytest.approx((1 + 1 / 3) / 2)
    assert s["fully_grounded_rate"] == 0.5
    assert s["abstention_rate"] == pytest.approx(1 / 3)
    assert s["claims_by_status"] == {GROUNDED: 3, UNSUPPORTED: 1, CONTRADICTED: 1}


def test_summary_of_an_empty_log(log) -> None:
    """No questions yet: rates are None, not a ZeroDivisionError."""
    s = log.summary()
    assert s["queries"] == 0 and s["mean_groundedness"] is None and s["abstention_rate"] is None


def test_worst_answers_skip_abstentions(filled) -> None:
    """'Not found' is correct behaviour, not a hotspot; worst answer comes first."""
    assert [r["question"] for r in filled.worst_answers()] == ["side effects?", "dose?"]


def test_region_stats_attribute_evidence_to_region_types(filled) -> None:
    """Grounded claims with best_source 2 were supported by the table passage."""
    stats = {r["region_label"]: r for r in filled.region_stats()}
    assert stats["table"]["grounded_claims"] == 2        # "dose?" claim 1, "side effects?" claim 1
    assert stats["text"]["grounded_claims"] == 1
    assert stats["page_header"]["grounded_claims"] == 0
    assert stats["table"]["cited_rate"] == 1.0 and stats["page_header"]["cited_rate"] == 0.0


@pytest.mark.parametrize("limit", [0, -1, 2.5, True])
def test_bad_limits_are_rejected(log, limit) -> None:
    """SQLite reads LIMIT -1 as 'no limit' — a bad value would silently return everything."""
    with pytest.raises(ValueError):
        log.recent(limit)


def test_newer_schema_version_is_refused(log) -> None:
    """Opening a log written by a different schema version fails with a clear message."""
    with sqlite3.connect(log.db_path) as conn:
        conn.execute("PRAGMA user_version = 99")
    with pytest.raises(RuntimeError, match="v99"):
        QueryLogger(log.db_path)
