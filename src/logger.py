"""
src/logger.py — SQLite log of every question, passage, answer and claim verdict.

WHY THIS MODULE EXISTS:
Every other module handles ONE question and forgets it. This one remembers.
Each interaction — the question, the passages retrieved, the answer, and the
NLI verdict on every claim — is written to a local SQLite file, so the system
can be judged over many questions instead of one demo at a time:
  "What fraction of answers are fully grounded?"
  "Which questions produce the most hallucinated claims?"
  "Do table regions actually supply evidence, or do they only get retrieved?"
Without it, both novelties stay anecdotes — there would be no way to show
that layout-aware chunking or NLI checking made a measurable difference.

LEARN THIS — A log is an evaluation dataset you get for free:
In CV you evaluate on a held-out labelled set. RAG systems rarely have one at
the start, so the usual practice is to log real interactions and mine them:
low-groundedness answers become the test cases for the next iteration. This
is why the log stores what the model actually SAW (passage text), not just
chunk ids — re-ingesting a document changes its chunks, and a log that can't
reproduce the model's input can't explain its output.

PRIVACY NOTE (INTERVIEW ALERT — "Any concerns with logging user queries?"):
Questions about medical documents can contain personal health data, which is
special-category data under GDPR. This log is local and git-ignored. A deployed
version would need consent, a retention period and a way to delete a user's
rows — ON DELETE CASCADE below makes that last part a single DELETE.
"""

from __future__ import annotations

import json                          # json: SQLite has no array type; the few
                                     # list-valued fields (citations, region filter)
                                     # are stored as JSON text. A child table for a
                                     # 0–3 element list would be overkill.

import sqlite3                       # sqlite3: Python standard library — no install,
                                     # no server, the whole database is one file.
                                     # Chosen over SQLAlchemy (an ORM is overkill for
                                     # 3 tables written by one module), a JSON-lines
                                     # file (no GROUP BY / JOIN for the analytics),
                                     # and Postgres (needs a running server).

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from config import SQLITE_BUSY_TIMEOUT_S, SQLITE_DB_PATH

if TYPE_CHECKING:                    # type hints only. Importing src.hallucination
    from src.generator import GeneratedAnswer        # for real would load torch
    from src.hallucination import VerificationReport  # just to write rows.


# Bump when the table layout changes. Stored in the file's PRAGMA user_version
# so an old database is detected instead of failing on a missing column later.
_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# SCHEMA
# ---------------------------------------------------------------------------

# INTERVIEW ALERT — "How did you design the log schema?"
#
# NAIVE APPROACH (don't do this):
#   CREATE TABLE log (id INTEGER, data TEXT)   -- json.dumps(everything)
#   PROBLEM: every analytics question becomes "load every row into Python and
#   parse JSON". "Which region types supply evidence for grounded claims?"
#   needs a join between claims and passages — impossible inside the database.
#
# BETTER APPROACH (what we do): normalize into three tables, one per entity.
#   queries           one row per question asked
#   retrieved_chunks  one row per passage shown to the LLM   (many per query)
#   claims            one row per verified sentence          (many per query)
#   Now that question is one SQL JOIN, and SQLite does the aggregation.
#
# The CHECK constraints make the database itself reject impossible values
# (a groundedness of 1.7, a status of "grounde") — a bug fails loudly at write
# time instead of quietly skewing every average computed later.
# The status list must match the constants in src/hallucination.py.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at        TEXT    NOT NULL,          -- UTC, ISO-8601
    question          TEXT    NOT NULL,
    answer            TEXT    NOT NULL,
    llm_model         TEXT    NOT NULL,
    nli_model         TEXT    NOT NULL,
    abstained         INTEGER NOT NULL CHECK (abstained IN (0, 1)),
    groundedness      REAL    NOT NULL CHECK (groundedness BETWEEN 0 AND 1),
    n_claims          INTEGER NOT NULL,
    n_flagged         INTEGER NOT NULL,
    invalid_citations TEXT    NOT NULL,          -- JSON list, e.g. "[7]"
    region_filter     TEXT,                      -- JSON list; NULL = default filter
    retrieval_ms      REAL,
    generation_ms     REAL,
    verification_ms   REAL
);

CREATE TABLE IF NOT EXISTS retrieved_chunks (
    query_id      INTEGER NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
    rank          INTEGER NOT NULL,              -- 1-based: the [n] the LLM saw
    chunk_id      TEXT    NOT NULL,
    doc_id        TEXT    NOT NULL,
    page_num      INTEGER NOT NULL,              -- 0-based, as in RetrievedChunk
    region_label  TEXT    NOT NULL,
    score         REAL    NOT NULL,
    cited         INTEGER NOT NULL CHECK (cited IN (0, 1)),
    text          TEXT    NOT NULL,
    PRIMARY KEY (query_id, rank)
);

CREATE TABLE IF NOT EXISTS claims (
    query_id          INTEGER NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
    position          INTEGER NOT NULL,          -- 1-based order in the answer
    text              TEXT    NOT NULL,
    status            TEXT    NOT NULL
                      CHECK (status IN ('grounded', 'unsupported', 'contradicted')),
    entailment        REAL    NOT NULL,
    contradiction     REAL    NOT NULL,
    best_source       INTEGER,                   -- = retrieved_chunks.rank
    citation_ok       INTEGER NOT NULL CHECK (citation_ok IN (0, 1)),
    cited             TEXT    NOT NULL,          -- JSON list
    evidence          TEXT    NOT NULL,
    counter_evidence  TEXT    NOT NULL,
    PRIMARY KEY (query_id, position)
);

CREATE INDEX IF NOT EXISTS idx_queries_created ON queries(created_at);
CREATE INDEX IF NOT EXISTS idx_chunks_region   ON retrieved_chunks(region_label);
CREATE INDEX IF NOT EXISTS idx_claims_status   ON claims(status);
"""


# ---------------------------------------------------------------------------
# ANALYTICS QUERIES
# ---------------------------------------------------------------------------

_SUMMARY_SQL = """
SELECT COUNT(*)                                                 AS queries,
       COALESCE(SUM(abstained), 0)                              AS abstained,
       AVG(CASE WHEN abstained = 0 THEN groundedness END)       AS mean_groundedness,
       AVG(CASE WHEN abstained = 0 THEN groundedness = 1.0 END) AS fully_grounded_rate
FROM queries
"""

_STATUS_SQL = "SELECT status, COUNT(*) AS n FROM claims GROUP BY status"

_WORST_SQL = """
SELECT id, created_at, question, groundedness, n_claims, n_flagged
FROM queries
WHERE abstained = 0 AND n_claims > 0
ORDER BY groundedness ASC, id DESC
LIMIT ?
"""

_RECENT_SQL = """
SELECT id, created_at, question, groundedness, n_claims, n_flagged, abstained
FROM queries
ORDER BY id DESC
LIMIT ?
"""

# For each region type: how often it is retrieved, how relevant it scores,
# how often the LLM cites it, and how many GROUNDED claims it supplied the
# evidence for. The last column is the Novelty-1 question: a region type that
# is retrieved a lot but never supports a grounded claim is noise in the index.
_REGION_SQL = """
SELECT rc.region_label           AS region_label,
       COUNT(*)                  AS retrieved,
       ROUND(AVG(rc.score), 3)   AS mean_score,
       ROUND(AVG(rc.cited), 3)   AS cited_rate,
       COALESCE(SUM(g.n), 0)     AS grounded_claims
FROM retrieved_chunks rc
LEFT JOIN (SELECT query_id, best_source, COUNT(*) AS n
           FROM claims WHERE status = 'grounded'
           GROUP BY query_id, best_source) g
       ON g.query_id = rc.query_id AND g.best_source = rc.rank
GROUP BY rc.region_label
ORDER BY retrieved DESC
"""


# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class StageTimings:
    """
    Wall-clock time of each pipeline stage for one question, in milliseconds.

    Measured by app.py and stored per query, so the log can answer "where does
    the time go?" — typically generation (a network call) dominates, and
    verification cost grows with answer length × passage count.

    Fields:
        retrieval_ms:    Embedding the question + vector search.
        generation_ms:   LLM call.
        verification_ms: NLI scoring of all claims.
    """
    retrieval_ms: float | None = None
    generation_ms: float | None = None
    verification_ms: float | None = None


# ---------------------------------------------------------------------------
# LOGGER
# ---------------------------------------------------------------------------

class QueryLogger:
    """
    Writes interactions to SQLite and answers analytics questions about them.

    LEARN THIS — Why every method opens its own connection:
    A sqlite3 connection refuses to be used from any thread other than the
    one that created it. Gradio runs each button click in a worker thread.
    NAIVE: self.conn = sqlite3.connect(...) in __init__ → the first click
        raises "SQLite objects created in a thread can only be used in that
        same thread". Silencing that with check_same_thread=False lets two
        threads interleave statements on ONE connection — a real race.
    BETTER: one short-lived connection per operation. For SQLite, "connecting"
        is opening a local file (no network, no handshake), so it costs about
        a millisecond, and each thread only ever touches its own connection.

    Usage:
        log = QueryLogger()
        query_id = log.log(answer, report, timings=StageTimings(12.0, 900.0, 270.0))
        log.summary()        # {'queries': 42, 'mean_groundedness': 0.91, ...}
        log.region_stats()   # which region types supply evidence
    """

    def __init__(self, db_path: Path | str = SQLITE_DB_PATH) -> None:
        """
        Open (or create) the log database and make sure its schema is current.

        Args:
            db_path: SQLite file to write to. Tests pass a temporary path.

        Raises:
            RuntimeError: If the file can't be opened, or was written by a
                different schema version.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        print(f"[QueryLogger] Logging to {self.db_path} ({self.count()} queries)")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """
        Yield a connection that commits on success and rolls back on error.

        GOTCHA: `with conn:` in sqlite3 manages the TRANSACTION, not the
        connection — it commits or rolls back but never closes. Forgetting
        that leaks one open file handle per logged question, hence the
        explicit close() in `finally`.

        Yields:
            An open connection with foreign keys enforced and rows as dicts.

        Raises:
            RuntimeError: Wrapping any sqlite3 error, with the file path.
        """
        try:
            conn = sqlite3.connect(self.db_path, timeout=SQLITE_BUSY_TIMEOUT_S)
        except sqlite3.Error as e:
            raise RuntimeError(f"Cannot open query log {self.db_path}: {e}") from e
        conn.row_factory = sqlite3.Row
        try:
            # GOTCHA: SQLite ignores REFERENCES/ON DELETE CASCADE unless this is
            # switched on — per connection, every time. Off by default for
            # backwards compatibility with databases from before 2009.
            conn.execute("PRAGMA foreign_keys = ON")
            with conn:
                yield conn
        except sqlite3.Error as e:
            raise RuntimeError(f"Query log operation failed on {self.db_path}: {e}") from e
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """
        Create the tables if missing, check the schema version, enable WAL.

        LEARN THIS — WAL (write-ahead logging):
        By default SQLite locks the whole file during a write, so the UI's
        analytics tab would stall while a question is being logged. In WAL
        mode new writes go to a side file (query_log.db-wal) and readers keep
        reading the last committed state — reads and one writer run in
        parallel. The setting is stored in the file, so it is set once here.

        Raises:
            RuntimeError: If the file was created by a different schema version.
        """
        with self._connect() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, _SCHEMA_VERSION):
                raise RuntimeError(
                    f"{self.db_path} uses log schema v{version}, this code expects "
                    f"v{_SCHEMA_VERSION}. Move the old file aside (or delete it) "
                    f"to start a new log."
                )
            conn.executescript(_SCHEMA)
            # PRAGMAs can't take ? parameters; this is a constant int, not input.
            conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
        if mode.lower() != "wal":
            print(f"[QueryLogger] WARNING: WAL mode unavailable on this filesystem "
                  f"(journal_mode={mode}). Logging works, but analytics reads may "
                  f"briefly wait while a question is being written.")

    # --- writing ------------------------------------------------------------

    def log(
        self,
        answer: GeneratedAnswer,
        report: VerificationReport,
        region_filter: list[str] | None = None,
        timings: StageTimings | None = None,
    ) -> int:
        """
        Record one full interaction.

        All three tables are written inside ONE transaction: either the query,
        its passages and its claims are all saved, or none are. A crash
        half-way can never leave a query row whose claims are missing — which
        would silently report it as having zero hallucinations.

        Args:
            answer:        The GeneratedAnswer from the generator.
            report:        The VerificationReport for that same answer.
            region_filter: Region types the user filtered retrieval to
                           (None = the default filter).
            timings:       Per-stage latency, if measured.

        Returns:
            The new query's id.

        Raises:
            ValueError:   If report was produced for a different answer.
            RuntimeError: If the database write fails.
        """
        if report.answer != answer.answer or report.question != answer.question:
            raise ValueError(
                "log() received a VerificationReport that belongs to a different "
                f"answer (question {report.question!r} vs {answer.question!r})."
            )
        with self._connect() as conn:
            query_id = _insert_query(conn, answer, report, region_filter,
                                     timings or StageTimings())
            _insert_chunks(conn, query_id, answer)
            _insert_claims(conn, query_id, report)
        return query_id

    # --- reading ------------------------------------------------------------

    def count(self) -> int:
        """Number of logged queries."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0]

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """
        Most recent queries first — the UI's history view.

        Args:
            limit: Maximum rows to return.

        Returns:
            One dict per query.

        Raises:
            ValueError: If limit is not a positive integer.
        """
        return self._select(_RECENT_SQL, (_check_limit(limit),))

    def worst_answers(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Answered queries with the lowest groundedness — the hallucination hotspots.

        Abstentions are excluded: "not found" is the CORRECT behaviour when the
        document lacks the answer, and counting it as perfect would hide the
        answers that actually need attention.

        Args:
            limit: Maximum rows to return.

        Returns:
            One dict per query, least grounded first.

        Raises:
            ValueError: If limit is not a positive integer.
        """
        return self._select(_WORST_SQL, (_check_limit(limit),))

    def region_stats(self) -> list[dict[str, Any]]:
        """
        Per region type: retrieval count, mean score, citation rate, grounded claims.

        Returns:
            One dict per region label, most retrieved first.
        """
        return self._select(_REGION_SQL)

    def summary(self) -> dict[str, Any]:
        """
        Headline numbers for the whole log.

        Returns:
            Dict with queries, abstained, abstention_rate, mean_groundedness
            and fully_grounded_rate (both over ANSWERED queries only; None if
            there are none yet) and claims_by_status.
        """
        with self._connect() as conn:
            row = dict(conn.execute(_SUMMARY_SQL).fetchone())
            statuses = {r["status"]: r["n"] for r in conn.execute(_STATUS_SQL)}
        row["abstention_rate"] = row["abstained"] / row["queries"] if row["queries"] else None
        row["claims_by_status"] = statuses
        return row

    def _select(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        """
        Run a read query and return plain dicts.

        Args:
            sql:    SELECT statement with ? placeholders.
            params: Values for the placeholders.

        Returns:
            Rows as dicts (JSON-serializable, ready for a Gradio table).
        """
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql, params)]


# ---------------------------------------------------------------------------
# INSERT HELPERS
# ---------------------------------------------------------------------------

# INTERVIEW ALERT — SQL injection:
# The question is typed by the user, and passage text comes from arbitrary PDFs.
# NAIVE (never do this):
#   conn.execute(f"INSERT INTO queries (question) VALUES ('{question}')")
#   A question containing  '); DROP TABLE queries; --  becomes SQL and runs.
# BETTER (what every insert below does): ? / :name placeholders. The driver
# sends the values separately from the SQL text, so they are only ever data.

def _insert_query(
    conn: sqlite3.Connection,
    answer: GeneratedAnswer,
    report: VerificationReport,
    region_filter: list[str] | None,
    timings: StageTimings,
) -> int:
    """
    Insert the queries row.

    Args:
        conn:          Open connection inside a transaction.
        answer:        The generated answer.
        report:        Its verification report.
        region_filter: Retrieval filter used, or None.
        timings:       Per-stage latency.

    Returns:
        The new row id.
    """
    cursor = conn.execute(
        """INSERT INTO queries (created_at, question, answer, llm_model, nli_model,
               abstained, groundedness, n_claims, n_flagged, invalid_citations,
               region_filter, retrieval_ms, generation_ms, verification_ms)
           VALUES (:created_at, :question, :answer, :llm_model, :nli_model,
               :abstained, :groundedness, :n_claims, :n_flagged, :invalid_citations,
               :region_filter, :retrieval_ms, :generation_ms, :verification_ms)""",
        {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "question": answer.question, "answer": answer.answer,
            "llm_model": answer.model, "nli_model": report.model,
            "abstained": int(report.abstained), "groundedness": report.groundedness,
            "n_claims": len(report.claims), "n_flagged": len(report.flagged),
            "invalid_citations": json.dumps(answer.invalid_citations),
            "region_filter": json.dumps(region_filter) if region_filter is not None else None,
            "retrieval_ms": timings.retrieval_ms, "generation_ms": timings.generation_ms,
            "verification_ms": timings.verification_ms,
        },
    )
    return cursor.lastrowid


def _insert_chunks(conn: sqlite3.Connection, query_id: int, answer: GeneratedAnswer) -> None:
    """
    Insert one retrieved_chunks row per passage shown to the LLM.

    `cited` comes from the answer's full citation list rather than from the
    verified claims, so a passage cited only by a sentence too short to verify
    still counts as cited.

    Args:
        conn:     Open connection inside a transaction.
        query_id: Parent query id.
        answer:   The generated answer (holds the sources and citations).
    """
    cited = set(answer.cited)
    conn.executemany(
        """INSERT INTO retrieved_chunks (query_id, rank, chunk_id, doc_id, page_num,
               region_label, score, cited, text)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [(query_id, rank, s.chunk_id, s.doc_id, s.page_num, s.region_label,
          s.score, int(rank in cited), s.text)
         for rank, s in enumerate(answer.sources, start=1)],
    )


def _insert_claims(conn: sqlite3.Connection, query_id: int, report: VerificationReport) -> None:
    """
    Insert one claims row per verified sentence.

    Args:
        conn:     Open connection inside a transaction.
        query_id: Parent query id.
        report:   The verification report.
    """
    conn.executemany(
        """INSERT INTO claims (query_id, position, text, status, entailment,
               contradiction, best_source, citation_ok, cited, evidence, counter_evidence)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [(query_id, pos, c.text, c.status, c.entailment, c.contradiction,
          c.best_source, int(c.citation_ok), json.dumps(c.cited),
          c.evidence, c.counter_evidence)
         for pos, c in enumerate(report.claims, start=1)],
    )


def _check_limit(limit: int) -> int:
    """
    Validate a row limit.

    Args:
        limit: Requested number of rows.

    Returns:
        The same limit.

    Raises:
        ValueError: If it is not a positive integer. SQLite treats LIMIT -1
            as "no limit", so a bad value would silently return everything.
    """
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError(f"limit must be a positive integer, got {limit!r}.")
    return limit
