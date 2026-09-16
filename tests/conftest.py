"""
tests/conftest.py — Shared fixtures and fakes for the whole test suite.

WHY THIS FILE EXISTS:
pytest loads conftest.py automatically before any test. Fixtures defined here
are available to every test file by NAME — a test that takes a parameter
called `nli_checker` gets the loaded model without importing anything.

LEARN THIS — Fixture scope (the ML-testing version of "load the model once"):
  scope="function" (default): rebuilt for every test — right for cheap,
      mutable things like a temp database, so tests can't leak state.
  scope="session": built once for the whole run — right for a model that
      takes seconds to load and is never mutated. Loading DeBERTa per test
      would turn a 20-second suite into a 5-minute one.

INTERVIEW ALERT — "How do you test code that depends on an ML model?"
Answer: "Two layers. Unit tests replace the model with a FAKE (an object with
the same method, returning fixed values), so the surrounding logic — parsing,
thresholds, bookkeeping — is tested fast and deterministically. A smaller set
of behaviour tests loads the REAL model and asserts on verdicts and
invariants, never exact floats — those vary by hardware and library version."
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Callable

import pytest

from src.vectorstore import RetrievedChunk


# Fixtures that load a real model. Any test that uses one (directly or through
# another fixture) is auto-marked `model` below, so the marker can't be forgotten.
_MODEL_FIXTURES = {"nli_checker", "embedder", "reranker"}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """
    Mark every test that depends on a real model with @pytest.mark.model.

    item.fixturenames is the test's full fixture closure, so a test using a
    fixture that itself uses `embedder` is caught too.

    Args:
        items: All collected tests.
    """
    for item in items:
        if _MODEL_FIXTURES & set(item.fixturenames):
            item.add_marker(pytest.mark.model)


# ---------------------------------------------------------------------------
# REAL MODELS (session-scoped: loaded once per test run)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def nli_checker():
    """The real DeBERTa NLI checker, loaded once for the whole session."""
    from src.hallucination import HallucinationChecker
    return HallucinationChecker()


@pytest.fixture(scope="session")
def embedder():
    """The real MiniLM embedder, loaded once for the whole session."""
    from src.embedder import Embedder
    return Embedder()


# ---------------------------------------------------------------------------
# FAKES AND FACTORIES
# ---------------------------------------------------------------------------

@pytest.fixture
def make_chunk() -> Callable[..., RetrievedChunk]:
    """
    Factory for RetrievedChunk objects with sensible defaults.

    LEARN THIS — Factory fixtures: a fixture can return a FUNCTION, so each
    test builds exactly the objects it needs while defaults stay in one place.

    Returns:
        make(text, label="text", page=0, score=0.7, chunk_id=None) -> RetrievedChunk
    """
    def make(text: str, label: str = "text", page: int = 0, score: float = 0.7,
             chunk_id: str | None = None) -> RetrievedChunk:
        return RetrievedChunk(chunk_id or f"c-{abs(hash(text)) % 10**8}", text,
                              "doc.pdf", page, label, [0.0, 0.0, 100.0, 20.0], score)
    return make


class FakeLLM:
    """
    Stand-in for Gemini: returns a fixed answer and counts calls.

    Implements only what Generator uses — `.chat(messages)` returning an object
    with `.message.content`, and a `.model` name. That is "duck typing": the
    Generator never checks the class, only that the method exists.
    """

    model = "fake-llm"

    def __init__(self, text: str) -> None:
        """Store the canned answer text."""
        self.text = text
        self.calls = 0

    def chat(self, messages: list) -> SimpleNamespace:
        """Return the canned answer, recording that the LLM was called."""
        self.calls += 1
        return SimpleNamespace(message=SimpleNamespace(content=self.text))


@pytest.fixture
def fake_llm() -> Callable[[str], FakeLLM]:
    """Factory: fake_llm("answer text [1].") -> FakeLLM."""
    return FakeLLM
