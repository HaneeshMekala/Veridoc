"""
tests/test_hallucination.py — NLI claim verification (Novelty 2).

WHY THIS FILE EXISTS:
A hallucination detector that silently stops detecting is worse than none —
the UI keeps painting sentences green. Three bugs of exactly that kind were
caught by hand while building this module, and each now has a test so it
can never come back quietly:
  1. NLI label order differs between model families (index 1 vs 2).
  2. Whole multi-sentence passages as premises → true facts scored "neutral".
  3. The "et al." regex guard also matched "meal.", merging two sentences.

Layout:
  - Unit tests (no model): splitting, premises, verdict rules, label lookup.
  - Behaviour tests (real DeBERTa, auto-marked `model`): a small labelled set
    covering each failure type, plus two properties of NLI itself.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from config import NLI_CONTRADICTION_INDEX, NLI_ENTAILMENT_INDEX
from src.generator import GeneratedAnswer
from src.hallucination import (
    CONTRADICTED,
    GROUNDED,
    UNSUPPORTED,
    HallucinationChecker,
    VerificationReport,
    VerifiedClaim,
    build_premises,
    is_checkable,
    premise_units,
    split_sentences,
    strip_citations,
)


def bare_checker(entail_idx: int = 1, contra_idx: int = 0) -> HallucinationChecker:
    """
    A checker WITHOUT a loaded model, for testing its pure logic.

    WHY __new__: HallucinationChecker.__init__ downloads and loads DeBERTa.
    __new__ creates the object without running __init__, so we can set just
    the attributes the method under test reads.
    """
    checker = HallucinationChecker.__new__(HallucinationChecker)
    checker.entail_idx, checker.contra_idx, checker.model_name = entail_idx, contra_idx, "fake-nli"
    return checker


# ---------------------------------------------------------------------------
# Answer splitting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("Give 0.5 mg daily [1]. Do not exceed 2 mg [2].",
     ["Give 0.5 mg daily [1].", "Do not exceed 2 mg [2]."]),                 # decimals survive
    ("Take it with food (e.g. breakfast). Next dose tomorrow.",
     ["Take it with food (e.g. breakfast).", "Next dose tomorrow."]),       # e.g.
    ("See Fig. 2 for the curve. Smith et al. 2019 agreed.",
     ["See Fig. 2 for the curve.", "Smith et al. 2019 agreed."]),           # Fig. / et al.
    ("Take with the evening meal. The dose rises weekly.",
     ["Take with the evening meal.", "The dose rises weekly."]),            # regression: "meal."
    ("- Dose is 500 mg [1]\n- Max is 2 g [2]",
     ["- Dose is 500 mg [1]", "- Max is 2 g [2]"]),                         # bullet lines
    ("The trial ran in 2020. 500 mg was the ceiling.",
     ["The trial ran in 2020.", "500 mg was the ceiling."]),                # digit-led sentence
])
def test_split_sentences(text: str, expected: list[str]) -> None:
    """Sentence boundaries in medical prose: decimals, abbreviations, bullets."""
    assert split_sentences(text) == expected


def test_split_is_conservative_on_lowercase_starts() -> None:
    """
    Known limitation, pinned on purpose: a sentence starting lowercase is not
    split. Under-splitting is the safe failure (two facts share one verdict);
    over-splitting would create fragments that all score "unsupported".
    """
    assert len(split_sentences("Dose is 500 mg. metformin is oral.")) == 1


@pytest.mark.parametrize("raw, clean", [
    ("Dose is 500 mg [1].", "Dose is 500 mg."),
    ("Dose is 500 mg [2][3].", "Dose is 500 mg."),
    ("Dose [1, 2] is 500 mg.", "Dose is 500 mg."),
])
def test_strip_citations_leaves_clean_prose(raw: str, clean: str) -> None:
    """The NLI model must see plain sentences, not "500 mg ." with a stray space."""
    assert strip_citations(raw) == clean


def test_short_fragments_are_not_checkable() -> None:
    """'See below.' is not a factual claim; scoring it would skew groundedness."""
    assert not is_checkable("See below.")
    assert is_checkable("Dose is 500 mg.")


# ---------------------------------------------------------------------------
# Premise construction (the SummaC fix)
# ---------------------------------------------------------------------------

def test_prose_premises_rejoin_pdf_line_wraps(make_chunk) -> None:
    """layout.py joins PDF lines with \\n mid-sentence; premises must be whole sentences."""
    passage = make_chunk("Metformin is initiated at 500 mg once daily with the\n"
                         "evening meal. The dose may be increased weekly.")
    units = premise_units(passage)
    assert units[:2] == ["Metformin is initiated at 500 mg once daily with the evening meal.",
                         "The dose may be increased weekly."]
    assert units[-1] == passage.text.strip()          # whole passage kept last


def test_table_premises_are_rows_even_when_short(make_chunk) -> None:
    """In a table, a line break is a row boundary and 'Glipizide 40 mg' is a full fact."""
    table = make_chunk("Drug Max dose\nMetformin 2000 mg\nGlipizide 40 mg", label="table")
    assert premise_units(table)[:3] == ["Drug Max dose", "Metformin 2000 mg", "Glipizide 40 mg"]


def test_single_sentence_passage_is_not_duplicated(make_chunk) -> None:
    """The sentence IS the whole passage — score it once, not twice."""
    assert premise_units(make_chunk("Metformin is taken orally once daily.")) == \
           ["Metformin is taken orally once daily."]


def test_build_premises_remembers_each_premise_source(make_chunk) -> None:
    """owner[i] maps premise i back to its passage — needed for citation checks."""
    texts, owner = build_premises([make_chunk("One fact here. Two facts here."),
                                   make_chunk("Third fact here.")])
    assert owner.tolist() == [0, 0, 0, 1]
    assert texts[3] == "Third fact here."


# ---------------------------------------------------------------------------
# Verdict logic with synthetic probabilities
# ---------------------------------------------------------------------------

# Rows are premises; columns follow the DeBERTa order [contradiction, entailment, neutral].
SUPPORTS, REFUTES, SILENT = [0.02, 0.95, 0.03], [0.95, 0.02, 0.03], [0.05, 0.05, 0.90]


def verdict(rows: list[list[float]], cited: list[int], owner: list[int]) -> VerifiedClaim:
    """Apply the checker's verdict rules to one claim with given premise scores."""
    claim = VerifiedClaim("A claim to check.", cited=cited)
    premises = [f"premise {i}" for i in range(len(rows))]
    bare_checker()._apply_scores([claim], np.array([rows]), premises, np.array(owner))
    return claim


def test_one_supporting_premise_makes_a_claim_grounded() -> None:
    """Entailment wins over a noisy contradiction elsewhere (e.g. another drug's row)."""
    claim = verdict([REFUTES, SUPPORTS], cited=[2], owner=[0, 1])
    assert (claim.status, claim.best_source, claim.evidence) == (GROUNDED, 2, "premise 1")


def test_refuted_and_unsupported_claims() -> None:
    """No support + refutation = contradicted; no support + silence = unsupported."""
    assert verdict([REFUTES, SILENT], [1], [0, 1]).status == CONTRADICTED
    assert verdict([SILENT, SILENT], [1], [0, 1]).status == UNSUPPORTED


def test_contradicted_claim_records_what_contradicted_it() -> None:
    """The UI shows this premise so a spurious 'contradiction' is visible as one."""
    claim = verdict([SILENT, REFUTES], cited=[1], owner=[0, 1])
    assert claim.counter_evidence == "premise 1"


def test_right_fact_wrong_citation_is_detected() -> None:
    """Supported by passage 2 while citing passage 1: grounded, but citation_ok is False."""
    claim = verdict([SILENT, SUPPORTS], cited=[1], owner=[0, 1])
    assert claim.grounded and not claim.citation_ok
    assert verdict([SILENT, SUPPORTS], cited=[2], owner=[0, 1]).citation_ok


def test_scores_stay_attached_to_the_right_claim_and_premise() -> None:
    """
    Regression guard for the reshape in _score_claims: pairs are built
    premise-major, so a wrong transpose would swap scores between claims.
    Each fake score encodes (premise, claim) so a swap is visible.
    """
    checker = bare_checker()
    checker.score_pairs = lambda prem, hyp: np.array(
        [[0.0, float(f"0.{p[-1]}{h[-1]}"), 0.0] for p, h in zip(prem, hyp)])
    claims = [VerifiedClaim("claim 1"), VerifiedClaim("claim 2")]
    probs = checker._score_claims(claims, ["premise 1", "premise 2", "premise 3"])
    assert probs.shape == (2, 3, 3)
    assert probs[1, :, 1].tolist() == [0.12, 0.22, 0.32]     # claim 2 vs premises 1..3


def test_groundedness_and_flagged() -> None:
    """Groundedness is the grounded fraction; an empty report (abstention) is 1.0."""
    claims = [VerifiedClaim("a b c", status=GROUNDED), VerifiedClaim("d e f", status=UNSUPPORTED)]
    report = VerificationReport("q", "a", claims, [])
    assert report.groundedness == 0.5
    assert [c.text for c in report.flagged] == ["d e f"]
    assert VerificationReport("q", "a", [], [], abstained=True).groundedness == 1.0


def test_abstention_is_not_sent_to_the_model(make_chunk) -> None:
    """'Not found' asserts nothing — verifying it would only waste a forward pass."""
    checker = bare_checker()
    checker.score_pairs = lambda *a: pytest.fail("NLI model was called for an abstention")
    answer = GeneratedAnswer("q", "I could not find this.", [make_chunk("x y z")], abstained=True)
    assert checker.verify(answer).claims == []


# ---------------------------------------------------------------------------
# Label order — the bug that doesn't crash
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("id2label, expected", [
    ({0: "contradiction", 1: "entailment", 2: "neutral"}, (1, 0)),   # cross-encoder/nli-deberta-v3
    ({0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}, (2, 0)),   # roberta-large-mnli
])
def test_label_order_is_read_from_the_checkpoint(id2label: dict, expected: tuple) -> None:
    """A hardcoded index would read NEUTRAL as ENTAILMENT for one of these families."""
    checker = bare_checker()
    checker.model = SimpleNamespace(config=SimpleNamespace(id2label=id2label))
    assert checker._resolve_label_indices() == expected


def test_unnamed_labels_fall_back_to_config_with_a_warning(capsys) -> None:
    """LABEL_0/1/2 carry no meaning: use config.py, and say so loudly."""
    checker = bare_checker()
    checker.model = SimpleNamespace(config=SimpleNamespace(
        id2label={0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}))
    assert checker._resolve_label_indices() == (NLI_ENTAILMENT_INDEX, NLI_CONTRADICTION_INDEX)
    assert "WARNING" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Behaviour tests — real DeBERTa (auto-marked `model`)
# ---------------------------------------------------------------------------

# NAIVE (don't do this):  assert claim.entailment == 0.9973
#   Exact floats differ between GPU and CPU, and between library versions.
# BETTER: assert the VERDICT (a threshold decision) — the thing the user sees.

@pytest.fixture(scope="module")
def sources():
    """Four passages shaped like real layout output: wrapped prose and a table."""
    from src.vectorstore import RetrievedChunk
    def c(i: int, text: str, label: str = "text") -> RetrievedChunk:
        return RetrievedChunk(f"c{i}", text, "doc.pdf", 0, label, [0.0, 0.0, 1.0, 1.0], 0.7)
    return [
        c(1, "Metformin is initiated at 500 mg once daily with the\n"
             "evening meal. The dose may be increased in 500 mg\nincrements weekly."),
        c(2, "The maximum recommended daily dose of metformin is 2000 mg.\n"
             "Doses above this provided no additional glycaemic benefit."),
        c(3, "Metformin is contraindicated in patients with an eGFR below\n"
             "30 mL/min/1.73 m2 because of the risk of lactic acidosis."),
        c(4, "Drug Maximum daily dose\nMetformin 2000 mg\nGlipizide 40 mg\nSitagliptin 100 mg",
          "table"),
    ]


@pytest.mark.parametrize("claim, expected, source", [
    # SummaC regression: paraphrase of sentence 1 of a 2-sentence, line-wrapped passage.
    ("Metformin is started at 500 mg once daily with the evening meal [1].", GROUNDED, 1),
    ("The maximum daily dose is 2000 mg [2].", GROUNDED, None),
    ("The maximum daily dose is 4000 mg [2].", CONTRADICTED, None),
    ("Patients should also take 81 mg of aspirin every morning [1].", UNSUPPORTED, None),
    ("It is contraindicated below an eGFR of 30 mL/min/1.73 m2 [3].", GROUNDED, 3),
    ("The maximum daily dose of glipizide is 40 mg [4].", GROUNDED, 4),   # from a table row
])
def test_labelled_claims(nli_checker, sources, claim: str, expected: str,
                         source: int | None) -> None:
    """One case per failure type the detector must tell apart."""
    [verified] = nli_checker.verify(GeneratedAnswer("q", claim, sources, model="fake")).claims
    assert verified.status == expected, f"{verified.text!r}: e={verified.entailment:.3f}"
    if source is not None:
        assert verified.best_source == source


def test_wrong_citation_on_real_model(nli_checker, sources) -> None:
    """The eGFR fact is in passage 3; citing [1] is grounded but a wrong citation."""
    answer = GeneratedAnswer("q", "It is contraindicated below an eGFR of 30 mL/min/1.73 m2 [1].",
                             sources, model="fake")
    [claim] = nli_checker.verify(answer).claims
    assert claim.grounded and claim.best_source == 3 and not claim.citation_ok


def test_real_checkpoint_label_order(nli_checker) -> None:
    """The DeBERTa-v3 cross-encoders put entailment at index 1, not 2."""
    assert (nli_checker.entail_idx, nli_checker.contra_idx) == (1, 0)


@pytest.mark.parametrize("specific, general", [
    ("Metformin is initiated at 500 mg once daily with the evening meal.",
     "Metformin is taken with a meal."),
    ("The maximum recommended daily dose of metformin is 2000 mg.",
     "Metformin has a maximum recommended daily dose."),
    ("Metformin is contraindicated in patients with an eGFR below 30 mL/min/1.73 m2.",
     "Metformin is contraindicated in some patients."),
    ("Lactic acidosis caused by metformin has resulted in death.",
     "Lactic acidosis can be fatal."),
])
def test_nli_is_not_symmetric(nli_checker, specific: str, general: str) -> None:
    """
    Premise = evidence, hypothesis = claim. Swapping them asks a different
    question: a specific statement entails a general one, not the reverse.

    NOTE: NLI is strict. "initiated at 500 mg once daily" does NOT entail
    "taken every day" (it could be stopped) — the model scores that 0.21.
    That strictness is what we want from a hallucination checker.
    """
    forward, backward = nli_checker.score_pairs([specific, general], [general, specific])
    assert forward[nli_checker.entail_idx] > 0.9
    assert backward[nli_checker.entail_idx] < 0.1


def test_scores_do_not_depend_on_batch_padding(nli_checker) -> None:
    """A pair scored alone or padded next to a long pair must get the same probabilities."""
    p, h = "The maximum daily dose is 2000 mg.", "The dose limit is 2000 mg."
    alone = nli_checker.score_pairs([p], [h])[0]
    batched = nli_checker.score_pairs([p, p * 12], [h, h])[0]
    assert np.allclose(alone, batched, atol=1e-3)
