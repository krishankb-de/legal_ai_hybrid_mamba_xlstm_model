"""The verifier (plan P7-H, P7-J; lexhybrid/verifier/), on chunks of the collectors' fixtures.

No NLI or LER model is loaded: the NLI step is a stand-in callable (a claim containing "erfunden" is
not entailed, every other one is), the LER step a stand-in name finder.
"""

import dataclasses
import json
import math
import time
import unicodedata
from types import SimpleNamespace

import pytest

from lexhybrid.data.corpus.scrub_ler import Entity, entity_hash, scrub_document
from lexhybrid.data.schema import Document, Section
from lexhybrid.renderer.render import NOTICE, REFUSAL, pointer_text, render_quote, render_segments
from lexhybrid.renderer.segments import QuotePointer, Text, parse_answer
from lexhybrid.retrieval.chunking import Chunk, chunk_document
from lexhybrid.verifier import verifier as verifier_module
from lexhybrid.verifier.leak import candidate_forms, leaked_entities
from lexhybrid.verifier.nli import ENTAILMENT_THRESHOLD, NLI_MODEL, MDebertaNLI, entailment_index
from lexhybrid.verifier.pointers import PointerIssue, check_pointers
from lexhybrid.verifier.quotes import (
    check_quote,
    check_quotes,
    extract_quotes,
    normalise,
    quote_matches,
    split_quote,
)
from lexhybrid.verifier.verifier import (
    REFUSED,
    asks_for_party,
    pooled_citation_precision,
    split_claims,
    verify,
)
from tests.test_retrieval import _fixture_corpus

ENDING = "\n\n" + NOTICE
KEY = b"test-scrub-key"
QUESTION = "Welche Form braucht die Kündigung eines Mietverhältnisses?"


def _entails(premise: str, hypothesis: str) -> float:
    return 0.1 if "erfunden" in hypothesis else 0.9


@pytest.fixture(scope="module")
def chunks() -> dict[str, Chunk]:
    return {c.id: c for d in _fixture_corpus() for c in chunk_document(d)}


@pytest.fixture(scope="module")
def retrieved(chunks) -> list[Chunk]:
    """A retrieved set in prompt order: passage K is retrieved[K - 1]."""
    return [
        chunks["gii:bgb:568#1"],  # <|c1|> BGB §568, one sentence
        chunks["gii:beurkg:3#1"],  # <|c2|> BeurkG §3, 19 units, one label each
        chunks["rii:KARE600065578#6"],  # <|c3|> BAG Rn. 6–8; s1 and s2 both "Rn. 6"
        chunks["gii:beurkg:2#1"],  # <|c4|> "außerhalb"
        chunks["fedlex:210:art_1#1"],  # <|c5|> ZGB Art. 1, three Absätze
    ]


# -- P7-H: pointers and quotes -----------------------------------------------------------------


def test_pointer_and_quote_checks_adversarial(retrieved):
    """P7-H: every pointer resolves (missing chunk, sentence past the end, gaps, repeats fail), and a
    rendered quote equals its source after NFC + whitespace normalisation: an off-by-one sentence,
    altered Unicode (homoglyph, ß -> ss, an invisible character), an added or dropped word, a
    mislabelled or missing quote fail; NFD and whitespace-only differences pass."""
    bgb568, beurkg3, bag, beurkg2, zgb1 = retrieved
    good = parse_answer(
        "Es gilt <|q|><|c2|><|s2|><|s3|> und <|cite|><|c5|><|s3|><|s1|> sowie <|cite|><|c3|>."
    )
    assert check_pointers(good, retrieved) == []
    adversarial = {
        "<|q|><|c6|><|s1|>": (6, None, "passage 6 is outside the 5 retrieved"),  # missing chunk
        "<|cite|><|c6|>": (6, None, "passage 6 is outside the 5 retrieved"),
        "<|q|><|c0|><|s1|>": (0, None, "passage 0 is outside"),
        "<|q|><|c1|><|s2|>": (
            1,
            2,
            "sentence 2 is outside passage 1 (gii:bgb:568#1), which has 1",
        ),  # one off
        "<|q|><|c2|><|s20|>": (2, 20, "which has 19"),
        "<|cite|><|c5|><|s4|>": (5, 4, "sentence 4 is outside passage 5"),
        "<|q|><|c2|><|s1|><|s3|>": (2, 3, "not consecutive: sentence 3 follows 1"),  # a gap
        "<|q|><|c2|><|s3|><|s2|>": (2, 2, "not consecutive: sentence 2 follows 3"),  # reversed
        "<|q|><|c2|><|s2|><|s2|>": (2, 2, "sentence 2 of passage 2 is named twice"),
        "<|cite|><|c5|><|s2|><|s2|>": (5, 2, "named twice"),
    }
    for pointer, (k, j, reason) in adversarial.items():
        (issue,) = check_pointers(parse_answer("Es gilt " + pointer), retrieved)
        assert (issue.segment, issue.passage, issue.sentence) == (1, k, j) and reason in issue.reason, pointer
    assert check_pointers([QuotePointer(1, ())], retrieved) == [
        PointerIssue(0, 1, None, "the quote of passage 1 names no sentence")
    ]
    assert len(check_pointers(parse_answer("<|q|><|c2|><|s1|><|s3|><|s99|>"), retrieved)) == 3

    segments = parse_answer("Es gilt: <|q|><|c3|><|s1|> Ferner <|q|><|c4|><|s1|> und <|q|><|c2|><|s2|><|s3|>")
    rendered = render_segments(segments, retrieved)
    assert check_quotes(segments, retrieved, rendered) == []
    span = beurkg3.sentences[1] + " " + beurkg3.sentences[2]
    assert extract_quotes(rendered) == [bag.sentences[0], beurkg2.sentences[0], span]

    def tampered(old: str, new: str) -> list:
        assert old in rendered and old != new
        return check_quotes(segments, retrieved, rendered.replace(old, new, 1))

    # off by one: the first quote carries sentence 2, which has the same label (Rn. 6)
    (issue,) = tampered(bag.sentences[0], bag.sentences[1])
    assert (issue.segment, issue.passage, issue.sentences) == (1, 3, (1,)) and issue.quoted == bag.sentences[
        1
    ]
    # off by one with the next sentences' labels: no quote of BeurkG §3 Abs. 1 Satz 1 ... Nr. 2 is left
    (issue,) = tampered(render_quote(beurkg3, (2, 3)), render_quote(beurkg3, (3, 4)))
    assert (
        issue.segment == 5
        and "no quote labelled (BeurkG §3 Abs. 1 Nr. 1; BeurkG §3 Abs. 1 Nr. 2)" in issue.reason
    )
    # altered Unicode: a Cyrillic е, ß written ss, a zero-width space (not whitespace)
    for old, new in [
        ("Die Revision", "Die Rеvision"),
        ("außerhalb", "ausserhalb"),
        ("außerhalb", "außer​halb"),
    ]:
        (issue,) = tampered(old, new)
        assert "differs" in issue.reason, new
    # a word added, a word dropped, a quote cut short
    assert len(tampered("Die Revision des Klägers", "Die Revision des klugen Klägers")) == 1
    assert len(tampered("Die Revision des Klägers", "Die Revision Klägers")) == 1
    assert len(tampered(beurkg2.sentences[0], beurkg2.sentences[0][:-10])) == 1
    # a mislabelled quote, a quote removed
    assert len(tampered("(BAG 3 AZR 158/22 Rn. 6)", "(BAG 3 AZR 158/22 Rn. 7)")) == 1
    assert [i.segment for i in tampered(render_quote(beurkg2, (1,)), "")] == [3]
    # whitespace-only differences and another Unicode normal form pass
    assert tampered("Die Revision des", "Die  Revision\n\tdes") == []
    assert tampered("Die Revision", "Die Revision") == []
    assert tampered("“ (BAG", "“\n(BAG") == []
    nfd = unicodedata.normalize("NFD", rendered)
    assert nfd != rendered and check_quotes(segments, retrieved, nfd) == []
    # a quote of a missing chunk, or of sentences it does not have
    (issue,) = check_quotes(parse_answer("Es gilt: <|q|><|c6|><|s1|>"), retrieved, "Es gilt: „x“ (y)")
    assert (issue.passage, issue.reason) == (6, "passage 6 is outside the 5 retrieved")
    (issue,) = check_quotes(parse_answer("<|q|><|c1|><|s2|>"), retrieved, "„x“ (y)")
    assert "not all in passage 1" in issue.reason

    # the single-quote checks the verifier runs on render_quote's output
    assert (
        normalise(" a\t\nb  c ") == "a b c"
        and normalise(unicodedata.normalize("NFD", "Kündigung")) == "Kündigung"
    )
    assert quote_matches(unicodedata.normalize("NFD", "Kündigung"), "Kündigung") and not quote_matches(
        "Kündigung", "Kundigung"
    )
    q = render_quote(bag, (1, 2))
    assert split_quote(q) == (normalise(bag.sentences[0] + " " + bag.sentences[1]), "BAG 3 AZR 158/22 Rn. 6")
    assert (
        check_quote(bag, (1, 2), q) is None
        and check_quote(bag, (1, 2), unicodedata.normalize("NFD", q)) is None
    )
    assert "differs from sentences [2, 3]" in check_quote(bag, (2, 3), q)  # one off
    assert "label" in check_quote(bag, (1, 2), q.replace("Rn. 6", "Rn. 7"))
    assert "not a quote" in check_quote(bag, (1,), bag.sentences[0])
    assert "not a quote" in check_quote(zgb1, (1,), f"„{zgb1.sentences[0]}“")  # the label is missing


def test_quotes_with_quotation_marks_inside_or_around_them(retrieved):
    """Quotation marks in the free text, or inside a quoted sentence (balanced or not), do not cut a
    quote short; ``extract_quotes`` keeps nested pairs inside their quote."""
    bgb568 = retrieved[0]
    segments = parse_answer("Der Begriff „Kündigung“ meint: <|q|><|c1|><|s1|>")
    rendered = render_segments(segments, retrieved)
    assert check_quotes(segments, retrieved, rendered) == []
    assert extract_quotes(rendered) == ["Kündigung", bgb568.sentences[0]]
    for sentence in ["Er sagte „ja“ und ging.", "Er sagte: „Ja.", "Er sagte: Nein“, und ging."]:
        chunk = dataclasses.replace(bgb568, sentences=(sentence,))
        segments = parse_answer("Laut Quelle: <|q|><|c1|><|s1|>")
        rendered = render_segments(segments, [chunk])
        assert check_quotes(segments, [chunk], rendered) == [], sentence
        (issue,) = check_quotes(segments, [chunk], rendered.replace("Er sagte", "Sie sagte"))
        assert "differs" in issue.reason
    assert extract_quotes("a „b „c“ d“ e „f“ „g") == ["b „c“ d", "f"]


# -- P7-J: the verifier ------------------------------------------------------------------------


def test_party_identity_refusal(retrieved):
    """P7-J: a question asking who a party was is refused; so is an answer whose text or quoted
    sentences hold a [PER_n] placeholder or a name the LER model finds. A normal answer passes."""
    answer = "Die Kündigung bedarf der Schriftform. <|q|><|c1|><|s1|>"
    ok = verify(QUESTION, answer, retrieved, _entails)
    assert not ok.refused and ok.reason is None
    assert ok.rendered == (
        "Die Kündigung bedarf der Schriftform. „(1) Die Kündigung des Mietverhältnisses bedarf der "
        "schriftlichen Form.“ (BGB §568 Abs. 1)" + ENDING
    )
    party_questions = [
        "Wer war der Kläger?",
        "Wer ist im Verfahren 3 AZR 158/22 die Beklagte?",
        "wer sind die Parteien des Rechtsstreits?",
        "Wer war Antragsgegner?",
        "Wer ist der Angeklagte?",
        "Wie heißt der Kläger?",
        "Wie hieß die Antragstellerin?",
        "Name des Beklagten?",
        "Wie lautet der Name des Klägers?",
        "Welchen Namen hat die Revisionsklägerin?",
        "Wer hat in diesem Verfahren geklagt?",
        "Wer ist die Beschwerdeführerin?",
        "Who was the defendant?",
    ]
    for question in party_questions:
        result = verify(question, answer, retrieved, _entails)
        assert result.refused and result.rendered == REFUSAL + ENDING == REFUSED, question
        assert result.reason == "party identity: the question asks who a party is" and result.claims == ()
    for question in [
        QUESTION,
        "Wer trägt die Kosten des Rechtsstreits?",
        "Wer ist beweispflichtig, der Kläger oder der Beklagte?",
        "Darf der Anwalt im Namen der Partei klagen?",
        "Wie lautet der Antrag des Klägers?",
    ]:
        assert not asks_for_party(question), question

    # a PER placeholder in the model's text, or in a quoted sentence
    result = verify(QUESTION, "[PER_1] muss schriftlich kündigen. <|q|><|c1|><|s1|>", retrieved, _entails)
    assert (
        result.refused
        and result.reason == "party identity: the answer contains the person placeholder [PER_1]"
    )
    bag = retrieved[2]
    scrubbed = dataclasses.replace(
        bag, sentences=("Der Kläger [PER_2] legte Revision ein.", *bag.sentences[1:])
    )
    result = verify(QUESTION, "Laut Gericht: <|q|><|c1|><|s1|>", [scrubbed], _entails)
    assert result.refused and "[PER_2]" in result.reason and result.rendered == REFUSED
    assert not verify(QUESTION, "Laut Gericht: <|q|><|c1|><|s2|>", [scrubbed], _entails).refused  # not quoted
    judge = dataclasses.replace(bag, sentences=("Vorsitzender war [RR_1].", *bag.sentences[1:]))
    assert not verify(
        QUESTION, "Laut Gericht: <|q|><|c1|><|s1|>", [judge], _entails
    ).refused  # a judge is no party

    # a name the LER model finds, in the text or in a quote; the reason does not repeat it
    def ner(text: str) -> list[str]:
        return [name for name in ("Hans Müller", "Erika Muster") if name in text]

    result = verify(
        QUESTION, "Hans Müller muss schriftlich kündigen. <|q|><|c1|><|s1|>", retrieved, _entails, ner=ner
    )
    assert result.refused and result.reason == "party identity: the answer names a person (LER)"
    assert "Müller" not in json.dumps(result.log_row())
    named = dataclasses.replace(bag, sentences=("Erika Muster legte Revision ein.", *bag.sentences[1:]))
    assert verify(QUESTION, "Laut Gericht: <|q|><|c1|><|s1|>", [named], _entails, ner=ner).refused
    assert verify(QUESTION, answer, retrieved, _entails, ner=ner) == ok  # nobody named: the same result


def _named_decision(doc_id: str, text: str) -> Document:
    return Document(
        id=doc_id, source="rii", jurisdiction="DE", doc_type="decision", licence="DE-UrhG-5",
        commercial_safe=True, text=text, citation_id="BGH VIII ZR 1/26",
        sections=[Section(label="Rn. 1", randnummer=1, text=text, part="Gründe")],
    )  # fmt: skip


def test_no_scrub_log_entity_in_index_chunks():
    """P7-J: no entity of the scrub log is left in an index chunk. A name planted in a decision is
    found among the fixture corpus' chunks by its HMAC under the test key (the log never holds the
    name); after the scrub replaced it, nothing is found; under another key nothing matches."""
    name = "Hans Müller"
    doc = _named_decision(
        "rii:named", f"Der Kläger {name} kündigte das Mietverhältnis fristlos. Die Beklagte widersprach."
    )
    start = doc.text.index(name)
    scrubbed, rows = scrub_document(doc, [Entity(start, start + len(name), "PER", 0.99)], key=KEY)
    assert [r["entity_hmac"] for r in rows] == [entity_hash(name, KEY)] and name not in json.dumps(rows)
    corpus = _fixture_corpus()
    leaky = [c for d in [*corpus, doc] for c in chunk_document(d)]
    assert leaked_entities(leaky, rows, KEY) == [("rii:named#1", entity_hash(name, KEY))]
    clean = [c for d in [*corpus, scrubbed] for c in chunk_document(d)]
    assert clean[-1].sentences[0] == "Der Kläger [PER_1] kündigte das Mietverhältnis fristlos."
    assert leaked_entities(clean, rows, KEY) == []
    assert leaked_entities(leaky, rows, b"another-key") == [] and leaked_entities(leaky, [], KEY) == []

    # the log is global: a logged surname left in another document's chunk is a leak too, once per chunk
    surname = {
        "doc_id": "rii:x",
        "label": "PER",
        "placeholder": "[PER_1]",
        "entity_hmac": entity_hash("Müller", KEY),
    }
    other = _named_decision("rii:other", "Die Zeugin Müller sagte aus. Frau Müller blieb dabei.")
    assert leaked_entities(chunk_document(other), [surname], KEY) == [("rii:other#1", surname["entity_hmac"])]
    forms = candidate_forms("Der Kläger Hans-Peter Müller und Dr. Bülow, nicht von Arnim.")
    assert {"Hans-Peter Müller", "Kläger Hans-Peter Müller", "Dr. Bülow", "Arnim"} <= set(forms)
    assert (
        "Der Kläger Hans-Peter Müller" not in forms
        and "Bülow, nicht" not in forms
        and "von Arnim" not in forms
    )


def test_failing_sentences_are_removed(retrieved):
    """A claim that NLI does not support, that cites nothing, or whose pointer does not resolve is
    removed; the rest is rendered with the notice, and the log says why each claim failed."""
    answer = (
        "Die Kündigung bedarf der Schriftform. <|cite|><|c1|> "
        "Mündlich genügt erfunden ebenso <|cite|><|c1|>. "
        "Das ist ohne Beleg so. "
        "Ein Notar wirkt nicht mit <|q|><|c2|><|s99|>. "
        "Der Notar wirkt in eigenen Angelegenheiten nicht mit: <|q|><|c2|><|s1|><|s2|>"
    )
    result = verify(QUESTION, answer, retrieved, _entails)
    beurkg3 = retrieved[1]
    assert not result.refused and result.rendered == (
        "Die Kündigung bedarf der Schriftform. (BGB §568) Der Notar wirkt in eigenen Angelegenheiten nicht mit: "
        f"„{beurkg3.sentences[0]} {beurkg3.sentences[1]}“ (BeurkG §3 Abs. 1 Satz 1; BeurkG §3 Abs. 1 Nr. 1)"
        + ENDING
    )
    assert [c.passed for c in result.claims] == [True, False, False, False, True]
    reasons = [c.reason for c in result.claims]
    assert reasons[0] is None and reasons[4] is None
    assert (
        reasons[1] == f"nli: entailment 0.100 < {ENTAILMENT_THRESHOLD}" and result.claims[1].entailment == 0.1
    )
    assert reasons[2] == "unsupported: the claim cites no passage"
    assert reasons[3] == "pointer: sentence 99 is outside passage 2 (gii:beurkg:3#1), which has 19"
    assert result.claims[3].text == "Ein Notar wirkt nicht mit <|q|><|c2|><|s99|>."  # unresolved: as written
    assert result.claims[0].text == "Die Kündigung bedarf der Schriftform. (BGB §568)"
    assert result.citation_precision == 0.5  # 2 of the 4 claims with a pointer


def test_main_claim_failure_refuses(retrieved):
    """The first claim failing -- not entailed, unsupported, or pointing outside the retrieved set --
    turns the whole answer into the refusal; the threshold is inclusive."""
    supported = " Die Kündigung bedarf der Schriftform <|cite|><|c1|>."
    for first, reason in [
        ("Mündlich genügt erfunden <|cite|><|c1|>.", "main claim failed: nli: entailment 0.100 < 0.5"),
        ("Ja.", "main claim failed: unsupported: the claim cites no passage"),
        ("Es gilt <|q|><|c6|><|s1|>.", "main claim failed: pointer: passage 6 is outside the 5 retrieved"),
    ]:
        result = verify(QUESTION, first + supported, retrieved, _entails)
        assert result.refused and result.rendered == REFUSED and result.reason == reason, first
        assert len(result.claims) == 2 and result.claims[1].passed
    answer = "Die Kündigung bedarf der Schriftform <|cite|><|c1|>."
    assert not verify(QUESTION, answer, retrieved, lambda p, h: ENTAILMENT_THRESHOLD).refused
    assert verify(QUESTION, answer, retrieved, lambda p, h: math.nextafter(0.5, 0)).refused
    assert verify(QUESTION, answer, retrieved, lambda p, h: 0.7, threshold=0.8).refused
    assert verify(QUESTION, answer, retrieved, lambda p, h: math.nan).refused


def test_a_claim_of_citations_alone_is_unsupported(retrieved):
    """Regression: an answer of only a citation sent NLI an empty hypothesis and passed when the
    model happened to score it high, rendering ``(BGB §568)`` as the answer. A claim must state
    something beyond its citations; a bare quote does (the quote is the statement)."""
    seen = []

    def recording(premise: str, hypothesis: str) -> float:
        seen.append(hypothesis)
        return 0.9

    result = verify(QUESTION, "<|cite|><|c1|>", retrieved, recording)
    assert result.refused and result.rendered == REFUSED and seen == []
    assert result.reason == "main claim failed: unsupported: the claim states nothing but citations"
    assert result.citation_precision == 0.0  # the claim cites, and failed
    result = verify(
        QUESTION,
        "Die Kündigung bedarf der Schriftform <|cite|><|c1|>. (<|cite|><|c2|>)",
        retrieved,
        recording,
    )
    assert not result.refused and [c.passed for c in result.claims] == [True, False]
    assert result.rendered == "Die Kündigung bedarf der Schriftform (BGB §568)." + ENDING
    assert not verify(QUESTION, "<|q|><|c1|><|s1|>", retrieved, recording).refused


def test_citation_precision_is_the_verifier_log(retrieved):
    """citation_precision = passed cited claims / cited claims (None when no claim cites), kept for a
    refused main claim; pooled over answers; the log row carries no answer text."""
    all_pass = verify(
        QUESTION,
        "Die Kündigung bedarf der Schriftform <|cite|><|c1|>. Ferner <|q|><|c1|><|s1|>",
        retrieved,
        _entails,
    )
    assert not all_pass.refused and all_pass.citation_precision == 1.0
    mixed = verify(
        QUESTION,
        "Schriftform <|cite|><|c1|>. Das ist erfunden <|cite|><|c1|>. Das ist erfunden <|cite|><|c5|>. Ohne Beleg.",
        retrieved,
        _entails,
    )
    assert not mixed.refused and mixed.citation_precision == pytest.approx(1 / 3)
    refused = verify(
        QUESTION, "Das ist erfunden <|cite|><|c1|>. Schriftform <|cite|><|c1|>.", retrieved, _entails
    )
    assert refused.refused and refused.citation_precision == 0.5
    uncited = verify(QUESTION, "Ja.", retrieved, _entails)
    assert uncited.refused and uncited.citation_precision is None and len(uncited.claims) == 1
    assert verify(QUESTION, "<|unanswerable|>", retrieved, _entails).citation_precision is None
    assert pooled_citation_precision([all_pass, mixed, refused, uncited]) == pytest.approx(
        (2 + 1 + 1) / (2 + 3 + 2)
    )
    assert pooled_citation_precision([uncited]) is None and pooled_citation_precision([]) is None

    row = mixed.log_row()
    assert (row["cited_claims"], row["passed_cited_claims"], row["refused"]) == (3, 1, False)
    assert [c["pointers"] for c in row["claims"]] == [
        ["<|cite|><|c1|>"],
        ["<|cite|><|c1|>"],
        ["<|cite|><|c5|>"],
        [],
    ]
    assert "Schriftform" not in json.dumps(row, ensure_ascii=False)


def test_unanswerable_and_malformed_answers_refuse(retrieved):
    for answer, reason in [
        ("<|unanswerable|><|endoftext|>", "unanswerable"),
        ("", "empty answer"),
        ("  \n ", "empty answer"),
    ]:
        result = verify(QUESTION, answer, retrieved, _entails)
        assert (result.rendered, result.refused, result.reason, result.claims) == (REFUSED, True, reason, ())
    for answer in ["Ja. <|unanswerable|>", "<|q|><|c1|> Text", "Text <|s3|>", "<|passage|>x"]:
        result = verify(QUESTION, answer, retrieved, _entails)
        assert (
            result.refused and result.rendered == REFUSED and result.reason.startswith("malformed answer: ")
        )


def test_claims_split_at_sentence_ends_and_pointers_attach(retrieved):
    """A claim ends at a sentence-final mark in a text segment, not after an abbreviation or an
    ordinal; a pointer directly after the mark belongs to the claim before it."""
    segments = parse_answer(
        "Nach § 573 Abs. 2 BGB gilt das <|cite|><|c1|>. <|q|><|c1|><|s1|> Ferner gilt z. B. Art. 1 ZGB! "
        "Wie oft? <|cite|><|c5|><|cite|><|c5|><|s1|>Am 5. Januar endet es seit 2001. Danach nicht <|cite|><|c1|>.  "
    )

    def shown(claim) -> str:
        return "".join(s.text if isinstance(s, Text) else pointer_text(s) for _, s in claim)

    claims = split_claims(segments)
    assert [shown(c) for c in claims] == [
        "Nach § 573 Abs. 2 BGB gilt das <|cite|><|c1|>. <|q|><|c1|><|s1|>",
        " Ferner gilt z. B. Art. 1 ZGB!",
        " Wie oft? <|cite|><|c5|><|cite|><|c5|><|s1|>",
        "Am 5. Januar endet es seit 2001.",
        " Danach nicht <|cite|><|c1|>.  ",
    ]
    assert [i for i, _ in claims[0]] == [0, 1, 2, 2, 3]  # the pieces of a text keep its segment index
    assert "".join(shown(c) for c in claims) == shown(list(enumerate(segments)))  # nothing lost
    assert split_claims(parse_answer("  ")) == [] and split_claims([]) == []


def test_claim_split_is_linear_in_the_answer():
    """Regression: the abbreviation test searched the whole text before every mark (quadratic: about
    26 s for 20 000 marks); it looks a few characters back now, with the same cuts, a word boundary
    and the digit lookbehind still seeing the text before the window."""
    unit = "Siehe Buchst. A. Seit 1999. Am 12. Mai. Nicht Xz. B. Ende. "
    ends = verifier_module._sentence_ends(unit)
    assert ends == [unit.index(s) + len(s) for s in ("A.", "1999.", "Mai.", "Xz.", " B.", "Ende.")]
    text = unit * 5000
    start = time.perf_counter()
    assert verifier_module._sentence_ends(text) == [e + k * len(unit) for k in range(5000) for e in ends]
    assert time.perf_counter() - start < 2.0


def test_nli_premise_and_hypothesis(retrieved):
    """The premise is the quoted and cited sentences, or the whole chunk text for a citation without
    sentences; the hypothesis is the claim's text with quotes inline, citations left out."""
    bgb568, _, _, beurkg2, zgb1 = retrieved
    seen = []

    def recording(premise: str, hypothesis: str) -> float:
        seen.append((premise, hypothesis))
        return 0.9

    answer = (
        "Es gilt: <|q|><|c1|><|s1|> Siehe <|cite|><|c5|><|s3|><|s1|> und <|cite|><|c4|> sowie <|cite|><|c4|>."
    )
    assert not verify(QUESTION, answer, retrieved, recording).refused
    ((premise, hypothesis),) = seen
    assert premise == "\n".join(
        [bgb568.sentences[0], f"{zgb1.sentences[2]} {zgb1.sentences[0]}", beurkg2.text]
    )
    assert hypothesis == f"Es gilt: „{bgb568.sentences[0]}“ Siehe und sowie ."


def test_quote_check_guards_against_renderer_drift(retrieved, monkeypatch):
    """A renderer whose quote, or whose final text, no longer equals the source refuses the claim or
    the answer instead of showing a wrong quote."""
    answer = "Die Kündigung bedarf der Schriftform. <|q|><|c1|><|s1|>"
    real_quote = verifier_module.render_quote
    monkeypatch.setattr(
        verifier_module, "render_quote", lambda c, s: real_quote(c, s).replace("Kündigung", "Kundigung")
    )
    result = verify(QUESTION, answer, retrieved, _entails)
    assert result.refused and result.reason.startswith("main claim failed: quote: the quoted text differs")
    monkeypatch.undo()
    monkeypatch.setattr(
        verifier_module,
        "render_segments",
        lambda s, c: render_segments(s, c).replace("schriftlichen", "mündlichen"),
    )
    result = verify(QUESTION, answer, retrieved, _entails)
    assert (
        result.refused
        and result.reason.startswith("renderer drift: the quote differs")
        and result.rendered == REFUSED
    )


def _stub_transformers(monkeypatch) -> SimpleNamespace:
    """Stand-in tokenizer and model for ``MDebertaNLI``: one token per word, three special tokens
    for a pair, and logits whose softmax gives the entailment class 0.6."""
    import torch
    import transformers

    seen = SimpleNamespace(loaded=[], pairs=[], forwards=0)

    class Tokenizer:
        def __call__(self, text, pair=None, **kwargs):
            if pair is None:  # one sequence: the hypothesis, counted
                return {"input_ids": [0] * len(text.split())}
            seen.pairs.append((text, pair, kwargs))
            return {"input_ids": torch.zeros(1, 4, dtype=torch.long)}

        def num_special_tokens_to_add(self, pair=False):
            return 3 if pair else 2

    class Model:
        config = SimpleNamespace(id2label={0: "contradiction", 1: "entailment", 2: "neutral"})

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **enc):
            seen.forwards += 1
            return SimpleNamespace(logits=torch.tensor([[0.0, math.log(3.0), 0.0]]))

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda name: seen.loaded.append(name) or Tokenizer()
    )
    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda name: seen.loaded.append(name) or Model(),
    )
    return seen


def test_nli_threshold_is_pre_registered_and_labels_are_read_by_name(monkeypatch):
    """ENTAILMENT_THRESHOLD is the pre-registered 0.5; MDebertaNLI loads lazily and takes the
    entailment probability at the index config.id2label names, whatever the order (stand-in model)."""
    assert ENTAILMENT_THRESHOLD == 0.5
    assert entailment_index({"0": "ENTAILMENT", "1": "neutral", "2": "contradiction"}) == 0
    with pytest.raises(ValueError, match="no single entailment label"):
        entailment_index({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})

    seen = _stub_transformers(monkeypatch)
    nli = MDebertaNLI(device="cpu")
    assert seen.loaded == []  # nothing is loaded before the first call
    assert nli("Prämisse", "Hypothese") == pytest.approx(0.6)  # softmax([0, ln 3, 0])[1]
    assert nli("Prämisse", "Hypothese") == pytest.approx(0.6)
    assert seen.loaded == [NLI_MODEL, NLI_MODEL] and seen.pairs[0][:2] == ("Prämisse", "Hypothese")
    assert seen.pairs[0][2]["max_length"] == 512


def test_nli_truncates_the_premise_never_the_claim(monkeypatch):
    """Regression: ``truncation=True`` is ``longest_first``, which cut the hypothesis's tail when it
    was the longer sequence (a claim quoting its premise always is), so the claim's own words could
    pass unchecked. Only the premise is truncated now, and a hypothesis that leaves no room for a
    premise is not entailed, without running the model."""
    seen = _stub_transformers(monkeypatch)
    nli = MDebertaNLI(device="cpu", max_length=8)
    assert nli("eine lange Prämisse " * 50, "vier Wörter lang hier") == pytest.approx(0.6)  # 4 + 3 < 8
    assert seen.pairs[-1][2]["truncation"] == "only_first" and seen.forwards == 1
    assert nli("Prämisse", "fünf Wörter sind hier zuviel") == 0.0  # 5 + 3 >= 8: nothing left for it
    assert seen.forwards == 1 and len(seen.pairs) == 1
