"""The renderer (lexhybrid/renderer/render.py), on chunks of the collectors' fixtures."""

import pytest

from lexhybrid.renderer.render import (
    NOTICE,
    REFUSAL,
    RenderError,
    pointer_text,
    render_answer,
    render_cite,
    render_quote,
    render_segments,
)
from lexhybrid.renderer.segments import (
    CitePointer,
    PointerSyntaxError,
    QuotePointer,
    Text,
    Unanswerable,
    parse_answer,
)
from lexhybrid.retrieval.chunking import Chunk, chunk_document
from tests.test_retrieval import _fixture_corpus

ENDING = "\n\n" + NOTICE


def _by_id() -> dict[str, Chunk]:
    return {c.id: c for d in _fixture_corpus() for c in chunk_document(d)}


@pytest.fixture(scope="module")
def chunks() -> dict[str, Chunk]:
    return _by_id()


@pytest.fixture(scope="module")
def retrieved(chunks) -> list[Chunk]:
    """A retrieved set in prompt order: passage K is retrieved[K - 1]."""
    return [
        chunks["gii:beurkg:3#1"],  # <|c1|> BeurkG §3, 19 units, Nr. 2a
        chunks["rii:KARE600065578#1"],  # <|c2|> BAG Leitsatz, cut after "sog."
        chunks["gii:bgb:568#1"],  # <|c3|> the planted BGB §568
        chunks["rii:KARE600065578#6"],  # <|c4|> BAG Rn. 6–8
        chunks["gii:beurkg:2#1"],  # <|c5|> "außerhalb"
        chunks["fedlex:210:art_1#1"],  # <|c6|> ZGB Art. 1
        chunks["bger:CH_BGer_004_4A-102-2026_2026-08-17#4"],  # <|c7|> BGer E. 1–2.1
    ]


def _quoted(rendered: str) -> list[str]:
    """The texts between „ and “ in a rendered answer."""
    out, rest = [], rendered
    while "„" in rest:
        rest = rest.split("„", 1)[1]
        text, rest = rest.split("“", 1)
        out.append(text)
    return out


def test_quote_is_verbatim(retrieved):
    """Every quote is the chunk's sentence byte for byte (§, umlauts, ß, abbreviation cuts), with
    that sentence's label; a span joins its sentences with one space and dedups its labels."""
    for k, chunk in enumerate(retrieved, start=1):
        for j, sentence in enumerate(chunk.sentences, start=1):
            rendered = render_answer(f"Es gilt: <|q|><|c{k}|><|s{j}|>", retrieved)
            assert rendered == f"Es gilt: „{sentence}“ ({chunk.labels[j - 1]})" + ENDING
    beurkg3, leitsatz, bgb568, bag6, beurkg2, zgb1, bger = retrieved
    assert any(ch in "".join(bgb568.sentences) for ch in "äöü") and "ß" in beurkg2.sentences[0]
    assert "§" in leitsatz.sentences[1] and leitsatz.sentences[0].endswith("sog.")

    out = render_answer("<|q|><|c1|><|s3|>", retrieved)
    assert out == f"„{beurkg3.sentences[2]}“ (BeurkG §3 Abs. 1 Nr. 2)" + ENDING
    assert render_answer("<|q|><|c1|><|s4|>", retrieved).endswith("(BeurkG §3 Abs. 1 Nr. 2a)" + ENDING)
    assert render_answer("<|q|><|c3|><|s1|>", retrieved).startswith(
        "„(1) Die Kündigung des Mietverhältnisses bedarf der schriftlichen Form.“ (BGB §568 Abs. 1)"
    )

    # a span: sentences in the pointer's order, one space between them, labels deduplicated
    span = render_answer("<|q|><|c2|><|s1|><|s2|>", retrieved)
    assert _quoted(span) == [leitsatz.sentences[0] + " " + leitsatz.sentences[1]]
    assert span.endswith("“ (BAG 3 AZR 158/22, Leitsatz)" + ENDING)
    span = render_answer("<|q|><|c4|><|s1|><|s2|><|s3|>", retrieved)
    assert _quoted(span) == [" ".join(bag6.sentences[:3])]
    assert span.endswith("“ (BAG 3 AZR 158/22 Rn. 6; BAG 3 AZR 158/22 Rn. 7)" + ENDING)
    span = render_answer("<|q|><|c6|><|s3|><|s1|>", retrieved)
    assert _quoted(span) == [zgb1.sentences[2] + " " + zgb1.sentences[0]]
    assert span.endswith("(ZGB Art. 1 Abs. 3; ZGB Art. 1 Abs. 1)" + ENDING)

    # a sentence that carries its own quotes and parentheses is not touched
    assert render_quote(bger, (3,)) == (
        "„Das Bundesgericht wendet das Recht von Amtes wegen an (Art. 106 Abs. 1 BGG).“"
        " (BGer 4A_102/2026 E. 2.1)"
    )


def test_notice_always_appended(retrieved):
    """Normal, citation-only, unanswerable, empty, whitespace-only and text-only answers all end
    with the notice, exactly once and after one blank line."""
    answers = {
        "normal": "Der Notar soll nicht mitwirken: <|q|><|c1|><|s2|> Siehe auch <|cite|><|c3|>.<|endoftext|>",
        "citation-only": "<|cite|><|c1|><|cite|><|c4|><|s1|>",
        "unanswerable": "<|unanswerable|><|endoftext|>",
        "unanswerable-padded": "  <|unanswerable|>\n",
        "empty": "",
        "eos-only": "<|endoftext|>",
        "whitespace": "  \n\t ",
        "text-only": "Die Frage betrifft das Mietrecht.\n",
    }
    for name, answer in answers.items():
        out = render_answer(answer, retrieved)
        assert out.endswith(ENDING), name
        assert out.count(NOTICE) == 1, name
        assert not out[: -len(ENDING)].endswith(("\n", " ")), name
    for name in ("unanswerable", "unanswerable-padded", "empty", "eos-only", "whitespace"):
        assert render_answer(answers[name], retrieved) == REFUSAL + ENDING, name
    assert render_answer(answers["text-only"], retrieved) == "Die Frage betrifft das Mietrecht." + ENDING
    assert (
        render_answer(answers["citation-only"], retrieved) == "(BeurkG §3)(BAG 3 AZR 158/22 Rn. 6)" + ENDING
    )
    assert render_answer("<|unanswerable|>", []) == REFUSAL + ENDING  # nothing retrieved
    assert render_segments([], retrieved) == REFUSAL + ENDING


def test_notice_written_by_the_model_is_text_and_the_notice_is_still_appended(retrieved):
    """Text is copied unchanged, the notice included: the renderer appends its own after it."""
    answer = "Nein. " + NOTICE
    assert render_answer(answer, retrieved) == answer + ENDING


def test_cite_renders_citation_id_or_sentence_labels(retrieved):
    beurkg3, leitsatz, bgb568, bag6, *_ = retrieved
    assert render_cite(bgb568) == "(BGB §568)"
    assert render_cite(beurkg3) == "(BeurkG §3)"
    assert render_cite(bag6) == "(BAG 3 AZR 158/22 Rn. 6–8)"
    assert render_cite(beurkg3, (3,)) == "(BeurkG §3 Abs. 1 Nr. 2)"
    assert render_cite(beurkg3, (2, 3)) == "(BeurkG §3 Abs. 1 Nr. 1; BeurkG §3 Abs. 1 Nr. 2)"
    assert render_cite(leitsatz, (1, 2)) == "(BAG 3 AZR 158/22, Leitsatz)"
    out = render_answer("Die Form ist streng <|cite|><|c3|>, ebenso <|cite|><|c1|><|s4|>.", retrieved)
    assert out == "Die Form ist streng (BGB §568), ebenso (BeurkG §3 Abs. 1 Nr. 2a)." + ENDING


def test_span_labels_are_deduplicated_in_order(retrieved):
    bag6 = retrieved[3]
    j = [i for i, label in enumerate(bag6.labels, start=1) if label.endswith("Rn. 7")]
    assert len(j) >= 2  # several sentences of one Randnummer
    first_rn6, last_rn7 = 1, j[-1]
    assert render_quote(bag6, j).endswith("“ (BAG 3 AZR 158/22 Rn. 7)")
    assert render_quote(bag6, (last_rn7, first_rn6, last_rn7)).endswith(
        "(BAG 3 AZR 158/22 Rn. 7; BAG 3 AZR 158/22 Rn. 6)"
    )
    assert render_cite(bag6, (*j, first_rn6)) == "(BAG 3 AZR 158/22 Rn. 7; BAG 3 AZR 158/22 Rn. 6)"


@pytest.mark.parametrize(
    "answer, message",
    [
        ("<|q|><|c0|><|s1|>", "passage 0"),
        ("<|q|><|c8|><|s1|>", "passage 8"),  # len(retrieved) + 1
        ("<|cite|><|c0|>", "passage 0"),
        ("Siehe <|cite|><|c8|>.", "passage 8"),
        ("<|q|><|c1|><|s20|>", "sentence 20"),  # BeurkG §3 has 19
        ("<|q|><|c1|><|s0|>", "sentence 0"),
        ("<|q|><|c5|><|s1|><|s2|>", "sentence 2"),  # BeurkG §2 has 1
        ("<|cite|><|c2|><|s3|>", "sentence 3"),  # the Leitsatz has 2
    ],
)
def test_pointer_outside_retrieved_set_raises(retrieved, answer, message):
    with pytest.raises(RenderError, match=message) as err:
        render_answer(answer, retrieved)
    assert pointer_text(next(s for s in parse_answer(answer) if not isinstance(s, Text))) in str(err.value)
    assert isinstance(err.value, ValueError)


def test_pointer_bounds_follow_the_retrieved_set(retrieved):
    """The last passage and the last sentence are in range; one past each is not."""
    n = len(retrieved)
    last = retrieved[-1]
    assert render_answer(f"<|q|><|c{n}|><|s{len(last.sentences)}|>", retrieved).startswith("„")
    with pytest.raises(RenderError, match=f"sentence {len(last.sentences) + 1}"):
        render_answer(f"<|q|><|c{n}|><|s{len(last.sentences) + 1}|>", retrieved)
    with pytest.raises(RenderError, match="passage 1 is outside the 0 retrieved"):
        render_answer("<|cite|><|c1|>", [])
    with pytest.raises(RenderError, match="sentence 20 is outside gii:beurkg:3#1"):
        render_quote(retrieved[0], (20,))
    with pytest.raises(RenderError, match="sentence 0"):
        render_cite(retrieved[0], (0,))
    with pytest.raises(RenderError, match="names no sentence"):
        render_quote(retrieved[0], ())


def test_segments_render_like_the_answer_and_syntax_errors_propagate(retrieved):
    answer = "Kein Mitwirken bei <|q|><|c1|><|s2|> (vgl. <|cite|><|c7|><|s3|>)."
    assert render_segments(parse_answer(answer), retrieved) == render_answer(answer, retrieved)
    with pytest.raises(PointerSyntaxError):
        render_answer("<|q|><|c1|> ohne Satz", retrieved)
    with pytest.raises(PointerSyntaxError):
        render_answer("Text <|unanswerable|>", retrieved)
    with pytest.raises(RenderError, match="whole answer"):
        render_segments([Text("Text "), Unanswerable()], retrieved)
    with pytest.raises(RenderError, match="names no sentence"):
        render_segments([QuotePointer(1, ())], retrieved)
    assert render_segments([CitePointer(2)], retrieved) == "(BAG 3 AZR 158/22, Leitsatz)" + ENDING


def test_text_segments_are_copied_unchanged(retrieved):
    text = "  Gemäß § 3 BeurkG – „Mitwirkungsverbot“ (Nr. 1):\n\n* eigene Angelegenheiten;"
    out = render_answer(text + " <|cite|><|c1|><|s2|>\n", retrieved)
    assert out == text.strip() + " (BeurkG §3 Abs. 1 Nr. 1)" + ENDING


def test_a_passage_held_by_several_sources_cites_every_source():
    """A quoted or cited passage whose text other sources
    also hold (Chunk.copies, linked from the pack's dedup log) names every source; the quoted text is
    unchanged, so the verifier's quote check still passes."""
    import dataclasses

    from lexhybrid.renderer.render import render_cite, render_quote, sources
    from lexhybrid.verifier.verifier import verify

    decision = next(c for d in _fixture_corpus() for c in chunk_document(d) if c.source == "rii")
    single = render_quote(decision, (1,))
    linked = dataclasses.replace(decision, copies=("oldp:123", "multilegalpile:9", "oldp:456"))
    assert sources(linked) == ["rii", "oldp", "multilegalpile"]
    quoted = render_quote(linked, (1,))
    want = "; Quellen: rechtsprechung-im-internet.de, openlegaldata.io, MultiLegalPile)"
    assert quoted == single[:-1] + want
    assert _quoted(quoted) == _quoted(single)  # the quote itself does not change
    assert (
        render_cite(linked).startswith(f"({linked.citation_id}; Quellen: ")
        and render_cite(decision) == f"({decision.citation_id})"
    )
    result = verify("Was gilt?", "Es gilt dies: <|q|><|c1|><|s1|>.", [linked], nli=lambda p, h: 0.99)
    assert not result.refused and "Quellen: " in result.rendered
