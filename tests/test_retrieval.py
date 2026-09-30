"""Retrieval (plan P7): chunking of statutes and decisions (P7-A, P7-B), on the collectors' fixtures."""

import json
from pathlib import Path

import pytest

from lexhybrid.data.schema import Document, Section, parse_citation
from lexhybrid.data.sft_format import Example, parse, render
from lexhybrid.retrieval.chunking import (
    Chunk,
    chunk_case_law,
    chunk_document,
    chunk_statute,
    estimate_tokens,
)

FIXTURES = Path(__file__).parent / "fixtures" / "collectors"


def _gii():
    from lexhybrid.data.corpus.collectors.gii import parse_gii_xml

    return {
        d.citation_id: d
        for d in parse_gii_xml((FIXTURES / "gii" / "beurkg_excerpt.xml").read_bytes(), "beurkg")
    }


def _fedlex(name):
    from lexhybrid.data.corpus.collectors.fedlex import parse_fedlex_xml

    return {d.citation_id: d for d in parse_fedlex_xml((FIXTURES / "fedlex" / f"{name}.xml").read_bytes())}


def _rii():
    from lexhybrid.data.corpus.collectors.rii import parse_rii_xml

    return parse_rii_xml((FIXTURES / "rii" / "jb-KARE600065578.xml").read_bytes())


def _bger():
    from lexhybrid.data.corpus.collectors.bger import parse_bger

    name = "CH_BGer_004_4A-102-2026_2026-08-17"
    meta = json.loads((FIXTURES / "bger" / f"{name}.json").read_text())
    return parse_bger((FIXTURES / "bger" / f"{name}.html").read_text(encoding="utf-8"), meta)


def _norm(text: str) -> str:
    return " ".join(text.split())


def _verbatim(chunk: Chunk, doc: Document) -> bool:
    """Every sentence is a stretch of the document's text (whitespace-normalised): quotable."""
    source = _norm(doc.text)
    return all(_norm(s) in source for s in chunk.sentences)


# -- P7-A: statutes ----------------------------------------------------------------------------


def test_statute_chunks_carry_hierarchy_and_dates():
    """P7-A: one chunk per provision, hierarchy Gesetz -> structural headings -> § (then Absatz,
    Satz and Nr. in the sentence labels), the version's validity window, sentences verbatim and
    numbered for <|sJ|>."""
    doc = _gii()["BeurkG §3"]
    (chunk,) = chunk_statute(doc)
    assert chunk.hierarchy == (
        "BeurkG",
        "Abschnitt 1 Allgemeine Vorschriften",
        "§ 3 Verbot der Mitwirkung als Notar",
    )
    assert (chunk.valid_from, chunk.valid_to) == (doc.valid_from, doc.valid_to) == ("2025-12-10", None)
    assert chunk.id == "gii:beurkg:3#1" and chunk.citation_id == "BeurkG §3" and chunk.doc_id == doc.id
    assert chunk.labels[:3] == ("BeurkG §3 Abs. 1 Satz 1", "BeurkG §3 Abs. 1 Nr. 1", "BeurkG §3 Abs. 1 Nr. 2")
    assert "BeurkG §3 Abs. 1 Nr. 2a" in chunk.labels and chunk.labels[-1] == "BeurkG §3 Abs. 3 Satz 2"
    assert chunk.sentences[0].startswith("(1) Ein Notar soll") and chunk.sentences[1].startswith("1. eigene")
    assert _verbatim(chunk, doc) and len(chunk.sentences) == len(chunk.labels) <= 64
    assert (chunk.licence, chunk.commercial_safe, chunk.jurisdiction) == ("DE-UrhG-5", True, "DE")
    # a one-sentence Absatz carries no "Satz"; a provision without Absätze cites the § itself
    (single,) = chunk_statute(_gii()["BeurkG §2"])
    assert single.labels == ("BeurkG §2",) and single.hierarchy[-1] == "§ 2 Überschreiten des Amtsbezirks"
    # Swiss articles: the act's Abteilung / Titel / Abschnitt path, the article's heading last
    (art,) = chunk_statute(_fedlex("or_excerpt")["OR Art. 97"])
    assert art.hierarchy[:4] == (
        "OR",
        "Erste Abteilung: Allgemeine Bestimmungen",
        "Zweiter Titel: Die Wirkung der Obligationen",
        "Zweiter Abschnitt: Die Folgen der Nichterfüllung",
    )
    assert art.hierarchy[4].startswith("Art. 97") and art.labels == ("OR Art. 97 Abs. 1", "OR Art. 97 Abs. 2")
    (schlt,) = chunk_statute(_fedlex("zgb_excerpt")["ZGB SchlT Art. 1"])
    assert schlt.hierarchy[0] == "ZGB SchlT" and schlt.hierarchy[1].startswith("Schlusstitel")


def test_statute_labels_are_citation_ids_and_long_provisions_continue():
    """Every sentence label parses as a statute citation id (the renderer prints it); a provision
    of more than 64 units continues in a second chunk with the same header and hierarchy."""
    for doc in [*_gii().values(), *_fedlex("or_excerpt").values(), *_fedlex("zgb_excerpt").values()]:
        for chunk in chunk_statute(doc):
            assert all(parse_citation(label)["kind"] == "statute" for label in chunk.labels), chunk.labels
            assert _verbatim(chunk, doc)
    items = "\n".join(f"{i}. Fall Nummer {i} mit Text." for i in range(1, 71))
    long = Document(
        id="gii:x:1", source="gii", jurisdiction="DE", doc_type="statute", licence="DE-UrhG-5",
        commercial_safe=True, text="§ 1 Lang\n(1) Es gilt:\n" + items, citation_id="XG §1",
        sections=[Section(label="§ 1 Abs. 1", absatz=1, text="(1) Es gilt:\n" + items)],
    )  # fmt: skip
    first, second = chunk_statute(long)
    assert (len(first.sentences), len(second.sentences)) == (64, 7)
    assert second.id == "gii:x:1#2" and second.hierarchy == first.hierarchy == ("XG", "§ 1 Lang")
    assert first.labels[1] == "XG §1 Abs. 1 Nr. 1" and second.labels[-1] == "XG §1 Abs. 1 Nr. 70"
    with pytest.raises(ValueError, match="not a statute"):
        chunk_statute(_rii())


# -- P7-B: decisions ---------------------------------------------------------------------------


def test_case_windows_keep_randnummer():
    """P7-B: part by part (Leitsatz, Tenor, Tatbestand, Entscheidungsgründe), windows of 300-500
    tokens anchored at the Randnummern they cover; every sentence labelled at its Rn.; nothing
    lost or repeated."""
    doc = _rii()
    chunks = chunk_case_law(doc)
    assert [c.part for c in chunks][:3] == ["Leitsatz", "Tenor", "Tatbestand"]
    assert (
        chunks[0].citation_id == "BAG 3 AZR 158/22, Leitsatz"
        and chunks[1].citation_id == "BAG 3 AZR 158/22, Tenor"
    )
    grounds = [c for c in chunks if c.part == "Entscheidungsgründe"]
    assert grounds[0].anchor == "Rn. 6–8" and grounds[0].citation_id == "BAG 3 AZR 158/22 Rn. 6–8"
    first_rn = [int(c.anchor.split()[1].split("–")[0]) for c in grounds]
    assert first_rn == sorted(first_rn), "windows follow the Randnummern"
    for c in chunks:
        assert c.hierarchy == ("BAG 3 AZR 158/22", c.part) and c.valid_from == doc.valid_from
        size = sum(estimate_tokens(s) for s in c.sentences)
        assert size <= 500 and _verbatim(c, doc)
        for label in c.labels:
            assert label.startswith("BAG 3 AZR 158/22")
            if " Rn. " in label:
                assert parse_citation(label)["randnummer"] == label.rsplit(" ", 1)[1]
    for part in ("Tatbestand", "Entscheidungsgründe"):
        windows = [c for c in chunks if c.part == part]
        sizes = [sum(estimate_tokens(s) for s in c.sentences) for c in windows]
        assert all(n >= 300 for n in sizes[1:-1]), (part, sizes)  # only the edges of a part may be short
    rn8 = [label for c in chunks for label in c.labels if label.endswith(" Rn. 8")]
    assert rn8 and all(label == "BAG 3 AZR 158/22 Rn. 8" for label in rn8)
    joined = _norm(" ".join(s for c in chunks for s in c.sentences))
    assert joined == _norm(" ".join(s.text for s in doc.sections)), "every sentence exactly once, in order"


def test_swiss_windows_anchor_at_erwaegungen_and_counts_are_injectable():
    doc = _bger()
    chunks = chunk_case_law(doc)
    assert [c.part for c in chunks][0] == "Kopf" and "Dispositiv" in {c.part for c in chunks}
    reasons = [c for c in chunks if c.part == "Erwägungen"]
    assert reasons[0].anchor.startswith("E. 1") and all(
        c.citation_id.startswith("BGer 4A_102/2026 E. ") for c in reasons
    )
    assert all(
        parse_citation(label)["kind"] == "decision" for c in reasons for label in c.labels if " E. " in label
    )
    words = chunk_case_law(doc, count_tokens=lambda s: len(s.split()))  # one token per word: fewer windows
    assert len(words) < len(chunks) and all(sum(len(s.split()) for s in c.sentences) <= 500 for c in words)


def test_an_overlong_sentence_is_cut_into_window_sized_verbatim_pieces():
    text = " ".join(f"wort{i}" for i in range(400))  # one "sentence", ~912 estimated tokens
    doc = Document(
        id="rii:x", source="rii", jurisdiction="DE", doc_type="decision", licence="DE-UrhG-5",
        commercial_safe=True, text=text, citation_id="BGH I ZR 1/26",
        sections=[Section(label="Rn. 1", randnummer=1, text=text, part="Gründe")],
    )  # fmt: skip
    chunks = chunk_case_law(doc)
    assert len(chunks) == 2 and all(sum(estimate_tokens(s) for s in c.sentences) <= 500 for c in chunks)
    assert " ".join(s for c in chunks for s in c.sentences) == text
    assert {c.anchor for c in chunks} == {"Rn. 1"} and chunks[0].labels == ("BGH I ZR 1/26 Rn. 1",)


# -- chunks and the prompt ---------------------------------------------------------------------


def test_chunks_are_prompt_passages_and_dispatch_by_type():
    """A chunk is one P3-Y passage (<= 64 sentences, no special-token markup) and round-trips
    through the prompt format; statutes and decisions dispatch, other types are not indexed."""
    chunks = [*chunk_document(_gii()["BeurkG §3"]), *chunk_document(_rii())[:3]]
    example = Example("Wann darf der Notar nicht mitwirken?", tuple(c.passage() for c in chunks))
    prompt, target = render(example)
    assert parse(prompt + target).passages == example.passages
    general = Document(
        id="fineweb2_de:x", source="fineweb2_de", jurisdiction="DE", doc_type="general",
        licence="ODC-By-1.0", commercial_safe=True, text="Ein Satz.",
    )  # fmt: skip
    assert chunk_document(general) == []
    with pytest.raises(ValueError, match="1..64"):
        Chunk(**{**chunks[0].__dict__, "sentences": (), "labels": ()})
    with pytest.raises(ValueError, match="one label per sentence"):
        Chunk(**{**chunks[0].__dict__, "labels": chunks[0].labels[:-1]})
    assert chunks[0].copies == () and "Abschnitt 1 Allgemeine Vorschriften" in chunks[0].text


# -- P7-C: BM25 --------------------------------------------------------------------------------


def _planted():
    text = "(1) Die Kündigung des Mietverhältnisses bedarf der schriftlichen Form."
    return Document(
        id="gii:bgb:568", source="gii", jurisdiction="DE", doc_type="statute", licence="DE-UrhG-5",
        commercial_safe=True, text="§ 568 Form und Inhalt der Kündigung\n" + text, citation_id="BGB §568",
        sections=[Section(label="§ 568 Abs. 1", absatz=1, text=text)], hierarchy=["Buch 2 Recht der Schuldverhältnisse"],
    )  # fmt: skip


def _fixture_corpus() -> list[Document]:
    return [
        *_gii().values(),
        *_fedlex("or_excerpt").values(),
        *_fedlex("zgb_excerpt").values(),
        _rii(),
        _bger(),
        _planted(),
    ]


def test_bm25_finds_planted_passage(tmp_path):
    """P7-C: German stopwords and stemming (inflected query words find the passage), numbers kept
    (`§ 1 BeurkG`), a stopword-only query finds nothing, and a saved index answers the same."""
    from lexhybrid.retrieval.bm25 import BM25Index, tokenize

    chunks = [c for d in _fixture_corpus() for c in chunk_document(d)]
    index = BM25Index.build(chunks)
    top = index.query("Kündigungen von Mietverhältnissen, schriftlich", k=5)
    assert top[0].chunk_id == "gii:bgb:568#1" and top[0].rank == 1 and top[0].score > 0
    assert [h.rank for h in top] == list(range(1, len(top) + 1)) and top == sorted(
        top, key=lambda h: -h.score
    )
    assert index.query("§ 1 BeurkG Geltungsbereich", k=3)[0].chunk_id == "gii:beurkg:1#1"
    assert index.query("OR Art. 97 Ausbleiben der Erfüllung", k=3)[0].chunk_id.startswith("fedlex:220:")
    assert index.query("und der die", k=5) == [] and index.query("", k=5) == []
    assert tokenize(["Mietverhältnisses über Wohnraum"]) == [["mietverhaltnis", "wohnraum"]]
    index.save(tmp_path / "bm25")
    again = BM25Index.load(tmp_path / "bm25")
    queries = ["Kündigung Mietverhältnis", "Notar Beurkundung Mitwirkung", "Schuldner Ersatz"]
    assert again.query_many(queries, k=4) == index.query_many(queries, k=4)
    assert again.query("Notar", k=len(chunks) + 50)  # k past the corpus size is clamped
    config = json.loads((tmp_path / "bm25" / "lexhybrid_bm25.json").read_text())
    (tmp_path / "bm25" / "lexhybrid_bm25.json").write_text(json.dumps({**config, "stopwords": "en"}))
    with pytest.raises(ValueError, match="other tokenizer settings"):
        BM25Index.load(tmp_path / "bm25")
    with pytest.raises(ValueError, match="unique"):
        BM25Index.build([chunks[0], chunks[0]])
    with pytest.raises(ValueError, match="no chunks"):
        BM25Index.build([])


def test_build_index_writes_chunks_manifest_and_a_queryable_index(tmp_path, script):
    """scripts/build_index.py --kind bm25: scrubbed documents in, chunks.jsonl + bm25/ + index.json
    out; general text is not indexed; a file under raw/ is refused (R11)."""
    from lexhybrid.retrieval.bm25 import BM25Index

    scrubbed = tmp_path / "scrubbed"
    scrubbed.mkdir()
    corpus = _fixture_corpus()
    general = Document(
        id="fineweb2_de:x", source="fineweb2_de", jurisdiction="DE", doc_type="general",
        licence="ODC-By-1.0", commercial_safe=True, text="Ein ganz allgemeiner Satz über das Wetter.",
    )  # fmt: skip
    (scrubbed / "legal.jsonl").write_text("".join(d.to_json() + "\n" for d in corpus))
    (scrubbed / "web.part-0-of-1.jsonl").write_text(general.to_json() + "\n")
    out = tmp_path / "index"
    bi = script("build_index")
    assert bi.main(["--kind", "bm25", "--docs", str(scrubbed / "legal.jsonl"), str(scrubbed / "web.part-0-of-1.jsonl"),
                    "--out", str(out)]) == 0  # fmt: skip
    manifest = json.loads((out / "index.json").read_text())
    chunks = [Chunk.from_json(ln) for ln in (out / "chunks.jsonl").read_text().splitlines()]
    assert manifest["documents"] == len(corpus) + 1 and manifest["chunks"] == len(chunks)
    assert (
        set(manifest["chunks_by_type"]) == {"statute", "decision"}
        and "fineweb2_de" not in manifest["chunks_by_source"]
    )
    assert chunks == [c for d in corpus for c in chunk_document(d)]
    by_id = {c.id: c for c in chunks}
    hit = BM25Index.load(out / "bm25").query("Schriftform der Kündigung eines Mietverhältnisses", k=1)[0]
    assert by_id[hit.chunk_id].labels == ("BGB §568 Abs. 1",)
    raw = tmp_path / "raw" / "gii"
    raw.mkdir(parents=True)
    (raw / "gii.jsonl").write_text(corpus[0].to_json() + "\n")
    with pytest.raises(SystemExit, match="not scrubbed"):
        bi.main(["--docs", str(raw / "gii.jsonl"), "--out", str(tmp_path / "x")])
