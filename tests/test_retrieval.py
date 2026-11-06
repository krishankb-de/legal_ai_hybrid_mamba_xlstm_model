"""Retrieval: chunking of statutes and decisions, on the collectors' fixtures."""

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


# -- statutes ----------------------------------------------------------------------------


def test_statute_chunks_carry_hierarchy_and_dates():
    """One chunk per provision, hierarchy Gesetz -> structural headings -> § (then Absatz,
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


# -- decisions ---------------------------------------------------------------------------


def test_case_windows_keep_randnummer():
    """Part by part (Leitsatz, Tenor, Tatbestand, Entscheidungsgründe), windows of 300-500
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
    """A chunk is one prompt passage (<= 64 sentences, no special-token markup) and round-trips
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


# -- BM25 --------------------------------------------------------------------------------


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
    """German stopwords and stemming (inflected query words find the passage), numbers kept
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
    out; general text is not indexed; a file under raw/ is refused."""
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


def test_rrf_and_rerank_orderings():
    """Reciprocal rank fusion (k 60) sums 1/(k + rank) over the rankings; ties break by best
    rank, then id. The reranker re-scores the fused top 50, sorts best first (ties keep the fused
    order) and keeps passages best first up to 8 chunks and 4,096 tokens, skipping one that does not fit."""
    import math

    from lexhybrid.retrieval.bm25 import Hit
    from lexhybrid.retrieval.chunking import chunk_document
    from lexhybrid.retrieval.fusion import reciprocal_rank_fusion
    from lexhybrid.retrieval.rerank import rerank, rerank_and_select, select

    fused = reciprocal_rank_fusion([["a", "b", "c"], [Hit("c", 9.0, 1), Hit("a", 2.0, 2), Hit("d", 1.0, 3)]])
    assert [h.chunk_id for h in fused] == ["a", "c", "b", "d"] and [h.rank for h in fused] == [1, 2, 3, 4]
    assert math.isclose(fused[0].score, 1 / 61 + 1 / 62) and math.isclose(fused[3].score, 1 / 63)
    assert [h.chunk_id for h in reciprocal_rank_fusion([["y"], ["x"]])] == ["x", "y"]  # tie: id
    assert [h.chunk_id for h in reciprocal_rank_fusion([["b", "a"], ["a"]])] == ["a", "b"]
    assert reciprocal_rank_fusion([["a", "a", "b"]])[1].score == 1 / 62  # a repeat counts once
    assert len(reciprocal_rank_fusion([list("abcdef")], top_n=3)) == 3

    chunks = [c for d in _fixture_corpus() for c in chunk_document(d)][:20]  # under the top-50 cut
    assert len(chunks) == 20 and len({c.id for c in chunks}) == 20
    by_len = {c.id: len(c.text) for c in chunks}
    longest_first = rerank("frage", chunks, lambda q, texts: [len(t) for t in texts])
    assert [r.chunk.id for r in longest_first] == sorted(by_len, key=lambda i: -by_len[i])
    flat = rerank("frage", chunks, lambda q, texts: [0.0] * len(texts))
    assert [r.chunk.id for r in flat] == [c.id for c in chunks]  # ties keep the fused order
    assert len(rerank("frage", chunks * 10, lambda q, t: [0.0] * len(t))) == 50  # the top 50 only

    ranked = rerank(
        "frage", chunks[:10], lambda q, t: list(range(len(t), 0, -1)), count_tokens=lambda t: 1000
    )
    assert [c.id for c in select(ranked)] == [c.id for c in chunks[:4]]  # 4 x 1,000 <= 4,096
    sizes = iter([3000, 2000, 900, 50, 50] + [10] * 5)
    ranked = rerank(
        "frage", chunks[:10], lambda q, t: list(range(len(t), 0, -1)), count_tokens=lambda t: next(sizes)
    )
    assert [c.id for c in select(ranked)] == [
        chunks[i].id for i in (0, 2, 3, 4, 5, 6, 7, 8)
    ]  # 2,000 skipped; 8 max
    assert rerank_and_select("frage", chunks[:2], lambda q, t: [1.0, 2.0]) == [chunks[1], chunks[0]]
    with pytest.raises(ValueError, match="scores for"):
        rerank("frage", chunks[:3], lambda q, t: [1.0])


# -- filters, index versions, re-sync ----------------------------------------------------


def test_valid_on_filter():
    """A chunk is valid on d when valid_from <= d <= valid_to (both inclusive, a missing bound
    open); undated chunks are always kept; jurisdiction and doc_type filters keep input order; an
    unknown jurisdiction, doc_type or malformed date is rejected."""
    import dataclasses
    from datetime import date, datetime

    from lexhybrid.retrieval.filters import filter_chunks, valid_on

    base = chunk_document(_planted())[0]
    assert base.valid_from is None and base.valid_to is None

    def at(name, **kw):
        return dataclasses.replace(base, id=f"x:{name}#1", doc_id=f"x:{name}", **kw)

    old = at("old", valid_from="2001-09-01", valid_to="2021-12-31")  # the repealed version
    new = at("new", valid_from="2022-01-01")  # in force, open end
    capped = at("capped", valid_to="1999-12-31")  # open start
    undated = at("undated", doc_type="decision")
    austrian = at("at", jurisdiction="AT", valid_from="2010-01-01")
    eu = at("eu", jurisdiction="EU", doc_type="regulation", valid_from="2018-05-25")
    chunks = [old, new, capped, undated, austrian, eu]

    def names(cs):
        return [c.doc_id.removeprefix("x:") for c in cs]

    assert valid_on(old, "2001-09-01") and valid_on(old, "2021-12-31")  # both ends inclusive
    assert not valid_on(old, "2001-08-31") and not valid_on(old, date(2022, 1, 1))
    assert valid_on(new, "9999-12-31") and valid_on(capped, "0001-01-01") and valid_on(undated, "1900-01-01")
    assert names(filter_chunks(chunks, valid_on="2021-12-31")) == ["old", "undated", "at", "eu"]
    in_2022 = ["new", "undated", "at", "eu"]
    assert names(filter_chunks(chunks, valid_on="2022-01-01")) == in_2022
    assert names(filter_chunks(chunks, valid_on=datetime(2022, 1, 1, 23, 59))) == in_2022  # its day
    assert names(filter_chunks(chunks, valid_on="1999-12-31")) == ["capped", "undated"]
    assert names(filter_chunks(chunks, valid_on=date(2000, 1, 1))) == ["undated"]

    assert filter_chunks(chunks) == chunks and filter_chunks(iter(chunks)) == chunks
    assert names(filter_chunks(chunks, jurisdictions=["EU", "AT"])) == ["at", "eu"]  # input order
    assert names(filter_chunks(chunks, jurisdictions="DE")) == ["old", "new", "capped", "undated"]
    assert names(filter_chunks(reversed(chunks), jurisdictions={"DE"})) == ["undated", "capped", "new", "old"]
    assert filter_chunks(chunks, jurisdictions=[]) == [] and filter_chunks(chunks, jurisdictions="CH") == []
    assert names(filter_chunks(chunks, doc_types=["decision", "regulation"])) == ["undated", "eu"]
    assert names(
        filter_chunks(chunks, jurisdictions=["DE"], valid_on="2022-06-30", doc_types=["statute"])
    ) == ["new"]

    for bad in (dict(jurisdictions=["DE", "FR"]), dict(jurisdictions="de"), dict(doc_types=["Statute"])):
        with pytest.raises(ValueError, match="unknown"):
            filter_chunks(chunks, **bad)
    with pytest.raises(ValueError):
        filter_chunks(chunks, valid_on="31.12.2021")
    with pytest.raises(TypeError):
        filter_chunks(chunks, valid_on=20211231)
    broken = at("broken", valid_from="2021-13-01")
    assert filter_chunks([broken]) == [broken]  # no date filter, no date read
    with pytest.raises(ValueError, match=r"x:broken#1: valid_from '2021-13-01'"):
        filter_chunks([broken], valid_on="2021-06-01")
    with pytest.raises(ValueError, match=r"x:broken#1"):
        valid_on(at("broken", valid_to="soon"), "2021-06-01")

    real = [c for d in _fixture_corpus() for c in chunk_document(d)]
    before = {c.doc_id for c in filter_chunks(real, valid_on="2025-12-09")}
    on = {c.doc_id for c in filter_chunks(real, valid_on="2025-12-10")}
    assert on - before == {"gii:beurkg:1", "gii:beurkg:2", "gii:beurkg:3"}  # BeurkG's Stand, inclusive
    swiss_cases = filter_chunks(real, jurisdictions=["CH"], doc_types=["decision"])
    assert {c.doc_id for c in swiss_cases} == {"bger:CH_BGer_004_4A-102-2026_2026-08-17"}
    assert filter_chunks(swiss_cases, valid_on="2026-08-16") == []  # not citable before it was decided


def _build_index(script, tmp_path, name: str, docs: list[Document]) -> Path:
    source = tmp_path / f"{name}.jsonl"
    source.write_text("".join(d.to_json() + "\n" for d in docs))
    out = tmp_path / name
    assert script("build_index").main(["--kind", "bm25", "--docs", str(source), "--out", str(out)]) == 0
    return out


def _snapshot(directory: Path) -> dict:
    return {p.relative_to(directory): p.read_bytes() for p in sorted(directory.rglob("*")) if p.is_file()}


def test_index_registry_versions_increment_idempotently_and_atomically(tmp_path, script, monkeypatch):
    """index.py: register records index.json + the sha256 of chunks.jsonl as the next version;
    the same directory with unchanged chunks is a no-op; a rebuild in place is a new version; a failed
    write leaves the registry as it was and no temporary file behind."""
    import hashlib

    from lexhybrid.retrieval import index as idx

    corpus = _fixture_corpus()
    a = _build_index(script, tmp_path, "a", corpus)
    registry = tmp_path / "registry" / "versions.json"
    assert idx.versions(registry) == [] and idx.latest(registry) is None

    v1 = idx.register(a, registry)
    manifest = json.loads((a / "index.json").read_text())
    assert v1 == idx.IndexVersion(
        version=1, kind="bm25", built_at=manifest["built_at"], index_dir=str(a.resolve()),
        chunks=manifest["chunks"], chunks_sha256=hashlib.sha256((a / "chunks.jsonl").read_bytes()).hexdigest(),
        documents=len(corpus), inputs=tuple(manifest["inputs"]),
    )  # fmt: skip
    written = registry.read_bytes()
    assert json.loads(written) == {"versions": [v1.to_dict()]}
    assert idx.register(a, registry) == v1 and registry.read_bytes() == written  # no-op, not rewritten
    assert idx.register(tmp_path / "x" / ".." / "a", registry) == v1  # the same directory, spelled otherwise

    b = _build_index(script, tmp_path, "b", corpus[:3])
    v2 = idx.register(b, registry)
    assert (v2.version, v2.documents, v2.index_dir) == (2, 3, str(b.resolve()))
    assert idx.versions(registry) == [v1, v2] and idx.latest(registry) == v2

    _build_index(script, tmp_path, "a", corpus[1:])  # rebuilt in place with other chunks
    v3 = idx.register(a, registry)
    assert v3.version == 3 and v3.index_dir == v1.index_dir and v3.chunks_sha256 != v1.chunks_sha256
    assert idx.register(a, registry) == v3 and [v.version for v in idx.versions(registry)] == [1, 2, 3]

    c = _build_index(script, tmp_path, "c", corpus[:1])
    before = registry.read_bytes()

    def fail(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(idx.os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        idx.register(c, registry)
    monkeypatch.undo()
    assert registry.read_bytes() == before
    assert [p.name for p in registry.parent.iterdir()] == ["versions.json"]  # no temporary file left
    assert idx.register(c, registry).version == 4
    assert [p.name for p in registry.parent.iterdir()] == ["versions.json"]

    with pytest.raises(FileNotFoundError, match="index.json"):
        idx.register(tmp_path / "registry", registry)
    registry.write_text("[]")
    with pytest.raises(ValueError, match="not an index registry"):
        idx.versions(registry)


def test_resync_index_records_what_would_change_and_leaves_the_index_alone(tmp_path, script, capsys):
    """scripts/resync_index.py: the documents an index was built from (documents.jsonl) against
    the current ones -- added, removed, changed (text or the metadata chunks carry; retrieved_at does
    not count) -- reported exactly, one RESYNC line printed, and the index directory unchanged."""
    import dataclasses
    import shutil

    corpus = _fixture_corpus()
    general = Document(
        id="fineweb2_de:x", source="fineweb2_de", jurisdiction="DE", doc_type="general",
        licence="ODC-By-1.0", commercial_safe=True, text="Ein ganz allgemeiner Satz über das Wetter.",
    )  # fmt: skip
    out = _build_index(script, tmp_path, "index", [*corpus, general])
    listed = [json.loads(ln) for ln in (out / "documents.jsonl").read_text().splitlines()]
    assert [r["id"] for r in listed] == [d.id for d in corpus]  # general text is not part of the index
    assert [r["sha256"] for r in listed] == [d.sha256 for d in corpus]
    assert all(set(r) == {"id", "sha256", "record_sha256"} for r in listed)
    snapshot = _snapshot(out)
    resync = script("resync_index")

    by_id = {d.id: d for d in corpus}
    planted = by_id["gii:bgb:568"]
    text = "(1) Die Kündigung des Mietverhältnisses bedarf der Textform."
    amended = dataclasses.replace(
        planted, text="§ 568 Form und Inhalt der Kündigung\n" + text, sha256="",
        sections=[Section(label="§ 568 Abs. 1", absatz=1, text=text)],
    )  # fmt: skip
    repealed = dataclasses.replace(by_id["fedlex:220:art_1"], valid_to="2030-12-31")  # same text
    recollected = dataclasses.replace(by_id["gii:beurkg:3"], retrieved_at="2026-10-27T03:00:00+00:00")
    added = dataclasses.replace(planted, id="gii:bgb:569", citation_id="BGB §569")
    replaced = {d.id: d for d in (amended, repealed, recollected)}
    current = [replaced.get(d.id, d) for d in corpus if d.id != "gii:beurkg:2"]
    first, second = tmp_path / "now" / "a.jsonl", tmp_path / "now" / "b.jsonl"
    first.parent.mkdir()
    first.write_text("".join(d.to_json() + "\n" for d in current[:5]))
    second.write_text("".join(d.to_json() + "\n" for d in [*current[5:], added, general]))
    report_path = tmp_path / "reports" / "resync.json"
    capsys.readouterr()

    argv = ["--index", str(out), "--docs", str(first), str(second), "--report", str(report_path)]
    assert resync.main(argv) == 0
    assert capsys.readouterr().out == "RESYNC added=1 removed=1 changed=2\n"
    report = json.loads(report_path.read_text())
    assert {k: v for k, v in report.items() if k != "checked_at"} == {
        "index": str(out), "built_at": json.loads((out / "index.json").read_text())["built_at"],
        "docs": [str(first), str(second)], "indexed_documents": len(corpus),
        "added": ["gii:bgb:569"], "removed": ["gii:beurkg:2"], "changed": ["fedlex:220:art_1", "gii:bgb:568"],
        "metadata_only": ["fedlex:220:art_1"], "unchanged": len(corpus) - 3, "would_change": True,
    }  # fmt: skip
    assert _snapshot(out) == snapshot  # the stub only records

    same = tmp_path / "same.jsonl"
    same.write_text("".join(d.to_json() + "\n" for d in [*corpus, general]))
    assert resync.main(["--index", str(out), "--docs", str(same), "--report", str(report_path)]) == 0
    assert capsys.readouterr().out == "RESYNC added=0 removed=0 changed=0\n"
    assert json.loads(report_path.read_text())["would_change"] is False

    with pytest.raises(SystemExit, match="inside the index"):
        resync.main(["--index", str(out), "--docs", str(same), "--report", str(out / "resync.json")])
    twice = tmp_path / "twice.jsonl"
    twice.write_text(same.read_text() + corpus[0].to_json() + "\n")
    with pytest.raises(SystemExit, match="appears twice"):
        resync.main(["--index", str(out), "--docs", str(twice), "--report", str(report_path)])
    raw = tmp_path / "raw" / "gii.jsonl"
    raw.parent.mkdir()
    raw.write_text(corpus[0].to_json() + "\n")
    with pytest.raises(SystemExit, match="not scrubbed"):
        resync.main(["--index", str(out), "--docs", str(raw), "--report", str(report_path)])
    assert _snapshot(out) == snapshot

    legacy = tmp_path / "legacy"
    shutil.copytree(out, legacy)
    (legacy / "documents.jsonl").unlink()
    with pytest.raises(SystemExit, match="documents.jsonl"):
        resync.main(["--index", str(legacy), "--docs", str(same), "--report", str(report_path)])


def test_index_registry_restoring_older_chunks_is_a_new_version(tmp_path, script):
    """A directory rebuilt X, Y, then X again is three versions: the no-op compares with that
    directory's latest entry, so its latest entry is always what the directory holds."""
    from lexhybrid.retrieval import index as idx

    corpus = _fixture_corpus()
    registry = tmp_path / "versions.json"
    a = _build_index(script, tmp_path, "a", corpus)
    v1 = idx.register(a, registry)
    b = _build_index(script, tmp_path, "b", corpus[:2])
    assert idx.register(b, registry).version == 2
    assert idx.register(_build_index(script, tmp_path, "a", corpus[1:]), registry).version == 3
    v4 = idx.register(_build_index(script, tmp_path, "a", corpus), registry)  # X restored
    assert (v4.version, v4.chunks_sha256) == (4, v1.chunks_sha256)
    assert idx.latest(registry) == v4 and idx.register(a, registry) == v4
    assert idx.register(b, registry).version == 2  # another directory's no-op is unaffected
    assert [v.version for v in idx.versions(registry)] == [1, 2, 3, 4]


def test_write_atomic_gives_the_file_the_mode_a_plain_write_would(tmp_path):
    """mkstemp creates its temporary file 0600; the registry and the re-sync report keep the mode a
    plain open() gives, so other readers of the index directory tree can read them."""
    import stat

    from lexhybrid.retrieval.index import write_atomic

    plain = tmp_path / "plain.json"
    plain.write_text("{}\n")
    write_atomic(tmp_path / "reg" / "versions.json", "{}\n")
    write_atomic(tmp_path / "reg" / "versions.json", '{"versions": []}\n')  # a replace, too
    mode = stat.S_IMODE((tmp_path / "reg" / "versions.json").stat().st_mode)
    assert mode == stat.S_IMODE(plain.stat().st_mode)
    assert (tmp_path / "reg" / "versions.json").read_text() == '{"versions": []}\n'


def test_a_failed_rebuild_leaves_no_index_to_register_or_resync(tmp_path, script):
    """A rebuild that fails after rewriting chunks.jsonl and documents.jsonl removes the old
    index.json, so register and the re-sync refuse the directory instead of recording the old
    manifest against the new, partial files."""
    from lexhybrid.retrieval import index as idx

    corpus = _fixture_corpus()
    out = _build_index(script, tmp_path, "index", corpus)
    registry = tmp_path / "versions.json"
    assert idx.register(out, registry).version == 1
    general = Document(
        id="fineweb2_de:x", source="fineweb2_de", jurisdiction="DE", doc_type="general",
        licence="ODC-By-1.0", commercial_safe=True, text="Ein Satz.",
    )  # fmt: skip
    web = tmp_path / "web.jsonl"
    web.write_text(general.to_json() + "\n")
    with pytest.raises(SystemExit, match="none of an indexed type"):
        script("build_index").main(["--docs", str(web), "--out", str(out)])
    assert not (out / "index.json").exists()
    with pytest.raises(FileNotFoundError, match="index.json"):
        idx.register(out, registry)
    assert [v.version for v in idx.versions(registry)] == [1]
    docs = tmp_path / "docs.jsonl"
    docs.write_text("".join(d.to_json() + "\n" for d in corpus))
    with pytest.raises(SystemExit, match="index.json"):
        script("resync_index").main(
            ["--index", str(out), "--docs", str(docs), "--report", str(tmp_path / "r.json")]
        )
    assert _build_index(script, tmp_path, "index", corpus) == out and idx.register(out, registry).version == 1


def test_cross_source_copies_are_grouped_and_reach_the_chunks(tmp_path, script):
    """Every copy is cited: dedup.jsonl rows with dropped false link copies across sources;
    linked documents group transitively (A-B, B-C -> A, B, C); dropped duplicates link nothing;
    build_index --dedup writes the group's other documents into every chunk's copies."""
    import json

    from lexhybrid.retrieval.chunking import Chunk
    from lexhybrid.retrieval.copies import copy_groups

    rows = [
        {"id": "oldp:1", "duplicate_of": "rii:a", "kind": "near", "jaccard": 0.9, "dropped": False},
        {
            "id": "multilegalpile:7",
            "duplicate_of": "oldp:1",
            "kind": "exact",
            "jaccard": 1.0,
            "dropped": False,
        },
        {"id": "oldp:2", "duplicate_of": "oldp:3", "kind": "exact", "jaccard": 1.0, "dropped": True},
    ]
    groups = copy_groups(rows)
    assert groups["rii:a"] == ("multilegalpile:7", "oldp:1") and groups["multilegalpile:7"] == (
        "oldp:1",
        "rii:a",
    )
    assert "oldp:2" not in groups and "oldp:3" not in groups

    corpus = _fixture_corpus()
    decision = next(d for d in corpus if d.doc_type == "decision")
    scrubbed = tmp_path / "scrubbed"
    scrubbed.mkdir()
    (scrubbed / "legal.jsonl").write_text("".join(d.to_json() + "\n" for d in corpus))
    dedup = tmp_path / "dedup.jsonl"
    dedup.write_text(
        json.dumps(
            {"id": "oldp:99", "duplicate_of": decision.id, "kind": "near", "jaccard": 0.95, "dropped": False}
        )
        + "\n"
    )
    out = tmp_path / "index"
    assert (
        script("build_index").main(
            ["--docs", str(scrubbed / "legal.jsonl"), "--out", str(out), "--dedup", str(dedup)]
        )
        == 0
    )
    chunks = [Chunk.from_json(line) for line in (out / "chunks.jsonl").read_text().splitlines()]
    linked = [c for c in chunks if c.copies]
    assert linked and all(c.doc_id == decision.id and c.copies == ("oldp:99",) for c in linked)
    assert json.loads((out / "index.json").read_text())["chunks_with_copies"] == len(linked)


# -- dense retrieval --------------------------------------------------------------------


def _bag_of_words(texts):
    """A stand-in encoder: hashed, stemmed bag of words, L2-normalised (no model download)."""
    import numpy as np

    from lexhybrid.retrieval.bm25 import tokenize

    out = np.zeros((len(texts), 64), np.float32)
    for row, words in enumerate(tokenize(list(texts))):
        for w in words:
            out[row, sum(map(ord, w)) % 64] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1.0, norms)


def test_dense_index_finds_planted_passage_and_blocks_match_brute_force(tmp_path):
    """DenseIndex: built through a memory-mapped float16 matrix in batches, saved and loaded; the
    blockwise top-k equals a full sort (ties to the earlier chunk); bad inputs are refused."""
    import numpy as np

    from lexhybrid.retrieval.dense import DenseIndex

    chunks = [c for d in _fixture_corpus() for c in chunk_document(d)]
    ids, texts = [c.id for c in chunks], [c.text for c in chunks]
    index = DenseIndex.build(tmp_path / "dense", ids, iter(texts), _bag_of_words, batch_size=3, model="bow")
    assert index.vectors.dtype == np.float16 and index.vectors.shape == (len(chunks), 64)
    assert index.config == {"kind": "dense", "n_chunks": len(chunks), "dim": 64, "model": "bow"}
    by_id = {c.id: c for c in chunks}
    hit = index.query("Schriftform der Kündigung eines Mietverhältnisses", _bag_of_words, k=1)[0]
    assert by_id[hit.chunk_id].citation_id == "BGB §568" and hit.rank == 1

    queries = _bag_of_words(["Kündigung Form", "Beurkundung Notar", "Vertrag"])
    scores = queries @ np.asarray(index.vectors, np.float32).T
    want = [[ids[j] for j in np.lexsort((np.arange(len(ids)), -s))[:7]] for s in scores]
    got = index.query_vectors(queries, k=7, block=4, query_block=2)
    assert [[h.chunk_id for h in r] for r in got] == want
    assert [h.chunk_id for h in DenseIndex.load(tmp_path / "dense").query_vectors(queries, k=7)[0]] == want[0]
    with pytest.raises(ValueError, match="unique"):
        DenseIndex.build(tmp_path / "dup", ["a", "a"], ["x", "y"], _bag_of_words)
    with pytest.raises(ValueError, match="texts for"):
        DenseIndex.build(tmp_path / "short", ["a", "b", "c"], ["x", "y"], _bag_of_words)


def test_dense_build_killed_at_its_time_limit_resumes_from_the_last_batch(tmp_path):
    from lexhybrid.retrieval.dense import PROGRESS_NAME, DenseIndex

    ids = [f"Vertrag {i}" for i in range(10)]
    seen = []

    def encode(batch):
        if len(seen) >= 6:
            raise TimeoutError
        seen.extend(batch)
        return _bag_of_words(batch)

    whole = DenseIndex.build(tmp_path / "whole", ids, ids, _bag_of_words, batch_size=3, model="m")
    with pytest.raises(TimeoutError):
        DenseIndex.build(tmp_path / "d", ids, iter(ids), encode, batch_size=3, model="m")
    seen.clear()
    resumed = DenseIndex.build(tmp_path / "d", ids, iter(ids), lambda b: seen.extend(b) or _bag_of_words(b),
                               batch_size=3, model="m")  # fmt: skip
    assert seen == ids[6:] and resumed.resumed == 6
    assert (resumed.vectors == whole.vectors).all() and not (tmp_path / "d" / PROGRESS_NAME).exists()

    seen.clear()
    with pytest.raises(TimeoutError):
        DenseIndex.build(tmp_path / "e", ids, ids, encode, batch_size=3, model="m")
    seen.clear()
    other = DenseIndex.build(tmp_path / "e", ids, ids, lambda b: seen.extend(b) or _bag_of_words(b),
                             batch_size=3, model="other")  # fmt: skip
    assert seen == ids and other.resumed == 0, "another model's vectors are not reused"


def test_build_index_dense_and_answer_retrieve_fuse_both_rankings(tmp_path, script, monkeypatch):
    """`build_index.py --kind dense` embeds the bm25 build's chunks.jsonl into dense/ and records
    size and speed in index.json; answer.py's retrieve fuses BM25 and dense when dense/ exists."""
    from lexhybrid.retrieval.dense import DenseIndex

    out = _build_index(script, tmp_path, "index", _fixture_corpus())
    bi = script("build_index")
    monkeypatch.setattr(bi, "BgeM3Encoder", lambda *a, **k: _bag_of_words)
    assert bi.main(["--kind", "dense", "--out", str(out)]) == 0
    chunks = [Chunk.from_json(ln) for ln in (out / "chunks.jsonl").read_text().splitlines()]
    dense = json.loads((out / "index.json").read_text())["dense"]
    assert (
        dense["n_chunks"] == len(chunks)
        and dense["bytes"] > len(chunks) * 64 * 2
        and dense["chunks_per_s"] > 0
    )
    assert DenseIndex.load(out / "dense").ids == [c.id for c in chunks]
    with pytest.raises(SystemExit, match="bm25 index there first"):
        bi.main(["--kind", "dense", "--out", str(tmp_path / "nothing")])

    answer = script("answer")
    calls = []
    encode = lambda texts: calls.append(list(texts)) or _bag_of_words(texts)  # noqa: E731
    question = "Schriftform der Kündigung eines Mietverhältnisses"
    flat = lambda q, texts: [0.0] * len(texts)  # noqa: E731
    got = answer.retrieve(question, out, flat, encode=encode)
    assert got[0].citation_id == "BGB §568" and calls == [[question]]
    assert [c.id for c in answer.retrieve(question, out, flat)] != [] and len(
        calls
    ) == 1  # no encoder: BM25 only


def _tiny_eval_set(tmp_path, extra_queries=()):
    """A set in the evalsets layout: q1 has the planted BGB §568 passage, q2 the first chunk."""
    from lexhybrid.data.evalsets.base import EvalSetInfo, Passage, Query, RetrievalSet, write

    chunks = [c for d in _fixture_corpus() for c in chunk_document(d)]
    planted = next(c for c in chunks if c.citation_id == "BGB §568")
    corpus = [Passage(c.id, "\n".join(c.sentences), c.citation_id) for c in chunks]
    queries = [Query("q1", "Schriftform der Kündigung eines Mietverhältnisses"), Query("q2", "Wetter heute")]
    queries += [Query(f"q{i}", text) for i, text in enumerate(extra_queries, start=3)]
    qrels = {"q1": {planted.id: 1}} | {q.id: {chunks[0].id: 1} for q in queries[1:]}
    info = EvalSetInfo(name="tiny", task="retrieval", licence="test", source="test")
    set_dir = tmp_path / "tiny"
    write(RetrievalSet(info, queries, qrels, corpus), set_dir, tmp_path / "manifests")
    return set_dir, corpus, queries


def test_eval_retrieval_scores_four_arms_and_reuses_the_corpus_vectors(tmp_path, script, monkeypatch, capsys):
    """scripts/eval_retrieval.py on a small set in the evalsets layout: bm25, dense, fused and reranked
    rankings scored against the qrels; the reranker's order decides its arm; a second run embeds only
    the queries (the corpus vectors in <set>/dense/ are reused)."""
    set_dir, corpus, queries = _tiny_eval_set(tmp_path)
    ev = script("eval_retrieval")
    embedded = []
    encode = lambda texts: embedded.append(len(texts)) or _bag_of_words(texts)  # noqa: E731
    monkeypatch.setattr(ev, "BgeM3Encoder", lambda *a, **k: encode)
    scorer = lambda q, texts: [float("schriftlichen Form" in t) for t in texts]  # noqa: E731 -- the planted sentence
    monkeypatch.setattr(ev, "CrossEncoderScorer", lambda *a, **k: scorer)
    out = tmp_path / "results.json"
    assert ev.main(["--set-dir", str(set_dir), "--out", str(out)]) == 0
    res = json.loads(out.read_text())["tiny"]
    assert set(res) == {"bm25", "dense", "fused", "reranked", "seconds"}
    assert all(res[a]["queries"] == 2 and res[a]["jurisdiction"] == "DE" for a in ev.ARMS)
    assert res["bm25"]["mrr"] >= 0.5, "BM25 ranks the planted passage first for q1"
    assert res["reranked"]["mrr"] >= 0.5, "the reranker lifts the planted passage to rank 1 for q1"
    assert "RETRIEVAL tiny bm25" in capsys.readouterr().out
    assert embedded == [len(corpus), len(queries)]
    embedded.clear()
    assert ev.main(["--set-dir", str(set_dir), "--out", str(out)]) == 0
    assert embedded == [len(queries)], "the corpus vectors are reused"


def test_eval_retrieval_rerank_shards_and_a_killed_run_leave_only_the_missing_queries(tmp_path, script):
    """Reranked lists are saved per query: after one shard and a killed shard, the scoring run reranks
    only the query left over, and its metrics equal those of one uninterrupted run."""
    import shutil

    set_dir, _, queries = _tiny_eval_set(tmp_path, ["Kündigung Mietverhältnis Form"])
    fresh = shutil.copytree(set_dir, tmp_path / "fresh")
    ev = script("eval_retrieval")
    asked = []

    def scorer(q, texts, limit=None):
        if limit is not None and len(asked) >= limit:
            raise TimeoutError("killed at the time limit")
        asked.append(q)
        return [float("schriftlichen Form" in t) for t in texts]

    assert ev.rank_set(set_dir, _bag_of_words, scorer, "bow", shard=(1, 2)) is None
    assert asked == [queries[1].text]
    asked.clear()
    with pytest.raises(TimeoutError):  # shard 0 owns q1 and q3; dies after q1
        ev.rank_set(set_dir, _bag_of_words, lambda q, t: scorer(q, t, limit=1), "bow", shard=(0, 2))
    asked.clear()
    resumed = ev.rank_set(set_dir, _bag_of_words, scorer, "bow")
    assert asked == [queries[2].text], "only the query the killed shard did not reach"
    assert resumed == ev.rank_set(fresh, _bag_of_words, scorer, "bow")
    shard_file = next((set_dir / "rerank").glob("*/shard_1_of_2.jsonl"))
    row = json.loads(shard_file.read_text())
    assert row["cand"][::-1] != row["cand"]
    shard_file.write_text(json.dumps(row | {"cand": row["cand"][::-1]}) + "\n")
    asked.clear()
    assert ev.rank_set(set_dir, _bag_of_words, scorer, "bow") == resumed
    assert asked == [], "the same candidates in another order (query vectors from another GPU) are reused"
    shard_file.write_text(json.dumps(row | {"cand": row["cand"][:-1] + ["elsewhere"]}) + "\n")
    for _ in range(2):
        assert ev.rank_set(set_dir, _bag_of_words, scorer, "bow") == resumed
    assert asked == [queries[1].text], (
        "another candidate set is reranked once; the stale row does not shadow it"
    )
    with pytest.raises(SystemExit):
        ev.main(["--set-dir", str(set_dir), "--out", str(tmp_path / "r.json"), "--shard", "2/2"])
    out = tmp_path / "saved.json"  # each saved file scored on its own, no model
    assert ev.main(["--set-dir", str(set_dir), "--out", str(out), "--saved-only"]) == 0
    saved = json.loads(out.read_text())[set_dir.name]
    assert sorted(k.split("/")[1] for k in saved) == [
        "rest.jsonl",
        "shard_0_of_2.jsonl",
        "shard_1_of_2.jsonl",
    ]
    shard0 = next(m for k, m in saved.items() if k.endswith("shard_0_of_2.jsonl"))
    assert shard0["queries"] == 1 and shard0["mrr"] == 1.0, "q1: the reranker puts the planted passage first"


def test_dense_ties_go_to_the_earlier_chunk_whatever_the_block_size(tmp_path):
    """Identical vectors (copies of a passage) tie exactly: the earlier chunk ranks first, and the
    ranking does not depend on how the corpus is blocked."""
    import numpy as np

    from lexhybrid.retrieval.dense import DenseIndex

    base = np.eye(4, dtype=np.float32)
    vecs = base[[0, 1, 0, 2, 0, 3, 0, 1, 0]]  # chunk 0, 2, 4, 6, 8 share one vector
    ids = [f"c{i}" for i in range(len(vecs))]
    index = DenseIndex.build(tmp_path / "d", ids, ids, lambda texts: vecs[[ids.index(t) for t in texts]])
    q = base[[0]]
    for block in (1, 2, 3, 4, 100):
        assert [h.chunk_id for h in index.query_vectors(q, k=4, block=block)[0]] == ["c0", "c2", "c4", "c6"]


def test_rebuilt_index_drops_stale_vectors_and_answer_ignores_partial_or_foreign_dense(
    tmp_path, script, monkeypatch
):
    from lexhybrid.retrieval.dense import CONFIG_NAME

    corpus = _fixture_corpus()
    out = _build_index(script, tmp_path, "index", corpus)
    bi = script("build_index")
    monkeypatch.setattr(bi, "BgeM3Encoder", lambda *a, **k: _bag_of_words)
    assert bi.main(["--kind", "dense", "--out", str(out)]) == 0
    _build_index(script, tmp_path, "index", corpus[:2])  # rebuilt in place with fewer chunks
    assert not (out / "dense").exists(), "vectors of the old chunks are removed"
    assert bi.main(["--kind", "dense", "--out", str(out)]) == 0
    (out / "dense" / CONFIG_NAME).unlink()  # an interrupted dense build writes its config last
    answer = script("answer")
    flat = lambda q, texts: [0.0] * len(texts)  # noqa: E731
    calls = []
    encode = lambda texts: calls.append(1) or _bag_of_words(texts)  # noqa: E731
    answer.retrieve("Kündigung", out, flat, encode=encode)
    assert calls == [], "a partial dense/ is not used"


def test_eval_retrieval_keeps_every_text_of_a_repeated_passage_id(script):
    from types import SimpleNamespace as P

    ev = script("eval_retrieval")
    rows = [P(id="a", text="x"), P(id="b", text="y"), P(id="a", text="x"), P(id="a", text="z")]
    assert [(p.id, p.text) for p in ev.unique_passages(rows, {"b"})] == [("a", "x"), ("b", "y"), ("a#1", "z")]
    with pytest.raises(SystemExit, match="more than one text"):
        ev.unique_passages(rows, {"a"})


def test_eval_retrieval_refuses_qrels_for_queries_it_cannot_rank(tmp_path, script):
    from lexhybrid.data.evalsets.base import EvalSetInfo, Passage, Query, RetrievalSet, write

    corpus = [Passage("p1", "Kündigung Mietvertrag Schriftform"), Passage("p2", "Testament Erblasser")]
    qrels = {"q1": {"p1": 1}, "ghost": {"p2": 1}}
    info = EvalSetInfo(name="tiny", task="retrieval", licence="test", source="test")
    write(
        RetrievalSet(info, [Query("q1", "Schriftform Kündigung")], qrels, corpus),
        tmp_path / "s",
        tmp_path / "m",
    )
    ev = script("eval_retrieval")
    with pytest.raises(SystemExit, match="missing from queries.jsonl"):
        ev.rank_set(tmp_path / "s", _bag_of_words, lambda q, t: [0.0] * len(t), "bow")


def test_eval_retrieval_rebuilds_vectors_when_texts_change_or_the_cache_is_broken(tmp_path):
    import importlib.util
    from types import SimpleNamespace

    from lexhybrid.retrieval.dense import IDS_NAME, VECTORS_NAME
    from tests.conftest import REPO_ROOT

    spec = importlib.util.spec_from_file_location("ev", REPO_ROOT / "scripts" / "eval_retrieval.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    calls = []
    encode = lambda texts: calls.append(len(texts)) or _bag_of_words(texts)  # noqa: E731
    passages = [SimpleNamespace(id=f"p{i}", text=f"Kündigung Frist {i}") for i in range(5)]
    d = tmp_path / "dense"
    ev.dense_index(d, passages, encode, "bow")
    ev.dense_index(d, passages, encode, "bow")
    assert calls == [5], "unchanged corpus: cached"
    passages[2] = SimpleNamespace(id="p2", text="Testament")
    ev.dense_index(d, passages, encode, "bow")
    assert calls == [5, 5], "same ids, other text: rebuilt"
    (d / IDS_NAME).write_text('["p0"]')  # ids no longer match the vectors
    ev.dense_index(d, passages, encode, "bow")
    assert calls == [5, 5, 5] and (d / VECTORS_NAME).exists()
