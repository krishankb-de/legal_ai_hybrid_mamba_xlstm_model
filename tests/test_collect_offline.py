"""The collectors' and fetchers' network loops, offline (plans P3-E..Q; CI has no network).

Each test replaces the module's ``http_get`` / ``http_post`` (or the dataset reader) with a fake web
that answers from the fixtures, and checks what the loop does with it: the order it asks in, the
paging and cursors it follows, where the limit stops it, what it skips. The live behaviour stays
with the ``network`` smokes of the weekly CI job.
"""

import gzip
import io
import json
import re
import zipfile

import pytest

from tests.conftest import REPO_ROOT

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "collectors"
EVALSETS = REPO_ROOT / "tests" / "fixtures" / "evalsets"


class Resp:
    def __init__(self, content: bytes = b"", data=None):
        self.content, self._data, self.status_code, self.encoding = content, data, 200, "utf-8"

    def json(self):
        return self._data if self._data is not None else json.loads(self.content)

    @property
    def text(self) -> str:
        return self.content.decode(self.encoding)


class FakeWeb:
    """``get``/``post`` answering from ``routes`` (regex on the URL -> function(url, params, json))."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def _answer(self, method, url, params=None, json_body=None):
        self.calls.append((method, url, params, json_body))
        for pattern, answer in self.routes:
            if re.search(pattern, url):
                return answer(url, params, json_body)
        raise AssertionError(f"unexpected {method} {url}")

    def get(self, url, params=None, headers=None, **kw):
        return self._answer("GET", url, params)

    def post(self, url, json=None, headers=None, **kw):
        return self._answer("POST", url, None, json)


def _zip(name: str, data: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, data)
    return buf.getvalue()


# -- statutes and decisions -------------------------------------------------------------------


def test_gii_collector_orders_core_codes_first_and_stops_at_the_limit(monkeypatch):
    from lexhybrid.data.corpus.collectors import gii

    law = _zip("x.xml", (FIXTURES / "gii" / "beurkg_excerpt.xml").read_bytes())
    web = FakeWeb([
        (r"gii-toc\.xml$", lambda *a: Resp((FIXTURES / "gii" / "toc_excerpt.xml").read_bytes())),
        (r"/xml\.zip$", lambda *a: Resp(law)),
    ])  # fmt: skip
    monkeypatch.setattr(gii, "http_get", web.get)
    docs = list(gii.GIICollector(first=("gg", "bgb")).iter_documents(limit=4))
    zips = [url for _, url, _, _ in web.calls if url.endswith("xml.zip")]
    assert zips == [
        "https://www.gesetze-im-internet.de/gg/xml.zip",
        "https://www.gesetze-im-internet.de/bgb/xml.zip",
    ]
    assert [d.id for d in docs] == ["gii:gg:1", "gii:gg:2", "gii:gg:3", "gii:bgb:1"]


def test_rii_collector_goes_newest_first(monkeypatch):
    from lexhybrid.data.corpus.collectors import rii

    xml = {p.stem.split("-", 1)[1]: p.read_bytes() for p in (FIXTURES / "rii").glob("jb-*.xml")}
    fallback = xml["KARE600065578"]
    web = FakeWeb([
        (r"rii-toc\.xml$", lambda *a: Resp((FIXTURES / "rii" / "toc_excerpt.xml").read_bytes())),
        (r"\.zip$", lambda url, *a: Resp(_zip("d.xml", xml.get(re.search(r"jb-(\w+)\.zip", url).group(1), fallback)))),
    ])  # fmt: skip
    monkeypatch.setattr(rii, "http_get", web.get)
    docs = list(rii.RIICollector().iter_documents(limit=2))
    assert web.calls[1][1].endswith("jb-JURE100055033.zip")  # 2010-01-14, the newest in the excerpt
    assert docs[0].id == "rii:JURE100055033" and len(docs) == 2


def test_rii_collector_skips_a_document_that_keeps_failing(monkeypatch, capsys):
    """Job 2589358_1: one document's connection dropped five times and the whole RII crawl (3.8 h,
    13,000 documents) died. That document is now skipped and logged; the crawl goes on."""
    from lexhybrid.data.corpus.collectors import rii
    from lexhybrid.data.corpus.collectors.base import FetchError

    xml = {p.stem.split("-", 1)[1]: p.read_bytes() for p in (FIXTURES / "rii").glob("jb-*.xml")}
    fallback = xml["KARE600065578"]

    def document(url, *a):
        if url.endswith("jb-JURE100055033.zip"):  # the newest, so the first fetched
            raise FetchError(f"GET {url} failed after 5 attempts (ConnectionError: RemoteDisconnected)")
        return Resp(_zip("d.xml", xml.get(re.search(r"jb-(\w+)\.zip", url).group(1), fallback)))

    web = FakeWeb([
        (r"rii-toc\.xml$", lambda *a: Resp((FIXTURES / "rii" / "toc_excerpt.xml").read_bytes())),
        (r"\.zip$", document),
    ])  # fmt: skip
    monkeypatch.setattr(rii, "http_get", web.get)
    docs = list(rii.RIICollector().iter_documents(limit=2))
    assert len(docs) == 2 and "rii:JURE100055033" not in {d.id for d in docs}
    assert "rii: skipping https://" in capsys.readouterr().err


def test_skip_failed_documents_stops_on_an_outage(capsys):
    from lexhybrid.data.corpus.collectors.base import FetchError, SkipFailedDocuments

    def fail(url):
        raise FetchError("GET x failed after 5 attempts (HTTP 503)")

    skip = SkipFailedDocuments("src", max_consecutive=3)
    assert skip(fail, "a") is None and skip(fail, "b") is None
    assert skip(lambda url: "ok", "c") == "ok" and skip.in_a_row == 0, "a success resets the run"
    assert skip(fail, "d") is None and skip(fail, "e") is None
    with pytest.raises(FetchError, match="3 document fetches failed in a row"):
        skip(fail, "f")
    assert skip.skipped == 5 and capsys.readouterr().err.count("src: skipping") == 5


def test_oldp_collector_reads_the_bulk_dump(monkeypatch, tmp_path):
    """Job 2589358_2: the OLDP API stopped paging past page 10, so the collector reads the bulk dump
    on the hub (gated parquet shards at a pinned revision) in shard order. A row carries the API's
    case fields, so the documents equal the parser's on the same cases."""
    import huggingface_hub
    import pyarrow as pa
    import pyarrow.parquet as pq

    from lexhybrid.data.corpus.collectors import oldp

    cases = [json.loads(p.read_text()) for p in sorted((FIXTURES / "oldp").glob("case_*.json"))]
    rows = [
        {
            "id": c["id"],
            "slug": c.get("slug"),
            "court": {"name": c["court"]["name"], "slug": c["court"].get("slug")},
            "file_number": c["file_number"],
            "date": c["date"],
            "content": c["content"],
            "markdown_content": "not read",
        }
        for c in cases
    ]
    shards = tmp_path / oldp.CONFIG
    shards.mkdir()
    pq.write_table(pa.Table.from_pylist(rows[:2]), shards / "train-00000-of-00002.parquet", row_group_size=1)
    pq.write_table(pa.Table.from_pylist(rows[2:]), shards / "train-00001-of-00002.parquet")
    globs = []

    class LocalHub:
        def glob(self, pattern):
            globs.append(pattern)
            return [str(p) for p in sorted(shards.glob("train-*.parquet"), reverse=True)]

        def open(self, path, mode):
            return open(path, mode)

    monkeypatch.setattr(huggingface_hub, "HfFileSystem", LocalHub)
    docs = list(oldp.OLDPCollector().iter_documents(limit=None))
    assert globs == [f"datasets/{oldp.REPO}@{oldp.REVISION}/{oldp.CONFIG}/train-*.parquet"]
    assert [d.id for d in docs] == [f"oldp:{c['id']}" for c in cases]
    assert [d.text for d in docs] == [oldp.parse_oldp_case(c).text for c in cases]
    assert len(list(oldp.OLDPCollector().iter_documents(limit=2))) == 2


def _ris_page(refs):
    return {"OgdSearchResult": {"OgdDocumentResults": {"OgdDocumentReference": refs}}}


def test_ris_collector_takes_norms_and_decisions_half_and_half(monkeypatch):
    from lexhybrid.data.corpus.collectors import ris

    refs = {
        n: json.loads((FIXTURES / "ris" / f"{n}.json").read_text())
        for n in ("NOR12018413", "NOR40172917", "NOR12019037")
    }
    jjt = json.loads((FIXTURES / "ris" / "JJT_20260917_OGH0002_0070OB00143_26V0000_000.json").read_text())

    def bundesrecht(url, params, _):
        return Resp(data=_ris_page(list(refs.values()) if params["Seitennummer"] == "1" else []))

    def judikatur(url, params, _):
        return Resp(data=_ris_page([jjt] if params["Seitennummer"] == "1" else []))

    def xml(url, *a):
        stem = url.rsplit("/", 1)[1].removesuffix(".xml")
        return Resp((FIXTURES / "ris" / f"{stem}.xml").read_bytes())

    web = FakeWeb([(r"/Bundesrecht$", bundesrecht), (r"/Judikatur$", judikatur), (r"\.xml$", xml)])
    monkeypatch.setattr(ris, "http_get", web.get)
    limited = list(ris.RISCollector(codes=("ABGB",)).iter_documents(limit=3))
    assert [d.doc_type for d in limited] == ["statute", "statute", "decision"]  # 2 norms (3+1)//2, 1 decision
    everything = list(ris.RISCollector(codes=("ABGB",)).iter_documents(limit=None))
    assert len(everything) == 4 and everything[-1].citation_id == "OGH 7 Ob 143/26v"


def test_fedlex_collector_lists_every_act_only_after_the_seeds(monkeypatch):
    from lexhybrid.data.corpus.collectors import fedlex

    versions = json.loads((FIXTURES / "fedlex" / "versions_220.json").read_text())
    xml = (FIXTURES / "fedlex" / "or_excerpt.xml").read_bytes()

    def sparql(url, params, _):
        q = params["query"]
        if "SELECT DISTINCT ?sr WHERE" in q:
            return Resp(data={"results": {"bindings": [{"sr": {"value": "220"}}, {"sr": {"value": "999"}}]}})
        return Resp(data=versions if 'str(?sr) = "220"' in q else {"results": {"bindings": []}})

    web = FakeWeb([(r"sparqlendpoint$", sparql), (r"filestore", lambda *a: Resp(xml))])
    monkeypatch.setattr(fedlex, "http_get", web.get)
    docs = list(fedlex.FedlexCollector(sr=("220",), as_of="2026-09-27").iter_documents(limit=2))
    assert [d.citation_id for d in docs] == ["OR Art. 1", "OR Art. 6a"]
    assert not any("SELECT DISTINCT ?sr WHERE" in (p or {}).get("query", "") for _, _, p, _ in web.calls)
    web.calls.clear()
    everything = list(fedlex.FedlexCollector(sr=("220",), as_of="2026-09-27").iter_documents(limit=None))
    assert (
        len(everything) == 5
        and sum("SELECT DISTINCT ?sr WHERE" in (p or {}).get("query", "") for _, _, p, _ in web.calls) == 1
    )
    assert fedlex.FedlexCollector().as_of  # defaults to today


def test_bger_collector_pages_by_search_after_and_skips_blocked_decisions(monkeypatch):
    from lexhybrid.data.corpus.collectors import bger

    names = sorted(p.stem for p in (FIXTURES / "bger").glob("CH_BGer_*.html"))
    hits = [
        {"_source": json.loads((FIXTURES / "bger" / f"{n}.json").read_text()), "sort": [i, n]}
        for i, n in enumerate(names)
    ]
    blocked = names[1]

    def search(url, params, body):
        return Resp(
            data={
                "hits": {
                    "hits": hits[:2]
                    if "search_after" not in body
                    else (hits[2:] if body["search_after"] == hits[1]["sort"] else [])
                }
            }
        )

    def html(url, *a):
        return Resp((FIXTURES / "bger" / url.rsplit("/", 1)[1]).read_bytes())

    web = FakeWeb([
        (r"Blockliste\.json$", lambda *a: Resp(data={"CH_BGer": [blocked]})),
        (r"_searchV2\.php$", search),
        (r"\.html$", html),
    ])  # fmt: skip
    monkeypatch.setattr(bger, "http_get", web.get)
    monkeypatch.setattr(bger, "http_post", web.post)
    docs = list(bger.BGerCollector(page_size=2).iter_documents(limit=None))
    assert [d.id for d in docs] == [f"bger:{n}" for n in names if n != blocked]
    posts = [body for method, _, _, body in web.calls if method == "POST"]
    assert "search_after" not in posts[0] and posts[1]["search_after"] == hits[1]["sort"] and len(posts) == 3


def test_dip_collector_follows_the_cursor(monkeypatch):
    from lexhybrid.data.corpus.collectors import dip

    records = [json.loads(p.read_text()) for p in sorted((FIXTURES / "dip").glob("*.json"))]

    def page(url, params, _):
        if "cursor" not in params:
            return Resp(data={"documents": records[:2], "cursor": "c1"})
        return Resp(data={"documents": records[2:], "cursor": "c1" if params["cursor"] == "c1" else "c2"})

    web = FakeWeb([(r"/drucksache-text$", page)])
    monkeypatch.setattr(dip, "http_get", web.get)
    monkeypatch.setenv("DIP_API_KEY", "personal-key")
    docs = list(dip.DIPCollector(types=("Gesetzentwurf",), zuordnung="BT").iter_documents(limit=None))
    assert len(docs) == 3 and len(web.calls) == 2  # the cursor did not move: the last page
    assert all(p["f.zuordnung"] == "BT" for _, _, p, _ in web.calls)
    assert dip.api_key() == "personal-key"
    assert len(list(dip.DIPCollector(types=("Gesetzentwurf",)).iter_documents(limit=1))) == 1


def test_fineweb_and_multilegalpile_collectors_read_their_rows(monkeypatch):
    import datasets

    from lexhybrid.data.corpus.collectors import fineweb2_de, multilegalpile

    rows = json.loads((FIXTURES / "fineweb2_de" / "rows.json").read_text())
    seen = {}

    class Stream:
        def __init__(self, items):
            self.items = items

        def __iter__(self):
            return iter(self.items)

        def shuffle(self, seed, buffer_size):
            seen.update(shuffle=(seed, buffer_size))
            return Stream(self.items[::-1])

    def load_dataset(path, **kw):
        seen.update(kw, path=path)
        return Stream([{"id": "<urn:uuid:empty>", "text": "  "}, *rows])

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    docs = list(fineweb2_de.FineWeb2DECollector().iter_documents(limit=2))
    assert (
        len(docs) == 2
        and seen["streaming"] is True
        and seen["path"] == "HuggingFaceFW/fineweb-2"
        and seen["name"] == "deu_Latn"
        and seen["shuffle"] == (0, 1000)  # sampled across the dataset's files (dumps), seeded
    )
    in_order = list(fineweb2_de.FineWeb2DECollector(shuffle_seed=None).iter_documents(limit=None))
    assert [d.id for d in docs] == [d.id for d in in_order[::-1][:2]]

    fixture = json.loads((FIXTURES / "multilegalpile" / "rows.json").read_text())
    by_subset = {tuple(k.split("_", 2)): v["row"] for k, v in fixture.items()}
    monkeypatch.setattr(
        multilegalpile, "iter_rows", lambda lang, typ, src, rev: iter([by_subset[(lang, typ, src)]] * 3)
    )
    collector = multilegalpile.MultiLegalPileCollector(subsets=list(by_subset)[:2])
    got = list(collector.iter_documents(limit=4))
    assert [d.id.rsplit(":", 1)[1] for d in got] == ["0", "1", "0", "1"] and all(d.research_only for d in got)
    assert len(list(collector.iter_documents(limit=None))) == 6


# -- evaluation sets --------------------------------------------------------------------------


def test_evalset_fetchers_and_cli(monkeypatch, tmp_path):
    from lexhybrid.data.evalsets import base, gerdalir, gerlayqa, gerlerb, legalquad

    corpus = gzip.compress((EVALSETS / "gerlerb" / "corpus.trec").read_bytes())
    files = {"topics.txt": (EVALSETS / "gerlerb" / "topics.txt").read_bytes(),
             "qrels.txt": (EVALSETS / "gerlerb" / "qrels.txt").read_bytes(), "corpus.trec.gz": corpus}  # fmt: skip
    web = FakeWeb(
        [
            (
                r"/files/([\w.]+)/content$",
                lambda url, *a: Resp(files[re.search(r"/files/([\w.]+)/content$", url).group(1)]),
            )
        ]
    )
    monkeypatch.setattr(gerlerb, "http_get", web.get)
    lerb = gerlerb.fetch(limit=2)
    assert [q.id for q in lerb.queries] == ["206", "207"] and len(lerb.corpus) == 2

    lines = {
        n: (EVALSETS / "gerdalir" / f"{n}.jsonl").read_text().splitlines(True)
        for n in ("qrels", "queries", "corpus")
    }
    monkeypatch.setattr(
        gerdalir, "_lines", lambda path: iter(lines[path.split("/")[0].removesuffix(".jsonl")])
    )
    assert len(gerdalir.fetch(limit=2).queries) == 2

    squad = json.loads((EVALSETS / "legalquad" / "LegalQuAD_excerpt.json").read_text())
    monkeypatch.setattr(legalquad, "http_get", lambda url, **kw: Resp(data=squad))
    rows = json.loads((EVALSETS / "gerlayqa" / "bgb_eval_excerpt.json").read_text())
    monkeypatch.setattr(gerlayqa, "http_get", lambda url, **kw: Resp(data=rows))
    assert len(legalquad.fetch(limit=2).items) == 2 and len(gerlayqa.fetch().items) == 3

    rc = base.cli(
        "gerlayqa",
        gerlayqa.fetch,
        ["--limit", "0", "--out", str(tmp_path / "set"), "--manifest-dir", str(tmp_path / "m")],
    )
    assert rc == 0 and (tmp_path / "set" / "items.jsonl").read_text().count("\n") == 3
    assert (tmp_path / "m" / "evalset_gerlayqa.jsonl").exists()


# -- tools: changelog, packing, scrub ---------------------------------------------------------


def _write_jsonl(path, docs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(d.to_json() + "\n" for d in docs))


def test_changelog_packing_and_scrub_entry_points(monkeypatch, tmp_path, capsys):
    from lexhybrid.data import packing
    from lexhybrid.data import tokenizer as T
    from lexhybrid.data.corpus import changelog, scrub_ler
    from lexhybrid.data.corpus.collectors.fedlex import parse_fedlex_xml
    from lexhybrid.data.corpus.collectors.oldp import parse_oldp_case
    from lexhybrid.data.corpus.scrub_ler import Entity
    from lexhybrid.data.schema import Document

    statutes = parse_fedlex_xml((FIXTURES / "fedlex" / "or_excerpt.xml").read_bytes())
    _write_jsonl(tmp_path / "raw" / "fedlex" / "fedlex.jsonl", statutes)
    out = tmp_path / "changelog.jsonl"
    assert changelog.main(["--since", "2020-01-01", "--raw", str(tmp_path / "raw"), "--out", str(out)]) == 0
    assert json.loads(out.read_text().splitlines()[0]) == {"since": "2020-01-01", "changes": 1}

    class ToyTok:
        def __call__(self, text, add_special_tokens=False, split_special_tokens=False):
            return {"input_ids": [min(ord(c), 999) + 1 for c in text]}

    monkeypatch.setattr(T, "load_tokenizer", lambda with_specials=True: ToyTok())
    cases = [
        parse_oldp_case(json.loads(p.read_text())) for p in sorted((FIXTURES / "oldp").glob("case_*.json"))
    ]
    _write_jsonl(tmp_path / "oldp.jsonl", cases)
    rc = packing.main(
        [
            "--in",
            str(tmp_path / "oldp.jsonl"),
            "--row-len",
            "4096",
            "--out",
            str(tmp_path / "shards"),
            "--max-rows",
            "2",
        ]
    )
    assert rc == 0 and len(packing.read_rows(tmp_path / "shards" / "shard-00000.parquet")) == 2

    with pytest.raises(FileNotFoundError, match="uv sync --locked --project envs/scrub"):
        scrub_ler.run_worker(cases[:1], python=tmp_path / "missing" / "python")  # the real worker's check
    entities = {cases[0].id: [Entity(0, 5, "PER", 0.9)]}  # "Tenor", tagged so it is replaced and logged

    def fake_stream(docs, python=None, args=(), worker=None, in_flight=1024, stats=None):
        docs = list(docs)
        stats.update({"docs": len(docs)})
        yield from ((d, entities.get(d.id, [])) for d in docs)

    monkeypatch.setattr(scrub_ler, "stream_worker", fake_stream)  # scrub_file's one seam to the worker
    key = "ab" * 32
    monkeypatch.setenv("LEXHYBRID_SCRUB_KEY", key)
    rc = scrub_ler.main(["--source", "oldp", "--in", str(tmp_path / "oldp.jsonl"), "--out", str(tmp_path / "s.jsonl"),
                         "--log", str(tmp_path / "log.jsonl")])  # fmt: skip
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0 and (summary["docs"], summary["entities"], summary["worker"]) == (3, 1, {"docs": 3})
    first = Document.from_json((tmp_path / "s.jsonl").read_text().splitlines()[0])
    assert first.text.startswith("[PER_1]\n") and "Tenor" not in first.text
    (row,) = [json.loads(x) for x in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert row["entity_hmac"] == scrub_ler.entity_hash(
        "Tenor", bytes.fromhex(key)
    ) and "Tenor" not in json.dumps(row)


# -- tokenizer glue ---------------------------------------------------------------------------


class FakeQwen:
    """Just enough of a Hugging Face tokenizer for ``load_tokenizer``'s checks."""

    def __init__(self, base=151669, eos=151643):
        self.vocab = {f"<|t{i}|>": i for i in range(3)}
        self.vocab["<|endoftext|>"] = eos
        self.base, self.eos_token, self.eos_token_id, self.unk_token_id = base, "<|endoftext|>", eos, None
        self.added: list[str] = []

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.vocab.get(tokens)
        return [self.vocab.get(t) for t in tokens]

    def add_special_tokens(self, specials, replace_extra_special_tokens=True):
        for t in specials["additional_special_tokens"]:
            self.vocab[t] = self.base + len(self.added)
            self.added.append(t)
        return len(self.added)

    def __len__(self):
        return self.base + len(self.added)

    def __call__(self, text, add_special_tokens=False, split_special_tokens=False):
        return {"input_ids": [1] * len(text) if split_special_tokens else [0]}


def test_tokenizer_loader_checks_eos_and_vocabulary(monkeypatch):
    import transformers

    from lexhybrid.data import tokenizer as T

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: FakeQwen())
    tok = T.load_tokenizer()
    ids = T.pointer_ids(tok)
    assert len(tok) == 151756 and ids["q"] == 151669 and ids["answer"] == 151755
    assert T.encode_document(tok, "abc") == [1, 1, 1]  # split_special_tokens is what encode_document asks for
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: FakeQwen(eos=7))
    with pytest.raises(ValueError, match="EOS"):
        T.load_tokenizer()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: FakeQwen(base=151900))
    with pytest.raises(ValueError, match="do not fit"):
        T.load_tokenizer()
    with pytest.raises(ValueError, match="missing"):
        T.pointer_ids(FakeQwen())  # no specials added


def test_tokenizer_cache_and_cli(monkeypatch, capsys):
    import huggingface_hub
    import transformers

    from lexhybrid.data import tokenizer as T

    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    assert T.cache_dir() == str(T.LOCAL_CACHE)
    monkeypatch.setenv("HF_HOME", "/scratch/.hf")
    assert T.cache_dir() is None
    fetched = {}
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda repo, **kw: fetched.update(repo=repo, **kw) or "/snap"
    )
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: FakeQwen())
    assert T.main(["--download"]) == 0
    assert fetched["repo"] == T.QWEN3_TOKENIZER and fetched["revision"] == T.QWEN3_REVISION
    assert "151756 tokens" in capsys.readouterr().out
