"""The data pipeline at smoke scale (plan P3): schema, licences, collectors on fixtures, scrub,
dedup, changelog, packing, splits, probes and the SFT format. Offline except ``network`` tests."""

import pytest

from lexhybrid.data.schema import Document, Section, parse_citation, sha256_text


def _doc(**kw):
    d = dict(
        id="gii:bgb:573",
        source="gii",
        jurisdiction="DE",
        doc_type="statute",
        licence="DE-UrhG-5",
        commercial_safe=True,
        text="(1) Der Vermieter kann nur kündigen, wenn er ein berechtigtes Interesse hat. (2) Ein Interesse liegt vor, wenn …",
        citation_id="BGB §573",
        url="https://www.gesetze-im-internet.de/bgb/__573.html",
        valid_from="2021-07-01",
        retrieved_at="2026-09-27T12:00:00+00:00",
        sections=[
            Section(
                "§ 573 Abs. 1",
                absatz=1,
                text="Der Vermieter kann nur kündigen, wenn er ein berechtigtes Interesse hat.",
            ),
            Section("§ 573 Abs. 2", absatz=2, text="Ein Interesse liegt vor, wenn …"),
        ],
    )
    d.update(kw)
    return Document(**d)


def test_document_roundtrip():
    """P3-A: to_json/from_json is lossless (umlauts, sections, dates), sha256 is filled and checked."""
    doc = _doc()
    assert doc.sha256 == sha256_text(doc.text)
    back = Document.from_json(doc.to_json())
    assert back == doc and back.sections[1].absatz == 2 and "kündigen" in back.text
    no_text = doc.to_dict(with_text=False)
    assert "text" not in no_text and all("text" not in s for s in no_text["sections"])


@pytest.mark.parametrize(
    "bad,match",
    [
        ({"jurisdiction": "US"}, "jurisdiction"),
        ({"doc_type": "blog"}, "doc_type"),
        ({"licence": ""}, "licence"),
        ({"research_only": True, "commercial_safe": True}, "research_only"),
        ({"valid_from": "01.07.2021"}, "ISO date"),
        ({"id": "rii:x"}, "source"),
        ({"sha256": "0" * 64}, "sha256"),
    ],
)
def test_document_rejects_invalid_records(bad, match):
    with pytest.raises(ValueError, match=match):
        _doc(**bad)


@pytest.mark.parametrize(
    "citation,parts",
    [
        (
            "BGB §573 Abs. 2 Nr. 1",
            {"kind": "statute", "code": "BGB", "paragraph": "573", "absatz": "2", "nummer": "1"},
        ),
        ("ABGB §1295", {"kind": "statute", "code": "ABGB", "paragraph": "1295"}),
        ("OR Art. 97", {"kind": "statute", "code": "OR", "article": "97"}),
        (
            "DSGVO Art. 6 Abs. 1 lit. f",
            {"kind": "statute", "code": "DSGVO", "article": "6", "absatz": "1", "litera": "f"},
        ),
        (
            "BGH VIII ZR 12/20 Rn. 14",
            {"kind": "decision", "court": "BGH", "docket": "VIII ZR 12/20", "randnummer": "14"},
        ),
        (
            "BGer 4A_123/2020 E. 3.2",
            {"kind": "decision", "court": "BGer", "docket": "4A_123/2020", "erwaegung": "3.2"},
        ),
        ("OGH 1 Ob 123/20x", {"kind": "decision", "court": "OGH", "docket": "1 Ob 123/20x"}),
        ("ZGB SchlT Art. 1", {"kind": "statute", "code": "ZGB", "part": "SchlT", "article": "1"}),
        ("SR 221.214.111 Art. 3", {"kind": "statute", "code": "SR 221.214.111", "article": "3"}),
        (
            "OR Art. 40a Abs. 2bis",
            {"kind": "statute", "code": "OR", "article": "40a", "absatz": "2bis"},
        ),
        ("ZGB Art. 334bis", {"kind": "statute", "code": "ZGB", "article": "334bis"}),
        ("BT-Drs. 21/8109", {"kind": "parliament", "series": "BT-Drs.", "number": "21/8109"}),
        (
            "BR-Drs. 506/26 S. 12",
            {"kind": "parliament", "series": "BR-Drs.", "number": "506/26", "page": "12"},
        ),
    ],
)
def test_citation_grammar(citation, parts):
    assert parse_citation(citation) == parts


def test_citation_grammar_rejects_prose():
    with pytest.raises(ValueError):
        parse_citation("see the tenancy rules")


# -- licences (P3-B) ---------------------------------------------------------------------------


def test_unknown_licence_refused():
    """R9 / P3-B: build_corpus stops on 'unknown' and on unregistered licence ids -- it never
    filters a licence problem away silently."""
    from lexhybrid.data.corpus.build import build_corpus
    from lexhybrid.data.corpus.licences import LicenceError, require_known

    with pytest.raises(LicenceError, match="unknown"):
        list(build_corpus([_doc(), _doc(id="gii:x", licence="unknown", commercial_safe=False)]))
    with pytest.raises(LicenceError, match="not in the register"):
        list(build_corpus([_doc(licence="CC-BY-SA-9.9")]))
    assert require_known("CC-BY-4.0").commercial_safe


def test_licence_flags_must_not_overclaim():
    from lexhybrid.data.corpus.build import build_corpus
    from lexhybrid.data.corpus.licences import LicenceError

    nc_marked_safe = _doc(id="gii:nc", licence="CC-BY-NC-SA-4.0", commercial_safe=True)
    with pytest.raises(LicenceError, match="does not permit"):
        list(build_corpus([nc_marked_safe]))
    nc_not_research = _doc(id="gii:nc2", licence="CC-BY-NC-SA-4.0", commercial_safe=False)
    with pytest.raises(LicenceError, match="research-only"):
        list(build_corpus([nc_not_research]))


def test_arms_admit_by_licence():
    """The shippable arm never sees research-only text; the research arm sees both."""
    from lexhybrid.data.corpus.build import build_corpus

    safe = _doc()
    research = _doc(id="gii:mlp", licence="CC-BY-NC-SA-4.0", commercial_safe=False, research_only=True)
    assert [d.id for d in build_corpus([safe, research], arm="commercial_safe")] == [safe.id]
    assert [d.id for d in build_corpus([safe, research], arm="research")] == [safe.id, research.id]
    with pytest.raises(ValueError, match="arm"):
        list(build_corpus([safe], arm="everything"))


def test_register_lists_every_collector_and_licence():
    """The prose register and the code register agree: every collector source has a row, and every
    licence id the register uses is defined in code."""
    import re

    from lexhybrid.data.corpus.licences import LICENCES
    from tests.conftest import REPO_ROOT

    text = (REPO_ROOT / "Docs" / "CORPUS_LICENCE_REGISTER.md").read_text()
    for source in (
        "gii",
        "rii",
        "oldp",
        "eurlex",
        "ris",
        "fedlex",
        "bger",
        "dip",
        "fineweb2_de",
        "multilegalpile",
    ):
        assert f"(`{source}`)" in text, f"no register row for {source}"
    used = set(re.findall(r"`((?:DE|CH)-U[a-zA-Z]+-\d|CC-[A-Z-]+-4\.0|ODbL-1\.0|ODC-By-1\.0)`", text))
    assert used and used <= set(LICENCES), used - set(LICENCES)


# -- collectors: shared machinery (P3-E) -------------------------------------------------------


def test_manifest_roundtrip(tmp_path):
    """P3-E: the manifest keeps every schema field but the text, one line per document."""
    from lexhybrid.data.corpus.manifest import read_manifest, write_manifest

    docs = [_doc(), _doc(id="gii:bgb:574", citation_id="BGB §574")]
    assert write_manifest(docs, tmp_path / "gii.jsonl") == 2
    rows = read_manifest(tmp_path / "gii.jsonl")
    assert [r["id"] for r in rows] == ["gii:bgb:573", "gii:bgb:574"]
    for row, doc in zip(rows, docs):
        assert "text" not in row and all("text" not in s for s in row["sections"])
        assert row["sha256"] == doc.sha256 and row["n_chars"] == len(doc.text)
        assert {k: v for k, v in row.items() if k != "n_chars"} == doc.to_dict(with_text=False)


class _FakeResponse:
    def __init__(self, status, headers=None):
        self.status_code, self.headers = status, headers or {}


class _FakeSession:
    def __init__(self, statuses):
        self.statuses, self.calls = list(statuses), 0

    def get(self, url, **kw):
        self.calls += 1
        status = self.statuses.pop(0)
        if isinstance(status, Exception):
            raise status
        return _FakeResponse(*status) if isinstance(status, tuple) else _FakeResponse(status)


def test_http_get_retries_transient_errors_and_honours_retry_after():
    import requests

    from lexhybrid.data.corpus.collectors.base import FetchError, RateLimiter, http_get

    slept = []
    session = _FakeSession([503, requests.ConnectionError("reset"), (429, {"Retry-After": "7"}), 200])
    out = http_get("https://example.org/x", session=session, limiter=RateLimiter(0), sleep=slept.append)
    assert out.status_code == 200 and session.calls == 4
    assert slept == [1.0, 2.0, 7.0]  # backoff 2**0, 2**1, then the server's Retry-After
    with pytest.raises(FetchError, match="HTTP 404"):
        http_get(
            "https://example.org/y", session=_FakeSession([404]), limiter=RateLimiter(0), sleep=slept.append
        )
    with pytest.raises(FetchError, match="after 3 attempts"):
        http_get(
            "https://example.org/z",
            session=_FakeSession([500] * 3),
            retries=2,
            limiter=RateLimiter(0),
            sleep=slept.append,
        )


def test_rate_limiter_spaces_requests_per_host():
    from lexhybrid.data.corpus.collectors.base import RateLimiter

    t, slept = [0.0], []

    def sleep(dt):
        slept.append(dt)
        t[0] += dt

    limiter = RateLimiter(1.5, clock=lambda: t[0], sleep=sleep)
    limiter.wait("a.org")
    limiter.wait("b.org")  # another host: no wait
    limiter.wait("a.org")
    assert slept == [1.5]


def test_collector_cli_writes_documents_and_manifest(tmp_path):
    from lexhybrid.data.corpus.collectors.base import cli
    from lexhybrid.data.corpus.manifest import read_manifest

    class Fake:
        name, licence, jurisdiction = "gii", "DE-UrhG-5", "DE"

        def iter_documents(self, limit=None):
            for i in range(limit or 5):
                yield _doc(id=f"gii:fake:{i}")

    assert (
        cli(Fake(), ["--limit", "3", "--out", str(tmp_path / "raw"), "--manifest-dir", str(tmp_path / "m")])
        == 0
    )
    lines = (tmp_path / "raw" / "gii.jsonl").read_text().splitlines()
    assert len(lines) == 3 and Document.from_json(lines[0]).id == "gii:fake:0"
    assert len(read_manifest(tmp_path / "m" / "gii.jsonl")) == 3
    # P4-K: `--limit all` collects everything the source has; nothing is left as .partial
    assert (
        cli(Fake(), ["--limit", "all", "--out", str(tmp_path / "raw"), "--manifest-dir", str(tmp_path / "m")])
        == 0
    )
    assert len((tmp_path / "raw" / "gii.jsonl").read_text().splitlines()) == 5
    assert [r["id"] for r in read_manifest(tmp_path / "m" / "gii.jsonl")] == [
        f"gii:fake:{i}" for i in range(5)
    ]
    assert not list(tmp_path.rglob("*.partial"))
    for bad in ("0", "-2", "many"):
        with pytest.raises(SystemExit):
            cli(Fake(), ["--limit", bad])


_ENTRY_POINT_SCRIPT = r"""
import sys, threading, time
from lexhybrid.data.corpus.collectors.base import main
from lexhybrid.data.schema import Document

class Collector:
    name, licence, jurisdiction = "gii", "DE-UrhG-5", "DE"

    def iter_documents(self, limit=None):
        mode = sys.argv[1]
        if mode == "raise":
            raise ConnectionError("network gone")
        # a download worker left behind, as a stopped datasets stream leaves them
        threading.Thread(target=time.sleep, args=(120,)).start()
        for i in range(0 if mode == "empty" else (limit or 3)):
            yield Document(id=f"gii:x:{i}", source="gii", jurisdiction="DE", doc_type="statute",
                           licence="DE-UrhG-5", commercial_safe=True, text=f"Satz {i}.")

main(Collector(), sys.argv[2:])
"""


@pytest.mark.parametrize(
    "mode, args, code",
    [("ok", ["--limit", "2"], 0), ("empty", [], 1), ("raise", [], 1), ("ok", ["--limit", "0"], 2)],
)
def test_collector_entry_point_ends_the_process_at_once(tmp_path, mode, args, code):
    """P4-K: a collector module ends its process right after writing its files (base.main), even
    with a worker thread still alive -- a stopped, shuffled datasets stream left one that kept the
    FineWeb-2 process alive past its summary line (2026-09-30) -- and exits with cli's code."""
    import subprocess
    import sys
    import time

    script = tmp_path / "collect.py"
    script.write_text(_ENTRY_POINT_SCRIPT)
    out, manifests = tmp_path / "raw", tmp_path / "m"
    start = time.monotonic()
    res = subprocess.run(
        [sys.executable, str(script), mode, *args, "--out", str(out), "--manifest-dir", str(manifests)],
        capture_output=True, text=True, timeout=90,
    )  # fmt: skip
    assert time.monotonic() - start < 60, "the lingering worker kept the process alive"
    assert res.returncode == code, res.stdout + res.stderr
    if code == 0:
        assert "gii: 2 documents" in res.stdout and len((out / "gii.jsonl").read_text().splitlines()) == 2
    if mode == "raise":
        assert "ConnectionError: network gone" in res.stderr


def test_every_collector_module_ends_through_the_entry_point():
    import re

    root = FIXTURES.parents[1].parent / "lexhybrid" / "data" / "corpus" / "collectors"
    modules = [p for p in root.glob("*.py") if p.name not in ("__init__.py", "base.py", "htmltext.py")]
    assert len(modules) == 10
    for p in modules:
        tail = p.read_text().split('if __name__ == "__main__":', 1)[1]
        assert re.fullmatch(r"\s*main\(\w+\(\)\)\s*", tail), p.name


def test_collection_that_dies_leaves_the_last_complete_files_alone(tmp_path):
    """A killed or failing at-scale collection (P4-K) streams into *.partial; the finished files of
    the previous run are only replaced when a collection completes."""
    from lexhybrid.data.corpus.collectors.base import run_collector

    class Dies:
        name, licence, jurisdiction = "gii", "DE-UrhG-5", "DE"

        def __init__(self, fail_after):
            self.fail_after = fail_after

        def iter_documents(self, limit=None):
            for i in range(10):
                if self.fail_after is not None and i == self.fail_after:
                    raise ConnectionError("network gone")
                yield _doc(id=f"gii:fake:{i}")

    raw, man = tmp_path / "raw", tmp_path / "m"
    assert run_collector(Dies(None), None, raw, man, progress_every=4) == 10
    with pytest.raises(ConnectionError):
        run_collector(Dies(3), None, raw, man)
    assert len((raw / "gii.jsonl").read_text().splitlines()) == 10  # the complete run survives
    assert len((raw / "gii.jsonl.partial").read_text().splitlines()) == 3


def test_http_cache_replays_responses_without_the_network(tmp_path):
    """--http-cache (P4-K): a restarted collection is served what it already fetched -- GET by URL
    and query, POST by JSON body -- without a request or a rate-limit wait; failures are not kept."""
    import requests

    from lexhybrid.data.corpus.collectors import base

    class Session:
        def __init__(self):
            self.calls = []

        def _reply(self, url, status=200, body=b'{"hits": [1, 2]}'):
            r = requests.Response()
            r.status_code, r._content, r.url, r.encoding = status, body, url, "utf-8"
            r.headers = requests.structures.CaseInsensitiveDict({"Content-Type": "application/json"})
            return r

        def get(self, url, params=None, **kw):
            self.calls.append(("GET", url, params))
            return self._reply(url, 404 if url.endswith("/missing") else 200)

        def post(self, url, json=None, **kw):
            self.calls.append(("POST", url, json))
            return self._reply(url, body=b'{"page": %d}' % json["page"])

    waits = []

    class Limiter:
        def wait(self, host):
            waits.append(host)

    session, limiter = Session(), Limiter()
    try:
        base.set_http_cache(tmp_path / "cache")
        first = base.http_get("https://ex.org/a", params={"q": 1}, session=session, limiter=limiter)
        again = base.http_get("https://ex.org/a", params={"q": 1}, session=session, limiter=limiter)
        other = base.http_get("https://ex.org/a", params={"q": 2}, session=session, limiter=limiter)
        assert first.json() == again.json() == other.json() == {"hits": [1, 2]}
        assert again.reason == "OK (http cache)" and again.headers["content-type"] == "application/json"
        assert [c[2] for c in session.calls] == [{"q": 1}, {"q": 2}] and len(waits) == 2
        p1 = base.http_post("https://ex.org/s", json={"page": 1}, session=session, limiter=limiter)
        p2 = base.http_post("https://ex.org/s", json={"page": 2}, session=session, limiter=limiter)
        assert (p1.json(), p2.json()) == ({"page": 1}, {"page": 2})
        assert base.http_post(
            "https://ex.org/s", json={"page": 1}, session=session, limiter=limiter
        ).json() == {"page": 1}
        assert len(session.calls) == 4
        for _ in range(2):
            with pytest.raises(base.FetchError):
                base.http_get(
                    "https://ex.org/missing", session=session, limiter=limiter, sleep=lambda s: None
                )
        assert len(session.calls) == 6, "a failed request is asked again, never replayed"
        assert not list((tmp_path / "cache").rglob("*.tmp*"))
    finally:
        base.set_http_cache(None)
    base.http_get("https://ex.org/a", params={"q": 1}, session=session, limiter=limiter)
    assert len(session.calls) == 7, "without a cache every call reaches the network"


# -- collector: Gesetze im Internet (P3-F) ------------------------------------------------------

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures" / "collectors"


def test_gii_parser_on_fixture():
    """P3-F: one document per provision, Absätze as sections, lists on their own lines, the Stand
    date as valid_from, licence and jurisdiction from the register; headings and TOC skipped."""
    from lexhybrid.data.corpus.collectors.gii import parse_gii_xml

    xml = (FIXTURES / "gii" / "beurkg_excerpt.xml").read_bytes()
    docs = parse_gii_xml(xml, "beurkg", retrieved_at="2026-09-27T16:00:00+00:00")
    assert [d.citation_id for d in docs] == ["BeurkG §1", "BeurkG §2", "BeurkG §3"]
    assert [d.id for d in docs] == ["gii:beurkg:1", "gii:beurkg:2", "gii:beurkg:3"]
    assert all(
        (d.licence, d.jurisdiction, d.commercial_safe, d.valid_from)
        == ("DE-UrhG-5", "DE", True, "2025-12-10")
        for d in docs
    )
    assert [s.label for s in docs[0].sections] == ["§ 1 Abs. 1", "§ 1 Abs. 2"]
    assert docs[0].sections[1].absatz == 2 and docs[0].sections[0].text.startswith("(1) Dieses Gesetz gilt")
    assert "\n1. eigene Angelegenheiten" in docs[2].text and "\n2a. Angelegenheiten" in docs[2].text
    assert docs[0].url == "https://www.gesetze-im-internet.de/beurkg/__1.html"
    for d in docs:
        parse_citation(d.citation_id)


def test_gii_toc_and_zip():
    import io
    import zipfile

    from lexhybrid.data.corpus.collectors.gii import parse_toc, read_zip_xml

    toc = parse_toc((FIXTURES / "gii" / "toc_excerpt.xml").read_bytes())
    assert toc == [
        ("beurkg", "https://www.gesetze-im-internet.de/beurkg/xml.zip"),
        ("bgb", "https://www.gesetze-im-internet.de/bgb/xml.zip"),
        ("gg", "https://www.gesetze-im-internet.de/gg/xml.zip"),
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("BJNR000010000.xml", b"<dokumente/>")
    assert read_zip_xml(buf.getvalue()) == b"<dokumente/>"


@pytest.mark.network
def test_gii_live_smoke():
    from lexhybrid.data.corpus.collectors.gii import GIICollector

    docs = list(GIICollector().iter_documents(limit=2))
    assert len(docs) == 2 and all(d.citation_id.startswith("BGB §") and d.valid_from for d in docs)


# -- collector: Rechtsprechung im Internet (P3-G) -----------------------------------------------


def test_rii_parser_on_fixtures():
    """P3-G: one document per decision; Randnummern become `Rn. N` sections; Tenor, Leitsatz,
    Tatbestand and Gründe keep their order; court + docket is the citation; the date is valid_from."""
    from lexhybrid.data.corpus.collectors.rii import parse_rii_xml

    docs = {
        f.name: parse_rii_xml(f.read_bytes(), retrieved_at="2026-09-27T16:00:00+00:00")
        for f in sorted((FIXTURES / "rii").glob("jb-*.xml"))
    }
    bgh, bag_u, bag_b = (
        docs["jb-JURE100055033.xml"],
        docs["jb-KARE600065578.xml"],
        docs["jb-KARE600065615.xml"],
    )
    assert [d.citation_id for d in (bgh, bag_u, bag_b)] == [
        "BGH IX ZB 72/08",
        "BAG 3 AZR 158/22",
        "BAG 2 AZN 22/23",
    ]
    assert [d.valid_from for d in (bgh, bag_u, bag_b)] == ["2010-01-14", "2023-01-17", "2023-02-28"]
    assert all(
        (d.licence, d.jurisdiction, d.doc_type) == ("DE-UrhG-5", "DE", "decision") for d in docs.values()
    )
    assert bag_u.sections[0].label == "Leitsatz" and "E-Mail-to-Fax-Verfahren" in bag_u.sections[0].text
    rn = [s.randnummer for s in bag_u.sections if s.randnummer]
    assert rn == sorted(rn) and rn[0] == 1 and len(rn) == 26
    assert bag_b.text.startswith("Tenor\n1. Die Beschwerde") and "\nGründe\n1 Die auf sämtliche" in bag_b.text
    for d in docs.values():
        parse_citation(d.citation_id)


def test_rii_toc_excerpt():
    from lexhybrid.data.corpus.collectors.rii import parse_toc

    items = parse_toc((FIXTURES / "rii" / "toc_excerpt.xml").read_bytes())
    assert [(i["court"], i["docket"], i["date"]) for i in items][2] == (
        "BGH 9. Zivilsenat",
        "IX ZB 72/08",
        "20100114",
    )
    assert all(i["link"].startswith("https://www.rechtsprechung-im-internet.de/") for i in items)


@pytest.mark.network
def test_rii_live_smoke():
    from lexhybrid.data.corpus.collectors.rii import RIICollector

    docs = list(RIICollector().iter_documents(limit=2))
    assert len(docs) == 2 and all(d.valid_from and d.sections for d in docs)


# -- collector: Open Legal Data (P3-H) ----------------------------------------------------------


def test_oldp_parser_on_fixtures():
    """P3-H: one document per case; court name -> citation abbreviation; HTML paragraphs as
    sections under their part headings; ODbL-1.0; the decision date as valid_from."""
    import json

    from lexhybrid.data.corpus.collectors.oldp import parse_oldp_case

    docs = [
        parse_oldp_case(json.loads(f.read_text()), retrieved_at="2026-09-27T16:00:00+00:00")
        for f in sorted((FIXTURES / "oldp").glob("case_*.json"))
    ]
    assert [d.citation_id for d in docs] == [
        "OLG Nürnberg 8 W 1488/26 Ver",
        "LG Stuttgart 7 O 102/26",
        "VG Bremen 2 K 1343/24",
    ]
    assert [d.valid_from for d in docs] == ["2026-08-25", "2026-08-24", "2026-08-26"]
    assert all(
        (d.licence, d.commercial_safe, d.doc_type, d.jurisdiction) == ("ODbL-1.0", True, "decision", "DE")
        for d in docs
    )
    assert {s.label for s in docs[1].sections} == {"Leitsatz", "Tenor", "Gründe"}
    assert all(s.text and "<" not in s.text for d in docs for s in d.sections)
    for d in docs:
        assert parse_citation(d.citation_id)["kind"] == "decision"


@pytest.mark.parametrize(
    "name,abbr",
    [
        ("Bundesgerichtshof", "BGH"),
        ("Oberlandesgericht Nürnberg", "OLG Nürnberg"),
        ("Brandenburgisches Oberlandesgericht", "OLG Brandenburg"),
        ("Hanseatisches Oberlandesgericht in Bremen", "OLG Bremen"),
        ("Hessischer Verwaltungsgerichtshof", "VGH Hessen"),
        ("Bayerisches Oberstes Landesgericht", "BayObLG"),
        ("Amtsgericht Frankfurt (Oder)", "AG Frankfurt (Oder)"),
        ("Verfassungsgericht des Landes Brandenburg", "VerfG Brandenburg"),
        ("Europäischer Gerichtshof", "EuGH"),
    ],
)
def test_oldp_court_abbreviations(name, abbr):
    from lexhybrid.data.corpus.collectors.oldp import court_abbreviation

    assert court_abbreviation(name) == abbr
    parse_citation(f"{abbr} 1 A 23/45")


def test_html_blocks():
    from lexhybrid.data.corpus.collectors.htmltext import html_blocks

    html = "<html><head><title>x</title><style>p{}</style></head><body><h2>Tenor</h2><p>Die Klage<br>wird abgewiesen.</p><p>&nbsp;</p><table><tr><td>a</td><td>b</td></tr></table></body></html>"
    assert html_blocks(html) == ["Tenor", "Die Klage\nwird abgewiesen.", "a b"]


@pytest.mark.network
def test_oldp_live_smoke():
    from huggingface_hub import get_token

    from lexhybrid.data.corpus.collectors.oldp import OLDPCollector

    if get_token() is None:
        pytest.skip(
            "openlegaldata/court-decisions-germany is gated: needs an HF token that accepted its terms"
        )
    docs = list(OLDPCollector().iter_documents(limit=2))
    assert len(docs) == 2 and all(d.valid_from and d.text for d in docs)


# -- collector: EUR-Lex (P3-I) ------------------------------------------------------------------


def test_eurlex_parser_on_fixture():
    """P3-I: one document per article of the German XHTML; paragraphs as `Art. N Abs. K` sections;
    lettered points on one line; short name as the citation code; the OJ date as valid_from."""
    from lexhybrid.data.corpus.collectors.eurlex import parse_eurlex_xhtml

    xhtml = (FIXTURES / "eurlex" / "32016R0679_excerpt.xhtml").read_bytes()
    docs = parse_eurlex_xhtml(xhtml, "32016R0679", retrieved_at="2026-09-27T16:00:00+00:00")
    assert [d.citation_id for d in docs] == ["DSGVO Art. 1", "DSGVO Art. 4", "DSGVO Art. 6"]
    assert [d.id for d in docs] == [
        "eurlex:32016R0679:art1",
        "eurlex:32016R0679:art4",
        "eurlex:32016R0679:art6",
    ]
    assert all(
        (d.jurisdiction, d.licence, d.valid_from, d.doc_type)
        == ("EU", "CC-BY-4.0", "2016-05-04", "regulation")
        for d in docs
    )
    art6 = docs[2]
    assert [s.label for s in art6.sections] == [
        "Art. 6 Abs. 1",
        "Art. 6 Abs. 2",
        "Art. 6 Abs. 3",
        "Art. 6 Abs. 4",
    ]
    assert "\na) Die betroffene Person hat ihre Einwilligung" in art6.sections[0].text
    assert docs[1].sections[0].label == "Art. 4" and "„personenbezogene Daten“" in docs[1].text
    for d in docs:
        parse_citation(d.citation_id)


@pytest.mark.parametrize(
    "celex,code",
    [
        ("32016R0679", "DSGVO"),
        ("32011L0083", "RL (EU) 2011/83"),
        ("32019L0770", "RL (EU) 2019/770"),
        ("32015R2120", "VO (EU) 2015/2120"),
    ],
)
def test_eurlex_act_codes(celex, code):
    from lexhybrid.data.corpus.collectors.eurlex import act_code

    assert act_code(celex) == code
    parse_citation(f"{code} Art. 3")


def test_eurlex_collector_skips_an_act_without_german_xhtml(monkeypatch, capsys):
    """An act CELLAR does not serve as German XHTML (HTTP 404, e.g. 31993L0013) is skipped with a
    message; the collection goes on with the next act."""
    from lexhybrid.data.corpus.collectors import eurlex
    from lexhybrid.data.corpus.collectors.base import FetchError

    xhtml = (FIXTURES / "eurlex" / "32016R0679_excerpt.xhtml").read_bytes()

    class _Response:
        content = xhtml

    def fake_get(url, **kwargs):
        if "31993L0013" in url:
            raise FetchError(f"GET {url} -> HTTP 404")
        return _Response()

    monkeypatch.setattr(eurlex, "http_get", fake_get)
    docs = list(eurlex.EURLexCollector(celex=("31993L0013", "32016R0679")).iter_documents(limit=2))
    assert [d.citation_id for d in docs] == ["DSGVO Art. 1", "DSGVO Art. 4"]
    assert "skipping 31993L0013" in capsys.readouterr().err


@pytest.mark.network
def test_eurlex_live_smoke():
    from lexhybrid.data.corpus.collectors.eurlex import EURLexCollector

    docs = list(EURLexCollector(celex=("32011L0083",)).iter_documents(limit=2))
    assert [d.citation_id for d in docs] == ["RL (EU) 2011/83 Art. 1", "RL (EU) 2011/83 Art. 2"]


# -- collector: RIS, Austria (P3-J) -------------------------------------------------------------


def _ris_fixture(name):
    import json

    return json.loads((FIXTURES / "ris" / f"{name}.json").read_text()), (
        FIXTURES / "ris" / f"{name}.xml"
    ).read_bytes()


def test_ris_norm_versions_on_fixtures():
    """P3-J: one document per norm VERSION with its validity window (the changelog's input);
    Absätze as sections even when the first starts with the paragraph symbol."""
    from lexhybrid.data.corpus.collectors.ris import parse_ris_norm

    old, new, multi = (
        parse_ris_norm(*_ris_fixture(n), retrieved_at="2026-09-27T16:00:00+00:00")
        for n in ("NOR12018413", "NOR40172917", "NOR12019037")
    )
    assert (old.citation_id, old.valid_from, old.valid_to) == ("ABGB §688", "1812-01-01", "2016-12-31")
    assert (new.citation_id, new.valid_from, new.valid_to) == ("ABGB §688", "2017-01-01", None)
    assert old.id != new.id and old.text != new.text
    assert [s.label for s in multi.sections] == ["§ 1295 Abs. 1", "§ 1295 Abs. 2"] and multi.sections[
        1
    ].absatz == 2
    assert all(
        (d.jurisdiction, d.licence, d.doc_type) == ("AT", "CC-BY-4.0", "statute") for d in (old, new, multi)
    )
    for d in (old, new, multi):
        parse_citation(d.citation_id)


def test_ris_norm_version_that_never_applied_is_skipped():
    """A version whose Ausserkrafttretensdatum precedes its Inkrafttretensdatum was replaced before it
    came into force: it never applied and is not collected (found live: NOR40246782)."""
    import copy

    from lexhybrid.data.corpus.collectors.ris import parse_ris_norm

    ref, xml = _ris_fixture("NOR40172917")
    never = copy.deepcopy(ref)
    brk = never["Data"]["Metadaten"]["Bundesrecht"]["BrKons"]
    brk["Inkrafttretensdatum"], brk["Ausserkrafttretensdatum"] = "2017-01-01", "2016-12-31"
    assert parse_ris_norm(never, xml) is None and parse_ris_norm(ref, xml) is not None


def test_ris_decision_on_fixture():
    from lexhybrid.data.corpus.collectors.ris import docket, parse_ris_decision

    doc = parse_ris_decision(
        *_ris_fixture("JJT_20260917_OGH0002_0070OB00143_26V0000_000"),
        retrieved_at="2026-09-27T16:00:00+00:00",
    )
    assert (doc.citation_id, doc.valid_from, doc.doc_type) == ("OGH 7 Ob 143/26v", "2026-09-17", "decision")
    assert [s.label for s in doc.sections[:2]] == ["Kopf", "Kopf"] and "Rechtliche Beurteilung" in {
        s.label for s in doc.sections
    }
    assert doc.text.startswith("Kopf\n") and "\nSpruch\n" in doc.text
    assert parse_citation(doc.citation_id) == {"kind": "decision", "court": "OGH", "docket": "7 Ob 143/26v"}
    assert docket("1Ob358/60; 4Ob639/71") == "1 Ob 358/60"


def test_ris_stops_at_the_last_page_the_search_reports(monkeypatch):
    """Job 2589358_4: RIS answers a page past the end with HTTP 500 ("Die Seitennummer ist höher als
    die Anzahl der verfügbaren Seiten"), not an empty list, so the whole RIS task died after ABGB's
    last page. The collector stops at the page count in ``Hits`` and never asks for the next page."""
    from lexhybrid.data.corpus.collectors import ris
    from lexhybrid.data.corpus.collectors.base import FetchError

    norm_ref, norm_xml = _ris_fixture("NOR40172917")
    dec_ref, dec_xml = _ris_fixture("JJT_20260917_OGH0002_0070OB00143_26V0000_000")
    xml = {ris._xml_url(norm_ref): norm_xml, ris._xml_url(dec_ref): dec_xml}
    asked = []

    class _Response:
        def __init__(self, body=None, content=b""):
            self.body, self.content = body, content

        def json(self):
            return self.body

    def fake_get(url, params=None, **kwargs):
        if url in xml:
            return _Response(content=xml[url])
        kind, page = url.rsplit("/", 1)[-1], int(params["Seitennummer"])
        asked.append((kind, page))
        if page > 2:  # 51 hits at 50 per page: 2 pages
            raise FetchError(f"GET {url} failed after 5 attempts (HTTP 500)")
        hits = {"@pageNumber": str(page), "@pageSize": "50", "#text": "51"}
        ref = norm_ref if kind == "Bundesrecht" else dec_ref
        return _Response(
            {"OgdSearchResult": {"OgdDocumentResults": {"Hits": hits, "OgdDocumentReference": ref}}}
        )

    monkeypatch.setattr(ris, "http_get", fake_get)
    docs = list(ris.RISCollector(codes=("ABGB",)).iter_documents())
    assert asked == [("Bundesrecht", 1), ("Bundesrecht", 2), ("Judikatur", 1), ("Judikatur", 2)]
    assert [d.doc_type for d in docs] == ["statute", "statute", "decision", "decision"]


@pytest.mark.network
def test_ris_live_smoke():
    from lexhybrid.data.corpus.collectors.ris import RISCollector

    docs = list(RISCollector(codes=("KSchG",)).iter_documents(limit=2))
    assert [d.doc_type for d in docs] == ["statute", "decision"] and all(d.valid_from for d in docs)


# -- collector: Fedlex, Switzerland (P3-K) ------------------------------------------------------


def _fedlex(name):
    from lexhybrid.data.corpus.collectors.fedlex import parse_fedlex_xml

    xml = (FIXTURES / "fedlex" / f"{name}.xml").read_bytes()
    return {d.citation_id: d for d in parse_fedlex_xml(xml, retrieved_at="2026-09-27T16:00:00+00:00")}


def test_fedlex_parser_on_fixtures():
    """P3-K: one document per article of the Akoma Ntoso XML; Absätze as sections (``Abs. 2bis``
    too, and the superscript-numbered ones of the final title); list items one per line; footnotes
    never in the text; repealed articles and the provisions of single amendments skipped."""
    or_, zgb = _fedlex("or_excerpt"), _fedlex("zgb_excerpt")
    assert list(or_) == ["OR Art. 1", "OR Art. 6a", "OR Art. 24", "OR Art. 40a", "OR Art. 97"]
    assert list(zgb) == [
        "ZGB Art. 1",
        "ZGB Art. 8",
        "ZGB Art. 334bis",
        "ZGB SchlT Art. 1",
        "ZGB SchlT Art. 59",
    ]
    art1 = or_["OR Art. 1"]
    assert art1.id == "fedlex:220:art_1"
    assert art1.url == "https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de#art_1"
    assert art1.text.splitlines() == [
        "Art. 1 Abschluss des Vertrages / Übereinstimmende Willensäusserung / Im Allgemeinen",
        "1 Zum Abschlusse eines Vertrages ist die übereinstimmende gegenseitige Willensäusserung der "
        "Parteien erforderlich.",
        "2 Sie kann eine ausdrückliche oder stillschweigende sein.",
    ]
    assert [(s.label, s.absatz) for s in or_["OR Art. 40a"].sections] == [
        ("Art. 40a Abs. 1", 1),
        ("Art. 40a Abs. 2", 2),
        ("Art. 40a Abs. 2bis", None),
        ("Art. 40a Abs. 3", 3),
    ]
    assert "folgenden Fällen ein wesentlicher:\n1. wenn der Irrende" in or_["OR Art. 24"].sections[0].text
    assert or_["OR Art. 97"].sections[1].text.endswith("der Zivilprozessordnung vom 19. Dezember 2008 (ZPO).")
    assert zgb["ZGB Art. 8"].sections[0].label == "Art. 8"  # one Absatz, no number
    schlt = zgb["ZGB SchlT Art. 1"]
    assert schlt.id == "fedlex:210:disp_u1:art_1"
    assert [s.label for s in schlt.sections] == ["Art. 1 Abs. 1", "Art. 1 Abs. 2", "Art. 1 Abs. 3"]
    assert schlt.sections[0].text.startswith("Die rechtlichen Wirkungen von Tatsachen")
    assert [s.label for s in zgb["ZGB SchlT Art. 59"].sections] == ["Art. 59 Abs. 1", "Art. 59 Abs. 3"]
    for d in (*or_.values(), *zgb.values()):
        assert "Fassung gemäss" not in d.text and "SR 281.1" not in d.text and "Eingefügt" not in d.text
        assert (d.jurisdiction, d.licence, d.doc_type, d.commercial_safe) == (
            "CH",
            "CH-URG-5",
            "statute",
            True,
        )
        parse_citation(d.citation_id)


def test_fedlex_valid_from_is_the_provisions_version():
    """P3-K / P3-T: valid_from is the latest in-force date in the footnotes of the article or of the
    headings above it, else the act's entry into force -- never after the version that prints it."""
    or_, zgb = _fedlex("or_excerpt"), _fedlex("zgb_excerpt")
    assert or_["OR Art. 1"].valid_from == "1912-01-01"  # original text: the act's entry into force
    assert or_["OR Art. 6a"].valid_from == "1991-07-01"  # "Eingefügt durch ..., in Kraft seit 1. Juli 1991"
    assert or_["OR Art. 97"].valid_from == "2011-01-01"  # Abs. 2 amended by the ZPO
    assert zgb["ZGB Art. 1"].valid_from == "2000-01-01"  # "Ausdruck gemäss ..., in Kraft seit 1. Jan. 2000"
    assert zgb["ZGB Art. 334bis"].valid_from == "1973-02-15"  # "... in Kraft seit 15. Febr. 1973"
    assert zgb["ZGB SchlT Art. 1"].valid_from == "2000-01-01"  # the final title's heading footnote
    assert all(d.valid_from <= "2026-01-01" and d.valid_to is None for d in or_.values())


def test_fedlex_versions_and_codes():
    """P3-K: the version collected is the latest one applicable on the day; an act without a
    usable German abbreviation is cited by its SR number."""
    import json

    from lexhybrid.data.corpus.collectors.fedlex import act_code, latest_versions, versions_query

    bindings = json.loads((FIXTURES / "fedlex" / "versions_220.json").read_text())["results"]["bindings"]
    date, file = latest_versions(bindings)["220"]
    assert date == "2026-01-01" and file.endswith(
        "/20260101/de/xml/fedlex-data-admin-ch-eli-cc-27-317_321_377-20260101-de-xml-12.xml"
    )
    query = versions_query("2026-09-27", "220")
    assert 'FILTER(str(?sr) = "220")' in query and '"2026-09-27"^^xsd:date' in query
    assert (act_code("OR", "220"), act_code("GebV BAR", "172.041.15"), act_code("", "0.101")) == (
        "OR",
        "SR 172.041.15",
        "SR 0.101",
    )
    for code in ("OR", "SR 172.041.15", "SR 0.101"):
        parse_citation(f"{code} Art. 3")


@pytest.mark.network
def test_fedlex_live_smoke():
    from lexhybrid.data.corpus.collectors.fedlex import FedlexCollector

    docs = list(FedlexCollector(sr=("151.1",), all_acts=False).iter_documents(limit=2))
    assert [d.citation_id for d in docs] == ["GlG Art. 1", "GlG Art. 2"] and all(d.valid_from for d in docs)


# -- collector: BGer, Swiss Federal Supreme Court (P3-L) ---------------------------------------


def _bger(name):
    import json

    from lexhybrid.data.corpus.collectors.bger import parse_bger

    meta = json.loads((FIXTURES / "bger" / f"{name}.json").read_text())
    html = (FIXTURES / "bger" / f"{name}.html").read_text(encoding="utf-8")
    return parse_bger(html, meta, retrieved_at="2026-09-27T16:00:00+00:00")


def test_bger_parser_on_fixtures():
    """P3-L: one document per decision; the heading, facts, considerations (``E. 4.4.2``), operative
    part and closing as sections; a numbered quotation inside a consideration stays inside it."""
    doc = _bger("CH_BGer_004_4A-102-2026_2026-08-17")
    assert (doc.id, doc.citation_id, doc.valid_from) == (
        "bger:CH_BGer_004_4A-102-2026_2026-08-17",
        "BGer 4A_102/2026",
        "2026-08-17",
    )
    assert (doc.jurisdiction, doc.doc_type, doc.licence, doc.commercial_safe) == (
        "CH",
        "decision",
        "CH-URG-5",
        True,
    )
    assert doc.url == "https://entscheidsuche.ch/docs/CH_BGer/CH_BGer_004_4A-102-2026_2026-08-17.html"
    labels = [s.label for s in doc.sections if s.label != "Kopf"]
    assert labels == [
        "Sachverhalt A", "Sachverhalt B", "Sachverhalt C",
        "E. 1", "E. 2.1", "E. 2.2", "E. 3", "E. 3.1", "E. 3.2", "E. 3.3", "E. 3.4",
        "E. 4", "E. 4.1", "E. 4.2", "E. 4.3", "E. 4.4", "E. 4.4.1", "E. 4.4.2", "E. 4.4.3", "E. 4.4.4", "E. 5",
        "Dispositiv Ziff. 1", "Dispositiv Ziff. 2", "Dispositiv Ziff. 3", "Dispositiv Ziff. 4", "Schluss",
    ]  # fmt: skip
    by = {s.label: s.text for s in doc.sections}
    assert "\n5. Honorar\n" in by["E. 4.2"]  # the contract's clause 5, not consideration 5
    assert by["E. 4.4.2"].startswith("Aus dem Umstand") and "\n" not in by["E. 4.4.2"]  # one paragraph
    assert by["Dispositiv Ziff. 1"] == "Die Beschwerde wird abgewiesen, soweit darauf eingetreten wird."
    assert doc.text.startswith("4A_102/2026\nUrteil vom 17. August 2026\n")  # court names dropped
    assert "\nErwägungen:\n1.\nDie Eintretensvoraussetzungen" in doc.text
    assert parse_citation(f"{doc.citation_id} E. 4.4.2")["erwaegung"] == "4.4.2"


def test_bger_parser_on_the_unmarked_older_layout():
    """Before ~2007 the numbers are not bold: a number opens a consideration only when it can
    follow the previous one, and the Erwägungen heading is the court's sentence."""
    from lexhybrid.data.corpus.collectors.bger import follows

    doc = _bger("CH_BGer_008_I-311-2000_2003-12-31")
    labels = [s.label for s in doc.sections if s.label != "Kopf"]
    assert labels == [
        "Sachverhalt A", "Sachverhalt B", "Sachverhalt C",
        "E. 1", "E. 2", "E. 2.1", "E. 2.2", "E. 3.1", "E. 3.2", "E. 4", "E. 5", "E. 5.1", "E. 5.2",
        "Dispositiv Ziff. 1", "Dispositiv Ziff. 2", "Dispositiv Ziff. 3", "Schluss",
    ]  # fmt: skip
    assert "\nDas Eidg. Versicherungsgericht zieht in Erwägung:\n" in doc.text
    assert doc.citation_id == "BGer I_311/2000" and doc.valid_from == "2003-12-31"
    short = _bger("CH_BGer_001_1C-464-2026_2026-09-16")  # "Demnach erkennt der Präsident:"
    assert [s.label for s in short.sections if s.label.startswith(("E.", "Dispositiv"))] == [
        "E. 1", "E. 2", "E. 3.1", "E. 3.2", "E. 3.3", "E. 4",
        "Dispositiv Ziff. 1", "Dispositiv Ziff. 2", "Dispositiv Ziff. 3",
    ]  # fmt: skip
    assert follows(None, (1,)) and follows((2,), (2, 1)) and follows((2, 1), (2, 2))
    assert follows((2, 1, 3), (2, 2)) and follows((2, 4), (3,))
    assert not follows(None, (2,)) and not follows((4, 2), (4, 4)) and not follows((1,), (2003,))


def test_bger_parser_on_an_order():
    """An order (Verfügung): unnumbered "in Erwägung, dass ..." considerations, then "verfügt die
    Präsidentin:" -- without "Demnach" -- opens the operative part, and the closing is its own."""
    doc = _bger("CH_BGer_008_8C-466-2026_2026-09-10")
    assert [s.label for s in doc.sections if s.label != "Kopf"] == [
        "Erwägungen", "Dispositiv Ziff. 1", "Dispositiv Ziff. 2", "Dispositiv Ziff. 3", "Schluss",
    ]  # fmt: skip
    by = {s.label: s.text for s in doc.sections}
    assert by["Erwägungen"].startswith("dass die Beschwerde") and by["Erwägungen"].count("\ndass ") == 1
    assert by["Dispositiv Ziff. 1"] == "Das Verfahren wird infolge Rückzugs der Beschwerde abgeschrieben."
    assert by["Schluss"].startswith("Luzern, 10. September 2026")


@pytest.mark.parametrize(
    "text,reasons,operative",
    [
        ("Erwägungen:", True, False),
        ("Erwägung:", True, False),
        ("Das Eidg. Versicherungsgericht zieht in Erwägung:", True, False),
        ("Der Präsident hat in Erwägung,", True, False),
        ("In Erwägung,", True, False),
        ("Das Obergericht zog in Erwägung, dass die Klägerin nicht legitimiert sei.", False, False),
        ("Demnach erkennt das Bundesgericht:", False, True),
        ("Demnach erkennt das Bundesgericht im", False, True),  # the heading runs on to the next line
        ("Demnach erkennt das Präsidium:", False, True),
        ("Demnach erkennt das präsidierende Mitglied:", False, True),
        ("verfügt die Präsidentin im Verfahren nach Art. 32 Abs. 2 BGG:", False, True),
        ("im Verfahren nach Art. 108 Abs. 1 lit. b BGG erkannt:", False, True),
        ("erkannt :", False, True),
        ("Die Vorinstanz erkennt zutreffend Folgendes:", False, False),
        ("Das Obergericht hat am 3. Mai 2020 wie folgt erkannt:", False, False),
        ("Demnach ist die Beschwerde abzuweisen.", False, False),
    ],
)
def test_bger_heading_forms(text, reasons, operative):
    """The heading forms met in 230 decisions from 2000 to 2026, and look-alikes inside reasons."""
    from lexhybrid.data.corpus.collectors.bger import OPERATIVE, REASONS

    assert (bool(REASONS.match(text)), bool(OPERATIVE.match(text))) == (reasons, operative)


def test_bger_search_body_blocklist_and_post_retries():
    import json

    import requests

    from lexhybrid.data.corpus.collectors.base import RateLimiter, http_post
    from lexhybrid.data.corpus.collectors.bger import blocked_ids, search_body

    body = search_body(50)
    assert body["size"] == 50 and "search_after" not in body
    assert {"term": {"hierarchy": "CH_BGer"}} in body["query"]["bool"]["filter"]
    assert {"term": {"attachment.language": "de"}} in body["query"]["bool"]["filter"]
    assert body["sort"] == [{"date": "desc"}, {"id": "desc"}]
    assert search_body(50, after=[1789516800000, "x"])["search_after"] == [1789516800000, "x"]
    blocklist = json.loads((FIXTURES / "bger" / "Blockliste.json").read_text())
    assert blocked_ids({"CH_BGer": ["CH_BGer_005_5A-1-2020_2020-01-02"], "ZH_Gerichte": ["ZH_x"]}) == {
        "CH_BGer_005_5A-1-2020_2020-01-02"
    }
    assert blocked_ids(blocklist) == set(blocklist.get("CH_BGer", []))

    class _PostSession(_FakeSession):
        def post(self, url, json=None, headers=None, timeout=None):
            self.bodies = getattr(self, "bodies", []) + [json]
            return self.get(url)

    session, slept = _PostSession([502, requests.ConnectionError("reset"), 200]), []
    out = http_post(
        "https://example.org/s", json={"q": 1}, session=session, limiter=RateLimiter(0), sleep=slept.append
    )
    assert out.status_code == 200 and session.bodies == [{"q": 1}] * 3 and slept == [1.0, 2.0]


@pytest.mark.network
def test_bger_live_smoke():
    from lexhybrid.data.corpus.collectors.bger import BGerCollector

    docs = list(BGerCollector(page_size=2).iter_documents(limit=2))
    assert len(docs) == 2 and all(d.citation_id.startswith("BGer ") and d.valid_from for d in docs)
    assert all(any(s.label.startswith("E. ") for s in d.sections) for d in docs)


# -- collector: DIP, Bundestag printed papers (P3-M) -------------------------------------------


def _dip(name):
    import json

    from lexhybrid.data.corpus.collectors.dip import parse_dip_drucksache

    record = json.loads((FIXTURES / "dip" / f"{name}.json").read_text())
    return parse_dip_drucksache(record, retrieved_at="2026-09-27T16:00:00+00:00")


def test_dip_parser_on_fixtures():
    """P3-M: one document per printed paper, cited "BT-Drs. 21/8109" as DIP's terms prescribe;
    sections are the paragraphs under the bill's headings; the watermark, the printer's
    boilerplate and the PDF's line breaks and hyphenation are gone."""
    bill, report, bundesrat = _dip("bt_21_8109"), _dip("bt_21_8163"), _dip("br_506_26")
    assert [d.citation_id for d in (bill, report, bundesrat)] == [
        "BT-Drs. 21/8109",
        "BT-Drs. 21/8163",
        "BR-Drs. 506/26",
    ]
    assert [d.valid_from for d in (bill, report, bundesrat)] == ["2026-09-22", "2026-09-23", "2026-09-04"]
    for d in (bill, report, bundesrat):
        assert (d.jurisdiction, d.doc_type, d.licence, d.commercial_safe) == (
            "DE",
            "parliament",
            "DE-UrhG-5",
            True,
        )
        assert "orabfassung" not in d.text and "Bundesanzeiger Verlag" not in d.text
        assert parse_citation(d.citation_id)["kind"] == "parliament"
    labels = {s.label for s in bill.sections}
    assert {"Kopf", "A. Problem", "B. Lösung", "Artikel 1", "II. Wesentlicher Inhalt des Entwurfs"} <= labels
    assert "Zu Artikel 1 / Zu Nummer 1 (§ 261 Absatz 1 StGB-E)" in labels  # a Zu-Nummer carries its article
    first = next(s.text for s in bill.sections if s.label == "A. Problem")
    assert first.startswith(
        "Finanzkriminalität ist ein direkter Angriff auf unseren Rechtsstaat und untergräbt das"
    )
    assert "\n" not in first  # the PDF's line breaks are rejoined inside a paragraph
    assert "erlangte Steuererstattungen bzw. Steuervergütungen" in bill.text  # hyphen across the watermark
    assert "der Sicherung von Vermögenabschöpfung" in bill.text  # a paragraph split by a page's footnote
    assert "Waffen- und Menschenhandel" in bill.text  # a real hyphen stays
    assert bill.url == "https://dserver.bundestag.de/btd/21/081/2108109.pdf" and bill.id == "dip:290950"
    assert "IV. Beratungsverlauf und Beratungsergebnisse im federführenden Ausschuss" in {
        s.label for s in report.sections
    }
    assert bundesrat.text.startswith("Bundesrat Drucksache 506/26\n")


def test_dip_paragraph_rules():
    from lexhybrid.data.corpus.collectors.dip import is_heading, paragraphs

    text = (
        "A. Problem\n"
        "Die Regelung ist unklar, weil die Erstattung und die ersparte Zahlung als Steuererstattung bzw.\n"
        "Steuervergütung unterschiedlich behandelt werden, obwohl sie wirtschaftlich gleich sind, und\n"
        "das führt zu Lücken.\n"
        "Zu Artikel 1 (Änderung des Strafgesetzbuches – StGB)\n"
        "Der Absatz wird neu gefasst, um die Verfol-\n"
        "gung zu stärken.\n"
    )
    assert paragraphs(text) == [
        (True, "A. Problem"),
        (
            False,
            "Die Regelung ist unklar, weil die Erstattung und die ersparte Zahlung als Steuererstattung bzw. "
            "Steuervergütung unterschiedlich behandelt werden, obwohl sie wirtschaftlich gleich sind, und das "
            "führt zu Lücken.",
        ),
        (True, "Zu Artikel 1 (Änderung des Strafgesetzbuches – StGB)"),
        (False, "Der Absatz wird neu gefasst, um die Verfolgung zu stärken."),
    ]
    assert is_heading("B. Lösung") and is_heading("Artikel 3") and is_heading("Begründung")
    assert not is_heading("§ 261 StGB einbezogen werden, soweit sie auf einer Steuerhinterziehung beruhen.")


@pytest.mark.network
def test_dip_live_smoke():
    from lexhybrid.data.corpus.collectors.dip import DIPCollector

    docs = list(DIPCollector(types=("Gesetzentwurf",)).iter_documents(limit=2))
    assert len(docs) == 2 and all(
        d.citation_id.startswith(("BT-Drs. ", "BR-Drs. ")) and d.valid_from for d in docs
    )


# -- collector: FineWeb-2 German (P3-N) --------------------------------------------------------


def test_fineweb2_parser_on_fixture_rows():
    """P3-N: one document per page; general German text, commercial-safe under ODC-By; the crawl
    date as valid_from; paragraphs as sections; no citation id."""
    import json

    from lexhybrid.data.corpus.collectors.fineweb2_de import parse_fineweb_row

    rows = json.loads((FIXTURES / "fineweb2_de" / "rows.json").read_text())
    docs = [parse_fineweb_row(r, retrieved_at="2026-09-27T16:00:00+00:00") for r in rows]
    assert [d.id for d in docs] == [
        "fineweb2_de:890004fd-4d76-4f82-83fb-0f2c840ec970",
        "fineweb2_de:7878752c-a6b2-43ab-861b-92d094cfa9f1",
        "fineweb2_de:9a1ad857-5f62-4fd0-95bf-55d90122bbdc",
    ]
    for d, r in zip(docs, rows, strict=True):
        assert (d.jurisdiction, d.doc_type, d.licence, d.commercial_safe, d.research_only) == (
            "DE",
            "general",
            "ODC-By-1.0",
            True,
            False,
        )
        assert d.valid_from == "2013-05-19" and d.url == r["url"] and d.citation_id == ""
        assert [s.text for s in d.sections] == d.text.split("\n") and all(
            s.label == "Text" for s in d.sections
        )
    assert parse_fineweb_row({"id": "<urn:uuid:x>", "text": " \n "}) is None


@pytest.mark.network
def test_fineweb2_live_smoke():
    from lexhybrid.data.corpus.collectors.fineweb2_de import FineWeb2DECollector

    docs = list(FineWeb2DECollector().iter_documents(limit=2))
    assert len(docs) == 2 and all(d.doc_type == "general" and d.valid_from for d in docs)


# -- collector: Multi Legal Pile, research-exception arm (P3-O) ---------------------------------


def test_multilegalpile_rows_are_research_only():
    """P3-O / decision 2: every Multi Legal Pile document is research_only and never commercial-safe
    (CC BY-NC-SA 4.0); the research arm admits it, the commercial-safe arm refuses it."""
    import json

    from lexhybrid.data.corpus.build import admits
    from lexhybrid.data.corpus.collectors.multilegalpile import parse_mlp_row

    fixture = json.loads((FIXTURES / "multilegalpile" / "rows.json").read_text())
    docs = {
        subset: parse_mlp_row(v["row"], subset, v["index"], retrieved_at="2026-09-27T16:00:00+00:00")
        for subset, v in fixture.items()
    }
    assert {s: (d.jurisdiction, d.doc_type) for s, d in docs.items()} == {
        "de_legislation_switzerland_lexfind": ("CH", "statute"),
        "de_legislation_germany_openlegaldata": ("DE", "statute"),
        "de_caselaw_switzerland_entscheidsuche": ("CH", "decision"),
        "de_caselaw_germany_openlegaldata": ("DE", "decision"),
    }
    for subset, d in docs.items():
        assert d.id == f"multilegalpile:{subset}:{fixture[subset]['index']}"
        assert (d.licence, d.research_only, d.commercial_safe) == ("CC-BY-NC-SA-4.0", True, False)
        assert admits("research", d) and not admits("commercial_safe", d)
        assert "\xa0" not in d.text and "\n\n" not in d.text and d.valid_from is None
    assert docs["de_legislation_switzerland_lexfind"].text.startswith("740.110\nDekret\n")
    assert (
        parse_mlp_row({"type": "contracts", "jurisdiction": "Germany", "text": "x"}, "de_contracts_x", 0)
        is None
    )


@pytest.mark.network
def test_multilegalpile_live_smoke():
    from lexhybrid.data.corpus.collectors.multilegalpile import MultiLegalPileCollector

    docs = list(
        MultiLegalPileCollector(subsets=(("de", "legislation", "germany_openlegaldata"),)).iter_documents(
            limit=2
        )
    )
    assert [d.id for d in docs] == [
        "multilegalpile:de_legislation_germany_openlegaldata:0",
        "multilegalpile:de_legislation_germany_openlegaldata:1",
    ] and all(d.research_only for d in docs)


# -- tokenizer (P3-C) --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qwen3():
    """Qwen3's tokenizer from the local cache (tests never reach the Hub unless marked `network`)."""
    from lexhybrid.data.tokenizer import load_tokenizer

    try:
        return load_tokenizer()
    except OSError as e:
        pytest.skip(
            f"Qwen3 tokenizer not cached ({type(e).__name__}); fetch it once with "
            "`.venv/bin/python -m lexhybrid.data.tokenizer --download` (the weekly network job downloads it)"
        )


def test_tokenizer_special_list_is_decision_7():
    from lexhybrid.data import tokenizer as T

    assert len(T.SPECIALS) == 87 == len(set(T.SPECIALS))
    assert T.SPECIALS[:3] == ("<|q|>", "<|cite|>", "<|c1|>") and T.SPECIALS[-1] == "<|answer|>"
    assert T.BASE_LEN + len(T.SPECIALS) <= T.VOCAB_SIZE == 151936 and T.EOS_ID == 151643


def _check_qwen3(tok):
    from lexhybrid.data.tokenizer import BASE_LEN, EOS_ID, SPECIALS, VOCAB_SIZE, encode_document, pointer_ids

    assert len(tok) == BASE_LEN + len(SPECIALS) == 151756 <= VOCAB_SIZE
    assert tok.eos_token_id == EOS_ID == tok.convert_tokens_to_ids("<|endoftext|>")
    ids = pointer_ids(tok)
    assert ids["q"] == 151669 and ids["cite"] == 151670 and ids["answer"] == 151755
    assert ids["passages"] == list(range(151671, 151687)) and ids["sentences"] == list(range(151687, 151751))
    for special in SPECIALS:  # each special is one token when the text means it
        assert tok(special, add_special_tokens=False)["input_ids"] == [tok.convert_tokens_to_ids(special)]
    text = "Nach § 573 Abs. 2 BGB <|endoftext|> gilt <|cite|> nicht."
    doc_ids = encode_document(tok, text)
    assert EOS_ID not in doc_ids and not set(doc_ids) & set(range(BASE_LEN, VOCAB_SIZE))
    assert tok.decode(doc_ids) == text


def test_tokenizer_specials_fit_vocab(qwen3):
    """P3-C: Qwen3's tokenizer plus decision 7's 87 specials fits the student's 151,936 rows; EOS is
    Qwen3's own 151,643; specials are single tokens but never come out of raw document text."""
    _check_qwen3(qwen3)


@pytest.mark.network
def test_tokenizer_download_and_specials(tmp_path, monkeypatch):
    """The weekly network job: a fresh download at the pinned revision passes the same checks."""
    from lexhybrid.data import tokenizer as T

    monkeypatch.setattr(T, "cache_dir", lambda: str(tmp_path))
    _check_qwen3(T.load_tokenizer())


# -- evaluation sets (P3-P, P3-Q) --------------------------------------------------------------

EVALSETS = FIXTURES.parent / "evalsets"


def test_gerlerb_parsers_on_fixture(tmp_path):
    """P3-P: GerLeRB's TSV topics and qrels and its TREC corpus; passage ids map onto GII's corpus;
    the set is evaluation-only and says why (GPT-4.1 questions, R10)."""
    import json

    from lexhybrid.data.evalsets import gerlerb
    from lexhybrid.data.evalsets.base import write

    topics = gerlerb.parse_topics((EVALSETS / "gerlerb" / "topics.txt").read_text())
    qrels = gerlerb.parse_qrels((EVALSETS / "gerlerb" / "qrels.txt").read_text())
    corpus = list(gerlerb.iter_corpus((EVALSETS / "gerlerb" / "corpus.trec").read_text()))
    assert [q.id for q in topics] == ["206", "207", "209"]
    assert (
        topics[0].text
        == "Kann die Ablehnung der Annahme einer Verfassungsbeschwerde ohne Begründung erfolgen?"
    )
    assert qrels == {"206": {"93d_bverfgg": 1}, "207": {"99_betrvg": 1}, "209": {"2b_beeg": 1}}
    assert {p.id for p in corpus} == {"1_kaeaano", "2_kaeaano", "2b_beeg", "93d_bverfgg", "99_betrvg"}
    assert gerlerb.gii_id("812_bgb") == "gii:bgb:812" and gerlerb.gii_id("8b_kstg_1977") == "gii:kstg_1977:8b"
    small = gerlerb.select(topics, qrels, corpus, limit=2)
    assert [q.id for q in small.queries] == ["206", "207"] and {p.id for p in small.corpus} == {
        "93d_bverfgg",
        "99_betrvg",
    }
    assert small.info.eval_only and "GPT-4.1" in small.info.note and small.info.licence == "CC-BY-4.0"
    assert write(small, tmp_path / "gerlerb", tmp_path / "manifests") == 2
    manifest = [
        json.loads(line)
        for line in (tmp_path / "manifests" / "evalset_gerlerb.jsonl").read_text().splitlines()
    ]
    assert manifest[0]["info"]["name"] == "gerlerb" and manifest[1]["relevant"] == ["93d_bverfgg"]
    assert "text" not in manifest[1] and (tmp_path / "gerlerb" / "qrels.jsonl").read_text().count("\n") == 2


def test_gerdalir_select_on_fixture():
    from lexhybrid.data.evalsets import gerdalir

    lines = {
        n: (EVALSETS / "gerdalir" / f"{n}.jsonl").read_text().splitlines(True)
        for n in ("qrels", "queries", "corpus")
    }
    full = gerdalir.select(lines["qrels"], lines["queries"], lines["corpus"], limit=None)
    assert [q.id for q in full.queries] == ["00000", "00001", "00002"]
    assert full.qrels["00000"] == {"SOlJarKNHX": 1} and len(full.corpus) == 3
    two = gerdalir.select(lines["qrels"], lines["queries"], lines["corpus"], limit=2)
    assert list(two.qrels) == ["00000", "00001"] and [p.id for p in two.corpus] == [
        "SOlJarKNHX",
        "iQ4h87ojuP",
    ]
    assert "[REF]" in " ".join(q.text for q in full.queries) and full.info.licence == "MIT"


def test_legalquad_and_gerlayqa_parsers_on_fixtures():
    """P3-Q: LegalQuAD's SQuAD items keep the answer span and the Open Legal Data decision id;
    GerLayQA's items carry their gold BGB paragraphs as citation ids and are research-only."""
    import json

    from lexhybrid.data.evalsets import gerlayqa, legalquad

    lq = legalquad.parse_legalquad(
        json.loads((EVALSETS / "legalquad" / "LegalQuAD_excerpt.json").read_text())
    )
    assert [i.id for i in lq] == ["legalquad:118133", "legalquad:120711", "legalquad:121833"]
    assert all(i.context_id.startswith("oldp:") and i.answers and i.context for i in lq)
    assert lq[0].answers[0] in lq[0].context and not legalquad.INFO.research_only
    gl = gerlayqa.parse_gerlayqa(json.loads((EVALSETS / "gerlayqa" / "bgb_eval_excerpt.json").read_text()))
    assert [i.gold for i in gl] == [
        ["BGB §573b"],
        ["BGB §138", "BGB §343", "BGB §313"],
        ["BGB §212", "BGB §826"],
    ]
    assert all(parse_citation(c)["code"] == "BGB" for i in gl for c in i.gold)
    assert gl[0].question.startswith("Sachverhalt: wir wohnen in einer Mietwohnung") and gl[0].answers[0]
    assert (
        gerlayqa.INFO.research_only and gerlayqa.INFO.eval_only and "non-commercial" in gerlayqa.INFO.licence
    )
    assert gerlayqa.bgb_citation("§ 573b") == "BGB §573b" and gerlayqa.bgb_citation("Art. 3 GG") is None


@pytest.mark.network
def test_evalsets_live_smoke():
    from lexhybrid.data.evalsets import gerdalir, gerlayqa, gerlerb, legalquad

    assert len(gerdalir.fetch(limit=2).queries) == 2
    assert len(legalquad.fetch(limit=2).items) == 2 and len(gerlayqa.fetch(limit=2).items) == 2
    lerb = gerlerb.fetch(limit=2)
    assert len(lerb.queries) == 2 and all(any(p.id in r for p in lerb.corpus) for r in lerb.qrels.values())


# -- LER scrub (P3-R) --------------------------------------------------------------------------


def _find(text, needle, start=0):
    i = text.index(needle, start)
    return i, i + len(needle)


def test_scrub_replaces_person_names():
    """P3-R: judges, lawyers and persons become numbered typed placeholders in the text and the
    sections; a repeated name is caught where the model did not tag it; the scrub log holds
    hashes, never the names."""
    from lexhybrid.data.corpus.scrub_ler import Entity, scrub_document

    doc = _bger("CH_BGer_004_4A-102-2026_2026-08-17")
    text = doc.text
    entities = [
        Entity(*_find(text, "Hurni"), "RR", 0.99),  # the signature "Der Präsident: Hurni" is left untagged
        Entity(*_find(text, "Enrico Moretti"), "AN", 0.98),
        Entity(*_find(text, "May Canellas"), "RR", 0.97),
        Entity(*_find(text, "Zürich"), "ST", 0.9),  # a city: not one of the scrubbed tags
    ]
    scrubbed, log = scrub_document(doc, entities, key=b"k" * 32)
    for name in ("Hurni", "Enrico Moretti", "May Canellas"):
        assert name not in scrubbed.text and not any(name in s.text for s in scrubbed.sections)
    assert "Bundesrichter [RR_1], Präsident," in scrubbed.text and "Der Präsident: [RR_1]" in scrubbed.text
    assert (
        "vertreten durch Rechtsanwälte [AN_1]" in scrubbed.text and "Bundesrichterin [RR_2]," in scrubbed.text
    )
    assert "Zürich" in scrubbed.text
    assert scrubbed.sha256 != doc.sha256 and (scrubbed.id, scrubbed.citation_id) == (doc.id, doc.citation_id)
    assert [(r["label"], r["placeholder"]) for r in log] == [
        ("RR", "[RR_1]"),
        ("RR", "[RR_2]"),
        ("AN", "[AN_1]"),
    ]
    assert all(len(r["entity_hmac"]) == 64 and "Hurni" not in str(r) for r in log)
    import hashlib

    from lexhybrid.data.corpus.scrub_ler import entity_hash

    assert log[0]["entity_hmac"] == entity_hash("Hurni", b"k" * 32) != hashlib.sha256(b"Hurni").hexdigest()
    assert entity_hash("Hurni", b"k" * 32) != entity_hash("Hurni", b"j" * 32)  # keyed: no dictionary check


def test_scrub_trims_the_punctuation_of_whitespace_tokens():
    from lexhybrid.data.corpus.scrub_ler import Entity, trim

    text = "Bundesrichter Haag, Präsident, Gerichtsschreiber Baur. Herr W. und (Müller)."
    spans = {s: text.index(s) for s in ("Haag,", "Baur.", "W.", "(Müller).")}
    got = {s: trim(Entity(i, i + len(s), "RR", 1.0), text) for s, i in spans.items()}
    assert {s: text[e.start : e.end] for s, e in got.items()} == {
        "Haag,": "Haag",
        "Baur.": "Baur",
        "W.": "W.",  # an initial keeps its period
        "(Müller).": "Müller",
    }


def test_scrub_short_forms_are_replaced_only_where_tagged():
    """A tagged initial is replaced where the model tagged it and nowhere else, so a heading such as
    ``A. Problem`` or an untagged ``W.`` elsewhere stays as it is."""
    from lexhybrid.data.corpus.scrub_ler import Entity, placeholders, scrub_text

    text = "Herr W. verstieß gegen § 36 IfSG. A. Problem: W. schwieg."
    entities = [Entity(5, 7, "PER", 0.99)]
    mapping = placeholders(text, entities)
    assert mapping == {"W.": "[PER_1]"}
    assert (
        scrub_text(text, entities, mapping)
        == "Herr [PER_1] verstieß gegen § 36 IfSG. A. Problem: W. schwieg."
    )


def test_scrub_worker_tags_the_model_card_example():
    """The real worker in the scrub environment (skipped until envs/scrub and the model are present)."""
    from lexhybrid.data.corpus.scrub_ler import REPO_ROOT, SCRUB_PYTHON, run_worker
    from lexhybrid.data.schema import Document

    model = next(
        (REPO_ROOT / "data" / "hf" / "hub").glob(
            "models--flair--ner-german-legal/snapshots/*/pytorch_model.bin"
        ),
        None,
    )
    if not SCRUB_PYTHON.exists() or model is None:
        pytest.skip(
            "scrub environment or flair/ner-german-legal absent (uv sync --locked --project envs/scrub)"
        )
    doc = Document(id="t:1", source="t", jurisdiction="DE", doc_type="decision", licence="DE-UrhG-5",
                   commercial_safe=True, text="Herr W. verstieß gegen § 36 Abs. 7 IfSG.")  # fmt: skip
    other = Document(id="t:2", source="t", jurisdiction="DE", doc_type="decision", licence="DE-UrhG-5",
                     commercial_safe=True, text="Die Klage wird abgewiesen.")  # fmt: skip
    # the cache the skip check looked in, whatever HF_HOME / HF_HUB_CACHE say in this shell
    cache = ("--cache-dir", str(REPO_ROOT / "data" / "hf" / "hub"))
    entities, stats = run_worker([doc], args=cache)
    (person,) = [e for e in entities["t:1"] if e.label == "PER"]
    assert doc.text[person.start : person.end] == "W." and person.score > 0.9 and stats["docs"] == 1
    batched, stats = run_worker(
        [doc, other], args=(*cache, "--docs-per-batch", "2")
    )  # P4-L: one predict call
    assert batched["t:1"] == entities["t:1"] and batched["t:2"] == [] and stats["docs"] == 2


def test_collectors_record_hierarchy_and_parts():
    """P7-A/B's inputs (schema fields added 2026-09-29): statute collectors carry the act's
    structural headings above each provision (Document.hierarchy, outermost first), decision
    collectors the part of the decision each paragraph belongs to (Section.part); records written
    before the fields existed still load, with the defaults."""
    import json

    from lexhybrid.data.corpus.collectors.gii import parse_gii_xml
    from lexhybrid.data.corpus.collectors.oldp import parse_oldp_case
    from lexhybrid.data.corpus.collectors.rii import parse_rii_xml
    from lexhybrid.data.corpus.collectors.ris import parse_ris_decision

    gii = parse_gii_xml((FIXTURES / "gii" / "beurkg_excerpt.xml").read_bytes(), "beurkg")
    assert {tuple(d.hierarchy) for d in gii} == {("Abschnitt 1 Allgemeine Vorschriften",)}
    fedlex = {**_fedlex("or_excerpt"), **_fedlex("zgb_excerpt")}
    assert fedlex["OR Art. 97"].hierarchy == [
        "Erste Abteilung: Allgemeine Bestimmungen",
        "Zweiter Titel: Die Wirkung der Obligationen",
        "Zweiter Abschnitt: Die Folgen der Nichterfüllung",
    ]
    assert fedlex["ZGB SchlT Art. 1"].hierarchy[0].startswith("Schlusstitel")
    rii = parse_rii_xml((FIXTURES / "rii" / "jb-KARE600065578.xml").read_bytes())
    parts = [s.part for s in rii.sections]
    assert parts[0] == "Leitsatz" and "Tenor" in parts and parts[-1] == "Entscheidungsgründe"
    assert parts == sorted(parts, key=["Leitsatz", "Tenor", "Tatbestand", "Entscheidungsgründe"].index)
    assert {s.part for s in rii.sections if s.label == "Rn. 3"} == {"Tatbestand"}
    bger = _bger("CH_BGer_004_4A-102-2026_2026-08-17")
    assert {s.part for s in bger.sections if s.label.startswith("E. ")} == {"Erwägungen"}
    assert {s.part for s in bger.sections if s.label.startswith("Sachverhalt")} == {"Sachverhalt"}
    oldp = parse_oldp_case(json.loads((FIXTURES / "oldp" / "case_521941.json").read_text()))
    assert {s.part for s in oldp.sections} >= {"Tenor"} and all(
        s.part == s.label or (s.part is None and s.label == "Text") for s in oldp.sections
    )
    ris = parse_ris_decision(*_ris_fixture("JJT_20260917_OGH0002_0070OB00143_26V0000_000"))
    assert all(s.part == s.label for s in ris.sections) and "Rechtliche Beurteilung" in {
        s.part for s in ris.sections
    }
    old = json.loads(gii[0].to_json())
    old.pop("hierarchy")
    for s in old["sections"]:
        s.pop("part")
    back = Document.from_dict(old)
    assert back.hierarchy == [] and back.sections[0].part is None and back.sha256 == gii[0].sha256


def test_scrub_worker_tags_a_batch_of_documents_in_one_predict_call(script, monkeypatch):
    """P4-L: `--docs-per-batch` puts the sentences of several documents into ONE predict call (short
    documents fill the GPU's mini-batches together) and yields the per-document entities unchanged."""
    import sys
    import types

    class Label:
        def __init__(self, value, score):
            self.value, self.score = value, score

    class Span:
        def __init__(self, start, end, value):
            self.start_position, self.end_position, self._label = start, end, Label(value, 0.99)

        def get_label(self, name):
            return self._label

    class Sentence:
        def __init__(self, text, use_tokenizer=True):
            assert use_tokenizer is False
            self.text, self.spans = text, []

        def get_spans(self, name):
            return self.spans

    class Tagger:
        def __init__(self):
            self.calls = []

        def predict(self, sentences, mini_batch_size):
            self.calls.append(len(sentences))
            for s in sentences:
                s.spans = [Span(m.start(), m.end(), "PER") for m in re.finditer(r"Müller|Schmidt", s.text)]

    import re

    monkeypatch.setitem(sys.modules, "flair", types.ModuleType("flair"))
    monkeypatch.setitem(sys.modules, "flair.data", types.SimpleNamespace(Sentence=Sentence))
    worker = script("scrub_ner_worker")
    docs = [
        {"id": "a", "text": "Herr Müller klagt.\nFrau Schmidt nicht."},
        {"id": "b", "text": "Ohne Namen."},
        {"id": "c", "text": "Müller und Schmidt; Müller zahlt."},
    ]
    batched = Tagger()
    together = worker.tag(batched, docs, max_tokens=200, batch_size=32)
    assert batched.calls == [4], "one predict call over the sentences (lines) of all three documents"
    alone = Tagger()
    assert together == [worker.tag(alone, [d], 200, 32)[0] for d in docs] and alone.calls == [2, 1, 1]
    assert [e[2] for e in together[2]["entities"]] == ["PER", "PER", "PER"]


FAKE_SCRUB_WORKER = r"""
import json, re, sys
argv = sys.argv[1:]
batch = int(argv[argv.index("--docs-per-batch") + 1]) if "--docs-per-batch" in argv else 1
die_after = int(argv[argv.index("--die-after") + 1]) if "--die-after" in argv else None
pending, n = [], 0
def flush():
    for doc in pending:
        ents = [[m.start(), m.end(), "PER", 0.99] for m in re.finditer(r"Müller", doc["text"])]
        print(json.dumps({"id": doc["id"], "entities": ents}, ensure_ascii=False), flush=True)
    pending.clear()
for line in sys.stdin:
    if not line.strip():
        continue
    n += 1
    if die_after is not None and n > die_after:
        print("worker died", file=sys.stderr); sys.exit(3)
    pending.append(json.loads(line))
    if len(pending) >= batch:
        flush()
flush()
print("some flair log line", file=sys.stderr)
print(json.dumps({"docs": n, "chars": 0, "seconds": 0.1, "docs_per_s": 1.0}), file=sys.stderr)
"""


def test_scrub_streams_parts_through_one_worker(tmp_path):
    """P4-L: stride parts of a source, streamed through one worker process with a bounded window
    of documents in flight; outputs named per part and renamed from *.partial on completion; a
    worker that dies leaves only the *.partial files; a consumer that stops early does not hang."""
    import itertools
    import sys

    from lexhybrid.data.corpus.scrub_ler import iter_part, part_name, scrub_file, stream_worker

    fake = tmp_path / "fake_worker.py"
    fake.write_text(FAKE_SCRUB_WORKER)
    docs = [_doc(id=f"gii:bgb:{i}", text=f"Herr Müller zahlt {i} Euro.") for i in range(50)]
    raw = tmp_path / "gii.jsonl"
    raw.write_text("".join(d.to_json() + "\n" for d in docs))
    assert [d.id for d in iter_part(raw, 1, 3)] == [f"gii:bgb:{i}" for i in range(1, 50, 3)]
    with pytest.raises(ValueError):
        next(iter_part(raw, 3, 3))
    assert part_name("gii", 0, 1) == "gii.jsonl" and part_name("gii", 1, 4) == "gii.part-1-of-4.jsonl"

    stats = {}
    streamed = list(
        stream_worker(iter(docs), sys.executable, ("--docs-per-batch", "4"), fake, in_flight=16, stats=stats)
    )
    assert [d.id for d, _ in streamed] == [d.id for d in docs] and stats["docs"] == 50
    assert all(e[0].label == "PER" for _, e in streamed)
    head = list(itertools.islice(stream_worker(iter(docs), sys.executable, (), fake, in_flight=4), 3))
    assert len(head) == 3  # the generator was closed early: the worker is killed, nothing hangs

    out_ids = []
    for k in (0, 1):
        name = part_name("gii", k, 2)
        s = scrub_file("gii", raw, tmp_path / "scrubbed" / name, tmp_path / "manifests" / f"scrub_{name}",
                       python=sys.executable, part=k, parts=2, worker=fake, in_flight=8)  # fmt: skip
        assert s["docs"] == 25 and s["entities"] == 25 and s["worker"]["docs"] == 25
        lines = (tmp_path / "scrubbed" / name).read_text().splitlines()
        assert all("[PER_1] zahlt" in Document.from_json(ln).text for ln in lines)
        out_ids += [Document.from_json(ln).id for ln in lines]
    assert sorted(out_ids) == sorted(d.id for d in docs)
    assert not list(tmp_path.rglob("*.partial"))

    dead = tmp_path / "scrubbed" / "dead.jsonl"
    with pytest.raises(RuntimeError, match="worker died"):
        scrub_file("gii", raw, dead, tmp_path / "manifests" / "scrub_dead.jsonl", python=sys.executable,
                   worker_args=("--die-after", "10"), worker=fake, in_flight=4)  # fmt: skip
    assert not dead.exists() and (tmp_path / "scrubbed" / "dead.jsonl.partial").exists()


# -- deduplication (P3-S) ----------------------------------------------------------------------


def _oldp_docs():
    import json

    from lexhybrid.data.corpus.collectors.oldp import parse_oldp_case

    return [
        parse_oldp_case(json.loads(p.read_text()), retrieved_at="2026-09-27T16:00:00+00:00")
        for p in sorted((FIXTURES / "oldp").glob("case_*.json"))
    ]


def test_dedup_catches_planted_near_duplicate():
    """P3-S: exact-hash pre-pass, then MinHash (5-gram shingles, 128 permutations, Jaccard >= 0.8).
    A near duplicate inside one source is dropped; the copy in another source (here the research-only
    Multi Legal Pile copy of an OLDP decision) is kept and linked (decision 13 as amended by the user,
    2026-09-29: "keep both and when cited cite both"); whatever the input order, the same answer."""
    import dataclasses
    import random

    from lexhybrid.data.corpus.dedup import deduplicate, minhash, shingles

    docs = _oldp_docs() + [_bger("CH_BGer_004_4A-102-2026_2026-08-17")]
    base = docs[1]
    words = base.text.split(" ")
    for i in random.Random(0).sample(range(len(words)), 3):
        words[i] = "geändert"
    near = dataclasses.replace(base, id="oldp:planted", text=" ".join(words), sha256="")
    exact_nc = dataclasses.replace(
        docs[0], id="multilegalpile:de_caselaw_germany_openlegaldata:7", source="multilegalpile",
        licence="CC-BY-NC-SA-4.0", commercial_safe=False, research_only=True,
    )  # fmt: skip
    true_jaccard = len(shingles(base.text) & shingles(near.text)) / len(
        shingles(base.text) | shingles(near.text)
    )
    assert 0.8 <= true_jaccard < 1.0 and minhash(base.text).jaccard(minhash(near.text)) >= 0.8

    kept, log = deduplicate([exact_nc, near, *docs])  # the research-only copy comes first on purpose
    # every original, then the research-only copy (visited last), and not the planted duplicate
    assert [d.id for d in kept] == [*sorted(d.id for d in docs), exact_nc.id]
    by_id = {entry.id: entry for entry in log}
    link = by_id["multilegalpile:de_caselaw_germany_openlegaldata:7"]
    assert (link.duplicate_of, link.kind, link.dropped) == (docs[0].id, "exact", False)
    assert by_id["oldp:planted"].kind == "near" and by_id["oldp:planted"].duplicate_of == base.id
    assert by_id["oldp:planted"].dropped is True
    assert set(by_id) == {"multilegalpile:de_caselaw_germany_openlegaldata:7", "oldp:planted"}
    shuffled = [exact_nc, near, *docs]
    random.Random(1).shuffle(shuffled)
    assert [d.id for d in deduplicate(shuffled)[0]] == [d.id for d in kept]


def test_dedup_at_scale_makes_the_same_decisions(tmp_path):
    """P4-L: signatures (no text, 512 bytes of MinHash) computed in worker processes from a stream,
    then the same visiting order and decisions as `deduplicate`."""
    import dataclasses

    import numpy as np

    from lexhybrid.data.corpus.dedup import deduplicate, deduplicate_stream, signature

    docs = _oldp_docs() + [_bger("CH_BGer_004_4A-102-2026_2026-08-17")]
    near = dataclasses.replace(docs[1], id="oldp:planted", text=docs[1].text.replace(" ", "  ", 2), sha256="")
    twin = dataclasses.replace(docs[0], id="oldp:exact-copy")
    web = dataclasses.replace(  # another source: a near copy that must stay, linked
        _fineweb_docs()[0], id="fineweb2_de:copy", text=docs[2].text.replace(" ", "  ", 1), sha256=""
    )
    corpus = [twin, near, web, *docs]
    kept, log = deduplicate(corpus)
    sig = signature(docs[0])
    assert len(sig.hashvalues) == 128 * 4 and np.frombuffer(sig.hashvalues, np.uint32).shape == (128,)
    for workers in (1, 2):
        ids, stream_log = deduplicate_stream(iter(corpus), workers=workers)
        assert ids == [d.id for d in kept] and stream_log == log
    assert {e.id for e in log if e.dropped} == {"oldp:exact-copy", "oldp:planted"}
    (linked,) = [e for e in log if not e.dropped]
    assert {linked.id, linked.duplicate_of} == {"fineweb2_de:copy", docs[2].id} and linked.kind == "near"
    assert "fineweb2_de:copy" in {d.id for d in kept}


def test_dedup_shingles():
    from lexhybrid.data.corpus.dedup import shingles

    assert shingles("Der Vermieter kann kündigen, wenn er ein Interesse hat.") >= {
        "der vermieter kann kündigen wenn"
    }
    assert shingles("Zu kurz") == {"zu kurz"} and shingles("") == set()


# -- currency changelog (P3-T) -----------------------------------------------------------------


def test_changelog_from_fixture_versions(tmp_path):
    """P3-T: RIS versions (linked to the version they replaced), Fedlex articles by their footnote
    dates, GII provisions by their act's Stand (granularity 'act'); nothing older than the date."""
    import json

    from lexhybrid.data.corpus.changelog import changes_since, write_changelog
    from lexhybrid.data.corpus.collectors.gii import parse_gii_xml
    from lexhybrid.data.corpus.collectors.ris import parse_ris_norm

    ris = [
        parse_ris_norm(*_ris_fixture(n), retrieved_at="2026-09-27T16:00:00+00:00")
        for n in ("NOR12018413", "NOR40172917", "NOR12019037")
    ]
    fedlex = list(_fedlex("or_excerpt").values())
    gii = parse_gii_xml(
        (FIXTURES / "gii" / "beurkg_excerpt.xml").read_bytes(),
        "beurkg",
        retrieved_at="2026-09-27T16:00:00+00:00",
    )
    changes = changes_since([*ris, *fedlex, *gii], since="2011-01-01")
    got = [(c.citation_id, c.valid_from, c.granularity, c.previous_id) for c in changes]
    assert got == [
        ("BeurkG §1", "2025-12-10", "act", None),
        ("BeurkG §2", "2025-12-10", "act", None),
        ("BeurkG §3", "2025-12-10", "act", None),
        ("OR Art. 40a", "2022-01-01", "provision", None),
        ("ABGB §688", "2017-01-01", "provision", "ris:NOR12018413"),  # the 1812 version it replaced
        ("OR Art. 97", "2011-01-01", "provision", None),
    ]
    assert all(c.valid_from >= "2011-01-01" for c in changes)
    assert changes_since([*ris, *fedlex, *gii], since="2030-01-01") == []
    write_changelog(changes, tmp_path / "changelog.jsonl", "2011-01-01")
    lines = (tmp_path / "changelog.jsonl").read_text().splitlines()
    assert json.loads(lines[0]) == {"since": "2011-01-01", "changes": 6} and len(lines) == 7


# -- packing (P3-U) ----------------------------------------------------------------------------


def _char_codec():
    """A toy tokenizer for the packing tests: one id per character, EOS = 0."""
    return (lambda text: [ord(c) + 1 for c in text]), (lambda ids: "".join(chr(i - 1) for i in ids)), 0


def _unpack(rows, eos_id, decode):
    """The documents back out of rows: real tokens concatenated and split after each EOS."""
    stream = [t for r in rows for t in r["input_ids"][: r["length"]]]
    docs, cur = [], []
    for t in stream:
        if t == eos_id:
            docs.append(decode(cur))
            cur = []
        else:
            cur.append(t)
    assert cur == []  # every document ends with its EOS
    return docs


def test_packing_roundtrip_doc_ids(tmp_path):
    """P3-U: EOS after every document, doc_ids stepping right after it, documents carried across
    rows (never dropped), the last row padded with EOS that joins its last document; shards
    round-trip through parquet."""
    from lexhybrid.data.packing import pack, read_rows, write_shards

    encode, decode, eos = _char_codec()
    docs = _oldp_docs()
    rows = list(pack(docs, encode, eos, row_len=1024))
    assert all(len(r.input_ids) == len(r.doc_ids) == 1024 for r in rows)
    assert all(r.length == 1024 for r in rows[:-1]) and rows[-1].length < 1024
    assert sum(r.length for r in rows) == sum(len(d.text) + 1 for d in docs)  # nothing dropped
    for r in rows:
        ids = r.doc_ids[: r.length]
        assert ids[0] == 0 and all(b - a in (0, 1) for a, b in zip(ids, ids[1:], strict=False))
        for i in range(1, r.length):  # a new local document starts exactly after an EOS
            assert (ids[i] != ids[i - 1]) == (r.input_ids[i - 1] == eos)
        assert len(r.documents) == ids[-1] + 1 and r.source == "oldp" and r.licence == "ODbL-1.0"
    last = rows[-1]
    assert set(last.input_ids[last.length :]) == {eos} and set(last.doc_ids[last.length :]) == {
        last.doc_ids[last.length - 1]
    }
    spanning = [r for r in rows[1:] if r.documents[0] == rows[rows.index(r) - 1].documents[-1]]
    assert spanning, "the fixtures are longer than a row: some document must continue into the next row"

    paths = write_shards(rows, tmp_path / "shards", rows_per_shard=4)
    back = [row for p in paths for row in read_rows(p)]
    assert len(paths) == -(-len(rows) // 4) and len(back) == len(rows)
    assert _unpack(back, eos, decode) == [d.text for d in docs]
    assert back[0]["commercial_safe"] is True and back[0]["research_only"] is False


def test_packing_keeps_one_source_per_pack():
    from lexhybrid.data.packing import pack

    encode, _, eos = _char_codec()
    with pytest.raises(ValueError, match="one source per pack"):
        list(pack([*_oldp_docs()[:1], _bger("CH_BGer_001_1C-464-2026_2026-09-16")], encode, eos, 256))


def test_packing_with_the_qwen3_tokenizer(qwen3):
    """The real tokenizer: every document decodes back exactly from 4,096-token rows."""
    from lexhybrid.data.packing import pack
    from lexhybrid.data.tokenizer import EOS_ID, encode_document

    docs = _oldp_docs()
    rows = [
        vars(r) | {"length": r.length} for r in pack(docs, lambda t: encode_document(qwen3, t), EOS_ID, 4096)
    ]
    assert _unpack(rows, EOS_ID, qwen3.decode) == [d.text for d in docs]


# -- datasets: split, mixture, arms (P3-V) -----------------------------------------------------


def _mlp_docs():
    import json

    from lexhybrid.data.corpus.collectors.multilegalpile import parse_mlp_row

    fixture = json.loads((FIXTURES / "multilegalpile" / "rows.json").read_text())
    return [
        parse_mlp_row(v["row"], s, v["index"], retrieved_at="2026-09-27T16:00:00+00:00")
        for s, v in fixture.items()
    ]


def _fineweb_docs():
    import json

    from lexhybrid.data.corpus.collectors.fineweb2_de import parse_fineweb_row

    rows = json.loads((FIXTURES / "fineweb2_de" / "rows.json").read_text())
    return [parse_fineweb_row(r, retrieved_at="2026-09-27T16:00:00+00:00") for r in rows]


def _build(tmp_path, docs, row_len=64, minimum=1):
    from lexhybrid.data.datasets import build_source_shards

    encode, _, eos = _char_codec()
    return build_source_shards(docs, encode, eos, row_len, tmp_path, fraction=0.001, minimum=minimum, seed=0)


def test_val_split_by_document(tmp_path):
    """P3-V: 0.1% of the documents or the minimum (2,000 by default), by seeded hash, disjoint from
    training, the same every time; a corpus too small for it is refused; packing happens after the
    split, so every row belongs to one side."""
    import pyarrow.parquet as pq

    from lexhybrid.data.datasets import split_by_document

    ids = [f"doc:{i}" for i in range(50_000)]
    val = split_by_document(ids, fraction=0.001, minimum=20, seed=0)
    assert len(val) == 50 and val <= set(ids)  # 0.1% beats the minimum here
    assert split_by_document(ids[:10_000], fraction=0.001, minimum=20) == split_by_document(
        ids[:10_000], 0.001, 20
    )
    assert len(split_by_document(ids[:10_000], fraction=0.001, minimum=20)) == 20  # the minimum binds
    assert split_by_document(ids, 0.001, 20, seed=1) != val
    assert len(split_by_document(ids[:10_000])) == 2000  # the defaults: 0.1%, at least 2,000
    with pytest.raises(ValueError, match="cannot give a validation split"):
        split_by_document(ids[:1_000])

    meta = _build(tmp_path, _oldp_docs())
    train_ids, val_ids = set(meta["train"]["document_ids"]), set(meta["val"]["document_ids"])
    assert len(val_ids) == 1 and len(train_ids) == 2 and not train_ids & val_ids
    for split, ids_ in (("train", train_ids), ("val", val_ids)):
        rows = [
            r
            for p in sorted((tmp_path / "64" / "oldp" / split).glob("*.parquet"))
            for r in pq.read_table(p).to_pylist()
        ]
        assert {d for r in rows for d in r["documents"]} == ids_


def test_no_research_only_in_commercial_safe_shard(tmp_path):
    """R9 / P3-V: the commercial-safe arm never serves a research-only row -- the mixture leaves an
    `arm: research` source out, and shards that are not commercial-safe are refused even when a
    config names them. The research arm serves both."""
    import yaml

    from lexhybrid.data.datasets import PackedMixture, SourceSpec, effective_weights, source_specs

    _build(tmp_path, _oldp_docs())
    _build(tmp_path, _mlp_docs())
    specs = [SourceSpec("oldp", "legal"), SourceSpec("multilegalpile", "legal", arm="research")]
    assert effective_weights({"legal": 1.0}, specs, "commercial_safe") == {"oldp": 1.0}
    assert effective_weights({"legal": 1.0}, specs, "research") == {"oldp": 0.5, "multilegalpile": 0.5}
    with pytest.raises(ValueError, match="not commercial-safe"):
        PackedMixture(
            tmp_path, 64, "train", {"oldp": 0.5, "multilegalpile": 0.5}, "commercial_safe", num_samples=10
        )
    safe = PackedMixture(tmp_path, 64, "train", {"oldp": 1.0}, "commercial_safe", num_samples=200)
    served = [safe.sources[n].row(k) for n, k in safe.index]
    assert all(r["commercial_safe"] and not r["research_only"] for r in served)
    research = PackedMixture(
        tmp_path, 64, "train", {"oldp": 0.5, "multilegalpile": 0.5}, "research", num_samples=200
    )
    assert {n for n, _ in research.index} == {"oldp", "multilegalpile"}
    shipped = yaml.safe_load(
        (FIXTURES.parents[1].parent / "configs" / "dataset" / "mixture_pretrain.yaml").read_text()
    )
    commercial = effective_weights(shipped["groups"], source_specs(shipped["sources"]), "commercial_safe")
    assert "multilegalpile" not in commercial and abs(sum(commercial.values()) - 1) < 1e-9


def test_mixture_draws_the_configured_shares(tmp_path):
    """Rows are drawn in the configured proportions on a seeded schedule that is the same every time
    and resumes by offset; labels past a row's real length are -100; validation reads every row."""
    import torch

    from lexhybrid.data.datasets import PackedMixture, effective_weights, source_specs

    _build(tmp_path, _oldp_docs())
    _build(tmp_path, _fineweb_docs())
    weights = effective_weights(
        {"legal": 0.7, "general": 0.3},
        source_specs({"oldp": {"group": "legal"}, "fineweb2_de": {"group": "general"}}),
        "commercial_safe",
    )
    full = PackedMixture(tmp_path, 64, "train", weights, "commercial_safe", num_samples=4000, seed=0)
    share = sum(n == "oldp" for n, _ in full.index) / len(full.index)
    assert abs(share - 0.7) < 0.03
    assert (
        PackedMixture(tmp_path, 64, "train", weights, "commercial_safe", num_samples=4000, seed=0).index
        == full.index
    )
    resumed = PackedMixture(
        tmp_path, 64, "train", weights, "commercial_safe", num_samples=3990, seed=0, start=10
    )
    assert resumed.index == full.index[10:]
    alone = PackedMixture(tmp_path, 64, "train", {"oldp": 1.0}, "commercial_safe", num_samples=500, seed=0)
    in_mix = [r for n, r in full.index if n == "oldp"]
    assert [r for _, r in alone.index] == in_mix[:500]  # a source's order does not depend on the others
    item = full[0]
    assert {k: tuple(v.shape) for k, v in item.items()} == {
        "input_ids": (64,),
        "labels": (64,),
        "doc_ids": (64,),
    }
    src = full.sources["oldp"]
    last = src.row(len(src) - 1)
    k = next(i for i, (n, r) in enumerate(full.index) if n == "oldp" and r == len(src) - 1)
    assert last["length"] < 64 and torch.all(full[k]["labels"][last["length"] :] == -100)
    val = PackedMixture(tmp_path, 64, "val", weights, "commercial_safe")
    assert len(val) == sum(len(s) for s in val.sources.values())


def test_streaming_build_equals_the_in_memory_build_at_every_row_len(tmp_path):
    """P4-L at scale: one pass, each document tokenised once, packed at two row lengths together --
    the same shards and meta.json as build_source_shards at each length on its own."""
    import json

    from lexhybrid.data.datasets import build_source_shards, build_source_shards_stream, split_by_document
    from lexhybrid.data.packing import read_rows

    encode, _, eos = _char_codec()
    docs = _oldp_docs()
    calls = []

    def counted(text):
        calls.append(text)
        return encode(text)

    val_ids = split_by_document([d.id for d in docs], 0.001, 1, 0)
    metas = build_source_shards_stream(
        iter(docs), val_ids, counted, eos, (32, 64), tmp_path / "stream", 0.001, 1, 0, {"scrubbed": True}
    )
    assert len(calls) == len(docs), "every document is tokenised once for both row lengths"
    for n in (32, 64):
        ref = build_source_shards(docs, encode, eos, n, tmp_path / f"ref{n}", 0.001, 1, 0, {"scrubbed": True})
        assert metas[n] == ref
        stream_dir, ref_dir = tmp_path / "stream" / str(n) / "oldp", tmp_path / f"ref{n}" / str(n) / "oldp"
        assert json.loads((stream_dir / "meta.json").read_text()) == ref
        for split in ("train", "val"):
            got = sorted((stream_dir / split).glob("*.parquet"))
            want = sorted((ref_dir / split).glob("*.parquet"))
            assert [p.name for p in got] == [p.name for p in want]
            assert [read_rows(p) for p in got] == [read_rows(p) for p in want]
    with pytest.raises(ValueError, match="no documents"):
        build_source_shards_stream(iter([]), set(), encode, eos, (32,), tmp_path / "empty")
    with pytest.raises(ValueError, match="one source"):
        build_source_shards_stream(
            iter(docs + _fineweb_docs()), val_ids, encode, eos, (32,), tmp_path / "mix"
        )


def _char_encoder():
    return _char_codec()[0]


def test_build_shards_streams_scrub_parts_dedups_and_packs_every_length(tmp_path, script):
    """P4-L's pack step (scripts/build_shards.py): a whole scrubbed file and a source in scrub-array
    parts, deduplicated together (a cross-source copy is dropped from its source before packing),
    packed at two row lengths; the summary P4-M reads; incomplete parts and unscrubbed sources
    are refused."""
    import argparse
    import dataclasses
    import json

    bs = script("build_shards")
    scrubbed, raw, root = tmp_path / "scrubbed", tmp_path / "raw", tmp_path / "shards"
    scrubbed.mkdir()
    oldp, web = _oldp_docs(), _fineweb_docs()
    copy = dataclasses.replace(web[0], id="fineweb2_de:copy-of-oldp", text=oldp[0].text, sha256="")
    twin = dataclasses.replace(oldp[1], id="oldp:twin")  # a duplicate inside one source: dropped
    (scrubbed / "oldp.jsonl").write_text("".join(d.to_json() + "\n" for d in [*oldp, twin]))
    parts = [[d for i, d in enumerate([*web, copy]) if i % 2 == k] for k in (0, 1)]
    for k, docs in enumerate(parts):
        (scrubbed / f"fineweb2_de.part-{k}-of-2.jsonl").write_text("".join(d.to_json() + "\n" for d in docs))
    args = argparse.Namespace(
        sources=["oldp", "fineweb2_de", "gii"], scrubbed=scrubbed, raw=raw, allow_unscrubbed=False,
        workers=1, row_len=[32, 64], root=root, val_fraction=0.001, min_val_docs=1, seed=0,
    )  # fmt: skip
    part_files = [scrubbed / f"fineweb2_de.part-{k}-of-2.jsonl" for k in (0, 1)]
    assert [d.id for d in bs.iter_documents(part_files)] == [d.id for d in [*web, copy]], "collection order"
    summary = bs.build(args, encoder_factory=_char_encoder, eos_id=0)
    assert (summary["duplicates_dropped"], summary["copies_linked"]) == (1, 1)
    assert summary["documents"] == len(oldp) + len(web) + 2
    for n in (32, 64):
        entries = [json.loads(ln) for ln in (root / str(n) / "dedup.jsonl").read_text().splitlines()]
        (drop,) = [e for e in entries if e["dropped"]]
        (entry,) = [e for e in entries if not e["dropped"]]
        assert (drop["id"], drop["duplicate_of"]) == ("oldp:twin", oldp[1].id)
        assert {entry["id"], entry["duplicate_of"]} == {copy.id, oldp[0].id} and entry["kind"] == "exact"
        shipped = set()
        for source in ("oldp", "fineweb2_de"):
            meta = json.loads((root / str(n) / source / "meta.json").read_text())
            assert meta["row_len"] == n and meta["scrubbed"] is True
            shipped |= set(meta["train"]["document_ids"] + meta["val"]["document_ids"])
            got = summary["sources"][source][str(n)]
            assert (
                got["train"]["documents"] + got["val"]["documents"]
                == len(meta["train"]["document_ids"] + meta["val"]["document_ids"])
                and "document_ids" not in got["train"]
            )
        assert {entry["id"], entry["duplicate_of"]} <= shipped, "a copy in another source is packed too"
        assert shipped == {d.id for d in [*oldp, *web, copy]}, "the within-source twin is not"
    assert json.loads((root / "build_summary.json").read_text()) == json.loads(json.dumps(summary))
    assert "gii" not in summary["sources"]  # no files: skipped
    (raw / "gii").mkdir(parents=True)
    (raw / "gii" / "gii.jsonl").write_text("{}\n")
    with pytest.raises(SystemExit, match="not scrubbed"):
        bs.build(args, encoder_factory=_char_encoder, eos_id=0)
    (scrubbed / "fineweb2_de.part-1-of-2.jsonl").rename(scrubbed / "fineweb2_de.part-1-of-3.jsonl")
    with pytest.raises(SystemExit, match="incomplete or mixed scrub parts"):
        bs.source_files("fineweb2_de", scrubbed, raw, False)
    with pytest.raises(SystemExit, match="oldp: .*cannot give a validation split"):
        bs.build(
            argparse.Namespace(**{**vars(args), "sources": ["oldp"], "min_val_docs": 10_000}),
            _char_encoder,
            0,
        )


def test_mixture_cuts_stored_rows_for_a_shorter_row_len(tmp_path):
    """The P5 screen trains on 2,048-token rows and P4-L packs 4,096 and 8,192: a row length with
    no shards of its own is served from the smallest packed multiple, each stored row cut into
    consecutive sub-rows that are exactly the stored row's pieces (here 32 from 64)."""
    import torch

    from lexhybrid.data.datasets import PackedMixture, SourceShards, shard_directory

    _build(tmp_path, _oldp_docs(), row_len=64)
    _build(tmp_path, _oldp_docs(), row_len=128)
    assert shard_directory(tmp_path, 64, "oldp") == (tmp_path / "64" / "oldp", 1)
    assert shard_directory(tmp_path, 32, "oldp") == (tmp_path / "64" / "oldp", 2)  # smallest multiple
    with pytest.raises(FileNotFoundError, match="multiple"):
        shard_directory(tmp_path, 48, "oldp")
    stored, cut = (
        SourceShards(tmp_path / "64" / "oldp", "train"),
        SourceShards(tmp_path / "64" / "oldp", "train", 2),
    )
    assert len(cut) == 2 * len(stored)
    for k in range(len(stored)):
        whole, a, b = stored.row(k), cut.row(2 * k), cut.row(2 * k + 1)
        assert a["input_ids"] + b["input_ids"] == whole["input_ids"]
        assert a["doc_ids"] + b["doc_ids"] == whole["doc_ids"]
        assert a["length"] + b["length"] == whole["length"] and a["documents"] == whole["documents"]
    last = cut.row(len(cut) - 1)
    assert 0 <= last["length"] <= 32 and last["length"] == max(0, stored.row(len(stored) - 1)["length"] - 32)
    with pytest.raises(IndexError):
        cut.row(len(cut))
    with pytest.raises(ValueError, match="cut"):
        SourceShards(tmp_path / "64" / "oldp", "train", 3)
    mix = PackedMixture(tmp_path, 32, "train", {"oldp": 1.0}, "commercial_safe", num_samples=50)
    item = mix[0]
    assert item["input_ids"].shape == (32,) and item["doc_ids"].shape == (32,)
    val = PackedMixture(tmp_path, 32, "val", {"oldp": 1.0}, "commercial_safe")
    val_cut = val.sources["oldp"]
    assert len(val) == len(val_cut) == 2 * len(SourceShards(tmp_path / "64" / "oldp", "val"))
    tail = val_cut.row(len(val_cut) - 1)["length"]  # val reads every sub-row once, in order
    assert torch.all(val[len(val) - 1]["labels"][tail:] == -100)


# -- probes: MQAR and statute recall (P3-W) ----------------------------------------------------


def test_mqar_generator_has_the_reference_semantics():
    from lexhybrid.data.probes.mqar import REFERENCE, MQARConfig, generate

    assert (REFERENCE.vocab_size, REFERENCE.num_kv_pairs, REFERENCE.num_queries) == (8192, 4, 4)
    assert (REFERENCE.power_a, REFERENCE.context_length, REFERENCE.seed) == (0.01, 2048, 42)
    items = generate(20)
    assert generate(20)[0].input_ids == items[0].input_ids  # seeded
    for it in items:
        context = it.input_ids[:8]
        pairs = dict(zip(context[0::2], context[1::2], strict=True))
        assert len(it.input_ids) == 2048 and len(it.query_positions) == 4
        for pos, value in zip(it.query_positions, it.values, strict=True):
            assert it.input_ids[pos] in pairs and pairs[it.input_ids[pos]] == value == it.input_ids[pos + 1]
        keys = set(pairs)
        filler = [t for i, t in enumerate(it.input_ids[8:], start=8) if i not in it.query_positions]
        assert not keys & set(filler)  # a key appears only in the context and at its query
        assert min(it.input_ids) >= REFERENCE.id_offset
    small = generate(
        2, MQARConfig(vocab_size=64, context_length=40, num_kv_pairs=3, num_queries=2, id_offset=4)
    )
    assert len(small[0].input_ids) == 40 and len(small[0].values) == 2


def test_mqar_score_counts_the_value_predictions():
    import torch

    from lexhybrid.data.probes.mqar import MQARConfig, generate, score

    class Oracle:  # predicts the next token of the sequence it is given
        def __init__(self, vocab, shift=0):
            self.vocab, self.shift = vocab, shift

        def eval(self):
            return self

        def backbone(self, ids):
            nxt = torch.cat([ids[:, 1:], ids[:, :1]], dim=1) + self.shift
            return nxt.unsqueeze(-1).float(), None

        def head(self, h):
            return torch.nn.functional.one_hot(h.squeeze(-1).long() % self.vocab, self.vocab).float()

    cfg = MQARConfig(vocab_size=64, context_length=48, id_offset=4)
    items = generate(5, cfg)
    assert score(Oracle(4 + 128), items, batch_size=2) == {"accuracy": 1.0, "queries": 20, "items": 5}
    assert score(Oracle(4 + 128, shift=1), items)["accuracy"] == 0.0


def _statute_docs():
    from lexhybrid.data.corpus.collectors.ris import parse_ris_norm

    ris = [parse_ris_norm(*_ris_fixture(n)) for n in ("NOR12018413", "NOR40172917", "NOR12019037")]
    return [*ris, *_fedlex("or_excerpt").values(), *_fedlex("zgb_excerpt").values()]


def test_statute_recall_items_need_the_context():
    """P3-W: the prompt holds the provision, more law after it, then the question; the answer, its
    Absatz 2, is in the prompt verbatim; exact match and char-F1 behave on strings."""
    from lexhybrid.data.probes.statute_recall import build_items, char_f1, exact_match

    encode = lambda t: list(t)  # noqa: E731 -- one token per character
    items = build_items(_statute_docs(), encode, context_tokens=3000)
    by = {i.citation_id: i for i in items}
    assert {"ABGB §1295", "OR Art. 1", "OR Art. 97", "ZGB Art. 1"} <= set(by)
    item = by["OR Art. 97"]
    assert item.prompt.endswith("\n\nOR Art. 97 Abs. 2 lautet:\n") and item.target in item.prompt
    assert item.prompt.startswith("Art. 97 ") and len(item.prompt) + len(item.target) <= 3000
    assert item.prompt.count("\n\n") > 1  # other provisions sit between the provision and the question
    assert exact_match(" Für  die Vollstreckung", "Für die Vollstreckung") and not exact_match(
        "Für", "Für die"
    )
    assert char_f1(item.target, item.target) == 1.0 and 0 < char_f1("Vollstreckung", item.target) < 1
    assert char_f1("", item.target) == 0.0
    half = item.target[: len(item.target) // 2]
    unrelated = by["ZGB Art. 1"].target[: len(item.target)]
    assert char_f1(half, item.target) > 0.5 > char_f1(unrelated, item.target)  # trigrams, not letters


def test_evaluate_probes_runs_on_a_random_model(script):
    """`scripts/evaluate_probes.py` end to end on a tiny random model with a toy tokenizer."""
    from lexhybrid import HybridConfig, HybridLanguageModel
    from lexhybrid.data.probes.mqar import MQARConfig

    probes = script("evaluate_probes")
    model = HybridLanguageModel(
        HybridConfig(
            vocab_size=1024,
            dim=32,
            num_layers=2,
            layer_pattern=["mamba3", "attention"],
            num_heads=2,
            head_dim=16,
            mamba3_head_dim=16,
            mamba3_d_state=16,
            max_position_embeddings=2048,
            tfla_impl="exact",
            mlstm_chunk_size=8,
        )  # fmt: skip
    ).eval()
    encode = lambda t: [min(ord(c), 1022) + 1 for c in t]  # noqa: E731
    decode = lambda ids: "".join(chr(max(i - 1, 32)) for i in ids)  # noqa: E731
    out = probes.run_probes(model, encode, decode, _statute_docs(), context=1500, mqar_items=4, recall_items=2,
                            mqar_cfg=MQARConfig(vocab_size=256, context_length=128, id_offset=4))  # fmt: skip
    assert out["mqar"]["queries"] == 16 and 0.0 <= out["mqar"]["accuracy"] <= 1.0
    assert out["statute_recall"]["items"] == 2 and 0.0 <= out["statute_recall"]["char_f1"] <= 1.0
    assert "multihop" in out and out["multihop"]["items"] >= 0


# -- probe: two-hop cross-references (P3-X) ----------------------------------------------------


def test_multihop_items_need_two_passages():
    """P3-X: A refers to exactly one provision B of the same act; the question does not name B; the
    answer (B's title) is not in A, is in B, and leaves the context when B is taken out, so both
    passages are needed; distractors come from the same act."""
    from lexhybrid.data.probes.multihop import build_items, same_act_references, title
    from lexhybrid.data.schema import Document

    path = FIXTURES.parent / "probes" / "bgb_multihop.jsonl"
    docs = {d.citation_id: d for d in (Document.from_json(line) for line in path.read_text().splitlines())}
    assert same_act_references(docs["BGB §169"]) == {"674", "729"}  # a list is several references
    items = build_items(list(docs.values()), n_items=10)
    assert [(i.source_citation, i.target_citation, i.answer) for i in items] == [
        ("BGB §168", "BGB §167", "Erteilung der Vollmacht"),
        ("BGB §843", "BGB §760", "Vorauszahlung"),
    ]
    for item in items:
        a, b = docs[item.source_citation], docs[item.target_citation]
        assert same_act_references(a) == {b.citation_id.split("§")[1]}
        assert item.answer == title(b) and item.answer not in a.text and item.answer in b.text
        assert item.target_citation not in item.question and item.source_citation in item.question
        assert a.text in item.context and b.text in item.context
        assert item.answer not in item.context.replace(b.text, "")  # without B the title is gone
        assert item.context.count("\n\n") == 3  # A, B and two distractors


# -- SFT format (P3-Y) -------------------------------------------------------------------------


def _sft_example():
    from lexhybrid.data.sft_format import Cite, Example, Passage, Quote, Text, split_sentences

    or_ = _fedlex("or_excerpt")
    zgb = _fedlex("zgb_excerpt")
    p1 = Passage("OR Art. 97", split_sentences(" ".join(s.text for s in or_["OR Art. 97"].sections)))
    p2 = Passage("ZGB Art. 8", split_sentences(zgb["ZGB Art. 8"].sections[0].text))
    answer = (
        Text("Der Schuldner haftet: "),
        Quote(1, 1),
        Text(" Die Beweislast regelt "),
        Cite(2),
        Text("."),
    )
    return Example("Wann haftet der Schuldner für Nichterfüllung?", (p1, p2), answer)


def test_sft_format_roundtrip():
    """P3-Y: the prompt is one document (no EOS inside), passages are sentence-numbered, answers are
    text with <|q|><|cK|><|sJ|> and <|cite|><|cK|> pointers ending on the EOS; render/parse round-trip,
    unanswerable included; malformed examples are refused."""
    import dataclasses

    from lexhybrid.data.sft_format import Example, Passage, Quote, parse, render

    ex = _sft_example()
    prompt, target = render(ex)
    assert prompt.startswith("<|question|>Wann haftet") and prompt.endswith("<|answer|>")
    assert "<|passage|><|c1|>OR Art. 97\n<|s1|>Kann die Erfüllung" in prompt and "<|endoftext|>" not in prompt
    assert (
        target == "Der Schuldner haftet: <|q|><|c1|><|s1|> Die Beweislast regelt <|cite|><|c2|>.<|endoftext|>"
    )
    assert parse(prompt + target) == ex
    none = Example(ex.question, ex.passages)
    assert render(none)[1] == "<|unanswerable|><|endoftext|>" and parse("".join(render(none))) == none
    bad = [
        dataclasses.replace(ex, passages=ex.passages * 9),  # 18 passages
        dataclasses.replace(ex, answer=(Quote(1, 99),)),
        dataclasses.replace(ex, answer=(Quote(3, 1),)),
        dataclasses.replace(ex, question="Was heisst <|cite|>?"),
        dataclasses.replace(ex, passages=(Passage("OR Art. 1", ()),)),
    ]
    for b in bad:
        with pytest.raises(ValueError):
            render(b)


def test_sft_encode_with_the_qwen3_tokenizer(qwen3):
    """The token form: specials are single ids, content cannot make one, one EOS at the very end,
    the prompt masked; decoding the ids gives back the rendered strings."""
    from lexhybrid.data.sft_format import encode, render
    from lexhybrid.data.tokenizer import EOS_ID, pointer_ids

    ex = _sft_example()
    enc = encode(ex, qwen3)
    ids = pointer_ids(qwen3)
    assert enc["input_ids"].count(EOS_ID) == 1 and enc["input_ids"][-1] == EOS_ID
    assert set(enc["doc_ids"]) == {0} and len(enc["labels"]) == len(enc["input_ids"])
    n_prompt = enc["labels"].index(next(label for label in enc["labels"] if label != -100))
    assert enc["input_ids"][n_prompt - 1] == ids["answer"] and all(
        x == -100 for x in enc["labels"][:n_prompt]
    )
    assert enc["labels"][n_prompt:] == enc["input_ids"][n_prompt:]
    assert ids["q"] in enc["input_ids"][n_prompt:] and ids["cite"] in enc["input_ids"][n_prompt:]
    assert qwen3.decode(enc["input_ids"]) == "".join(render(ex))


def test_sft_split_sentences():
    from lexhybrid.data.sft_format import split_sentences

    text = "(1) Die Frist beträgt nach § 573c Abs. 1 Nr. 1. BGB drei Monate. Sie verlängert sich. (2) Weiteres gilt."
    assert split_sentences(text) == (
        "(1) Die Frist beträgt nach § 573c Abs. 1 Nr. 1. BGB drei Monate.",
        "Sie verlängert sich.",
        "(2) Weiteres gilt.",
    )


def test_scrub_worker_reads_the_jobs_model_cache(script, monkeypatch):
    """On the cluster the worker must find flair's weights where fetch_hf.sh put them ($HF_HOME)."""
    worker = script("scrub_ner_worker")
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", "/sc/scratch/u/lexhybrid/.hf")
    assert worker.default_cache_dir() == "/sc/scratch/u/lexhybrid/.hf/hub"
    monkeypatch.setenv("HF_HUB_CACHE", "/elsewhere/hub")
    assert worker.default_cache_dir() == "/elsewhere/hub"
    monkeypatch.delenv("HF_HUB_CACHE")
    monkeypatch.delenv("HF_HOME")
    assert worker.default_cache_dir().endswith("data/hf/hub")
