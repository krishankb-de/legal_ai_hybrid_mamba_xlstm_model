"""Fedlex: Swiss federal law, the classified compilation (SR), in German (plan P3-K).

Source: the Federal Chancellery's linked-data platform. Its SPARQL endpoint
``https://fedlex.data.admin.ch/sparqlendpoint`` (JOLux ontology) lists every act of the SR
(``jolux:ConsolidationAbstract``, classified by its SR number, ``inForceStatus`` 0 = in force) and
the act's consolidations, one per **version** (``jolux:dateApplicability``), each with a German
Akoma Ntoso XML file in the filestore. The collector takes, per act, the latest version applicable
on ``as_of`` (default today; the endpoint also lists versions that apply only in the future).

The XML: ``<article eId="art_97">`` holds ``<paragraph eId="art_97/para_1">`` (``<num>1</num>``,
the text in ``content``), lists as ``blockList``/``item``, and footnotes as ``authorialNote`` --
SR references and the amendment history (``Fassung gemäss ..., in Kraft seit 1. Jan. 2011``), never
part of the text. The marginal titles (``Abschluss des Vertrages / Übereinstimmende
Willensäusserung / Im Allgemeinen``) are the ``level``s above an article; the header (``meta``)
carries the SR number, the German title and abbreviation, and the version's dates.

One ``Document`` per article (``OR Art. 97``); its sections are the Absätze (``Art. 97 Abs. 1``,
``Art. 6 Abs. 2bis``). ``valid_from`` is the article's own version date: the latest ``in Kraft
seit`` / ``mit Wirkung seit`` date in the footnotes of the article or of the headings above it,
else the act's entry into force -- per provision, which the currency changelog (P3-T) needs. A
heading's footnote counts for every article under it, so an article may be dated later than its
text last changed, never earlier. Repealed articles and Absätze (``…``) are skipped. The citation
code is the act's German abbreviation (``OR``, ``ZGB``, ``SchKG``), else ``SR <number>``; articles
of the ZGB's final title are ``ZGB SchlT Art. N``; the final and transitional provisions of single
amendments, which have no abbreviation of their own, are skipped. Ids name the provision
(``fedlex:220:art_97``), so one corpus holds one ``as_of``. Licence: Art. 5 URG (register row
``fedlex``).
"""

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import UTC, date, datetime

from lexhybrid.data.corpus.collectors.base import http_get, licence_flags, main, utc_now
from lexhybrid.data.schema import Document, Section

SPARQL = "https://fedlex.data.admin.ch/sparqlendpoint"
SPARQL_HEADERS = {"Accept": "application/sparql-results+json"}
SITE = "https://www.fedlex.admin.ch"
AKN = "{http://docs.oasis-open.org/legaldocml/ns/akn/3.0}"
FEDLEX = "{http://fedlex.admin.ch/}"
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
# Civil and commercial law first, then procedure, constitution, criminal law, data protection.
SEED_SR = (
    "210", "220", "272", "281.1", "101", "311.0", "312.0", "235.1", "291", "241", "221.229.1", "173.110",
)  # fmt: skip
_PREFIXES = """PREFIX jolux: <http://data.legilux.public.lu/resource/ontology/jolux#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""
_ARTICLE = re.compile(r"^(?:(disp_u\d+)/)?art_(\d+)(?:_([a-z]+))?$")
_CODE = re.compile(r"^[A-ZÄÖÜ][\wÄÖÜäöüß\-/]*$")
_ABSATZ_NUM = re.compile(r"\d+[a-z]*")
_ELI = re.compile(r"^https://fedlex\.data\.admin\.ch/(eli/cc/[^/]+/[^/]+)")
_MONTHS = {
    "jan": 1, "feb": 2, "mär": 3, "mar": 3, "apr": 4, "mai": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "okt": 10, "nov": 11, "dez": 12,
}  # fmt: skip
_IN_FORCE = re.compile(r"(?:in Kraft|Wirkung) seit\s+(\d{1,2})\.\s*([A-Za-zÄäÖöÜü]+)\.?\s+(\d{4})")
# Containers whose headings sit above an article (their footnotes date it).
_CONTAINERS = {"book", "part", "title", "chapter", "section", "level", "proviso", "transitional"}
# The containers that are the act's structure (Document.hierarchy); a "level" is a marginal note.
_STRUCTURE = {"book", "part", "title", "chapter", "section", "proviso", "transitional"}
# Block elements: each starts on a new line.
_BLOCKS = {"p", "listIntroduction", "listWrapUp", "item", "blockList", "tr", "table", "content"}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def render(elem: ET.Element) -> str:
    """Text of an Akoma Ntoso block without its footnotes: paragraphs, list introductions and list
    items on their own lines (an item keeps its number: ``1. wenn der Irrende ...``), a table row
    on one line with its cells space-separated."""
    parts: list[str] = []

    def walk(e: ET.Element, inline: bool) -> None:
        tag = _local(e.tag)
        if tag == "authorialNote":
            return
        block = not inline and tag in _BLOCKS
        if block:
            parts.append("\n")
        if tag == "br":
            parts.append(" " if inline else "\n")
        if e.text:
            parts.append(e.text)
        for child in e:
            # an item's number and text share a line, and so do a table row's cells
            walk(child, inline or tag == "tr" or (tag == "item" and _local(child.tag) in ("num", "p")))
            if child.tail:
                parts.append(child.tail)
        if tag in ("td", "th"):
            parts.append(" ")
        if block:
            parts.append("\n")

    walk(elem, False)
    lines = (" ".join(line.replace("\xa0", " ").split()) for line in "".join(parts).split("\n"))
    return "\n".join(line for line in lines if line)


def _inline(elem: ET.Element | None) -> str:
    return " ".join(render(elem).split()) if elem is not None else ""


def _note_dates(elem: ET.Element) -> list[str]:
    """ISO dates of every ``in Kraft seit`` / ``mit Wirkung seit`` in the footnotes under ``elem``."""
    out = []
    for note in elem.iter(f"{AKN}authorialNote"):
        for d, month, y in _IN_FORCE.findall(" ".join("".join(note.itertext()).split())):
            m = _MONTHS.get(month.lower()[:3])
            if m is not None:
                out.append(date(int(y), m, int(d)).isoformat())
    return out


def _is_text(text: str) -> bool:
    """False for a repealed unit, which Fedlex prints as ``…``."""
    return bool(text.replace("…", "").strip())


def act_meta(root: ET.Element) -> dict:
    """SR number, German title and abbreviation, entry into force and version date from ``meta``."""
    work = root.find(f".//{AKN}meta/{AKN}identification/{AKN}FRBRWork")
    dates = {e.get("name"): e.get("date") for e in work.findall(f"{AKN}FRBRdate")}
    name = next(
        (e for e in work.findall(f"{AKN}FRBRname") if e.get(XML_LANG) == "de"),
        None,
    )
    eli = _ELI.match(work.find(f"{AKN}FRBRuri").get("value"))
    return {
        "sr": work.find(f"{AKN}FRBRnumber").get("value"),
        "title": name.get("value", "") if name is not None else "",
        "short": (name.get("shortForm") or "").strip() if name is not None else "",
        "entry_into_force": dates.get("jolux:dateEntryInForce"),
        "applicability": dates.get("jolux:dateApplicability"),
        "eli": eli.group(1) if eli else None,
    }


def act_code(short: str, sr: str) -> str:
    """The citation code: the German abbreviation when it fits the grammar, else ``SR <number>``."""
    return short if _CODE.match(short) else f"SR {sr}"


def _numbered_ps(para: ET.Element) -> list[tuple[str, ET.Element]] | None:
    """A paragraph without a number whose ``p``s each open with a superscript number
    (``<p><sup>1</sup> Die rechtlichen ...``, the markup of older parts such as the ZGB's final
    title) holds one Absatz per ``p``; else None. Repealed stubs (``…``) do not count."""
    content = para.find(f"{AKN}content")
    ps = [
        c for c in (content if content is not None else para) if _local(c.tag) == "p" and _is_text(render(c))
    ]
    out = []
    for p in ps:
        k = _inline(p[0]) if len(p) and _local(p[0].tag) == "sup" and not (p.text or "").strip() else ""
        if not _ABSATZ_NUM.fullmatch(k):
            return None
        out.append((k, p))
    return out if len(out) > 1 else None


def article_sections(article: ET.Element, number: str) -> tuple[list[Section], list[str]]:
    """The article's Absätze as sections, and its body lines (each Absatz with its number, as
    printed: ``2 Für die Vollstreckung ...``). Repealed Absätze are left out."""
    units: list[tuple[str, str]] = []  # (Absatz number or "", text)
    paragraphs = list(article.iter(f"{AKN}paragraph"))
    for para in paragraphs:
        k = _inline(para.find(f"{AKN}num"))
        numbered = None if k else _numbered_ps(para)
        if numbered:
            units += [(n, render(p).removeprefix(n).strip()) for n, p in numbered]
        else:
            units.append((k, "\n".join(render(c) for c in para if _local(c.tag) != "num")))
    if not paragraphs:  # a one-Absatz article: the text sits in the article's own content
        units.append(("", "\n".join(render(c) for c in article if _local(c.tag) not in ("num", "heading"))))
    sections, lines = [], []
    for k, text in units:
        if not _is_text(text):
            continue
        label = f"Art. {number} Abs. {k}" if k else f"Art. {number}"
        sections.append(Section(label=label, absatz=int(k) if k.isdigit() else None, text=text))
        lines.append(f"{k} {text}" if k else text)
    return sections, lines


def parse_fedlex_xml(xml_bytes: bytes, retrieved_at: str | None = None) -> list[Document]:
    """Every article of one act's German consolidation (Akoma Ntoso) as a ``Document``."""
    root = ET.fromstring(xml_bytes)
    meta = act_meta(root)
    code = act_code(meta["short"], meta["sr"])
    first_word = (meta["title"].split() or [""])[0].lower()
    doc_type = "regulation" if first_word.endswith(("verordnung", "reglement")) else "statute"
    floor, ceiling = meta["entry_into_force"], meta["applicability"]
    flags = licence_flags("CH-URG-5")
    retrieved_at = retrieved_at or utc_now()
    parent = {child: p for p in root.iter() for child in p}
    docs = []
    for article in root.iter(f"{AKN}article"):
        eid = article.get("eId") or ""
        m = _ARTICLE.match(eid)
        if not m:
            continue  # "art_627_628": a repealed pair
        block, number = m.group(1), m.group(2) + (m.group(3) or "")
        ancestors, p = [], parent.get(article)
        while p is not None:
            ancestors.append(p)
            p = parent.get(p)
        qualifier = ""
        if block:  # final or transitional provisions: only the ZGB's final title has a citation code
            top = next(a for a in ancestors if a.get("eId") == block)
            if not _inline(top.find(f"{AKN}heading")).startswith("Schlusstitel"):
                continue
            qualifier = " SchlT"
        sections, lines = article_sections(article, number)
        if not sections:
            continue  # repealed
        marginal = [
            _inline(a.find(f"{AKN}heading"))
            for a in reversed(ancestors)
            if _local(a.tag) == "level" and a.get(f"{FEDLEX}role") == "marginal"
        ]
        title = _inline(article.find(f"{AKN}heading")) or " / ".join(t for t in marginal if t)
        hierarchy = [  # "Erste Abteilung: Allgemeine Bestimmungen", ..., outermost first
            " ".join(f"{_inline(a.find(f'{AKN}num'))} {_inline(a.find(f'{AKN}heading'))}".split())
            for a in reversed(ancestors)
            if _local(a.tag) in _STRUCTURE
        ]
        dates = _note_dates(article) + [
            d
            for a in ancestors
            if _local(a.tag) in _CONTAINERS
            for child in a
            if _local(child.tag) in ("num", "heading")
            for d in _note_dates(child)
        ]
        valid_from = max([d for d in [floor, *dates] if d], default=ceiling)
        if ceiling and valid_from and valid_from > ceiling:
            valid_from = ceiling  # a footnote cannot date a text after the version that prints it
        docs.append(
            Document(
                id=f"fedlex:{meta['sr']}:{eid.replace('/', ':')}",
                source="fedlex",
                jurisdiction="CH",
                doc_type=doc_type,
                text="\n".join([f"Art. {number} {title}".strip(), *lines]),
                citation_id=f"{code}{qualifier} Art. {number}",
                url=f"{SITE}/{meta['eli']}/de#{eid}" if meta["eli"] else "",
                valid_from=valid_from,
                retrieved_at=retrieved_at,
                sections=sections,
                hierarchy=[h for h in hierarchy if h],
                **flags,
            )
        )
    return docs


# ---------------------------------------------------------------------------------------------
# which version of which act
# ---------------------------------------------------------------------------------------------


def acts_query() -> str:
    """SPARQL: the SR number of every act in force."""
    return (
        _PREFIXES
        + """SELECT DISTINCT ?sr WHERE {
  ?act a jolux:ConsolidationAbstract ;
       jolux:inForceStatus <https://fedlex.data.admin.ch/vocabulary/enforcement-status/0> ;
       jolux:classifiedByTaxonomyEntry/skos:notation ?sr .
}"""
    )


def versions_query(as_of: str, sr: str) -> str:
    """SPARQL: the German XML consolidations of the act in force with SR number ``sr`` that apply
    on ``as_of``; the latest of them is the version to collect. (One act per query: an aggregate
    over every act at once returned wrong maxima from this endpoint.)"""
    return (
        _PREFIXES
        + f"""SELECT DISTINCT ?sr ?date ?file WHERE {{
  ?act a jolux:ConsolidationAbstract ;
       jolux:inForceStatus <https://fedlex.data.admin.ch/vocabulary/enforcement-status/0> ;
       jolux:classifiedByTaxonomyEntry/skos:notation ?sr .
  FILTER(str(?sr) = "{sr}")
  ?cons jolux:isMemberOf ?act ; jolux:dateApplicability ?date ; jolux:isRealizedBy ?expr .
  ?expr jolux:language <http://publications.europa.eu/resource/authority/language/DEU> ;
        jolux:isEmbodiedBy ?manif .
  ?manif jolux:userFormat <https://fedlex.data.admin.ch/vocabulary/user-format/xml> ;
         jolux:isExemplifiedBy ?file .
  FILTER(?date <= "{as_of}"^^xsd:date)
}}"""
    )


def latest_versions(bindings: list[dict]) -> dict[str, tuple[str, str]]:
    """``{sr: (date, file)}`` from SPARQL result bindings: per SR number the latest version
    (ties: the last file name, the most recent upload)."""
    best: dict[str, tuple[str, str]] = {}
    for b in bindings:
        sr, d, f = b["sr"]["value"], b["date"]["value"], b["file"]["value"]
        if sr not in best or (d, f) > best[sr]:
            best[sr] = (d, f)
    return best


class FedlexCollector:
    name = "fedlex"
    licence = "CH-URG-5"
    jurisdiction = "CH"

    def __init__(self, sr: tuple[str, ...] = SEED_SR, all_acts: bool = True, as_of: str | None = None):
        """``sr`` are collected first; with ``all_acts`` every other act in force follows."""
        self.sr, self.all_acts = sr, all_acts
        self.as_of = as_of or datetime.now(UTC).date().isoformat()

    def _sparql(self, query: str) -> list[dict]:
        response = http_get(SPARQL, params={"query": query}, headers=SPARQL_HEADERS, timeout=300)
        return response.json()["results"]["bindings"]

    def _numbers(self) -> Iterator[str]:
        """The seeds, then (lazily: only once the seeds are used up) every other act in force."""
        yield from self.sr
        if self.all_acts:
            listed = sorted({b["sr"]["value"] for b in self._sparql(acts_query())})
            yield from (sr for sr in listed if sr not in self.sr)

    def _acts(self) -> Iterator[tuple[str, str]]:
        """``(sr, file)`` of the version to collect for each act."""
        for sr in self._numbers():
            version = latest_versions(self._sparql(versions_query(self.as_of, sr))).get(sr)
            if version is not None:
                yield sr, version[1]

    def iter_documents(self, limit: int | None = None) -> Iterator[Document]:
        n = 0
        for _, file in self._acts():
            for doc in parse_fedlex_xml(http_get(file, timeout=300).content):
                yield doc
                n += 1
                if limit is not None and n >= limit:
                    return


if __name__ == "__main__":
    main(FedlexCollector())
