"""RIS, the Austrian federal legal information system: consolidated federal law and Supreme Court
decisions (plan P3-J).

Source: the OGD API v2.6 at ``https://data.bka.gv.at/ris/api/v2.6/`` (CC BY 4.0, no registration).

* ``Bundesrecht?Applikation=BrKons`` lists consolidated norms, one record per **version** of a
  provision: ``Abkuerzung`` (``ABGB``), ``ArtikelParagraphAnlage`` (``§ 688``),
  ``Inkrafttretensdatum`` / ``Ausserkrafttretensdatum`` -- the validity window, which the currency
  changelog (P3-T) needs -- and links to the version's XML.
* ``Judikatur?Applikation=Justiz`` lists decisions; with ``Dokumenttyp.SucheInEntscheidungstexten``
  the full texts (``JJT_...``): ``Geschaeftszahl``, ``Entscheidungsdatum``, ECLI and the XML.

Both XMLs are ``risdok`` documents: metadata paragraphs, then the text as ``absatz`` elements whose
``ct`` attribute names the part (``text`` for a norm; ``kopf``, ``spruch``, ``text``,
``rechtlichebeurteilung`` for a decision). One ``Document`` per norm version (``ABGB §688``, sections
= Absätze) and per decision (``OGH 7 Ob 143/26v``, sections = the parts' paragraphs).
"""

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator

from lexhybrid.data.corpus.collectors.base import http_get, licence_flags, main, utc_now
from lexhybrid.data.schema import Document, Section

API = "https://data.bka.gv.at/ris/api/v2.6"
NS = "{http://www.bka.gv.at}"
SEED_CODES = ("ABGB", "KSchG", "MRG", "UGB", "StGB", "ZPO", "B-VG", "DSG", "ASVG", "EO")
DECISION_PARTS = {
    "kopf": "Kopf",
    "spruch": "Spruch",
    "text": "Begründung",
    "begruendung": "Begründung",
    "rechtlichebeurteilung": "Rechtliche Beurteilung",
}
_ABSATZ = re.compile(r"^\((\d+[a-z]?)\)\s*")
_GLDSYM = re.compile(r"^(§|Art\.?)\s*\d+[a-z]*\.\s*")
_GZ = re.compile(r"^(\d+)\s*([A-Za-z]+)\s*(\d+/\d+[a-z]?)$")


def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _xml_url(ref: dict) -> str | None:
    urls = ref["Data"].get("Dokumentliste", {}).get("ContentReference", {})
    for content in _as_list(urls):
        for u in _as_list(content.get("Urls", {}).get("ContentUrl")):
            if u.get("DataType") == "Xml":
                return u["Url"]
    return None


def _clean(elem: ET.Element) -> str:
    return " ".join("".join(elem.itertext()).replace("\xa0", " ").split())


def text_paragraphs(xml_bytes: bytes) -> list[tuple[str, str]]:
    """``(ct, text)`` of every ``absatz``/``ueberschrift`` after the metadata, in document order."""
    root = ET.fromstring(xml_bytes)
    out = []
    for elem in root.iter():
        tag = elem.tag.replace(NS, "")
        ct = elem.get("ct")
        if tag in ("absatz", "ueberschrift") and ct:
            text = _clean(elem)
            if text:
                out.append((ct, text))
    return out


def norm_citation(code: str, apa: str) -> str | None:
    """``ABGB`` + ``§ 688`` -> ``ABGB §688``; ``Art. 5`` -> ``B-VG Art. 5``; annexes -> None."""
    code = code.strip().replace(" ", "-")
    m = re.match(r"^(§|Art\.?)\s*(\d+[a-z]*)$", " ".join(apa.split()))
    if not m:
        return None
    return f"{code} §{m.group(2)}" if m.group(1) == "§" else f"{code} Art. {m.group(2)}"


def parse_ris_norm(ref: dict, xml_bytes: bytes, retrieved_at: str | None = None) -> Document | None:
    """One consolidated norm version from its API reference and its risdok XML."""
    md = ref["Data"]["Metadaten"]
    brk = md["Bundesrecht"]["BrKons"]
    citation = norm_citation(brk.get("Abkuerzung") or "", brk.get("ArtikelParagraphAnlage") or "")
    if citation is None:
        return None
    heading, sections = "", []
    for ct, text in text_paragraphs(xml_bytes):
        if ct != "text":
            continue
        if not sections and not _ABSATZ.match(text) and not re.match(r"^(§|Art)", text):
            heading = text  # the provision's title line
            continue
        m = _ABSATZ.match(_GLDSYM.sub("", text))  # "§ 1295. (1) Jedermann ..." opens Absatz 1
        label = f"{brk['ArtikelParagraphAnlage']} Abs. {m.group(1)}" if m else brk["ArtikelParagraphAnlage"]
        if sections and not m:
            sections[-1].text += "\n" + text  # a list item or sentence continues its Absatz
            continue
        number = m.group(1) if m else None
        sections.append(
            Section(label=label, absatz=int(number) if number and number.isdigit() else None, text=text)
        )
    if not sections:
        return None
    valid_from, valid_to = brk.get("Inkrafttretensdatum") or None, brk.get("Ausserkrafttretensdatum") or None
    if valid_from and valid_to and valid_to < valid_from:
        return None  # a version replaced before it came into force never applied
    typ = brk.get("Typ", "")
    return Document(
        id=f"ris:{md['Technisch']['ID']}",
        source="ris",
        jurisdiction="AT",
        doc_type="regulation" if typ.startswith("V") else "statute",
        text=(heading + "\n" if heading else "") + "\n".join(s.text for s in sections),
        citation_id=citation,
        url=md["Allgemein"].get("DokumentUrl", ""),
        valid_from=valid_from,
        valid_to=valid_to,
        retrieved_at=retrieved_at or utc_now(),
        sections=sections,
        **licence_flags("CC-BY-4.0"),
    )


def docket(geschaeftszahl: str) -> str:
    """``7Ob143/26v`` -> ``7 Ob 143/26v`` (the spaced form courts print)."""
    first = geschaeftszahl.split(";")[0].strip()
    m = _GZ.match(first.replace(" ", ""))
    return f"{m.group(1)} {m.group(2)} {m.group(3)}" if m else first


def parse_ris_decision(ref: dict, xml_bytes: bytes, retrieved_at: str | None = None) -> Document | None:
    """One full-text decision (``JJT_...``) from its API reference and its risdok XML."""
    md = ref["Data"]["Metadaten"]
    ju = md["Judikatur"]
    court = (ju.get("Justiz") or {}).get("Gericht") or md["Technisch"].get("Organ") or "OGH"
    gz = docket(" ".join(_as_list((ju.get("Geschaeftszahl") or {}).get("item"))))
    sections, lines, last = [], [], None
    for ct, text in text_paragraphs(xml_bytes):
        part = DECISION_PARTS.get(ct)
        if part is None:
            continue
        if part != last:
            lines.append(part)
            last = part
        sections.append(Section(label=part, text=text, part=part))
        lines.append(text)
    if not sections:
        return None
    return Document(
        id=f"ris:{md['Technisch']['ID']}",
        source="ris",
        jurisdiction="AT",
        doc_type="decision",
        text="\n".join(lines),
        citation_id=f"{court} {gz}",
        url=md["Allgemein"].get("DokumentUrl", ""),
        valid_from=ju.get("Entscheidungsdatum") or None,
        retrieved_at=retrieved_at or utc_now(),
        sections=sections,
        **licence_flags("CC-BY-4.0"),
    )


def _refs(page: dict) -> list[dict]:
    return _as_list(page.get("OgdSearchResult", {}).get("OgdDocumentResults", {}).get("OgdDocumentReference"))


class RISCollector:
    name = "ris"
    licence = "CC-BY-4.0"
    jurisdiction = "AT"

    def __init__(self, codes: tuple[str, ...] = SEED_CODES, courts: tuple[str, ...] = ("OGH",)):
        self.codes, self.courts = codes, courts

    def norms(self) -> Iterator[Document]:
        for code in self.codes:
            page_no = 1
            while True:
                params = {
                    "Applikation": "BrKons",
                    "Titel": code,
                    "DokumenteProSeite": "Fifty",
                    "Seitennummer": str(page_no),
                }
                refs = _refs(http_get(f"{API}/Bundesrecht", params=params).json())
                if not refs:
                    break
                for ref in refs:
                    brk = ref["Data"]["Metadaten"].get("Bundesrecht", {}).get("BrKons", {})
                    url = _xml_url(ref)
                    if brk.get("Abkuerzung") != code or url is None:
                        continue
                    doc = parse_ris_norm(ref, http_get(url).content)
                    if doc is not None:
                        yield doc
                page_no += 1

    def decisions(self) -> Iterator[Document]:
        for court in self.courts:
            page_no = 1
            while True:
                params = {
                    "Applikation": "Justiz",
                    "Gericht": court,
                    "DokumenteProSeite": "Fifty",
                    "Seitennummer": str(page_no),
                    "Dokumenttyp.SucheInRechtssaetzen": "false",
                    "Dokumenttyp.SucheInEntscheidungstexten": "true",
                }
                refs = _refs(http_get(f"{API}/Judikatur", params=params).json())
                if not refs:
                    break
                for ref in refs:
                    url = _xml_url(ref)
                    if url is None:
                        continue
                    doc = parse_ris_decision(ref, http_get(url).content)
                    if doc is not None:
                        yield doc
                page_no += 1

    def iter_documents(self, limit: int | None = None) -> Iterator[Document]:
        """Norms and decisions, half the limit each (all of both without a limit)."""
        if limit is None:
            yield from self.norms()
            yield from self.decisions()
            return
        for source, share in ((self.norms(), (limit + 1) // 2), (self.decisions(), limit // 2)):
            for n, doc in enumerate(source, start=1):
                yield doc
                if n >= share:
                    break


if __name__ == "__main__":
    main(RISCollector())
