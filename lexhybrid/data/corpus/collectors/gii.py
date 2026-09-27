"""Gesetze im Internet: federal statutes and ordinances of Germany (plan P3-F).

Source: ``https://www.gesetze-im-internet.de/gii-toc.xml`` lists every law with a link to its
``xml.zip``; the zip holds one XML file in the gii-norm format: ``<dokumente>`` of ``<norm>``s.
The first norm is the law's header (``jurabk``/``amtabk``, ``ausfertigung-datum``, the ``Stand``
comment ``Zuletzt geändert durch Art. 3 G v. 10.12.2025 I Nr. 320``); the others are the table of
contents, structural headings (``gliederungseinheit``) and the provisions (``enbez`` ``§ 573`` or
``Art 5``) with their text in ``textdaten/text/Content/P``, one ``<P>`` per Absatz.

One ``Document`` per provision (``BGB §573``); its sections are the Absätze. ``valid_from`` is the
date of the last amendment named in the law's ``Stand`` comment -- the version this text is --
falling back to the date of the law's issue. Repealed provisions (``(weggefallen)``) and empty
headings are skipped. Licence: official work under § 5 UrhG (register row ``gii``).
"""

import io
import re
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterator
from datetime import date

from lexhybrid.data.corpus.collectors.base import cli, http_get, licence_flags, utc_now
from lexhybrid.data.schema import Document, Section

TOC_URL = "https://www.gesetze-im-internet.de/gii-toc.xml"
BASE_URL = "https://www.gesetze-im-internet.de"
_ABSATZ = re.compile(r"^\((\d+[a-z]?)\)\s*")
_DATE = re.compile(r"v\.\s*(\d{1,2})\.(\d{1,2})\.(\d{4})")
_ENBEZ = re.compile(r"^(§|Art\.?)\s*(\d+[a-z]*)$")
# Inline block elements: a line break after each, so lists and tables stay readable.
_BREAK_AFTER = {"DD", "LA", "row", "Title", "Ident"}


def render(elem: ET.Element) -> str:
    """Text of a gii-norm content element: list items on their own lines, whitespace normalised."""
    parts: list[str] = []

    def walk(e: ET.Element) -> None:
        if e.tag in ("BR", "DL"):
            parts.append("\n")  # a list starts on its own line, even mid-sentence
        if e.text:
            parts.append(e.text)
        for child in e:
            walk(child)
            if child.tail:
                parts.append(child.tail)
        if e.tag in ("DT", "entry"):
            parts.append(" ")
        elif e.tag in _BREAK_AFTER:
            parts.append("\n")

    walk(elem)
    lines = (" ".join(line.split()) for line in "".join(parts).split("\n"))
    return "\n".join(line for line in lines if line)


def stand_date(header: ET.Element) -> str | None:
    """The date of the last amendment in the header's ``Stand`` comment, else the issue date."""
    md = header.find("metadaten")
    for standangabe in md.findall("standangabe"):
        if (standangabe.findtext("standtyp") or "").strip() == "Stand":
            dates = _DATE.findall(standangabe.findtext("standkommentar") or "")
            if dates:
                d, m, y = dates[-1]
                return date(int(y), int(m), int(d)).isoformat()
    issued = (md.findtext("ausfertigung-datum") or "").strip()
    return issued or None


def parse_gii_xml(xml_bytes: bytes, slug: str, retrieved_at: str | None = None) -> list[Document]:
    """Every provision of one law as a ``Document``.

    Args:
        xml_bytes: the gii-norm XML of one law (the file inside ``xml.zip``).
        slug: the law's path segment on the site (``bgb``), used in ids and URLs.
    """
    root = ET.fromstring(xml_bytes)
    norms = root.findall("norm")
    if not norms:
        return []
    header = norms[0]
    hmd = header.find("metadaten")
    code = (hmd.findtext("amtabk") or hmd.findtext("jurabk") or slug).strip()
    title = (hmd.findtext("langue") or "").strip()
    doc_type = "regulation" if "verordnung" in title.lower() else "statute"
    valid_from = stand_date(header)
    flags = licence_flags("DE-UrhG-5")
    retrieved_at = retrieved_at or utc_now()
    docs = []
    for norm in norms[1:]:
        md = norm.find("metadaten")
        enbez = " ".join((md.findtext("enbez") or "").split())
        m = _ENBEZ.match(enbez)
        content = norm.find("textdaten/text/Content")
        if not m or content is None:
            continue  # table of contents, structural headings, annexes
        heading = " ".join((md.findtext("titel") or "").split())
        if "weggefallen" in heading.lower():
            continue
        sections = []
        for p in content.findall("P"):
            text = render(p)
            if not text:
                continue
            absatz = _ABSATZ.match(text)
            label = f"{enbez} Abs. {absatz.group(1)}" if absatz else enbez
            number = absatz.group(1) if absatz else None
            sections.append(
                Section(label=label, absatz=int(number) if number and number.isdigit() else None, text=text)
            )
        if not sections or all(s.text in ("-", "(weggefallen)") for s in sections):
            continue
        symbol, number = m.groups()
        citation = f"{code} §{number}" if symbol == "§" else f"{code} Art. {number}"
        head = f"{enbez} {heading}".strip()
        docs.append(
            Document(
                id=f"gii:{slug}:{'art' if symbol != '§' else ''}{number}",
                source="gii",
                jurisdiction="DE",
                doc_type=doc_type,
                text=head + "\n" + "\n".join(s.text for s in sections),
                citation_id=citation,
                url=f"{BASE_URL}/{slug}/{'__' if symbol == '§' else 'art_'}{number}.html",
                valid_from=valid_from,
                retrieved_at=retrieved_at,
                sections=sections,
                **flags,
            )
        )
    return docs


def parse_toc(xml_bytes: bytes) -> list[tuple[str, str]]:
    """``(slug, xml.zip URL)`` for every law in ``gii-toc.xml``."""
    out = []
    for item in ET.fromstring(xml_bytes).findall("item"):
        link = (item.findtext("link") or "").strip().replace("http://", "https://", 1)
        m = re.search(r"gesetze-im-internet\.de/([^/]+)/xml\.zip$", link)
        if m:
            out.append((m.group(1), link))
    return out


def read_zip_xml(zip_bytes: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".xml"))
        return zf.read(name)


class GIICollector:
    name = "gii"
    licence = "DE-UrhG-5"
    jurisdiction = "DE"

    def __init__(self, first: tuple[str, ...] = ("bgb", "stgb", "zpo", "gg", "hgb")):
        self.first = first  # core codes first, then the rest of the table of contents

    def iter_documents(self, limit: int | None = None) -> Iterator[Document]:
        toc = parse_toc(http_get(TOC_URL).content)
        ordered = sorted(
            toc,
            key=lambda item: (
                item[0] not in self.first,
                self.first.index(item[0]) if item[0] in self.first else 0,
            ),
        )
        n = 0
        for slug, url in ordered:
            for doc in parse_gii_xml(read_zip_xml(http_get(url).content), slug):
                yield doc
                n += 1
                if limit is not None and n >= limit:
                    return


if __name__ == "__main__":
    raise SystemExit(cli(GIICollector()))
