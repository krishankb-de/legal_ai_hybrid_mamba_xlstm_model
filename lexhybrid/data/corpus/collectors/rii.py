"""Rechtsprechung im Internet: decisions of the German federal courts (plan P3-G).

Source: ``https://www.rechtsprechung-im-internet.de/rii-toc.xml`` (~23 MB) lists every published
decision (court, date, docket, zip link). Each zip holds one XML ``<dokument>``: ``gertyp`` (BGH,
BVerwG, BFH, BAG, BSG, BVerfG, BPatG), ``spruchkoerper``, ``entsch-datum`` (YYYYMMDD),
``aktenzeichen``, ``doktyp`` (Urteil, Beschluss), ``ecli``, the cited ``norm``s, and the parts
``leitsatz``, ``tenor``, ``tatbestand``, ``entscheidungsgruende``/``gruende``, ``abwmeinung``. Every
part is a sequence of ``<dl class="RspDL">`` entries: a ``<dt>`` that may carry the Randnummer as
``<a name="rd_N">N</a>`` and a ``<dd>`` with the paragraph.

One ``Document`` per decision; each non-empty entry is a ``Section`` labelled ``Rn. N`` when it has
a Randnummer, otherwise by its part (``Tenor``, ``Leitsatz``). ``valid_from`` is the decision date.
The published decisions are already anonymised; the LER scrub (P3-R) still runs over them.
Licence: official work under § 5 UrhG (register row ``rii``).
"""

import io
import re
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterator

from lexhybrid.data.corpus.collectors.base import cli, http_get, licence_flags, utc_now
from lexhybrid.data.schema import Document, Section

TOC_URL = "https://www.rechtsprechung-im-internet.de/rii-toc.xml"
PARTS = (
    ("leitsatz", "Leitsatz"),
    ("tenor", "Tenor"),
    ("tatbestand", "Tatbestand"),
    ("entscheidungsgruende", "Entscheidungsgründe"),
    ("gruende", "Gründe"),
    ("abwmeinung", "Abweichende Meinung"),
)
_RD = re.compile(r"rd_(\d+)")


def _text(elem: ET.Element) -> str:
    """Paragraph text: ``<p>``s become lines, ``<br/>`` breaks, whitespace normalised."""
    parts: list[str] = []

    def walk(e: ET.Element) -> None:
        if e.tag == "br":
            parts.append("\n")
        if e.text:
            parts.append(e.text)
        for child in e:
            walk(child)
            if child.tail:
                parts.append(child.tail)
        if e.tag in ("p", "div", "tr"):
            parts.append("\n")
        elif e.tag in ("td", "th"):
            parts.append(" ")

    walk(elem)
    lines = (" ".join(line.split()) for line in "".join(parts).split("\n"))
    return "\n".join(line for line in lines if line)


def _iso(yyyymmdd: str) -> str | None:
    d = (yyyymmdd or "").strip()
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}" if re.fullmatch(r"\d{8}", d) else None


def parse_rii_xml(xml_bytes: bytes, url: str = "", retrieved_at: str | None = None) -> Document | None:
    """One decision as a ``Document`` (None if it has no text)."""
    root = ET.fromstring(xml_bytes)
    court = (root.findtext("gertyp") or "").strip()
    docket = " ".join((root.findtext("aktenzeichen") or "").split())
    doknr = (root.findtext("doknr") or "").strip()
    sections: list[Section] = []
    lines: list[str] = []
    for tag, name in PARTS:
        part = root.find(tag)
        if part is None:
            continue
        first = True
        for dl in part.iter("dl"):
            dd = dl.find("dd")
            text = _text(dd) if dd is not None else ""
            if not text:
                continue
            anchor = dl.find("dt/a")
            rd = _RD.fullmatch(anchor.get("name", "")) if anchor is not None else None
            number = int(rd.group(1)) if rd else None
            sections.append(Section(label=f"Rn. {number}" if number else name, randnummer=number, text=text))
            if first:
                lines.append(name)
                first = False
            lines.append(f"{number} {text}" if number else text)
    if not sections or not court or not docket:
        return None
    return Document(
        id=f"rii:{doknr}",
        source="rii",
        jurisdiction="DE",
        doc_type="decision",
        text="\n".join(lines),
        citation_id=f"{court} {docket}",
        url=(root.findtext("identifier") or url).strip(),
        valid_from=_iso(root.findtext("entsch-datum")),
        retrieved_at=retrieved_at or utc_now(),
        sections=sections,
        **licence_flags("DE-UrhG-5"),
    )


def parse_toc(xml_bytes: bytes) -> list[dict]:
    """Every ``<item>`` of ``rii-toc.xml`` as a dict (court, date, docket, link)."""
    out = []
    for item in ET.fromstring(xml_bytes).findall("item"):
        link = (item.findtext("link") or "").strip().replace("http://", "https://", 1)
        if link.endswith(".zip"):
            out.append(
                {
                    "court": (item.findtext("gericht") or "").strip(),
                    "date": (item.findtext("entsch-datum") or "").strip(),
                    "docket": (item.findtext("aktenzeichen") or "").strip(),
                    "link": link,
                }
            )
    return out


def read_zip_xml(zip_bytes: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        return zf.read(next(n for n in zf.namelist() if n.endswith(".xml")))


class RIICollector:
    name = "rii"
    licence = "DE-UrhG-5"
    jurisdiction = "DE"

    def __init__(self, newest_first: bool = True):
        self.newest_first = newest_first

    def iter_documents(self, limit: int | None = None) -> Iterator[Document]:
        items = parse_toc(http_get(TOC_URL, timeout=180).content)
        if self.newest_first:
            items.sort(key=lambda it: it["date"], reverse=True)
        n = 0
        for item in items:
            doc = parse_rii_xml(read_zip_xml(http_get(item["link"]).content), url=item["link"])
            if doc is None:
                continue
            yield doc
            n += 1
            if limit is not None and n >= limit:
                return


if __name__ == "__main__":
    raise SystemExit(cli(RIICollector()))
