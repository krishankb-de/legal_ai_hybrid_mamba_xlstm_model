"""DIP, the Bundestag's documentation system: printed papers (Drucksachen) in full text (plan P3-M).

Source: the DIP API ``https://search.dip.bundestag.de/api/v1/drucksache-text`` (OpenAPI 1.5): per
request ten papers with their metadata (``dokumentnummer`` ``21/8109``, ``drucksachetyp``,
``herausgeber`` BT or BR, ``fundstelle.datum``, the PDF's URL) and ``text``, the text extracted from
the PDF, paged by ``cursor``. Every request carries an API key: ``DIP_API_KEY`` if set, else the
public key the DIP help page publishes (valid until the end of May 2027); a personal key is issued on
request (parlamentsdokumentation@bundestag.de). Not more than 25 concurrent requests (the API's
information sheet); this collector sends one at a time.

The extracted text keeps the PDF's line breaks, hyphenation and, in preliminary versions, a margin
watermark spliced into the text ("Vorabfassung – wird durch die lektorierte Fassung ersetzt.").
``paragraphs`` removes the watermark and the printer's boilerplate, rejoins hyphenated words and
lines, and splits paragraphs at short sentence-final lines and at the headings of a bill or report
(``A. Problem``, ``Artikel 1``, ``Begründung``, ``Zu Artikel 1 (...)``, ``I. ...``).

One ``Document`` per paper, cited as the terms of use prescribe: ``BT-Drs. 21/8109`` or
``BR-Drs. 506/26``; sections are the paragraphs, labelled by the heading above them (``Kopf`` before
the first; a ``Zu Nummer`` heading carries its ``Zu Artikel``). ``valid_from`` is the paper's date. By default the legislative types are collected, bills
first. Licence: official works (§ 5 UrhG); DIP's terms of use (27 February 2023) allow the API data
to be used and processed "umfassend in jeglicher Form", with the source ("Deutscher
Bundestag/Bundesrat – DIP") named on redistribution and, for commercial use, a note that the data
are free in DIP (register row ``dip``).
"""

import os
import re
from collections.abc import Iterator

from lexhybrid.data.corpus.collectors.base import cli, http_get, licence_flags, utc_now
from lexhybrid.data.schema import Document, Section

API = "https://search.dip.bundestag.de/api/v1"
# The public key of the DIP API help page (https://dip.bundestag.de/über-dip/hilfe/api), valid until
# the end of May 2027; set DIP_API_KEY to use a personal one.
PUBLIC_API_KEY = "R2BZaee.DjdCyihKZMf8AOjtScubP2EVydegzjmBIQ"
TYPES = ("Gesetzentwurf", "Beschlussempfehlung und Bericht")
SERIES = {"BT": "BT-Drs.", "BR": "BR-Drs."}

_WATERMARK = re.compile(
    r"\n?[ \t]*" + r"\s*".join("Vorabfassung") + r"\s*[–-]\s*" + r"\s*".join("wird")
    + r"\s+durch\s+die\s+lektorierte\s+Fassung\s+ersetzt\s*\.[ \t]*\n?"
)  # fmt: skip
_BOILERPLATE = re.compile(r"^(?:Vertrieb: Bundesanzeiger Verlag|Telefon \(02 21\)|ISSN \d{4}-\d{3}[\dX])")
_PAGE_NUMBER = re.compile(r"^[–-]\s*\d+\s*[–-]$")
_HEADING = re.compile(
    r"^(?:[A-H](?:\.\d+)?\.?\s+[A-ZÄÖÜ][^.;,]*"  # A. Problem, E.1 Erfüllungsaufwand für ...
    r"|[IVX]+\.\s+[A-ZÄÖÜ][^.;,]*"  # I. Zielsetzung und Notwendigkeit der Regelungen
    r"|Artikel \d+[a-z]?"
    r"|Zu (?:Artikel|Nummer|Buchstabe|Doppelbuchstabe|Absatz|§|den Nummern|den Artikeln)\b[^.;]*"
    r"|Begründung|Vorblatt|Stellungnahme des Bundesrates|Gegenäußerung der Bundesregierung|Anlage \d*)$"
)
_ITEM = re.compile(r"^(?:\d+\.|[a-z]\)|[a-z]{2}\)|[–•])\s")
_SENTENCE_END = re.compile(r"[.:;!?][)\"“”»]*$")  # "... gilt.", "... ersetzt.“" -- not "„für“"
_LIST_MARK = re.compile(r"^(?:[a-z]{1,4}|\d+)[.)]\s")  # "b. stellt ...", "ii. die ...", "2) ..."
_FOOTNOTE = re.compile(r"^\d{1,3} [A-ZÄÖÜ„\"]")  # "1 BGH, Urteil v. 28.07.2021, ..." at a page's foot
# A line ending in one of these abbreviations does not end a sentence ("... Steuererstattung bzw.").
_ABBREVIATION_END = re.compile(
    r"(?:\b(?:bzw|vgl|ggf|gem|Nr|Abs|Art|Buchst|Ziff|Rn|Rz|Anm|Bd|S|Dr|Prof|ca|Mio|Mrd|Tsd|etc|usw|z\. B|"
    r"d\. h|u\. a|i\. V\. m|i\. S\. d|a\. F|n\. F)\.)$"
)
_NOT_A_SPLIT = ("und ", "oder ", "bis ", "sowie ", "bzw. ", "beziehungsweise ")


def is_heading(line: str) -> bool:
    return len(line) <= 140 and bool(_HEADING.match(line))


def paragraphs(text: str) -> list[tuple[bool, str]]:
    """The paper's paragraphs as ``(is_heading, text)``, cleaned of the PDF's layout (module doc)."""
    text = _WATERMARK.sub("\n", text.replace("\xa0", " ").replace("\r", ""))
    lines = [" ".join(line.split()) for line in text.split("\n")]
    lines = [ln for ln in lines if not _BOILERPLATE.match(ln) and not _PAGE_NUMBER.match(ln)]
    widths = sorted(len(ln) for ln in lines if ln)
    width = max(60, widths[int(0.9 * (len(widths) - 1))]) if widths else 60
    out: list[tuple[bool, str]] = []
    current = ""

    def flush():
        nonlocal current
        if current:
            out.append((False, current))
        current = ""

    for line in lines:
        if not line:
            flush()
            continue
        if is_heading(line):
            flush()
            out.append((True, line))
            continue
        if _ITEM.match(line):
            flush()
        if not current:
            current = line
        elif re.search(r"[a-zäöüß]-$", current) and line[0].islower() and not line.startswith(_NOT_A_SPLIT):
            current = current[:-1] + line  # "Steu-" + "ererstattungen"
        else:
            current += " " + line
        if _SENTENCE_END.search(line) and len(line) < 0.85 * width and not _ABBREVIATION_END.search(line):
            flush()  # a short sentence-final line ends its paragraph
    flush()
    return _rejoin_across_footnotes(out)


def _rejoin_across_footnotes(blocks: list[tuple[bool, str]]) -> list[tuple[bool, str]]:
    """A page's footnotes interrupt the paragraph that runs on to the next page ("... der Siche-",
    "1 BGH, Urteil ...", "rung der ..."): join the two halves and put the footnotes after them."""
    out: list[tuple[bool, str]] = []
    for heading, text in blocks:
        if not heading and text[:1].islower() and not _LIST_MARK.match(text) and out:
            k = len(out) - 1
            while k >= 0 and not out[k][0] and _FOOTNOTE.match(out[k][1]):
                k -= 1
            if k >= 0 and not out[k][0] and not _SENTENCE_END.search(out[k][1]):
                head = out[k][1]
                joined = head[:-1] + text if re.search(r"[a-zäöüß]-$", head) else head + " " + text
                out[k : k + 1] = [(False, joined)]
                continue
        out.append((heading, text))
    return out


def api_key() -> str:
    return os.environ.get("DIP_API_KEY") or PUBLIC_API_KEY


def parse_dip_drucksache(doc: dict, retrieved_at: str | None = None) -> Document | None:
    """One printed paper from a ``drucksache-text`` API record; None without text."""
    blocks = paragraphs(doc.get("text") or "")
    if not any(not heading for heading, _ in blocks):
        return None
    sections, label, article = [], "Kopf", None
    for heading, text in blocks:
        if not heading:
            sections.append(Section(label=label, text=text))
        elif text.startswith("Zu Artikel"):
            label, article = text, text.split(" (")[0]
        elif text.startswith("Zu ") and article:
            label = f"{article} / {text}"  # "Zu Artikel 1 / Zu Nummer 1 (§ 261 Absatz 1 StGB-E)"
        else:
            label, article = text, None
    fundstelle = doc.get("fundstelle") or {}
    series = SERIES.get(doc.get("herausgeber") or fundstelle.get("herausgeber"), "BT-Drs.")
    return Document(
        id=f"dip:{doc['id']}",
        source="dip",
        jurisdiction="DE",
        doc_type="parliament",
        text="\n".join(text for _, text in blocks),
        citation_id=f"{series} {doc['dokumentnummer']}",
        url=fundstelle.get("pdf_url", ""),
        valid_from=fundstelle.get("datum") or doc.get("datum") or None,
        retrieved_at=retrieved_at or utc_now(),
        sections=sections,
        **licence_flags("DE-UrhG-5"),
    )


class DIPCollector:
    name = "dip"
    licence = "DE-UrhG-5"
    jurisdiction = "DE"

    def __init__(self, types: tuple[str, ...] = TYPES, zuordnung: str | None = None):
        """``types``: the Drucksache types, collected in turn, newest first; ``zuordnung``: ``BT``
        or ``BR`` to keep one chamber (default both)."""
        self.types, self.zuordnung = types, zuordnung

    def pages(self, typ: str) -> Iterator[list[dict]]:
        params = {"f.drucksachetyp": typ, "format": "json"}
        if self.zuordnung:
            params["f.zuordnung"] = self.zuordnung
        headers = {"Authorization": f"ApiKey {api_key()}"}
        cursor = None
        while True:
            page = http_get(
                f"{API}/drucksache-text", params={**params, **({"cursor": cursor} if cursor else {})},
                headers=headers, timeout=180,
            ).json()  # fmt: skip
            docs = page.get("documents") or []
            if not docs:
                return
            yield docs
            if page.get("cursor") in (None, cursor):
                return
            cursor = page["cursor"]

    def iter_documents(self, limit: int | None = None) -> Iterator[Document]:
        n = 0
        for typ in self.types:
            for docs in self.pages(typ):
                for record in docs:
                    doc = parse_dip_drucksache(record)
                    if doc is None:
                        continue
                    yield doc
                    n += 1
                    if limit is not None and n >= limit:
                        return


if __name__ == "__main__":
    raise SystemExit(cli(DIPCollector()))
