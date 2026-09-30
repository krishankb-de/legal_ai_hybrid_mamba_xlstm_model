"""Swiss Federal Supreme Court (Bundesgericht) decisions in German, via entscheidsuche.ch (plan P3-L).

Source: entscheidsuche.ch republishes every decision the court puts online (``CH_BGer``: 178,635
since 2000, 109,262 of them in German, on 2026-09-27), one HTML file per decision under
``https://entscheidsuche.ch/docs/CH_BGer/``. Its search API (``_searchV2.php``, an Elasticsearch
query body by POST) lists them newest first with the metadata the parser needs: ``id``, ``date``,
``reference`` (the docket, ``4A_102/2026``), ``attachment.language`` and ``attachment.content_url``.
The directory listing (116 MB) and the daily job reports (58 MB each) are never fetched; the
block list (``Blockliste.json``, decisions withdrawn from publication) is, and its entries are skipped.
The published terms (the site's "Daten" page and its API documentation of 3.7.2022) allow any use
of the API and the decisions, ask for moderate load, entscheidsuche.ch named as the source, and a
voluntary donation for commercial use (register row ``bger``).

The HTML is the court's own layout, one ``<div class="para">`` per line: the heading (court, docket,
``Urteil vom ...``, Besetzung, Verfahrensbeteiligte -- parties already anonymised as ``A.________``
-- and Gegenstand), ``Sachverhalt:`` with lettered parts, ``Erwägungen:`` with numbered
considerations (``3.1.``), ``Demnach erkennt das Bundesgericht:`` with the numbered operative part,
and the closing (``Lausanne, 17. August 2026``, signatures). Orders (Verfügungen) and older
decisions word the headings differently (``Das Eidg. Versicherungsgericht zieht in Erwägung:``,
``in Erwägung, dass ...``, ``verfügt die Präsidentin:``, ``... im Verfahren nach Art. 36a OG
erkannt:``); ``REASONS`` and ``OPERATIVE`` hold the forms seen in 230 decisions from 2000 to 2026.
Since about 2007 the numbers are bold
(``<b>3.1.</b>``), which tells a consideration from a numbered quotation inside it; older decisions
have no markup, so a number opens a consideration there only when it can follow the previous one
(``2.1`` after ``2``, ``3`` after ``2.4``).

One ``Document`` per decision (``BGer 4A_102/2026``); sections are the heading lines (``Kopf``), the
facts (``Sachverhalt A``), the considerations (``E. 3.1``, the unit Swiss decisions are cited by),
the operative part (``Dispositiv Ziff. 1``) and the closing (``Schluss``). ``valid_from`` is the
decision date. Licence: decisions are unprotected (Art. 5 Abs. 1 lit. c URG; register row
``bger``, whose note records that entscheidsuche.ch's own terms are confirmed before P4).
"""

import re
from collections.abc import Iterator
from html.parser import HTMLParser

from lexhybrid.data.corpus.collectors.base import http_get, http_post, licence_flags, main, utc_now
from lexhybrid.data.schema import Document, Section

SEARCH = "https://entscheidsuche.ch/_searchV2.php"
# Decisions withdrawn from publication (by file name, per collection); never collected.
BLOCKLIST = "https://entscheidsuche.ch/docs/Blockliste.json"
# Court names printed above the docket, in the four national languages (and the former Federal
# Insurance Court's): boilerplate, not text.
_COURT_LINES = {
    "Bundesgericht", "Tribunal fédéral", "Tribunale federale", "Tribunal federal",
    "Eidgenössisches Versicherungsgericht", "Tribunal fédéral des assurances",
    "Tribunale federale delle assicurazioni", "Tribunal federal d'assicuranzas",
}  # fmt: skip
_FACTS = re.compile(r"^(?:Sachverhalt|Tatsachen|Prozessgeschichte)\s*:?$")
# "Erwägungen:", "Aus den Erwägungen:", "Das Bundesgericht zieht in Erwägung:", "Der Präsident hat
# in Erwägung," and an order's "in Erwägung, dass ..." -- not "Das Obergericht zog in Erwägung, ...".
REASONS = re.compile(
    r"^(?:(?:Aus den )?Erwägung(?:en)?\s*:?|in Erwägung\b.*|.{0,60}\b(?:zieht|hat|haben) in Erwägung\s*[,:]?)$",
    re.IGNORECASE,
)
# The operative part's heading names the deciding body ("Demnach erkennt das Bundesgericht:",
# "verfügt die Präsidentin im Verfahren nach Art. 32 Abs. 2 BGG:", "Das Eidg. Versicherungsgericht
# erkennt:", "Demnach erkennt das präsidierende Mitglied:") or ends the older formula ("in
# Erwägung, ... im Verfahren nach Art. 36a OG erkannt:", "Demnach wird erkannt:").
_BODY = (
    r"(?:das|die|der)\s(?:[\w.]+\s){0,3}?"
    r"(?:Bundesgericht|Versicherungsgericht|Gericht|(?:Vize)?[Pp]räsident(?:in)?|Mitglied|Einzelrichter(?:in)?"
    r"|Instruktionsrichter(?:in)?|Richter(?:in)?|Präsidium|Abteilung|Kammer)"
)
_VERB = r"(?:erkennt|verfügt|beschliesst|entscheidet)"
_DECIDES = (
    rf"(?:(?:{_VERB}\s+{_BODY}|{_BODY}\s+{_VERB})(?:\s+im Verfahren nach .{{0,60}})?"
    r"|(?:und\s+|wird\s+|im Verfahren nach .{0,60}\s)?(?:erkannt|verfügt|beschlossen))"
)
# After "Demnach" the verb suffices: the heading may run on to the next line ("Demnach erkennt das
# Bundesgericht im" / "Verfahren nach Art. 36a OG:") and name any body ("das Präsidium").
OPERATIVE = re.compile(
    rf"^(?:Demnach\s+(?:{_VERB}|wird)\b.{{0,80}}|{_DECIDES}\s*:)$",
    re.IGNORECASE,
)
_CLOSING = re.compile(r"^[A-ZÄÖÜ][\w. -]*, (?:den )?\d{1,2}\. [A-Za-zäöü]+ \d{4}$")
_NUMBER = re.compile(r"^(\d+(?:\.\d+)*)\.?(?=\s|$)")
_LETTER = re.compile(r"^([A-Z](?:\.[a-z]+)?)\.?$")
_ITEM = re.compile(r"^(\d+)\.(?:\s*-\s*|\s+|$)")  # "1.", "1. Text", "1.-Text"


class _Paras(HTMLParser):
    """The ``<div class="para">`` blocks as ``(bold, text)``: ``bold`` is the text of a ``<b>`` that
    opens the block (``3.1.``, ``Erwägungen:``), else None. Source newlines and ``<br>`` are spaces."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.paras: list[tuple[str | None, str]] = []
        self._text: list[str] | None = None  # None outside a para
        self._bold: list[str] | None = None
        self._lead: str | None = None
        self._nested = 0

    def handle_starttag(self, tag, attrs):
        if self._text is None:
            if tag == "div" and "para" in (dict(attrs).get("class") or "").split():
                self._text, self._bold, self._lead, self._nested = [], None, None, 0
            return
        if tag == "div":
            self._nested += 1
        elif tag == "b" and self._lead is None and self._bold is None and not "".join(self._text).strip():
            self._bold = []
        elif tag == "br":
            self._text.append(" ")

    def handle_endtag(self, tag):
        if self._text is None:
            return
        if tag == "b" and self._bold is not None:
            self._lead = " ".join("".join(self._bold).split()) or None
            self._bold = None
        elif tag == "div":
            if self._nested:
                self._nested -= 1
            else:
                text = " ".join("".join(self._text).replace("\xa0", " ").split())
                if text:
                    self.paras.append((self._lead, text))
                self._text = None

    def handle_data(self, data):
        if self._text is not None:
            self._text.append(data)
            if self._bold is not None:
                self._bold.append(data)


def paras(html: str) -> list[tuple[str | None, str]]:
    parser = _Paras()
    parser.feed(html)
    parser.close()
    return parser.paras


def follows(prev: tuple[int, ...] | None, cur: tuple[int, ...]) -> bool:
    """Whether consideration ``cur`` can follow ``prev``: the first child (2 -> 2.1), the next
    sibling (2.1 -> 2.2) or the next of an ancestor (2.1.3 -> 2.2, 2.4 -> 3); the first is 1."""
    if prev is None:
        return cur == (1,)
    return cur == prev + (1,) or any(cur == prev[:d] + (prev[d] + 1,) for d in range(len(prev)))


def parse_bger(html: str, meta: dict, retrieved_at: str | None = None) -> Document | None:
    """One decision from its HTML and its search-API record (``id``, ``date``, ``reference``,
    ``attachment.content_url``); None when it has no text."""
    blocks = paras(html)
    marked = any(bold for bold, _ in blocks)  # numbers are bold (since ~2007), else plain text
    part, label, last_e, last_item = "Kopf", "Kopf", None, 0
    units: list[tuple[str, str, str]] = []  # (section label, text, part), in order; consecutive labels merge
    lines: list[str] = []
    for bold, text in blocks:
        if part == "Kopf" and not units and text in _COURT_LINES:
            continue
        lines.append(text)
        if _FACTS.match(text):
            part, label = "Sachverhalt", "Sachverhalt"
            continue
        if part in ("Kopf", "Sachverhalt") and REASONS.match(text):
            part, label = "Erwägungen", "Erwägungen"
            continue
        if part != "Schluss" and OPERATIVE.match(text):
            part, label = "Dispositiv", "Dispositiv"
            continue
        if part != "Kopf" and _CLOSING.match(text):
            part, label = "Schluss", "Schluss"
        lead = bold if marked else text
        body = text
        if part == "Sachverhalt":
            m = _LETTER.match(lead or "")
            if m and (marked or lead == text):
                label, body = f"Sachverhalt {m.group(1)}", ""
        elif part == "Erwägungen":
            m = _NUMBER.match(lead or "")
            number = tuple(int(x) for x in m.group(1).split(".")) if m else None
            if m and (marked or follows(last_e, number)):
                label, body, last_e = f"E. {m.group(1)}", text[m.end() :].strip(), number
        elif part == "Dispositiv":
            m = _ITEM.match(lead or "")
            if m and (marked or int(m.group(1)) == last_item + 1):
                last_item = int(m.group(1))
                label, body = f"Dispositiv Ziff. {last_item}", text[m.end() :].strip()
        if body:
            if units and units[-1][0] == label and part != "Kopf":
                units[-1] = (label, units[-1][1] + "\n" + body, part)
            else:
                units.append((label, body, part))
    if not any(lbl.startswith(("E. ", "Erwägungen", "Dispositiv")) for lbl, _, _ in units):
        return None  # no reasons and no operative part: not a decision text
    sections = [Section(label=lbl, text=t, part=p) for lbl, t, p in units]
    reference = (meta.get("reference") or [""])[0]
    return Document(
        id=f"bger:{meta['id']}",
        source="bger",
        jurisdiction="CH",
        doc_type="decision",
        text="\n".join(lines),
        citation_id=f"BGer {reference}",
        url=(meta.get("attachment") or {}).get("content_url", ""),
        valid_from=meta.get("date") or None,
        retrieved_at=retrieved_at or utc_now(),
        sections=sections,
        **licence_flags("CH-URG-5"),
    )


def blocked_ids(blocklist: dict, spider: str = "CH_BGer") -> set[str]:
    """The ids ``Blockliste.json`` withdraws from ``spider``'s collection (file names without
    their extension, the same as the search API's ``id``)."""
    return {
        str(name).rsplit("/", 1)[-1].removesuffix(".json").removesuffix(".html")
        for name in blocklist.get(spider, [])
    }


def search_body(size: int, after: list | None = None) -> dict:
    """The search-API query: German decisions of the Federal Supreme Court, newest first."""
    body = {
        "size": size,
        "_source": {"excludes": ["attachment.content"]},
        "query": {
            "bool": {"filter": [{"term": {"hierarchy": "CH_BGer"}}, {"term": {"attachment.language": "de"}}]}
        },
        "sort": [{"date": "desc"}, {"id": "desc"}],
    }
    if after is not None:
        body["search_after"] = after
    return body


class BGerCollector:
    name = "bger"
    licence = "CH-URG-5"
    jurisdiction = "CH"

    def __init__(self, page_size: int = 50):
        self.page_size = page_size

    def iter_documents(self, limit: int | None = None) -> Iterator[Document]:
        n, after = 0, None
        blocked = blocked_ids(http_get(BLOCKLIST).json())
        while True:
            hits = http_post(SEARCH, json=search_body(self.page_size, after)).json()["hits"]["hits"]
            if not hits:
                return
            for hit in hits:
                meta = hit["_source"]
                url = (meta.get("attachment") or {}).get("content_url")
                if not url or not meta.get("reference") or meta.get("id") in blocked:
                    continue
                response = http_get(url)
                response.encoding = "utf-8"
                doc = parse_bger(response.text, meta)
                if doc is None:
                    continue
                yield doc
                n += 1
                if limit is not None and n >= limit:
                    return
            after = hits[-1]["sort"]


if __name__ == "__main__":
    main(BGerCollector())
