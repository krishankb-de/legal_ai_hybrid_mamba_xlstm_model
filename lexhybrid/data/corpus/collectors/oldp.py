"""Open Legal Data: German court decisions of every instance (plan P3-H).

Source: the REST API at ``https://de.openlegaldata.io/api/``. ``/api/cases/`` pages through case
metadata (``ordering=-date`` newest first); ``/api/cases/<id>/`` adds ``content``, the decision
as HTML, already anonymised by the courts. No API key is needed for reading.

One ``Document`` per decision. The court's full name becomes its citation abbreviation
(``Verwaltungsgericht Bremen`` -> ``VG Bremen``), so the citation id is ``VG Bremen 2 K 1343/24``.
Sections are the HTML paragraphs, grouped under the part headings the decision uses (``Tenor``,
``Tatbestand``, ``Entscheidungsgründe``, ``Gründe``). Licence: the database is ODbL 1.0 (register
row ``oldp``); the decisions themselves are official works.
"""

import re
from collections.abc import Iterator

from lexhybrid.data.corpus.collectors.base import cli, http_get, licence_flags, utc_now
from lexhybrid.data.corpus.collectors.htmltext import html_blocks
from lexhybrid.data.schema import Document, Section

API = "https://de.openlegaldata.io/api"
SITE = "https://de.openlegaldata.io"
PART_HEADINGS = {
    "tenor",
    "tatbestand",
    "entscheidungsgründe",
    "gründe",
    "leitsatz",
    "leitsätze",
    "rechtsmittelbelehrung",
}

# Full court name -> citation abbreviation. Longest prefixes first.
_COURTS = (
    ("Bundesgerichtshof", "BGH"),
    ("Bundesverfassungsgericht", "BVerfG"),
    ("Bundesverwaltungsgericht", "BVerwG"),
    ("Bundesfinanzhof", "BFH"),
    ("Bundesarbeitsgericht", "BAG"),
    ("Bundessozialgericht", "BSG"),
    ("Bundespatentgericht", "BPatG"),
    ("Bayerisches Oberstes Landesgericht", "BayObLG"),
    ("Bayerischer Verwaltungsgerichtshof", "BayVGH"),
    ("Kammergericht", "KG"),
    ("Oberlandesgericht", "OLG"),
    ("Oberverwaltungsgericht", "OVG"),
    ("Verwaltungsgerichtshof", "VGH"),
    ("Verwaltungsgericht", "VG"),
    ("Landesarbeitsgericht", "LAG"),
    ("Arbeitsgericht", "ArbG"),
    ("Landessozialgericht", "LSG"),
    ("Sozialgericht", "SG"),
    ("Finanzgericht", "FG"),
    ("Verfassungsgerichtshof", "VerfGH"),
    ("Staatsgerichtshof", "StGH"),
    ("Landgericht", "LG"),
    ("Amtsgericht", "AG"),
)


# A regional adjective before the court type stands for its seat or state
# ("Brandenburgisches Oberlandesgericht" -> "OLG Brandenburg").
_REGIONS = {
    "Bayerisch": "Bayern", "Brandenburgisch": "Brandenburg", "Hamburgisch": "Hamburg",
    "Hanseatisch": "Hamburg", "Hessisch": "Hessen", "Niedersächsisch": "Niedersachsen",
    "Pfälzisch": "Pfalz", "Saarländisch": "Saarland", "Sächsisch": "Sachsen",
    "Schleswig-Holsteinisch": "Schleswig-Holstein", "Thüringer": "Thüringen",
    "Rheinland-Pfälzisch": "Rheinland-Pfalz", "Mecklenburg-Vorpommersch": "Mecklenburg-Vorpommern",
}  # fmt: skip
_SPECIAL = {
    "Europäischer Gerichtshof": "EuGH",
    "Landesverfassungsgericht": "LVerfG",
    "Anwaltsgerichtshof": "AGH",
    "Anwaltsgericht": "AnwG",
    "Dienstgerichtshof": "DGH",
    "Dienstgericht": "DG",
    "Landesberufsgericht": "LBerufsG",
    "Berufsgericht": "BerufsG",
    "Schifffahrtsobergericht": "SchOG",
    "Schifffahrtsgericht": "SchG",
    "Rheinschifffahrtsobergericht": "RhSchOG",
    "Rheinschifffahrtsgericht": "RhSchG",
    "Moselschifffahrtsobergericht": "MoSchOG",
    "Verfassungsgericht": "VerfG",
}


def court_abbreviation(name: str) -> str:
    """``Oberlandesgericht Nürnberg`` -> ``OLG Nürnberg``; ``Hessischer Verwaltungsgerichtshof`` ->
    ``VGH Hessen``; ``Hanseatisches Oberlandesgericht in Bremen`` -> ``OLG Bremen``. A court the
    tables do not know keeps its name."""
    name = " ".join(name.split())
    if name in _SPECIAL:
        return _SPECIAL[name]
    for full, abbr in _COURTS:  # multi-word full names ("Bayerisches Oberstes Landesgericht")
        if " " in full and (name == full or name.startswith(full + " ")):
            return f"{abbr} {name[len(full) + 1 :]}".strip()
    words = name.split(" ")
    for i, word in enumerate(words):
        abbr = next((a for full, a in _COURTS if word == full), None) or next(
            (a for full, a in _SPECIAL.items() if word == full and " " not in full), None
        )
        if abbr is None:
            continue
        rest = [w for w in words[i + 1 :] if w not in ("in", "für", "des", "der", "Landes", "Kammerbezirk")]
        if rest:
            return f"{abbr} {' '.join(rest)}"
        if i > 0:
            stem = words[i - 1].rstrip("rs").removesuffix("e")
            region = next((r for adj, r in _REGIONS.items() if stem == adj or words[i - 1] == adj), None)
            if region:
                return f"{abbr} {region}"
        return abbr
    return name


def parse_oldp_case(case: dict, retrieved_at: str | None = None) -> Document | None:
    """One decision from an ``/api/cases/<id>/`` JSON object (None without text)."""
    html = case.get("content") or ""
    blocks = html_blocks(html)
    if not blocks:
        return None
    court = court_abbreviation((case.get("court") or {}).get("name", ""))
    docket = " ".join((case.get("file_number") or "").split())
    sections, part = [], "Text"
    for block in blocks:
        if block.strip(" :").lower() in PART_HEADINGS:
            part = block.strip(" :")
            continue
        sections.append(Section(label=part, text=block))
    if not sections:
        return None
    return Document(
        id=f"oldp:{case['id']}",
        source="oldp",
        jurisdiction="DE",
        doc_type="decision",
        text="\n".join(blocks),
        citation_id=f"{court} {docket}".strip(),
        url=f"{SITE}/case/{case.get('slug') or case['id']}",
        valid_from=case.get("date") or None,
        retrieved_at=retrieved_at or utc_now(),
        sections=sections,
        **licence_flags("ODbL-1.0"),
    )


class OLDPCollector:
    name = "oldp"
    licence = "ODbL-1.0"
    jurisdiction = "DE"

    def __init__(self, page_size: int = 50):
        self.page_size = page_size

    def iter_documents(self, limit: int | None = None) -> Iterator[Document]:
        url, params, n = (
            f"{API}/cases/",
            {"format": "json", "page_size": self.page_size, "ordering": "-date"},
            0,
        )
        while url:
            page = http_get(url, params=params).json()
            params = None  # `next` already carries them
            for item in page.get("results", []):
                case = http_get(f"{API}/cases/{item['id']}/", params={"format": "json"}).json()
                doc = parse_oldp_case(case)
                if doc is None:
                    continue
                yield doc
                n += 1
                if limit is not None and n >= limit:
                    return
            url = page.get("next")
            if url:
                url = re.sub(r"^http://", "https://", url)


if __name__ == "__main__":
    raise SystemExit(cli(OLDPCollector()))
