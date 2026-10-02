"""Open Legal Data: German court decisions of every instance (plan P3-H).

Source: Open Legal Data's bulk dump on the Hugging Face hub, ``openlegaldata/court-decisions-germany``
(config ``dump-20260520``: 423,941 decisions in 54 parquet shards), read at a pinned revision with
``huggingface_hub``'s file system one row group at a time. The dataset is gated (automatic approval):
the account behind ``HF_TOKEN`` must have accepted its terms. The REST API
(``https://de.openlegaldata.io/api/``) was the source until 2026-09-30, when it began refusing any page
past the tenth ("Page 20 exceeds maximum of 10. For bulk access use
https://huggingface.co/openlegaldata/datasets"; job 2589358_2), i.e. at most the 500 newest cases.
A row carries the API's case fields (``id``, ``slug``, ``court``, ``file_number``, ``date``,
``content`` as HTML, already anonymised by the courts), so the parser is unchanged.

One ``Document`` per decision. The court's full name becomes its citation abbreviation
(``Verwaltungsgericht Bremen`` -> ``VG Bremen``), so the citation id is ``VG Bremen 2 K 1343/24``.
Sections are the HTML paragraphs, grouped under the part headings the decision uses (``Tenor``,
``Tatbestand``, ``Entscheidungsgründe``, ``Gründe``; ``Section.part``, None before the first heading). Licence: the database is ODbL 1.0 (register
row ``oldp``); the decisions themselves are official works.
"""

from collections.abc import Iterator

from lexhybrid.data.corpus.collectors.base import licence_flags, main, utc_now
from lexhybrid.data.corpus.collectors.htmltext import html_blocks
from lexhybrid.data.schema import Document, Section

REPO = "openlegaldata/court-decisions-germany"
REVISION = "408eb916fab64080580e4d67bf4242fb10cdedf5"  # main on 2026-09-30
CONFIG = "dump-20260520"
COLUMNS = ["id", "slug", "court", "file_number", "date", "content"]
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
    """One decision from a dump row (the API's case fields; None without text)."""
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
        sections.append(Section(label=part, text=block, part=None if part == "Text" else part))
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


def iter_rows(config: str = CONFIG, revision: str = REVISION) -> Iterator[dict]:
    """The dump's rows in shard order, one parquet row group in memory at a time."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    for path in sorted(fs.glob(f"datasets/{REPO}@{revision}/{config}/train-*.parquet")):
        with fs.open(path, "rb") as f:
            shard = pq.ParquetFile(f)
            for group in range(shard.num_row_groups):
                yield from shard.read_row_group(group, columns=COLUMNS).to_pylist()


class OLDPCollector:
    name = "oldp"
    licence = "ODbL-1.0"
    jurisdiction = "DE"

    def __init__(self, config: str = CONFIG, revision: str = REVISION):
        self.config, self.revision = config, revision

    def iter_documents(self, limit: int | None = None) -> Iterator[Document]:
        n = 0
        for row in iter_rows(self.config, self.revision):
            doc = parse_oldp_case(row)
            if doc is None:
                continue
            yield doc
            n += 1
            if limit is not None and n >= limit:
                return


if __name__ == "__main__":
    main(OLDPCollector())
