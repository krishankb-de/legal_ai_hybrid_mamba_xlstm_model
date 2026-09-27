"""EUR-Lex: EU regulations and directives in German (plan P3-I).

Source: the Publications Office's CELLAR repository, which serves each act's German XHTML by CELEX
number (``Accept: application/xhtml+xml``, ``Accept-Language: deu``). The eur-lex.europa.eu site
itself answers scripted requests with an empty 202 (a bot challenge), so it is only linked.

The XHTML (CONVEX, ELI) marks every article as ``<div class="eli-subdivision" id="art_N">`` with
``<p class="oj-ti-art">Artikel N</p>``, its title in ``p.oj-sti-art``, and one ``<div id="00N.00K">``
per paragraph (Absatz K); lettered points are small tables (``a)`` | text).

One ``Document`` per article (``DSGVO Art. 6``); sections are its paragraphs. Acts with a common
German short name use it; the rest are cited ``VO (EU) 2016/679`` / ``RL (EU) 2019/770`` from the
CELEX number. ``valid_from`` is the Official Journal date in the act's header. Licence: CC BY 4.0
under the EUR-Lex reuse notice (register row ``eurlex``).
"""

import re
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import date

from lexhybrid.data.corpus.collectors.base import FetchError, cli, http_get, licence_flags, utc_now
from lexhybrid.data.schema import Document, Section

CELLAR = "https://publications.europa.eu/resource/celex/{celex}"
EURLEX = "https://eur-lex.europa.eu/legal-content/DE/TXT/?uri=CELEX:{celex}"
XHTML_HEADERS = {"Accept": "application/xhtml+xml", "Accept-Language": "deu"}

# German short names that fit the citation grammar (no spaces).
SHORT_NAMES = {
    "32016R0679": "DSGVO",
    "32022R2065": "DSA",
    "32024R1689": "KI-VO",
    "32008R0593": "Rom-I-VO",
    "32007R0864": "Rom-II-VO",
    "32012R1215": "EuGVVO",
    "32004R0261": "FluggastrechteVO",
}
# Private-law core of the DACH corpus, collected first.
SEED_CELEX = (
    "32016R0679", "32011L0083", "32019L0771", "32019L0770", "31993L0013", "32008R0593",
    "32007R0864", "32012R1215", "32022R2065", "32024R1689", "32004R0261", "32015L2302",
)  # fmt: skip
_CELEX = re.compile(r"^3(\d{4})([RLD])(\d{4})$")
_KINDS = {"R": "VO", "L": "RL", "D": "Beschluss"}
_PARA_ID = re.compile(r"^\d{3}\.(\d{3})$")
_OJ_DATE = re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$")


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def act_code(celex: str) -> str:
    """The citation code of an act: its short name, else ``VO (EU) YYYY/N`` or ``RL (EU) YYYY/N``."""
    if celex in SHORT_NAMES:
        return SHORT_NAMES[celex]
    m = _CELEX.match(celex)
    if not m:
        raise ValueError(f"not a CELEX number of a regulation, directive or decision: {celex!r}")
    year, kind, number = m.groups()
    return f"{_KINDS[kind]} (EU) {year}/{int(number)}"


def _inline(elem: ET.Element) -> str:
    """All text under ``elem`` on one line."""
    return " ".join("".join(elem.itertext()).replace("\xa0", " ").split())


def render(elem: ET.Element) -> str:
    """Text of an XHTML block: ``<p>``s become lines; a table row (a lettered point, ``a)`` | text)
    becomes one line with its cells space-separated."""
    parts: list[str] = []

    def walk(e: ET.Element) -> None:
        tag = _local(e.tag)
        if tag == "tr":
            parts.append(
                "\n" + " ".join(c for c in (_inline(td) for td in e if _local(td.tag) == "td") if c) + "\n"
            )
            return
        if tag == "br":
            parts.append("\n")
        if e.text:
            parts.append(e.text)
        for child in e:
            walk(child)
            if child.tail:
                parts.append(child.tail)
        if tag in ("p", "div"):
            parts.append("\n")

    walk(elem)
    lines = (" ".join(line.replace("\xa0", " ").split()) for line in "".join(parts).split("\n"))
    return "\n".join(line for line in lines if line)


def _find(root: ET.Element, cls: str) -> ET.Element | None:
    return next((e for e in root.iter() if e.get("class") == cls), None)


def oj_date(root: ET.Element) -> str | None:
    header = _find(root, "oj-hd-date")
    m = _OJ_DATE.match(" ".join("".join(header.itertext()).split())) if header is not None else None
    if not m:
        return None
    d, mth, y = map(int, m.groups())
    return date(y, mth, d).isoformat()


def parse_eurlex_xhtml(xhtml: bytes, celex: str, retrieved_at: str | None = None) -> list[Document]:
    """Every article of one act as a ``Document``."""
    root = ET.fromstring(xhtml)
    code = act_code(celex)
    doc_type = "decision" if _CELEX.match(celex) and _CELEX.match(celex).group(2) == "D" else "regulation"
    valid_from = oj_date(root)
    flags = licence_flags("CC-BY-4.0")
    retrieved_at = retrieved_at or utc_now()
    docs = []
    for div in root.iter():
        art_id = div.get("id") or ""
        if (
            _local(div.tag) != "div"
            or div.get("class") != "eli-subdivision"
            or not re.fullmatch(r"art_\d+[a-z]*", art_id)
        ):
            continue
        number = art_id[4:]
        title_p = _find(div, "oj-sti-art")
        title = " ".join("".join(title_p.itertext()).split()) if title_p is not None else ""
        sections = []
        for child in div:
            m = _PARA_ID.match(child.get("id") or "")
            if _local(child.tag) == "div" and m:
                text = render(child)
                if text:
                    k = int(m.group(1))
                    sections.append(Section(label=f"Art. {number} Abs. {k}", absatz=k, text=text))
        if not sections:  # a one-paragraph article: its text sits directly in the article div
            body = [
                render(c)
                for c in div
                if c.get("class") not in ("oj-ti-art",) and not (c.get("class") or "").startswith("eli-title")
            ]
            text = "\n".join(b for b in body if b)
            if text:
                sections.append(Section(label=f"Art. {number}", text=text))
        if not sections:
            continue
        docs.append(
            Document(
                id=f"eurlex:{celex}:art{number}",
                source="eurlex",
                jurisdiction="EU",
                doc_type=doc_type,
                text=f"Artikel {number} {title}".strip() + "\n" + "\n".join(s.text for s in sections),
                citation_id=f"{code} Art. {number}",
                url=EURLEX.format(celex=celex),
                valid_from=valid_from,
                retrieved_at=retrieved_at,
                sections=sections,
                **flags,
            )
        )
    return docs


class EURLexCollector:
    name = "eurlex"
    licence = "CC-BY-4.0"
    jurisdiction = "EU"

    def __init__(self, celex: tuple[str, ...] = SEED_CELEX):
        self.celex = celex

    def iter_documents(self, limit: int | None = None) -> Iterator[Document]:
        n = 0
        for celex in self.celex:
            try:
                xhtml = http_get(CELLAR.format(celex=celex), headers=XHTML_HEADERS, timeout=180).content
            except FetchError as e:  # older acts may have no German XHTML manifestation (HTTP 404)
                print(f"eurlex: skipping {celex}: {e}", file=sys.stderr)
                continue
            for doc in parse_eurlex_xhtml(xhtml, celex):
                yield doc
                n += 1
                if limit is not None and n >= limit:
                    return


if __name__ == "__main__":
    raise SystemExit(cli(EURLexCollector()))
