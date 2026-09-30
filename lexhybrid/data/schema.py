"""The one document record every collector emits and every later stage reads (plan P3-A).

A ``Document`` is a whole statute provision, decision, parliamentary paper or general-web page, with
its provenance (source, licence, URL, retrieval time, content hash), its legal coordinates
(jurisdiction, type, validity window, citation id) and its text split into ``Section``s -- the
units the retriever chunks (P7) and the renderer quotes (``BGB §573 Abs. 2 Nr. 1``).

Licence discipline (R9) starts here: ``licence`` is required, ``commercial_safe`` is explicit, and
``research_only`` marks the non-commercial research-exception arm (decision 2), which can never be
commercial-safe. ``Docs/DOCUMENT_SCHEMA.md`` documents every field and the ``citation_id`` grammar.
"""

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime

JURISDICTIONS = ("DE", "AT", "CH", "EU")
DOC_TYPES = ("statute", "regulation", "decision", "parliament", "general")


def sha256_text(text: str) -> str:
    """Hex SHA-256 of the UTF-8 text: the identity dedup and the manifests key on."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Section:
    """One quotable unit of a document.

    Attributes:
        label: its citable label inside the document: ``"§ 573 Abs. 2 Nr. 1"``, ``"Art. 97 Abs. 1"``,
            ``"Rn. 14"``, ``"E. 3.2"``, ``"Tenor"``.
        absatz: paragraph (Absatz) number, when the section is one.
        satz_idx: sentence number within the Absatz, when the section is one sentence.
        randnummer: margin number (Randnummer / Rz.) of a decision, when it has one.
        text: the section's exact text (what a quote is checked against).
        part: the part of a decision the section belongs to -- ``"Leitsatz"``, ``"Tenor"``,
            ``"Tatbestand"``, ``"Entscheidungsgründe"``, ``"Gründe"``, ``"Sachverhalt"``,
            ``"Erwägungen"``, ``"Spruch"``, ``"Begründung"`` -- when the source marks it (P7-B chunks a
            decision part by part); None for statutes and sources without parts.
    """

    label: str
    absatz: int | None = None
    satz_idx: int | None = None
    randnummer: int | None = None
    text: str = ""
    part: str | None = None


@dataclass
class Document:
    """A legal or general document with provenance, licence and sections. See the module docstring."""

    id: str
    source: str
    jurisdiction: str
    doc_type: str
    licence: str
    commercial_safe: bool
    text: str
    citation_id: str = ""
    url: str = ""
    valid_from: str | None = None  # ISO date, e.g. "2021-07-01"
    valid_to: str | None = None
    retrieved_at: str = ""  # ISO timestamp (UTC)
    sha256: str = ""  # filled from ``text`` when empty
    research_only: bool = False
    sections: list[Section] = field(default_factory=list)
    # The act's structural headings above this provision, outermost first, when the source has them
    # (P7-A): ["Buch 2 Recht der Schuldverhältnisse", "Abschnitt 8 Einzelne Schuldverhältnisse", ...].
    hierarchy: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.sections = [s if isinstance(s, Section) else Section(**s) for s in self.sections]
        if not self.sha256:
            self.sha256 = sha256_text(self.text)
        problems = self.problems()
        if problems:
            raise ValueError(f"invalid Document {self.id!r}: " + "; ".join(problems))

    def problems(self) -> list[str]:
        out = []
        if not self.id or ":" not in self.id:
            out.append("id must be '<source>:<native id>'")
        elif not self.id.startswith(self.source + ":"):
            out.append(f"id {self.id!r} does not start with its source {self.source!r}")
        if self.jurisdiction not in JURISDICTIONS:
            out.append(f"jurisdiction {self.jurisdiction!r} not in {JURISDICTIONS}")
        if self.doc_type not in DOC_TYPES:
            out.append(f"doc_type {self.doc_type!r} not in {DOC_TYPES}")
        if not self.licence:
            out.append("licence is required (use 'unknown' explicitly, which the corpus build refuses)")
        if self.research_only and self.commercial_safe:
            out.append("a research_only document can never be commercial_safe")
        for name in ("valid_from", "valid_to"):
            value = getattr(self, name)
            if value is not None:
                try:
                    date.fromisoformat(value)
                except ValueError:
                    out.append(f"{name} {value!r} is not an ISO date")
        if self.valid_from and self.valid_to and self.valid_to < self.valid_from:
            out.append("valid_to precedes valid_from")
        if self.retrieved_at:
            try:
                datetime.fromisoformat(self.retrieved_at)
            except ValueError:
                out.append(f"retrieved_at {self.retrieved_at!r} is not an ISO timestamp")
        if self.sha256 != sha256_text(self.text):
            out.append("sha256 does not match text")
        return out

    def to_dict(self, with_text: bool = True) -> dict:
        d = asdict(self)
        if not with_text:
            d.pop("text")
            for s in d["sections"]:
                s.pop("text")
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        return cls(**d)

    @classmethod
    def from_json(cls, line: str) -> "Document":
        return cls.from_dict(json.loads(line))


# ---------------------------------------------------------------------------------------------
# citation ids (Docs/DOCUMENT_SCHEMA.md)
# ---------------------------------------------------------------------------------------------

_NUM = r"\d+[a-z]*"  # 573, 573a, 334bis
_STATUTE = re.compile(
    rf"^(?P<code>SR \d+(?:\.\d+)*|[A-ZÄÖÜ][\wÄÖÜäöüß\-/]*(?: \([A-Z]+\) \d{{4}}/\d+)?)"
    rf"(?: (?P<part>SchlT))? "
    rf"(?:§ ?(?P<paragraph>{_NUM})|Art\. (?P<article>{_NUM}))"
    rf"(?: Abs\. (?P<absatz>{_NUM}))?(?: (?:S\.|Satz) (?P<satz>\d+))?"
    rf"(?: Nr\. (?P<nummer>{_NUM}))?(?: lit\. (?P<litera>[a-z]{{1,2}}))?$"
)
_DECISION = re.compile(
    r"^(?P<court>BGH|BVerfG|BVerwG|BFH|BAG|BSG|BPatG|OGH|VfGH|VwGH|BGer|BVGer|EuGH|EuG|[A-Z][A-Za-z]*G|[A-Z][A-Za-z]*GH) "
    r"(?P<docket>[\w\s/.\-()]+?)(?: (?:Rn\.|Rz\.) (?P<randnummer>\d+)| E\. (?P<erwaegung>\d+(?:\.\d+)*))?$"
)


_PAPER = re.compile(
    r"^(?P<series>BT-Drs\.|BR-Drs\.|BT-PlPr\.|BR-PlPr\.) (?P<number>\d+/\d+)(?: S\. (?P<page>\d+))?$"
)


def parse_citation(citation_id: str) -> dict:
    """The parts of a citation id; raises ``ValueError`` if it matches neither grammar.

    Statutes: ``<code> [SchlT] §<n>|Art. <n> [Abs. <k>] [Satz|S. <s>] [Nr. <m>] [lit. <x>]``
    (``BGB §573 Abs. 2 Nr. 1``, ``ABGB §1295``, ``OR Art. 97``, ``DSGVO Art. 6 Abs. 1 lit. f``,
    ``ZGB SchlT Art. 1``, ``SR 221.214.111 Art. 3``, ``ZGB Art. 334bis``).
    Decisions: ``<court> <docket> [Rn.|Rz. <n>] [E. <n.n>]`` (``BGH VIII ZR 12/20 Rn. 14``,
    ``BGer 4A_123/2020 E. 3.2``); a lower court's seat is part of the docket (``VG Bremen 2 K 1343/24``).
    Parliamentary papers: ``BT-Drs.|BR-Drs.|BT-PlPr.|BR-PlPr. <n>/<n> [S. <p>]`` (``BT-Drs. 21/8109``).
    """
    m = _STATUTE.match(citation_id)
    if m:
        return {"kind": "statute", **{k: v for k, v in m.groupdict().items() if v is not None}}
    m = _DECISION.match(citation_id)
    if m:
        return {"kind": "decision", **{k: v.strip() for k, v in m.groupdict().items() if v is not None}}
    m = _PAPER.match(citation_id)
    if m:
        return {"kind": "parliament", **{k: v for k, v in m.groupdict().items() if v is not None}}
    raise ValueError(f"{citation_id!r} is not a statute, decision or parliamentary-paper citation id")
