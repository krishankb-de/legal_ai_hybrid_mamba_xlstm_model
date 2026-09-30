"""Retrieval chunks (plan P7-A, P7-B): the units the retriever indexes and a prompt quotes.

A chunk is what one ``<|passage|>`` of the SFT prompt (P3-Y) holds: a citation header and at most
64 numbered sentences (``<|s1|>`` .. ``<|s64|>``), each with the label the renderer prints beside a
quote of it (``BeurkG §3 Abs. 1 Nr. 2``, ``BAG 3 AZR 158/22 Rn. 14``). The sentences are the
source's own text, whitespace-normalised, so a quote can be checked against it (P7-H).

**Statutes** (P7-A, ``chunk_statute``): one chunk per § / Artikel -- the statute collectors emit one
document per provision -- with its hierarchy, outermost first: the act (``BGB``), the structural
headings the source records (``Document.hierarchy``: Buch, Abschnitt, Titel), the provision
(``§ 573 Ordentliche Kündigung des Vermieters``); below it every Absatz (``Section.absatz``) is cut
into its sentences (``Satz``) and its numbered or lettered items (``Nr.``, ``lit.``), each labelled
down to that grain. ``valid_from`` / ``valid_to`` are the version's validity window. A provision of
more than 64 units continues in the next chunk.

**Decisions** (P7-B, ``chunk_case_law``): part by part (``Section.part``: Leitsatz, Tenor,
Tatbestand, Entscheidungsgründe or Gründe; Swiss Sachverhalt, Erwägungen, Dispositiv; Austrian
Spruch, Begründung), each part cut into windows of 300 to 500 tokens: a window ends at a paragraph
boundary once it holds 300 and the next paragraph would not fit, and inside a paragraph only at a
sentence end when the next sentence would pass 500 (a single sentence longer than that is cut at
spaces); only a part's last window, or a short part, holds fewer than 300. A window's anchor is the
Randnummern (or Erwägungen) it covers, ``Rn. 7–12``, and its header cites the decision there. Tokens are counted by ``count_tokens`` (the index build passes
the student tokenizer's length); the default estimates 2.28 Qwen3 tokens per whitespace word, the
pooled DACH legal fertility P3-D measured.

``copies`` lists the documents that hold the same text in other sources (decision 13 as amended:
every copy is kept); the index fills it from the dedup links, and the renderer cites all of them
(``decisions.cite_every_copy``).
"""

import json
import math
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from typing import get_origin

from lexhybrid.data.schema import Document, Section, parse_citation
from lexhybrid.data.sft_format import Passage, split_sentences
from lexhybrid.data.tokenizer import N_SENTENCES

TOKENS_PER_WORD = 2.28  # Qwen3 tokens per whitespace word on the 1M-word legal sample (P3-D)
MIN_TOKENS, MAX_TOKENS = 300, 500  # P7-B's window
STATUTE_TYPES = ("statute", "regulation")
_ITEM = re.compile(r"^(\d+[a-z]?)\.\s+\S")  # "1. eigene Angelegenheiten"
_LETTER = re.compile(r"^([a-z]{1,2})\)\s+\S")  # "a) die betroffene Person"
_NUMBERED = re.compile(r"^(Rn\.|Rz\.|E\.)\s")  # section labels that cite with the decision


def estimate_tokens(text: str) -> int:
    return math.ceil(TOKENS_PER_WORD * len(text.split()))


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage."""

    id: str  # "<document id>#<n>", n from 1
    doc_id: str
    source: str
    jurisdiction: str
    doc_type: str
    citation_id: str  # the passage header: the document's citation, at the window's anchor
    hierarchy: tuple[str, ...]  # outermost first
    sentences: tuple[str, ...]  # <|sJ|> is sentences[J - 1]
    labels: tuple[str, ...]  # the citation label of each sentence
    valid_from: str | None
    valid_to: str | None
    licence: str
    commercial_safe: bool
    research_only: bool
    url: str
    part: str | None = None  # decisions: the part the window comes from
    anchor: str = ""  # decisions: "Rn. 7–12", "E. 3.1–3.4"
    copies: tuple[str, ...] = ()  # documents with the same text in other sources (filled by the index)

    def __post_init__(self):
        if not 1 <= len(self.sentences) <= N_SENTENCES:
            raise ValueError(f"{self.id}: {len(self.sentences)} sentences; a passage holds 1..{N_SENTENCES}")
        if len(self.labels) != len(self.sentences):
            raise ValueError(f"{self.id}: one label per sentence")

    @property
    def text(self) -> str:
        """What the lexical and dense indexes see: the header, the hierarchy, the sentences."""
        return "\n".join([self.citation_id, " › ".join(self.hierarchy), *self.sentences])

    def passage(self) -> Passage:
        """The P3-Y prompt passage: ``<|passage|><|cK|>`` header, then the numbered sentences."""
        return Passage(self.citation_id, self.sentences)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> "Chunk":
        d = json.loads(line)
        tuples = {f.name for f in fields(cls) if get_origin(f.type) is tuple}
        return cls(**{k: tuple(v) if k in tuples else v for k, v in d.items()})


def _common(doc: Document) -> dict:
    return dict(
        doc_id=doc.id, source=doc.source, jurisdiction=doc.jurisdiction, doc_type=doc.doc_type,
        valid_from=doc.valid_from, valid_to=doc.valid_to, licence=doc.licence,
        commercial_safe=doc.commercial_safe, research_only=doc.research_only, url=doc.url,
    )  # fmt: skip


# ---------------------------------------------------------------------------------------------
# statutes (P7-A)
# ---------------------------------------------------------------------------------------------


def statute_units(section: Section) -> list[tuple[str, str]]:
    """``(grain, text)`` for one Absatz: its sentences (``Satz 2``) and its items (``Nr. 1``,
    ``lit. a``), in order. A one-sentence Absatz carries no ``Satz``."""
    units: list[tuple[str, str]] = []
    for line in section.text.split("\n"):
        item = _ITEM.match(line) or _LETTER.match(line)
        if item:
            grain = f"Nr. {item.group(1)}" if item.re is _ITEM else f"lit. {item.group(1)}"
            units.append((grain, " ".join(line.split())))
        else:
            units += [("Satz", s) for s in split_sentences(line)]
    n_sentences = sum(1 for g, _ in units if g == "Satz")
    out, satz = [], 0
    for grain, text in units:
        if grain == "Satz":
            satz += 1
            grain = f"Satz {satz}" if n_sentences > 1 else ""
        out.append((grain, text))
    return out


def act_of(citation_id: str) -> str:
    """``BGB`` of ``BGB §573``, ``ZGB SchlT`` of ``ZGB SchlT Art. 1``; the text before the § or the
    Art. when the id is outside the grammar."""
    try:
        parts = parse_citation(citation_id)
    except ValueError:
        return re.split(r" ?§| Art\. ", citation_id, maxsplit=1)[0]
    return parts.get("code", citation_id) + (f" {parts['part']}" if parts.get("part") else "")


def provision_heading(doc: Document) -> str:
    """``§ 1 Geltungsbereich``: the text's first line when it is a heading, not the first Absatz."""
    first = doc.text.split("\n", 1)[0].strip()
    if doc.sections and doc.sections[0].text.startswith(first):
        return doc.sections[0].label.split(" Abs.")[0]
    return first


def chunk_statute(doc: Document) -> list[Chunk]:
    """The chunks of one statute provision (P7-A); see the module docstring."""
    if doc.doc_type not in STATUTE_TYPES:
        raise ValueError(f"{doc.id}: doc_type {doc.doc_type!r} is not a statute or regulation")
    hierarchy = (act_of(doc.citation_id), *doc.hierarchy, provision_heading(doc))
    labelled = []
    for section in doc.sections:
        base = f"{doc.citation_id} Abs. {section.absatz}" if section.absatz else doc.citation_id
        labelled += [(f"{base} {grain}".strip(), text) for grain, text in statute_units(section)]
    if not labelled:
        raise ValueError(f"{doc.id}: no text to chunk")
    return [
        Chunk(
            id=f"{doc.id}#{k + 1}",
            citation_id=doc.citation_id,
            hierarchy=hierarchy,
            sentences=tuple(text for _, text in labelled[start : start + N_SENTENCES]),
            labels=tuple(label for label, _ in labelled[start : start + N_SENTENCES]),
            **_common(doc),
        )
        for k, start in enumerate(range(0, len(labelled), N_SENTENCES))
    ]


# ---------------------------------------------------------------------------------------------
# decisions (P7-B)
# ---------------------------------------------------------------------------------------------


def _anchor(sections: list[Section]) -> str:
    """``Rn. 7–12`` / ``Rn. 7`` / ``E. 3.1–3.4`` over the numbered sections of a window."""
    numbered = [s.label for s in sections if _NUMBERED.match(s.label)]
    if not numbered:
        return ""
    first, last = numbered[0], numbered[-1]
    if first == last:
        return first
    prefix = first.split(" ", 1)[0]
    return f"{first}–{last.split(' ', 1)[1]}" if last.startswith(prefix + " ") else f"{first}–{last}"


def _label(citation: str, section: Section) -> str:
    """How one paragraph is cited: at its Randnummer or Erwägung, else by its label (``Tenor``)."""
    return f"{citation} {section.label}" if _NUMBERED.match(section.label) else f"{citation}, {section.label}"


def _units(section: Section, count: Callable[[str], int], max_tokens: int) -> list[str]:
    """A paragraph's sentences; one longer than a window (a table, a list of motions) is cut at
    spaces into window-sized pieces, each still a verbatim stretch of the paragraph."""
    out = []
    for sentence in split_sentences(section.text.replace("\n", " ")):
        if count(sentence) <= max_tokens:
            out.append(sentence)
            continue
        piece: list[str] = []
        for word in sentence.split(" "):
            if piece and count(" ".join([*piece, word])) > max_tokens:
                out.append(" ".join(piece))
                piece = []
            piece.append(word)
        if piece:
            out.append(" ".join(piece))
    return out


def _windows(
    sections: list[Section], count: Callable[[str], int], min_tokens: int, max_tokens: int
) -> list[list[tuple[Section, str]]]:
    """``(section, sentence)`` windows of one part: a window closes at a paragraph boundary once it
    holds ``min_tokens`` and the next paragraph would not fit, and inside a paragraph only when the
    next sentence would pass ``max_tokens`` (or the 64 sentence ids); a short last window joins the
    one before it when the two fit together."""
    windows: list[list[tuple[Section, str]]] = [[]]
    sizes = [0]
    for section in sections:
        units = [(u, count(u)) for u in _units(section, count, max_tokens)]
        whole = sum(n for _, n in units)
        if sizes[-1] >= min_tokens and sizes[-1] + whole > max_tokens:
            windows.append([])
            sizes.append(0)
        for text, n in units:
            if windows[-1] and (sizes[-1] + n > max_tokens or len(windows[-1]) == N_SENTENCES):
                windows.append([])
                sizes.append(0)
            windows[-1].append((section, text))
            sizes[-1] += n
    if not windows[-1]:
        windows.pop()
        sizes.pop()
    if len(windows) > 1 and sizes[-1] < min_tokens:
        if sizes[-2] + sizes[-1] <= max_tokens and len(windows[-2]) + len(windows[-1]) <= N_SENTENCES:
            windows[-2:] = [windows[-2] + windows[-1]]
    return windows


def chunk_case_law(
    doc: Document,
    count_tokens: Callable[[str], int] = estimate_tokens,
    min_tokens: int = MIN_TOKENS,
    max_tokens: int = MAX_TOKENS,
) -> list[Chunk]:
    """The chunks of one decision (P7-B); see the module docstring."""
    if doc.doc_type != "decision":
        raise ValueError(f"{doc.id}: doc_type {doc.doc_type!r} is not a decision")
    groups: list[tuple[str, list[Section]]] = []  # consecutive sections of one part
    for s in doc.sections:
        part = s.part or "Text"
        if groups and groups[-1][0] == part:
            groups[-1][1].append(s)
        else:
            groups.append((part, [s]))
    chunks: list[Chunk] = []
    for part, sections in groups:
        for window in _windows(sections, count_tokens, min_tokens, max_tokens):
            covered: list[Section] = []
            for section, _ in window:
                if not covered or covered[-1] is not section:
                    covered.append(section)
            anchor = _anchor(covered)
            header = f"{doc.citation_id} {anchor}" if anchor else f"{doc.citation_id}, {part}"
            chunks.append(
                Chunk(
                    id=f"{doc.id}#{len(chunks) + 1}",
                    citation_id=header,
                    hierarchy=(doc.citation_id, part),
                    sentences=tuple(text for _, text in window),
                    labels=tuple(_label(doc.citation_id, s) for s, _ in window),
                    part=part,
                    anchor=anchor,
                    **_common(doc),
                )
            )
    return chunks


def chunk_document(doc: Document, count_tokens: Callable[[str], int] = estimate_tokens) -> list[Chunk]:
    """Statutes and regulations by provision, decisions by window; other types are not indexed."""
    if doc.doc_type in STATUTE_TYPES:
        return chunk_statute(doc)
    if doc.doc_type == "decision":
        return chunk_case_law(doc, count_tokens)
    return []
