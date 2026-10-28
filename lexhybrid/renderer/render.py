"""The renderer: the model's pointer answer as the text a reader sees.

The answer (``lexhybrid/renderer/segments.py`` reads its grammar) is rendered against the retrieved
passages in prompt order, passage ``K`` being ``chunks[K - 1]``:

* ``<|q|><|cK|><|sJ1|>…<|sJn|>`` -> ``„<sentences J1..Jn of passage K, joined by one space>“ (<label>)``,
  German quotes U+201E / U+201C. The sentences are the chunk's own strings byte for byte: no
  normalisation, no truncation, so a quote is checkable against the source. The label of one
  sentence is ``chunks[K - 1].labels[J - 1]`` (``BeurkG §3 Abs. 1 Nr. 2``); of a span, the span's
  labels in order with duplicates removed, joined by ``"; "``;
* ``<|cite|><|cK|>`` -> ``(<chunks[K - 1].citation_id>)`` (``(BGB §573)``); with sentences,
  ``<|cite|><|cK|><|sJ|>…`` -> ``(<their labels, deduplicated, joined by "; ">)``
  (``(BGB §573 Abs. 2 Nr. 1)``);
* a passage held by more than one source (``Chunk.copies``) adds every
  source to its label: ``(BGH VIII ZR 12/20 Rn. 14; Quellen: rechtsprechung-im-internet.de,
  openlegaldata.io)``. The quoted text itself is unchanged, so the quote check is unaffected;
* text between pointers is copied unchanged;
* ``<|unanswerable|>``, and an answer whose rendered body is empty or whitespace (it answers
  nothing), render as ``REFUSAL``.

The rendered body, stripped of surrounding whitespace, is followed by ``"\\n\\n" + NOTICE``: every
answer, a refusal included, ends with the notice, appended once. Text is not searched for it, so an
answer that writes the notice itself shows it twice. A pointer to a passage outside
``1..len(chunks)`` or to a sentence outside ``1..len(chunk.sentences)`` is outside the retrieved set
and raises ``RenderError`` naming the pointer; a malformed answer raises the parser's
``PointerSyntaxError``. Both are ``ValueError``. The verifier reuses ``render_quote`` and
``render_cite`` so that what it checks is what the reader sees.
"""

from collections.abc import Sequence

from lexhybrid.data.sft_format import CITE, UNANSWERABLE, Q
from lexhybrid.renderer.segments import CitePointer, QuotePointer, Segment, Text, Unanswerable, parse_answer
from lexhybrid.retrieval.chunking import Chunk

NOTICE = "Hinweis: Dies ist eine rechtliche Information, keine Rechtsberatung."
REFUSAL = "Die bereitgestellten Quellen beantworten diese Frage nicht."
OPEN_QUOTE, CLOSE_QUOTE = "„", "“"  # „ “
LABEL_SEPARATOR = "; "


class RenderError(ValueError):
    pass


def pointer_text(pointer: QuotePointer | CitePointer) -> str:
    """The pointer as the model wrote it: ``<|q|><|c2|><|s1|><|s2|>``, ``<|cite|><|c3|>``."""
    marker = Q if isinstance(pointer, QuotePointer) else CITE
    return marker + f"<|c{pointer.passage}|>" + "".join(f"<|s{j}|>" for j in pointer.sentences)


def _check_sentences(chunk: Chunk, sentences: Sequence[int]) -> None:
    for j in sentences:
        if not 1 <= j <= len(chunk.sentences):
            raise RenderError(f"sentence {j} is outside {chunk.id}, which has {len(chunk.sentences)}")


# display names of the sources a citation names; an unknown source id is shown as is
SOURCE_NAMES = {
    "gii": "gesetze-im-internet.de", "rii": "rechtsprechung-im-internet.de", "oldp": "openlegaldata.io",
    "dip": "dip.bundestag.de", "eurlex": "eur-lex.europa.eu", "ris": "ris.bka.gv.at", "fedlex": "fedlex.admin.ch",
    "bger": "bger.ch", "multilegalpile": "MultiLegalPile", "fineweb2_de": "FineWeb-2",
}  # fmt: skip


def sources(chunk: Chunk) -> list[str]:
    """Every source holding the passage: the chunk's own first, then its copies' (document ids are
    ``<source>:<native id>``), each once."""
    return list(dict.fromkeys([chunk.source, *(c.split(":", 1)[0] for c in chunk.copies)]))


def _with_sources(chunk: Chunk, label: str) -> str:
    names = sources(chunk)
    if len(names) < 2:
        return label
    return label + LABEL_SEPARATOR + "Quellen: " + ", ".join(SOURCE_NAMES.get(n, n) for n in names)


def _labels(chunk: Chunk, sentences: Sequence[int]) -> str:
    return LABEL_SEPARATOR.join(dict.fromkeys(chunk.labels[j - 1] for j in sentences))


def render_quote(chunk: Chunk, sentences: Sequence[int]) -> str:
    """``„<the sentences, verbatim, joined by one space>“ (<their labels>)``."""
    if not sentences:
        raise RenderError(f"a quote of {chunk.id} names no sentence")
    _check_sentences(chunk, sentences)
    text = " ".join(chunk.sentences[j - 1] for j in sentences)
    return f"{OPEN_QUOTE}{text}{CLOSE_QUOTE} ({_with_sources(chunk, _labels(chunk, sentences))})"


def render_cite(chunk: Chunk, sentences: Sequence[int] = ()) -> str:
    """``(<citation_id>)``, or ``(<the sentences' labels>)`` when the citation names sentences."""
    _check_sentences(chunk, sentences)
    return f"({_with_sources(chunk, _labels(chunk, sentences) if sentences else chunk.citation_id)})"


def _resolve(pointer: QuotePointer | CitePointer, chunks: Sequence[Chunk]) -> Chunk:
    """The pointer's passage, after checking the pointer stays inside the retrieved set."""
    if not 1 <= pointer.passage <= len(chunks):
        raise RenderError(
            f"{pointer_text(pointer)}: passage {pointer.passage} is outside the {len(chunks)} retrieved"
        )
    chunk = chunks[pointer.passage - 1]
    if isinstance(pointer, QuotePointer) and not pointer.sentences:
        raise RenderError(f"{pointer_text(pointer)}: a quote names no sentence")
    for j in pointer.sentences:
        if not 1 <= j <= len(chunk.sentences):
            raise RenderError(
                f"{pointer_text(pointer)}: sentence {j} is outside passage {pointer.passage}"
                f" ({chunk.id}), which has {len(chunk.sentences)}"
            )
    return chunk


def render_segments(segments: Sequence[Segment], chunks: Sequence[Chunk]) -> str:
    """The rendered answer of parsed segments against the retrieved passages (prompt order)."""
    if any(isinstance(s, Unanswerable) for s in segments):
        if len(segments) != 1:
            raise RenderError(f"{UNANSWERABLE} must be the whole answer")
        return REFUSAL + "\n\n" + NOTICE
    parts: list[str] = []
    for segment in segments:
        if isinstance(segment, Text):
            parts.append(segment.text)
        elif isinstance(segment, QuotePointer):
            parts.append(render_quote(_resolve(segment, chunks), segment.sentences))
        elif isinstance(segment, CitePointer):
            parts.append(render_cite(_resolve(segment, chunks), segment.sentences))
        else:
            raise TypeError(f"not an answer segment: {segment!r}")
    body = "".join(parts).strip()
    return (body or REFUSAL) + "\n\n" + NOTICE


def render_answer(answer: str, chunks: Sequence[Chunk]) -> str:
    """The rendered answer of the model's raw answer text (``<|endoftext|>`` allowed at the end)."""
    return render_segments(parse_answer(answer), chunks)
