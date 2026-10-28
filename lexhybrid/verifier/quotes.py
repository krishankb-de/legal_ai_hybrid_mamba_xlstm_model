"""Quote fidelity: a rendered quote equals its source after NFC + whitespace normalisation.

The renderer writes a ``QuotePointer(K, (J1..Jn))`` as ``„<sentences J1..Jn of passage K joined by
one space>“ (<label>)``, German quotes U+201E / U+201C, the label being the span's sentence labels in
order, deduplicated, joined by ``"; "``. The quote characters and the label rule are taken from that
contract here, not imported from the renderer, so a renderer that drifts from it fails this check.

``normalise`` is Unicode NFC followed by collapsing every whitespace run (``str.split``: spaces, tabs,
newlines, no-break spaces) to one space and stripping. Two texts match when their normal forms are
equal, so the same text in NFD form, or with a line break or a doubled space, passes, while any other
change fails: a homoglyph (Cyrillic ``а`` for Latin ``a``), ``ß`` written ``ss``, a word added or
dropped, an invisible character that is not whitespace (U+200B), a sentence one off.

``check_quotes`` reads the quotes of a whole rendered answer back in pointer order: the ``i``-th
``QuotePointer``'s quote is the first ``„…“ (<its label>)`` after the previous one's, and its body must
match the source sentences. The body runs from the nearest ``„`` before the closing ``“ (<label>)``
whose text matches, so quotation marks inside a sentence or in the free text before it do not cut
it short; a quote that is missing, mislabelled, cut, extended or altered is an issue.
"""

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from lexhybrid.renderer.segments import QuotePointer, Segment
from lexhybrid.retrieval.chunking import Chunk

OPEN, CLOSE = "„", "“"  # „ “
LABEL_SEPARATOR = "; "


def normalise(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def quote_matches(rendered_quote: str, source: str) -> bool:
    """A quote's body as rendered (the text between „ and “) equals the source after normalisation."""
    return normalise(rendered_quote) == normalise(source)


def source_text(chunk: Chunk, sentences: Sequence[int]) -> str:
    """Sentences ``J1..Jn`` of the chunk joined by one space: what a quote of them must say."""
    return " ".join(chunk.sentences[j - 1] for j in sentences)


def quote_label(chunk: Chunk, sentences: Sequence[int]) -> str:
    """The label a quote of these sentences carries: their labels in order, deduplicated, then, for
    a passage held by several sources, every source. Computed here,
    not taken from the renderer, so that renderer drift shows; only the source data is shared."""
    from lexhybrid.renderer.render import SOURCE_NAMES, sources

    label = LABEL_SEPARATOR.join(dict.fromkeys(chunk.labels[j - 1] for j in sentences))
    names = sources(chunk)
    if len(names) > 1:
        label += LABEL_SEPARATOR + "Quellen: " + ", ".join(SOURCE_NAMES.get(n, n) for n in names)
    return label


def extract_quotes(rendered: str) -> list[str]:
    """The bodies of the top-level ``„…“`` quotes in order, nested pairs kept inside their quote; an
    unclosed ``„`` opens no quote."""
    out: list[str] = []
    depth, start = 0, 0
    for i, ch in enumerate(rendered):
        if ch == OPEN:
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch == CLOSE and depth:
            depth -= 1
            if depth == 0:
                out.append(rendered[start:i])
    return out


def split_quote(rendered_quote: str) -> tuple[str, str]:
    """``(body, label)`` of one rendered quote ``„<body>“ (<label>)``, normalised; ``ValueError`` if it
    has another shape."""
    text = normalise(rendered_quote)
    marker = CLOSE + " ("
    if not (text.startswith(OPEN) and text.endswith(")")) or marker not in text:
        raise ValueError(f"not a quote of the form „…“ (label): {rendered_quote!r}")
    at = text.rindex(marker)
    return normalise(text[1:at]), normalise(text[at + len(marker) : -1])


def check_quote(chunk: Chunk, sentences: Sequence[int], rendered_quote: str) -> str | None:
    """Why one rendered quote of ``chunk``'s ``sentences`` is not faithful, or ``None`` when it is."""
    try:
        body, label = split_quote(rendered_quote)
    except ValueError as e:
        return str(e)
    if not quote_matches(body, source_text(chunk, sentences)):
        return f"the quoted text differs from sentences {list(sentences)} of {chunk.id}"
    if label != normalise(quote_label(chunk, sentences)):
        return f"the label ({label}) differs from ({quote_label(chunk, sentences)})"
    return None


@dataclass(frozen=True)
class QuoteIssue:
    segment: int  # index of the QuotePointer in the answer's segments
    passage: int
    sentences: tuple[int, ...]
    reason: str
    quoted: str | None = None  # what the rendered text quotes instead (normalised), when found


def _balanced_body(text: str, start: int, close: int) -> str:
    """The body before ``close`` from its depth-balanced ``„`` (for the issue; ``text[start:close]``
    when there is none)."""
    depth = 0
    for i in range(close - 1, start - 1, -1):
        if text[i] == CLOSE:
            depth += 1
        elif text[i] == OPEN:
            if depth == 0:
                return text[i + 1 : close].strip()
            depth -= 1
    return text[start:close].strip()


def _find_quote(text: str, start: int, label: str, expected: str) -> tuple[int, str | None] | None:
    """The first quote labelled ``label`` at or after ``start`` of the normalised ``text``:
    ``(its end, None)`` when its body is ``expected``, ``(its end, the body)`` when not, ``None`` when
    there is no such quote."""
    marker = f"{CLOSE} ({label})"
    close = text.find(marker, start)
    if close < 0:
        return None
    end = close + len(marker)
    for i in range(close - 1, start - 1, -1):
        if text[i] == OPEN and normalise(text[i + 1 : close]) == expected:
            return end, None
    return end, _balanced_body(text, start, close)


def check_quotes(segments: Sequence[Segment], chunks: Sequence[Chunk], rendered: str) -> list[QuoteIssue]:
    """Every ``QuotePointer`` whose quote in ``rendered`` is not its source, in segment order."""
    text = normalise(rendered)
    issues: list[QuoteIssue] = []
    cursor = 0
    for i, segment in enumerate(segments):
        if not isinstance(segment, QuotePointer):
            continue
        k, sentences = segment.passage, segment.sentences
        if not 1 <= k <= len(chunks):
            issues.append(QuoteIssue(i, k, sentences, f"passage {k} is outside the {len(chunks)} retrieved"))
            continue
        chunk = chunks[k - 1]
        if not sentences or any(not 1 <= j <= len(chunk.sentences) for j in sentences):
            reason = f"sentences {list(sentences)} are not all in passage {k} ({chunk.id}), which has {len(chunk.sentences)}"
            issues.append(QuoteIssue(i, k, sentences, reason))
            continue
        label = normalise(quote_label(chunk, sentences))
        found = _find_quote(text, cursor, label, normalise(source_text(chunk, sentences)))
        if found is None:
            issues.append(
                QuoteIssue(i, k, sentences, f"no quote labelled ({label}) follows the previous quote")
            )
            continue
        cursor, quoted = found
        if quoted is not None:
            reason = f"the quote differs from sentences {list(sentences)} of {chunk.id}"
            issues.append(QuoteIssue(i, k, sentences, reason, quoted))
    return issues
