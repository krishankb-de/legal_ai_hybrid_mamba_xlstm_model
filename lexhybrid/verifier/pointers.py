"""Pointer resolution (plan P7-H): every pointer of an answer names a passage and sentences that exist.

The answer's pointers (``lexhybrid/renderer/segments.py``) are 1-based into the retrieved passages in
prompt order, passage ``K`` being ``chunks[K - 1]``. The parser checks only their syntax; here each
``QuotePointer`` and ``CitePointer`` is checked against the retrieved set. A pointer resolves when

* its passage ``K`` is in ``1..len(chunks)`` (else nothing else about it is checked);
* every sentence ``J`` is in ``1..len(chunks[K - 1].sentences)``;
* no sentence is named twice;
* a quote names at least one sentence, and a quote of several is a span ``J, J + 1, ...``: strictly
  consecutive and increasing, because the renderer joins the sentences into one quotation and a gap
  or a reversal would put words side by side that the source does not. A citation of several
  sentences may name them in any order.

``check_pointers`` lists every failure in segment order and never raises; an empty list means
every pointer resolves.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from lexhybrid.renderer.segments import CitePointer, QuotePointer, Segment
from lexhybrid.retrieval.chunking import Chunk


@dataclass(frozen=True)
class PointerIssue:
    segment: int  # index of the pointer in the answer's segments
    passage: int  # K as the pointer names it
    sentence: int | None  # the offending J; None when the passage itself fails
    reason: str


def check_pointer(
    index: int, pointer: QuotePointer | CitePointer, chunks: Sequence[Chunk]
) -> list[PointerIssue]:
    """The failures of one pointer (segment ``index``) against the retrieved passages."""
    k = pointer.passage
    if not 1 <= k <= len(chunks):
        return [PointerIssue(index, k, None, f"passage {k} is outside the {len(chunks)} retrieved")]
    chunk = chunks[k - 1]
    issues: list[PointerIssue] = []
    if isinstance(pointer, QuotePointer) and not pointer.sentences:
        issues.append(PointerIssue(index, k, None, f"the quote of passage {k} names no sentence"))
    seen: set[int] = set()
    for n, j in enumerate(pointer.sentences):
        reasons = []
        if not 1 <= j <= len(chunk.sentences):
            reasons.append(
                f"sentence {j} is outside passage {k} ({chunk.id}), which has {len(chunk.sentences)}"
            )
        if j in seen:
            reasons.append(f"sentence {j} of passage {k} is named twice")
        elif isinstance(pointer, QuotePointer) and n and j != pointer.sentences[n - 1] + 1:
            reasons.append(
                f"the quote span is not consecutive: sentence {j} follows {pointer.sentences[n - 1]}"
            )
        issues += [PointerIssue(index, k, j, reason) for reason in reasons]
        seen.add(j)
    return issues


def check_pointers(segments: Sequence[Segment], chunks: Sequence[Chunk]) -> list[PointerIssue]:
    """Every failing pointer of the answer, in segment order; ``[]`` when all resolve."""
    return [
        issue
        for i, segment in enumerate(segments)
        if isinstance(segment, QuotePointer | CitePointer)
        for issue in check_pointer(i, segment, chunks)
    ]
