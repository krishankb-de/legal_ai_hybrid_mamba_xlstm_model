"""The model's answer as segments: text, quote pointers, citation pointers (plan P7-G, P7-H, P7-J).

The renderer and the verifier read the same grammar, defined once here. The answer is the target
after ``<|answer|>``, optionally ending in the EOS token. Pointers:

* ``<|q|><|cK|><|sJ|>…``: a quote of one or more sentences ``J`` of passage ``K``, as the P3-Y format
  writes it (one sentence) and the pointer FSM allows it (a span, ``lexhybrid/decoding/pointer_constraints.py``);
* ``<|cite|><|cK|>`` optionally followed by ``<|sJ|>…``: a citation of passage ``K``, or of those of its
  sentences. ``sft_format.Cite`` writes it without a sentence, and the FSM ends it after one;
* ``<|unanswerable|>``: the whole answer, alone.

``K`` and ``J`` are 1-based. Any other ``<|…|>`` in the answer, a pointer cut short (``<|q|>`` without
a passage, a quote without a sentence) or text around ``<|unanswerable|>`` raises ``PointerSyntaxError``.
Whether ``K`` and ``J`` exist in the retrieved set is the verifier's check
(``lexhybrid/verifier/pointers.py``), not the parser's.
"""

import re
from dataclasses import dataclass

from lexhybrid.data.sft_format import CITE, UNANSWERABLE, Q
from lexhybrid.data.tokenizer import EOS_TOKEN

_TOKEN = re.compile(r"<\|[^|<>]*\|>")
_PASSAGE = re.compile(r"<\|c(\d+)\|>")
_SENTENCE = re.compile(r"<\|s(\d+)\|>")


class PointerSyntaxError(ValueError):
    pass


@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True)
class QuotePointer:
    passage: int
    sentences: tuple[int, ...]


@dataclass(frozen=True)
class CitePointer:
    passage: int
    sentences: tuple[int, ...] = ()


@dataclass(frozen=True)
class Unanswerable:
    pass


Segment = Text | QuotePointer | CitePointer | Unanswerable


def parse_answer(answer: str) -> list[Segment]:
    """The answer's segments in order; adjacent text is kept as one ``Text``."""
    if answer.endswith(EOS_TOKEN):
        answer = answer[: -len(EOS_TOKEN)]
    if answer.strip() == UNANSWERABLE:
        return [Unanswerable()]
    if UNANSWERABLE in answer:
        raise PointerSyntaxError("<|unanswerable|> must be the whole answer")
    tokens = list(_TOKEN.finditer(answer))
    segments: list[Segment] = []
    pos, i = 0, 0

    def text(upto: int) -> None:
        if upto > pos:
            if segments and isinstance(segments[-1], Text):
                segments[-1] = Text(segments[-1].text + answer[pos:upto])
            else:
                segments.append(Text(answer[pos:upto]))

    while i < len(tokens):
        tok = tokens[i]
        if tok.group() not in (Q, CITE):
            raise PointerSyntaxError(f"stray token {tok.group()!r} at {tok.start()}")
        text(tok.start())
        kind, end = tok.group(), tok.end()
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        passage = _PASSAGE.fullmatch(nxt.group()) if nxt is not None and nxt.start() == end else None
        if passage is None:
            raise PointerSyntaxError(f"{kind} at {tok.start()} is not followed by a passage <|cK|>")
        end, i = nxt.end(), i + 2
        sentences = []
        while i < len(tokens) and tokens[i].start() == end and (s := _SENTENCE.fullmatch(tokens[i].group())):
            sentences.append(int(s.group(1)))
            end, i = tokens[i].end(), i + 1
        k = int(passage.group(1))
        if kind == Q:
            if not sentences:
                raise PointerSyntaxError(f"quote of passage {k} at {tok.start()} names no sentence")
            segments.append(QuotePointer(k, tuple(sentences)))
        else:
            segments.append(CitePointer(k, tuple(sentences)))
        pos = end
    text(len(answer))
    return segments


def pointers(segments: list[Segment]) -> list[QuotePointer | CitePointer]:
    return [s for s in segments if isinstance(s, QuotePointer | CitePointer)]
