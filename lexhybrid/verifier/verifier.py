"""The answer verifier: claim by claim, pointer -> quote -> NLI, behind a party-identity rule.

``verify(question, answer, chunks, nli, ner)`` takes the model's raw pointer answer and the retrieved
passages in prompt order (passage ``K`` is ``chunks[K - 1]``) and returns what the reader sees:

1. **Parse** (``segments.parse_answer``): a malformed answer is refused; ``<|unanswerable|>`` is the
   model's own refusal.
2. **Party identity** (no person is ever named): a question asking who a party was
   (``Wer war der Kläger?``, ``Wie heißt die Beklagte?``, ``Name des Antragstellers``; see
   ``asks_for_party``) is refused, and so is an answer whose text or quoted sentences hold a ``[PER_n]``
   placeholder or a name the LER model finds (``ner(text) -> [person names]``, optional). Refusal
   reasons never repeat a name. A question pattern matches within one clause, so ``Wer ist
   beweispflichtig, der Kläger oder der Beklagte?`` is answered, while a generic ``Wer ist Partei
   eines Mietvertrags?`` is refused: the rule errs towards refusing.
3. **Claims** (``split_claims``): a claim is the run of segments up to and including a sentence-final
   ``.``, ``!`` or ``?`` in a text segment (one followed by whitespace and an upper-case letter, a digit,
   ``(``, ``„`` or ``§``, or ending the segment; not after an abbreviation such as ``Abs.``, a lone
   lower-case letter (``z. B.``) or a one- or two-digit ordinal, so a claim is cut too late rather than
   too early). A pointer belongs to the claim it sits in, or to the claim it directly follows (only
   whitespace between). Text after the last mark is a claim of its own.
4. **Each claim** passes when it has at least one pointer and says something beyond its citations
   (else it is unsupported: ``(BGB §568)`` alone is no answer, and NLI is not asked about an empty
   hypothesis), every pointer
   resolves (``pointers.check_pointers``), every quote rendered by ``render_quote`` equals its source
   (``quotes.check_quote``: this guards against renderer drift), and the NLI model gives the
   hypothesis -- the claim's text with each quote inline as ``„<source>“``, citations left out --
   an entailment probability of at least ``threshold`` (``ENTAILMENT_THRESHOLD``)
   given the premise: the quoted and cited sentences, or the whole chunk text (``Chunk.text``) for a
   citation of a passage without sentences.
5. **Result**: failing claims are removed. If the first claim (the main claim) fails, the answer is
   ``REFUSAL``. The kept segments are rendered (``render_segments``: the body, then ``"\\n\\n" + NOTICE``)
   and their quotes read back from the rendered text (``quotes.check_quotes``); a mismatch refuses.

Every refusal renders as ``REFUSAL + "\\n\\n" + NOTICE``. ``citation_precision`` is the share of claims
with a pointer that passed (``None`` when no claim has one); it is computed for a refused main claim
too, since it measures the model, and ``pooled_citation_precision`` pools it over a run. The verifier
log is ``VerifierResult.log_row()``: pointers, reasons and scores, never answer text.
"""

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from lexhybrid.renderer.render import (
    NOTICE,
    REFUSAL,
    RenderError,
    pointer_text,
    render_cite,
    render_quote,
    render_segments,
)
from lexhybrid.renderer.segments import (
    CitePointer,
    PointerSyntaxError,
    QuotePointer,
    Segment,
    Text,
    Unanswerable,
    parse_answer,
)
from lexhybrid.retrieval.chunking import Chunk
from lexhybrid.verifier.nli import ENTAILMENT_THRESHOLD, NLI
from lexhybrid.verifier.pointers import check_pointers
from lexhybrid.verifier.quotes import CLOSE, OPEN, check_quote, check_quotes, normalise, source_text

NER = Callable[[str], list[str]]
REFUSED = REFUSAL + "\n\n" + NOTICE
PER_PLACEHOLDER = re.compile(r"\[PER_\d+\]")

_ROLE = (
    r"\w*(?:kläger|beklagt|partei|antragsteller|antragsgegner|angeklagt|beschuldigt|beschwerdeführer"
    r"|beschwerdegegner|gesuchsteller|gesuchsgegner|berufungswerber|revisionswerber)\w*"
)
_ROLE_EN = r"(?:plaintiff|defendant|claimant|respondent|appellant|applicant|accused|part(?:y|ies))"
_CLAUSE = r"[^?!,;:]*?"  # the rest of one clause: "Wer ist beweispflichtig, der Kläger ...?" is not asked
PARTY_QUESTION = re.compile(
    "|".join(
        [
            rf"\bwer\s+(?:war|ist|sind|waren|wäre|wären)\b{_CLAUSE}\b{_ROLE}",
            rf"\bwie\s+(?:hei(?:ß|ss)t|hie(?:ß|ss)|hei(?:ß|ss)en|hie(?:ß|ss)en)\b{_CLAUSE}\b{_ROLE}",
            rf"(?<!\bim\s)\bnamen?\s+(?:des|der|den|von)\b{_CLAUSE}\b{_ROLE}",
            rf"\bwelchen\s+namen\b{_CLAUSE}\b{_ROLE}",
            rf"\bwer\s+(?:hat|hatte|haben|hatten)\b{_CLAUSE}\b(?:geklagt|verklagt|angeklagt)\b",
            r"\bwer\s+(?:klagt|klagte|wurde\s+(?:verklagt|angeklagt))\b",
            rf"\bwho\s+(?:is|was|are|were)\b{_CLAUSE}\b{_ROLE_EN}\b",
            rf"\bnames?\s+of\s+the\s+{_ROLE_EN}\b",
        ]
    ),
    re.IGNORECASE,
)

_MARK = re.compile(r"[.!?]+[“”\"'»)\]]*")
_NOT_AN_END = re.compile(  # an abbreviation, a lone lower-case letter ("z." of "z. B."), an ordinal
    r"(?:\b(?:Abs|Art|Nr|Nrn|Ziff|lit|Buchst|bzw|vgl|ggf|gem|Rn|Rz|S|Bd|Kap|Aufl|Dr|Prof|ca|insb|sog|"
    r"z\. B|i\. V|[a-zäöüß])|(?<!\d)\d{1,2})\.$"
)
_LOOKBACK = 8  # covers the longest abbreviation ("Buchst."); \b and (?<!\d) still see the text before


@dataclass(frozen=True)
class ClaimCheck:
    index: int  # 0 is the main claim
    text: str  # the claim as the reader would see it (whitespace-normalised)
    pointers: tuple[str, ...]  # as the model wrote them: "<|q|><|c1|><|s2|>"
    passed: bool
    reason: str | None  # why it failed; None when it passed
    entailment: float | None = None  # the NLI probability, when the claim reached that step


@dataclass(frozen=True)
class VerifierResult:
    rendered: str
    refused: bool
    reason: str | None
    claims: tuple[ClaimCheck, ...] = ()
    citation_precision: float | None = None

    def log_row(self) -> dict:
        """The verifier log line: no answer text, so it can be kept beside the manifests."""
        cited = [c for c in self.claims if c.pointers]
        return {
            "refused": self.refused,
            "reason": self.reason,
            "citation_precision": self.citation_precision,
            "cited_claims": len(cited),
            "passed_cited_claims": sum(c.passed for c in cited),
            "claims": [
                {"index": c.index, "pointers": list(c.pointers), "passed": c.passed, "reason": c.reason,
                 "entailment": c.entailment}
                for c in self.claims
            ],
        }  # fmt: skip


def pooled_citation_precision(results: Iterable[VerifierResult]) -> float | None:
    """Passed cited claims over cited claims, pooled over many answers (``None`` if none cites)."""
    cited = passed = 0
    for result in results:
        for claim in result.claims:
            if claim.pointers:
                cited += 1
                passed += claim.passed
    return passed / cited if cited else None


def asks_for_party(question: str) -> bool:
    """The question asks who a party of a case is (German or English patterns)."""
    return PARTY_QUESTION.search(question) is not None


def _sentence_ends(text: str) -> list[int]:
    """Offsets just past each sentence-final mark of one text segment (see the module docstring)."""
    ends = []
    for m in _MARK.finditer(text):
        rest = text[m.end() :]
        if rest and not rest[0].isspace():
            continue
        nxt = rest.lstrip()[:1]
        if nxt and not (nxt.isupper() or nxt.isdigit() or nxt in "(„§"):
            continue
        if m.group()[0] == "." and _NOT_AN_END.search(text, max(0, m.start() - _LOOKBACK), m.start() + 1):
            continue
        ends.append(m.end())
    return ends


def split_claims(segments: Sequence[Segment]) -> list[list[tuple[int, Segment]]]:
    """The answer's claims as ``(segment index, segment)`` runs; text segments are cut at the claims'
    ends, so a claim's pieces of text are new ``Text`` segments carrying their segment's index."""
    claims: list[list[tuple[int, Segment]]] = []
    current: list[tuple[int, Segment]] = []
    attach = False  # the last claim ended on a mark and only whitespace has followed since
    for i, segment in enumerate(segments):
        if not isinstance(segment, Text):
            if attach and claims:
                claims[-1] += [*current, (i, segment)]
                current = []
            else:
                current.append((i, segment))
            continue
        ends = _sentence_ends(segment.text)
        bounds = [0, *ends, len(segment.text)]
        for a, b in zip(bounds, bounds[1:]):
            piece = segment.text[a:b]
            if not piece:
                continue
            if piece.strip():
                attach = False
            current.append((i, Text(piece)))
            if b in ends:
                claims.append(current)
                current = []
                attach = True
    if any(not isinstance(s, Text) or s.text.strip() for _, s in current):
        claims.append(current)
    elif claims:
        claims[-1] += current
    return claims


def _resolves(pointer: QuotePointer | CitePointer, chunks: Sequence[Chunk]) -> bool:
    return 1 <= pointer.passage <= len(chunks) and all(
        1 <= j <= len(chunks[pointer.passage - 1].sentences) for j in pointer.sentences
    )


def _party_named(segments: Sequence[Segment], chunks: Sequence[Chunk], ner: NER | None) -> str | None:
    """Why the answer identifies a person, or ``None``: the model's text and the quoted sentences."""
    pieces = [" ".join(s.text for s in segments if isinstance(s, Text))]
    pieces += [
        source_text(chunks[s.passage - 1], s.sentences)
        for s in segments
        if isinstance(s, QuotePointer) and s.sentences and _resolves(s, chunks)
    ]
    for piece in pieces:
        if m := PER_PLACEHOLDER.search(piece):
            return f"the answer contains the person placeholder {m.group()}"
    if ner is not None and any(piece.strip() and ner(piece) for piece in pieces):
        return "the answer names a person (LER)"
    return None


def _shown(segment: Segment, chunks: Sequence[Chunk]) -> str:
    """One segment as the reader sees it; a pointer that does not resolve as the model wrote it."""
    if isinstance(segment, Text):
        return segment.text
    try:
        chunk = chunks[segment.passage - 1] if 1 <= segment.passage <= len(chunks) else None
        if chunk is not None and isinstance(segment, QuotePointer):
            return render_quote(chunk, segment.sentences)
        if chunk is not None and isinstance(segment, CitePointer):
            return render_cite(chunk, segment.sentences)
    except RenderError:
        pass
    return pointer_text(segment)


def _premise(pointer: QuotePointer | CitePointer, chunk: Chunk) -> str:
    return source_text(chunk, pointer.sentences) if pointer.sentences else chunk.text


def _hypothesis(claim: Sequence[tuple[int, Segment]], chunks: Sequence[Chunk]) -> str:
    parts = []
    for _, s in claim:
        if isinstance(s, Text):
            parts.append(s.text)
        elif isinstance(s, QuotePointer):
            parts.append(f"{OPEN}{source_text(chunks[s.passage - 1], s.sentences)}{CLOSE}")
    return normalise("".join(parts))


def check_claim(
    index: int,
    claim: Sequence[tuple[int, Segment]],
    chunks: Sequence[Chunk],
    nli: NLI,
    threshold: float = ENTAILMENT_THRESHOLD,
) -> ClaimCheck:
    """One claim (``split_claims``) through pointer -> quote -> NLI."""
    text = normalise("".join(_shown(s, chunks) for _, s in claim))
    found = [(i, s) for i, s in claim if isinstance(s, QuotePointer | CitePointer)]
    pointers = tuple(pointer_text(s) for _, s in found)
    if not found:
        return ClaimCheck(index, text, (), False, "unsupported: the claim cites no passage")
    if bad := check_pointers([s for _, s in found], chunks):
        return ClaimCheck(index, text, pointers, False, f"pointer: {bad[0].reason}")
    for _, s in found:
        if isinstance(s, QuotePointer):
            chunk = chunks[s.passage - 1]
            problem = check_quote(chunk, s.sentences, render_quote(chunk, s.sentences))
            if problem:
                return ClaimCheck(index, text, pointers, False, f"quote: {problem}")
    hypothesis = _hypothesis(claim, chunks)
    if not any(ch.isalnum() for ch in hypothesis):
        return ClaimCheck(index, text, pointers, False, "unsupported: the claim states nothing but citations")
    premise = "\n".join(dict.fromkeys(_premise(s, chunks[s.passage - 1]) for _, s in found))
    p = float(nli(premise, hypothesis))
    if not p >= threshold:
        return ClaimCheck(index, text, pointers, False, f"nli: entailment {p:.3f} < {threshold}", p)
    return ClaimCheck(index, text, pointers, True, None, p)


def _refusal(
    reason: str, claims: tuple[ClaimCheck, ...] = (), precision: float | None = None
) -> VerifierResult:
    return VerifierResult(REFUSED, True, reason, claims, precision)


def _precision(claims: Sequence[ClaimCheck]) -> float | None:
    cited = [c for c in claims if c.pointers]
    return sum(c.passed for c in cited) / len(cited) if cited else None


def verify(
    question: str,
    answer: str,
    chunks: Sequence[Chunk],
    nli: NLI,
    ner: NER | None = None,
    threshold: float = ENTAILMENT_THRESHOLD,
) -> VerifierResult:
    """The verified, rendered answer of the model's raw ``answer`` (see the module docstring)."""
    try:
        segments = parse_answer(answer)
    except PointerSyntaxError as e:
        return _refusal(f"malformed answer: {e}")
    if any(isinstance(s, Unanswerable) for s in segments):
        return _refusal("unanswerable")
    if asks_for_party(question):
        return _refusal("party identity: the question asks who a party is")
    if named := _party_named(segments, chunks, ner):
        return _refusal(f"party identity: {named}")
    claims = split_claims(segments)
    if not claims:
        return _refusal("empty answer")
    checks = tuple(check_claim(n, claim, chunks, nli, threshold) for n, claim in enumerate(claims))
    precision = _precision(checks)
    if not checks[0].passed:
        return _refusal(f"main claim failed: {checks[0].reason}", checks, precision)
    kept = [s for check, claim in zip(checks, claims) if check.passed for _, s in claim]
    rendered = render_segments(kept, chunks)
    if drift := check_quotes(kept, chunks, rendered):
        return _refusal(f"renderer drift: {drift[0].reason}", checks, precision)
    return VerifierResult(rendered, False, None, checks, precision)
