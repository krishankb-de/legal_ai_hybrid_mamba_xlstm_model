"""The retrieval-grounded SFT format (plan P3-Y; decision 7): one prompt, pointer answers.

The prompt is **one document** -- no ``<|endoftext|>`` inside it, so per-document positions and the
recurrent caches see question, passages and answer as one sequence::

    <|question|>Wann darf der Vermieter kündigen?
    <|passage|><|c1|>BGB §573
    <|s1|>(1) Der Vermieter kann nur kündigen, wenn ... <|s2|>Die Kündigung zum Zwecke der ...<|/passage|>
    <|passage|><|c2|>BGB §573c
    <|s1|>...<|/passage|>
    <|answer|>

The target mixes free text with pointers and ends on the EOS::

    Er braucht ein berechtigtes Interesse: <|q|><|c1|><|s1|> Die Frist regelt <|cite|><|c2|>.<|endoftext|>

``<|q|><|cK|><|sJ|>`` quotes sentence J of passage K verbatim (the renderer, P7, replaces it by the
sentence and its citation label; the model never writes the sentence itself), ``<|cite|><|cK|>``
cites passage K, and ``<|unanswerable|>`` is the whole answer when the passages do not support one.
At most 16 passages of at most 64 sentences (``<|c1|>..<|c16|>``, ``<|s1|>..<|s64|>``).

``encode`` inserts the special tokens as ids and tokenizes every piece of content with
``encode_document``, so no content can turn into a pointer; the prompt is masked from the loss.
``render`` and ``parse`` are the string forms and round-trip.
"""

import re
from dataclasses import dataclass, field

from lexhybrid.data.tokenizer import EOS_ID, EOS_TOKEN, N_PASSAGES, N_SENTENCES, encode_document, pointer_ids

Q, CITE, UNANSWERABLE = "<|q|>", "<|cite|>", "<|unanswerable|>"
PASSAGE_OPEN, PASSAGE_CLOSE, QUESTION, ANSWER = "<|passage|>", "<|/passage|>", "<|question|>", "<|answer|>"


@dataclass(frozen=True)
class Passage:
    citation_id: str
    sentences: tuple[str, ...]


@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True)
class Quote:
    passage: int  # 1-based, as the tokens are numbered
    sentence: int


@dataclass(frozen=True)
class Cite:
    passage: int


@dataclass(frozen=True)
class Example:
    question: str
    passages: tuple[Passage, ...]
    answer: tuple = field(default=())  # Text / Quote / Cite segments; () = unanswerable

    @property
    def unanswerable(self) -> bool:
        return not self.answer


def validate(ex: Example) -> None:
    if not 1 <= len(ex.passages) <= N_PASSAGES:
        raise ValueError(f"{len(ex.passages)} passages; the format holds 1..{N_PASSAGES}")
    for k, p in enumerate(ex.passages, start=1):
        if not 1 <= len(p.sentences) <= N_SENTENCES:
            raise ValueError(f"passage {k}: {len(p.sentences)} sentences; the format holds 1..{N_SENTENCES}")
    pieces = [
        ex.question,
        *(p.citation_id for p in ex.passages),
        *(s for p in ex.passages for s in p.sentences),
    ]
    pieces += [seg.text for seg in ex.answer if isinstance(seg, Text)]
    if any("<|" in s for s in pieces):
        raise ValueError("content may not contain special-token markup ('<|')")
    for seg in ex.answer:
        if isinstance(seg, Quote | Cite) and not 1 <= seg.passage <= len(ex.passages):
            raise ValueError(f"pointer to passage {seg.passage}; there are {len(ex.passages)}")
        if isinstance(seg, Quote) and not 1 <= seg.sentence <= len(ex.passages[seg.passage - 1].sentences):
            raise ValueError(f"quote of sentence {seg.sentence} of passage {seg.passage}: no such sentence")


def render(ex: Example) -> tuple[str, str]:
    """``(prompt, target)`` as strings with the special-token markup."""
    validate(ex)
    prompt = [f"{QUESTION}{ex.question}\n"]
    for k, p in enumerate(ex.passages, start=1):
        body = " ".join(f"<|s{j}|>{s}" for j, s in enumerate(p.sentences, start=1))
        prompt.append(f"{PASSAGE_OPEN}<|c{k}|>{p.citation_id}\n{body}{PASSAGE_CLOSE}\n")
    prompt.append(ANSWER)
    if ex.unanswerable:
        return "".join(prompt), UNANSWERABLE + EOS_TOKEN
    target = []
    for seg in ex.answer:
        if isinstance(seg, Text):
            target.append(seg.text)
        elif isinstance(seg, Quote):
            target.append(f"{Q}<|c{seg.passage}|><|s{seg.sentence}|>")
        else:
            target.append(f"{CITE}<|c{seg.passage}|>")
    return "".join(prompt), "".join(target) + EOS_TOKEN


_PASSAGE = re.compile(
    re.escape(PASSAGE_OPEN) + r"<\|c(\d+)\|>(.*?)\n(.*?)" + re.escape(PASSAGE_CLOSE) + r"\n", re.S
)
_SENTENCE = re.compile(r"<\|s(\d+)\|>(.*?)(?= <\|s\d+\|>|$)", re.S)
_POINTER = re.compile(r"<\|q\|><\|c(\d+)\|><\|s(\d+)\|>|<\|cite\|><\|c(\d+)\|>")


def parse(text: str) -> Example:
    """The ``Example`` back from ``prompt + target`` (as ``render`` writes them)."""
    if not text.startswith(QUESTION) or ANSWER not in text or not text.endswith(EOS_TOKEN):
        raise ValueError("not an SFT example: question, answer marker and final EOS expected")
    prompt, target = text[: -len(EOS_TOKEN)].split(ANSWER, 1)
    question, rest = prompt[len(QUESTION) :].split("\n", 1)
    passages = []
    for k, (num, citation, body) in enumerate(_PASSAGE.findall(rest), start=1):
        if int(num) != k:
            raise ValueError(f"passage {num} out of order")
        sentences = [(int(j), s) for j, s in _SENTENCE.findall(body)]
        if [j for j, _ in sentences] != list(range(1, len(sentences) + 1)):
            raise ValueError(f"passage {k}: sentences out of order")
        passages.append(Passage(citation, tuple(s for _, s in sentences)))
    if target == UNANSWERABLE:
        return Example(question, tuple(passages))
    answer, pos = [], 0
    for m in _POINTER.finditer(target):
        if m.start() > pos:
            answer.append(Text(target[pos : m.start()]))
        answer.append(Quote(int(m.group(1)), int(m.group(2))) if m.group(1) else Cite(int(m.group(3))))
        pos = m.end()
    if pos < len(target):
        answer.append(Text(target[pos:]))
    ex = Example(question, tuple(passages), tuple(answer))
    validate(ex)
    return ex


def encode(ex: Example, tok) -> dict[str, list[int]]:
    """Token ids of one example: ``input_ids``, ``labels`` (-100 over the prompt) and ``doc_ids``
    (all 0: one document); exactly one EOS, at the end."""
    validate(ex)
    ids = pointer_ids(tok)
    text = lambda s: encode_document(tok, s)  # noqa: E731 -- content can never become a special token
    nl = text("\n")
    prompt = [ids["question"], *text(ex.question), *nl]
    for k, p in enumerate(ex.passages, start=1):
        prompt += [ids["passage_open"], ids["passages"][k - 1], *text(p.citation_id), *nl]
        for j, s in enumerate(p.sentences, start=1):
            prompt += ([*text(" ")] if j > 1 else []) + [ids["sentences"][j - 1], *text(s)]
        prompt += [ids["passage_close"], *nl]
    prompt.append(ids["answer"])
    if ex.unanswerable:
        target = [ids["unanswerable"]]
    else:
        target = []
        for seg in ex.answer:
            if isinstance(seg, Text):
                target += text(seg.text)
            elif isinstance(seg, Quote):
                target += [ids["q"], ids["passages"][seg.passage - 1], ids["sentences"][seg.sentence - 1]]
            else:
                target += [ids["cite"], ids["passages"][seg.passage - 1]]
    target.append(EOS_ID)
    return {
        "input_ids": prompt + target,
        "labels": [-100] * len(prompt) + target,
        "doc_ids": [0] * (len(prompt) + len(target)),
    }


_ABBREVIATION = re.compile(
    r"(?:\b(?:Abs|Art|Nr|Ziff|lit|Buchst|bzw|vgl|ggf|gem|Rn|Rz|S|z\. B|d\. h|u\. a|i\. V\. m)|\d)\.$"
)


def split_sentences(text: str) -> tuple[str, ...]:
    """German sentences of a passage: split after ``.``, ``!`` or ``?`` before an upper-case letter or
    an opening parenthesis, not after an abbreviation (``Abs.``, ``Nr.``) or a number (``1.``)."""
    out, start = [], 0
    for m in re.finditer(r"[.!?](?=\s+[A-ZÄÖÜ(])", text):
        if _ABBREVIATION.search(text[start : m.end()]):
            continue
        out.append(" ".join(text[start : m.end()].split()))
        start = m.end()
    tail = " ".join(text[start:].split())
    return tuple(s for s in [*out, tail] if s)
