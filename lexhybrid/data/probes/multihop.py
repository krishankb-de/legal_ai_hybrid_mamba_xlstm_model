"""Two-hop cross-reference probe (plan P3-X): the answer needs both passages.

An item takes a provision A that refers to exactly one other provision B of the same act
(``§ 280 Abs. 1`` inside the BGB, ``Art. 97`` inside the OR, ``Artikel 6`` inside the GDPR) and
asks for the title of "the provision A refers to" without naming it. The context holds A, B and
provisions of the same act as distractors, in seeded order. Hop one reads A to learn which provision
is meant; hop two reads that provision's title, which A does not contain. The answer is B's title as
printed (GII ``§ 280 Schadensersatz wegen Pflichtverletzung``, Fedlex's marginal titles, EUR-Lex's
article titles); RIS is left out because most of its provisions carry no title.

Scored by exact match of the first line of a greedy continuation (whitespace-normalised).
"""

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from lexhybrid.data.schema import Document, parse_citation

SOURCES = ("gii", "fedlex", "eurlex")
_TITLE = re.compile(r"^(?:§\s*\d+[a-z]*|Art\.\s*\d+[a-z]*|Artikel\s+\d+[a-z]*)\s+(?P<title>.+)$")
_TAIL = r"(?:\s+(?:Abs\.|Absatz)\s*\d+[a-z]*)?(?:\s+(?:Satz|S\.|Ziff\.|Unterabsatz)\s*\d+)?(?:\s+(?:Nr\.|Nummer|Buchstabe|lit\.)\s*\w+)?"
_GERMAN_REF = re.compile(
    r"(?<![\w§])§\s*(?P<n>\d+[a-z]?)(?![\w.]*\d)"
    + _TAIL
    + r"(?:\s+(?P<code>[A-ZÄÖÜ][A-Za-zÄÖÜäöüß-]*[A-Z][\w-]*))?"
)
_ARTICLE_REF = re.compile(
    r"\b(?:Art\.|Artikel)\s*(?P<n>\d+[a-z]*)"
    + _TAIL
    + r"(?:\s+(?P<code>[A-ZÄÖÜ][A-Za-zÄÖÜäöüß-]*[A-Z][\w-]*|der\s+(?:Richtlinie|Verordnung)|des\s+(?:Beschlusses|Übereinkommens|Vertrags)))?"
)


@dataclass
class MultihopItem:
    id: str
    source_citation: str
    target_citation: str
    context: str
    question: str
    answer: str


# Lists and ranges: "§§ 674, 729", "§§ 535 bis 580", "Artikel 15 bis 22", "Art. 6 und 7".
_LIST = re.compile(
    r"(?:(?<![\w§])§§\s*|\b(?:Art\.|Artikel)\s*)(?P<list>\d+[a-z]*(?:\s*(?:,|und|oder|bis)\s*\d+[a-z]*)+)"
)


def title(doc: Document) -> str | None:
    m = _TITLE.match(doc.text.split("\n", 1)[0])
    return " ".join(m.group("title").split()) if m else None


def same_act_references(doc: Document) -> set[str]:
    """Numbers of the provisions of the same act that ``doc`` refers to (never itself)."""
    code = parse_citation(doc.citation_id).get("code", "")
    pattern = _GERMAN_REF if "§" in doc.citation_id else _ARTICLE_REF
    own = parse_citation(doc.citation_id)
    own_n = own.get("paragraph") or own.get("article")
    refs = set()
    for m in pattern.finditer(doc.text.split("\n", 1)[-1]):  # the body, not the title line
        other = m.group("code")
        if other is None or other == code:
            refs.add(m.group("n"))
    for m in _LIST.finditer(doc.text.split("\n", 1)[-1]):  # several at once: never a unique reference
        refs.update(re.findall(r"\d+[a-z]*", m.group("list")))
    refs.discard(own_n)
    return refs


def build_items(
    documents: Sequence[Document], n_items: int = 200, distractors: int = 2, seed: int = 0
) -> list[MultihopItem]:
    # The ZGB's final title (``ZGB SchlT Art. 1``) numbers its articles afresh; it is left out so that
    # an article number always names one provision of an act.
    docs = [
        d for d in documents
        if d.source in SOURCES and d.citation_id and title(d) and "part" not in parse_citation(d.citation_id)
    ]  # fmt: skip
    by_act: dict[str, dict[str, Document]] = {}
    for d in docs:
        c = parse_citation(d.citation_id)
        by_act.setdefault(c.get("code", ""), {})[c.get("paragraph") or c.get("article")] = d
    rng = random.Random(seed)
    order = sorted(docs, key=lambda d: d.id)
    rng.shuffle(order)
    items = []
    for a in order:
        act = by_act[parse_citation(a.citation_id).get("code", "")]
        refs = [n for n in same_act_references(a) if n in act]
        if len(refs) != 1:
            continue
        b = act[refs[0]]
        answer = title(b)
        if not answer or answer in a.text or answer == title(a):
            continue  # A alone must not give the answer
        pool = [d for n, d in sorted(act.items()) if d.id not in (a.id, b.id) and title(d) != answer]
        if len(pool) < distractors:
            continue
        passages = [a, b, *rng.sample(pool, distractors)]
        rng.shuffle(passages)
        items.append(
            MultihopItem(
                id=f"multihop:{a.id}->{b.id}",
                source_citation=a.citation_id,
                target_citation=b.citation_id,
                context="\n\n".join(p.text for p in passages),
                question=f"\n\n{a.citation_id} verweist auf eine andere Vorschrift. Wie lautet deren Überschrift?\n",
                answer=answer,
            )
        )
        if len(items) == n_items:
            break
    return items


@torch.no_grad()
def score(
    model, items: Sequence[MultihopItem], encode, decode, device: str = "cpu", max_new_tokens: int = 32
) -> dict:
    """Exact match of the first line of a greedy continuation against the title."""
    from lexhybrid.decoding.generate import greedy, greedy_cached

    run = greedy_cached if model.supports_cached_decode() else greedy
    hits = 0
    for item in items:
        prompt = torch.tensor([encode(item.context + item.question)], device=device)
        out = run(model, prompt, max_new_tokens=max_new_tokens)
        first_line = decode(out[0, prompt.shape[1] :].tolist()).strip().split("\n", 1)[0]
        hits += " ".join(first_line.split()) == item.answer
    return {"exact_match": hits / len(items) if items else float("nan"), "items": len(items)}
