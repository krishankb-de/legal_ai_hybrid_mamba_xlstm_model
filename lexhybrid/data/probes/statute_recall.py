"""Statute recall (plan P3-W): read a provision, read more law, then reproduce its Absatz 2 verbatim.

An item is built from a statute ``Document`` with an Absatz 2 section: the prompt is the provision
itself, then other provisions (seeded order) until the prompt reaches ``context_tokens`` less the
answer, then the question ``<citation> Abs. 2 lautet:``. The answer must be copied from far back
in the context. The model decodes greedily exactly as many tokens as the reference Absatz has;
``exact_match`` compares the whitespace-normalised strings and ``char_f1`` is the F1 of their
character-trigram multisets.
"""

import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from lexhybrid.data.schema import Document


@dataclass
class RecallItem:
    id: str
    citation_id: str
    prompt: str
    target: str


def _norm(text: str) -> str:
    return " ".join(text.split())


def exact_match(prediction: str, target: str) -> bool:
    return _norm(prediction) == _norm(target)


def _trigrams(text: str) -> Counter:
    s = _norm(text)
    return Counter(s[i : i + 3] for i in range(max(len(s) - 2, 1))) if s else Counter()


def char_f1(prediction: str, target: str) -> float:
    """F1 of the character-trigram multisets. (Single characters would not do: unrelated German text
    scores 0.74 against a BGB sentence, above a half-correct copy at 0.66; trigrams give 0.14 and 0.66.)"""
    p, t = _trigrams(prediction), _trigrams(target)
    common = sum((p & t).values())
    if not common:
        return 0.0
    precision, recall = common / sum(p.values()), common / sum(t.values())
    return 2 * precision * recall / (precision + recall)


def build_items(
    documents: Sequence[Document],
    encode: Callable[[str], list[int]],
    context_tokens: int = 2048,
    seed: int = 0,
) -> list[RecallItem]:
    """One item per statute that has an Absatz 2 and fits the context with room to spare."""
    statutes = [d for d in documents if d.doc_type in ("statute", "regulation")]
    items = []
    for doc in statutes:
        absatz2 = next((s for s in doc.sections if s.absatz == 2), None)
        if absatz2 is None or not doc.citation_id:
            continue
        question = f"\n\n{doc.citation_id} Abs. 2 lautet:\n"
        budget = context_tokens - len(encode(absatz2.text)) - len(encode(question))
        if len(encode(doc.text)) >= budget:
            continue
        others = [d for d in statutes if d.id != doc.id]
        random.Random(f"{seed}:{doc.id}").shuffle(others)
        parts = [doc.text]
        used = len(encode(doc.text))
        for other in others:
            cost = len(encode("\n\n" + other.text))
            if used + cost > budget:
                break
            parts.append(other.text)
            used += cost
        items.append(RecallItem(doc.id, doc.citation_id, "\n\n".join(parts) + question, absatz2.text))
    return items


@torch.no_grad()
def score(
    model,
    items: Sequence[RecallItem],
    encode: Callable[[str], list[int]],
    decode: Callable[[list[int]], str],
    device: str = "cpu",
) -> dict:
    """Mean exact match and char-F1 over the items (greedy decoding; cached where supported)."""
    from lexhybrid.decoding.generate import greedy, greedy_cached

    run = greedy_cached if model.supports_cached_decode() else greedy
    em, f1 = [], []
    for item in items:
        prompt = torch.tensor([encode(item.prompt)], device=device)
        n = len(encode(item.target))
        out = run(model, prompt, max_new_tokens=n)
        prediction = decode(out[0, prompt.shape[1] :].tolist())
        em.append(exact_match(prediction, item.target))
        f1.append(char_f1(prediction, item.target))
    k = len(items)
    return {
        "exact_match": sum(em) / k if k else float("nan"),
        "char_f1": sum(f1) / k if k else float("nan"),
        "items": k,
    }
