#!/usr/bin/env python3
"""Build the judge's gold-labelled calibration and validation items from the evaluation sets.

    .venv/bin/python scripts/build_judge_calibration.py --qa-dir data/evalsets/legalquad \\
        --qa-dir data/evalsets/gerlayqa --retrieval-dir data/evalsets/gerlerb \\
        --retrieval-dir data/evalsets/gerdalir --nli-pairs data/manifests/nli_calibration.jsonl

Set directories are what ``lexhybrid.data.evalsets`` writes. One JSONL item per line (``id``, ``type``,
``variant``, ``group``, ``split``, ``question``, ``answer``, ``references``, ``contexts``, ``label``);
ids and groups start with the set name. Every type is put to the judge with the prompt that scores
answers in production (``scripts/score_judge_calibration.py``). From the QA sets (items with a
non-empty gold answer):

* **correctness**: the gold answer as the candidate (label 1); another question's gold answer,
  preferring one that rests on a shared provision (label 0); and the gold answer perturbed (label 0):
  one § number changed (``section``), one other number changed (``number``; never one inside a §
  reference such as ``§§ 535, 536`` or ``§ 573 Abs. 2``, nor a list marker such as ``1.``), or the
  statement negated (``negation``: ``nicht`` dropped, ``kein`` -> ``ein``, or ``nicht`` after the
  first auxiliary), each only where the answer has something to change.
* **answer_relevancy**: the question with its gold answer (1) and with another question's gold answer,
  preferring one on other provisions (0).
* **context_precision** (items with a context, with the reference answers): the gold context (1) and
  the best BM25 match among the other contexts of the set that does not hold the answer (0).
* **context_recall** (items with a context): the gold answer against its gold context (1) and against
  a random other context that does not hold it (0).

From the retrieval sets, **context_relevance** (no reference answer: the sets have none): a
qrel-relevant passage (1) and the best BM25 match over the set's corpus not judged relevant (0).
From the NLI pairs (the file's ``review`` must be ``done``), **faithfulness**: premise as the one
passage, claim as the answer (with the pair's ``question`` where it has one), the pair's label.

The split is fixed by a hash of the item's group, the source sibling questions share: a QA item's
context (``context_id``), else its first gold provision in sorted order, else the item; a retrieval
query's first relevant passage; an NLI pair's provision. So every item on one context or provision
lands in the same split, the same in every run: ``validation`` (reported) for
``--validation-share`` of the groups, else ``calibration`` (prompt tuning only). Negatives built from
other questions or passages come from the same split, so no context or passage appears in both.
The items hold evaluation text only, so the default output is under the git-ignored
``data/evalsets/``.
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from lexhybrid.retrieval.bm25 import BM25Index

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "evalsets" / "judge_calibration.jsonl"
BM25_DEPTH = 50
_SECTION = re.compile(r"(§§?\s*)(\d+)")
_REFERENCE = re.compile(  # a § reference with its lists and sub-units: "§§ 535, 536", "§ 573 Abs. 2 Satz 1"
    r"§§?\s*\d+[a-z]*(?:(?:\s*(?:,|und|bis|-|–)\s*|\s+(?:Abs\.|Satz|S\.|Nr\.|Alt\.)\s*)\d+[a-z]*)*"
)
_NUMBER = re.compile(r"(?<!\w)\d+(?![.)\d])")  # not a list marker "1."
_NICHT = re.compile(r" ?\bnicht\b(?! nur\b)")
_KEIN = re.compile(r"\b([Kk])ein(e|er|es|en|em)?\b")
_AUX = re.compile(
    r"\b(ist|sind|kann|können|muss|müssen|darf|dürfen|hat|haben|wird|werden|soll|sollen|war|waren)\b"
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def split_of(group: str, validation_share: float) -> str:
    h = int(hashlib.sha256(group.encode()).hexdigest()[:12], 16) / 16**12
    return "validation" if h < validation_share else "calibration"


def perturbations(text: str) -> dict[str, str]:
    """The gold answer with one § number, one other number, or its polarity changed (where possible)."""
    out = {"section": _SECTION.sub(lambda m: f"{m.group(1)}{int(m.group(2)) + 1}", text, count=1)}
    refs = [m.span() for m in _REFERENCE.finditer(text)]
    n = next((m for m in _NUMBER.finditer(text) if not any(a <= m.start() < b for a, b in refs)), None)
    if n:
        out["number"] = f"{text[: n.start()]}{int(n.group()) + 1}{text[n.end() :]}"
    if _NICHT.search(text):
        out["negation"] = _NICHT.sub("", text, count=1)
    elif _KEIN.search(text):
        out["negation"] = _KEIN.sub(lambda m: f"{'E' if m[1] == 'K' else 'e'}in{m[2] or ''}", text, count=1)
    elif "nicht" not in text:  # "nicht nur" stays: a second "nicht" would not negate it
        out["negation"] = _AUX.sub(lambda m: f"{m.group()} nicht", text, count=1)
    return {k: v for k, v in out.items() if v != text}


def _item(type_: str, variant: str, label: int, *, group: str, qid: str, question: str, answer: str = "",
          references=(), contexts=()) -> dict:  # fmt: skip
    return {"id": f"{qid}:{type_}:{variant}", "type": type_, "variant": variant, "group": group,
            "question": question, "answer": answer, "references": list(references),
            "contexts": list(contexts), "label": label}  # fmt: skip


def qa_items(name: str, rows: list[dict], share: float) -> list[dict]:
    qa = []
    for row in rows:
        answers = [a for a in row["answers"] if a.strip()]
        if not answers:
            continue
        qid = str(row["id"]) if str(row["id"]).startswith(f"{name}:") else f"{name}:{row['id']}"
        # shortcut: one provision per item, so items sharing only a later one can split; a union over
        # all of them would chain most of the BGB into one group
        key = row.get("context_id") or (min(row["gold"]) if row.get("gold") else qid)
        group = f"{name}:{key}" if key != qid else qid
        qa.append({**row, "id": qid, "answers": answers, "group": group, "split": split_of(group, share)})
    contexts = {r["context_id"]: r["context"] for r in qa if r.get("context") and r.get("context_id")}
    ranked = [[] for _ in qa]
    if contexts:
        index = BM25Index.build([SimpleNamespace(id=c, text=t) for c, t in contexts.items()])
        ranked = index.query_many([r["question"] for r in qa], k=BM25_DEPTH)
    items = []
    for r, hits in zip(qa, ranked, strict=True):
        gold, provisions, rng = r["answers"][0], set(r.get("gold") or ()), random.Random(r["id"])
        q = dict(group=r["group"], qid=r["id"], question=r["question"], references=r["answers"])
        same = [o for o in qa if o["split"] == r["split"] and o["answers"][0] not in r["answers"]]
        shared = [o for o in same if provisions & set(o.get("gold") or ())]
        unshared = [o for o in same if not provisions & set(o.get("gold") or ())]
        items.append(_item("correctness", "gold", 1, answer=gold, **q))
        for variant, answer in perturbations(gold).items():
            items.append(_item("correctness", variant, 0, answer=answer, **q))
        if same:
            other = rng.choice(shared or same)["answers"][0]
            items.append(_item("correctness", "other_answer", 0, answer=other, **q))
            items.append(_item("answer_relevancy", "gold", 1, answer=gold, **q))
            other = rng.choice(unshared or same)["answers"][0]
            items.append(_item("answer_relevancy", "other_answer", 0, answer=other, **q))
        if not r.get("context"):
            continue
        others = {c: t for c, t in contexts.items() if c != r["context_id"] and gold not in t
                  and split_of(f"{name}:{c}", share) == r["split"]}  # fmt: skip
        items.append(_item("context_precision", "gold_context", 1, contexts=[r["context"]], **q))
        hard = next((others[h.chunk_id] for h in hits if h.chunk_id in others), None)
        if hard is not None:
            items.append(_item("context_precision", "bm25_negative", 0, contexts=[hard], **q))
        items.append(_item("context_recall", "gold_context", 1, contexts=[r["context"]], **q))
        if others:
            unrelated = others[rng.choice(sorted(others))]
            items.append(_item("context_recall", "unrelated", 0, contexts=[unrelated], **q))
    return items


def relevance_items(
    name: str, queries: list[dict], corpus: list[dict], qrels: list[dict], share: float
) -> list[dict]:
    texts = {str(p["id"]): f"{p['title']}\n{p['text']}" if p.get("title") else p["text"] for p in corpus}
    rel: dict[str, dict[str, int]] = {}
    for r in qrels:
        rel.setdefault(str(r["query_id"]), {})[str(r["passage_id"])] = int(r["relevance"])
    judged = [
        (q, sorted(p for p, g in rel.get(str(q["id"]), {}).items() if g > 0 and p in texts)) for q in queries
    ]
    judged = [(q, relevant) for q, relevant in judged if relevant]
    if not judged:
        return []
    index = BM25Index.build([SimpleNamespace(id=p, text=t) for p, t in texts.items()])
    items = []
    for (q, relevant), hits in zip(
        judged, index.query_many([q["text"] for q, _ in judged], k=BM25_DEPTH), strict=True
    ):
        group, split = f"{name}:{relevant[0]}", split_of(f"{name}:{relevant[0]}", share)
        candidates = (h.chunk_id for h in hits if h.chunk_id not in relevant)
        negative = next((p for p in candidates if split_of(f"{name}:{p}", share) == split), None)
        if negative is None:
            continue
        ctx = dict(group=group, qid=f"{name}:{q['id']}", question=q["text"])
        items.append(_item("context_relevance", "relevant", 1, contexts=[texts[relevant[0]]], **ctx))
        items.append(_item("context_relevance", "bm25_negative", 0, contexts=[texts[negative]], **ctx))
    return items


def faithfulness_items(path: Path) -> list[dict]:
    lines = read_jsonl(path)
    if not lines or "info" not in lines[0]:
        raise SystemExit(f"{path}: the first line must be the info block")
    if lines[0]["info"].get("review") != "done":
        raise SystemExit(
            f"{path}: review is {lines[0]['info'].get('review')!r}; a person checks the labels first"
        )
    return [_item("faithfulness", p["kind"], int(p["label"]), group=f"nli:{p['source']}", qid=f"nli:{p['id']}",
                  question=p.get("question", ""), answer=p["hypothesis"], contexts=[p["premise"]])
            for p in lines[1:]]  # fmt: skip


def _name(directory: Path) -> str:
    return json.loads((directory / "info.json").read_text(encoding="utf-8"))["name"]


def build(qa_dirs, retrieval_dirs, nli_pairs: Path | None, validation_share: float) -> list[dict]:
    items = [i for d in qa_dirs for i in qa_items(_name(d), read_jsonl(d / "items.jsonl"), validation_share)]
    for d in retrieval_dirs:
        files = (read_jsonl(d / f) for f in ("queries.jsonl", "corpus.jsonl", "qrels.jsonl"))
        items += relevance_items(_name(d), *files, validation_share)
    if nli_pairs is not None:
        items += faithfulness_items(nli_pairs)
    duplicates = [i for i, n in Counter(i["id"] for i in items).items() if n > 1]
    if duplicates:
        raise SystemExit(f"duplicate item ids (one set given twice?): {duplicates[:3]}")
    for item in items:
        item["split"] = split_of(item["group"], validation_share)
    return items


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--qa-dir", type=Path, action="append", default=[])
    p.add_argument("--retrieval-dir", type=Path, action="append", default=[])
    p.add_argument("--nli-pairs", type=Path, default=None)
    p.add_argument("--validation-share", type=float, default=0.5)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    items = build(args.qa_dir, args.retrieval_dir, args.nli_pairs, args.validation_share)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    for (kind, split), n in sorted(Counter((i["type"], i["split"]) for i in items).items()):
        print(f"JUDGE-CAL {kind} {split} items={n}")
    print(f"JUDGE-CAL {len(items)} items -> {args.out}")
    return 0 if items else 1


if __name__ == "__main__":
    raise SystemExit(main())
