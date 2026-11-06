#!/usr/bin/env python3
"""Retrieval quality on benchmark sets: BM25, dense, fused and reranked rankings over each set's own corpus.

    .venv/bin/python scripts/eval_retrieval.py --set-dir data/evalsets/gerlerb \\
        --set-dir data/evalsets/gerdalir --out logs/eval_retrieval/results.json

A set directory is what ``lexhybrid.data.evalsets`` writes (``queries.jsonl``, ``corpus.jsonl``,
``qrels.jsonl``). Every arm ranks the top 100 passages for each judged query: **bm25**
(``BM25Index``), **dense** (bge-m3; the corpus vectors are kept in ``<set>/dense/`` and reused),
**fused** (reciprocal rank fusion of the two) and **reranked** (the fused top 50 re-scored by the
cross-encoder, followed by fused ranks 51-100). Metrics: nDCG@10, R@5/10/20, MRR
(``lexhybrid.eval.retrieval_metrics``). Both benchmarks are German law (jurisdiction DE).
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

from lexhybrid.eval.retrieval_metrics import evaluate
from lexhybrid.retrieval.bm25 import BM25Index
from lexhybrid.retrieval.dense import CONFIG_NAME, DENSE_MODEL, BgeM3Encoder, DenseIndex
from lexhybrid.retrieval.fusion import reciprocal_rank_fusion
from lexhybrid.retrieval.rerank import CANDIDATES, RERANK_MODEL, CrossEncoderScorer, rerank

DEPTH = 100
ARMS = ("bm25", "dense", "fused", "reranked")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def load_set(directory: Path):
    qrels: dict[str, dict[str, int]] = {}
    for r in read_jsonl(directory / "qrels.jsonl"):
        qrels.setdefault(str(r["query_id"]), {})[str(r["passage_id"])] = int(r["relevance"])
    queries = [q for q in read_jsonl(directory / "queries.jsonl") if str(q["id"]) in qrels]
    missing = set(qrels) - {str(q["id"]) for q in queries}
    if missing:  # a judged query without its text cannot be ranked; scoring it 0 would bias every arm
        raise SystemExit(
            f"{directory}: {len(missing)} qrels queries are missing from queries.jsonl, e.g. {sorted(missing)[:3]}"
        )
    passages = [
        SimpleNamespace(id=str(p["id"]), text=f"{p['title']}\n{p['text']}" if p.get("title") else p["text"])
        for p in read_jsonl(directory / "corpus.jsonl")
    ]
    return queries, unique_passages(passages, {pid for r in qrels.values() for pid in r}), qrels


def unique_passages(passages: list, judged: set[str]) -> list:
    """One row per id. A published corpus can repeat ids (GerLeRB's repeats 460 docnos, none judged):
    an exact repeat is dropped, other text stays searchable as ``<id>#<k>``. A judged id with two
    texts has no defined gold passage, so it stops the run."""
    texts: dict[str, set[str]] = {}
    out = []
    for p in passages:
        seen = texts.setdefault(p.id, set())
        if p.text in seen:
            continue
        if seen and p.id in judged:
            raise SystemExit(f"judged passage {p.id} has more than one text")
        out.append(SimpleNamespace(id=f"{p.id}#{len(seen)}", text=p.text) if seen else p)
        seen.add(p.text)
    if len(out) != len(passages):
        print(
            f"corpus: {len(passages)} rows -> {len(out)} passages, {len(texts)} distinct source ids",
            flush=True,
        )
    return out


def dense_index(directory: Path, passages, encode, model: str) -> DenseIndex:
    """The corpus vectors, embedded once: rebuilt when the passages' ids or texts, the model or the
    encoder's truncation changed, or when the cached directory is unreadable."""
    ids = [p.id for p in passages]
    digest = hashlib.sha256("\x1f".join(p.text for p in passages).encode()).hexdigest()
    key = {"model": model, "max_length": getattr(encode, "max_length", None), "texts_sha256": digest}
    if (directory / CONFIG_NAME).exists():
        try:
            index = DenseIndex.load(directory)
        except (ValueError, OSError):
            index = None
        if index is not None and index.ids == ids and all(index.config.get(k) == v for k, v in key.items()):
            return index
    return DenseIndex.build(directory, ids, (p.text for p in passages), encode, **key)


def rank_set(directory: Path, encode, scorer, model: str, depth: int = DEPTH, rerank_top: int = CANDIDATES):
    queries, passages, qrels = load_set(directory)
    ids, texts = [str(q["id"]) for q in queries], [q["text"] for q in queries]
    lexical = BM25Index.build(passages).query_many(texts, k=depth)
    dense = dense_index(directory / "dense", passages, encode, model).query_vectors(encode(texts), k=depth)
    fused = [reciprocal_rank_fusion([a, b], top_n=depth) for a, b in zip(lexical, dense, strict=True)]
    by_id = {p.id: p for p in passages}
    reranked = []
    for text, hits in zip(texts, fused, strict=True):
        head = rerank(text, [by_id[h.chunk_id] for h in hits], scorer, limit=rerank_top)
        reranked.append([r.chunk.id for r in head] + [h.chunk_id for h in hits[rerank_top:]])
    rankings = {
        "bm25": [[h.chunk_id for h in r] for r in lexical],
        "dense": [[h.chunk_id for h in r] for r in dense],
        "fused": [[h.chunk_id for h in r] for r in fused],
        "reranked": reranked,
    }
    return {
        arm: {"jurisdiction": "DE", **evaluate(dict(zip(ids, rankings[arm], strict=True)), qrels)}
        for arm in ARMS
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--set-dir", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dense-model", default=DENSE_MODEL)
    p.add_argument("--reranker", default=RERANK_MODEL)
    p.add_argument("--device", default=None)
    p.add_argument("--depth", type=int, default=DEPTH)
    args = p.parse_args(argv)
    encode = BgeM3Encoder(args.dense_model, device=args.device)
    scorer = CrossEncoderScorer(args.reranker, device=args.device)
    results = {"settings": {"dense_model": args.dense_model, "reranker": args.reranker, "depth": args.depth,
                            "rerank_top": CANDIDATES}}  # fmt: skip
    for directory in args.set_dir:
        start = time.time()
        name = json.loads((directory / "info.json").read_text())["name"]
        results[name] = rank_set(directory, encode, scorer, args.dense_model, args.depth)
        results[name]["seconds"] = round(time.time() - start, 1)
        for arm in ARMS:
            m = results[name][arm]
            print(
                f"RETRIEVAL {name} {arm:8s} DE queries={m['queries']} ndcg@10={m['ndcg@10']:.4f} "
                f"R@5={m['recall@5']:.4f} R@10={m['recall@10']:.4f} R@20={m['recall@20']:.4f} mrr={m['mrr']:.4f}"
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
