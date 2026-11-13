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

Each query's reranked list is saved in ``<set>/rerank/`` as it is made, so a run stopped at its time
limit is finished by the next one. ``--shard i/N`` reranks only every N-th query (from the i-th) and
scores nothing; N shards can run side by side, and a run without ``--shard`` then scores everything.
``--saved-only`` scores each saved file on its own, without a model, to compare shards run on
different machines.
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
from lexhybrid.utils.jsonl import read_complete_lines

DEPTH = 100
ARMS = ("bm25", "dense", "fused", "reranked")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def read_qrels(directory: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for r in read_jsonl(directory / "qrels.jsonl"):
        qrels.setdefault(str(r["query_id"]), {})[str(r["passage_id"])] = int(r["relevance"])
    return qrels


def load_set(directory: Path):
    qrels = read_qrels(directory)
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
    key = {
        "model": model,
        "max_length": getattr(encode, "max_length", None),
        "texts_sha256": texts_digest(passages),
    }
    if (directory / CONFIG_NAME).exists():
        try:
            index = DenseIndex.load(directory)
        except (ValueError, OSError):
            index = None
        if index is not None and index.ids == ids and all(index.config.get(k) == v for k, v in key.items()):
            return index
    return DenseIndex.build(directory, ids, (p.text for p in passages), encode, **key)


def texts_digest(passages) -> str:
    return hashlib.sha256("\x1f".join(p.text for p in passages).encode()).hexdigest()


def reranked_lists(directory: Path, ids, texts, fused, by_id, scorer, rerank_top: int, shard=None) -> list:
    """Per query: the fused top ``rerank_top`` re-scored, then the rest of the fused ranks. Each list
    is appended to ``directory`` when made, and a saved list is reused only for the same candidate set
    (in any order: query vectors from another GPU can swap near-tied fused ranks, and the reranker's
    order depends only on its scores). ``shard=(i, n)`` reranks queries i, i+n, ... and reads only its
    own file, since the other shards may still be writing theirs. It returns None for the queries it
    skips."""
    directory.mkdir(parents=True, exist_ok=True)
    own = directory / (f"shard_{shard[0]}_of_{shard[1]}.jsonl" if shard else "rest.jsonl")
    saved: dict[str, list[dict]] = {}
    for f in [own] if shard else sorted(directory.glob("*.jsonl")):
        if f.exists():
            for row in map(json.loads, read_complete_lines(f)):
                saved.setdefault(row["q"], []).append(row)
    out = []
    with open(own, "a", encoding="utf-8") as log:
        for n, (qid, text, hits) in enumerate(zip(ids, texts, fused, strict=True)):
            cand = [h.chunk_id for h in hits[:rerank_top]]
            row = next((r for r in saved.get(qid, []) if set(r["cand"]) == set(cand)), None)
            if row is None:
                if shard and n % shard[1] != shard[0]:
                    out.append(None)
                    continue
                head = rerank(text, [by_id[c] for c in cand], scorer, limit=rerank_top)
                row = {"q": qid, "cand": cand, "ranked": [r.chunk.id for r in head]}
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                log.flush()
            out.append(row["ranked"] + [h.chunk_id for h in hits[rerank_top:]])
    return out


def saved_rerank_metrics(directory: Path) -> dict:
    """Metrics of each saved rerank file on its own (its reranked top only), loading no model. Shards
    hold interleaved queries, so their means should agree; one far below the others points at the
    machine that wrote it."""
    qrels = read_qrels(directory)
    out = {}
    for f in sorted(directory.glob("rerank/*/*.jsonl")):
        ranked = {row["q"]: row["ranked"] for row in map(json.loads, read_complete_lines(f))}
        out[f"{f.parent.name}/{f.name}"] = evaluate(ranked, {q: qrels[q] for q in ranked if q in qrels})
    return out


def rank_set(directory: Path, encode, scorer, model: str, depth: int = DEPTH, rerank_top: int = CANDIDATES,
             shard=None):  # fmt: skip
    """Every arm's metrics, or None for a ``shard``, which only saves its reranked lists."""
    queries, passages, qrels = load_set(directory)
    ids, texts = [str(q["id"]) for q in queries], [q["text"] for q in queries]
    lexical = BM25Index.build(passages).query_many(texts, k=depth)
    dense = dense_index(directory / "dense", passages, encode, model).query_vectors(encode(texts), k=depth)
    fused = [reciprocal_rank_fusion([a, b], top_n=depth) for a, b in zip(lexical, dense, strict=True)]
    by_id = {p.id: p for p in passages}
    settings = [getattr(scorer, "model_name", None), getattr(scorer, "max_length", None), rerank_top]
    key = hashlib.sha256(json.dumps([*settings, texts_digest(passages)]).encode()).hexdigest()[:16]
    reranked = reranked_lists(directory / "rerank" / key, ids, texts, fused, by_id, scorer, rerank_top, shard)
    if shard:
        return None
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
    p.add_argument(
        "--shard", type=lambda s: tuple(map(int, s.split("/"))), help="i/N: rerank only, every N-th query"
    )
    p.add_argument("--saved-only", action="store_true", help="score each saved rerank file on its own (CPU)")
    args = p.parse_args(argv)
    if args.shard and not (len(args.shard) == 2 and 0 <= args.shard[0] < args.shard[1]):
        p.error("--shard must be i/N with 0 <= i < N")
    if args.saved_only:
        saved = {d.name: saved_rerank_metrics(d) for d in args.set_dir}
        for name, files in saved.items():
            for f, m in files.items():
                print(f"RERANK SAVED {name} {f} queries={m['queries']} ndcg@10={m['ndcg@10']:.4f} "
                      f"R@10={m['recall@10']:.4f} mrr={m['mrr']:.4f}")  # fmt: skip
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(saved, indent=1) + "\n")
        return 0
    encode = BgeM3Encoder(args.dense_model, device=args.device)
    scorer = CrossEncoderScorer(args.reranker, device=args.device)
    if args.shard:
        for directory in args.set_dir:
            rank_set(directory, encode, scorer, args.dense_model, args.depth, shard=args.shard)
            print(f"RERANK SHARD {args.shard[0]}/{args.shard[1]} {directory.name} DONE", flush=True)
        return 0
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
