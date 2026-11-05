"""Ranking metrics over qrels, as trec_eval computes them: nDCG@k (gain = relevance label,
log2 discount), recall@k, and reciprocal rank (MRR over queries). A query without a relevant passage is
skipped, so every mean is over judged queries."""

import math
from collections.abc import Mapping, Sequence


def ndcg_at_k(ranking: Sequence[str], rel: Mapping[str, int], k: int) -> float:
    top = list(dict.fromkeys(ranking))[:k]  # a passage counts once, at its best rank
    dcg = sum(rel.get(pid, 0) / math.log2(r + 1) for r, pid in enumerate(top, start=1))
    ideal = sorted((g for g in rel.values() if g > 0), reverse=True)[:k]
    return dcg / sum(g / math.log2(r + 1) for r, g in enumerate(ideal, start=1)) if ideal else 0.0


def recall_at_k(ranking: Sequence[str], rel: Mapping[str, int], k: int) -> float:
    relevant = {pid for pid, g in rel.items() if g > 0}
    return len(relevant & set(ranking[:k])) / len(relevant)


def reciprocal_rank(ranking: Sequence[str], rel: Mapping[str, int]) -> float:
    return next((1.0 / r for r, pid in enumerate(ranking, start=1) if rel.get(pid, 0) > 0), 0.0)


def evaluate(rankings: Mapping[str, Sequence[str]], qrels: Mapping[str, Mapping[str, int]]) -> dict:
    """Mean nDCG@10, R@5/10/20 and MRR over the judged queries (a query with no ranking scores 0)."""
    judged = [q for q, rel in qrels.items() if any(g > 0 for g in rel.values())]
    if not judged:
        raise ValueError("no query has a relevant passage")
    rows = [(rankings.get(q, []), qrels[q]) for q in judged]
    mean = lambda f: sum(f(r, rel) for r, rel in rows) / len(rows)  # noqa: E731
    return {
        "queries": len(rows),
        "ndcg@10": mean(lambda r, rel: ndcg_at_k(r, rel, 10)),
        "recall@5": mean(lambda r, rel: recall_at_k(r, rel, 5)),
        "recall@10": mean(lambda r, rel: recall_at_k(r, rel, 10)),
        "recall@20": mean(lambda r, rel: recall_at_k(r, rel, 20)),
        "mrr": mean(reciprocal_rank),
    }
