"""Reciprocal rank fusion of retrieval rankings (plan P7-E; Cormack, Clarke and Büttcher 2009).

A chunk's fused score is the sum, over the rankings that contain it, of ``1 / (k + rank)``, with
1-based ranks and ``k = 60``. Only ranks enter, never the raw scores, so BM25's unbounded scores and
the dense index's cosine similarities fuse without calibration. Ties break by the best single rank
the chunk reached in any ranking, then by chunk id, so the result is deterministic.
"""

from collections.abc import Iterable, Sequence

from lexhybrid.retrieval.bm25 import Hit

RRF_K = 60


def _ids(ranking: Iterable[Hit | str]) -> list[str]:
    return [h.chunk_id if isinstance(h, Hit) else str(h) for h in ranking]


def reciprocal_rank_fusion(
    rankings: Sequence[Iterable[Hit | str]], k: int = RRF_K, top_n: int | None = None
) -> list[Hit]:
    """Fuse rankings (each a list of ``Hit`` or chunk ids, best first) into one ranking of ``Hit``s
    whose ``score`` is the RRF score. A chunk repeated within one ranking counts once, at its best
    rank, and the chunks after it are ranked as if the repeat were not there."""
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    score: dict[str, float] = {}
    best: dict[str, int] = {}
    for ranking in rankings:
        unique = list(
            dict.fromkeys(_ids(ranking))
        )  # a repeat keeps its first (best) place; later ones move up
        for rank, cid in enumerate(unique, start=1):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + rank)
            best[cid] = min(best.get(cid, rank), rank)
    order = sorted(score, key=lambda c: (-score[c], best[c], c))
    if top_n is not None:
        order = order[:top_n]
    return [Hit(cid, score[cid], i) for i, cid in enumerate(order, start=1)]
