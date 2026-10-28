"""Cross-encoder reranking of fused candidates into the prompt's passages (plan P7-E).

The fused ranking's top ``candidates`` (50) are scored against the question by a cross-encoder,
``BAAI/bge-reranker-v2-m3`` in production. Chunks are then taken best first while two limits hold:
at most ``max_chunks`` (8), and at most ``token_budget`` (4,096) tokens of passage text in total. A
chunk that would overflow the budget is skipped, not cut, because a passage is quoted by
sentence. The plan's lower bound of 3 chunks is what a normal question gets; a question with fewer
candidates, or whose candidates are long, gets fewer. ``select`` never pads with chunks that do
not fit.

The scorer is any callable ``(question, passage texts) -> scores`` (higher is better), so tests and
the CPU pipeline use a stand-in; ``CrossEncoderScorer`` loads the model lazily, on first use.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from lexhybrid.retrieval.chunking import Chunk, estimate_tokens

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
CANDIDATES, MIN_CHUNKS, MAX_CHUNKS, TOKEN_BUDGET = 50, 3, 8, 4096

Scorer = Callable[[str, Sequence[str]], Sequence[float]]


@dataclass(frozen=True)
class Ranked:
    chunk: Chunk
    score: float
    tokens: int


def rerank(question: str, candidates: Sequence[Chunk], scorer: Scorer, limit: int = CANDIDATES,
           count_tokens: Callable[[str], int] = estimate_tokens) -> list[Ranked]:  # fmt: skip
    """The first ``limit`` candidates re-scored and sorted best first (ties keep the fused order)."""
    pool = list(candidates[:limit])
    if not pool:
        return []
    scores = list(scorer(question, [c.text for c in pool]))
    if len(scores) != len(pool):
        raise ValueError(f"scorer returned {len(scores)} scores for {len(pool)} passages")
    ranked = [Ranked(c, float(s), count_tokens(c.text)) for c, s in zip(pool, scores, strict=True)]
    order = sorted(range(len(ranked)), key=lambda i: (-ranked[i].score, i))
    return [ranked[i] for i in order]


def select(
    ranked: Sequence[Ranked], max_chunks: int = MAX_CHUNKS, token_budget: int = TOKEN_BUDGET
) -> list[Chunk]:
    """Best first, while there are fewer than ``max_chunks`` and the passages fit ``token_budget``."""
    chosen, used = [], 0
    for r in ranked:
        if len(chosen) == max_chunks:
            break
        if used + r.tokens > token_budget:
            continue
        chosen.append(r.chunk)
        used += r.tokens
    return chosen


def rerank_and_select(question: str, candidates: Sequence[Chunk], scorer: Scorer, **limits) -> list[Chunk]:
    """P7-E's step from the fused top 50 to the 3-8 passages (<= 4K tokens) of the prompt."""
    select_keys = {k: limits.pop(k) for k in ("max_chunks", "token_budget") if k in limits}
    return select(rerank(question, candidates, scorer, **limits), **select_keys)


class CrossEncoderScorer:
    """``bge-reranker-v2-m3`` as a ``Scorer``: one logit per (question, passage) pair, in batches.
    Loaded on first call from the Hugging Face cache (offline on the cluster)."""

    def __init__(self, model_name: str = RERANK_MODEL, device: str | None = None, batch_size: int = 16,
                 max_length: int = 1024):  # fmt: skip
        self.model_name, self.batch_size, self.max_length = model_name, batch_size, max_length
        self.device = device
        self._model = self._tok = None

    def _load(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = (
            AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device).eval()
        )

    def __call__(self, question: str, passages: Sequence[str]) -> list[float]:
        import torch

        if self._model is None:
            self._load()
        scores: list[float] = []
        for start in range(0, len(passages), self.batch_size):
            batch = [[question, p] for p in passages[start : start + self.batch_size]]
            enc = self._tok(
                batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            )
            with torch.no_grad():
                logits = self._model(**{k: v.to(self.device) for k, v in enc.items()}).logits
            scores += logits.view(-1).float().tolist()
        return scores
