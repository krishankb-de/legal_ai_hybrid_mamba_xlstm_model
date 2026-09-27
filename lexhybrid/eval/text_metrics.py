"""Sentence ROUGE-L and corpus BLEU, stdlib only.

Ported verbatim from the reference ``scripts/evaluate_report_generation.py`` (which is otherwise
dropped) so ``scripts/bootstrap_compare.py`` reproduces the reference's point estimates exactly.
These are self-contained implementations, not sacrebleu/rouge-score; compare systems scored by the
same code, and do not quote the numbers as sacrebleu-comparable.
"""

import math
from collections import Counter


def _lcs_length(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l_score(hyp_tokens: list[str], ref_tokens: list[str], beta: float = 1.2) -> float:
    """Sentence-level ROUGE-L F-measure (Lin, 2004)."""
    if not hyp_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(hyp_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    p = lcs / len(hyp_tokens)
    r = lcs / len(ref_tokens)
    denom = r + (beta**2) * p
    return ((1 + beta**2) * p * r) / denom if denom > 0 else 0.0


def _ngram_counts(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(hyps: list[list[str]], refs: list[list[str]], max_n: int = 4) -> float:
    """Corpus-level BLEU-N with brevity penalty, one reference per hypothesis."""
    weights = [1.0 / max_n] * max_n
    precisions = []
    for n in range(1, max_n + 1):
        match, total = 0, 0
        for hyp, ref in zip(hyps, refs):
            hyp_ngrams = _ngram_counts(hyp, n)
            ref_ngrams = _ngram_counts(ref, n)
            match += sum(min(c, ref_ngrams.get(g, 0)) for g, c in hyp_ngrams.items())
            total += sum(hyp_ngrams.values())
        precisions.append(match / total if total > 0 else 0.0)

    if any(p == 0.0 for p in precisions):
        geo_mean = 0.0
    else:
        geo_mean = math.exp(sum(w * math.log(p) for w, p in zip(weights, precisions)))

    hyp_len = sum(len(h) for h in hyps)
    ref_len = sum(len(r) for r in refs)
    if hyp_len == 0:
        bp = 0.0
    elif hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / hyp_len)
    return bp * geo_mean
