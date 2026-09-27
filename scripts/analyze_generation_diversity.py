#!/usr/bin/env python3
"""Boilerplate / duplicate-template analysis for generated answers, with controls.

Ported from the reference ``scripts/analyze_generation_diversity.py`` (Phase 14B): exact-duplicate
clusters, distinct-n, type-token ratio and sampled self-BLEU-4, for the generator and -- the point
of the tool -- for the reference answers and a copy baseline scored the same way. Stdlib only.
Self-BLEU here is not sacrebleu-comparable; compare only rows scored in the same run.

    .venv/bin/python scripts/analyze_generation_diversity.py --hyps hyps.txt --refs refs.txt \
        --baseline copy_floor.txt --output analysis/diversity.md
"""

import argparse
import collections
import math
import random
import sys
from collections.abc import Sequence
from pathlib import Path


def read_lines(path: str) -> list[str]:
    """Read one generation per line, dropping blanks and normalising whitespace."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = [" ".join(line.strip().split()) for line in fh]
    return [line for line in lines if line]


def duplicate_clusters(texts: Sequence[str]) -> tuple[int, int, float, list[int]]:
    """Exact-duplicate clustering — the same measurement Phase 11C reported.

    Returns:
        (num_clusters, num_texts_in_a_cluster, fraction_in_a_cluster, top_sizes)
        where a "cluster" is a distinct text occurring 2+ times.
    """
    counts = collections.Counter(texts)
    sizes = sorted((c for c in counts.values() if c > 1), reverse=True)
    in_cluster = sum(sizes)
    frac = in_cluster / len(texts) if texts else 0.0
    return len(sizes), in_cluster, frac, sizes[:5]


def distinct_n(texts: Sequence[str], n: int) -> float:
    """Ratio of unique n-grams to total n-grams across the corpus (higher = more diverse)."""
    total, unique = 0, set()
    for text in texts:
        tokens = text.split()
        for i in range(len(tokens) - n + 1):
            unique.add(tuple(tokens[i : i + n]))
            total += 1
    return len(unique) / total if total else 0.0


def type_token_ratio(texts: Sequence[str]) -> float:
    """Unique tokens / total tokens across the corpus."""
    total, vocab = 0, set()
    for text in texts:
        tokens = text.split()
        vocab.update(tokens)
        total += len(tokens)
    return len(vocab) / total if total else 0.0


def _ngram_counts(tokens: Sequence[str], n: int) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = collections.defaultdict(int)
    for i in range(len(tokens) - n + 1):
        counts[tuple(tokens[i : i + n])] += 1
    return counts


def _sentence_bleu4(hyp: Sequence[str], refs: Sequence[Sequence[str]]) -> float:
    """Self-contained BLEU-4 with add-one smoothing on higher-order n-grams.

    Not sacrebleu-comparable. Used only to compare corpora scored identically within
    one run of this script.
    """
    if not hyp:
        return 0.0
    log_precisions = []
    for n in range(1, 5):
        hyp_counts = _ngram_counts(hyp, n)
        if not hyp_counts:
            return 0.0
        max_ref: dict[tuple[str, ...], int] = {}
        for ref in refs:
            for gram, count in _ngram_counts(ref, n).items():
                if count > max_ref.get(gram, 0):
                    max_ref[gram] = count
        overlap = sum(min(c, max_ref.get(g, 0)) for g, c in hyp_counts.items())
        total = sum(hyp_counts.values())
        # add-one smoothing for n>1 so a single zero order doesn't zero the score
        if n > 1:
            overlap, total = overlap + 1, total + 1
        if overlap == 0:
            return 0.0
        log_precisions.append(math.log(overlap / total))

    closest = min((abs(len(r) - len(hyp)), len(r)) for r in refs)[1]
    brevity = 1.0 if len(hyp) > closest else math.exp(1 - closest / max(len(hyp), 1))
    return brevity * math.exp(sum(log_precisions) / 4.0)


def self_bleu4(texts: Sequence[str], sample: int = 300, refs_per: int = 40, seed: int = 0) -> float:
    """Mean BLEU-4 of each text against a sample of the OTHERS. Higher = more repetitive.

    Exhaustive self-BLEU is O(n^2) (7M pairs at n=2663), so this samples: `sample`
    hypotheses each scored against `refs_per` random others. Sampling is seeded, and
    every corpus in a run gets the same treatment, which is what makes the
    generator/reference/baseline comparison valid.
    """
    if len(texts) < 2:
        return 0.0
    rng = random.Random(seed)
    tokenised = [t.split() for t in texts]
    indices = list(range(len(tokenised)))
    chosen = rng.sample(indices, min(sample, len(indices)))

    scores = []
    for i in chosen:
        pool = [j for j in rng.sample(indices, min(refs_per + 1, len(indices))) if j != i]
        if not pool:
            continue
        scores.append(_sentence_bleu4(tokenised[i], [tokenised[j] for j in pool[:refs_per]]))
    return sum(scores) / len(scores) if scores else 0.0


def analyse(name: str, texts: Sequence[str], seed: int = 0) -> dict[str, object]:
    n_clusters, in_cluster, frac, top = duplicate_clusters(texts)
    return {
        "name": name,
        "n": len(texts),
        "unique": len(set(texts)),
        "duplicate_clusters": n_clusters,
        "in_duplicate_cluster": in_cluster,
        "pct_in_duplicate_cluster": 100.0 * frac,
        "largest_clusters": top,
        "distinct_1": distinct_n(texts, 1),
        "distinct_2": distinct_n(texts, 2),
        "distinct_3": distinct_n(texts, 3),
        "distinct_4": distinct_n(texts, 4),
        "type_token_ratio": type_token_ratio(texts),
        "self_bleu_4": self_bleu4(texts, seed=seed),
        "mean_tokens": (sum(len(t.split()) for t in texts) / len(texts)) if texts else 0.0,
    }


def render(results: list[dict[str, object]], historical_pct: float | None = None) -> str:
    """Render the comparison table plus the pre-registered verdict."""
    out = []
    out.append("# Generation diversity / boilerplate analysis\n")
    out.append("Produced by `scripts/analyze_generation_diversity.py`.\n")
    out.append(
        "\n**Why the controls matter.** A duplication rate is only interpretable against the "
        "reference corpus (how templated the real answers are) and against a copy baseline "
        "(what a system that returns real text scores). Supply both with --refs and --baseline.\n"
    )

    out.append("\n## Headline\n")
    out.append("\n| corpus | n | unique | dup. clusters | % in a dup. cluster | largest clusters |")
    out.append("\n|---|---|---|---|---|---|")
    for r in results:
        largest = ", ".join(str(s) for s in r["largest_clusters"]) or "—"
        out.append(
            f"\n| {r['name']} | {int(r['n'])} | {int(r['unique'])} | {int(r['duplicate_clusters'])} | "
            f"**{r['pct_in_duplicate_cluster']:.1f}%** | {largest} |"
        )

    out.append("\n\n## Lexical diversity (higher = more varied)\n")
    out.append(
        "\n| corpus | distinct-1 | distinct-2 | distinct-3 | distinct-4 | TTR | self-BLEU-4 | mean tokens |"
    )
    out.append("\n|---|---|---|---|---|---|---|---|")
    for r in results:
        out.append(
            f"\n| {r['name']} | {r['distinct_1']:.4f} | {r['distinct_2']:.4f} | {r['distinct_3']:.4f} | "
            f"{r['distinct_4']:.4f} | {r['type_token_ratio']:.4f} | {r['self_bleu_4']:.4f} | "
            f"{r['mean_tokens']:.1f} |"
        )
    out.append(
        "\n\nSelf-BLEU is a self-contained implementation (stdlib only, sampled) and is "
        "**not** sacrebleu-comparable -- use it only to compare the rows above, which "
        "were all scored identically in one run.\n"
    )

    gen = next((r for r in results if r["name"].startswith("generated")), results[0])
    controls = [r for r in results if r is not gen]
    out.append("\n## Verdict\n")
    if historical_pct is not None:
        pct = gen["pct_in_duplicate_cluster"]
        out.append(
            f"\n- A previous measurement recorded **{historical_pct:.1f}%**; this run measures "
            f"**{pct:.1f}%** (**{pct - historical_pct:+.1f} pp**).\n"
        )
    for c in controls:
        delta = gen["pct_in_duplicate_cluster"] - c["pct_in_duplicate_cluster"]
        out.append(
            f"- vs **{c['name']}** control: {gen['pct_in_duplicate_cluster']:.1f}% vs "
            f"{c['pct_in_duplicate_cluster']:.1f}% (**{delta:+.1f} pp**).\n"
        )

    # Pre-registered interpretation (thresholds carried over from the reference).
    above_all = (
        all(gen["pct_in_duplicate_cluster"] > c["pct_in_duplicate_cluster"] + 10.0 for c in controls)
        if controls
        else False
    )
    if gen["pct_in_duplicate_cluster"] >= 70.0 and above_all:
        out.append(
            "\n**PRE-REGISTERED OUTCOME: QUALIFIER REQUIRED.** The generator is >=70% templated "
            "AND materially above every control; any headline about it must carry that qualifier.\n"
        )
    elif controls and gen["pct_in_duplicate_cluster"] <= max(c["pct_in_duplicate_cluster"] for c in controls):
        out.append(
            "\n**PRE-REGISTERED OUTCOME: POSITIVE FINDING.** The generator's duplication rate is at "
            "or below its controls', i.e. no more templated than the corpus it models.\n"
        )
    else:
        out.append(
            "\n**PRE-REGISTERED OUTCOME: INTERMEDIATE.** Neither trigger fired. Report the numbers "
            "with the controls alongside and do not round the interpretation in either direction.\n"
        )
    if not controls:
        out.append(
            "\n**No controls were supplied.** Re-run with --refs and --baseline; a bare duplication "
            "rate is not interpretable.\n"
        )
    return "".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hyps", required=True, help="generated reports, one per line")
    ap.add_argument("--refs", default=None, help="reference reports (control)")
    ap.add_argument("--baseline", default=None, help="copy-baseline outputs (control)")
    ap.add_argument("--output", default=None, help="write the markdown report here")
    ap.add_argument(
        "--historical-pct",
        type=float,
        default=None,
        help="an earlier measured duplication rate to compare against",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    results = [analyse("generated (model)", read_lines(args.hyps), seed=args.seed)]
    if args.refs:
        results.append(analyse("references (human)", read_lines(args.refs), seed=args.seed))
    if args.baseline:
        results.append(analyse("copy baseline (real text)", read_lines(args.baseline), seed=args.seed))

    report = render(results, historical_pct=args.historical_pct)
    print(report)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
