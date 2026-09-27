#!/usr/bin/env python3
"""Paired bootstrap comparison of two systems scored on the same items.

Ported from the reference ``scripts/bootstrap_compare.py`` (Phase 14A-6) with the CheXbert parts
dropped and a generic per-item metric input added. The bootstrap is PAIRED: both systems answered
the same items, so each resample draws one set of item indices and scores both on it, and the
difference is computed within the resample. That removes the "some items are just harder"
variance an unpaired comparison would leave in.

Inputs (either or both, all aligned by line):
  --hyps-a/--hyps-b/--refs   one generation per line -> rouge_l (per-item mean), bleu_1, bleu_4
                             (corpus-level, recomputed on every resample)
  --metrics-a/--metrics-b    JSONL, one object of numeric per-item metrics per line
                             (e.g. {"citation_precision": 1.0, "refusal_correct": 0}); the mean
                             of every key present in both files is compared

Decision 16: the bootstrap CI is the sampling uncertainty over items; the seed SD is the training
uncertainty. Report both; a claim needs the paired mean above one baseline seed SD and the sign at
>= 2/3 seeds -- this script supplies the CI, not the claim.

    .venv/bin/python scripts/bootstrap_compare.py --metrics-a hybrid_s42.jsonl --metrics-b xfmr_s42.jsonl \\
        --name-a hybrid --name-b transformer --output analysis/bootstrap_hybrid_vs_xfmr_s42.md
"""

import argparse
import json
import random
import sys
from collections.abc import Sequence
from pathlib import Path

from lexhybrid.eval.text_metrics import corpus_bleu, rouge_l_score


def read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return [line.rstrip("\n") for line in fh]


def read_metrics(path: str) -> list[dict[str, float]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_cache(hyps: Sequence[str] | None, refs: Sequence[str] | None, metrics: list[dict] | None) -> dict:
    cache: dict = {"n": None}
    if hyps is not None:
        cache["rouge"] = [rouge_l_score(h.split(), r.split()) for h, r in zip(hyps, refs)]
        cache["hyp_toks"] = [h.split() for h in hyps]
        cache["ref_toks"] = [r.split() for r in refs]
        cache["n"] = len(hyps)
    if metrics is not None:
        cache["metrics"] = metrics
        cache["n"] = len(metrics) if cache["n"] is None else cache["n"]
    return cache


def evaluate_subset(idx: Sequence[int], cache: dict) -> dict[str, float]:
    """All metrics for one system on one resample of item indices."""
    out: dict[str, float] = {}
    if "rouge" in cache:
        out["rouge_l"] = sum(cache["rouge"][i] for i in idx) / len(idx)
        hyp_toks = [cache["hyp_toks"][i] for i in idx]
        ref_toks = [cache["ref_toks"][i] for i in idx]
        out["bleu_1"] = corpus_bleu(hyp_toks, ref_toks, max_n=1)
        out["bleu_4"] = corpus_bleu(hyp_toks, ref_toks, max_n=4)
    if "metrics" in cache:
        keys = sorted({k for row in cache["metrics"] for k in row})
        for k in keys:
            vals = [float(cache["metrics"][i][k]) for i in idx if k in cache["metrics"][i]]
            if vals:
                out[k] = sum(vals) / len(vals)
    return out


def paired_bootstrap(cache_a: dict, cache_b: dict, n_samples: int, seed: int, alpha: float = 0.05):
    """Resample item indices ONCE per draw and score both systems on them."""
    n = cache_a["n"]
    rng = random.Random(seed)
    point_a = evaluate_subset(range(n), cache_a)
    point_b = evaluate_subset(range(n), cache_b)
    metrics = [m for m in point_a if m in point_b]

    diffs: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_samples):
        idx = [rng.randrange(n) for _ in range(n)]
        sub_a = evaluate_subset(idx, cache_a)
        sub_b = evaluate_subset(idx, cache_b)
        for m in metrics:
            diffs[m].append(sub_a[m] - sub_b[m])

    results = {}
    lo_q, hi_q = alpha / 2, 1 - alpha / 2
    for m in metrics:
        d = sorted(diffs[m])
        lo = d[max(0, int(lo_q * len(d)) - 1)]
        hi = d[min(len(d) - 1, int(hi_q * len(d)))]
        observed = point_a[m] - point_b[m]
        if observed > 0:
            n_opposite = sum(1 for x in d if x <= 0)
        elif observed < 0:
            n_opposite = sum(1 for x in d if x >= 0)
        else:
            n_opposite = len(d)
        results[m] = {
            "a": point_a[m],
            "b": point_b[m],
            "diff": observed,
            "ci_low": lo,
            "ci_high": hi,
            "significant": (lo > 0) or (hi < 0),
            "frac_sign_flipped": n_opposite / len(d),
        }
    return results, {"n": n, "bootstrap_samples": n_samples, "seed": seed}


def render(results: dict, meta: dict, name_a: str, name_b: str) -> str:
    out = [f"# Paired bootstrap: {name_a} vs {name_b}\n"]
    out.append(
        f"\nn={meta['n']} items, {meta['bootstrap_samples']} paired bootstrap resamples, seed {meta['seed']}. "
        "Both systems were scored on the SAME items, so each resample draws one set of indices and "
        "scores both on it.\n"
    )
    out.append(
        f"\nA positive difference favours **{name_a}**. A result is called only when the 95% CI excludes zero.\n"
    )

    def verdict(r: dict) -> str:
        if r["significant"]:
            return f"**{name_a if r['diff'] > 0 else name_b} wins**"
        return "tie (CI spans 0)"

    out.append(f"\n| metric | {name_a} | {name_b} | diff | 95% CI | sign flipped | verdict |\n")
    out.append("|---|---|---|---|---|---|---|\n")
    for m, r in results.items():
        out.append(
            f"| {m} | {r['a']:.4f} | {r['b']:.4f} | {r['diff']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] "
            f"| {r['frac_sign_flipped']:.1%} | {verdict(r)} |\n"
        )
    wins_a = [m for m, r in results.items() if r["significant"] and r["diff"] > 0]
    wins_b = [m for m, r in results.items() if r["significant"] and r["diff"] < 0]
    ties = [m for m, r in results.items() if not r["significant"]]
    out.append("\n## Summary\n")
    out.append(f"\n- **{name_a} wins (CI excludes 0):** {', '.join(wins_a) or 'none'}\n")
    out.append(f"- **{name_b} wins (CI excludes 0):** {', '.join(wins_b) or 'none'}\n")
    out.append(f"- **Ties (CI spans 0):** {', '.join(ties) or 'none'}\n")
    return "".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hyps-a")
    ap.add_argument("--hyps-b")
    ap.add_argument("--refs")
    ap.add_argument("--metrics-a")
    ap.add_argument("--metrics-b")
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--bootstrap-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    text_mode = any([args.hyps_a, args.hyps_b, args.refs])
    metric_mode = any([args.metrics_a, args.metrics_b])
    if text_mode and not all([args.hyps_a, args.hyps_b, args.refs]):
        raise SystemExit("text mode needs --hyps-a, --hyps-b and --refs")
    if metric_mode and not all([args.metrics_a, args.metrics_b]):
        raise SystemExit("metric mode needs --metrics-a and --metrics-b")
    if not (text_mode or metric_mode):
        raise SystemExit("give --hyps-a/--hyps-b/--refs and/or --metrics-a/--metrics-b")

    hyps_a = read_lines(args.hyps_a) if text_mode else None
    hyps_b = read_lines(args.hyps_b) if text_mode else None
    refs = read_lines(args.refs) if text_mode else None
    met_a = read_metrics(args.metrics_a) if metric_mode else None
    met_b = read_metrics(args.metrics_b) if metric_mode else None

    lengths = {len(x) for x in (hyps_a, hyps_b, refs, met_a, met_b) if x is not None}
    if len(lengths) != 1:
        raise SystemExit(
            f"pairing requires equal lengths, got {sorted(lengths)}: both systems must be scored on the same items"
        )

    cache_a = build_cache(hyps_a, refs, met_a)
    cache_b = build_cache(hyps_b, refs, met_b)
    results, meta = paired_bootstrap(cache_a, cache_b, args.bootstrap_samples, args.seed)
    report = render(results, meta, args.name_a, args.name_b)
    print(report)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
