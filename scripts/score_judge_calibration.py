#!/usr/bin/env python3
"""Validate the judge on gold-labelled items before it scores any system.

    .venv/bin/python scripts/score_judge_calibration.py --items data/evalsets/judge_calibration.jsonl \\
        --out logs/judge/calibration.json --base-url http://localhost:8000/v1 --model <judge model> \\
        --min-kappa <k> --min-accuracy <a> [--split validation] [--api-key-env JUDGE_API_KEY]

The items are ``scripts/build_judge_calibration.py``'s. Each is put to the judge with the prompt that
scores its metric in production (``lexhybrid.eval.judge``) and predicted 1 when the score reaches
``THRESHOLDS[type]``: **correctness** for the verdict ``correct``; **answer_relevancy** for a rating of
at least 0.5; **context_precision** (with the reference answers) and **context_relevance** (without)
for a useful passage; **context_recall** for at least half the reference attributed;
**faithfulness** for every statement supported. A reply that does not parse counts as a wrong
prediction.

Reports per item type, on ``--split`` (``validation`` by default; ``calibration`` while tuning
prompts): accuracy, Cohen's kappa, precision and recall of label 1, accuracy per variant, and whether
the type meets the bar: kappa >= ``--min-kappa`` and accuracy >= ``--min-accuracy``. Exits 0 only
when every type meets it.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from lexhybrid.eval.judge import (
    METRICS,
    PROMPT_VERSION,
    add_judge_args,
    correctness_prompt,
    faithfulness_prompt,
    judge_from_args,
    precision_prompt,
    read_attributed,
    read_relevancy,
    read_supported,
    read_useful,
    read_verdict,
    recall_prompt,
    relevancy_prompt,
)

DEFAULT_ITEMS = Path(__file__).resolve().parents[1] / "data" / "evalsets" / "judge_calibration.jsonl"
THRESHOLDS = {"correctness": 1.0, "answer_relevancy": 0.5, "context_precision": 1.0, "context_relevance": 1.0,
              "context_recall": 0.5, "faithfulness": 1.0}  # fmt: skip


def predict(judge, item: dict) -> tuple[float | None, dict]:
    q, answer, refs, contexts = item["question"], item["answer"], item["references"], item["contexts"]
    kind = item["type"]
    if kind == "correctness":
        return judge.ask(correctness_prompt(q, answer, refs), read_verdict)
    if kind == "answer_relevancy":
        return judge.ask(relevancy_prompt(q, answer), read_relevancy)
    if kind in ("context_precision", "context_relevance"):  # relevance items carry no references
        return judge.ask(precision_prompt(q, contexts, refs), read_useful(len(contexts)))
    if kind == "context_recall":
        return judge.ask(recall_prompt(q, refs[0], contexts), read_attributed)
    if kind == "faithfulness":
        return judge.ask(faithfulness_prompt(q, answer, contexts), read_supported)
    raise ValueError(f"{item['id']}: unknown type {kind!r}")


def agreement(labels: list[int], preds: list[int]) -> dict:
    """Accuracy, Cohen's kappa (``None`` when chance agreement is 1), precision and recall of label 1."""
    n = len(labels)
    tp = sum(y and p for y, p in zip(labels, preds, strict=True))
    fp = sum(p and not y for y, p in zip(labels, preds, strict=True))
    fn = sum(y and not p for y, p in zip(labels, preds, strict=True))
    tn = n - tp - fp - fn
    observed = (tp + tn) / n
    chance = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n**2
    return {
        "n": n,
        "accuracy": observed,
        "kappa": (observed - chance) / (1 - chance) if chance < 1 else None,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def sample(items: list[dict], cap: int | None) -> list[dict]:
    """At most ``cap`` items per type, taken a whole group at a time (an item with its negatives) in
    the order of a hash of the group, so every run takes the same sample."""
    if not cap:
        return items
    groups = defaultdict(list)
    for item in items:
        groups[item["type"], item["group"]].append(item)
    out, taken = [], defaultdict(int)
    for (kind, _group), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], _sample_rank(kv[0][1]))):
        if taken[kind] + len(rows) <= cap:
            out += rows
            taken[kind] += len(rows)
    return out


def _sample_rank(group: str) -> str:
    return hashlib.sha256(f"sample:{group}".encode()).hexdigest()  # salted: the split uses the plain hash


def score(items: list[dict], judge, min_kappa: float, min_accuracy: float) -> dict:
    per_item, by_type = [], defaultdict(list)
    for item in items:
        s, _ = predict(judge, item)
        pred = 1 - item["label"] if s is None else int(s >= THRESHOLDS[item["type"]])
        per_item.append({"id": item["id"], "label": item["label"], "pred": pred, "parsed": s is not None})
        by_type[item["type"]].append((item, pred, s is not None))
    types = {}
    for kind, rows in sorted(by_type.items()):
        m = agreement([i["label"] for i, _, _ in rows], [p for _, p, _ in rows])
        variants = defaultdict(list)
        for i, p, _ in rows:
            variants[i["variant"]].append(p == i["label"])
        m["unparsed"] = sum(not ok for _, _, ok in rows)
        m["accuracy_by_variant"] = {v: sum(x) / len(x) for v, x in sorted(variants.items())}
        m["meets"] = m["kappa"] is not None and m["kappa"] >= min_kappa and m["accuracy"] >= min_accuracy
        types[kind] = m
    missing = sorted(set(METRICS) - set(types))  # a metric nobody validated licenses no claim
    return {"types": types, "missing_types": missing, "meets": not missing and all(m["meets"] for m in types.values()),
            "per_item": per_item}  # fmt: skip


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    p.add_argument("--out", type=Path, required=True)
    add_judge_args(p)
    p.add_argument("--split", choices=("validation", "calibration"), default="validation")
    p.add_argument("--min-kappa", type=float, required=True)
    p.add_argument("--min-accuracy", type=float, required=True)
    p.add_argument("--max-per-type", type=int, default=None, help="sample at most this many items per type")
    args = p.parse_args(argv)
    lines = args.items.read_text(encoding="utf-8").split("\n")
    items = [i for i in map(json.loads, filter(str.strip, lines)) if i["split"] == args.split]
    items = sample(items, args.max_per_type)
    judge = judge_from_args(args)
    report = {"judge": {"model": args.model, "base_url": args.base_url}, "split": args.split,
              "prompt_version": PROMPT_VERSION, "max_per_type": args.max_per_type,
              "bar": {"min_kappa": args.min_kappa, "min_accuracy": args.min_accuracy},
              **score(items, judge, args.min_kappa, args.min_accuracy)}  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    for kind, m in report["types"].items():
        kappa = f"{m['kappa']:.3f}" if m["kappa"] is not None else "-"
        print(f"JUDGE-CAL {args.split} {kind} n={m['n']} accuracy={m['accuracy']:.3f} kappa={kappa} "
              f"precision={m['precision']:.3f} recall={m['recall']:.3f} unparsed={m['unparsed']} "
              f"meets={'yes' if m['meets'] else 'no'}")  # fmt: skip
    if report["missing_types"]:
        print(f"JUDGE-CAL no items of {report['missing_types']}: those metrics are not validated")
    print(f"JUDGE-CAL bar {'met' if report['meets'] else 'NOT met'} ({args.split}, {len(items)} items)")
    return 0 if report["meets"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
