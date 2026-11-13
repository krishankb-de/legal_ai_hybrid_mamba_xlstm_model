#!/usr/bin/env python3
"""Calibrate the verifier's entailment threshold on hand-labelled premise/claim pairs.

    .venv/bin/python scripts/calibrate_nli.py --pairs data/manifests/nli_calibration.jsonl \\
        --out logs/nli_calibration.json

The pairs file starts with an ``{"info": ...}`` line whose ``review`` must be ``"done"``: the labels are
checked by a person before any model run, and the script refuses an unreviewed set. Each pair:
``id``, ``kind``, ``premise``, ``hypothesis``, ``label`` (1 = the premise entails the claim).

Reports precision, recall and F1 of "entailed" at the fixed threshold (``ENTAILMENT_THRESHOLD``) and at
the best threshold (highest F1 over the observed probabilities; ties go to the higher threshold),
accuracy per kind at the fixed threshold, and every pair's probability. The fixed threshold ships
unless its precision is below ``SHIP_PRECISION``; then the choice is left to a person.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from lexhybrid.verifier.nli import ENTAILMENT_THRESHOLD, NLI_MODEL, MDebertaNLI

SHIP_PRECISION = 0.85


def read_pairs(path: Path) -> tuple[dict, list[dict]]:
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    if not lines or "info" not in lines[0]:
        raise SystemExit(f"{path}: the first line must be the info block")
    info, pairs = lines[0]["info"], lines[1:]
    ids = [p["id"] for p in pairs]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{path}: duplicate pair ids")
    bad = [
        p["id"]
        for p in pairs
        if type(p.get("label")) is not int
        or p["label"] not in (0, 1)
        or not p.get("premise")
        or not p.get("hypothesis")
    ]
    if bad:
        raise SystemExit(f"{path}: pairs without premise, hypothesis or a 0/1 label: {bad[:5]}")
    if not pairs:
        raise SystemExit(f"{path}: no pairs")
    return info, pairs


def scores(probs: list[float], labels: list[int], threshold: float) -> dict:
    predicted = [p >= threshold for p in probs]
    tp = sum(pr and y for pr, y in zip(predicted, labels, strict=True))
    fp = sum(pr and not y for pr, y in zip(predicted, labels, strict=True))
    fn = sum(not pr and y for pr, y in zip(predicted, labels, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def calibrate(pairs: list[dict], nli) -> dict:
    probs = [float(nli(p["premise"], p["hypothesis"])) for p in pairs]
    labels = [p["label"] for p in pairs]
    fixed = scores(probs, labels, ENTAILMENT_THRESHOLD)
    best = max(
        (scores(probs, labels, t) for t in sorted(set(probs))), key=lambda s: (s["f1"], s["threshold"])
    )
    by_kind: dict[str, list[bool]] = defaultdict(list)
    for p, prob in zip(pairs, probs, strict=True):
        by_kind[p["kind"]].append((prob >= ENTAILMENT_THRESHOLD) == bool(p["label"]))
    return {
        "pairs": len(pairs),
        "fixed": fixed,
        "best": best,
        "best_is_in_sample": True,  # chosen and scored on the same pairs: optimistic, never the shipped value
        "ships": ENTAILMENT_THRESHOLD if fixed["precision"] >= SHIP_PRECISION else None,
        "accuracy_by_kind": {k: sum(v) / len(v) for k, v in sorted(by_kind.items())},
        "per_pair": [
            {"id": p["id"], "label": p["label"], "p": prob} for p, prob in zip(pairs, probs, strict=True)
        ],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=NLI_MODEL)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    info, pairs = read_pairs(args.pairs)
    if info.get("review") != "done":
        raise SystemExit(
            f"{args.pairs}: review is {info.get('review')!r}; a person checks the labels before any run"
        )
    report = {"model": args.model, **calibrate(pairs, MDebertaNLI(args.model, device=args.device))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    for name in ("fixed", "best"):
        s = report[name]
        print(
            f"NLI {name} threshold={s['threshold']:.3f} precision={s['precision']:.3f} recall={s['recall']:.3f} f1={s['f1']:.3f}"
        )
    verdict = (
        f"SHIP {ENTAILMENT_THRESHOLD}"
        if report["ships"] is not None
        else "PRECISION BELOW 0.85: a person decides"
    )
    print(f"NLI {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
