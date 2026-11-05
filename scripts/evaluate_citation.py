#!/usr/bin/env python3
"""Citation metrics of answer records: per-question metrics for the bootstrap, a summary per system, seed and arm.

    .venv/bin/python scripts/evaluate_citation.py logs/answers/hybrid_s42.jsonl \\
        --metrics-out logs/citation/hybrid_s42.metrics.jsonl --summary-out logs/citation/hybrid_s42.summary.json

Inputs are ``AnswerRecord`` JSONL files (``lexhybrid.eval.records``), read in the order given; the
metrics are ``lexhybrid.eval.citation_metrics``'. ``--metrics-out`` has one line per input record, in
input order, its ``id`` and numeric fields (a metric that does not apply is left out), as
``scripts/bootstrap_compare.py --metrics-a`` reads it. ``--summary-out`` is a JSON list with one
summary per (system, seed, arm). Citation precision there is system-internal: the system was tuned
against this verifier.
"""

import argparse
import json
from pathlib import Path

from lexhybrid.eval.citation_metrics import record_metrics, summarise
from lexhybrid.eval.records import read_records


def _fmt(value) -> str:
    return "-" if value is None else f"{value:.4f}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("records", type=Path, nargs="+", help="AnswerRecord JSONL files")
    p.add_argument("--metrics-out", type=Path, required=True)
    p.add_argument("--summary-out", type=Path, required=True)
    args = p.parse_args(argv)
    records = [r for path in args.records for r in read_records(path)]
    for path in (args.metrics_out, args.summary_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(
        "".join(json.dumps({"id": r.id, **record_metrics(r)}) + "\n" for r in records)
    )
    summaries = summarise(records)
    args.summary_out.write_text(json.dumps(summaries, indent=1, ensure_ascii=False) + "\n")
    for s in summaries:
        print(
            f"CITATION {s['system']} seed={s['seed']} arm={s['arm']} records={s['records']} "
            f"precision(system-internal)={_fmt(s['citation_precision'])} recall={_fmt(s['citation_recall'])} "
            f"fidelity={_fmt(s['quote_fidelity'])} refusal_f1={_fmt(s['refusal_f1'])} "
            f"em={_fmt(s['exact_match'])} char_f1={_fmt(s['char_f1'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
