#!/usr/bin/env python3
"""Score answer records with the independent judge: correctness and the four RAGAS metrics.

    .venv/bin/python scripts/judge_answers.py --records logs/answers/hybrid_s42.jsonl \\
        --out logs/judge/hybrid_s42.jsonl --base-url http://localhost:8000/v1 --model <judge model> \\
        --calibration <score_judge_calibration.py report of this judge> \\
        [--api-key-env JUDGE_API_KEY] [--cache-dir data/cache/judge] [--limit 20] [--dry-run]

``--records`` is ``AnswerRecord`` JSONL (``lexhybrid.eval.records``). Writes, for the first
``--limit`` records (all by default):

* ``--out``: one line per record, in input order, holding its ``id`` and numeric scores
  (``correctness``, ``faithfulness``, ``answer_relevancy``, ``context_precision``,
  ``context_recall``; a score that does not apply or did not parse is left out), so
  ``scripts/bootstrap_compare.py --metrics-a`` reads it directly;
* ``<out>.details.jsonl``: per record its ids and the judge's reply objects (verdicts, reasons,
  statements; ``{"raw": ...}`` for a reply that did not parse);
* ``<out>.summary.json``: per (system, seed, arm) the record and refusal counts and, per metric, the
  mean over the records it applies to, that count and the number of unparsed replies.

The judge is ``lexhybrid.eval.judge`` behind its reply cache, so a rerun asks only for what did not
parse or was not reached; an HTTP failure that outlasts the client's retries stops the run with what
was judged so far in the cache. ``--dry-run`` prints the first prompt the judge would be sent and
calls nothing.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from lexhybrid.eval.judge import (
    METRICS,
    PROMPT_VERSION,
    add_judge_args,
    judge_from_args,
    judge_prompts,
    judge_record,
)
from lexhybrid.eval.records import read_records


def summarize(records, results) -> list[dict]:
    groups = defaultdict(list)
    for rec, (scores, details) in zip(records, results, strict=True):
        groups[(rec.system, rec.seed, rec.arm)].append((rec, scores, details))
    out = []
    for (system, seed, arm), rows in groups.items():
        entry = {"system": system, "seed": seed, "arm": arm, "n": len(rows),
                 "refused": sum(r.refused for r, _, _ in rows)}  # fmt: skip
        for m in METRICS:
            vals = [s[m] for _, s, _ in rows if s[m] is not None]
            unparsed = sum("raw" in d.get(m, {}) for _, _, d in rows)
            entry[m] = {"mean": sum(vals) / len(vals) if vals else None, "n": len(vals), "unparsed": unparsed}
        out.append(entry)
    return out


def check_validated(path: Path, model: str, base_url: str) -> None:
    """Refuse to score with a judge whose validation report (``score_judge_calibration.py``) is not this
    judge's or did not pass: its scores would license claims nobody checked."""
    report = json.loads(path.read_text(encoding="utf-8"))
    judge = report.get("judge", {})
    problems = [
        *([] if report.get("meets") else ["the bar was not met"]),
        *([] if report.get("split") == "validation" else [f"split {report.get('split')!r}, not validation"]),
        *([] if judge.get("model") == model else [f"model {judge.get('model')!r}, not {model!r}"]),
        *([] if str(judge.get("base_url", "")).rstrip("/") == base_url.rstrip("/") else [f"endpoint {judge.get('base_url')!r}, not {base_url!r}"]),
        *([] if report.get("prompt_version") == PROMPT_VERSION else [f"prompt version {report.get('prompt_version')!r}, not {PROMPT_VERSION}"]),
        *([] if set(METRICS) <= set(report.get("types", {})) else [f"no validation of {sorted(set(METRICS) - set(report.get('types', {})))}"]),
    ]  # fmt: skip
    if problems:
        raise SystemExit(f"{path}: this judge is not validated: " + "; ".join(problems))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--calibration", type=Path, help="the judge's validation report; required to score")
    p.add_argument("--out", type=Path, required=True)
    add_judge_args(p)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    records = read_records(args.records)[: args.limit]
    if args.dry_run:
        print(next((prompt for r in records for prompt, _ in judge_prompts(r).values()), "(no prompt)"))
        return 0
    if args.calibration is None:
        p.error("--calibration is required: score_judge_calibration.py validates a judge before it scores")
    check_validated(args.calibration, args.model, args.base_url)
    judge = judge_from_args(args)
    results = [judge_record(judge, r) for r in records]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r, (scores, _) in zip(records, results, strict=True):
            row = {"id": r.id, **{m: v for m, v in scores.items() if v is not None}}
            f.write(json.dumps(row, sort_keys=True) + "\n")
    with open(args.out.with_suffix(".details.jsonl"), "w", encoding="utf-8") as f:
        for r, (_, details) in zip(records, results, strict=True):
            row = {
                "id": r.id,
                "set": r.set,
                "system": r.system,
                "seed": r.seed,
                "arm": r.arm,
                "details": details,
            }
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    groups = summarize(records, results)
    summary = {"judge": {"model": args.model, "base_url": args.base_url}, "records": str(args.records),
               "new_calls": judge.calls, "groups": groups}  # fmt: skip
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for g in groups:
        means = " ".join(f"{m}={g[m]['mean']:.4f}" if g[m]["mean"] is not None else f"{m}=-" for m in METRICS)
        print(
            f"JUDGE {g['system']} seed={g['seed']} arm={g['arm']} n={g['n']} refused={g['refused']} {means}"
        )
    print(f"JUDGE new_calls={judge.calls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
