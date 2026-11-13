#!/usr/bin/env python3
"""Truncation audit: the share of decodes that did not end on the EOS within their budget.

    .venv/bin/python scripts/audit_truncation.py logs/eval_system_ablation/<run>/*.jsonl

Reads AnswerRecords. A decode is truncated when its raw markup does not end in ``<|endoftext|>``:
it spent its ``max_new_tokens`` without producing the EOS, or stopped without one. Refused answers
count by their raw decode. Prints one TRUNCATION line per (system, seed, arm) with the truncated ids.
Exit status: 0 when every rate is 0 (the target), 1 when one is above it, 2 when the audit itself
fails (an unreadable file), so a crash is never read as truncation.
"""

import argparse
import traceback
from collections import defaultdict
from pathlib import Path

from lexhybrid.data.tokenizer import EOS_TOKEN
from lexhybrid.eval.records import read_records


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("records", type=Path, nargs="+", help="AnswerRecord JSONL files")
    args = p.parse_args(argv)
    groups = defaultdict(list)
    for path in args.records:
        for r in read_records(path):
            groups[r.system, r.seed, r.arm].append(r)
    failed = False
    for (system, seed, arm), rows in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        cut = [r.id for r in rows if not r.raw_answer.endswith(EOS_TOKEN)]
        failed |= bool(cut)
        print(f"TRUNCATION system={system} seed={seed} arm={arm} records={len(rows)} truncated={len(cut)} "
              f"rate={len(cut) / len(rows):.4f} ids={','.join(cut)}")  # fmt: skip
    return int(failed)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:  # Python's own exit status for a crash is 1, the truncation verdict
        traceback.print_exc()
        raise SystemExit(2) from None
