#!/usr/bin/env python3
"""The P5 screen's seed table, per-arm means and the pre-registered decision rule (P5-F, P5-H, P5-I).

    .venv/bin/python scripts/screen_table.py --prefix screen2_ \\
        --val-logs cluster/logs/screen_metrics_2662834.log cluster/logs/screen_metrics_2662842.log \\
        --train-logs cluster/logs/screen_2637454_*.log cluster/logs/screen_2637455_*.log ... \\
        --eval-dirs cluster/logs/eval_2663318 cluster/logs/eval_2663319 --bar 0.10

Assembles every column from the job outputs, so nothing is copied by hand:

* val PPL: the ``val_ppl`` of the run's validation at ``--final-step`` (``VAL`` lines of
  ``screen_metrics.py``, which computes exp(``val_loss``) at full precision; a later log wins for
  the same run);
* late-25% PPL (``decisions.late25_ppl``): ``late_doc_ppl`` of ``<run>.valppl.json``, ``late_row_ppl``
  beside it;
* MQAR accuracy, statute-recall exact match (the probe's score: Absatz 2 reproduced verbatim) with
  char-F1 beside it, multi-hop exact match: ``<run>.probes.json``;
* s/step and peak GB: the ``STEPSTATS … final=1`` lines of the run's training logs, or, for a leg
  killed at its walltime (no final line), its last ``STEPSTATS`` line. A run in
  several legs (a continuation after a TIMEOUT) gets the step-weighted mean s/step and the largest
  peak.

Then the rule as written in the plan, with the bar from P5-D: the simplest arm, S1, advances unless
another arm beats it on the two-seed mean val PPL by more than the bar and does not lose
statute-recall or multi-hop by more than the bar. A lever that wins PPL but loses recall is
recorded and not adopted. An arm missing a column is reported as incomplete, never decided.
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

COLUMNS = ("val_ppl", "late_doc_ppl", "late_row_ppl", "mqar", "statute_em", "statute_f1", "multihop_em", "s_per_step", "peak_gb")  # fmt: skip
_RUN = re.compile(r"^(?P<prefix>.*?)(?P<arm>S\d+)_s(?P<seed>\d+)$")


def parse_run(name: str, prefix: str) -> tuple[str, int] | None:
    m = _RUN.match(name)
    return (m.group("arm"), int(m.group("seed"))) if m and m.group("prefix") == prefix else None


def read_val(paths, prefix: str, final_step: int) -> dict:
    out = {}
    for path in paths:
        for m in re.finditer(
            r"^VAL (\S+) step=(\d+) val_loss=\S+ val_ppl=(\S+)", Path(path).read_text(), re.M
        ):
            run = parse_run(m.group(1), prefix)
            if run and int(m.group(2)) == final_step:
                out[run] = float(m.group(3))
    return out


def read_train(paths, prefix: str) -> dict:
    legs: dict[tuple, list] = {}
    for path in paths:
        text = Path(path).read_text(errors="replace")
        exp = re.search(r"^experiment: (\S+)", text, re.M)
        run = parse_run(exp.group(1), prefix) if exp else None
        stats = re.findall(
            r"STEPSTATS rank=0 step=\d+ s_per_step=(\S+) n=(\d+) peak_alloc_gb=(\S+) \S+ final=([01])", text
        )
        final = [s for s in stats if s[3] == "1"] or stats  # a leg killed at its walltime has no final line
        if run and final:
            s, n, gb, _ = final[-1]
            legs.setdefault(run, []).append((float(s), int(n), float(gb)))
    return {
        run: {
            "s_per_step": sum(s * n for s, n, _ in v) / sum(n for _, n, _ in v),
            "peak_gb": max(gb for _, _, gb in v),
            "legs": len(v),
        }
        for run, v in legs.items()
    }


def read_eval(dirs, prefix: str) -> dict:
    out: dict[tuple, dict] = {}
    for d in dirs:
        for path in sorted(Path(d).glob("*.json")):
            name, kind = path.name.rsplit(".", 2)[:2]
            run = parse_run(name, prefix)
            if not run:
                continue
            r = json.loads(path.read_text())
            row = out.setdefault(run, {})
            if kind == "valppl":
                row.update(late_doc_ppl=r["late_doc_ppl"], late_row_ppl=r["late_row_ppl"], eval_ppl=r["ppl"],
                           late_doc_tokens=r["late_doc_tokens"], late_row_tokens=r["late_row_tokens"])  # fmt: skip
            elif kind == "probes":
                row.update(mqar=r["mqar"]["accuracy"], statute_em=r["statute_recall"]["exact_match"],
                           statute_f1=r["statute_recall"]["char_f1"], multihop_em=r["multihop"]["exact_match"],
                           statute_items=r["statute_recall"]["items"], multihop_items=r["multihop"]["items"])  # fmt: skip
    return out


def build_rows(val, train, evals) -> dict:
    rows = {}
    for run in sorted(set(val) | set(train) | set(evals), key=lambda r: (int(r[0][1:]), r[1])):
        rows[run] = {"val_ppl": val.get(run), **train.get(run, {}), **evals.get(run, {})}
    return rows


def arm_means(rows) -> dict:
    by_arm: dict[str, list] = {}
    for (arm, _), row in rows.items():
        by_arm.setdefault(arm, []).append(row)
    out = {}
    for arm, seeds in by_arm.items():
        out[arm] = {"seeds": len(seeds)}
        for c in COLUMNS:
            vals = [r[c] for r in seeds if r.get(c) is not None]
            if len(vals) == len(seeds) == 2:
                out[arm][c] = (statistics.mean(vals), statistics.stdev(vals))
    return out


def apply_rule(means: dict, bar: float, base: str = "S1") -> list[str]:
    need = ("val_ppl", "statute_em", "multihop_em")
    if any(c not in means.get(base, {}) for c in need):
        return [
            f"RULE incomplete: {base} lacks two seeds of {', '.join(c for c in need if c not in means.get(base, {}))}"
        ]
    b = means[base]
    lines, adopt = [], []
    for arm in sorted(means, key=lambda a: int(a[1:])):
        if arm == base:
            continue
        m = means[arm]
        missing = [c for c in need if c not in m]
        if missing:
            lines.append(f"RULE {arm}: incomplete ({', '.join(missing)} missing), not decided")
            continue
        gain = b["val_ppl"][0] - m["val_ppl"][0]  # > 0: the arm has the lower PPL
        loss_sr = b["statute_em"][0] - m["statute_em"][0]
        loss_mh = b["multihop_em"][0] - m["multihop_em"][0]
        beats = gain > bar
        loses = loss_sr > bar or loss_mh > bar
        verdict = "candidate: adopt" if beats and not loses else (
            "wins PPL but loses recall: recorded, not adopted" if beats else "does not beat S1 by more than the bar")  # fmt: skip
        lines.append(
            f"RULE {arm}: val PPL S1-{arm} = {gain:+.4f} (bar {bar}); statute-recall loss {loss_sr:+.4f}, "
            f"multi-hop loss {loss_mh:+.4f} -> {verdict}"
        )
        if beats and not loses:
            adopt.append(arm)
    lines.append(f"RULE winner: {base} (no lever beats it as the rule requires)" if not adopt
                 else f"RULE candidates over {base}: {', '.join(adopt)} (more than one: the user decides)" if len(adopt) > 1
                 else f"RULE winner: {adopt[0]}")  # fmt: skip
    return lines


def _f(x, spec=".4f") -> str:
    return "—" if x is None else format(x, spec)


def render(rows, means, rule) -> str:
    head = "| arm | seed | val PPL | late-25% PPL (doc) | late PPL (row) | MQAR | statute-recall EM | char-F1 | multi-hop EM | s/step | peak GB |"
    out = [head, "|" + "---|" * 11]
    for (arm, seed), r in rows.items():
        out.append(
            f"| {arm} | {seed} | {_f(r.get('val_ppl'))} | {_f(r.get('late_doc_ppl'))} | {_f(r.get('late_row_ppl'))} | "
            f"{_f(r.get('mqar'), '.3f')} | {_f(r.get('statute_em'), '.3f')} | {_f(r.get('statute_f1'), '.3f')} | "
            f"{_f(r.get('multihop_em'), '.3f')} | {_f(r.get('s_per_step'))} | {_f(r.get('peak_gb'), '.2f')} |"
        )
    out += ["", "| arm | " + " | ".join(COLUMNS) + " |", "|" + "---|" * (len(COLUMNS) + 1)]
    for arm in sorted(means, key=lambda a: int(a[1:])):
        cells = [
            "—" if means[arm].get(c) is None else "{:.4f} ± {:.4f}".format(*means[arm][c]) for c in COLUMNS
        ]
        out.append(f"| {arm} | " + " | ".join(cells) + " |")
    return "\n".join([*out, "", *rule])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="screen2_")
    ap.add_argument("--val-logs", nargs="*", default=[])
    ap.add_argument("--train-logs", nargs="*", default=[])
    ap.add_argument("--eval-dirs", nargs="*", default=[])
    ap.add_argument("--bar", type=float, required=True, help="pre_registered.P5.bar_value")
    ap.add_argument("--final-step", type=int, default=11999)
    args = ap.parse_args(argv)
    rows = build_rows(
        read_val(args.val_logs, args.prefix, args.final_step),
        read_train(args.train_logs, args.prefix),
        read_eval(args.eval_dirs, args.prefix),
    )
    means = arm_means(rows)
    print(render(rows, means, apply_rule(means, args.bar)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
