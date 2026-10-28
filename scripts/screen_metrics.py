#!/usr/bin/env python3
"""Validation and training curves of the screen runs, read from their TensorBoard events.

    .venv/bin/python scripts/screen_metrics.py --outputs $SCRATCH_ROOT/outputs --window 250

``val/perplexity`` reaches only the progress bar (not rendered into a SLURM log) and TensorBoard
under ``outputs/<exp>/logs/<exp>/version_<n>/`` on scratch, which is never pulled. This
prints, per run directory matching ``--prefix``, from its highest ``version_<n>`` (an earlier
version is an earlier attempt of the same experiment):

    RUN <exp> version=<n> last_step=<s>
    VAL <exp> step=<s> val_loss=<x> val_ppl=<exp(x)> logged_ppl=<y>   one per validation
    CURVE <exp> steps=<a>-<b> loss=… ce=… kl=… grad_norm=… grad_norm_max=… nan_skips=… lr=…

``val_ppl`` is ``exp(val_loss)``: ``val/loss`` is the token-weighted mean CE, but the logged
``val/perplexity`` (``logged_ppl``) is the token-weighted mean of per-batch ``exp(CE)``, which is
larger by Jensen's inequality and noisy (e.g. 31.69 against exp(loss) = 8.19); it is not the
validation PPL. ``CURVE`` rows are means over ``--window`` optimizer steps (``grad_norm_max`` the maximum,
``nan_skips`` and ``lr`` the last value); a tag the run never logged is left out.

``--dump REGEX`` adds every logged value of the matching tags, optionally only in ``--steps A-B``
(e.g. the ``diag/`` tags of a spike-diagnosis run, ``lexhybrid/training/diagnostics.py``):

    SCALAR <exp> <tag> step=<s> value=<v>
"""

import argparse
import math
import re
import sys
from pathlib import Path

CURVE_TAGS = {"loss": "train/loss", "ce": "train/ce", "kl": "train/kl", "grad_norm": "train/grad_norm"}
LAST_TAGS = {"nan_skips": "train/nan_skips", "lr": "train/lr"}


def latest_version(run: Path) -> Path | None:
    """``<run>/logs/<exp>/version_<n>`` with the highest n that holds an events file."""
    versions = [
        d
        for d in (run / "logs" / run.name).glob("version_*")
        if re.fullmatch(r"version_\d+", d.name) and any(d.glob("events.out.tfevents.*"))
    ]
    return max(versions, key=lambda d: int(d.name.split("_")[1]), default=None)


def read_scalars(version: Path) -> dict[str, list[tuple[int, float]]]:
    from tensorboard.backend.event_processing.event_accumulator import SCALARS, EventAccumulator

    acc = EventAccumulator(str(version), size_guidance={SCALARS: 0})
    acc.Reload()
    return {tag: [(e.step, e.value) for e in acc.Scalars(tag)] for tag in acc.Tags()["scalars"]}


def _fmt(x: float) -> str:
    return f"{x:.4f}" if math.isfinite(x) else str(x)


def report(
    run: Path, window: int, dump: str | None = None, steps: tuple[int, int] | None = None
) -> list[str]:
    version = latest_version(run)
    if version is None:
        return [f"RUN {run.name} version=none (no events file)"]
    scalars = read_scalars(version)
    last = max((s for series in scalars.values() for s, _ in series), default=-1)
    out = [f"RUN {run.name} version={version.name.split('_')[1]} last_step={last}"]
    logged = dict(scalars.get("val/perplexity", []))
    for step, loss in scalars.get("val/loss", []):
        ppl = math.exp(loss) if loss < 700 else math.inf
        line = f"VAL {run.name} step={step} val_loss={_fmt(loss)} val_ppl={_fmt(ppl)}"
        out.append(line + (f" logged_ppl={_fmt(logged[step])}" if step in logged else ""))
    windows: dict[int, dict[str, list[float]]] = {}
    for key, tag in {**CURVE_TAGS, **LAST_TAGS}.items():
        for step, value in scalars.get(tag, []):
            windows.setdefault(step // window, {}).setdefault(key, []).append(value)
    for w in sorted(windows):
        cols = windows[w]
        parts = [f"{k}={_fmt(sum(cols[k]) / len(cols[k]))}" for k in CURVE_TAGS if k in cols]
        if "grad_norm" in cols:
            parts.append(f"grad_norm_max={_fmt(max(cols['grad_norm']))}")
        parts += [f"{k}={_fmt(cols[k][-1])}" for k in LAST_TAGS if k in cols]
        out.append(f"CURVE {run.name} steps={w * window}-{(w + 1) * window - 1} " + " ".join(parts))
    if dump:
        lo, hi = steps or (0, math.inf)
        for tag in sorted(t for t in scalars if re.search(dump, t)):
            out += [
                f"SCALAR {run.name} {tag} step={s} value={_fmt(v)}" for s, v in scalars[tag] if lo <= s <= hi
            ]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", required=True, help="the outputs/ directory on scratch")
    ap.add_argument("--prefix", default="screen_", help="run directories to read")
    ap.add_argument("--window", type=int, default=250, help="optimizer steps per CURVE row")
    ap.add_argument("--dump", default=None, help="regex: print every value of the matching tags")
    ap.add_argument("--steps", default=None, help="A-B: limit --dump to these optimizer steps")
    args = ap.parse_args(argv)
    steps = tuple(int(x) for x in args.steps.split("-")) if args.steps else None
    runs = sorted(p for p in Path(args.outputs).glob(f"{args.prefix}*") if p.is_dir())
    if not runs:
        print(f"FATAL: no {args.prefix}* run under {args.outputs}", file=sys.stderr)
        return 1
    for run in runs:
        print("\n".join(report(run, args.window, args.dump, steps)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
