#!/usr/bin/env python3
"""Parameter counts of the legal model configs (plan P2-W) -> ``analysis/param_counts.md``.

    .venv/bin/python scripts/param_counts.py            # print the table
    .venv/bin/python scripts/param_counts.py --write    # (re)write analysis/param_counts.md
    .venv/bin/python scripts/param_counts.py --check    # exit 1 if the file is stale

Models are built on the ``meta`` device, so no weights are allocated. Counts follow decision 11:
``non_embedding = total - embedding - (lm_head if untied)``. The MTP head is training-only (it is
discarded at inference), so it is reported in its own column and left out of the inference counts
the parameter bands compare. ``tests/test_config_invariants.py::test_param_bands`` asserts the bands
and that the committed file equals this script's output.
"""

import argparse
import sys
from pathlib import Path

import torch

from lexhybrid import HybridLanguageModel
from lexhybrid.config import load_model_config

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "analysis" / "param_counts.md"
BASE = "hybrid_legal_base"
# (config, role) in table order.
CONFIGS = [
    ("hybrid_legal_base", "base; screen arm S1 (control)"),
    ("transformer_legal_base", "matched Transformer (plan §4)"),
    ("hybrid_legal_legacy", "legacy-operator ablation (decision 17; arm S5)"),
    ("hybrid_legal_attn0", "screen arm S0: no attention"),
    ("hybrid_legal_attn4", "screen arm S2: four attention layers"),
    ("hybrid_legal_ds64", "screen arm S3: Mamba-3 d_state 64"),
    ("hybrid_legal_mtp", "screen arm S4: multi-token prediction"),
]


def count(name: str) -> dict:
    with torch.device("meta"):
        model = HybridLanguageModel(load_model_config(name))
    mtp = sum(p.numel() for p in model.mtp_head.parameters()) if model.mtp_head is not None else 0
    types = model.get_layer_types()
    return {
        "name": name,
        "non_embedding": model.get_num_params(non_embedding=True) - mtp,
        "total": model.get_num_params(non_embedding=False) - mtp,
        "mtp_head": mtp,
        "layers": ", ".join(
            f"{k}x{types.count(k)}" for k in ("mamba3", "attention", "mlstm", "mamba") if k in types
        ),
    }


def all_counts() -> list[dict]:
    rows = [count(name) | {"role": role} for name, role in CONFIGS]
    base = next(r for r in rows if r["name"] == BASE)["non_embedding"]
    for r in rows:
        r["delta_pct"] = 100.0 * (r["non_embedding"] - base) / base
    return rows


def render(rows: list[dict]) -> str:
    lines = [
        "# Parameter counts of the legal model configs",
        "",
        "> Computed from the model definitions by `scripts/param_counts.py` (models built on the `meta`",
        "> device; no job, no measurement). `tests/test_config_invariants.py::test_param_bands` fails if",
        "> this file is stale and asserts the plan's bands (P2-W). Counts follow decision 11:",
        "> non-embedding = total - embedding - (lm_head if untied). The MTP head is training-only and is",
        "> reported in its own column, outside the inference counts.",
        "",
        "| config | role | layers | non-embedding | total | MTP head (training only) | Δ non-embedding vs base |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        mtp = f"{r['mtp_head']:,}" if r["mtp_head"] else "—"
        lines.append(
            f"| `{r['name']}` | {r['role']} | {r['layers']} | {r['non_embedding']:,} | {r['total']:,} "
            f"| {mtp} | {r['delta_pct']:+.2f}% |"
        )
    lines += [
        "",
        "Bands (P2-W): Transformer within ±2.5% of the base; each screen variant within ±3%, its delta",
        "reported. The legacy ablation (S5) was pre-registered at <= 1% but measures +2.36% for the",
        "9 x mamba + 3 x mlstm shape; by the user's decision of 2026-09-27 (plan §14) it is reported as a",
        "stated difference inside ±2.5%, not reshaped: S5 measures the old recipe, PPL/probe-only, and",
        "the difference favours legacy, so a hybrid win is conservative and a legacy win inside the",
        "margin is inconclusive.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="write analysis/param_counts.md")
    mode.add_argument("--check", action="store_true", help="exit 1 if analysis/param_counts.md is stale")
    args = parser.parse_args(argv)
    text = render(all_counts())
    if args.write:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(text, encoding="utf-8")
        print(f"wrote {OUTPUT.relative_to(REPO_ROOT)}")
        return 0
    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != text:
            print(
                f"{OUTPUT.relative_to(REPO_ROOT)} is stale: run scripts/param_counts.py --write",
                file=sys.stderr,
            )
            return 1
        print(f"{OUTPUT.relative_to(REPO_ROOT)} is current")
        return 0
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
