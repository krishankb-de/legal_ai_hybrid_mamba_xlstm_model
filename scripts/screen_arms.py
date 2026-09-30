#!/usr/bin/env python3
"""The P5 screen's arm table: one source for preflight, submission and the ARCH check (plan P2-X).

    .venv/bin/python scripts/screen_arms.py list
    .venv/bin/python scripts/screen_arms.py env S1-s43        # what the SLURM wrapper evals on the node
    .venv/bin/python scripts/screen_arms.py verify --reduced  # small width, one forward/backward per arm
    .venv/bin/python scripts/screen_arms.py verify --full     # full size on `meta`, plus parameter bands

Pattern of the reference's ``scripts/mamba3_arms.py`` (port map §11.5): every arm is a config, the
model overrides that make it, the one lever it isolates (R3), the tokens its ARCH line must show
(so a lever that silently fails to reach the model stops the job at step 0 -- FM5), and a walltime.
``ARM`` names are ``<arm>-s<seed>`` and are resolved on the compute node, never on the login node.

The screen settings (plan P5) are one table, ``SCREEN``, emitted as Hydra overrides by ``env``.
S5 (legacy recipe) is a stated difference, +2.36% inside ±2.5% (the user's decision of 2026-09-27,
plan §14): it measures the old operators, PPL/probe-only, and is not reshaped to match.
"""

import argparse
import dataclasses
import shlex
import sys
from pathlib import Path
from typing import Any, NamedTuple

import torch

from lexhybrid import HybridConfig, HybridLanguageModel
from lexhybrid.config import load_model_config

REPO_ROOT = Path(__file__).resolve().parent.parent
SEEDS = (42, 43)
BASE_ARM = "S1"

# Plan P5: 12,000 steps, 2,048-token rows, micro-batch 8 x accum 2 (32,768 tokens/step), warmup 500,
# LR 4e-4, clip 1.0, WSD, SAVE_TOP_K 0, validation every 4,000 optimizer steps.
SCREEN: dict[str, Any] = {
    "trainer.max_steps": 12000,
    "trainer.accumulate_grad_batches": 2,
    "trainer.val_every_opt_steps": 4000,
    "dataset.row_len": 2048,
    "dataset.batch_size": 8,
    "model.warmup_steps": 500,
    "model.learning_rate": 4.0e-4,
    "model.gradient_clip_val": 1.0,
    "model.scheduler": "wsd",
    "callbacks.checkpoint.save_top_k": 0,
}


class Arm(NamedTuple):
    config: str
    overrides: dict[str, Any]  # model fields on top of the yaml
    isolates: str  # the one lever against S1
    expect: list[str]  # substrings the ARCH line must contain
    forbid: list[str] = []  # substrings it must not contain
    walltime: str = "12:00:00"
    extra: dict[str, Any] = {}  # non-model Hydra overrides (e.g. the KD weight)
    band_pct: float | None = 3.0  # |non-embedding delta vs S1| allowed; None for S1


BASE_TOKENS = [
    "tfla_impl=exact",
    "vocab=151936",
    "tied=True",
    "mlstm(chunk_size=128, forget_bias=3.0, fallback=error)",
    "attn(rope_theta=500000",
]

ARMS: dict[str, Arm] = {
    "S1": Arm(
        "hybrid_legal_base",
        {},
        "control",
        ["mamba3x7", "attentionx2", "mlstmx3", *BASE_TOKENS],
        band_pct=None,
    ),
    "S0": Arm(
        "hybrid_legal_attn0",
        {},
        "no attention",
        ["mamba3x9", "mlstmx3", "tfla_impl=exact"],
        forbid=["attention"],
    ),
    "S2": Arm(
        "hybrid_legal_attn4",
        {},
        "four attention layers",
        ["mamba3x5", "attentionx4", "mlstmx3", *BASE_TOKENS],
    ),
    "S3": Arm(
        "hybrid_legal_ds64", {}, "mamba3_d_state 64", ["mamba3(d_state=64", "attentionx2", *BASE_TOKENS]
    ),
    "S4": Arm("hybrid_legal_mtp", {}, "multi-token prediction", ["mtp(n=2", "attentionx2", *BASE_TOKENS]),
    "S5": Arm(
        "hybrid_legal_legacy",
        {},
        "incumbent recipe (legacy operators); PPL/probe only",
        ["mambax9", "mlstmx3", "scan_impl=legacy", "tfla_impl=legacy"],
        forbid=["attention", "mamba3("],
        walltime="24:00:00",
        band_pct=2.5,  # +2.36%, delta reported (user decision 2026-09-27)
    ),
    "S6": Arm(
        "hybrid_legal_base",
        {},
        "no KD (alpha 0)",
        ["mamba3x7", "attentionx2", "mlstmx3", *BASE_TOKENS],
        extra={"distill.alpha": 0.0},
        band_pct=0.0,
    ),
}


def parse_name(name: str) -> tuple[str, int]:
    """``"S1-s43"`` -> ("S1", 43); a bare ``"S1"`` takes the first seed."""
    arm, _, seed = name.partition("-s")
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; arms are {sorted(ARMS)}")
    seed = int(seed) if seed else SEEDS[0]
    if seed not in SEEDS:
        raise ValueError(f"seed {seed} is not a screen seed {SEEDS}")
    return arm, seed


def _fmt(value) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value)


def hydra_overrides(arm_name: str) -> list[str]:
    arm = ARMS[arm_name]
    out = [f"{k}={_fmt(v)}" for k, v in SCREEN.items()]
    out += [f"model.{k}={_fmt(v)}" for k, v in sorted(arm.overrides.items())]
    out += [f"{k}={_fmt(v)}" for k, v in sorted(arm.extra.items())]
    return out


def env_lines(name: str) -> list[str]:
    """The exports a SLURM wrapper ``eval``s on the compute node."""
    arm_name, seed = parse_name(name)
    arm = ARMS[arm_name]
    exports = {
        "ARM": f"{arm_name}-s{seed}",
        "MODEL_CONFIG": arm.config,
        "SEED": str(seed),
        "EXPERIMENT": f"screen_{arm_name}_s{seed}",
        "SAVE_TOP_K": "0",
        "EXTRA_OVERRIDES": " ".join(hydra_overrides(arm_name)),
        "ARM_EXPECT": "|".join(arm.expect),
        "ARM_FORBID": "|".join(arm.forbid),
        "ARM_WALLTIME": arm.walltime,
    }
    return [f"export {k}={shlex.quote(v)}" for k, v in exports.items()]


def build_config(arm_name: str, reduced: bool = False) -> HybridConfig:
    arm = ARMS[arm_name]
    fields = {f.name for f in dataclasses.fields(HybridConfig)}
    unknown = sorted(set(arm.overrides) - fields)
    if unknown:
        raise ValueError(f"{arm_name}: overrides {unknown} are not HybridConfig fields")
    overrides = dict(arm.overrides)
    if reduced:  # width only: every lever keeps its configured value
        overrides.update(dim=64, num_heads=2, head_dim=32, mamba3_head_dim=32, max_position_embeddings=128)
    return load_model_config(arm.config, **overrides)


def check_tokens(arm_name: str, fingerprint: str) -> list[str]:
    arm = ARMS[arm_name]
    problems = [f"{arm_name}: ARCH lacks {tok!r}" for tok in arm.expect if tok not in fingerprint]
    problems += [f"{arm_name}: ARCH shows forbidden {tok!r}" for tok in arm.forbid if tok in fingerprint]
    return problems


def _inference_non_embedding(model: HybridLanguageModel) -> int:
    mtp = sum(p.numel() for p in model.mtp_head.parameters()) if model.mtp_head is not None else 0
    return model.get_num_params(non_embedding=True) - mtp


def verify(reduced: bool, full: bool) -> list[str]:
    problems: list[str] = []
    counts: dict[str, int] = {}
    for arm_name in ARMS:
        cfg = build_config(arm_name, reduced=reduced and not full)
        if full:
            with torch.device("meta"):
                model = HybridLanguageModel(cfg)
            counts[arm_name] = _inference_non_embedding(model)
        else:
            torch.manual_seed(0)
            model = HybridLanguageModel(cfg)
            ids = torch.randint(0, cfg.vocab_size, (1, 64))
            doc = torch.zeros(1, 64, dtype=torch.long)
            doc[:, 29:] = 1
            out = model(ids, labels=ids, doc_ids=doc)
            out.loss.backward()
            if not torch.isfinite(out.loss):
                problems.append(f"{arm_name}: non-finite loss at reduced width")
        fp = model.architecture_fingerprint()
        problems += check_tokens(arm_name, fp)
        print(f"{arm_name:3s} {'ok ' if not check_tokens(arm_name, fp) else 'BAD'} {fp}")
    if full:
        base = counts[BASE_ARM]
        for arm_name, arm in ARMS.items():
            delta = 100.0 * (counts[arm_name] - base) / base
            ok = arm.band_pct is None or abs(delta) <= arm.band_pct + 1e-9
            band = "control" if arm.band_pct is None else f"band +-{arm.band_pct:g}%"
            print(
                f"    {arm_name}: non-embedding {counts[arm_name]:,} ({delta:+.2f}%, {band}) {'ok' if ok else 'FAIL'}"
            )
            if not ok:
                problems.append(
                    f"{arm_name}: non-embedding delta {delta:+.2f}% outside the pre-registered {band}"
                )
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p_env = sub.add_parser("env")
    p_env.add_argument("arm", help="e.g. S1-s43")
    p_ver = sub.add_parser("verify")
    size = p_ver.add_mutually_exclusive_group()
    size.add_argument("--reduced", action="store_true", help="small width, real forward/backward")
    size.add_argument("--full", action="store_true", help="full size on meta, plus parameter bands")
    args = parser.parse_args(argv)

    if args.cmd == "list":
        for name, arm in ARMS.items():
            print(f"{name}  {arm.config:22s} {arm.isolates:45s} walltime {arm.walltime}")
        return 0
    if args.cmd == "env":
        print("\n".join(env_lines(args.arm)))
        return 0
    problems = verify(reduced=args.reduced or not args.full, full=args.full)
    for p in problems:
        print("PROBLEM", p, file=sys.stderr)
    print(f"{len(ARMS)} arms, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
