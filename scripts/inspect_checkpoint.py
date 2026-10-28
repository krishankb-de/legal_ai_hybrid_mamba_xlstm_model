#!/usr/bin/env python3
"""Per-parameter weights and AdamW state of a training checkpoint (decisions.p5_divergence).

    .venv/bin/python scripts/inspect_checkpoint.py --checkpoint outputs/screen_S1_s42/checkpoints/last.ckpt \\
        --model-config hybrid_legal_base --label screen_S1_s42

Screen attempt 1 left a collapsed ``last.ckpt`` (S1-s42, S3-s43) next to healthy ones of the same
configs (S1-s43, S3-s42). Comparing them shows what the collapse left behind: a norm weight that
went to zero, a gate bias that ran away, an AdamW second moment that is tiny where the gradient now
is not. Prints one line per parameter, then one per part (``embed``, ``L<i>_<type>``,
``final_norm``, ``mtp``)::

    PARAM <label> <name> shape=<s> norm=… max=… mean=… v_sqrt_max=… v_sqrt_median=… m_norm=… adam_rms=…
    PART <label> <part> norm=… max=… adam_rms=…

``v_sqrt`` is ``sqrt(exp_avg_sq)``. ``adam_rms`` is the RMS of ``exp_avg / (sqrt(exp_avg_sq) + eps)``,
the size of the next AdamW update in units of the LR, before bias correction. The optimizer state is
mapped to names through ``get_parameter_groups``, the grouping the run was trained with.
"""

import argparse
import math
import re
import sys

import torch

from lexhybrid.config.loading import load_model_config
from lexhybrid.models.hybrid_lm import HybridLanguageModel
from lexhybrid.training.optimizer import get_parameter_groups
from lexhybrid.utils.checkpoint import strip_prefixes

_LAYER = re.compile(r"layers\.(\d+)\.")


def part_of(name: str, types: list[str]) -> str:
    m = _LAYER.match(name)
    if m:
        return f"L{int(m.group(1))}_{types[int(m.group(1))]}"
    for prefix, part in (("embeddings.", "embed"), ("final_norm.", "final_norm"), ("mtp_head.", "mtp")):
        if name.startswith(prefix):
            return part
    return "other"


def optimizer_names(model: torch.nn.Module) -> list[str]:
    """Parameter names in the order AdamW's state indexes them (group by group)."""
    name_of = {id(p): n for n, p in model.named_parameters()}
    return [name_of[id(p)] for group in get_parameter_groups(model) for p in group["params"]]


def _f(x: float) -> str:
    return f"{x:.4g}" if math.isfinite(x) else str(x)


def inspect(checkpoint: str, model_config: str, label: str, eps: float = 1e-8) -> list[str]:
    config = load_model_config(model_config)
    with torch.device("meta"):
        model = HybridLanguageModel(config)
    types = model.get_layer_types()
    names = optimizer_names(model)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = strip_prefixes(ckpt.get("state_dict", ckpt))
    opt_state = {}
    if ckpt.get("optimizer_states"):
        state = ckpt["optimizer_states"][0]
        n_indexed = sum(len(g["params"]) for g in state["param_groups"])
        if n_indexed != len(names):
            raise RuntimeError(f"optimizer holds {n_indexed} parameters, the model {len(names)}")
        opt_state = {names[i]: s for i, s in state["state"].items()}
    out = [f"CKPT {label} {checkpoint} global_step={ckpt.get('global_step')} params={len(names)}"]
    parts: dict[str, dict[str, float]] = {}
    for name in names:
        w = weights[name].float()
        line = f"PARAM {label} {name} shape={'x'.join(map(str, w.shape))} norm={_f(w.norm().item())}"
        line += f" max={_f(w.abs().max().item())} mean={_f(w.mean().item())}"
        agg = parts.setdefault(part_of(name, types), {"sq": 0.0, "max": 0.0, "adam_sq": 0.0, "n": 0})
        agg["sq"] += w.norm().item() ** 2
        agg["max"] = max(agg["max"], w.abs().max().item())
        s = opt_state.get(name)
        if s is not None and "exp_avg_sq" in s:
            v_sqrt = s["exp_avg_sq"].float().sqrt()
            ratio = s["exp_avg"].float() / (v_sqrt + eps)
            line += f" v_sqrt_max={_f(v_sqrt.max().item())} v_sqrt_median={_f(v_sqrt.median().item())}"
            line += f" m_norm={_f(s['exp_avg'].float().norm().item())} adam_rms={_f(ratio.pow(2).mean().sqrt().item())}"
            agg["adam_sq"] += ratio.pow(2).sum().item()
            agg["n"] += ratio.numel()
        out.append(line)
    for part, agg in parts.items():
        adam = math.sqrt(agg["adam_sq"] / agg["n"]) if agg["n"] else math.nan
        out.append(
            f"PART {label} {part} norm={_f(math.sqrt(agg['sq']))} max={_f(agg['max'])} adam_rms={_f(adam)}"
        )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-config", required=True, help="configs/model/<name>.yaml or a yaml path")
    ap.add_argument("--label", default=None)
    args = ap.parse_args(argv)
    print("\n".join(inspect(args.checkpoint, args.model_config, args.label or args.checkpoint)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
