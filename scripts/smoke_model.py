#!/usr/bin/env python3
"""validate.sh gate 5: a CPU forward/backward over every mixer type with the shipping operators.

Builds a 4-layer model with ``[mamba3, mlstm, attention, mamba]``, exact operators and HybridNorm,
runs a packed batch with ``doc_ids`` and labels, and asserts a finite loss and that EVERY trainable
parameter receives a gradient -- the check that catches a mixer wired up halfway. (The reference
harness built this model with ``use_tfla=False``, i.e. a different mLSTM operator than the one that
trains; this one runs the shipping path.)
"""

import sys

import torch

from lexhybrid import HybridConfig, HybridLanguageModel


def run() -> str:
    torch.manual_seed(0)
    cfg = HybridConfig(
        vocab_size=1024,
        dim=64,
        num_layers=4,
        layer_pattern=["mamba3", "mlstm", "attention", "mamba"],
        max_position_embeddings=64,
        mamba3_d_state=16,
        mamba3_head_dim=32,
        head_dim=32,
        num_heads=2,
        state_size=8,
        tfla_impl="exact",
        scan_impl="exact",
        norm_topology="hybrid",
        mlstm_forget_gate_bias_init=3.0,
    )
    model = HybridLanguageModel(cfg)
    model.train()
    ids = torch.randint(0, 1024, (2, 32))
    doc_ids = torch.tensor([[0] * 13 + [1] * 19, [0] * 32])
    out = model(ids, labels=ids, doc_ids=doc_ids)
    if out.loss is None or not torch.isfinite(out.loss):
        raise AssertionError("loss missing or not finite")
    if not torch.isfinite(out.logits).all():
        raise AssertionError("logits contain NaN/Inf")
    out.loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    if missing:
        raise AssertionError(f"parameters without gradient: {missing[:5]}")
    return f"OK  loss={out.loss.item():.4f}  logits={tuple(out.logits.shape)}  params={model.get_num_params(False):,}"


def main() -> int:
    try:
        print(run())
    except AssertionError as exc:
        print(f"FAIL: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
