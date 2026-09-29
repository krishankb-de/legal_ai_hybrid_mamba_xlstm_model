#!/usr/bin/env python3
"""Locate compile-vs-eager drift per mixer and per batch row (plan P4-J2 follow-up, job 2588779).

Job 2588779: on the H100, ``torch.compile`` of the reduced ``hybrid_legal_base`` matched eager in
batch row 0 (~1e-7) and drifted in row 1 (max 0.0575) at every position, both rows having the same
document layout. This builds the same reduced model with one mixer at a time (and the attention
mixer with each packed-row implementation), runs eager and compiled on a batch of two, and prints
the max |compiled - eager| per row, with and without ``doc_ids`` and with the rows swapped.

    .venv/bin/python scripts/diagnose_compile.py                      # CUDA, Inductor
    .venv/bin/python scripts/diagnose_compile.py --device cpu --backend eager --seq-len 64
"""

import argparse

import torch

from lexhybrid import HybridLanguageModel
from lexhybrid.config import load_model_config

VARIANTS = {
    "full": {},
    "mamba3": {"layer_pattern": ["mamba3"], "num_layers": 2},
    "mlstm": {"layer_pattern": ["mlstm"], "num_layers": 2},
    "attention-flex": {"layer_pattern": ["attention"], "num_layers": 2, "attn_impl": "flex"},
    "attention-sdpa": {"layer_pattern": ["attention"], "num_layers": 2, "attn_impl": "sdpa"},
}


def build(variant: str, device: str, dim: int) -> HybridLanguageModel:
    torch.manual_seed(0)
    cfg = load_model_config(
        "hybrid_legal_base",
        dim=dim,
        num_heads=2,
        head_dim=dim // 2,
        mamba3_head_dim=dim // 2,
        vocab_size=1024,
        max_position_embeddings=4096,
        **VARIANTS[variant],
    )
    return HybridLanguageModel(cfg).to(device).eval()


def row_drift(model, ids, doc, backend: str) -> list[float]:
    with torch.no_grad():
        eager = model(ids, doc_ids=doc).logits
        torch._dynamo.reset()
        got = torch.compile(model, dynamic=False, backend=backend)(ids, doc_ids=doc).logits
    return [(got[b] - eager[b]).abs().max().item() for b in range(ids.shape[0])]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", default="inductor")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    args = parser.parse_args(argv)

    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print("GPU:", torch.cuda.get_device_name(0))
    print(f"torch {torch.__version__}  backend={args.backend}  seq_len={args.seq_len}  dim={args.dim}")

    torch.manual_seed(1)
    ids = torch.randint(0, 1024, (2, args.seq_len), device=args.device)
    doc = torch.zeros(2, args.seq_len, dtype=torch.long, device=args.device)
    doc[:, args.seq_len * 2 // 5 :] = 1
    cases = {"packed": (ids, doc), "unpacked": (ids, None), "packed-swapped": (ids.flip(0), doc)}

    for variant in args.variants:
        model = build(variant, args.device, args.dim)
        for case, (x, d) in cases.items():
            r0, r1 = row_drift(model, x, d, args.backend)
            flag = "DRIFT" if max(r0, r1) >= 1e-4 else "ok"
            print(f"CASE variant={variant:15s} input={case:15s} row0={r0:.3e} row1={r1:.3e} {flag}")
    print("DIAGNOSE_COMPILE DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
