#!/usr/bin/env python3
"""Rule R1 -- the equivalence gate every operator or speed change has to pass.

Ported from the reference ``scripts/check_operator_equivalence.py``. A faster operator that
changes a decoded token would invalidate every downstream number, so speed is measured second:

  * a variant must match the current operator to rel-max-err <= 1e-4 in fp32, and
  * it must agree with ``ssd_sequential_reference`` (fp64) at least as well as the current operator
    does (within a factor of 2, or 1e-4), and
  * a mismatch on a document boundary is an automatic fail.

Two levels are checked because they fail differently: the operator level catches a wrong
recurrence; the model level catches everything else a compiler might do to a forward pass.

    .venv/bin/python scripts/check_operator_equivalence.py --device cpu
    .venv/bin/python scripts/check_operator_equivalence.py --device cuda --compile

Exit code is 0 only if every check passes, so a SLURM wrapper can gate a measurement on it.
"""

import argparse
import sys

import torch

from lexhybrid.config.loading import load_model_config
from lexhybrid.kernels.ssd import ssd_chunked_scan, ssd_sequential_reference
from lexhybrid.layers.mamba3_block import Mamba3Block
from lexhybrid.models.hybrid_lm import HybridLanguageModel

R1_TOLERANCE = 1e-4


def rel_max_err(a, b):
    """max |a-b| / max(|b|, eps), computed in fp64."""
    a, b = a.double(), b.double()
    denom = b.abs().max().clamp(min=1e-12)
    return ((a - b).abs().max() / denom).item()


def _operands(batch, seqlen, nheads, headdim, ngroups, dstate, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def mk(*shape):
        return torch.randn(*shape, generator=g).to(device=device, dtype=torch.float32)

    x = mk(batch, seqlen, nheads, headdim)
    # dt is positive and A negative, as softplus / -exp produce in the real block.
    dt = torch.nn.functional.softplus(mk(batch, seqlen, nheads))
    A = -torch.exp(mk(nheads))
    B = mk(batch, seqlen, ngroups, dstate)
    C = mk(batch, seqlen, ngroups, dstate)
    D = mk(nheads)
    return x, dt, A, B, C, D


def check_operator(device, chunk_sizes, verbose=True, baseline_chunk=64):
    """Operator level: does changing chunk_size change the function the scan computes?"""
    failures = []
    batch, seqlen, nheads, headdim, ngroups, dstate = 2, 256, 4, 32, 1, 64
    # A document boundary that falls INSIDE a chunk for every chunk size tested -- where the
    # state-reset masks are hardest and where the Mamba-1 defect lived.
    seg = torch.zeros(batch, seqlen, dtype=torch.long, device=device)
    seg[:, 100:] = 1
    seg[1, 173:] = 2
    cases = [("contiguous", None), ("doc_ids (boundary mid-chunk)", seg)]

    for case_name, ids in cases:
        x, dt, A, B, C, D = _operands(batch, seqlen, nheads, headdim, ngroups, dstate, device)
        oracle = ssd_sequential_reference(x, dt, A, B, C, D=D, doc_ids=ids).float()
        baseline = ssd_chunked_scan(x, dt, A, B, C, D=D, chunk_size=baseline_chunk, doc_ids=ids)
        base_err = rel_max_err(baseline, oracle)
        if verbose:
            print(f"\n  case: {case_name}")
            print(f"    chunk_size={baseline_chunk:>3} (baseline) vs fp64 oracle : {base_err:.3e}")
        for cs in chunk_sizes:
            variant = ssd_chunked_scan(x, dt, A, B, C, D=D, chunk_size=cs, doc_ids=ids)
            vs_base = rel_max_err(variant, baseline)
            vs_oracle = rel_max_err(variant, oracle)
            ok = vs_base <= R1_TOLERANCE and vs_oracle <= max(base_err * 2.0, R1_TOLERANCE)
            if verbose:
                print(
                    f"    chunk_size={cs:>3}            vs baseline   : {vs_base:.3e}   "
                    f"vs oracle: {vs_oracle:.3e}   {'PASS' if ok else 'FAIL'}"
                )
            if not ok:
                failures.append(f"operator/{case_name}/chunk_size={cs}")
    return failures


def set_mamba3_chunk_size(model, chunk_size):
    """The sweep is over the SSD chunk (``mamba3_chunk_size``) only. The mLSTM mixers also carry a
    ``chunk_size`` (the TFLA chunk, ``mlstm_chunk_size``), which may differ: ref_hybrid_m3 runs
    Mamba-3 at 64 and mLSTM at 128, and resetting both to 64 moved its logits by 4.5e-4 (job 2589360)."""
    for layer in model.layers:
        if isinstance(layer.mixer, Mamba3Block):
            layer.mixer.chunk_size = chunk_size


def check_model(device, model_name, chunk_sizes, seq_length, do_compile, verbose=True, dim=None):
    """Model level: do the logits move? This is what a decoded token actually sees."""
    failures = []
    torch.manual_seed(0)
    overrides = {}
    if dim is not None:  # reduced-size variant for CPU tests
        overrides = dict(
            dim=dim, num_heads=max(1, dim // 32), head_dim=32, mamba3_head_dim=32, mamba3_d_state=32
        )
    config = load_model_config(model_name, **overrides)
    baseline_chunk = config.mamba3_chunk_size
    model = HybridLanguageModel(config).to(device=device, dtype=torch.float32).eval()
    input_ids = torch.randint(0, config.vocab_size, (2, seq_length), device=device)

    with torch.no_grad():
        baseline = model(input_ids).logits.float()
    if verbose:
        print(f"\n  model {model_name} @ L={seq_length}, chunk_size={baseline_chunk} (baseline)")

    for cs in chunk_sizes:
        set_mamba3_chunk_size(model, cs)
        with torch.no_grad():
            variant = model(input_ids).logits.float()
        err = rel_max_err(variant, baseline)
        ok = err <= R1_TOLERANCE
        if verbose:
            print(f"    chunk_size={cs:>3} logits vs baseline : {err:.3e}   {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"model/chunk_size={cs}")
    set_mamba3_chunk_size(model, baseline_chunk)

    if do_compile:
        compiled = torch.compile(model)
        with torch.no_grad():
            got = compiled(input_ids).logits.float()
        err = rel_max_err(got, baseline)
        ok = err <= R1_TOLERANCE
        if verbose:
            print(f"    torch.compile  logits vs baseline : {err:.3e}   {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append("model/torch.compile")
    return failures


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model", default="ref_hybrid_m3")
    ap.add_argument("--chunk-sizes", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument(
        "--seq-length",
        type=int,
        default=512,
        help="model-level check length; must exceed the largest chunk size for the comparison to mean anything",
    )
    ap.add_argument("--dim", type=int, default=None, help="shrink the model's width (CPU tests)")
    ap.add_argument(
        "--compile", dest="do_compile", action="store_true", help="also check torch.compile logits"
    )
    ap.add_argument("--skip-model", action="store_true", help="operator level only")
    args = ap.parse_args(argv)

    print("=" * 78)
    print(f"R1 equivalence gate   device={args.device}  tolerance={R1_TOLERANCE:.0e}")
    print("=" * 78)
    failures = check_operator(args.device, args.chunk_sizes)
    if not args.skip_model:
        failures += check_model(
            args.device, args.model, args.chunk_sizes, args.seq_length, args.do_compile, dim=args.dim
        )

    print("\n" + "=" * 78)
    if failures:
        print(f"R1 FAILED: {', '.join(failures)}")
        print("These variants change the function the model computes and must NOT be adopted.")
        print("=" * 78)
        return 1
    print("R1 PASSED: every variant computes the same function as the baseline operator.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
