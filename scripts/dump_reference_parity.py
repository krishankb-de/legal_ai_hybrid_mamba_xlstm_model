#!/usr/bin/env python3
"""Dump the reference-parity fixtures (plan P1-X) -- run once, while Reference/ still exists.

Imports the reference package from ``Reference/hybrid_model_mamba_xlstm`` WITHOUT touching
``sys.path``: an empty package module is registered under the reference's own name with its
``__path__`` pointing into Reference/, so its absolute imports resolve there (and its heavy package
``__init__`` never runs). For each mixer it builds the REFERENCE block for several flag
combinations, seeds it, and records the state dict, a packed input with document ids, and the
outputs with and without the ids (and of ``step()`` where the block has one). ``model.pt`` does the
same for a whole 4-layer model (logits, loss) and records cached beam-search tokens for a decodable
pattern.

``tests/test_reference_parity.py`` loads each state dict into the NEW code and asserts the outputs
agree to 1e-6, forever -- after P1-Z these files are the record of what the reference computed.
This is the one file allowed to name the reference package (tests/test_port_map.py).

    .venv/bin/python scripts/dump_reference_parity.py            # writes tests/fixtures/reference_parity/
    .venv/bin/python scripts/dump_reference_parity.py --check    # regenerate in memory and compare
"""

import argparse
import importlib
import sys
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_ROOT = REPO_ROOT / "Reference" / "hybrid_model_mamba_xlstm"
REF_PKG = "hybrid_xmamba"
OUT_DIR = REPO_ROOT / "tests" / "fixtures" / "reference_parity"
REF_DOC_KWARG = "cu_seqlens"  # the reference's name for doc_ids

BATCH, SEQ, DIM = 2, 96, 64
DOC_SPLIT = 40  # doc_ids = [0]*40 + [1]*56 -- the boundary falls inside a chunk for every chunk size

# (label, kwargs) per mixer. Small enough to keep every fixture well under the 5 MB hygiene limit.
MAMBA3_CASES = [
    ("default", dict(d_state=16, head_dim=32, chunk_size=64)),
    (
        "trapezoid_rope",
        dict(d_state=16, head_dim=32, chunk_size=64, use_trapezoid=True, use_rope=True, theta_max=0.2),
    ),
    (
        "bias_noconv_dd",
        dict(
            d_state=16,
            head_dim=32,
            chunk_size=32,
            bc_bias="one_init",
            use_conv=False,
            a_mode="data_dependent",
        ),
    ),
]
MLSTM_CASES = [
    ("exact", dict(head_dim=32, tfla_impl="exact")),
    ("legacy", dict(head_dim=32, tfla_impl="legacy")),
    ("exact_hybridnorm", dict(head_dim=32, tfla_impl="exact", use_hybrid_norm=True)),
]
ATTENTION_CASES = [
    ("pre_rms", dict(num_heads=2)),
    ("qk_norm", dict(num_heads=2, use_hybrid_norm=True)),
]
MAMBA_CASES = [
    ("legacy", dict(state_size=8, scan_impl="legacy")),
    ("exact_hybrid", dict(state_size=8, scan_impl="exact", use_hybrid_norm=True, use_dt_norm=False)),
]
MODEL_CASES = [
    (
        "full",
        dict(
            vocab_size=256,
            dim=64,
            num_layers=4,
            layer_pattern=["mamba3", "mlstm", "attention", "mamba"],
            state_size=8,
            mamba3_d_state=16,
            mamba3_head_dim=32,
            head_dim=32,
            num_heads=2,
            max_position_embeddings=128,
            tfla_impl="exact",
            scan_impl="exact",
            norm_topology="hybrid",
            dropout=0.0,
        ),
    ),
    (
        "decode",
        dict(
            vocab_size=256,
            dim=64,
            num_layers=4,
            layer_pattern=["mamba3", "mlstm"],
            mamba3_d_state=16,
            mamba3_head_dim=32,
            head_dim=32,
            num_heads=2,
            max_position_embeddings=128,
            tfla_impl="exact",
            norm_topology="hybrid",
            dropout=0.0,
        ),
    ),
]


def import_reference():
    """Register the reference package without running its __init__ and without sys.path edits."""
    if not REFERENCE_ROOT.exists():
        raise SystemExit(f"{REFERENCE_ROOT} is gone (P1-Z); the committed fixtures are the record")
    if REF_PKG not in sys.modules:
        pkg = types.ModuleType(REF_PKG)
        pkg.__path__ = [str(REFERENCE_ROOT / REF_PKG)]
        sys.modules[REF_PKG] = pkg
    return {
        "mamba3": importlib.import_module(f"{REF_PKG}.layers.mamba3_block").Mamba3Block,
        "mlstm": importlib.import_module(f"{REF_PKG}.layers.mlstm_block").mLSTMBlock,
        "attention": importlib.import_module(f"{REF_PKG}.layers.attention_block").AttentionBlock,
        "mamba": importlib.import_module(f"{REF_PKG}.layers.mamba_block").MambaBlock,
        "config": importlib.import_module(f"{REF_PKG}.models.configuration_hybrid").HybridConfig,
        "model": importlib.import_module(f"{REF_PKG}.models.hybrid_lm").HybridLanguageModel,
    }


def inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(BATCH, SEQ, DIM, generator=g)
    doc_ids = torch.tensor([[0] * DOC_SPLIT + [1] * (SEQ - DOC_SPLIT)] * BATCH)
    return x, doc_ids


def block_cases(ref, kind: str, cases):
    out = []
    for label, kwargs in cases:
        torch.manual_seed(0)
        block = ref[kind](DIM, **kwargs).eval()
        x, doc_ids = inputs()
        with torch.no_grad():
            case = {
                "label": label,
                "kwargs": kwargs,
                "state_dict": {k: v.clone() for k, v in block.state_dict().items()},
                "input": x,
                "doc_ids": doc_ids,
                "output_doc": block(x, **{REF_DOC_KWARG: doc_ids}),
                "output_nodoc": block(x),
                "step_output": None,
            }
            if getattr(block, "supports_step", False):
                cache = block.allocate_inference_cache(BATCH)
                case["step_output"] = torch.stack([block.step(x[:, t], cache) for t in range(SEQ)], dim=1)
        out.append(case)
    return out


def model_cases(ref):
    out = []
    g = torch.Generator().manual_seed(1)
    for label, kwargs in MODEL_CASES:
        torch.manual_seed(0)
        model = ref["model"](ref["config"](**kwargs)).eval()
        ids = torch.randint(0, kwargs["vocab_size"], (BATCH, SEQ), generator=g)
        doc_ids = torch.tensor([[0] * DOC_SPLIT + [1] * (SEQ - DOC_SPLIT)] * BATCH)
        with torch.no_grad():
            with_doc = model(ids, labels=ids, **{REF_DOC_KWARG: doc_ids})
            without_doc = model(ids, labels=ids)
            case = {
                "label": label,
                "kwargs": kwargs,
                "state_dict": {k: v.clone() for k, v in model.state_dict().items()},
                "input_ids": ids,
                "doc_ids": doc_ids,
                "logits_doc": with_doc.logits,
                "loss_doc": with_doc.loss,
                "logits_nodoc": without_doc.logits,
                "loss_nodoc": without_doc.loss,
                "beam_prompt": None,
                "beam_tokens": None,
            }
            if model.supports_cached_decode():
                prompt = ids[:1, :6]
                case["beam_prompt"] = prompt
                case["beam_tokens"] = model.beam_search_cached(prompt, beam_size=3, max_new_tokens=10)
        out.append(case)
    return out


def build_all() -> dict[str, list]:
    ref = import_reference()
    return {
        "mamba3": block_cases(ref, "mamba3", MAMBA3_CASES),
        "mlstm": block_cases(ref, "mlstm", MLSTM_CASES),
        "attention": block_cases(ref, "attention", ATTENTION_CASES),
        "mamba": block_cases(ref, "mamba", MAMBA_CASES),
        "model": model_cases(ref),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="compare a fresh dump with the committed fixtures")
    args = ap.parse_args(argv)
    fixtures = build_all()
    if args.check:
        bad = []
        for name, cases in fixtures.items():
            stored = torch.load(OUT_DIR / f"{name}.pt", weights_only=True)
            for fresh, old in zip(cases, stored):
                for key, value in fresh.items():
                    if torch.is_tensor(value) and not torch.equal(value, old[key]):
                        bad.append(f"{name}/{fresh['label']}/{key}")
        print("fixtures reproduce" if not bad else f"fixtures differ: {bad}")
        return 1 if bad else 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, cases in fixtures.items():
        path = OUT_DIR / f"{name}.pt"
        torch.save(cases, path)
        print(f"wrote {path.relative_to(REPO_ROOT)} ({len(cases)} cases, {path.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
