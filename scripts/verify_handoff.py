#!/usr/bin/env python3
"""Verify that a training checkpoint loads cleanly into the model its yaml builds.

Ported from the reference ``verify_stage0_handoff.py``: the static guard for the "silent fresh
load" (``strict=False`` hides a config mismatch, the backbone loads nothing, the metrics are
wrong). PASS means 0 missing and 0 unexpected keys after taking the ``model.``-prefixed entries of
the Lightning checkpoint. Exit code 0 on PASS, 1 otherwise, so a SLURM wrapper can gate on it.

    .venv/bin/python scripts/verify_handoff.py outputs/<run>/checkpoints/last.ckpt configs/model/hybrid_legal_base.yaml
"""

import argparse
import sys

import torch

from lexhybrid.config.loading import load_model_config
from lexhybrid.models.hybrid_lm import HybridLanguageModel


def model_state_from_checkpoint(ckpt_path: str) -> dict:
    """The ``model.``-prefixed keys of a Lightning checkpoint (or a bare state dict), prefix removed."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    prefixed = {k[len("model.") :]: v for k, v in sd.items() if k.startswith("model.")}
    return prefixed or dict(sd)


def verify(ckpt_path: str, model_yaml: str) -> bool:
    config = load_model_config(model_yaml)
    model = HybridLanguageModel(config)
    state = model_state_from_checkpoint(ckpt_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("  " + model.architecture_fingerprint())
    print(f"  checkpoint keys: {len(state)}   model keys: {len(model.state_dict())}")
    print(f"  MISSING   (in model, not in checkpoint): {len(missing)}  {list(missing)[:8]}")
    print(f"  UNEXPECTED(in checkpoint, not in model): {len(unexpected)}  {list(unexpected)[:8]}")
    ok = not missing and not unexpected and len(state) > 0
    print(
        "RESULT: " + ("PASS -- clean exact-match handoff" if ok else "FAIL -- see missing/unexpected above")
    )
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint")
    ap.add_argument("model_yaml")
    args = ap.parse_args(argv)
    return 0 if verify(args.checkpoint, args.model_yaml) else 1


if __name__ == "__main__":
    sys.exit(main())
