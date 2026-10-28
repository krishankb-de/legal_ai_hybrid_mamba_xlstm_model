#!/usr/bin/env python3
"""Validation PPL of a checkpoint, overall and late in context (plan P5-F's late-25% PPL; P6-G, P6-K).

    .venv/bin/python scripts/evaluate_val_ppl.py --checkpoint outputs/screen2_S1_s42/checkpoints/last.ckpt \\
        --model-config hybrid_legal_base --row-len 2048 --label screen2_S1_s42

Reads the same validation rows as training (``configs/dataset/mixture_pretrain.yaml``, split
``val``, at ``--row-len``) and scores every supervised next-token prediction the way training's
``val/loss`` does: boundary-masked targets (``boundary_masked_labels``), bf16 autocast on CUDA, CE in
fp32. ``ppl`` should therefore equal exp(``val/loss``) of the run's last validation. Two late
subsets, both at ``--late-from`` (default 0.75 x row length = 1,536 of 2,048):

* ``late_doc`` (**the late-25% PPL**, user's choice of 2026-10-13): predicted tokens with at least
  ``late_from`` tokens of their own document before them in the row. Packed rows restart position
  and recurrent state at every document, so this is the context the model can actually use.
* ``late_row``: predicted tokens at row positions ``>= late_from``, whatever their document. It is
  reported for comparison only, because a token late in the row may be early in its document.

Prints one line, ``VALPPL <label> tokens=… ppl=… late_doc_tokens=… late_doc_ppl=… late_row_tokens=…
late_row_ppl=… rows=…``, and the same numbers as JSON with ``--out``.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from lexhybrid.models.hybrid_lm import boundary_masked_labels

REPO_ROOT = Path(__file__).resolve().parent.parent


def document_offsets(doc_ids: torch.Tensor) -> torch.Tensor:
    """(B, L): how many tokens of the same document precede each position in its row."""
    B, L = doc_ids.shape
    idx = torch.arange(L, device=doc_ids.device).expand(B, L)
    starts = torch.ones_like(doc_ids, dtype=torch.bool)
    starts[:, 1:] = doc_ids[:, 1:] != doc_ids[:, :-1]
    return idx - torch.where(starts, idx, torch.zeros_like(idx)).cummax(dim=1).values


@torch.no_grad()
def late_context_sums(model, batches, late_from: int, device: str = "cpu", slab: int = 512) -> dict:
    """Summed CE and token counts: all supervised predictions, ``late_doc`` and ``late_row``."""
    sums = {k: [0.0, 0] for k in ("all", "late_doc", "late_row")}
    rows = 0
    use_amp = str(device).startswith("cuda")
    for batch in batches:
        input_ids = batch["input_ids"].to(device)
        labels = batch.get("labels", batch["input_ids"]).to(device)
        doc_ids = batch.get("doc_ids")
        doc_ids = doc_ids.to(device) if doc_ids is not None else torch.zeros_like(input_ids)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            residual, _ = model.backbone(input_ids, doc_ids=doc_ids)
        targets = boundary_masked_labels(labels, doc_ids)  # (B, L-1): position t predicts t+1
        offsets = document_offsets(doc_ids)[:, 1:]  # of the predicted token
        positions = torch.arange(1, input_ids.shape[1], device=device).expand_as(targets)
        hidden = residual[:, :-1]
        for start in range(0, hidden.shape[1], slab):
            t = targets[:, start : start + slab]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model.head(hidden[:, start : start + slab])
            ce = F.cross_entropy(
                logits.float().flatten(0, 1), t.flatten(), ignore_index=-100, reduction="none"
            ).view_as(t)
            supervised = t != -100
            for key, mask in (
                ("all", supervised),
                ("late_doc", supervised & (offsets[:, start : start + slab] >= late_from)),
                ("late_row", supervised & (positions[:, start : start + slab] >= late_from)),
            ):
                sums[key][0] += ce[mask].double().sum().item()
                sums[key][1] += int(mask.sum())
        rows += input_ids.shape[0]
    return {"rows": rows, **{k: {"ce_sum": s, "tokens": n} for k, (s, n) in sums.items()}}


def summarise(sums: dict, label: str) -> tuple[str, dict]:
    def ppl(k):
        s, n = sums[k]["ce_sum"], sums[k]["tokens"]
        return math.exp(s / n) if n else math.nan

    out = {"label": label, "rows": sums["rows"]}
    for k, name in (("all", ""), ("late_doc", "late_doc_"), ("late_row", "late_row_")):
        out[f"{name}tokens"], out[f"{name}ppl"] = sums[k]["tokens"], ppl(k)
    line = (
        f"VALPPL {label} tokens={out['tokens']} ppl={out['ppl']:.4f} late_doc_tokens={out['late_doc_tokens']} "
        f"late_doc_ppl={out['late_doc_ppl']:.4f} late_row_tokens={out['late_row_tokens']} "
        f"late_row_ppl={out['late_row_ppl']:.4f} rows={out['rows']}"
    )
    return line, out


def load_model(model_config: str, checkpoint: str, device: str):
    """The model from its yaml with the checkpoint's weights (the guarded loader), in eval mode."""
    from lexhybrid import HybridLanguageModel
    from lexhybrid.config import load_model_config
    from lexhybrid.utils.checkpoint import load_state_dict_guarded, strip_prefixes

    model = HybridLanguageModel(load_model_config(model_config))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    load_state_dict_guarded(model, strip_prefixes(state.get("state_dict", state)))
    return model.to(device).eval()


def val_dataset(row_len: int, dataset_yaml: Path):
    from omegaconf import OmegaConf

    from lexhybrid.data.datasets import mixture_from_config

    d = OmegaConf.load(dataset_yaml)
    d.row_len = row_len
    return mixture_from_config(d, "val")


def main(argv=None) -> int:
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--row-len", type=int, default=2048)
    ap.add_argument("--late-from", type=int, default=None, help="default: 0.75 x row length")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-rows", type=int, default=None, help="a smoke; omit for the full split")
    ap.add_argument("--dataset-yaml", type=Path, default=REPO_ROOT / "configs/dataset/mixture_pretrain.yaml")
    ap.add_argument("--label", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    late_from = args.late_from if args.late_from is not None else int(0.75 * args.row_len)
    model = load_model(args.model_config, args.checkpoint, args.device)
    data = val_dataset(args.row_len, args.dataset_yaml)
    if args.max_rows:
        data = torch.utils.data.Subset(data, range(min(args.max_rows, len(data))))
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    line, out = summarise(
        late_context_sums(model, loader, late_from, args.device), args.label or args.checkpoint
    )
    out.update(checkpoint=args.checkpoint, late_from=late_from, row_len=args.row_len)
    print(line, flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
