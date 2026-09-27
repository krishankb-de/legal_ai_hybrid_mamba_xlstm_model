#!/usr/bin/env python3
"""Probes on a checkpoint (plan P3-W, P3-X; P5-G and P6-M run it on the cluster): MQAR, statute
recall and two-hop cross-references.

    .venv/bin/python scripts/evaluate_probes.py --model-config hybrid_legal_base \
        --checkpoint outputs/screen_S1_s42/checkpoints/last.ckpt --context 2048 \
        --statutes data/scrubbed/gii.jsonl data/scrubbed/fedlex.jsonl --out analysis/probes_S1_s42.json

Without ``--checkpoint`` the model is a fresh random init (a smoke of the pipeline, not a result).
Prints one JSON object: MQAR accuracy, statute-recall exact match and char-F1, multi-hop exact
match, the context length, the checkpoint and the model's ARCH fingerprint.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from lexhybrid import HybridLanguageModel
from lexhybrid.config import load_model_config
from lexhybrid.data.probes import mqar, multihop, statute_recall
from lexhybrid.data.schema import Document


def run_probes(model, encode, decode, statutes, context: int, mqar_items: int, recall_items: int,
               device: str = "cpu", mqar_cfg: mqar.MQARConfig | None = None, multihop_items: int = 200) -> dict:  # fmt: skip
    cfg = mqar_cfg or mqar.MQARConfig(context_length=context)
    items = statute_recall.build_items(statutes, encode, context_tokens=context)[:recall_items]
    hops = [
        h
        for h in multihop.build_items(statutes, n_items=multihop_items)
        if len(encode(h.context + h.question)) < context
    ]
    return {
        "context": context,
        "mqar": mqar.score(model, mqar.generate(mqar_items, cfg), device=device),
        "statute_recall": statute_recall.score(model, items, encode, decode, device=device),
        "multihop": multihop.score(model, hops, encode, decode, device=device),
    }


def load_model(model_config: str, checkpoint: str | None, device: str) -> HybridLanguageModel:
    model = HybridLanguageModel(load_model_config(model_config))
    if checkpoint:
        from lexhybrid.utils.checkpoint import load_state_dict_guarded, strip_prefixes

        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        load_state_dict_guarded(model, strip_prefixes(state.get("state_dict", state)))
    return model.to(device).eval()


def main(argv=None) -> int:
    from lexhybrid.data.tokenizer import encode_document, load_tokenizer

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-config", default="hybrid_legal_base")
    parser.add_argument("--checkpoint")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--mqar-items", type=int, default=256)
    parser.add_argument("--recall-items", type=int, default=100)
    parser.add_argument("--multihop-items", type=int, default=200)
    parser.add_argument("--statutes", nargs="*", type=Path, default=[])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    model = load_model(args.model_config, args.checkpoint, args.device)
    tok = load_tokenizer(with_specials=False)
    statutes = [
        Document.from_json(line)
        for p in args.statutes
        for line in p.read_text(encoding="utf-8").splitlines()
        if line
    ]
    results = run_probes(
        model, lambda t: encode_document(tok, t), tok.decode, statutes, args.context, args.mqar_items,
        args.recall_items, args.device, multihop_items=args.multihop_items,
    )  # fmt: skip
    results.update(checkpoint=args.checkpoint, arch=model.architecture_fingerprint())
    text = json.dumps(results, ensure_ascii=False, indent=1)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
