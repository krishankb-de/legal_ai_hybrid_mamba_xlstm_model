#!/usr/bin/env python3
"""Language-model evaluation: perplexity, bits per token, throughput and peak memory.

Ported from the reference ``scripts/evaluate_lm.py``. Changes: the model is built from its yaml
through ``HybridConfig.from_hydra`` (the reference hard-coded a kwarg list -- the FM5 bug class);
the checkpoint loads through the guarded loader (>50% missing keys is a hard failure; the
reference only warned, defect 13) and the architecture read off the checkpoint must equal the
yaml's; data is a packed parquet shard (``input_ids``, optional ``doc_ids``) or ``synthetic``;
the metric is called bits per token, which is what ``loss / ln 2`` is.

    .venv/bin/python scripts/evaluate_lm.py --checkpoint last.ckpt --model-config hybrid_legal_base \\
        --data data/shards/val_4096_000.parquet --max-length 4096 --throughput --output-dir eval/
"""

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from lexhybrid.config.loading import load_model_config
from lexhybrid.data.synthetic import SyntheticPackedDataset
from lexhybrid.models.hybrid_lm import HybridLanguageModel
from lexhybrid.utils.checkpoint import infer_layer_types, load_state_dict_guarded, strip_prefixes


class ParquetRows(Dataset):
    """Rows of a packed parquet shard: ``input_ids`` (and ``doc_ids`` when present), cut to ``max_length``."""

    def __init__(self, path: str, max_length: int):
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        self.input_ids = table.column("input_ids").to_pylist()
        self.doc_ids = table.column("doc_ids").to_pylist() if "doc_ids" in table.column_names else None
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        row = {"input_ids": torch.tensor(self.input_ids[i][: self.max_length], dtype=torch.long)}
        if self.doc_ids is not None:
            row["doc_ids"] = torch.tensor(self.doc_ids[i][: self.max_length], dtype=torch.long)
        return row


def load_model(checkpoint: str, model_config: str, device: str) -> tuple[HybridLanguageModel, int]:
    """Build the model from its yaml and load the checkpoint through the guarded loader."""
    config = load_model_config(model_config)
    model = HybridLanguageModel(config)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = strip_prefixes(ckpt.get("state_dict", ckpt))
    inferred = infer_layer_types(state)
    built = model.get_layer_types()
    if inferred != built:
        raise RuntimeError(f"checkpoint layer pattern {inferred} does not match {model_config} ({built})")
    missing, unexpected = load_state_dict_guarded(model, state)
    print(f"  loaded {checkpoint}: {len(missing)} missing, {len(unexpected)} unexpected keys")
    print("  " + model.architecture_fingerprint())
    return model.to(device).eval(), sum(p.numel() for p in model.parameters())


@torch.no_grad()
def evaluate_perplexity(model, dataloader, device, max_batches=None) -> dict:
    """Token-weighted mean next-token loss; each row contributes ``L - 1`` predictions."""
    total_loss, total_tokens = 0.0, 0
    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        doc_ids = batch.get("doc_ids")
        doc_ids = doc_ids.to(device) if doc_ids is not None else None
        loss = model(input_ids, labels=input_ids, doc_ids=doc_ids, return_dict=True).loss
        n = input_ids.shape[0] * (input_ids.shape[1] - 1)
        total_loss += loss.item() * n
        total_tokens += n
    avg = total_loss / max(total_tokens, 1)
    return {
        "perplexity": math.exp(avg),
        "loss": avg,
        "bits_per_token": avg / math.log(2),
        "num_tokens": total_tokens,
    }


@torch.no_grad()
def measure_throughput(model, device, vocab_size, seq_lengths, batch_size=4, warmup=3, trials=10) -> dict:
    """Forward tokens/s on random ids at each sequence length."""

    def sync():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

    results = {}
    for seq_len in seq_lengths:
        ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        for _ in range(warmup):
            model(ids, return_dict=True)
        sync()
        start = time.perf_counter()
        for _ in range(trials):
            model(ids, return_dict=True)
        sync()
        elapsed = time.perf_counter() - start
        tokens = batch_size * seq_len * trials
        results[seq_len] = {
            "tokens_per_second": tokens / elapsed,
            "ms_per_token": elapsed / tokens * 1000,
            "ms_per_batch": elapsed / trials * 1000,
        }
        print(
            f"  seq_len={seq_len:>6}: {tokens / elapsed:>12,.0f} tok/s  ({elapsed / trials * 1000:.1f} ms/batch)"
        )
    return results


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-config", required=True, help="configs/model/<name>.yaml or a yaml path")
    ap.add_argument("--data", default="synthetic", help="'synthetic' or a packed parquet shard")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--synthetic-rows", type=int, default=16)
    ap.add_argument("--throughput", action="store_true")
    ap.add_argument("--seq-lengths", type=int, nargs="+", default=[128, 256, 512, 1024])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args(argv)

    model, num_params = load_model(args.checkpoint, args.model_config, args.device)
    if args.data == "synthetic":
        dataset = SyntheticPackedDataset(
            num_rows=args.synthetic_rows, row_len=args.max_length, vocab_size=model.config.vocab_size
        )
    else:
        dataset = ParquetRows(args.data, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    ppl = evaluate_perplexity(model, loader, args.device, args.max_batches)
    print(
        f"  perplexity {ppl['perplexity']:.3f}  loss {ppl['loss']:.4f}  bits/token {ppl['bits_per_token']:.4f}"
    )

    throughput = {}
    if args.throughput:
        throughput = measure_throughput(
            model, args.device, model.config.vocab_size, args.seq_lengths, args.batch_size
        )
    peak = torch.cuda.max_memory_allocated() / 1e9 if str(args.device).startswith("cuda") else 0.0

    results = {
        "model_config": args.model_config,
        "checkpoint": str(args.checkpoint),
        "data": args.data,
        "total_params": num_params,
        "perplexity": ppl["perplexity"],
        "loss": ppl["loss"],
        "bits_per_token": ppl["bits_per_token"],
        "num_tokens_evaluated": ppl["num_tokens"],
        "peak_gpu_memory_gb": peak,
        "throughput": throughput,
        "timestamp": datetime.now().isoformat(),
    }
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
        print(f"  wrote {out / 'results.json'}")
    return results


if __name__ == "__main__":
    main()
