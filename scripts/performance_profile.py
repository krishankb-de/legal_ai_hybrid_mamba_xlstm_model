#!/usr/bin/env python3
"""Performance profiling: latency, throughput, peak memory and scaling exponents.

Ported from the reference ``scripts/performance_profile.py``. Modes:

  single (default)  one (batch_size, seq_length) point.
  --sweep           efficiency curves: sweep sequence length (and optionally batch size) across
                    model configs and fit the log-log slope of latency and peak memory.
  --decode          prefill / time-to-first-token and per-token decode, cached vs full recompute.
  --per-layer       split a forward by mixer type and by ``ssd_chunked_scan``, and print the
                    Amdahl bound.

Protocol (plan R7): every compiled number is measured one shape per process with its own
``TORCHINDUCTOR_CACHE_DIR`` -- the SLURM wrapper does that -- because a shared Inductor cache once
timed two chunk sizes identically to the microsecond in the reference. The chunk sizes are read back
off the built module and recorded in every row.

Packed rows and the training loss (plan P4-P). ``--doc-len N`` gives every row ``doc_ids`` with a
new document every N tokens (each row's layout shifted, as packed rows are), so attention takes its
packed path -- the flex block mask on CUDA -- and every mixer resets at the boundaries; without it
rows are unpacked (causal SDPA). ``--loss slab`` times the training step the way
``PretrainLightningModule`` computes it: the backbone (compiled on its own under ``--compile``), then
the boundary-masked slab-wise cross-entropy through the head, so ``(B, L, V)`` logits never exist.
The default ``--loss logits`` is the reference's ``logits.float().mean()``, kept for the sanity
points; at the Qwen3 vocabulary its logits alone are 10 GB for a (4, 8192) batch in bf16.

    .venv/bin/python scripts/performance_profile.py --sweep --models hybrid_legal_base transformer_legal_base \
        --seq-lengths 2048 8192 16384 --dtype bf16 --output-dir analysis/profile
    .venv/bin/python scripts/performance_profile.py --sweep --models hybrid_legal_base --backward \
        --loss slab --doc-len 1000 --seq-lengths 4096 --dtype bf16 --output-dir analysis/profile/train
"""

import argparse
import csv
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from lexhybrid.config.loading import available_model_configs, load_model_config
from lexhybrid.models.hybrid_lm import HybridLanguageModel, boundary_masked_labels

DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
LOSSES = ("logits", "slab")
SLAB = 512  # positions per slab, as configs/distill/*.yaml


@contextmanager
def timer(name):
    """Simple timing context manager."""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{name}: {(end - start) * 1000:.2f}ms")


def available_configs():
    """Names of every model config that can be profiled (the yamls under configs/model/)."""
    return available_model_configs()


def load_config(name):
    """``HybridConfig`` for configs/model/<name>.yaml, through ``from_hydra``."""
    return load_model_config(name)


def _sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _reset_peak_memory(device):
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def _peak_memory_gb(device):
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 1e9
    return float("nan")


def packed_doc_ids(batch_size, seq_length, doc_len, device):
    """(B, L) document ids with a boundary every ``doc_len`` tokens; row b's layout is shifted by
    ``b * doc_len / B``, so the rows of a batch are packed differently (as real packed rows are)."""
    if doc_len < 1:
        raise ValueError(f"doc_len must be >= 1, got {doc_len}")
    pos = torch.arange(seq_length, device=device)
    offsets = [(b * doc_len) // batch_size for b in range(batch_size)]
    return torch.stack([(pos + off) // doc_len for off in offsets]).to(torch.long)


def effective_chunk_sizes(model):
    """``chunk_size`` read off the first built Mamba-3 and the first mLSTM mixer (None if absent)."""
    base = getattr(model, "_orig_mod", model)
    found = {"Mamba3Block": None, "mLSTMBlock": None}
    for layer in base.layers:
        kind = type(layer.mixer).__name__
        if kind in found and found[kind] is None:
            found[kind] = layer.mixer.chunk_size
    return found["Mamba3Block"], found["mLSTMBlock"]


def packed_attention(config, device, doc_len):
    """What the attention layers run on this point: causal SDPA unpacked, else the packed kernel."""
    if "attention" not in config.layer_pattern:
        return "none"
    if doc_len is None:
        return "causal-sdpa"
    impl = config.attn_impl
    if impl == "auto":
        impl = "flex" if device.startswith("cuda") else "sdpa"
    return f"packed-{impl}"


def measure_point(
    model,
    batch_size,
    seq_length,
    num_iterations,
    device,
    vocab_size,
    backward=False,
    warmup=3,
    doc_len=None,
    loss="logits",
    backbone=None,
):
    """Time one (batch_size, seq_length) point.

    ``doc_len`` packs every row (``packed_doc_ids``); ``loss="slab"`` times the training step of
    ``PretrainLightningModule`` (``backbone`` -- e.g. a compiled ``model.backbone`` -- then the
    slab-wise cross-entropy through ``model.head``) instead of ``logits.float().mean()``.

    Returns a dict of timings in seconds and peak memory in GB, or a dict with
    `oom=True` if the point does not fit. Peak memory is reset per point so the
    number is attributable to this point and not to the largest earlier one.
    """
    if loss not in LOSSES:
        raise ValueError(f"loss must be one of {LOSSES}, got {loss!r}")
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    doc_ids = None if doc_len is None else packed_doc_ids(batch_size, seq_length, doc_len, device)
    base = getattr(model, "_orig_mod", model)
    backbone = backbone or base.backbone
    targets = boundary_masked_labels(input_ids, doc_ids)

    def _run():
        if backward and loss == "slab":
            from lexhybrid.training.distill import slab_ce_kl

            model.zero_grad(set_to_none=True)
            residual, _ = backbone(input_ids, doc_ids=doc_ids)
            slab_ce_kl(residual[:, :-1], targets, None, base.head, slab=SLAB)["loss"].backward()
        elif backward:
            model.zero_grad(set_to_none=True)
            out = model(input_ids, doc_ids=doc_ids)
            # forward() returns a CausalLMOutput dataclass, not a tensor or dict.
            if hasattr(out, "logits"):
                logits = out.logits
            elif isinstance(out, dict):
                logits = out["logits"]
            else:
                logits = out
            loss_value = logits.float().mean()
            loss_value.backward()
        else:
            with torch.no_grad():
                model(input_ids, doc_ids=doc_ids)

    try:
        for _ in range(warmup):
            _run()
        _sync(device)
        _reset_peak_memory(device)

        times = []
        for _ in range(num_iterations):
            _sync(device)
            start = time.perf_counter()
            _run()
            _sync(device)
            times.append(time.perf_counter() - start)
    except torch.cuda.OutOfMemoryError:
        model.zero_grad(set_to_none=True)
        _reset_peak_memory(device)
        return {"oom": True}
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        model.zero_grad(set_to_none=True)
        _reset_peak_memory(device)
        return {"oom": True}

    times.sort()
    mean = sum(times) / len(times)
    median = times[len(times) // 2]
    var = sum((t - mean) ** 2 for t in times) / max(len(times) - 1, 1)
    tokens = batch_size * seq_length
    model.zero_grad(set_to_none=True)
    return {
        "oom": False,
        "latency_mean_s": mean,
        "latency_median_s": median,
        "latency_std_s": math.sqrt(var),
        "latency_min_s": times[0],
        "tokens_per_s": tokens / median,
        "peak_memory_gb": _peak_memory_gb(device),
    }


def fit_log_slope(xs, ys):
    """Least-squares slope of log(y) vs log(x) — the empirical scaling exponent.

    Returns None if fewer than two finite positive points are available.
    """
    pts = [
        (math.log(x), math.log(y))
        for x, y in zip(xs, ys)
        if x > 0 and y is not None and y > 0 and math.isfinite(y)
    ]
    if len(pts) < 2:
        return None
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    denom = sum((p[0] - mx) ** 2 for p in pts)
    if denom == 0:
        return None
    return sum((p[0] - mx) * (p[1] - my) for p in pts) / denom


def build_model(config, device, dtype):
    model = HybridLanguageModel(config)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


def profile_model(
    config,
    batch_size=4,
    seq_length=2048,
    num_iterations=10,
    device="cuda",
    dtype=torch.float32,
    backward=False,
    attn_backend="auto",
    compile_model=False,
    doc_len=None,
    loss="logits",
):
    """Profile a single (batch_size, seq_length) point and print a report."""
    print("=" * 80)
    print("Model Profiling")
    print("=" * 80)

    model = build_model(config, device, dtype)
    num_params = model.get_num_params(non_embedding=True)
    model, backbone, compile_s = compile_for(model, compile_model, backward and loss == "slab")
    print(f"Model: {num_params / 1e6:.1f}M parameters (non-embedding)")
    print(
        "Attention backend: {}   torch.compile: {}".format(
            attn_backend, f"{compile_s:.1f}s to wrap" if compile_model else "off"
        )
    )
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_length}")
    print(f"Device: {device}  dtype: {dtype}")
    print("Pass: {}".format(f"forward+backward (loss {loss})" if backward else "forward"))
    print(f"Rows: {'unpacked' if doc_len is None else f'packed, a document every {doc_len} tokens'}")
    print()

    print("Warming up and profiling...")
    with attention_backend(attn_backend):
        res = measure_point(
            model,
            batch_size,
            seq_length,
            num_iterations,
            device,
            config.vocab_size,
            backward=backward,
            doc_len=doc_len,
            loss=loss,
            backbone=backbone,
        )
    if res["oom"]:
        print("OUT OF MEMORY at this point.")
        return res

    print("\n" + "=" * 80)
    print("Results:")
    print(
        "Median forward time: {:.2f}ms  (mean {:.2f} +/- {:.2f}ms)".format(
            res["latency_median_s"] * 1000,
            res["latency_mean_s"] * 1000,
            res["latency_std_s"] * 1000,
        )
    )
    print(
        "Throughput: {:.0f} tokens/second ({:.2f}k)".format(res["tokens_per_s"], res["tokens_per_s"] / 1000)
    )
    if device.startswith("cuda"):
        print("Peak memory allocated: {:.2f} GB".format(res["peak_memory_gb"]))
    print("=" * 80)
    return res


def run_sweep(
    model_names,
    seq_lengths,
    batch_sizes,
    num_iterations,
    device,
    dtype,
    backward,
    output_dir,
    attn_backend="auto",
    compile_model=False,
    chunk_size=None,
    mlstm_chunk_size=None,
    doc_len=None,
    loss="logits",
):
    """Sweep sequence length (x batch size) across models and fit exponents.

    `attn_backend` (E0-D) and `compile_model` (E1-B) are recorded on every row so
    two sweeps written to different directories can be compared arm by arm; the
    published 14A-7 numbers are the `auto` / uncompiled arm. So are the packing
    (`doc_len`, the attention path it takes) and the training `loss` (P4-P).
    """
    rows: list[dict[str, Any]] = []

    for name in model_names:
        config = load_config(name)
        if chunk_size is not None and hasattr(config, "mamba3_chunk_size"):
            config.mamba3_chunk_size = chunk_size
        if mlstm_chunk_size is not None and hasattr(config, "mlstm_chunk_size"):
            config.mlstm_chunk_size = mlstm_chunk_size
        model = build_model(config, device, dtype)
        num_params = model.get_num_params(non_embedding=True)
        # Read the chunk size back OFF THE BUILT MODULE, not off the config. In the
        # E1 run (job 2580198) the compiled cs=64 and cs=128 arms timed identically
        # to 1 microsecond at L=4096 and L=8192 while the uncompiled arms differed by
        # 45%, which is not a plausible coincidence. Either the override stopped
        # reaching the operator under compile, or those points genuinely plateau.
        # Printing the effective value is what tells those two apart.
        effective_chunk, effective_mlstm_chunk = effective_chunk_sizes(model)
        model, backbone, compile_s = compile_for(model, compile_model, backward and loss == "slab")
        if compile_model:
            print(
                f"torch.compile: wrapped in {compile_s:.1f}s (graph build happens on the "
                "first forward of each new shape)"
            )
        if effective_chunk is not None or effective_mlstm_chunk is not None:
            print(
                f"effective chunk_size on the built module: mamba3 {effective_chunk}, "
                f"mlstm {effective_mlstm_chunk}"
            )
        attn_path = packed_attention(config, device, doc_len)
        pattern = ",".join(config.layer_pattern)
        print("\n" + "=" * 80)
        print(
            f"{name}  |  {num_params / 1e6:.1f}M non-emb params  |  dim={config.dim} layers={config.num_layers}  |  [{pattern}]"
        )
        print("=" * 80)

        for batch_size in batch_sizes:
            for seq_length in seq_lengths:
                with attention_backend(attn_backend):
                    res = measure_point(
                        model,
                        batch_size,
                        seq_length,
                        num_iterations,
                        device,
                        config.vocab_size,
                        backward=backward,
                        doc_len=doc_len,
                        loss=loss,
                        backbone=backbone,
                    )
                row = {
                    "model": name,
                    "params_non_emb_m": round(num_params / 1e6, 2),
                    "dim": config.dim,
                    "num_layers": config.num_layers,
                    "layer_pattern": pattern,
                    "device": device,
                    "dtype": str(dtype).replace("torch.", ""),
                    "pass": "forward+backward" if backward else "forward",
                    "loss": loss if backward else "",
                    "batch_size": batch_size,
                    "seq_length": seq_length,
                    "doc_len": doc_len,
                    "attn_path": attn_path,
                    "attn_backend": attn_backend,
                    "compiled": bool(compile_model),
                    "chunk_size": getattr(config, "mamba3_chunk_size", None),
                    "effective_chunk_size": effective_chunk,
                    "mlstm_chunk_size": getattr(config, "mlstm_chunk_size", None),
                    "effective_mlstm_chunk_size": effective_mlstm_chunk,
                    "oom": res["oom"],
                }
                if res["oom"]:
                    print(f"  bs={batch_size:<4} L={seq_length:<6} OOM")
                else:
                    row.update(
                        {
                            "latency_median_ms": round(res["latency_median_s"] * 1000, 3),
                            "latency_mean_ms": round(res["latency_mean_s"] * 1000, 3),
                            "latency_std_ms": round(res["latency_std_s"] * 1000, 3),
                            "tokens_per_s": round(res["tokens_per_s"], 1),
                            "peak_memory_gb": round(res["peak_memory_gb"], 4),
                        }
                    )
                    print(
                        "  bs={:<4} L={:<6} {:9.2f}ms  {:10.0f} tok/s  {:7.3f} GB".format(
                            batch_size,
                            seq_length,
                            res["latency_median_s"] * 1000,
                            res["tokens_per_s"],
                            res["peak_memory_gb"],
                        )
                    )
                rows.append(row)

        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # --- scaling exponents, per (model, batch_size) -------------------------
    exponents = []
    for name in model_names:
        for batch_size in batch_sizes:
            sel = [r for r in rows if r["model"] == name and r["batch_size"] == batch_size and not r["oom"]]
            if len(sel) < 2:
                continue
            xs = [r["seq_length"] for r in sel]
            lat = fit_log_slope(xs, [r["latency_median_ms"] for r in sel])
            mem = fit_log_slope(xs, [r.get("peak_memory_gb") for r in sel])
            exponents.append(
                {
                    "model": name,
                    "batch_size": batch_size,
                    "seq_lengths": xs,
                    "latency_exponent": None if lat is None else round(lat, 3),
                    "memory_exponent": None if mem is None else round(mem, 3),
                }
            )

    print("\n" + "=" * 80)
    print("SCALING EXPONENTS  (slope of log(y) vs log(seq_length))")
    print("  ~1.0 = linear in sequence length; ~2.0 = quadratic (softmax attention)")
    print("=" * 80)
    print("{:<26} {:>4}  {:>10}  {:>10}".format("model", "bs", "latency", "memory"))
    for e in exponents:
        print(
            "{:<26} {:>4}  {:>10}  {:>10}".format(
                e["model"],
                e["batch_size"],
                "n/a" if e["latency_exponent"] is None else "{:.3f}".format(e["latency_exponent"]),
                "n/a" if e["memory_exponent"] is None else "{:.3f}".format(e["memory_exponent"]),
            )
        )
    print("=" * 80)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "model",
            "params_non_emb_m",
            "dim",
            "num_layers",
            "layer_pattern",
            "device",
            "dtype",
            "pass",
            "loss",
            "batch_size",
            "seq_length",
            "doc_len",
            "attn_path",
            "attn_backend",
            "compiled",
            "chunk_size",
            "effective_chunk_size",
            "mlstm_chunk_size",
            "effective_mlstm_chunk_size",
            "oom",
            "latency_median_ms",
            "latency_mean_ms",
            "latency_std_ms",
            "tokens_per_s",
            "peak_memory_gb",
        ]
        csv_path = out / "efficiency_curves.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        json_path = out / "efficiency_curves.json"
        with open(json_path, "w") as f:
            json.dump({"points": rows, "scaling_exponents": exponents}, f, indent=2)
        print(f"\nWrote {csv_path}\n      {json_path}")

    return rows, exponents


def profile_decode(
    config, prompt_len=256, new_tokens=64, batch_size=1, device="cpu", dtype=torch.float32, beam_size=1
):
    """Prefill, time-to-first-token and per-token decode, cached vs full recompute (reference M6-E).

    Reports, for each path:
        prefill / TTFT  -- seconds to the first sampled token
        decode          -- seconds per token thereafter, and its growth from the first half of
                           the run to the second (an O(1) path is flat; a recomputing one is not)
    """
    import time

    model = build_model(config, device, dtype).eval()
    vocab = config.vocab_size
    ids = torch.randint(0, vocab, (batch_size, prompt_len), device=device)

    def _time(fn):
        _sync(device)
        t0 = time.perf_counter()
        out = fn()
        _sync(device)
        return time.perf_counter() - t0, out

    print("\n" + "=" * 70)
    print(
        f"DECODE PROFILE  prompt={prompt_len}  new_tokens={new_tokens}  batch={batch_size}  beam={beam_size}"
    )
    print("=" * 70)

    rows = {}
    with torch.no_grad():
        # ---- full recompute: what generate() does today -------------------------------
        hidden = model.embeddings(ids)
        ttft, _ = _time(lambda: model(inputs_embeds=hidden).logits[:, -1])
        per_token = []
        seq = hidden
        for _ in range(new_tokens):
            dt, logits = _time(lambda seq=seq: model(inputs_embeds=seq).logits[:, -1])
            per_token.append(dt)
            seq = torch.cat([seq, model.embeddings(logits.argmax(-1, keepdim=True))], dim=1)
        rows["full recompute"] = (ttft, per_token)

        # ---- cached ------------------------------------------------------------------
        if model.supports_cached_decode():
            caches = model.allocate_inference_cache(batch_size, device=device, dtype=dtype)
            ttft_c, logits = _time(lambda: model.prefill(model.embeddings(ids), caches))
            per_token_c = []
            for _ in range(new_tokens):
                nxt = logits.argmax(-1, keepdim=True)
                dt, logits = _time(lambda nxt=nxt: model.step_logits(model.embeddings(nxt)[:, 0], caches))
                per_token_c.append(dt)
            rows["cached (O(1))"] = (ttft_c, per_token_c)

    print("{:<18} {:>10} {:>12} {:>12} {:>10}".format("path", "TTFT s", "s/token", "2nd/1st half", "tok/s"))
    for name, (ttft, per_token) in rows.items():
        half = len(per_token) // 2
        first = sum(per_token[:half]) / max(half, 1)
        second = sum(per_token[half:]) / max(len(per_token) - half, 1)
        mean = sum(per_token) / len(per_token)
        print(
            "{:<18} {:>10.4f} {:>12.5f} {:>12.2f}x {:>10.1f}".format(
                name, ttft, mean, second / first if first else float("nan"), 1.0 / mean
            )
        )

    if len(rows) == 2:
        (f_ttft, f_tok), (c_ttft, c_tok) = rows["full recompute"], rows["cached (O(1))"]

        def _mean(xs):
            return sum(xs) / len(xs)

        half = len(c_tok) // 2
        growth = _mean(c_tok[half:]) / _mean(c_tok[:half]) if half else float("nan")
        print(f"\n  per-token speedup      {_mean(f_tok) / _mean(c_tok):.2f}x")
        print(
            f"  TTFT ratio             {c_ttft / f_ttft:.2f}x  (>1 means the cached prefill is SLOWER: it "
            "steps"
        )
        print("                              token by token -- see prefill()'s docstring)")
        print(f"  cached growth 2nd/1st  {growth:.2f}x  (1.00 = O(1) in context, which is the claim)")
    return rows


# ---------------------------------------------------------------------------
# Where does the time go? (reference efficiency plan E0.) Split by mixer type
# (the Amdahl bound), split the Mamba-3 block into `ssd_chunked_scan` and the
# rest, sweep the chunk size, and re-time attention with the fused SDPA
# backends disabled -- which separates the algorithm from kernel engineering.
# ---------------------------------------------------------------------------

SDPA_BACKENDS = ("auto", "math", "flash", "efficient")


@contextmanager
def attention_backend(name):
    """Restrict `F.scaled_dot_product_attention` to one backend (E0-D).

    `auto` is PyTorch's own dispatch, i.e. the fused FlashAttention kernel on an
    H100 -- that is the arm every published efficiency number was measured in.
    `math` forces the unfused reference path, which materialises the (L, L)
    attention matrix. Comparing the two isolates how much of the Transformer's
    speed is its algorithm and how much is a hand-written kernel.

    Expect `math` to OOM at the top of the sequence ladder. That is not a bug in
    the harness: b=4, 12 heads, L=16384 needs ~26 GB in bf16 for one attention
    matrix. `measure_point` records it as `oom=True` and the sweep continues,
    and the OOM is itself the FlashAttention memory story stated as data.
    """
    if name == "auto":
        yield
        return
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:  # torch < 2.3
        flags = {
            "math": dict(enable_math=True, enable_flash=False, enable_mem_efficient=False),
            "flash": dict(enable_math=False, enable_flash=True, enable_mem_efficient=False),
            "efficient": dict(enable_math=False, enable_flash=False, enable_mem_efficient=True),
        }[name]
        with torch.backends.cuda.sdp_kernel(**flags):
            yield
        return
    backend = {
        "math": SDPBackend.MATH,
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
    }[name]
    with sdpa_kernel([backend]):
        yield


def maybe_compile(model, enabled):
    """`torch.compile` the model for inference only (E1-B).

    The reference pinned compile off because its Mamba-1 and mLSTM blocks loop in
    Python over (row, segment) for document boundaries. SSD handles boundaries
    with masks, so that reason does not transfer; this flag is how it is tested.
    Compile time is reported because a long compile for a small steady-state gain
    is a null.
    """
    if not enabled:
        return model, 0.0
    start = time.perf_counter()
    compiled = torch.compile(model)
    return compiled, time.perf_counter() - start


def compile_for(model, enabled, slab_training):
    """``(model, backbone, seconds)``: the slab-loss training step compiles the backbone alone, as
    ``PretrainLightningModule`` does; every other pass compiles the whole model (``backbone`` None)."""
    if slab_training:
        backbone, seconds = maybe_compile(model.backbone, enabled)
        return model, backbone, seconds
    model, seconds = maybe_compile(model, enabled)
    return model, None, seconds


def _mark(cuda):
    """Start/stop marker: a CUDA event on GPU, a wall clock on CPU."""
    if cuda:
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        return ev
    return time.perf_counter()


def _delta_ms(cuda, a, b):
    return a.elapsed_time(b) if cuda else (b - a) * 1000.0


class LayerSplit:
    """Per-`HybridBlock` timing, aggregated by mixer type (E0-A).

    Markers are queued on the same stream as the work, so on CUDA this costs two
    event records per layer per iteration and does not serialise the forward.
    Read the totals only after a synchronise.
    """

    def __init__(self, model, device):
        self.cuda = device.startswith("cuda")
        self.layer_types = [layer.layer_type for layer in model.layers]
        self.enabled = False
        self._open = {}
        self._records = []
        self._handles = []
        for idx, layer in enumerate(model.layers):
            self._handles.append(layer.register_forward_pre_hook(self._pre(idx)))
            self._handles.append(layer.register_forward_hook(self._post(idx)))

    def _pre(self, idx):
        def hook(module, args):
            if self.enabled:
                self._open[idx] = _mark(self.cuda)

        return hook

    def _post(self, idx):
        def hook(module, args, output):
            if self.enabled and idx in self._open:
                self._records.append((idx, self._open.pop(idx), _mark(self.cuda)))

        return hook

    def reset(self):
        self._open.clear()
        self._records.clear()

    def totals_ms(self, iterations):
        """{mixer type: mean ms per forward}, plus a per-layer-index breakdown."""
        by_type, by_index = {}, {}
        for idx, start, end in self._records:
            ms = _delta_ms(self.cuda, start, end) / max(iterations, 1)
            lt = self.layer_types[idx]
            by_type[lt] = by_type.get(lt, 0.0) + ms
            by_index[idx] = by_index.get(idx, 0.0) + ms
        return by_type, by_index

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


class ScanSplit:
    """Time every `ssd_chunked_scan` call inside the Mamba-3 blocks (E0-B).

    `mamba3_block.py` imports the symbol directly, so the patch target is that
    module's attribute, not the kernel package's. Restores on exit even if the
    body raises -- a profiler that leaves a monkeypatch behind would silently
    corrupt every later measurement in the same process.
    """

    def __init__(self, device):
        self.cuda = device.startswith("cuda")
        self.enabled = False
        self._pairs = []
        self._module = None
        self._orig = None

    def __enter__(self):
        from lexhybrid.layers import mamba3_block as _m3

        self._module = _m3
        self._orig = _m3.ssd_chunked_scan

        def timed(*args, **kwargs):
            if not self.enabled:
                return self._orig(*args, **kwargs)
            start = _mark(self.cuda)
            out = self._orig(*args, **kwargs)
            self._pairs.append((start, _mark(self.cuda)))
            return out

        _m3.ssd_chunked_scan = timed
        return self

    def __exit__(self, *exc):
        if self._module is not None and self._orig is not None:
            self._module.ssd_chunked_scan = self._orig
        return False

    def reset(self):
        self._pairs.clear()

    def total_ms(self, iterations):
        if not self._pairs:
            return None
        return sum(_delta_ms(self.cuda, a, b) for a, b in self._pairs) / max(iterations, 1)


def run_layer_split(
    model_names,
    seq_lengths,
    batch_size,
    num_iterations,
    device,
    dtype,
    attn_backend="auto",
    chunk_size=None,
    output_dir=None,
    warmup=3,
):
    """E0-A/E0-B: split a forward pass by mixer type and by the SSD scan.

    Prints, per model and sequence length, the milliseconds and share of the
    forward spent in each mixer type, the share inside `ssd_chunked_scan`, and
    the Amdahl bound (E0-F) that follows: the best speedup available from making
    the Mamba-3 path free is 1 / (1 - its share).
    """
    rows = []
    cuda = device.startswith("cuda")

    for name in model_names:
        config = load_config(name)
        if chunk_size is not None and hasattr(config, "mamba3_chunk_size"):
            config.mamba3_chunk_size = chunk_size
        model = build_model(config, device, dtype)
        pattern = ",".join(config.layer_pattern)
        counts = {}
        for lt in [layer.layer_type for layer in model.layers]:
            counts[lt] = counts.get(lt, 0) + 1

        print("\n" + "=" * 80)
        print(f"{name}  |  dim={config.dim} layers={config.num_layers}  |  [{pattern}]")
        print("  layer counts: {}".format(", ".join(f"{v}x{k}" for k, v in sorted(counts.items()))))
        if chunk_size is not None:
            print("  mamba3_chunk_size = {}".format(getattr(config, "mamba3_chunk_size", "n/a")))
        print(f"  attention backend = {attn_backend}")
        print("=" * 80)

        split = LayerSplit(model, device)
        try:
            for seq_length in seq_lengths:
                input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)
                row = {
                    "model": name,
                    "seq_length": seq_length,
                    "batch_size": batch_size,
                    "dtype": str(dtype).replace("torch.", ""),
                    "attn_backend": attn_backend,
                    "chunk_size": getattr(config, "mamba3_chunk_size", None),
                    "layer_counts": counts,
                    "oom": False,
                }
                try:
                    with attention_backend(attn_backend), ScanSplit(device) as scan:
                        with torch.no_grad():
                            for _ in range(warmup):
                                model(input_ids)
                        _sync(device)
                        split.reset()
                        scan.reset()
                        # Rule R2 is checked against this: a chunk size or a compile arm that
                        # buys speed by exceeding the Transformer's peak memory is rejected.
                        _reset_peak_memory(device)
                        split.enabled = scan.enabled = True
                        wall_start = time.perf_counter()
                        with torch.no_grad():
                            for _ in range(num_iterations):
                                model(input_ids)
                        _sync(device)
                        wall_ms = (time.perf_counter() - wall_start) * 1000.0 / num_iterations
                        split.enabled = scan.enabled = False
                        by_type, by_index = split.totals_ms(num_iterations)
                        scan_ms = scan.total_ms(num_iterations)
                except torch.cuda.OutOfMemoryError:
                    row["oom"] = True
                    rows.append(row)
                    print(f"  L={seq_length:<6} OOM")
                    _reset_peak_memory(device)
                    continue
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    row["oom"] = True
                    rows.append(row)
                    print(f"  L={seq_length:<6} OOM")
                    _reset_peak_memory(device)
                    continue

                row.update(
                    {
                        "forward_ms": round(wall_ms, 3),
                        "peak_memory_gb": round(_peak_memory_gb(device), 4),
                        "by_type_ms": {k: round(v, 3) for k, v in by_type.items()},
                        "by_layer_ms": {str(k): round(v, 3) for k, v in sorted(by_index.items())},
                        "ssd_scan_ms": None if scan_ms is None else round(scan_ms, 3),
                    }
                )
                mixer_total = sum(by_type.values())
                row["outside_mixers_ms"] = round(max(wall_ms - mixer_total, 0.0), 3)

                print(
                    "  L={:<6} forward {:8.2f} ms   peak {:7.3f} GB".format(
                        seq_length, wall_ms, row["peak_memory_gb"]
                    )
                )
                for lt, ms in sorted(by_type.items(), key=lambda kv: -kv[1]):
                    print(
                        f"      {lt:<10} {ms:8.2f} ms  {100.0 * ms / wall_ms:5.1f}%  ({counts.get(lt, 0)} layers)"
                    )
                print(
                    "      {:<10} {:8.2f} ms  {:5.1f}%  (embed/head/norm)".format(
                        "other", row["outside_mixers_ms"], 100.0 * row["outside_mixers_ms"] / wall_ms
                    )
                )
                if scan_ms is not None:
                    print(
                        f"      -> of which ssd_chunked_scan: {scan_ms:.2f} ms  {100.0 * scan_ms / wall_ms:.1f}% of forward"
                    )
                    row["ssd_scan_share"] = round(scan_ms / wall_ms, 4)

                # E0-F, stated as data rather than left to the reader.
                m3_ms = by_type.get("mamba3", 0.0) + by_type.get("mamba", 0.0)
                if m3_ms > 0:
                    share = m3_ms / wall_ms
                    bound = float("inf") if share >= 1.0 else 1.0 / (1.0 - share)
                    row["amdahl_bound_if_ssd_free"] = round(bound, 3)
                    print(
                        f"      Amdahl: SSD path is {100.0 * share:.1f}% of the forward, so making it FREE "
                        f"caps the speedup at {bound:.2f}x"
                    )
                rows.append(row)
        finally:
            split.remove()
            del model
            if cuda:
                torch.cuda.empty_cache()

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / "layer_split.json"
        with open(json_path, "w") as f:
            json.dump({"points": rows}, f, indent=2)
        csv_path = out / "layer_split.csv"
        types = sorted({t for r in rows for t in r.get("by_type_ms", {})})
        fieldnames = (
            [
                "model",
                "seq_length",
                "batch_size",
                "dtype",
                "attn_backend",
                "chunk_size",
                "oom",
                "forward_ms",
                "peak_memory_gb",
            ]
            + ["ms_" + t for t in types]
            + ["outside_mixers_ms", "ssd_scan_ms", "ssd_scan_share", "amdahl_bound_if_ssd_free"]
        )
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                flat = {k: v for k, v in r.items() if k in fieldnames}
                for t in types:
                    flat["ms_" + t] = r.get("by_type_ms", {}).get(t, "")
                writer.writerow({k: flat.get(k, "") for k in fieldnames})
        print(f"\nWrote {csv_path}\n      {json_path}")

    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Profile hybrid model")
    choices = available_configs()
    parser.add_argument(
        "--model",
        type=str,
        default="ref_hybrid_m3",
        choices=choices,
        help="Model config to profile (single-point mode)",
    )
    parser.add_argument(
        "--sweep", action="store_true", help="Sweep sequence lengths and fit scaling exponents"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        choices=choices,
        help="Model configs to compare in --sweep mode (default: the --model value)",
    )
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048, 4096],
        help="Sequence lengths to sweep",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Batch sizes to sweep (default: the --batch_size value)",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=2048, help="Sequence length (single-point mode)")
    parser.add_argument("--num_iterations", type=int, default=10, help="Timed iterations per point")
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=sorted(DTYPES),
        help="Compute dtype (use bf16 on H100/A100)",
    )
    parser.add_argument(
        "--backward",
        action="store_true",
        help="Time forward+backward (training step) instead of forward-only inference",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Write efficiency_curves.csv/.json here (--sweep)"
    )
    parser.add_argument(
        "--decode",
        action="store_true",
        help="Profile autoregressive decode: prefill/TTFT and per-token latency, cached vs full recompute",
    )
    parser.add_argument("--prompt-len", type=int, default=256, help="Prompt length for --decode")
    parser.add_argument("--new-tokens", type=int, default=64, help="Tokens to generate for --decode")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on"
    )
    # --- per-layer split, attention backend, chunk size, compile ----------
    parser.add_argument(
        "--per-layer",
        action="store_true",
        help="E0-A/E0-B: split the forward by mixer type and by ssd_chunked_scan, and print the Amdahl bound",
    )
    parser.add_argument(
        "--attn-backend",
        type=str,
        default="auto",
        choices=list(SDPA_BACKENDS),
        help="E0-D: restrict scaled_dot_product_attention to one "
        "backend. `auto` is the fused kernel every published "
        "number used; `math` is the unfused reference and is "
        "what separates algorithm from kernel engineering",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="E0-C: override mamba3_chunk_size. The chunked "
        "decomposition is exact for any chunk size, so this is "
        "a pure performance knob -- but it changes float "
        "association, so equivalence (rule R1) is not optional",
    )
    parser.add_argument(
        "--compile",
        dest="compile_model",
        action="store_true",
        help="torch.compile the model (measure one shape per process with its own Inductor cache, plan R7)",
    )
    parser.add_argument(
        "--mlstm-chunk-size", type=int, default=None, help="override mlstm_chunk_size (--sweep)"
    )
    parser.add_argument(
        "--doc-len",
        type=int,
        default=None,
        help="pack every row with doc_ids, a document every N tokens (flex attention on CUDA); "
        "default: unpacked rows",
    )
    parser.add_argument(
        "--loss",
        choices=LOSSES,
        default="logits",
        help="with --backward: 'slab' is the training loss (backbone + slab-wise CE, no (B, L, V) "
        "logits); 'logits' is the reference's logits.float().mean()",
    )

    args = parser.parse_args(argv)
    dtype = DTYPES[args.dtype]

    if args.per_layer:
        run_layer_split(
            model_names=args.models if args.models else [args.model],
            seq_lengths=sorted(args.seq_lengths),
            batch_size=args.batch_size,
            num_iterations=args.num_iterations,
            device=args.device,
            dtype=dtype,
            attn_backend=args.attn_backend,
            chunk_size=args.chunk_size,
            output_dir=args.output_dir,
        )
    elif args.decode:
        profile_decode(
            config=load_config(args.model),
            prompt_len=args.prompt_len,
            new_tokens=args.new_tokens,
            batch_size=args.batch_size,
            device=args.device,
            dtype=dtype,
        )
    elif args.sweep:
        model_names = args.models if args.models else [args.model]
        batch_sizes = args.batch_sizes if args.batch_sizes else [args.batch_size]
        run_sweep(
            model_names=model_names,
            seq_lengths=sorted(args.seq_lengths),
            batch_sizes=batch_sizes,
            num_iterations=args.num_iterations,
            device=args.device,
            dtype=dtype,
            backward=args.backward,
            output_dir=args.output_dir,
            attn_backend=args.attn_backend,
            compile_model=args.compile_model,
            chunk_size=args.chunk_size,
            mlstm_chunk_size=args.mlstm_chunk_size,
            doc_len=args.doc_len,
            loss=args.loss,
        )
    else:
        profile_model(
            config=load_config(args.model),
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            num_iterations=args.num_iterations,
            device=args.device,
            dtype=dtype,
            backward=args.backward,
            attn_backend=args.attn_backend,
            compile_model=args.compile_model,
            doc_len=args.doc_len,
            loss=args.loss,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
