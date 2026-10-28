"""Loss-spike diagnostics for one training run (decisions.p5_divergence, 2026-10-11).

Two screen runs (S1-s42, S3-s43; jobs 2626202, 2629409) collapsed to unigram-level loss after one
gradient spike at constant LR. ``SpikeDiagnosticsCallback`` logs, to the run's TensorBoard logger,
what is needed to see which part of the model moved first. ``<part>`` is ``embed``,
``L<i>_<type>`` (one per block), ``final_norm`` or ``mtp``.

Every optimizer step (rank 0):

* ``diag/grad/<part>``: gradient norm before clipping (callbacks run before the module's clip);
* ``diag/update/<part>``: ``||w_after - w_before|| / ||w_before||`` of the step just taken.

Every ``every_n_steps``, from forward hooks on the first micro-batch of that step (with activation
checkpointing a hook also fires on the recompute; every statistic is a maximum, so it is unchanged):

* ``diag/act/<part>_rms`` / ``_max``: the block's output (the residual stream);
* ``diag/mlp/<part>_max``: the MLP output (``norm_topology: hybrid`` feeds it the un-normalised stream);
* ``diag/mlstm/L<i>_ilogit_max`` / ``_flogit_min``: gate logits before the tanh soft cap;
* ``diag/mlstm/L<i>_half_range``: the largest per-chunk half-range of the cumulative log forget
  gate, the quantity the exact TFLA path tests against 40, here without the restarts at document
  boundaries (an upper bound on packed rows); ``_row_block_min`` the smallest row block
  ``_row_block`` chose that step (0 when the one-factor path held, -1 when no block fit);
* ``diag/attn/L<i>_logit_bound``: ``max ||q|| * max ||k|| / sqrt(head_dim)`` after QK-norm (before
  RoPE, which preserves norms): an upper bound on the attention logits;
* ``diag/mamba/L<i>_dt_max``: the largest step size after softplus and the clamp;
* ``diag/head/logit_max``: the largest |logit| over the step's loss slabs;
* ``diag/param/<part>``: parameter norm.

A step whose total gradient norm exceeds ``spike_factor`` times the median of the last 200 prints
``DIAG SPIKE step=<s> grad_norm=<g> median=<m> top=<part>:<norm>,...`` to stdout. ``stop_at_step``
> 0 ends training after that optimizer step (the schedule still follows ``max_steps``).
"""

from __future__ import annotations

import math
import re
import statistics
from collections import deque

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback

from lexhybrid.kernels.tfla import tfla_interface

_LAYER = re.compile(r"layers\.(\d+)\.")


class SpikeDiagnosticsCallback(Callback):
    def __init__(self, every_n_steps: int = 10, spike_factor: float = 5.0, stop_at_step: int = 0):
        self.every_n_steps = max(1, int(every_n_steps))
        self.spike_factor = float(spike_factor)
        self.stop_at_step = int(stop_at_step)
        self.parts: dict[str, list[torch.nn.Parameter]] = {}
        self.history: deque[float] = deque(maxlen=200)
        self.spikes: list[int] = []
        self._snapshot: dict[str, list[torch.Tensor]] | None = None
        self._snap_step: int | None = None
        self._active_step: int | None = None
        self._hooked_step: int | None = None
        self._stats: dict[str, float] = {}
        self._head_dims: dict[str, int] = {}
        self._handles: list = []
        self._current_mlstm: str | None = None
        self._patched: tuple | None = None

    # ------------------------------------------------------------------ set-up
    def on_fit_start(self, trainer, pl_module):
        model = pl_module.model
        types = model.get_layer_types()
        self.parts = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            m = _LAYER.match(name)
            if m:
                part = f"L{int(m.group(1))}_{types[int(m.group(1))]}"
            elif name.startswith("embeddings."):
                part = "embed"
            elif name.startswith("final_norm."):
                part = "final_norm"
            elif name.startswith("mtp_head."):
                part = "mtp"
            else:
                part = "other"
            self.parts.setdefault(part, []).append(p)
        for i, layer in enumerate(model.layers):
            self._hook_layer(f"L{i}_{layer.layer_type}", f"L{i}", layer)
        if getattr(model, "lm_head", None) is not None:
            self._handles.append(model.lm_head.register_forward_hook(self._max_abs("head/logit_max")))
        self._patch_tfla()

    def teardown(self, trainer, pl_module, stage):
        for h in self._handles:
            h.remove()
        self._handles = []
        if self._patched is not None:
            tfla_interface.tfla_forward_parallel, tfla_interface._row_block = self._patched
            self._patched = None
        self._snapshot = None

    def _hook_layer(self, part: str, short: str, layer) -> None:
        self._handles.append(layer.register_forward_hook(self._block_out(part)))
        if getattr(layer, "mlp", None) is not None:
            self._handles.append(layer.mlp.register_forward_hook(self._max_abs(f"mlp/{part}_max")))
        mixer = layer.mixer
        if layer.layer_type == "mlstm":
            self._handles.append(mixer.register_forward_pre_hook(self._enter_mlstm(short)))
            self._handles.append(
                mixer.i_gate_proj.register_forward_hook(self._extreme(f"mlstm/{short}_ilogit_max", max))
            )
            self._handles.append(
                mixer.f_gate_proj.register_forward_hook(self._extreme(f"mlstm/{short}_flogit_min", min))
            )
        if layer.layer_type == "attention" and getattr(mixer, "q_norm", None) is not None:
            self._head_dims[short] = int(mixer.head_dim)
            self._handles.append(mixer.q_norm.register_forward_hook(self._norm_max(f"_q/{short}")))
            self._handles.append(mixer.k_norm.register_forward_hook(self._norm_max(f"_k/{short}")))
        if layer.layer_type == "mamba3" and hasattr(mixer, "_split") and hasattr(mixer, "dt_bias"):
            self._handles.append(
                mixer.in_proj.register_forward_hook(self._mamba_dt(f"mamba/{short}_dt_max", mixer))
            )

    # ------------------------------------------------------------------ hooks
    def _put(self, key: str, value: float, reduce=max) -> None:
        self._stats[key] = reduce(self._stats[key], value) if key in self._stats else value

    @staticmethod
    def _tensor(output) -> torch.Tensor:
        return output[0] if isinstance(output, tuple) else output

    def _block_out(self, part):
        def hook(module, args, output):
            if self._active_step is None:
                return
            x = self._tensor(output).detach().float()
            self._put(f"act/{part}_rms", x.pow(2).mean().sqrt().item())
            self._put(f"act/{part}_max", x.abs().max().item())

        return hook

    def _max_abs(self, key):
        def hook(module, args, output):
            if self._active_step is not None:
                self._put(key, self._tensor(output).detach().float().abs().max().item())

        return hook

    def _extreme(self, key, reduce):
        def hook(module, args, output):
            if self._active_step is not None:
                x = self._tensor(output).detach().float()
                self._put(key, (x.max() if reduce is max else x.min()).item(), reduce)

        return hook

    def _norm_max(self, key):
        def hook(module, args, output):
            if self._active_step is not None:
                self._put(key, self._tensor(output).detach().float().norm(dim=-1).max().item())

        return hook

    def _mamba_dt(self, key, mixer):
        def hook(module, args, output):
            if self._active_step is None:
                return
            dt_raw = torch.split(self._tensor(output).detach(), mixer._split, dim=-1)[4]
            dt = F.softplus(dt_raw.float() + mixer.dt_bias.detach().float())
            if getattr(mixer, "dt_limit", None) is not None:
                dt = dt.clamp(max=mixer.dt_limit)
            self._put(key, dt.max().item())

        return hook

    def _enter_mlstm(self, short):
        def hook(module, args):
            self._current_mlstm = short

        return hook

    def _patch_tfla(self) -> None:
        forward, row_block = tfla_interface.tfla_forward_parallel, tfla_interface._row_block
        self._patched = (forward, row_block)

        def traced_forward(q, k, v, i_gate, f_gate, *args, **kwargs):
            if self._active_step is not None and self._current_mlstm is not None:
                chunk = int(kwargs.get("chunk_size", args[0] if args else 64))
                with torch.no_grad():
                    log_f = torch.log(f_gate.detach().float().clamp(min=1e-6))
                    n = log_f.shape[2] // chunk * chunk
                    if n:
                        cum = log_f[:, :, :n].unflatten(2, (n // chunk, chunk)).cumsum(3)
                        half = 0.5 * cum.amin(dim=3).abs().max().item()
                        self._put(f"mlstm/{self._current_mlstm}_half_range", half)
                self._put(f"mlstm/{self._current_mlstm}_row_block_min", 0.0)
            return forward(q, k, v, i_gate, f_gate, *args, **kwargs)

        def traced_row_block(log_f_cum, min_block=1):
            block = row_block(log_f_cum, min_block)
            if self._active_step is not None and self._current_mlstm is not None:
                key = f"mlstm/{self._current_mlstm}_row_block_min"
                value = float(block) if block is not None else -1.0
                self._stats[key] = value if self._stats.get(key, 0.0) == 0.0 else min(self._stats[key], value)
            return block

        tfla_interface.tfla_forward_parallel = traced_forward
        tfla_interface._row_block = traced_row_block

    # ------------------------------------------------------------------ training loop
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        step = int(trainer.global_step)
        if step % self.every_n_steps == 0 and step != self._hooked_step:
            self._hooked_step = step
            self._active_step = step
            self._stats = {}

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        step = int(trainer.global_step)
        metrics, total_sq = {}, 0.0
        for part, params in self.parts.items():
            grads = [p.grad for p in params if p.grad is not None]
            if not grads:
                continue
            norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads))).item()
            metrics[f"diag/grad/{part}"] = norm
            total_sq += norm * norm
        total = math.sqrt(total_sq)
        median = statistics.median(self.history) if self.history else math.nan
        if not math.isfinite(total) or (len(self.history) >= 20 and total > self.spike_factor * median):
            self.spikes.append(step)
            top = sorted(
                ((v, k.split("/")[-1]) for k, v in metrics.items()),
                key=lambda x: -x[0] if math.isfinite(x[0]) else -math.inf,
            )[:4]
            print(
                f"DIAG SPIKE step={step} grad_norm={total:.3f} median={median:.3f} top="
                + ",".join(f"{name}:{v:.3f}" for v, name in top),
                flush=True,
            )
        if math.isfinite(total):
            self.history.append(total)
        if step % self.every_n_steps == 0:
            for part, params in self.parts.items():
                metrics[f"diag/param/{part}"] = torch.linalg.vector_norm(
                    torch.stack(torch._foreach_norm([p.detach() for p in params]))
                ).item()
        if self._snapshot is None:
            self._snapshot = {
                part: [torch.empty_like(p) for p in params] for part, params in self.parts.items()
            }
        for part, params in self.parts.items():
            torch._foreach_copy_(self._snapshot[part], [p.detach() for p in params])
        self._snap_step = step
        self._log(trainer, metrics, step)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if self._active_step is not None:
            stats = self._finish_forward_stats()
            self._log(trainer, {f"diag/{k}": v for k, v in stats.items()}, self._active_step)
            self._active_step = None
        if self._snap_step is not None and step > self._snap_step:
            metrics = {}
            for part, params in self.parts.items():
                before = self._snapshot[part]
                w = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(before))).item()
                torch._foreach_sub_(before, [p.detach() for p in params])  # in place: before - after
                d = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(before))).item()
                metrics[f"diag/update/{part}"] = d / w if w > 0 else 0.0
            self._log(trainer, metrics, self._snap_step)
            self._snap_step = None
        if self.stop_at_step > 0 and step >= self.stop_at_step:
            print(f"DIAG STOP step={step} spikes={self.spikes}", flush=True)
            trainer.should_stop = True

    def _finish_forward_stats(self) -> dict[str, float]:
        stats = {k: v for k, v in self._stats.items() if not k.startswith("_")}
        for short, dim in self._head_dims.items():
            q, k = self._stats.get(f"_q/{short}"), self._stats.get(f"_k/{short}")
            if q is not None and k is not None:
                stats[f"attn/{short}_logit_bound"] = q * k / math.sqrt(dim)
        return stats

    @staticmethod
    def _log(trainer: pl.Trainer, metrics: dict, step: int) -> None:
        if metrics and trainer.logger is not None and trainer.is_global_zero:
            trainer.logger.log_metrics(metrics, step=step)
