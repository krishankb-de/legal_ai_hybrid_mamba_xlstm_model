"""Training callbacks: checkpoint-on-signal (ported from the reference) and step statistics.

``SignalCheckpointCallback`` saves ``<checkpoint_dir>/interrupt.ckpt`` on SIGTERM / SIGUSR1 (SLURM
preemption) or on an uncaught exception, once. Pair it with a wrapper that resumes from
``last.ckpt``: the reference's wrappers never passed a resume path, so a requeued job restarted from
step 0 (defect 12).

``StepStatsCallback`` prints seconds per optimizer step and peak GPU memory for every rank as
``STEPSTATS`` lines, the numbers the KD probe (P4-T) and the screen tables (P5-F) record.
"""

import os
import signal
import statistics
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback


class SignalCheckpointCallback(Callback):
    """Write ``interrupt.ckpt`` when the process is signalled; a no-op after the first save."""

    def __init__(self, checkpoint_dir: str, filename: str = "interrupt.ckpt"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.filename = filename
        self._trainer: pl.Trainer | None = None
        self._saved = False

    def setup(self, trainer, pl_module, stage):
        self._trainer = trainer
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            for sig in (signal.SIGTERM, signal.SIGUSR1):
                try:
                    signal.signal(sig, self._handle)
                except (ValueError, OSError):
                    # Not in the main thread (some launchers); skip.
                    pass

    def on_exception(self, trainer, pl_module, exception):
        self._save("exception")

    def _handle(self, signum, frame):
        self._save(f"signal_{signum}")
        # Re-raise as SystemExit so Lightning unwinds cleanly and SLURM can requeue.
        raise SystemExit(0)

    def _save(self, reason: str):
        if self._saved or self._trainer is None:
            return
        self._saved = True
        target = self.checkpoint_dir / self.filename
        try:
            self._trainer.save_checkpoint(str(target))
            print(f"[SignalCheckpointCallback] Saved {target} ({reason}).")
        except Exception as e:  # checkpointing must not mask the original exit
            print(f"[SignalCheckpointCallback] Save failed on {reason}: {e}")


class StepStatsCallback(Callback):
    """Seconds per optimizer step and peak GPU memory, per rank (plan P4-T, P5-F).

    Every ``every_n_steps`` optimizer steps, and once when training ends, each rank prints::

        STEPSTATS rank=0 step=20 s_per_step=1.2345 n=15 peak_alloc_gb=31.20 peak_reserved_gb=33.00 final=1

    ``s_per_step`` is the median over the steps timed so far. The first ``skip_steps`` steps
    (compilation, allocator warm-up) are not timed, and the clock restarts after every validation
    run, so validation never counts toward a step. The clock reads after a CUDA synchronise, so
    queued kernels count toward the step that launched them. Peak memory is the process's high-water mark since start (``max_memory_*``);
    ``n/a`` without CUDA.
    """

    def __init__(self, every_n_steps: int = 500, skip_steps: int = 5):
        self.every_n_steps = int(every_n_steps)
        self.skip_steps = int(skip_steps)
        self.durations: list[float] = []
        self._last_step: int | None = None
        self._last_time: float | None = None

    @staticmethod
    def _now() -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step == self._last_step:  # a gradient-accumulation micro-batch, not an optimizer step
            return
        now = self._now()
        if self._last_time is not None and self._last_step is not None and step > self.skip_steps:
            self.durations.append((now - self._last_time) / (step - self._last_step))
        self._last_step, self._last_time = step, now
        if self.every_n_steps > 0 and step > 0 and step % self.every_n_steps == 0:
            self.report(trainer, final=False)

    def on_validation_end(self, trainer, pl_module):
        # restart the clock after the validation run, so the next step is timed without it
        self._last_time = self._now() if self._last_step is not None else None

    def on_train_end(self, trainer, pl_module):
        self.report(trainer, final=True)

    def summary(self) -> dict:
        cuda = torch.cuda.is_available()
        return {
            "s_per_step": statistics.median(self.durations) if self.durations else None,
            "n": len(self.durations),
            "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1e9 if cuda else None,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9 if cuda else None,
        }

    def report(self, trainer, final: bool) -> str:
        s = self.summary()

        def fmt(value, spec):
            return "n/a" if value is None else format(value, spec)

        line = (
            f"STEPSTATS rank={trainer.global_rank} step={int(trainer.global_step)} "
            f"s_per_step={fmt(s['s_per_step'], '.4f')} n={s['n']} "
            f"peak_alloc_gb={fmt(s['peak_alloc_gb'], '.2f')} "
            f"peak_reserved_gb={fmt(s['peak_reserved_gb'], '.2f')} final={int(final)}"
        )
        print(line, flush=True)
        return line
