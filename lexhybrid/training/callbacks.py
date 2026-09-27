"""SLURM-friendly checkpoint-on-signal callback (ported from the reference).

Saves ``<checkpoint_dir>/interrupt.ckpt`` on SIGTERM / SIGUSR1 (SLURM preemption) or on an
uncaught exception, once. Pair it with a wrapper that resumes from ``last.ckpt``: the reference's
wrappers never passed a resume path, so a requeued job restarted from step 0 (defect 12).
"""

import os
import signal
from pathlib import Path

import pytorch_lightning as pl
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
