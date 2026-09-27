"""Warmup-Stable-Decay (WSD) learning-rate schedule and the beta2 anneal (ported verbatim).

WSD: linear warmup (1% of max_steps by default, or an absolute count), a constant stable phase
(85%), and a decay phase (14%) with factor ``1 - sqrt(p)``. The beta2 helper anneals AdamW's beta2
linearly from 0.999 to 0.974 across the decay phase; the training loop applies it every step.

Reference values: 120,000 steps with warmup 2,000 -> decay 16,800, stable 101,200, decay start
103,200; 12,000 steps with warmup 500 -> decay 1,680, decay start 10,320.
"""

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def wsd_factor(
    step: int,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    min_lr_ratio: float = 0.0,
    warmup_start_factor: float = 0.01,
) -> float:
    """WSD learning-rate multiplier at ``step``.

    warmup: linear from ``warmup_start_factor`` to 1.0; stable: 1.0; decay: ``1 - sqrt(p)``,
    clamped at ``min_lr_ratio``.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return warmup_start_factor + (1.0 - warmup_start_factor) * (step / warmup_steps)
    decay_start = warmup_steps + stable_steps
    if step < decay_start:
        return 1.0
    if decay_steps <= 0:
        return min_lr_ratio
    p = min(1.0, (step - decay_start) / decay_steps)
    return max(min_lr_ratio, 1.0 - math.sqrt(p))


class WSDScheduler(LambdaLR):
    """Warmup-Stable-Decay scheduler. Ratios must sum to 1; an absolute ``warmup_steps`` keeps the
    decay ratio and absorbs the remainder into the stable phase."""

    def __init__(
        self,
        optimizer: Optimizer,
        max_steps: int,
        warmup_ratio: float = 0.01,
        stable_ratio: float = 0.85,
        decay_ratio: float = 0.14,
        min_lr_ratio: float = 0.0,
        warmup_start_factor: float = 0.01,
        warmup_steps: int | None = None,
        last_epoch: int = -1,
    ):
        total = warmup_ratio + stable_ratio + decay_ratio
        if abs(total - 1.0) >= 1e-6:
            raise ValueError(f"WSD ratios must sum to 1.0, got {total!r}")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        self.max_steps = max_steps
        if warmup_steps is not None:
            self.warmup_steps = int(warmup_steps)
            self.decay_steps = int(round(max_steps * decay_ratio))
            self.stable_steps = max(0, max_steps - self.warmup_steps - self.decay_steps)
        else:
            self.warmup_steps = int(round(max_steps * warmup_ratio))
            self.stable_steps = int(round(max_steps * stable_ratio))
            self.decay_steps = max(0, max_steps - self.warmup_steps - self.stable_steps)
        self.decay_start = self.warmup_steps + self.stable_steps
        self.min_lr_ratio = min_lr_ratio
        self.warmup_start_factor = warmup_start_factor

        def _lambda(step: int) -> float:
            return wsd_factor(
                step,
                self.warmup_steps,
                self.stable_steps,
                self.decay_steps,
                min_lr_ratio=self.min_lr_ratio,
                warmup_start_factor=self.warmup_start_factor,
            )

        super().__init__(optimizer, lr_lambda=_lambda, last_epoch=last_epoch)


def beta2_for_step(
    step: int,
    decay_start: int,
    decay_steps: int,
    beta2_start: float = 0.999,
    beta2_end: float = 0.974,
) -> float:
    """Linear beta2 anneal across the decay phase; ``beta2_start`` before it."""
    if decay_steps <= 0 or step < decay_start:
        return beta2_start
    p = min(1.0, (step - decay_start) / decay_steps)
    return beta2_start + (beta2_end - beta2_start) * p


def apply_beta2_schedule(
    optimizer: Optimizer,
    step: int,
    decay_start: int,
    decay_steps: int,
    beta2_start: float = 0.999,
    beta2_end: float = 0.974,
) -> float:
    """Set every param group's beta2 to the scheduled value in place; returns it for logging."""
    b2 = beta2_for_step(step, decay_start, decay_steps, beta2_start, beta2_end)
    for pg in optimizer.param_groups:
        if "betas" in pg:
            b1, _ = pg["betas"]
            pg["betas"] = (b1, b2)
    return b2
