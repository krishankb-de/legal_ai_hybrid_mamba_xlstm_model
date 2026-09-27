"""Cross-entropy over a large vocabulary without materialising ``(B, L, V)`` logits.

At the Qwen3 vocabulary (151,936) the logits of one 4,096-token row are 622M floats. The loss is
computed slab by slab along the sequence, each slab under activation checkpointing, so at most one
``(B, slab, V)`` block is alive in the forward pass and it is recomputed in the backward pass.
The distillation loss (``lexhybrid.training.distill``) and the multi-token-prediction head use it.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

HeadFn = Callable[[torch.Tensor], torch.Tensor]  # hidden (B, S, D) -> logits (B, S, V)


def _slab_ce_sum(hidden: torch.Tensor, targets: torch.Tensor, head_fn: HeadFn) -> torch.Tensor:
    logits = head_fn(hidden).float()
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100, reduction="sum"
    )


def slab_cross_entropy(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    head_fn: HeadFn,
    slab: int = 512,
    checkpoint: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed cross-entropy of ``head_fn(hidden)`` against ``targets`` (-100 ignored), slab-wise.

    Args:
        hidden: (B, L, D) states that ``head_fn`` maps to logits.
        targets: (B, L) target ids, -100 where nothing is supervised.
        head_fn: e.g. ``lambda h: lm_head(final_norm(h))``.
        slab: sequence positions per slab.
        checkpoint: recompute each slab's logits in the backward pass instead of keeping them.

    Returns:
        ``(ce_sum, n_supervised)``; the mean is ``ce_sum / n_supervised.clamp(min=1)``.
    """
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, hidden.shape[1], slab):
        h, t = hidden[:, start : start + slab], targets[:, start : start + slab]
        if checkpoint and torch.is_grad_enabled() and h.requires_grad:
            total = total + torch.utils.checkpoint.checkpoint(
                _slab_ce_sum, h, t, head_fn, use_reentrant=False
            )
        else:
            total = total + _slab_ce_sum(h, t, head_fn)
    return total, (targets != -100).sum()
