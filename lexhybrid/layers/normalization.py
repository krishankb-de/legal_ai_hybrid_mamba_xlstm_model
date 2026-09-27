"""Normalization layers.

Ported unchanged from the reference. ``RMSNorm`` computes in the input dtype with no fp32
upcast; that is recorded defect 10 and is fixed in P2-E, not here, so the P1 parity fixtures hold.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization.

    Args:
        dim: size of the normalized (last) dimension.
        eps: added to the mean square before the square root.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.weight * x_norm


def get_norm_layer(norm_type: str, dim: int, eps: float = 1e-6) -> nn.Module:
    """``RMSNorm`` for ``"rms"``, ``nn.LayerNorm`` for ``"layer"``."""
    norm_type = norm_type.lower()
    if norm_type in ("rms", "rmsnorm"):
        return RMSNorm(dim, eps=eps)
    if norm_type in ("layer", "layernorm"):
        return nn.LayerNorm(dim, eps=eps)
    raise ValueError(f"Unknown normalization type: {norm_type}")
