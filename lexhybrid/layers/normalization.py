"""Normalization layers.

Ported from the reference. P2-E (defect 10): ``RMSNorm`` computes in fp32 and casts back to the
input dtype. The reference squared and averaged in the activation dtype, so a bf16 model lost the
mean-square to bf16's 8-bit mantissa. For fp32 inputs the computation is unchanged bit for bit.
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
        x32 = x.float()
        rms = torch.sqrt(torch.mean(x32**2, dim=-1, keepdim=True) + self.eps)
        return (self.weight.float() * (x32 / rms)).to(x.dtype)


def get_norm_layer(norm_type: str, dim: int, eps: float = 1e-6) -> nn.Module:
    """``RMSNorm`` for ``"rms"``, ``nn.LayerNorm`` for ``"layer"``."""
    norm_type = norm_type.lower()
    if norm_type in ("rms", "rmsnorm"):
        return RMSNorm(dim, eps=eps)
    if norm_type in ("layer", "layernorm"):
        return nn.LayerNorm(dim, eps=eps)
    raise ValueError(f"Unknown normalization type: {norm_type}")
