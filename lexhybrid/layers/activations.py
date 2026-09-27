"""Activation functions used by the mixers."""

import torch
import torch.nn.functional as F


def exponential_activation(x: torch.Tensor) -> torch.Tensor:
    """Exponential gate activation of the mLSTM input gate."""
    return torch.exp(x)


def silu_activation(x: torch.Tensor) -> torch.Tensor:
    """SiLU (swish) activation."""
    return F.silu(x)
