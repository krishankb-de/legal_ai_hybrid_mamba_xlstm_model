"""Optimizer construction with weight-decay parameter groups (ported from the reference)."""

from typing import Any

import torch
from torch.optim import SGD, Adam, AdamW, Optimizer

# Substrings of lower-cased parameter names that are excluded from weight decay.
_NO_DECAY_BIAS = ("bias",)
_NO_DECAY_NORM = ("norm", "ln", "layernorm", "layer_norm", "rmsnorm")
_NO_DECAY_EMBEDDING = ("embedding", "embed", "pos_embed", "token_embedding")


def get_parameter_groups(
    model: torch.nn.Module,
    weight_decay: float = 0.1,
    no_decay_bias: bool = True,
    no_decay_norm: bool = True,
    no_decay_embedding: bool = True,
) -> list[dict[str, Any]]:
    """Split trainable parameters into a decayed and a non-decayed group by name substring."""
    no_decay_names: set[str] = set()
    if no_decay_bias:
        no_decay_names.update(_NO_DECAY_BIAS)
    if no_decay_norm:
        no_decay_names.update(_NO_DECAY_NORM)
    if no_decay_embedding:
        no_decay_names.update(_NO_DECAY_EMBEDDING)

    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name_lower = name.lower()
        if any(nd in name_lower for nd in no_decay_names):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def configure_optimizer(
    model: torch.nn.Module,
    optimizer_name: str = "adamw",
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    momentum: float = 0.9,
    foreach: bool = True,
    **kwargs,
) -> Optimizer:
    """Build an optimizer over ``get_parameter_groups``; ``foreach`` is used only on CUDA."""
    param_groups = get_parameter_groups(model, weight_decay=weight_decay)
    name = optimizer_name.lower()
    use_foreach = foreach and torch.cuda.is_available()
    if name == "adamw":
        return AdamW(param_groups, lr=learning_rate, betas=betas, eps=eps, foreach=use_foreach, **kwargs)
    if name == "adam":
        return Adam(param_groups, lr=learning_rate, betas=betas, eps=eps, foreach=use_foreach, **kwargs)
    if name == "sgd":
        return SGD(param_groups, lr=learning_rate, momentum=momentum, **kwargs)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")
