"""Training and evaluation metrics (ported from the reference)."""

import math

import torch


def compute_perplexity(loss: torch.Tensor) -> torch.Tensor:
    """exp(cross-entropy)."""
    return torch.exp(loss)


def compute_bits_per_token(loss: torch.Tensor) -> torch.Tensor:
    """Cross-entropy in bits per token (``loss / ln 2``).

    The reference named this "bits per byte", but nothing divides by bytes per token (defect 13).
    """
    return loss / math.log(2)


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Token-level argmax accuracy over positions whose label is not ``ignore_index``."""
    predictions = torch.argmax(logits, dim=-1)
    mask = labels != ignore_index
    correct = (predictions == labels) & mask
    return correct.sum().float() / mask.sum().float()


def compute_mqar_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    query_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Accuracy at the query positions of a multi-query associative-recall sequence."""
    predictions = torch.argmax(logits, dim=-1)
    if query_positions is None:
        query_positions = torch.ones_like(labels, dtype=torch.bool)
    correct = (predictions == labels) & query_positions
    return correct.sum().float() / query_positions.sum().float()
