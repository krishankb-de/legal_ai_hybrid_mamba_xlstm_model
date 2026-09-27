"""Multi-token prediction (plan P2-S; screen arm S4). Off by default (``mtp_n = 1``).

With ``mtp_n = D + 1``, depth ``k`` (1..D) predicts the token ``k + 1`` positions ahead, in the
sequential form of DeepSeek-V3 (arXiv:2412.19437, Sec. 2.2): module ``k`` combines the previous
depth's hidden state at position ``i`` with the embedding of token ``i + k``,

    h_k[i] = Block_k( W_k [ RMSNorm(h_{k-1}[i]) ; RMSNorm(Emb(t_{i+k})) ] )

and the model's own final norm and (tied) head turn ``h_k`` into logits for ``t_{i+k+1}``. The
embedding and the head are the main model's -- shared, not copied, and not registered here, so the
state dict holds them once. ``h_0`` is the main model's last residual stream.

The loss of depth ``k`` supervises position ``i`` only when ``i`` and ``i + k + 1`` lie in the same
document (the P2-L rule, extended), and the total added to the language-model loss is
``mtp_loss_weight * mean_k(CE_k)``. Nothing here runs at inference time.
"""

import torch
import torch.nn as nn

from lexhybrid.kernels.segments import segment_ids
from lexhybrid.layers.normalization import RMSNorm
from lexhybrid.models.slab_loss import slab_cross_entropy


class MTPModule(nn.Module):
    """One prediction depth: two norms, a 2D -> D projection, and one ``HybridBlock``."""

    def __init__(self, block: nn.Module, dim: int):
        super().__init__()
        self.norm_hidden = RMSNorm(dim)
        self.norm_embed = RMSNorm(dim)
        self.proj = nn.Linear(2 * dim, dim, bias=False)
        self.block = block

    def forward(
        self, hidden: torch.Tensor, embedded: torch.Tensor, doc_ids: torch.Tensor | None
    ) -> torch.Tensor:
        x = self.proj(torch.cat([self.norm_hidden(hidden), self.norm_embed(embedded)], dim=-1))
        return self.block(x, cache=None, doc_ids=doc_ids)


class MTPHead(nn.Module):
    """``depths`` sequential ``MTPModule``s; ``blocks`` are built by the model's block factory."""

    def __init__(self, blocks: list[nn.Module], dim: int):
        super().__init__()
        self.depths = nn.ModuleList(MTPModule(block, dim) for block in blocks)

    def loss(self, hidden, input_ids, embed, head_fn, doc_ids=None, labels=None, slab: int = 512):
        """Mean over depths of each depth's boundary-masked cross-entropy.

        Args:
            hidden: (B, L, D) main-model residual stream (before the final norm).
            input_ids: (B, L) the tokens (their embeddings feed each depth).
            embed: the shared token embedding module.
            head_fn: ``lambda h: lm_head(final_norm(h))``.
            doc_ids: optional (B, L) document ids.
            labels: optional (B, L) targets (``input_ids`` if None); -100 is ignored.
        """
        labels = input_ids if labels is None else labels
        seg = segment_ids(doc_ids, *input_ids.shape, input_ids.device)
        per_depth = []
        h = hidden
        for k, module in enumerate(self.depths, start=1):
            length = input_ids.shape[1] - k - 1
            if length <= 0:
                break
            ids_k = None if doc_ids is None else doc_ids[:, :length]
            h = module(h[:, :length], embed(input_ids[:, k : k + length]), ids_k)
            targets = labels[:, k + 1 : k + 1 + length].clone()
            targets = targets.masked_fill(seg[:, k + 1 : k + 1 + length] != seg[:, :length], -100)
            ce_sum, n = slab_cross_entropy(h, targets, head_fn, slab=slab)
            per_depth.append(ce_sum / n.clamp(min=1))
        if not per_depth:
            return hidden.new_zeros((), dtype=torch.float32)
        return torch.stack(per_depth).mean()
