"""Document segments inside packed rows, shared by the SSD and TFLA kernels.

A packed row holds several documents; ``doc_ids`` (batch, seqlen) gives each position's document.
The kernels need two derived forms: a monotone segment index (positions sharing a value share a
document, even if two different documents happen to reuse an id) and the boolean "a new document
starts here" mask.

Resets are applied as boolean masks, never by folding a large negative number into a cumulative
log-decay: adding -1e30 to a cumulative sum annihilates the finite part, so every within-document
decay would be destroyed along with the cross-document one.
"""

import torch


def segment_starts(doc_ids: torch.Tensor) -> torch.Tensor:
    """(batch, seqlen) bool, True where a new document begins; position 0 is False."""
    starts = torch.zeros_like(doc_ids, dtype=torch.bool)
    starts[:, 1:] = doc_ids[:, 1:] != doc_ids[:, :-1]
    return starts


def segment_ids(doc_ids: torch.Tensor | None, batch: int, seqlen: int, device: torch.device) -> torch.Tensor:
    """Monotone segment index per position, (batch, seqlen) long; all zeros without ``doc_ids``."""
    if doc_ids is None:
        return torch.zeros(batch, seqlen, dtype=torch.long, device=device)
    return segment_starts(doc_ids).long().cumsum(dim=1)


def pad_segments(seg: torch.Tensor, pad: int) -> torch.Tensor:
    """Extend ``seg`` by ``pad`` positions that join the LAST real segment.

    Padding carries forget gates of 1 (or zero decay) and zero inputs, so it leaves the state
    untouched. Giving it the last real segment's id means the state that reaches the end of the
    padded row is the state after the last real token; a fresh id (the reference's choice, which
    also cost a host sync) would have discarded it (defect 7).
    """
    if pad <= 0:
        return seg
    return torch.cat([seg, seg[:, -1:].expand(seg.shape[0], pad)], dim=1)


def forward_fill_from_starts(values: torch.Tensor, starts: torch.Tensor, dim: int) -> torch.Tensor:
    """For each position, ``values`` at the most recent start at or before it along ``dim``.

    ``starts`` has ``values``' shape without its trailing dimensions (it is broadcast), and a
    position before any start takes index 0 -- callers make ``values`` there the neutral element.
    """
    n = starts.shape[dim]
    shape = [1] * starts.dim()
    shape[dim] = n
    idx = torch.arange(n, device=starts.device).view(shape).expand_as(starts)
    last = torch.where(starts, idx, torch.zeros_like(idx)).cummax(dim=dim).values
    while last.dim() < values.dim():
        last = last.unsqueeze(-1)
    return torch.gather(values, dim, last.expand_as(values))
