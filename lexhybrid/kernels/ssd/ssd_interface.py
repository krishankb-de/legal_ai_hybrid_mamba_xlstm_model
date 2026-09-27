"""Chunked SSD scan -- the Mamba-2/3 state space dual (reference M2-B).

Why this exists rather than a patch to ``selective_scan``: Mamba-1 parameterizes ``A`` as
``(d_inner, dstate)``, one decay per channel *and* state, so the exact intra-chunk pairwise decay
is a ``(chunk, chunk, d_inner, dstate)`` tensor -- 19.3 GB at chunk 64 / batch 48. Mamba-2/3 make
``A`` a scalar per head, so the same tensor is ``(chunk, chunk)`` per head: 19 MB, shaped as a
matmul. Correctness is affordable here and is not there.

The algorithm is the standard chunked decomposition: within a chunk the output is
``(L . C B^T) X`` with ``L`` the lower-triangular decay mask; across chunks a linear recurrence
carries the state. Nothing is divided by a decay, so there is no clamp and no annihilation.

Shapes follow ``ssd_reference``:

    x   (batch, seqlen, nheads, headdim)
    dt  (batch, seqlen, nheads)            sets the decay exp(dt * A)
    A   (nheads,)                          strictly negative
    B   (batch, seqlen, ngroups, dstate)
    C   (batch, seqlen, ngroups, dstate)
    D   (nheads,) or None

P2-F (defect 7): padding joins the last real document instead of a fresh segment id (which cost
a ``.item()`` host sync and made the state after a partial last chunk come back as zero), and
``return_final_state`` hands back the fp32 state after the last token -- what ``ssd_step``
continues from, so a forward pass can fill the decode cache. P2-E (defect 16): the carried state
is fp32 by declaration.
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from lexhybrid.kernels.segments import pad_segments, segment_ids

Term = tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # (coefficient, B, x)


def segsum(v: torch.Tensor) -> torch.Tensor:
    """Lower-triangular segment sums: ``out[..., t, s] = sum_{i=s+1}^{t} v_i`` for ``s <= t``.

    Entries above the diagonal are ``-inf``, so ``exp(segsum(v))`` is the causal decay mask
    directly. The diagonal is an empty sum, hence ``exp(0) = 1``: a token always contributes to its
    own state at full weight, whatever the decay -- the property the divide-and-clamp form loses.
    """
    T = v.size(-1)
    vv = v.unsqueeze(-1).expand(*v.shape, T)  # [..., d, e] = v[d]
    lower = torch.ones(T, T, dtype=torch.bool, device=v.device).tril(-1)
    vv = vv.masked_fill(~lower, 0)
    out = vv.cumsum(dim=-2)
    causal = torch.ones(T, T, dtype=torch.bool, device=v.device).tril(0)
    return out.masked_fill(~causal, -float("inf"))


def _segment_ids(doc_ids: torch.Tensor | None, batch: int, seqlen: int, device: torch.device) -> torch.Tensor:
    """Monotone segment index per position (see ``lexhybrid.kernels.segments``)."""
    return segment_ids(doc_ids, batch, seqlen, device)


def ssd_chunked_scan(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    chunk_size: int = 64,
    doc_ids: torch.Tensor | None = None,
    coeff: torch.Tensor | None = None,
    extra_terms: Sequence[Term] | None = None,
    return_final_state: bool = False,
):
    """Chunk-parallel SSD scan.

        h_t = exp(dt_t A) h_{t-1} + sum_terms coeff_t (B_t x_t^T)
        y_t = C_t^T h_t + D x_t

    Args:
        chunk_size: intra-chunk block; the mask is ``(chunk_size, chunk_size)`` per head.
        doc_ids: optional (batch, seqlen) document ids; the state resets where the id changes.
        coeff: state-input coefficient, (batch, seqlen, nheads). Defaults to ``dt`` (Mamba-2's
            exponential-Euler rule). Separate from ``dt`` because the exponential-trapezoidal rule
            scales the input by ``gamma = lambda * dt`` while the decay stays ``exp(dt A)``.
        extra_terms: additional ``(coefficient, B, x)`` triples summed into the state -- how the
            trapezoidal rule's ``beta * B_{t-1} x_{t-1}`` term is added without a second scan.
        return_final_state: also return the fp32 state after the last token,
            (batch, nheads, headdim, dstate).

    Returns:
        (batch, seqlen, nheads, headdim), or ``(y, final_state)`` with ``return_final_state``.
    """
    batch, seqlen, nheads, headdim = x.shape
    ngroups, dstate = B.shape[-2], B.shape[-1]
    if nheads % ngroups != 0:
        raise ValueError(f"nheads ({nheads}) must be divisible by ngroups ({ngroups})")
    rep = nheads // ngroups
    chunk_size = max(1, min(chunk_size, seqlen))

    terms: list[Term] = [(dt if coeff is None else coeff, B, x)] + list(extra_terms or [])
    dA = dt * A  # (batch, seqlen, nheads)
    seg = _segment_ids(doc_ids, batch, seqlen, x.device)

    pad = (chunk_size - seqlen % chunk_size) % chunk_size
    if pad:
        dA = F.pad(dA, (0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, 0, 0, pad))
        # Padding has zero decay and zero input, so it joins the last real document: the state that
        # reaches the end of the padded row is then the state after the last real token.
        seg = pad_segments(seg, pad)
        terms = [
            (F.pad(co, (0, 0, 0, pad)), F.pad(bb, (0, 0, 0, 0, 0, pad)), F.pad(xx, (0, 0, 0, 0, 0, pad)))
            for co, bb, xx in terms
        ]
    padded = dA.shape[1]
    nc = padded // chunk_size

    dA_c = dA.reshape(batch, nc, chunk_size, nheads)
    A_cum = dA_c.cumsum(dim=2)  # decay from the chunk's start
    seg_c = seg.reshape(batch, nc, chunk_size)
    C_c = C.reshape(batch, nc, chunk_size, ngroups, dstate)

    # Three boolean masks encode every document reset:
    #   same     -- token s may influence token t at all
    #   carry_ok -- the state carried in from earlier chunks is still valid at t
    #   end_ok   -- token t's contribution survives to the chunk boundary
    same = seg_c.unsqueeze(-1) == seg_c.unsqueeze(-2)  # (batch, nc, cs, cs)
    prev_seg = torch.full((batch, nc), -1, dtype=seg_c.dtype, device=seg_c.device)
    if nc > 1:
        prev_seg[:, 1:] = seg_c[:, :-1, -1]
    carry_ok = seg_c == prev_seg.unsqueeze(-1)  # (batch, nc, cs)
    end_ok = seg_c == seg_c[:, :, -1:]  # (batch, nc, cs)

    chunked: list[Term] = [
        (
            co.reshape(batch, nc, chunk_size, nheads),
            bb.reshape(batch, nc, chunk_size, ngroups, dstate),
            xx.reshape(batch, nc, chunk_size, nheads, headdim),
        )
        for co, bb, xx in terms
    ]

    mask = torch.exp(segsum(dA_c.permute(0, 1, 3, 2)))  # (batch, nc, nheads, cs, cs)
    mask = mask * same.unsqueeze(2)

    # The decay factors are built in fp32 by policy while x/B/C carry the model's dtype, so every
    # einsum below is handed operands that already agree: einsum does NOT promote. Under autocast
    # this cast is exactly what autocast applies; without autocast (a `.to(bfloat16)` model) it is
    # the difference between running and a dtype error (reference job 2560261).
    cdtype = x.dtype

    # --- intra-chunk: y[t] = sum_{s<=t} mask[t,s] <C_t, B_s> coeff_s x_s ---------------------
    y = torch.zeros(batch, nc, chunk_size, nheads, headdim, device=x.device, dtype=x.dtype)
    for co_c, B_c, x_ct in chunked:
        CB = torch.einsum("bctgn,bcsgn->bcgts", C_c, B_c)  # (batch, nc, ngroups, cs, cs)
        CB = CB.repeat_interleave(rep, dim=2)  # broadcast groups over heads
        y = y + torch.einsum(
            "bchts,bcsh,bcshp->bcthp", (mask * CB).to(cdtype), co_c.to(cdtype), x_ct.to(cdtype)
        )

    # --- inter-chunk: carry the state across chunks (nc sequential steps) --------------------
    # fp32 by declaration (defect 16, P2-E): the reference allocated it in the activation dtype and
    # it became fp32 only because the first decay multiplication promoted it.
    state = torch.zeros(batch, nheads, headdim, dstate, device=x.device, dtype=torch.float32)
    C_h = C_c.repeat_interleave(rep, dim=3)  # (batch, nc, cs, nheads, dstate)
    offsets = []
    for ci in range(nc):
        A_cum_ci = A_cum[:, ci]  # (batch, cs, nheads)
        carry_gate = (torch.exp(A_cum_ci) * carry_ok[:, ci].unsqueeze(-1)).unsqueeze(-1)
        offsets.append(torch.einsum("bhpn,bthn->bthp", state.to(cdtype), C_h[:, ci].to(cdtype)) * carry_gate)

        keep = end_ok[:, ci].unsqueeze(-1)  # (batch, cs, 1)
        decay_to_end = torch.exp(A_cum_ci[:, -1:] - A_cum_ci) * keep
        state = (torch.exp(A_cum_ci[:, -1]) * carry_ok[:, ci, -1].unsqueeze(-1)).unsqueeze(-1).unsqueeze(
            -1
        ) * state
        for co_c, B_c, x_ct in chunked:
            state = state + torch.einsum(
                "bth,bth,bthp,bthn->bhpn",
                decay_to_end.to(cdtype),
                co_c[:, ci].to(cdtype),
                x_ct[:, ci].to(cdtype),
                B_c[:, ci].repeat_interleave(rep, dim=2).to(cdtype),
            )

    y = (y + torch.stack(offsets, dim=1)).reshape(batch, padded, nheads, headdim)
    if D is not None:
        y = y + D.view(1, 1, -1, 1) * (F.pad(x, (0, 0, 0, 0, 0, pad)) if pad else x)
    # Return in the dtype we were handed: the fp32 decay factors and fp32 `D` promote `y`.
    y = y.to(cdtype)
    y = y[:, :seqlen] if pad else y
    return (y, state) if return_final_state else y
