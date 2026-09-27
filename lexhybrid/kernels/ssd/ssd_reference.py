"""Sequential reference for the SSD recurrence, and the single-token decode step (reference M2-A).

* ``ssd_sequential_reference`` is the oracle every chunked-scan test is checked against. It runs
  in float64 by default and is a plain Python loop, because an oracle has to be obviously correct
  rather than fast.
* ``ssd_step`` advances that same recurrence by exactly one token given a carried state. That is
  the decode path, so the oracle and the decode kernel cannot drift apart.

Shape conventions, shared with ``ssd_interface``:

    x   (batch, seqlen, nheads, headdim)     SSM input, one vector per head
    dt  (batch, seqlen, nheads)              per-head step size, already positive
    A   (nheads,)                            per-head state transition, strictly negative
    B   (batch, seqlen, ngroups, dstate)     state-input projection, shared across heads
    C   (batch, seqlen, ngroups, dstate)     state-output projection, shared across heads
    D   (nheads,) or None                    skip connection
    state (batch, nheads, headdim, dstate)   the recurrent state
"""

from collections.abc import Sequence

import torch


def _heads_per_group(nheads: int, ngroups: int) -> int:
    if nheads % ngroups != 0:
        raise ValueError(f"nheads ({nheads}) must be divisible by ngroups ({ngroups})")
    return nheads // ngroups


def ssd_step(
    x_t: torch.Tensor,
    dt_t: torch.Tensor,
    A: torch.Tensor,
    B_t: torch.Tensor,
    C_t: torch.Tensor,
    state: torch.Tensor,
    D: torch.Tensor | None = None,
    coeff: torch.Tensor | None = None,
    extra_terms: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance the SSD recurrence by one token.

        h_t = exp(dt_t * A) h_{t-1} + sum_terms coeff_t (B_t x_t^T)
        y_t = C_t^T h_t + D * x_t

    Args:
        x_t: (batch, nheads, headdim)
        dt_t: (batch, nheads) -- sets the decay exp(dt_t A)
        A: (nheads,) or (batch, nheads)
        B_t, C_t: (batch, ngroups, dstate)
        state: (batch, nheads, headdim, dstate) -- the state after token t-1
        D: (nheads,) or None
        coeff: (batch, nheads) state-input coefficient; defaults to ``dt_t`` (Euler).
        extra_terms: additional (coefficient, B, x) triples summed into the state.

    Returns:
        (y_t, new_state) with y_t (batch, nheads, headdim).
    """
    nheads = x_t.shape[1]
    ngroups = B_t.shape[1]
    rep = _heads_per_group(nheads, ngroups)

    decay = torch.exp(dt_t * A).unsqueeze(-1).unsqueeze(-1)  # (batch, nheads, 1, 1)
    C_h = C_t.repeat_interleave(rep, dim=1)

    terms = [(dt_t if coeff is None else coeff, B_t, x_t)] + list(extra_terms or [])
    new_state = decay * state
    for co, bb, xx in terms:
        b_h = bb.repeat_interleave(_heads_per_group(nheads, bb.shape[1]), dim=1)
        new_state = new_state + (co.unsqueeze(-1).unsqueeze(-1) * xx.unsqueeze(-1) * b_h.unsqueeze(-2))
    y_t = torch.einsum("bhpn,bhn->bhp", new_state, C_h)
    if D is not None:
        y_t = y_t + D.view(1, -1, 1) * x_t
    return y_t, new_state


def ssd_sequential_reference(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    doc_ids: torch.Tensor | None = None,
    coeff: torch.Tensor | None = None,
    extra_terms: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
    dtype: torch.dtype = torch.float64,
    return_final_state: bool = False,
):
    """Ground-truth SSD scan: a plain loop, in float64 by default.

    Args:
        doc_ids: optional (batch, seqlen) document ids; the state is reset wherever the id changes.
        coeff: state-input coefficient, defaulting to ``dt``.
        extra_terms: additional ``(coefficient, B, x)`` triples summed into the state.
        return_final_state: also return the state after the last token.

    Returns:
        (batch, seqlen, nheads, headdim), or ``(y, final_state)`` with ``return_final_state``.
    """
    x, dt, A, B, C = (t.to(dtype) for t in (x, dt, A, B, C))
    D = None if D is None else D.to(dtype)
    terms: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = [
        ((dt if coeff is None else coeff.to(dtype)), B, x)
    ] + [(co.to(dtype), bb.to(dtype), xx.to(dtype)) for co, bb, xx in (extra_terms or [])]
    batch, seqlen, nheads, headdim = x.shape
    dstate = B.shape[-1]
    rep = _heads_per_group(nheads, B.shape[-2])

    state = torch.zeros(batch, nheads, headdim, dstate, dtype=dtype, device=x.device)
    ys = []
    for t in range(seqlen):
        if doc_ids is not None and t > 0:
            # Zero the carried state for any row whose document changed at this position.
            new_doc = (doc_ids[:, t] != doc_ids[:, t - 1]).view(batch, 1, 1, 1)
            state = torch.where(new_doc, torch.zeros_like(state), state)
        state = torch.exp(dt[:, t] * A).view(batch, nheads, 1, 1) * state
        for co, bb, xx in terms:
            B_h = bb[:, t].repeat_interleave(rep, dim=1)  # (batch, nheads, dstate)
            state = state + (
                co[:, t].unsqueeze(-1).unsqueeze(-1) * xx[:, t].unsqueeze(-1) * B_h.unsqueeze(-2)
            )
        C_h = C[:, t].repeat_interleave(rep, dim=1)
        y_t = torch.einsum("bhpn,bhn->bhp", state, C_h)
        if D is not None:
            y_t = y_t + D.view(1, -1, 1) * x[:, t]
        ys.append(y_t)
    out = torch.stack(ys, dim=1)
    return (out, state) if return_final_state else out
