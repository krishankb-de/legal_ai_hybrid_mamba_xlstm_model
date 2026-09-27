"""Mamba-1 selective scan (legacy incumbent; used only by the ``hybrid_legal_legacy`` arm).

Three operators (reference M1 / Phase 14C):
- ``selective_scan_parallel`` -- ``scan_impl="legacy"``: the chunk-parallel scan every reference
  checkpoint was trained with. It divides by ``A_cum.clamp(min=1e-8)`` and so does not compute the
  specified recurrence where the clamp fires (rel-max-err 0.92 at the reference's init Delta of
  0.705). Kept byte-identical for the legacy ablation.
- ``selective_scan_exact`` -- ``scan_impl="exact"``: division-free chunked scan, exact to fp32.
- ``selective_scan_sequential_reference`` -- float64 O(L) oracle, measurement only.

There is no Triton path: everything is plain PyTorch in fp32, and backward is autograd.
"""

import os

import torch
import torch.nn.functional as F

# No Triton kernel exists. The reference's `scan_triton.py` was imported and never dispatched,
# then deleted; this flag keeps that fact visible to anything that checks for it.
TRITON_AVAILABLE = False


def selective_scan_parallel(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Legacy chunk-parallel selective scan (divide-and-clamp; see module docstring).

    Args:
        x: (B, L, D) input; dt: (B, L, D) positive steps; A: (D, N) negative transition;
        B, C: (B, L, N); D: (D,) skip; chunk_size: chunk length.

    Returns:
        (B, L, D)
    """
    batch, seq_len, dim = x.shape
    _, _, state_size = B.shape
    device = x.device
    dtype = x.dtype

    pad_len = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad_len > 0:
        x = F.pad(x, (0, 0, 0, pad_len))
        dt = F.pad(dt, (0, 0, 0, pad_len))
        B = F.pad(B, (0, 0, 0, pad_len))
        C = F.pad(C, (0, 0, 0, pad_len))

    L = x.shape[1]
    num_chunks = L // chunk_size

    x_c = x.reshape(batch, num_chunks, chunk_size, dim)
    dt_c = dt.reshape(batch, num_chunks, chunk_size, dim)
    B_c = B.reshape(batch, num_chunks, chunk_size, state_size)
    C_c = C.reshape(batch, num_chunks, chunk_size, state_size)

    # dA[t] = dt[t] * A, the per-step log-decay; dA: (B, nc, cs, D, N). (The reference also built
    # exp(dA) here and never used it; dropping it changes no output.)
    dA = dt_c.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    dB = dt_c.unsqueeze(-1) * B_c.unsqueeze(-2)  # (B, nc, cs, D, N)
    Bx = dB * x_c.unsqueeze(-1)  # (B, nc, cs, D, N)

    log_A_cum = torch.cumsum(dA, dim=2)
    A_cum = torch.exp(log_A_cum)

    h = torch.zeros(batch, dim, state_size, device=device, dtype=dtype)
    all_outputs = []
    for ci in range(num_chunks):
        A_cum_ci = A_cum[:, ci]  # (B, cs, D, N)
        Bx_ci = Bx[:, ci]
        C_ci = C_c[:, ci]
        x_ci = x_c[:, ci]

        h_prev_contribution = A_cum_ci * h.unsqueeze(1)
        # The defect: intra[t] = A_cum[t] * cumsum(Bx / A_cum.clamp(1e-8))[t].
        A_cum_safe = A_cum_ci.clamp(min=1e-8)
        Bx_weighted = Bx_ci / A_cum_safe
        Bx_cum = torch.cumsum(Bx_weighted, dim=1)
        h_intra = A_cum_ci * Bx_cum

        h_chunk = h_prev_contribution + h_intra  # (B, cs, D, N)
        y_ci = torch.einsum("btdn, btn -> btd", h_chunk, C_ci)
        y_ci = y_ci + D.unsqueeze(0).unsqueeze(0) * x_ci
        all_outputs.append(y_ci)
        h = h_chunk[:, -1, :, :]

    output = torch.cat(all_outputs, dim=1)
    if pad_len > 0:
        output = output[:, :seq_len, :]
    return output


def selective_scan_sequential_reference(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """Exact selective scan -- the recurrence the Mamba-1 block is specified to compute.

        A_disc[t] = exp(dt[t] * A);  Bx[t] = dt[t] * B[t] * x[t]
        h[t] = A_disc[t] * h[t-1] + Bx[t],  h[-1] = 0
        y[t] = sum_n C[t,n] * h[t,:,n] + D * x[t]

    Sequential float64, for measurement only. Returns (B, L, D) in ``x``'s dtype.
    """
    in_dtype = x.dtype
    xd, dtd = x.double(), dt.double()
    Ad, Bd, Cd, Dd = A.double(), B.double(), C.double(), D.double()

    batch, seq_len, dim = xd.shape
    h = torch.zeros(batch, dim, Ad.shape[1], dtype=torch.float64, device=xd.device)
    outputs = []
    for t in range(seq_len):
        a_disc = torch.exp(dtd[:, t].unsqueeze(-1) * Ad.unsqueeze(0))
        bx = dtd[:, t].unsqueeze(-1) * Bd[:, t].unsqueeze(-2) * xd[:, t].unsqueeze(-1)
        h = a_disc * h + bx
        outputs.append(torch.einsum("bdn,bn->bd", h, Cd[:, t]) + Dd.unsqueeze(0) * xd[:, t])
    return torch.stack(outputs, dim=1).to(in_dtype)


def selective_scan_exact(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Exact chunked selective scan -- no division, no clamp (reference M1-E).

    Flips which axis is parallel: (1) scan within each chunk from a zero state, all chunks in
    parallel; (2) carry the chunk-end states across chunks; (3) add ``A_cum * carry`` back into
    every position. Depth is ``chunk_size + L/chunk_size``, minimized near ``sqrt(L)``, which the
    default picks. ``A_cum`` may underflow to zero; it only ever multiplies, never divides.

    Args/Returns: as ``selective_scan_parallel``.
    """
    batch, seq_len, dim = x.shape
    state_size = B.shape[-1]
    device, dtype = x.device, x.dtype

    if chunk_size is None:
        target = max(1.0, float(seq_len)) ** 0.5
        chunk_size = min(64, max(8, 1 << int(target).bit_length() - 1))
    chunk_size = min(chunk_size, seq_len) if seq_len > 0 else chunk_size

    pad_len = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad_len > 0:
        x = F.pad(x, (0, 0, 0, pad_len))
        dt = F.pad(dt, (0, 0, 0, pad_len))
        B = F.pad(B, (0, 0, 0, pad_len))
        C = F.pad(C, (0, 0, 0, pad_len))

    padded_len = x.shape[1]
    num_chunks = padded_len // chunk_size

    x_c = x.reshape(batch, num_chunks, chunk_size, dim)
    dt_c = dt.reshape(batch, num_chunks, chunk_size, dim)
    B_c = B.reshape(batch, num_chunks, chunk_size, state_size)
    C_c = C.reshape(batch, num_chunks, chunk_size, state_size)

    dA = dt_c.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (B, nc, cs, D, N)
    A_disc = torch.exp(dA)
    A_cum = torch.exp(torch.cumsum(dA, dim=2))
    dB = dt_c.unsqueeze(-1) * B_c.unsqueeze(-2)
    Bx = dB * x_c.unsqueeze(-1)

    # (1) intra-chunk scan from a zero state, every chunk in parallel.
    h = torch.zeros(batch, num_chunks, dim, state_size, device=device, dtype=dtype)
    h_local_steps = []
    for t in range(chunk_size):
        h = A_disc[:, :, t] * h + Bx[:, :, t]
        h_local_steps.append(h)
    h_local = torch.stack(h_local_steps, dim=2)  # (B, nc, cs, D, N)

    # (2) carry chunk-end states forward; carry_prev[c] is the state entering chunk c.
    chunk_decay = A_cum[:, :, -1]
    chunk_end = h_local[:, :, -1]
    carry = torch.zeros(batch, dim, state_size, device=device, dtype=dtype)
    carries = []
    for ci in range(num_chunks):
        carries.append(carry)
        carry = chunk_decay[:, ci] * carry + chunk_end[:, ci]
    carry_prev = torch.stack(carries, dim=1)

    # (3) h[t] = A_cum[t] * (state entering the chunk) + (chunk-local scan).
    h_full = h_local + A_cum * carry_prev.unsqueeze(2)
    y = torch.einsum("bctdn,bctn->bctd", h_full, C_c)
    y = y + D.view(1, 1, 1, -1) * x_c
    y = y.reshape(batch, padded_len, dim)
    return y[:, :seq_len] if pad_len > 0 else y


def selective_scan(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: torch.Tensor | None = None,
    scan_impl: str = "legacy",
) -> torch.Tensor:
    """Public selective-scan entry point, computed in fp32 and cast back.

    ``scan_impl="legacy"`` reproduces the reference operator bit for bit (defect included);
    ``"exact"`` selects the division-free scan. ``HYBRID_EXACT_SCAN=1`` in the environment swaps
    in the float64 sequential reference, for measurement only.
    """
    if scan_impl not in ("legacy", "exact"):
        raise ValueError(f"scan_impl must be 'legacy' or 'exact', got {scan_impl!r}")
    seq_len = x.shape[1]
    if seq_len <= 128:
        chunk_size = 32
    elif seq_len <= 2048:
        chunk_size = 64
    else:
        chunk_size = 128

    # fp32 guard: under bf16 the chunked scan's cumulative decays underflow and their gradients
    # spike; the reference Mamba keeps the scan in fp32 for exactly this reason.
    in_dtype = x.dtype
    if os.environ.get("HYBRID_EXACT_SCAN", "0") == "1":
        y = selective_scan_sequential_reference(
            x.float(), dt.float(), A.float(), B.float(), C.float(), D.float()
        ).to(in_dtype)
        return y * z if z is not None else y

    impl = selective_scan_parallel if scan_impl == "legacy" else selective_scan_exact
    y = impl(
        x.float(),
        dt.float(),
        A.float(),
        B.float(),
        C.float(),
        D.float(),
        chunk_size=chunk_size if scan_impl == "legacy" else None,
    ).to(in_dtype)
    return y * z if z is not None else y
