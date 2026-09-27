"""Chunk-parallel TFLA (tiled linear attention) for the mLSTM.

Forward is O(L/C) sequential steps with O(C^2 * D) parallel work per chunk; backward is plain
autograd through the chunked ops.

Two intra-chunk implementations (reference M1-C / M1-H):
- ``"legacy"`` absorbs the decay into the query and its reciprocal into the key and clamps the
  reciprocal's denominator at 1e-6. Where ``f_cum`` underflows that clamp (70.9% of entries at the
  shipped forget-gate init) the decay collapses to ~0 even for nearby positions: rel-max-err 0.882
  against the fp64 recurrence. Kept only for the ``hybrid_legal_legacy`` ablation arm.
- ``"exact"`` re-centres both sides on half the chunk's total log-decay, so nothing is clamped;
  when even that would overflow fp32 it falls back to a scan within the chunk.

P1 port: unchanged apart from names. Recorded for P2 (defects 9 and 15): the chunk size is
hard-wired by sequence length in ``apply_tfla``; the ``m_state`` block is computed and never
used; at chunk 128 with forget-gate bias 0 the exact path starts on the sequential fallback;
document boundaries are handled by the block's per-segment loop rather than inside the kernel.
"""

import torch
import torch.nn.functional as F

# fp32 overflows at exp(~88). 40 leaves ~20 orders of magnitude of headroom for the q/k
# magnitudes that multiply these factors before the einsum accumulates them.
_EXP_SAFE = 40.0


def _intra_chunk_sequential(q_c, k_gated, v_c, f_c):
    """Exact intra-chunk term: sequential over positions, parallel over every chunk at once.

    Used only when re-centring the decay would overflow fp32 (see ``tfla_forward_parallel``). Runs
    the mLSTM recurrence from a zero state inside each chunk -- exactly what the intra-chunk term
    is -- so it forms no exponential of a large number and is exact for any gate values.

    Returns (num, den) shaped (B, H, nc, C, D) and (B, H, nc, C, 1).
    """
    batch, heads, num_chunks, chunk_size, head_dim = q_c.shape
    C_state = torch.zeros(batch, heads, num_chunks, head_dim, head_dim, device=q_c.device, dtype=q_c.dtype)
    n_state = torch.zeros(batch, heads, num_chunks, head_dim, device=q_c.device, dtype=q_c.dtype)
    nums, dens = [], []
    for t in range(chunk_size):
        f_t = f_c[:, :, :, t]  # (B, H, nc, D)
        ki = k_gated[:, :, :, t]  # (B, H, nc, D)
        C_state = f_t.unsqueeze(-1) * C_state + torch.einsum("bhcd, bhce -> bhcde", ki, v_c[:, :, :, t])
        n_state = f_t * n_state + ki
        q_t = q_c[:, :, :, t]
        nums.append(torch.einsum("bhcd, bhcde -> bhce", q_t, C_state))
        dens.append(torch.einsum("bhcd, bhcd -> bhc", q_t, n_state))
    return torch.stack(nums, dim=3), torch.stack(dens, dim=3).unsqueeze(-1)


def tfla_forward_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i_gate: torch.Tensor,
    f_gate: torch.Tensor,
    chunk_size: int = 64,
    tfla_impl: str = "legacy",
) -> torch.Tensor:
    """Chunk-parallel TFLA.

    Args:
        q, k, v: (B, H, L, D)
        i_gate: input gates (B, H, L, D), already exponentiated
        f_gate: forget gates (B, H, L, D), already sigmoided
        chunk_size: chunk length
        tfla_impl: ``"legacy"`` | ``"exact"``

    Returns:
        (B, H, L, D)
    """
    if tfla_impl not in ("legacy", "exact"):
        raise ValueError(f"tfla_impl must be 'legacy' or 'exact', got {tfla_impl!r}")
    batch, num_heads, seq_len, head_dim = q.shape
    device = q.device
    dtype = q.dtype

    # Pad the sequence to a multiple of chunk_size; a forget gate of 1 carries the state unchanged.
    pad_len = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        i_gate = F.pad(i_gate, (0, 0, 0, pad_len))
        f_gate = F.pad(f_gate, (0, 0, 0, pad_len), value=1.0)

    padded_len = q.shape[2]
    num_chunks = padded_len // chunk_size

    q_c = q.reshape(batch, num_heads, num_chunks, chunk_size, head_dim)
    k_c = k.reshape(batch, num_heads, num_chunks, chunk_size, head_dim)
    v_c = v.reshape(batch, num_heads, num_chunks, chunk_size, head_dim)
    i_c = i_gate.reshape(batch, num_heads, num_chunks, chunk_size, head_dim)
    f_c = f_gate.reshape(batch, num_heads, num_chunks, chunk_size, head_dim)

    # Cumulative forget-gate products within each chunk, in log space.
    log_f = torch.log(f_c.clamp(min=1e-6))
    log_f_cum = torch.cumsum(log_f, dim=3)  # (B, H, nc, C, D)
    f_cum = torch.exp(log_f_cum)

    causal_mask = torch.triu(torch.ones(chunk_size, chunk_size, device=device, dtype=torch.bool), diagonal=1)

    # ---- intra-chunk (parallel across all chunks) --------------------------------------------
    # decay[i,j,d] = f_cum[i,d] / f_cum[j,d] (i >= j), absorbed into the query/key scaling.
    k_gated_intra = k_c * i_c  # per-dimension input gating
    if tfla_impl == "legacy":
        # M1-C: `f_cum` underflows 1e-6 within a chunk for any realistic forget gate, and the
        # clamp then pins the denominator, so decay[i, j] collapses to ~0 even for j close to i.
        f_cum_safe = f_cum.clamp(min=1e-6)
        q_weighted = q_c * f_cum
        k_weighted = k_gated_intra / f_cum_safe
    else:
        # Same factorization, no reciprocal: decay[i, j] = exp(log_f_cum[i] - m) * exp(m - log_f_cum[j])
        # for any m; m = log_f_cum[-1]/2 splits the range evenly, so nothing is clamped.
        m = 0.5 * log_f_cum[:, :, :, -1:, :]  # (B, H, nc, 1, D)
        half_range = 0.5 * log_f_cum[:, :, :, -1, :].abs().max()
        if half_range < _EXP_SAFE:
            q_weighted = q_c * torch.exp(log_f_cum - m)
            k_weighted = k_gated_intra * torch.exp(m - log_f_cum)
        else:
            # Re-centring halves the dynamic range but cannot remove it; a chunk whose total
            # log-decay exceeds 2 * _EXP_SAFE still overflows fp32 on one side and inf * 0 = NaN.
            # Scan within the chunk instead of factorizing across it.
            q_weighted = k_weighted = None

    if q_weighted is None:
        h_intra_num, h_intra_den = _intra_chunk_sequential(q_c, k_gated_intra, v_c, f_c)
    else:
        scores = torch.einsum("bhcid, bhcjd -> bhcij", q_weighted, k_weighted)  # (B, H, nc, C, C)
        scores = scores.masked_fill(causal_mask, 0.0)
        h_intra_num = torch.einsum("bhcij, bhcjd -> bhcid", scores, v_c)  # (B, H, nc, C, D)
        h_intra_den = scores.sum(dim=-1, keepdim=True)  # (B, H, nc, C, 1)

    # ---- inter-chunk recurrence (sequential across chunks only) ------------------------------
    C_state = torch.zeros(batch, num_heads, head_dim, head_dim, device=device, dtype=dtype)
    n_state = torch.zeros(batch, num_heads, head_dim, device=device, dtype=dtype)
    # m_state: computed as in the reference and never applied to the output (defect 9, P2-C).
    m_state = torch.full((batch, num_heads, head_dim), float("-inf"), device=device, dtype=dtype)

    recurrent_num_list = []
    recurrent_den_list = []

    for ci in range(num_chunks):
        f_cum_ci = f_cum[:, :, ci]
        log_f_cum_ci = log_f_cum[:, :, ci]

        # Contribution of the carried state to this chunk.
        q_f = q_c[:, :, ci] * f_cum_ci  # (B, H, C, D)
        h_rec_num = torch.einsum("bhld, bhde -> bhle", q_f, C_state)  # (B, H, C, D)
        h_rec_den = torch.einsum("bhld, bhd -> bhl", q_f, n_state).unsqueeze(-1)  # (B, H, C, 1)
        recurrent_num_list.append(h_rec_num)
        recurrent_den_list.append(h_rec_den)

        # Update the carried state for the next chunk.
        total_f_last = f_cum_ci[:, :, -1, :]  # (B, H, D)
        C_state = total_f_last.unsqueeze(-1) * C_state
        n_state = total_f_last * n_state

        # decay_to_end[t] = prod(f[t+1..end]) = f_cum[-1] / f_cum[t], in log space.
        decay_to_end = torch.exp(log_f_cum_ci[:, :, -1:, :] - log_f_cum_ci)  # (B, H, C, D)
        k_gated_update = k_c[:, :, ci] * i_c[:, :, ci] * decay_to_end
        C_state = C_state + torch.einsum("bhld, bhle -> bhde", k_gated_update, v_c[:, :, ci])
        n_state = n_state + k_gated_update.sum(dim=2)

        log_i_ci = torch.log(i_c[:, :, ci].clamp(min=1e-30))
        log_alpha_ci = log_i_ci - log_f_cum_ci
        m_ci = log_alpha_ci.amax(dim=2)
        log_f_last = log_f_cum_ci[:, :, -1, :]
        m_state = torch.max(log_f_last + m_state, m_ci)

    h_rec_num_all = torch.stack(recurrent_num_list, dim=2)  # (B, H, nc, C, D)
    h_rec_den_all = torch.stack(recurrent_den_list, dim=2)  # (B, H, nc, C, 1)

    # Joint normalization: (intra_num + rec_num) / max(intra_den + rec_den, 1) -- a signed clamp.
    total_num = h_intra_num + h_rec_num_all
    total_den = (h_intra_den + h_rec_den_all).clamp(min=1.0)
    output = total_num / total_den
    output = output.reshape(batch, num_heads, padded_len, head_dim)

    if pad_len > 0:
        output = output[:, :, :seq_len, :]
    return output


def apply_tfla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i_gate: torch.Tensor,
    f_gate: torch.Tensor,
    tfla_impl: str = "legacy",
) -> torch.Tensor:
    """Public TFLA entry point; picks the chunk size from the sequence length (as the reference did).

    Args:
        q, k, v, i_gate, f_gate: (B, H, L, D)
        tfla_impl: ``"legacy"`` | ``"exact"``

    Returns:
        (B, H, L, D)
    """
    seq_len = q.shape[2]
    if seq_len <= 128:
        chunk_size = 32
    elif seq_len <= 2048:
        chunk_size = 64
    else:
        chunk_size = 128
    return tfla_forward_parallel(q, k, v, i_gate, f_gate, chunk_size=chunk_size, tfla_impl=tfla_impl)
