"""Chunk-parallel TFLA (tiled linear attention) for the mLSTM.

Forward is O(L/C) sequential steps with O(C^2 * D) parallel work per chunk; backward is plain
autograd through the chunked ops.

Two intra-chunk implementations (reference M1-C / M1-H):
- ``"legacy"`` absorbs the decay into the query and its reciprocal into the key and clamps the
  reciprocal's denominator at 1e-6. Where ``f_cum`` underflows that clamp (70.9% of entries at the
  reference's forget-gate init) the decay collapses to ~0 even for nearby positions: rel-max-err
  0.882 against the fp64 recurrence. Kept only for the ``hybrid_legal_legacy`` ablation arm.
- ``"exact"`` re-centres both sides on half the chunk's widest log-decay, so nothing is clamped;
  when even that would overflow fp32 it either raises or scans within the chunk (``fallback``).

P2-C (defect 9): the chunk size comes from the block (``mlstm_chunk_size``) instead of being picked
from the sequence length; the reference's ``m_state`` stabiliser, computed and never applied, is
gone; the sequential fallback is a configured choice, an error by default. At chunk 128 with
forget-gate bias 0 the exact path needed the fallback from the first step (half-range
0.5 * 128 * ln 2 = 44.4 > 40); the shipped forget bias 3.0 gives 3.1.

P2-D (defect 15): document boundaries are handled here, with the three boolean masks of
``kernels/ssd/ssd_interface.py`` -- ``same`` on the intra-chunk scores, ``carry_ok`` on the carried
numerator/denominator, ``end_ok`` on the chunk-end state -- and a cumulative log-decay that restarts
at every boundary inside a chunk. It replaces the block's per-(row, document) Python loop.
``return_state`` hands back the state after the last token for the decode cache (P2-G).
"""

import torch
import torch.nn.functional as F

from lexhybrid.kernels.segments import forward_fill_from_starts, pad_segments, segment_ids

# fp32 overflows at exp(~88). 40 leaves ~20 orders of magnitude of headroom for the q/k
# magnitudes that multiply these factors before the einsum accumulates them.
_EXP_SAFE = 40.0
TFLA_FALLBACKS = ("error", "sequential")


class TFLAFallbackError(RuntimeError):
    """The exact factorisation would overflow fp32 and ``tfla_fallback`` is ``"error"``."""


def _intra_chunk_sequential(q_c, k_gated, v_c, f_c, starts_c=None):
    """Exact intra-chunk term: sequential over positions, parallel over every chunk at once.

    Used only when re-centring the decay would overflow fp32 (see ``tfla_forward_parallel``). Runs
    the mLSTM recurrence from a zero state inside each chunk -- exactly what the intra-chunk term
    is -- so it forms no exponential of a large number and is exact for any gate values. With
    ``starts_c`` (batch, nc, C) the state also restarts where a new document begins.

    Returns (num, den) shaped (B, H, nc, C, D) and (B, H, nc, C, 1).
    """
    batch, heads, num_chunks, chunk_size, head_dim = q_c.shape
    C_state = torch.zeros(batch, heads, num_chunks, head_dim, head_dim, device=q_c.device, dtype=q_c.dtype)
    n_state = torch.zeros(batch, heads, num_chunks, head_dim, device=q_c.device, dtype=q_c.dtype)
    nums, dens = [], []
    for t in range(chunk_size):
        f_t = f_c[:, :, :, t]  # (B, H, nc, D)
        if starts_c is not None:
            f_t = f_t * (~starts_c[:, None, :, t, None]).to(f_t.dtype)  # a new document forgets all
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
    fallback: str = "error",
    doc_ids: torch.Tensor | None = None,
    return_state: bool = False,
):
    """Chunk-parallel TFLA.

    Args:
        q, k, v: (B, H, L, D)
        i_gate: input gates (B, H, L, D), already exponentiated
        f_gate: forget gates (B, H, L, D), already sigmoided
        chunk_size: chunk length
        tfla_impl: ``"legacy"`` | ``"exact"``
        fallback: what ``exact`` does when a chunk's log-decay is too wide to factorise in fp32:
            ``"error"`` raises ``TFLAFallbackError``; ``"sequential"`` scans within the chunk
            (exact, and ``chunk_size`` times slower for the intra-chunk term).
        doc_ids: optional (B, L) document ids; the recurrence restarts where the id changes.
        return_state: also return ``(C, n)`` after the last token, shaped (B, H, D, D) and
            (B, H, D) -- the state ``mLSTMBlock.step`` continues from.

    Returns:
        (B, H, L, D), or ``(h, C, n)`` with ``return_state``.
    """
    if tfla_impl not in ("legacy", "exact"):
        raise ValueError(f"tfla_impl must be 'legacy' or 'exact', got {tfla_impl!r}")
    if fallback not in TFLA_FALLBACKS:
        raise ValueError(f"fallback must be one of {TFLA_FALLBACKS}, got {fallback!r}")
    batch, num_heads, seq_len, head_dim = q.shape
    device = q.device
    docs = doc_ids is not None

    # Pad the sequence to a multiple of chunk_size; a forget gate of 1 carries the state unchanged
    # and a zero input gate adds nothing, and the padding joins the last real document.
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

    if docs:
        seg = pad_segments(segment_ids(doc_ids, batch, seq_len, device), pad_len)
        seg_c = seg.reshape(batch, num_chunks, chunk_size)
        starts_c = torch.zeros_like(seg_c, dtype=torch.bool)
        starts_c[..., 1:] = seg_c[..., 1:] != seg_c[..., :-1]
        # Restart the cumulative log-decay at every boundary inside a chunk. Every product the
        # kernel forms is either within one document (where the baseline cancels) or masked, so
        # this changes no value -- it only keeps the factorised range per document, not per chunk.
        baseline = torch.where(starts_c[:, None, :, :, None], log_f_cum - log_f, torch.zeros_like(log_f))
        log_f_cum = log_f_cum - forward_fill_from_starts(
            baseline, starts_c[:, None].expand(-1, num_heads, -1, -1), 3
        )
        # The three masks of ssd_interface.py.
        same = seg_c.unsqueeze(-1) == seg_c.unsqueeze(-2)  # (B, nc, C, C): j may influence i
        prev_seg = torch.full((batch, num_chunks), -1, dtype=seg_c.dtype, device=device)
        if num_chunks > 1:
            prev_seg[:, 1:] = seg_c[:, :-1, -1]
        carry_ok = seg_c == prev_seg.unsqueeze(-1)  # (B, nc, C): the carried state is valid at i
        end_ok = seg_c == seg_c[:, :, -1:]  # (B, nc, C): t's contribution reaches the chunk end
    else:
        starts_c = same = carry_ok = end_ok = None

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
        # Same factorisation, no reciprocal: decay[i, j] = exp(L[i] - m) * exp(m - L[j]) for any m.
        # m = half the chunk's lowest cumulative log-decay splits the range evenly; without
        # documents that is half the last value, since the cumulative sum only decreases.
        low = log_f_cum.amin(dim=3, keepdim=True) if docs else log_f_cum[:, :, :, -1:, :]
        m = 0.5 * low  # (B, H, nc, 1, D)
        half_range = 0.5 * low.abs().max()
        if half_range < _EXP_SAFE:
            q_weighted = q_c * torch.exp(log_f_cum - m)
            k_weighted = k_gated_intra * torch.exp(m - log_f_cum)
        elif fallback == "error":
            raise TFLAFallbackError(
                f"exact TFLA at chunk {chunk_size}: a chunk's log-decay half-range is "
                f"{float(half_range):.1f} >= {_EXP_SAFE} and would overflow fp32. Keep the factorised "
                "path by configuration (mlstm_forget_gate_bias_init 3.0 gives 3.1 at chunk 128 at "
                "init; a smaller mlstm_chunk_size halves the range), or set tfla_fallback: sequential "
                "(exact, slow)."
            )
        else:
            # Re-centring halves the dynamic range but cannot remove it; a chunk whose log-decay
            # exceeds 2 * _EXP_SAFE still overflows fp32 on one side and inf * 0 = NaN. Scan
            # within the chunk instead of factorising across it.
            q_weighted = k_weighted = None

    if q_weighted is None:
        h_intra_num, h_intra_den = _intra_chunk_sequential(q_c, k_gated_intra, v_c, f_c, starts_c)
    else:
        scores = torch.einsum("bhcid, bhcjd -> bhcij", q_weighted, k_weighted)  # (B, H, nc, C, C)
        scores = scores.masked_fill(causal_mask, 0.0)
        if docs:
            scores = scores.masked_fill(~same.unsqueeze(1), 0.0)
        h_intra_num = torch.einsum("bhcij, bhcjd -> bhcid", scores, v_c)  # (B, H, nc, C, D)
        h_intra_den = scores.sum(dim=-1, keepdim=True)  # (B, H, nc, C, 1)

    # ---- inter-chunk recurrence (sequential across chunks only) ------------------------------
    # The carried state is fp32 whatever the activation dtype (P2-E, as in ssd_interface.py): it
    # accumulates over the whole row. It is read in the activation dtype for each einsum, because
    # einsum does not promote, and every update is added in fp32.
    C_state = torch.zeros(batch, num_heads, head_dim, head_dim, device=device, dtype=torch.float32)
    n_state = torch.zeros(batch, num_heads, head_dim, device=device, dtype=torch.float32)

    recurrent_num_list = []
    recurrent_den_list = []

    for ci in range(num_chunks):
        f_cum_ci = f_cum[:, :, ci]
        log_f_cum_ci = log_f_cum[:, :, ci]

        # Contribution of the carried state to this chunk (decay from the chunk start).
        q_f = q_c[:, :, ci] * f_cum_ci  # (B, H, C, D)
        h_rec_num = torch.einsum("bhld, bhde -> bhle", q_f, C_state.to(q_f.dtype))  # (B, H, C, D)
        h_rec_den = torch.einsum("bhld, bhd -> bhl", q_f, n_state.to(q_f.dtype)).unsqueeze(-1)
        if docs:
            ok = carry_ok[:, ci][:, None, :, None].to(h_rec_num.dtype)
            h_rec_num = h_rec_num * ok
            h_rec_den = h_rec_den * ok
        recurrent_num_list.append(h_rec_num)
        recurrent_den_list.append(h_rec_den)

        # Update the carried state for the next chunk: it survives only if no document starts in
        # this chunk, and only this chunk's last document contributes to it.
        total_f_last = f_cum_ci[:, :, -1, :]  # (B, H, D)
        if docs:
            total_f_last = total_f_last * carry_ok[:, ci, -1][:, None, None].to(total_f_last.dtype)
        C_state = total_f_last.float().unsqueeze(-1) * C_state
        n_state = total_f_last.float() * n_state

        # decay_to_end[t] = prod(f[t+1..end]) = f_cum[-1] / f_cum[t], in log space. Masked in log
        # space before the exponential: across documents the difference can be large and positive.
        log_to_end = log_f_cum_ci[:, :, -1:, :] - log_f_cum_ci
        if docs:
            log_to_end = log_to_end.masked_fill(~end_ok[:, ci][:, None, :, None], float("-inf"))
        decay_to_end = torch.exp(log_to_end)  # (B, H, C, D)
        k_gated_update = k_c[:, :, ci] * i_c[:, :, ci] * decay_to_end
        C_state = C_state + torch.einsum("bhld, bhle -> bhde", k_gated_update, v_c[:, :, ci]).float()
        n_state = n_state + k_gated_update.sum(dim=2).float()

    h_rec_num_all = torch.stack(recurrent_num_list, dim=2)  # (B, H, nc, C, D)
    h_rec_den_all = torch.stack(recurrent_den_list, dim=2)  # (B, H, nc, C, 1)

    # Joint normalisation: (intra_num + rec_num) / max(intra_den + rec_den, 1) -- a signed clamp.
    total_num = h_intra_num + h_rec_num_all
    total_den = (h_intra_den + h_rec_den_all).clamp(min=1.0)
    output = total_num / total_den
    output = output.reshape(batch, num_heads, padded_len, head_dim)

    if pad_len > 0:
        output = output[:, :, :seq_len, :]
    if return_state:
        return output, C_state, n_state
    return output


def apply_tfla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i_gate: torch.Tensor,
    f_gate: torch.Tensor,
    tfla_impl: str = "legacy",
    chunk_size: int = 128,
    fallback: str = "error",
    doc_ids: torch.Tensor | None = None,
    return_state: bool = False,
):
    """Public TFLA entry point, with the chunk size the block was configured with.

    The reference chose 32/64/128 from the sequence length here, so the operator a model trained
    with depended on its row length. A sequence shorter than one chunk runs as one chunk of its
    own length, which is identical to padding it with forget gates of 1.

    Args:
        q, k, v, i_gate, f_gate: (B, H, L, D)
        tfla_impl: ``"legacy"`` | ``"exact"``
        chunk_size: the block's ``chunk_size`` (config ``mlstm_chunk_size``)
        fallback: ``"error"`` | ``"sequential"`` (config ``tfla_fallback``)
        doc_ids: optional (B, L) document ids
        return_state: also return the final ``(C, n)``

    Returns:
        (B, H, L, D), or ``(h, C, n)`` with ``return_state``.
    """
    chunk = max(1, min(int(chunk_size), q.shape[2]))
    return tfla_forward_parallel(
        q,
        k,
        v,
        i_gate,
        f_gate,
        chunk_size=chunk,
        tfla_impl=tfla_impl,
        fallback=fallback,
        doc_ids=doc_ids,
        return_state=return_state,
    )
