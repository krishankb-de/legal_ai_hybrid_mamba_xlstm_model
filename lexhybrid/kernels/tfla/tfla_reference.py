"""float64 sequential oracle for the recurrence ``tfla_forward_parallel`` computes.

Moved out of the reference test suite (``tests/test_mamba3_numerics.py::sequential_mlstm_fp64``)
into the package so kernels, tests and the P2 boundary-masking work share one definition.

    C_t[d,e] = f_t[d] C_{t-1}[d,e] + i_t[d] k_t[d] v_t[e]
    n_t[d]   = f_t[d] n_{t-1}[d]   + i_t[d] k_t[d]
    y_t[e]   = (sum_d q_t[d] C_t[d,e]) / max(sum_d q_t[d] n_t[d], 1)

``f`` is clamped at 1e-6 exactly as the chunked path does, so a comparison isolates the chunked
algorithm rather than that clamp. The denominator clamp is signed (not ``abs``), matching TFLA.
"""

import torch


def sequential_mlstm_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i_gate: torch.Tensor,
    f_gate: torch.Tensor,
    dtype: torch.dtype = torch.float64,
    doc_ids: torch.Tensor | None = None,
    return_state: bool = False,
):
    """Plain-loop mLSTM recurrence; inputs (B, H, L, D), output (B, H, L, D) in ``dtype``.

    Args:
        doc_ids: optional (B, L) document ids; ``C`` and ``n`` restart at zero wherever the id
            changes, exactly as running each document on its own.
        return_state: also return the final ``(C, n)``.
    """
    q, k, v, i_gate = (t.to(dtype) for t in (q, k, v, i_gate))
    f = f_gate.to(dtype).clamp(min=1e-6)
    batch, heads, seq_len, dim = q.shape

    C = torch.zeros(batch, heads, dim, dim, dtype=dtype, device=q.device)
    n = torch.zeros(batch, heads, dim, dtype=dtype, device=q.device)
    ys = []
    for t in range(seq_len):
        if doc_ids is not None and t > 0:
            new_doc = (doc_ids[:, t] != doc_ids[:, t - 1]).view(batch, 1, 1)
            C = torch.where(new_doc.unsqueeze(-1), torch.zeros_like(C), C)
            n = torch.where(new_doc, torch.zeros_like(n), n)
        ki = k[:, :, t] * i_gate[:, :, t]
        C = f[:, :, t].unsqueeze(-1) * C + torch.einsum("bhd,bhe->bhde", ki, v[:, :, t])
        n = f[:, :, t] * n + ki
        num = torch.einsum("bhd,bhde->bhe", q[:, :, t], C)
        den = torch.einsum("bhd,bhd->bh", q[:, :, t], n).unsqueeze(-1).clamp(min=1.0)
        ys.append(num / den)
    out = torch.stack(ys, dim=2)
    return (out, C, n) if return_state else out
