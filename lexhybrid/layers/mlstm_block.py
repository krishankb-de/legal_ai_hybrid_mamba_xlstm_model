"""mLSTM (matrix LSTM) mixer from xLSTM, with exponential input gating and chunked TFLA.

P1 port of the reference block with two removals: ``use_tfla``/``_slow_forward`` (the sequential
path computed a *different* function from TFLA -- LSE-stabilised with an ``abs`` clamp, 0.42 max
abs apart at L=24 -- and no checkpoint ever trained on it), and the sLSTM sibling. Everything else
is unchanged, including three recorded defects fixed in P2: the ``mlstm_*`` config keys never reach
this block (defect 2), the gate-bias init is zeroed by the model's weight pass (defect 3), and
document boundaries are handled by a per-(row, segment) Python loop (defect 15).
"""

import torch
import torch.nn as nn
from einops import rearrange

from lexhybrid.kernels.tfla import apply_tfla
from lexhybrid.layers.activations import exponential_activation
from lexhybrid.layers.normalization import RMSNorm


def _tanh_soft_cap(x: torch.Tensor, cap: float) -> torch.Tensor:
    """Soft cap via tanh: maps x to ``tanh(x / cap) * cap``, range (-cap, cap)."""
    return torch.tanh(x / cap) * cap


class mLSTMBlock(nn.Module):
    """mLSTM mixer: per-dimension exponential input gate, sigmoid forget gate, matrix memory.

    Args:
        dim: model dimension.
        head_dim: dimension per head.
        num_heads: number of heads (``max(1, dim // head_dim)`` when None).
        tfla_impl: ``"legacy"`` | ``"exact"`` TFLA intra-chunk operator.
        proj_factor: input projection factor (2 splits into the mLSTM input and a gate).
        gate_soft_cap: tanh soft cap on the raw i/f gate pre-activations.
        input_gate_bias_init: initial bias of ``i_gate_proj`` (negative keeps early gates small).
        forget_gate_bias_init: initial bias of ``f_gate_proj``.
        use_hybrid_norm: RMSNorm the values as well as the queries and keys.
    """

    # Declared capability read by `HybridBlock`; `_forward_segmented` re-runs the block per document.
    supports_doc_ids = True

    def __init__(
        self,
        dim: int,
        head_dim: int = 64,
        num_heads: int | None = None,
        tfla_impl: str = "legacy",
        proj_factor: int = 2,
        gate_soft_cap: float = 15.0,
        input_gate_bias_init: float = -10.0,
        forget_gate_bias_init: float = 0.0,
        use_hybrid_norm: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.gate_soft_cap = gate_soft_cap
        self.use_hybrid_norm = use_hybrid_norm
        self.num_heads = max(1, dim // head_dim) if num_heads is None else num_heads
        self.inner_dim = self.num_heads * head_dim

        # "legacy" keeps the reference numerics (defect included); "exact" removes the reciprocal.
        if tfla_impl not in ("legacy", "exact"):
            raise ValueError(f"tfla_impl must be 'legacy' or 'exact', got {tfla_impl!r}")
        self.tfla_impl = tfla_impl
        self.proj_factor = proj_factor

        self.in_proj = nn.Linear(dim, self.inner_dim * proj_factor, bias=False)
        self.q_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=False)

        # Gates are per dimension, not one scalar per head.
        self.i_gate_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=True)
        self.f_gate_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=True)
        self.o_gate_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=True)

        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.v_norm = RMSNorm(head_dim) if use_hybrid_norm else None

        self.out_proj = nn.Linear(self.inner_dim, dim, bias=False)

        # Unused by the computation; kept because it is a persistent buffer in the reference state
        # dict, and the P1 parity fixtures load strictly. P2 may drop it with a fixture note.
        self.register_buffer("eps", torch.tensor(1e-6))

        # Negative input-gate bias keeps early gates small (prevents exp overflow).
        nn.init.constant_(self.i_gate_proj.bias, input_gate_bias_init)
        nn.init.constant_(self.f_gate_proj.bias, forget_gate_bias_init)

    def _gates(self, x_inner: torch.Tensor):
        """Soft-capped gates: exp input gate (computed in fp32), sigmoid forget and output gates."""
        i_logit = _tanh_soft_cap(self.i_gate_proj(x_inner), self.gate_soft_cap)
        f_logit = _tanh_soft_cap(self.f_gate_proj(x_inner), self.gate_soft_cap)
        # exp of the soft-capped logit reaches exp(15) ~ 3.3M; in bf16 that value and its gradient
        # lose precision and spike, so exp is taken in fp32 and cast back.
        gate_dtype = i_logit.dtype
        i_gate = exponential_activation(i_logit.float()).to(gate_dtype)
        f_gate = torch.sigmoid(f_logit)
        o_gate = torch.sigmoid(self.o_gate_proj(x_inner))
        return i_gate, f_gate, o_gate

    def forward(
        self,
        x: torch.Tensor,
        cache: dict | None = None,
        doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """(batch, seq_len, dim) -> (batch, seq_len, dim).

        Args:
            x: input.
            cache: accepted for signature parity; the chunked forward does not write it (P2-G).
            doc_ids: optional (batch, seq_len) per-position document ids. When given, each row is
                split into per-document segments so the matrix memory does not cross a boundary.
        """
        if doc_ids is not None:
            return self._forward_segmented(x, doc_ids)

        x_proj = self.in_proj(x)
        if self.proj_factor == 2:
            x_inner, x_gate = x_proj.chunk(2, dim=-1)
        else:
            x_inner = x_proj
            x_gate = x_inner

        q = rearrange(self.q_proj(x_inner), "b l (h d) -> b h l d", h=self.num_heads)
        k = rearrange(self.k_proj(x_inner), "b l (h d) -> b h l d", h=self.num_heads)
        v = rearrange(self.v_proj(x_inner), "b l (h d) -> b h l d", h=self.num_heads)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.v_norm is not None:
            v = self.v_norm(v)

        i_gate, f_gate, o_gate = self._gates(x_inner)
        i_gate = rearrange(i_gate, "b l (h d) -> b h l d", h=self.num_heads)
        f_gate = rearrange(f_gate, "b l (h d) -> b h l d", h=self.num_heads)
        o_gate = rearrange(o_gate, "b l (h d) -> b h l d", h=self.num_heads)

        h = apply_tfla(q, k, v, i_gate, f_gate, tfla_impl=self.tfla_impl)
        h = h * o_gate
        h = rearrange(h, "b h l d -> b l (h d)")
        h = h * torch.sigmoid(x_gate)
        return self.out_proj(h)

    # -- O(1) recurrent decode (reference M6-B) ---------------------------------------------------
    supports_step = True

    def allocate_inference_cache(self, batch_size, device=None, dtype=torch.float32):
        """mLSTM decode state: the matrix cell ``C`` and the normalizer ``n``, both zeros.

        No ``m``: TFLA computes an ``m_state`` and never applies it, so the recurrence this cache
        continues has no stabilizer to carry.
        """
        device = device or self.out_proj.weight.device
        H, D = self.num_heads, self.head_dim
        return {
            "C": torch.zeros(batch_size, H, D, D, device=device, dtype=dtype),
            "n": torch.zeros(batch_size, H, D, device=device, dtype=dtype),
            "seen": 0,
        }

    def step(self, x_t: torch.Tensor, cache: dict) -> torch.Tensor:
        """Advance one token. ``(batch, dim)`` or ``(batch, 1, dim)`` in, ``(batch, dim)`` out.

        Reproduces the recurrence TFLA computes (the one every checkpoint trained with):

            C_t[d,e] = f_t[d] C_{t-1}[d,e] + i_t[d] k_t[d] v_t[e]
            n_t[d]   = f_t[d] n_{t-1}[d]   + i_t[d] k_t[d]
            y_t[e]   = (sum_d q_t[d] C_t[d,e]) / max(sum_d q_t[d] n_t[d], 1)

        The denominator clamp is signed, matching TFLA's joint normalization. Only the ``exact``
        TFLA equals this recurrence; the ``legacy`` kernel computes no recurrence a step could
        reproduce.
        """
        if x_t.dim() == 3:
            if x_t.shape[1] != 1:
                raise ValueError(f"step() takes one token, got seqlen {x_t.shape[1]}")
            x_t = x_t[:, 0]
        H = self.num_heads

        x_proj = self.in_proj(x_t)
        if self.proj_factor == 2:
            x_inner, x_gate = x_proj.chunk(2, dim=-1)
        else:
            x_inner = x_proj
            x_gate = x_inner

        def heads(t: torch.Tensor) -> torch.Tensor:
            return rearrange(t, "b (h d) -> b h d", h=H)

        q = self.q_norm(heads(self.q_proj(x_inner)))
        k = self.k_norm(heads(self.k_proj(x_inner)))
        v = heads(self.v_proj(x_inner))
        if self.v_norm is not None:
            v = self.v_norm(v)

        i_gate, f_gate, o_gate = (heads(g) for g in self._gates(x_inner))

        # f is clamped at 1e-6 exactly as TFLA does when it takes log f.
        f_eff = f_gate.clamp(min=1e-6)
        ki = i_gate * k

        # The state is cast to the activation dtype here (defect 16, P2-E keeps it fp32).
        C = f_eff.unsqueeze(-1) * cache["C"].to(v.dtype) + torch.einsum("bhd,bhe->bhde", ki, v)
        n = f_eff * cache["n"].to(v.dtype) + ki
        h = torch.einsum("bhde,bhd->bhe", C, q)
        denom = (n * q).sum(dim=-1, keepdim=True).clamp(min=1.0)
        h = h / denom

        cache["C"], cache["n"] = C, n
        cache["seen"] += 1

        h = rearrange(h * o_gate, "b h d -> b (h d)")
        return self.out_proj(h * torch.sigmoid(x_gate))

    def _forward_segmented(self, x: torch.Tensor, doc_ids: torch.Tensor) -> torch.Tensor:
        """Per-(row, segment) wrapper that resets the mLSTM state at document boundaries.

        A Python loop with a host sync per row (defect 15); P2-D moves the reset inside TFLA.
        """
        B, L, _ = x.shape
        out_parts = []
        for b in range(B):
            ids = doc_ids[b]
            change = torch.ones(L, dtype=torch.bool, device=ids.device)
            change[1:] = ids[1:] != ids[:-1]
            starts = torch.nonzero(change, as_tuple=False).flatten().tolist()
            starts.append(L)
            row_pieces = []
            for i in range(len(starts) - 1):
                s, e = starts[i], starts[i + 1]
                row_pieces.append(self.forward(x[b : b + 1, s:e, :]))
            out_parts.append(torch.cat(row_pieces, dim=1))
        return torch.cat(out_parts, dim=0)
