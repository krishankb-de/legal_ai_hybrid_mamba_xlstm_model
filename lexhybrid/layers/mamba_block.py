"""Mamba-1 mixer with the selective scan -- the legacy incumbent.

Only the ``hybrid_legal_legacy`` ablation arm uses this mixer (decision 17): it shows that the
operator correction still matters on legal text. It has no ``step()``, so a model containing it
cannot use the cached decode path, and its document-boundary handling is a per-(row, segment)
Python loop. Ported unchanged from the reference apart from the ``doc_ids`` name.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from lexhybrid.kernels.selective_scan import selective_scan
from lexhybrid.kernels.selective_scan.scan_interface import selective_scan_exact, selective_scan_parallel
from lexhybrid.layers.normalization import RMSNorm


class MambaBlock(nn.Module):
    """Mamba-1 mixer: input projection, depthwise causal conv, selective SSM, gated output.

    Args:
        dim: model dimension.
        state_size: SSM state size N.
        conv_size: depthwise convolution width.
        expand_factor: inner-dimension expansion.
        dt_rank: rank of the dt projection (``dim // 16`` when None).
        use_fast_path: ``selective_scan`` (adaptive chunk) versus ``_slow_forward`` (chunk
            ``min(64, L)``); both share one implementation.
        use_hybrid_norm: B/C RMSNorms (and the Delta norm unless ``use_dt_norm`` is False).
        scan_impl: ``"legacy"`` (reference numerics, defect included) | ``"exact"``.
        dt_init_strategy: ``"none"`` | ``"mamba"`` (reference logU[dt_min, dt_max] init).
        dt_min, dt_max: Delta init range for ``dt_init_strategy="mamba"``.
        use_dt_norm: whether to RMSNorm Delta (defaults to ``use_hybrid_norm``).
    """

    # Declared capability read by `HybridBlock`; `_forward_segmented` re-runs the block per document.
    supports_doc_ids = True

    def __init__(
        self,
        dim: int,
        state_size: int = 16,
        conv_size: int = 4,
        expand_factor: int = 2,
        dt_rank: int | None = None,
        use_fast_path: bool = True,
        use_hybrid_norm: bool = False,
        scan_impl: str = "legacy",
        dt_init_strategy: str = "none",
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        use_dt_norm: bool | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.state_size = state_size
        self.conv_size = conv_size
        self.expand_factor = expand_factor
        if scan_impl not in ("legacy", "exact"):
            raise ValueError(f"scan_impl must be 'legacy' or 'exact', got {scan_impl!r}")
        self.scan_impl = scan_impl
        # M1-F: the reference Mamba dt init only takes effect if (1) it is re-applied after the
        # model's weight pass zeroes every bias (post_model_init) and (2) the Delta RMSNorm is off,
        # because that norm rescales Delta to unit RMS and discards the bias offset.
        if dt_init_strategy not in ("none", "mamba"):
            raise ValueError(f"dt_init_strategy must be 'none' or 'mamba', got {dt_init_strategy!r}")
        self.dt_init_strategy = dt_init_strategy
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.inner_dim = dim * expand_factor
        self.use_fast_path = use_fast_path
        self.use_hybrid_norm = use_hybrid_norm
        self.dt_rank = max(1, dim // 16) if dt_rank is None else dt_rank

        self.in_proj = nn.Linear(dim, self.inner_dim * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.inner_dim,
            out_channels=self.inner_dim,
            kernel_size=conv_size,
            padding=conv_size - 1,
            groups=self.inner_dim,
            bias=True,
        )
        self.x_proj = nn.Linear(self.inner_dim, self.dt_rank + state_size * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.inner_dim, bias=True)

        # HybridNorm: per-projection pre-norm on Delta/B/C. `hybrid_bc` sets use_dt_norm False.
        self._use_dt_norm = use_hybrid_norm if use_dt_norm is None else bool(use_dt_norm)
        if use_hybrid_norm:
            self.dt_norm = RMSNorm(self.inner_dim) if self._use_dt_norm else None
            self.B_norm = RMSNorm(state_size)
            self.C_norm = RMSNorm(state_size)
        else:
            self.dt_norm = None
            self.B_norm = None
            self.C_norm = None

        A = torch.arange(1, state_size + 1, dtype=torch.float32).repeat(self.inner_dim, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.inner_dim))
        self.out_proj = nn.Linear(self.inner_dim, dim, bias=False)
        self.activation = nn.SiLU()

    def post_model_init(self) -> None:
        """Re-apply the Mamba dt init after the model's global weight pass (no-op unless ``"mamba"``)."""
        if self.dt_init_strategy != "mamba":
            return
        with torch.no_grad():
            dt = torch.exp(
                torch.rand(self.inner_dim) * (math.log(self.dt_max) - math.log(self.dt_min))
                + math.log(self.dt_min)
            ).clamp(min=1e-4)
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(
        self,
        x: torch.Tensor,
        cache: dict | None = None,
        doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """(batch, seq_len, dim) -> (batch, seq_len, dim); ``doc_ids`` resets state per document."""
        if doc_ids is not None:
            return self._forward_segmented(x, doc_ids)
        seq_len = x.shape[1]

        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        x_conv = rearrange(x_inner, "b l d -> b d l")
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = rearrange(x_conv, "b d l -> b l d")
        x_conv = self.activation(x_conv)

        dt, B, C = torch.split(self.x_proj(x_conv), [self.dt_rank, self.state_size, self.state_size], dim=-1)
        dt = self.dt_proj(dt)
        # B/C norms are gated on their own modules: under `hybrid_bc` the Delta norm is absent
        # while the B/C norms stay.
        if self.dt_norm is not None:
            dt = self.dt_norm(dt)
        if self.B_norm is not None:
            B = self.B_norm(B)
            C = self.C_norm(C)
        dt = F.softplus(dt)

        A = -torch.exp(self.A_log.float())  # (inner_dim, N)
        if self.use_fast_path:
            y = selective_scan(x_conv, dt, A, B, C, self.D.float(), z=None, scan_impl=self.scan_impl)
        else:
            y = self._slow_forward(x_conv, dt, A, B, C)

        y = y * self.activation(z)
        return self.out_proj(y)

    def _slow_forward(self, x, dt, A, B, C) -> torch.Tensor:
        """Non-fast-path scan: the same implementation as ``selective_scan`` with chunk ``min(64, L)``.

        The reference once carried a second copy of the scan here, including a second copy of the
        divide-and-clamp defect, and the pre-push harness only ever exercised that copy.
        """
        chunk_size = min(64, x.shape[1])
        impl = selective_scan_parallel if self.scan_impl == "legacy" else selective_scan_exact
        in_dtype = x.dtype
        y = impl(
            x.float(), dt.float(), A.float(), B.float(), C.float(), self.D.float(), chunk_size=chunk_size
        )
        return y.to(in_dtype)

    def _forward_segmented(self, x: torch.Tensor, doc_ids: torch.Tensor) -> torch.Tensor:
        """Per-(row, segment) wrapper that resets the SSM state at document boundaries."""
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
