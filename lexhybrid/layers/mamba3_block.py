"""Mamba-3 mixer (reference M2-C).

With every feature flag off this is **exactly Mamba-2 SSD**: scalar-per-head ``A``, multi-value
``B``/``C`` shared across heads, a short depthwise convolution, and the chunked scan in
``kernels/ssd``. The Mamba-3 additions (arXiv:2603.15569) each sit behind a flag that reduces to
that baseline, so every ablation arm differs from its predecessor in exactly one variable:

    use_trapezoid   Sec 3.1  exponential-trapezoidal discretization
    use_rope        Sec 3.2  complex state via data-dependent rotation
    bc_bias         Sec 3.4  learnable head-wise B/C biases
    use_conv=False  Sec 4.2  drop the short convolution
    mimo_rank > 1   Sec 3.3  MIMO -- plumbed, never run

Two deliberate departures from stock Mamba-2, recorded so they are not mistaken for bugs:

* **No post-gate RMSNorm** by default (``use_outproj_norm=False``): Mamba-3 removes it because
  BCNorm stabilizes training on its own (Sec 3.4).
* **The projection is sized for every flag at once.** The ``trap``, ``dd_A`` and rotation-angle
  slices of ``in_proj`` exist whether or not their flags are on, so parameter count is identical
  across arms and turning a flag off is exactly "zero that slice".

P1 port: unchanged apart from the ``doc_ids`` name. ``forward`` does not yet write the decode
cache (defect 7, P2-F).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lexhybrid.kernels.ssd import ssd_chunked_scan
from lexhybrid.kernels.ssd.ssd_reference import ssd_step
from lexhybrid.layers.normalization import RMSNorm
from lexhybrid.layers.rotary import apply_rotary, cumulative_angles


def heavy_tail_activation(x: torch.Tensor) -> torch.Tensor:
    """``1 + x`` for ``x >= 0``, ``1 / (1 - x)`` otherwise -- a map onto (0, inf) for the data-dependent A.

    It grows linearly rather than exponentially on the positive side, which the reference Mamba-3
    implementation notes improves stability during WSD training and at higher learning rates.
    """
    return torch.where(x >= 0, 1.0 + x, 1.0 / (1.0 - x))


def _shift_right(v: torch.Tensor) -> torch.Tensor:
    """Shift a (batch, seqlen, ...) tensor one step right along seqlen, zero-filling position 0."""
    return torch.cat([torch.zeros_like(v[:, :1]), v[:, :-1]], dim=1)


class Mamba3Block(nn.Module):
    """Mamba-3 sequence mixer. Flags all off == Mamba-2 SSD.

    Args:
        dim: model dimension.
        d_state: SSM state size N (128 is the reference default: 8x the Mamba-1 setting for
            +0.24% parameters, because B/C are shared across heads).
        head_dim: SSM head dimension P; ``nheads = dim * expand_factor / head_dim``.
        expand_factor: inner-dimension expansion.
        ngroups: number of B/C groups (1 = every head shares one B and C).
        chunk_size: SSD chunk length.
        use_conv / conv_size: short depthwise causal convolution over the concatenated x, B, C.
        use_trapezoid, use_rope, bc_bias, mimo_rank: the Mamba-3 features listed above.
        a_mode: "static" learns one ``A`` per head (Mamba-2); "data_dependent" projects it per token.
        dt_limit: hard clamp on Delta; softplus is otherwise unbounded above.
    """

    # Declares that this mixer honours packed-document resets, so `HybridBlock` dispatches on a
    # capability rather than on a hard-coded tuple of layer names (reference M2-E).
    supports_doc_ids = True

    def __init__(
        self,
        dim: int,
        d_state: int = 128,
        head_dim: int = 64,
        expand_factor: int = 2,
        ngroups: int = 1,
        chunk_size: int = 64,
        use_conv: bool = True,
        conv_size: int = 4,
        use_trapezoid: bool = False,
        use_rope: bool = False,
        rope_fraction: float = 0.5,
        bc_bias: str = "none",
        mimo_rank: int = 1,
        a_mode: str = "static",
        a_floor: float = 1e-4,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_init_floor: float = 1e-4,
        dt_limit: float = 1.0,
        theta_max: float = 1.0,
        use_outproj_norm: bool = False,
        use_hybrid_norm: bool = False,  # accepted for interface parity; BCNorm is unconditional
        **unused,
    ):
        super().__init__()
        if bc_bias not in ("none", "zero_init", "one_init"):
            raise ValueError(f"bc_bias must be 'none', 'zero_init' or 'one_init', got {bc_bias!r}")
        if a_mode not in ("static", "data_dependent"):
            raise ValueError(f"a_mode must be 'static' or 'data_dependent', got {a_mode!r}")
        if rope_fraction not in (0.5, 1.0):
            raise ValueError(f"rope_fraction must be 0.5 or 1.0, got {rope_fraction!r}")
        if mimo_rank != 1:
            raise NotImplementedError(
                "MIMO is plumbed but deliberately not implemented: rank 4 costs +3.2% parameters, "
                "leaving the parameter-matched regime, and its payoff (decode arithmetic intensity) "
                "is not measured in this project."
            )

        self.dim = dim
        self.d_state = d_state
        self.head_dim = head_dim
        self.expand_factor = expand_factor
        self.inner_dim = dim * expand_factor
        if self.inner_dim % head_dim != 0:
            raise ValueError(f"inner_dim ({self.inner_dim}) must be divisible by head_dim")
        self.nheads = self.inner_dim // head_dim
        if self.nheads % ngroups != 0:
            raise ValueError(f"nheads ({self.nheads}) must be divisible by ngroups ({ngroups})")
        self.ngroups = ngroups
        self.chunk_size = chunk_size
        self.use_conv = use_conv
        self.conv_size = conv_size
        self.use_trapezoid = use_trapezoid
        self.use_rope = use_rope
        self.rope_fraction = rope_fraction
        self.bc_bias = bc_bias
        self.mimo_rank = mimo_rank
        self.a_mode = a_mode
        self.a_floor = a_floor
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init_floor = dt_init_floor
        self.dt_limit = dt_limit
        # theta = theta_max * tanh(proj(x)) bounds the angular rate; with the projection slice
        # near zero at init the rotation starts as ~identity, so rope-on and rope-off arms start
        # from the same place.
        self.theta_max = theta_max

        # Rotation angles are shared across heads and cover `rope_fraction` of d_state, one angle
        # per 2-D rotation pair.
        self.n_rope_angles = int(d_state * rope_fraction) // 2 * ngroups

        self.bc_dim = d_state * ngroups
        d_in_proj = (
            2 * self.inner_dim  # z (gate) and x (SSM input)
            + 2 * self.bc_dim  # B and C
            + 3 * self.nheads  # dt, A, trapezoid lambda
            + self.n_rope_angles
        )
        self.in_proj = nn.Linear(dim, d_in_proj, bias=False)
        self._split = [
            self.inner_dim,
            self.inner_dim,
            self.bc_dim,
            self.bc_dim,
            self.nheads,
            self.nheads,
            self.nheads,
            self.n_rope_angles,
        ]

        if use_conv:
            conv_channels = self.inner_dim + 2 * self.bc_dim
            self.conv1d = nn.Conv1d(
                conv_channels,
                conv_channels,
                kernel_size=conv_size,
                groups=conv_channels,
                padding=conv_size - 1,
                bias=True,
            )
        else:
            self.conv1d = None

        # BCNorm (paper Sec 3.4), unconditional: part of the layer, not a degree of freedom.
        self.B_norm = RMSNorm(d_state)
        self.C_norm = RMSNorm(d_state)
        if bc_bias == "none":
            self.B_bias = None
            self.C_bias = None
        else:
            init = 0.0 if bc_bias == "zero_init" else 1.0
            self.B_bias = nn.Parameter(torch.full((self.nheads, d_state), init))
            self.C_bias = nn.Parameter(torch.full((self.nheads, d_state), init))

        # Parameters owned by one setting are registered conditionally, so "every parameter
        # receives a gradient" stays a strict check. lambda = sigmoid(proj(x) + trap_bias); init 0
        # gives lambda = 0.5 (classical trapezoid); lambda = 1 recovers Euler exactly.
        self.trap_bias = nn.Parameter(torch.zeros(self.nheads)) if use_trapezoid else None
        self.A_log = nn.Parameter(torch.empty(self.nheads)) if a_mode == "static" else None
        self.dt_bias = nn.Parameter(torch.empty(self.nheads))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.out_norm = RMSNorm(self.inner_dim) if use_outproj_norm else None
        self.out_proj = nn.Linear(self.inner_dim, dim, bias=False)
        self.activation = nn.SiLU()
        self.post_model_init()

    def post_model_init(self) -> None:
        """(Re-)apply the SSM initializations after the parent model's global weight pass.

        ``HybridLanguageModel.__init__`` ends with ``self.apply(self._init_weights)``, which zeroes
        every ``nn.Linear`` bias. ``dt_bias`` and ``A_log`` are bare Parameters so they survive that,
        but this hook is called explicitly afterwards anyway: the Mamba-1 block lost its dt init to
        exactly this ordering (reference M1-F).
        """
        with torch.no_grad():
            # Delta ~ logU[dt_min, dt_max], inverted through softplus.
            dt = torch.exp(
                torch.rand(self.nheads) * (math.log(self.dt_max) - math.log(self.dt_min))
                + math.log(self.dt_min)
            ).clamp(min=self.dt_init_floor)
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            # A ~ U[1, 16] per head, stored in log space; A = -exp(A_log) is strictly negative.
            if self.A_log is not None:
                self.A_log.copy_(torch.log(torch.rand(self.nheads) * 15.0 + 1.0))

    def _compute_a(self, a_raw: torch.Tensor) -> torch.Tensor:
        """Per-head state transition, strictly negative. Shape (nheads,) or (batch, len, nheads)."""
        if self.a_mode == "static":
            return -torch.exp(self.A_log.float())
        return torch.clamp(-heavy_tail_activation(a_raw.float()), max=-self.a_floor)

    def forward(
        self,
        x: torch.Tensor,
        cache: dict | None = None,
        doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """(batch, seqlen, dim) -> (batch, seqlen, dim).

        Argument order is load-bearing: ``hybrid_lm.py`` passes these positionally through
        ``torch.utils.checkpoint.checkpoint``.
        """
        batch, seqlen, _ = x.shape
        proj = self.in_proj(x)
        z, xs, B, C, dt_raw, a_raw, trap_raw, angles = torch.split(proj, self._split, dim=-1)

        if self.conv1d is not None:
            xbc = torch.cat([xs, B, C], dim=-1).transpose(1, 2)
            xbc = self.conv1d(xbc)[..., :seqlen].transpose(1, 2)
            xbc = self.activation(xbc)
            if doc_ids is not None:
                xbc = self._mask_conv_across_documents(torch.cat([xs, B, C], dim=-1), xbc, doc_ids)
            xs, B, C = torch.split(xbc, [self.inner_dim, self.bc_dim, self.bc_dim], dim=-1)

        dt = F.softplus(dt_raw.float() + self.dt_bias)
        if self.dt_limit is not None:
            # Replaces the RMSNorm that used to bound Delta by accident; without it gamma =
            # lambda * Delta scales the state input without limit.
            dt = dt.clamp(max=self.dt_limit)
        A = self._compute_a(a_raw)

        B = self.B_norm(B.view(batch, seqlen, self.ngroups, self.d_state))
        C = self.C_norm(C.view(batch, seqlen, self.ngroups, self.d_state))

        xs = xs.view(batch, seqlen, self.nheads, self.head_dim)
        y = self._scan(xs, dt, A, B, C, trap_raw, angles, doc_ids)

        y = y.reshape(batch, seqlen, self.inner_dim)
        if self.out_norm is not None:
            y = self.out_norm(y)
        return self.out_proj(y * self.activation(z))

    def _scan(self, xs, dt, A, B, C, trap_raw, angles, doc_ids):
        """SSD scan, plus the trapezoidal term and the complex-state rotation when enabled."""
        if self.use_rope:
            # Rotate BEFORE the trapezoid shifts anything (paper Prop. 4): the beta term's B_{t-1}
            # must carry its own Theta_{t-1}.
            theta = self.theta_max * torch.tanh(angles.float())
            theta_angles = cumulative_angles(dt[..., :1], theta, doc_ids=doc_ids)
            B = apply_rotary(B, theta_angles, self.rope_fraction)
            C = apply_rotary(C, theta_angles, self.rope_fraction)
        if self.B_bias is not None:
            B = B.repeat_interleave(self.nheads // self.ngroups, dim=2) + self.B_bias
            C = C.repeat_interleave(self.nheads // self.ngroups, dim=2) + self.C_bias

        coeff, extra = dt, None
        if self.use_trapezoid:
            coeff, extra = self._trapezoid_terms(xs, dt, A, B, trap_raw, doc_ids)

        return ssd_chunked_scan(
            xs,
            dt,
            A,
            B,
            C,
            self.D.float(),
            chunk_size=self.chunk_size,
            doc_ids=doc_ids,
            coeff=coeff,
            extra_terms=extra,
        )

    def _trapezoid_terms(self, xs, dt, A, B, trap_raw, doc_ids):
        """Exponential-trapezoidal discretization (paper Prop. 1).

            h_t = alpha_t h_{t-1} + beta_t B_{t-1} x_{t-1} + gamma_t B_t x_t
            alpha_t = exp(dt_t A),  beta_t = (1 - lambda_t) dt_t alpha_t,  gamma_t = lambda_t dt_t

        The recurrence is linear in its state input, so the beta term is one more
        ``(coefficient, B, x)`` triple over the same decay mask -- no second scan.

        Returns ``(gamma, [(beta, shift(B), shift(x))])``.
        """
        lam = torch.sigmoid(trap_raw.float() + self.trap_bias)  # (batch, seqlen, nheads)
        alpha = torch.exp(dt * A)
        gamma = lam * dt
        beta = (1.0 - lam) * dt * alpha

        # beta reaches one token back, so at a document's first position it would pull in the
        # previous document's last token -- the leak the decay mask prevents for the state.
        if doc_ids is not None:
            starts = torch.zeros_like(doc_ids, dtype=torch.bool)
            starts[:, 1:] = doc_ids[:, 1:] != doc_ids[:, :-1]
            starts[:, 0] = True
            beta = beta.masked_fill(starts.unsqueeze(-1), 0.0)
        else:
            beta = beta.clone()
            beta[:, 0] = 0.0

        return gamma, [(beta, _shift_right(B), _shift_right(xs))]

    def _mask_conv_across_documents(self, raw, conved, doc_ids):
        """Recompute the first ``conv_size - 1`` positions of each document with a zeroed history.

        A causal depthwise convolution of width k lets each document see the last k-1 tokens of the
        previous one. Only those positions are wrong, so the fix is a masked re-run of the window.
        """
        k = self.conv_size
        if k <= 1:
            return conved
        batch, seqlen, _ = raw.shape
        pos = torch.arange(seqlen, device=raw.device)
        windows = F.pad(raw.transpose(1, 2), (k - 1, 0)).unfold(-1, k, 1)  # (b, ch, L, k)
        offsets = torch.arange(k - 1, -1, -1, device=raw.device)  # k-1 .. 0 steps back
        src = (pos.view(-1, 1) - offsets.view(1, -1)).clamp(min=0)  # (L, k)
        same = doc_ids.gather(1, src.reshape(1, -1).expand(batch, -1)).view(batch, seqlen, k)
        same = (same == doc_ids.unsqueeze(-1)) & ((pos.view(1, -1, 1) - offsets.view(1, 1, -1)) >= 0)
        w = self.conv1d.weight.view(1, -1, 1, k)
        masked = (windows * same.unsqueeze(1) * w).sum(-1) + self.conv1d.bias.view(1, -1, 1)
        masked = self.activation(masked.transpose(1, 2))
        # Only the first k-1 positions of each document differ; splice those in.
        starts = torch.zeros_like(doc_ids, dtype=torch.bool)
        starts[:, 1:] = doc_ids[:, 1:] != doc_ids[:, :-1]
        starts[:, 0] = True
        near_start = torch.zeros_like(starts)
        for shift in range(k - 1):
            near_start[:, shift:] |= starts[:, : seqlen - shift] if shift else starts
        return torch.where(near_start.unsqueeze(-1), masked, conved)

    # -- O(1) recurrent decode (reference M6-A) ---------------------------------------------------
    supports_step = True

    def allocate_inference_cache(self, batch_size, device=None, dtype=torch.float32):
        """Everything the recurrence needs to continue from token t to t+1, and nothing that grows.

        ``nheads * head_dim * d_state`` state elements, a conv window of ``conv_size - 1``, and,
        with RoPE, one accumulated angle per rotated pair (fp64, as the chunked path accumulates).
        """
        device = device or self.in_proj.weight.device
        cache = {
            "ssm_state": torch.zeros(
                batch_size, self.nheads, self.head_dim, self.d_state, device=device, dtype=dtype
            ),
            "seen": 0,
        }
        if self.conv1d is not None:
            cache["conv_state"] = torch.zeros(
                batch_size,
                self.conv1d.weight.shape[0],
                max(self.conv_size - 1, 0),
                device=device,
                dtype=dtype,
            )
        if self.use_rope:
            cache["angle_state"] = torch.zeros(
                batch_size, 1, self.n_rope_angles, device=device, dtype=torch.float64
            )
        if self.use_trapezoid:
            # Allocated as zeros so the cache has a fixed structure from step zero; `seen == 0`
            # suppresses the term. B is head-indexed once a B/C bias is on, group-indexed otherwise.
            b_groups = self.nheads if self.B_bias is not None else self.ngroups
            cache["B_prev"] = torch.zeros(batch_size, b_groups, self.d_state, device=device, dtype=dtype)
            cache["x_prev"] = torch.zeros(batch_size, self.nheads, self.head_dim, device=device, dtype=dtype)
        return cache

    def step(self, x_t: torch.Tensor, cache: dict) -> torch.Tensor:
        """Advance one token. ``(batch, dim)`` or ``(batch, 1, dim)`` in, ``(batch, dim)`` out.

        Mirrors ``forward`` exactly -- same projection split, same conv, same rotate-then-bias-then-
        shift ordering -- but carries state in ``cache``. The recurrence is ``ssd_step``, the fp64
        oracle's inner loop, and the state stays fp32 regardless of autocast.
        """
        if x_t.dim() == 3:
            if x_t.shape[1] != 1:
                raise ValueError(f"step() takes one token, got seqlen {x_t.shape[1]}")
            x_t = x_t[:, 0]
        batch = x_t.shape[0]
        out_dtype = x_t.dtype

        proj = self.in_proj(x_t)
        z, xs, B, C, dt_raw, a_raw, trap_raw, angles = torch.split(proj, self._split, dim=-1)

        if self.conv1d is not None:
            xbc = torch.cat([xs, B, C], dim=-1)  # (batch, channels)
            window = torch.cat(
                [cache["conv_state"].to(xbc.dtype), xbc.unsqueeze(-1)], dim=-1
            )  # (batch, ch, k)
            # Depthwise, so the convolution is an elementwise product summed over the window;
            # w[0] hits the oldest position, as Conv1d's left-padded cross-correlation does.
            conv = (window * self.conv1d.weight.squeeze(1).unsqueeze(0)).sum(-1)
            conv = conv + self.conv1d.bias
            cache["conv_state"] = window[..., 1:].detach()
            xbc = self.activation(conv)
            xs, B, C = torch.split(xbc, [self.inner_dim, self.bc_dim, self.bc_dim], dim=-1)

        dt = F.softplus(dt_raw.float() + self.dt_bias)  # (batch, nheads)
        if self.dt_limit is not None:
            dt = dt.clamp(max=self.dt_limit)
        A = self._compute_a(a_raw)

        B = self.B_norm(B.view(batch, self.ngroups, self.d_state)).float()
        C = self.C_norm(C.view(batch, self.ngroups, self.d_state)).float()
        xs = xs.view(batch, self.nheads, self.head_dim).float()

        if self.use_rope:
            theta = self.theta_max * torch.tanh(angles.float())  # (batch, n_angles)
            increment = (dt[..., :1].double() * theta.double()).unsqueeze(1)
            cache["angle_state"] = torch.remainder(cache["angle_state"] + increment, 2.0 * math.pi)
            ang = cache["angle_state"].to(B.dtype)  # (batch, 1, n_angles)
            B = apply_rotary(B.unsqueeze(1), ang, self.rope_fraction).squeeze(1)
            C = apply_rotary(C.unsqueeze(1), ang, self.rope_fraction).squeeze(1)
        if self.B_bias is not None:
            B = B.repeat_interleave(self.nheads // self.ngroups, dim=1) + self.B_bias
            C = C.repeat_interleave(self.nheads // self.ngroups, dim=1) + self.C_bias

        coeff, extra = dt, None
        if self.use_trapezoid:
            lam = torch.sigmoid(trap_raw.float() + self.trap_bias)  # (batch, nheads)
            coeff = lam * dt  # gamma
            if cache["seen"] > 0:
                beta = (1.0 - lam) * dt * torch.exp(dt * A)
                extra = [(beta, cache["B_prev"], cache["x_prev"])]
            # At the first token beta is zero, exactly as the chunked path masks beta[:, 0].
            cache["B_prev"], cache["x_prev"] = B, xs

        y, cache["ssm_state"] = ssd_step(
            xs,
            dt,
            A,
            B,
            C,
            cache["ssm_state"].to(xs.dtype),
            self.D.float(),
            coeff=coeff,
            extra_terms=extra,
        )
        cache["seen"] += 1

        y = y.reshape(batch, self.inner_dim).to(out_dtype)
        if self.out_norm is not None:
            y = self.out_norm(y)
        return self.out_proj(y * self.activation(z))

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, d_state={self.d_state}, head_dim={self.head_dim}, "
            f"nheads={self.nheads}, ngroups={self.ngroups}, conv={self.use_conv}, "
            f"trapezoid={self.use_trapezoid}, rope={self.use_rope}, bc_bias={self.bc_bias}, "
            f"a_mode={self.a_mode}"
        )
