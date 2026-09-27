"""Data-dependent rotary embeddings for the complex-valued SSM (reference M4).

Mamba-3 Sec 3.2 makes the state transition complex. Proposition 2 shows a complex SSM of state
size N/2 is equivalent to a real one of size N whose transition is a scalar decay times a
block-diagonal of 2x2 rotations, and Proposition 3 shows that system is in turn equivalent to a
*real* scalar-transition SSM with a data-dependent rotation applied to B and C:

    h_t = exp(dt_t A) h_{t-1} + (prod_{i<=t} R_i^T) dt_t B_t x_t
    y_t = [(prod_{i<=t} R_i^T) C_t]^T h_t

So the rotation is applied *outside* the scan -- the SSD kernel never learns that its state is
complex. ``C_t^T B_s`` then carries ``R(Theta_s - Theta_t)``, a relative rotation: RoPE with a
data-dependent angle schedule instead of a fixed geometric one.

Numerics. Theta accumulates, and a naive fp32 cumsum of many terms drifts by ~1e-2 rad by
position 512. Three cheap measures keep that irrelevant: accumulate in float64, wrap into
[0, 2pi) before the fp32 sin/cos (exact for rotations), and reset per document so Theta never
grows past one document.
"""

import torch

TWO_PI = 2.0 * torch.pi


def cumulative_angles(
    dt: torch.Tensor,
    theta: torch.Tensor,
    doc_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Theta_t = sum_{i<=t} dt_i * theta_i, accumulated in float64 and wrapped to [0, 2pi).

    Args:
        dt: (batch, seqlen, 1) or (batch, seqlen, n_angles) step sizes.
        theta: (batch, seqlen, n_angles) per-token angular rates.
        doc_ids: optional (batch, seqlen) document ids; the accumulator restarts per document.

    Returns:
        (batch, seqlen, n_angles) angles in [0, 2pi), in ``theta``'s dtype.
    """
    step = dt.double() * theta.double()
    if doc_ids is None:
        angles = step.cumsum(dim=1)
    else:
        # Subtract each segment's running total at its start, which restarts the accumulator
        # without a Python loop over segments.
        total = step.cumsum(dim=1)
        starts = torch.zeros_like(doc_ids, dtype=torch.bool)
        starts[:, 1:] = doc_ids[:, 1:] != doc_ids[:, :-1]
        before = total - step  # exclusive prefix sum
        baseline = torch.where(starts.unsqueeze(-1), before, torch.zeros_like(before))
        angles = total - _segment_baseline(baseline, starts)
    # Wrapping is exact for rotations and keeps sin/cos away from large arguments.
    return torch.remainder(angles, TWO_PI).to(theta.dtype)


def _segment_baseline(baseline: torch.Tensor, starts: torch.Tensor) -> torch.Tensor:
    """Forward-fill each segment's start value across the segment."""
    idx = torch.arange(starts.shape[1], device=starts.device).expand_as(starts)
    last_start = torch.where(starts, idx, torch.zeros_like(idx)).cummax(dim=1).values
    return torch.gather(baseline, 1, last_start.unsqueeze(-1).expand_as(baseline))


def apply_rotary(v: torch.Tensor, angles: torch.Tensor, rope_fraction: float = 0.5) -> torch.Tensor:
    """Rotate the leading ``rope_fraction`` of ``v``'s last dimension as 2-D pairs.

    Args:
        v: (..., seqlen, groups, dstate) tensor to rotate (B or C).
        angles: (batch, seqlen, n_angles) from ``cumulative_angles``, one angle per rotated pair.
        rope_fraction: 0.5 (the reference default) rotates half the state and leaves the rest as
            pure real decay.

    Returns:
        Same shape as ``v``.
    """
    dstate = v.shape[-1]
    rot_dim = int(dstate * rope_fraction) // 2 * 2
    if rot_dim == 0:
        return v
    groups = v.shape[-2]
    ang = angles.view(angles.shape[0], angles.shape[1], groups, rot_dim // 2)
    cos, sin = torch.cos(ang).unsqueeze(-1), torch.sin(ang).unsqueeze(-1)

    rot, keep = v[..., :rot_dim], v[..., rot_dim:]
    pairs = rot.reshape(*rot.shape[:-1], rot_dim // 2, 2)
    a, b = pairs[..., 0:1], pairs[..., 1:2]
    rotated = torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)
    return torch.cat([rotated.reshape(*rot.shape), keep], dim=-1)
