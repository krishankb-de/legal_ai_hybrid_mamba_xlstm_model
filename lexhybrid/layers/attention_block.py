"""Causal self-attention mixer with RoPE (reference Phase 14A).

A *mixer*, not a model class: registered as the ``"attention"`` layer type, so
``layer_pattern: ["attention"]`` turns ``HybridLanguageModel`` into an ordinary pre-norm
Transformer decoder that goes through the identical pipeline as the hybrid (same embeddings, MLP,
trainer, decoding, scorer). The legal hybrid interleaves two of these among its recurrent layers.

Design choices (comparison fairness):
- RoPE, not learned position embeddings: the recurrent mixers spend zero parameters on position.
- No biases on qkv/out, matching ``HybridBlock``'s MLP.
- Document-boundary masking from ``doc_ids``: the recurrent mixers reset at a boundary, so
  attention must not attend across one either.
- ``use_hybrid_norm`` is honoured as QK-norm.

P1 port: unchanged. Recorded for P2: no ``step()``/KV cache (defect 5); RoPE positions are
absolute within a packed row and ``_rope_tables`` rebuilds with the default theta rather than
``rope_theta`` (defect 8); the document mask is a dense ``(B, 1, L, L)`` tensor (defect 18).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lexhybrid.layers.normalization import RMSNorm


def build_rope_cache(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute rotary cos/sin tables, each (max_seq_len, head_dim), rotate-half layout."""
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # (L, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (L, head_dim)
    cos, sin = emb.cos().to(dtype), emb.sin().to(dtype)
    if device is not None:
        cos, sin = cos.to(device), sin.to(device)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to a (B, H, L, head_dim) tensor with (L, head_dim) tables."""
    cos = cos.unsqueeze(0).unsqueeze(0).to(x.dtype)
    sin = sin.unsqueeze(0).unsqueeze(0).to(x.dtype)
    half = x.shape[-1] // 2
    x_rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + x_rot * sin


def build_doc_boundary_attn_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    """Additive causal mask that also blocks attention across documents.

    Args:
        doc_ids: (B, L) integer per-position document ids.

    Returns:
        (B, 1, L, L) float mask: 0.0 where attention is allowed, -inf where it is not.
    """
    batch, seq_len = doc_ids.shape
    device = doc_ids.device
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
    same_doc = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)  # (B, L, L)
    allowed = same_doc & causal.unsqueeze(0)
    mask = torch.zeros(batch, 1, seq_len, seq_len, device=device, dtype=torch.float32)
    return mask.masked_fill(~allowed.unsqueeze(1), float("-inf"))


class AttentionBlock(nn.Module):
    """Multi-head causal self-attention with RoPE.

    Args:
        dim: model dimension.
        num_heads: number of heads (``dim // head_dim`` when None).
        head_dim: per-head size used only to default ``num_heads``; the actual head size is
            ``dim // num_heads``.
        attn_dropout: dropout on attention weights (training only).
        rope_theta: RoPE base frequency.
        max_position_embeddings: size of the precomputed RoPE tables.
        use_hybrid_norm: RMSNorm Q and K (the HybridNorm analogue).
    """

    # Capability read by HybridBlock: this mixer consumes doc_ids (document-boundary mask).
    supports_doc_ids = True

    def __init__(
        self,
        dim: int,
        num_heads: int | None = None,
        head_dim: int = 64,
        attn_dropout: float = 0.0,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 1024,
        use_hybrid_norm: bool = False,
    ):
        super().__init__()
        if num_heads is None:
            num_heads = max(1, dim // head_dim)
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_dropout = attn_dropout
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.use_hybrid_norm = use_hybrid_norm
        if use_hybrid_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        cos, sin = build_rope_cache(self.head_dim, max_position_embeddings, theta=rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _rope_tables(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin covering ``seq_len``, extending the cache if needed (with the default theta: defect 8)."""
        if seq_len > self.rope_cos.shape[0]:
            cos, sin = build_rope_cache(self.head_dim, seq_len, device=device, dtype=self.rope_cos.dtype)
            self.rope_cos, self.rope_sin = cos, sin
        return self.rope_cos[:seq_len].to(device), self.rope_sin[:seq_len].to(device)

    def forward(
        self,
        x: torch.Tensor,
        cache: dict | None = None,
        doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Causal self-attention, (B, L, dim) -> (B, L, dim).

        Args:
            x: input.
            cache: unused in P1 (no KV cache yet; P2-I).
            doc_ids: optional (B, L) document ids; attention is blocked across boundaries.
        """
        batch, seq_len, _ = x.shape
        qkv = self.qkv_proj(x).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, H, L, hd)

        if self.use_hybrid_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        cos, sin = self._rope_tables(seq_len, x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        dropout_p = self.attn_dropout if self.training else 0.0
        if doc_ids is None:
            attn = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        else:
            # The explicit mask carries causality AND the boundaries, so is_causal must be False.
            mask = build_doc_boundary_attn_mask(doc_ids).to(q.dtype)
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)

        attn = attn.transpose(1, 2).reshape(batch, seq_len, self.dim)
        return self.out_proj(attn)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_heads={self.num_heads}, head_dim={self.head_dim}, "
            f"hybrid_norm={self.use_hybrid_norm}"
        )
