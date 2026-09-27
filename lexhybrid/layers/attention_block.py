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

P2-H (defect 8): RoPE positions restart at every document in a packed row, so a document sees
the positions it would see alone, and ``_rope_tables`` extends with the configured ``rope_theta``
(the reference rebuilt with 10,000). The config default is 500,000: 10,000 is a Llama-1 setting
whose slowest frequency wraps within ~63K tokens and resolves long offsets poorly.
P2-I (defect 5): ``step()`` with a preallocated KV cache ``{k, v: (B, H, Lmax, Dh), seen,
doc_start}``, filled in one pass by ``forward(x, cache)``, so a hybrid containing attention decodes
with the cache instead of recomputing (the reference's ``supports_cached_decode`` was all-or-nothing
and attention made it false). Keys are cached after RoPE; a token's position is its offset inside
its document, as in the packed forward.
P2-J (defect 18): packed rows can attend through ``torch.nn.attention.flex_attention`` with a
block mask built from ``doc_ids`` (``attn_impl``), instead of a dense ``(B, 1, L, L)`` fp32 mask --
256 MB per row at 8K, and it pushes SDPA off the flash kernel. ``"auto"`` takes flex on CUDA and
keeps the dense mask on CPU, where it is the reference path (compiled flex needs Inductor, which
runs on Linux CPUs and on GPUs but not on macOS).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lexhybrid.kernels.segments import segment_starts
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
    """Apply rotary embeddings to a (B, H, L, head_dim) tensor.

    The tables are (L, head_dim), shared by every row, or (B, L, head_dim) per row -- the packed
    case, where each row's positions restart at its own document boundaries.
    """
    if cos.dim() == 2:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    cos = cos.unsqueeze(1).to(x.dtype)
    sin = sin.unsqueeze(1).to(x.dtype)
    half = x.shape[-1] // 2
    x_rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + x_rot * sin


def positions_within_documents(doc_ids: torch.Tensor) -> torch.Tensor:
    """(B, L) position of each token inside its own document: 0 at every document start."""
    starts = segment_starts(doc_ids)
    starts[:, 0] = True
    idx = torch.arange(doc_ids.shape[1], device=doc_ids.device).expand_as(doc_ids)
    return idx - torch.where(starts, idx, torch.zeros_like(idx)).cummax(dim=1).values


ATTN_IMPLS = ("auto", "sdpa", "flex")
_COMPILED_FLEX = None


def compiled_flex_attention():
    """``flex_attention`` under ``torch.compile``, built once. Uncompiled it materialises the full
    score matrix, which is exactly what the block mask exists to avoid."""
    global _COMPILED_FLEX
    if _COMPILED_FLEX is None:
        from torch.nn.attention.flex_attention import flex_attention

        _COMPILED_FLEX = torch.compile(flex_attention, dynamic=False)
    return _COMPILED_FLEX


def build_doc_block_mask(doc_ids: torch.Tensor):
    """FlexAttention ``BlockMask`` for causal attention inside each document of a packed row.

    Fully masked (row-block, column-block) tiles are skipped by the kernel, so the cost follows the
    documents' own lengths rather than the row's.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    batch, seq_len = doc_ids.shape

    def same_document_causal(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (doc_ids[b, q_idx] == doc_ids[b, kv_idx])

    return create_block_mask(
        same_document_causal, B=batch, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=doc_ids.device
    )


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
        attn_impl: packed-row attention: ``"flex"`` (block mask), ``"sdpa"`` (dense mask), or
            ``"auto"`` (flex on CUDA, dense elsewhere). Unpacked rows always use causal SDPA.
    """

    # Capability read by HybridBlock: this mixer consumes doc_ids (document-boundary mask).
    supports_doc_ids = True

    def __init__(
        self,
        dim: int,
        num_heads: int | None = None,
        head_dim: int = 64,
        attn_dropout: float = 0.0,
        rope_theta: float = 500000.0,
        max_position_embeddings: int = 1024,
        use_hybrid_norm: bool = False,
        attn_impl: str = "auto",
    ):
        super().__init__()
        if attn_impl not in ATTN_IMPLS:
            raise ValueError(f"attn_impl must be one of {ATTN_IMPLS}, got {attn_impl!r}")
        self.attn_impl = attn_impl
        if num_heads is None:
            num_heads = max(1, dim // head_dim)
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_dropout = attn_dropout
        self.rope_theta = float(rope_theta)
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
        """cos/sin covering ``seq_len``, extending the cache with this block's ``rope_theta``."""
        if seq_len > self.rope_cos.shape[0]:
            cos, sin = build_rope_cache(
                self.head_dim, seq_len, theta=self.rope_theta, device=device, dtype=self.rope_cos.dtype
            )
            self.rope_cos, self.rope_sin = cos, sin
        return self.rope_cos[:seq_len].to(device), self.rope_sin[:seq_len].to(device)

    def _rope_for(self, seq_len: int, doc_ids: torch.Tensor | None, device: torch.device):
        """Tables for absolute positions, or per-row tables for positions within each document."""
        cos, sin = self._rope_tables(seq_len, device)
        if doc_ids is None:
            return cos, sin
        pos = positions_within_documents(doc_ids)
        return cos[pos], sin[pos]  # (B, L, head_dim)

    def forward(
        self,
        x: torch.Tensor,
        cache: dict | None = None,
        doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Causal self-attention, (B, L, dim) -> (B, L, dim).

        Args:
            x: input.
            cache: an EMPTY cache from ``allocate_inference_cache``; when given, the keys (after
                RoPE) and values are written into it and ``step`` continues from the last token.
            doc_ids: optional (B, L) document ids; attention is blocked across boundaries and
                RoPE positions restart at each document.
        """
        if cache is not None and cache.get("seen", 0) != 0:
            raise ValueError(
                "AttentionBlock.forward fills an empty cache (a prefill); continue a filled cache with step()"
            )
        batch, seq_len, _ = x.shape
        qkv = self.qkv_proj(x).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, H, L, hd)

        if self.use_hybrid_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        cos, sin = self._rope_for(seq_len, doc_ids, x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if cache is not None:
            self._fill_cache(cache, k, v, doc_ids)

        dropout_p = self.attn_dropout if self.training else 0.0
        if doc_ids is None:
            attn = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        elif self._packed_impl(q, dropout_p) == "flex":
            attn = compiled_flex_attention()(q, k, v, block_mask=build_doc_block_mask(doc_ids))
        else:
            # The explicit mask carries causality AND the boundaries, so is_causal must be False.
            mask = build_doc_boundary_attn_mask(doc_ids).to(q.dtype)
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)

        attn = attn.transpose(1, 2).reshape(batch, seq_len, self.dim)
        return self.out_proj(attn)

    def _packed_impl(self, q: torch.Tensor, dropout_p: float) -> str:
        """Which kernel a packed row uses. FlexAttention has no attention dropout, and in torch 2.11 no
        backward pass on CPU (CI run 36330688065): a CPU training step with ``flex`` is refused here,
        before it could fail inside autograd; ``auto`` never picks flex on a CPU."""
        if self.attn_impl == "flex":
            if dropout_p > 0:
                raise ValueError(
                    "attn_impl='flex' has no attention dropout; set attn_dropout 0 or use 'sdpa'"
                )
            if not q.is_cuda and torch.is_grad_enabled() and q.requires_grad:
                raise NotImplementedError(
                    "FlexAttention has no CPU backward in this torch version; train on CPU with "
                    "attn_impl 'sdpa' or 'auto' (flex runs forward-only on CPU)"
                )
            return "flex"
        if self.attn_impl == "auto" and q.is_cuda and dropout_p == 0:
            return "flex"
        return "sdpa"

    # -- KV-cache decode (P2-I) ------------------------------------------------------------------
    supports_step = True

    def allocate_inference_cache(self, batch_size, device=None, dtype=torch.float32, max_seq_len=None):
        """A KV cache for ``max_seq_len`` tokens (default ``max_position_embeddings``).

        Unlike the recurrent mixers this grows with the context: ``2 * B * H * Lmax * Dh`` elements.
        ``doc_start`` is each row's first position of the document being continued; keys before it
        belong to earlier documents and are masked out of every step.
        """
        device = device or self.qkv_proj.weight.device
        max_seq_len = int(max_seq_len or self.max_position_embeddings)
        self._rope_tables(max_seq_len, device)  # extend the tables once, not per step
        shape = (batch_size, self.num_heads, max_seq_len, self.head_dim)
        return {
            "k": torch.zeros(shape, device=device, dtype=dtype),
            "v": torch.zeros(shape, device=device, dtype=dtype),
            "seen": 0,
            "doc_start": torch.zeros(batch_size, dtype=torch.long, device=device),
            "masked": False,  # True once a row's current document starts after position 0
        }

    def _fill_cache(self, cache: dict, k: torch.Tensor, v: torch.Tensor, doc_ids: torch.Tensor | None):
        length = k.shape[2]
        if length > cache["k"].shape[2]:
            raise ValueError(f"prompt of {length} tokens exceeds the KV cache ({cache['k'].shape[2]})")
        cache["k"][:, :, :length] = k.to(cache["k"].dtype)
        cache["v"][:, :, :length] = v.to(cache["v"].dtype)
        cache["seen"] = length
        if doc_ids is not None:
            start = length - 1 - positions_within_documents(doc_ids)[:, -1]
            cache["doc_start"] = start.to(cache["doc_start"].device)
            cache["masked"] = bool((start > 0).any())

    def step(self, x_t: torch.Tensor, cache: dict) -> torch.Tensor:
        """Advance one token. ``(batch, dim)`` or ``(batch, 1, dim)`` in, ``(batch, dim)`` out.

        Mirrors ``forward``: same projection, QK-norm and RoPE, attention over the cached keys of
        the current document plus this token's own.
        """
        if x_t.dim() == 3:
            if x_t.shape[1] != 1:
                raise ValueError(f"step() takes one token, got seqlen {x_t.shape[1]}")
            x_t = x_t[:, 0]
        batch = x_t.shape[0]
        seen = cache["seen"]
        if seen >= cache["k"].shape[2]:
            raise RuntimeError(f"KV cache full at {seen} tokens: allocate with a larger max_seq_len")
        q, k, v = self.qkv_proj(x_t).view(batch, 3, self.num_heads, self.head_dim).unbind(1)
        if self.use_hybrid_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        pos = seen - cache["doc_start"]  # (B,) position inside the current document
        cos, sin = self._rope_tables(seen + 1, x_t.device)
        cos, sin = cos[pos].unsqueeze(1), sin[pos].unsqueeze(1)  # (B, 1, Dh)
        q = apply_rope(q.unsqueeze(2), cos, sin).squeeze(2)
        k = apply_rope(k.unsqueeze(2), cos, sin).squeeze(2)
        cache["k"][:, :, seen] = k.to(cache["k"].dtype)
        cache["v"][:, :, seen] = v.to(cache["v"].dtype)
        keys, values = cache["k"][:, :, : seen + 1], cache["v"][:, :, : seen + 1]
        mask = None
        if cache["masked"]:
            before = torch.arange(seen + 1, device=x_t.device).view(1, -1) < cache["doc_start"].view(-1, 1)
            mask = torch.zeros(batch, 1, 1, seen + 1, device=x_t.device, dtype=keys.dtype)
            mask = mask.masked_fill(before.view(batch, 1, 1, -1), float("-inf"))
        attn = F.scaled_dot_product_attention(q.unsqueeze(2).to(keys.dtype), keys, values, attn_mask=mask)
        cache["seen"] = seen + 1
        return self.out_proj(attn.reshape(batch, self.dim).to(x_t.dtype))

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_heads={self.num_heads}, head_dim={self.head_dim}, "
            f"hybrid_norm={self.use_hybrid_norm}, rope_theta={self.rope_theta:g}, attn_impl={self.attn_impl}"
        )
