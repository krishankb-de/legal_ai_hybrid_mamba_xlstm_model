"""``HybridBlock``: pre-norm -> mixer -> residual, then the MLP, for every mixer type.

Ported from the reference with the sLSTM branch removed. Two behaviours are kept on purpose for
the P1 parity fixtures and fixed in P2-A: the ``mlstm`` allow-list does not strip the ``mlstm_``
prefix, so ``mlstm_gate_soft_cap`` and friends never reach the mixer (defect 2); and Mamba-3 reads
the shared ``expand_factor`` (defect 4).
"""

import functools
import inspect

import torch
import torch.nn as nn

from lexhybrid.config.hybrid_config import LayerType
from lexhybrid.layers.attention_block import AttentionBlock
from lexhybrid.layers.mamba3_block import Mamba3Block
from lexhybrid.layers.mamba_block import MambaBlock
from lexhybrid.layers.mlstm_block import mLSTMBlock
from lexhybrid.layers.normalization import RMSNorm


@functools.lru_cache(maxsize=1)
def _mamba3_params() -> frozenset:
    """The ``mamba3_*`` options this dispatcher forwards, read off the block's signature.

    Derived rather than listed: two hand-maintained copies in the reference both omitted
    ``theta_max``, so its screen could only run the block default. Adding a parameter to the block
    is enough to make it reachable from ``HybridConfig``.
    """
    params = inspect.signature(Mamba3Block.__init__).parameters
    return frozenset(
        name
        for name, p in params.items()
        if name not in ("self", "dim") and p.kind is not inspect.Parameter.VAR_KEYWORD
    )


_MAMBA_PARAMS = frozenset(
    {
        "state_size",
        "conv_size",
        "expand_factor",
        "dt_rank",
        "use_fast_path",
        "use_hybrid_norm",
        "use_dt_norm",
        "scan_impl",
        "dt_init_strategy",
        "dt_min",
        "dt_max",
    }
)
# Defect 2 (kept for P1 parity): unprefixed names, so the `mlstm_*` config keys never match.
_MLSTM_PARAMS = frozenset(
    {
        "head_dim",
        "num_heads",
        "tfla_impl",
        "proj_factor",
        "gate_soft_cap",
        "input_gate_bias_init",
        "forget_gate_bias_init",
        "use_hybrid_norm",
    }
)
_ATTENTION_PARAMS = frozenset(
    {"num_heads", "head_dim", "attn_dropout", "rope_theta", "max_position_embeddings", "use_hybrid_norm"}
)


class HybridBlock(nn.Module):
    """One decoder block: ``x + mixer(norm1(x))``, then the MLP (pre-norm, or HybridNorm post-norm).

    Args:
        dim: model dimension.
        layer_type: ``"mamba"`` | ``"mamba3"`` | ``"mlstm"`` | ``"attention"``.
        norm_type: ``"rms"`` or ``"layer"``.
        use_mlp: include the MLP sublayer.
        mlp_ratio: MLP expansion ratio.
        norm_topology: ``"pre_rms"`` | ``"hybrid"`` | ``"hybrid_bc"``.
        is_first_block: the first block stays pure pre-norm under HybridNorm.
        **layer_kwargs: the flat bag of every config field; each mixer takes what it knows.
    """

    def __init__(
        self,
        dim: int,
        layer_type: LayerType = "mamba",
        norm_type: str = "rms",
        use_mlp: bool = True,
        mlp_ratio: float = 4.0,
        norm_topology: str = "pre_rms",
        is_first_block: bool = False,
        **layer_kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.layer_type = layer_type.lower()
        self.use_mlp = use_mlp
        # HybridNorm: per-projection norms in the mixer, and the FFN post-norm post-residual for
        # blocks >= 1. `hybrid_bc` is `hybrid` minus the Mamba-1 Delta norm.
        self.norm_topology = norm_topology
        self.is_first_block = is_first_block
        self._use_hybrid_norm = norm_topology in ("hybrid", "hybrid_bc")
        self._ffn_post_norm = self._use_hybrid_norm and not is_first_block
        layer_kwargs = dict(layer_kwargs)
        layer_kwargs["use_hybrid_norm"] = self._use_hybrid_norm
        layer_kwargs.setdefault("use_dt_norm", norm_topology != "hybrid_bc")

        self.norm1 = RMSNorm(dim) if norm_type.lower() == "rms" else nn.LayerNorm(dim)

        # A kwarg that names a mixer family but matches nothing is always a mistake -- a typo, or
        # a field added to HybridConfig and never wired through. Unprefixed kwargs are dropped
        # quietly, because the flat bag deliberately carries every mixer's fields.
        known = {
            "mamba3_": _mamba3_params(),
            "mlstm_": frozenset({"gate_soft_cap", "input_gate_bias_init", "forget_gate_bias_init"}),
        }
        for key in layer_kwargs:
            for prefix, names in known.items():
                if key.startswith(prefix) and key[len(prefix) :] not in names:
                    raise ValueError(
                        f"unknown mixer option {key!r}: {key[len(prefix) :]!r} is not a "
                        f"{prefix.rstrip('_')} parameter. Known: {sorted(names)}"
                    )

        if self.layer_type == "mamba":
            filtered = {k: v for k, v in layer_kwargs.items() if k in _MAMBA_PARAMS}
            self.mixer = MambaBlock(dim, **filtered)
        elif self.layer_type == "mamba3":
            # `mamba3_*` keys are stripped so the block's signature stays readable and does not
            # collide with the Mamba-1 names. Keys are processed in dataclass order, so the
            # `mamba3_` form of conv_size/dt_min/dt_max wins over the Mamba-1 one.
            mamba3_params = _mamba3_params()
            renamed = {}
            for key, value in layer_kwargs.items():
                stripped = key[len("mamba3_") :] if key.startswith("mamba3_") else key
                if stripped in mamba3_params:
                    renamed[stripped] = value
            # `head_dim` is shared with mLSTM in the flat bag; the mamba3_ form wins.
            if "mamba3_head_dim" in layer_kwargs:
                renamed["head_dim"] = layer_kwargs["mamba3_head_dim"]
            self.mixer = Mamba3Block(dim, **renamed)
        elif self.layer_type == "mlstm":
            filtered = {k: v for k, v in layer_kwargs.items() if k in _MLSTM_PARAMS}
            self.mixer = mLSTMBlock(dim, **filtered)
        elif self.layer_type == "attention":
            # `head_dim`/`num_heads` are shared with mLSTM on purpose: the matched Transformer reuses
            # the hybrid's head geometry.
            filtered = {k: v for k, v in layer_kwargs.items() if k in _ATTENTION_PARAMS}
            self.mixer = AttentionBlock(dim, **filtered)
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

        # Dispatch on a capability the mixer declares, so a new layer type cannot inherit a
        # "forgot to pass doc_ids" bug by omission (the reference's sLSTM leaked state that way).
        self._mixer_takes_doc_ids = bool(getattr(self.mixer, "supports_doc_ids", False))

        if use_mlp:
            self.norm2 = RMSNorm(dim) if norm_type.lower() == "rms" else nn.LayerNorm(dim)
            mlp_dim = int(dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_dim, bias=False),
                nn.GELU(),
                nn.Linear(mlp_dim, dim, bias=False),
            )
        else:
            self.norm2 = None
            self.mlp = None

    def forward(
        self,
        x: torch.Tensor,
        cache: dict | None = None,
        doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """(batch, seq_len, dim) -> (batch, seq_len, dim). Argument order matches checkpointing."""
        residual = x
        x = self.norm1(x)
        if self._mixer_takes_doc_ids:
            x = self.mixer(x, cache=cache, doc_ids=doc_ids)
        else:  # pragma: no cover - every shipped mixer takes doc_ids
            x = self.mixer(x, cache=cache)
        x = residual + x

        if self.use_mlp:
            if self._ffn_post_norm:
                x = self.norm2(x + self.mlp(x))
            else:
                residual = x
                x = self.norm2(x)
                x = self.mlp(x)
                x = residual + x
        return x

    def allocate_inference_cache(self, batch_size, device=None, dtype=torch.float32):
        """Per-layer decode state, or None if this mixer has no O(1) step."""
        allocate = getattr(self.mixer, "allocate_inference_cache", None)
        if allocate is None:
            return None
        return allocate(batch_size, device=device, dtype=dtype)

    def step(self, x_t: torch.Tensor, cache) -> torch.Tensor:
        """One token through mixer + MLP with residuals, mirroring ``forward``'s topology exactly."""
        if not getattr(self.mixer, "supports_step", False):
            raise NotImplementedError(
                f"{type(self.mixer).__name__} has no step(); cached decode is unavailable for a model "
                f"containing '{self.layer_type}' layers"
            )
        squeeze = x_t.dim() == 3
        if squeeze:
            x_t = x_t[:, 0]
        x_t = x_t + self.mixer.step(self.norm1(x_t), cache)
        if self.use_mlp:
            if self._ffn_post_norm:
                x_t = self.norm2(x_t + self.mlp(x_t))
            else:
                x_t = x_t + self.mlp(self.norm2(x_t))
        return x_t.unsqueeze(1) if squeeze else x_t

    def get_layer_info(self) -> dict:
        return {
            "layer_type": self.layer_type,
            "dim": self.dim,
            "use_mlp": self.use_mlp,
            "mixer_class": self.mixer.__class__.__name__,
        }


def create_hybrid_blocks(
    dim: int,
    num_layers: int,
    layer_pattern: list[LayerType],
    norm_topology: str = "pre_rms",
    **kwargs,
) -> nn.ModuleList:
    """``num_layers`` blocks cycling through ``layer_pattern``; block 0 is marked first."""
    blocks = nn.ModuleList()
    for i in range(num_layers):
        blocks.append(
            HybridBlock(
                dim=dim,
                layer_type=layer_pattern[i % len(layer_pattern)],
                norm_topology=norm_topology,
                is_first_block=(i == 0),
                **kwargs,
            )
        )
    return blocks
