"""Configuration dataclass for the hybrid decoder.

Ported from the reference package's ``models/configuration_hybrid.py`` (branch ``h100_efficiency``,
HEAD ``900acee``). Phase ids in comments (M1-F, M2-F, FM5, Phase 9, ...) refer to the reference
project's plans; ``Docs/REFERENCE_PORT_MAP.md`` summarises them.

P1 ports the dataclass faithfully, minus the fields that only served the medical retrieval and
report-generation stack (sLSTM, pooling, bidirectional encoding, projection-head dropout,
``use_cache``, ``use_tfla``). Defaults are unchanged, including ``scan_impl``/``tfla_impl`` =
``"legacy"``, so the P1 parity fixtures hold; P2 adds the legal defaults.
"""

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Literal

LayerType = Literal["mamba", "mamba3", "mlstm", "attention"]
VALID_LAYER_TYPES = ("mamba", "mamba3", "mlstm", "attention")


@dataclass
class HybridConfig:
    """Every architectural knob of the hybrid decoder.

    One dataclass serves every architecture: ``layer_pattern`` selects the mixer of each block
    (cycled over ``num_layers``), and the per-mixer fields are only read by the mixers that use
    them. ``from_hydra`` is the only way the training entry points build it (FM5).
    """

    # Architecture
    vocab_size: int = 50257
    dim: int = 768
    num_layers: int = 12
    layer_pattern: list[LayerType] = field(default_factory=lambda: ["mamba", "mamba", "mlstm"])

    # Mamba-1 parameters (the legacy incumbent; only `hybrid_legal_legacy` uses this mixer)
    state_size: int = 16
    conv_size: int = 4
    expand_factor: int = 2
    dt_rank: int | None = None  # dim // 16 when None
    use_fast_path: bool = True
    # M1: all four default to the reference's historical behaviour, so adding them changed nothing.
    scan_impl: str = "legacy"  # "legacy" | "exact"
    dt_init_strategy: str = "none"  # "none" | "mamba" (reference logU[dt_min, dt_max] init)
    dt_min: float = 1e-3
    dt_max: float = 1e-1
    tfla_impl: str = "legacy"  # "legacy" | "exact" -- the mLSTM counterpart (M1-H)

    # Mamba-3 (M2). Every flag defaults to the Mamba-2 reduction, so a `mamba3` layer built from
    # defaults is exactly Mamba-2 SSD and each arm moves one variable.
    mamba3_d_state: int = 128
    mamba3_head_dim: int = 64
    mamba3_ngroups: int = 1  # >1 leaves the parameter-matched regime
    mamba3_chunk_size: int = 64
    mamba3_use_conv: bool = True
    mamba3_conv_size: int = 4
    mamba3_use_trapezoid: bool = False  # paper Sec 3.1 (M3)
    mamba3_use_rope: bool = False  # paper Sec 3.2 (M4)
    mamba3_rope_fraction: float = 0.5
    # theta_max bounds the per-token angular rate: theta = theta_max * tanh(proj(x)), and the
    # rotation angle is Delta_t * theta_t. It was missing from the reference dataclass until
    # 2026-09-06, so the block default of 1.0 was the only value its screen could run: with
    # dt_limit=1.0 that is 1 rad/token, 81 full turns over 512 tokens, and every rope-on arm
    # collapsed to ~1166 val PPL. A relative rotation is a positional signal only inside one turn.
    mamba3_theta_max: float = 1.0
    # Delta's init range; the pair sets where the rotation can operate at all.
    mamba3_dt_min: float = 1e-3
    mamba3_dt_max: float = 1e-1
    mamba3_dt_init_floor: float = 1e-4
    mamba3_bc_bias: str = "none"  # "none" | "zero_init" | "one_init" (paper Sec 3.4, M5)
    mamba3_mimo_rank: int = 1  # plumbed, never run
    mamba3_a_mode: str = "static"  # "static" (Mamba-2) | "data_dependent" (Mamba-3)
    mamba3_a_floor: float = 1e-4
    mamba3_dt_limit: float = 1.0
    mamba3_use_outproj_norm: bool = False

    # mLSTM parameters
    head_dim: int = 64
    num_heads: int | None = None  # max(1, dim // head_dim) when None
    proj_factor: int = 2
    mlstm_gate_soft_cap: float = 15.0
    mlstm_input_gate_bias_init: float = -10.0
    mlstm_forget_gate_bias_init: float = 0.0

    # Attention parameters
    attn_dropout: float = 0.0
    rope_theta: float = 10000.0

    # Shared parameters
    norm_type: str = "rms"
    # "pre_rms" (canonical pre-norm) | "hybrid" (per-projection pre-norms + FFN post-norm
    # post-residual from block 1; the first block stays pre-norm) | "hybrid_bc" ("hybrid" minus
    # the Mamba-1 Delta norm).
    norm_topology: str = "pre_rms"
    use_mlp: bool = True
    mlp_ratio: float = 4.0

    # Training parameters
    max_position_embeddings: int = 2048
    dropout: float = 0.1
    initializer_range: float = 0.02

    # Output head
    tie_word_embeddings: bool = False

    # Memory optimisation
    use_gradient_checkpointing: bool = False

    # Model type identifier
    model_type: str = "lexhybrid"

    def __post_init__(self):
        """Validate and auto-compute derived parameters."""
        if self.num_heads is None:
            self.num_heads = max(1, self.dim // self.head_dim)
        if self.dt_rank is None:
            self.dt_rank = max(1, self.dim // 16)

        for layer_type in self.layer_pattern:
            if layer_type not in VALID_LAYER_TYPES:
                raise ValueError(
                    f"Invalid layer type '{layer_type}'. Must be one of {set(VALID_LAYER_TYPES)}"
                )

        # M1-F: norm_topology used to be unvalidated, so a typo silently behaved as "pre_rms" --
        # the silent-drop class that cost the reference project a run in Phase 9.
        if self.norm_topology not in ("pre_rms", "hybrid", "hybrid_bc"):
            raise ValueError(
                f"norm_topology must be 'pre_rms', 'hybrid' or 'hybrid_bc', got {self.norm_topology!r}"
            )
        if self.scan_impl not in ("legacy", "exact"):
            raise ValueError(f"scan_impl must be 'legacy' or 'exact', got {self.scan_impl!r}")
        if self.tfla_impl not in ("legacy", "exact"):
            raise ValueError(f"tfla_impl must be 'legacy' or 'exact', got {self.tfla_impl!r}")
        if self.mamba3_bc_bias not in ("none", "zero_init", "one_init"):
            raise ValueError(
                f"mamba3_bc_bias must be 'none', 'zero_init' or 'one_init', got {self.mamba3_bc_bias!r}"
            )
        if self.mamba3_a_mode not in ("static", "data_dependent"):
            raise ValueError(
                f"mamba3_a_mode must be 'static' or 'data_dependent', got {self.mamba3_a_mode!r}"
            )
        if self.dt_init_strategy not in ("none", "mamba"):
            raise ValueError(f"dt_init_strategy must be 'none' or 'mamba', got {self.dt_init_strategy!r}")

    def get_layer_config(self, layer_idx: int) -> dict:
        """The per-mixer arguments for layer ``layer_idx`` (a readable summary; not used to build)."""
        layer_type = self.layer_pattern[layer_idx % len(self.layer_pattern)]
        base_config = {
            "dim": self.dim,
            "norm_type": self.norm_type,
            "use_mlp": self.use_mlp,
            "mlp_ratio": self.mlp_ratio,
        }
        if layer_type == "mamba":
            base_config.update(
                {
                    "state_size": self.state_size,
                    "conv_size": self.conv_size,
                    "expand_factor": self.expand_factor,
                    "dt_rank": self.dt_rank,
                    "use_fast_path": self.use_fast_path,
                    "scan_impl": self.scan_impl,
                    "dt_init_strategy": self.dt_init_strategy,
                    "dt_min": self.dt_min,
                    "dt_max": self.dt_max,
                }
            )
        elif layer_type == "mamba3":
            base_config.update(
                {
                    name[len("mamba3_") :]: getattr(self, name)
                    for name in (f.name for f in dataclasses.fields(self))
                    if name.startswith("mamba3_")
                }
            )
            base_config["expand_factor"] = self.expand_factor
        elif layer_type == "mlstm":
            base_config.update(
                {
                    "head_dim": self.head_dim,
                    "num_heads": self.num_heads,
                    "tfla_impl": self.tfla_impl,
                    "proj_factor": self.proj_factor,
                    "gate_soft_cap": self.mlstm_gate_soft_cap,
                    "input_gate_bias_init": self.mlstm_input_gate_bias_init,
                    "forget_gate_bias_init": self.mlstm_forget_gate_bias_init,
                }
            )
        elif layer_type == "attention":
            base_config.update(
                {
                    "num_heads": self.num_heads,
                    "head_dim": self.head_dim,
                    "attn_dropout": self.attn_dropout,
                    "rope_theta": self.rope_theta,
                    "max_position_embeddings": self.max_position_embeddings,
                }
            )
        else:  # pragma: no cover - __post_init__ rejects unknown types first
            raise ValueError(f"Unknown layer type {layer_type!r} at layer {layer_idx}")
        return base_config

    @classmethod
    def from_hydra(cls, model_cfg, **overrides) -> "HybridConfig":
        """Build a config from a Hydra ``cfg.model`` node, keeping every field the dataclass has.

        M2-F / FM5. Every training entry point in the reference used to spell out ~25
        ``field=cfg.model.field`` lines by hand, so a new field silently fell back to its default
        unless all twelve were edited. It happened twice: ``norm_topology`` in Phase 9, and on
        2026-09-06 ``scan_impl``/``tfla_impl``/``dt_init_strategy`` reached the yaml but not the
        model (job 2513007 trained an arm with its defects intact; only the ARCH fingerprint
        caught it). Filtering against ``dataclasses.fields`` removes the class of bug: a field
        added to the dataclass and set in a yaml arrives with no further wiring.

        Keys the dataclass does not know (``learning_rate``, ``warmup_steps``, ...) are ignored,
        because a model yaml also carries training settings.

        Args:
            model_cfg: Hydra ``cfg.model`` node, or any mapping.
            **overrides: applied after the config, for values a caller computes itself.
        """
        valid = {f.name for f in dataclasses.fields(cls)}
        # `model_type` exists on both sides and means different things (the yaml names the
        # Hydra target); keep the dataclass's own value.
        raw = {k: v for k, v in dict(model_cfg).items() if k in valid and k != "model_type"}
        raw.update({k: v for k, v in overrides.items() if k in valid})
        # Hydra hands back ListConfig for sequences; the dataclass validates plain strings.
        if "layer_pattern" in raw and raw["layer_pattern"] is not None:
            raw["layer_pattern"] = [str(x) for x in raw["layer_pattern"]]
        return cls(**raw)

    def to_dict(self) -> dict:
        """Convert the config to a plain dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, config_dict: dict) -> "HybridConfig":
        """Create a config from a dictionary produced by ``to_dict``."""
        return cls(**config_dict)

    def save_pretrained(self, save_directory: str) -> str:
        """Write ``config.json`` into ``save_directory`` and return its path."""
        os.makedirs(save_directory, exist_ok=True)
        path = os.path.join(save_directory, "config.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path
