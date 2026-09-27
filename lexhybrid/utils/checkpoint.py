"""Checkpoint loading: prefix stripping, a guarded ``load_state_dict``, architecture inference.

* ``strip_prefixes`` is the reference ``evaluate_lm.strip_state_dict_prefixes``: torch.compile
  (``_orig_mod.``) and Lightning (``model.``) wrappers, in either order.
* ``load_state_dict_guarded`` is the reference's hard-fail guard: ``strict=False`` loading with the
  wrong yaml once loaded a mismatched model without error and cost more than the whole
  architecture gap in the reference's headline metric. More than 50% missing keys raises; more
  than 5% warns. "Missing keys: 0" is not evidence that the yaml matches -- the ARCH fingerprint
  and ``infer_architecture`` are.
* ``infer_layer_types`` / ``infer_architecture`` read the mixer family of each layer off one
  parameter only that family owns, and refuse ambiguous layers rather than guess.
"""

import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

# (layer type, parameter names -- the FIRST segment after "mixer." -- that only this mixer owns).
FINGERPRINTS: Sequence[tuple[str, tuple[str, ...]]] = (
    ("mamba3", ("dt_bias", "B_bias")),  # Mamba3Block: per-head dt_bias; B_bias when bc_bias != none
    ("mamba", ("dt_proj",)),  # MambaBlock: the Mamba-1 dt projection
    ("mlstm", ("i_gate_proj",)),
    ("attention", ("qkv_proj",)),
)

MISSING_FAIL_FRACTION = 0.5
MISSING_WARN_FRACTION = 0.05


def strip_prefixes(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise ``model._orig_mod.`` / ``_orig_mod.model.`` / ``_orig_mod.`` / ``model.`` / ``lm.`` keys."""
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("model._orig_mod."):
            new_k = k[len("model._orig_mod.") :]
        elif k.startswith("_orig_mod.model."):
            new_k = k[len("_orig_mod.model.") :]
        elif k.startswith("_orig_mod."):
            new_k = k[len("_orig_mod.") :]
        elif k.startswith("model."):
            new_k = k[len("model.") :]
        else:
            new_k = k
        if new_k.startswith("projection_head.") or new_k == "logit_scale":
            continue  # retrieval-era heads, not part of the language model
        if new_k.startswith("lm."):
            new_k = new_k[len("lm.") :]
        cleaned[new_k] = v
    return cleaned


def load_state_dict_guarded(
    model: torch.nn.Module,
    state_dict: Mapping[str, Any],
    fail_fraction: float = MISSING_FAIL_FRACTION,
    warn_fraction: float = MISSING_WARN_FRACTION,
) -> tuple[list[str], list[str]]:
    """``load_state_dict(strict=False)`` that refuses a load missing more than ``fail_fraction`` of keys.

    Returns ``(missing, unexpected)``.
    """
    n_keys = max(1, len(model.state_dict()))
    missing, unexpected = model.load_state_dict(dict(state_dict), strict=False)
    missing_frac = len(missing) / n_keys
    if missing_frac > fail_fraction:
        raise RuntimeError(
            f"{len(missing)}/{n_keys} model keys ({missing_frac:.0%}) are missing from the checkpoint -- "
            f"the model yaml does not match it. First missing: {list(missing)[:5]}"
        )
    if missing_frac > warn_fraction:
        warnings.warn(
            f"{len(missing)}/{n_keys} model keys ({missing_frac:.0%}) missing from the checkpoint: {list(missing)[:5]}",
            RuntimeWarning,
            stacklevel=2,
        )
    return list(missing), list(unexpected)


@dataclass
class InferredArchitecture:
    """What a state dict says about the model that produced it."""

    layer_pattern: list[str]
    norm_topology: str
    num_layers: int
    state_size: int | None = None
    conv_size: int | None = None
    expand_factor: int | None = None
    dt_rank: int | None = None
    mamba3_d_state: int | None = None
    mamba3_head_dim: int | None = None
    mamba3_conv_size: int | None = None
    mamba3_use_conv: bool | None = None

    def size_kwargs(self) -> dict[str, Any]:
        out = {}
        for name in (
            "state_size",
            "conv_size",
            "expand_factor",
            "dt_rank",
            "mamba3_d_state",
            "mamba3_head_dim",
            "mamba3_conv_size",
            "mamba3_use_conv",
        ):
            v = getattr(self, name)
            if v is not None:
                out[name] = v
        return out

    def config_kwargs(self) -> dict[str, Any]:
        out = {
            "layer_pattern": list(self.layer_pattern),
            "norm_topology": self.norm_topology,
            "num_layers": self.num_layers,
        }
        out.update(self.size_kwargs())
        return out


def detect_prefix(state: Mapping[str, Any]) -> str:
    """The ``...layers.`` prefix of a state dict, whatever wrapper stripping left behind."""
    pat = re.compile(r"^(.*?layers\.)\d+\.mixer\.")
    seen = {m.group(1) for k in state for m in [pat.match(k)] if m}
    if len(seen) == 1:
        return seen.pop()
    if not seen:
        raise ValueError(
            f"no '<prefix>layers.<i>.mixer.*' keys in this checkpoint; first keys: {list(state)[:5]}"
        )
    raise ValueError(f"ambiguous layer prefixes {sorted(seen)} -- strip the wrapper first")


def _mixer_params(state: Mapping[str, Any], prefix: str) -> dict[int, dict[str, Any]]:
    pat = re.compile(r"^" + re.escape(prefix) + r"(\d+)\.mixer\.(.+)$")
    layers: dict[int, dict[str, Any]] = {}
    for k, v in state.items():
        m = pat.match(k)
        if m:
            layers.setdefault(int(m.group(1)), {})[m.group(2)] = v
    return layers


def _first_segments(params: Mapping[str, Any]) -> set:
    return {k.split(".")[0] for k in params}


def infer_layer_types(state: Mapping[str, Any], prefix: str | None = None) -> list[str]:
    """Mixer family of every layer, from the one parameter only that family owns."""
    prefix = detect_prefix(state) if prefix is None else prefix
    layers = _mixer_params(state, prefix)
    if not layers:
        raise ValueError(f"no '{prefix}<i>.mixer.*' keys found; first keys: {list(state)[:5]}")
    pattern: list[str] = []
    for i in range(max(layers) + 1):
        names = _first_segments(layers.get(i, {}))
        hits = [t for t, owned in FINGERPRINTS if any(n in names for n in owned)]
        if len(hits) != 1:
            kind = "ambiguous" if hits else "no fingerprint"
            raise ValueError(f"layer {i}: {kind} mixer fingerprint {hits} among parameters {sorted(names)}")
        pattern.append(hits[0])
    return pattern


def infer_architecture(state: Mapping[str, Any], prefix: str | None = None) -> InferredArchitecture:
    """Layer pattern, norm topology and tensor-derived sizes of a checkpoint.

    Cannot recover parameter-invisible flags (``scan_impl``, ``tfla_impl``, ``dt_init_strategy``);
    a loader must pin those from the yaml.
    """
    prefix = detect_prefix(state) if prefix is None else prefix
    layers = _mixer_params(state, prefix)
    pattern = infer_layer_types(state, prefix)
    names_of = {i: _first_segments(layers.get(i, {})) for i in range(len(pattern))}

    # Mamba3Block normalises B/C unconditionally, so only Mamba-1 layers can tell hybrid from
    # hybrid_bc, and only a HybridNorm parameter tells hybrid from pre_rms.
    mamba1 = [i for i, t in enumerate(pattern) if t == "mamba"]
    if any("dt_norm" in names_of[i] for i in mamba1):
        topo = "hybrid"
    elif any("B_norm" in names_of[i] for i in mamba1):
        topo = "hybrid_bc"
    elif any("v_norm" in names_of[i] for i, t in enumerate(pattern) if t == "mlstm") or any(
        "q_norm" in names_of[i] for i, t in enumerate(pattern) if t == "attention"
    ):
        topo = "hybrid"
    else:
        topo = "pre_rms"

    arch = InferredArchitecture(layer_pattern=pattern, norm_topology=topo, num_layers=len(pattern))

    def shape(i: int, sub: str):
        v = layers.get(i, {}).get(sub)
        return tuple(v.shape) if hasattr(v, "shape") else None

    for i, t in enumerate(pattern):
        if t == "mamba" and arch.state_size is None:
            a = shape(i, "A_log")
            op = shape(i, "out_proj.weight")
            cv = shape(i, "conv1d.weight")
            xp = shape(i, "x_proj.weight")
            if a:
                arch.state_size = int(a[1])
            if op:
                arch.expand_factor = int(op[1] // op[0])
            if cv:
                arch.conv_size = int(cv[-1])
            if xp and a:
                arch.dt_rank = int(xp[0] - 2 * a[1])
        elif t == "mamba3" and arch.mamba3_d_state is None:
            bn = shape(i, "B_norm.weight")
            op = shape(i, "out_proj.weight")
            db = shape(i, "dt_bias")
            cv = shape(i, "conv1d.weight")
            if bn:
                arch.mamba3_d_state = int(bn[0])
            if op and db:
                arch.mamba3_head_dim = int(op[1] // db[0])
            arch.mamba3_use_conv = cv is not None
            if cv:
                arch.mamba3_conv_size = int(cv[-1])
    return arch
