"""The one-line ``ARCH`` summary every model logs at construction (reference M2-I).

The expensive failure mode is not a crash but training the wrong thing for days: a lever reaches
the yaml and never the mixer. The fingerprint prints at step 0 of every job, so "is this really the
configuration I asked for?" is answered by reading the log. SLURM wrappers grep it against each
arm's expected tokens.
"""


def architecture_fingerprint(model) -> str:
    """``"ARCH layers=[...] | norm_topology=... | ... | params=N"`` for a ``HybridLanguageModel``."""
    config = model.config
    types = model.get_layer_types()
    counts = ", ".join(f"{k}x{types.count(k)}" for k in sorted(set(types)))
    parts = [
        f"layers=[{counts}]",
        f"norm_topology={config.norm_topology}",
        f"scan_impl={config.scan_impl}",
        f"tfla_impl={config.tfla_impl}",
        f"dt_init={config.dt_init_strategy}",
    ]
    if "mamba3" in types:
        parts.append(
            f"mamba3(d_state={config.mamba3_d_state}, head_dim={config.mamba3_head_dim}, ngroups={config.mamba3_ngroups}, conv={config.mamba3_use_conv}, trapezoid={config.mamba3_use_trapezoid}, rope={config.mamba3_use_rope}, "
            f"bc_bias={config.mamba3_bc_bias}, a_mode={config.mamba3_a_mode}, mimo_rank={config.mamba3_mimo_rank})"
        )
    parts.append(f"params={sum(p.numel() for p in model.parameters()):,}")
    return "ARCH " + " | ".join(parts)
