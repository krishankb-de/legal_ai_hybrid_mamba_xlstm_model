"""The one-line ``ARCH`` summary every model logs at construction (reference M2-I).

The expensive failure mode is not a crash but training the wrong thing for days: a lever reaches
the yaml and never the mixer. The fingerprint prints at step 0 of every job, so "is this really the
configuration I asked for?" is answered by reading the log. SLURM wrappers grep it against each
arm's expected tokens.
"""


def architecture_fingerprint(model) -> str:
    """``"ARCH layers=[...] | norm_topology=... | ... | params=N | params_nonembed=M"``.

    Every lever an arm can move appears as a ``key=value`` token, so a wrapper can grep an arm's
    expected tokens at step 0. Mixer-specific groups appear only when the mixer is present.
    """
    config = model.config
    types = model.get_layer_types()
    counts = ", ".join(f"{k}x{types.count(k)}" for k in sorted(set(types)))
    parts = [
        f"layers=[{counts}]",
        f"norm_topology={config.norm_topology}",
        f"scan_impl={config.scan_impl}",
        f"tfla_impl={config.tfla_impl}",
        f"dt_init={config.dt_init_strategy}",
        f"vocab={config.vocab_size}",
        f"tied={config.tie_word_embeddings}",
    ]
    if "mamba3" in types:
        parts.append(
            f"mamba3(d_state={config.mamba3_d_state}, head_dim={config.mamba3_head_dim}, "
            f"ngroups={config.mamba3_ngroups}, conv={config.mamba3_use_conv}, "
            f"trapezoid={config.mamba3_use_trapezoid}, rope={config.mamba3_use_rope}, "
            f"bc_bias={config.mamba3_bc_bias}, a_mode={config.mamba3_a_mode}, "
            f"mimo_rank={config.mamba3_mimo_rank}, expand={config.mamba3_expand_factor}, "
            f"chunk={config.mamba3_chunk_size})"
        )
    if "mlstm" in types:
        parts.append(
            f"mlstm(chunk_size={config.mlstm_chunk_size}, forget_bias={config.mlstm_forget_gate_bias_init}, "
            f"fallback={config.tfla_fallback})"
        )
    if "attention" in types:
        attn = next(layer.mixer for layer in model.layers if layer.layer_type == "attention")
        parts.append(
            f"attn(rope_theta={config.rope_theta:g}, impl={config.attn_impl}, "
            f"kv_cache={bool(getattr(attn, 'supports_step', False))})"
        )
    mtp = getattr(model, "mtp_head", None)
    if mtp is not None:
        parts.append(f"mtp(n={config.mtp_n}, weight={config.mtp_loss_weight:g})")
    parts.append(f"params={model.get_num_params(non_embedding=False):,}")
    parts.append(f"params_nonembed={model.get_num_params(non_embedding=True):,}")
    return "ARCH " + " | ".join(parts)
