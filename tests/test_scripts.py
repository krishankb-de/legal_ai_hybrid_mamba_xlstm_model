"""The ported command-line scripts run end to end on CPU with tiny models (plan P1-R..T)."""

import json
import math

import pytest
import torch
import yaml

from lexhybrid import HybridConfig, HybridLanguageModel


@pytest.fixture
def tiny_run(tmp_path):
    """A tiny model yaml and a Lightning-style checkpoint for it."""
    cfg = dict(
        vocab_size=64,
        dim=64,
        num_layers=2,
        layer_pattern=["mamba3", "mlstm"],
        mamba3_d_state=16,
        mamba3_head_dim=32,
        head_dim=32,
        num_heads=2,
        max_position_embeddings=64,
        tfla_impl="exact",
        scan_impl="exact",
        learning_rate=1e-3,
    )
    yaml_path = tmp_path / "tiny.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg))
    torch.manual_seed(0)
    model = HybridLanguageModel(HybridConfig.from_hydra(cfg))
    ckpt = tmp_path / "last.ckpt"
    torch.save(
        {"state_dict": {"model." + k: v for k, v in model.state_dict().items()}, "global_step": 3}, ckpt
    )
    return yaml_path, ckpt, model


def test_evaluate_lm_on_synthetic_data(script, tiny_run, tmp_path):
    yaml_path, ckpt, _ = tiny_run
    results = script("evaluate_lm").main(
        [
            "--checkpoint",
            str(ckpt),
            "--model-config",
            str(yaml_path),
            "--data",
            "synthetic",
            "--max-length",
            "32",
            "--batch-size",
            "2",
            "--synthetic-rows",
            "4",
            "--throughput",
            "--seq-lengths",
            "16",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path / "eval"),
        ]
    )
    assert results["num_tokens_evaluated"] == 4 * 31 and results["perplexity"] > 1.0
    assert json.loads((tmp_path / "eval" / "results.json").read_text())["bits_per_token"] == pytest.approx(
        results["bits_per_token"]
    )


def test_evaluate_lm_refuses_a_checkpoint_of_another_architecture(script, tiny_run, tmp_path):
    _, ckpt, _ = tiny_run
    other = tmp_path / "other.yaml"
    other.write_text(
        yaml.safe_dump(
            dict(
                vocab_size=64,
                dim=64,
                num_layers=2,
                layer_pattern=["attention"],
                num_heads=2,
                max_position_embeddings=64,
            )
        )
    )
    with pytest.raises(RuntimeError, match="does not match"):
        script("evaluate_lm").main(
            ["--checkpoint", str(ckpt), "--model-config", str(other), "--device", "cpu"]
        )


def test_evaluate_lm_reads_a_parquet_shard(script, tiny_run, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    yaml_path, ckpt, _ = tiny_run
    rows = [list(range(i, i + 20)) for i in range(3)]
    pq.write_table(
        pa.table({"input_ids": rows, "doc_ids": [[0] * 10 + [1] * 10] * 3}), tmp_path / "s.parquet"
    )
    results = script("evaluate_lm").main(
        [
            "--checkpoint",
            str(ckpt),
            "--model-config",
            str(yaml_path),
            "--data",
            str(tmp_path / "s.parquet"),
            "--max-length",
            "20",
            "--device",
            "cpu",
        ]
    )
    assert results["num_tokens_evaluated"] == 3 * 19


def test_verify_handoff_pass_and_fail(script, tiny_run, tmp_path):
    yaml_path, ckpt, _ = tiny_run
    mod = script("verify_handoff")
    assert mod.main([str(ckpt), str(yaml_path)]) == 0
    other = tmp_path / "other.yaml"
    other.write_text(
        yaml.safe_dump(
            dict(
                vocab_size=64,
                dim=64,
                num_layers=3,
                layer_pattern=["mamba3", "mlstm"],
                mamba3_d_state=16,
                mamba3_head_dim=32,
                head_dim=32,
                num_heads=2,
            )
        )
    )
    assert mod.main([str(ckpt), str(other)]) == 1


def test_check_operator_equivalence_passes_on_cpu(script):
    rc = script("check_operator_equivalence").main(
        ["--device", "cpu", "--chunk-sizes", "32", "128", "--seq-length", "160", "--dim", "64"]
    )
    assert rc == 0


def test_check_operator_equivalence_sweeps_only_the_mamba3_chunk(script, monkeypatch, capsys):
    """Job 2589360: ref_hybrid_m3 runs Mamba-3 at chunk 64 and mLSTM at 128. The sweep set every
    mixer's chunk_size, so "chunk 64 vs the chunk-64 baseline" moved the mLSTM to 64 and failed at
    4.5e-4, and the reset left it there for the compile check. Re-running the baseline chunk must
    reproduce the baseline exactly, and the mLSTM chunk must not move."""
    mod = script("check_operator_equivalence")
    cfg = HybridConfig(
        vocab_size=64,
        dim=64,
        num_layers=2,
        layer_pattern=["mamba3", "mlstm"],
        mamba3_d_state=16,
        mamba3_head_dim=32,
        num_heads=2,
        head_dim=32,
        max_position_embeddings=128,
        tfla_impl="exact",
        mamba3_chunk_size=16,
        mlstm_chunk_size=32,
    )
    monkeypatch.setattr(mod, "load_model_config", lambda name, **kw: HybridConfig.from_dict(cfg.to_dict()))
    assert mod.check_model("cpu", "tiny", [16], 96, do_compile=False) == []
    assert "chunk_size= 16 logits vs baseline : 0.000e+00" in capsys.readouterr().out
    model = HybridLanguageModel(HybridConfig.from_dict(cfg.to_dict()))
    mod.set_mamba3_chunk_size(model, 8)
    assert [layer.mixer.chunk_size for layer in model.layers] == [8, 32]


def test_performance_profile_single_point_and_sweep(script, tmp_path):
    pp = script("performance_profile")
    cfg = HybridConfig(
        vocab_size=64,
        dim=64,
        num_layers=2,
        layer_pattern=["mamba3", "attention"],
        mamba3_d_state=16,
        mamba3_head_dim=32,
        num_heads=2,
        max_position_embeddings=128,
    )
    res = pp.profile_model(cfg, batch_size=1, seq_length=32, num_iterations=2, device="cpu")
    assert res["oom"] is False and res["tokens_per_s"] > 0
    assert pp.fit_log_slope([1, 2, 4], [1, 2, 4]) == pytest.approx(1.0)
    assert pp.fit_log_slope([1], [1]) is None


def test_packed_doc_ids_shift_the_layout_per_row(script):
    pp = script("performance_profile")
    ids = pp.packed_doc_ids(4, 40, 10, "cpu")
    assert ids.shape == (4, 40) and ids.dtype == torch.long
    starts = [(r[1:] != r[:-1]).nonzero().flatten().add(1).tolist() for r in ids]
    assert starts[0] == [10, 20, 30]
    assert all(len(s) >= 3 and all(b - a == 10 for a, b in zip(s, s[1:])) for s in starts)
    assert len({tuple(s) for s in starts}) == 4, "every row of the batch is packed differently"
    with pytest.raises(ValueError):
        pp.packed_doc_ids(1, 8, 0, "cpu")


def test_performance_profile_packed_training_rows_with_the_slab_loss(script, tmp_path, monkeypatch):
    """P4-P's training rows: packed doc_ids, the slab-wise loss (no (B, L, V) logits), both chunk
    sizes read back off the built module and written to every row."""
    pp = script("performance_profile")
    cfg = HybridConfig(
        vocab_size=64,
        dim=64,
        num_layers=3,
        layer_pattern=["mamba3", "attention", "mlstm"],
        mamba3_d_state=16,
        mamba3_head_dim=32,
        num_heads=2,
        head_dim=32,
        max_position_embeddings=128,
        tfla_impl="exact",
        mamba3_chunk_size=16,
        mlstm_chunk_size=16,
    )
    monkeypatch.setattr(pp, "load_config", lambda name: HybridConfig.from_dict(cfg.to_dict()))
    rows, _ = pp.run_sweep(
        ["tiny"], [32, 64], [2], 1, "cpu", torch.float32, True, tmp_path,
        chunk_size=8, mlstm_chunk_size=8, doc_len=20, loss="slab",
    )  # fmt: skip
    assert [r["oom"] for r in rows] == [False, False]
    for r in rows:
        assert (r["effective_chunk_size"], r["effective_mlstm_chunk_size"]) == (8, 8)
        assert (r["loss"], r["doc_len"], r["attn_path"]) == ("slab", 20, "packed-sdpa")
    header = (tmp_path / "efficiency_curves.csv").read_text().splitlines()[0].split(",")
    assert {"loss", "doc_len", "attn_path", "effective_mlstm_chunk_size"} <= set(header)
    unpacked, _ = pp.run_sweep(["tiny"], [32], [1], 1, "cpu", torch.float32, False, None)
    assert unpacked[0]["attn_path"] == "causal-sdpa" and unpacked[0]["loss"] == ""
    with pytest.raises(ValueError):
        pp.measure_point(None, 1, 8, 1, "cpu", 64, backward=True, loss="mean")


def test_slab_training_step_is_the_pretrain_loss(script):
    """The profiler's slab step computes what PretrainLightningModule trains on: the model's own
    boundary-masked CE."""
    from lexhybrid.models.hybrid_lm import boundary_masked_labels
    from lexhybrid.training.distill import slab_ce_kl

    pp = script("performance_profile")
    cfg = HybridConfig(
        vocab_size=64, dim=32, num_layers=2, layer_pattern=["mamba3", "mlstm"], mamba3_d_state=16,
        mamba3_head_dim=16, num_heads=2, head_dim=16, max_position_embeddings=64, tfla_impl="exact",
        mlstm_chunk_size=8,
    )  # fmt: skip
    model = HybridLanguageModel(cfg).eval()
    ids = torch.randint(0, 64, (2, 24))
    doc = pp.packed_doc_ids(2, 24, 10, "cpu")
    residual, _ = model.backbone(ids, doc_ids=doc)
    slab = slab_ce_kl(residual[:, :-1], boundary_masked_labels(ids, doc), None, model.head, slab=8)["loss"]
    assert slab.item() == pytest.approx(model(ids, labels=ids, doc_ids=doc).loss.item(), rel=1e-5)


def test_bootstrap_paired_ci_sign(script, tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text("\n".join(json.dumps({"citation_precision": 1.0}) for _ in range(40)) + "\n")
    b.write_text("\n".join(json.dumps({"citation_precision": float(i % 2)}) for i in range(40)) + "\n")
    mod = script("bootstrap_compare")
    cache_a = mod.build_cache(None, None, mod.read_metrics(str(a)))
    cache_b = mod.build_cache(None, None, mod.read_metrics(str(b)))
    results, meta = mod.paired_bootstrap(cache_a, cache_b, n_samples=300, seed=0)
    r = results["citation_precision"]
    assert r["diff"] == pytest.approx(0.5) and r["significant"] and r["ci_low"] > 0 and meta["n"] == 40
    assert mod.main(["--metrics-a", str(a), "--metrics-b", str(b), "--bootstrap-samples", "50"]) == 0
    with pytest.raises(SystemExit):
        mod.main(["--hyps-a", str(a)])


def test_analyze_generation_diversity(script, tmp_path):
    hyps = tmp_path / "h.txt"
    hyps.write_text("\n".join(["the claim fails"] * 8 + ["another answer here"] * 2) + "\n")
    mod = script("analyze_generation_diversity")
    res = mod.analyse("generated (model)", mod.read_lines(str(hyps)))
    assert res["duplicate_clusters"] == 2 and res["pct_in_duplicate_cluster"] == pytest.approx(100.0)
    assert mod.main(["--hyps", str(hyps), "--output", str(tmp_path / "d.md")]) == 0


# -- screen arms (P2-X) ---------------------------------------------------------------------


def test_screen_arms_verify_reduced(script):
    """P2-X: every arm builds through HybridConfig.from_hydra at reduced width, runs a packed
    forward/backward, and shows its expected ARCH tokens (and none of its forbidden ones)."""
    arms = script("screen_arms")
    assert set(arms.ARMS) == {"S0", "S1", "S2", "S3", "S4", "S5", "S6"}
    assert arms.main(["verify", "--reduced"]) == 0


def test_screen_arms_verify_full_passes_with_the_decided_s5_band(script, capsys):
    """Full size on `meta` with the bands: every arm passes; S5 carries +2.36% as a stated difference
    inside ±2.5% (user decision 2026-09-27, plan §14)."""
    arms = script("screen_arms")
    assert arms.verify(reduced=False, full=True) == []
    assert arms.ARMS["S5"].band_pct == 2.5
    assert "S5: non-embedding 106,527,072 (+2.36%, band +-2.5%) ok" in capsys.readouterr().out


def test_screen_arm_env_is_what_the_wrapper_evals(script):
    import shlex

    arms = script("screen_arms")
    env = dict(line.removeprefix("export ").split("=", 1) for line in arms.env_lines("S6-s43"))
    assert env["SEED"] == "43" and env["EXPERIMENT"] == "screen2_S6_s43" and env["SAVE_TOP_K"] == "0"
    overrides = shlex.split(shlex.split(env["EXTRA_OVERRIDES"])[0])
    assert "distill.alpha=0.0" in overrides and "trainer.max_steps=12000" in overrides
    assert "callbacks.checkpoint.save_top_k=0" in overrides
    assert arms.env_lines("S1") == arms.env_lines("S1-s42")
    with pytest.raises(KeyError):
        arms.env_lines("S9-s42")
    with pytest.raises(ValueError):
        arms.env_lines("S1-s7")


def test_screen_arm_expect_tokens_catch_a_lever_that_did_not_arrive(script):
    """The FM5 defence: if an arm's lever never reaches the model, its ARCH line lacks the token."""
    arms = script("screen_arms")
    import torch

    from lexhybrid import HybridLanguageModel

    with torch.device("meta"):
        base = HybridLanguageModel(arms.build_config("S1"))
    assert arms.check_tokens("S3", base.architecture_fingerprint()) == ["S3: ARCH lacks 'mamba3(d_state=64'"]
    assert (
        arms.check_tokens("S0", base.architecture_fingerprint())[-1] == "S0: ARCH shows forbidden 'attention'"
    )


# -- tokenizer fertility (P3-D) -----------------------------------------------------------------


def test_fertility_sampling_and_arithmetic(script):
    """P3-D's method as pre-registered: whole documents until the word budget, documents over the cap
    skipped; tokens/word per source and pooled word-weighted."""
    from types import SimpleNamespace

    mf = script("measure_fertility")
    docs = [SimpleNamespace(text=" ".join(["w"] * n)) for n in (4, 100, 3, 5, 2)]
    assert [len(d.text.split()) for d in mf.take(docs, words=8, max_doc_words=50)] == [4, 3, 5]
    rows = mf.measure({"a": ["x y", "z"], "b": ["p q r s"]}, {"chars": lambda t: list(t.replace(" ", ""))})
    assert [(r["source"], r["words"], r["tokens"], r["fertility"]) for r in rows] == [
        ("a", 3, 3, 1.0),
        ("b", 4, 4, 1.0),
    ]
    pooled = mf.pooled(rows, ("a", "b"), "chars")
    assert (pooled["words"], pooled["tokens"], pooled["docs"]) == (7, 7, 3)


def test_fertility_reads_every_collector_fixture_offline(script):
    mf = script("measure_fertility")
    texts = mf.fixture_texts()
    assert set(texts) == set(mf.LEGAL)
    assert all(texts[s] and all(t.strip() for t in texts[s]) for s in mf.LEGAL)


def _tb_run(root, exp, version, scalars):
    """Write ``{tag: [(step, value), ...]}`` as TensorBoard events where TensorBoardLogger puts them."""
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=str(root / exp / "logs" / exp / f"version_{version}"))
    for tag, points in scalars.items():
        for step, value in points:
            writer.add_scalar(tag, value, step)
    writer.close()


def test_screen_metrics_reads_val_and_curves_from_the_latest_version(tmp_path, script, capsys):
    """P5-D/F: val PPL is only in TensorBoard; the reader takes the run's highest version (the
    earlier one is a failed attempt), reports exp(val/loss) as the PPL (the logged val/perplexity is a
    batch-mean of exp(CE), job 2636358) and averages the training curve per window."""
    sm = script("screen_metrics")
    _tb_run(tmp_path, "screen_S1_s42", 0, {"train/loss": [(25, 99.0)]})
    _tb_run(
        tmp_path,
        "screen_S1_s42",
        1,
        {
            "train/loss": [(25, 4.0), (50, 2.0), (125, 9.0)],
            "train/grad_norm": [(25, 1.0), (50, 3.0), (125, 0.5)],
            "train/nan_skips": [(25, 0.0), (125, 2.0)],
            "val/loss": [(100, 2.0), (200, 1.5)],
            "val/perplexity": [(100, 9.5), (200, math.exp(1.5))],  # a batch-mean of exp(CE)
        },
    )
    _tb_run(tmp_path, "screen_S0_s42", 0, {"val/loss": [(100, 3.0)]})
    (tmp_path / "screen_S2_s42").mkdir()
    (tmp_path / "other_run").mkdir()
    assert sm.main(["--outputs", str(tmp_path), "--window", "100"]) == 0
    lines = capsys.readouterr().out.splitlines()
    s1 = [ln for ln in lines if " screen_S1_s42 " in ln]
    assert s1[0] == "RUN screen_S1_s42 version=1 last_step=200"
    assert "VAL screen_S1_s42 step=100 val_loss=2.0000 val_ppl=7.3891 logged_ppl=9.5000" in s1
    assert "VAL screen_S1_s42 step=200 val_loss=1.5000 val_ppl=4.4817 logged_ppl=4.4817" in s1
    assert (
        "CURVE screen_S1_s42 steps=0-99 loss=3.0000 grad_norm=2.0000 grad_norm_max=3.0000 nan_skips=0.0000"
        in s1
    )
    assert (
        "CURVE screen_S1_s42 steps=100-199 loss=9.0000 grad_norm=0.5000 grad_norm_max=0.5000 nan_skips=2.0000"
        in s1
    )
    assert "VAL screen_S0_s42 step=100 val_loss=3.0000 val_ppl=20.0855" in lines  # nothing logged
    assert "RUN screen_S2_s42 version=none (no events file)" in lines
    assert not any("other_run" in ln for ln in lines)


def test_screen_metrics_fails_without_runs(tmp_path, script):
    assert script("screen_metrics").main(["--outputs", str(tmp_path)]) == 1


def test_screen_metrics_dumps_matching_tags_in_a_step_range(tmp_path, script, capsys):
    """Spike diagnosis: --dump prints every value of the matching tags, limited to --steps."""
    _tb_run(
        tmp_path,
        "diag_S1_s42",
        0,
        {"diag/grad/L1_attention": [(1, 0.5), (5, 9.0), (9, 0.4)], "train/loss": [(5, 3.0)]},
    )
    sm = script("screen_metrics")
    assert (
        sm.main(["--outputs", str(tmp_path), "--prefix", "diag_", "--dump", "^diag/", "--steps", "2-9"]) == 0
    )
    dumped = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("SCALAR")]
    assert dumped == [
        "SCALAR diag_S1_s42 diag/grad/L1_attention step=5 value=9.0000",
        "SCALAR diag_S1_s42 diag/grad/L1_attention step=9 value=0.4000",
    ]


def _screen_outputs(tmp_path, runs):
    """Fake job outputs for screen_table.py: a reader log, training logs, eval JSON.
    runs = {(arm, seed): (val_loss, late_doc, statute_em, multihop_em, [(s_per_step, n), ...])}."""
    import json
    import math

    val = tmp_path / "metrics.log"
    val.write_text("".join(f"VAL screen2_{a}_s{s} step=11999 val_loss={v[0]:.4f} val_ppl={math.exp(v[0]):.4f}\n" for (a, s), v in runs.items())
                   + "VAL screen_S1_s42 step=11999 val_loss=9.0 val_ppl=0\n")  # fmt: skip
    (tmp_path / "eval").mkdir()
    train = []
    for (a, s), (_, late, em, mh, legs) in runs.items():
        name = f"screen2_{a}_s{s}"
        (tmp_path / "eval" / f"{name}.valppl.json").write_text(json.dumps(
            {"ppl": 1.0, "late_doc_ppl": late, "late_row_ppl": late + 1, "late_doc_tokens": 10, "late_row_tokens": 20}))  # fmt: skip
        (tmp_path / "eval" / f"{name}.probes.json").write_text(json.dumps(
            {"mqar": {"accuracy": 0.5}, "statute_recall": {"exact_match": em, "char_f1": em + 0.2, "items": 100},
             "multihop": {"exact_match": mh, "items": 200}}))  # fmt: skip
        for k, (sps, n) in enumerate(legs):
            log = tmp_path / f"train_{name}_{k}.log"
            log.write_text(f"experiment: {name}  model: x\nSTEPSTATS rank=0 step=500 s_per_step=99 n=1 peak_alloc_gb=1.00 "
                           f"peak_reserved_gb=1 final=0\nSTEPSTATS rank=0 step=12000 s_per_step={sps} n={n} "
                           f"peak_alloc_gb={40 + k}.00 peak_reserved_gb=1 final={0 if len(legs) > 1 and k == 0 else 1}\n")  # fmt: skip
            train.append(str(log))
    return ["--val-logs", str(val), "--train-logs", *train, "--eval-dirs", str(tmp_path / "eval")]


def test_screen_table_applies_the_pre_registered_rule(tmp_path, script, capsys):
    """P5-F/H/I: columns come from the job outputs (val PPL = exp(val_loss) at step 11,999; a
    continuation's s/step is step-weighted), and the rule is applied as written: S1 advances unless
    an arm beats its mean val PPL by more than the bar without losing statute-recall or multi-hop by
    more than the bar; a PPL win that loses recall is recorded, not adopted."""
    import math

    st = script("screen_table")
    runs = {
        ("S1", 42): (1.70, 9.0, 0.30, 0.40, [(2.8, 100)]),
        ("S1", 43): (1.70, 9.2, 0.32, 0.40, [(2.8, 100)]),
        ("S2", 42): (1.65, 9.0, 0.30, 0.40, [(2.7, 100)]),  # PPL lower by 0.267 and recall kept: adopt
        ("S2", 43): (1.65, 9.0, 0.30, 0.40, [(2.7, 100)]),
        ("S3", 42): (1.65, 9.0, 0.10, 0.40, [(2.7, 100)]),  # wins PPL, loses statute-recall by 0.21
        ("S3", 43): (1.65, 9.0, 0.10, 0.40, [(2.7, 100)]),
        ("S4", 42): (1.69, 9.0, 0.30, 0.40, [(3.0, 100)]),  # PPL lower by ~0.05 < bar
        ("S4", 43): (1.69, 9.0, 0.30, 0.40, [(3.0, 100)]),
        ("S5", 42): (
            2.0,
            9.0,
            0.30,
            0.40,
            [(9.0, 8000), (10.0, 4000)],
        ),  # two legs, the first killed (no final=1)
    }
    assert st.main(["--bar", "0.10", *_screen_outputs(tmp_path, runs)]) == 0
    out = capsys.readouterr().out
    s1 = next(ln for ln in out.splitlines() if ln.startswith("| S1 | 42 |")).split(" | ")
    assert s1[2] == f"{math.exp(1.70):.4f}" and s1[3] == "9.0000" and s1[6] == "0.300" and s1[-2] == "2.8000"
    s5 = next(ln for ln in out.splitlines() if ln.startswith("| S5 | 42 |")).split(" | ")
    assert s5[-2] == f"{(9.0 * 8000 + 10.0 * 4000) / 12000:.4f}" and s5[-1] == "41.00 |"
    assert "| S1 | 5.4739 ± 0.0000 | 9.1000 ± 0.1414 |" in out  # two-seed mean ± sample SD
    rule = [ln for ln in out.splitlines() if ln.startswith("RULE")]
    assert (
        "S2: val PPL S1-S2 = +0.2669" in rule[0]
    )  # 5.4739 - 5.2070 and rule[0].endswith("candidate: adopt")
    assert (
        rule[1].startswith("RULE S3:")
        and "statute-recall loss +0.2100" in rule[1]
        and rule[1].endswith("not adopted")
    )
    assert rule[2].endswith("does not beat S1 by more than the bar")
    assert rule[3] == "RULE S5: incomplete (val_ppl, statute_em, multihop_em missing), not decided"
    assert rule[4] == "RULE winner: S2"
    assert "screen_S1_s42" not in out  # attempt 1 is not read under the screen2_ prefix


def test_screen_table_never_decides_without_the_control(tmp_path, script, capsys):
    st = script("screen_table")
    assert (
        st.main(["--bar", "0.10", *_screen_outputs(tmp_path, {("S2", 42): (1.6, 9.0, 0.3, 0.4, [(2.7, 1)])})])
        == 0
    )
    assert (
        "RULE incomplete: S1 lacks two seeds of val_ppl, statute_em, multihop_em" in capsys.readouterr().out
    )
