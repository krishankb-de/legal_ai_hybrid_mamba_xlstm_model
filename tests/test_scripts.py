"""The ported command-line scripts run end to end on CPU with tiny models (plan P1-R..T)."""

import json

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
