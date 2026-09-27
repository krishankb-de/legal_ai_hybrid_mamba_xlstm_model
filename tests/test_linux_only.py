"""Checks that need Linux (plan §8.3): they run in CI and are skipped, with the reason, on the Mac."""

import pytest
import torch

from lexhybrid import HybridConfig, HybridLanguageModel


@pytest.mark.linux_only
@pytest.mark.slow
def test_inductor_cpu_compile_matches_eager():
    """torch.compile on the Inductor CPU backend (gcc) reproduces the eager logits.

    The reference could only ever test compile on the H100; this proves the compile path on every
    push. P2-Y extends it to the legal model at chunk 128 and asserts the TFLA fast path.
    """
    torch.manual_seed(0)
    cfg = HybridConfig(
        vocab_size=128,
        dim=64,
        num_layers=3,
        layer_pattern=["mamba3", "mlstm", "attention"],
        mamba3_d_state=16,
        mamba3_head_dim=32,
        head_dim=32,
        num_heads=2,
        tfla_impl="exact",
        max_position_embeddings=128,
        dropout=0.0,
    )
    model = HybridLanguageModel(cfg).eval()
    ids = torch.randint(0, 128, (2, 96))
    with torch.no_grad():
        eager = model(ids).logits
        compiled = torch.compile(model)(ids).logits
    rel = ((compiled - eager).abs().max() / eager.abs().max()).item()
    assert rel <= 1e-4, f"compiled logits differ from eager by rel {rel:.3e}"
