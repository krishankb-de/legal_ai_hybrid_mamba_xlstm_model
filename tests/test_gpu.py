"""CUDA checks (plan §8.1 "GPU" layer). Marked ``cuda``: they collect everywhere and skip, with the
reason, on the Mac and in CI; the cluster runs them (``scripts/slurm/gpu_tests.sh``, P4-J1).

P2-J adds the FlexAttention equivalence; P2-Y1 adds the rest of the GPU layer (SDPA flash vs math,
SSD/TFLA under autocast, compile vs eager, cached decode, peak memory).
"""

import pytest
import torch

from lexhybrid import HybridLanguageModel
from lexhybrid.config import load_model_config
from lexhybrid.layers.attention_block import AttentionBlock
from lexhybrid.layers.mamba3_block import Mamba3Block
from lexhybrid.layers.mlstm_block import mLSTMBlock

pytestmark = pytest.mark.cuda


def _packed_doc_ids(batch: int, seq_len: int, device) -> torch.Tensor:
    ids = torch.zeros(batch, seq_len, dtype=torch.long, device=device)
    ids[0, 300:] = 1
    ids[0, 1100:] = 2
    if batch > 1:
        ids[1, 1:] = 1
        ids[1, 700:] = 2
    return ids


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_flex_block_mask_equals_dense_on_gpu(dtype):
    """Defect 18 (P2-J), CUDA variant of tests/test_layers.py::test_flex_block_mask_equals_dense:
    the compiled FlexAttention kernel equals the dense-mask path on packed 2,048-token rows."""
    torch.manual_seed(0)
    dense = AttentionBlock(dim=256, num_heads=4, attn_impl="sdpa").cuda().to(dtype)
    flex = AttentionBlock(dim=256, num_heads=4, attn_impl="flex").cuda().to(dtype)
    flex.load_state_dict(dense.state_dict())
    x = torch.randn(2, 2048, 256, device="cuda", dtype=dtype)
    ids = _packed_doc_ids(2, 2048, "cuda")
    want, got = dense(x, doc_ids=ids), flex(x, doc_ids=ids)
    tol = 1e-4 if dtype is torch.float32 else 3e-2
    assert (got.float() - want.float()).abs().max() / want.float().abs().max() < tol
    # Backward too: the CPU variant cannot check it (no FlexAttention backward on CPU).
    g_want = torch.autograd.grad(want.float().square().sum(), dense.qkv_proj.weight)[0]
    g_got = torch.autograd.grad(got.float().square().sum(), flex.qkv_proj.weight)[0]
    assert (g_got.float() - g_want.float()).abs().max() / g_want.float().abs().max() < 10 * tol


def test_attn_impl_auto_takes_flex_on_gpu():
    block = AttentionBlock(dim=64, num_heads=4).cuda()
    assert block._packed_impl(torch.zeros(1, 4, 8, 16, device="cuda"), 0.0) == "flex"


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


def _reduced_base(**kw):
    torch.manual_seed(0)
    cfg = load_model_config(
        "hybrid_legal_base",
        dim=128,
        num_heads=2,
        head_dim=64,
        mamba3_head_dim=64,
        vocab_size=1024,
        max_position_embeddings=4096,
        **kw,
    )
    return HybridLanguageModel(cfg).cuda().eval()


def test_sdpa_flash_equals_math_bf16():
    """The fused flash kernel and the math reference agree in bf16 on causal attention."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.manual_seed(0)
    block = AttentionBlock(dim=256, num_heads=4).cuda().to(torch.bfloat16).eval()
    x = torch.randn(2, 1024, 256, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            flash = block(x)
        with sdpa_kernel(SDPBackend.MATH):
            math_ = block(x)
    assert _rel(flash, math_) < 2e-2


@pytest.mark.parametrize("kind", ["mamba3", "mlstm"])
def test_ssd_tfla_under_bf16_autocast_match_fp32(kind):
    """SSD and TFLA keep fp32 state under bf16 autocast (P2-E); outputs stay within bf16 noise."""
    torch.manual_seed(0)
    if kind == "mamba3":
        block = Mamba3Block(dim=256, d_state=64, head_dim=64, chunk_size=128).cuda().eval()
    else:
        block = mLSTMBlock(dim=256, head_dim=64, tfla_impl="exact", chunk_size=128).cuda().eval()
    x = torch.randn(2, 1024, 256, device="cuda")
    doc = torch.zeros(2, 1024, dtype=torch.long, device="cuda")
    doc[:, 300:] = 1
    with torch.no_grad():
        want = block(x, doc_ids=doc)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            got = block(x, doc_ids=doc)
    assert _rel(got, want) < 5e-2


def test_compile_equals_eager_on_gpu():
    """P2-Y's CUDA variant: the compiled legal base equals eager at 1e-4 in fp32 (TF32 off)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = _reduced_base()
    ids = torch.randint(0, 1024, (2, 512), device="cuda")
    doc = torch.zeros(2, 512, dtype=torch.long, device="cuda")
    doc[:, 200:] = 1
    with torch.no_grad():
        eager = model(ids, doc_ids=doc).logits
        torch._dynamo.reset()
        got = torch.compile(model, dynamic=False)(ids, doc_ids=doc).logits
    assert (got - eager).abs().max() < 1e-4


def test_cached_decode_on_gpu():
    """Cached greedy equals the uncached reference on GPU in fp32, attention layers included."""
    from lexhybrid.decoding import greedy, greedy_cached

    torch.backends.cuda.matmul.allow_tf32 = False
    model = _reduced_base()
    ids = torch.randint(0, 1024, (3, 64), device="cuda")
    assert torch.equal(greedy_cached(model, ids, max_new_tokens=32), greedy(model, ids, max_new_tokens=32))


def test_peak_memory_slab_loss_on_gpu():
    """The slab loss's peak memory follows the slab, not the row: at the Qwen3 vocabulary the
    materialised logits of one 2,048-token row are 1.24 GB in fp32."""
    from lexhybrid.training.distill import slab_ce_kl

    vocab, length, dim = 151936, 2048, 64
    head = torch.nn.Linear(dim, vocab, bias=False).cuda()
    hidden = torch.randn(1, length, dim, device="cuda", requires_grad=True)
    targets = torch.randint(0, vocab, (1, length), device="cuda")

    def peak(fn):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        fn().backward()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - base

    slab = peak(lambda: slab_ce_kl(hidden, targets, None, head, slab=256)["loss"])
    full = peak(
        lambda: torch.nn.functional.cross_entropy(head(hidden).float().view(-1, vocab), targets.view(-1))
    )
    assert slab < 0.5 * full, (slab, full)


def test_compiled_flex_attention_matches_eager_per_row():
    """Job 2588784: under an outer torch.compile, a block mask traced inside the graph was wrong for
    batch row 1 (drift 0.99). Rows with different document layouts, every row within 1e-4."""
    torch.backends.cuda.matmul.allow_tf32 = False
    model = _reduced_base(layer_pattern=["attention"], num_layers=2, attn_impl="flex")
    ids = torch.randint(0, 1024, (3, 512), device="cuda")
    doc = torch.zeros(3, 512, dtype=torch.long, device="cuda")
    doc[0, 200:] = 1
    doc[1, 64:] = 1
    doc[1, 300:] = 2
    doc[2, 128:] = 1  # boundary on the 128 block edge
    with torch.no_grad():
        eager = model(ids, doc_ids=doc).logits
        torch._dynamo.reset()
        got = torch.compile(model, dynamic=False)(ids, doc_ids=doc).logits
    drift = [(got[b] - eager[b]).abs().max().item() for b in range(3)]
    assert max(drift) < 1e-4, drift
