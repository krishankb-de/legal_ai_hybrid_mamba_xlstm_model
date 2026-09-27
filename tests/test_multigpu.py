"""Multi-GPU checks (plan §8.1 "Multi-GPU" layer). Marked ``multigpu``: they collect everywhere and
skip, with the reason, unless at least two CUDA devices are visible (``scripts/slurm/multigpu_tests.sh``,
P4-J1).

The DDP step uses the settings ``scripts/train_pretrain.py`` builds (``DDPStrategy(
find_unused_parameters=False, gradient_as_bucket_view=True)``): with ``find_unused_parameters``
off, a parameter that receives no gradient makes the second iteration fail, so two iterations
also prove every parameter of the legal base (MTP head included) takes part.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.multigpu

WORLD = 2


def _ddp_worker(rank: int, world: int, port: int, results):
    from torch.nn.parallel import DistributedDataParallel as DDP

    from lexhybrid import HybridLanguageModel
    from lexhybrid.config import load_model_config

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    torch.manual_seed(0)
    cfg = load_model_config(
        "hybrid_legal_base", dim=128, num_heads=2, head_dim=64, mamba3_head_dim=64, vocab_size=1024, mtp_n=2
    )
    model = DDP(
        HybridLanguageModel(cfg).cuda(),
        device_ids=[rank],
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )
    g = torch.Generator().manual_seed(100 + rank)  # every rank sees different data
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        ids = torch.randint(0, 1024, (2, 256), generator=g).cuda()
        doc = torch.zeros_like(ids)
        doc[:, 120:] = 1
        model(ids, labels=ids, doc_ids=doc).loss.backward()
    flat = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    gathered = [torch.empty_like(flat) for _ in range(world)]
    dist.all_gather(gathered, flat)
    if rank == 0:
        results.put(bool(all(torch.equal(gathered[0], other) for other in gathered[1:])))
    dist.barrier()
    dist.destroy_process_group()


def test_ddp_step_gradients_identical_across_ranks():
    """Two processes, different data, one all-reduced gradient: identical on every rank."""
    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    port = 29500 + os.getpid() % 1000
    mp.spawn(_ddp_worker, args=(WORLD, port, results), nprocs=WORLD, join=True)
    assert results.get(timeout=60) is True
