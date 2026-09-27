"""The port computes exactly what the reference computed (plan P1-Y). Never skipped.

Each fixture in ``tests/fixtures/reference_parity/`` was produced by the REFERENCE code
(``scripts/dump_reference_parity.py``): a state dict, a packed input with document ids, and the
outputs. Here the state dict is loaded strictly into the NEW block or model and every output must
agree to 1e-6 in fp32; cached beam search must produce the identical tokens. P2 fixes defects on
paths these fixtures pin; a fix that moves one of them updates the fixture in the same box, with a
note saying why.
"""

import pytest
import torch

from lexhybrid import HybridConfig, HybridLanguageModel
from lexhybrid.layers.attention_block import AttentionBlock
from lexhybrid.layers.mamba3_block import Mamba3Block
from lexhybrid.layers.mamba_block import MambaBlock
from lexhybrid.layers.mlstm_block import mLSTMBlock
from tests.conftest import REPO_ROOT

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "reference_parity"
TOL = 1e-6
BLOCKS = {"mamba3": Mamba3Block, "mlstm": mLSTMBlock, "attention": AttentionBlock, "mamba": MambaBlock}


def load(name):
    return torch.load(FIXTURES / f"{name}.pt", weights_only=True)


def block_params():
    for name in BLOCKS:
        for i, case in enumerate(load(name)):
            yield pytest.param(name, i, id=f"{name}-{case['label']}")


def model_params():
    for i, case in enumerate(load("model")):
        yield pytest.param(i, id=case["label"])


def test_all_five_fixtures_exist():
    assert sorted(p.name for p in FIXTURES.glob("*.pt")) == [
        "attention.pt",
        "mamba.pt",
        "mamba3.pt",
        "mlstm.pt",
        "model.pt",
    ]


@pytest.mark.parametrize("name,index", list(block_params()))
def test_block_matches_the_reference(name, index):
    case = load(name)[index]
    block = BLOCKS[name](case["input"].shape[-1], **case["kwargs"]).eval()
    block.load_state_dict(case["state_dict"], strict=True)
    with torch.no_grad():
        got_doc = block(case["input"], doc_ids=case["doc_ids"])
        got_nodoc = block(case["input"])
    assert (got_doc - case["output_doc"]).abs().max().item() <= TOL
    assert (got_nodoc - case["output_nodoc"]).abs().max().item() <= TOL
    if case["step_output"] is not None:
        cache = block.allocate_inference_cache(case["input"].shape[0])
        with torch.no_grad():
            stepped = torch.stack(
                [block.step(case["input"][:, t], cache) for t in range(case["input"].shape[1])], dim=1
            )
        assert (stepped - case["step_output"]).abs().max().item() <= TOL


@pytest.mark.parametrize("index", list(model_params()))
def test_model_matches_the_reference(index):
    case = load("model")[index]
    model = HybridLanguageModel(HybridConfig(**case["kwargs"])).eval()
    model.load_state_dict(case["state_dict"], strict=True)
    with torch.no_grad():
        with_doc = model(case["input_ids"], labels=case["input_ids"], doc_ids=case["doc_ids"])
        without_doc = model(case["input_ids"], labels=case["input_ids"])
    assert (with_doc.logits - case["logits_doc"]).abs().max().item() <= TOL
    assert (without_doc.logits - case["logits_nodoc"]).abs().max().item() <= TOL
    assert abs(with_doc.loss.item() - case["loss_doc"].item()) <= TOL
    assert abs(without_doc.loss.item() - case["loss_nodoc"].item()) <= TOL
    if case["beam_tokens"] is not None:
        tokens = model.beam_search_cached(case["beam_prompt"], beam_size=3, max_new_tokens=10)
        assert torch.equal(tokens, case["beam_tokens"])
