"""Preference optimisation: pair selection and balance, the answer-token log-probabilities, the
incremental pair build, resuming a run, the Hugging Face baseline, and the CPU path from sampling to
pairs to DPO/KTO optimizer steps."""

import argparse
import json
import math
import shutil
from itertools import count
from types import SimpleNamespace

import pytest
import torch

from lexhybrid import HybridConfig, HybridLanguageModel
from lexhybrid.data.datasets import _unit
from lexhybrid.data.sft_format import Cite, Example, Passage, Quote, Text, render
from lexhybrid.data.tokenizer import BASE_LEN, EOS_ID, SPECIALS
from lexhybrid.training.dpo_module import (
    DPOLightningModule,
    PairSchedule,
    answer_logps,
    collate_pairs,
    load_sft_model,
)
from lexhybrid.verifier.verifier import verify
from tests.conftest import load_script
from tests.test_pipeline import ByteTokenizer
from tests.test_training import _compose

B = load_script("build_dpo_pairs")
EOS = "<|endoftext|>"
NOTHING = (frozenset(), frozenset())
GOLD = (frozenset({1}), frozenset({(1, 2)}))  # the gold target quotes sentence 2 of passage 1


def _s(markup: str, refused: bool, claims=(True,)) -> dict:
    verdict = {"refused": refused, "reason": None, "citation_precision": None,
               "claims": [{"index": i, "pointers": [], "passed": p} for i, p in enumerate(claims)]}  # fmt: skip
    return {"tokens": list(markup.encode()), "markup": markup, "verifier": verdict}


KEPT = _s(f"Ja <|cite|><|c1|>.{EOS}", refused=False)
UNSUPPORTED = _s(f"Ja.{EOS}", refused=True, claims=(False,))
REFUSAL = _s(B.REFUSAL_MARKUP, refused=True, claims=())


# -- pair selection ------------------------------------------------------------------------------


def test_a_good_answer_keeps_every_claim_and_points_into_a_gold_passage():
    """A kept answer that lost a secondary claim, a quote of a passage the gold target does not cite,
    a truncated answer and a refusal are all bad; the rest are good."""
    lost_a_claim = _s(f"Ja <|cite|><|c1|>. Nein.{EOS}", refused=False, claims=(True, False))
    off_topic = _s(f"So <|q|><|c2|><|s1|>.{EOS}", refused=False)
    truncated = _s("Ja <|cite|><|c1|>.", refused=False)
    samples = [REFUSAL, lost_a_claim, off_topic, truncated, KEPT]
    assert [B.is_good(s, False, GOLD) for s in samples] == [False, False, False, False, True]


def test_the_good_rule_on_the_real_verifiers_verdicts():
    """Through ``verify`` (stub NLI): an uncited second claim leaves the answer kept with citation
    precision 1.0, and a verbatim quote of an off-topic passage is kept and entailed; neither is
    good. An answer whose every claim passes, pointing into the gold passage, is."""
    passages = (Passage("BGB §1", ("Satz eins.", "Satz zwei.")), Passage("BGB §2", ("Etwas anderes.",)))
    item = {"id": "x", "source": "t", "kind": "answerable",
            "text": "".join(render(Example("Frage?", passages, (Text("Ja: "), Quote(1, 2)))))}  # fmt: skip
    chunks = [B.passage_chunk(item, k, p) for k, p in enumerate(passages, start=1)]
    gold = B.gold_pointers(item)
    assert gold == (frozenset({1}), frozenset({(1, 2)}))

    def sample(markup):
        verdict = verify("Frage?", markup, chunks, lambda premise, hypothesis: 1.0).log_row()
        return {"tokens": [], "markup": markup, "verifier": verdict}

    uncited = sample(f"Ja <|cite|><|c1|>. Noch etwas.{EOS}")
    assert not uncited["verifier"]["refused"] and uncited["verifier"]["citation_precision"] == 1.0
    assert not B.is_good(uncited, False, gold)
    off_topic = sample(f"So: <|q|><|c2|><|s1|>{EOS}")
    assert not off_topic["verifier"]["refused"] and not B.is_good(off_topic, False, gold)
    assert B.is_good(sample(f"Ja: <|q|><|c1|><|s2|>{EOS}"), False, gold)
    assert B.gold_pointers(
        {"text": "".join(render(Example("Frage?", passages, (Text("Ja "), Cite(2)))))}
    ) == (
        frozenset({2}),
        frozenset(),
    )


def test_chosen_overlaps_the_gold_pointers_most_and_rejected_is_the_first_failure():
    """Sentences count before passages, then draw order; the rejected sample is the first bad one."""
    cite = _s(f"Ja <|cite|><|c1|>.{EOS}", refused=False)
    other_sentence = _s(f"Ja <|q|><|c1|><|s1|>.{EOS}", refused=False)
    gold_quote = _s(f"Ja <|q|><|c1|><|s2|>.{EOS}", refused=False)
    late_quote = _s(f"Doch <|q|><|c1|><|s2|>.{EOS}", refused=False)
    assert B.select_pair([REFUSAL, cite, UNSUPPORTED, gold_quote, late_quote], False, GOLD) == (
        gold_quote,
        REFUSAL,
    )
    assert B.select_pair([REFUSAL, cite, other_sentence], False, GOLD) == (cite, REFUSAL)


def test_a_should_refuse_prompt_pairs_the_models_own_refusal_with_an_attempted_answer():
    """Only the exact ``<|unanswerable|>`` is a good refusal: a kept answer is bad, and so is one the
    verifier refuses for it (a party question refuses every answer)."""
    assert B.select_pair([KEPT, UNSUPPORTED, REFUSAL], True, NOTHING) == (REFUSAL, KEPT)
    assert not B.is_good(_s(f" <|unanswerable|> {EOS}", refused=True), True, NOTHING)
    assert not B.is_good(UNSUPPORTED, True, NOTHING) and not B.is_good(REFUSAL, False, GOLD)


def test_a_prompt_without_a_good_or_a_bad_sample_gives_no_pair():
    assert B.select_pair([KEPT, KEPT], False, GOLD) is None
    assert B.select_pair([UNSUPPORTED, REFUSAL], False, GOLD) is None
    assert B.select_pair([REFUSAL], True, NOTHING) is None


def _pairs(kinds):
    return [{"id": f"q{i}", "kind": k} for i, k in enumerate(kinds)]


def test_balance_cuts_the_over_represented_side_to_the_should_refuse_share():
    """Target 1/4 of refusal pairs: 1 refusal + 6 answer pairs keep 3 answer pairs (share 1/4); 3 refusal
    + 1 answer pair keep 1 + 1 (rounded towards keeping pairs: a side never empties the other while it
    has pairs). Order is kept, the subsample is seeded, and a one-sided input is left alone."""
    many_answers = _pairs(["answerable"] * 3 + ["party"] + ["answerable"] * 3)
    kept = B.balance(many_answers, refuse_prompts=2, prompts=8, seed=0)
    assert [p["kind"] for p in kept].count("party") == 1 and len(kept) == 4
    assert kept == sorted(kept, key=lambda p: int(p["id"][1:]))
    assert kept == B.balance(many_answers, 2, 8, seed=0)
    assert {p["id"] for s in range(5) for p in B.balance(many_answers, 2, 8, seed=s)} != {
        p["id"] for p in kept
    }

    many_refusals = _pairs(["unanswerable", "false_premise", "party", "answerable"])
    assert sorted(p["kind"] in B.REFUSE_KINDS for p in B.balance(many_refusals, 1, 4)) == [False, True]
    assert B.balance(_pairs(["answerable"] * 3), refuse_prompts=1, prompts=4) == []
    assert B.balance(_pairs(["answerable"] * 3), refuse_prompts=0, prompts=4) == _pairs(["answerable"] * 3)


def _item(i, kind) -> dict:
    answer = (Text("Ja "), Cite(1)) if kind == "answerable" else ()
    passage = Passage("BGB §1", ("Satz eins.", "Satz zwei."))
    return {"id": f"q{i}", "question_id": f"Q{i}", "source": "s", "kind": kind,
            "text": "".join(render(Example(f"Frage {i}?", (passage,), answer)))}  # fmt: skip


def test_build_pairs_keeps_one_pair_per_prompt_balances_and_counts():
    drawn = [(_item(i, "answerable"), [i], [UNSUPPORTED, KEPT, KEPT]) for i in range(6)]
    drawn += [
        (_item(6, "party"), [6], [KEPT, REFUSAL]),
        (_item(7, "unanswerable"), [7], [REFUSAL]),  # no bad sample
        (_item(8, "answerable"), [8], [REFUSAL, UNSUPPORTED]),  # no good sample
    ]
    pairs, stats = B.build_pairs(drawn, seed=0)
    assert stats == {
        "prompts": 9, "should_refuse_prompts": 2, "distinct_samples": 23, "skipped_no_good": 1,
        "skipped_no_bad": 1, "refusal_pairs_before_balance": 1, "answer_pairs_before_balance": 6,
        "dropped_for_balance": 2, "pairs": 5, "refusal_pairs": 1,
        "refusal_share": 0.2, "target_refusal_share": 2 / 9,
    }  # fmt: skip
    assert len({p["id"] for p in pairs}) == len(pairs)
    party = next(p for p in pairs if p["kind"] == "party")
    assert (party["chosen"], party["rejected"], party["prompt_ids"]) == (
        B.REFUSAL_MARKUP,
        KEPT["markup"],
        [6],
    )
    assert party["chosen_ids"] == REFUSAL["tokens"] and party["chosen_verifier"]["refused"]
    with pytest.raises(ValueError, match="kind"):
        B.build_pairs([(_item(9, "yes"), [9], [KEPT])])


# -- answer-token log-probabilities -------------------------------------------------------------


def test_answer_logps_sum_the_answer_tokens_only():
    """With the identity as head the residual is the logits: the sum runs over the positions that
    predict an answer token, and nothing at a prompt-only or padding position counts."""
    residual = torch.randn(2, 6, 5)
    labels = torch.tensor([[-100, -100, 3, 1, -100, -100], [-100, 4, 0, 2, 2, -100]])
    lp = residual.log_softmax(-1)
    want = torch.stack([lp[0, 1, 3] + lp[0, 2, 1], lp[1, 0, 4] + lp[1, 1, 0] + lp[1, 2, 2] + lp[1, 3, 2]])
    assert torch.allclose(answer_logps(lambda h: h, residual, labels), want)
    noisy = residual.clone()
    noisy[0, [0, 3, 4, 5]] += 9.0
    noisy[1, [4, 5]] += 9.0
    assert torch.allclose(answer_logps(lambda h: h, noisy, labels), want)


def test_answer_logps_run_the_head_in_fp32_under_autocast():
    """Under bf16 autocast the head would round its logits to bf16 (the backbone's stream stays bf16,
    as autocast leaves it): the result equals the same computation with autocast off, exactly."""
    head = torch.nn.Linear(8, 50, bias=False)
    residual = torch.randn(2, 6, 8).bfloat16()
    labels = torch.tensor([[-100, -100, 3, 1, 2, -100], [-100, 4, 0, 2, 2, 7]])
    plain = answer_logps(head, residual, labels)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        got = answer_logps(head, residual, labels)
        rounded = head(residual.float())
    assert rounded.dtype == torch.bfloat16, "autocast is on here"
    assert got.dtype == torch.float32 and torch.equal(got, plain)


def test_collated_pairs_score_as_their_rows_alone():
    """``collate_pairs`` lays out chosen rows, then rejected rows, right-padded with -100 labels off
    the answers, and KTO's mismatched rows; on a real model each padded row scores as it does alone."""
    pairs = [
        {"prompt_ids": [1, 2, 3, 4], "chosen_ids": [5, 6], "rejected_ids": [7]},
        {"prompt_ids": [8], "chosen_ids": [9, 10, 11, 12, 13], "rejected_ids": [14, 15]},
    ]
    batch = collate_pairs(pairs, pad_id=0)
    assert batch["input_ids"].tolist() == [[1, 2, 3, 4, 5, 6], [8, 9, 10, 11, 12, 13], [1, 2, 3, 4, 7, 0], [8, 14, 15, 0, 0, 0]]  # fmt: skip
    assert batch["labels"][2].tolist() == [-100, -100, -100, -100, 7, -100]
    assert batch["kl_input_ids"].tolist() == [[1, 2, 3, 4, 9, 10, 11, 12, 13], [8, 5, 6, 0, 0, 0, 0, 0, 0]]
    assert batch["kl_labels"][1].tolist() == [-100, 5, 6, *[-100] * 6]

    model = HybridLanguageModel(_tiny_config()).eval()
    rows = [(p["prompt_ids"], p["chosen_ids"]) for p in pairs] + [
        (p["prompt_ids"], p["rejected_ids"]) for p in pairs
    ]
    with torch.no_grad():
        got = DPOLightningModule._logps(model, batch["input_ids"], batch["labels"])
        for i, (p, a) in enumerate(rows):
            lp = model(torch.tensor([p + a])).logits[0].log_softmax(-1)
            assert got[i] == pytest.approx(
                sum(lp[len(p) - 1 + j, t].item() for j, t in enumerate(a)), abs=1e-5
            )


def _tiny_config() -> HybridConfig:
    return HybridConfig(vocab_size=64, dim=32, num_layers=2, layer_pattern=["mamba3", "mlstm"], head_dim=16,
                        num_heads=2, tfla_impl="exact", mamba3_d_state=16, mamba3_head_dim=16, mamba3_chunk_size=8)  # fmt: skip


def test_sft_loads_refuse_a_missing_or_an_unexpected_key(tmp_path):
    """Policy, reference and sampler start from ``load_sft_model``: every key on both sides."""
    state = {f"model.{k}": v for k, v in HybridLanguageModel(_tiny_config()).state_dict().items()}
    path = tmp_path / "sft.ckpt"
    torch.save({"state_dict": state}, path)
    assert isinstance(load_sft_model(path, _tiny_config()), HybridLanguageModel)
    torch.save({"state_dict": {**state, "model.extra.weight": torch.zeros(1)}}, path)
    with pytest.raises(RuntimeError, match=r"1 unexpected keys \['extra.weight'\]"):
        load_sft_model(path, _tiny_config())
    torch.save({"state_dict": {k: v for k, v in state.items() if k != "model.final_norm.weight"}}, path)
    with pytest.raises(RuntimeError, match=r"1 missing keys \['final_norm.weight'\]"):
        load_sft_model(path, _tiny_config())


def test_pair_schedule_is_seeded_and_reads_every_pair_once_per_pass():
    pairs = [{"id": i} for i in range(4)]
    schedule = [PairSchedule(pairs, num_samples=10, seed=3)[i]["id"] for i in range(10)]
    assert schedule == [PairSchedule(pairs, num_samples=10, seed=3)[i]["id"] for i in range(10)]
    assert sorted(schedule[:4]) == sorted(schedule[4:8]) == [0, 1, 2, 3]
    assert [PairSchedule(pairs)[i]["id"] for i in range(4)] == [0, 1, 2, 3]


# -- the input and the data loaders -------------------------------------------------------------


def test_the_pair_builder_reads_only_the_sft_builders_splits(sft_run):
    """A generator file has no manifest beside it, or is not the row count the manifest gives."""
    items = B.read_items(sft_run / "sft" / "train.jsonl")
    assert len(items) == 7 and {i["kind"] for i in items} == {"answerable", "unanswerable"}
    with pytest.raises(ValueError, match="not a generator file"):
        B.read_items(sft_run / "generator.jsonl")
    shutil.copy(sft_run / "generator.jsonl", sft_run / "sft" / "dev.jsonl")
    with pytest.raises(ValueError, match="not a generator file"):
        B.read_items(sft_run / "sft" / "dev.jsonl")
    assert B.read_items(sft_run / "sft" / "train.jsonl") == items


def _pair_rows(path, n: int):
    rows = [
        {"id": f"p{i}", "question_id": f"q{i}", "kind": "answerable", "prompt_ids": [i + 1, 2, 3],
         "chosen_ids": [4, 5, EOS_ID], "rejected_ids": [6, EOS_ID]}
        for i in range(n)
    ]  # fmt: skip
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _fake_trainer(world=1, accumulate=1):
    return SimpleNamespace(world_size=world, accumulate_grad_batches=accumulate, max_steps=2)


def test_train_dpo_refuses_pairs_that_cannot_fill_a_step(tmp_path, script):
    """4 ranks x 2 pairs need 8 pairs for one optimizer step (and twice that with 2 micro-batches)."""
    train = script("train_dpo")
    cfg = _compose("+dpo=default", "dpo.sft_checkpoint=x", f"dpo.pairs={_pair_rows(tmp_path / 'p.jsonl', 7)}",
                   "dpo.batch_size=2", "dpo.num_workers=0")  # fmt: skip
    with pytest.raises(ValueError, match="cannot fill one optimizer step"):
        train.build_dataloaders(cfg, _fake_trainer(world=4))
    _pair_rows(tmp_path / "p.jsonl", 8)
    assert train.build_dataloaders(cfg, _fake_trainer(world=4))[1] is None
    with pytest.raises(ValueError, match="2 micro-batches"):
        train.build_dataloaders(cfg, _fake_trainer(world=4, accumulate=2))


def test_kto_validation_never_sees_a_batch_of_one(tmp_path, script):
    """KTO's mismatched rows need two pairs: its dev loader drops a short last batch and refuses dev
    pairs that fill none; DPO keeps it. A KTO batch size of 1 is refused outright."""
    train = script("train_dpo")
    common = ["+dpo=default", "dpo.sft_checkpoint=x", f"dpo.pairs={_pair_rows(tmp_path / 'p.jsonl', 8)}",
              f"dpo.val_pairs={_pair_rows(tmp_path / 'd.jsonl', 5)}", "dpo.batch_size=2", "dpo.num_workers=0"]  # fmt: skip

    def sizes(*extra):
        val = train.build_dataloaders(_compose(*common, *extra), _fake_trainer())[1]
        return [len(b["input_ids"]) // 2 for b in val]

    assert sizes("dpo.loss=kto") == [2, 2]
    assert sizes("dpo.loss=dpo") == [2, 2, 1]
    _pair_rows(tmp_path / "d.jsonl", 1)
    with pytest.raises(ValueError, match="without a validation batch"):
        sizes("dpo.loss=kto")
    assert sizes("dpo.loss=dpo") == [1]
    with pytest.raises(ValueError, match="batch_size must be >= 2"):
        sizes("dpo.loss=kto", "dpo.batch_size=1")


def test_train_dpo_needs_its_config_and_top_k_needs_validation(tmp_path, script):
    train = script("train_dpo")
    with pytest.raises(ValueError, match=r"\+dpo=default"):
        train.run(_compose())
    cfg = _compose("+dpo=default", "dpo.sft_checkpoint=x", "dpo.pairs=x", "callbacks.checkpoint.save_top_k=1",
                   f"output_dir={tmp_path}")  # fmt: skip
    with pytest.raises(ValueError, match="give dpo.val_pairs"):
        train.build_callbacks(cfg, True)


# -- CPU end to end ------------------------------------------------------------------------------

TINY = [
    "model.dim=32", "model.num_layers=2", "model.layer_pattern=[mamba3,mlstm]", "model.num_heads=2",
    "model.head_dim=16", "model.mamba3_head_dim=16", "model.mamba3_d_state=16", "model.mamba3_chunk_size=8",
    "model.tfla_impl=exact", "model.mlstm_chunk_size=8", "model.tie_word_embeddings=false", "model.dropout=0.0",
]  # fmt: skip


def _rig_unigram(blocks, tails, embedding, norm, head, tok) -> None:
    """A real model's next-token distribution made uniform at every position over EOS, ``a``,
    ``<|unanswerable|>``, ``<|cite|>`` and ``<|c1|>`` (the rest at logit -30, and the pointer FSM masks
    what the position forbids): every block writes nothing to the residual stream (zero output
    projections), every token embeds alike, and the head maps that one stream to the logits."""
    with torch.no_grad():
        for name, p in blocks.named_parameters():
            if name.endswith(tails):
                p.zero_()
        embedding.weight[:] = torch.randn(embedding.weight.shape[1])
        h = norm(embedding.weight[:1])[0]
        target = torch.full((head.weight.shape[0],), -30.0)
        target[[EOS_ID, ord("a"), *(tok.ids[n] for n in ("<|unanswerable|>", "<|cite|>", "<|c1|>"))]] = 0.0
        head.weight[:] = target[:, None] * h / h.dot(h)


@pytest.fixture
def sft_run(tmp_path, script):
    """A tiny SFT checkpoint and its yaml, and the SFT builder's real splits of twelve generator rows
    on one passage: train holds 6 answerable and 1 unanswerable, dev 3 and 2. One row has no chunk ids
    (the verifier reads the prompt's passages)."""
    from omegaconf import OmegaConf

    cfg = _compose(*TINY)
    torch.manual_seed(0)
    model = HybridLanguageModel(HybridConfig.from_hydra(cfg.model)).eval()
    _rig_unigram(model.layers, ("mixer.out_proj.weight", "mlp.2.weight"), model.embeddings.token_embedding,
                 model.final_norm, model.lm_head, ByteTokenizer())  # fmt: skip
    torch.save(
        {"state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}}, tmp_path / "sft.ckpt"
    )
    (tmp_path / "tiny.yaml").write_text(OmegaConf.to_yaml(cfg.model))

    passage = Passage("BGB §568", ("(1) Die Kündigung des Mietverhältnisses bedarf der schriftlichen Form.",))
    sft = script("build_sft_data")
    rows = []
    for i in range(12):
        unanswerable = i % 4 == 0
        answer = () if unanswerable else (Text("Schriftlich: "), Quote(1, 1))
        # the builder splits on the question's own text: pick a wording that lands on this row's side
        dev = _unit(0, f"q{i}") < 0.5
        question = next(q for n in count() if (_unit(0, sft.question_id(q := f"Welche Form braucht Fall {i}.{n}?")) < 0.5) == dev)  # fmt: skip
        rows.append({"id": f"sft{i}", "question_id": f"q{i}", "source": "test",
                     "kind": "unanswerable" if unanswerable else "answerable", "chunk_ids": ["c#1"] if i else [],
                     "text": "".join(render(Example(question, (passage,), answer)))})  # fmt: skip
    (tmp_path / "generator.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "evalset_x.jsonl").write_text(json.dumps({"info": {"name": "x"}}) + "\n")
    sft.build([tmp_path / "generator.jsonl"], [], tmp_path / "sft", ByteTokenizer(), tmp_path / "manifests", 0.5)  # fmt: skip
    return tmp_path


def _stub(build, monkeypatch) -> None:
    monkeypatch.setattr(build, "load_tokenizer", ByteTokenizer)
    monkeypatch.setattr(build, "MDebertaNLI", lambda *a, **k: lambda premise, hypothesis: 1.0)


def _argv(run, split: str, *extra, n: int = 32, model=None) -> list[str]:
    return [
        "--sft-data", str(run / "sft" / f"{split}.jsonl"), "--checkpoint", str(run / "sft.ckpt"),
        *(model or ["--model-config", str(run / "tiny.yaml")]), "--out", str(run / "pairs" / f"{split}.jsonl"),
        "--n", str(n), "--max-new-tokens", "6", *extra,
    ]  # fmt: skip


def _dpo_overrides(run, loss="dpo", *extra) -> list[str]:
    return [
        *TINY, "trainer=cpu_debug", "+dpo=default", f"dpo.sft_checkpoint={run / 'sft.ckpt'}",
        f"dpo.pairs={run / 'pairs' / 'train.jsonl'}", f"dpo.val_pairs={run / 'pairs' / 'dev.jsonl'}",
        f"dpo.loss={loss}", "dpo.batch_size=2", "dpo.num_workers=0", "dpo.learning_rate=1e-3",
        "dpo.warmup_steps=0", f"output_dir={run / 'out' / loss}", "+logger=false", *extra,
    ]  # fmt: skip


def test_sampling_to_pairs_to_preference_steps_on_cpu(sft_run, script, monkeypatch):
    """The SFT model's FSM-constrained samples, verified (stub NLI), give one pair per prompt and the
    refusal share of each split's prompts; two DPO steps from that SFT checkpoint move the policy
    towards the chosen answers, two KTO steps run with a finite loss, and the reference is neither
    trained nor checkpointed."""
    build, train = script("build_dpo_pairs"), script("train_dpo")
    _stub(build, monkeypatch)
    assert build.main(_argv(sft_run, "train")) == 0
    assert build.main(_argv(sft_run, "dev")) == 0
    stats = json.loads((sft_run / "pairs" / "train.stats.json").read_text())
    assert (stats["prompts"], stats["should_refuse_prompts"], stats["pairs"], stats["refusal_pairs"]) == (
        7,
        1,
        7,
        1,
    )
    stats = json.loads((sft_run / "pairs" / "dev.stats.json").read_text())
    assert (stats["prompts"], stats["should_refuse_prompts"], stats["pairs"], stats["refusal_pairs"]) == (
        5,
        2,
        5,
        2,
    )
    pairs = train.read_pairs(sft_run / "pairs" / "train.jsonl")
    for pair in pairs:
        if pair["kind"] == "unanswerable":
            assert pair["chosen"] == B.REFUSAL_MARKUP
        else:
            assert pair["chosen"].endswith(EOS) and all(
                c["passed"] for c in pair["chosen_verifier"]["claims"]
            )
        assert pair["rejected"] != pair["chosen"]

    # DPO: all the pairs in one batch, so each step descends the loss the end of the test reads
    for loss, batch_size in (("dpo", 7), ("kto", 2)):
        cfg = _compose(*_dpo_overrides(sft_run, loss, f"dpo.batch_size={batch_size}"))
        trainer = train.run(cfg)
        module = trainer.lightning_module
        assert trainer.global_step == 2
        assert all(math.isfinite(trainer.callback_metrics[k]) for k in ("train/loss", "val/loss"))
        assert not any(k.startswith("reference") or "_reference" in k for k in module.state_dict())
        policy, reference = module.model.state_dict(), module.reference.state_dict()
        assert any(not torch.equal(policy[k], reference[k]) for k in policy), "the policy trained"
        sft = train.build_policy(cfg).state_dict()
        assert all(torch.equal(reference[k], sft[k]) for k in sft), "the reference stays the SFT model"
        if loss == "dpo":
            with torch.no_grad():
                assert module.compute_loss(collate_pairs(pairs))["loss"] < math.log(2)


def test_a_restart_skips_finished_prompts_and_draws_what_an_unbroken_run_would(sft_run, script, monkeypatch):
    """A job killed after two prompts, in the middle of writing the third record, keeps the two; the
    restart draws only the others and ends with the unbroken run's pairs. Other sampling settings
    over the same file are refused."""
    build = script("build_dpo_pairs")
    _stub(build, monkeypatch)
    argv = _argv(sft_run, "train", n=8)
    assert build.main(argv) == 0
    whole = (sft_run / "pairs" / "train.jsonl").read_text()
    samples_path = sft_run / "pairs" / "train.samples.jsonl"
    lines = samples_path.read_text().split("\n")
    assert json.loads(lines[0])["params"]["n"] == 8 and len(lines) == 1 + 7 + 1, (
        "a header, 7 records, a newline"
    )
    assert set(json.loads(lines[1])) == {"id", "samples"}
    assert set(json.loads(lines[1])["samples"][0]) == {"tokens", "markup", "verifier"}

    (sft_run / "pairs" / "train.jsonl").unlink()
    samples_path.write_text("\n".join(lines[:3]) + "\n" + lines[3][:20])  # the kill
    drawn, real = [], build.draw_samples

    def counted(item, *a, **k):
        drawn.append(item["id"])
        return real(item, *a, **k)

    monkeypatch.setattr(build, "draw_samples", counted)
    assert build.main(argv) == 0
    assert drawn == [json.loads(line)["id"] for line in lines[3:8]], "the unfinished prompts, in order"
    assert (sft_run / "pairs" / "train.jsonl").read_text() == whole
    assert samples_path.read_text().split("\n") == lines

    drawn.clear()
    assert build.main(argv) == 0 and drawn == [], "nothing left to draw: selection runs on the file"
    with pytest.raises(ValueError, match="give another --out"):
        build.main(_argv(sft_run, "train", n=16))


def test_a_run_with_no_pairs_exits_non_zero_and_says_why(sft_run, script, monkeypatch, capsys):
    """No prompt with a pair, and balancing that empties the set, both exit 1 and name the cause;
    the stats are written and no pairs file is."""
    build = script("build_dpo_pairs")
    _stub(build, monkeypatch)
    monkeypatch.setattr(build, "draw_samples", lambda item, *a, **k: [UNSUPPORTED])
    assert build.main(_argv(sft_run, "train")) == 1
    err = capsys.readouterr().err
    assert "no prompt gave a pair: of 7 prompts, 7 had no good sample and 0 no bad sample" in err
    assert not (sft_run / "pairs" / "train.jsonl").exists()
    assert json.loads((sft_run / "pairs" / "train.stats.json").read_text())["pairs"] == 0

    # every should-refuse prompt refuses well, so only answer pairs exist: balancing drops them all
    def refuses_well(item, *a, **k):
        return [REFUSAL] if item["kind"] == "unanswerable" else [UNSUPPORTED, KEPT]

    monkeypatch.setattr(build, "draw_samples", refuses_well)
    assert build.main(_argv(sft_run, "train", "--out", str(sft_run / "pairs2" / "train.jsonl"))) == 1
    assert (
        "balancing emptied the set: there are no refusal pairs, so the 6 others were dropped"
        in capsys.readouterr().err
    )

    def answers_nothing(item, *a, **k):
        return [REFUSAL, KEPT] if item["kind"] == "unanswerable" else [REFUSAL]

    monkeypatch.setattr(build, "draw_samples", answers_nothing)
    assert build.main(_argv(sft_run, "train", "--out", str(sft_run / "pairs3" / "train.jsonl"))) == 1
    assert "there are no answer pairs, so the 1 others were dropped" in capsys.readouterr().err


def test_a_requeue_reads_on_through_the_pair_schedule(sft_run, script, monkeypatch):
    """A run stopped at step 2 and resumed from last.ckpt reads the rows an unbroken run reads:
    nothing repeated, nothing skipped."""
    train = script("train_dpo")
    pairs = _pair_rows(sft_run / "synthetic_pairs.jsonl", 6)
    seen, real = [], train.DPOLightningModule.training_step

    def recorded(self, batch, batch_idx):
        seen.extend(batch["input_ids"][:, 0].tolist())
        return real(self, batch, batch_idx)

    monkeypatch.setattr(train.DPOLightningModule, "training_step", recorded)

    def fit(name, *extra):
        overrides = _dpo_overrides(sft_run, "dpo", f"dpo.pairs={pairs}", "dpo.val_pairs=null", "dpo.batch_size=1",
                                   "trainer.max_steps=4", "trainer.enable_checkpointing=true",
                                   "callbacks.checkpoint.every_n_train_steps=1", f"output_dir={sft_run / name}", *extra)  # fmt: skip
        seen.clear()
        trainer = train.run(_compose(*overrides))
        return trainer.global_step, list(seen)

    steps, whole = fit("whole")
    assert steps == 4 and len(whole) == 8, "4 steps of chosen + rejected rows"
    steps, first = fit("split", "trainer.stop_at_step=2")
    assert steps == 2
    last = sft_run / "split" / "checkpoints" / "last.ckpt"
    steps, second = fit("split", f"+resume_from_checkpoint={last}")
    assert steps == 4 and first + second == whole


def test_resuming_refuses_a_checkpoint_of_other_settings(sft_run, script, monkeypatch):
    """A last.ckpt saved with another loss, beta, learning rate or SFT checkpoint is another run's."""
    build, train = script("build_dpo_pairs"), script("train_dpo")
    _stub(build, monkeypatch)
    assert build.main(_argv(sft_run, "train", n=8)) == 0
    out = sft_run / "first"
    common = ["trainer.enable_checkpointing=true", "callbacks.checkpoint.every_n_train_steps=1",
              "dpo.val_pairs=null", f"output_dir={out}"]  # fmt: skip
    assert train.run(_compose(*_dpo_overrides(sft_run, "dpo", *common))).global_step == 2
    last = f"+resume_from_checkpoint={out / 'checkpoints' / 'last.ckpt'}"
    shutil.copy(sft_run / "sft.ckpt", sft_run / "other.ckpt")
    for override, name in [
        ("dpo.beta=0.2", "beta"),
        ("dpo.learning_rate=2e-3", "learning_rate"),
        ("dpo.loss=kto", "loss"),
        (f"dpo.sft_checkpoint={sft_run / 'other.ckpt'}", "sft_checkpoint"),
        ("trainer.max_steps=3", "max_steps"),  # another schedule, and the sampler offset would shift
        ("dpo.batch_size=1", "batch_size"),
        (f"dpo.pairs={sft_run / 'other_pairs.jsonl'}", "pairs"),
    ]:
        with pytest.raises(ValueError, match=rf"other settings .*'{name}'"):
            train.run(_compose(*_dpo_overrides(sft_run, "dpo", *common, override, last)))
    assert train.run(_compose(*_dpo_overrides(sft_run, "dpo", *common, last))).global_step == 2


def test_the_best_checkpoint_holds_the_weights_its_val_loss_scored(sft_run, script, monkeypatch):
    """The step-based checkpoint writes last.ckpt only. The top-k one saves at validation end, so the
    weights in the best file give the best score when scored again: a checkpoint saved before that
    step's validation pairs each score with the weights one validation later."""
    build, train = script("build_dpo_pairs"), script("train_dpo")
    _stub(build, monkeypatch)
    assert build.main(_argv(sft_run, "train", n=8)) == 0
    assert build.main(_argv(sft_run, "dev", n=8)) == 0
    out = sft_run / "best"
    trainer = train.run(_compose(*_dpo_overrides(
        sft_run, "dpo", "trainer.enable_checkpointing=true", "trainer.max_steps=3", "trainer.limit_val_batches=1.0",
        "callbacks.checkpoint.every_n_train_steps=1", "callbacks.checkpoint.save_top_k=1", f"output_dir={out}",
        "dpo.learning_rate=1e-2",
    )))  # fmt: skip
    (best,) = [c for c in trainer.checkpoint_callbacks if getattr(c, "save_top_k", 0) > 0]
    assert best.monitor == "val/loss"
    files = sorted(p.name for p in (out / "checkpoints").glob("*.ckpt"))
    assert files == [p for p in files if p == "last.ckpt" or p.startswith("best-step")] and len(files) == 2
    module = trainer.lightning_module
    state = torch.load(best.best_model_path, map_location="cpu", weights_only=False)["state_dict"]
    module.load_state_dict(state)
    module.eval()
    with torch.no_grad():
        rescored = module.compute_loss(collate_pairs(train.read_pairs(sft_run / "pairs" / "dev.jsonl")))[
            "loss"
        ]
    assert rescored.item() == pytest.approx(best.best_model_score.item(), abs=1e-5)


@pytest.mark.parametrize("crash_in", ["train", "val"])
def test_a_requeue_resumes_with_every_validation_ranked(sft_run, script, monkeypatch, crash_in):
    """A 6-step run validating every 2 steps with a scripted val/loss of 2, 1, 3 is killed after its
    step-4 validation (in step 5) or during it, then resumed from last.ckpt as a requeue does. Every
    step is validated once its run gets there, and the ranking keeps step 4 as the only best file. A
    last.ckpt saved at step 4 before its validation would skip that validation on resume and restore
    the ranking of step 2's."""
    from pytorch_lightning.callbacks import ModelCheckpoint

    train = script("train_dpo")
    losses, seen, crash = {2: 2.0, 4: 1.0, 6: 3.0}, [], {"on": True}

    def scripted(self, batch, batch_idx):
        if crash["on"] and crash_in == "val" and self.global_step == 4:
            raise RuntimeError("preempted")
        if batch_idx == 0:
            seen.append(self.global_step)
        self.log("val/loss", losses[self.global_step], batch_size=1)

    real = DPOLightningModule.training_step

    def training_step(self, batch, batch_idx):
        if crash["on"] and crash_in == "train" and self.global_step == 4:
            raise RuntimeError("preempted")
        return real(self, batch, batch_idx)

    monkeypatch.setattr(DPOLightningModule, "validation_step", scripted)
    monkeypatch.setattr(DPOLightningModule, "training_step", training_step)
    out = sft_run / "requeue"
    ov = _dpo_overrides(sft_run, "dpo", f"dpo.pairs={_pair_rows(sft_run / 'p.jsonl', 6)}",
                        f"dpo.val_pairs={_pair_rows(sft_run / 'd.jsonl', 2)}", "trainer.max_steps=6",
                        "trainer.val_every_opt_steps=2", "callbacks.checkpoint.every_n_train_steps=2",
                        "trainer.enable_checkpointing=true", "callbacks.checkpoint.save_top_k=1", f"output_dir={out}")  # fmt: skip
    with pytest.raises(RuntimeError, match="preempted"):
        train.run(_compose(*ov))
    ckpts = out / "checkpoints"
    first = seen.copy()
    seen.clear()
    crash["on"] = False
    trainer = train.run(_compose(*ov, f"+resume_from_checkpoint={ckpts / 'last.ckpt'}"))
    assert (first, seen) == (([2, 4], [6]) if crash_in == "train" else ([2], [4, 6]))
    (best,) = [c for c in trainer.callbacks if isinstance(c, ModelCheckpoint) and c.monitor == "val/loss"]
    assert (best.best_model_path, float(best.best_model_score)) == (str(ckpts / "best-step000004.ckpt"), 1.0)
    assert sorted(p.name for p in ckpts.glob("best-step*.ckpt")) == ["best-step000004.ckpt"]
    assert torch.load(ckpts / "last.ckpt", map_location="cpu", weights_only=False)["global_step"] == 6


# -- the Hugging Face baseline --------------------------------------------------------------------


@pytest.fixture
def hf_run(sft_run):
    """A tiny Qwen3 SFT checkpoint on the same splits: its config has the tokenizer's base vocabulary,
    and the SFT run resized the embedding to hold the pointer specials."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(vocab_size=BASE_LEN, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=8, tie_word_embeddings=False)  # fmt: skip
    config.save_pretrained(sft_run / "hf")
    torch.manual_seed(0)
    lm = Qwen3ForCausalLM(config)
    lm.resize_token_embeddings(BASE_LEN + len(SPECIALS))
    _rig_unigram(lm.model.layers, ("self_attn.o_proj.weight", "mlp.down_proj.weight"), lm.model.embed_tokens,
                 lm.model.norm, lm.lm_head, ByteTokenizer())  # fmt: skip
    torch.save({"state_dict": {f"model.lm.{k}": v for k, v in lm.state_dict().items()}}, sft_run / "sft.ckpt")
    return sft_run


def test_the_hf_baseline_goes_through_pairs_and_dpo(hf_run, script, monkeypatch):
    """The baseline's SFT checkpoint loads with every key (the embedding at the checkpoint's size),
    gives pairs under the pointer FSM, and trains against its frozen copy."""
    build, train = script("build_dpo_pairs"), script("train_dpo")
    _stub(build, monkeypatch)
    model = ["--hf-model", str(hf_run / "hf")]
    assert build.main(_argv(hf_run, "train", model=model)) == 0
    assert build.main(_argv(hf_run, "dev", model=model)) == 0
    assert json.loads((hf_run / "pairs" / "train.stats.json").read_text())["refusal_pairs"] == 1
    with pytest.raises(SystemExit):
        build.main(_argv(hf_run, "train", "--hf-model", str(hf_run / "hf")))
    cfg = _compose(*_dpo_overrides(hf_run, "dpo", f"dpo.hf_model={hf_run / 'hf'}"))
    trainer = train.run(cfg)
    module = trainer.lightning_module
    assert trainer.global_step == 2 and math.isfinite(trainer.callback_metrics["val/loss"])
    policy, reference = module.model.state_dict(), module.reference.state_dict()
    assert any(not torch.equal(policy[k], reference[k]) for k in policy)
    (hf_run / "out" / "broken").mkdir(parents=True)
    state = torch.load(hf_run / "sft.ckpt", map_location="cpu", weights_only=False)["state_dict"]
    del state["model.lm.lm_head.weight"]
    torch.save({"state_dict": state}, hf_run / "out" / "broken" / "sft.ckpt")
    with pytest.raises(RuntimeError, match=r"1 missing keys \['lm_head.weight'\]"):
        load_sft_model(hf_run / "out" / "broken" / "sft.ckpt", hf_model=str(hf_run / "hf"))
    torch.save(
        {"state_dict": {"model.layers.0.norm1.weight": torch.zeros(1)}},
        hf_run / "out" / "broken" / "sft.ckpt",
    )
    with pytest.raises(RuntimeError, match="missing from the checkpoint"):  # not the baseline's checkpoint
        load_sft_model(hf_run / "out" / "broken" / "sft.ckpt", hf_model=str(hf_run / "hf"))


def test_hf_sampling_through_its_cache_matches_the_uncached_reference():
    """Same logits, same generator: the KV-cached sampler and the uncached ``sample`` decode alike."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from lexhybrid.decoding.generate import sample, sample_cached
    from lexhybrid.training.sft_module import CausalLMAdapter

    torch.manual_seed(0)
    lm = Qwen3ForCausalLM(
        Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                    num_attention_heads=2, num_key_value_heads=1, head_dim=8)
    ).eval()  # fmt: skip
    ids = torch.randint(0, 64, (3, 7))
    kw = dict(max_new_tokens=9, temperature=1.0, top_p=0.95, eos_token_id=5)
    cached = sample_cached(CausalLMAdapter(lm), ids, generator=torch.Generator().manual_seed(1), **kw)
    assert torch.equal(cached, sample(lm, ids, generator=torch.Generator().manual_seed(1), **kw))


# -- the SLURM wrappers, run in a sandbox ---------------------------------------------------------

FAKE_PYTHON = r"""#!/usr/bin/env bash
# `-c <gpu check> N`: passes with at least N fake GPUs; an entry point: one line per call
if [[ "$1" == -c ]]; then [[ "${FAKE_GPUS:-4}" -ge "${!#}" ]]; exit $?; fi
(IFS=$'\x1f'; printf '%s\n' "$*" >> "$CALLS")
exit "${FAKE_EXIT:-0}"
"""


@pytest.fixture
def wrapper(tmp_path):
    """``run(script, **env) -> (CompletedProcess, calls: argument lists)`` in a repository copy whose
    ``.venv/bin/python`` records the entry points it starts; ``run.sft_ckpt`` is a run's checkpoint."""
    import stat
    import subprocess

    from tests.conftest import REPO_ROOT

    repo, bin_dir = tmp_path / "repo", tmp_path / "bin"
    shutil.copytree(REPO_ROOT / "scripts" / "slurm", repo / "scripts" / "slurm")
    (repo / ".venv" / "bin").mkdir(parents=True)
    bin_dir.mkdir()
    for path, text in [
        (repo / ".venv" / "bin" / "python", FAKE_PYTHON),
        (bin_dir / "nvidia-smi", "#!/usr/bin/env bash\n"),
    ]:
        path.write_text(text)
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    scratch = tmp_path / "scratch"
    sft_ckpt = scratch / "outputs" / "sft_x" / "checkpoints" / "last.ckpt"
    sft_ckpt.parent.mkdir(parents=True)
    sft_ckpt.write_bytes(b"0")

    def run(script: str, **env):
        calls = tmp_path / "calls.txt"
        calls.unlink(missing_ok=True)
        full = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), "USER": "tester", "CALLS": str(calls),
                "SLURM_SUBMIT_DIR": str(repo), "SCRATCH_ROOT": str(scratch), **{k: str(v) for k, v in env.items()}}  # fmt: skip
        res = subprocess.run(
            ["bash", f"scripts/slurm/{script}"], env=full, cwd=repo, capture_output=True, text=True
        )
        lines = calls.read_text().splitlines() if calls.exists() else []
        return res, [line.split("\x1f") for line in lines]

    run.scratch, run.sft_ckpt = scratch, sft_ckpt
    return run


def _write_pairs(wrapper) -> None:
    pairs = wrapper.scratch / "data" / "dpo" / "sft_x"
    pairs.mkdir(parents=True)
    (pairs / "train.jsonl").write_text("{}\n")
    (pairs / "dev.jsonl").write_text("{}\n")


def test_dpo_wrapper_stops_before_training_without_its_inputs(wrapper):
    res, calls = wrapper("train_dpo_4gpu.sh")
    assert res.returncode != 0 and "SFT_CKPT" in res.stderr and calls == []
    res, calls = wrapper("train_dpo_4gpu.sh", SFT_CKPT="elsewhere.ckpt")
    assert res.returncode == 1 and "is not a file <sft run>/checkpoints/<file>" in res.stdout and calls == []
    res, calls = wrapper("train_dpo_4gpu.sh", SFT_CKPT=wrapper.sft_ckpt)
    assert res.returncode == 1 and "data/dpo/sft_x/train.jsonl does not exist" in res.stdout and calls == []
    _write_pairs(wrapper)
    res, calls = wrapper("train_dpo_4gpu.sh", SFT_CKPT=wrapper.sft_ckpt, FAKE_GPUS=3)
    assert res.returncode == 1 and "fewer than NUM_GPUS=4" in res.stdout and calls == []


def test_dpo_wrapper_names_the_run_after_its_settings_and_composes_under_hydra(wrapper):
    """The defaults read the pairs of that SFT run and match configs/dpo/default.yaml; the experiment
    carries the SFT run, loss, LR, beta and seed, so a sweep point never shares a directory; a requeue
    passes last.ckpt; real Hydra composes the argument list."""
    from omegaconf import OmegaConf

    from tests.conftest import REPO_ROOT

    _write_pairs(wrapper)
    res, ((entry, *args),) = wrapper("train_dpo_4gpu.sh", SFT_CKPT=wrapper.sft_ckpt)
    name = "dpo_sft_x_dpo_lr5e-7_beta0.1_s42"
    assert res.returncode == 0 and f"DPO DONE {name}" in res.stdout, res.stdout + res.stderr
    assert entry == "scripts/train_dpo.py" and not any(a.startswith("+resume_from_checkpoint") for a in args)
    cfg, yaml = _compose(*args), OmegaConf.load(REPO_ROOT / "configs" / "dpo" / "default.yaml")
    pairs = wrapper.scratch / "data" / "dpo" / "sft_x"
    d = cfg.dpo
    assert (d.sft_checkpoint, d.pairs, d.val_pairs, d.hf_model) == (str(wrapper.sft_ckpt), str(pairs / "train.jsonl"), str(pairs / "dev.jsonl"), None)  # fmt: skip
    assert (d.loss, d.learning_rate, d.beta) == (yaml.loss, yaml.learning_rate, yaml.beta)
    assert (cfg.trainer.devices, cfg.trainer.strategy, cfg.callbacks.checkpoint.save_top_k) == (4, "ddp", 1)
    assert cfg.output_dir == str(wrapper.scratch / "outputs" / name)

    sweep = dict(SFT_CKPT=wrapper.sft_ckpt, LOSS="kto", LR="1e-6", BETA="0.2")
    out = wrapper.scratch / "outputs" / "dpo_sft_x_kto_lr1e-6_beta0.2_s42"
    (out / "checkpoints").mkdir(parents=True)
    (out / "checkpoints" / "last.ckpt").write_bytes(b"0")
    res, ((_, *args),) = wrapper("train_dpo_4gpu.sh", **sweep, EXTRA_OVERRIDES="dpo.batch_size=8")
    assert res.returncode == 0 and f"+resume_from_checkpoint={out}/checkpoints/last.ckpt" in args
    d = _compose(*args).dpo
    assert (d.loss, d.learning_rate, d.beta, d.batch_size) == ("kto", 1e-6, 0.2, 8)


def test_wrappers_take_the_hf_baseline(wrapper):
    _write_pairs(wrapper)
    env = dict(SFT_CKPT=wrapper.sft_ckpt, HF_BASELINE=1)
    res, ((_, *args),) = wrapper("train_dpo_4gpu.sh", **env)
    assert res.returncode == 0 and _compose(*args).dpo.hf_model == "Qwen/Qwen3-0.6B-Base"
    res, calls = wrapper("build_dpo_pairs.sh", **env)
    assert res.returncode == 0
    assert all(
        c[c.index("--hf-model") + 1] == "Qwen/Qwen3-0.6B-Base" and "--model-config" not in c for c in calls
    )


def test_sft_wrapper_trains_the_lora_ceiling(wrapper):
    res, ((entry, *args),) = wrapper("train_sft_4gpu.sh", LORA=1, LR="1e-4", SEED=42)
    assert res.returncode == 0, res.stdout + res.stderr
    assert entry == "scripts/train_sft_hf.py"
    assert {"sft.lora_rank=16", "sft.hf_model=Qwen/Qwen3-8B-Base", "sft.learning_rate=1e-4"} <= set(args)
    assert "experiment_name=sft_lora_qwen3_8b_s42_lr1e-4" in args
    assert not any(a.startswith("lm_checkpoint=") for a in args)
    _, ((_, *plain),) = wrapper("train_sft_4gpu.sh", HF_BASELINE=1, SEED=42)
    assert not any("lora" in a for a in plain), "the baseline stays a full fine-tune"


def test_evaluation_takes_a_hugging_face_system(hf_run, script, monkeypatch):
    """The evaluation's loader and generator run the baseline's checkpoint under the pointer FSM,
    best of n through the KV cache; a run names either a config or a repository, never both."""
    from types import SimpleNamespace

    from lexhybrid.retrieval.chunking import Chunk
    from lexhybrid.training.sft_module import CausalLMAdapter
    from tests.test_pipeline import ByteTokenizer

    answer = script("answer")
    model = answer.load_model(None, str(hf_run / "sft.ckpt"), "cpu", str(hf_run / "hf"))
    assert isinstance(model, CausalLMAdapter) and not model.training
    chunk = Chunk(id="d#0", doc_id="d", source="s", jurisdiction="DE", doc_type="statute", citation_id="BGB §1",
                  hierarchy=(), sentences=("Ein Satz.",), labels=("BGB §1",), valid_from=None, valid_to=None,
                  licence="", commercial_safe=True, research_only=False, url="")  # fmt: skip
    monkeypatch.setattr(answer, "verify", lambda *a: SimpleNamespace(refused=False, citation_precision=1.0))
    gen = answer.generate("Frage?", [chunk], model, ByteTokenizer(), nli=None, n=2, max_new_tokens=4)
    assert 0 < gen.new_tokens <= 4

    parser = argparse.ArgumentParser()
    answer.add_arguments(parser)
    base = ["--index", "i", "--checkpoint", "c"]
    assert parser.parse_args([*base, "--hf-model", "r"]).hf_model == "r"
    for bad in (base, [*base, "--hf-model", "r", "--model-config", "m"]):
        with pytest.raises(SystemExit):
            parser.parse_args(bad)


def test_pairs_wrapper_builds_both_splits_of_the_sft_builder_for_that_run(wrapper):
    """Train, then dev, from the SFT builder's output into a directory named after the SFT run; a
    split with no pairs (exit 1) stops the job."""
    res, calls = wrapper("build_dpo_pairs.sh", SFT_CKPT=wrapper.sft_ckpt, N=16)
    assert res.returncode == 0 and "DPO PAIRS DONE" in res.stdout, res.stdout + res.stderr
    assert [c[0] for c in calls] == ["scripts/build_dpo_pairs.py"] * 2
    for call, split in zip(calls, ("train", "dev"), strict=True):
        flags = dict(zip(call[1::2], call[2::2]))
        assert flags["--sft-data"] == str(wrapper.scratch / "data" / "sft" / f"{split}.jsonl")
        assert flags["--out"] == str(wrapper.scratch / "data" / "dpo" / "sft_x" / f"{split}.jsonl")
        assert (flags["--device"], flags["--n"], flags["--model-config"]) == (
            "cuda",
            "16",
            "hybrid_legal_base",
        )
    res, calls = wrapper("build_dpo_pairs.sh", SFT_CKPT=wrapper.sft_ckpt, FAKE_EXIT=1)
    assert res.returncode == 1 and len(calls) == 1 and "DPO PAIRS DONE" not in res.stdout
    res, calls = wrapper("build_dpo_pairs.sh", SFT_CKPT=wrapper.sft_ckpt, FAKE_GPUS=0)
    assert res.returncode == 1 and "no CUDA device" in res.stdout and calls == []


def test_a_copy_of_a_gold_passage_counts_as_gold(script):
    """The index keeps every copy of a text, so a prompt can hold a gold passage twice: a quote of the copy
    answers as well as a quote of the original, for the rule and for the tie-break."""
    build = script("build_dpo_pairs")
    same = ("(1) Die Kündigung bedarf der schriftlichen Form.",)
    passages = (
        Passage("BGB §568", same),
        Passage("Kopie", same),
        Passage("BGB §90", ("Sachen sind körperlich.",)),
    )
    item = {"text": "".join(render(Example("Welche Form?", passages, (Text("Schriftlich: "), Quote(1, 1)))))}
    gold = build.gold_pointers(item)
    assert gold == (frozenset({1, 2}), frozenset({(1, 1), (2, 1)}))
    passed = {"refused": False, "claims": [{"passed": True}]}
    for k, good in ((1, True), (2, True), (3, False)):
        sample = {"markup": f"Schriftlich: <|q|><|c{k}|><|s1|><|endoftext|>", "verifier": passed}
        assert build.is_good(sample, False, gold) is good, k


def test_samples_of_older_rows_or_an_older_checkpoint_are_not_reused(sft_run, script, monkeypatch):
    """Rebuilding the rows or retraining into the same paths changes what would be sampled: the
    resume cache refuses, instead of pairing new prompts with old samples. A copied file of equal
    count that the builder did not write is refused too."""
    build = script("build_dpo_pairs")
    _stub(build, monkeypatch)
    assert build.main(_argv(sft_run, "train", n=4)) == 0
    assert build.main(_argv(sft_run, "train", n=4)) == 0, "unchanged inputs resume"
    ckpt = sft_run / "sft.ckpt"
    ckpt.write_bytes(ckpt.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="give another --out"):
        build.main(_argv(sft_run, "train", n=4))
    rows = sft_run / "sft" / "train.jsonl"
    lines = rows.read_text(encoding="utf-8").splitlines()
    rows.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not hold these rows"):
        build.main(_argv(sft_run, "train", n=4))
