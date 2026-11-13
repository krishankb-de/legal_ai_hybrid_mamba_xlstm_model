"""End-to-end answer pipeline on CPU: index -> retrieve -> prompt -> constrained best-of-n -> verify."""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace

import pytest
import torch

from lexhybrid import HybridConfig, HybridLanguageModel
from lexhybrid.data.tokenizer import BASE_LEN, EOS_ID, SPECIALS, VOCAB_SIZE
from lexhybrid.renderer.render import NOTICE, REFUSAL
from tests.conftest import REPO_ROOT

QUESTION = "Welche Form braucht die Kündigung des Mietverhältnisses?"


class ByteTokenizer:
    """UTF-8 bytes as ids 0..255 and the pointer specials from ``BASE_LEN`` on, as in Qwen3's vocabulary."""

    unk_token_id = None

    def __init__(self):
        self.ids = {name: BASE_LEN + i for i, name in enumerate(SPECIALS)}

    def __call__(self, text, add_special_tokens=False, split_special_tokens=True):
        return {"input_ids": list(text.encode())}

    def convert_tokens_to_ids(self, names):
        return [self.ids.get(n) for n in names]

    def decode(self, ids):
        return bytes(i for i in ids if i < 256).decode(errors="replace")


@pytest.fixture
def tiny_setup(tmp_path):
    from lexhybrid.retrieval.bm25 import BM25Index
    from lexhybrid.retrieval.chunking import chunk_document
    from lexhybrid.retrieval.index import CHUNKS_NAME
    from tests.test_retrieval import _gii, _planted

    chunks = chunk_document(_planted()) + [c for d in list(_gii().values())[:3] for c in chunk_document(d)]
    index = tmp_path / "index"
    index.mkdir()
    (index / CHUNKS_NAME).write_text("".join(c.to_json() + "\n" for c in chunks), encoding="utf-8")
    BM25Index.build(chunks).save(index / "bm25")

    cfg = dict(vocab_size=VOCAB_SIZE, dim=32, num_layers=2, layer_pattern=["mamba3", "mlstm"], head_dim=16,
               num_heads=2, tfla_impl="exact", mamba3_d_state=16, mamba3_head_dim=16, mamba3_chunk_size=8)  # fmt: skip
    torch.manual_seed(0)
    model = HybridLanguageModel(HybridConfig(**cfg))
    with torch.no_grad():  # only bytes, EOS and pointers can win, so samples decode to text and pointers
        keep = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
        keep[[*range(256), EOS_ID, *range(BASE_LEN, BASE_LEN + len(SPECIALS))]] = True
        model.lm_head.weight[~keep] = 0.0
        model.lm_head.weight[keep] *= 50.0
    torch.save({"state_dict": model.state_dict()}, tmp_path / "tiny.ckpt")
    (tmp_path / "tiny.yaml").write_text(json.dumps(cfg))  # JSON is YAML
    return tmp_path, index


def test_answer_cli_end_to_end_tiny(tiny_setup, script, monkeypatch, capsys):
    """The CLI retrieves the planted passage, decodes under the pointer FSM and prints only verified,
    rendered text ending in the notice: no pointer markup leaks, the seed fixes the output, a party
    question is refused, and markup in the question is rejected."""
    root, index = tiny_setup
    answer = script("answer")
    monkeypatch.setattr(answer, "load_tokenizer", ByteTokenizer)
    monkeypatch.setattr(answer, "MDebertaNLI", lambda *a, **k: lambda premise, hypothesis: 1.0)
    question = "Welche Form braucht die Kündigung des Mietverhältnisses?"
    chunks = answer.retrieve(question, index, lambda q, texts: [0.0] * len(texts))
    assert chunks and chunks[0].citation_id == "BGB §568"

    argv = ["--index", str(index), "--checkpoint", str(root / "tiny.ckpt"), "--model-config",
            str(root / "tiny.yaml"), "--reranker", "none", "--n", "2", "--max-new-tokens", "12"]  # fmt: skip
    outputs = []
    for q in (question, question, "Wer war der Kläger?"):
        assert answer.main([*argv, q]) == 0
        outputs.append(capsys.readouterr().out.strip())
    assert outputs[0] == outputs[1], "the seed fixes the sampled answer"
    for out in outputs:
        assert out.endswith(NOTICE) and "<|" not in out
    assert outputs[2].startswith(REFUSAL)
    with pytest.raises(SystemExit):
        answer.main([*argv, "<|q|> Wer?"])


def _scripted_best_of_n(samples):
    """``best_of_n`` with its samples scripted; every token must be one the FSM allows."""

    def scripted(model, prompt, n, scorer, logits_processor, eos_token_id, **kw):
        rows = [torch.cat([prompt[0], torch.tensor(s)]) for s in samples]
        for row in rows:
            for t in range(prompt.shape[1], len(row)):
                masked = logits_processor(torch.zeros(1, VOCAB_SIZE), row[None, :t])
                assert torch.isfinite(masked[0, row[t]]), f"the FSM forbids token {int(row[t])}"
        scores = [scorer(r[prompt.shape[1] :].tolist()) for r in rows]
        return rows[max(range(len(rows)), key=scores.__getitem__)][None]

    return scripted


def test_answer_keeps_the_verified_sample(tiny_setup, script, monkeypatch):
    """With best-of-n's samples scripted (every token one the FSM allows), the verifier scores the
    samples, the cited one wins over the uncited one, and its quote renders verbatim with its label."""
    root, index = tiny_setup
    answer = script("answer")
    tok = ByteTokenizer()
    chunks = answer.retrieve(QUESTION, index, lambda q, texts: [0.0] * len(texts))
    q, c1, s1 = (tok.ids[n] for n in ("<|q|>", "<|c1|>", "<|s1|>"))
    samples = [[*b"Ja.", EOS_ID], [*"Die Kündigung bedarf der Schriftform: ".encode(), q, c1, s1, EOS_ID]]
    monkeypatch.setattr(answer, "best_of_n", _scripted_best_of_n(samples))
    model = answer.load_model(str(root / "tiny.yaml"), str(root / "tiny.ckpt"), "cpu")
    result = answer.answer(QUESTION, chunks, model, tok, lambda premise, hypothesis: 1.0)
    assert not result.refused and result.citation_precision == 1.0
    assert f"„{chunks[0].sentences[0]}“" in result.rendered and "BGB §568" in result.rendered
    assert result.rendered.endswith(NOTICE) and "<|" not in result.rendered


def test_best_of_n_refuses_when_at_least_half_the_samples_are_the_models_refusal(
    tiny_setup, script, monkeypatch
):
    """Half of the samples or more being the model's own refusal decides, though a kept sample would
    outscore it; fewer than half leave the choice to the verifier's score."""
    root, index = tiny_setup
    answer = script("answer")
    tok = ByteTokenizer()
    chunks = answer.retrieve(QUESTION, index, answer.fused_order)
    q, c1, s1, no = (tok.ids[n] for n in ("<|q|>", "<|c1|>", "<|s1|>", "<|unanswerable|>"))
    cited, refusal = [*b"Schriftform: ", q, c1, s1, EOS_ID], [no, EOS_ID]
    model = answer.load_model(str(root / "tiny.yaml"), str(root / "tiny.ckpt"), "cpu")
    nli = lambda premise, hypothesis: 1.0  # noqa: E731

    def decode(samples):
        monkeypatch.setattr(answer, "best_of_n", _scripted_best_of_n(samples))
        kept = answer.answer(QUESTION, chunks, model, tok, nli, n=len(samples))
        return answer.generate(QUESTION, chunks, model, tok, nli, n=len(samples)), kept

    for samples in ([cited, refusal, cited, refusal], [refusal, refusal, refusal, cited], [refusal]):
        gen, kept = decode(samples)
        assert (gen.raw, gen.new_tokens) == ("<|unanswerable|><|endoftext|>", 2)
        assert kept.refused and kept.reason == "unanswerable"
    gen, kept = decode([refusal, cited, cited, cited])
    assert gen.raw == "Schriftform: <|q|><|c1|><|s1|><|endoftext|>" and gen.new_tokens == len(cited)
    assert not kept.refused and kept.citation_precision == 1.0


def _ablation(root, script, monkeypatch):
    """eval_system_ablation over a 51-chunk BM25 + dense index (room for 16 passages) with stand-in
    models, whose reranker lifts fused rank 2: ``(module, index, chunks by id, fused ids, scorer,
    the shared options)``."""
    from lexhybrid.retrieval.bm25 import BM25Index
    from lexhybrid.retrieval.chunking import chunk_document
    from lexhybrid.retrieval.dense import DenseIndex
    from lexhybrid.retrieval.fusion import reciprocal_rank_fusion
    from lexhybrid.retrieval.index import CHUNKS_NAME
    from tests.test_retrieval import _bag_of_words, _fixture_corpus

    chunks = [c for d in _fixture_corpus() for c in chunk_document(d)]
    index = root / "corpus"
    index.mkdir()
    (index / CHUNKS_NAME).write_text("".join(c.to_json() + "\n" for c in chunks), encoding="utf-8")
    BM25Index.build(chunks).save(index / "bm25")
    DenseIndex.build(index / "dense", [c.id for c in chunks], [c.text for c in chunks], _bag_of_words)
    rankings = [BM25Index.load(index / "bm25").query(QUESTION, k=50),
                DenseIndex.load(index / "dense").query(QUESTION, _bag_of_words, k=50)]  # fmt: skip
    fused = [h.chunk_id for h in reciprocal_rank_fusion(rankings)]
    by_id = {c.id: c for c in chunks}
    scorer = lambda q, texts: [float(t == by_id[fused[1]].text) for t in texts]  # noqa: E731

    ab = script("eval_system_ablation")
    stand_ins = {"load_tokenizer": ByteTokenizer, "MDebertaNLI": lambda *a, **k: lambda premise, hypothesis: 1.0,
                 "CrossEncoderScorer": lambda *a, **k: scorer, "BgeM3Encoder": lambda *a, **k: _bag_of_words}  # fmt: skip
    for name, stand_in in stand_ins.items():
        monkeypatch.setattr(ab.answer, name, stand_in)
    settings = ["--index", str(index), "--checkpoint", str(root / "tiny.ckpt"), "--model-config",
                str(root / "tiny.yaml"), "--n", "2", "--max-new-tokens", "8", "--seed", "3"]  # fmt: skip
    return ab, index, by_id, fused, scorer, settings


def test_ablation_arms_differ_only_in_their_component(tiny_setup, script, monkeypatch, capsys):
    """scripts/eval_system_ablation.py over BM25 + dense: every arm answers the same questions with the
    same seed; A1 and A2 quote the top of the fused-order and of the reranked selection whole and verify,
    the reranker's order deciding A2; A3 is answer.py's output and A4_sel the same selection without the
    reranker; A4_k prompts are the fused top k in order; A5 decodes the gold prompt sets as A3 does the
    same passages; prompts past the trained length are labelled; a prompt set without passages is
    refused undecoded; the summary names what each comparison isolates; markup in a question is rejected."""
    from lexhybrid.data.sft_format import UNANSWERABLE
    from lexhybrid.data.tokenizer import EOS_TOKEN
    from lexhybrid.eval.records import AnswerRecord, chunk_dict, read_records, write_records
    from lexhybrid.retrieval.rerank import rerank_and_select
    from tests.test_retrieval import _bag_of_words

    root, _ = tiny_setup
    ab, index, by_id, fused, scorer, settings = _ablation(root, script, monkeypatch)
    monkeypatch.setattr(ab, "TRAINED_CONTEXT", 2000)  # in bytes: 16 passages are past it, none is for a copy
    party = "Wer war der Kläger?"
    rows = [{"id": "q1", "question": QUESTION, "answers": ["Schriftform"], "gold": ["BGB §568 Abs. 1"]},
            {"id": "q2", "question": party, "kind": "party", "gold_chunk_ids": ["nowhere#1"]},
            {"id": "q3"}]  # fmt: skip
    (root / "tiny.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    reranked = ab.answer.retrieve(QUESTION, index, scorer, encode=_bag_of_words)
    gold = [AnswerRecord("q1", "tiny", "gold", "gold_hard_k4", QUESTION, chunks=[chunk_dict(c) for c in reranked]),
            AnswerRecord("q2", "tiny", "gold", "gold_hard_k4", party, kind="party")]  # fmt: skip
    write_records(root / "gold.jsonl", gold)
    out = root / "ablation"
    argv = ["--questions", str(root / "tiny.jsonl"), "--gold-prompts", str(root / "gold.jsonl"), "--out", str(out)]  # fmt: skip
    assert ab.main([*argv, *settings]) == 0
    log = capsys.readouterr().out
    assert "QUESTIONS rows=3 skipped=1 with_gold_answers=1 with_gold_chunks=1 gold_prompts=2" in log
    assert "ABLATION A1 records=2 refused=1" in log

    arms = {p.stem: read_records(p) for p in out.glob("*.jsonl")}
    assert set(arms) == {"A1", "A2", "A3", "A4_sel", "A4_k5", "A4_k10", "A4_k16", "A5_gold_hard_k4"}
    for arm, recs in arms.items():
        assert [r.id for r in recs] == ["q1", "q2"] and [r.kind for r in recs] == ["answerable", "party"]
        assert {(r.set, r.system, r.seed, r.arm) for r in recs} == {("tiny", "hybrid", 3, arm)}
        assert recs[1].refused, "a party question is refused in every arm"
        for r in recs:
            budget = 0 if arm in ("A1", "A2") or not r.chunks else 8
            assert r.max_new_tokens == budget and r.new_tokens <= budget
            assert r.raw_answer.endswith(EOS_TOKEN) or r.new_tokens == budget
    assert arms["A1"][0].gold_chunk_ids == ["gii:bgb:568#1"] and arms["A1"][0].gold_answers == ["Schriftform"]

    second = by_id[fused[1]]
    assert by_id[fused[0]].citation_id == "BGB §568" and reranked[0] == second
    for arm, top in (("A1", by_id[fused[0]]), ("A2", second)):
        r = arms[arm][0]
        span = "".join(f"<|s{j}|>" for j in range(1, len(top.sentences) + 1))
        assert r.chunks == [chunk_dict(top)]
        assert r.raw_answer == f"<|q|><|c1|>{span}{EOS_TOKEN}" and r.new_tokens == 0
        assert not r.refused and r.verifier["citation_precision"] == 1.0
        assert f"„{' '.join(top.sentences)}“" in r.rendered and r.rendered.endswith(NOTICE)
    for k in (5, 10, 16):
        assert [c["id"] for c in arms[f"A4_k{k}"][0].chunks] == fused[:k]

    a3, a5 = arms["A3"], arms["A5_gold_hard_k4"]
    plain = rerank_and_select(QUESTION, [by_id[i] for i in fused], lambda q, texts: [0.0] * len(texts))
    assert arms["A4_sel"][0].chunks == [chunk_dict(c) for c in plain], "A3's limits in the fused order"
    assert [c.id for c in plain] == [i for i in fused if i in {c.id for c in plain}] and len(plain) <= 8
    assert {c["id"] for c in arms["A4_sel"][0].chunks} != {second.id} and a3[0].chunks != arms["A4_sel"][
        0
    ].chunks
    assert a3[0].chunks == a5[0].chunks == [chunk_dict(c) for c in reranked]
    assert (a5[0].raw_answer, a5[0].rendered) == (a3[0].raw_answer, a3[0].rendered), (
        "same decoding, same seed"
    )
    assert a5[1].chunks == [] and a5[1].raw_answer == UNANSWERABLE + EOS_TOKEN and a5[1].new_tokens == 0
    assert ab.answer.main([*settings, QUESTION]) == 0
    assert capsys.readouterr().out == a3[0].rendered + "\n", "A3 is the shipped pipeline"

    summary = json.loads((out / "summary.json").read_text())
    assert summary["settings"]["seed"] == 3 and summary["trained_context"] == 2000
    assert summary["questions"] == {"rows": 3, "skipped": 1, "with_gold_answers": 1, "with_gold_chunks": 1,
                                    "gold_prompts": 2}  # fmt: skip
    assert "the reranker" in summary["comparisons"]["A1 vs A2"]
    assert "the reranker" in summary["comparisons"]["A4_sel vs A3"]
    for arm, recs in arms.items():
        s = summary["arms"][arm]
        assert (s["records"], s["refused"]) == (2, sum(r.refused for r in recs))
        assert s["mean_new_tokens"] == pytest.approx(sum(r.new_tokens for r in recs) / 2)
    assert summary["arms"]["A4_k16"]["extrapolated"] == ["q1", "q2"]
    assert summary["arms"]["A1"]["extrapolated"] == summary["arms"]["A2"]["extrapolated"] == []
    assert summary["arms"]["A5_gold_hard_k4"]["extrapolated"] == ["q1"], "no passages, no prompt"

    (root / "bad.jsonl").write_text(json.dumps({"id": "q3", "question": "<|q|> Wer?"}) + "\n")
    with pytest.raises(SystemExit):
        ab.main(["--questions", str(root / "bad.jsonl"), "--out", str(out), *settings])


def test_ablation_copies_the_selection_top_and_takes_any_generator(tiny_setup, script, monkeypatch, tmp_path):
    """A passage over the token budget can be the fused top but is never selected: A1 and A4_sel skip
    it as A3's selection does, while A4_k keeps it, so A1 vs A2 and A4_sel vs A3 differ only by the
    reranker. The model is one callable: a stand-in generator fills every model arm."""
    from lexhybrid.eval.records import AnswerRecord, read_records

    _, index = tiny_setup
    ab = script("eval_system_ablation")
    small = ab.answer._index(index)[0]["gii:bgb:568#1"]
    big = replace(small, id="big#1", sentences=("Satz " * 5000,))
    monkeypatch.setattr(ab.answer, "candidates", lambda *a, **k: [big, small])
    prompts = []

    def generator(question, chunks):
        prompts.append([c.id for c in chunks])
        return ab.answer.Generation("Ja.<|endoftext|>", 2, 8)

    nli = lambda premise, hypothesis: 1.0  # noqa: E731
    q = AnswerRecord("q1", "tiny", "other", "", QUESTION, seed=0)
    ab.run([q], [], index, ab.answer.fused_order, None, generator, nli, tmp_path)
    arms = {p.stem: read_records(p)[0] for p in tmp_path.glob("*.jsonl")}
    assert [c["id"] for c in arms["A1"].chunks] == [small.id] == [c["id"] for c in arms["A2"].chunks]
    assert [c["id"] for c in arms["A4_sel"].chunks] == [small.id] == [c["id"] for c in arms["A3"].chunks]
    for k in (5, 10, 16):
        assert [c["id"] for c in arms[f"A4_k{k}"].chunks] == [big.id, small.id]
    assert prompts == [[small.id], [small.id], *[[big.id, small.id]] * 3], "only model arms decode"
    assert arms["A3"].raw_answer == "Ja.<|endoftext|>" and arms["A3"].new_tokens == 2
    assert arms["A3"].max_new_tokens == 8 and arms["A3"].system == "other"


def test_ablation_reads_the_evalsets_qa_format_and_counts_its_gold(tiny_setup, script, tmp_path):
    """Questions in either form: ``answers`` are gold answers; ``gold`` citations resolve to the index's
    chunks through ``citation_id`` or a sentence label; ``gold_chunk_ids`` the index holds win; a row
    without an id or a question is skipped and counted; gold-less rows need no index at all."""
    _, index = tiny_setup
    ab = script("eval_system_ablation")
    rows = [{"id": "a", "question": "?", "answers": ["x"], "gold": ["BGB §568"]},
            {"id": 7, "question": "?", "gold": ["BeurkG §1 Abs. 2", "BeurkG §2", "nowhere"]},
            {"id": "c", "question": "?", "gold_answers": ["y"], "gold_chunk_ids": ["gii:beurkg:3#1", "nowhere#1"],
             "gold": ["BGB §568"]},
            {"id": "d", "question": "?", "gold_chunk_ids": ["nowhere#1"], "gold": ["BGB §568"]},
            {"id": "e", "question": "?", "kind": "unanswerable"},
            {"id": "f", "question": ""}, {"question": "no id"}]  # fmt: skip
    path = tmp_path / "qa.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows) + "\n")
    questions, counts = ab.read_questions(path, "tiny", "hybrid", 3, index)
    assert [(q.id, q.gold_answers, q.gold_chunk_ids) for q in questions] == [
        ("a", ["x"], ["gii:bgb:568#1"]),
        ("7", [], ["gii:beurkg:1#1", "gii:beurkg:2#1"]),
        ("c", ["y"], ["gii:beurkg:3#1"]),
        ("d", [], ["gii:bgb:568#1"]),
        ("e", [], []),
    ]
    assert {(q.set, q.system, q.seed) for q in questions} == {("tiny", "hybrid", 3)}
    assert questions[-1].kind == "unanswerable"
    assert counts == {"rows": 7, "skipped": 2, "with_gold_answers": 2, "with_gold_chunks": 4}
    path.write_text(json.dumps({"id": "e", "question": "?"}) + "\n")
    assert ab.read_questions(path, "tiny", "hybrid", 3, tmp_path / "missing")[1]["with_gold_chunks"] == 0


def test_ablation_appends_each_record_and_resumes_after_preemption(tiny_setup, script, monkeypatch):
    """Records reach the arm files as they are produced, so a run cut mid-question keeps its work; the
    rerun drops a cut last line, decodes only the missing (id, arm) pairs and ends byte-identical to an
    uninterrupted run; a rerun with other settings is refused rather than mixed in."""
    root, _ = tiny_setup
    ab, _, _, _, _, settings = _ablation(root, script, monkeypatch)
    rows = [{"id": "q1", "question": QUESTION}, {"id": "q2", "question": "Wann verjährt der Anspruch?"}]
    (root / "tiny.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    questions = ["--questions", str(root / "tiny.jsonl")]
    reference, out = root / "reference", root / "out"
    assert ab.main([*questions, "--out", str(reference), *settings]) == 0

    real, calls, cut_at = ab.answer.generate, [], [2]

    def flaky(question, chunks, **kw):
        if len(calls) == cut_at[0]:
            raise RuntimeError("preempted")
        calls.append(question)
        return real(question, chunks, **kw)

    monkeypatch.setattr(ab.answer, "generate", flaky)
    with pytest.raises(RuntimeError, match="preempted"):
        ab.main([*questions, "--out", str(out), *settings])
    assert {p.stem for p in out.glob("*.jsonl")} == {"A1", "A2", "A3", "A4_sel"}
    for path in out.glob("*.jsonl"):  # q1's copies and its first two decodes, as the arm order has them
        assert path.read_text() == (reference / path.name).read_text().splitlines(keepends=True)[0]
    with open(out / "A3.jsonl", "a", encoding="utf-8") as f:
        f.write('{"id": "q2", "arm": "A3", "ques')

    calls.clear()
    cut_at[0] = None
    assert ab.main([*questions, "--out", str(out), *settings]) == 0
    assert len(calls) == 5 * len(rows) - 2, "five model arms per question, two of them already written"
    assert {p.name for p in out.glob("*.jsonl")} == {p.name for p in reference.glob("*.jsonl")}
    for path in reference.glob("*.jsonl"):
        assert (out / path.name).read_text() == path.read_text()
    summary = json.loads((out / "summary.json").read_text())
    assert all(s["records"] == len(rows) for s in summary["arms"].values())

    with pytest.raises(SystemExit):
        ab.main([*questions, "--out", str(out), *settings, "--seed", "4"])
    assert (out / "A3.jsonl").read_text() == (reference / "A3.jsonl").read_text()


def test_ablation_on_an_empty_questions_file_writes_an_empty_summary(tmp_path, script, monkeypatch):
    """No questions and no gold prompts: a clean exit and an empty summary, with no model or index read."""
    ab = script("eval_system_ablation")
    monkeypatch.setattr(ab.answer, "load_tokenizer", ByteTokenizer)
    (tmp_path / "empty.jsonl").write_text("")
    out = tmp_path / "new" / "out"
    argv = ["--questions", str(tmp_path / "empty.jsonl"), "--out", str(out), "--index", str(tmp_path / "missing"),
            "--checkpoint", str(tmp_path / "missing.ckpt"), "--model-config", "missing"]  # fmt: skip
    assert ab.main(argv) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["arms"] == {} and summary["questions"]["rows"] == 0


def test_ablation_wrapper_fails_on_an_audit_crash_but_not_on_truncation(tmp_path):
    """Exit 1 from the audit (a rate above 0) is reported and the job succeeds, any other failure fails
    the job, and a run with no records skips the audit."""
    repo, scratch = tmp_path / "repo", tmp_path / "scratch"
    (repo / "scripts" / "slurm").mkdir(parents=True)
    (repo / ".venv" / "bin").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "scripts" / "slurm" / "eval_system_ablation.sh", repo / "scripts" / "slurm")
    fake = repo / ".venv" / "bin" / "python"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == scripts/audit_truncation.py ]]; then exit "$AUDIT_EXIT"; fi\n'
        "mkdir -p logs/eval_system_ablation_local\n"
        '[[ -n "${NO_RECORDS:-}" ]] || touch logs/eval_system_ablation_local/A1.jsonl\n'
    )
    fake.chmod(0o755)
    ckpt = scratch / "outputs" / "run" / "checkpoints" / "last.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.touch()
    (scratch / "data" / "index" / "v1").mkdir(parents=True)
    (scratch / "data" / "index" / "v1" / "index.json").touch()
    (repo / "q.jsonl").touch()
    base = {k: v for k, v in os.environ.items() if not k.startswith("SLURM")}
    base |= {"SCRATCH_ROOT": str(scratch), "EXPERIMENT": "run", "QUESTIONS": "q.jsonl"}

    def wrapper(**env):
        return subprocess.run(["bash", "scripts/slurm/eval_system_ablation.sh"], cwd=repo, env={**base, **env},
                              capture_output=True, text=True)  # fmt: skip

    no_records = wrapper(AUDIT_EXIT="2", NO_RECORDS="1")  # an audit it ran would fail the job
    assert no_records.returncode == 0 and "EVAL SYSTEM ABLATION DONE" in no_records.stdout
    clean = wrapper(AUDIT_EXIT="0")
    assert clean.returncode == 0 and "TRUNCATION above" not in clean.stdout
    truncated = wrapper(AUDIT_EXIT="1")
    assert truncated.returncode == 0 and "TRUNCATION above 0%" in truncated.stdout
    assert "EVAL SYSTEM ABLATION DONE" in truncated.stdout
    crashed = wrapper(AUDIT_EXIT="2")
    assert crashed.returncode == 2 and "FATAL" in crashed.stdout
    assert "EVAL SYSTEM ABLATION DONE" not in crashed.stdout and "TRUNCATION above" not in crashed.stdout
    fake.write_text(fake.read_text() + 'echo "$*" >> calls.log\n')
    assert wrapper(AUDIT_EXIT="0", MODEL_CONFIG="hf=Qwen/Qwen3-8B-Base").returncode == 0
    hf_call = (repo / "calls.log").read_text().splitlines()[0]
    assert "--hf-model Qwen/Qwen3-8B-Base" in hf_call and "--model-config" not in hf_call
    assert wrapper(AUDIT_EXIT="0").returncode == 0
    assert "--model-config hybrid_legal_base" in (repo / "calls.log").read_text().splitlines()[-1]


def test_audit_truncation_flags_decodes_without_a_final_eos(tmp_path, script, capsys):
    """A decode is truncated when its raw markup does not end on the EOS, refused or not (a budget spent
    exactly on the EOS is not); the rate is per system, seed and arm; any truncation exits 1."""
    from lexhybrid.eval.records import AnswerRecord, write_records

    audit = script("audit_truncation")

    def rec(id, arm, raw, seed=0, new_tokens=3, refused=False):
        return AnswerRecord(id, "tiny", "hybrid", arm, "?", seed=seed, raw_answer=raw, refused=refused,
                            new_tokens=new_tokens, max_new_tokens=8)  # fmt: skip

    clean = [rec("q1", "A3", "Ja <|cite|><|c1|>.<|endoftext|>", new_tokens=8),
             rec("q2", "A3", "<|unanswerable|><|endoftext|>", refused=True)]  # fmt: skip
    cut = [rec("q1", "A4_k5", "Ja <|q|><|c1|>", new_tokens=8, refused=True),  # budget spent inside a pointer
           rec("q2", "A4_k5", "Ja.<|endoftext|>"), rec("q3", "A4_k5", "Ja."),  # q3 stopped without the EOS
           rec("q1", "A4_k5", "Ja.<|endoftext|>", seed=1)]  # fmt: skip
    write_records(tmp_path / "clean.jsonl", clean)
    write_records(tmp_path / "cut.jsonl", cut)
    assert audit.main([str(tmp_path / "clean.jsonl")]) == 0
    clean_line = "TRUNCATION system=hybrid seed=0 arm=A3 records=2 truncated=0 rate=0.0000 ids="
    assert capsys.readouterr().out == clean_line + "\n"
    assert audit.main([str(tmp_path / "clean.jsonl"), str(tmp_path / "cut.jsonl")]) == 1
    assert capsys.readouterr().out.splitlines() == [
        clean_line,
        "TRUNCATION system=hybrid seed=0 arm=A4_k5 records=3 truncated=2 rate=0.6667 ids=q1,q3",
        "TRUNCATION system=hybrid seed=1 arm=A4_k5 records=1 truncated=0 rate=0.0000 ids=",
    ]


def test_audit_truncation_crash_is_not_a_truncation_verdict(tmp_path):
    """Python exits 1 on an uncaught exception, which is the audit's verdict for truncation: a crash
    exits 2, so a broken file is never read as a truncated decode, and a clean audit still exits 0."""
    from lexhybrid.eval.records import AnswerRecord, write_records

    audit = str(REPO_ROOT / "scripts" / "audit_truncation.py")

    def status(*files):
        return subprocess.run([sys.executable, audit, *files], capture_output=True, text=True)

    missing = status(str(tmp_path / "missing.jsonl"))
    assert missing.returncode == 2 and "FileNotFoundError" in missing.stderr
    (tmp_path / "broken.jsonl").write_text('{"id": "q1", "ar\n')
    assert status(str(tmp_path / "broken.jsonl")).returncode == 2
    rec = AnswerRecord("q1", "tiny", "hybrid", "A3", "?", seed=0, raw_answer="Ja.<|endoftext|>")
    write_records(tmp_path / "ok.jsonl", [rec])
    assert status(str(tmp_path / "ok.jsonl")).returncode == 0
    write_records(tmp_path / "cut.jsonl", [replace(rec, raw_answer="Ja")])
    assert status(str(tmp_path / "cut.jsonl")).returncode == 1


def test_best_of_n_never_prefers_a_truncated_sample(tiny_setup, script, monkeypatch):
    """A sample cut off by the budget scores below any finished one, even a refused one."""
    root, index = tiny_setup
    answer = script("answer")
    tok = ByteTokenizer()
    question = "Welche Form braucht die Kündigung des Mietverhältnisses?"
    chunks = answer.retrieve(question, index, lambda q, texts: [0.0] * len(texts))
    q, c1, s1 = (tok.ids[n] for n in ("<|q|>", "<|c1|>", "<|s1|>"))
    words = [*"Die Kündigung bedarf der Schriftform: ".encode(), q, c1, s1]
    seen = {}

    def scripted(model, prompt, n, scorer, **kw):
        cut, finished = words, [*words, EOS_ID]
        seen["cut"], seen["finished"] = scorer(cut), scorer(finished)
        return torch.tensor([[*prompt[0].tolist(), *finished]])

    monkeypatch.setattr(answer, "best_of_n", scripted)
    model = answer.load_model(str(root / "tiny.yaml"), str(root / "tiny.ckpt"), "cpu")
    answer.answer(question, chunks, model, tok, lambda premise, hypothesis: 1.0)
    assert seen["cut"] < 0 < seen["finished"]


def test_answer_refuses_a_partial_checkpoint(tiny_setup, script):
    root, _ = tiny_setup
    state = torch.load(root / "tiny.ckpt", weights_only=False)
    state["state_dict"].pop(next(iter(state["state_dict"])))
    torch.save(state, root / "partial.ckpt")
    with pytest.raises(RuntimeError, match="1 missing keys"):
        script("answer").load_model(str(root / "tiny.yaml"), str(root / "partial.ckpt"), "cpu")
