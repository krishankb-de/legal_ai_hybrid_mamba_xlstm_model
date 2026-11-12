"""Text metrics ported from the reference."""

import pytest

from lexhybrid.eval.text_metrics import _lcs_length, corpus_bleu, rouge_l_score


def test_lcs_length():
    assert _lcs_length("a b c d".split(), "a c d e".split()) == 3


def test_rouge_l_known_values():
    assert rouge_l_score([], ["a"]) == 0.0
    assert rouge_l_score("a b".split(), "c d".split()) == 0.0
    assert rouge_l_score("a b c d".split(), "a b c d".split()) == pytest.approx(1.0)
    p = r = 0.75  # lcs 3 of 4 on both sides
    assert rouge_l_score("a b c d".split(), "a c d e".split()) == pytest.approx(
        (1 + 1.44) * p * r / (r + 1.44 * p)
    )


def test_corpus_bleu_known_values():
    assert corpus_bleu([["a", "b", "c", "d"]], [["a", "b", "c", "d"]]) == pytest.approx(1.0)
    assert corpus_bleu([["x"]], [["y"]]) == 0.0
    assert corpus_bleu([[]], [["a"]], max_n=1) == 0.0
    # BLEU-1 with a brevity penalty: 2 of 2 unigrams match, hyp 2 vs ref 4 -> bp = exp(1 - 2)
    assert corpus_bleu([["a", "b"]], [["a", "b", "c", "d"]], max_n=1) == pytest.approx(0.36787944117144233)


def test_retrieval_metrics_hand_computed():
    """nDCG@10 with graded gains and log2 discount, recall@k and MRR, means over judged queries only."""
    import math

    from lexhybrid.eval.retrieval_metrics import evaluate, ndcg_at_k, recall_at_k, reciprocal_rank

    rel = {"a": 2, "b": 1, "z": 0}
    ranking = ["x", "a", "y", "b"]
    dcg = 2 / math.log2(3) + 1 / math.log2(5)
    assert math.isclose(ndcg_at_k(ranking, rel, 10), dcg / (2 + 1 / math.log2(3)))
    assert ndcg_at_k(["a", "b"], rel, 10) == 1.0 and ndcg_at_k(["b", "a"], rel, 1) == 0.5
    assert recall_at_k(ranking, rel, 2) == 0.5 and recall_at_k(ranking, rel, 5) == 1.0
    assert reciprocal_rank(ranking, rel) == 0.5 and reciprocal_rank(["z", "y"], rel) == 0.0
    m = evaluate({"q1": ["a"], "q2": ["n"]}, {"q1": {"a": 1}, "q2": {"b": 1}, "q3": {"c": 0}})
    assert m["queries"] == 2 and m["mrr"] == 0.5 and m["ndcg@10"] == 0.5 and m["recall@5"] == 0.5
    assert evaluate({}, {"q": {"a": 1}})["ndcg@10"] == 0.0
    with pytest.raises(ValueError):
        evaluate({}, {"q": {"a": 0}})


def test_answer_record_round_trips_with_its_chunks(tmp_path):
    """An AnswerRecord carries its prompt's passages in full and survives the JSONL round trip;
    an unknown kind is refused."""
    from lexhybrid.eval.records import AnswerRecord, chunk_dict, read_records, write_records
    from lexhybrid.retrieval.chunking import Chunk

    chunk = Chunk(
        id="gii:bgb:568#1", doc_id="gii:bgb:568", source="gii", jurisdiction="DE", doc_type="statute",
        citation_id="BGB §568", hierarchy=("Buch 2",), sentences=("Die Kündigung bedarf der Form.",),
        labels=("BGB §568 Abs. 1",), valid_from=None, valid_to=None, licence="DE-UrhG-5",
        commercial_safe=True, research_only=False, url="",
    )  # fmt: skip
    rec = AnswerRecord("q1", "tiny", "hybrid", "default", "Welche Form?", seed=42, chunks=[chunk_dict(chunk)])
    party = AnswerRecord("q2", "tiny", "hybrid", "default", "Wer?", kind="party")
    assert write_records(tmp_path / "r.jsonl", [rec, party]) == 2
    back = read_records(tmp_path / "r.jsonl")
    assert back == [rec, party] and back[0].chunk_objects() == [chunk]
    assert not back[0].should_refuse and back[1].should_refuse
    with pytest.raises(ValueError, match="kind"):
        AnswerRecord("q3", "tiny", "hybrid", "default", "?", kind="other")


def test_judge_parses_scores_and_caches(tmp_path):
    """The judge reads the first JSON object of a reply and retries an unparseable one (or one listing no
    statement) once, then keeps None and the raw text; correctness and the RAGAS arithmetic; only parsed
    replies are cached, keyed by endpoint, model and prompt, so a rerun asks only what did not parse and
    the next one makes zero calls; a corrupt cache file is a miss; refusals are scored without the judge
    and never get faithfulness."""
    import json

    from lexhybrid.eval.judge import (
        PROMPT_VERSION,
        RETRY,
        Judge,
        average_precision,
        first_json_object,
        judge_prompts,
        judge_record,
    )
    from lexhybrid.eval.records import AnswerRecord, chunk_dict
    from lexhybrid.renderer.render import NOTICE, REFUSAL
    from lexhybrid.retrieval.chunking import Chunk

    assert first_json_object('x {bad} ```json\n{"a": {"b": 1}}``` {"c": 2}') == {"a": {"b": 1}}
    assert first_json_object("[1, 2]") is None
    assert average_precision([True, False, True]) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision([False, False]) == 0.0

    def chunk(n, sentence):
        return chunk_dict(Chunk(
            id=f"gii:bgb:{n}#1", doc_id=f"gii:bgb:{n}", source="gii", jurisdiction="DE", doc_type="statute",
            citation_id=f"BGB §{n}", hierarchy=(), sentences=(sentence,), labels=(f"BGB §{n} Abs. 1",),
            valid_from=None, valid_to=None, licence="DE-UrhG-5", commercial_safe=True, research_only=False, url="",
        ))  # fmt: skip

    replies = {
        "You grade": ('Sure, the verdict is "partial".', '```json\n{"verdict": "partial", "reason": "Nur die Form."}\n```'),
        "You check whether": ('{"statements": []}', '{"statements": [{"statement": "a", "supported": true}, '
                              '{"statement": "b", "supported": true}, {"statement": "c", "supported": false}]}'),
        "You rate": ('{"relevancy": 0.8, "reason": "Passt."}', ""),
        "You judge which": ('{"useful": [false, true]}', ""),
        "You check how much": ("no json here", "still no json"),
    }  # fmt: skip
    sent = []

    def complete(prompt):
        sent.append(prompt)
        first, retry = next(v for k, v in replies.items() if prompt.startswith(k))
        return retry if prompt.endswith(RETRY) else first

    def fail(prompt):
        raise AssertionError("not cached")

    gold = ["Die Kündigung bedarf der schriftlichen Form."]
    chunks = [chunk(573, "Unrelated."), chunk(568, "Form.")]
    rec = AnswerRecord("q1", "tiny", "hybrid", "default", "Welche Form?", seed=42, gold_answers=gold, chunks=chunks,
                       rendered="Schriftlich (BGB §568 Abs. 1).\n\n" + NOTICE)  # fmt: skip
    cache = tmp_path / "cache"
    judge = Judge(complete, "judge-model", cache, base_url="http://a/v1")
    scores, details = judge_record(judge, rec)
    assert scores == {"correctness": 0.5, "faithfulness": pytest.approx(2 / 3), "answer_relevancy": 0.8,
                      "context_precision": 0.5, "context_recall": None}  # fmt: skip
    assert details["correctness"]["reason"] == "Nur die Form."
    assert details["context_recall"] == {"raw": "still no json"}
    assert judge.calls == len(sent) == 8  # five prompts, three of them retried
    assert "Welche Form?" in sent[0] and gold[0] in sent[0] and NOTICE not in sent[0]
    assert "<passage 2>\nBGB §568\n" in sent[2] and sent[0].endswith('"reason": "<one sentence>"}')
    files = sorted(cache.glob("*.json"))
    assert len(files) == 4  # the unparsed recall reply is not cached
    entry = json.loads(files[0].read_text())
    assert (entry["base_url"], entry["model"]) == ("http://a/v1", "judge-model")
    assert entry["prompt_version"] == PROMPT_VERSION and entry["timestamp"] and "response" in entry

    replies["You check how much"] = ('{"statements": [{"statement": "s", "attributed": true}, '
                                     '{"statement": "t", "attributed": false}]}', "")  # fmt: skip
    rerun = Judge(complete, "judge-model", cache, base_url="http://a/v1/")
    scores = judge_record(rerun, rec)[0]
    assert rerun.calls == 1 and scores["context_recall"] == 0.5  # only the unparsed prompt is asked again
    third = Judge(fail, "judge-model", cache, base_url="http://a/v1")
    assert judge_record(third, rec)[0] == scores and third.calls == 0
    prompt, read = judge_prompts(rec)["answer_relevancy"]
    other = Judge(complete, "judge-model", cache, base_url="http://b/v1")
    assert other.ask(prompt, read)[0] == 0.8 and other.calls == 1  # another endpoint: its own verdicts

    files = sorted(cache.glob("*.json"))
    files[0].write_text("{")
    for f in files[1:]:
        f.write_text('{"model": "judge-model"}')
    healed = Judge(complete, "judge-model", cache, base_url="http://a/v1")
    assert judge_record(healed, rec)[0] == scores and healed.calls == 7  # every file a miss, rewritten
    assert judge_record(third, rec)[0] == scores and third.calls == 0

    refusal = REFUSAL + "\n\n" + NOTICE
    unanswerable = AnswerRecord("q2", "tiny", "hybrid", "default", "Wer?", kind="unanswerable", refused=True,
                                rendered=refusal)  # fmt: skip
    assert judge_record(third, unanswerable)[0] == {"correctness": 1.0, **dict.fromkeys(list(scores)[1:])}
    no_gold = AnswerRecord("q4", "tiny", "hybrid", "default", "Was gilt?", refused=True, rendered=refusal)
    assert judge_record(third, no_gold) == (dict.fromkeys(scores), {})  # no gold: refusing scores None
    refused = AnswerRecord(**{**rec.__dict__, "refused": True, "rendered": refusal})
    scores, details = judge_record(third, refused)  # passages and gold unchanged: served from the cache
    assert scores["correctness"] == 0.0 and scores["context_precision"] == 0.5 and third.calls == 0
    assert scores["faithfulness"] is scores["answer_relevancy"] is None and "faithfulness" not in details
    answered = AnswerRecord(
        "q3", "tiny", "hybrid", "default", "Wer war der Kläger?", kind="party", rendered="X."
    )
    scores, _ = judge_record(judge, answered)
    assert scores["correctness"] == 0.0 and scores["faithfulness"] == pytest.approx(2 / 3)


# -- citation metrics and the gold-passage sanity check --------------------------------------

from tests.test_pipeline import ByteTokenizer, tiny_setup  # noqa: E402,F401 -- shared tiny fixtures


def _test_chunk(cid: str, citation: str, sentences=("Ein Satz.",), labels=None, copies=()):
    from lexhybrid.retrieval.chunking import Chunk

    doc = cid.split("#")[0]
    return Chunk(
        id=cid, doc_id=doc, source=doc.split(":")[0], jurisdiction="DE", doc_type="statute", citation_id=citation,
        hierarchy=(), sentences=tuple(sentences), labels=tuple(labels or [citation] * len(sentences)),
        valid_from=None, valid_to=None, licence="DE-UrhG-5", commercial_safe=True, research_only=False, url="",
        copies=tuple(copies),
    )  # fmt: skip


def test_citation_metrics_on_synthetic(tmp_path, script):
    """Per-question metrics and the run summary on verified synthetic records: a correct cited answer,
    a wrong citation, a refusal on an answerable item, a correct refusal, an item without gold, a
    tampered quote, and a run that never refuses. Precision is the verifier log's (system-internal),
    recall the gold passages the rendered answer cites, refusal P/R/F1 take "should refuse" as positive
    (0, not None, for a run that never refuses), and a verbatim quote of the gold sentence is EM 1; the
    metrics file has one numeric line per record, as the bootstrap reads it."""
    import json
    from dataclasses import replace

    from lexhybrid.eval.citation_metrics import SYSTEM_INTERNAL, record_metrics
    from lexhybrid.eval.records import AnswerRecord, chunk_dict, write_records
    from lexhybrid.verifier.verifier import verify

    chunks = [
        _test_chunk("gii:bgb:568#1", "BGB §568", ["Die Kündigung des Mietverhältnisses bedarf der schriftlichen Form."],
                    ["BGB §568 Abs. 1"]),
        _test_chunk("gii:bgb:573#1", "BGB §573", ["Der Vermieter braucht ein berechtigtes Interesse."]),
    ]  # fmt: skip
    g568, g573 = chunks[0].id, chunks[1].id

    def nli(premise, hypothesis):
        return float("Kündigung" in premise)  # only BGB §568 entails anything

    def record(id, raw, kind="answerable", gold_chunks=(), gold_answers=()):
        r = verify("Welche Form?", raw, chunks, nli)
        return AnswerRecord(
            id, "tiny", "hybrid", "default", "Welche Form?", seed=42, kind=kind, gold_answers=list(gold_answers),
            gold_chunk_ids=list(gold_chunks), chunks=[chunk_dict(c) for c in chunks], raw_answer=raw,
            rendered=r.rendered, refused=r.refused, verifier=r.log_row(),
        )  # fmt: skip

    correct = record(
        "correct", "<|q|><|c1|><|s1|><|endoftext|>", gold_chunks=[g568], gold_answers=[chunks[0].sentences[0]]
    )
    records = [
        correct,
        record("wrong", "Sie bedarf der Schriftform <|cite|><|c1|>. Sie braucht einen Grund <|cite|><|c2|>.",
               gold_chunks=[g573]),
        record("refused", "<|unanswerable|>", gold_chunks=[g568], gold_answers=["Schriftlich."]),
        record("unanswerable", "<|unanswerable|><|endoftext|>", kind="unanswerable"),
        record("no_gold", "Die Kündigung ist schriftlich <|cite|><|c1|>."),
        replace(correct, id="tampered", seed=43, rendered=correct.rendered.replace("schriftlichen", "mündlichen")),
        replace(record("never", "Die Kündigung ist schriftlich <|cite|><|c1|>.", kind="unanswerable"), seed=44),
    ]  # fmt: skip
    ok = {"refused": 0.0, "should_refuse": 0.0, "refusal_correct": 1.0}
    assert [record_metrics(r) for r in records[:5]] == [
        {**ok, "citation_precision": 1.0, "citation_recall": 1.0, "quote_fidelity": 1.0, "exact_match": 1.0,
         "char_f1": 1.0},
        {**ok, "citation_precision": 0.5, "citation_recall": 0.0},  # the second claim fails and is dropped
        {"refused": 1.0, "should_refuse": 0.0, "refusal_correct": 0.0, "citation_recall": 0.0, "exact_match": 0.0,
         "char_f1": 0.0},
        {"refused": 1.0, "should_refuse": 1.0, "refusal_correct": 1.0},
        {**ok, "citation_precision": 1.0},
    ]  # fmt: skip
    tampered = record_metrics(records[5])
    assert tampered["quote_fidelity"] == 0.0 and tampered["exact_match"] == 1.0  # EM reads the pointers

    write_records(tmp_path / "answers.jsonl", records)
    metrics, summary = tmp_path / "out" / "metrics.jsonl", tmp_path / "out" / "summary.json"
    argv = [str(tmp_path / "answers.jsonl"), "--metrics-out", str(metrics), "--summary-out", str(summary)]
    assert script("evaluate_citation").main(argv) == 0
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    assert rows == [{"id": r.id, **record_metrics(r)} for r in records]
    assert all(type(v) is float for row in rows for k, v in row.items() if k != "id")
    bootstrap = script("bootstrap_compare")
    means = bootstrap.evaluate_subset(
        range(7), bootstrap.build_cache(None, None, bootstrap.read_metrics(str(metrics)))
    )
    assert means["citation_recall"] == pytest.approx(0.5) and means["quote_fidelity"] == 0.5

    first, second, third = json.loads(summary.read_text())
    assert (first["system"], first["seed"], first["arm"], first["sets"], first["records"]) == (
        "hybrid", 42, "default", ["tiny"], 5)  # fmt: skip
    assert first["citation_precision"] == 0.75 and (first["cited_claims"], first["passed_cited_claims"]) == (
        4,
        3,
    )
    assert first["citation_precision_scope"] == SYSTEM_INTERNAL and "system-internal" in SYSTEM_INTERNAL
    assert first["citation_recall"] == pytest.approx(1 / 3) and first["recall_records"] == 3
    assert first["quote_fidelity"] == 1.0 and first["quote_records"] == 1
    assert (first["refused"], first["should_refuse"], first["correct_refusals"]) == (2, 1, 1)
    assert first["refusal_precision"] == 0.5 and first["refusal_recall"] == 1.0
    assert first["refusal_f1"] == pytest.approx(2 / 3) and first["refusal_accuracy"] == 0.8
    assert first["exact_match"] == 0.5 and first["char_f1"] == 0.5 and first["answer_records"] == 2
    assert second["seed"] == 43 and second["quote_fidelity"] == 0.0
    assert second["refusal_precision"] is second["refusal_recall"] is second["refusal_f1"] is None
    assert (third["seed"], third["refused"], third["should_refuse"]) == (44, 0, 1)
    assert third["refusal_precision"] == third["refusal_recall"] == third["refusal_f1"] == 0.0


def _verified_record(chunks, raw, gold_chunks=(), gold_answers=()):
    from lexhybrid.eval.records import AnswerRecord, chunk_dict
    from lexhybrid.verifier.verifier import verify

    r = verify("Welche Form?", raw, chunks, lambda premise, hypothesis: 1.0)
    return AnswerRecord(
        "q", "tiny", "hybrid", "default", "Welche Form?", gold_answers=list(gold_answers),
        gold_chunk_ids=list(gold_chunks), chunks=[chunk_dict(c) for c in chunks], raw_answer=raw,
        rendered=r.rendered, refused=r.refused, verifier=r.log_row(),
    )  # fmt: skip


def test_citation_recall_counts_copies_of_the_gold_and_never_wraps_a_pointer():
    """A cited chunk counts for a gold passage when it is that passage, shares its document or is linked
    to it through copies (either direction), also when the gold chunk is not in the prompt. A passage
    index outside the prompt cites nothing: K = 0 does not wrap to the last chunk, K too large does not
    raise, and a kept quote of such a passage fails fidelity."""
    from dataclasses import replace

    from lexhybrid.eval.citation_metrics import record_metrics

    gold_id = "gii:bgb:568#1"
    same_doc = _test_chunk("gii:bgb:568#2", "BGB §568")
    links_back = _test_chunk("rii:k#1", "BGH VIII ZR 1/20 Rn. 1", copies=["gii:bgb:568"])
    gold_in_prompt = _test_chunk("gii:bgb:573#1", "BGB §573", copies=["oldp:m"])
    listed_by_gold = _test_chunk("oldp:m#1", "LG Berlin 1 S 2/21 Rn. 3")
    other = _test_chunk("gii:bgb:574#1", "BGB §574")
    cite_first = "Das gilt <|cite|><|c1|>."
    cases = [
        ([same_doc, other], [gold_id], cite_first, 1.0),
        ([links_back, other], [gold_id], cite_first, 1.0),
        ([listed_by_gold, gold_in_prompt], [gold_in_prompt.id], cite_first, 1.0),
        ([other, same_doc], [gold_id], cite_first, 0.0),
        ([same_doc, other], [gold_id, gold_in_prompt.id], cite_first, 0.5),
    ]
    for chunks, gold, raw, recall in cases:
        assert record_metrics(_verified_record(chunks, raw, gold))["citation_recall"] == recall, chunks[0].id

    logged = {"claims": [{"index": 0, "passed": True}], "citation_precision": 1.0}
    record = _verified_record([gold_in_prompt, other], cite_first, [other.id], ["Siehe und."])
    outside = "Siehe <|cite|><|c0|> und <|cite|><|c3|> <|q|><|c5|><|s1|>."
    m = record_metrics(replace(record, raw_answer=outside, refused=False, verifier=logged))
    assert m["citation_recall"] == 0.0 and m["quote_fidelity"] == 0.0 and m["exact_match"] == 0.0


def test_em_and_char_f1_score_the_answers_own_words():
    """EM and char-F1 read the kept text and the quoted sentences: no quotation marks, labels, source
    lists or notice; a claim the verifier dropped and a refusal add no words."""
    from lexhybrid.eval.citation_metrics import answer_words, kept_segments, record_metrics
    from lexhybrid.renderer.render import NOTICE
    from lexhybrid.verifier.verifier import verify

    sentence = "Die Kündigung bedarf der Form."
    chunks = [_test_chunk("gii:bgb:568#1", "BGB §568", [sentence], ["BGB §568 Abs. 1"], copies=["oldp:m"])]
    words = f"Sie ist schriftlich. {sentence}"
    raw = "Sie ist schriftlich <|cite|><|c1|>. <|q|><|c1|><|s1|>"
    record = _verified_record(chunks, raw, gold_answers=[words])
    assert "Quellen:" in record.rendered and "„" in record.rendered and record.rendered.endswith(NOTICE)
    assert answer_words(kept_segments(record), record.chunk_objects()) == words
    m = record_metrics(record)
    assert m["exact_match"] == 1.0 and m["char_f1"] == 1.0

    raw = "Sie ist schriftlich <|cite|><|c1|>. Das stimmt nicht."  # the last claim cites nothing: dropped
    r = verify("Welche Form?", raw, chunks, lambda premise, hypothesis: 1.0)
    assert not r.refused and r.claims[1].passed is False
    dropped = _verified_record(chunks, raw, gold_answers=["Sie ist schriftlich."])
    assert record_metrics(dropped)["exact_match"] == 1.0
    refused = _verified_record(chunks, "<|unanswerable|>", gold_answers=[words])
    assert record_metrics(refused)["exact_match"] == 0.0 and record_metrics(refused)["char_f1"] == 0.0


def test_hard_distractors_exclude_gold_relevant_and_copies():
    """Gold-relevant chunks (explicit ids, or citation ids matched on a chunk's citation or a sentence
    label) are never distractors, nor is any copy of one (same document, or linked through copies in
    either direction); hard distractors keep the retriever's order, easy ones come from other acts; a
    tier with too few distractors is skipped; the gold position is seeded and uniform."""
    from lexhybrid.eval.gold_passages import build_sets

    gold = _test_chunk("gii:bgb:568#1", "BGB §568", copies=["rii:copy"])
    continued = _test_chunk("gii:bgb:568#2", "BGB §568")  # the provision's next chunk: same document
    second = _test_chunk(
        "gii:bgb:573#1", "BGB §573", ["Satz 1.", "Satz 2."], ["BGB §573 Abs. 1", "BGB §573 Abs. 2"]
    )
    copy = _test_chunk("rii:copy#1", "BGH VIII ZR 1/20 Rn. 1")  # listed in gold.copies
    back = _test_chunk(
        "oldp:back#1", "LG Berlin 1 S 2/21 Rn. 3", copies=["gii:bgb:568"]
    )  # links back to gold
    d1, d2 = _test_chunk("gii:bgb:569#1", "BGB §569"), _test_chunk("gii:bgb:570#1", "BGB §570")
    others = [_test_chunk(f"gii:{act.lower()}:1#1", f"{act} §1") for act in ("BeurkG", "ZPO", "HGB")]
    chunks = [gold, continued, second, copy, back, d1, d2, *others]
    ranking = [gold.id, copy.id, d1.id, continued.id, second.id, back.id, d2.id, *(c.id for c in others)]
    items = [
        {
            "id": "q1",
            "question": "Form?",
            "answers": ["Schriftlich."],
            "gold": ["BGB §568", "BGB §573 Abs. 2"],
        },
        {
            "id": "q2",
            "question": "Notar?",
            "gold_chunk_ids": [others[0].id, "gii:missing#1"],
            "gold": ["BGB §568"],
        },
        {"id": "q3", "question": "?", "gold": ["XYZ §1"]},  # resolves to nothing: skipped
    ]
    sets, counts = build_sets(items, chunks, lambda question: ranking, (1, 4), seed=7, set_name="tiny")
    assert (sets, counts) == build_sets(items, chunks, lambda question: ranking, (1, 4), 7, "tiny")
    assert counts == {
        "items": 3,
        "with_gold": 2,
        "sets": {
            "gold_easy_k1": {"built": 2, "skipped": 0},
            "gold_easy_k4": {"built": 1, "skipped": 1},  # q1 has three chunks of other acts
            "gold_hard_k1": {"built": 2, "skipped": 0},
            "gold_hard_k4": {"built": 2, "skipped": 0},
        },
    }
    by = {(r.id, r.arm): r for r in sets}
    assert sorted(by) == [("q1", "gold_easy_k1"), ("q1", "gold_hard_k1"), ("q1", "gold_hard_k4"),
                          ("q2", "gold_easy_k1"), ("q2", "gold_easy_k4"), ("q2", "gold_hard_k1"),
                          ("q2", "gold_hard_k4")]  # fmt: skip
    related = {gold.id, continued.id, second.id, copy.id, back.id}
    for r in sets:
        ids = [c["id"] for c in r.chunks]
        k = int(r.arm.rsplit("_k", 1)[1])
        assert len(ids) == k + 1 and ids.count(r.gold_chunk_ids[0]) == 1 and r.set == "tiny"
        assert r.system == "" and not r.raw_answer and r.kind == "answerable"
    for arm in ("gold_easy_k1", "gold_hard_k1", "gold_hard_k4"):
        r = by["q1", arm]
        assert r.gold_chunk_ids == [gold.id] and r.gold_answers == ["Schriftlich."]
        assert not related & ({c["id"] for c in r.chunks} - {gold.id})
    assert [c["id"] for c in by["q1", "gold_hard_k4"].chunks if c["id"] != gold.id] == [d1.id, d2.id,
                                                                                         others[0].id, others[1].id]  # fmt: skip
    assert {c["id"] for c in by["q1", "gold_easy_k1"].chunks} - {gold.id} <= {c.id for c in others}
    q2 = by["q2", "gold_hard_k4"]
    assert q2.gold_chunk_ids == [others[0].id] and [c["id"] for c in q2.chunks if c["id"] != others[0].id] == [
        gold.id, copy.id, d1.id, continued.id]  # fmt: skip
    easy = [c for c in by["q2", "gold_easy_k4"].chunks if c["id"] != others[0].id]
    assert len({c["id"] for c in easy}) == 4 and all(c["citation_id"] != "BeurkG §1" for c in easy)
    positions = set()
    for seed in range(40):
        built, _ = build_sets(items[:1], chunks, lambda q: ranking, (4,), seed)
        r = next(r for r in built if r.arm == "gold_hard_k4")
        positions.add([c["id"] for c in r.chunks].index(gold.id) + 1)
    assert positions == {1, 2, 3, 4, 5}
    with pytest.raises(ValueError, match="at most 16"):
        build_sets(items, chunks, lambda q: ranking, k_values=(16,))


def _gold_items(tmp_path, rows=None):
    import json

    rows = rows or [
        {"id": "q1", "question": "Wann ist die Mitwirkung des Notars ausgeschlossen?", "answers": ["Bei Befangenheit."],
         "gold": ["BeurkG §3"]},
        {"id": "q2", "question": "Welche Form braucht die Kündigung des Mietverhältnisses?", "gold": ["BGB §568"]},
        {"id": "q3", "question": "Was regelt das?", "gold": []},
    ]  # fmt: skip
    items = tmp_path / "gerlayqa" / "items.jsonl"
    items.parent.mkdir(exist_ok=True)
    items.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return items


def _decode_argv(root, sets, out, *extra):
    return ["decode", "--sets", str(sets), "--system", "hybrid", "--seed", "42", "--model-config",
            str(root / "tiny.yaml"), "--checkpoint", str(root / "tiny.ckpt"), "--max-new-tokens", "8",
            "--out", str(out), "--summary", str(out.with_suffix(".summary.json")), *extra]  # fmt: skip


def _sft_row(id, chunk_ids, answer=(), question="Welche Form braucht die Kündigung?", passages=None):
    """An SFT row as scripts/build_sft_data.py writes dev.jsonl: one passage per chunk id, in order."""
    from lexhybrid.data.sft_format import Example, Passage, render

    passages = tuple(
        passages or (Passage(f"BGB §{560 + k}", ("Ein Satz.", "Noch einer.")) for k in range(len(chunk_ids)))
    )
    return {"id": id, "question_id": "h", "source": "synthetic", "kind": "answerable" if answer else "unanswerable",
            "chunk_ids": list(chunk_ids), "text": "".join(render(Example(question, passages, tuple(answer))))}  # fmt: skip


def test_gold_items_from_sft_rows():
    """An SFT row's gold is the chunks its answer quotes or cites, each once in order of citation, never
    the passage pointers of its prompt; a refusal has none, and a pointer past its chunk ids is an error."""
    from lexhybrid.data.sft_format import Cite, Quote, Text
    from lexhybrid.eval.gold_passages import sft_item

    ids = ["a#1", "b#1", "c#1"]
    answer = (Text("Schriftlich "), Cite(3), Text(", denn "), Quote(1, 2), Cite(3), Text("."))
    row = _sft_row("r1", ids, answer, question="  Welche Form?  ")
    assert row["text"].count("<|c2|>") == 1  # only the prompt points at passage 2
    item = {"id": "r1", "question": "Welche Form?", "answers": [], "gold_chunk_ids": ["c#1", "a#1"]}
    assert sft_item(row) == item
    assert sft_item(_sft_row("r2", ids))["gold_chunk_ids"] == []
    with pytest.raises(ValueError, match="cites past its 2 chunk ids"):
        sft_item({**row, "chunk_ids": ids[:2]})


def test_eval_gold_passages_sft_rows_and_limited_decode(tiny_setup, script, monkeypatch, tmp_path, capsys):  # noqa: F811
    """build takes SFT rows (their answers' citations are the gold, a refusal is skipped); decode --limit N
    decodes only N sets and writes no summary, so a job does not skip the run as scored, and a run without
    it resumes the rest."""
    import json

    from lexhybrid.data.sft_format import Cite, Quote
    from lexhybrid.eval.gold_passages import read_chunks
    from lexhybrid.eval.records import read_records

    root, index = tiny_setup
    gold_passages = script("eval_gold_passages")
    chunks = read_chunks(index)
    notar = next(c for c in chunks if c.citation_id == "BeurkG §3")
    form = next(c for c in chunks if c.citation_id == "BGB §568")
    rows = [
        _sft_row("s1", [form.id, notar.id], (Quote(2, 1),), "Wann ist die Mitwirkung des Notars ausgeschlossen?"),
        _sft_row("s2", [notar.id, form.id], (Cite(2),), "Welche Form braucht die Kündigung des Mietverhältnisses?"),
        _sft_row("s3", [form.id]),
    ]  # fmt: skip
    sft = tmp_path / "sft" / "dev.jsonl"
    sft.parent.mkdir()
    sft.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    sets = tmp_path / "sets.jsonl"
    build = ["build", "--items", str(sft), "--index", str(index), "--k", "1", "2", "--set-name", "sft_dev"]
    assert gold_passages.main([*build, "--out", str(sets)]) == 0
    built = read_records(sets)
    assert {r.id for r in built} == {"s1", "s2"} and {r.set for r in built} == {"sft_dev"} and len(built) > 2
    for r in built:
        assert r.gold_chunk_ids == [notar.id if r.id == "s1" else form.id]

    monkeypatch.setattr(gold_passages, "load_tokenizer", ByteTokenizer)
    monkeypatch.setattr(gold_passages, "MDebertaNLI", lambda *a, **k: lambda premise, hypothesis: 1.0)
    out, summary = tmp_path / "run.jsonl", tmp_path / "run.summary.json"
    capsys.readouterr()
    assert gold_passages.main(_decode_argv(root, sets, out, "--limit", "2")) == 0
    assert [(r.id, r.arm) for r in read_records(out)] == [(r.id, r.arm) for r in built[:2]]
    assert f"0 resumed, 2 decoded, {len(built) - 2} left" in capsys.readouterr().out and not summary.exists()
    assert gold_passages.main(_decode_argv(root, sets, out)) == 0
    assert [(r.id, r.arm) for r in read_records(out)] == [(r.id, r.arm) for r in built]
    assert f"2 resumed, {len(built) - 2} decoded, 0 left" in capsys.readouterr().out and summary.exists()


def test_as_is_sets_decode_each_sft_rows_own_prompt(tiny_setup, script, monkeypatch, tmp_path):  # noqa: F811
    """build --as-is writes one set per SFT row: its own question and passages in chunk_ids order, its
    kind, as gold the chunks its answer cites, loading no encoder and no retriever and writing no
    counts; a chunk id the index lacks, or a passage the index does not hold as the prompt shows it,
    stops the build. decode answers each set, its summary scores no gold position for them, and the
    citation metrics and the truncation audit read the records."""
    import json

    from lexhybrid.data.sft_format import Cite, Quote
    from lexhybrid.eval.gold_passages import read_chunks
    from lexhybrid.eval.records import read_records

    root, index = tiny_setup
    gold_passages = script("eval_gold_passages")
    chunks = read_chunks(index)
    notar = next(c for c in chunks if c.citation_id == "BeurkG §3")
    form = next(c for c in chunks if c.citation_id == "BGB §568")

    def row(id, prompt, answer=(), **kw):
        kind = kw.pop("kind", None)
        r = _sft_row(id, [c.id for c in prompt], answer, passages=[c.passage() for c in prompt], **kw)
        return {**r, "kind": kind or r["kind"]}

    rows = [row("s1", [form, notar], (Quote(2, 1),)), row("s2", [notar, form], (Cite(2), Cite(1))),
            row("s3", [form], question="  Gilt das?  ", kind="false_premise")]  # fmt: skip
    sft = tmp_path / "sft" / "dev.jsonl"
    sft.parent.mkdir()
    sft.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    def no_retrieval(*args, **kwargs):
        raise AssertionError("an as-is set retrieves nothing")

    monkeypatch.setattr(gold_passages, "BgeM3Encoder", no_retrieval)
    monkeypatch.setattr(gold_passages, "fused_retriever", no_retrieval)
    sets = tmp_path / "sets.jsonl"
    build = ["build", "--items", str(sft), "--index", str(index), "--as-is", "--set-name", "sft_dev"]
    assert gold_passages.main([*build, "--out", str(sets)]) == 0
    built = read_records(sets)
    assert [(r.id, r.arm, r.set, r.kind, r.gold_chunk_ids, [c["id"] for c in r.chunks]) for r in built] == [
        ("s1", "as_is", "sft_dev", "answerable", [notar.id], [form.id, notar.id]),
        ("s2", "as_is", "sft_dev", "answerable", [form.id, notar.id], [notar.id, form.id]),
        ("s3", "as_is", "sft_dev", "false_premise", [], [form.id]),
    ]
    assert [r.question for r in built] == ["Welche Form braucht die Kündigung?"] * 2 + ["  Gilt das?  "]
    assert (
        all(r.system == "" and not r.raw_answer for r in built)
        and not (tmp_path / "sets.counts.json").exists()
    )
    for bad, match in (
        ({**rows[0], "chunk_ids": [form.id, "gii:missing#1"]}, "not in the index"),
        ({**rows[0], "chunk_ids": [notar.id, form.id]}, "passages are not the index's"),
    ):
        sft.write_text(json.dumps(bad, ensure_ascii=False) + "\n")
        with pytest.raises(ValueError, match=match):
            gold_passages.main([*build, "--out", str(tmp_path / "bad.jsonl")])

    monkeypatch.setattr(gold_passages, "load_tokenizer", ByteTokenizer)
    monkeypatch.setattr(gold_passages, "MDebertaNLI", lambda *a, **k: lambda premise, hypothesis: 1.0)
    out, summary = tmp_path / "run.jsonl", tmp_path / "run.summary.json"
    assert gold_passages.main(_decode_argv(root, sets, out)) == 0
    decoded = read_records(out)
    assert [(r.id, r.kind, r.chunks, r.gold_chunk_ids) for r in decoded] == [
        (r.id, r.kind, r.chunks, r.gold_chunk_ids) for r in built
    ]
    assert (
        all(r.system == "hybrid" and r.raw_answer for r in decoded) and json.loads(summary.read_text()) == []
    )
    citation = tmp_path / "run.citation.summary.json"
    metrics = ["--metrics-out", str(tmp_path / "run.citation.metrics.jsonl"), "--summary-out", str(citation)]
    assert script("evaluate_citation").main([str(out), *metrics]) == 0
    [s] = json.loads(citation.read_text())
    assert (s["arm"], s["records"], s["should_refuse"], s["recall_records"]) == ("as_is", 3, 1, 2)
    assert script("audit_truncation").main([str(out)]) in (0, 1)


def test_eval_gold_passages_end_to_end_tiny(tiny_setup, script, monkeypatch, tmp_path, capsys):  # noqa: F811
    """build writes the prompt sets from a real index (BM25 ranks the hard tier), counts per tier and k
    the items it skipped for too few distractors and warns about them; decode answers each set greedily
    under the pointer FSM, verifies it and scores it per tier and k; score counts first-pointer
    selection, any gold citation, the model's own refusal, the verifier's pass and accuracy by gold
    position. A build without a single set fails and leaves no sets file a later job could reuse."""
    import json
    import re
    from dataclasses import replace

    from lexhybrid.eval.gold_passages import score
    from lexhybrid.eval.records import read_records

    root, index = tiny_setup
    gold_passages = script("eval_gold_passages")
    items = _gold_items(tmp_path)
    sets = tmp_path / "sets.jsonl"
    build = ["build", "--items", str(items), "--index", str(index), "--k", "1", "2"]
    assert gold_passages.main([*build, "--out", str(sets)]) == 0
    built = read_records(sets)
    # q1: BM25 ranks two other BeurkG chunks, and only BGB §568 is of another act; q2: BM25 finds only the gold
    assert [(r.id, r.arm) for r in built] == [("q1", "gold_easy_k1"), ("q1", "gold_hard_k1"), ("q1", "gold_hard_k2"),
                                              ("q2", "gold_easy_k1"), ("q2", "gold_easy_k2")]  # fmt: skip
    assert {r.set for r in built} == {"gerlayqa"} and all(r.system == "" and not r.raw_answer for r in built)
    assert json.loads((tmp_path / "sets.counts.json").read_text()) == {
        "items": 3,
        "with_gold": 2,
        "sets": {
            "gold_easy_k1": {"built": 2, "skipped": 0},
            "gold_easy_k2": {"built": 1, "skipped": 1},
            "gold_hard_k1": {"built": 1, "skipped": 1},
            "gold_hard_k2": {"built": 1, "skipped": 1},
        },
    }
    printed = capsys.readouterr()
    assert "GOLD SETS gold_easy_k1 built=2 skipped=0" in printed.out
    assert "GOLD SETS gold_hard_k2 built=1 skipped=1" in printed.out
    assert "WARNING: gold_hard_k2 lost 1 of 2 items" in printed.err and "gold_easy_k1 lost" not in printed.err
    assert not (tmp_path / "sets.tmp").exists()
    none = tmp_path / "none" / "sets.jsonl"
    no_gold = _gold_items(tmp_path, [{"id": "q", "question": "?"}])
    build = ["build", "--items", str(no_gold), "--index", str(index)]
    assert gold_passages.main([*build, "--out", str(none)]) == 1 and not none.exists()

    monkeypatch.setattr(gold_passages, "load_tokenizer", ByteTokenizer)
    monkeypatch.setattr(gold_passages, "MDebertaNLI", lambda *a, **k: lambda premise, hypothesis: 1.0)
    out, summary = tmp_path / "hybrid_s42.jsonl", tmp_path / "hybrid_s42.summary.json"
    assert gold_passages.main(_decode_argv(root, sets, out)) == 0
    decoded = read_records(out)
    assert [(r.id, r.arm, r.chunks, r.gold_chunk_ids) for r in decoded] == [
        (r.id, r.arm, r.chunks, r.gold_chunk_ids) for r in built
    ]
    for r in decoded:
        assert (
            r.system == "hybrid"
            and r.seed == 42
            and r.raw_answer
            and 0 < r.new_tokens <= r.max_new_tokens == 8
        )
        assert r.verifier["refused"] == r.refused and r.rendered
        assert all(1 <= int(k) <= len(r.chunks) for k in re.findall(r"<\|c(\d+)\|>", r.raw_answer)), "the FSM"
    summaries = json.loads(summary.read_text())
    assert summaries == score(decoded)
    assert [(s["tier"], s["k"], s["records"]) for s in summaries] == [("easy", 1, 2), ("hard", 1, 1),
                                                                      ("hard", 2, 1), ("easy", 2, 1)]  # fmt: skip
    for s in summaries:
        assert 0 <= s["selection_accuracy"] <= s["any_gold_cited"] <= 1 and 0 <= s["verifier_pass_rate"] <= 1
        assert sum(p["records"] for p in s["by_position"].values()) == s["records"]

    r = next(r for r in decoded if r.arm == "gold_hard_k2")
    g = [c["id"] for c in r.chunks].index(r.gold_chunk_ids[0]) + 1
    other = 1 if g != 1 else 2
    scripted = [
        replace(r, raw_answer=f"Ja <|cite|><|c{g}|>.", refused=False),
        replace(r, raw_answer=f"Ja <|cite|><|c{other}|> und <|q|><|c{g}|><|s1|>.", refused=True),
        replace(r, raw_answer="<|unanswerable|>", refused=True),
        replace(r, raw_answer="<|q|> kaputt", refused=True),
    ]
    [s] = score(scripted)
    assert (s["selection_accuracy"], s["any_gold_cited"], s["refusal_rate"], s["verifier_pass_rate"]) == (
        0.25, 0.5, 0.25, 0.25)  # fmt: skip
    assert s["by_position"] == {str(g): {"records": 4, "selection_accuracy": 0.25}}


def test_gold_selection_counts_an_invalid_first_pointer():
    """A first pointer outside the prompt selects nothing: the next pointer does not take its place, K = 0
    does not wrap to the last passage, and K too large does not raise."""
    from dataclasses import replace

    from lexhybrid.eval.gold_passages import outcome
    from lexhybrid.eval.records import AnswerRecord, chunk_dict

    chunks = [_test_chunk("gii:bgb:569#1", "BGB §569"), _test_chunk("gii:bgb:568#1", "BGB §568")]
    record = AnswerRecord("q", "tiny", "hybrid", "gold_hard_k1", "?", gold_chunk_ids=[chunks[1].id],
                          chunks=[chunk_dict(c) for c in chunks])  # fmt: skip

    def seen(raw):
        o = outcome(replace(record, raw_answer=raw))
        return o["selected"], o["any_gold"]

    assert seen("Ja <|cite|><|c2|>.") == (True, True)
    assert seen("Ja <|cite|><|c1|> <|cite|><|c2|>.") == (False, True)
    assert seen("Ja <|cite|><|c0|> <|cite|><|c2|>.") == (False, True)
    assert seen("Ja <|cite|><|c0|>.") == (False, False)
    assert seen("Ja <|cite|><|c3|>.") == (False, False)


def test_eval_gold_passages_decode_appends_and_resumes(tiny_setup, script, monkeypatch, tmp_path, capsys):  # noqa: F811
    """decode appends each record as it is decoded, so a job stopped midway leaves whole records and no
    summary; a rerun cuts a line left short (even inside a character), decodes only the rest and ends
    with what an uninterrupted run writes, a finished file loads no model, and records of another system,
    budget or prompt set are refused rather than mixed in."""
    from dataclasses import replace

    from lexhybrid.eval.records import read_records, write_records

    class Preempted(Exception):
        pass

    root, index = tiny_setup
    gold_passages = script("eval_gold_passages")
    sets = tmp_path / "sets.jsonl"
    build = ["build", "--items", str(_gold_items(tmp_path)), "--index", str(index), "--k", "1", "2"]
    assert gold_passages.main([*build, "--out", str(sets)]) == 0
    built = read_records(sets)
    monkeypatch.setattr(gold_passages, "load_tokenizer", ByteTokenizer)
    monkeypatch.setattr(gold_passages, "MDebertaNLI", lambda *a, **k: lambda premise, hypothesis: 1.0)
    real, calls = gold_passages.decode, []

    def counting(stop_at=None):
        def decode(record, *args):
            calls.append((record.id, record.arm))
            if len(calls) == stop_at:
                raise Preempted
            return real(record, *args)

        return decode

    out, summary = tmp_path / "run.jsonl", tmp_path / "run.summary.json"
    monkeypatch.setattr(gold_passages, "decode", counting(stop_at=3))
    with pytest.raises(Preempted):
        gold_passages.main(_decode_argv(root, sets, out))
    assert [(r.id, r.arm) for r in read_records(out)] == [(r.id, r.arm) for r in built[:2]]
    assert not summary.exists()

    with open(out, "ab") as f:
        f.write('{"id": "q1", "question": "Kü'.encode()[:-1])  # the cut falls inside the ü
    calls.clear()
    capsys.readouterr()
    monkeypatch.setattr(gold_passages, "decode", counting())
    assert gold_passages.main(_decode_argv(root, sets, out)) == 0
    assert calls == [(r.id, r.arm) for r in built[2:]]
    assert "GOLD DECODE 2 resumed, 3 decoded" in capsys.readouterr().out
    assert summary.exists()
    fresh = tmp_path / "fresh.jsonl"
    assert gold_passages.main(_decode_argv(root, sets, fresh)) == 0
    assert read_records(out) == read_records(fresh) and out.read_bytes() == fresh.read_bytes()

    def no_model(*args):
        raise AssertionError("nothing is left to decode")

    monkeypatch.setattr(gold_passages, "load_model", no_model)
    capsys.readouterr()
    assert gold_passages.main(_decode_argv(root, sets, out)) == 0
    assert "GOLD DECODE 5 resumed, 0 decoded" in capsys.readouterr().out

    before = out.read_bytes()
    other_sets = tmp_path / "other_sets.jsonl"
    write_records(other_sets, [replace(r, chunks=r.chunks[::-1]) for r in built])
    for argv in (
        [*_decode_argv(root, sets, out), "--system", "transformer"],
        [*_decode_argv(root, sets, out), "--seed", "43"],
        [*_decode_argv(root, sets, out), "--max-new-tokens", "9"],
        _decode_argv(root, other_sets, out),
    ):
        with pytest.raises(SystemExit, match="not resuming"):
            gold_passages.main(argv)
    assert out.read_bytes() == before


def test_eval_gold_passages_decodes_in_the_requested_dtype(script, monkeypatch):
    """--dtype bfloat16 casts the loaded model's parameters (a large model's longest prompts then fit the GPU)
    but not its fp32 buffers such as RoPE's inv_freq; the default keeps fp32, and decode passes the flag
    through to the loader."""
    import pytest
    import torch

    gold_passages = script("eval_gold_passages")

    def stub(*args):
        model = torch.nn.Linear(2, 2).train()
        model.register_buffer("inv_freq", torch.full((2,), 1 / 3), persistent=False)  # as RoPE's
        return model

    monkeypatch.setattr(gold_passages, "load_sft_model", stub)
    assert gold_passages.load_model(None, "x.ckpt", "cpu", "Qwen/x").weight.dtype == torch.float32
    model = gold_passages.load_model(None, "x.ckpt", "cpu", "Qwen/x", dtype="bfloat16")
    assert model.weight.dtype == torch.bfloat16 and not model.training
    assert model.inv_freq.dtype == torch.float32  # a buffer rounded to bf16 garbles long-range positions
    seen = []

    def loader(*args):
        seen.append(args[-1])
        raise RuntimeError("stop after loading")

    from types import SimpleNamespace

    monkeypatch.setattr(
        gold_passages, "read_records", lambda path: [SimpleNamespace(id="q", arm="gold_easy_k1")]
    )
    monkeypatch.setattr(gold_passages, "resumed", lambda args, sets: [])
    monkeypatch.setattr(gold_passages, "load_model", loader)
    argv = ["decode", "--sets", "s.jsonl", "--system", "lora", "--hf-model", "Qwen/x", "--checkpoint", "c",
            "--out", "o.jsonl", "--summary", "s.json"]  # fmt: skip
    for extra, dtype in (([], "float32"), (["--dtype", "bfloat16"], "bfloat16")):
        with pytest.raises(RuntimeError, match="stop after loading"):
            gold_passages.main([*argv, *extra])
        assert seen[-1] == dtype
    with pytest.raises(SystemExit):
        gold_passages.main([*argv, "--dtype", "float16"])


def test_eval_gold_passages_on_the_arm_node_runs_from_its_own_environment(tmp_path):
    """On aarch64 every Python step (decode and both scorers) runs from .venv-arm with the repository on
    PYTHONPATH, since the x86 .venv cannot run there."""
    import os
    import shutil
    import subprocess

    from tests.conftest import REPO_ROOT

    repo, scratch, calls, bin_dir = (
        tmp_path / "repo",
        tmp_path / "scratch",
        tmp_path / "calls.log",
        tmp_path / "bin",
    )
    (repo / "scripts" / "slurm").mkdir(parents=True)
    wrapper = repo / "scripts" / "slurm" / "eval_gold_passages.sh"
    shutil.copy(REPO_ROOT / "scripts" / "slurm" / "eval_gold_passages.sh", wrapper)
    bin_dir.mkdir()
    (bin_dir / "uname").write_text("#!/usr/bin/env bash\necho aarch64\n")
    arm = repo / ".venv-arm" / "bin" / "python"
    arm.parent.mkdir(parents=True)
    arm.write_text(
        '#!/usr/bin/env bash\necho "arm:$PYTHONPATH $1" >> "$CALLS"\necho "$*" >> "$CALLS.args"\n'
        "while [[ $# -gt 0 ]]; do\n"
        '  if [[ "$1" == --summary ]]; then echo "{}" > "$2"; fi\n  shift\ndone\n'
    )
    for exe in (bin_dir / "uname", arm):
        exe.chmod(0o755)
    ckpt = scratch / "outputs" / "sft_hybrid_s42" / "checkpoints" / "last.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.touch()
    sets = repo / "logs" / "shared" / "sets.jsonl"
    sets.parent.mkdir(parents=True)
    sets.touch()
    result = subprocess.run(
        ["bash", str(wrapper)], cwd=repo, capture_output=True, text=True,
        env={"PATH": f"{bin_dir}:{os.environ['PATH']}", "HOME": str(tmp_path), "USER": "tester",
             "SLURM_SUBMIT_DIR": str(repo), "SLURM_JOB_ID": "5", "SCRATCH_ROOT": str(scratch), "CALLS": str(calls),
             "RUNS": "hybrid:42:sft_hybrid_s42:hybrid_legal_base", "SETS": str(sets), "DTYPE": "bfloat16"},
    )  # fmt: skip
    assert result.returncode == 0 and result.stdout.endswith("EVAL GOLD PASSAGES DONE\n"), (
        result.stdout + result.stderr
    )
    assert calls.read_text().splitlines() == [
        f"arm:{repo} scripts/{s}.py" for s in ("eval_gold_passages", "evaluate_citation", "audit_truncation")
    ]
    decode = calls.with_name("calls.log.args").read_text().splitlines()[0].split()
    assert decode[decode.index("--dtype") + 1] == "bfloat16"


def test_eval_gold_passages_wrapper_keeps_runs_apart_and_reuses_sets(tmp_path):
    """Two runs of one system and seed (SFT and DPO) get their own outputs, so the second is not skipped as
    already scored; a requeue skips the runs that have a summary and does not rebuild the sets; SETS makes
    another job decode those sets, without needing the items or the index, and a missing SETS is built.
    Missing GerLayQA items are fetched (online for that command only) and moved into place; other missing
    items are fatal. SET=sft_dev builds from the SFT dev split; LIMIT reaches every decode. Every run decoded
    whole, and only such a run, gets its citation metrics beside its records and a truncation audit, also on
    a requeue; truncated decodes are a finding, a failed audit fails the job. AS_IS=1 builds the SFT rows'
    own prompts and needs SET=sft_dev."""
    import os
    import shutil
    import subprocess

    from tests.conftest import REPO_ROOT

    repo, scratch, calls = tmp_path / "repo", tmp_path / "scratch", tmp_path / "calls.log"
    (repo / "scripts" / "slurm").mkdir(parents=True)
    wrapper = repo / "scripts" / "slurm" / "eval_gold_passages.sh"
    shutil.copy(REPO_ROOT / "scripts" / "slurm" / "eval_gold_passages.sh", wrapper)
    fake = repo / ".venv" / "bin" / "python"
    fake.parent.mkdir(parents=True)
    fake.write_text(
        '#!/usr/bin/env bash\necho "$* offline=$HF_HUB_OFFLINE" >> "$CALLS"\nfetch="$1"\n'
        'if [[ "$1" == scripts/audit_truncation.py ]]; then exit "${AUDIT_EXIT:-0}"; fi\nwhile [[ $# -gt 0 ]]; do\n'
        '  if [[ "$1" == --out && "$fetch" == -m ]]; then mkdir -p "$2"; touch "$2/items.jsonl" "$2/info.json"\n'
        '  elif [[ "$1" == --limit ]]; then limited=1\n'  # sets are left: no summary
        '  elif [[ "$1" == --out || ( "$1" == --summary && -z "${limited:-}" ) ]]; then\n'
        '    mkdir -p "$(dirname "$2")"; echo "{}" > "$2"\n  fi\n'
        "  shift\ndone\n"
    )
    fake.chmod(0o755)
    for experiment in ("sft_hybrid_s42", "dpo_hybrid_s42"):
        ckpt = scratch / "outputs" / experiment / "checkpoints" / "last.ckpt"
        ckpt.parent.mkdir(parents=True)
        ckpt.touch()
    items = scratch / "data" / "evalsets" / "gerlayqa" / "items.jsonl"
    index = scratch / "data" / "index" / "v1" / "index.json"
    for path in (items, index):
        path.parent.mkdir(parents=True)
        path.touch()
    runs = "hybrid:42:sft_hybrid_s42:hybrid_legal_base hybrid:42:dpo_hybrid_s42:hybrid_legal_base"

    def call(job, **env):
        calls.unlink(missing_ok=True)
        return subprocess.run(
            ["bash", str(wrapper)], cwd=repo, capture_output=True, text=True,
            env={"PATH": os.environ["PATH"], "HOME": str(tmp_path), "USER": "tester", "SLURM_SUBMIT_DIR": str(repo),
                 "SLURM_JOB_ID": job, "SCRATCH_ROOT": str(scratch), "CALLS": str(calls), "RUNS": runs, **env},
        )  # fmt: skip

    def called(*scripts):
        """The arguments of each call of these scripts (default: the gold-passage script and the fetch)."""
        lines = [line.split() for line in calls.read_text().splitlines()] if calls.exists() else []
        return [line[1:] for line in lines if line[0] in (scripts or ("scripts/eval_gold_passages.py", "-m"))]

    def scored():
        """The records each scorer read, in call order."""
        return [[c[0] for c in called(f"scripts/{s}.py")] for s in ("evaluate_citation", "audit_truncation")]

    def run(job, **env):
        result = call(job, **env)
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.endswith("EVAL GOLD PASSAGES DONE\n")
        return called()

    def value(call, flag):
        return call[call.index(flag) + 1]

    hf = run("6", RUNS="lora:42:sft_hybrid_s42:hf=Qwen/Qwen3-8B-Base")[-1]
    assert value(hf, "--hf-model") == "Qwen/Qwen3-8B-Base" and "--model-config" not in hf
    first = run("7")
    assert [c[0] for c in first] == ["build", "decode", "decode"]
    assert value(first[0], "--out") == "logs/eval_gold_passages_7/sets.jsonl"
    records = [
        f"logs/eval_gold_passages_7/hybrid_s42_{e}.jsonl" for e in ("sft_hybrid_s42", "dpo_hybrid_s42")
    ]
    assert [value(c, "--out") for c in first[1:]] == records
    assert scored() == [records, records]
    citation = called("scripts/evaluate_citation.py")[0]
    assert (value(citation, "--metrics-out"), value(citation, "--summary-out")) == (
        "logs/eval_gold_passages_7/hybrid_s42_sft_hybrid_s42.citation.metrics.jsonl",
        "logs/eval_gold_passages_7/hybrid_s42_sft_hybrid_s42.citation.summary.json",
    )
    assert run("7") == [], "a requeue decodes nothing and keeps the sets"
    assert scored() == [records, records], "and scores every decoded run"

    items.unlink()
    index.unlink()
    shared = "logs/eval_gold_passages_7/sets.jsonl"
    reused = run("8", SETS=shared)
    assert [c[0] for c in reused] == ["decode", "decode"]
    assert {value(c, "--sets") for c in reused} == {shared}
    assert value(reused[0], "--out") == "logs/eval_gold_passages_8/hybrid_s42_sft_hybrid_s42.jsonl"

    failed = call("9", SETS="logs/shared/sets.jsonl")
    assert failed.returncode == 1 and "FATAL: no index" in failed.stdout
    [fetch] = called()
    assert fetch[0] == "lexhybrid.data.evalsets.gerlayqa" and value(fetch, "--limit") == "0"
    assert fetch[-1] == "offline=0"
    assert items.is_file() and (items.parent / "info.json").is_file()
    assert not list(items.parent.parent.glob(".fetch.*")), "the temporary directory is removed"
    index.touch()
    built = run("9", SETS="logs/shared/sets.jsonl", LIMIT="2")
    assert [c[0] for c in built] == ["build", "decode", "decode"]
    assert value(built[0], "--out") == "logs/shared/sets.jsonl"
    assert value(built[0], "--set-name") == "gerlayqa" and {c[-1] for c in built} == {"offline=1"}
    assert {value(c, "--sets") for c in built[1:]} == {"logs/shared/sets.jsonl"}
    assert {value(c, "--limit") for c in built[1:]} == {"2"}
    assert scored() == [[], []], "nothing is scored while sets are left"
    assert all("--limit" not in c for c in first), "no LIMIT, no limit"

    failed = call("10", SET="gerlerb")
    assert failed.returncode == 1 and "FATAL: no items" in failed.stdout and called() == []
    dev = scratch / "data" / "sft" / "dev.jsonl"
    dev.parent.mkdir(parents=True)
    dev.touch()
    sft = run("11", SET="sft_dev")
    assert value(sft[0], "--items") == str(dev) and value(sft[0], "--set-name") == "sft_dev"
    assert value(sft[0], "--k") == "1" and "--as-is" not in sft[0]

    failed = call("12", AS_IS="1")
    assert failed.returncode == 1 and "FATAL: AS_IS=1" in failed.stdout and called() == []
    as_is = run("12", SET="sft_dev", AS_IS="1")
    assert [c[0] for c in as_is] == ["build", "decode", "decode"] and "--k" not in as_is[0]
    assert (
        "--as-is" in as_is[0]
        and value(as_is[0], "--items") == str(dev)
        and value(as_is[0], "--set-name") == "sft_dev"
    )
    assert scored() == [[value(c, "--out") for c in as_is[1:]]] * 2
    finding = call("13", SET="sft_dev", AS_IS="1", AUDIT_EXIT="1")
    assert finding.returncode == 0 and finding.stdout.endswith("EVAL GOLD PASSAGES DONE\n")
    assert finding.stdout.count("FINDING: hybrid_s42_") == 2 and len(scored()[1]) == 2, (
        "a finding stops nothing"
    )
    failed = call("14", SET="sft_dev", AS_IS="1", AUDIT_EXIT="2")
    assert failed.returncode == 2 and "FATAL: the truncation audit of hybrid_s42_sft" in failed.stdout
    assert len(scored()[1]) == 1, "the job stops at the failed audit"


def test_answer_records_keep_unicode_line_separators_and_string_ids(tmp_path):
    """JSON keeps U+2028/U+0085 raw under ensure_ascii=False: the reader splits on newlines only.
    Numeric ids become strings, so every consumer joins on one type."""
    from lexhybrid.eval.records import AnswerRecord, read_records, write_records

    rec = AnswerRecord(7, "tiny", "hybrid", "default", "Satz eins Satz zwei\u0085Ende")
    write_records(tmp_path / "r.jsonl", [rec])
    (back,) = read_records(tmp_path / "r.jsonl")
    assert back.question == rec.question and back.id == "7"
