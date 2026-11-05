"""The judge's HTTP client, its CLI, and its calibration items and scoring (all offline)."""

import json
from types import SimpleNamespace

import pytest
import requests

from lexhybrid.eval.judge import METRICS, Judge, OpenAICompatible, judge_from_args, read_relevancy


class FakeResponse:
    def __init__(self, content="", status_code=200):
        self.content, self.status_code = content, status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        return {"choices": [{"message": {"content": self.content}}]}


def fake_post(monkeypatch, reply):
    """Patch ``requests.post``; ``reply(prompt)`` gives the completion. Returns the list of requests."""
    calls = []

    def post(url, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json})
        return FakeResponse(reply(json["messages"][0]["content"]))

    monkeypatch.setattr(requests, "post", post)
    return calls


def test_openai_compatible_client(monkeypatch):
    """One user message at temperature 0 to <base>/chat/completions, the key as a bearer token only
    when there is one; the key comes from the named environment variable."""
    calls = fake_post(monkeypatch, lambda prompt: f"echo {prompt}")
    assert OpenAICompatible("http://localhost:8000/v1/", "m", "k")("hallo") == "echo hallo"
    assert calls[0] == {
        "url": "http://localhost:8000/v1/chat/completions",
        "headers": {"Authorization": "Bearer k"},
        "json": {"model": "m", "messages": [{"role": "user", "content": "hallo"}], "temperature": 0},
    }
    OpenAICompatible("http://h/v1", "m")("x")
    assert calls[1]["headers"] == {}
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(None))
    assert OpenAICompatible("http://h/v1", "m")("x") == ""

    import argparse

    args = argparse.Namespace(base_url="http://h/v1", model="m", api_key_env="JUDGE_TEST_KEY", cache_dir="c")
    monkeypatch.delenv("JUDGE_TEST_KEY", raising=False)
    with pytest.raises(SystemExit, match="JUDGE_TEST_KEY"):
        judge_from_args(args)
    monkeypatch.setenv("JUDGE_TEST_KEY", "secret")
    judge = judge_from_args(args)
    assert judge._complete.api_key == "secret" and judge.base_url == "http://h/v1"


def test_http_failures_back_off_then_raise(tmp_path, monkeypatch):
    """429, 5xx, lost connections and timeouts wait 2, 4, 8, 16 s between five tries and then raise; any
    other 4xx raises at once; replies judged before the failure stay cached."""
    sleeps = []
    monkeypatch.setattr("lexhybrid.eval.judge.time.sleep", sleeps.append)

    def script_post(*steps):
        steps = list(steps)

        def post(*a, **k):
            step = steps.pop(0)
            if isinstance(step, Exception):
                raise step
            return step

        monkeypatch.setattr(requests, "post", post)
        return steps

    client = OpenAICompatible("http://h/v1", "m")
    script_post(FakeResponse(status_code=503), requests.ConnectionError(), requests.Timeout(),
                FakeResponse(status_code=429), FakeResponse("ok"))  # fmt: skip
    assert client("x") == "ok" and sleeps == [2.0, 4.0, 8.0, 16.0]

    sleeps.clear()
    left = script_post(*[FakeResponse(status_code=500) for _ in range(5)], FakeResponse("never"))
    with pytest.raises(requests.HTTPError):
        client("x")
    assert len(left) == 1 and sleeps == [2.0, 4.0, 8.0, 16.0]
    script_post(*[requests.ConnectionError() for _ in range(5)])
    with pytest.raises(requests.ConnectionError):
        client("x")

    sleeps.clear()
    left = script_post(FakeResponse(status_code=401), FakeResponse("never"))
    with pytest.raises(requests.HTTPError):
        client("x")
    assert len(left) == 1 and not sleeps

    judge = Judge(client, "m", tmp_path, base_url="http://h/v1")
    script_post(FakeResponse('{"relevancy": 1.0}'), *[requests.Timeout() for _ in range(5)])
    assert judge.ask("erste Frage", read_relevancy)[0] == 1.0
    with pytest.raises(requests.Timeout):
        judge.ask("zweite Frage", read_relevancy)
    script_post(FakeResponse('{"relevancy": 0.5}'))  # one reply left: the first prompt must not ask
    assert judge.ask("erste Frage", read_relevancy)[0] == 1.0
    assert judge.ask("zweite Frage", read_relevancy)[0] == 0.5


def test_cache_key_and_bad_cache_entries(tmp_path, monkeypatch):
    """Another endpoint, model or prompt version never sees a cached verdict (a trailing slash is the same
    endpoint), and a cached reply that does not parse is asked again and replaced, never served."""
    asked = []

    def complete(prompt):
        asked.append(prompt)
        return '{"relevancy": 1.0}'

    def ask(model="m", url="http://a/v1"):
        return Judge(complete, model, tmp_path, base_url=url).ask("p", read_relevancy)

    assert ask() == (1.0, {"relevancy": 1.0}) and len(asked) == 1
    ask(url="http://a/v1/")
    assert len(asked) == 1
    ask(url="http://b/v1")
    assert len(asked) == 2
    ask(model="n")
    assert len(asked) == 3
    monkeypatch.setattr("lexhybrid.eval.judge.PROMPT_VERSION", 99)
    ask()
    assert len(asked) == 4
    monkeypatch.undo()

    for entry in tmp_path.glob("*.json"):
        entry.write_text('{"response": "kaputt"}')
    assert ask() == (1.0, {"relevancy": 1.0}) and len(asked) == 5
    ask()
    assert len(asked) == 5


def _reply(prompt):
    if prompt.startswith("You grade"):
        return (
            '{"verdict": "correct", "reason": "-"}' if "Antwort A" in prompt else '{"verdict": "incorrect"}'
        )
    if prompt.startswith("You judge which"):
        return '{"useful": [true]}'
    if prompt.startswith("You check how much"):
        return '{"statements": [{"statement": "s", "attributed": true}]}'
    if prompt.startswith("You check whether"):
        return '{"statements": [{"statement": "s", "supported": true}]}'
    return '{"relevancy": 1.0, "reason": "-"}'


def _validated_report(**changes) -> dict:
    from lexhybrid.eval.judge import METRICS, PROMPT_VERSION

    return {"judge": {"model": "m", "base_url": "http://h/v1/"}, "split": "validation", "meets": True,
            "prompt_version": PROMPT_VERSION, "types": {m: {"meets": True} for m in METRICS}, **changes}  # fmt: skip


def test_judge_answers_cli(tmp_path, monkeypatch, script, capsys):
    """Scores JSONL aligned with the input and numeric only (bootstrap_compare reads it), details and a
    summary per (system, seed, arm); a rerun makes no request; --dry-run prints a prompt and sends nothing."""
    from lexhybrid.eval.records import AnswerRecord, chunk_dict, write_records
    from lexhybrid.retrieval.chunking import Chunk

    chunk = chunk_dict(Chunk(
        id="gii:bgb:568#1", doc_id="gii:bgb:568", source="gii", jurisdiction="DE", doc_type="statute",
        citation_id="BGB §568", hierarchy=(), sentences=("Die Kündigung bedarf der Form.",), labels=("BGB §568 Abs. 1",),
        valid_from=None, valid_to=None, licence="DE-UrhG-5", commercial_safe=True, research_only=False, url="",
    ))  # fmt: skip
    records = [
        AnswerRecord("q1", "tiny", "hybrid", "default", "Welche Form?", seed=1, gold_answers=["schriftlich"],
                     chunks=[chunk], rendered="Antwort A."),
        AnswerRecord("q2", "tiny", "hybrid", "default", "Wer?", seed=1, kind="unanswerable", refused=True),
        AnswerRecord("q1", "tiny", "transformer", "default", "Welche Form?", seed=1, gold_answers=["schriftlich"],
                     chunks=[chunk], rendered="Antwort B."),
    ]  # fmt: skip
    write_records(tmp_path / "answers.jsonl", records)
    calls = fake_post(monkeypatch, _reply)
    report = tmp_path / "validated.json"
    report.write_text(json.dumps(_validated_report()))
    argv = ["--records", str(tmp_path / "answers.jsonl"), "--out", str(tmp_path / "judge" / "s.jsonl"),
            "--base-url", "http://h/v1", "--model", "m", "--cache-dir", str(tmp_path / "cache"),
            "--calibration", str(report)]  # fmt: skip
    assert script("judge_answers").main(argv) == 0
    rows = script("bootstrap_compare").read_metrics(str(tmp_path / "judge" / "s.jsonl"))
    full = dict.fromkeys(("answer_relevancy", "context_precision", "context_recall", "faithfulness"), 1.0)
    assert rows == [{"id": "q1", "correctness": 1.0, **full}, {"id": "q2", "correctness": 1.0},
                    {"id": "q1", "correctness": 0.0, **full}]  # fmt: skip
    assert len(calls) == 8  # precision and recall of q1 are asked once: the second system hits the cache
    details = [json.loads(line) for line in (tmp_path / "judge" / "s.details.jsonl").read_text().splitlines()]
    assert details[2]["system"] == "transformer"
    assert details[2]["details"]["correctness"]["verdict"] == "incorrect"
    summary = json.loads((tmp_path / "judge" / "s.summary.json").read_text())
    hybrid, transformer = summary["groups"]
    assert (hybrid["system"], hybrid["seed"], hybrid["n"], hybrid["refused"]) == ("hybrid", 1, 2, 1)
    assert hybrid["correctness"] == {"mean": 1.0, "n": 2, "unparsed": 0} and hybrid["faithfulness"]["n"] == 1
    assert transformer["correctness"]["mean"] == 0.0 and summary["judge"]["model"] == "m"

    assert script("judge_answers").main(argv) == 0 and len(calls) == 8
    scores = dict.fromkeys(METRICS)
    [g] = script("judge_answers").summarize(records[:1], [(scores, {"faithfulness": {"raw": "kaputt"}})])
    assert g["faithfulness"] == {"mean": None, "n": 0, "unparsed": 1}
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("dry run sent a request"))
    capsys.readouterr()
    assert script("judge_answers").main([*argv, "--dry-run", "--limit", "1"]) == 0
    assert capsys.readouterr().out.startswith("You grade a system's answer")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def test_perturbations_change_the_gold_answer(script):
    perturb = script("build_judge_calibration").perturbations
    assert perturb("Nach § 573 BGB beträgt die Frist 14 Tage.") == {
        "section": "Nach § 574 BGB beträgt die Frist 14 Tage.",
        "number": "Nach § 573 BGB beträgt die Frist 15 Tage.",
    }
    assert perturb("Der Vermieter kann kündigen.") == {"negation": "Der Vermieter kann nicht kündigen."}
    assert perturb("Eine Frist ist nicht vorgesehen.")["negation"] == "Eine Frist ist vorgesehen."
    assert perturb("Keine Form ist nötig.")["negation"] == "Eine Form ist nötig."
    assert perturb("Der Vermieter darf das nicht.")["negation"] == "Der Vermieter darf das."
    assert perturb("1. Die Frist ist um.") == {"negation": "1. Die Frist ist nicht um."}
    assert perturb("keinen Anspruch hat er.")["negation"] == "einen Anspruch hat er."
    assert perturb("Ja.") == {}
    # a number inside a § reference is the section's to change, never the number variant's
    assert perturb("Nach §§ 535, 536 BGB gilt das.") == {"section": "Nach §§ 536, 536 BGB gilt das."}
    assert perturb("§ 573 Abs. 2 Satz 1 BGB: 14 Tage.") == {
        "section": "§ 574 Abs. 2 Satz 1 BGB: 14 Tage.",
        "number": "§ 573 Abs. 2 Satz 1 BGB: 15 Tage.",
    }
    # no second "nicht" on "nicht nur", no garbled word from "Keinesfalls"
    assert perturb("Das ist nicht nur hier so.") == {}
    assert perturb("Keinesfalls gilt das.") == {}


def _set_dir(root, name, **files):
    d = root / name
    d.mkdir(parents=True)
    (d / "info.json").write_text(json.dumps({"name": name}))
    for stem, rows in files.items():
        _write_jsonl(d / f"{stem}.jsonl", rows)
    return d


def _qa_row(id, question, answers, context="", context_id="", gold=()):
    return {"id": id, "question": question, "answers": answers, "context": context, "context_id": context_id,
            "gold": list(gold)}  # fmt: skip


def test_calibration_items_and_split(tmp_path, script):
    """Every production prompt gets its items; the split goes by context or provision (sibling questions
    stay together), is the same in every run, and no answer, context or passage crosses it; ids carry the
    set name; empty gold answers are skipped; only reviewed NLI pairs are used."""
    b = script("build_judge_calibration")
    legalquad = [
        _qa_row(f"legalquad:q{i}", f"Was gilt für Thema{i}?", [f"Ergebnis{i}"],
                f"Entscheidung {i} über Thema{i} ergab Ergebnis{i}.", f"oldp:{i}")
        for i in range(12)
    ]  # fmt: skip
    legalquad += [
        _qa_row("legalquad:sibling", "Was ergab Thema0?", ["über Thema0"], legalquad[0]["context"], "oldp:0"),
        _qa_row("legalquad:empty", "Keine Antwort?", ["  "], "Leer.", "oldp:empty"),
    ]
    gerlayqa = [
        _qa_row("gerlayqa:a", "Wann darf er kündigen?", ["Nach § 573 BGB kann der Vermieter kündigen."],
                gold=["BGB §573"]),
        _qa_row("gerlayqa:b", "Welche Frist?", ["Die Frist beträgt 3 Monate."], gold=["BGB §573c", "BGB §573"]),
        _qa_row("gerlayqa:c", "Kaution?", ["Eine Kaution ist nicht zulässig."], gold=["BGB §551"]),
        _qa_row("gerlayqa:d", "Form?", ["Keine Form ist nötig."], gold=["BGB §125"]),
        _qa_row("gerlayqa:e", "Miete?", ["Die Miete ist fällig."], gold=["BGB §556b"]),
    ]  # fmt: skip
    qa_dirs = [
        _set_dir(tmp_path, "legalquad", items=legalquad),
        _set_dir(tmp_path, "gerlayqa", items=gerlayqa),
    ]
    queries = [{"id": f"q{i}", "text": f"Worum geht es bei Sache{i}?"} for i in range(8)]
    corpus = [{"id": f"p{i}", "text": f"Sache{i} ist geregelt.", "title": ""} for i in range(8)]
    qrels = [{"query_id": f"q{i}", "passage_id": f"p{i}", "relevance": 1} for i in range(8)]
    # the corpus shares one term, so BM25 has a negative for every query in either split
    corpus = [{**p, "text": p["text"] + " Worum es geht."} for p in corpus]
    ret = _set_dir(tmp_path, "gerlerb", queries=queries, corpus=corpus, qrels=qrels)
    nli = tmp_path / "nli.jsonl"
    pairs = [{"id": f"nli-{i}", "kind": "k", "premise": f"P{i}.", "hypothesis": f"H{i}.", "label": i % 2,
              "source": f"bgb{i // 2}"} for i in range(4)]  # fmt: skip
    pairs[0]["question"] = "Welche Form?"
    _write_jsonl(nli, [{"info": {"review": "done"}}, *pairs])

    argv = ["--retrieval-dir", str(ret), "--nli-pairs", str(nli)]
    argv += [a for d in qa_dirs for a in ("--qa-dir", str(d))]
    assert b.main([*argv, "--out", str(tmp_path / "a.jsonl")]) == 0
    assert b.main([*argv, "--out", str(tmp_path / "b.jsonl")]) == 0
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    rows = [json.loads(line) for line in (tmp_path / "a.jsonl").read_text().splitlines()]
    items = {i["id"]: i for i in rows}
    assert len(items) == len(rows)
    assert {i["type"] for i in rows} == {"correctness", "answer_relevancy", "context_precision", "context_recall",
                                         "context_relevance", "faithfulness"}  # fmt: skip
    assert all(i["id"].split(":")[0] in ("legalquad", "gerlayqa", "gerlerb", "nli") for i in rows)
    assert not any(i["id"].startswith("legalquad:empty") for i in rows)
    assert all(a.strip() for i in rows for a in i["references"])

    splits = {}
    for i in rows:
        splits.setdefault(i["group"], set()).add(i["split"])
    assert all(len(s) == 1 for s in splits.values())
    assert {s for v in splits.values() for s in v} == {"calibration", "validation"}
    assert (
        items["legalquad:sibling:correctness:gold"]["group"]
        == items["legalquad:q0:correctness:gold"]["group"]
    )
    assert items["legalquad:q0:correctness:gold"]["group"] == "legalquad:oldp:0"
    # by the sorted first provision: both rest on BGB §573, whatever order the set lists them in
    assert items["gerlayqa:a:correctness:gold"]["group"] == items["gerlayqa:b:correctness:gold"]["group"]
    assert {i["split"] for i in b.build(qa_dirs, [], None, 0.0)} == {"calibration"}
    assert {i["split"] for i in b.build(qa_dirs, [], None, 1.0)} == {"validation"}

    def texts_by_split(field):
        out = {"calibration": set(), "validation": set()}
        for i in rows:
            out[i["split"]].update(i[field] if isinstance(i[field], list) else [i[field]])
        return out

    assert not set.intersection(*texts_by_split("contexts").values())
    gold_split = {i["answer"]: i["split"] for i in rows if i["variant"] == "gold"}
    others = [i for i in rows if i["variant"] == "other_answer"]
    assert others and all(gold_split[i["answer"]] == i["split"] for i in others)
    corr = [i for i in rows if i["type"] == "correctness"]
    assert {i["variant"] for i in corr} == {"gold", "other_answer", "section", "number", "negation"}
    assert all((i["label"] == 1) == (i["answer"] in i["references"]) for i in corr)
    assert (
        items["gerlayqa:a:correctness:negation"]["answer"]
        == "Nach § 573 BGB kann nicht der Vermieter kündigen."
    )
    assert (
        items["nli:nli-1:faithfulness:k"]["label"] == 1
        and items["nli:nli-1:faithfulness:k"]["group"] == "nli:bgb0"
    )
    assert items["nli:nli-0:faithfulness:k"]["question"] == "Welche Form?"
    assert items["nli:nli-1:faithfulness:k"]["question"] == ""

    _write_jsonl(nli, [{"info": {"review": "pending"}}, *pairs])
    with pytest.raises(SystemExit, match="review"):
        b.main([*argv, "--out", str(tmp_path / "c.jsonl")])


def test_calibration_ids_do_not_collide_across_sets(tmp_path, script):
    b = script("build_judge_calibration")
    rows = [_qa_row("1", "Frage?", ["Antwort eins."]), _qa_row("2", "Noch eine?", ["Antwort zwei."])]
    first, second = _set_dir(tmp_path, "eins", items=rows), _set_dir(tmp_path, "zwei", items=rows)
    ids = [i["id"] for i in b.build([first, second], [], None, 0.5)]
    assert len(ids) == len(set(ids))
    assert "eins:1:correctness:gold" in ids and "zwei:1:correctness:gold" in ids
    with pytest.raises(SystemExit, match="duplicate"):
        b.build([first, first], [], None, 0.5)


def test_calibration_negatives_come_from_bm25(tmp_path, script):
    """Precision and relevance negatives are the top-ranked passage that is not relevant, not a random
    one; a context holding the gold answer is never a negative; recall negatives need not rank."""
    b = script("build_judge_calibration")
    question = "Wann verjährt der Anspruch auf Kaufpreiszahlung?"
    gold_context = "Der Anspruch auf Kaufpreiszahlung verjährt in drei Jahren."
    near = "Der Anspruch auf Kaufpreiszahlung wurde wirksam abgetreten."
    holds_answer = (
        "Die Kaufpreiszahlung und der Anspruch verjähren stets in drei Jahren, der Anspruch verjährt."
    )
    far = "Das Grundstück liegt in Köln."
    rows = [
        _qa_row("q", question, ["in drei Jahren"], gold_context, "oldp:gold"),
        _qa_row("near", "Abtretung?", ["wirksam"], near, "oldp:near"),
        _qa_row("holds", "Dauer?", ["stets"], holds_answer, "oldp:holds"),
        _qa_row("far", "Wo?", ["in Köln"], far, "oldp:far"),
    ]
    items = {i["id"]: i for i in b.build([_set_dir(tmp_path, "legalquad", items=rows)], [], None, 1.0)}
    assert items["legalquad:q:context_precision:gold_context"]["contexts"] == [gold_context]
    assert items["legalquad:q:context_precision:gold_context"]["references"] == ["in drei Jahren"]
    assert items["legalquad:q:context_precision:bm25_negative"]["contexts"] == [near]
    assert items["legalquad:q:context_precision:bm25_negative"]["references"] == ["in drei Jahren"]
    assert items["legalquad:q:context_recall:gold_context"]["label"] == 1
    assert items["legalquad:q:context_recall:unrelated"]["label"] == 0
    assert items["legalquad:q:context_recall:unrelated"]["contexts"][0] in (near, far)

    queries = [{"id": "q1", "text": "Kündigung des Mietvertrags"}]
    corpus = [
        {"id": "p1", "text": "Die Kündigung des Mietvertrags bedarf der Schriftform.", "title": ""},
        {"id": "p2", "text": "Die Kündigung des Mietvertrags durch den Erben.", "title": "Erbrecht"},
        {"id": "p3", "text": "Das Grundstück liegt in Köln.", "title": ""},
    ]
    qrels = [{"query_id": "q1", "passage_id": "p1", "relevance": 1}]
    ret = _set_dir(tmp_path, "gerlerb", queries=queries, corpus=corpus, qrels=qrels)
    items = {i["id"]: i for i in b.build([], [ret], None, 1.0)}
    assert items["gerlerb:q1:context_relevance:relevant"]["contexts"] == [corpus[0]["text"]]
    assert items["gerlerb:q1:context_relevance:bm25_negative"]["contexts"] == [
        f"Erbrecht\n{corpus[1]['text']}"
    ]
    assert items["gerlerb:q1:context_relevance:bm25_negative"]["group"] == "gerlerb:p1"
    assert items["gerlerb:q1:context_relevance:relevant"]["references"] == []


def test_agreement_on_a_hand_example(script):
    """Cohen's kappa on the 2x2 table (20, 5 / 10, 15): p_o = 0.7, p_e = 0.5, kappa = 0.4."""
    agreement = script("score_judge_calibration").agreement
    labels = [1] * 25 + [0] * 25
    preds = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    m = agreement(labels, preds)
    assert m["n"] == 50 and m["accuracy"] == pytest.approx(0.7) and m["kappa"] == pytest.approx(0.4)
    assert m["precision"] == pytest.approx(20 / 30) and m["recall"] == pytest.approx(0.8)
    assert agreement([1, 1], [1, 1])["kappa"] is None


def test_score_judge_calibration_cli(tmp_path, monkeypatch, script):
    """Each type goes through the prompt its metric uses in production, on the chosen split, predicted by
    its threshold; an unparsed reply counts as wrong and is asked again next time; the exit code says
    whether every type meets the bar."""

    def item(i, kind, label, answer="", contexts=("Text",), refs=("Ref",), split="validation"):
        return {"id": f"x{i}", "type": kind, "variant": "v", "group": f"g{i}", "split": split, "question": "Q?",
                "answer": answer, "references": list(refs), "contexts": list(contexts), "label": label}  # fmt: skip

    items = [
        *(item(i, "correctness", y, a) for i, (y, a) in enumerate([(1, "Antwort A"), (0, "Antwort B"),
                                                                    (1, "Antwort A"), (0, "Antwort A")])),
        item(4, "context_relevance", 1, contexts=["Kontext R"], refs=()),
        item(5, "context_relevance", 0, contexts=["Kontext R2"], refs=()),
        item(6, "context_relevance", 0, contexts=["Kontext R3"], refs=(), split="calibration"),
        item(7, "faithfulness", 1),
        item(8, "context_precision", 1, contexts=["Kontext Y"]),
        item(9, "context_precision", 0, contexts=["Kontext N"]),
        item(10, "answer_relevancy", 1, "Antwort A"),
        item(11, "answer_relevancy", 0, "Antwort B"),
        item(12, "context_recall", 1, contexts=["Kontext C"]),
        item(13, "context_recall", 0, contexts=["Kontext D"]),
    ]  # fmt: skip
    _write_jsonl(tmp_path / "items.jsonl", items)

    def reply(p):
        if p.startswith("You check whether"):
            return "kaputt"
        if p.startswith("You rate"):
            return '{"relevancy": 0.5}' if "Antwort A" in p else '{"relevancy": 0.4}'
        if p.startswith("You judge which") and "Kontext N" in p:
            return '{"useful": [false]}'
        if p.startswith("You check how much"):  # half the reference reaches the threshold
            return ('{"statements": [{"statement": "a", "attributed": true}, '
                    '{"statement": "b", "attributed": false}]}')  # fmt: skip
        return _reply(p)

    calls = fake_post(monkeypatch, reply)
    argv = ["--items", str(tmp_path / "items.jsonl"), "--out", str(tmp_path / "report.json"), "--base-url", "http://h/v1",
            "--model", "m", "--cache-dir", str(tmp_path / "cache"), "--min-kappa", "0.5", "--min-accuracy", "0.75"]  # fmt: skip
    assert script("score_judge_calibration").main(argv) == 1
    report = json.loads((tmp_path / "report.json").read_text())
    assert set(report["types"]) == {"correctness", "answer_relevancy", "context_precision", "context_recall",
                                    "context_relevance", "faithfulness"}  # fmt: skip
    c, x, f = (report["types"][k] for k in ("correctness", "context_relevance", "faithfulness"))
    assert (c["accuracy"], c["kappa"], c["precision"], c["recall"]) == (0.75, 0.5, 2 / 3, 1.0) and c["meets"]
    assert (x["n"], x["accuracy"], x["kappa"], x["meets"]) == (2, 0.5, 0.0, False)
    assert (f["unparsed"], f["accuracy"]) == (1, 0.0) and not report["meets"]
    assert report["types"]["answer_relevancy"]["accuracy"] == 1.0  # 0.5 reaches the threshold, 0.4 does not
    assert report["types"]["context_precision"]["accuracy"] == 1.0
    assert report["types"]["context_recall"]["recall"] == 1.0  # one of two statements is enough

    prompts = [c["json"]["messages"][0]["content"] for c in calls]
    relevance = next(p for p in prompts if "Kontext R" in p)
    assert "Reference answers" not in relevance and "answering the question" in relevance
    assert not any("Kontext R3" in p for p in prompts)  # the calibration split is not asked
    assert "Reference answers:\n- Ref" in next(p for p in prompts if "Kontext Y" in p)
    assert "Reference answer:\nRef" in next(p for p in prompts if "Kontext C" in p)

    calls.clear()
    loose = [a if a not in ("0.5", "0.75") else "0" for a in argv]
    assert script("score_judge_calibration").main(loose) == 0
    asked = [c["json"]["messages"][0]["content"] for c in calls]
    assert len(asked) == 2 and all(p.startswith("You check whether") for p in asked)  # only the unparsed


def test_a_judge_scores_only_after_its_own_validation_passed(tmp_path, script):
    """No report, a failed bar, the calibration split, another model or endpoint, older prompts or an
    unvalidated metric: scoring is refused, naming the reason."""
    from lexhybrid.eval.judge import METRICS, PROMPT_VERSION

    ja = script("judge_answers")
    report = tmp_path / "r.json"
    report.write_text(json.dumps(_validated_report()))
    ja.check_validated(report, "m", "http://h/v1")
    partial = {m: {"meets": True} for m in METRICS[:-1]}
    for changes, reason in [
        ({"meets": False}, "bar was not met"),
        ({"split": "calibration"}, "not validation"),
        ({"judge": {"model": "other", "base_url": "http://h/v1"}}, "model 'other'"),
        ({"judge": {"model": "m", "base_url": "http://x/v1"}}, "endpoint"),
        ({"prompt_version": PROMPT_VERSION - 1}, "prompt version"),
        ({"types": partial}, f"no validation of .*{METRICS[-1]}"),
    ]:
        report.write_text(json.dumps(_validated_report(**changes)))
        with pytest.raises(SystemExit, match=reason):
            ja.check_validated(report, "m", "http://h/v1")
    argv = ["--records", str(tmp_path / "none.jsonl"), "--out", str(tmp_path / "o.jsonl"), "--base-url", "http://h/v1", "--model", "m"]  # fmt: skip
    (tmp_path / "none.jsonl").write_text("")
    with pytest.raises(SystemExit):
        ja.main(argv)


def test_the_bar_needs_every_metric_validated(script, tmp_path):
    """Items of one type alone never meet the bar: the metrics without items are named."""
    from lexhybrid.eval.judge import METRICS, Judge

    item = {"id": "a", "type": "correctness", "label": 1, "variant": "gold", "question": "Q", "answer": "A",
            "references": ["A"], "contexts": []}  # fmt: skip
    judge = Judge(lambda prompt: '{"verdict": "correct", "reason": "r"}', "m", tmp_path)
    report = script("score_judge_calibration").score(
        [item, {**item, "id": "b", "label": 0, "variant": "x"}], judge, 0.0, 0.0
    )
    assert not report["meets"] and report["missing_types"] == sorted(set(METRICS) - {"correctness"})


def test_a_reasoning_draft_and_a_malformed_success_are_not_verdicts(monkeypatch):
    """A draft verdict inside a reasoning block is not the reply; a 200 without choices counts as an
    unparsed reply instead of ending a long run."""
    from lexhybrid.eval.judge import OpenAICompatible, first_json_object

    draft = '<think>{"verdict": "correct"}</think>\n{"verdict": "incorrect"}'
    assert first_json_object(draft) == {"verdict": "incorrect"}
    assert first_json_object('<think>{"verdict": "correct"} und dann abgeschnitten') is None
    response = SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: {"error": "busy"})
    monkeypatch.setattr(requests, "post", lambda *a, **k: response)
    assert OpenAICompatible("http://h/v1", "m")("p") == ""
