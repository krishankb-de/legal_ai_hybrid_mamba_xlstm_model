"""Synthetic QA (scripts/generate_synthetic_qa.py) and the vLLM jobs, offline.

The generator runs on the retrieval fixture corpus (the Swiss decision marked research-only, the ZGB
not commercial-safe) with a stub teacher and a stub NLI; the wrappers run in a repository copy
against a fake ``vllm serve`` that answers /health and completions, and a fake ``.venv`` python.
"""

import json
import os
import random
import re
import shutil
import socket
import stat
import subprocess
import sys
import threading
import zlib
from collections import Counter
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from lexhybrid.data.sft_format import UNANSWERABLE, Cite, Quote, Text, parse, render
from lexhybrid.data.tokenizer import EOS_TOKEN
from lexhybrid.retrieval.bm25 import BM25Index
from lexhybrid.retrieval.chunking import chunk_document
from lexhybrid.retrieval.index import CHUNKS_NAME
from lexhybrid.verifier.verifier import asks_for_party
from tests.conftest import REPO_ROOT, load_script
from tests.test_retrieval import _bag_of_words, _fixture_corpus, _planted

gen = load_script("generate_synthetic_qa")
NAMED = re.compile(r"nach (.+?)(?: keine Form nötig)?\?$")  # the source citation in a stub question
QUOTE = [{"text": "Es gilt: "}, {"quote": [1, 1]}, {"text": "."}]  # passes the stub NLI
UNENTAILED = [{"text": "Das regelt die Passage."}, {"cite": 1}]  # fails the stub NLI


@pytest.fixture
def index(tmp_path):
    """``(index directory, {chunk id: chunk})``: chunks.jsonl and bm25/ as build_index.py writes them."""
    chunks = [
        replace(c, research_only=True) if c.source == "bger"
        else replace(c, commercial_safe=False) if c.citation_id.startswith("ZGB") else c
        for d in _fixture_corpus()
        for c in chunk_document(d)
    ]  # fmt: skip
    out = tmp_path / "index"
    out.mkdir()
    (out / CHUNKS_NAME).write_text("".join(c.to_json() + "\n" for c in chunks), encoding="utf-8")
    BM25Index.build(chunks).save(out / "bm25")
    return out, {c.id: c for c in chunks}


def teacher(prompt: str, schema: dict) -> str:
    """Pure, so threads may call it in any order. A question names its source's citation (a party
    question is sometimes not one the rule recognises, and is never asked for an answer: its citation
    lookup would raise). Every other reply follows a hash of the prompt, whose passages differ from
    attempt to attempt. With the source among the passages: malformed, a quote of a sentence that does
    not exist, a citation the stub NLI does not entail, a declared refusal, or a verbatim quote of the
    source. Without it: a declared refusal, or a verbatim quote of a distractor."""
    if schema is gen.QUESTION_SCHEMA:
        citation = prompt.split("Abschnitt:\n", 1)[1].split("\n", 1)[0]
        if gen._ASK["party"] in prompt:
            question = "Worum ging es?" if zlib.crc32(citation.encode()) % 4 == 0 else "Wer war der Kläger?"
        elif gen._ASK["false_premise"] in prompt:
            question = f"Warum ist nach {citation} keine Form nötig?"
        else:
            question = f"Was gilt nach {citation}?"
        return json.dumps({"question": question})
    pick = zlib.crc32(prompt.encode())
    if schema is gen.PREMISE_SCHEMA:
        return json.dumps({"contradicts": pick % 4 != 0})
    question = re.search(r"^Frage: (.*)$", prompt, re.M).group(1)
    citation = NAMED.search(question).group(1)
    source = [int(n) for n, c in re.findall(r"^Passage (\d+): (.*)$", prompt, re.M) if c == citation]
    if not source:
        return json.dumps({"answer": QUOTE if pick % 3 == 0 else []})
    k = source[0]
    return [
        "kein JSON",
        json.dumps({"answer": [{"text": "Es gilt: "}, {"quote": [k, 99]}]}),
        json.dumps({"answer": [{"text": "Das regelt die Passage."}, {"cite": k}]}),
        json.dumps({"answer": []}),
        json.dumps({"answer": [{"text": "Es gilt: "}, {"quote": [k, 1]}, {"text": "."}]}),
        json.dumps({"answer": [{"text": "Es gilt: "}, {"quote": [k, 1]}, {"text": "."}]}),
    ][pick % 6]


def nli(premise: str, hypothesis: str) -> float:
    return 1.0 if premise in hypothesis else 0.0  # a verbatim quote is entailed, a bare citation is not


def _rows(out):
    return [json.loads(line) for line in (out / gen.ROWS_NAME).read_text(encoding="utf-8").splitlines()]


def test_rows_are_verified_sft_examples_with_a_quarter_refusals(index, tmp_path, monkeypatch, capsys):
    """Through the CLI: 40 rows that round-trip through sft_format.parse; answerable ones quote their
    source, refusals end on <|unanswerable|> (the unanswerable question's document is not among its
    passages, the false premise's source is); no research-only or non-commercial chunk is ever shown
    to the teacher or put in a prompt; the stats add up and name the rejections."""
    index_dir, chunks = index
    prompts = []
    monkeypatch.setattr(
        gen, "http_teacher", lambda endpoint, model, seed: lambda p, s: prompts.append(p) or teacher(p, s)
    )
    monkeypatch.setattr(gen, "MDebertaNLI", lambda device=None: nli)
    out = tmp_path / "out"
    argv = [
        "--index",
        str(index_dir),
        "--count",
        "40",
        "--seed",
        "3",
        "--out",
        str(out),
        "--concurrency",
        "4",
    ]
    assert gen.main(argv) == 0
    assert "SYNTH kept 40/40" in capsys.readouterr().out
    rows, stats = _rows(out), json.loads((out / gen.STATS_NAME).read_text())
    kinds = Counter(r["kind"] for r in rows)
    assert len(rows) == stats["kept"] == stats["requested"] == 40 and kinds["answerable"] == 30
    assert stats["kept_by_kind"] == {k: kinds[k] for k in gen.KINDS} and all(kinds[k] for k in gen.KINDS)
    assert 0.2 <= stats["refusal_share"] == 10 / 40 <= 0.3
    assert (stats["seed"], stats["teacher_model"]) == (3, "Qwen/Qwen3-32B")
    assert (
        stats["attempts"]
        == stats["verified"] + stats["rejected"]
        == len((out / gen.ATTEMPTS_NAME).read_text().splitlines())
    )
    assert stats["verified"] == 40 and stats["dropped_over_count"] == stats["dropped_for_share"] == 0
    assert stats["rejected"] == sum(stats["rejected_by_reason"].values())
    assert {"malformed reply", "invalid example: quote of sentence N of passage N: no such sentence",
            "party question not recognised", "teacher: declared unanswerable",
            "refusal: the teacher's answer verifies",
            } <= set(stats["rejected_by_reason"])  # fmt: skip
    assert any(r.startswith("verifier: main claim failed: nli") for r in stats["rejected_by_reason"])
    excluded = {i for i, c in chunks.items() if c.research_only or not c.commercial_safe}
    assert excluded and not any(chunks[i].sentences[0] in p for i in excluded for p in prompts)
    assert len({r["id"] for r in rows}) == 40
    for row in rows:
        assert (
            set(row) == {"id", "question_id", "source", "kind", "chunk_ids", "text"}
            and row["source"] == "synthetic"
        )
        ex = parse(row["text"])
        assert "".join(render(ex)) == row["text"]
        passages = [chunks[i] for i in row["chunk_ids"]]
        assert ex.passages == tuple(c.passage() for c in passages) and 1 <= len(passages) <= 8
        assert not excluded & set(row["chunk_ids"])
        assert (
            row["text"].endswith(UNANSWERABLE + EOS_TOKEN) == ex.unanswerable == (row["kind"] != "answerable")
        )
        if row["kind"] == "party":
            assert asks_for_party(ex.question)
            continue
        named = NAMED.search(ex.question).group(1)
        if row["kind"] == "answerable":
            (quote,) = [s for s in ex.answer if isinstance(s, Quote)]
            assert passages[quote.passage - 1].citation_id == named and quote.sentence == 1
        elif row["kind"] == "false_premise":
            assert named in [c.citation_id for c in passages]
        else:
            source_docs = {c.doc_id for c in chunks.values() if c.citation_id == named}
            assert not source_docs & {c.doc_id for c in passages}
    with pytest.raises(SystemExit):
        gen.main([*argv[:6], "--out", str(tmp_path / "x"), "--endpoint", "https://api.example.com/v1"])


def test_a_rerun_resumes_from_the_attempts_and_the_seed_fixes_the_rows(index, tmp_path):
    """Quotas met: no teacher call, same rows. A log cut mid-line by a kill loses only that line and
    the run completes after the last attempt. Another directory, same seed: the same rows; a log of
    another seed is refused."""
    index_dir, _ = index
    calls = []

    def counting(prompt, schema):
        calls.append(prompt)
        return teacher(prompt, schema)

    a, b = tmp_path / "a", tmp_path / "b"
    gen.generate(index_dir, a, counting, nli, 16, seed=1, concurrency=4)
    first, n = (a / gen.ROWS_NAME).read_text(), len(calls)
    gen.generate(index_dir, a, counting, nli, 16, seed=1, concurrency=4)
    assert len(calls) == n and (a / gen.ROWS_NAME).read_text() == first
    log = a / gen.ATTEMPTS_NAME
    lines = log.read_text().splitlines(keepends=True)
    log.write_text("".join(lines[:5]) + lines[5][:20])
    stats = gen.generate(index_dir, a, counting, nli, 16, seed=1, concurrency=4)
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["attempt"] for r in records] == list(range(len(records))) and len(calls) > n
    assert stats["kept"] == len(_rows(a)) == 16 and records[:5] == [json.loads(line) for line in lines[:5]]
    gen.generate(index_dir, b, teacher, nli, 16, seed=1, concurrency=2)
    assert (b / gen.ROWS_NAME).read_text() == first
    with pytest.raises(SystemExit, match="'seed': 1"):
        gen.generate(index_dir, a, teacher, nli, 16, seed=2)


VERIFIES = "refusal: the teacher's answer verifies"


@pytest.mark.parametrize(
    "kind, answer, contradicts, reason",
    [
        ("unanswerable", [], None, None),  # the teacher declares it unanswerable
        ("unanswerable", UNENTAILED, None, None),  # the verifier refuses the answer
        ("unanswerable", [{"quote": [1, 99]}], None, None),  # a sentence that does not exist
        ("unanswerable", QUOTE, None, VERIFIES),  # a distractor answers it
        ("unanswerable", [*QUOTE, {"text": " Das ist falsch. "}, {"cite": 1}], None, VERIFIES),  # main claim
        ("unanswerable", "kein JSON", None, "malformed reply"),
        ("false_premise", [], True, None),
        ("false_premise", UNENTAILED, True, None),
        ("false_premise", QUOTE, True, VERIFIES),
        ("false_premise", [], False, "false premise not confirmed"),
        ("false_premise", [], "ja", "malformed reply"),
        ("party", None, None, None),  # never asked: the verifier refuses it whatever the answer
        ("answerable", QUOTE, None, None),
        ("answerable", [], None, "teacher: declared unanswerable"),
        ("answerable", UNENTAILED, None, "verifier: main claim failed: nli"),
        ("answerable", [*QUOTE, {"text": " Das ist falsch. "}, {"cite": 1}], None, "verifier: a later claim"),
    ],
)  # fmt: skip
def test_what_keeps_an_item(kind, answer, contradicts, reason):
    """The teacher answers over the prompt's passages in the normal format. A refusal survives only an
    answer that is no answer, and a false premise also needs a yes; an answerable item is the reverse."""
    chunk = chunk_document(_planted())[0]

    def scripted(prompt, schema):
        assert kind != "party", "a party question is never asked for an answer"
        if schema is gen.PREMISE_SCHEMA:
            assert chunk.sentences[0] in prompt and kind == "false_premise"
            return json.dumps({"contradicts": contradicts})
        return answer if isinstance(answer, str) else json.dumps({"answer": answer})

    question = "Wer war der Kläger?" if kind == "party" else "Welche Form braucht die Kündigung?"
    a = gen.Attempt(0, kind, chunk, random.Random(0), question=question, passages=[chunk])
    a.ask_answer(scripted)
    record = gen.finish(a, nli, {"seed": 0})
    if reason is not None:
        assert record["row"] is None and record["reason"].startswith(reason)
        return
    assert record["reason"] is None
    ex = parse(record["row"]["text"])
    assert ex.question == question and ex.unanswerable == (kind != "answerable")
    assert (
        render(ex)[1] == UNANSWERABLE + EOS_TOKEN
        if kind != "answerable"
        else any(isinstance(s, Quote) for s in ex.answer)
    )


def test_an_unreachable_teacher_stops_the_run_and_a_small_failed_round_does_not(index, tmp_path, monkeypatch):
    """Rounds smaller than the limit add up: 32 errors in a row stop the run, whatever the round size,
    and the attempts are logged. Four failed calls that are one whole round do not."""
    index_dir, _ = index

    def down(prompt, schema):
        raise gen.TeacherError("down")

    with pytest.raises(SystemExit, match="last 32 attempts got no teacher reply"):
        gen.generate(index_dir, tmp_path / "down", down, nli, 8, seed=0)
    assert (
        len((tmp_path / "down" / gen.ATTEMPTS_NAME).read_text().splitlines()) == gen.MAX_TEACHER_ERRORS == 32
    )

    monkeypatch.setattr(gen, "MAX_ATTEMPTS", 40)
    calls = []

    def flaky(prompt, schema):
        calls.append(prompt)
        if len(calls) <= 4:
            raise gen.TeacherError("still starting")
        return teacher(prompt, schema)

    stats = gen.generate(index_dir, tmp_path / "flaky", flaky, nli, 4, seed=0, concurrency=1)
    assert stats["kept"] == 4 and stats["rejected_by_reason"]["teacher error"] == 4


def test_a_low_yield_is_fitted_to_the_share_and_the_run_keeps_its_rows(index, tmp_path, monkeypatch):
    """Attempts run out with too few answerable rows: a seeded sample of refusals goes, the rows are
    written and the stats say so. The other way round the sample comes from the answerable rows."""
    index_dir, _ = index
    monkeypatch.setattr(gen, "MAX_ATTEMPTS", 1)
    stats = gen.generate(index_dir, tmp_path / "low", teacher, nli, 40, seed=0, concurrency=4)
    rows = _rows(tmp_path / "low")
    kinds = Counter(r["kind"] for r in rows)
    assert stats["attempts"] == 40 and stats["dropped_for_share"] > 0 and stats["kept"] == len(rows)
    assert stats["verified"] == len(rows) + stats["dropped_for_share"]
    assert kinds["answerable"] == sum(
        r["row"]["kind"] == "answerable"
        for r in map(json.loads, (tmp_path / "low" / gen.ATTEMPTS_NAME).read_text().splitlines())
        if r["row"]
    ), "only refusals are dropped"
    assert 0.2 <= stats["refusal_share"] <= 0.3
    gen.generate(index_dir, tmp_path / "again", teacher, nli, 40, seed=0, concurrency=2)
    assert _rows(tmp_path / "again") == rows, "the sample is seeded"
    monkeypatch.undo()

    def answers_everything(prompt, schema):
        if schema is gen.QUESTION_SCHEMA:
            return teacher(prompt, schema)
        if schema is gen.PREMISE_SCHEMA:
            return json.dumps({"contradicts": True})
        return json.dumps({"answer": QUOTE})

    stats = gen.generate(index_dir, tmp_path / "high", answers_everything, nli, 40, seed=0, concurrency=4)
    kinds = Counter(r["kind"] for r in _rows(tmp_path / "high"))
    assert (
        stats["attempts"] == 160 and stats["refusal_share"] == 0.2 and set(kinds) == {"answerable", "party"}
    )
    assert kinds == {"answerable": 12, "party": 3} and stats["dropped_for_share"] == 18
    assert stats["rejected_by_reason"][VERIFIES] == stats["rejected"] - stats["rejected_by_reason"].get(
        "party question not recognised", 0
    )


def test_a_rerun_with_a_smaller_count_writes_no_more_and_too_small_counts_are_refused(index, tmp_path):
    """Rows are capped kind by kind at the new count's quotas, with no teacher call. A count no integer
    number of refusals can fit is refused before anything runs. No rows at all fail the share."""
    index_dir, _ = index
    out = tmp_path / "out"
    gen.generate(index_dir, out, teacher, nli, 40, seed=0, concurrency=4)

    def not_called(prompt, schema):
        raise AssertionError("the attempts already hold enough rows")

    stats = gen.generate(index_dir, out, not_called, nli, 16, seed=0)
    rows = _rows(out)
    assert len(rows) == stats["kept"] == 16 and stats["dropped_over_count"] == 24 and stats["verified"] == 40
    assert Counter(r["kind"] for r in rows) == {
        "answerable": 12, "unanswerable": 2, "false_premise": 1, "party": 1,
    }  # fmt: skip
    for count in (0, 1, 2, 3, 6):
        small = tmp_path / f"count{count}"
        with pytest.raises(SystemExit, match=f"--count {count} cannot hold"):
            gen.generate(index_dir, small, not_called, nli, count, seed=0)
        assert not small.exists()
    with pytest.raises(SystemExit, match="refusal share 0.000 outside"):
        gen.generate(index_dir, tmp_path / "none", lambda p, s: "kein JSON", nli, 8, seed=0)
    assert (tmp_path / "none" / gen.STATS_NAME).exists() and not (tmp_path / "none" / gen.ROWS_NAME).exists()


def test_the_dense_index_joins_retrieval_when_present(index, tmp_path):
    from lexhybrid.retrieval.dense import DenseIndex

    index_dir, chunks = index
    DenseIndex.build(index_dir / "dense", list(chunks), [c.text for c in chunks.values()], _bag_of_words)
    batches = []
    encode = lambda texts: batches.append(list(texts)) or _bag_of_words(texts)  # noqa: E731
    stats = gen.generate(index_dir, tmp_path / "out", teacher, nli, 8, seed=0, encode=encode, concurrency=4)
    assert stats["kept"] == 8 and batches and all(q.endswith("?") for batch in batches for q in batch)
    shown = {i for row in _rows(tmp_path / "out") for i in row["chunk_ids"]}
    assert not any(chunks[i].research_only or not chunks[i].commercial_safe for i in shown)


def test_http_teacher_retries_and_the_reply_parsers_are_strict(monkeypatch):
    """Retries with backoff on a refused connection and a 503, then the content of the reply; the
    request holds the reply to the schema and turns Qwen3's thinking off; three timeouts are a
    TeacherError. Parsed answers join adjacent text; anything off-schema is a ValueError."""
    import requests

    posts, sleeps = [], []

    class Reply:
        def __init__(self, status, body=None):
            self.status, self.body = status, body

        def raise_for_status(self):
            if self.status >= 400:
                raise requests.HTTPError(str(self.status))

        def json(self):
            return self.body

    replies = [requests.ConnectionError("refused"), Reply(503),
               Reply(200, {"choices": [{"message": {"content": '{"question": "Q?"}'}}]})]  # fmt: skip

    def post(url, json, timeout):
        posts.append((url, json, timeout))
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(gen.time, "sleep", sleeps.append)
    ask = gen.http_teacher("http://127.0.0.1:8000/v1/", "Qwen/Qwen3-32B", timeout=5)
    assert ask("Frage?", gen.QUESTION_SCHEMA) == '{"question": "Q?"}' and len(posts) == 3 and sleeps == [1, 2]
    url, body, timeout = posts[-1]
    assert (url, timeout, body["model"]) == ("http://127.0.0.1:8000/v1/chat/completions", 5, "Qwen/Qwen3-32B")
    assert body["messages"] == [{"role": "user", "content": "Frage?"}]
    assert body["response_format"]["json_schema"]["schema"] == gen.QUESTION_SCHEMA
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    replies[:] = [requests.Timeout("slow")] * 3
    with pytest.raises(gen.TeacherError, match="3 tries: Timeout"):
        ask("Frage?", gen.ANSWER_SCHEMA)

    reply = json.dumps(
        {"answer": [{"text": "A"}, {"text": ""}, {"text": "B "}, {"cite": 2}, {"quote": [1, 3]}]}
    )
    assert gen.parse_answer(reply) == (Text("AB "), Cite(2), Quote(1, 3))
    assert gen.parse_answer('{"answer": []}') == (), "an empty list declares it unanswerable"
    assert gen.parse_question('{"question": " Wann\\n endet die Frist? "}') == "Wann endet die Frist?"
    for bad in (
        '{"answer": [{"text": ""}]}',
        '{"answer": [{"text": "x", "cite": 1}]}',
        '{"answer": [{"quote": [1]}]}',
    ):
        with pytest.raises(ValueError):
            gen.parse_answer(bad)
    with pytest.raises(ValueError):
        gen.parse_question('{"question": " "}')
    assert gen.parse_contradicts('{"contradicts": false}') is False
    with pytest.raises(ValueError):
        gen.parse_contradicts('{"contradicts": "ja"}')


# -- the jobs, against a fake vLLM server ----------------------------------------------------------

FAKE_VLLM = """#!PYTHON
# Fake `vllm serve`: records its pid, arguments and HF settings, then serves /health and completions
# (FAKE_VLLM=crash exits at once, FAKE_VLLM=hang never listens, FAKE_VLLM=foreign dies at once while a
# child it leaves behind, standing in for another job's server, answers on the port for ten seconds).
import json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer

args = sys.argv[1:]
record = {"pid": os.getpid(), "args": args, **{k: os.environ.get(k) for k in ("HF_HOME", "HF_HUB_OFFLINE")}}
with open(os.environ["FAKE_VLLM_RECORD"], "w") as f:
    json.dump(record, f)
if os.environ.get("FAKE_VLLM") == "crash":
    sys.exit(1)
if os.environ.get("FAKE_VLLM") == "hang":
    time.sleep(600)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        body = json.dumps({"choices": [{"text": " das Schuldrecht."}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


server = HTTPServer(("127.0.0.1", int(args[args.index("--port") + 1])), Handler)
if os.environ.get("FAKE_VLLM") == "foreign":
    if os.fork():
        os._exit(1)
    threading.Timer(10, os._exit, [0]).start()
server.serve_forever()
"""
# the GPU check passes; the generator records its arguments and exits with FAKE_GEN_EXIT
FAKE_MAIN_PYTHON = """#!/usr/bin/env bash
if [[ "$1" == -c ]]; then exit 0; fi
printf '%s\\n' "$@" > "$FAKE_GEN_RECORD"
exit "${FAKE_GEN_EXIT:-0}"
"""


def _executable(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _stopped(pid):
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def _free_port() -> int:
    """A free port of the range the jobs derive theirs from."""
    for port in range(10000 + os.getpid() % 19000, 30000):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free port")


@pytest.fixture
def jobs(tmp_path):
    """``run(script, **env) -> (CompletedProcess, server record | None, generator argv)`` in a repo copy.
    The job id is the one whose derived port (``run.port``) is free here."""
    repo, scratch = tmp_path / "repo", tmp_path / "scratch"
    shutil.copytree(REPO_ROOT / "scripts" / "slurm", repo / "scripts" / "slurm")
    _executable(repo / ".venv-vllm" / "bin" / "vllm", FAKE_VLLM.replace("PYTHON", sys.executable))
    _executable(
        repo / ".venv-vllm" / "bin" / "python", f'#!/usr/bin/env bash\nexec "{sys.executable}" "$@"\n'
    )
    _executable(repo / ".venv" / "bin" / "python", FAKE_MAIN_PYTHON)
    (scratch / "data" / "index" / "v1").mkdir(parents=True)
    (scratch / "data" / "index" / "v1" / "index.json").write_text("{}")
    port = _free_port()
    server, generator = tmp_path / "server.json", tmp_path / "generator.txt"

    def run(script, **env):
        server.unlink(missing_ok=True)
        generator.unlink(missing_ok=True)
        full = {
            "PATH": os.environ["PATH"], "HOME": str(tmp_path / "home"), "USER": "tester",
            "SLURM_SUBMIT_DIR": str(repo), "SCRATCH_ROOT": str(scratch), "SLURM_JOB_ID": str(port - 10000),
            "FAKE_VLLM_RECORD": str(server), "FAKE_GEN_RECORD": str(generator),
            **{k: str(v) for k, v in env.items()},
        }  # fmt: skip
        res = subprocess.run(["bash", str(repo / "scripts" / "slurm" / script)], env=full, cwd=repo,
                             capture_output=True, text=True, timeout=120)  # fmt: skip
        record = json.loads(server.read_text()) if server.exists() else None
        return res, record, generator.read_text().splitlines() if generator.exists() else []

    run.port, run.scratch = port, scratch
    return run


def test_vllm_smoke_serves_one_completion_and_always_stops_the_server(jobs):
    res, server, _ = jobs("vllm_smoke.sh")
    assert res.returncode == 0 and "VLLM SMOKE OK" in res.stdout, res.stdout + res.stderr
    assert server["args"][:2] == ["serve", "Qwen/Qwen3-0.6B-Base"]
    assert server["args"][server["args"].index("--port") + 1] == str(jobs.port), "the port follows the job id"
    assert (server["HF_HOME"], server["HF_HUB_OFFLINE"]) == (str(jobs.scratch / ".hf"), "1")
    _stopped(server["pid"])
    for mode, message in (
        ("crash", "FATAL: vllm exited before /health"),
        ("hang", "FATAL: no /health after 1s"),
    ):
        res, server, _ = jobs("vllm_smoke.sh", FAKE_VLLM=mode, VLLM_TIMEOUT=1)
        assert res.returncode == 1 and message in res.stdout and "VLLM SMOKE OK" not in res.stdout
        _stopped(server["pid"])


def test_another_server_on_the_port_is_never_taken_for_ours(jobs):
    """Already answering (with any status): the job stops before launching. Answering only after our
    server has died: its /health is not trusted, because our process is gone."""
    other = HTTPServer(("127.0.0.1", jobs.port), BaseHTTPRequestHandler)  # answers every GET with a 501
    threading.Thread(target=other.serve_forever, daemon=True).start()
    try:
        res, server, _ = jobs("vllm_smoke.sh")
    finally:
        other.shutdown()
        other.server_close()
    assert res.returncode == 1 and f"FATAL: port {jobs.port} already answers" in res.stdout
    assert server is None and "VLLM SMOKE OK" not in res.stdout
    res, server, _ = jobs("vllm_smoke.sh", FAKE_VLLM="foreign")
    assert res.returncode == 1 and "FATAL: vllm exited before /health" in res.stdout
    assert "VLLM SMOKE OK" not in res.stdout
    _stopped(server["pid"])


def test_synth_qa_runs_the_generator_against_the_server_and_always_stops_it(jobs):
    res, server, argv = jobs("synth_qa.sh", COUNT=100, SEED=2)
    assert res.returncode == 0 and "SYNTH QA DONE" in res.stdout, res.stdout + res.stderr
    assert server["args"][:2] == ["serve", "Qwen/Qwen3-32B"] and server["HF_HUB_OFFLINE"] == "1"
    assert server["args"][server["args"].index("--tensor-parallel-size") + 1] == "2"
    _stopped(server["pid"])
    value = lambda flag: argv[argv.index(flag) + 1]  # noqa: E731
    assert argv[0] == "scripts/generate_synthetic_qa.py" and value("--model") == "Qwen/Qwen3-32B"
    assert value("--endpoint") == f"http://127.0.0.1:{jobs.port}/v1"
    assert (value("--index"), value("--out")) == (
        str(jobs.scratch / "data" / "index" / "v1"),
        str(jobs.scratch / "data" / "synthetic_qa" / "v1_s2"),
    )
    assert (value("--count"), value("--seed")) == ("100", "2")
    res, server, _ = jobs("synth_qa.sh", FAKE_GEN_EXIT=1)
    assert res.returncode == 1 and "SYNTH QA DONE" not in res.stdout
    _stopped(server["pid"])
    res, server, _ = jobs("synth_qa.sh", INDEX_VERSION="v9")
    assert res.returncode == 1 and "FATAL: no index" in res.stdout and server is None


def test_attempts_of_another_teacher_or_index_are_refused(tmp_path):
    """One rows file has one provenance: a rerun with another teacher or index does not merge into it."""
    gen = load_script("generate_synthetic_qa")
    path = tmp_path / "attempts.jsonl"
    identity = {"seed": 1, "teacher": "a", "index": "v1"}
    path.write_text(
        json.dumps({"attempt": 0, **identity, "kind": "party", "reason": None, "row": None}) + "\n"
    )
    assert len(gen.read_attempts(path, identity)) == 1
    for other in ({"teacher": "b"}, {"index": "v2"}, {"seed": 2}):
        with pytest.raises(SystemExit, match="was written with"):
            gen.read_attempts(path, identity | other)
