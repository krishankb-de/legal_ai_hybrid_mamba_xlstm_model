#!/usr/bin/env python3
"""Synthetic retrieval-grounded QA for SFT: an open-weight teacher writes it, the verifier keeps it.

    .venv/bin/python scripts/generate_synthetic_qa.py --index data/index/v1 --endpoint http://127.0.0.1:8000/v1 \\
        --model Qwen/Qwen3-32B --count 20000 --seed 0 --out data/synthetic_qa/v1_s0

The teacher is a local OpenAI-compatible server (vLLM, started in the same job from ``.venv-vllm``);
any other host is refused, so no commercial API can write training data. Only commercial-safe chunks
are used: a chunk with ``research_only`` set or ``commercial_safe`` unset is never shown to the
teacher nor put in a prompt. Each attempt samples a source chunk (statutes and decisions equally
often) and a kind:

* ``answerable``: the teacher writes a German question the chunk answers. The prompt's passages are
  the chunk and the chunks retrieved for the question (BM25, fused by RRF with ``dense/`` when the
  index has one) as distractors, 3 to 8 within the reranker's token budget, shuffled. The teacher
  answers over exactly those passages in JSON (``{"answer": [{"text": ...}, {"quote": [K, J]},
  {"cite": K}]}``, constrained by a JSON schema; ``{"answer": []}`` declares the question
  unanswerable), which becomes ``sft_format`` segments; the item is kept only when ``verify`` (the
  mDeBERTa NLI in production) passes every claim.
* ``unanswerable``: a question of the same kind, with the chunk and every copy of its document left
  out of the passages;
* ``false_premise``: a question presupposing what the chunk contradicts, the chunk in the passages;
* ``party``: a question asking who a party of the decision was (decision chunks only; it must match
  the verifier's party rule, ``asks_for_party``).

The three refusal kinds target ``<|unanswerable|>``. Their labels are checked: the teacher also
answers ``unanswerable`` and ``false_premise`` questions over the prompt's exact passages, and the
item is kept only when it declares them unanswerable or ``verify`` refuses that answer (a distractor
may answer the question, and a decision may quote the statute). A ``false_premise`` also needs the
teacher's yes that the source contradicts the question. ``party`` needs no answer: ``verify`` refuses
a party question whatever the answer says.

Refusals make up ``REFUSAL_SHARE`` of the rows. A ``--count`` no integer number of refusals can fit
to ``SHARE_RANGE`` is refused up front. A low yield does not lose the run: when the attempts run out,
a seeded sample of whichever side is over-represented is dropped so the share fits, and ``stats.json``
says how many; the run fails only when no rows can fit. Rows of a kind never exceed its quota, so a
rerun with a smaller ``--count`` writes no more than it asks for. A malformed teacher reply, an HTTP
failure after the retries, a pointer to a sentence that does not exist and a verifier refusal are
counted rejections, never a crash; ``MAX_TEACHER_ERRORS`` teacher errors in a row stop the run.
``--out`` gets:

* ``synthetic_qa.jsonl``: one row per kept item, ``{"id", "question_id", "source": "synthetic",
  "kind", "chunk_ids", "text"}``; ``chunk_ids`` are the passages in ``<|cK|>`` order, ``text`` is
  ``prompt + target`` of ``sft_format.render`` (``sft_format.parse`` reads it back), ``question_id``
  hashes the question, so a question the teacher wrote twice shares it;
* ``stats.json``: requested, kept (the rows written), attempts, verified (attempts that passed),
  rejected by reason (digits folded to N), dropped over the count and for the share, kept by kind,
  the refusal share, teacher model, seed;
* ``attempts.jsonl``: every attempt as it finished; a rerun (a requeued job) resumes after the last.
"""

import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from lexhybrid.data.sft_format import Cite, Example, Quote, Text, render
from lexhybrid.retrieval.bm25 import BM25Index
from lexhybrid.retrieval.chunking import Chunk, estimate_tokens
from lexhybrid.retrieval.dense import BgeM3Encoder, DenseIndex
from lexhybrid.retrieval.fusion import reciprocal_rank_fusion
from lexhybrid.retrieval.index import CHUNKS_NAME, write_atomic
from lexhybrid.retrieval.rerank import CANDIDATES, MAX_CHUNKS, MIN_CHUNKS, TOKEN_BUDGET, Ranked, select
from lexhybrid.utils.jsonl import read_complete_lines
from lexhybrid.verifier.nli import NLI, MDebertaNLI
from lexhybrid.verifier.verifier import asks_for_party, verify

TEACHER = "Qwen/Qwen3-32B"
KINDS = ("answerable", "unanswerable", "false_premise", "party")
REFUSAL_SHARE, SHARE_RANGE = 0.25, (0.2, 0.3)
MAX_ATTEMPTS = 4  # attempts per requested row before the run stops short
BATCH = 256  # attempts per round; fixed, so the rows do not depend on --concurrency
MAX_TEACHER_ERRORS = 32  # in a row: the server is gone (a small round failing whole proves nothing)
ROWS_NAME, STATS_NAME, ATTEMPTS_NAME = "synthetic_qa.jsonl", "stats.json", "attempts.jsonl"
LOCAL_HOSTS = ("127.0.0.1", "localhost")

Teacher = Callable[[str, dict], str]  # (prompt, JSON schema of the reply) -> the reply's text


def _object(key: str, schema: dict) -> dict:
    """The JSON schema of ``{key: <schema>}`` and nothing else."""
    return {"type": "object", "properties": {key: schema}, "required": [key], "additionalProperties": False}


QUESTION_SCHEMA = _object("question", {"type": "string"})
ANSWER_SCHEMA = _object("answer", {"type": "array", "items": {"anyOf": [
    _object("text", {"type": "string"}),
    _object("quote", {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}),
    _object("cite", {"type": "integer"}),
]}})  # fmt: skip
PREMISE_SCHEMA = _object("contradicts", {"type": "boolean"})

_ASK = {
    "answerable": "Formuliere eine konkrete Rechtsfrage, die sich allein mit diesem Abschnitt beantworten "
    "lässt, so wie ein Ratsuchender sie stellen würde. Nenne die Fundstelle nicht.",
    "false_premise": "Formuliere eine Rechtsfrage, die eine falsche Annahme voraussetzt, der dieser Abschnitt "
    "widerspricht (etwa eine falsche Frist, Form, Zuständigkeit oder Rechtsfolge), so dass sie wie "
    "gestellt nicht beantwortet werden kann. Nenne die Fundstelle nicht.",
    "party": "Formuliere eine Frage nach dem Namen einer Partei dieses Verfahrens, etwa „Wer war der "
    "Kläger?“ oder „Wie heißt die Beklagte?“.",
}
_ASK["unanswerable"] = _ASK["answerable"]
QUESTION_PROMPT = """Du erstellst Trainingsdaten für ein juristisches Frage-Antwort-System auf Deutsch.

Abschnitt:
{passage}

{task}
Antworte nur mit JSON: {{"question": "..."}}"""
ANSWER_PROMPT = """Beantworte die Frage ausschließlich anhand der nummerierten Passagen.

Frage: {question}

{passages}

Antworte nur mit JSON: {{"answer": [...]}}, einer Liste von Teilen in Lesereihenfolge:
- {{"text": "..."}}: eigener Text auf Deutsch; Leerzeichen und Satzzeichen gehören in den Text.
- {{"quote": [K, J]}}: zitiert Satz [J] der Passage K wörtlich; schreibe den Satz nicht selbst.
- {{"cite": K}}: verweist auf Passage K.
Belege jede Aussage direkt danach mit quote oder cite. Antworte knapp (ein bis drei Sätze) und nenne
keine Personen. Lässt sich die Frage mit den Passagen nicht beantworten oder setzt sie etwas voraus,
dem die Passagen widersprechen, antworte {{"answer": []}}."""
PREMISE_PROMPT = """Abschnitt:
{passage}

Frage: {question}

Setzt die Frage etwas voraus, dem der Abschnitt widerspricht?
Antworte nur mit JSON: {{"contradicts": true}} für ja, {{"contradicts": false}} für nein."""


class TeacherError(RuntimeError):
    """The teacher gave no reply after the retries."""


class Rejected(Exception):
    """An attempt that yields no row; the message is the reason the stats count."""


def http_teacher(
    endpoint: str, model: str, seed: int = 0, timeout: float = 600.0, retries: int = 3
) -> Teacher:
    """The ``Teacher`` of an OpenAI-compatible server: one chat completion per call, the reply held to
    the JSON schema, Qwen3's thinking off and its recommended non-thinking sampling."""
    import requests

    url = endpoint.rstrip("/") + "/chat/completions"

    def ask(prompt: str, schema: dict) -> str:
        body = {
            "model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1024,
            "temperature": 0.7, "top_p": 0.8, "top_k": 20, "seed": seed,
            "response_format": {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema}},
            "chat_template_kwargs": {"enable_thinking": False},
        }  # fmt: skip
        error: Exception | None = None
        for attempt in range(retries):
            try:
                response = requests.post(url, json=body, timeout=timeout)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as e:
                error = e
                if attempt + 1 < retries:
                    time.sleep(2**attempt)
        raise TeacherError(f"no reply after {retries} tries: {type(error).__name__}")

    return ask


def parse_question(reply: str) -> str:
    question = json.loads(reply)["question"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("no question")
    return " ".join(question.split())  # one line: the prompt format ends the question at the newline


def parse_answer(reply: str) -> tuple:
    """``sft_format`` segments of the teacher's JSON answer, adjacent text joined and empty text
    dropped; ``()`` when the teacher declares the question unanswerable (an empty list)."""
    parts = json.loads(reply)["answer"]
    if parts == []:
        return ()
    segments: list = []
    for part in parts:
        match part:
            case {"text": str(text)} if len(part) == 1:
                if segments and isinstance(segments[-1], Text):
                    segments[-1] = Text(segments[-1].text + text)
                elif text:
                    segments.append(Text(text))
            case {"quote": [int(k), int(j)]} if len(part) == 1:
                segments.append(Quote(k, j))
            case {"cite": int(k)} if len(part) == 1:
                segments.append(Cite(k))
            case _:
                raise ValueError(f"not an answer part: {part!r}")
    if not segments:
        raise ValueError("empty answer")
    return tuple(segments)


def parse_contradicts(reply: str) -> bool:
    contradicts = json.loads(reply)["contradicts"]
    if not isinstance(contradicts, bool):
        raise ValueError("not a yes or no")
    return contradicts


def _reply(teacher: Teacher, prompt: str, schema: dict, parse):
    try:
        reply = teacher(prompt, schema)
    except TeacherError:
        raise Rejected("teacher error") from None
    try:
        return parse(reply)
    except (ValueError, KeyError, TypeError):
        raise Rejected("malformed reply") from None


def _show(chunk: Chunk) -> str:
    return chunk.citation_id + "\n" + "\n".join(f"[{j}] {s}" for j, s in enumerate(chunk.sentences, start=1))


@dataclass
class Attempt:
    n: int
    kind: str
    source: Chunk
    rng: random.Random
    question: str = ""
    passages: list[Chunk] = field(default_factory=list)
    answer: tuple = ()
    contradicts: bool = False
    reason: str | None = None

    def ask_question(self, teacher: Teacher) -> None:
        prompt = QUESTION_PROMPT.format(passage=_show(self.source), task=_ASK[self.kind])
        try:
            self.question = _reply(teacher, prompt, QUESTION_SCHEMA, parse_question)
        except Rejected as e:
            self.reason = str(e)
        else:
            if self.kind == "party" and not asks_for_party(self.question):
                self.reason = "party question not recognised"

    def ask_answer(self, teacher: Teacher) -> None:
        """The teacher's answer over the prompt's passages (which a refusal's label must survive) and,
        for a false premise, whether the source contradicts the question."""
        if self.reason is not None or self.kind == "party":  # verify refuses it whatever the answer
            return
        shown = "\n\n".join(f"Passage {k}: {_show(c)}" for k, c in enumerate(self.passages, start=1))
        prompt = ANSWER_PROMPT.format(question=self.question, passages=shown)
        try:
            self.answer = _reply(teacher, prompt, ANSWER_SCHEMA, parse_answer)
            if self.kind == "false_premise":
                prompt = PREMISE_PROMPT.format(passage=_show(self.source), question=self.question)
                self.contradicts = _reply(teacher, prompt, PREMISE_SCHEMA, parse_contradicts)
        except Rejected as e:
            self.reason = str(e)


def next_kinds(remaining: dict[str, int], n: int) -> list[str]:
    """Up to ``n`` kinds, none more often than it is still missing, interleaved in proportion."""
    slots = sorted(((j + 0.5) / r, KINDS.index(k), k) for k, r in remaining.items() for j in range(r))
    return [k for *_, k in slots[:n]]


def fused_rankings(
    questions: list[str], bm25: BM25Index, dense: DenseIndex | None, encode
) -> list[list[str]]:
    """Chunk ids per question, best first: BM25, fused by RRF with the dense ranking when there is one."""
    rankings = [bm25.query_many(questions, k=CANDIDATES)]
    if dense is not None:
        rankings.append(dense.query_vectors(encode(questions), k=CANDIDATES))
    return [[h.chunk_id for h in reciprocal_rank_fusion(per_question)] for per_question in zip(*rankings)]


def add_passages(a: Attempt, ranked: list[str], usable: dict[str, Chunk]) -> None:
    """The prompt's passages: the source (left out for ``unanswerable``, with its document's copies)
    and the best retrieved others, 3 to 8 within the token budget, shuffled."""
    own = [] if a.kind == "unanswerable" else [a.source]
    left_out = {a.source.doc_id, *a.source.copies} if a.kind == "unanswerable" else set()
    others = [
        usable[i] for i in ranked if i in usable and i != a.source.id and usable[i].doc_id not in left_out
    ]
    budget = TOKEN_BUDGET - sum(estimate_tokens(c.text) for c in own)
    n = a.rng.randint(MIN_CHUNKS, MAX_CHUNKS) - len(own)
    a.passages = own + select([Ranked(c, 0.0, estimate_tokens(c.text)) for c in others], n, budget)
    a.rng.shuffle(a.passages)
    if not a.passages:
        a.reason = "no passages"


def check_answer(a: Attempt, nli: NLI) -> tuple[str | None, str | None]:
    """``(why the teacher's answer is no answer, why a later claim of it failed)``, ``None`` for each
    that does not apply: a declared refusal, a pointer that does not exist or a refused main claim
    make it no answer."""
    if not a.answer:
        return "teacher: declared unanswerable", None
    try:
        _, target = render(Example(a.question, tuple(c.passage() for c in a.passages), a.answer))
    except ValueError as e:
        return f"invalid example: {e}", None
    result = verify(a.question, target, a.passages, nli)
    if result.refused:
        return f"verifier: {result.reason}", None
    return None, next((c.reason for c in result.claims if not c.passed), None)


def finish(a: Attempt, nli: NLI, identity: dict) -> dict:
    """The attempt's record: the row when an answerable answer passes every claim, or a refusal's label
    survives its checks (a main claim that verifies is enough to overturn it)."""
    row = None
    if a.reason is None and a.kind != "party":
        no_answer, later = check_answer(a, nli)
        if a.kind == "answerable":
            a.reason = no_answer or (later and f"verifier: a later claim failed: {later}")
        elif no_answer is None:
            a.reason = "refusal: the teacher's answer verifies"
        elif a.kind == "false_premise" and not a.contradicts:
            a.reason = "false premise not confirmed"
    if a.reason is None:
        answer = a.answer if a.kind == "answerable" else ()
        try:
            prompt, target = render(Example(a.question, tuple(c.passage() for c in a.passages), answer))
        except ValueError as e:
            a.reason = f"invalid example: {e}"
        else:
            row = {
                "id": f"synthetic:s{identity['seed']}:{a.n}",
                "question_id": "q:" + hashlib.sha256(a.question.encode("utf-8")).hexdigest()[:16],
                "source": "synthetic", "kind": a.kind, "chunk_ids": [c.id for c in a.passages],
                "text": prompt + target,
            }  # fmt: skip
    return {"attempt": a.n, **identity, "kind": a.kind, "reason": a.reason, "row": row}


def cap_rows(rows: list[dict], quota: dict[str, int]) -> list[dict]:
    """The first rows of each kind up to its quota: a rerun with a smaller count writes no more."""
    seen: Counter = Counter()
    capped = []
    for row in rows:
        seen[row["kind"]] += 1
        if seen[row["kind"]] <= quota.get(row["kind"], 0):
            capped.append(row)
    return capped


def fit_share(rows: list[dict], seed: int) -> list[dict]:
    """The rows, in order, without a seeded sample of the over-represented side, so the refusal share
    lies in ``SHARE_RANGE``; no rows when no subset of them can fit."""
    answerable = [r for r in rows if r["kind"] == KINDS[0]]
    refusals = [r for r in rows if r["kind"] != KINDS[0]]
    n_answerable, n_refusals = len(answerable), len(refusals)
    while n_refusals and n_refusals / (n_answerable + n_refusals) > SHARE_RANGE[1]:
        n_refusals -= 1
    while n_answerable and n_refusals / (n_answerable + n_refusals) < SHARE_RANGE[0]:
        n_answerable -= 1
    rng = random.Random(f"{seed}:share")
    kept = {r["id"] for r in [*rng.sample(answerable, n_answerable), *rng.sample(refusals, n_refusals)]}
    return [r for r in rows if r["id"] in kept]


def read_attempts(path: Path, identity: dict) -> list[dict]:
    """The attempts a previous run finished; a line cut off by a kill is dropped from the file. Attempts
    of another seed, teacher or index are refused, so one rows file has one provenance."""
    if not path.exists():
        return []
    records = [json.loads(line) for line in read_complete_lines(path)]
    if records and (saved := {k: records[0].get(k) for k in identity}) != identity:
        raise SystemExit(f"{path} was written with {saved}, not {identity}")
    return records


def generate(index_dir: Path, out_dir: Path, teacher: Teacher, nli: NLI, count: int, seed: int,
             encode=None, concurrency: int = 32, model: str = TEACHER) -> dict:  # fmt: skip
    """Write ``count`` verified rows to ``out_dir`` (see the module docstring); return the stats."""
    n_refusal = round(count * REFUSAL_SHARE)
    if count < 1 or not SHARE_RANGE[0] <= n_refusal / count <= SHARE_RANGE[1]:
        raise SystemExit(f"--count {count} cannot hold a refusal share in {list(SHARE_RANGE)}")
    usable: dict[str, Chunk] = {}
    with open(index_dir / CHUNKS_NAME, encoding="utf-8") as f:
        for line in f:
            if line.strip() and (c := Chunk.from_json(line)).commercial_safe and not c.research_only:
                usable[c.id] = c
    decisions = [c for c in usable.values() if c.doc_type == "decision"]
    pools = [p for p in ([c for c in usable.values() if c.doc_type != "decision"], decisions) if p]
    if not pools:
        raise SystemExit(f"{index_dir}: no commercial-safe chunks")
    bm25 = BM25Index.load(index_dir / "bm25")
    dense = (
        DenseIndex.load(index_dir / "dense")
        if encode is not None and (index_dir / "dense").exists()
        else None
    )
    refusals = KINDS[1:] if decisions else KINDS[1:3]  # no decision, no party question
    quota = {KINDS[0]: count - n_refusal}
    quota |= {k: n_refusal // len(refusals) + (i < n_refusal % len(refusals)) for i, k in enumerate(refusals)}

    out_dir.mkdir(parents=True, exist_ok=True)
    identity = {"seed": seed, "teacher": model, "index": str(index_dir)}
    records = read_attempts(out_dir / ATTEMPTS_NAME, identity)
    with open(out_dir / ATTEMPTS_NAME, "a", encoding="utf-8") as log, ThreadPoolExecutor(concurrency) as pool:
        while len(records) < MAX_ATTEMPTS * count:
            kept = Counter(r["kind"] for r in records if r["row"])
            remaining = {k: q - kept[k] for k, q in quota.items() if q > kept[k]}
            if not remaining:
                break
            batch = []
            for kind in next_kinds(remaining, BATCH):
                rng = random.Random(f"{seed}:{len(records) + len(batch)}")
                source = rng.choice(decisions if kind == "party" else rng.choice(pools))
                batch.append(Attempt(len(records) + len(batch), kind, source, rng))
            list(pool.map(lambda a: a.ask_question(teacher), batch))
            live = [a for a in batch if a.reason is None]
            if live:
                for a, ranked in zip(live, fused_rankings([a.question for a in live], bm25, dense, encode)):
                    add_passages(a, ranked, usable)
            list(pool.map(lambda a: a.ask_answer(teacher), batch))
            new = [finish(a, nli, identity) for a in batch]
            log.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in new)
            log.flush()
            records += new
            errors = next(
                (i for i, r in enumerate(reversed(records)) if r["reason"] != "teacher error"), len(records)
            )
            if errors >= MAX_TEACHER_ERRORS:
                raise SystemExit(f"the last {errors} attempts got no teacher reply: is the server up?")

    verified = [r["row"] for r in records if r["row"]]
    capped = cap_rows(verified, quota)
    rows = fit_share(capped, seed)
    kinds = Counter(r["kind"] for r in rows)
    share = (len(rows) - kinds[KINDS[0]]) / len(rows) if rows else 0.0
    rejected = Counter(re.sub(r"\d+", "N", r["reason"]) for r in records if r["reason"])
    stats = {
        "requested": count, "kept": len(rows), "attempts": len(records), "verified": len(verified),
        "rejected": sum(rejected.values()), "rejected_by_reason": dict(rejected.most_common()),
        "dropped_over_count": len(verified) - len(capped), "dropped_for_share": len(capped) - len(rows),
        "kept_by_kind": {k: kinds[k] for k in KINDS}, "refusal_share": share, "teacher_model": model,
        "seed": seed,
    }  # fmt: skip
    write_atomic(out_dir / STATS_NAME, json.dumps(stats, indent=1, ensure_ascii=False) + "\n")
    if not SHARE_RANGE[0] <= share <= SHARE_RANGE[1]:
        raise SystemExit(f"refusal share {share:.3f} outside {list(SHARE_RANGE)}: {stats['kept_by_kind']}")
    write_atomic(out_dir / ROWS_NAME, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--index", type=Path, required=True, help="an index directory written by scripts/build_index.py"
    )
    p.add_argument("--endpoint", default="http://127.0.0.1:8000/v1", help="the local server's OpenAI API")
    p.add_argument("--model", default=TEACHER, help="the open-weight teacher the server serves")
    p.add_argument("--count", type=int, required=True, help="rows to write")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--concurrency", type=int, default=32, help="teacher requests in flight")
    p.add_argument("--device", default=None, help="for the NLI model and bge-m3 (default: cuda if available)")
    args = p.parse_args(argv)
    if urlparse(args.endpoint).hostname not in LOCAL_HOSTS:
        p.error(
            f"--endpoint must be a server in this job ({', '.join(LOCAL_HOSTS)}): open-weight teachers only"
        )
    stats = generate(
        args.index, args.out, http_teacher(args.endpoint, args.model, args.seed), MDebertaNLI(device=args.device),
        args.count, args.seed, BgeM3Encoder(device=args.device), args.concurrency, args.model,
    )  # fmt: skip
    print(f"SYNTH kept {stats['kept']}/{stats['requested']} in {stats['attempts']} attempts, "
          f"refusal share {stats['refusal_share']:.3f}, {stats['kept_by_kind']}, dropped "
          f"{stats['dropped_over_count']} over the count and {stats['dropped_for_share']} for the share "
          f"-> {args.out}")  # fmt: skip
    for reason, n in stats["rejected_by_reason"].items():
        print(f"  rejected {n}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
