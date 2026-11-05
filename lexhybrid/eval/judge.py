"""The independent answer judge: a strong LLM, not the verifier, scores the answers the headline reports.

The verifier (pointer, quote, NLI) also shaped training, so its verdicts cannot be the headline. The
judge is any model behind ``Complete`` (prompt in, completion text out); ``OpenAICompatible`` is the
one client, an OpenAI-compatible chat-completions endpoint (a hosted API or a local vLLM server) at
temperature 0. The judge is validated against gold labels before it scores any system
(``scripts/build_judge_calibration.py``, ``scripts/score_judge_calibration.py``).

The judge only ever sees evaluation material: evaluation questions, the prompt's passages, the
system's rendered answer and the gold answers. Training data is never sent to it.

Scores per answer (``judge_record``), each in [0, 1] or ``None``:

* **correctness**: the answer against the gold answers, ``correct`` 1 / ``partial`` 0.5 /
  ``incorrect`` 0, with a one-sentence reason. An answer to a question a correct system refuses
  (``AnswerRecord.should_refuse``), and a refusal of a question with gold answers, are scored
  without the judge: 1 when the answer refuses exactly when it should. A refusal of a question with
  neither scores ``None``, as an answer to it would.
* **faithfulness** (RAGAS): the judge splits the answer into atomic statements and marks each
  supported or not by the passages; the share supported. Not asked of a refusal.
* **answer_relevancy**: the judge-rated variant: the judge rates the answer's relevance to the
  question from 0 to 1 (RAGAS instead embeds questions generated from the answer). Not asked of a
  refusal.
* **context_precision** (RAGAS): the judge marks each passage useful or not for reaching the gold
  answer; precision@k averaged over the positions k of the useful passages (0 when none is).
* **context_recall** (RAGAS): the share of the first gold answer's statements the judge attributes to
  the passages.

Every prompt asks for one JSON object; the first JSON object in the reply is read. A reply that does
not parse, lacks a field or lists no statement gets one follow-up (the prompt plus ``RETRY``); if
that fails too the score is ``None`` and the raw reply is kept, so a long run never stops on a bad
reply. ``OpenAICompatible`` retries rate limits, server errors, lost connections and timeouts with
exponential backoff (``TRIES`` attempts), then raises; what was judged so far stays in the cache.

A reply that parsed is cached under ``cache_dir`` as ``<sha256>.json``, keyed by (endpoint, model
id, ``PROMPT_VERSION``, prompt) and holding those, a timestamp and the raw reply, written
atomically, so a rerun is free and identical. A reply that did not parse is never cached (a rerun
asks again), and a corrupt or incomplete cache file is a miss.
"""

import argparse
import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import requests

from lexhybrid.eval.records import AnswerRecord
from lexhybrid.renderer.render import NOTICE
from lexhybrid.retrieval.index import write_atomic

Complete = Callable[[str], str]
Read = Callable[[dict], float]  # raises KeyError, TypeError or ValueError on a bad reply

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "judge"
PROMPT_VERSION = 1  # part of the cache key: bump it whenever a prompt below changes
TRIES, BACKOFF_S = 5, 2.0  # HTTP attempts; waits 2, 4, 8, 16 s between them
METRICS = ("correctness", "faithfulness", "answer_relevancy", "context_precision", "context_recall")
VERDICTS = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
RETRY = "\n\nYour previous reply could not be parsed. Reply with the JSON object only, nothing else."
_JSON_ONLY = "Reply with one JSON object and nothing else:"
_THINK = re.compile(r"<think>.*?(?:</think>|$)", re.S)


class OpenAICompatible:
    """``POST {base_url}/chat/completions``: one user message, temperature 0; returns the reply text."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 300.0):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key, self.timeout = model, api_key, timeout

    def __call__(self, prompt: str) -> str:
        for attempt in range(TRIES):
            try:
                r = requests.post(
                    self.url,
                    headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                    json={"model": self.model, "messages": [{"role": "user", "content": prompt}],
                          "temperature": 0},
                    timeout=self.timeout,
                )  # fmt: skip
                if r.status_code == 429 or r.status_code >= 500:
                    r.raise_for_status()
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
                if attempt == TRIES - 1:
                    raise
                time.sleep(BACKOFF_S * 2**attempt)
                continue
            r.raise_for_status()
            try:
                return r.json()["choices"][0]["message"]["content"] or ""
            except (ValueError, KeyError, IndexError, TypeError):
                return ""  # a malformed success: unparsed like any bad reply, not the end of a long run


def first_json_object(text: str) -> dict | None:
    """The first ``{...}`` in ``text`` that parses as a JSON object (code fences and prose around it
    ignored). A reasoning block goes first: a draft verdict in it is not the reply."""
    text = _THINK.sub("", text)
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text, i)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def _parse(raw: str, read: Read) -> tuple[float, dict]:
    obj = first_json_object(raw)
    if obj is None:
        raise ValueError("no JSON object in the reply")
    return read(obj), obj


class Judge:
    """A ``Complete`` behind the reply cache; ``calls`` counts the completions actually requested.
    ``base_url`` names the endpoint in the cache key, so two endpoints never share verdicts."""

    def __init__(self, complete: Complete, model: str, cache_dir: str | Path, base_url: str = ""):
        self._complete, self.model, self.cache_dir = complete, model, Path(cache_dir)
        self.base_url = base_url.rstrip("/")
        self.calls = 0

    def ask(self, prompt: str, read: Read) -> tuple[float | None, dict]:
        """``(read(reply object), reply object)``, retried once; ``(None, {"raw": reply})`` when both fail."""
        key_fields = [self.base_url, self.model, PROMPT_VERSION, prompt]
        key = hashlib.sha256(json.dumps(key_fields, ensure_ascii=False).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        try:
            return _parse(json.loads(path.read_text(encoding="utf-8"))["response"], read)
        except (OSError, KeyError, TypeError, ValueError):
            pass  # no entry, or a corrupt one: ask
        raw = ""
        for attempt, p in enumerate((prompt, prompt + RETRY)):
            raw = self._complete(p)
            self.calls += 1
            try:
                parsed = _parse(raw, read)
            except (KeyError, TypeError, ValueError):
                continue
            entry = {"base_url": self.base_url, "model": self.model, "prompt_version": PROMPT_VERSION,
                     "timestamp": datetime.now(UTC).isoformat(timespec="seconds"), "retried": bool(attempt),
                     "response": raw}  # fmt: skip
            write_atomic(path, json.dumps(entry, ensure_ascii=False) + "\n")
            return parsed
        return None, {"raw": raw}


# ---------------------------------------------------------------------------------------------
# prompts and their readers


def _numbered(texts: Sequence[str]) -> str:
    return "\n\n".join(f"[{i}] {t}" for i, t in enumerate(texts, start=1)) or "(none)"


def _listed(texts: Sequence[str]) -> str:
    return "\n".join(f"- {t}" for t in texts)


def correctness_prompt(question: str, answer: str, references: Sequence[str]) -> str:
    return f"""You grade a system's answer to a German legal question against reference answers.

Question:
{question}

Reference answers (agreeing with any one of them is enough):
{_listed(references)}

Answer to grade:
{answer}

Verdicts:
- "correct": the answer states the substance of a reference answer and contradicts none of it.
- "partial": the answer states part of that substance and contradicts none of it.
- "incorrect": the answer misses the substance, or contradicts a reference answer in any point: another provision (§), another number, period or amount, or the opposite statement.
Grade the content only, not wording, length or style.

{_JSON_ONLY}
{{"verdict": "correct" | "partial" | "incorrect", "reason": "<one sentence>"}}"""


def read_verdict(obj: dict) -> float:
    return VERDICTS[obj["verdict"]]


def faithfulness_prompt(question: str, answer: str, contexts: Sequence[str]) -> str:
    asked = f"Question:\n{question}\n\n" if question else ""
    return f"""You check whether an answer is supported by the passages it was given.

{asked}Passages:
{_numbered(contexts)}

Answer:
{answer}

Split the answer into atomic statements: each a single self-contained claim, in the answer's language. A citation alone, such as "(BGB §573 Abs. 2)", is not a statement. Mark a statement supported only if it can be inferred directly from the passages, without outside knowledge.

{_JSON_ONLY}
{{"statements": [{{"statement": "<atomic statement>", "supported": true | false}}]}}"""


def _share(obj: dict, flag: str) -> float:
    flags = [s[flag] for s in obj["statements"]]
    if not flags or not all(isinstance(f, bool) for f in flags):
        raise ValueError(f"statements must be a non-empty list, each {flag} true or false")
    return sum(flags) / len(flags)


def read_supported(obj: dict) -> float:
    return _share(obj, "supported")


def relevancy_prompt(question: str, answer: str) -> str:
    return f"""You rate how relevant an answer is to a German legal question.

Question:
{question}

Answer:
{answer}

Rate relevance only, not correctness: 1.0 when the answer addresses exactly what is asked; lower when it addresses only part of it or adds unrelated material; 0.0 when it does not address the question or is evasive.

{_JSON_ONLY}
{{"relevancy": <number from 0.0 to 1.0>, "reason": "<one sentence>"}}"""


def read_relevancy(obj: dict) -> float:
    x = float(obj["relevancy"])
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"relevancy {x} outside [0, 1]")
    return x


def precision_prompt(question: str, contexts: Sequence[str], references: Sequence[str] = ()) -> str:
    """Which passages are useful: for reaching the reference answers, or for answering the question when
    there are none (the retrieval calibration sets have no answers)."""
    goal = "arriving at the reference answer" if references else "answering the question"
    given = f"Reference answers:\n{_listed(references)}\n\n" if references else ""
    return f"""You judge which passages are useful for a German legal question.

Question:
{question}

{given}Passages:
{_numbered(contexts)}

For each passage, in order, decide whether it is useful for {goal}: it holds information the answer rests on, not merely the same topic.

{_JSON_ONLY}
{{"useful": [true | false, ...]}} with exactly {len(contexts)} entries, one per passage."""


def average_precision(useful: Sequence[bool]) -> float:
    """RAGAS context precision: precision@k averaged over the positions k of the useful passages."""
    hits, total = 0, 0.0
    for k, u in enumerate(useful, start=1):
        if u:
            hits += 1
            total += hits / k
    return total / hits if hits else 0.0


def read_useful(n: int) -> Read:
    def read(obj: dict) -> float:
        useful = obj["useful"]
        if len(useful) != n or not all(isinstance(u, bool) for u in useful):
            raise ValueError(f"useful must be {n} true/false entries")
        return average_precision(useful)

    return read


def recall_prompt(question: str, reference: str, contexts: Sequence[str]) -> str:
    return f"""You check how much of a reference answer to a German legal question the passages cover.

Question:
{question}

Reference answer:
{reference}

Passages:
{_numbered(contexts)}

Split the reference answer into its statements (sentences or atomic claims). Mark a statement attributed if the passages state it or it follows directly from them.

{_JSON_ONLY}
{{"statements": [{{"statement": "<statement>", "attributed": true | false}}]}}"""


def read_attributed(obj: dict) -> float:
    return _share(obj, "attributed")


# ---------------------------------------------------------------------------------------------
# records


def answer_text(record: AnswerRecord) -> str:
    """What the reader sees, without the fixed legal-information notice."""
    return record.rendered.removesuffix(NOTICE).strip()


def judge_prompts(record: AnswerRecord) -> dict[str, tuple[str, Read]]:
    """The prompts a record needs, by metric, each with the reader of its reply."""
    contexts = [c.text for c in record.chunk_objects()]
    q, answer, gold = record.question, answer_text(record), record.gold_answers
    out: dict[str, tuple[str, Read]] = {}
    if gold and not (record.refused or record.should_refuse):
        out["correctness"] = (correctness_prompt(q, answer, gold), read_verdict)
    if not record.refused:
        out["faithfulness"] = (faithfulness_prompt(q, answer, contexts), read_supported)
        out["answer_relevancy"] = (relevancy_prompt(q, answer), read_relevancy)
    if gold and contexts:
        out["context_precision"] = (precision_prompt(q, contexts, gold), read_useful(len(contexts)))
        out["context_recall"] = (recall_prompt(q, gold[0], contexts), read_attributed)
    return out


def judge_record(judge: Judge, record: AnswerRecord) -> tuple[dict[str, float | None], dict[str, dict]]:
    """``(scores, details)``: a score per metric (``None`` when not applicable or not parsed) and, per
    metric asked, the judge's reply object (``{"raw": text}`` when it did not parse)."""
    scores: dict[str, float | None] = dict.fromkeys(METRICS)
    details: dict[str, dict] = {}
    if record.should_refuse or (record.refused and record.gold_answers):
        ok = record.refused == record.should_refuse
        scores["correctness"] = float(ok)
        did = "refused" if record.refused else "answered"
        should = "should" if record.should_refuse else "should not"
        details["correctness"] = {"verdict": "correct" if ok else "incorrect",
                                  "reason": f"{did} a question that {should} be refused"}  # fmt: skip
    for metric, (prompt, read) in judge_prompts(record).items():
        scores[metric], details[metric] = judge.ask(prompt, read)
    return scores, details


# ---------------------------------------------------------------------------------------------
# command line


def add_judge_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--base-url", required=True, help="OpenAI-compatible endpoint, e.g. http://host:8000/v1"
    )
    parser.add_argument("--model", required=True, help="the judge model id the endpoint serves")
    parser.add_argument("--api-key-env", default=None, help="environment variable holding the API key")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)


def judge_from_args(args: argparse.Namespace) -> Judge:
    api_key = None
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise SystemExit(f"${args.api_key_env} is not set")
    complete = OpenAICompatible(args.base_url, args.model, api_key)
    return Judge(complete, args.model, args.cache_dir, base_url=args.base_url)
