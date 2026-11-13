#!/usr/bin/env python3
"""System ablation: the same questions through arms that differ in one component each.

    .venv/bin/python scripts/eval_system_ablation.py --questions data/qa/<set>.jsonl \\
        --index data/index/<version> --model-config hybrid_legal_base \\
        --checkpoint outputs/<run>/checkpoints/last.ckpt --gold-prompts <gold prompt sets>.jsonl \\
        --out logs/eval_system_ablation/<run>

Arms, one AnswerRecord JSONL each (``<out>/<arm>.jsonl``). *Selection* is A3's step from the fused
candidates to the prompt (``rerank.select``: at most 8 passages within 4,096 tokens), taken in the
reranker's order or, with the reranker off (every score tied), in the fused order:

* ``A1`` retriever only: the top chunk of the fused-order selection, verbatim, as a quote of all its
  sentences from passage 1 (``<|q|><|c1|><|s1|>…<|sN|>`` + EOS), so it renders and verifies like a
  model answer;
* ``A2`` retriever + reranker: the top chunk of the reranked selection (passage 1 of A3), the same way;
* ``A3`` retriever + reranker + model: ``answer.py``'s pipeline;
* ``A4_sel`` retriever + model: the fused-order selection as the prompt;
* ``A4_k5``, ``A4_k10``, ``A4_k16`` retriever + model: the fused top k, no selection limits;
* ``A5_<arm>`` model on gold passages: each record of ``--gold-prompts`` decoded as it stands.

``COMPARISONS`` (also in the summary) names what each comparison isolates. Every model arm decodes
with the same generator (decoding settings and seed) and every arm verifies through the same verifier.

``--questions`` rows carry ``id`` and ``question``, and their gold as AnswerRecord fields (``kind``,
``gold_answers``, ``gold_chunk_ids``) or in the evalsets' QA format (``answers``; ``gold``: citations
such as ``BGB §573b``, resolved to the index's chunks). A row without an id or a question is skipped.

Each record is appended to its arm's file as soon as it is verified, and a rerun into the same
``--out`` skips the (id, arm) pairs already there (a last line cut by preemption is dropped first), so
a requeued job resumes; a rerun with other settings (``<out>/settings.json``) is refused, since its
records would mix with the old ones. ``<out>/summary.json`` holds the settings, the question counts
(rows, skipped, with gold answers, with gold chunks) and per arm the records, refusals, mean new
tokens and the ids whose prompt (in the hybrid model's tokens) is longer than the 8,192 it was trained
at (``extrapolated``). Scoring is left to the scripts that read the records.
"""

import argparse
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from statistics import fmean

from lexhybrid.data.sft_format import Q, prompt_ids
from lexhybrid.data.tokenizer import EOS_TOKEN
from lexhybrid.eval.gold_passages import gold_chunk_ids
from lexhybrid.eval.records import AnswerRecord, chunk_dict, read_records
from lexhybrid.retrieval.rerank import rerank_and_select
from lexhybrid.utils.jsonl import read_complete_lines

_spec = importlib.util.spec_from_file_location("answer", Path(__file__).with_name("answer.py"))
answer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(answer)

FUSED_K = (5, 10, 16)  # the pointer format holds at most 16 passages
QUESTION_ARMS = ("A1", "A2", "A3", "A4_sel", *(f"A4_k{k}" for k in FUSED_K))
TRAINED_CONTEXT = 8192
COMPARISONS = {
    "A1 vs A2": "the reranker, answering with the top chunk copied",
    "A4_sel vs A3": "the reranker, with the model answering from the selection",
    "A2 vs A3": "the model, against copying the reranked top chunk",
    "A4_k5 vs A4_k10 vs A4_k16": "how many fused candidates the model reads, without selection limits",
    "A5_* vs A3": "retrieval: gold passages among distractors in place of the retrieved ones",
}


def copy_top(question: str, chunks) -> "answer.Generation":
    """Retrieval's own answer in the pointer format: passage 1 quoted whole, nothing decoded."""
    span = "".join(f"<|s{j}|>" for j in range(1, len(chunks[0].sentences) + 1))
    return answer.Generation(f"{Q}<|c1|>{span}{EOS_TOKEN}", 0, 0)


def read_questions(path: Path, set_name: str, system: str, seed: int, index_dir: Path):
    """``(questions, counts)``: the rows as AnswerRecords to fill, and how many carry gold."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    kept = [r for r in rows if "id" in r and r.get("question")]
    needs_index = any("gold" in r or "gold_chunk_ids" in r for r in kept)
    gold = gold_chunk_ids(kept, answer._index(index_dir)[0].values() if needs_index else ())
    questions = [
        AnswerRecord(
            r["id"], set_name, system, "", r["question"], seed=seed, kind=r.get("kind", "answerable"),
            gold_answers=list(r.get("gold_answers", r.get("answers", []))), gold_chunk_ids=ids,
        )
        for r, ids in zip(kept, gold, strict=True)
    ]  # fmt: skip
    counts = {"rows": len(rows), "skipped": len(rows) - len(kept),
              "with_gold_answers": sum(bool(q.gold_answers) for q in questions),
              "with_gold_chunks": sum(bool(q.gold_chunk_ids) for q in questions)}  # fmt: skip
    return questions, counts


def written(path: Path) -> set[str]:
    """The ids an arm's file already holds; a last line cut by preemption is dropped from the file."""
    if not path.exists():
        return set()
    return {AnswerRecord.from_json(line).id for line in read_complete_lines(path)}


def run(questions, gold, index_dir: Path, scorer, encode, generator, nli, out: Path, jurisdictions=None,
        valid_on=None) -> None:  # fmt: skip
    """Append each arm's missing records to ``<out>/<arm>.jsonl``: A1-A4 per question, A5 per gold prompt."""
    done: dict[str, set[str]] = {}

    def todo(base: AnswerRecord, arm: str) -> bool:
        if arm not in done:
            done[arm] = written(out / f"{arm}.jsonl")
        return base.id not in done[arm]

    def add(base: AnswerRecord, arm: str, chunks, gen) -> None:
        g, result = answer.respond(base.question, chunks, gen, nli)
        record = replace(
            base, arm=arm, chunks=[chunk_dict(c) for c in chunks], raw_answer=g.raw, rendered=result.rendered,
            refused=result.refused, verifier=result.log_row(), new_tokens=g.new_tokens,
            max_new_tokens=g.max_new_tokens,
        )  # fmt: skip
        with open(out / f"{arm}.jsonl", "a", encoding="utf-8") as f:
            f.write(record.to_json() + "\n")

    for q in questions:
        if not (arms := [a for a in QUESTION_ARMS if todo(q, a)]):
            continue
        fused = answer.candidates(q.question, index_dir, jurisdictions, valid_on, encode)
        plain = rerank_and_select(q.question, fused, answer.fused_order)  # A3's selection, reranker off
        reranked = rerank_and_select(q.question, fused, scorer)  # answer.retrieve on the same candidates
        prompts = {"A1": (plain[:1], copy_top), "A2": (reranked[:1], copy_top), "A3": (reranked, generator),
                   "A4_sel": (plain, generator)} | {f"A4_k{k}": (fused[:k], generator) for k in FUSED_K}  # fmt: skip
        for arm in arms:
            add(q, arm, *prompts[arm])
    for g in gold:
        if todo(g, arm := f"A5_{g.arm}"):
            add(g, arm, g.chunk_objects(), generator)


def prompt_tokens(record: AnswerRecord, tok) -> int:
    """A decoded record's prompt length in the hybrid model's tokens (0 when nothing was decoded)."""
    if not record.max_new_tokens:
        return 0
    return len(prompt_ids(record.question, [c.passage() for c in record.chunk_objects()], tok))


def summary(records: list[AnswerRecord], tok) -> dict:
    return {
        "records": len(records),
        "refused": sum(r.refused for r in records),
        "mean_new_tokens": fmean(r.new_tokens for r in records) if records else 0.0,
        "extrapolated": [r.id for r in records if prompt_tokens(r, tok) > TRAINED_CONTEXT],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--questions", type=Path, required=True, help="JSONL: id, question and gold (see above)")
    p.add_argument("--set", help="the evaluation set's name (default: the questions file's stem)")
    p.add_argument(
        "--gold-prompts", type=Path, action="append", default=[], help="gold-passage prompt sets; repeatable"
    )
    p.add_argument("--system", default="hybrid")
    p.add_argument("--out", type=Path, required=True, help="directory for <arm>.jsonl and summary.json")
    answer.add_arguments(p)
    args = p.parse_args(argv)
    set_name = args.set or args.questions.stem
    questions, counts = read_questions(args.questions, set_name, args.system, args.seed, args.index)
    gold = [
        replace(r, system=args.system, seed=args.seed)
        for path in args.gold_prompts
        for r in read_records(path)
    ]
    if bad := [r.id for r in questions + gold if "<|" in r.question]:
        p.error(f"questions may not contain special-token markup ('<|'): {bad}")
    counts["gold_prompts"] = len(gold)
    print("QUESTIONS " + " ".join(f"{k}={v}" for k, v in counts.items()))
    args.out.mkdir(parents=True, exist_ok=True)
    settings = json.dumps(vars(args), indent=1, default=str, sort_keys=True) + "\n"
    if (args.out / "settings.json").exists() and (args.out / "settings.json").read_text() != settings:
        p.error(f"{args.out} holds a run with other settings (settings.json); use another --out")
    (args.out / "settings.json").write_text(settings)
    if questions or gold:
        scorer, encode, nli, generator = answer.pipeline(args)
        run(questions, gold, args.index, scorer, encode, generator, nli, args.out, args.jurisdiction,
            args.valid_on)  # fmt: skip
    tok = answer.load_tokenizer()
    arms = {path.stem: summary(read_records(path), tok) for path in sorted(args.out.glob("*.jsonl"))}
    for arm, s in arms.items():
        print(f"ABLATION {arm} records={s['records']} refused={s['refused']} "
              f"mean_new_tokens={s['mean_new_tokens']:.1f} extrapolated={len(s['extrapolated'])}")  # fmt: skip
    summary_json = {"settings": vars(args), "trained_context": TRAINED_CONTEXT, "questions": counts,
                    "comparisons": COMPARISONS, "arms": arms}  # fmt: skip
    (args.out / "summary.json").write_text(json.dumps(summary_json, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
