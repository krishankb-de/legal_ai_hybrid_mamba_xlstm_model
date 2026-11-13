#!/usr/bin/env python3
"""Gold-passage sanity check: each item's gold passage among k distractors; does the model cite it?

    .venv/bin/python scripts/eval_gold_passages.py build --items data/evalsets/gerlayqa/items.jsonl \\
        --index data/index/v1 --out logs/gold_passages/sets.jsonl
    .venv/bin/python scripts/eval_gold_passages.py decode --sets logs/gold_passages/sets.jsonl \\
        --system hybrid --seed 42 --model-config hybrid_legal_base \\
        --checkpoint outputs/<run>/checkpoints/last.ckpt --out logs/gold_passages/hybrid_s42.jsonl \\
        --summary logs/gold_passages/hybrid_s42.summary.json

``build`` writes the prompt sets (``lexhybrid.eval.gold_passages``: easy and hard tiers, k = 1, 4, 9 by
default) as answer records without an answer; build once, so every system decodes the same sets. The
items are QA JSONL rows (``id``, ``question``, optional ``answers``, and ``gold_chunk_ids`` or
``gold`` citation ids such as ``BGB §573b``), as ``lexhybrid.data.evalsets`` writes ``items.jsonl``,
or SFT rows (``scripts/build_sft_data.py``'s ``dev.jsonl``), whose gold is what their answer cites
(``gold_passages.sft_item``); the index is a ``scripts/build_index.py`` directory (bge-m3 joins BM25
when it has ``dense/``). Beside the sets, ``<out>.counts.json`` (``sets.counts.json``) holds how many
items got a set and how many were skipped per tier and k; a tier and k that lost items is warned about
on stderr. The sets file is written last and whole.
``decode`` answers every set greedily under the pointer FSM, verifies it with the NLI model and
appends each record to ``--out`` as it is decoded; a rerun (a requeued job) keeps the whole records
already there and decodes only the rest; ``--limit N`` decodes at most N of the rest (a smoke). It
then prints one GOLD line per tier and k and, once every set is decoded, writes them as a JSON list of
summaries (selection accuracy, any gold cited, refusal rate, verifier pass rate, selection accuracy by
gold position).
``build --as-is`` takes SFT rows and writes one set per row instead, its own prompt as it stands (arm
``as_is``: its passages in ``chunk_ids`` order from the index, its ``kind``, the gold its answer cites;
``gold_passages.as_is_sets``), retrieving nothing and writing no counts. ``decode`` treats them like any
set; its summary, which scores gold positions, lists none of them, so score them with
``scripts/evaluate_citation.py`` and ``scripts/audit_truncation.py``.
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

from lexhybrid.config import load_model_config
from lexhybrid.data.tokenizer import load_tokenizer
from lexhybrid.eval.gold_passages import (
    K_VALUES,
    as_is_sets,
    build_sets,
    decode,
    fused_retriever,
    read_chunks,
    score,
    sft_item,
)
from lexhybrid.eval.records import AnswerRecord, read_records, write_records
from lexhybrid.retrieval.dense import DENSE_MODEL, BgeM3Encoder
from lexhybrid.retrieval.rerank import CANDIDATES
from lexhybrid.training.dpo_module import load_sft_model
from lexhybrid.utils.jsonl import read_complete_lines
from lexhybrid.verifier.nli import NLI_MODEL, MDebertaNLI


def load_model(model_config: str | None, checkpoint: str, device: str, hf_model: str | None = None,
               hf_revision: str = "main", dtype: str = "float32"):  # fmt: skip
    config = None if hf_model else load_model_config(model_config)
    model = load_sft_model(checkpoint, config, hf_model, hf_revision)
    for (
        p
    ) in model.parameters():  # not Module.to(dtype): it would also round fp32 buffers such as RoPE's inv_freq
        p.data = p.data.to(getattr(torch, dtype))
    return model.to(device).eval()


def gold_sets(args, items: list[dict]) -> list[AnswerRecord]:
    """The gold-passage prompt sets, with their counts written beside ``--out`` and printed."""
    items = [r if "question" in r else sft_item(r) for r in items]  # an SFT row carries its question in text
    encode = BgeM3Encoder(args.dense_model, device=args.device)  # loads only if the index has dense/
    retrieve = fused_retriever(args.index, encode, depth=args.depth)
    sets, counts = build_sets(
        items, read_chunks(args.index), retrieve, args.k, args.seed, args.set_name or args.items.parent.name
    )
    counts_path = args.out.with_suffix(".counts.json")
    counts_path.parent.mkdir(parents=True, exist_ok=True)
    counts_path.write_text(json.dumps(counts, indent=1) + "\n")
    print(f"GOLD SETS items={counts['items']} with_gold={counts['with_gold']} -> {counts_path}")
    for arm, c in counts["sets"].items():
        print(f"GOLD SETS {arm} built={c['built']} skipped={c['skipped']}")
        if c["skipped"]:
            print(f"WARNING: {arm} lost {c['skipped']} of {counts['with_gold']} items with a gold passage "
                  "(fewer distractors than k)", file=sys.stderr)  # fmt: skip
    return sets


def build(args) -> int:
    items = [json.loads(line) for line in args.items.read_text(encoding="utf-8").split("\n") if line.strip()]
    if args.as_is:  # nothing is retrieved, so no encoder is loaded
        sets = as_is_sets(items, read_chunks(args.index), args.set_name or args.items.parent.name)
    else:
        sets = gold_sets(args, items)
    if not sets:
        print(f"FATAL: no item got a prompt set; {args.out} not written", file=sys.stderr)
        return 1
    tmp = args.out.with_suffix(".tmp")
    write_records(tmp, sets)
    tmp.replace(args.out)  # whole or not at all: a preempted build leaves no partial sets file
    print(
        f"GOLD SETS {len(sets)} prompt sets from {len({r.id for r in sets})} of {len(items)} items -> {args.out}"
    )
    return 0


def resumed(args, sets: list[AnswerRecord]) -> list[AnswerRecord]:
    """The records an earlier, preempted decode wrote whole; a line it cut short is cut from the file.
    Records of another run (system, seed, budget) or another prompt set are refused, not mixed in."""
    if not args.out.exists():
        return []
    done = [AnswerRecord.from_json(line) for line in read_complete_lines(args.out)]
    prompts = {(r.id, r.arm): r.chunks for r in sets}
    for r in done:
        if (r.system, r.seed, r.max_new_tokens) != (args.system, args.seed, args.max_new_tokens):
            raise SystemExit(
                f"{args.out} holds records of another system, seed or budget; not resuming into it"
            )
        if prompts.get((r.id, r.arm)) != r.chunks:
            raise SystemExit(
                f"{args.out}: {r.id} {r.arm} is not a prompt set of {args.sets}; not resuming into it"
            )
    return done


def run_decode(args) -> int:
    sets = read_records(args.sets)
    records = resumed(args, sets)
    finished = {(r.id, r.arm) for r in records}
    pending = [r for r in sets if (r.id, r.arm) not in finished]
    todo = pending[: args.limit]
    left = len(pending) - len(todo)
    if todo:
        model = load_model(
            args.model_config, args.checkpoint, args.device, args.hf_model, args.hf_revision, args.dtype
        )
        tok, nli = load_tokenizer(), MDebertaNLI(args.nli_model, device=args.device)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a", encoding="utf-8") as f:
            for r in todo:  # appended as decoded: a requeued job resumes after the last whole record
                records.append(decode(replace(r, system=args.system, seed=args.seed), model, tok, nli,
                                      args.max_new_tokens))  # fmt: skip
                f.write(records[-1].to_json() + "\n")
                f.flush()
    print(f"GOLD DECODE {len(finished)} resumed, {len(todo)} decoded, {left} left -> {args.out}")
    summaries = score(records)
    if left:  # no summary until every set is decoded: a job that finds one skips the run as scored
        print(f"GOLD DECODE {args.summary} not written: {left} sets left")
    else:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summaries, indent=1, ensure_ascii=False) + "\n")
    for s in summaries:
        print(
            f"GOLD {s['system']} seed={s['seed']} {s['tier']} k={s['k']} records={s['records']} "
            f"selection={s['selection_accuracy']:.4f} any_gold={s['any_gold_cited']:.4f} "
            f"refusal={s['refusal_rate']:.4f} verifier_pass={s['verifier_pass_rate']:.4f}"
        )
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="write the prompt sets")
    b.add_argument("--items", type=Path, required=True, help="QA JSONL (gold_chunk_ids or gold) or SFT rows")
    b.add_argument("--index", type=Path, required=True, help="a scripts/build_index.py directory")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--k", type=int, nargs="+", default=list(K_VALUES), help="distractor counts")
    b.add_argument(
        "--as-is", action="store_true", help="SFT rows: one set per row, its own prompt (arm as_is)"
    )
    b.add_argument("--seed", type=int, default=0, help="seeds the easy draws and the gold positions")
    b.add_argument("--set-name", help="the records' set (default: the items file's directory name)")
    b.add_argument("--depth", type=int, default=CANDIDATES, help="ranked chunks per retriever")
    b.add_argument("--dense-model", default=DENSE_MODEL, help="used when the index has dense/")
    b.add_argument("--device", default=None)
    d = sub.add_parser("decode", help="answer and verify the prompt sets, then score them")
    d.add_argument("--sets", type=Path, required=True, help="the output of build")
    d.add_argument("--system", required=True, help="the records' system name")
    d.add_argument("--seed", type=int, help="the records' seed (the checkpoint's training seed)")
    d.add_argument("--checkpoint", required=True)
    model = d.add_mutually_exclusive_group(required=True)
    model.add_argument("--model-config", help="a name under configs/model/ or a yaml path")
    model.add_argument(
        "--hf-model", help="the Hugging Face baseline's repository (its SFT checkpoint: --checkpoint)"
    )
    d.add_argument("--hf-revision", default="main")
    d.add_argument("--nli-model", default=NLI_MODEL)
    d.add_argument("--max-new-tokens", type=int, default=256)
    d.add_argument(
        "--dtype", choices=("float32", "bfloat16"), default="float32",
        help="the weights' dtype while decoding (bfloat16: a large model's longest prompts fit the GPU)",
    )  # fmt: skip
    d.add_argument("--limit", type=int, help="decode at most this many sets not yet in --out (default: all)")
    d.add_argument("--device", default="cpu")
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--summary", type=Path, required=True)
    args = p.parse_args(argv)
    if getattr(args, "limit", None) is not None and args.limit < 1:
        p.error("--limit must be at least 1")  # a negative slice would decode all but the last |N|
    return build(args) if args.command == "build" else run_decode(args)


if __name__ == "__main__":
    raise SystemExit(main())
