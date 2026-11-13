"""Preference pairs for DPO/KTO from the SFT model's own samples, labelled by the verifier.

The input is a split ``scripts/build_sft_data.py`` wrote, ``<sft out>/train.jsonl`` for the training
pairs and ``dev.jsonl`` for the validation pairs: rows ``{"id", "question_id", "source", "kind",
"chunk_ids", "text"}`` that passed its evaluation-overlap and kind guards, the dev split's questions
held out. A raw generator file is refused (the split must be counted in the ``manifest.json`` beside
it). Every row gives one prompt, exactly its SFT prompt. The SFT checkpoint draws ``n`` samples under
the pointer FSM, and every distinct sample is verified against the prompt's own passages, what the
model saw (a row's ``chunk_ids`` may be empty). The prompt carries no sentence labels or hierarchy,
which change only the rendered text, never a verdict: each sentence is labelled with its passage's
citation. A prompt yields at most one pair:

* A sample is **good** when it ends in EOS and does what the item's ``kind`` asks for. For a
  should-refuse item (``unanswerable``, ``false_premise``, ``party``) that is the model's own refusal,
  exactly ``<|unanswerable|>``. For an answerable item it is an answer the verifier keeps whole, with
  at least one pointer into a passage the item's gold target cites: every claim passed (citation
  precision is not enough: it ignores a claim without a pointer, and a kept answer may have lost failed
  secondary claims), and a verbatim quote of an off-topic passage, which NLI always entails, does
  not count. Every other sample is bad: a refusal of an answerable item, an attempted answer to a
  should-refuse one, a malformed, unsupported or truncated answer.
* **chosen** is the good sample overlapping the gold target's pointers most (sentences first, then
  passages; ties: the first drawn), **rejected** the first bad sample drawn (an unbiased draw from the
  model's own failures). A prompt with no good or no bad sample yields no pair and is counted as skipped.
* **Coverage is kept balanced.** A pair's chosen answer is a refusal exactly when its prompt should
  be refused, so the share of refusal pairs would follow whichever kind the model gets *half* right,
  not the data: a model that already refuses well leaves few refusal pairs, and the answer pairs
  (many with a refusal as the rejected answer) would teach it never to refuse. So the
  over-represented side is subsampled (seeded) to the share of should-refuse prompts in the input,
  keeping as many pairs as possible: the share ends within one pair of the target, on the
  over-represented side. A side with no pairs at all empties the other.

The job is preemptible and long, so sampling is incremental: ``<out stem>.samples.jsonl`` gets one
record per prompt as it is verified (a first line with the sampling settings, then ``{"id", "samples":
[{"tokens", "markup", "verifier"}]}``). A restart skips the finished ids, and each prompt's seed depends
on its id alone, so it draws what an unbroken run would; other settings over the same file are refused.
Selection and balance run at the end, over that file.

Output: the pairs JSONL (``prompt_ids``, ``chosen``/``rejected`` markup and ``*_ids`` answer tokens,
item ids, ``kind``, both ``VerifierResult.log_row()``) and ``<out stem>.stats.json`` (prompts, pairs,
skipped prompts, dropped pairs, the refusal share and its target). A run with no pairs writes the
stats, says why and exits 1.

    .venv/bin/python scripts/build_dpo_pairs.py --sft-data data/sft/train.jsonl \\
        --model-config hybrid_legal_base --checkpoint outputs/<sft run>/checkpoints/last.ckpt \\
        --out data/dpo/<sft run>/train.jsonl --device cuda

``--hf-model Qwen/Qwen3-0.6B-Base`` takes the place of ``--model-config`` for the Hugging Face
baseline's SFT checkpoint: the same selection core (``_token_loop``), advanced through its KV cache.
"""

import argparse
import hashlib
import json
import math
import random
import sys
import zlib
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch

from lexhybrid.config import load_model_config
from lexhybrid.data.sft_format import (
    UNANSWERABLE,
    Cite,
    Passage,
    Quote,
    decode_answer,
    parse,
    prompt_ids,
)
from lexhybrid.data.tokenizer import EOS_ID, EOS_TOKEN, load_tokenizer
from lexhybrid.decoding.generate import generated_tokens, sample_cached
from lexhybrid.decoding.pointer_constraints import PointerFSM, SpecialTokens
from lexhybrid.eval.records import KINDS, REFUSE_KINDS
from lexhybrid.renderer.segments import CitePointer, QuotePointer, parse_answer
from lexhybrid.retrieval.chunking import Chunk
from lexhybrid.training.dpo_module import load_sft_model
from lexhybrid.utils.jsonl import read_complete_lines
from lexhybrid.verifier.nli import NLI_MODEL, MDebertaNLI
from lexhybrid.verifier.verifier import verify

REFUSAL_MARKUP = UNANSWERABLE + EOS_TOKEN
Pointers = tuple[frozenset[int], frozenset[tuple[int, int]]]  # passages, (passage, sentence)s


# -- pair selection (pure): samples are {"tokens", "markup", "verifier": VerifierResult.log_row()} --


def gold_pointers(item: dict) -> Pointers:
    """What the item's gold target points at."""
    ex = parse(item["text"])
    cited = {s.passage for s in ex.answer if isinstance(s, Quote | Cite)}
    # The index keeps every copy of a text, so a prompt can hold a gold passage twice: the copy answers as well.
    same = {g: [k for k, p in enumerate(ex.passages, start=1) if p.sentences == ex.passages[g - 1].sentences]
            for g in cited}  # fmt: skip
    return (
        frozenset(k for g in cited for k in same[g]),
        frozenset((k, s.sentence) for s in ex.answer if isinstance(s, Quote) for k in same[s.passage]),
    )


def answer_pointers(markup: str) -> Pointers:
    """What a parsable answer points at."""
    found = [s for s in parse_answer(markup) if isinstance(s, QuotePointer | CitePointer)]
    return frozenset(s.passage for s in found), frozenset((s.passage, j) for s in found for j in s.sentences)


def is_good(sample: dict, should_refuse: bool, gold: Pointers) -> bool:
    """The sample finished and behaves as the item's kind asks (module docstring)."""
    if should_refuse:
        return sample["markup"] == REFUSAL_MARKUP
    verdict = sample["verifier"]
    if not sample["markup"].endswith(EOS_TOKEN) or verdict["refused"]:
        return False
    # a claim without a pointer never passes (the verifier's "unsupported")
    return all(c["passed"] for c in verdict["claims"]) and bool(
        answer_pointers(sample["markup"])[0] & gold[0]
    )


def select_pair(samples: Sequence[dict], should_refuse: bool, gold: Pointers) -> tuple[dict, dict] | None:
    """``(chosen, rejected)`` from one prompt's verified samples in draw order, or None."""
    good = [s for s in samples if is_good(s, should_refuse, gold)]
    bad = [s for s in samples if not is_good(s, should_refuse, gold)]
    if not good or not bad:
        return None

    def overlap(sample: dict) -> tuple[int, int]:
        passages, sentences = answer_pointers(sample["markup"])
        return len(sentences & gold[1]), len(passages & gold[0])

    return max(good, key=overlap), bad[0]  # max keeps the first of equals


def balance(pairs: Sequence[dict], refuse_prompts: int, prompts: int, seed: int = 0) -> list[dict]:
    """The pairs with the over-represented side subsampled to the share ``refuse_prompts / prompts``
    of refusal pairs (rounded towards keeping pairs); order is kept."""
    answer_prompts = prompts - refuse_prompts
    if refuse_prompts == 0 or answer_prompts == 0:  # one side only, by construction
        return list(pairs)
    refuse = [i for i, p in enumerate(pairs) if p["kind"] in REFUSE_KINDS]
    answer = [i for i, p in enumerate(pairs) if p["kind"] not in REFUSE_KINDS]
    rng = random.Random(seed)
    # integer form of len(refuse) / len(pairs) > refuse_prompts / prompts
    if len(refuse) * answer_prompts > len(answer) * refuse_prompts:
        refuse = rng.sample(refuse, math.ceil(len(answer) * refuse_prompts / answer_prompts))
    else:
        answer = rng.sample(answer, math.ceil(len(refuse) * answer_prompts / refuse_prompts))
    kept = {*refuse, *answer}
    return [p for i, p in enumerate(pairs) if i in kept]


def pair_record(item: dict, prompt: Sequence[int], chosen: dict, rejected: dict) -> dict:
    return {
        "id": item["id"], "question_id": item.get("question_id"), "source": item.get("source"),
        "kind": item["kind"], "chunk_ids": item.get("chunk_ids", []), "prompt_ids": list(prompt),
        "chosen": chosen["markup"], "chosen_ids": list(chosen["tokens"]), "chosen_verifier": chosen["verifier"],
        "rejected": rejected["markup"], "rejected_ids": list(rejected["tokens"]),
        "rejected_verifier": rejected["verifier"],
    }  # fmt: skip


def build_pairs(drawn: Iterable[tuple[dict, Sequence[int], Sequence[dict]]], seed: int = 0):
    """``(pairs, stats)`` from ``(item, prompt ids, verified samples)`` per prompt."""
    pairs, prompts, refuse_prompts, samples, no_good, no_bad = [], 0, 0, 0, 0, 0
    for item, prompt, verified in drawn:
        if item["kind"] not in KINDS:
            raise ValueError(f"{item['id']}: kind {item['kind']!r} is not one of {KINDS}")
        should_refuse, gold = item["kind"] in REFUSE_KINDS, gold_pointers(item)
        prompts += 1
        refuse_prompts += should_refuse
        samples += len(verified)
        pair = select_pair(verified, should_refuse, gold)
        if pair is None:
            if any(is_good(s, should_refuse, gold) for s in verified):
                no_bad += 1
            else:
                no_good += 1
            continue
        pairs.append(pair_record(item, prompt, *pair))
    kept = balance(pairs, refuse_prompts, prompts, seed)
    refusals_before = sum(p["kind"] in REFUSE_KINDS for p in pairs)
    refusals = sum(p["kind"] in REFUSE_KINDS for p in kept)
    stats = {
        "prompts": prompts, "should_refuse_prompts": refuse_prompts, "distinct_samples": samples,
        "skipped_no_good": no_good, "skipped_no_bad": no_bad,
        "refusal_pairs_before_balance": refusals_before, "answer_pairs_before_balance": len(pairs) - refusals_before,
        "dropped_for_balance": len(pairs) - len(kept), "pairs": len(kept), "refusal_pairs": refusals,
        "refusal_share": refusals / len(kept) if kept else None,
        "target_refusal_share": refuse_prompts / prompts if prompts else None,
    }  # fmt: skip
    return kept, stats


def no_pairs_reason(stats: dict) -> str:
    """Why a run produced no pairs, from its stats."""
    refusals, answers = stats["refusal_pairs_before_balance"], stats["answer_pairs_before_balance"]
    if not refusals and not answers:
        return (
            f"no prompt gave a pair: of {stats['prompts']} prompts, {stats['skipped_no_good']} had no good "
            f"sample and {stats['skipped_no_bad']} no bad sample"
        )
    missing, others = ("refusal", answers) if not refusals else ("answer", refusals)
    return (
        f"balancing emptied the set: there are no {missing} pairs, so the {others} others were dropped "
        "(sample more: --n, --temperature)"
    )


# -- sampling (model-driven) --------------------------------------------------------------------


def passage_chunk(item: dict, k: int, passage: Passage) -> Chunk:
    """Passage ``k`` of an item's prompt as the verifier's ``Chunk``."""
    return Chunk(
        id=f"{item['id']}#{k}", doc_id=item["id"], source=item.get("source") or "", jurisdiction="",
        doc_type="", citation_id=passage.citation_id, hierarchy=(), sentences=passage.sentences,
        labels=(passage.citation_id,) * len(passage.sentences), valid_from=None, valid_to=None,
        licence="", commercial_safe=False, research_only=False, url="",
    )  # fmt: skip


def draw_samples(item: dict, model, tok, nli, n: int, max_new_tokens: int, temperature: float, top_p: float,
                 seed: int) -> list[dict]:  # fmt: skip
    """One prompt's distinct verified samples in draw order (as ``answer.answer`` decodes); the
    generator's seed depends on the item alone, so a restart draws what an unbroken run would."""
    ex = parse(item["text"])
    chunks = [passage_chunk(item, k, p) for k, p in enumerate(ex.passages, start=1)]
    device = next(model.parameters()).device
    prompt = prompt_ids(ex.question, ex.passages, tok)
    retrieved = {k: range(1, len(c.sentences) + 1) for k, c in enumerate(chunks, start=1)}
    fsm = PointerFSM(SpecialTokens.from_tokenizer(tok), retrieved, model.config.vocab_size, EOS_ID)
    rows = sample_cached(
        model, torch.tensor([prompt] * n, device=device), max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p, eos_token_id=EOS_ID,
        generator=torch.Generator(device=device).manual_seed(zlib.crc32(f"{seed}:{item['id']}".encode())),
        logits_processor=fsm.processor(len(prompt)),
    )  # fmt: skip
    samples: dict[tuple[int, ...], dict] = {}
    for row in rows:
        tokens = tuple(generated_tokens(row, len(prompt), EOS_ID))
        if tokens not in samples:
            markup = decode_answer(tokens, tok)
            verdict = verify(ex.question, markup, chunks, nli).log_row()
            samples[tokens] = {"tokens": list(tokens), "markup": markup, "verifier": verdict}
    return list(samples.values())


# -- files --------------------------------------------------------------------------------------


def read_items(path: Path) -> list[dict]:
    """The rows of an SFT builder's split; anything the builder's manifest does not count is refused."""
    manifest = path.with_name("manifest.json")
    saved = json.loads(manifest.read_text()).get("synthetic_rows_sha256", {}) if manifest.exists() else {}
    data = path.read_bytes()
    if saved.get(path.name) != hashlib.sha256(data).hexdigest():
        raise ValueError(
            f"{path}: {manifest} does not hold these rows: give build_sft_data.py's train.jsonl or "
            "dev.jsonl as it wrote them, not a generator file (no evaluation guard ran on it)"
        )
    items = [json.loads(line) for line in data.decode("utf-8").split("\n") if line.strip()]
    if len({item["id"] for item in items}) != len(items):
        raise ValueError(f"{path}: duplicate ids")
    return items


def read_samples(path: Path, params: dict) -> dict[str, dict]:
    """The records a previous attempt finished, by id. Every record ends in a newline, so one a kill
    cut short is cut off here; a file written with other sampling settings is refused."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    lines = [json.loads(line) for line in read_complete_lines(path)]
    if not lines:
        path.write_text(json.dumps({"params": params}, ensure_ascii=False) + "\n", encoding="utf-8")
        return {}
    if lines[0].get("params") != params:
        raise ValueError(
            f"{path} was sampled with {lines[0].get('params')}, not {params}: give another --out"
        )
    return {r["id"]: r for r in lines[1:]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sft-data", type=Path, required=True, help="<sft out>/train.jsonl or dev.jsonl")
    p.add_argument("--checkpoint", required=True, help="the SFT checkpoint")
    p.add_argument("--model-config", help="the hybrid: a name under configs/model/ or a yaml path")
    p.add_argument("--hf-model", help="the Hugging Face baseline: the model the SFT run trained")
    p.add_argument("--hf-revision", default="main")
    p.add_argument("--out", type=Path, required=True, help="pairs JSONL; <stem>.samples.jsonl, <stem>.stats.json beside it")  # fmt: skip
    p.add_argument("--nli-model", default=NLI_MODEL)
    p.add_argument("--n", type=int, default=8, help="samples per prompt")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0, help="seeds the samples and the balancing subsample")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    if (args.model_config is None) == (args.hf_model is None):
        p.error("give exactly one of --model-config and --hf-model")

    items = read_items(args.sft_data)
    samples_path = args.out.with_name(args.out.stem + ".samples.jsonl")
    ckpt = Path(args.checkpoint).stat()
    params = {  # paths alone would let samples of an older checkpoint or older rows pass as these
        "sft_data": str(args.sft_data), "sft_data_sha256": hashlib.sha256(args.sft_data.read_bytes()).hexdigest(),
        "checkpoint": args.checkpoint, "checkpoint_stat": [ckpt.st_size, ckpt.st_mtime_ns],
        "model_config": args.model_config,
        "hf_model": args.hf_model, "hf_revision": args.hf_revision, "nli_model": args.nli_model, "n": args.n,
        "max_new_tokens": args.max_new_tokens, "temperature": args.temperature, "top_p": args.top_p,
        "seed": args.seed,
    }  # fmt: skip
    finished = read_samples(samples_path, params)
    todo = [item for item in items if item["id"] not in finished]
    tok = load_tokenizer()
    print(f"{len(items)} prompts, {len(items) - len(todo)} sampled by an earlier attempt", flush=True)
    if todo:
        config = load_model_config(args.model_config) if args.model_config else None
        model = (
            load_sft_model(args.checkpoint, config, args.hf_model, args.hf_revision).to(args.device).eval()
        )
        nli = MDebertaNLI(args.nli_model, device=args.device)
        with open(samples_path, "a", encoding="utf-8") as f:
            for k, item in enumerate(todo, start=1):
                samples = draw_samples(item, model, tok, nli, args.n, args.max_new_tokens, args.temperature,
                                       args.top_p, args.seed)  # fmt: skip
                f.write(json.dumps({"id": item["id"], "samples": samples}, ensure_ascii=False) + "\n")
                f.flush()
                if k % 100 == 0:
                    print(f"{k}/{len(todo)} prompts sampled", flush=True)
        finished = read_samples(samples_path, params)

    def drawn():
        for item in items:
            ex = parse(item["text"])
            yield item, prompt_ids(ex.question, ex.passages, tok), finished[item["id"]]["samples"]

    pairs, stats = build_pairs(drawn(), seed=args.seed)
    stats_path = args.out.with_name(args.out.stem + ".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats), flush=True)
    if not pairs:
        print(f"FATAL: {args.sft_data}: {no_pairs_reason(stats)}", file=sys.stderr, flush=True)
        return 1
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pairs), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
