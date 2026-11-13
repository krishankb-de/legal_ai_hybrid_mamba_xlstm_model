"""The gold-passage sanity check: with retrieval assumed solved, does the model pick the right passage?

An item's **gold-relevant** chunks are its ``gold_chunk_ids`` that the index holds, else every chunk
whose ``citation_id`` or one of whose sentence ``labels`` is one of its ``gold`` citations
(``BGB §573b``), in the order of the item's citations, then of the index. The first is the item's
**gold passage**. A chunk is *related* to the gold when it shares a document with a gold-relevant
chunk or is linked to one through ``Chunk.copies`` (either direction; ``citation_metrics.gold_relation``,
which citation recall shares): a copy of the passage is no distractor.

``build_sets`` puts the gold passage among ``k`` distractors (k = 1, 4, 9), at a position drawn
uniformly from ``1..k + 1`` (seeded per item, tier and k), in two tiers:

* **easy**: chunks drawn uniformly at random from the index, neither related nor of the act of a
  gold-relevant chunk (``act_of(citation_id)``; a decision's ``act_of`` is its whole citation, so a
  decision is told apart by its document);
* **hard**: the retriever's best unrelated chunks for the question, in rank order (``fused_retriever``:
  BM25, fused by reciprocal rank fusion with the dense index when the index has ``dense/``; no
  filters, no reranking).

An item without a gold passage is skipped, and so is a tier and k with fewer than k distractors;
``build_sets`` counts both, per tier and k, beside the sets. Each prompt set is an ``AnswerRecord``
without system or answer (arm ``gold_<tier>_k<k>``, ``chunks`` in prompt order, ``gold_chunk_ids`` the
gold passage, whose position in ``chunks`` is the gold position), the input of any decoder. ``decode``
answers one greedily under the pointer FSM (no best-of-n, so the verifier never picks the passage for
the model) and fills it with the verified answer. ``score`` summarises decoded records per (system,
seed, arm): **selection accuracy** (the answer's first pointer cites the gold passage; a first pointer
that does not resolve selects nothing), **any gold cited**, **refusal rate** (the model's own
``<|unanswerable|>``), **verifier pass rate** (the verifier kept an answer) and selection accuracy by
gold position.

``as_is_sets`` makes the same undecoded records from SFT rows without retrieval: each row's own prompt
(arm ``as_is``), for the citation metrics and the truncation audit of the decoded records; ``score``
leaves them out, as they have no gold position.
"""

import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from pathlib import Path

import torch

from lexhybrid.data.sft_format import Cite, Quote, decode_answer, parse, prompt_ids
from lexhybrid.data.tokenizer import EOS_ID, N_PASSAGES
from lexhybrid.decoding.generate import generated_tokens, greedy_cached
from lexhybrid.decoding.pointer_constraints import PointerFSM, SpecialTokens
from lexhybrid.eval.citation_metrics import cited_passages, gold_relation
from lexhybrid.eval.records import AnswerRecord, chunk_dict
from lexhybrid.renderer.segments import (
    PointerSyntaxError,
    Unanswerable,
    parse_answer,
)
from lexhybrid.retrieval.bm25 import BM25Index
from lexhybrid.retrieval.chunking import Chunk, act_of
from lexhybrid.retrieval.dense import DenseIndex, Encoder
from lexhybrid.retrieval.fusion import reciprocal_rank_fusion
from lexhybrid.retrieval.index import CHUNKS_NAME
from lexhybrid.retrieval.rerank import CANDIDATES
from lexhybrid.verifier.nli import NLI
from lexhybrid.verifier.verifier import verify

K_VALUES = (1, 4, 9)
TIERS = ("easy", "hard")
Retrieve = Callable[[str], list[str]]  # question -> chunk ids, best first


def read_chunks(index_dir: Path) -> list[Chunk]:
    lines = (Path(index_dir) / CHUNKS_NAME).read_text(encoding="utf-8").split("\n")
    return [Chunk.from_json(line) for line in lines if line.strip()]


def fused_retriever(index_dir: Path, encode: Encoder | None = None, depth: int = CANDIDATES) -> Retrieve:
    """The index's ranking: BM25, fused with ``dense/`` when it exists and ``encode`` is given."""
    index_dir = Path(index_dir)
    bm25 = BM25Index.load(index_dir / "bm25")
    has_dense = encode is not None and (index_dir / "dense").exists()
    dense = DenseIndex.load(index_dir / "dense") if has_dense else None

    def retrieve(question: str) -> list[str]:
        rankings = [bm25.query(question, k=depth)]
        if dense is not None:
            rankings.append(dense.query(question, encode, k=depth))
        return [h.chunk_id for h in reciprocal_rank_fusion(rankings)]

    return retrieve


def _sets_of(item: dict, relevant: list[Chunk], chunks: Sequence[Chunk], by_id: dict[str, Chunk],
             retrieve: Retrieve, k_values: Sequence[int], seed: int, set_name: str) -> list[AnswerRecord]:  # fmt: skip
    gold = relevant[0]
    relation = gold_relation((c.doc_id for c in relevant), (d for c in relevant for d in c.copies))
    acts = {act_of(c.citation_id) for c in relevant}

    def related(c: Chunk) -> bool:
        return relation(c.doc_id, c.copies)

    ranked = [c for c in map(by_id.get, retrieve(item["question"])) if c is not None and not related(c)]

    def easy(k: int, rng: random.Random) -> list[Chunk]:
        picked, seen = [], set()  # uniform without replacement, without listing the whole index
        while len(picked) < k and len(seen) < len(chunks):
            i = rng.randrange(len(chunks))
            if i not in seen:
                seen.add(i)
                if not related(chunks[i]) and act_of(chunks[i].citation_id) not in acts:
                    picked.append(chunks[i])
        return picked

    out = []
    for tier in TIERS:
        for k in k_values:
            rng = random.Random(f"{seed}:{item['id']}:{tier}:{k}")
            distractors = ranked[:k] if tier == "hard" else easy(k, rng)
            if len(distractors) < k:
                continue
            at = rng.randrange(k + 1)
            prompt = [*distractors[:at], gold, *distractors[at:]]
            out.append(AnswerRecord(
                id=str(item["id"]), set=set_name, system="", arm=f"gold_{tier}_k{k}", question=item["question"],
                gold_answers=list(item.get("answers", [])), gold_chunk_ids=[gold.id],
                chunks=[chunk_dict(c) for c in prompt],
            ))  # fmt: skip
    return out


def gold_chunk_ids(items: Sequence[dict], chunks: Iterable[Chunk]) -> list[list[str]]:
    """Each item's gold-relevant chunk ids: its ``gold_chunk_ids`` the index holds, else every chunk whose
    ``citation_id`` or a sentence label is one of its ``gold`` citations (citation, then index order)."""
    chunks = list(chunks)
    known = {c.id for c in chunks}
    wanted = {g for item in items for g in item.get("gold", ())}
    by_citation: dict[str, list[str]] = {}
    for c in chunks:
        for citation in wanted.intersection((c.citation_id, *c.labels)):
            by_citation.setdefault(citation, []).append(c.id)
    out = []
    for item in items:
        held = [i for i in item.get("gold_chunk_ids", ()) if i in known]
        cited = [i for g in item.get("gold", ()) for i in by_citation.get(g, ())]
        out.append(list(dict.fromkeys(held or cited)))
    return out


def sft_item(row: dict) -> dict:
    """A QA item from an SFT row (``id``, ``chunk_ids`` in prompt order, ``text`` as ``sft_format.render``
    writes it): its question, and as ``gold_chunk_ids`` the chunks its answer quotes or cites, in order
    of citation; a refusal cites none. The passage pointers of the prompt are no citations."""
    ex = parse(row["text"])
    cited = [seg.passage for seg in ex.answer if isinstance(seg, Quote | Cite)]
    if any(k > len(row["chunk_ids"]) for k in cited):
        raise ValueError(f"row {row['id']!r} cites past its {len(row['chunk_ids'])} chunk ids")
    gold = dict.fromkeys(row["chunk_ids"][k - 1] for k in cited)
    return {"id": row["id"], "question": ex.question.strip(), "answers": [], "gold_chunk_ids": list(gold)}


def as_is_sets(rows: Iterable[dict], chunks: Iterable[Chunk], set_name: str = "") -> list[AnswerRecord]:
    """Each SFT row's own prompt as a record without system or answer (arm ``as_is``): its question,
    its passages in ``chunk_ids`` order as the index's ``chunks``, its ``kind``, and as gold the chunks
    its answer cites (``sft_item``). A chunk id the index lacks, or a chunk that no longer reads as the
    row's prompt passage, is an error: the prompt would not be the row's."""
    by_id = {c.id: c for c in chunks}
    out = []
    for row in rows:
        if missing := [i for i in row["chunk_ids"] if i not in by_id]:
            raise ValueError(f"row {row['id']!r}: chunk ids not in the index: {missing}")
        ex, prompt = parse(row["text"]), [by_id[i] for i in row["chunk_ids"]]
        if [c.passage() for c in prompt] != list(ex.passages):
            raise ValueError(
                f"row {row['id']!r}: its prompt's passages are not the index's {row['chunk_ids']}"
            )
        out.append(AnswerRecord(
            id=str(row["id"]), set=set_name, system="", arm="as_is", question=ex.question, kind=row["kind"],
            gold_chunk_ids=sft_item(row)["gold_chunk_ids"], chunks=[chunk_dict(c) for c in prompt],
        ))  # fmt: skip
    return out


def build_sets(items: Iterable[dict], chunks: Sequence[Chunk], retrieve: Retrieve,
               k_values: Sequence[int] = K_VALUES, seed: int = 0,
               set_name: str = "") -> tuple[list[AnswerRecord], dict]:  # fmt: skip
    """``(sets, counts)``: the prompt sets of every item with a gold passage (see the module docstring)
    and ``{"items", "with_gold", "sets": {arm: {"built", "skipped"}}}``, an item being skipped for an
    arm when that tier has fewer than k distractors for it. ``items`` are QA rows (``id``,
    ``question``, optional ``answers``, ``gold_chunk_ids`` or ``gold``), ``chunks`` the index's."""
    if max(k_values) + 1 > N_PASSAGES:
        raise ValueError(f"k up to {max(k_values)}: a prompt holds at most {N_PASSAGES} passages")
    items = list(items)
    by_id = {c.id: c for c in chunks}
    records, with_gold = [], 0
    for item, ids in zip(items, gold_chunk_ids(items, chunks), strict=True):
        if ids:
            with_gold += 1
            relevant = [by_id[i] for i in ids]
            records += _sets_of(item, relevant, chunks, by_id, retrieve, k_values, seed, set_name)
    arms = [f"gold_{tier}_k{k}" for tier in TIERS for k in k_values]
    built = {arm: sum(r.arm == arm for r in records) for arm in arms}
    sets = {arm: {"built": built[arm], "skipped": with_gold - built[arm]} for arm in arms}
    return records, {"items": len(items), "with_gold": with_gold, "sets": sets}


def decode(record: AnswerRecord, model, tok, nli: NLI, max_new_tokens: int = 256) -> AnswerRecord:
    """The record with the model's greedy, FSM-constrained answer and its verification filled in."""
    chunks = record.chunk_objects()
    device = next(model.parameters()).device
    prompt = torch.tensor([prompt_ids(record.question, [c.passage() for c in chunks], tok)], device=device)
    retrieved = {k: range(1, len(c.sentences) + 1) for k, c in enumerate(chunks, start=1)}
    fsm = PointerFSM(SpecialTokens.from_tokenizer(tok), retrieved, model.config.vocab_size, EOS_ID)
    row = greedy_cached(model, prompt, max_new_tokens, eos_token_id=EOS_ID,
                        logits_processor=fsm.processor(prompt.shape[1]))  # fmt: skip
    tokens = generated_tokens(row[0], prompt.shape[1], EOS_ID)
    raw = decode_answer(tokens, tok)
    result = verify(record.question, raw, chunks, nli)
    return replace(record, raw_answer=raw, rendered=result.rendered, refused=result.refused,
                   verifier=result.log_row(), new_tokens=len(tokens), max_new_tokens=max_new_tokens)  # fmt: skip


def outcome(record: AnswerRecord) -> dict:
    """What one decoded set shows: the gold position (1-based), whether the first pointer and any
    pointer cite the gold passage, the model's own refusal, and whether the verifier kept an answer."""
    try:
        segments = parse_answer(record.raw_answer)
    except PointerSyntaxError:
        segments = []
    passages = cited_passages(segments, record.chunk_objects())
    cited = [None if k is None else record.chunks[k]["id"] for k in passages]
    gold = set(record.gold_chunk_ids)
    return {
        "position": [c["id"] for c in record.chunks].index(record.gold_chunk_ids[0]) + 1,
        "selected": bool(cited) and cited[0] in gold,
        "any_gold": not gold.isdisjoint(cited),
        "model_refused": any(isinstance(s, Unanswerable) for s in segments),
        "passed": not record.refused,
    }


def _share(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows)


def score(records: Iterable[AnswerRecord]) -> list[dict]:
    """One summary per (system, seed, gold arm), in order of first appearance (see the module docstring)."""
    groups: dict[tuple, list[AnswerRecord]] = {}
    for r in records:
        if r.arm.startswith("gold_"):  # an as-is prompt has no gold position
            groups.setdefault((r.system, r.seed, r.arm), []).append(r)
    out = []
    for (system, seed, arm), rs in groups.items():
        rows = [outcome(r) for r in rs]
        _, tier, k = arm.split("_")
        by_position = {}
        for p in sorted({row["position"] for row in rows}):
            at = [row for row in rows if row["position"] == p]
            by_position[str(p)] = {"records": len(at), "selection_accuracy": _share(at, "selected")}
        out.append({
            "system": system, "seed": seed, "arm": arm, "tier": tier, "k": int(k[1:]),
            "sets": sorted({r.set for r in rs}), "records": len(rs),
            "selection_accuracy": _share(rows, "selected"), "any_gold_cited": _share(rows, "any_gold"),
            "refusal_rate": _share(rows, "model_refused"), "verifier_pass_rate": _share(rows, "passed"),
            "by_position": by_position,
        })  # fmt: skip
    return out
