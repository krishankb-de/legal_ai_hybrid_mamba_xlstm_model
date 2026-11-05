"""Answer one legal question end to end and print the verified answer.

question -> BM25 (+ bge-m3 when the index has ``dense/``) -> fusion -> filters -> rerank to 3-8 passages -> the SFT prompt -> best-of-n
sampling under the pointer FSM, each sample scored by the verifier -> the best sample's verified,
rendered answer (with the notice) on stdout. Runs on CPU.

Best of n: a sample scores 1 when the verifier keeps it, plus its citation precision. The model's own
refusal (``<|unanswerable|>`` + EOS) scores 0 and would lose to any kept sample, so refusals are
counted first: when at least half of the n samples are that refusal, it is the answer.

    .venv/bin/python scripts/answer.py --index data/index/<version> --model-config hybrid_legal_base \\
        --checkpoint outputs/<run>/checkpoints/last.ckpt "Wann darf der Vermieter kündigen?"

Decoding is a ``Generator``, ``(question, chunks) -> Generation``: the raw pointer markup over the
prompt's passages (passage ``K`` is ``chunks[K - 1]``, 1..16 of them), ending in ``<|endoftext|>``
when the decode stopped on its own, with its token counts. ``generate`` bound by ``functools.partial``
is the hybrid model's; another model answers through the same verifier by being another generator.
"""

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path

import torch

from lexhybrid.config import load_model_config
from lexhybrid.data.sft_format import UNANSWERABLE, decode_answer, prompt_ids
from lexhybrid.data.tokenizer import EOS_ID, EOS_TOKEN, load_tokenizer
from lexhybrid.decoding import best_of_n
from lexhybrid.decoding.generate import generated_tokens
from lexhybrid.decoding.pointer_constraints import PointerFSM, SpecialTokens
from lexhybrid.retrieval.bm25 import BM25Index
from lexhybrid.retrieval.chunking import Chunk
from lexhybrid.retrieval.dense import CONFIG_NAME as DENSE_CONFIG
from lexhybrid.retrieval.dense import DENSE_MODEL, BgeM3Encoder, DenseIndex
from lexhybrid.retrieval.filters import filter_chunks
from lexhybrid.retrieval.fusion import reciprocal_rank_fusion
from lexhybrid.retrieval.index import CHUNKS_NAME
from lexhybrid.retrieval.rerank import CANDIDATES, RERANK_MODEL, CrossEncoderScorer, rerank_and_select
from lexhybrid.training.dpo_module import load_sft_model
from lexhybrid.verifier.nli import NLI_MODEL, MDebertaNLI
from lexhybrid.verifier.verifier import VerifierResult, verify


@dataclass(frozen=True)
class Generation:
    raw: str  # the pointer markup, EOS included when the decode produced it
    new_tokens: int  # answer tokens decoded
    max_new_tokens: int  # the decoding budget


Generator = Callable[[str, Sequence[Chunk]], Generation]
REFUSAL_MARKUP = UNANSWERABLE + EOS_TOKEN
NO_PASSAGES = Generation(REFUSAL_MARKUP, 0, 0)  # nothing to answer from: refuse, decode nothing


def fused_order(question: str, texts: Sequence[str]) -> list[float]:
    """The reranker switched off: every score ties, so the selection keeps the fused order."""
    return [0.0] * len(texts)


@lru_cache(maxsize=1)
def _index(index_dir: Path):
    """Chunks, BM25 and (when built) dense parts, read once per process: the ablation asks per question."""
    lines = (index_dir / CHUNKS_NAME).read_text(encoding="utf-8").split("\n")
    chunks = {c.id: c for c in map(Chunk.from_json, filter(str.strip, lines))}
    dense = None
    if (index_dir / "dense" / DENSE_CONFIG).exists():  # written last: an interrupted build is ignored
        dense = DenseIndex.load(index_dir / "dense")
        if set(dense.ids) != set(chunks):
            raise SystemExit(
                f"{index_dir}: dense/ was built from other chunks; rebuild it (build_index.py --kind dense)"
            )
    return chunks, BM25Index.load(index_dir / "bm25"), dense


def candidates(question: str, index_dir: Path, jurisdictions=None, valid_on=None, encode=None) -> list[Chunk]:
    """The fused BM25 (+ dense with ``encode``) top-50 rankings after the filters, best first."""
    chunks, bm25, dense = _index(index_dir)
    rankings = [bm25.query(question, k=CANDIDATES)]
    if encode is not None and dense is not None:
        rankings.append(dense.query(question, encode, k=CANDIDATES))
    fused = reciprocal_rank_fusion(rankings)
    return filter_chunks([chunks[h.chunk_id] for h in fused], jurisdictions, valid_on)


def retrieve(
    question: str, index_dir: Path, scorer, jurisdictions=None, valid_on=None, encode=None
) -> list[Chunk]:
    """The prompt's passages, best first."""
    return rerank_and_select(
        question, candidates(question, index_dir, jurisdictions, valid_on, encode), scorer
    )


def generate(question: str, chunks: Sequence[Chunk], model, tok, nli, n: int = 4, max_new_tokens: int = 256,
             temperature: float = 0.7, top_p: float = 0.95, seed: int = 0, ner=None) -> Generation:  # fmt: skip
    """The model's generator: best of ``n`` FSM-constrained samples, scored by the verifier; the
    model's own refusal when at least half of them are it (see the module docstring)."""
    device = next(model.parameters()).device
    prompt = torch.tensor([prompt_ids(question, [c.passage() for c in chunks], tok)], device=device)
    retrieved = {k: range(1, len(c.sentences) + 1) for k, c in enumerate(chunks, start=1)}
    fsm = PointerFSM(SpecialTokens.from_tokenizer(tok), retrieved, model.config.vocab_size, EOS_ID)

    refusals: list[list[int]] = []

    def score(tokens: list[int]) -> float:
        raw = decode_answer(tokens, tok)
        if raw == REFUSAL_MARKUP:
            refusals.append(tokens)
        if tokens[-1:] != [EOS_ID]:  # cut off by the budget: any finished sample beats it
            return -1.0
        r = verify(question, raw, chunks, nli, ner)
        return (0.0 if r.refused else 1.0) + (r.citation_precision or 0.0)

    best = best_of_n(
        model, prompt, n, score, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
        eos_token_id=EOS_ID, logits_processor=fsm.processor(prompt.shape[1]),
        generator=torch.Generator(device=device).manual_seed(seed),
    )  # fmt: skip
    tokens = refusals[0] if 2 * len(refusals) >= n else generated_tokens(best[0], prompt.shape[1], EOS_ID)
    return Generation(decode_answer(tokens, tok), len(tokens), max_new_tokens)


def respond(question: str, chunks: Sequence[Chunk], generator: Generator, nli, ner=None):
    """``(Generation, VerifierResult)``: the generator's answer over ``chunks``, verified."""
    gen = generator(question, chunks) if chunks else NO_PASSAGES
    return gen, verify(question, gen.raw, chunks, nli, ner)


def answer(question: str, chunks: Sequence[Chunk], model, tok, nli, n: int = 4, max_new_tokens: int = 256,
           temperature: float = 0.7, top_p: float = 0.95, seed: int = 0, ner=None) -> VerifierResult:  # fmt: skip
    """The verified answer: best of ``n`` FSM-constrained samples, scored by the verifier."""
    bound = partial(generate, model=model, tok=tok, nli=nli, n=n, max_new_tokens=max_new_tokens,
                    temperature=temperature, top_p=top_p, seed=seed, ner=ner)  # fmt: skip
    return respond(question, chunks, bound, nli, ner)[1]


def load_model(model_config: str | None, checkpoint: str, device: str, hf_model: str | None = None,
               hf_revision: str = "main"):  # fmt: skip
    config = None if hf_model else load_model_config(model_config)
    return load_sft_model(checkpoint, config, hf_model, hf_revision).to(device).eval()


def add_arguments(p: argparse.ArgumentParser) -> None:
    """The retrieval, model and decoding options, shared with scripts/eval_system_ablation.py."""
    p.add_argument(
        "--index", type=Path, required=True, help="an index directory written by scripts/build_index.py"
    )
    p.add_argument("--checkpoint", required=True)
    model = p.add_mutually_exclusive_group(required=True)
    model.add_argument("--model-config", help="a name under configs/model/ or a yaml path")
    model.add_argument(
        "--hf-model", help="the Hugging Face baseline's repository (its SFT checkpoint: --checkpoint)"
    )
    p.add_argument("--hf-revision", default="main")
    p.add_argument(
        "--reranker", default=RERANK_MODEL, help="cross-encoder model, or 'none' to keep the fused order"
    )
    p.add_argument("--dense-model", default=DENSE_MODEL, help="used when the index has dense/")
    p.add_argument("--nli-model", default=NLI_MODEL)
    p.add_argument("--jurisdiction", action="append", help="DE, AT, CH or EU; repeatable (default: all)")
    p.add_argument("--valid-on", help="YYYY-MM-DD: only passages in force on that day")
    p.add_argument("--n", type=int, default=4, help="samples for best-of-n")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")


def pipeline(args):
    """``(scorer, encode, nli, generator)`` from ``add_arguments``'s options; the models load here or on first use."""
    scorer = fused_order if args.reranker == "none" else CrossEncoderScorer(args.reranker, device=args.device)
    encode = BgeM3Encoder(args.dense_model, device=args.device)  # loads on first use only
    nli = MDebertaNLI(args.nli_model, device=args.device)
    generator = partial(
        generate, model=load_model(args.model_config, args.checkpoint, args.device, args.hf_model, args.hf_revision),
        tok=load_tokenizer(),
        nli=nli, n=args.n, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
        seed=args.seed,
    )  # fmt: skip
    return scorer, encode, nli, generator


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("question")
    add_arguments(p)
    args = p.parse_args(argv)
    if "<|" in args.question:
        p.error("the question may not contain special-token markup ('<|')")
    scorer, encode, nli, generator = pipeline(args)
    chunks = retrieve(args.question, args.index, scorer, args.jurisdiction, args.valid_on, encode)
    print(respond(args.question, chunks, generator, nli)[1].rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
