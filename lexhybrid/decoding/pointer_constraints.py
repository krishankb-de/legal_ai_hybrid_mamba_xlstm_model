"""Pointer-constrained decoding (plan P2-Q, decision 7).

The legal model never generates statute text: it emits *pointers* into the numbered retrieved
context, ``<|q|><|c3|><|s2|>`` ("quote chunk 3, sentence 2") or ``<|cite|><|c3|><|s2|>``, and the
renderer (P7) replaces each pointer with the exact sentence or its label. ``PointerFSM`` makes an
invalid pointer impossible to decode rather than something the verifier has to catch:

    FREE --(<|q|> or <|cite|>)--> MARKER --(<|cK|>, K retrieved)--> CHUNK(K)
    CHUNK(K) --(<|sJ|>, J a retrieved sentence of chunk K)--> SENTENCE(K)
    SENTENCE(K) --(<|sJ'|> of chunk K)--> SENTENCE(K)       a span of sentences
    SENTENCE(K) --(ordinary token, EOS, marker)--> as from FREE

* a chunk id may only follow a marker, and only a retrieved chunk's id;
* a sentence id may only follow its chunk id (or another sentence of that chunk);
* inside a pointer (MARKER, CHUNK) nothing else is allowed, so a pointer is never left dangling;
* prompt-structure tokens (``<|passage|>``, ``<|/passage|>``, ``<|question|>``, ``<|answer|>``)
  are never generated; ``<|unanswerable|>`` is an ordinary answer token;
* with nothing retrieved, the markers themselves are masked.

The machine is stateless across calls: ``mask_logits`` replays each row's generated tokens, so it
follows beam reordering without being told (a pointer is a handful of tokens).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

N_CHUNKS = 16
N_SENTENCES = 64

FREE, MARKER, CHUNK, SENTENCE = "free", "marker", "chunk", "sentence"


@dataclass(frozen=True)
class SpecialTokens:
    """Ids of the 87 SFT-only special tokens of decision 7 (never used in KD)."""

    q: int
    cite: int
    chunks: tuple[int, ...]  # <|c1|> .. <|c16|>
    sentences: tuple[int, ...]  # <|s1|> .. <|s64|>
    unanswerable: int
    passage_open: int
    passage_close: int
    question: int
    answer: int

    @staticmethod
    def names() -> list[str]:
        """The token strings, in registry order (the order ``from_offset`` assigns ids)."""
        return (
            ["<|q|>", "<|cite|>"]
            + [f"<|c{k}|>" for k in range(1, N_CHUNKS + 1)]
            + [f"<|s{j}|>" for j in range(1, N_SENTENCES + 1)]
            + ["<|unanswerable|>", "<|passage|>", "<|/passage|>", "<|question|>", "<|answer|>"]
        )

    @classmethod
    def from_ids(cls, ids: Sequence[int]) -> "SpecialTokens":
        ids = list(ids)
        if len(ids) != len(cls.names()) or len(set(ids)) != len(ids):
            raise ValueError(f"need {len(cls.names())} distinct ids, got {len(ids)}")
        c0, s0 = 2, 2 + N_CHUNKS
        tail = ids[s0 + N_SENTENCES :]
        return cls(ids[0], ids[1], tuple(ids[c0:s0]), tuple(ids[s0 : s0 + N_SENTENCES]), *tail)

    @classmethod
    def from_offset(cls, first_id: int) -> "SpecialTokens":
        """Consecutive ids from ``first_id`` -- tests, and any vocabulary that appends them in order."""
        return cls.from_ids(range(first_id, first_id + len(cls.names())))

    @classmethod
    def from_tokenizer(cls, tokenizer) -> "SpecialTokens":
        """Ids from a tokenizer that has the tokens added (P3-C adds them to the Qwen3 tokenizer)."""
        ids = tokenizer.convert_tokens_to_ids(cls.names())
        unknown = [n for n, i in zip(cls.names(), ids) if i is None or i == tokenizer.unk_token_id]
        if unknown:
            raise ValueError(f"tokenizer lacks the pointer tokens {unknown[:3]}...")
        return cls.from_ids(ids)

    def all_ids(self) -> list[int]:
        return [
            self.q,
            self.cite,
            *self.chunks,
            *self.sentences,
            self.unanswerable,
            self.passage_open,
            self.passage_close,
            self.question,
            self.answer,
        ]

    def prompt_only(self) -> tuple[int, ...]:
        return (self.passage_open, self.passage_close, self.question, self.answer)

    def chunk_number(self, token: int) -> int | None:
        """K for ``<|cK|>``, else None."""
        return self.chunks.index(token) + 1 if token in self.chunks else None

    def sentence_number(self, token: int) -> int | None:
        """J for ``<|sJ|>``, else None."""
        return self.sentences.index(token) + 1 if token in self.sentences else None


class PointerFSM:
    """The pointer grammar for one retrieved context.

    Args:
        specials: the special-token ids.
        retrieved: ``{chunk K: [sentence J, ...]}`` (1-based, as numbered in the prompt).
        vocab_size: the model's vocabulary (logit width).
        eos_token_id: allowed wherever a pointer is complete.
    """

    def __init__(
        self,
        specials: SpecialTokens,
        retrieved: Mapping[int, Sequence[int]],
        vocab_size: int,
        eos_token_id: int | None = None,
    ):
        for k, sentences in retrieved.items():
            if not 1 <= k <= N_CHUNKS or any(not 1 <= j <= N_SENTENCES for j in sentences):
                raise ValueError(
                    f"chunk {k} / sentences {list(sentences)} outside c1..c{N_CHUNKS}, s1..s{N_SENTENCES}"
                )
        self.specials = specials
        self.retrieved = {int(k): tuple(sorted(set(int(j) for j in v))) for k, v in retrieved.items() if v}
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self._masks: dict[tuple[str, int | None], torch.Tensor] = {}

    # -- transitions ----------------------------------------------------------------------------

    def step(self, state: tuple[str, int | None], token: int) -> tuple[str, int | None]:
        """The state after ``token``; raises ``ValueError`` if the grammar forbids it."""
        if not self.allowed(state)[token]:
            raise ValueError(f"token {token} is not allowed in state {state}")
        kind, chunk = state
        sp = self.specials
        if token in (sp.q, sp.cite):
            return (MARKER, None)
        if kind == MARKER:
            return (CHUNK, sp.chunk_number(token))
        if kind in (CHUNK, SENTENCE) and sp.sentence_number(token) is not None:
            return (SENTENCE, chunk)
        return (FREE, None)

    def replay(self, tokens: Sequence[int]) -> tuple[str, int | None]:
        state: tuple[str, int | None] = (FREE, None)
        for token in tokens:
            state = self.step(state, int(token))
        return state

    # -- masks ------------------------------------------------------------------------------------

    def allowed(self, state: tuple[str, int | None]) -> torch.Tensor:
        """Boolean (vocab,) mask of the tokens the grammar allows next."""
        if state not in self._masks:
            self._masks[state] = self._build(state)
        return self._masks[state]

    def _build(self, state: tuple[str, int | None]) -> torch.Tensor:
        kind, chunk = state
        sp = self.specials
        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        if kind == MARKER:
            mask[[sp.chunks[k - 1] for k in self.retrieved]] = True
            return mask
        if kind == CHUNK:
            mask[[sp.sentences[j - 1] for j in self.retrieved[chunk]]] = True
            return mask
        # FREE or SENTENCE: ordinary text, EOS, <|unanswerable|>, and a new pointer if possible.
        mask[:] = True
        mask[list(sp.chunks) + list(sp.sentences) + list(sp.prompt_only())] = False
        if not self.retrieved:
            mask[[sp.q, sp.cite]] = False
        if kind == SENTENCE:
            mask[[sp.sentences[j - 1] for j in self.retrieved[chunk]]] = True
        return mask

    def mask_logits(self, logits: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
        """Set every forbidden logit to ``-inf``, row by row; ``generated`` (B, T) are the tokens
        decoded so far (not the prompt)."""
        masks = torch.stack([self.allowed(self.replay(row.tolist())) for row in generated]).to(logits.device)
        return logits.masked_fill(~masks, float("-inf"))

    def processor(self, prompt_len: int):
        """A ``logits_processor(logits, seqs)`` for the decoders in ``lexhybrid.decoding.generate``."""

        def process(logits: torch.Tensor, seqs: torch.Tensor) -> torch.Tensor:
            return self.mask_logits(logits, seqs[:, prompt_len:])

        return process


def validate_pointers(
    tokens: Sequence[int], specials: SpecialTokens, retrieved: Mapping[int, Sequence[int]]
) -> list[str]:
    """Problems with the pointers in a decoded sequence; empty when every pointer is valid.

    Independent of ``PointerFSM`` on purpose: the test checks the machine against it.
    """
    problems, i, tokens = [], 0, [int(t) for t in tokens]
    markers = (specials.q, specials.cite)
    while i < len(tokens):
        t = tokens[i]
        k = specials.chunk_number(t)
        if specials.sentence_number(t) is not None:
            problems.append(f"sentence pointer outside a pointer at {i}")
        elif k is not None:
            if i == 0 or tokens[i - 1] not in markers:
                problems.append(f"chunk pointer c{k} without a marker at {i}")
            if k not in retrieved:
                problems.append(f"chunk c{k} was not retrieved (at {i})")
            j = i + 1
            if j >= len(tokens) or specials.sentence_number(tokens[j]) is None:
                problems.append(f"chunk pointer c{k} without a sentence at {i}")
            while j < len(tokens) and specials.sentence_number(tokens[j]) is not None:
                s = specials.sentence_number(tokens[j])
                if s not in retrieved.get(k, ()):
                    problems.append(f"sentence s{s} is not in chunk c{k} (at {j})")
                j += 1
            i = j
            continue
        elif t in markers and (i + 1 >= len(tokens) or specials.chunk_number(tokens[i + 1]) is None):
            problems.append(f"marker without a chunk pointer at {i}")
        i += 1
    return problems
