"""Decoding: uncached references and the shared selection logic of every decoder.

Each decoder is split into two parts:

* a **selection core** (``_token_loop`` for greedy and sampling, ``_beam_core`` for beam search)
  that owns every decision -- repetition penalty, EOS, finished rows, beam ranking, length
  normalisation, stopping;
* an **advance** function that turns "these rows now end in these tokens" into next-token logits.

The uncached references (``greedy``, ``sample``, ``beam_search``) advance by recomputing the full
forward pass over every row; the cached decoders (P2-O, P2-P) advance by one ``step`` through the
recurrent state and KV caches. Sharing the core means a cached decoder can differ from its reference
only through the logits, which the cache-equivalence tests bound -- never through a subtly
different rule. The reference project's cached beam was tested against an uncached one living in a
dropped report-generation script (defect 11); this module is the new reference.

Conventions (as in Hugging Face ``generate``):

* ``repetition_penalty`` (CTRL, Keskar et al. 2019): for every id already in a row's history
  (prompt and generated), a positive logit is divided by the penalty and a negative one multiplied.
* ``eos_token_id``: a row that emits it is finished; the EOS is kept, later positions are
  ``pad_token_id`` (default: the EOS id). Decoding stops early once every row is finished.
* Beam search keeps ``beam_size`` live hypotheses ranked by summed log-probability; a hypothesis
  that emits EOS within the top ``beam_size`` candidates is finished and scored
  ``sum_logprob / generated_length ** length_penalty``. It stops when ``beam_size`` hypotheses are
  finished and no live one can still beat the worst of them (HF's ``early_stopping=False``).
"""

from collections.abc import Callable

import torch

Advance = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
# (logits (B, V), seqs (B, T)) -> logits; e.g. ``PointerFSM.processor`` masks invalid pointers.
LogitsProcessor = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


# ---------------------------------------------------------------------------------------------
# logit processing
# ---------------------------------------------------------------------------------------------


def apply_repetition_penalty(logits: torch.Tensor, history: torch.Tensor, penalty: float) -> torch.Tensor:
    """CTRL repetition penalty over each row's ``history`` (B, T) of token ids; ``logits`` (B, V)."""
    if penalty == 1.0:
        return logits
    if penalty <= 0:
        raise ValueError(f"repetition_penalty must be > 0, got {penalty}")
    seen = torch.gather(logits, 1, history)
    seen = torch.where(seen > 0, seen / penalty, seen * penalty)
    return logits.scatter(1, history, seen)


def filter_logits(
    logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None
) -> torch.Tensor:
    """Temperature, then top-k, then nucleus (top-p) filtering; filtered entries become ``-inf``."""
    logits = logits / temperature
    if top_k is not None:
        logits = logits.masked_fill(logits < torch.topk(logits, top_k)[0][..., -1, None], float("-inf"))
    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        logits = logits.masked_fill(remove.scatter(1, sorted_indices, remove), float("-inf"))
    return logits


# ---------------------------------------------------------------------------------------------
# selection cores
# ---------------------------------------------------------------------------------------------


def _token_loop(
    first_logits: torch.Tensor,
    advance: Advance,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    choose: Callable[[torch.Tensor], torch.Tensor],
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
    repetition_penalty: float = 1.0,
    logits_processor: LogitsProcessor | None = None,
) -> torch.Tensor:
    """Greedy/sampling core: ``choose`` maps processed logits (B, V) to next ids (B,)."""
    ids = input_ids
    pad = eos_token_id if pad_token_id is None else pad_token_id
    finished = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
    logits = first_logits
    for step in range(max_new_tokens):
        processed = apply_repetition_penalty(logits.float(), ids, repetition_penalty)
        if logits_processor is not None:
            processed = logits_processor(processed, ids)
        nxt = choose(processed)
        if eos_token_id is not None:
            nxt = torch.where(finished, torch.full_like(nxt, pad), nxt)
        ids = torch.cat([ids, nxt.unsqueeze(-1)], dim=1)
        if eos_token_id is not None:
            finished = finished | (nxt == eos_token_id)
            if bool(finished.all()):
                break
        if step + 1 < max_new_tokens:
            logits = advance(torch.arange(ids.shape[0], device=ids.device), nxt, ids)
    return ids


def _beam_core(
    first_logits: torch.Tensor,
    advance: Advance,
    input_ids: torch.Tensor,
    beam_size: int,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    length_penalty: float = 1.0,
    repetition_penalty: float = 1.0,
    logits_processor: LogitsProcessor | None = None,
) -> torch.Tensor:
    """Beam-search core over ``beam_size`` rows; ``first_logits`` is (beam, V) for the prompt.

    ``advance(beam_idx, tokens, seqs)`` must return (beam, V) logits after row ``i`` has become
    ``seqs[i]`` = the old row ``beam_idx[i]`` extended by ``tokens[i]``.
    """
    device = input_ids.device
    seqs = input_ids.expand(beam_size, -1).contiguous()
    # Only beam 0 is live at first: the rest start at -inf, so the first expansion takes the true
    # top-k of one hypothesis rather than k copies of it.
    scores = torch.full((beam_size,), float("-inf"), device=device)
    scores[0] = 0.0
    finished: list[tuple[float, torch.Tensor]] = []
    logits = first_logits
    for step in range(max_new_tokens):
        gen_len = step + 1
        processed = apply_repetition_penalty(logits.float(), seqs, repetition_penalty)
        if logits_processor is not None:
            processed = logits_processor(processed, seqs)
        log_probs = torch.log_softmax(processed, dim=-1)
        total = scores.unsqueeze(-1) + log_probs  # (beam, V)
        vocab = total.shape[-1]
        top_scores, flat = total.view(-1).topk(min(2 * beam_size, total.numel()))
        beams, tokens, new_scores = [], [], []
        for rank, (s, idx) in enumerate(zip(top_scores.tolist(), flat.tolist())):
            if s == float("-inf"):
                break
            b, t = divmod(idx, vocab)
            if eos_token_id is not None and t == eos_token_id:
                if rank < beam_size:  # HF: only an EOS within the top beam_size candidates finishes
                    finished.append((s / gen_len**length_penalty, torch.cat([seqs[b], seqs.new_tensor([t])])))
                continue
            beams.append(b)
            tokens.append(t)
            new_scores.append(s)
            if len(beams) == beam_size:
                break
        if not beams:
            break
        while len(beams) < beam_size:  # tiny vocabularies: keep the batch shape with dead rows
            beams.append(beams[0])
            tokens.append(tokens[0])
            new_scores.append(float("-inf"))
        beam_idx = torch.tensor(beams, device=device)
        token_idx = torch.tensor(tokens, device=device, dtype=seqs.dtype)
        scores = torch.tensor(new_scores, device=device)
        seqs = torch.cat([seqs.index_select(0, beam_idx), token_idx.unsqueeze(-1)], dim=1)
        if len(finished) >= beam_size:
            finished = sorted(finished, key=lambda f: f[0], reverse=True)[:beam_size]
            best_live = max(new_scores) / gen_len**length_penalty
            if finished[-1][0] >= best_live:
                break
        if step + 1 < max_new_tokens:
            logits = advance(beam_idx, token_idx, seqs)
    gen_len = seqs.shape[1] - input_ids.shape[1]
    for s, row in zip(scores.tolist(), seqs):
        if s != float("-inf"):
            finished.append((s / max(gen_len, 1) ** length_penalty, row))
    best = max(finished, key=lambda f: f[0])
    return best[1].unsqueeze(0)


def _check_one_sample(input_ids: torch.Tensor, name: str) -> None:
    if input_ids.shape[0] != 1:
        raise ValueError(f"{name} operates on one sample at a time (got batch size {input_ids.shape[0]})")


# ---------------------------------------------------------------------------------------------
# uncached references
# ---------------------------------------------------------------------------------------------


def _recompute(model) -> Advance:
    def advance(beam_idx, tokens, seqs):
        return model(seqs, return_dict=True).logits[:, -1]

    return advance


@torch.no_grad()
def greedy(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
    repetition_penalty: float = 1.0,
    logits_processor: LogitsProcessor | None = None,
) -> torch.Tensor:
    """Uncached greedy decoding: the full forward is recomputed for every new token (O(L)/token)."""
    model.eval()
    first = model(input_ids, return_dict=True).logits[:, -1]
    return _token_loop(
        first,
        _recompute(model),
        input_ids,
        max_new_tokens,
        lambda lg: lg.argmax(-1),
        eos_token_id,
        pad_token_id,
        repetition_penalty,
        logits_processor,
    )


@torch.no_grad()
def sample(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
    repetition_penalty: float = 1.0,
    generator: torch.Generator | None = None,
    logits_processor: LogitsProcessor | None = None,
) -> torch.Tensor:
    """Uncached sampling with temperature / top-k / top-p; ``generator`` makes it reproducible."""
    model.eval()
    first = model(input_ids, return_dict=True).logits[:, -1]
    return _token_loop(
        first,
        _recompute(model),
        input_ids,
        max_new_tokens,
        _sampler(temperature, top_k, top_p, generator),
        eos_token_id,
        pad_token_id,
        repetition_penalty,
        logits_processor,
    )


def _sampler(temperature, top_k, top_p, generator):
    def choose(logits):
        probs = torch.softmax(filter_logits(logits, temperature, top_k, top_p), dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)

    return choose


@torch.no_grad()
def beam_search(
    model,
    input_ids: torch.Tensor,
    beam_size: int = 3,
    max_new_tokens: int = 100,
    eos_token_id: int | None = None,
    length_penalty: float = 1.0,
    repetition_penalty: float = 1.0,
    logits_processor: LogitsProcessor | None = None,
) -> torch.Tensor:
    """Uncached beam search, one sample at a time: every live beam is re-run in full each step.

    Returns the best hypothesis, (1, L + n) with ``n <= max_new_tokens`` (it ends at its EOS).
    """
    _check_one_sample(input_ids, "beam_search")
    model.eval()
    first = model(input_ids, return_dict=True).logits[:, -1].expand(beam_size, -1)
    return _beam_core(
        first,
        _recompute(model),
        input_ids,
        beam_size,
        max_new_tokens,
        eos_token_id,
        length_penalty,
        repetition_penalty,
        logits_processor,
    )


@torch.no_grad()
def beam_search_uncached(
    model,
    input_ids: torch.Tensor,
    beam_size: int = 3,
    max_new_tokens: int = 100,
    length_penalty: float = 1.0,
    eos_token_id: int | None = None,
    repetition_penalty: float = 1.0,
    logits_processor: LogitsProcessor | None = None,
) -> torch.Tensor:
    """The P1 name of ``beam_search`` (ported from the reference's ``beam_search_decode``)."""
    _check_one_sample(input_ids, "beam_search_uncached")
    return beam_search(
        model,
        input_ids,
        beam_size=beam_size,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        logits_processor=logits_processor,
    )


# ---------------------------------------------------------------------------------------------
# cached decoders (P2-O, P2-P): the same cores, advanced one step through the caches
# ---------------------------------------------------------------------------------------------


def _require_cache(model) -> None:
    if not model.supports_cached_decode():
        raise NotImplementedError(
            f"cached decode needs every mixer to implement step(); layer types are {model.get_layer_types()}"
        )


def _prefilled(model, input_ids: torch.Tensor, max_new_tokens: int, doc_ids: torch.Tensor | None = None):
    """Allocate caches for prompt + ``max_new_tokens`` and consume the prompt in one pass."""
    _require_cache(model)
    param = model.lm_head.weight
    caches = model.allocate_inference_cache(
        input_ids.shape[0],
        device=param.device,
        dtype=param.dtype,
        max_seq_len=input_ids.shape[1] + max_new_tokens,
    )
    return caches, model.prefill(model.embeddings(input_ids), caches, doc_ids=doc_ids)


def _step_advance(model, caches) -> Advance:
    """Advance by one cached step; rows keep their cache slot (no reordering)."""

    def advance(beam_idx, tokens, seqs):
        return model.step_logits(model.embeddings(tokens.unsqueeze(-1))[:, 0], caches)

    return advance


@torch.no_grad()
def greedy_cached(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
    repetition_penalty: float = 1.0,
    doc_ids: torch.Tensor | None = None,
    logits_processor: LogitsProcessor | None = None,
) -> torch.Tensor:
    """``greedy`` with the prompt prefilled once and O(1) recurrent steps (KV for attention).

    Rows finish independently: a finished row keeps stepping (so the batch stays aligned) but
    emits ``pad_token_id``. ``doc_ids`` lets a packed prompt continue each row's last document.
    """
    model.eval()
    caches, first = _prefilled(model, input_ids, max_new_tokens, doc_ids)
    return _token_loop(
        first,
        _step_advance(model, caches),
        input_ids,
        max_new_tokens,
        lambda lg: lg.argmax(-1),
        eos_token_id,
        pad_token_id,
        repetition_penalty,
        logits_processor,
    )


@torch.no_grad()
def sample_cached(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
    repetition_penalty: float = 1.0,
    generator: torch.Generator | None = None,
    doc_ids: torch.Tensor | None = None,
    logits_processor: LogitsProcessor | None = None,
) -> torch.Tensor:
    """``sample`` over the caches; same processing, same finished-row rule as ``greedy_cached``."""
    model.eval()
    caches, first = _prefilled(model, input_ids, max_new_tokens, doc_ids)
    return _token_loop(
        first,
        _step_advance(model, caches),
        input_ids,
        max_new_tokens,
        _sampler(temperature, top_k, top_p, generator),
        eos_token_id,
        pad_token_id,
        repetition_penalty,
        logits_processor,
    )


@torch.no_grad()
def beam_search_cached(
    model,
    input_ids: torch.Tensor,
    beam_size: int = 3,
    max_new_tokens: int = 100,
    eos_token_id: int | None = None,
    length_penalty: float = 1.0,
    repetition_penalty: float = 1.0,
    logits_processor: LogitsProcessor | None = None,
) -> torch.Tensor:
    """``beam_search`` over the caches (P2-P, defect 11), one sample at a time.

    The prompt is prefilled ONCE on one row and the caches are replicated to ``beam_size`` rows;
    every step reorders all of them (recurrent states, conv windows, KV caches, document starts)
    along the batch axis to follow the surviving beams, then advances one cached step. Finished
    hypotheses, their length normalisation and the stopping rule are ``_beam_core``'s, shared with
    the uncached reference.
    """
    _check_one_sample(input_ids, "beam_search_cached")
    model.eval()
    caches, first = _prefilled(model, input_ids, max_new_tokens)
    state = {
        "caches": model.reorder_cache(
            caches, torch.zeros(beam_size, dtype=torch.long, device=input_ids.device)
        )
    }

    def advance(beam_idx, tokens, seqs):
        state["caches"] = model.reorder_cache(state["caches"], beam_idx)
        return model.step_logits(model.embeddings(tokens.unsqueeze(-1))[:, 0], state["caches"])

    return _beam_core(
        first.expand(beam_size, -1),
        advance,
        input_ids,
        beam_size,
        max_new_tokens,
        eos_token_id,
        length_penalty,
        repetition_penalty,
        logits_processor,
    )


def generated_tokens(row: torch.Tensor, prompt_len: int, eos_token_id: int | None = None) -> list[int]:
    """The answer part of a decoded row: after the prompt, up to and including the first EOS."""
    out = row[prompt_len:].tolist()
    if eos_token_id is not None and eos_token_id in out:
        out = out[: out.index(eos_token_id) + 1]
    return out


@torch.no_grad()
def best_of_n(
    model,
    input_ids: torch.Tensor,
    n: int,
    scorer: Callable[[list[int]], float],
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | None = None,
    repetition_penalty: float = 1.0,
    logits_processor: LogitsProcessor | None = None,
    generator: torch.Generator | None = None,
    return_scores: bool = False,
):
    """Draw ``n`` cached samples for one prompt and return the one ``scorer`` rates highest (P2-R).

    The n samples decode as one batch of n rows over the caches. ``scorer`` receives each answer's
    generated tokens (prompt removed, cut after the first EOS) and returns a float; the P7 verifier
    (citation precision, quote fidelity, NLI) is the scorer the plan intends. Ties go to the
    earliest sample, so a constant scorer returns sample 0.

    Returns:
        (1, L + T) best row, or ``(best_row, scores)`` with ``return_scores``.
    """
    _check_one_sample(input_ids, "best_of_n")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    rows = sample_cached(
        model,
        input_ids.expand(n, -1).contiguous(),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=eos_token_id,
        repetition_penalty=repetition_penalty,
        generator=generator,
        logits_processor=logits_processor,
    )
    scores = [float(scorer(generated_tokens(row, input_ids.shape[1], eos_token_id))) for row in rows]
    best = max(range(n), key=lambda i: (scores[i], -i))
    return (rows[best : best + 1], scores) if return_scores else rows[best : best + 1]
