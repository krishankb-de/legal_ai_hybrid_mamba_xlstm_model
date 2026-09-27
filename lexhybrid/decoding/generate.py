"""Uncached reference decoding.

``beam_search_uncached`` is ported from the reference's ``beam_search_decode`` (the dropped
report-generation script), minus the image prefix. It re-runs the full forward for every beam at
every step -- O(beam * L) per token -- and exists as the reference the cached beam search is tested
against. P2-N adds EOS termination and repetition control; P2-O/P hold the cached paths to it.
"""

import torch


@torch.no_grad()
def beam_search_uncached(
    model,
    input_ids: torch.Tensor,
    beam_size: int = 3,
    max_new_tokens: int = 100,
    length_penalty: float = 1.0,
) -> torch.Tensor:
    """Standard beam search over full forward passes, one sample at a time.

    Args:
        model: a ``HybridLanguageModel`` (any mixer pattern; no cache needed).
        input_ids: (1, L) prompt.
        beam_size: number of beams.
        max_new_tokens: tokens to generate (no EOS stop yet).
        length_penalty: score / length**length_penalty ranks hypotheses.

    Returns:
        (1, L + max_new_tokens) best hypothesis.
    """
    if input_ids.shape[0] != 1:
        raise ValueError(
            f"beam_search_uncached operates on one sample at a time (got batch size {input_ids.shape[0]})"
        )
    device = input_ids.device
    base_hidden = model.embeddings(input_ids)

    # Each beam: (hidden_states, token_ids, cumulative_log_prob)
    beams = [(base_hidden, input_ids, 0.0)]

    def _ranked(candidate):
        return candidate[2] / (candidate[1].shape[1] ** length_penalty)

    for _ in range(max_new_tokens):
        candidates = []
        for hidden_states, token_ids, score in beams:
            logits = model.forward(inputs_embeds=hidden_states, return_dict=True).logits
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)
            topk_logp, topk_idx = log_probs.topk(beam_size)
            for lp, idx in zip(topk_logp.tolist(), topk_idx.tolist()):
                next_token = torch.tensor([[idx]], device=device, dtype=token_ids.dtype)
                new_hidden = torch.cat([hidden_states, model.embeddings(next_token)], dim=1)
                new_tokens = torch.cat([token_ids, next_token], dim=1)
                candidates.append((new_hidden, new_tokens, score + lp))
        candidates.sort(key=_ranked, reverse=True)
        beams = candidates[:beam_size]

    best = max(beams, key=_ranked)
    return best[1]
