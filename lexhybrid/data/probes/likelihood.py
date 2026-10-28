"""Gold-answer likelihood: a continuous probe score that an exact-match floor cannot hide.

In the screening runs, statute recall and multi-hop scored 0.000 by exact match: a 220M model at
393M tokens does not reproduce an Absatz verbatim, nor answer the two-hop question exactly. The
teacher-forced negative log-likelihood of the reference answer after the prompt moves before exact
match does. Lower is better, and a model that has learned to copy from far back in the context
gets a lower score than one that guesses.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def gold_nll(model, prompt_ids: list[int], target_ids: list[int], device: str = "cpu") -> float:
    """Mean per-token NLL of ``target_ids`` following ``prompt_ids`` (natural log). The backbone
    runs over prompt + target, and the head runs only over the positions that predict the target,
    so an 8,192-token prompt never materialises (L, vocab) logits."""
    if not target_ids:
        raise ValueError("an empty target has no likelihood")
    ids = torch.tensor([prompt_ids + target_ids], device=device)
    residual, _ = model.backbone(ids)
    start = len(prompt_ids) - 1  # position t predicts token t + 1
    logits = model.head(residual[:, start : start + len(target_ids)]).float()
    return F.cross_entropy(logits[0], torch.tensor(target_ids, device=device)).item()
