"""Online logit distillation from an open teacher (plan P2-T, decision 9).

    loss = (1 - alpha) * CE(student, labels) + alpha * T^2 * KL(softmax(teacher / T) || softmax(student / T))

with alpha = 0.5 and T = 2 by default. Both terms are means over the same supervised positions: the
document-boundary label mask of P2-L applies to the KL too, because at a document's last token
neither model should be graded on the next document's first.

The reference distilled mean-pooled hidden-state cosine; logit KL at a 151,936-token vocabulary is
new code with two memory rules:

* **Never materialise (B, L, V).** ``slab_ce_kl`` walks the sequence in slabs of ``slab`` positions,
  each under activation checkpointing: the student's and the teacher's slab logits exist only
  inside the slab and are recomputed for the backward pass. The teacher contributes hidden states
  (``teacher_forward_packed``) and its own head, applied per slab under ``no_grad``.
* **The teacher sees documents, not rows.** A packed row goes to the teacher with per-document
  ``position_ids`` and a block-diagonal causal 4-D mask, so document i+1's teacher distribution is
  exactly the one it would have alone (``tests/test_training.py::test_teacher_position_ids_isolation``).
  Position ids alone are not enough with SDPA: attention would still cross the boundary.
"""

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from lexhybrid.layers.attention_block import positions_within_documents


def teacher_inputs_packed(input_ids: torch.Tensor, doc_ids: torch.Tensor | None):
    """``(position_ids, attention_mask)`` for a Hugging Face decoder on packed rows.

    ``attention_mask`` is (B, 1, L, L) bool, True where query i may attend key j (same document,
    j <= i); None without ``doc_ids`` (plain causal attention, positions 0..L-1).
    """
    if doc_ids is None:
        return None, None
    seq_len = input_ids.shape[1]
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device).tril()
    same = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)
    return positions_within_documents(doc_ids), (same & causal).unsqueeze(1)


@torch.no_grad()
def teacher_forward_packed(
    teacher, input_ids: torch.Tensor, doc_ids: torch.Tensor | None = None
) -> torch.Tensor:
    """The teacher's final hidden states (after its final norm), (B, L, D_teacher), without logits.

    ``teacher`` is a Hugging Face causal LM (``Qwen3ForCausalLM`` for the plan's
    ``Qwen/Qwen3-1.7B-Base``); its decoder is called directly so the (B, L, V) logits are never
    formed here -- ``slab_ce_kl`` applies ``teacher.get_output_embeddings()`` slab by slab.
    """
    position_ids, mask = teacher_inputs_packed(input_ids, doc_ids)
    decoder = teacher.get_decoder() if hasattr(teacher, "get_decoder") else teacher.model
    out = decoder(input_ids=input_ids, attention_mask=mask, position_ids=position_ids, use_cache=False)
    return out.last_hidden_state


def _slab_terms(h, teacher_h, targets, head, teacher_head, temperature):
    """Summed CE and summed masked KL for one slab (recomputed in the backward pass)."""
    student = head(h).float()
    with torch.no_grad():
        teacher = teacher_head(teacher_h).float()
    if teacher.shape[-1] != student.shape[-1]:
        raise ValueError(f"teacher vocabulary {teacher.shape[-1]} != student vocabulary {student.shape[-1]}")
    vocab = student.shape[-1]
    ce = F.cross_entropy(student.reshape(-1, vocab), targets.reshape(-1), ignore_index=-100, reduction="sum")
    log_q = F.log_softmax(student / temperature, dim=-1)
    log_p = F.log_softmax(teacher / temperature, dim=-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(-1)  # (B, S)
    kl = (kl * (targets != -100)).sum()
    return ce, kl


def slab_ce_kl(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    teacher_hidden: torch.Tensor | None,
    lm_head,
    teacher_lm_head=None,
    alpha: float = 0.5,
    T: float = 2.0,
    slab: int = 512,
    checkpoint: bool = True,
) -> dict:
    """Slab-wise ``(1 - alpha) * CE + alpha * T^2 * KL`` without (B, L, V) logits.

    Args:
        hidden: (B, L, D) student states aligned with ``targets`` (position t predicts
            ``targets[:, t]``); ``lm_head`` maps them to logits (e.g. ``model.head``).
        targets: (B, L) next-token ids with -100 where nothing is supervised -- already shifted and
            boundary-masked (``lexhybrid.models.hybrid_lm.boundary_masked_labels``).
        teacher_hidden: (B, L, D_t) from ``teacher_forward_packed`` at the same positions, or None
            for plain CE (``alpha`` is then ignored).
        teacher_lm_head: the teacher's output layer.
        alpha, T: decision 9's weight and temperature.
        slab: positions per slab.

    Returns:
        dict of 0-dim tensors: ``loss``, ``ce`` and ``kl`` (means over supervised positions) and
        ``n_supervised``.
    """
    ce_sum = hidden.new_zeros((), dtype=torch.float32)
    kl_sum = hidden.new_zeros((), dtype=torch.float32)
    distil = teacher_hidden is not None
    for start in range(0, hidden.shape[1], slab):
        h, t = hidden[:, start : start + slab], targets[:, start : start + slab]
        if distil:
            th = teacher_hidden[:, start : start + slab].detach()
            args = (h, th, t, lm_head, teacher_lm_head, T)
            if checkpoint and torch.is_grad_enabled() and h.requires_grad:
                ce, kl = torch.utils.checkpoint.checkpoint(_slab_terms, *args, use_reentrant=False)
            else:
                ce, kl = _slab_terms(*args)
            kl_sum = kl_sum + kl
        else:
            from lexhybrid.models.slab_loss import slab_cross_entropy

            ce, _ = slab_cross_entropy(h, t, lm_head, slab=slab, checkpoint=checkpoint)
        ce_sum = ce_sum + ce
    n = (targets != -100).sum()
    denom = n.clamp(min=1)
    ce_mean, kl_mean = ce_sum / denom, kl_sum / denom
    loss = (1.0 - alpha) * ce_mean + alpha * T**2 * kl_mean if distil else ce_mean
    return {"loss": loss, "ce": ce_mean, "kl": kl_mean, "n_supervised": n}
