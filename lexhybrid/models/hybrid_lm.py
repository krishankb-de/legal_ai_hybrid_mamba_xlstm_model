"""``HybridLanguageModel``: embedding -> N x HybridBlock -> RMSNorm -> LM head.

Ported from the reference ``models/hybrid_lm.py`` (lines 1-600) without the retrieval encoder
(``AttentionPooling``, ``HybridTextEncoder``) and without the image-prefix arguments of the decode
methods. P2 fixes recorded here:

* P2-K (defect 7): ``prefill`` is one chunked forward per layer that fills the decode caches; the
  reference stepped the prompt token by token.
* P2-L (defect 6): the loss no longer asks a document's last token to predict the next document's
  first.
* P2-M (defect 17): the weight pass runs before the head is tied (tying first drew the shared
  matrix twice), and ``tie_word_embeddings`` defaults to True.
* P2-N..P (defect 11): every decoder lives in ``lexhybrid.decoding`` with EOS, repetition and
  length penalties; ``generate``, ``generate_cached`` and ``beam_search_cached`` delegate there.
"""

import dataclasses
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from lexhybrid.config.hybrid_config import HybridConfig
from lexhybrid.layers.hybrid_block import create_hybrid_blocks
from lexhybrid.layers.normalization import RMSNorm
from lexhybrid.utils.arch_fingerprint import architecture_fingerprint

logger = logging.getLogger(__name__)

# Fields of HybridConfig that HybridBlock/create_hybrid_blocks take explicitly.
_RESERVED_BLOCK_ARGS = (
    "dim",
    "num_layers",
    "layer_pattern",
    "norm_type",
    "norm_topology",
    "use_mlp",
    "mlp_ratio",
)
# Model-level fields a block never reads.
_MODEL_ONLY_ARGS = ("mtp_n", "mtp_loss_weight", "mtp_layer_type")


@dataclass
class CausalLMOutput:
    """Output of a causal language model forward pass."""

    loss: torch.Tensor | None = None
    logits: torch.Tensor = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None
    # Targets that entered the loss (P2-L): the boundary mask and -100 labels are excluded. A 0-dim
    # tensor, so reading it forces no host sync until someone logs it.
    n_supervised_tokens: torch.Tensor | None = None
    # The multi-token-prediction term before its weight (P2-S); None when MTP is off.
    mtp_loss: torch.Tensor | None = None


def boundary_masked_labels(labels: torch.Tensor, doc_ids: torch.Tensor | None) -> torch.Tensor:
    """Next-token targets ``labels[:, 1:]`` with every cross-document prediction set to -100.

    Position t predicts token t+1; when the two sit in different documents (t is the last token of
    one -- normally its EOS -- and t+1 the first of the next) the target is masked (defect 6). The
    EOS itself stays a target, predicted from inside its own document, so the model still learns
    to end a document. A row that starts mid-document needs nothing special: no position predicts
    into position 0. The distillation KL uses the same mask (P2-T).
    """
    shifted = labels[:, 1:].clone()
    if doc_ids is not None:
        shifted = shifted.masked_fill(doc_ids[:, 1:] != doc_ids[:, :-1], -100)
    return shifted


class HybridEmbedding(nn.Module):
    """Token embedding followed by dropout. There is no positional embedding: position comes from
    the recurrent mixers' order and the attention mixer's RoPE."""

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.token_embedding(input_ids))


class HybridLanguageModel(nn.Module):
    """Hybrid Mamba-3 / mLSTM / attention causal language model."""

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        self.embeddings = HybridEmbedding(config)

        # Every dataclass field reaches the block factory (M2-F): hand-writing ~20 `name=config.name`
        # lines is how a new field silently fell back to its default in the reference (Phase 9).
        layer_kwargs = {f.name: getattr(config, f.name) for f in dataclasses.fields(config)}
        for reserved in _RESERVED_BLOCK_ARGS + _MODEL_ONLY_ARGS:
            layer_kwargs.pop(reserved, None)
        self.layers = create_hybrid_blocks(
            dim=config.dim,
            num_layers=config.num_layers,
            layer_pattern=config.layer_pattern,
            norm_type=config.norm_type,
            norm_topology=config.norm_topology,
            use_mlp=config.use_mlp,
            mlp_ratio=config.mlp_ratio,
            **layer_kwargs,
        )

        self.final_norm = (
            RMSNorm(config.dim) if config.norm_type.lower() == "rms" else nn.LayerNorm(config.dim)
        )
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Initialise, THEN tie (defect 17): tying first made the weight pass draw the shared matrix
        # twice (as an embedding, then as a linear weight), so the tied init depended on module
        # order. Now the embedding's draw is the tied matrix.
        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embeddings.token_embedding.weight

        # `_init_weights` zeroes every nn.Linear bias, which erases per-mixer inits done in the
        # block's own __init__ (the Mamba dt bias is exactly that). Mixers that need to re-assert an
        # init expose `post_model_init`.
        for layer in self.layers:
            mixer = getattr(layer, "mixer", None)
            if hasattr(mixer, "post_model_init"):
                mixer.post_model_init()

        # Built last, so the main model's weights (and RNG draws) are the same with MTP on or off.
        self.mtp_head = self._build_mtp_head(config, layer_kwargs) if config.mtp_n > 1 else None

        logger.info(self.architecture_fingerprint())

    def _build_mtp_head(self, config: HybridConfig, layer_kwargs: dict):
        from lexhybrid.layers.hybrid_block import HybridBlock
        from lexhybrid.models.mtp_head import MTPHead

        blocks = [
            HybridBlock(
                dim=config.dim,
                layer_type=config.mtp_layer_type,
                norm_type=config.norm_type,
                use_mlp=config.use_mlp,
                mlp_ratio=config.mlp_ratio,
                norm_topology=config.norm_topology,
                is_first_block=False,
                **layer_kwargs,
            )
            for _ in range(config.mtp_n - 1)
        ]
        head = MTPHead(blocks, config.dim)
        head.apply(self._init_weights)
        for module in head.modules():
            if module is not head and hasattr(module, "post_model_init"):
                module.post_model_init()
        return head

    def architecture_fingerprint(self) -> str:
        """One-line summary of what was actually built; see ``lexhybrid.utils.arch_fingerprint``."""
        return architecture_fingerprint(self)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> CausalLMOutput | tuple:
        """Language-model forward pass.

        Args:
            input_ids: (B, L) token ids; mutually exclusive with ``inputs_embeds``.
            inputs_embeds: (B, L, D) precomputed hidden states.
            labels: (B, L) targets; the loss is the shifted next-token cross-entropy, ignoring -100
                and, with ``doc_ids``, every prediction that crosses a document boundary.
            attention_mask: (B, L), 1 for real tokens and 0 for padding; padded positions are
                zeroed at the embedding so recurrent state does not absorb them.
            doc_ids: (B, L) per-position document ids for packed rows; every mixer resets at a
                change of id.
            output_hidden_states: also return every block's input and the final hidden state.
            return_dict: return ``CausalLMOutput`` rather than a tuple.
        """
        residual_stream, all_hidden_states = self.backbone(
            input_ids, inputs_embeds, attention_mask, doc_ids, output_hidden_states=output_hidden_states
        )
        token_ids = input_ids
        hidden_states = self.final_norm(residual_stream)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        logits = self.lm_head(hidden_states)

        loss = n_supervised = None
        if labels is not None:
            shift_labels = boundary_masked_labels(labels, doc_ids)
            n_supervised = (shift_labels != -100).sum()
            ce_sum = F.cross_entropy(
                logits[..., :-1, :].reshape(-1, self.config.vocab_size),
                shift_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            # Mean over supervised targets; a batch with none gives 0 rather than 0/0 = NaN.
            loss = ce_sum / n_supervised.clamp(min=1)

        mtp_loss = None
        if labels is not None and self.mtp_head is not None:
            if token_ids is None:
                raise ValueError("multi-token prediction needs input_ids (it embeds the tokens ahead)")
            mtp_loss = self.mtp_head.loss(
                residual_stream,
                token_ids,
                self.embeddings.token_embedding,
                self.head,
                doc_ids=doc_ids,
                labels=labels,
            )
            loss = loss + self.config.mtp_loss_weight * mtp_loss

        if not return_dict:
            output = (logits,)
            if output_hidden_states:
                output = output + (all_hidden_states,)
            return ((loss,) + output) if loss is not None else output
        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=all_hidden_states,
            n_supervised_tokens=n_supervised,
            mtp_loss=mtp_loss,
        )

    def backbone(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ):
        """Embedding and every block: the residual stream (B, L, D) BEFORE the final norm.

        Training at the Qwen3 vocabulary uses this with ``head`` and the slab-wise losses
        (``lexhybrid.models.slab_loss``, ``lexhybrid.training.distill``), so ``(B, L, V)`` logits
        never exist; ``forward`` is ``head(backbone(...))`` plus the loss.

        Returns:
            ``(residual_stream, all_hidden_states)``; the tuple is None unless requested.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Exactly one of input_ids or inputs_embeds must be provided")
        hidden_states = self.embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.to(hidden_states.dtype).unsqueeze(-1)

        all_hidden_states = () if output_hidden_states else None
        use_ckpt = self.config.use_gradient_checkpointing and self.training
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if use_ckpt:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, None, doc_ids, use_reentrant=False
                )
            else:
                hidden_states = layer(hidden_states, cache=None, doc_ids=doc_ids)
        return hidden_states, all_hidden_states

    def head(self, residual_stream: torch.Tensor) -> torch.Tensor:
        """Final norm and LM head: residual stream -> logits."""
        return self.lm_head(self.final_norm(residual_stream))

    # -- sampling helpers --------------------------------------------------------------------------

    def _filter_logits(self, logits, temperature, top_k, top_p):
        """Shared sampling filter (``lexhybrid.decoding.filter_logits``)."""
        from lexhybrid.decoding.generate import filter_logits

        return filter_logits(logits, temperature, top_k, top_p)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        **decode_kwargs,
    ) -> torch.Tensor:
        """Uncached sampling (``lexhybrid.decoding.sample``): the full forward per new token."""
        from lexhybrid.decoding.generate import sample

        return sample(self, input_ids, max_new_tokens, temperature, top_k, top_p, **decode_kwargs)

    # -- O(1) cached decode (reference M6-C) -----------------------------------------------------

    def supports_cached_decode(self) -> bool:
        """True when every mixer has an O(1) ``step``; all-or-nothing by design, because one
        recomputing layer keeps the whole model O(L) per token."""
        return all(getattr(layer.mixer, "supports_step", False) for layer in self.layers)

    def allocate_inference_cache(self, batch_size, device=None, dtype=torch.float32, max_seq_len=None):
        """One cache per layer. Recurrent mixers hold a fixed-size state; attention layers hold a KV
        cache for ``max_seq_len`` tokens (default ``max_position_embeddings``)."""
        device = device or self.lm_head.weight.device
        return [
            layer.allocate_inference_cache(batch_size, device=device, dtype=dtype, max_seq_len=max_seq_len)
            for layer in self.layers
        ]

    def step_logits(self, hidden_t: torch.Tensor, caches) -> torch.Tensor:
        """One token of hidden state -> next-token logits, advancing every layer's cache."""
        for layer, cache in zip(self.layers, caches):
            hidden_t = layer.step(hidden_t, cache)
        return self.lm_head(self.final_norm(hidden_t))

    def prefill(self, hidden: torch.Tensor, caches, doc_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Consume a prompt in ONE chunked forward per layer, filling every cache (P2-K, defect 7).

        Each mixer's forward writes its decode state when handed an empty cache (Mamba-3: SSM state,
        conv window, trapezoid and rotation state; mLSTM: C and n; attention: K and V), so the
        prompt costs one parallel pass instead of the reference's ``L`` sequential steps.

        Args:
            hidden: (batch, L, dim) prompt embeddings.
            caches: from ``allocate_inference_cache``, empty.
            doc_ids: optional (batch, L); decoding then continues each row's last document.

        Returns:
            The logits at the final prompt position, (batch, vocab).
        """
        for layer, cache in zip(self.layers, caches):
            hidden = layer(hidden, cache=cache, doc_ids=doc_ids)
        return self.lm_head(self.final_norm(hidden[:, -1]))

    @torch.no_grad()
    def generate_cached(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        **decode_kwargs,
    ) -> torch.Tensor:
        """``generate`` over the caches (``lexhybrid.decoding.sample_cached``): one prefill pass,
        then O(1) recurrent steps (KV for attention layers)."""
        from lexhybrid.decoding.generate import sample_cached

        return sample_cached(self, input_ids, max_new_tokens, temperature, top_k, top_p, **decode_kwargs)

    @staticmethod
    def reorder_cache(caches, index: torch.Tensor):
        """Reindex every cached tensor along the batch axis (beam search).

        Beams reorder and duplicate every step, so their recurrent state must follow. Scalars such
        as ``seen`` are shared across beams by construction and are copied through.
        """
        out = []
        for cache in caches:
            if cache is None:
                out.append(None)
                continue
            out.append({k: (v.index_select(0, index) if torch.is_tensor(v) else v) for k, v in cache.items()})
        return out

    @torch.no_grad()
    def beam_search_cached(
        self,
        input_ids: torch.Tensor,
        beam_size: int = 3,
        max_new_tokens: int = 100,
        length_penalty: float = 1.0,
        eos_token_id: int | None = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Beam search over the caches (``lexhybrid.decoding.beam_search_cached``): EOS-finished
        hypotheses scored ``sum_logprob / generated ** length_penalty`` (defect 11: the reference
        had no EOS and its length penalty never changed a ranking)."""
        from lexhybrid.decoding.generate import beam_search_cached

        return beam_search_cached(
            self,
            input_ids,
            beam_size=beam_size,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
        )

    # -- introspection -----------------------------------------------------------------------------

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Parameter count (decision 11): ``total`` counts every distinct tensor once, so a tied head
        is not double-counted; ``non_embedding = total - embedding - (lm_head if untied)``."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embeddings.token_embedding.weight.numel()
            if self.lm_head.weight is not self.embeddings.token_embedding.weight:
                n_params -= self.lm_head.weight.numel()
        return n_params

    def get_layer_types(self) -> list[str]:
        """The mixer type of every block, in order."""
        return [layer.layer_type for layer in self.layers]
