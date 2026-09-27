"""``HybridLanguageModel``: embedding -> N x HybridBlock -> RMSNorm -> LM head.

Ported from the reference ``models/hybrid_lm.py`` (lines 1-600) without the retrieval encoder
(``AttentionPooling``, ``HybridTextEncoder``) and without the image-prefix arguments of the decode
methods. Recorded defects kept for the P1 parity fixtures and fixed in P2: the loss does not mask
the prediction across a document boundary (defect 6), prefill steps token by token (defect 7),
beam search has no EOS handling (defect 11), and the head is tied before the weight pass draws
the tied matrix (defect 17).
"""

import dataclasses
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
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


@dataclass
class CausalLMOutput:
    """Output of a causal language model forward pass."""

    loss: torch.Tensor | None = None
    logits: torch.Tensor = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


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
        for reserved in _RESERVED_BLOCK_ARGS:
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
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embeddings.token_embedding.weight

        self.apply(self._init_weights)

        # `_init_weights` zeroes every nn.Linear bias, which erases per-mixer inits done in the
        # block's own __init__ (the Mamba dt bias is exactly that). Mixers that need to re-assert an
        # init expose `post_model_init`.
        for layer in self.layers:
            mixer = getattr(layer, "mixer", None)
            if hasattr(mixer, "post_model_init"):
                mixer.post_model_init()

        logger.info(self.architecture_fingerprint())

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
            labels: (B, L) targets; the loss is the shifted next-token cross-entropy.
            attention_mask: (B, L), 1 for real tokens and 0 for padding; padded positions are
                zeroed at the embedding so recurrent state does not absorb them.
            doc_ids: (B, L) per-position document ids for packed rows; every mixer resets at a
                change of id.
            output_hidden_states: also return every block's input and the final hidden state.
            return_dict: return ``CausalLMOutput`` rather than a tuple.
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

        hidden_states = self.final_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n. (Defect 6: no document-boundary mask yet.)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,)
            if output_hidden_states:
                output = output + (all_hidden_states,)
            return ((loss,) + output) if loss is not None else output
        return CausalLMOutput(loss=loss, logits=logits, hidden_states=all_hidden_states)

    # -- sampling helpers --------------------------------------------------------------------------

    def _filter_logits(self, logits, temperature, top_k, top_p):
        """Shared sampling filter, so the cached and uncached paths cannot diverge on it."""
        logits = logits / temperature
        if top_k is not None:
            logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = float("-inf")
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumulative > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = 0
            logits[remove.scatter(1, sorted_indices, remove)] = float("-inf")
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """Uncached sampling: re-runs the full forward for every new token (O(L) per token)."""
        self.eval()
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids, return_dict=True).logits[:, -1, :]
            filtered = self._filter_logits(logits.clone(), temperature, top_k, top_p)
            next_token = torch.multinomial(torch.softmax(filtered, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids

    # -- O(1) cached decode (reference M6-C) -----------------------------------------------------

    def supports_cached_decode(self) -> bool:
        """True when every mixer has an O(1) ``step``; all-or-nothing by design, because one
        recomputing layer keeps the whole model O(L) per token."""
        return all(getattr(layer.mixer, "supports_step", False) for layer in self.layers)

    def allocate_inference_cache(self, batch_size, device=None, dtype=torch.float32):
        """One cache per layer, independent of context length for the recurrent mixers."""
        device = device or self.lm_head.weight.device
        return [
            layer.allocate_inference_cache(batch_size, device=device, dtype=dtype) for layer in self.layers
        ]

    def step_logits(self, hidden_t: torch.Tensor, caches) -> torch.Tensor:
        """One token of hidden state -> next-token logits, advancing every layer's cache."""
        for layer, cache in zip(self.layers, caches):
            hidden_t = layer.step(hidden_t, cache)
        return self.lm_head(self.final_norm(hidden_t))

    def prefill(self, hidden: torch.Tensor, caches) -> torch.Tensor:
        """Consume a prompt token by token, leaving the caches after its last token (defect 7).

        Returns the logits at the final prompt position, (batch, vocab).
        """
        logits = None
        for t in range(hidden.shape[1]):
            logits = self.step_logits(hidden[:, t], caches)
        return logits

    @torch.no_grad()
    def generate_cached(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """``generate`` with an O(1)-per-token recurrent cache instead of full recomputation."""
        if not self.supports_cached_decode():
            raise NotImplementedError(
                f"cached decode needs every mixer to implement step(); layer types are {self.get_layer_types()}"
            )
        self.eval()
        param = self.lm_head.weight
        caches = self.allocate_inference_cache(input_ids.shape[0], device=param.device, dtype=param.dtype)
        generated_ids = input_ids
        logits = self.prefill(self.embeddings(input_ids), caches)
        for _ in range(max_new_tokens):
            filtered = self._filter_logits(logits.clone(), temperature, top_k, top_p)
            next_token = torch.multinomial(torch.softmax(filtered, dim=-1), num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            logits = self.step_logits(self.embeddings(next_token)[:, 0], caches)
        return generated_ids

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
    ) -> torch.Tensor:
        """Beam search over the recurrent cache, one sample at a time.

        Beams live in the batch axis of one cache, so a step costs O(beam) and the prompt is
        consumed once. Tie-breaking and the length-penalty convention match
        ``lexhybrid.decoding.generate.beam_search_uncached``, so the two return identical tokens.
        """
        if input_ids.shape[0] != 1:
            raise ValueError(
                f"beam_search_cached operates on one sample at a time (got batch {input_ids.shape[0]})"
            )
        if not self.supports_cached_decode():
            raise NotImplementedError("cached decode needs every mixer to implement step()")
        self.eval()
        device = input_ids.device
        param = self.lm_head.weight

        hidden = self.embeddings(input_ids)
        # All beams start from the same prompt, so prefill once, replicated.
        hidden = hidden.expand(beam_size, -1, -1).contiguous()
        caches = self.allocate_inference_cache(beam_size, device=param.device, dtype=param.dtype)
        logits = self.prefill(hidden, caches)  # (beam, vocab)

        tokens = input_ids.expand(beam_size, -1).contiguous()
        # Only beam 0 is live at the start; the rest are -inf so the first expansion picks the true
        # top-k of a single hypothesis rather than k copies of it.
        scores = torch.full((beam_size,), float("-inf"), device=device)
        scores[0] = 0.0

        for _ in range(max_new_tokens):
            log_probs = torch.log_softmax(logits.float(), dim=-1)  # (beam, vocab)
            total = scores.unsqueeze(-1) + log_probs
            length = tokens.shape[1] + 1
            ranked = total / (length**length_penalty)
            _, flat_idx = ranked.view(-1).topk(beam_size)
            beam_idx = torch.div(flat_idx, log_probs.shape[-1], rounding_mode="floor")
            token_idx = flat_idx % log_probs.shape[-1]

            scores = total.view(-1)[flat_idx]
            tokens = torch.cat([tokens.index_select(0, beam_idx), token_idx.unsqueeze(-1)], dim=1)
            caches = self.reorder_cache(caches, beam_idx)
            logits = self.step_logits(self.embeddings(token_idx.unsqueeze(-1))[:, 0], caches)

        best = int(torch.argmax(scores / (tokens.shape[1] ** length_penalty)))
        return tokens[best : best + 1]

    # -- introspection -----------------------------------------------------------------------------

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Parameter count; ``non_embedding`` subtracts the token embedding (P2-M refines this)."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embeddings.token_embedding.weight.numel()
        return n_params

    def get_layer_types(self) -> list[str]:
        """The mixer type of every block, in order."""
        return [layer.layer_type for layer in self.layers]
