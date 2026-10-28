"""Entailment of a claim by its cited sources: the NLI step of the verifier.

An NLI model is any callable ``(premise, hypothesis) -> float``, the probability that the premise
entails the hypothesis; the verifier passes a claim when it is at least ``ENTAILMENT_THRESHOLD``.
0.5 is the default entailment threshold; calibrate it on held-out pairs before relying on it.

``MDebertaNLI`` is the production model, ``MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7``
(multilingual, German included), loaded lazily on first call from the Hugging Face cache (offline on
the cluster). Its three classes are read by name from ``model.config.id2label``, never by position,
so a checkpoint with another label order still yields the entailment probability. The claim under
test is never cut: a pair longer than ``max_length`` loses tokens from the premise only
(``truncation="only_first"``; ``longest_first`` would cut the hypothesis's tail whenever it is the
longer of the two, as it is for a claim that quotes its premise, and the dropped words would pass
unchecked), and a hypothesis that leaves no room for any premise is not entailed (0.0). Tests use
stand-in callables; the model is never loaded offline.
"""

from collections.abc import Callable, Mapping

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
ENTAILMENT_THRESHOLD = 0.5  # fixed before calibration

NLI = Callable[[str, str], float]


def entailment_index(id2label: Mapping[int | str, str]) -> int:
    """The class index whose label is ``entailment`` (any case); ``ValueError`` unless exactly one."""
    hits = [int(i) for i, name in id2label.items() if str(name).strip().lower() == "entailment"]
    if len(hits) != 1:
        raise ValueError(f"no single entailment label in {dict(id2label)}")
    return hits[0]


class MDebertaNLI:
    """mDeBERTa XNLI as an ``NLI`` callable: softmax over the classes, the entailment probability."""

    def __init__(self, model_name: str = NLI_MODEL, device: str | None = None, max_length: int = 512):
        self.model_name, self.device, self.max_length = model_name, device, max_length
        self._model = self._tok = None
        self._entail = -1

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = (
            AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device).eval()
        )
        self._entail = entailment_index(self._model.config.id2label)

    def __call__(self, premise: str, hypothesis: str) -> float:
        import torch

        if self._model is None:
            self._load()
        hypothesis_tokens = len(self._tok(hypothesis, add_special_tokens=False)["input_ids"])
        if hypothesis_tokens + self._tok.num_special_tokens_to_add(pair=True) >= self.max_length:
            return 0.0  # the claim alone fills the window: it cannot be checked, so it is not entailed
        enc = self._tok(
            premise, hypothesis, truncation="only_first", max_length=self.max_length, return_tensors="pt"
        )
        with torch.no_grad():
            logits = self._model(**{k: v.to(self.device) for k, v in enc.items()}).logits
        return float(torch.softmax(logits.float(), dim=-1)[0, self._entail])
