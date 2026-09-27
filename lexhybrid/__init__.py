"""lexhybrid -- a DACH legal, citation-faithful, retrieval-gated hybrid Mamba-3 / mLSTM / attention decoder.

Every operator is plain PyTorch; there are no hand-written CUDA or Triton kernels. Triton code
appears only when a model runs under ``torch.compile``, because Inductor generates it.

Heavy submodules are imported lazily so ``import lexhybrid`` stays cheap for tooling.
"""

__version__ = "0.1.0"

__all__ = ["HybridConfig", "HybridLanguageModel", "__version__"]


def __getattr__(name):
    if name == "HybridConfig":
        from lexhybrid.config.hybrid_config import HybridConfig

        return HybridConfig
    if name == "HybridLanguageModel":
        from lexhybrid.models.hybrid_lm import HybridLanguageModel

        return HybridLanguageModel
    raise AttributeError(f"module 'lexhybrid' has no attribute {name!r}")
