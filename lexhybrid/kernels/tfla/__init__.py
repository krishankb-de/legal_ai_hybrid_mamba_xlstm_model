"""TFLA (tiled linear attention) kernels for the mLSTM."""

from lexhybrid.kernels.tfla.tfla_interface import apply_tfla, tfla_forward_parallel
from lexhybrid.kernels.tfla.tfla_reference import sequential_mlstm_reference

__all__ = ["apply_tfla", "tfla_forward_parallel", "sequential_mlstm_reference"]
