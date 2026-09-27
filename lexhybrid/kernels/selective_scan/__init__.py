"""Mamba-1 selective scan (legacy ablation only)."""

from lexhybrid.kernels.selective_scan.scan_interface import (
    selective_scan,
    selective_scan_exact,
    selective_scan_parallel,
    selective_scan_sequential_reference,
)

__all__ = [
    "selective_scan",
    "selective_scan_exact",
    "selective_scan_parallel",
    "selective_scan_sequential_reference",
]
