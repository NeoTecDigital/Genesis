"""Interpolation algorithms for wavetable matrices."""

from .trilinear import trilinear_interpolate, trilinear_interpolate_batch
from .bilinear import bilinear_interpolate, extract_slice, sample_slice_2d
from .nearest import nearest_neighbor, nearest_neighbor_batch, nearest_neighbor_fill

__all__ = [
    # Trilinear (3D)
    'trilinear_interpolate',
    'trilinear_interpolate_batch',
    # Bilinear (2D)
    'bilinear_interpolate',
    'extract_slice',
    'sample_slice_2d',
    # Nearest neighbor (fast)
    'nearest_neighbor',
    'nearest_neighbor_batch',
    'nearest_neighbor_fill',
]
