"""Compression codecs for wavetable matrices."""

from .codec import WavetableCodec, CompressedWavetable
from .gaussian import GaussianMixtureCodec, GaussianMixtureParams

__all__ = [
    'WavetableCodec',
    'CompressedWavetable',
    'GaussianMixtureCodec',
    'GaussianMixtureParams',
]
