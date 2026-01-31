"""
Wavecube: Multi-Resolution Wavetable Matrix Library

High-performance 3D grid storage with trilinear interpolation and advanced
compression for multimodal data (text, audio, image, video).

Key Features:
- Variable resolution (8�8 to 4096�4096 or higher)
- Advanced compression (100-10000� with Gaussian mixtures)
- Real-time trilinear interpolation
- Multi-octave hierarchies
- GPU acceleration (optional)
- Modality-agnostic foundation

Quick Start:
    >>> from wavecube import WavetableMatrix
    >>> matrix = WavetableMatrix(width=10, height=10, depth=10, resolution=256)
    >>> matrix.set_node(5, 5, 5, wavetable)
    >>> result = matrix.sample(x=5.5, y=5.3, z=5.7)
"""

__version__ = "0.1.0-alpha"
__author__ = "Alembic Project"
__license__ = "MIT"

# Core (Phase 1 - IMPLEMENTED)
from .core.matrix import WavetableMatrix
from .core.node import WavetableNode, NodeMetadata

# Interpolation (Phase 1 - IMPLEMENTED)
from .interpolation.trilinear import trilinear_interpolate, trilinear_interpolate_batch

# I/O (Phase 1 - IMPLEMENTED)
from .io.serialization import save_matrix, load_matrix

# Compression will be implemented in Phase 2
# from .compression.codec import WavetableCodec, CompressedWavetable
# from .compression.gaussian import GaussianMixtureCodec
# from .compression.dct import DCTCodec
# from .compression.fft import FFTCodec

__all__ = [
    # Core (Phase 1 - IMPLEMENTED)
    "WavetableMatrix",
    "WavetableNode",
    "NodeMetadata",

    # Interpolation (Phase 1 - IMPLEMENTED)
    "trilinear_interpolate",
    "trilinear_interpolate_batch",

    # I/O (Phase 1 - IMPLEMENTED)
    "save_matrix",
    "load_matrix",

    # Phase 2 - TODO
    # "WavetableCodec",
    # "CompressedWavetable",
    # "GaussianMixtureCodec",
]
