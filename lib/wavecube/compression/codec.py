"""
Base codec interface for wavetable compression.

Provides abstract base class and data structures for implementing
compression methods like Gaussian mixtures, DCT, FFT, etc.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class CompressedWavetable:
    """
    Compressed representation of a wavetable.

    Attributes:
        method: Compression method used ('gaussian', 'dct', 'fft', etc.)
        params: Method-specific compression parameters
        original_shape: Original wavetable shape (H, W, C)
        dtype: Original data type
        metadata: Additional compression metadata
    """
    method: str
    params: Any
    original_shape: tuple[int, int, int]
    dtype: np.dtype
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_memory_usage(self) -> int:
        """
        Get memory usage of compressed representation in bytes.

        Returns:
            Memory usage in bytes
        """
        # Handle dataclass params (like GaussianMixtureParams)
        if hasattr(self.params, 'to_dict'):
            params_dict = self.params.to_dict()
            total = 0
            for value in params_dict.values():
                if isinstance(value, np.ndarray):
                    total += value.nbytes
                elif isinstance(value, int):
                    total += 4
                elif isinstance(value, float):
                    total += 8
            return total
        # Estimate based on params
        elif isinstance(self.params, np.ndarray):
            return self.params.nbytes
        elif isinstance(self.params, dict):
            total = 0
            for value in self.params.values():
                if isinstance(value, np.ndarray):
                    total += value.nbytes
                elif isinstance(value, (int, float)):
                    total += 8  # Assume 64-bit
            return total
        else:
            return 0

    def get_compression_ratio(self) -> float:
        """
        Calculate compression ratio vs uncompressed size.

        Returns:
            Compression ratio (original_size / compressed_size)
        """
        h, w, c = self.original_shape
        original_size = h * w * c * self.dtype.itemsize
        compressed_size = self.get_memory_usage()

        if compressed_size == 0:
            return float('inf')

        return original_size / compressed_size


class WavetableCodec(ABC):
    """
    Abstract base class for wavetable compression codecs.

    All compression methods inherit from this class and implement
    encode() and decode() methods.
    """

    def __init__(self, **kwargs):
        """
        Initialize codec with optional parameters.

        Args:
            **kwargs: Codec-specific parameters
        """
        self.params = kwargs

    @abstractmethod
    def encode(
        self,
        wavetable: np.ndarray,
        quality: float = 0.95
    ) -> CompressedWavetable:
        """
        Compress wavetable to parametric representation.

        Args:
            wavetable: Input wavetable of shape (H, W, C)
            quality: Quality level (0-1), higher = better quality

        Returns:
            CompressedWavetable with method-specific parameters

        Raises:
            ValueError: If wavetable shape is invalid
        """
        raise NotImplementedError

    @abstractmethod
    def decode(
        self,
        compressed: CompressedWavetable
    ) -> np.ndarray:
        """
        Reconstruct wavetable from compressed representation.

        Args:
            compressed: CompressedWavetable to decompress

        Returns:
            Reconstructed wavetable of shape (H, W, C)

        Raises:
            ValueError: If compressed data is invalid
        """
        raise NotImplementedError

    def validate_wavetable(self, wavetable: np.ndarray) -> None:
        """
        Validate wavetable shape and type.

        Args:
            wavetable: Wavetable to validate

        Raises:
            ValueError: If wavetable is invalid
        """
        if not isinstance(wavetable, np.ndarray):
            raise ValueError(f"Wavetable must be ndarray, got {type(wavetable)}")

        if wavetable.ndim != 3:
            raise ValueError(
                f"Wavetable must be 3D (H, W, C), got shape {wavetable.shape}"
            )

        h, w, c = wavetable.shape
        if h < 1 or w < 1 or c < 1:
            raise ValueError(f"Invalid wavetable shape: {wavetable.shape}")

    def compute_error(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute reconstruction error metrics.

        Args:
            original: Original wavetable
            reconstructed: Reconstructed wavetable

        Returns:
            Dict with error metrics (mse, mae, max_error, psnr)
        """
        mse = np.mean((original - reconstructed) ** 2)
        mae = np.mean(np.abs(original - reconstructed))
        max_error = np.max(np.abs(original - reconstructed))

        # PSNR (Peak Signal-to-Noise Ratio)
        if mse > 0:
            data_range = original.max() - original.min()
            psnr = 20 * np.log10(data_range / np.sqrt(mse))
        else:
            psnr = float('inf')

        return {
            'mse': float(mse),
            'mae': float(mae),
            'max_error': float(max_error),
            'psnr': float(psnr)
        }
