"""
Wavetable node data structure.

Stores metadata and information about a single wavetable in the matrix.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..compression import CompressedWavetable


@dataclass
class WavetableNode:
    """
    Single node in a wavetable matrix.

    Stores a wavetable (N-dimensional array) along with metadata about
    its position, resolution, compression status, and other properties.

    Attributes:
        wavetable: The actual wavetable data (H, W, C) array
        coordinates: Grid position (x, y, z)
        resolution: Wavetable resolution (height, width)
        channels: Number of channels (typically 4 for XYZW quaternions)
        compressed: Whether this node is compressed
        compression_method: Compression codec used ('gaussian', 'dct', 'fft', None)
        compressed_params: Compressed parameters (if compressed=True)
        metadata: Additional user-defined metadata
    """

    wavetable: Optional[np.ndarray] = None
    coordinates: tuple[int, int, int] = (0, 0, 0)
    resolution: tuple[int, int] = (512, 512)
    channels: int = 4
    compressed: bool = False
    compression_method: Optional[str] = None
    compressed_params: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate node configuration."""
        if self.wavetable is not None:
            # Validate wavetable shape
            if self.wavetable.ndim != 3:
                raise ValueError(
                    f"Wavetable must be 3D (H, W, C), got {self.wavetable.ndim}D"
                )

            h, w, c = self.wavetable.shape

            # Update resolution and channels from wavetable if not compressed
            if not self.compressed:
                self.resolution = (h, w)
                self.channels = c

        # Validate compression state
        if self.compressed:
            if self.compression_method is None:
                raise ValueError("Compressed node must have compression_method")
            if self.compressed_params is None:
                raise ValueError("Compressed node must have compressed_params")

    @property
    def shape(self) -> tuple[int, int, int]:
        """Get wavetable shape (height, width, channels)."""
        if self.wavetable is not None:
            return self.wavetable.shape
        else:
            return (*self.resolution, self.channels)

    @property
    def size(self) -> int:
        """Get total number of elements in wavetable."""
        h, w = self.resolution
        return h * w * self.channels

    @property
    def memory_bytes(self) -> int:
        """
        Estimate memory usage in bytes.

        Returns:
            Approximate memory usage (uncompressed size or compressed params size)
        """
        if self.compressed and self.compressed_params is not None:
            # Use CompressedWavetable's method if available
            if hasattr(self.compressed_params, 'get_memory_usage'):
                return self.compressed_params.get_memory_usage()
            elif isinstance(self.compressed_params, np.ndarray):
                return self.compressed_params.nbytes
            elif isinstance(self.compressed_params, dict):
                # Rough estimate for dict-based params
                total = 0
                for v in self.compressed_params.values():
                    if isinstance(v, np.ndarray):
                        total += v.nbytes
                    else:
                        total += 8  # Assume 8 bytes per scalar
                return total
            else:
                return 0
        elif self.wavetable is not None:
            return self.wavetable.nbytes
        else:
            # Estimate based on resolution and dtype (assume float32)
            return self.size * 4

    def is_valid(self) -> bool:
        """
        Check if node is in a valid state.

        Returns:
            True if node has either wavetable or compressed params
        """
        return (
            (self.wavetable is not None) or
            (self.compressed and self.compressed_params is not None)
        )

    def __repr__(self) -> str:
        """String representation."""
        status = "compressed" if self.compressed else "uncompressed"
        coord_str = f"({self.coordinates[0]}, {self.coordinates[1]}, {self.coordinates[2]})"
        res_str = f"{self.resolution[0]}×{self.resolution[1]}×{self.channels}"
        mem_str = f"{self.memory_bytes / 1024:.2f}KB"

        return (
            f"WavetableNode(coords={coord_str}, "
            f"resolution={res_str}, "
            f"status={status}, "
            f"memory={mem_str})"
        )


@dataclass
class NodeMetadata:
    """
    Extended metadata for wavetable nodes.

    Optional additional information that can be attached to nodes
    for specific use cases (Genesis integration, clustering, etc.).
    """

    # Genesis-specific
    octave: Optional[int] = None
    frequency: Optional[np.ndarray] = None  # (H, W, 2) [magnitude, phase]
    fundamental_freq: Optional[float] = None
    harmonic_signature: Optional[np.ndarray] = None  # 10 harmonic coefficients

    # Clustering
    cluster_id: Optional[int] = None
    cluster_distance: Optional[float] = None
    resonance_strength: int = 1

    # Multi-modal
    modality: Optional[str] = None  # 'text', 'audio', 'image', 'video'
    source_path: Optional[str] = None

    # Temporal
    timestamp: Optional[float] = None
    version: int = 1

    # Quality metrics
    compression_ratio: Optional[float] = None
    reconstruction_error: Optional[float] = None

    # Adaptive resolution
    density_level: Optional[str] = None  # 'low', 'medium', 'high'
    target_resolution: Optional[tuple] = None  # (H, W, C)
    adaptation_mse: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                if isinstance(value, np.ndarray):
                    result[key] = value.tolist()
                else:
                    result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NodeMetadata':
        """Create from dictionary."""
        # Convert lists back to arrays
        if 'frequency' in data and isinstance(data['frequency'], list):
            data['frequency'] = np.array(data['frequency'])
        if 'harmonic_signature' in data and isinstance(data['harmonic_signature'], list):
            data['harmonic_signature'] = np.array(data['harmonic_signature'])

        return cls(**data)
