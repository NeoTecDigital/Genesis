"""
Core wavetable matrix implementation.

Provides WavetableMatrix class for storing and interpolating wavetables
in a 3D grid structure with support for dense/sparse storage and compression.
"""

from typing import Optional, Dict, Tuple, Union, List, Any
import numpy as np
from pathlib import Path

from .node import WavetableNode, NodeMetadata
from ..compression import WavetableCodec, CompressedWavetable, GaussianMixtureCodec


class WavetableMatrix:
    """
    3D grid of wavetables with trilinear interpolation.

    A wavetable matrix stores wavetables at integer grid positions (x, y, z)
    and supports sampling at fractional coordinates using trilinear interpolation.

    Each node can store a wavetable of shape (height, width, channels), typically
    (512, 512, 4) for Genesis integration, but supports variable resolutions.

    Attributes:
        width: Grid width (X dimension)
        height: Grid height (Y dimension)
        depth: Grid depth (Z dimension)
        resolution: Default wavetable resolution (height, width) or single int
        channels: Number of channels (typically 4 for XYZW quaternions)
        dtype: Data type for wavetables (default: float32)
        sparse: Use sparse storage (only allocate populated nodes)
        compression: Default compression method ('gaussian', 'dct', 'fft', None)
    """

    def __init__(
        self,
        width: int,
        height: int,
        depth: int,
        resolution: Union[Tuple[int, int], int] = 512,
        channels: int = 4,
        dtype: np.dtype = np.float32,
        sparse: bool = True,
        compression: Optional[str] = None
    ):
        """
        Initialize wavetable matrix.

        Args:
            width: Grid width (X dimension)
            height: Grid height (Y dimension)
            depth: Grid depth (Z dimension)
            resolution: Wavetable resolution (H, W) or single int for square
            channels: Number of channels (4 for XYZW quaternions)
            dtype: Data type (float32, float16)
            sparse: Use sparse storage (only allocate populated nodes)
            compression: Compression method ('gaussian', 'dct', 'fft', None)
        """
        if width <= 0 or height <= 0 or depth <= 0:
            raise ValueError(f"Grid dimensions must be positive: {width}×{height}×{depth}")

        self.width = width
        self.height = height
        self.depth = depth
        self.channels = channels
        self.dtype = dtype
        self.sparse = sparse
        self.compression = compression

        # Handle resolution
        if isinstance(resolution, int):
            self.resolution = (resolution, resolution)
        else:
            self.resolution = resolution

        # Storage: sparse dict or dense 3D array
        if sparse:
            # Sparse: dict mapping (x, y, z) -> WavetableNode
            self._nodes: Dict[Tuple[int, int, int], WavetableNode] = {}
        else:
            # Dense: pre-allocate full grid (memory intensive)
            # Store as nested dict for now (convert to proper 3D array if needed)
            self._nodes: Dict[Tuple[int, int, int], WavetableNode] = {}

        # Statistics
        self._stats = {
            'num_nodes': 0,
            'total_memory_bytes': 0,
            'compressed_nodes': 0
        }

    def _validate_coordinates(self, x: int, y: int, z: int) -> None:
        """Validate grid coordinates are in bounds."""
        if not (0 <= x < self.width):
            raise IndexError(f"X coordinate {x} out of bounds [0, {self.width})")
        if not (0 <= y < self.height):
            raise IndexError(f"Y coordinate {y} out of bounds [0, {self.height})")
        if not (0 <= z < self.depth):
            raise IndexError(f"Z coordinate {z} out of bounds [0, {self.depth})")

    def set_node(
        self,
        x: int, y: int, z: int,
        wavetable: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        auto_compress: bool = None
    ) -> None:
        """
        Set wavetable at grid position (x, y, z).

        Args:
            x, y, z: Grid coordinates
            wavetable: Wavetable array (H, W, C)
            metadata: Optional metadata dict
            auto_compress: Automatically compress if compression enabled
                (default: use matrix compression setting)
        """
        self._validate_coordinates(x, y, z)

        if wavetable.ndim != 3:
            raise ValueError(f"Wavetable must be 3D (H, W, C), got {wavetable.ndim}D")

        # Determine if we should compress
        should_compress = (
            auto_compress if auto_compress is not None
            else (self.compression is not None)
        )

        # Create node
        node = WavetableNode(
            wavetable=wavetable.astype(self.dtype),
            coordinates=(x, y, z),
            resolution=wavetable.shape[:2],
            channels=wavetable.shape[2],
            compressed=False,
            metadata=metadata or {}
        )

        # Store node
        coords = (x, y, z)
        old_node = self._nodes.get(coords)

        self._nodes[coords] = node

        # Update statistics
        if old_node is None:
            self._stats['num_nodes'] += 1
        else:
            self._stats['total_memory_bytes'] -= old_node.memory_bytes

        self._stats['total_memory_bytes'] += node.memory_bytes

        # Compress if requested
        if should_compress:
            self.compress_node(x, y, z, method=self.compression, quality=0.95)

    def get_node(self, x: int, y: int, z: int) -> Optional[np.ndarray]:
        """
        Get wavetable at grid position (x, y, z).

        Args:
            x, y, z: Grid coordinates

        Returns:
            Wavetable array or None if node doesn't exist
        """
        self._validate_coordinates(x, y, z)

        coords = (x, y, z)
        node = self._nodes.get(coords)

        if node is None:
            return None

        # If compressed, decompress
        if node.compressed:
            return self.decompress_node(x, y, z)

        return node.wavetable

    def has_node(self, x: int, y: int, z: int) -> bool:
        """
        Check if node exists at position.

        Args:
            x, y, z: Grid coordinates

        Returns:
            True if node exists
        """
        self._validate_coordinates(x, y, z)
        return (x, y, z) in self._nodes

    def delete_node(self, x: int, y: int, z: int) -> None:
        """
        Delete node at position (for sparse matrices).

        Args:
            x, y, z: Grid coordinates
        """
        self._validate_coordinates(x, y, z)

        coords = (x, y, z)
        node = self._nodes.get(coords)

        if node is not None:
            del self._nodes[coords]
            self._stats['num_nodes'] -= 1
            self._stats['total_memory_bytes'] -= node.memory_bytes
            if node.compressed:
                self._stats['compressed_nodes'] -= 1

    def compress_node(
        self,
        x: int, y: int, z: int,
        method: str = 'gaussian',
        quality: float = 0.95
    ) -> None:
        """
        Compress node in-place.

        Args:
            x, y, z: Grid coordinates
            method: Compression method ('gaussian', 'dct', 'fft')
            quality: Compression quality (0-1), higher = better
        """
        self._validate_coordinates(x, y, z)

        coords = (x, y, z)
        node = self._nodes.get(coords)

        if node is None:
            raise ValueError(f"No node at position ({x}, {y}, {z})")

        if node.compressed:
            # Already compressed
            return

        # Get codec
        codec = self._get_codec(method)

        # Compress wavetable
        compressed = codec.encode(node.wavetable, quality=quality)

        # Update node
        old_memory = node.memory_bytes

        node.compressed = True
        node.compression_method = method
        node.compressed_params = compressed
        node.wavetable = None  # Free uncompressed data

        # Update statistics
        new_memory = compressed.get_memory_usage()
        self._stats['total_memory_bytes'] += new_memory - old_memory
        self._stats['compressed_nodes'] += 1

    def decompress_node(self, x: int, y: int, z: int) -> np.ndarray:
        """
        Decompress node and return wavetable.

        Args:
            x, y, z: Grid coordinates

        Returns:
            Decompressed wavetable

        Note: Does not decompress in-place, just returns the wavetable
        """
        self._validate_coordinates(x, y, z)

        coords = (x, y, z)
        node = self._nodes.get(coords)

        if node is None:
            raise ValueError(f"No node at position ({x}, {y}, {z})")

        if not node.compressed:
            # Not compressed, return wavetable
            return node.wavetable

        # Get codec
        codec = self._get_codec(node.compression_method)

        # Decompress
        wavetable = codec.decode(node.compressed_params)

        return wavetable

    def decompress_node_in_place(self, x: int, y: int, z: int) -> None:
        """
        Decompress node in-place (replace compressed with uncompressed).

        Args:
            x, y, z: Grid coordinates
        """
        self._validate_coordinates(x, y, z)

        coords = (x, y, z)
        node = self._nodes.get(coords)

        if node is None or not node.compressed:
            return

        # Decompress
        wavetable = self.decompress_node(x, y, z)

        # Update node
        old_memory = node.compressed_params.get_memory_usage()

        node.wavetable = wavetable
        node.compressed = False
        node.compressed_params = None
        node.compression_method = None

        # Update statistics
        new_memory = wavetable.nbytes
        self._stats['total_memory_bytes'] += new_memory - old_memory
        self._stats['compressed_nodes'] -= 1

    def compress_all(self, method: str = 'gaussian', quality: float = 0.95) -> None:
        """
        Compress all uncompressed nodes.

        Args:
            method: Compression method
            quality: Compression quality (0-1)
        """
        for coords in list(self._nodes.keys()):
            node = self._nodes[coords]
            if not node.compressed:
                self.compress_node(*coords, method=method, quality=quality)

    def decompress_all(self) -> None:
        """Decompress all compressed nodes in-place."""
        for coords in list(self._nodes.keys()):
            node = self._nodes[coords]
            if node.compressed:
                self.decompress_node_in_place(*coords)

    def _get_codec(self, method: str) -> WavetableCodec:
        """
        Get codec instance for compression method.

        Args:
            method: Compression method name

        Returns:
            WavetableCodec instance
        """
        if method == 'gaussian':
            return GaussianMixtureCodec()
        # TODO: Add other codecs (DCT, FFT, etc.) in Phase 2 Week 4
        else:
            raise ValueError(f"Unknown compression method: {method}")

    def sample(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Sample wavetable at fractional coordinates using trilinear interpolation.

        Args:
            x, y, z: Fractional coordinates in [0, width/height/depth]

        Returns:
            Interpolated wavetable [resolution_h, resolution_w, channels]
        """
        # Import here to avoid circular dependency
        from ..interpolation.trilinear import trilinear_interpolate

        return trilinear_interpolate(self, x, y, z)

    def sample_batch(self, coords: np.ndarray) -> np.ndarray:
        """
        Batch sampling for multiple coordinates.

        Args:
            coords: (N, 3) array of [x, y, z] coordinates

        Returns:
            (N, resolution_h, resolution_w, channels) interpolated wavetables
        """
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coords must be (N, 3), got {coords.shape}")

        results = []
        for i in range(len(coords)):
            x, y, z = coords[i]
            result = self.sample(x, y, z)
            results.append(result)

        return np.array(results)

    def get_resolution(self, x: int, y: int, z: int) -> Tuple[int, int]:
        """
        Get resolution of node at (x, y, z).

        Args:
            x, y, z: Grid coordinates

        Returns:
            (height, width) resolution tuple
        """
        self._validate_coordinates(x, y, z)

        node = self._nodes.get((x, y, z))
        if node is None:
            return self.resolution

        return node.resolution

    def set_global_resolution(self, resolution: Union[Tuple[int, int], int]) -> None:
        """
        Change default resolution for all future nodes.

        Args:
            resolution: New resolution (H, W) or single int
        """
        if isinstance(resolution, int):
            self.resolution = (resolution, resolution)
        else:
            self.resolution = resolution

    def get_memory_usage(self) -> Dict[str, int]:
        """
        Get memory usage statistics.

        Returns:
            Dict with memory statistics in bytes
        """
        return {
            'total_bytes': self._stats['total_memory_bytes'],
            'total_kb': self._stats['total_memory_bytes'] / 1024,
            'total_mb': self._stats['total_memory_bytes'] / (1024 ** 2),
            'num_nodes': self._stats['num_nodes'],
            'avg_bytes_per_node': (
                self._stats['total_memory_bytes'] / self._stats['num_nodes']
                if self._stats['num_nodes'] > 0 else 0
            ),
            'compressed_nodes': self._stats['compressed_nodes']
        }

    def get_compression_ratio(self) -> float:
        """
        Get overall compression ratio.

        Returns:
            Compression ratio (uncompressed_size / compressed_size)
        """
        if self._stats['compressed_nodes'] == 0:
            return 1.0

        # Calculate total uncompressed size if all nodes were uncompressed
        total_uncompressed = 0
        total_compressed = 0

        for node in self._nodes.values():
            if node.compressed:
                h, w = node.resolution
                c = node.channels
                uncompressed_size = h * w * c * self.dtype(0).itemsize
                compressed_size = node.compressed_params.get_memory_usage()

                total_uncompressed += uncompressed_size
                total_compressed += compressed_size

        if total_compressed == 0:
            return 1.0

        return total_uncompressed / total_compressed

    def get_populated_nodes(self) -> List[Tuple[int, int, int]]:
        """
        Get list of all populated node coordinates.

        Returns:
            List of (x, y, z) tuples
        """
        return list(self._nodes.keys())

    def clear(self) -> None:
        """Clear all nodes from the matrix."""
        self._nodes.clear()
        self._stats = {
            'num_nodes': 0,
            'total_memory_bytes': 0,
            'compressed_nodes': 0
        }

    def save(self, path: str, format: str = 'npz') -> None:
        """
        Save matrix to disk.

        Args:
            path: File path
            format: Format ('npz', 'hdf5', 'wavecube')
        """
        # Import here to avoid circular dependency
        from ..io.serialization import save_matrix

        save_matrix(self, path, format=format)

    @staticmethod
    def load(path: str) -> 'WavetableMatrix':
        """
        Load matrix from disk.

        Args:
            path: File path

        Returns:
            Loaded WavetableMatrix
        """
        # Import here to avoid circular dependency
        from ..io.serialization import load_matrix

        return load_matrix(path)

    def __repr__(self) -> str:
        """String representation."""
        mem_mb = self._stats['total_memory_bytes'] / (1024 ** 2)
        storage = "sparse" if self.sparse else "dense"
        compression_str = f", compression={self.compression}" if self.compression else ""

        return (
            f"WavetableMatrix("
            f"grid={self.width}×{self.height}×{self.depth}, "
            f"resolution={self.resolution[0]}×{self.resolution[1]}×{self.channels}, "
            f"nodes={self._stats['num_nodes']}, "
            f"memory={mem_mb:.2f}MB, "
            f"storage={storage}{compression_str})"
        )

    def __getitem__(self, key: Tuple[int, int, int]) -> Optional[np.ndarray]:
        """Allow dict-like access: matrix[x, y, z]"""
        return self.get_node(*key)

    def __setitem__(self, key: Tuple[int, int, int], value: np.ndarray) -> None:
        """Allow dict-like assignment: matrix[x, y, z] = wavetable"""
        self.set_node(*key, value)

    def __contains__(self, key: Tuple[int, int, int]) -> bool:
        """Allow 'in' operator: (x, y, z) in matrix"""
        return self.has_node(*key)
