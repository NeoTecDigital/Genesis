"""
Chunked WaveCube matrix implementation for efficient memory management.

Uses video-game-style chunking to load/unload regions dynamically.
Chunks are 16×16×16 by default (configurable).
"""

from typing import Optional, Dict, Tuple, Set, List, Any
import numpy as np
from pathlib import Path

from .matrix import WavetableMatrix
from ..spatial.density_analyzer import DensityAnalyzer
from .adaptive_resolution import AdaptiveResolutionManager


class ChunkedWaveCube:
    """
    WaveCube with video-game-style chunking for efficient memory management.

    Divides the 3D space into chunks of configurable size (default 16×16×16).
    Keeps active chunk and adjacent chunks loaded, offloads distant chunks.

    Attributes:
        chunk_size: Size of each chunk (x, y, z) dimensions
        chunks: Dict mapping chunk coordinates to WavetableMatrix instances
        active_chunks: Set of currently loaded chunk coordinates
        active_position: Current position in world space
        cache_radius: Number of chunks to keep cached around active position
        resolution: Default wavetable resolution
        channels: Number of channels (4 for XYZW quaternions)
        compression: Compression method for offloaded chunks
    """

    def __init__(
        self,
        chunk_size: Tuple[int, int, int] = (16, 16, 16),
        resolution: int = 512,
        channels: int = 4,
        cache_radius: int = 1,
        compression: Optional[str] = 'gaussian',
        adaptive_resolution: bool = False
    ):
        """
        Initialize chunked WaveCube.

        Args:
            chunk_size: Size of each chunk (x, y, z)
            resolution: Wavetable resolution (512×512 default)
            channels: Number of channels (4 for XYZW)
            cache_radius: Chunks to keep loaded around active position
            compression: Compression method for inactive chunks
            adaptive_resolution: Enable adaptive resolution based on density
        """
        self.chunk_size = chunk_size
        self.resolution = resolution
        self.channels = channels
        self.cache_radius = cache_radius
        self.compression = compression
        self.adaptive_resolution = adaptive_resolution

        # Storage: chunk coordinates -> WavetableMatrix
        self.chunks: Dict[Tuple[int, int, int], WavetableMatrix] = {}

        # Track loaded chunks
        self.active_chunks: Set[Tuple[int, int, int]] = set()

        # Current active position (chunk coordinates)
        self.active_position: Optional[Tuple[int, int, int]] = None

        # Adaptive resolution components
        if adaptive_resolution:
            self.density_analyzer = DensityAnalyzer()
            self.resolution_manager = AdaptiveResolutionManager(
                default_resolution=(resolution, resolution, channels)
            )
            # Track chunk densities
            self.chunk_densities: Dict[Tuple[int, int, int], float] = {}
            # Track chunk resolutions
            self.chunk_resolutions: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
        else:
            self.density_analyzer = None
            self.resolution_manager = None
            self.chunk_densities = {}
            self.chunk_resolutions = {}

        # Statistics
        self.stats = {
            'chunks_total': 0,
            'chunks_loaded': 0,
            'chunks_compressed': 0,
            'total_nodes': 0
        }

    def _world_to_chunk(self, x: int, y: int, z: int) -> Tuple[int, int, int]:
        """Convert world coordinates to chunk coordinates."""
        cx = x // self.chunk_size[0]
        cy = y // self.chunk_size[1]
        cz = z // self.chunk_size[2]
        return (cx, cy, cz)

    def _world_to_local(self, x: int, y: int, z: int) -> Tuple[int, int, int]:
        """Convert world coordinates to local chunk coordinates."""
        lx = x % self.chunk_size[0]
        ly = y % self.chunk_size[1]
        lz = z % self.chunk_size[2]
        return (lx, ly, lz)

    def _get_adjacent_chunks(
        self,
        chunk_coords: Tuple[int, int, int],
        radius: int = 1
    ) -> Set[Tuple[int, int, int]]:
        """Get chunk coordinates within radius of given chunk."""
        cx, cy, cz = chunk_coords
        adjacent = set()

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    adjacent.add((cx + dx, cy + dy, cz + dz))

        return adjacent

    def _ensure_chunk_loaded(self, chunk_coords: Tuple[int, int, int]) -> WavetableMatrix:
        """Ensure chunk is loaded, creating if necessary."""
        if chunk_coords not in self.chunks:
            # Create new chunk
            self.chunks[chunk_coords] = WavetableMatrix(
                width=self.chunk_size[0],
                height=self.chunk_size[1],
                depth=self.chunk_size[2],
                resolution=self.resolution,
                channels=self.channels,
                sparse=True,
                compression=None  # Don't auto-compress active chunks
            )
            self.stats['chunks_total'] += 1

        # Ensure it's in active set
        if chunk_coords not in self.active_chunks:
            # Decompress if needed
            chunk = self.chunks[chunk_coords]
            if hasattr(chunk, '_is_offloaded') and chunk._is_offloaded:
                if self.compression:
                    chunk.decompress_all()
                    self.stats['chunks_compressed'] -= 1
                chunk._is_offloaded = False

            self.active_chunks.add(chunk_coords)
            self.stats['chunks_loaded'] += 1

        return self.chunks[chunk_coords]

    def _offload_chunk(self, chunk_coords: Tuple[int, int, int]) -> None:
        """Compress and offload chunk from active memory."""
        if chunk_coords not in self.chunks:
            return

        if chunk_coords in self.active_chunks:
            chunk = self.chunks[chunk_coords]

            # Compress all nodes if compression is enabled
            if self.compression:
                # Compress all nodes in the chunk
                chunk.compress_all(method=self.compression, quality=0.95)
                chunk._is_offloaded = True
                self.stats['chunks_compressed'] += 1

                # Clear uncompressed data to free memory
                # WavetableMatrix uses _nodes internally
                if hasattr(chunk, '_nodes'):
                    for node in chunk._nodes.values():
                        if hasattr(node, 'is_compressed') and node.is_compressed:
                            # Clear the uncompressed data field to save memory
                            node.data = None
            else:
                # Even without compression, mark as offloaded for testing
                chunk._is_offloaded = True

            # Remove from active set
            self.active_chunks.remove(chunk_coords)
            self.stats['chunks_loaded'] -= 1

    def set_active_position(self, x: int, y: int, z: int) -> None:
        """
        Set active position and update loaded chunks.

        Args:
            x, y, z: World coordinates of active position
        """
        chunk_coords = self._world_to_chunk(x, y, z)

        # If position hasn't changed significantly, skip
        if self.active_position == chunk_coords:
            return

        self.active_position = chunk_coords

        # Determine which chunks should be loaded
        chunks_to_load = self._get_adjacent_chunks(chunk_coords, self.cache_radius)

        # Offload distant chunks
        chunks_to_offload = self.active_chunks - chunks_to_load
        for chunk in chunks_to_offload:
            self._offload_chunk(chunk)

        # Load nearby chunks
        for chunk in chunks_to_load:
            if chunk in self.chunks:
                self._ensure_chunk_loaded(chunk)

    def set_node(
        self,
        x: int, y: int, z: int,
        wavetable: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Set wavetable at world position.

        Args:
            x, y, z: World coordinates
            wavetable: Wavetable array (H, W, C)
            metadata: Optional metadata
        """
        # Get chunk and local coordinates
        chunk_coords = self._world_to_chunk(x, y, z)
        local_coords = self._world_to_local(x, y, z)

        # Ensure chunk is loaded
        chunk = self._ensure_chunk_loaded(chunk_coords)

        # Set node in chunk
        chunk.set_node(*local_coords, wavetable, metadata=metadata)
        self.stats['total_nodes'] = sum(
            len(c.get_populated_nodes())
            for c in self.chunks.values()
        )

    def get_node(self, x: int, y: int, z: int) -> Optional[np.ndarray]:
        """
        Get wavetable at world position.

        Args:
            x, y, z: World coordinates

        Returns:
            Wavetable array or None if not found
        """
        # Get chunk and local coordinates
        chunk_coords = self._world_to_chunk(x, y, z)
        local_coords = self._world_to_local(x, y, z)

        # Check if chunk exists
        if chunk_coords not in self.chunks:
            return None

        # Ensure chunk is loaded
        chunk = self._ensure_chunk_loaded(chunk_coords)

        # Get node from chunk
        return chunk.get_node(*local_coords)

    def has_node(self, x: int, y: int, z: int) -> bool:
        """
        Check if node exists at world position.

        Args:
            x, y, z: World coordinates

        Returns:
            True if node exists
        """
        chunk_coords = self._world_to_chunk(x, y, z)

        if chunk_coords not in self.chunks:
            return False

        local_coords = self._world_to_local(x, y, z)
        chunk = self.chunks[chunk_coords]

        return chunk.has_node(*local_coords)

    def delete_node(self, x: int, y: int, z: int) -> None:
        """
        Delete node at world position.

        Args:
            x, y, z: World coordinates
        """
        chunk_coords = self._world_to_chunk(x, y, z)

        if chunk_coords not in self.chunks:
            return

        local_coords = self._world_to_local(x, y, z)
        chunk = self._ensure_chunk_loaded(chunk_coords)

        chunk.delete_node(*local_coords)
        self.stats['total_nodes'] = sum(
            len(c.get_populated_nodes())
            for c in self.chunks.values()
        )

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        total_memory = sum(
            chunk.get_memory_usage()['total_bytes']
            for chunk in self.chunks.values()
        )

        active_memory = sum(
            self.chunks[coords].get_memory_usage()['total_bytes']
            for coords in self.active_chunks
            if coords in self.chunks
        )

        return {
            'total_bytes': total_memory,
            'total_mb': total_memory / (1024 ** 2),
            'active_bytes': active_memory,
            'active_mb': active_memory / (1024 ** 2),
            'chunks_total': self.stats['chunks_total'],
            'chunks_loaded': self.stats['chunks_loaded'],
            'chunks_compressed': self.stats['chunks_compressed'],
            'total_nodes': self.stats['total_nodes']
        }

    def analyze_chunk_density(
        self,
        chunk_coords: Tuple[int, int, int]
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze density of a chunk.

        Args:
            chunk_coords: Chunk coordinates

        Returns:
            Density analysis dict or None if chunk doesn't exist
        """
        if not self.adaptive_resolution or chunk_coords not in self.chunks:
            return None

        chunk = self.chunks[chunk_coords]
        num_nodes = len(chunk.get_populated_nodes())
        chunk_volume = self.chunk_size[0] * self.chunk_size[1] * self.chunk_size[2]

        # Analyze with density analyzer
        analysis = self.density_analyzer.analyze_chunk(num_nodes, chunk_volume)

        # Store density and resolution
        self.chunk_densities[chunk_coords] = analysis['density']
        self.chunk_resolutions[chunk_coords] = analysis['resolution']

        return analysis

    def adapt_chunk_resolution(
        self,
        chunk_coords: Tuple[int, int, int],
        force_reanalyze: bool = False
    ) -> None:
        """
        Adapt chunk resolution based on density.

        Args:
            chunk_coords: Chunk coordinates
            force_reanalyze: Force reanalysis even if already analyzed
        """
        if not self.adaptive_resolution:
            return

        if chunk_coords not in self.chunks:
            return

        # Analyze density if needed
        if force_reanalyze or chunk_coords not in self.chunk_densities:
            analysis = self.analyze_chunk_density(chunk_coords)
            if analysis is None:
                return
            target_resolution = analysis['resolution']
        else:
            target_resolution = self.chunk_resolutions[chunk_coords]

        # Adapt all nodes in chunk
        chunk = self.chunks[chunk_coords]
        for coords in chunk.get_populated_nodes():
            node = chunk._nodes[coords]
            if node.wavetable is not None:
                current_shape = node.wavetable.shape
                if current_shape != target_resolution:
                    # Resize wavetable
                    result = self.resolution_manager.adapt_wavetable(
                        node.wavetable,
                        target_resolution,
                        track_error=True
                    )
                    node.wavetable = result['wavetable']
                    node.resolution = target_resolution[:2]

    def get_chunk_density(
        self,
        chunk_coords: Tuple[int, int, int]
    ) -> Optional[float]:
        """
        Get cached density for chunk.

        Args:
            chunk_coords: Chunk coordinates

        Returns:
            Density value or None
        """
        return self.chunk_densities.get(chunk_coords)

    def get_chunk_resolution(
        self,
        chunk_coords: Tuple[int, int, int]
    ) -> Optional[Tuple[int, int, int]]:
        """
        Get target resolution for chunk.

        Args:
            chunk_coords: Chunk coordinates

        Returns:
            Resolution tuple or None
        """
        return self.chunk_resolutions.get(chunk_coords)

    def offload_all_inactive(self) -> None:
        """Offload all chunks except active and adjacent."""
        if self.active_position is None:
            # No active position, offload everything
            for chunk_coords in list(self.active_chunks):
                self._offload_chunk(chunk_coords)
        else:
            # Keep only chunks near active position
            chunks_to_keep = self._get_adjacent_chunks(
                self.active_position,
                self.cache_radius
            )
            chunks_to_offload = self.active_chunks - chunks_to_keep

            for chunk_coords in chunks_to_offload:
                self._offload_chunk(chunk_coords)

    def load_all(self) -> None:
        """Load all chunks (for debugging/analysis)."""
        for chunk_coords in self.chunks:
            self._ensure_chunk_loaded(chunk_coords)

    def clear(self) -> None:
        """Clear all chunks."""
        self.chunks.clear()
        self.active_chunks.clear()
        self.active_position = None
        self.stats = {
            'chunks_total': 0,
            'chunks_loaded': 0,
            'chunks_compressed': 0,
            'total_nodes': 0
        }

    def __repr__(self) -> str:
        """String representation."""
        mem_stats = self.get_memory_usage()
        return (
            f"ChunkedWaveCube("
            f"chunk_size={self.chunk_size}, "
            f"chunks={self.stats['chunks_total']}, "
            f"loaded={self.stats['chunks_loaded']}, "
            f"nodes={self.stats['total_nodes']}, "
            f"memory={mem_stats['active_mb']:.2f}MB active / "
            f"{mem_stats['total_mb']:.2f}MB total)"
        )