"""
Layered WaveCube system for multi-tier memory hierarchy.

Provides three independent WavetableMatrix layers for categorical memory
management in Genesis coherence-driven synthesis architecture.
"""

from typing import Dict, Optional, Tuple, Any, Literal
import numpy as np
import time

from .matrix import WavetableMatrix

# Type alias for layer names
LayerType = Literal['proto_unity', 'experiential', 'io']

# Default layer configurations
DEFAULT_PROTO_UNITY_CONFIG = {
    'chunk_size': (16, 16, 16),
    'cache_radius': 0,
    'compression': 'gaussian',
    'quality': 0.98
}

DEFAULT_EXPERIENTIAL_CONFIG = {
    'chunk_size': (64, 64, 64),
    'cache_radius': 2,
    'compression': 'gaussian',
    'quality': 0.90
}

DEFAULT_IO_CONFIG = {
    'chunk_size': (32, 32, 32),
    'cache_radius': 1,
    'compression': 'gaussian',
    'quality': 0.85
}


class LayeredWaveCube:
    """
    Three-layer WaveCube storage system.

    Manages three independent WavetableMatrix layers for categorical
    memory hierarchy:
    - proto_unity: Stable reference (proto-unity carrier γ ∪ ε, consolidated)
    - experiential: Working memory (EVOLUTION state, active queries)
    - io: Short-term buffer (transient/io state, immediate operations)

    Each layer has independent configuration for chunk size, cache radius,
    compression method, and quality settings.

    Attributes:
        layers: Dict mapping layer names to WavetableMatrix instances
        layer_metadata: Per-layer metadata tracking (resonance, access, etc)
        width: Grid width (X dimension)
        height: Grid height (Y dimension)
        depth: Grid depth (Z dimension)
        resolution: Default wavetable resolution
        channels: Number of channels (4 for XYZW quaternions)
    """

    def __init__(
        self,
        proto_unity_config: Optional[Dict] = None,
        experiential_config: Optional[Dict] = None,
        io_config: Optional[Dict] = None,
        width: int = 128,
        height: int = 128,
        depth: int = 128,
        resolution: int = 512,
        channels: int = 4
    ):
        """
        Initialize LayeredWaveCube with three independent layers.

        Args:
            proto_unity_config: Configuration for proto_unity layer
            experiential_config: Configuration for experiential layer
            io_config: Configuration for io layer
            width: Grid width (X dimension)
            height: Grid height (Y dimension)
            depth: Grid depth (Z dimension)
            resolution: Default wavetable resolution
            channels: Number of channels (4 for XYZW quaternions)
        """
        self.width = width
        self.height = height
        self.depth = depth
        self.resolution = resolution
        self.channels = channels

        # Initialize layers and metadata
        self.layers = self._initialize_layers(
            proto_unity_config,
            experiential_config,
            io_config
        )
        self.layer_metadata = self._initialize_metadata()
        self.stats = self._initialize_stats()

    def _initialize_layers(
        self,
        proto_unity_config: Optional[Dict],
        experiential_config: Optional[Dict],
        io_config: Optional[Dict]
    ) -> Dict[str, WavetableMatrix]:
        """
        Initialize the three WavetableMatrix layers.

        Args:
            proto_unity_config: Proto unity layer config
            experiential_config: Experiential layer config
            io_config: IO layer config

        Returns:
            Dict mapping layer names to WavetableMatrix instances
        """
        proto_cfg = {**DEFAULT_PROTO_UNITY_CONFIG, **(proto_unity_config or {})}
        exp_cfg = {**DEFAULT_EXPERIENTIAL_CONFIG, **(experiential_config or {})}
        io_cfg = {**DEFAULT_IO_CONFIG, **(io_config or {})}

        return {
            'proto_unity': self._create_layer('proto_unity', proto_cfg),
            'experiential': self._create_layer('experiential', exp_cfg),
            'io': self._create_layer('io', io_cfg)
        }

    def _initialize_metadata(self) -> Dict[str, Dict[Tuple[int, int, int], Dict[str, Any]]]:
        """
        Initialize layer metadata storage.

        Returns:
            Dict mapping layer names to coordinate->metadata dicts
        """
        return {
            'proto_unity': {},
            'experiential': {},
            'io': {}
        }

    def _initialize_stats(self) -> Dict[str, Any]:
        """
        Initialize statistics tracking.

        Returns:
            Stats dictionary
        """
        return {
            'total_nodes': 0,
            'queries': 0,
            'layer_accesses': {'proto_unity': 0, 'experiential': 0, 'io': 0}
        }

    def _create_layer(self, layer_name: str, config: Dict) -> WavetableMatrix:
        """
        Create a WavetableMatrix layer with configuration.

        Args:
            layer_name: Name of the layer
            config: Layer configuration

        Returns:
            Configured WavetableMatrix instance
        """
        compression = config.get('compression', 'gaussian')

        return WavetableMatrix(
            width=self.width,
            height=self.height,
            depth=self.depth,
            resolution=self.resolution,
            channels=self.channels,
            sparse=True,
            compression=compression
        )

    def set_node(
        self,
        x: int, y: int, z: int,
        wavetable: np.ndarray,
        layer: LayerType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store wavetable in specific layer with metadata.

        Args:
            x, y, z: Grid coordinates
            wavetable: Wavetable array (H, W, C)
            layer: Target layer ('proto_unity', 'experiential', 'io')
            metadata: Optional metadata dict

        Raises:
            ValueError: If layer name is invalid
        """
        if layer not in self.layers:
            raise ValueError(f"Invalid layer '{layer}'. Must be one of: {list(self.layers.keys())}")

        # Store in the target layer
        layer_matrix = self.layers[layer]
        layer_matrix.set_node(x, y, z, wavetable, metadata=metadata)

        # Initialize layer metadata for this node
        coords = (x, y, z)
        if coords not in self.layer_metadata[layer]:
            self.layer_metadata[layer][coords] = {
                'resonance': 0.5,
                'access_count': 0,
                'timestamp': time.time()
            }
            self.stats['total_nodes'] += 1

        # Merge additional metadata if provided
        if metadata:
            self.layer_metadata[layer][coords].update(metadata)

    def get_node(
        self,
        x: int, y: int, z: int,
        layer: Optional[LayerType] = None
    ) -> Optional[np.ndarray]:
        """
        Retrieve wavetable from layer (or search all if layer=None).

        Args:
            x, y, z: Grid coordinates
            layer: Specific layer or None to search all layers

        Returns:
            Wavetable array or None if not found
        """
        coords = (x, y, z)

        if layer is not None:
            # Get from specific layer
            if layer not in self.layers:
                raise ValueError(f"Invalid layer '{layer}'")

            result = self.layers[layer].get_node(x, y, z)

            if result is not None:
                self._update_access(coords, layer)

            return result
        else:
            # Search all layers in priority order: io -> experiential -> proto_unity
            search_order: list[LayerType] = ['io', 'experiential', 'proto_unity']

            for search_layer in search_order:
                result = self.layers[search_layer].get_node(x, y, z)

                if result is not None:
                    self._update_access(coords, search_layer)
                    return result

            return None

    def get_layer(self, layer: LayerType) -> WavetableMatrix:
        """
        Get underlying WavetableMatrix for specific layer.

        Args:
            layer: Layer name

        Returns:
            WavetableMatrix instance

        Raises:
            ValueError: If layer name is invalid
        """
        if layer not in self.layers:
            raise ValueError(f"Invalid layer '{layer}'. Must be one of: {list(self.layers.keys())}")

        return self.layers[layer]

    def has_node(self, x: int, y: int, z: int, layer: LayerType) -> bool:
        """
        Check if node exists in specific layer.

        Args:
            x, y, z: Grid coordinates
            layer: Layer name

        Returns:
            True if node exists in layer

        Raises:
            ValueError: If layer name is invalid
        """
        if layer not in self.layers:
            raise ValueError(f"Invalid layer '{layer}'")

        return self.layers[layer].has_node(x, y, z)

    def remove_node(
        self,
        x: int, y: int, z: int,
        layer: LayerType
    ) -> bool:
        """
        Remove node from specific layer.

        Args:
            x, y, z: Grid coordinates
            layer: Layer name

        Returns:
            True if node was removed, False if it didn't exist

        Raises:
            ValueError: If layer name is invalid
        """
        if layer not in self.layers:
            raise ValueError(f"Invalid layer '{layer}'")

        coords = (x, y, z)

        # Check if node exists
        if not self.layers[layer].has_node(x, y, z):
            return False

        # Remove from underlying matrix
        self.layers[layer].delete_node(x, y, z)

        # Remove metadata
        if coords in self.layer_metadata[layer]:
            del self.layer_metadata[layer][coords]
            self.stats['total_nodes'] -= 1

        return True

    def update_resonance(
        self,
        x: int, y: int, z: int,
        layer: LayerType,
        resonance: float
    ) -> None:
        """
        Update resonance strength in layer metadata.

        Args:
            x, y, z: Grid coordinates
            layer: Layer name
            resonance: New resonance value

        Raises:
            ValueError: If layer name is invalid or node doesn't exist
        """
        if layer not in self.layers:
            raise ValueError(f"Invalid layer '{layer}'")

        coords = (x, y, z)

        if coords not in self.layer_metadata[layer]:
            raise ValueError(f"Node at {coords} does not exist in layer '{layer}'")

        self.layer_metadata[layer][coords]['resonance'] = resonance

    def clear_layer(self, layer: LayerType) -> None:
        """
        Clear all nodes in specific layer.

        Args:
            layer: Layer name

        Raises:
            ValueError: If layer name is invalid
        """
        if layer not in self.layers:
            raise ValueError(f"Invalid layer '{layer}'")

        # Clear the matrix
        num_nodes = len(self.layer_metadata[layer])
        self.layers[layer].clear()

        # Clear metadata
        self.layer_metadata[layer].clear()

        # Update stats
        self.stats['total_nodes'] -= num_nodes

    def query_with_interference(
        self,
        fx: float,
        fy: float,
        fz: float
    ) -> Optional[np.ndarray]:
        """
        Query with standing wave interference across layers.

        Note: This is a stub for now. Full implementation will be provided
        by StandingWaveInterference in a separate module.

        Args:
            fx, fy, fz: Fractional coordinates

        Returns:
            None (stub implementation)
        """
        self.stats['queries'] += 1
        # Stub: Will be implemented by StandingWaveInterference
        return None

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Return memory usage per layer.

        Returns:
            Dict with per-layer memory statistics
        """
        usage = {}

        for layer_name, layer_matrix in self.layers.items():
            layer_usage = layer_matrix.get_memory_usage()
            usage[layer_name] = layer_usage

        # Add total across all layers
        total_bytes = sum(u['total_bytes'] for u in usage.values())
        usage['total'] = {
            'total_bytes': total_bytes,
            'total_kb': total_bytes / 1024,
            'total_mb': total_bytes / (1024 ** 2)
        }

        return usage

    def get_layer_stats(self) -> Dict[str, Any]:
        """
        Return statistics per layer (node count, avg resonance, etc).

        Returns:
            Dict with per-layer statistics
        """
        stats = {}

        for layer_name in self.layers:
            metadata = self.layer_metadata[layer_name]
            node_count = len(metadata)

            if node_count > 0:
                avg_resonance = sum(m['resonance'] for m in metadata.values()) / node_count
                avg_access = sum(m['access_count'] for m in metadata.values()) / node_count
                total_accesses = sum(m['access_count'] for m in metadata.values())
            else:
                avg_resonance = 0.0
                avg_access = 0.0
                total_accesses = 0

            stats[layer_name] = {
                'node_count': node_count,
                'avg_resonance': avg_resonance,
                'avg_access_count': avg_access,
                'total_accesses': total_accesses
            }

        return stats

    def _update_access(self, coords: Tuple[int, int, int], layer: str) -> None:
        """
        Update access metadata for a node.

        Args:
            coords: Node coordinates
            layer: Layer name
        """
        if coords in self.layer_metadata[layer]:
            self.layer_metadata[layer][coords]['access_count'] += 1
            self.layer_metadata[layer][coords]['timestamp'] = time.time()

        self.stats['layer_accesses'][layer] += 1

    def __repr__(self) -> str:
        """String representation."""
        layer_stats = self.get_layer_stats()
        memory = self.get_memory_usage()

        proto_count = layer_stats['proto_unity']['node_count']
        exp_count = layer_stats['experiential']['node_count']
        io_count = layer_stats['io']['node_count']

        total_mb = memory['total']['total_mb']

        return (
            f"LayeredWaveCube(\n"
            f"  Grid: {self.width}×{self.height}×{self.depth}\n"
            f"  Resolution: {self.resolution}×{self.resolution}×{self.channels}\n"
            f"  Layers:\n"
            f"    proto_unity: {proto_count} nodes\n"
            f"    experiential: {exp_count} nodes\n"
            f"    io: {io_count} nodes\n"
            f"  Total Memory: {total_mb:.2f} MB\n"
            f")"
        )
