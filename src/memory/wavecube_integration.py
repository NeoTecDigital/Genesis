"""
WaveCube integration for Genesis memory hierarchy.

Bridges the LayeredWaveCube system with VoxelCloud storage for efficient
multi-layer memory management.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Any, Literal
import sys
import os

# Add lib/wavecube to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../lib/wavecube'))

from lib.wavecube.core.layered_matrix import LayeredWaveCube, LayerType
from lib.wavecube.core.layer_manager import LayerManager
from lib.wavecube.spatial.interference import StandingWaveInterference, InterferenceMode

from src.memory.voxel_cloud import VoxelCloud, ProtoIdentityEntry


class WaveCubeMemoryBridge:
    """
    Bridge between LayeredWaveCube and VoxelCloud memory systems.

    Maps VoxelCloud entries to WaveCube layers with automatic management
    of promotion, demotion, and interference patterns.
    """

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        depth: int = 128,
        use_wavecube: bool = True,
        enable_auto_management: bool = True
    ):
        """
        Initialize WaveCube memory bridge.

        Args:
            width: Proto-identity width
            height: Proto-identity height
            depth: Voxel cloud depth
            use_wavecube: Enable WaveCube storage backend
            enable_auto_management: Enable automatic layer management
        """
        self.width = width
        self.height = height
        self.depth = depth
        self.use_wavecube = use_wavecube
        self.enable_auto_management = enable_auto_management

        if use_wavecube:
            # Initialize LayeredWaveCube with Genesis-optimized settings
            self.wavecube = LayeredWaveCube(
                proto_unity_config={
                    'chunk_size': (16, 16, 16),
                    'cache_radius': 0,
                    'compression': 'gaussian',
                    'quality': 0.98
                },
                experiential_config={
                    'chunk_size': (64, 64, 64),
                    'cache_radius': 2,
                    'compression': 'gaussian',
                    'quality': 0.90
                },
                io_config={
                    'chunk_size': (32, 32, 32),
                    'cache_radius': 1,
                    'compression': 'gaussian',
                    'quality': 0.85
                },
                resolution=width,
                channels=4  # XYZW quaternions
            )

            # Initialize layer manager for automatic transitions
            if enable_auto_management:
                self.layer_manager = LayerManager(
                    self.wavecube,
                    promotion_config={
                        'resonance_threshold': 0.8,
                        'access_threshold': 10,
                        'check_interval': 50
                    },
                    demotion_config={
                        'access_threshold': 2,
                        'time_threshold': 1000,
                        'check_interval': 100
                    },
                    eviction_config={
                        'memory_threshold_mb': 900.0,
                        'resonance_threshold': 0.3
                    }
                )
            else:
                self.layer_manager = None

            # Initialize interference system
            self.interference = StandingWaveInterference(
                carrier_weight=1.0,
                modulation_weight=0.5,
                io_weight=0.3,
                phase_coherence=0.9
            )
        else:
            self.wavecube = None
            self.layer_manager = None
            self.interference = None

        # Mapping from VoxelCloud entries to WaveCube coordinates
        self.entry_to_coords: Dict[int, Tuple[int, int, int]] = {}
        self.coords_to_entry: Dict[Tuple[int, int, int], ProtoIdentityEntry] = {}

        # Statistics
        self.stats = {
            'entries_stored': 0,
            'entries_retrieved': 0,
            'layer_transitions': 0,
            'interference_queries': 0
        }

    def store_entry(
        self,
        entry: ProtoIdentityEntry,
        layer: LayerType = 'experiential'
    ) -> Tuple[int, int, int]:
        """
        Store VoxelCloud entry in WaveCube layer.

        Args:
            entry: ProtoIdentityEntry from VoxelCloud
            layer: Target layer ('proto_unity', 'experiential', 'io')

        Returns:
            WaveCube coordinates (x, y, z)
        """
        if not self.use_wavecube or self.wavecube is None:
            return (0, 0, 0)

        # Extract voxel position from proto-identity
        voxel_pos = self._extract_voxel_position(entry.proto_identity)

        # Convert to integer coordinates for WaveCube
        x = int(voxel_pos[0] * self.depth / self.width)
        y = int(voxel_pos[1] * self.depth / self.height)
        z = int(voxel_pos[2])

        # Ensure coordinates are in bounds
        x = max(0, min(x, 127))
        y = max(0, min(y, 127))
        z = max(0, min(z, 127))

        # Prepare metadata
        metadata = {
            'text_hash': entry.metadata.get('text_hash', ''),
            'octave': entry.metadata.get('octave', 0),
            'resonance': entry.resonance_strength,
            'frequency_band': entry.metadata.get('frequency_band', None)
        }

        # Store in WaveCube
        self.wavecube.set_node(
            x, y, z,
            entry.proto_identity,
            layer=layer,
            metadata=metadata
        )

        # Update resonance in layer metadata
        self.wavecube.update_resonance(x, y, z, layer, entry.resonance_strength)

        # Track mapping
        entry_id = id(entry)
        self.entry_to_coords[entry_id] = (x, y, z)
        self.coords_to_entry[(x, y, z)] = entry

        # Notify layer manager
        if self.layer_manager:
            self.layer_manager.on_store(x, y, z, layer, entry.resonance_strength)

        self.stats['entries_stored'] += 1
        return (x, y, z)

    def retrieve_entry(
        self,
        x: int, y: int, z: int,
        layer: Optional[LayerType] = None
    ) -> Optional[ProtoIdentityEntry]:
        """
        Retrieve entry from WaveCube.

        Args:
            x, y, z: WaveCube coordinates
            layer: Specific layer or None to search all

        Returns:
            ProtoIdentityEntry or None
        """
        if not self.use_wavecube or self.wavecube is None:
            return None

        # Get from WaveCube
        proto = self.wavecube.get_node(x, y, z, layer=layer)

        if proto is None:
            return None

        # Check if we have the original entry
        if (x, y, z) in self.coords_to_entry:
            entry = self.coords_to_entry[(x, y, z)]

            # Notify layer manager of access
            if self.layer_manager:
                # Determine which layer it was found in
                for check_layer in ['io', 'experiential', 'proto_unity']:
                    if self.wavecube.get_layer(check_layer).has_node(x, y, z):
                        self.layer_manager.on_access(x, y, z, check_layer)
                        break

            self.stats['entries_retrieved'] += 1
            return entry

        # Reconstruct entry from proto-identity
        entry = ProtoIdentityEntry(
            proto_identity=proto,
            frequency=np.zeros((self.height, self.width, 2)),  # Placeholder
            metadata={},
            resonance_strength=0.5
        )

        self.stats['entries_retrieved'] += 1
        return entry

    def query_with_interference(
        self,
        query_proto: np.ndarray,
        max_results: int = 10,
        mode: InterferenceMode = InterferenceMode.MODULATION
    ) -> List[ProtoIdentityEntry]:
        """
        Query using standing wave interference across layers.

        Args:
            query_proto: Query proto-identity
            max_results: Maximum results
            mode: Interference mode

        Returns:
            List of matching entries
        """
        if not self.use_wavecube or self.wavecube is None:
            return []

        # Extract position from query
        query_pos = self._extract_voxel_position(query_proto)

        # Convert to fractional coordinates for sampling
        fx = query_pos[0] * self.depth / self.width
        fy = query_pos[1] * self.depth / self.height
        fz = query_pos[2]

        # Query with interference
        combined = self.wavecube.query_with_interference(fx, fy, fz)

        if combined is None:
            return []

        # Find similar entries by comparing combined result
        results = []
        for coords, entry in self.coords_to_entry.items():
            similarity = self._compute_similarity(combined, entry.proto_identity)

            if similarity > 0.5:  # Threshold
                results.append((similarity, entry))

        # Sort and limit
        results.sort(key=lambda x: x[0], reverse=True)

        self.stats['interference_queries'] += 1
        return [entry for _, entry in results[:max_results]]

    def migrate_voxel_cloud(
        self,
        voxel_cloud: VoxelCloud,
        target_layer: LayerType = 'experiential'
    ) -> int:
        """
        Migrate existing VoxelCloud entries to WaveCube.

        Args:
            voxel_cloud: VoxelCloud instance to migrate
            target_layer: Target layer for migration

        Returns:
            Number of entries migrated
        """
        if not self.use_wavecube:
            return 0

        migrated = 0
        for entry in voxel_cloud.entries:
            coords = self.store_entry(entry, layer=target_layer)
            if coords != (0, 0, 0):
                migrated += 1

        return migrated

    def optimize_layers(self) -> Dict[str, int]:
        """
        Run layer optimization (promotion/demotion/eviction).

        Returns:
            Statistics of operations performed
        """
        if self.layer_manager:
            return self.layer_manager.optimize_layers()

        return {'promoted': 0, 'demoted': 0, 'evicted': 0}

    def get_layer_distribution(self) -> Dict[str, int]:
        """
        Get distribution of entries across layers.

        Returns:
            Count of entries in each layer
        """
        if not self.use_wavecube or self.wavecube is None:
            return {}

        return {
            'proto_unity': len(self.wavecube.layer_metadata['proto_unity']),
            'experiential': len(self.wavecube.layer_metadata['experiential']),
            'io': len(self.wavecube.layer_metadata['io'])
        }

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics.

        Returns:
            Memory usage and statistics
        """
        if not self.use_wavecube or self.wavecube is None:
            return self.stats

        wavecube_stats = self.wavecube.get_memory_usage()
        layer_stats = self.wavecube.get_layer_stats()

        return {
            **self.stats,
            'memory': wavecube_stats,
            'layers': layer_stats,
            'distribution': self.get_layer_distribution()
        }

    def _extract_voxel_position(self, proto: np.ndarray) -> np.ndarray:
        """Extract 3D position from proto-identity."""
        z_channel = proto[:, :, 2]
        h, w = z_channel.shape

        y_coords = np.arange(h).reshape(-1, 1)
        x_coords = np.arange(w).reshape(1, -1)

        total_mass = z_channel.sum()
        if total_mass > 1e-8:
            center_y = (z_channel * y_coords).sum() / total_mass
            center_x = (z_channel * x_coords).sum() / total_mass
        else:
            center_y, center_x = h / 2, w / 2

        center_z = z_channel.mean() * 100.0

        return np.array([center_x, center_y, center_z], dtype=np.float32)

    def _compute_similarity(self, proto1: np.ndarray, proto2: np.ndarray) -> float:
        """Compute similarity between two proto-identities."""
        # Flatten and normalize
        flat1 = proto1.flatten()
        flat2 = proto2.flatten()

        norm1 = np.linalg.norm(flat1)
        norm2 = np.linalg.norm(flat2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Cosine similarity
        similarity = np.dot(flat1, flat2) / (norm1 * norm2)
        return float(similarity)

    def set_active_position(self, x: int, y: int, z: int) -> None:
        """Update active position for chunk loading."""
        if self.wavecube:
            self.wavecube.set_active_position(x, y, z)

    def clear_layer(self, layer: LayerType) -> None:
        """Clear specific layer."""
        if self.wavecube:
            self.wavecube.clear_layer(layer)

            # Clean up mappings
            to_remove = []
            for coords, entry in list(self.coords_to_entry.items()):
                if self.wavecube.get_node(*coords, layer=layer) is None:
                    to_remove.append(coords)

            for coords in to_remove:
                entry = self.coords_to_entry.pop(coords)
                entry_id = id(entry)
                if entry_id in self.entry_to_coords:
                    del self.entry_to_coords[entry_id]

    def __repr__(self) -> str:
        """String representation."""
        if not self.use_wavecube:
            return "WaveCubeMemoryBridge(disabled)"

        stats = self.get_memory_stats()
        return (
            f"WaveCubeMemoryBridge(\n"
            f"  Stored: {stats['entries_stored']}, "
            f"Retrieved: {stats['entries_retrieved']}\n"
            f"  Distribution: {stats.get('distribution', {})}\n"
            f"  Memory: {stats.get('memory', {}).get('total_mb', 0):.2f}MB\n"
            f")"
        )