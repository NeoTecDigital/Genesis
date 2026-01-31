"""
MultiOctaveEncoder with ChunkedWaveCube integration.

This module extends the existing encoder to use ChunkedWaveCube for
efficient storage and retrieval of proto-identities.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from src.pipeline.multi_octave_encoder import MultiOctaveEncoder, OctaveUnit
from lib.wavecube.core.chunked_matrix import ChunkedWaveCube
from lib.wavecube.io.chunk_storage import ChunkStorage
from lib.wavecube.spatial.spatial_index import SpatialIndex
from lib.wavecube.spatial.coordinates import QuaternionicCoord, Modality


class ChunkedMultiOctaveEncoder(MultiOctaveEncoder):
    """
    MultiOctaveEncoder with ChunkedWaveCube storage backend.

    Extends the base encoder to:
    - Store proto-identities in ChunkedWaveCube
    - Use spatial indexing for O(k) queries
    - Persist chunks to disk
    - Maintain backward compatibility
    """

    def __init__(
        self,
        carrier: np.ndarray,
        width: int = 512,
        height: int = 512,
        layers: int = 4,
        chunk_size: Tuple[int, int, int] = (16, 16, 16),
        cache_dir: Optional[Path] = None,
        enable_persistence: bool = True
    ):
        """
        Initialize chunked encoder.

        Args:
            carrier: Carrier (kept for compatibility)
            width: Proto-identity width
            height: Proto-identity height
            layers: Number of layers (4 for XYZW)
            chunk_size: Size of chunks for spatial partitioning
            cache_dir: Directory for chunk persistence
            enable_persistence: Enable disk persistence
        """
        super().__init__(carrier, width, height, layers)

        # Initialize ChunkedWaveCube
        self.cube = ChunkedWaveCube(
            chunk_size=chunk_size,
            resolution=max(width, height),
            channels=layers,
            cache_radius=2,
            compression='gaussian'
        )

        # Initialize spatial index
        self.spatial_index = SpatialIndex(
            chunk_size=chunk_size,
            cache_radius=2,
            query_cache_size=100
        )

        # Initialize storage if persistence enabled
        self.storage = None
        if enable_persistence:
            if cache_dir is None:
                cache_dir = Path.home() / '.cache' / 'genesis' / 'chunks'

            self.storage = ChunkStorage(
                cache_dir=cache_dir,
                max_threads=4,
                auto_save=True
            )

        # Track proto locations for migration
        self.proto_locations: Dict[int, QuaternionicCoord] = {}
        self.next_position = [0, 0, 0]  # Auto-incrementing position

    def encode_and_store(
        self,
        text: str,
        octaves: List[int] = [4, 0],
        modality: Modality = Modality.TEXT,
        base_position: Optional[Tuple[int, int, int]] = None
    ) -> List[Tuple[OctaveUnit, QuaternionicCoord]]:
        """
        Encode text and store in ChunkedWaveCube.

        Args:
            text: Input text
            octaves: Octave levels to encode
            modality: Modality for phase-locking
            base_position: Optional base position for storage

        Returns:
            List of (OctaveUnit, QuaternionicCoord) tuples
        """
        # Encode at multiple octaves
        units = self.encode_text_hierarchical(text, octaves)

        results = []
        for unit in units:
            # Determine storage position
            if base_position:
                x, y, z = base_position
            else:
                x, y, z = self._get_next_position(unit.octave)

            # Create quaternionic coordinate with modality phase
            coord = QuaternionicCoord.from_modality(x, y, z, modality)

            # Store in ChunkedWaveCube
            metadata = {
                'octave': unit.octave,
                'modality': modality.name,
                'text_hash': hash(text),
                'timestamp': np.datetime64('now')
            }

            self.cube.set_node(
                coord.x, coord.y, coord.z,
                unit.proto_identity,
                metadata=metadata
            )

            # Update spatial index
            chunk_coords = self.cube._world_to_chunk(coord.x, coord.y, coord.z)
            self.spatial_index.add_chunk(
                chunk_coords,
                node_count=1,
                is_loaded=True
            )

            # Track location
            proto_id = id(unit)
            self.proto_locations[proto_id] = coord

            results.append((unit, coord))

        # Persist if enabled
        if self.storage:
            self._persist_dirty_chunks()

        return results

    def query_similar(
        self,
        query_proto: np.ndarray,
        k: int = 10,
        max_distance: float = 100.0,
        similarity_threshold: float = 0.9
    ) -> List[Tuple[np.ndarray, float, Dict[str, Any]]]:
        """
        Find similar proto-identities using spatial indexing.

        Args:
            query_proto: Query proto-identity
            k: Number of neighbors
            max_distance: Maximum search distance
            similarity_threshold: Minimum similarity

        Returns:
            List of (proto, similarity, metadata) tuples
        """
        # First, find approximate position based on proto features
        query_pos = self._estimate_position_from_proto(query_proto)

        # Node getter for spatial index
        def get_nodes(chunk_coords):
            if chunk_coords not in self.cube.chunks:
                return []

            chunk = self.cube._ensure_chunk_loaded(chunk_coords)
            nodes = []

            for local_coords in chunk.get_populated_nodes():
                # Convert to world coordinates
                wx = chunk_coords[0] * self.cube.chunk_size[0] + local_coords[0]
                wy = chunk_coords[1] * self.cube.chunk_size[1] + local_coords[1]
                wz = chunk_coords[2] * self.cube.chunk_size[2] + local_coords[2]
                world_pos = (wx, wy, wz)

                # Get node data
                node_data = chunk.get_node(*local_coords)
                if node_data is not None:
                    nodes.append((world_pos, node_data))

            return nodes

        # Perform k-NN query
        query_results = self.spatial_index.knn_query(
            query_pos,
            k=k * 2,  # Get more candidates for similarity filtering
            max_distance=max_distance,
            node_getter=get_nodes
        )

        # Filter by similarity
        results = []
        for result in query_results:
            proto = result.data

            # Calculate similarity
            similarity = self._calculate_similarity(query_proto, proto)

            if similarity >= similarity_threshold:
                # Get metadata from cube
                pos = result.position
                chunk_coords = self.cube._world_to_chunk(int(pos[0]), int(pos[1]), int(pos[2]))
                if chunk_coords in self.cube.chunks:
                    chunk = self.cube.chunks[chunk_coords]
                    local_coords = self.cube._world_to_local(int(pos[0]), int(pos[1]), int(pos[2]))
                    node = chunk.nodes.get(local_coords)
                    metadata = node.metadata if node else {}
                else:
                    metadata = {}

                results.append((proto, similarity, metadata))

        # Sort by similarity and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def _get_next_position(self, octave: int) -> Tuple[int, int, int]:
        """Get next available position for storing proto."""
        # Distribute by octave level
        z_offset = (octave + 4) * 10  # Map octave to Z level

        x = self.next_position[0]
        y = self.next_position[1]
        z = z_offset

        # Increment position (grid pattern)
        self.next_position[0] += 10
        if self.next_position[0] >= 100:
            self.next_position[0] = 0
            self.next_position[1] += 10
            if self.next_position[1] >= 100:
                self.next_position[1] = 0
                self.next_position[2] += 1

        return (x, y, z)

    def _estimate_position_from_proto(self, proto: np.ndarray) -> Tuple[float, float, float]:
        """Estimate spatial position from proto features."""
        # Simple heuristic: use center of mass of proto energy
        energy = np.sum(proto, axis=2)  # Sum across channels

        # Find center of mass
        total = np.sum(energy)
        if total > 0:
            y_coords, x_coords = np.ogrid[:proto.shape[0], :proto.shape[1]]
            cx = np.sum(x_coords * energy) / total
            cy = np.sum(y_coords * energy) / total

            # Map to spatial coordinates
            x = (cx / proto.shape[1]) * 100
            y = (cy / proto.shape[0]) * 100
            z = 0

            return (x, y, z)

        return (50, 50, 0)  # Default center

    def _calculate_similarity(self, proto1: np.ndarray, proto2: np.ndarray) -> float:
        """Calculate similarity between proto-identities."""
        # Flatten and normalize
        p1_flat = proto1.flatten()
        p2_flat = proto2.flatten()

        p1_norm = p1_flat / (np.linalg.norm(p1_flat) + 1e-8)
        p2_norm = p2_flat / (np.linalg.norm(p2_flat) + 1e-8)

        # Cosine similarity
        return float(np.dot(p1_norm, p2_norm))

    def _persist_dirty_chunks(self):
        """Save dirty chunks to disk."""
        if not self.storage:
            return

        for chunk_coords, chunk in self.cube.chunks.items():
            if hasattr(chunk, '_is_dirty') and chunk._is_dirty:
                # Prepare chunk data for storage
                chunk_data = {
                    'node_indices': list(chunk.nodes.keys()),
                    'node_data': {},
                    'metadata': {}
                }

                for coords, node in chunk.nodes.items():
                    key = f"{coords[0]}_{coords[1]}_{coords[2]}"
                    chunk_data['node_data'][key] = node.data
                    chunk_data['metadata'][key] = node.metadata

                # Save async
                self.storage.save_chunk(chunk_coords, chunk_data, async_save=True)
                chunk._is_dirty = False

    def migrate_from_voxel_cloud(
        self,
        voxel_cloud,
        batch_size: int = 100
    ) -> int:
        """
        Migrate existing proto-identities from VoxelCloud to ChunkedWaveCube.

        Args:
            voxel_cloud: VoxelCloud instance
            batch_size: Number of entries to migrate at once

        Returns:
            Number of entries migrated
        """
        migrated = 0

        for i, entry in enumerate(voxel_cloud.entries):
            # Get position for this entry
            x, y, z = self._get_next_position(entry.metadata.get('octave', 0))

            # Create coordinate
            modality_str = entry.metadata.get('modality', 'TEXT')
            modality = Modality[modality_str] if modality_str in Modality.__members__ else Modality.TEXT
            coord = QuaternionicCoord.from_modality(x, y, z, modality)

            # Store in ChunkedWaveCube
            self.cube.set_node(
                coord.x, coord.y, coord.z,
                entry.proto_identity,
                metadata=entry.metadata
            )

            # Update voxel cloud with reference
            voxel_cloud.set_wavecube_reference(i, coord.to_tuple())

            migrated += 1

            # Periodic persistence
            if migrated % batch_size == 0:
                self._persist_dirty_chunks()
                print(f"Migrated {migrated}/{len(voxel_cloud.entries)} entries...")

        # Final persistence
        self._persist_dirty_chunks()

        print(f"Migration complete: {migrated} entries migrated to ChunkedWaveCube")
        return migrated

    def get_statistics(self) -> Dict[str, Any]:
        """Get encoder statistics."""
        cube_stats = self.cube.get_memory_usage()
        index_stats = self.spatial_index.get_stats()

        stats = {
            'cube': cube_stats,
            'index': index_stats,
            'proto_count': len(self.proto_locations),
            'storage': None
        }

        if self.storage:
            stats['storage'] = self.storage.get_summary_stats()

        return stats

    def shutdown(self):
        """Clean shutdown with persistence."""
        if self.storage:
            print("Persisting remaining chunks...")
            self._persist_dirty_chunks()
            self.storage.wait_pending(timeout=10.0)
            self.storage.shutdown()

        print(f"Encoder shutdown complete. Stats: {self.get_statistics()}")