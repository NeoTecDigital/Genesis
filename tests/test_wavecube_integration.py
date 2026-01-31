"""
Test WaveCube-Genesis integration (Phase 1).

Tests:
1. ChunkedWaveCube load/unload behavior
2. Quaternionic coordinate system
3. VoxelCloud WaveCube references
4. Basic encode → store → query → decode flow
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
# import pytest  # Not needed for basic test run
from pathlib import Path

# WaveCube imports
from lib.wavecube.core.chunked_matrix import ChunkedWaveCube
from lib.wavecube.spatial.coordinates import QuaternionicCoord, Modality
from lib.wavecube.spatial.phase_locking import (
    phase_shift, find_phase_locked, cross_modal_bind
)

# Genesis imports
from src.memory.voxel_cloud import VoxelCloud, ProtoIdentityEntry
from src.pipeline.multi_octave_encoder import MultiOctaveEncoder
from src.pipeline.multi_octave_decoder import MultiOctaveDecoder


class TestChunkedWaveCube:
    """Test ChunkedWaveCube functionality."""

    def test_chunk_creation(self):
        """Test basic chunk creation and management."""
        cube = ChunkedWaveCube(
            chunk_size=(8, 8, 8),
            resolution=256,
            channels=4
        )

        # Create a test wavetable
        wavetable = np.random.randn(256, 256, 4).astype(np.float32)

        # Set node in chunk (0, 0, 0)
        cube.set_node(1, 2, 3, wavetable)

        # Verify chunk was created
        assert len(cube.chunks) == 1
        assert (0, 0, 0) in cube.chunks
        assert cube.stats['chunks_total'] == 1
        assert cube.stats['total_nodes'] == 1

        # Retrieve node
        retrieved = cube.get_node(1, 2, 3)
        assert retrieved is not None
        assert np.allclose(retrieved, wavetable)

    def test_chunk_loading_unloading(self):
        """Test dynamic chunk loading/unloading."""
        cube = ChunkedWaveCube(
            chunk_size=(4, 4, 4),
            cache_radius=1,
            compression=None  # Disable compression for this test
        )

        # Create nodes in different chunks
        wavetable1 = np.random.randn(512, 512, 4).astype(np.float32)
        wavetable2 = np.random.randn(512, 512, 4).astype(np.float32)

        # Chunk (0,0,0)
        cube.set_node(1, 1, 1, wavetable1)

        # Chunk (2,2,2) - far away
        cube.set_node(10, 10, 10, wavetable2)

        # Both chunks should be loaded initially
        assert cube.stats['chunks_loaded'] == 2

        # Set active position to chunk (0,0,0)
        cube.set_active_position(1, 1, 1)

        # Chunk (2,2,2) should be offloaded
        assert cube.stats['chunks_loaded'] == 1
        assert (0, 0, 0) in cube.active_chunks
        assert (2, 2, 2) not in cube.active_chunks

        # Move to chunk (2,2,2)
        cube.set_active_position(10, 10, 10)

        # Now chunk (0,0,0) should be offloaded
        assert cube.stats['chunks_loaded'] == 1
        assert (2, 2, 2) in cube.active_chunks
        assert (0, 0, 0) not in cube.active_chunks

        # Verify we can still retrieve from offloaded chunk
        retrieved = cube.get_node(1, 1, 1)
        assert retrieved is not None
        # Without compression, should be exact
        assert np.allclose(retrieved, wavetable1)

    def test_adjacent_chunk_caching(self):
        """Test that adjacent chunks remain cached."""
        cube = ChunkedWaveCube(
            chunk_size=(4, 4, 4),
            cache_radius=1
        )

        # Create nodes in adjacent chunks
        for x in [0, 4, 8]:
            for y in [0, 4]:
                for z in [0]:
                    wavetable = np.random.randn(256, 256, 4).astype(np.float32)
                    cube.set_node(x, y, z, wavetable)

        # Set active position to center chunk
        cube.set_active_position(4, 2, 2)

        # Check which chunks are loaded
        # With radius=1, chunks (0,0,0), (1,0,0), (2,0,0), (0,1,0), (1,1,0) should be loaded
        # Since our nodes are at x=0,4,8 and y=0,4, chunks would be:
        # (0,0,0), (1,0,0), (2,0,0), (0,1,0), (1,1,0)
        assert cube.stats['chunks_loaded'] >= 3  # At least center and some adjacent


class TestQuaternionicCoordinates:
    """Test quaternionic coordinate system."""

    def test_coordinate_creation(self):
        """Test creating quaternionic coordinates."""
        # Direct creation
        coord1 = QuaternionicCoord(10, 20, 30, 45.0)
        assert coord1.x == 10
        assert coord1.y == 20
        assert coord1.z == 30
        assert coord1.w == 45.0

        # From modality
        coord2 = QuaternionicCoord.from_modality(5, 10, 15, Modality.AUDIO)
        assert coord2.w == 90.0

        # Phase normalization
        coord3 = QuaternionicCoord(1, 2, 3, 400.0)
        assert coord3.w == 40.0  # 400 % 360

    def test_distance_metrics(self):
        """Test distance calculations."""
        coord1 = QuaternionicCoord(0, 0, 0, 0.0)
        coord2 = QuaternionicCoord(3, 4, 0, 90.0)

        # Spatial distance (3-4-5 triangle)
        assert coord1.spatial_distance(coord2) == 5.0

        # Phase distance
        assert coord1.phase_distance(coord2) == 90.0

        # Total distance
        total = coord1.total_distance(coord2, phase_weight=1.0/360.0)
        expected = 5.0 + 90.0/360.0  # 5.25
        assert abs(total - expected) < 0.001

    def test_phase_operations(self):
        """Test phase shifting and locking."""
        coord = QuaternionicCoord(10, 20, 30, 0.0)

        # Phase shift
        shifted = phase_shift(coord, 90.0)
        assert shifted.w == 90.0
        assert shifted.x == 10  # Spatial coords unchanged

        # Find phase-locked position
        locked = find_phase_locked(coord, Modality.IMAGE, search_radius=2)
        assert locked.w == 180.0

    def test_cross_modal_binding(self):
        """Test binding multiple modalities."""
        # Create test proto-identities
        proto1 = np.random.randn(256, 256, 4).astype(np.float32)
        proto2 = np.random.randn(256, 256, 4).astype(np.float32)
        proto3 = np.random.randn(256, 256, 4).astype(np.float32)

        protos = [proto1, proto2, proto3]
        modalities = [Modality.TEXT, Modality.AUDIO, Modality.IMAGE]

        # Bind them
        coords = cross_modal_bind(protos, modalities, base_position=(10, 10, 10))

        # Check phase offsets
        assert len(coords) == 3
        assert coords[0].w == 0.0    # TEXT
        assert coords[1].w == 90.0   # AUDIO
        assert coords[2].w == 180.0  # IMAGE

        # All should be near base position
        for coord in coords:
            distance = coord.spatial_distance(QuaternionicCoord(10, 10, 10, 0))
            assert distance <= 3  # Within search radius


class TestVoxelCloudIntegration:
    """Test VoxelCloud with WaveCube references."""

    def test_wavecube_reference_storage(self):
        """Test storing and retrieving WaveCube references."""
        voxel = VoxelCloud(width=512, height=512, depth=128)

        # Create a test proto-identity
        proto = np.random.randn(512, 512, 4).astype(np.float32)
        freq = np.random.randn(512, 512, 2).astype(np.float32)
        metadata = {"text": "test", "octave": 0}

        # Add to voxel cloud
        voxel.add(proto, freq, metadata)

        # Set WaveCube reference
        wavecube_coords = (10, 20, 30, 45.0)
        voxel.set_wavecube_reference(0, wavecube_coords)

        # Retrieve reference
        retrieved = voxel.get_wavecube_reference(0)
        assert retrieved == wavecube_coords

        # Check entry has reference
        entry = voxel.entries[0]
        assert entry.wavecube_coords == wavecube_coords


class TestFullIntegrationFlow:
    """Test complete encode → store → query → decode flow."""

    def test_basic_flow(self):
        """Test basic integration flow."""
        # Initialize components
        cube = ChunkedWaveCube(
            chunk_size=(16, 16, 16),
            resolution=512,
            channels=4,
            compression='gaussian'
        )

        voxel = VoxelCloud(width=512, height=512, depth=128)

        # Create encoder/decoder
        # Note: carrier is not actually used but kept for API compatibility
        dummy_carrier = np.zeros((512, 512, 2), dtype=np.float32)
        encoder = MultiOctaveEncoder(carrier=dummy_carrier, width=512, height=512)
        decoder = MultiOctaveDecoder(carrier=dummy_carrier)

        # Test text
        test_text = "Hello, WaveCube integration!"

        # Encode at multiple octaves
        units = encoder.encode_text_hierarchical(test_text, octaves=[4, 0])

        # Process first unit (should be character level from octave +4)
        if len(units) > 0:
            unit = units[0]  # First unit
            proto = unit.proto_identity
            freq = unit.frequency

            # Add to VoxelCloud
            metadata = {"octave": unit.octave, "modality": "text"}
            voxel.add(proto, freq, metadata)

            # Store in WaveCube
            coord = QuaternionicCoord.from_modality(5, 5, 5, Modality.TEXT)
            cube.set_node(coord.x, coord.y, coord.z, proto, metadata=metadata)

            # Store reference in VoxelCloud
            voxel.set_wavecube_reference(0, coord.to_tuple())

            # Query: Retrieve from WaveCube via VoxelCloud reference
            ref = voxel.get_wavecube_reference(0)
            assert ref is not None

            x, y, z, w = ref
            retrieved_proto = cube.get_node(int(x), int(y), int(z))
            assert retrieved_proto is not None

            # Verify similarity (should be identical or very close with compression)
            similarity = np.dot(
                proto.flatten() / np.linalg.norm(proto.flatten()),
                retrieved_proto.flatten() / np.linalg.norm(retrieved_proto.flatten())
            )
            assert similarity > 0.95, f"Low similarity: {similarity}"

            print(f"✓ Basic flow test passed - similarity: {similarity:.4f}")

    def test_multi_octave_storage(self):
        """Test storing multiple octave levels."""
        cube = ChunkedWaveCube(chunk_size=(8, 8, 8))
        voxel = VoxelCloud()
        dummy_carrier = np.zeros((512, 512, 2), dtype=np.float32)
        encoder = MultiOctaveEncoder(carrier=dummy_carrier, width=512, height=512)

        test_text = "Multi-octave test"
        units = encoder.encode_text_hierarchical(test_text, octaves=[4, 0, -2])

        stored_count = 0

        # Store protos from different octaves (limit to first few)
        for i, unit in enumerate(units[:6]):
            # Create coordinates at different Z levels per octave
            z_level = 10 + unit.octave * 2
            coord = QuaternionicCoord(i * 10, i * 10, max(0, z_level), 0.0)

            # Store in WaveCube
            cube.set_node(coord.x, coord.y, coord.z, unit.proto_identity)

            # Add to VoxelCloud with reference
            metadata = {"octave": unit.octave}
            voxel.add(unit.proto_identity, unit.frequency, metadata)
            voxel.set_wavecube_reference(stored_count, coord.to_tuple())

            stored_count += 1

        # Verify storage
        assert stored_count > 0
        assert cube.stats['total_nodes'] == stored_count
        assert len(voxel.entries) == stored_count

        print(f"✓ Multi-octave test passed - stored {stored_count} protos")

    def test_compression_efficiency(self):
        """Test compression efficiency with WaveCube."""
        cube = ChunkedWaveCube(
            chunk_size=(8, 8, 8),
            compression='gaussian'
        )

        # Create and store multiple protos
        num_protos = 10
        for i in range(num_protos):
            proto = np.random.randn(512, 512, 4).astype(np.float32)
            cube.set_node(i * 2, i * 2, i, proto)

        # Get baseline memory
        baseline_mem = cube.get_memory_usage()

        # Offload all chunks
        cube.offload_all_inactive()

        # Get compressed memory
        compressed_mem = cube.get_memory_usage()

        # Calculate compression ratio
        if baseline_mem['total_bytes'] > 0:
            ratio = baseline_mem['total_bytes'] / compressed_mem['total_bytes']
            print(f"✓ Compression test - ratio: {ratio:.2f}x")
            print(f"  Baseline: {baseline_mem['total_mb']:.2f} MB")
            print(f"  Compressed: {compressed_mem['total_mb']:.2f} MB")
        else:
            print("✓ Compression test - no data to compress")


if __name__ == "__main__":
    # Run tests
    print("=" * 60)
    print("Testing WaveCube-Genesis Integration (Phase 1)")
    print("=" * 60)

    # Test ChunkedWaveCube
    print("\n1. Testing ChunkedWaveCube...")
    chunk_tests = TestChunkedWaveCube()
    chunk_tests.test_chunk_creation()
    chunk_tests.test_chunk_loading_unloading()
    chunk_tests.test_adjacent_chunk_caching()
    print("   ✓ All chunk tests passed")

    # Test Quaternionic Coordinates
    print("\n2. Testing Quaternionic Coordinates...")
    coord_tests = TestQuaternionicCoordinates()
    coord_tests.test_coordinate_creation()
    coord_tests.test_distance_metrics()
    coord_tests.test_phase_operations()
    coord_tests.test_cross_modal_binding()
    print("   ✓ All coordinate tests passed")

    # Test VoxelCloud Integration
    print("\n3. Testing VoxelCloud Integration...")
    voxel_tests = TestVoxelCloudIntegration()
    voxel_tests.test_wavecube_reference_storage()
    print("   ✓ VoxelCloud integration passed")

    # Test Full Flow
    print("\n4. Testing Full Integration Flow...")
    flow_tests = TestFullIntegrationFlow()
    flow_tests.test_basic_flow()
    flow_tests.test_multi_octave_storage()
    flow_tests.test_compression_efficiency()
    print("   ✓ All integration tests passed")

    print("\n" + "=" * 60)
    print("SUCCESS: All Phase 1 tests passed!")
    print("=" * 60)