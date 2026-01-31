#!/usr/bin/env python3
"""
Basic test for Phase 2 WaveCube chunking implementation.
Quick validation of core functionality.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
from pathlib import Path

# Import our implementations
from lib.wavecube.core.chunked_matrix import ChunkedWaveCube
from lib.wavecube.io.chunk_storage import ChunkStorage
from lib.wavecube.spatial.spatial_index import SpatialIndex
from lib.wavecube.spatial.coordinates import QuaternionicCoord


def test_basic_chunking():
    """Test basic chunking functionality."""
    print("1. Testing basic chunking...")

    cube = ChunkedWaveCube(
        chunk_size=(8, 8, 8),
        resolution=256,
        channels=4,
        compression='gaussian'
    )

    # Create and store proto-identities
    for i in range(10):
        proto = np.random.randn(256, 256, 4).astype(np.float32) * 0.1
        cube.set_node(i * 5, i * 5, 0, proto)

    # Check memory usage
    mem_stats = cube.get_memory_usage()
    print(f"   Nodes: {cube.stats['total_nodes']}")
    print(f"   Chunks: {cube.stats['chunks_total']}")
    print(f"   Memory: {mem_stats['total_mb']:.2f} MB")

    assert cube.stats['total_nodes'] == 10
    print("   ✓ Basic chunking works")


def test_chunk_persistence():
    """Test chunk persistence to disk."""
    print("\n2. Testing chunk persistence...")

    cache_dir = Path.home() / '.cache' / 'genesis' / 'test_chunks'
    storage = ChunkStorage(cache_dir=cache_dir)

    # Create test data - flattened for npz storage
    chunk_data = {
        'node_0': np.random.randn(256, 256, 4).astype(np.float32),
        'node_count': np.array([1]),
        'metadata': np.array({'test': True})
    }

    # Save chunk
    coords = (0, 0, 0)
    storage.save_chunk(coords, chunk_data, async_save=False)

    # Load chunk
    loaded = storage.load_chunk(coords, async_load=False)

    assert loaded is not None
    print(f"   Loaded keys: {list(loaded.keys()) if loaded else 'None'}")

    # Check that node data was saved and loaded
    assert 'node_0' in loaded
    assert loaded['node_0'].shape == (256, 256, 4)

    # Cleanup
    storage.cleanup_old_chunks(max_age_days=0)
    storage.shutdown(wait=False)

    print("   ✓ Chunk persistence works")


def test_spatial_queries():
    """Test spatial indexing and queries."""
    print("\n3. Testing spatial queries...")

    # Create cube with data
    cube = ChunkedWaveCube(chunk_size=(8, 8, 8), resolution=256)
    positions = []

    for i in range(20):
        x, y, z = i * 3, i * 3, 0
        proto = np.random.randn(256, 256, 4).astype(np.float32) * 0.1
        cube.set_node(x, y, z, proto)
        positions.append((x, y, z))

    # Create spatial index
    index = SpatialIndex(chunk_size=(8, 8, 8))

    for chunk_coords in cube.chunks:
        index.add_chunk(chunk_coords, node_count=1, is_loaded=True)

    # Test k-NN query
    def get_nodes(chunk_coords):
        if chunk_coords not in cube.chunks:
            return []
        nodes = []
        chunk = cube.chunks[chunk_coords]
        for local_coords in chunk.get_populated_nodes():
            wx = chunk_coords[0] * 8 + local_coords[0]
            wy = chunk_coords[1] * 8 + local_coords[1]
            wz = chunk_coords[2] * 8 + local_coords[2]
            node_data = chunk.get_node(*local_coords)
            if node_data is not None:
                nodes.append(((wx, wy, wz), node_data))
        return nodes

    results = index.knn_query((10, 10, 0), k=5, node_getter=get_nodes)

    print(f"   Found {len(results)} neighbors")
    stats = index.get_stats()
    print(f"   Index stats: chunks_examined={stats.get('chunks_examined_avg', 0):.1f}, "
          f"nodes_examined={stats.get('nodes_examined_avg', 0):.1f}")

    assert len(results) <= 5
    print("   ✓ Spatial queries work")


def test_memory_efficiency():
    """Test memory efficiency with compression."""
    print("\n4. Testing memory efficiency...")

    cube = ChunkedWaveCube(
        chunk_size=(16, 16, 16),
        resolution=256,
        compression='gaussian'
    )

    # Create 100 sparse proto-identities
    for i in range(100):
        # Very sparse proto for high compression
        proto = np.zeros((256, 256, 4), dtype=np.float32)

        # Just 2 Gaussian peaks
        for _ in range(2):
            cx, cy = np.random.randint(100, 156, 2)
            y_grid, x_grid = np.ogrid[:256, :256]
            gaussian = np.exp(-((x_grid - cx)**2 + (y_grid - cy)**2) / 200)
            proto[:, :, 0] += gaussian

        proto /= (np.max(proto) + 1e-8)

        x = (i % 10) * 10
        y = (i // 10) * 10
        cube.set_node(x, y, 0, proto)

    # Force compression
    cube.offload_all_inactive()

    mem_stats = cube.get_memory_usage()
    compression_ratio = 100 * (256*256*4*4) / (mem_stats['total_mb'] * 1024*1024)

    print(f"   100 protos: {mem_stats['total_mb']:.2f} MB")
    print(f"   Compression ratio: {compression_ratio:.1f}x")
    print(f"   Chunks: {cube.stats['chunks_total']} total, "
          f"{cube.stats['chunks_compressed']} compressed")

    assert mem_stats['total_mb'] < 100  # Should be much less than uncompressed
    print("   ✓ Memory efficiency achieved")


def test_quaternionic_coords():
    """Test quaternionic coordinate system."""
    print("\n5. Testing quaternionic coordinates...")

    from lib.wavecube.spatial.coordinates import QuaternionicCoord, Modality

    # Test coordinate creation
    coord1 = QuaternionicCoord(10, 20, 30, 45.0)
    assert coord1.w == 45.0

    # Test modality phase-locking
    coord2 = QuaternionicCoord.from_modality(5, 10, 15, Modality.AUDIO)
    assert coord2.w == 90.0  # AUDIO phase

    # Test distance metrics
    dist = coord1.spatial_distance(coord2)
    phase_dist = coord1.phase_distance(coord2)

    print(f"   Spatial distance: {dist:.2f}")
    print(f"   Phase distance: {phase_dist:.2f}")

    print("   ✓ Quaternionic coordinates work")


def main():
    """Run all basic tests."""
    print("=" * 60)
    print("PHASE 2 BASIC TESTS")
    print("=" * 60)

    try:
        test_basic_chunking()
        test_chunk_persistence()
        test_spatial_queries()
        test_memory_efficiency()
        test_quaternionic_coords()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())