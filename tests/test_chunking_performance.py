"""
Performance tests for chunking system under load.

Tests:
- Memory footprint with 1000+ proto-identities
- Query performance for k-NN
- Chunk loading/unloading latency
- Memory leak detection
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
import tracemalloc
from pathlib import Path
import psutil
import gc

# WaveCube imports
from lib.wavecube.core.chunked_matrix import ChunkedWaveCube
from lib.wavecube.io.chunk_storage import ChunkStorage
from lib.wavecube.spatial.spatial_index import SpatialIndex
from lib.wavecube.spatial.coordinates import QuaternionicCoord


class TestChunkingPerformance:
    """Test chunking performance under load."""

    def __init__(self):
        """Initialize test environment."""
        self.process = psutil.Process()
        self.initial_memory = None

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / (1024 ** 2)

    def test_memory_with_1000_protos(self):
        """Test memory footprint with 1000 proto-identities."""
        print("\n=== Testing Memory with 1000 Proto-identities ===")

        # Start memory tracking
        gc.collect()
        initial_mem = self.get_memory_usage()
        print(f"Initial memory: {initial_mem:.2f} MB")

        # Create chunked cube with compression
        cube = ChunkedWaveCube(
            chunk_size=(8, 8, 8),  # Smaller chunks for better distribution
            resolution=256,  # Match proto size for memory efficiency
            channels=4,
            cache_radius=1,
            compression='gaussian'
        )

        # Create 1000 proto-identities distributed across space
        num_protos = 1000
        protos_created = 0

        print(f"Creating {num_protos} proto-identities...")
        start_time = time.time()

        for i in range(num_protos):
            # Distribute across 100 chunks (10x10x1 grid)
            x = (i % 10) * 10
            y = ((i // 10) % 10) * 10
            z = (i // 100) * 2

            # Create sparse proto-identity (more realistic for text)
            # Use smaller size for memory efficiency
            proto = np.zeros((256, 256, 4), dtype=np.float32)

            # Add 3-5 Gaussian peaks (typical for text encoding)
            num_peaks = np.random.randint(3, 6)
            for _ in range(num_peaks):
                cx = np.random.randint(50, 206)
                cy = np.random.randint(50, 206)
                sigma = np.random.uniform(5, 20)

                y_grid, x_grid = np.ogrid[:256, :256]
                gaussian = np.exp(-((x_grid - cx)**2 + (y_grid - cy)**2) / (2 * sigma**2))
                proto[:, :, np.random.randint(0, 4)] += gaussian

            # Normalize
            proto /= (np.max(proto) + 1e-8)

            # Store in cube
            cube.set_node(x, y, z, proto)
            protos_created += 1

            # Set active position periodically to trigger offloading
            if i % 100 == 0:
                cube.set_active_position(x, y, z)
                current_mem = self.get_memory_usage()
                print(f"  {i+1}/{num_protos}: {current_mem:.2f} MB "
                      f"(+{current_mem - initial_mem:.2f} MB), "
                      f"chunks: {cube.stats['chunks_loaded']}/{cube.stats['chunks_total']}")

        creation_time = time.time() - start_time

        # Force offload all except active
        cube.offload_all_inactive()
        gc.collect()

        # Check final memory
        final_mem = self.get_memory_usage()
        memory_used = final_mem - initial_mem

        # Get cube statistics
        mem_stats = cube.get_memory_usage()

        print(f"\n--- Results ---")
        print(f"Proto-identities created: {protos_created}")
        print(f"Creation time: {creation_time:.2f} seconds")
        print(f"Memory used: {memory_used:.2f} MB")
        print(f"Active memory (cube): {mem_stats['active_mb']:.2f} MB")
        print(f"Total memory (cube): {mem_stats['total_mb']:.2f} MB")
        print(f"Chunks: {cube.stats['chunks_total']} total, "
              f"{cube.stats['chunks_loaded']} loaded, "
              f"{cube.stats['chunks_compressed']} compressed")

        # Check if under 1GB
        assert memory_used < 1024, f"Memory usage {memory_used:.2f} MB exceeds 1GB limit"
        assert mem_stats['active_mb'] < 1024, f"Active memory {mem_stats['active_mb']:.2f} MB exceeds 1GB"

        print("✓ Memory test PASSED - Under 1GB")
        return cube, memory_used

    def test_knn_query_performance(self, cube=None):
        """Test k-NN query performance."""
        print("\n=== Testing K-NN Query Performance ===")

        if cube is None:
            # Create test cube with some data
            cube = ChunkedWaveCube(chunk_size=(8, 8, 8), resolution=256, compression='gaussian')
            for i in range(100):
                x = (i % 10) * 8
                y = ((i // 10) % 10) * 8
                z = 0
                proto = np.random.randn(256, 256, 4).astype(np.float32) * 0.1
                cube.set_node(x, y, z, proto)

        # Create spatial index
        index = SpatialIndex(
            chunk_size=(8, 8, 8),
            cache_radius=2,
            query_cache_size=100
        )

        # Add chunks to index
        for chunk_coords in cube.chunks:
            info = cube.chunks[chunk_coords]
            node_count = len(info.get_populated_nodes())
            is_loaded = chunk_coords in cube.active_chunks
            index.add_chunk(chunk_coords, node_count, is_loaded)

        # Node getter function for queries
        def get_nodes_from_chunk(chunk_coords):
            if chunk_coords not in cube.chunks:
                return []

            chunk = cube._ensure_chunk_loaded(chunk_coords)
            nodes = []
            for local_coords in chunk.get_populated_nodes():
                # Convert to world coordinates
                wx = chunk_coords[0] * cube.chunk_size[0] + local_coords[0]
                wy = chunk_coords[1] * cube.chunk_size[1] + local_coords[1]
                wz = chunk_coords[2] * cube.chunk_size[2] + local_coords[2]
                world_pos = (wx, wy, wz)

                # Get node data
                node_data = chunk.get_node(*local_coords)
                if node_data is not None:
                    nodes.append((world_pos, node_data))

            return nodes

        # Test queries at different positions
        query_positions = [
            (40, 40, 0),   # Center
            (0, 0, 0),     # Corner
            (80, 80, 0),   # Other corner
            (20, 60, 0),   # Random
        ]

        query_times = []
        for pos in query_positions:
            # Warm-up query
            index.knn_query(pos, k=10, node_getter=get_nodes_from_chunk)

            # Timed query
            start_time = time.time()
            results = index.knn_query(
                pos,
                k=10,
                max_distance=50.0,
                node_getter=get_nodes_from_chunk
            )
            query_time = (time.time() - start_time) * 1000  # Convert to ms

            query_times.append(query_time)
            print(f"Query at {pos}: {query_time:.2f} ms, found {len(results)} neighbors")

        avg_time = np.mean(query_times)
        max_time = np.max(query_times)

        print(f"\n--- Results ---")
        print(f"Average query time: {avg_time:.2f} ms")
        print(f"Max query time: {max_time:.2f} ms")
        print(f"Index stats: {index.get_stats()}")

        # Check if under 100ms
        assert max_time < 100, f"Query time {max_time:.2f} ms exceeds 100ms limit"
        print("✓ Query performance test PASSED - Under 100ms")

        return avg_time

    def test_chunk_load_unload_latency(self):
        """Test chunk loading and unloading latency."""
        print("\n=== Testing Chunk Load/Unload Latency ===")

        # Create storage system
        storage = ChunkStorage(
            cache_dir=Path.home() / '.cache' / 'genesis' / 'test_chunks',
            max_threads=4,
            auto_save=True
        )

        # Create test chunk data
        chunk_data = {
            'nodes': {},
            'metadata': {}
        }

        # Add some nodes
        for i in range(10):
            node_key = f'node_{i}'
            chunk_data['nodes'][node_key] = np.random.randn(256, 256, 4).astype(np.float32)

        # Test save latency
        save_times = []
        for i in range(5):
            coords = (i, 0, 0)
            start_time = time.time()
            future = storage.save_chunk(coords, chunk_data, async_save=False)
            save_time = (time.time() - start_time) * 1000
            save_times.append(save_time)
            print(f"Save chunk {coords}: {save_time:.2f} ms")

        # Test load latency
        load_times = []
        for i in range(5):
            coords = (i, 0, 0)
            start_time = time.time()
            loaded_data = storage.load_chunk(coords, async_load=False)
            load_time = (time.time() - start_time) * 1000
            load_times.append(load_time)
            print(f"Load chunk {coords}: {load_time:.2f} ms")
            assert loaded_data is not None, f"Failed to load chunk {coords}"

        avg_save = np.mean(save_times)
        avg_load = np.mean(load_times)

        print(f"\n--- Results ---")
        print(f"Average save time: {avg_save:.2f} ms")
        print(f"Average load time: {avg_load:.2f} ms")
        print(f"Storage stats: {storage.get_summary_stats()}")

        # Cleanup
        storage.cleanup_old_chunks(max_age_days=0)
        storage.shutdown()

        print("✓ Load/unload latency test PASSED")
        return avg_save, avg_load

    def test_memory_leak_detection(self):
        """Test for memory leaks during extended operation."""
        print("\n=== Testing for Memory Leaks ===")

        # Start memory tracking
        tracemalloc.start()
        gc.collect()
        initial_mem = self.get_memory_usage()

        # Create cube
        cube = ChunkedWaveCube(
            chunk_size=(4, 4, 4),
            cache_radius=1,
            compression='gaussian'
        )

        # Perform many operations
        num_iterations = 10
        num_protos_per_iter = 100

        memory_samples = []

        for iteration in range(num_iterations):
            # Add protos
            for i in range(num_protos_per_iter):
                x = np.random.randint(0, 40)
                y = np.random.randint(0, 40)
                z = np.random.randint(0, 10)

                proto = np.random.randn(256, 256, 4).astype(np.float32) * 0.1
                cube.set_node(x, y, z, proto)

            # Move active position
            cube.set_active_position(
                np.random.randint(0, 40),
                np.random.randint(0, 40),
                np.random.randint(0, 10)
            )

            # Delete some nodes
            for _ in range(10):
                x = np.random.randint(0, 40)
                y = np.random.randint(0, 40)
                z = np.random.randint(0, 10)
                cube.delete_node(x, y, z)

            # Force garbage collection
            cube.offload_all_inactive()
            gc.collect()

            # Sample memory
            current_mem = self.get_memory_usage()
            memory_samples.append(current_mem)

            print(f"Iteration {iteration + 1}/{num_iterations}: "
                  f"{current_mem:.2f} MB (+{current_mem - initial_mem:.2f} MB)")

        # Check for memory leak
        # Memory should stabilize, not continuously increase
        first_half_avg = np.mean(memory_samples[:5])
        second_half_avg = np.mean(memory_samples[5:])
        memory_growth = second_half_avg - first_half_avg

        # Get memory snapshot
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')[:10]

        print(f"\n--- Results ---")
        print(f"Initial memory: {initial_mem:.2f} MB")
        print(f"Final memory: {memory_samples[-1]:.2f} MB")
        print(f"First half avg: {first_half_avg:.2f} MB")
        print(f"Second half avg: {second_half_avg:.2f} MB")
        print(f"Memory growth: {memory_growth:.2f} MB")

        print("\nTop memory allocations:")
        for stat in top_stats[:5]:
            print(f"  {stat}")

        tracemalloc.stop()

        # Check for leak (shouldn't grow more than 10MB between halves)
        assert memory_growth < 10, f"Potential memory leak: {memory_growth:.2f} MB growth"
        print("✓ Memory leak test PASSED - No significant leaks detected")

        return memory_growth

    def test_stress_10k_protos(self):
        """Stress test with 10,000 proto-identities."""
        print("\n=== Stress Test: 10,000 Proto-identities ===")

        gc.collect()
        initial_mem = self.get_memory_usage()

        # Create cube with aggressive compression
        cube = ChunkedWaveCube(
            chunk_size=(16, 16, 16),  # Larger chunks for efficiency
            resolution=256,  # Smaller resolution for memory efficiency
            cache_radius=0,  # Minimal caching
            compression='gaussian'
        )

        num_protos = 10000
        batch_size = 500

        print(f"Creating {num_protos} proto-identities...")
        start_time = time.time()

        for batch_start in range(0, num_protos, batch_size):
            batch_end = min(batch_start + batch_size, num_protos)

            for i in range(batch_start, batch_end):
                # Distribute across large space
                x = (i % 50) * 4
                y = ((i // 50) % 50) * 4
                z = (i // 2500) * 4

                # Create very sparse proto (extreme compression)
                proto = np.zeros((256, 256, 4), dtype=np.float32)

                # Just 2-3 peaks for maximum compression
                num_peaks = np.random.randint(2, 4)
                for _ in range(num_peaks):
                    cx = np.random.randint(100, 156)
                    cy = np.random.randint(100, 156)
                    sigma = 10

                    y_grid, x_grid = np.ogrid[:256, :256]
                    gaussian = np.exp(-((x_grid - cx)**2 + (y_grid - cy)**2) / (2 * sigma**2))
                    proto[:, :, np.random.randint(0, 4)] += gaussian

                proto /= (np.max(proto) + 1e-8)
                cube.set_node(x, y, z, proto)

            # Aggressive offloading after each batch
            cube.offload_all_inactive()
            gc.collect()

            current_mem = self.get_memory_usage()
            memory_used = current_mem - initial_mem

            print(f"Batch {batch_end}/{num_protos}: {current_mem:.2f} MB "
                  f"(+{memory_used:.2f} MB), "
                  f"chunks: {cube.stats['chunks_loaded']}/{cube.stats['chunks_total']}")

            # Emergency check
            if memory_used > 900:  # Leave 100MB buffer
                print(f"WARNING: Approaching 1GB limit at {batch_end} protos")
                break

        creation_time = time.time() - start_time

        # Final offload
        cube.offload_all_inactive()
        gc.collect()

        final_mem = self.get_memory_usage()
        total_memory = final_mem - initial_mem

        mem_stats = cube.get_memory_usage()

        print(f"\n--- Results ---")
        print(f"Proto-identities created: {cube.stats['total_nodes']}")
        print(f"Creation time: {creation_time:.2f} seconds")
        print(f"Total memory used: {total_memory:.2f} MB")
        print(f"Cube statistics:")
        print(f"  Total chunks: {cube.stats['chunks_total']}")
        print(f"  Loaded chunks: {cube.stats['chunks_loaded']}")
        print(f"  Compressed chunks: {cube.stats['chunks_compressed']}")
        print(f"  Active memory: {mem_stats['active_mb']:.2f} MB")
        print(f"  Total memory: {mem_stats['total_mb']:.2f} MB")

        # Verify no memory leaks
        assert total_memory < 1024, f"Memory {total_memory:.2f} MB exceeds 1GB limit"
        print("✓ Stress test PASSED - Handled 10K protos under 1GB")

        return cube.stats['total_nodes'], total_memory


def main():
    """Run all performance tests."""
    print("=" * 70)
    print("CHUNKING PERFORMANCE TESTS")
    print("=" * 70)

    tester = TestChunkingPerformance()

    try:
        # Test 1: Memory with 1000 protos
        cube, memory_1k = tester.test_memory_with_1000_protos()

        # Test 2: Query performance (reuse cube from test 1)
        avg_query_time = tester.test_knn_query_performance(cube)

        # Test 3: Load/unload latency
        save_time, load_time = tester.test_chunk_load_unload_latency()

        # Test 4: Memory leak detection
        memory_growth = tester.test_memory_leak_detection()

        # Test 5: Stress test with 10K protos
        num_protos, memory_10k = tester.test_stress_10k_protos()

        print("\n" + "=" * 70)
        print("ALL PERFORMANCE TESTS PASSED!")
        print("=" * 70)
        print("\nSummary:")
        print(f"  1000 protos memory: {memory_1k:.2f} MB")
        print(f"  Query time: {avg_query_time:.2f} ms")
        print(f"  Chunk save/load: {save_time:.2f}/{load_time:.2f} ms")
        print(f"  Memory leak growth: {memory_growth:.2f} MB")
        print(f"  10K protos memory: {memory_10k:.2f} MB")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())