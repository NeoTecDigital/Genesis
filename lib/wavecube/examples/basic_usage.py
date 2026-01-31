#!/usr/bin/env python3
"""
Basic usage examples for Wavecube library.

Demonstrates core features:
- Creating matrices
- Adding wavetables
- Interpolation
- Save/load
- Memory management
"""

import numpy as np
from pathlib import Path
import sys

# Add wavecube to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wavecube import WavetableMatrix
from wavecube.interpolation import nearest_neighbor
from wavecube.utils import run_full_benchmark_suite


def example_1_basic_operations():
    """Example 1: Basic matrix operations."""
    print("=" * 60)
    print("Example 1: Basic Matrix Operations")
    print("=" * 60)

    # Create matrix
    print("\n1. Creating 10×10×10 matrix with 256×256×4 wavetables...")
    matrix = WavetableMatrix(
        width=10,
        height=10,
        depth=10,
        resolution=256,
        channels=4,
        sparse=True
    )
    print(f"   {matrix}")

    # Add wavetables
    print("\n2. Adding wavetables at random positions...")
    for i in range(5):
        x, y, z = np.random.randint(0, 10, 3)
        wavetable = np.random.randn(256, 256, 4).astype(np.float32)
        matrix.set_node(x, y, z, wavetable)
        print(f"   Added at ({x}, {y}, {z})")

    # Memory stats
    stats = matrix.get_memory_usage()
    print(f"\n3. Memory usage:")
    print(f"   Nodes: {stats['num_nodes']}")
    print(f"   Total: {stats['total_mb']:.2f} MB")
    print(f"   Avg per node: {stats['avg_bytes_per_node'] / 1024:.2f} KB")

    # Dictionary-style access
    print("\n4. Dictionary-style access:")
    if (5, 5, 5) in matrix:
        print(f"   Node exists at (5, 5, 5)")
    else:
        print(f"   No node at (5, 5, 5)")
        wavetable = np.ones((256, 256, 4), dtype=np.float32)
        matrix[5, 5, 5] = wavetable
        print(f"   Created node at (5, 5, 5)")

    result = matrix[5, 5, 5]
    print(f"   Retrieved: shape={result.shape}, mean={result.mean():.3f}")

    print("\n" + "=" * 60 + "\n")


def example_2_interpolation():
    """Example 2: Interpolation."""
    print("=" * 60)
    print("Example 2: Interpolation")
    print("=" * 60)

    # Create simple matrix with known values
    print("\n1. Creating matrix with gradient values...")
    matrix = WavetableMatrix(width=3, height=3, depth=3, resolution=64, channels=4)

    for x in range(3):
        for y in range(3):
            for z in range(3):
                # Value increases with coordinates
                value = x + y*3 + z*9
                wavetable = np.full((64, 64, 4), value, dtype=np.float32)
                matrix.set_node(x, y, z, wavetable)

    print(f"   Created {matrix.get_memory_usage()['num_nodes']} nodes")

    # Exact sampling
    print("\n2. Sampling at exact grid point (1, 1, 1):")
    result = matrix.sample(1.0, 1.0, 1.0)
    expected = 1 + 1*3 + 1*9  # = 13
    print(f"   Result: {result[0, 0, 0]}")
    print(f"   Expected: {expected}")

    # Interpolated sampling
    print("\n3. Sampling at fractional position (1.5, 1.5, 1.5):")
    result = matrix.sample(1.5, 1.5, 1.5)
    # Average of all 8 corners
    corners = [
        1 + 1*3 + 1*9,  # (1,1,1) = 13
        2 + 1*3 + 1*9,  # (2,1,1) = 14
        1 + 2*3 + 1*9,  # (1,2,1) = 16
        2 + 2*3 + 1*9,  # (2,2,1) = 17
        1 + 1*3 + 2*9,  # (1,1,2) = 22
        2 + 1*3 + 2*9,  # (2,1,2) = 23
        1 + 2*3 + 2*9,  # (1,2,2) = 25
        2 + 2*3 + 2*9,  # (2,2,2) = 26
    ]
    expected = sum(corners) / 8
    print(f"   Result: {result[0, 0, 0]}")
    print(f"   Expected (average of 8 corners): {expected}")

    # Batch sampling
    print("\n4. Batch sampling at 100 random positions:")
    coords = np.random.rand(100, 3) * 3  # Random coords in [0, 3]
    results = matrix.sample_batch(coords)
    print(f"   Results shape: {results.shape}")
    print(f"   Value range: [{results.min():.2f}, {results.max():.2f}]")

    print("\n" + "=" * 60 + "\n")


def example_3_save_load():
    """Example 3: Save and load."""
    print("=" * 60)
    print("Example 3: Save and Load")
    print("=" * 60)

    # Create and populate matrix
    print("\n1. Creating matrix with metadata...")
    matrix = WavetableMatrix(width=5, height=5, depth=5, resolution=128)

    for i in range(10):
        x, y, z = np.random.randint(0, 5, 3)
        wavetable = np.random.randn(128, 128, 4).astype(np.float32)
        metadata = {
            'octave': i % 4,
            'modality': ['text', 'audio', 'image'][i % 3],
            'index': i
        }
        matrix.set_node(x, y, z, wavetable, metadata=metadata)

    print(f"   Created {matrix.get_memory_usage()['num_nodes']} nodes")

    # Save
    print("\n2. Saving to test_matrix.npz...")
    matrix.save('test_matrix.npz')
    file_size = Path('test_matrix.npz').stat().st_size / 1024**2
    print(f"   File size: {file_size:.2f} MB")

    # Load
    print("\n3. Loading from disk...")
    loaded = WavetableMatrix.load('test_matrix.npz')
    print(f"   Loaded: {loaded}")
    print(f"   Nodes: {loaded.get_memory_usage()['num_nodes']}")

    # Verify
    print("\n4. Verifying metadata preservation...")
    for coords in loaded.get_populated_nodes()[:3]:
        node = loaded._nodes[coords]
        print(f"   Node {coords}: {node.metadata}")

    # Cleanup
    Path('test_matrix.npz').unlink()
    print("\n   Cleaned up test file")

    print("\n" + "=" * 60 + "\n")


def example_4_sparse_vs_dense():
    """Example 4: Sparse vs dense storage comparison."""
    print("=" * 60)
    print("Example 4: Sparse vs Dense Storage")
    print("=" * 60)

    # Sparse matrix
    print("\n1. Creating sparse matrix (10×10×10)...")
    sparse = WavetableMatrix(width=10, height=10, depth=10, resolution=64, sparse=True)

    # Populate 10% of nodes
    for i in range(100):
        x, y, z = np.random.randint(0, 10, 3)
        wavetable = np.random.randn(64, 64, 4).astype(np.float32)
        sparse.set_node(x, y, z, wavetable)

    sparse_stats = sparse.get_memory_usage()
    print(f"   Nodes: {sparse_stats['num_nodes']}")
    print(f"   Memory: {sparse_stats['total_mb']:.2f} MB")

    # Dense would be
    total_nodes = 10 * 10 * 10
    node_size = 64 * 64 * 4 * 4  # resolution * channels * float32
    dense_size = total_nodes * node_size / 1024**2

    print(f"\n2. Dense matrix would use:")
    print(f"   Nodes: {total_nodes}")
    print(f"   Memory: {dense_size:.2f} MB")

    savings = (1 - sparse_stats['total_mb'] / dense_size) * 100
    print(f"\n3. Savings: {savings:.1f}%")

    print("\n" + "=" * 60 + "\n")


def example_5_performance():
    """Example 5: Performance benchmarks."""
    print("=" * 60)
    print("Example 5: Performance Benchmarks")
    print("=" * 60)

    print("\nRunning full benchmark suite...")
    print("(This may take a minute...)\n")

    results = run_full_benchmark_suite(verbose=True)


def main():
    """Run all examples."""
    examples = [
        example_1_basic_operations,
        example_2_interpolation,
        example_3_save_load,
        example_4_sparse_vs_dense,
        example_5_performance,
    ]

    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Wavecube Library - Basic Usage Examples".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")

    for i, example in enumerate(examples, 1):
        print(f"\nRunning Example {i}...")
        example()
        if i < len(examples):
            input("Press Enter to continue...")

    print("\n")
    print("=" * 60)
    print("All examples complete!")
    print("=" * 60)
    print("\n")


if __name__ == "__main__":
    main()
