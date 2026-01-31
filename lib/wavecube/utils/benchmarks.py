"""
Performance benchmarks for wavecube operations.

Measures interpolation speed, memory usage, and compression ratios.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Callable
from pathlib import Path
import tempfile

from ..core import WavetableMatrix
from ..interpolation import (
    trilinear_interpolate,
    bilinear_interpolate,
    nearest_neighbor
)


def benchmark_interpolation(
    matrix: WavetableMatrix,
    num_samples: int = 1000,
    method: str = 'trilinear'
) -> Dict[str, float]:
    """
    Benchmark interpolation performance.

    Args:
        matrix: WavetableMatrix to benchmark
        num_samples: Number of samples to interpolate
        method: Interpolation method ('trilinear', 'nearest')

    Returns:
        Dict with timing statistics
    """
    # Generate random coordinates
    coords = np.random.rand(num_samples, 3)
    coords[:, 0] *= matrix.width
    coords[:, 1] *= matrix.height
    coords[:, 2] *= matrix.depth

    # Warm-up (JIT, caching, etc.)
    for i in range(min(10, num_samples)):
        x, y, z = coords[i]
        if method == 'trilinear':
            try:
                matrix.sample(x, y, z)
            except RuntimeError:
                pass
        elif method == 'nearest':
            try:
                nearest_neighbor(matrix, x, y, z)
            except RuntimeError:
                pass

    # Benchmark
    start = time.perf_counter()

    for i in range(num_samples):
        x, y, z = coords[i]
        try:
            if method == 'trilinear':
                matrix.sample(x, y, z)
            elif method == 'nearest':
                nearest_neighbor(matrix, x, y, z)
        except RuntimeError:
            # Sparse matrix - node doesn't exist
            pass

    end = time.perf_counter()

    elapsed = end - start

    return {
        'method': method,
        'num_samples': num_samples,
        'total_time_s': elapsed,
        'avg_time_ms': (elapsed / num_samples) * 1000,
        'samples_per_second': num_samples / elapsed
    }


def benchmark_batch_interpolation(
    matrix: WavetableMatrix,
    batch_sizes: List[int] = [10, 100, 1000]
) -> List[Dict[str, float]]:
    """
    Benchmark batch interpolation at different batch sizes.

    Args:
        matrix: WavetableMatrix to benchmark
        batch_sizes: List of batch sizes to test

    Returns:
        List of timing dicts for each batch size
    """
    results = []

    for batch_size in batch_sizes:
        coords = np.random.rand(batch_size, 3)
        coords[:, 0] *= matrix.width
        coords[:, 1] *= matrix.height
        coords[:, 2] *= matrix.depth

        # Warm-up
        try:
            matrix.sample_batch(coords[:min(10, batch_size)])
        except:
            pass

        # Benchmark
        start = time.perf_counter()
        try:
            matrix.sample_batch(coords)
        except RuntimeError:
            pass
        end = time.perf_counter()

        elapsed = end - start

        results.append({
            'batch_size': batch_size,
            'total_time_ms': elapsed * 1000,
            'avg_time_per_sample_ms': (elapsed / batch_size) * 1000,
            'samples_per_second': batch_size / elapsed
        })

    return results


def benchmark_memory_usage(
    grid_sizes: List[Tuple[int, int, int]] = [(5, 5, 5), (10, 10, 10), (20, 20, 20)],
    resolution: int = 256,
    sparse_fill_ratio: float = 0.5
) -> List[Dict[str, float]]:
    """
    Benchmark memory usage at different grid sizes.

    Args:
        grid_sizes: List of (width, height, depth) tuples
        resolution: Wavetable resolution
        sparse_fill_ratio: Fraction of nodes to populate (0-1)

    Returns:
        List of memory usage dicts
    """
    results = []

    for width, height, depth in grid_sizes:
        matrix = WavetableMatrix(
            width=width, height=height, depth=depth,
            resolution=resolution,
            sparse=True
        )

        # Populate nodes
        num_nodes = int(width * height * depth * sparse_fill_ratio)
        populated = 0

        while populated < num_nodes:
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            z = np.random.randint(0, depth)

            if not matrix.has_node(x, y, z):
                wavetable = np.random.randn(resolution, resolution, 4).astype(np.float32)
                matrix.set_node(x, y, z, wavetable)
                populated += 1

        stats = matrix.get_memory_usage()

        results.append({
            'grid_size': f"{width}×{height}×{depth}",
            'total_nodes': width * height * depth,
            'populated_nodes': stats['num_nodes'],
            'fill_ratio': stats['num_nodes'] / (width * height * depth),
            'total_mb': stats['total_mb'],
            'avg_mb_per_node': stats['avg_bytes_per_node'] / (1024**2),
        })

    return results


def benchmark_save_load(
    matrix: WavetableMatrix,
    num_trials: int = 5
) -> Dict[str, float]:
    """
    Benchmark save/load performance.

    Args:
        matrix: WavetableMatrix to benchmark
        num_trials: Number of trials to average

    Returns:
        Dict with save/load timing and file size
    """
    save_times = []
    load_times = []
    file_sizes = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for trial in range(num_trials):
            path = Path(tmpdir) / f"test_{trial}.npz"

            # Benchmark save
            start = time.perf_counter()
            matrix.save(str(path))
            save_time = time.perf_counter() - start
            save_times.append(save_time)

            # Get file size
            file_sizes.append(path.stat().st_size)

            # Benchmark load
            start = time.perf_counter()
            loaded = WavetableMatrix.load(str(path))
            load_time = time.perf_counter() - start
            load_times.append(load_time)

    return {
        'avg_save_time_ms': np.mean(save_times) * 1000,
        'avg_load_time_ms': np.mean(load_times) * 1000,
        'avg_file_size_mb': np.mean(file_sizes) / (1024**2),
        'std_save_time_ms': np.std(save_times) * 1000,
        'std_load_time_ms': np.std(load_times) * 1000,
    }


def run_full_benchmark_suite(
    verbose: bool = True
) -> Dict[str, any]:
    """
    Run complete benchmark suite.

    Args:
        verbose: Print results to console

    Returns:
        Dict with all benchmark results
    """
    if verbose:
        print("=" * 60)
        print("Wavecube Performance Benchmark Suite")
        print("=" * 60)

    results = {}

    # Create test matrix
    if verbose:
        print("\nCreating test matrix (10×10×10, resolution=256)...")

    matrix = WavetableMatrix(width=10, height=10, depth=10, resolution=256, sparse=True)

    # Populate 50% of nodes
    num_nodes = 500
    for i in range(num_nodes):
        x = np.random.randint(0, 10)
        y = np.random.randint(0, 10)
        z = np.random.randint(0, 10)
        wavetable = np.random.randn(256, 256, 4).astype(np.float32)
        matrix.set_node(x, y, z, wavetable)

    if verbose:
        stats = matrix.get_memory_usage()
        print(f"  Populated: {stats['num_nodes']} nodes")
        print(f"  Memory: {stats['total_mb']:.2f} MB")

    # Benchmark 1: Single-sample interpolation
    if verbose:
        print("\n1. Single-sample interpolation (1000 samples)...")

    results['trilinear_single'] = benchmark_interpolation(matrix, 1000, 'trilinear')
    results['nearest_single'] = benchmark_interpolation(matrix, 1000, 'nearest')

    if verbose:
        print(f"  Trilinear: {results['trilinear_single']['avg_time_ms']:.3f} ms/sample")
        print(f"  Nearest:   {results['nearest_single']['avg_time_ms']:.3f} ms/sample")

    # Benchmark 2: Batch interpolation
    if verbose:
        print("\n2. Batch interpolation...")

    results['batch'] = benchmark_batch_interpolation(matrix, [10, 100, 1000])

    if verbose:
        for r in results['batch']:
            print(f"  Batch size {r['batch_size']:4d}: {r['total_time_ms']:6.2f} ms total, "
                  f"{r['avg_time_per_sample_ms']:.3f} ms/sample")

    # Benchmark 3: Memory usage
    if verbose:
        print("\n3. Memory usage scaling...")

    results['memory'] = benchmark_memory_usage()

    if verbose:
        for r in results['memory']:
            print(f"  Grid {r['grid_size']:>11s}: {r['populated_nodes']:4d} nodes, "
                  f"{r['total_mb']:6.2f} MB, {r['avg_mb_per_node']:.3f} MB/node")

    # Benchmark 4: Save/load
    if verbose:
        print("\n4. Save/load performance...")

    results['save_load'] = benchmark_save_load(matrix, num_trials=5)

    if verbose:
        print(f"  Save: {results['save_load']['avg_save_time_ms']:.2f} ms "
              f"(± {results['save_load']['std_save_time_ms']:.2f} ms)")
        print(f"  Load: {results['save_load']['avg_load_time_ms']:.2f} ms "
              f"(± {results['save_load']['std_load_time_ms']:.2f} ms)")
        print(f"  File size: {results['save_load']['avg_file_size_mb']:.2f} MB")

    if verbose:
        print("\n" + "=" * 60)
        print("Benchmark complete!")
        print("=" * 60)

    return results


if __name__ == "__main__":
    # Run benchmark suite
    results = run_full_benchmark_suite(verbose=True)
