#!/usr/bin/env python3
"""
Compression demonstration for Wavecube library.

Demonstrates:
- Gaussian mixture compression
- Compression ratio measurement
- Memory savings
- Round-trip accuracy
"""

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path (works from any location)
wavecube_root = Path(__file__).parent.parent
if str(wavecube_root) not in sys.path:
    sys.path.insert(0, str(wavecube_root.parent))

from wavecube import WavetableMatrix
from wavecube.compression import GaussianMixtureCodec


def demo_1_basic_compression():
    """Demo 1: Basic compression and decompression."""
    print("=" * 60)
    print("Demo 1: Basic Compression")
    print("=" * 60)

    # Create codec
    codec = GaussianMixtureCodec(num_gaussians=8)

    # Create sparse frequency pattern
    print("\n1. Creating sparse frequency pattern (256×256×4)...")
    wavetable = np.zeros((256, 256, 4), dtype=np.float32)

    # Add Gaussian peaks
    y, x = np.ogrid[0:256, 0:256]

    # Peak 1: fundamental frequency
    peak1 = np.exp(-((x - 128)**2 + (y - 128)**2) / 200)
    wavetable[:, :, 0] = peak1 * 1.0

    # Peak 2: first harmonic
    peak2 = np.exp(-((x - 192)**2 + (y - 64)**2) / 150)
    wavetable[:, :, 1] = peak2 * 0.7

    # Peak 3: second harmonic
    peak3 = np.exp(-((x - 64)**2 + (y - 192)**2) / 100)
    wavetable[:, :, 2] = peak3 * 0.5

    # Encode
    print("\n2. Compressing with Gaussian mixture (8 peaks)...")
    compressed = codec.encode(wavetable, quality=0.95)

    original_size = wavetable.nbytes
    compressed_size = compressed.get_memory_usage()
    ratio = compressed.get_compression_ratio()

    print(f"   Original size: {original_size:,} bytes ({original_size / 1024:.1f} KB)")
    print(f"   Compressed size: {compressed_size:,} bytes")
    print(f"   Compression ratio: {ratio:.1f}×")

    # Decode
    print("\n3. Decompressing...")
    reconstructed = codec.decode(compressed)

    # Measure accuracy
    mse = np.mean((wavetable - reconstructed) ** 2)
    mae = np.mean(np.abs(wavetable - reconstructed))
    max_error = np.max(np.abs(wavetable - reconstructed))

    print(f"   Reconstruction quality:")
    print(f"   - MSE: {mse:.6f}")
    print(f"   - MAE: {mae:.6f}")
    print(f"   - Max error: {max_error:.6f}")

    print("\n" + "=" * 60 + "\n")


def demo_2_matrix_compression():
    """Demo 2: Compression in WavetableMatrix."""
    print("=" * 60)
    print("Demo 2: Matrix Compression")
    print("=" * 60)

    # Create matrix with auto-compression
    print("\n1. Creating matrix with auto-compression...")
    matrix = WavetableMatrix(
        width=5, height=5, depth=5,
        resolution=256,
        compression='gaussian'  # Auto-compress on set_node
    )

    # Add nodes with sparse patterns
    print("\n2. Adding 10 sparse wavetables...")
    for i in range(10):
        x = i % 5
        y = (i // 5) % 5
        z = 0

        # Create sparse pattern
        wavetable = np.zeros((256, 256, 4), dtype=np.float32)
        y_grid, x_grid = np.ogrid[0:256, 0:256]

        # Random peaks
        for _ in range(3):
            cx = np.random.randint(50, 200)
            cy = np.random.randint(50, 200)
            sigma = np.random.randint(50, 150)
            channel = np.random.randint(0, 4)

            peak = np.exp(-((x_grid - cx)**2 + (y_grid - cy)**2) / sigma)
            wavetable[:, :, channel] += peak * np.random.rand()

        # Set node (auto-compresses)
        matrix.set_node(x, y, z, wavetable)

    # Get stats
    stats = matrix.get_memory_usage()
    ratio = matrix.get_compression_ratio()

    print(f"\n3. Compression statistics:")
    print(f"   Nodes: {stats['num_nodes']}")
    print(f"   Compressed nodes: {stats['compressed_nodes']}")
    print(f"   Total memory: {stats['total_mb']:.2f} MB")
    print(f"   Avg per node: {stats['avg_bytes_per_node'] / 1024:.2f} KB")
    print(f"   Compression ratio: {ratio:.1f}×")

    # Uncompressed would be
    uncompressed_size = stats['num_nodes'] * (256 * 256 * 4 * 4)
    print(f"\n4. Memory comparison:")
    print(f"   Uncompressed: {uncompressed_size / 1024**2:.2f} MB")
    print(f"   Compressed: {stats['total_mb']:.2f} MB")
    print(f"   Savings: {(1 - stats['total_mb'] / (uncompressed_size / 1024**2)) * 100:.1f}%")

    # Test retrieval
    print("\n5. Testing compressed retrieval...")
    # Get first populated node
    coords = matrix.get_populated_nodes()[0]
    result = matrix.get_node(*coords)
    print(f"   Retrieved from {coords}: shape={result.shape}, dtype={result.dtype}")

    print("\n" + "=" * 60 + "\n")


def demo_3_compression_quality():
    """Demo 3: Quality vs compression trade-off."""
    print("=" * 60)
    print("Demo 3: Quality vs Compression Trade-off")
    print("=" * 60)

    codec = GaussianMixtureCodec(num_gaussians=8)

    # Create test wavetable
    wavetable = np.zeros((128, 128, 4), dtype=np.float32)
    y, x = np.ogrid[0:128, 0:128]
    peak = np.exp(-((x - 64)**2 + (y - 64)**2) / 100)
    wavetable[:, :, 0] = peak

    print("\n1. Testing different quality levels...")
    print(f"   Original size: {wavetable.nbytes:,} bytes\n")

    qualities = [0.5, 0.7, 0.9, 0.95, 0.99]

    for quality in qualities:
        compressed = codec.encode(wavetable, quality=quality)
        reconstructed = codec.decode(compressed)

        size = compressed.get_memory_usage()
        ratio = compressed.get_compression_ratio()
        mse = np.mean((wavetable - reconstructed) ** 2)

        print(f"   Quality {quality:.2f}:")
        print(f"     - Size: {size:,} bytes")
        print(f"     - Ratio: {ratio:.1f}×")
        print(f"     - MSE: {mse:.6f}")

    print("\n" + "=" * 60 + "\n")


def demo_4_genesis_simulation():
    """Demo 4: Simulating Genesis proto-identity storage."""
    print("=" * 60)
    print("Demo 4: Genesis Proto-Identity Simulation")
    print("=" * 60)

    print("\n1. Simulating Genesis frequency field storage...")

    # Create octave hierarchy (like Genesis)
    octaves = {
        +4: WavetableMatrix(width=10, height=10, depth=10, resolution=128, compression='gaussian'),
        0: WavetableMatrix(width=10, height=10, depth=10, resolution=256, compression='gaussian'),
        -2: WavetableMatrix(width=10, height=10, depth=10, resolution=512, compression='gaussian'),
        -4: WavetableMatrix(width=10, height=10, depth=10, resolution=1024, compression='gaussian'),
    }

    print(f"   Created 4 octave levels: +4, 0, -2, -4")

    # Add proto-identities to each octave
    print("\n2. Adding 100 proto-identities across octaves...")

    total_uncompressed = 0
    total_compressed = 0

    for i in range(100):
        # Choose random octave and position
        octave = np.random.choice([+4, 0, -2, -4])
        x = np.random.randint(0, 10)
        y = np.random.randint(0, 10)
        z = np.random.randint(0, 10)

        matrix = octaves[octave]
        res = matrix.resolution[0]

        # Create sparse frequency pattern
        wavetable = np.zeros((res, res, 4), dtype=np.float32)
        y_grid, x_grid = np.ogrid[0:res, 0:res]

        # 2-4 dominant frequencies
        num_peaks = np.random.randint(2, 5)
        for _ in range(num_peaks):
            cx = np.random.randint(res // 4, 3 * res // 4)
            cy = np.random.randint(res // 4, 3 * res // 4)
            sigma = np.random.randint(res // 10, res // 5)
            channel = np.random.randint(0, 4)

            peak = np.exp(-((x_grid - cx)**2 + (y_grid - cy)**2) / sigma)
            wavetable[:, :, channel] += peak * np.random.rand()

        # Track sizes
        total_uncompressed += wavetable.nbytes

        # Add to matrix (auto-compresses)
        matrix.set_node(x, y, z, wavetable)

    # Calculate total compressed size
    for octave, matrix in octaves.items():
        stats = matrix.get_memory_usage()
        total_compressed += stats['total_bytes']

        print(f"\n   Octave {octave:+2d}:")
        print(f"     Nodes: {stats['num_nodes']}")
        print(f"     Memory: {stats['total_mb']:.2f} MB")
        print(f"     Ratio: {matrix.get_compression_ratio():.1f}×")

    overall_ratio = total_uncompressed / total_compressed

    print(f"\n3. Overall statistics:")
    print(f"   Total uncompressed: {total_uncompressed / 1024**2:.2f} MB")
    print(f"   Total compressed: {total_compressed / 1024**2:.2f} MB")
    print(f"   Overall ratio: {overall_ratio:.1f}×")
    print(f"   Memory savings: {(1 - total_compressed / total_uncompressed) * 100:.1f}%")

    print("\n" + "=" * 60 + "\n")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Wavecube Compression Demonstrations".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")

    demos = [
        demo_1_basic_compression,
        demo_2_matrix_compression,
        demo_3_compression_quality,
        demo_4_genesis_simulation,
    ]

    for i, demo in enumerate(demos, 1):
        print(f"\nRunning Demo {i}...")
        demo()
        if i < len(demos):
            input("Press Enter to continue...")

    print("\n")
    print("=" * 60)
    print("All compression demos complete!")
    print("=" * 60)
    print("\n")


if __name__ == "__main__":
    main()
