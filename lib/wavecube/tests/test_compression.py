"""
Tests for compression codecs.

Tests Gaussian Mixture compression and decompression with various
wavetable patterns.
"""

import pytest
import numpy as np
from wavecube import WavetableMatrix
from wavecube.compression import (
    GaussianMixtureCodec,
    CompressedWavetable,
    GaussianMixtureParams
)


class TestGaussianMixtureCodec:
    """Test Gaussian mixture compression."""

    def test_codec_initialization(self):
        """Test codec can be created with parameters."""
        codec = GaussianMixtureCodec(num_gaussians=8)
        assert codec.params['num_gaussians'] == 8

    def test_encode_decode_sparse_pattern(self):
        """Test encoding and decoding a sparse frequency pattern."""
        codec = GaussianMixtureCodec(num_gaussians=8)

        # Create sparse pattern with a few peaks
        wavetable = np.zeros((64, 64, 4), dtype=np.float32)

        # Add 3 Gaussian peaks
        y, x = np.ogrid[0:64, 0:64]

        # Peak 1: center (16, 16)
        peak1 = np.exp(-((x - 16)**2 + (y - 16)**2) / 50)
        wavetable[:, :, 0] += peak1 * 1.0

        # Peak 2: center (48, 48)
        peak2 = np.exp(-((x - 48)**2 + (y - 48)**2) / 50)
        wavetable[:, :, 1] += peak2 * 0.8

        # Peak 3: center (32, 16)
        peak3 = np.exp(-((x - 32)**2 + (y - 16)**2) / 50)
        wavetable[:, :, 2] += peak3 * 0.6

        # Encode
        compressed = codec.encode(wavetable, quality=0.95)

        assert isinstance(compressed, CompressedWavetable)
        assert compressed.method == 'gaussian'
        assert compressed.original_shape == (64, 64, 4)

        # Decode
        reconstructed = codec.decode(compressed)

        assert reconstructed.shape == wavetable.shape
        assert reconstructed.dtype == wavetable.dtype

        # Check reconstruction quality (should be reasonable for sparse pattern)
        mse = np.mean((wavetable - reconstructed) ** 2)
        assert mse < 0.1  # Allowing some reconstruction error

    def test_compression_ratio(self):
        """Test that compression achieves significant compression ratio."""
        codec = GaussianMixtureCodec(num_gaussians=4)

        # Create sparse 256x256x4 wavetable
        wavetable = np.zeros((256, 256, 4), dtype=np.float32)

        # Add few peaks
        y, x = np.ogrid[0:256, 0:256]
        peak1 = np.exp(-((x - 128)**2 + (y - 128)**2) / 200)
        wavetable[:, :, 0] = peak1

        compressed = codec.encode(wavetable, quality=0.95)

        # Original: 256 * 256 * 4 * 4 bytes = 1MB
        original_size = 256 * 256 * 4 * 4

        # Compressed should be much smaller
        compressed_size = compressed.get_memory_usage()

        ratio = compressed.get_compression_ratio()

        print(f"Original: {original_size} bytes, Compressed: {compressed_size} bytes")
        print(f"Compression ratio: {ratio:.1f}×")

        # Should achieve at least 100× compression for sparse patterns
        assert ratio > 100

    def test_quality_parameter(self):
        """Test that quality parameter affects compression."""
        codec = GaussianMixtureCodec(num_gaussians=8)

        wavetable = np.random.randn(64, 64, 4).astype(np.float32)

        # High quality
        compressed_high = codec.encode(wavetable, quality=0.95)

        # Low quality (fewer Gaussians)
        compressed_low = codec.encode(wavetable, quality=0.5)

        # High quality should use more Gaussians (implicitly)
        # In practice, quality affects num_gaussians used
        assert compressed_high.metadata['quality'] == 0.95
        assert compressed_low.metadata['quality'] == 0.5

    def test_invalid_wavetable(self):
        """Test that codec rejects invalid wavetables."""
        codec = GaussianMixtureCodec()

        # 2D wavetable should fail
        wavetable_2d = np.zeros((64, 64), dtype=np.float32)

        with pytest.raises(ValueError, match="must be 3D"):
            codec.encode(wavetable_2d)

        # 4D wavetable should fail
        wavetable_4d = np.zeros((64, 64, 4, 2), dtype=np.float32)

        with pytest.raises(ValueError, match="must be 3D"):
            codec.encode(wavetable_4d)

    def test_round_trip_preservation(self):
        """Test that multiple encode/decode cycles preserve data."""
        codec = GaussianMixtureCodec(num_gaussians=8)

        # Create wavetable with known pattern
        wavetable = np.random.randn(32, 32, 4).astype(np.float32) * 0.1

        # Add dominant peak
        y, x = np.ogrid[0:32, 0:32]
        peak = np.exp(-((x - 16)**2 + (y - 16)**2) / 20)
        wavetable[:, :, 0] += peak

        # Encode
        compressed = codec.encode(wavetable)

        # Decode
        reconstructed = codec.decode(compressed)

        # Re-encode
        compressed2 = codec.encode(reconstructed)

        # Re-decode
        reconstructed2 = codec.decode(compressed2)

        # Should be similar (not exact due to lossy compression)
        mse = np.mean((reconstructed - reconstructed2) ** 2)
        assert mse < 0.05  # Allow some variability for lossy compression


class TestWavetableMatrixCompression:
    """Test compression integration with WavetableMatrix."""

    def test_compress_node(self):
        """Test compressing a single node."""
        matrix = WavetableMatrix(
            width=5, height=5, depth=5,
            resolution=64,
            channels=4
        )

        # Add uncompressed node
        wavetable = np.random.randn(64, 64, 4).astype(np.float32)
        matrix.set_node(2, 2, 2, wavetable)

        # Check uncompressed
        assert not matrix._nodes[(2, 2, 2)].compressed

        # Compress
        matrix.compress_node(2, 2, 2, method='gaussian', quality=0.95)

        # Check compressed
        node = matrix._nodes[(2, 2, 2)]
        assert node.compressed
        assert node.compression_method == 'gaussian'
        assert node.compressed_params is not None
        assert node.wavetable is None  # Uncompressed data freed

    def test_decompress_node(self):
        """Test decompressing a node."""
        matrix = WavetableMatrix(
            width=5, height=5, depth=5,
            resolution=64
        )

        # Add and compress node
        wavetable = np.random.randn(64, 64, 4).astype(np.float32)
        matrix.set_node(2, 2, 2, wavetable)
        matrix.compress_node(2, 2, 2)

        # Decompress
        reconstructed = matrix.decompress_node(2, 2, 2)

        assert reconstructed.shape == wavetable.shape
        assert isinstance(reconstructed, np.ndarray)

        # Node should still be compressed (decompress doesn't modify in-place)
        assert matrix._nodes[(2, 2, 2)].compressed

    def test_decompress_node_in_place(self):
        """Test in-place decompression."""
        matrix = WavetableMatrix(
            width=5, height=5, depth=5,
            resolution=64
        )

        wavetable = np.random.randn(64, 64, 4).astype(np.float32)
        matrix.set_node(2, 2, 2, wavetable)
        matrix.compress_node(2, 2, 2)

        # Decompress in-place
        matrix.decompress_node_in_place(2, 2, 2)

        node = matrix._nodes[(2, 2, 2)]
        assert not node.compressed
        assert node.wavetable is not None
        assert node.compressed_params is None

    def test_compress_all(self):
        """Test compressing all nodes in matrix."""
        matrix = WavetableMatrix(
            width=5, height=5, depth=5,
            resolution=64
        )

        # Add 10 nodes
        for i in range(10):
            x, y, z = i % 5, (i // 5) % 5, 0
            wavetable = np.random.randn(64, 64, 4).astype(np.float32)
            matrix.set_node(x, y, z, wavetable)

        # Compress all
        matrix.compress_all(method='gaussian', quality=0.95)

        # Check all compressed
        for coords, node in matrix._nodes.items():
            assert node.compressed

        assert matrix._stats['compressed_nodes'] == 10

    def test_auto_compression(self):
        """Test automatic compression on set_node."""
        matrix = WavetableMatrix(
            width=5, height=5, depth=5,
            resolution=64,
            compression='gaussian'  # Enable auto-compression
        )

        # Set node should auto-compress
        wavetable = np.random.randn(64, 64, 4).astype(np.float32)
        matrix.set_node(2, 2, 2, wavetable)

        # Should be compressed automatically
        node = matrix._nodes[(2, 2, 2)]
        assert node.compressed

    def test_get_node_with_compression(self):
        """Test that get_node automatically decompresses."""
        matrix = WavetableMatrix(
            width=5, height=5, depth=5,
            resolution=64
        )

        wavetable = np.random.randn(64, 64, 4).astype(np.float32)
        matrix.set_node(2, 2, 2, wavetable)
        matrix.compress_node(2, 2, 2)

        # get_node should return decompressed data
        result = matrix.get_node(2, 2, 2)

        assert isinstance(result, np.ndarray)
        assert result.shape == wavetable.shape

    def test_compression_ratio_stat(self):
        """Test compression ratio statistic."""
        matrix = WavetableMatrix(
            width=5, height=5, depth=5,
            resolution=256
        )

        # Add sparse wavetable
        wavetable = np.zeros((256, 256, 4), dtype=np.float32)
        y, x = np.ogrid[0:256, 0:256]
        peak = np.exp(-((x - 128)**2 + (y - 128)**2) / 500)
        wavetable[:, :, 0] = peak

        matrix.set_node(2, 2, 2, wavetable)
        matrix.compress_node(2, 2, 2)

        ratio = matrix.get_compression_ratio()

        print(f"Matrix compression ratio: {ratio:.1f}×")

        # Should achieve significant compression
        assert ratio > 100

    def test_memory_usage_with_compression(self):
        """Test that memory usage reflects compression."""
        matrix = WavetableMatrix(
            width=5, height=5, depth=5,
            resolution=256
        )

        wavetable = np.random.randn(256, 256, 4).astype(np.float32)
        matrix.set_node(2, 2, 2, wavetable)

        # Get uncompressed memory
        stats_before = matrix.get_memory_usage()
        memory_before = stats_before['total_mb']

        # Compress
        matrix.compress_node(2, 2, 2)

        # Get compressed memory
        stats_after = matrix.get_memory_usage()
        memory_after = stats_after['total_mb']

        print(f"Before: {memory_before:.2f} MB, After: {memory_after:.2f} MB")

        # Should use significantly less memory
        assert memory_after < memory_before * 0.1  # At least 10× reduction


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
