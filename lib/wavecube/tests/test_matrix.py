"""
Unit tests for WavetableMatrix core functionality.

Tests node operations, sparse storage, interpolation, and I/O.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from wavecube.core import WavetableMatrix, WavetableNode


class TestWavetableNode:
    """Test WavetableNode dataclass."""

    def test_create_node(self):
        """Test basic node creation."""
        wavetable = np.random.randn(64, 64, 4).astype(np.float32)
        node = WavetableNode(
            wavetable=wavetable,
            coordinates=(0, 0, 0),
        )

        assert node.wavetable.shape == (64, 64, 4)
        assert node.coordinates == (0, 0, 0)
        assert node.resolution == (64, 64)
        assert node.channels == 4
        assert not node.compressed

    def test_node_validation(self):
        """Test node validation on creation."""
        # Invalid wavetable dimension
        with pytest.raises(ValueError, match="must be 3D"):
            WavetableNode(wavetable=np.random.randn(64, 64))

    def test_node_properties(self):
        """Test node properties."""
        wavetable = np.random.randn(128, 128, 4).astype(np.float32)
        node = WavetableNode(wavetable=wavetable)

        assert node.shape == (128, 128, 4)
        assert node.size == 128 * 128 * 4
        assert node.memory_bytes == wavetable.nbytes
        assert node.is_valid()

    def test_compressed_node_validation(self):
        """Test validation for compressed nodes."""
        # Missing compression_method
        with pytest.raises(ValueError, match="must have compression_method"):
            WavetableNode(
                compressed=True,
                compressed_params={'foo': 'bar'}
            )

        # Missing compressed_params
        with pytest.raises(ValueError, match="must have compressed_params"):
            WavetableNode(
                compressed=True,
                compression_method='gaussian'
            )


class TestWavetableMatrix:
    """Test WavetableMatrix core functionality."""

    def test_create_matrix(self):
        """Test basic matrix creation."""
        matrix = WavetableMatrix(
            width=10, height=10, depth=10,
            resolution=256,
            channels=4
        )

        assert matrix.width == 10
        assert matrix.height == 10
        assert matrix.depth == 10
        assert matrix.resolution == (256, 256)
        assert matrix.channels == 4
        assert matrix.sparse

    def test_matrix_validation(self):
        """Test matrix parameter validation."""
        # Invalid dimensions
        with pytest.raises(ValueError, match="must be positive"):
            WavetableMatrix(width=0, height=10, depth=10)

    def test_set_get_node(self):
        """Test setting and getting nodes."""
        matrix = WavetableMatrix(width=10, height=10, depth=10, resolution=64)

        wavetable = np.random.randn(64, 64, 4).astype(np.float32)
        matrix.set_node(5, 5, 5, wavetable)

        retrieved = matrix.get_node(5, 5, 5)
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved, wavetable)

    def test_has_node(self):
        """Test node existence checking."""
        matrix = WavetableMatrix(width=10, height=10, depth=10)

        wavetable = np.random.randn(64, 64, 4).astype(np.float32)
        matrix.set_node(3, 3, 3, wavetable)

        assert matrix.has_node(3, 3, 3)
        assert not matrix.has_node(0, 0, 0)

    def test_delete_node(self):
        """Test node deletion."""
        matrix = WavetableMatrix(width=10, height=10, depth=10)

        wavetable = np.random.randn(64, 64, 4).astype(np.float32)
        matrix.set_node(2, 2, 2, wavetable)

        assert matrix.has_node(2, 2, 2)

        matrix.delete_node(2, 2, 2)

        assert not matrix.has_node(2, 2, 2)
        assert matrix.get_node(2, 2, 2) is None

    def test_bounds_checking(self):
        """Test coordinate bounds validation."""
        matrix = WavetableMatrix(width=5, height=5, depth=5)

        wavetable = np.random.randn(64, 64, 4).astype(np.float32)

        # Out of bounds
        with pytest.raises(IndexError, match="out of bounds"):
            matrix.set_node(10, 0, 0, wavetable)

        with pytest.raises(IndexError, match="out of bounds"):
            matrix.get_node(0, 10, 0)

    def test_dict_like_access(self):
        """Test dictionary-style access."""
        matrix = WavetableMatrix(width=10, height=10, depth=10)

        wavetable = np.random.randn(64, 64, 4).astype(np.float32)

        # Set using []
        matrix[3, 3, 3] = wavetable

        # Get using []
        retrieved = matrix[3, 3, 3]
        np.testing.assert_array_equal(retrieved, wavetable)

        # Check using 'in'
        assert (3, 3, 3) in matrix
        assert (0, 0, 0) not in matrix

    def test_memory_tracking(self):
        """Test memory usage tracking."""
        matrix = WavetableMatrix(width=10, height=10, depth=10, resolution=128)

        # Initially empty
        stats = matrix.get_memory_usage()
        assert stats['num_nodes'] == 0
        assert stats['total_bytes'] == 0

        # Add a node
        wavetable = np.random.randn(128, 128, 4).astype(np.float32)
        matrix.set_node(0, 0, 0, wavetable)

        stats = matrix.get_memory_usage()
        assert stats['num_nodes'] == 1
        assert stats['total_bytes'] == wavetable.nbytes

    def test_clear(self):
        """Test matrix clearing."""
        matrix = WavetableMatrix(width=10, height=10, depth=10)

        # Add some nodes
        for i in range(3):
            wavetable = np.random.randn(64, 64, 4).astype(np.float32)
            matrix.set_node(i, i, i, wavetable)

        assert matrix.get_memory_usage()['num_nodes'] == 3

        # Clear
        matrix.clear()

        assert matrix.get_memory_usage()['num_nodes'] == 0
        assert not matrix.has_node(0, 0, 0)


class TestInterpolation:
    """Test trilinear interpolation."""

    def test_exact_sampling(self):
        """Test sampling at exact grid points (no interpolation)."""
        matrix = WavetableMatrix(width=10, height=10, depth=10, resolution=64)

        wavetable = np.random.randn(64, 64, 4).astype(np.float32)
        matrix.set_node(5, 5, 5, wavetable)

        # Sample at exact position
        result = matrix.sample(5.0, 5.0, 5.0)

        np.testing.assert_array_almost_equal(result, wavetable, decimal=5)

    def test_linear_interpolation(self):
        """Test interpolation between two nodes."""
        matrix = WavetableMatrix(width=10, height=10, depth=10, resolution=32)

        # Create two simple wavetables
        w0 = np.zeros((32, 32, 4), dtype=np.float32)
        w1 = np.ones((32, 32, 4), dtype=np.float32)

        matrix.set_node(0, 0, 0, w0)
        matrix.set_node(1, 0, 0, w1)

        # Sample at midpoint
        result = matrix.sample(0.5, 0.0, 0.0)

        # Should be 0.5 everywhere
        expected = np.full((32, 32, 4), 0.5, dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_trilinear_interpolation(self):
        """Test full 3D interpolation."""
        matrix = WavetableMatrix(width=10, height=10, depth=10, resolution=16)

        # Create 8 corner nodes with known values
        for x in [0, 1]:
            for y in [0, 1]:
                for z in [0, 1]:
                    value = x + 2*y + 4*z  # Unique value for each corner
                    w = np.full((16, 16, 4), value, dtype=np.float32)
                    matrix.set_node(x, y, z, w)

        # Sample at center
        result = matrix.sample(0.5, 0.5, 0.5)

        # Expected value is average of all corners
        expected_value = (0 + 1 + 2 + 3 + 4 + 5 + 6 + 7) / 8.0
        expected = np.full((16, 16, 4), expected_value, dtype=np.float32)

        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_interpolation_bounds(self):
        """Test interpolation bounds validation."""
        matrix = WavetableMatrix(width=5, height=5, depth=5)

        wavetable = np.random.randn(32, 32, 4).astype(np.float32)
        matrix.set_node(0, 0, 0, wavetable)

        # Out of bounds
        with pytest.raises(ValueError, match="out of bounds"):
            matrix.sample(10.0, 0.0, 0.0)

    def test_empty_region_interpolation(self):
        """Test interpolation in empty region raises error."""
        matrix = WavetableMatrix(width=10, height=10, depth=10, sparse=True)

        # No nodes in matrix
        with pytest.raises(RuntimeError, match="No nodes found"):
            matrix.sample(5.5, 5.5, 5.5)

    def test_batch_sampling(self):
        """Test batch sampling."""
        matrix = WavetableMatrix(width=10, height=10, depth=10, resolution=32)

        wavetable = np.random.randn(32, 32, 4).astype(np.float32)
        matrix.set_node(5, 5, 5, wavetable)

        # Sample multiple coordinates
        coords = np.array([
            [5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0],
        ])

        results = matrix.sample_batch(coords)

        assert results.shape == (3, 32, 32, 4)

        # All should be equal to wavetable
        for i in range(3):
            np.testing.assert_array_almost_equal(results[i], wavetable, decimal=5)


class TestSerialization:
    """Test matrix save/load."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)

    def test_save_load_npz(self, temp_dir):
        """Test NPZ save/load round-trip."""
        # Create matrix
        matrix = WavetableMatrix(width=5, height=5, depth=5, resolution=64, channels=4)

        # Add some nodes
        for i in range(3):
            wavetable = np.random.randn(64, 64, 4).astype(np.float32)
            matrix.set_node(i, i, i, wavetable)

        # Save
        path = Path(temp_dir) / "test_matrix.npz"
        matrix.save(str(path))

        assert path.exists()

        # Load
        loaded = WavetableMatrix.load(str(path))

        # Verify
        assert loaded.width == matrix.width
        assert loaded.height == matrix.height
        assert loaded.depth == matrix.depth
        assert loaded.resolution == matrix.resolution
        assert loaded.channels == matrix.channels

        # Check nodes
        for i in range(3):
            original = matrix.get_node(i, i, i)
            loaded_node = loaded.get_node(i, i, i)
            np.testing.assert_array_equal(loaded_node, original)

    def test_save_with_metadata(self, temp_dir):
        """Test saving nodes with metadata."""
        matrix = WavetableMatrix(width=5, height=5, depth=5)

        wavetable = np.random.randn(64, 64, 4).astype(np.float32)
        metadata = {'octave': 0, 'modality': 'text'}
        matrix.set_node(0, 0, 0, wavetable, metadata=metadata)

        # Save and load
        path = Path(temp_dir) / "test_metadata.npz"
        matrix.save(str(path))
        loaded = WavetableMatrix.load(str(path))

        # Verify metadata is preserved
        node = loaded._nodes[(0, 0, 0)]
        assert node.metadata == metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
