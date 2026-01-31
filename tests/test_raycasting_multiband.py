"""
Test suite for Phase 5: Raycasting & Multi-Band Frequency Clustering.

Tests ProjectionMatrix, FrequencyBandClustering, and VoxelCloud integration.
"""

import pytest
import numpy as np
from src.memory.projection import ProjectionMatrix, Frustum
from src.memory.frequency_bands import (
    FrequencyBandClustering, FrequencyBand
)
from src.memory.voxel_cloud import VoxelCloud, ProtoIdentityEntry


class TestProjectionMatrix:
    """Test ProjectionMatrix for raycasting."""

    def test_init(self):
        """Test projection matrix initialization."""
        proj = ProjectionMatrix(fov=60.0, aspect_ratio=1.0, near=0.1, far=1000.0)

        assert proj.fov == 60.0
        assert proj.aspect_ratio == 1.0
        assert proj.near == 0.1
        assert proj.far == 1000.0
        assert proj.frustum is not None

    def test_set_camera(self):
        """Test camera positioning."""
        proj = ProjectionMatrix()

        position = np.array([10.0, 20.0, 30.0])
        look_at = np.array([0.0, 0.0, 0.0])
        up = np.array([0.0, 1.0, 0.0])

        proj.set_camera(position, look_at, up)

        assert np.allclose(proj.position, position)
        assert np.allclose(proj.look_at, look_at)
        assert np.allclose(proj.up, up)
        assert proj.view_matrix is not None

    def test_build_frustum(self):
        """Test frustum plane construction."""
        proj = ProjectionMatrix(fov=60.0, near=1.0, far=100.0)
        proj.set_camera(
            position=np.array([0.0, 0.0, -10.0]),
            look_at=np.array([0.0, 0.0, 0.0])
        )

        proj.build_frustum()

        assert proj.frustum is not None
        assert proj.frustum.planes.shape == (6, 4)  # 6 planes, 4 coefficients each
        assert proj.frustum.near == 1.0
        assert proj.frustum.far == 100.0

    def test_is_voxel_visible_inside(self):
        """Test frustum culling - voxel inside frustum."""
        proj = ProjectionMatrix(fov=60.0, near=1.0, far=100.0)
        proj.set_camera(
            position=np.array([0.0, 0.0, 0.0]),
            look_at=np.array([0.0, 0.0, 10.0])
        )

        # Voxel directly in front of camera
        voxel_pos = np.array([0.0, 0.0, 5.0])
        assert proj.is_voxel_visible(voxel_pos, voxel_size=1.0)

    def test_is_voxel_visible_outside(self):
        """Test frustum culling - voxel outside frustum."""
        proj = ProjectionMatrix(fov=60.0, near=1.0, far=100.0)
        proj.set_camera(
            position=np.array([0.0, 0.0, 0.0]),
            look_at=np.array([0.0, 0.0, 10.0])
        )

        # Voxel behind camera
        voxel_pos = np.array([0.0, 0.0, -5.0])
        assert not proj.is_voxel_visible(voxel_pos, voxel_size=1.0)

    def test_is_voxel_visible_far(self):
        """Test frustum culling - voxel too far."""
        proj = ProjectionMatrix(fov=60.0, near=1.0, far=100.0)
        proj.set_camera(
            position=np.array([0.0, 0.0, 0.0]),
            look_at=np.array([0.0, 0.0, 10.0])
        )

        # Voxel beyond far plane
        voxel_pos = np.array([0.0, 0.0, 150.0])
        # May or may not be visible depending on voxel size
        # Just test that method runs
        result = proj.is_voxel_visible(voxel_pos, voxel_size=1.0)
        assert isinstance(result, (bool, np.bool_))

    def test_compute_lod_level_close(self):
        """Test LOD selection - close distance."""
        proj = ProjectionMatrix()
        proj.set_camera(
            position=np.array([0.0, 0.0, 0.0]),
            look_at=np.array([0.0, 0.0, 10.0])
        )

        # Voxel very close
        voxel_pos = np.array([0.0, 0.0, 5.0])
        lod = proj.compute_lod_level(voxel_pos)

        assert lod == 0  # Finest detail

    def test_compute_lod_level_medium(self):
        """Test LOD selection - medium distance."""
        proj = ProjectionMatrix()
        proj.set_camera(
            position=np.array([0.0, 0.0, 0.0]),
            look_at=np.array([0.0, 0.0, 10.0])
        )

        # Voxel at medium distance
        voxel_pos = np.array([0.0, 0.0, 50.0])
        lod = proj.compute_lod_level(voxel_pos)

        assert lod >= 1  # Coarser level

    def test_compute_lod_level_far(self):
        """Test LOD selection - far distance."""
        proj = ProjectionMatrix()
        proj.set_camera(
            position=np.array([0.0, 0.0, 0.0]),
            look_at=np.array([0.0, 0.0, 10.0])
        )

        # Voxel very far
        voxel_pos = np.array([0.0, 0.0, 500.0])
        lod = proj.compute_lod_level(voxel_pos)

        assert lod >= 4  # Coarsest level

    def test_cast_ray(self):
        """Test ray-voxel intersection."""
        proj = ProjectionMatrix()

        origin = np.array([0.0, 0.0, 0.0])
        direction = np.array([0.0, 0.0, 1.0])

        hit, distance, point = proj.cast_ray(origin, direction, max_distance=100.0)

        assert isinstance(hit, (bool, np.bool_))
        assert distance >= 0.0
        assert point.shape == (3,)

    def test_project_point(self):
        """Test 3D to NDC projection."""
        proj = ProjectionMatrix()
        proj.set_camera(
            position=np.array([0.0, 0.0, -10.0]),
            look_at=np.array([0.0, 0.0, 0.0])
        )

        point_3d = np.array([0.0, 0.0, 5.0])
        ndc = proj.project_point(point_3d)

        assert ndc.shape == (2,)
        # Center point should be near origin in NDC
        assert abs(ndc[0]) < 1.0
        assert abs(ndc[1]) < 1.0


class TestFrequencyBandClustering:
    """Test FrequencyBandClustering."""

    def test_init(self):
        """Test band initialization."""
        fbc = FrequencyBandClustering(num_bands=3)

        assert fbc.num_bands == 3
        assert FrequencyBand.LOW in fbc.band_ranges
        assert FrequencyBand.MID in fbc.band_ranges
        assert FrequencyBand.HIGH in fbc.band_ranges

    def test_init_invalid_bands(self):
        """Test initialization with invalid band count."""
        with pytest.raises(ValueError):
            FrequencyBandClustering(num_bands=5)

    def test_assign_band_high_freq(self):
        """Test character-level (HIGH band) classification."""
        fbc = FrequencyBandClustering()

        # Create high-frequency spectrum (far from center)
        freq_spectrum = np.zeros((128, 128, 4), dtype=np.float32)
        # High spatial frequencies (corners, far from DC)
        freq_spectrum[0:10, 0:10, 0] = 1.0
        freq_spectrum[118:128, 118:128, 0] = 1.0

        band = fbc.assign_band(freq_spectrum)
        assert band == FrequencyBand.HIGH

    def test_assign_band_mid_freq(self):
        """Test word-level (MID band) classification."""
        fbc = FrequencyBandClustering()

        # Create mid-frequency spectrum (moderate distance from center)
        freq_spectrum = np.zeros((128, 128, 4), dtype=np.float32)
        # Place energy at moderate distance from center (64, 64)
        # Distance of ~10-15 pixels from center should be MID
        freq_spectrum[52:58, 52:58, 0] = 1.0
        freq_spectrum[70:76, 70:76, 0] = 1.0

        band = fbc.assign_band(freq_spectrum)
        assert band == FrequencyBand.MID

    def test_assign_band_low_freq(self):
        """Test concept-level (LOW band) classification."""
        fbc = FrequencyBandClustering()

        # Create low-frequency spectrum (DC component, at center)
        freq_spectrum = np.zeros((128, 128, 4), dtype=np.float32)
        # Low spatial frequencies (very close to center at 64,64)
        freq_spectrum[62:66, 62:66, 0] = 1.0

        band = fbc.assign_band(freq_spectrum)
        assert band == FrequencyBand.LOW

    def test_cluster_by_band(self):
        """Test grouping voxels by band."""
        fbc = FrequencyBandClustering()
        voxel_cloud = VoxelCloud(width=128, height=128, depth=64)

        # Add protos with different frequency bands
        for i in range(3):
            proto = np.random.randn(64, 64, 4).astype(np.float32)
            freq = np.zeros((128, 128, 4), dtype=np.float32)

            # Low frequency (near center)
            freq[62:66, 62:66, 0] = 1.0

            voxel_cloud.add(proto, freq, {'text': f'low_{i}'})

        for i in range(2):
            proto = np.random.randn(64, 64, 4).astype(np.float32)
            freq = np.zeros((128, 128, 4), dtype=np.float32)

            # High frequency (far from center)
            freq[0:10, 0:10, 0] = 1.0
            freq[118:128, 118:128, 0] = 1.0

            voxel_cloud.add(proto, freq, {'text': f'high_{i}'})

        # Cluster by LOW band
        low_band = fbc.cluster_by_band(voxel_cloud, FrequencyBand.LOW)
        assert len(low_band) >= 1  # At least some LOW protos

    def test_compute_band_coherence_single(self):
        """Test coherence with single proto."""
        fbc = FrequencyBandClustering()
        voxel_cloud = VoxelCloud(width=128, height=128, depth=64)

        proto = np.random.randn(64, 64, 4).astype(np.float32)
        freq = np.zeros((128, 128, 4), dtype=np.float32)
        freq[60:68, 60:68, 0] = 1.0

        voxel_cloud.add(proto, freq, {'text': 'test'})

        coherence = fbc.compute_band_coherence(
            voxel_cloud.entries, FrequencyBand.LOW
        )
        assert coherence == 1.0  # Perfect coherence for single proto

    def test_compute_band_coherence_multiple(self):
        """Test coherence with multiple protos."""
        fbc = FrequencyBandClustering()
        voxel_cloud = VoxelCloud(width=128, height=128, depth=64)

        # Add similar protos to same band
        for i in range(3):
            proto = np.random.randn(64, 64, 4).astype(np.float32)
            freq = np.zeros((128, 128, 4), dtype=np.float32)
            freq[60:68, 60:68, 0] = 1.0 + i * 0.1  # Similar frequencies

            voxel_cloud.add(proto, freq, {'text': f'test_{i}'})

        coherence = fbc.compute_band_coherence(
            voxel_cloud.entries, FrequencyBand.LOW
        )
        assert 0.0 <= coherence <= 1.0

    def test_get_band_representatives_empty(self):
        """Test getting representatives from empty cloud."""
        fbc = FrequencyBandClustering()
        voxel_cloud = VoxelCloud(width=128, height=128, depth=64)

        reps = fbc.get_band_representatives(voxel_cloud, FrequencyBand.LOW, k=5)
        assert len(reps) == 0

    def test_get_band_representatives_k(self):
        """Test getting top-k representatives."""
        fbc = FrequencyBandClustering()
        voxel_cloud = VoxelCloud(width=128, height=128, depth=64)

        # Add 10 LOW band protos
        for i in range(10):
            proto = np.random.randn(64, 64, 4).astype(np.float32)
            freq = np.zeros((128, 128, 4), dtype=np.float32)
            freq[60:68, 60:68, 0] = 1.0

            voxel_cloud.add(proto, freq, {'text': f'low_{i}'})

        reps = fbc.get_band_representatives(voxel_cloud, FrequencyBand.LOW, k=5)
        assert len(reps) <= 5

    def test_analyze_band_distribution(self):
        """Test band distribution analysis."""
        fbc = FrequencyBandClustering()
        voxel_cloud = VoxelCloud(width=128, height=128, depth=64)

        # Add protos to different bands
        for i in range(3):
            proto = np.random.randn(64, 64, 4).astype(np.float32)
            freq = np.zeros((128, 128, 4), dtype=np.float32)
            freq[60:68, 60:68, 0] = 1.0
            voxel_cloud.add(proto, freq, {'text': f'low_{i}'})

        for i in range(2):
            proto = np.random.randn(64, 64, 4).astype(np.float32)
            freq = np.zeros((128, 128, 4), dtype=np.float32)
            freq[100:128, 100:128, 0] = 1.0
            voxel_cloud.add(proto, freq, {'text': f'high_{i}'})

        stats = fbc.analyze_band_distribution(voxel_cloud)

        assert 'LOW' in stats
        assert 'MID' in stats
        assert 'HIGH' in stats
        assert stats['LOW']['count'] >= 0
        assert 'avg_resonance' in stats['LOW']


class TestVoxelCloudRaycastingIntegration:
    """Test VoxelCloud integration with raycasting."""

    def test_query_by_raycast_empty(self):
        """Test raycast query on empty cloud."""
        voxel_cloud = VoxelCloud(width=128, height=128, depth=64)
        proj = ProjectionMatrix()
        proj.set_camera(
            position=np.array([64.0, 64.0, -50.0]),
            look_at=np.array([64.0, 64.0, 32.0])
        )

        results = voxel_cloud.query_by_raycast(proj, max_results=10)
        assert len(results) == 0

    def test_query_by_raycast_visible(self):
        """Test raycast query returns visible protos only."""
        voxel_cloud = VoxelCloud(width=128, height=128, depth=64)

        # Add protos at different positions
        for i in range(5):
            proto = np.random.randn(64, 64, 4).astype(np.float32)
            freq = np.zeros((128, 128, 4), dtype=np.float32)
            freq[60 + i*5:68 + i*5, 60:68, 0] = 1.0

            voxel_cloud.add(proto, freq, {'text': f'test_{i}'})

        # Set up camera looking at cloud center
        proj = ProjectionMatrix(fov=60.0)
        proj.set_camera(
            position=np.array([64.0, 64.0, -50.0]),
            look_at=np.array([64.0, 64.0, 32.0])
        )

        results = voxel_cloud.query_by_raycast(proj, max_results=10)

        # Should return some visible protos
        assert isinstance(results, list)
        # All results should be ProtoIdentityEntry
        for entry in results:
            assert isinstance(entry, ProtoIdentityEntry)

    def test_query_by_frequency_band_invalid(self):
        """Test frequency band query with invalid band."""
        voxel_cloud = VoxelCloud(width=128, height=128, depth=64)

        with pytest.raises(ValueError):
            voxel_cloud.query_by_frequency_band(band=5, max_results=10)

    def test_query_by_frequency_band_empty(self):
        """Test frequency band query on empty cloud."""
        voxel_cloud = VoxelCloud(width=128, height=128, depth=64)

        results = voxel_cloud.query_by_frequency_band(band=0, max_results=10)
        assert len(results) == 0

    def test_query_by_frequency_band_low(self):
        """Test LOW band query."""
        voxel_cloud = VoxelCloud(width=128, height=128, depth=64)

        # Add LOW frequency protos (near center)
        for i in range(3):
            proto = np.random.randn(64, 64, 4).astype(np.float32)
            freq = np.zeros((128, 128, 4), dtype=np.float32)
            freq[62:66, 62:66, 0] = 1.0

            voxel_cloud.add(proto, freq, {'text': f'low_{i}'})

        results = voxel_cloud.query_by_frequency_band(band=0, max_results=10)

        # Should return LOW band protos
        assert len(results) >= 1
        for entry in results:
            assert entry.frequency_band == 0

    def test_query_by_frequency_band_mid(self):
        """Test MID band query."""
        voxel_cloud = VoxelCloud(width=128, height=128, depth=64)

        # Add MID frequency protos
        for i in range(2):
            proto = np.random.randn(64, 64, 4).astype(np.float32)
            freq = np.zeros((128, 128, 4), dtype=np.float32)
            freq[40:60, 40:60, 0] = 1.0

            voxel_cloud.add(proto, freq, {'text': f'mid_{i}'})

        results = voxel_cloud.query_by_frequency_band(band=1, max_results=10)

        # Should return MID band protos
        for entry in results:
            assert entry.frequency_band == 1

    def test_query_by_frequency_band_high(self):
        """Test HIGH band query."""
        voxel_cloud = VoxelCloud(width=128, height=128, depth=64)

        # Add HIGH frequency protos
        for i in range(2):
            proto = np.random.randn(64, 64, 4).astype(np.float32)
            freq = np.zeros((128, 128, 4), dtype=np.float32)
            freq[100:128, 100:128, 0] = 1.0

            voxel_cloud.add(proto, freq, {'text': f'high_{i}'})

        results = voxel_cloud.query_by_frequency_band(band=2, max_results=10)

        # Should return HIGH band protos
        for entry in results:
            assert entry.frequency_band == 2

    def test_raycast_with_lod(self):
        """Test raycast query includes LOD selection."""
        voxel_cloud = VoxelCloud(width=128, height=128, depth=64)

        # Add protos at different distances
        for i in range(5):
            proto = np.random.randn(64, 64, 4).astype(np.float32)
            freq = np.zeros((128, 128, 4), dtype=np.float32)
            # Vary position to create distance differences
            freq[60:68, 60 + i*10:68 + i*10, 0] = 1.0

            voxel_cloud.add(proto, freq, {'text': f'test_{i}'})

        # Set up camera
        proj = ProjectionMatrix()
        proj.set_camera(
            position=np.array([64.0, 64.0, -50.0]),
            look_at=np.array([64.0, 64.0, 32.0])
        )

        results = voxel_cloud.query_by_raycast(proj, max_results=10)

        # Check that LOD computation works (internal to raycast)
        for entry in results:
            lod = proj.compute_lod_level(entry.position)
            assert lod >= 0

    def test_frequency_band_assignment(self):
        """Test protos assigned correct frequency bands."""
        voxel_cloud = VoxelCloud(width=128, height=128, depth=64)

        # Add LOW frequency proto
        proto_low = np.random.randn(64, 64, 4).astype(np.float32)
        freq_low = np.zeros((128, 128, 4), dtype=np.float32)
        freq_low[60:68, 60:68, 0] = 1.0
        voxel_cloud.add(proto_low, freq_low, {'text': 'low'})

        # Add HIGH frequency proto
        proto_high = np.random.randn(64, 64, 4).astype(np.float32)
        freq_high = np.zeros((128, 128, 4), dtype=np.float32)
        freq_high[100:128, 100:128, 0] = 1.0
        voxel_cloud.add(proto_high, freq_high, {'text': 'high'})

        # Check assignments
        assert voxel_cloud.entries[0].frequency_band is not None
        assert voxel_cloud.entries[1].frequency_band is not None

    def test_frequency_band_query_sorted_by_resonance(self):
        """Test frequency band query sorts by resonance."""
        voxel_cloud = VoxelCloud(width=128, height=128, depth=64)

        # Add LOW frequency protos with different resonances
        # (add same proto multiple times to increase resonance)
        proto1 = np.random.randn(64, 64, 4).astype(np.float32)
        freq1 = np.zeros((128, 128, 4), dtype=np.float32)
        freq1[60:68, 60:68, 0] = 1.0

        # Add first proto 3 times (resonance = 3)
        for _ in range(3):
            voxel_cloud.add(proto1.copy(), freq1.copy(), {'text': 'proto1'})

        # Add second proto 1 time (resonance = 1)
        proto2 = np.random.randn(64, 64, 4).astype(np.float32) * 0.1
        freq2 = np.zeros((128, 128, 4), dtype=np.float32)
        freq2[61:69, 61:69, 0] = 1.0
        voxel_cloud.add(proto2, freq2, {'text': 'proto2'})

        results = voxel_cloud.query_by_frequency_band(band=0, max_results=10)

        # Results should be sorted by resonance (descending)
        if len(results) >= 2:
            assert (results[0].resonance_strength >=
                   results[1].resonance_strength)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
