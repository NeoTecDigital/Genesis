"""Unit tests for quaternionic vector extraction."""

import numpy as np
import pytest
from src.origin import Origin


class TestQuaternionExtraction:
    """Test quaternionic vector extraction from standing waves."""

    def test_extract_quaternion_unit_norm(self):
        """Verify extracted quaternion is unit norm."""
        origin = Origin(64, 64, use_gpu=False)

        # Create test standing wave
        standing_wave = np.random.randn(64, 64, 4).astype(np.float32)

        # Extract quaternion
        quaternion = origin.proto_manager.extract_quaternion(standing_wave)

        # Verify unit norm
        norm = np.linalg.norm(quaternion)
        assert abs(norm - 1.0) < 1e-6, f"Quaternion not unit norm: {norm}"

        # Verify shape
        assert quaternion.shape == (4,), f"Wrong shape: {quaternion.shape}"

    def test_extract_quaternion_energy_weighted(self):
        """Verify quaternion extraction is energy-weighted."""
        origin = Origin(64, 64, use_gpu=False)

        # Create standing wave with concentrated energy
        standing_wave = np.zeros((64, 64, 4), dtype=np.float32)

        # Add energy spike in one region
        standing_wave[32, 32, :] = [1.0, 0.5, 0.3, 0.1]

        # Extract quaternion
        quaternion = origin.proto_manager.extract_quaternion(standing_wave)

        # Should reflect the concentrated energy region
        # w channel should be dominant
        assert quaternion[0] > abs(quaternion[1])
        assert quaternion[0] > abs(quaternion[2])
        assert quaternion[0] > abs(quaternion[3])

    def test_multi_octave_quaternions(self):
        """Verify multi-octave quaternion extraction."""
        origin = Origin(128, 128, use_gpu=False)

        # Create test standing wave
        standing_wave = np.random.randn(128, 128, 4).astype(np.float32)

        # Extract multi-octave quaternions
        octave_quaternions = origin.proto_manager.extract_multi_octave_quaternions(
            standing_wave, num_octaves=5
        )

        # Verify we got 5 octave levels
        assert len(octave_quaternions) == 5

        # Verify each is unit norm
        for octave, quaternion in octave_quaternions.items():
            norm = np.linalg.norm(quaternion)
            assert abs(norm - 1.0) < 1e-6, f"Octave {octave} not unit norm: {norm}"

    def test_act_returns_quaternionic_vector(self):
        """Verify Act() returns quaternionic vector."""
        origin = Origin(64, 64, use_gpu=False)

        # Create test standing wave
        gamma_params = {
            'amplitude': 1.0,
            'base_frequency': 2.0,
            'envelope_sigma': 0.45,
            'num_harmonics': 12,
            'harmonic_decay': 0.75,
            'initial_phase': 0.0
        }
        iota_params = {
            'harmonic_coeffs': [1.0]*10,
            'global_amplitude': 1.0,
            'frequency_range': 2.0
        }

        # Create standing wave via Gen
        input_n = np.random.randn(64, 64, 4).astype(np.float32) * 0.1
        standing_wave = origin.Gen(gamma_params, iota_params, input_n=input_n)

        # Act should return quaternionic vector
        result = origin.Act(standing_wave)

        assert hasattr(result, 'quaternionic_vector')
        assert result.quaternionic_vector is not None
        assert result.quaternionic_vector.shape == (4,)
        assert abs(np.linalg.norm(result.quaternionic_vector) - 1.0) < 1e-6

        assert hasattr(result, 'multi_octave_quaternions')
        assert result.multi_octave_quaternions is not None
        assert isinstance(result.multi_octave_quaternions, dict)

    def test_zero_standing_wave_quaternion(self):
        """Verify zero standing wave returns default quaternion."""
        origin = Origin(64, 64, use_gpu=False)

        # Zero standing wave
        standing_wave = np.zeros((64, 64, 4), dtype=np.float32)

        # Should return default [1, 0, 0, 0]
        quaternion = origin.proto_manager.extract_quaternion(standing_wave)

        expected = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert np.allclose(quaternion, expected)

    def test_average_pooling_downsampling(self):
        """Verify average pooling works correctly."""
        origin = Origin(64, 64, use_gpu=False)

        # Create test array with known values
        x = np.ones((64, 64, 4), dtype=np.float32)

        # Downsample by factor 2
        downsampled = origin.proto_manager._average_pool(x, factor=2)

        # Should be half size
        assert downsampled.shape == (32, 32, 4)

        # Values should still be 1.0 (average of 1.0s)
        assert np.allclose(downsampled, 1.0)

        # Downsample by factor 4
        downsampled = origin.proto_manager._average_pool(x, factor=4)
        assert downsampled.shape == (16, 16, 4)

    def test_quaternion_extraction_from_proto_identity(self):
        """Verify quaternion extraction from actual proto-identity."""
        origin = Origin(64, 64, use_gpu=False)

        # Create proto-identity
        gamma_params = {
            'amplitude': 1.0,
            'base_frequency': 2.0,
            'envelope_sigma': 0.45,
            'num_harmonics': 12,
            'harmonic_decay': 0.75,
            'initial_phase': 0.0
        }
        epsilon_params = {
            'extraction_rate': 0.0,
            'focus_sigma': 2.222,
            'threshold': 0.1,
            'preserve_peaks': True
        }

        # Create proto-identity (standing wave from Gen ∪ Res)
        proto_identity = origin.create_proto_identity(gamma_params, epsilon_params)

        # Extract quaternion
        quaternion = origin.proto_manager.extract_quaternion(proto_identity)

        # Should be unit norm
        norm = np.linalg.norm(quaternion)
        assert abs(norm - 1.0) < 1e-6, f"Quaternion not unit norm: {norm}"

        # Should be non-zero (proto-identity is non-trivial)
        assert not np.allclose(quaternion, [1, 0, 0, 0])

    def test_multi_octave_decreasing_resolution(self):
        """Verify multi-octave quaternions have decreasing resolution."""
        origin = Origin(256, 256, use_gpu=False)

        # Create test standing wave with fine detail
        standing_wave = np.random.randn(256, 256, 4).astype(np.float32)

        # Extract multi-octave quaternions
        octave_quaternions = origin.proto_manager.extract_multi_octave_quaternions(
            standing_wave, num_octaves=6
        )

        # Verify we got 6 octave levels
        assert len(octave_quaternions) == 6

        # All should be unit norm
        for octave in range(6):
            norm = np.linalg.norm(octave_quaternions[octave])
            assert abs(norm - 1.0) < 1e-6

    def test_quaternion_consistency_across_octaves(self):
        """Verify quaternions at different octaves are similar for uniform input."""
        origin = Origin(128, 128, use_gpu=False)

        # Create uniform standing wave (should be similar across octaves)
        standing_wave = np.ones((128, 128, 4), dtype=np.float32)
        standing_wave[:, :, 0] = 1.0  # w
        standing_wave[:, :, 1] = 0.5  # x
        standing_wave[:, :, 2] = 0.3  # y
        standing_wave[:, :, 3] = 0.2  # z

        # Extract multi-octave quaternions
        octave_quaternions = origin.proto_manager.extract_multi_octave_quaternions(
            standing_wave, num_octaves=4
        )

        # All octaves should produce similar quaternions (uniform input)
        base_quaternion = octave_quaternions[0]
        for octave in range(1, 4):
            diff = np.linalg.norm(octave_quaternions[octave] - base_quaternion)
            assert diff < 0.1, f"Octave {octave} differs too much from base: {diff}"
