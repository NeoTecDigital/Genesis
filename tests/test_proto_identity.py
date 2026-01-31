"""Unit tests for proto-identity creation and validation."""

import numpy as np
import pytest
from src.proto_identity import ProtoIdentityManager
from src.origin import Origin


class TestProtoIdentityCreation:
    """Test Gen ∪ Res convergence."""

    def test_create_proto_identity_standing_wave(self):
        """Verify proto-identity has nodes and antinodes."""
        origin = Origin(64, 64, use_gpu=False)

        gamma_params = {
            'amplitude': 1.0,
            'base_frequency': 2.0,
            'envelope_sigma': 0.45,
            'num_harmonics': 12,
            'harmonic_decay': 0.75,
            'initial_phase': 0.0
        }

        epsilon_params = origin.proto_manager.derive_epsilon_from_gamma(gamma_params)
        proto = origin.create_proto_identity(gamma_params, epsilon_params)

        # Validate standing wave
        assert origin.proto_manager.validate_standing_wave(proto), \
            "Proto-identity should be valid standing wave"

        # Check amplitude distribution (use relative thresholds like validation)
        amplitude = np.sqrt(np.sum(proto**2, axis=-1))
        max_amp = amplitude.max()
        nodes_fraction = (amplitude < 0.3 * max_amp).sum() / amplitude.size
        antinodes_fraction = (amplitude > 0.7 * max_amp).sum() / amplitude.size

        assert nodes_fraction > 0.10, \
            f"Need >10% nodes (< 0.3×max), got {nodes_fraction:.1%}"
        assert antinodes_fraction > 0.01, \
            f"Need >1% antinodes (> 0.7×max), got {antinodes_fraction:.1%}"

    def test_gen_res_convergence(self):
        """Verify Gen and Res create SAME proto-identity."""
        origin = Origin(64, 64, use_gpu=False)

        gamma_params = {
            'amplitude': 1.0,
            'base_frequency': 2.0,
            'envelope_sigma': 0.45,
            'num_harmonics': 12,
            'harmonic_decay': 0.75,
            'initial_phase': 0.0
        }

        epsilon_params = origin.proto_manager.derive_epsilon_from_gamma(gamma_params)

        # Gen path (query mode)
        iota_params = {
            'harmonic_coeffs': [1.0] * 10,
            'phase_shifts': [0.0] * 10,
            'amplitude_mods': [1.0] * 10
        }
        proto_gen = origin.Gen(gamma_params, iota_params, input_n=None)

        # Res path (query mode)
        tau_params = {
            'projection_strength': 1.0,
            'frequency_bands': 10,
            'adaptive': True
        }
        proto_res = origin.Res(epsilon_params, tau_params, input_n=None)

        # Should be same proto-identity
        similarity = np.dot(proto_gen.flatten(), proto_res.flatten()) / \
                    (np.linalg.norm(proto_gen) * np.linalg.norm(proto_res))

        assert similarity > 0.9, \
            f"Gen/Res should converge, got similarity {similarity:.2f}"

    def test_proto_identity_projection(self):
        """Verify projection preserves energy."""
        origin = Origin(64, 64, use_gpu=False)

        gamma_params = {
            'amplitude': 1.0,
            'base_frequency': 2.0,
            'envelope_sigma': 0.45,
            'num_harmonics': 12,
            'harmonic_decay': 0.75,
            'initial_phase': 0.0
        }

        epsilon_params = origin.proto_manager.derive_epsilon_from_gamma(gamma_params)
        proto = origin.create_proto_identity(gamma_params, epsilon_params)

        # Create test input
        input_n = np.random.randn(64, 64, 4).astype(np.float32)

        # Project
        standing_wave = origin.project_proto_identity(proto, input_n)

        # Verify energy conservation
        proto_energy = np.linalg.norm(proto)
        n_energy = np.linalg.norm(input_n)
        standing_energy = np.linalg.norm(standing_wave)
        geometric_mean = (proto_energy * n_energy) ** 0.5

        # Standing wave energy should be <= geometric mean (with 20% tolerance)
        assert standing_energy <= geometric_mean * 1.2, \
            f"Energy {standing_energy:.2f} exceeds limit {geometric_mean * 1.2:.2f}"

    def test_parameter_derivation(self):
        """Verify gamma ↔ epsilon parameter derivation."""
        origin = Origin(64, 64, use_gpu=False)

        gamma_params = {
            'amplitude': 1.0,
            'base_frequency': 2.0,
            'envelope_sigma': 0.45,
            'num_harmonics': 12,
            'harmonic_decay': 0.75,
            'initial_phase': 0.0
        }

        # Derive epsilon from gamma
        epsilon_params = origin.proto_manager.derive_epsilon_from_gamma(gamma_params)

        # Check inverse relationship
        assert epsilon_params['extraction_rate'] == pytest.approx(
            1.0 - gamma_params['amplitude']
        )
        assert epsilon_params['focus_sigma'] == pytest.approx(
            1.0 / gamma_params['envelope_sigma'], rel=1e-5
        )

        # Round trip
        gamma_params_2 = origin.proto_manager.derive_gamma_from_epsilon(epsilon_params)
        assert gamma_params_2['amplitude'] == pytest.approx(
            gamma_params['amplitude']
        )
        assert gamma_params_2['envelope_sigma'] == pytest.approx(
            gamma_params['envelope_sigma'], rel=1e-5
        )

    def test_zero_proto_identity(self):
        """Test edge case: zero proto-identity."""
        origin = Origin(64, 64, use_gpu=False)

        # Zero proto
        proto = np.zeros((64, 64, 4), dtype=np.float32)
        input_n = np.random.randn(64, 64, 4).astype(np.float32)

        # Should return zeros
        standing_wave = origin.project_proto_identity(proto, input_n)
        assert np.linalg.norm(standing_wave) < 1e-6, \
            "Zero proto should produce zero standing wave"

    def test_zero_input(self):
        """Test edge case: zero input."""
        origin = Origin(64, 64, use_gpu=False)

        gamma_params = {
            'amplitude': 1.0,
            'base_frequency': 2.0,
            'envelope_sigma': 0.45,
            'num_harmonics': 12,
            'harmonic_decay': 0.75,
            'initial_phase': 0.0
        }

        epsilon_params = origin.proto_manager.derive_epsilon_from_gamma(gamma_params)
        proto = origin.create_proto_identity(gamma_params, epsilon_params)

        # Zero input
        input_n = np.zeros((64, 64, 4), dtype=np.float32)

        # Should return zeros
        standing_wave = origin.project_proto_identity(proto, input_n)
        assert np.linalg.norm(standing_wave) < 1e-6, \
            "Zero input should produce zero standing wave"

    def test_backward_compatibility(self):
        """Test that old Gen/Res API still works (no input_n)."""
        origin = Origin(64, 64, use_gpu=False)

        gamma_params = {
            'amplitude': 1.0,
            'base_frequency': 2.0,
            'envelope_sigma': 0.45,
            'num_harmonics': 12,
            'harmonic_decay': 0.75,
            'initial_phase': 0.0
        }

        iota_params = {
            'harmonic_coeffs': [1.0] * 10,
            'phase_shifts': [0.0] * 10,
            'amplitude_mods': [1.0] * 10
        }

        # Old API: Gen returns proto when input_n=None
        proto_gen = origin.Gen(gamma_params, iota_params, input_n=None)

        # Should be valid proto-identity
        assert origin.proto_manager.validate_standing_wave(proto_gen), \
            "Old API should still produce valid proto-identity"
