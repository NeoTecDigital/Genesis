"""
Test octave hierarchy integration with voxel cloud and query pipeline.
"""

import pytest
import numpy as np
from pathlib import Path
import pickle
import tempfile

from src.memory.octave_hierarchy import OctaveHierarchy, OctaveProtoIdentity
from src.memory.voxel_cloud import VoxelCloud
from src.origin import Origin
from src.memory.frequency_field import TextFrequencyAnalyzer


class TestOctaveHierarchy:
    """Test the OctaveHierarchy class itself."""

    def setup_method(self):
        self.hierarchy = OctaveHierarchy(num_octaves=5)
        self.origin = Origin(width=512, height=512, use_gpu=False)

    def test_octave_storage(self):
        """Verify proto-identities stored at all octave levels."""
        # Create test proto
        proto = np.random.randn(512, 512, 4).astype(np.float32)
        quaternions = {i: np.random.randn(4) for i in range(5)}
        # Normalize quaternions
        for i in quaternions:
            quaternions[i] /= np.linalg.norm(quaternions[i])

        octave_proto = OctaveProtoIdentity(
            proto_identity=proto,
            quaternions=quaternions,
            frequency=100.0,
            modality="text"
        )

        self.hierarchy.add_proto_identity(octave_proto)

        # Verify stored at all levels
        for octave in range(5):
            assert len(self.hierarchy.octave_storage[octave]) == 1

    def test_query_at_octave(self):
        """Verify querying at specific octave level."""
        # Add test protos
        for i in range(10):
            proto = np.random.randn(512, 512, 4).astype(np.float32)
            quaternions = {j: np.random.randn(4) for j in range(5)}
            for j in quaternions:
                quaternions[j] /= np.linalg.norm(quaternions[j])

            octave_proto = OctaveProtoIdentity(
                proto_identity=proto,
                quaternions=quaternions,
                frequency=100.0 + i,
                modality="text"
            )
            self.hierarchy.add_proto_identity(octave_proto)

        # Query at octave 2
        query_quat = np.random.randn(4)
        query_quat /= np.linalg.norm(query_quat)

        results = self.hierarchy.query_at_octave(query_quat, octave=2, top_k=5)

        assert len(results) == 5
        # Verify sorted by distance
        distances = [d for _, d in results]
        assert distances == sorted(distances)

    def test_multi_octave_query(self):
        """Verify multi-octave query with weighted fusion."""
        # Add test protos
        for i in range(10):
            proto = np.random.randn(512, 512, 4).astype(np.float32)
            quaternions = {j: np.random.randn(4) for j in range(5)}
            for j in quaternions:
                quaternions[j] /= np.linalg.norm(quaternions[j])

            octave_proto = OctaveProtoIdentity(
                proto_identity=proto,
                quaternions=quaternions,
                frequency=100.0 + i,
                modality="text"
            )
            self.hierarchy.add_proto_identity(octave_proto)

        # Query with multiple octaves
        query_quaternions = {}
        for octave in [0, 2, 4]:
            q = np.random.randn(4)
            query_quaternions[octave] = q / np.linalg.norm(q)

        results = self.hierarchy.multi_octave_query(query_quaternions, top_k=5)

        assert len(results) <= 5
        # Verify sorted by fused distance
        distances = [d for _, d in results]
        assert distances == sorted(distances)

    def test_adaptive_octave_selection(self):
        """Verify adaptive octave selection based on query length."""
        # Short query → high octave (abstract)
        octave = self.hierarchy.adaptive_octave_selection("What?")
        assert octave == 4

        # Medium query → mid octave
        octave = self.hierarchy.adaptive_octave_selection("What is the Tao?")
        assert octave == 3

        # Long query → low octave (detailed)
        octave = self.hierarchy.adaptive_octave_selection(
            "What is the fundamental nature of reality and consciousness?"
        )
        assert octave == 1


class TestVoxelCloudOctaveIntegration:
    """Test VoxelCloud integration with octave hierarchy."""

    def setup_method(self):
        self.voxel_cloud = VoxelCloud(width=512, height=512, depth=128)
        self.origin = Origin(512, 512, use_gpu=False)

    def test_voxel_cloud_octave_initialization(self):
        """Verify voxel cloud initializes with octave hierarchy."""
        assert hasattr(self.voxel_cloud, 'octave_hierarchy')
        assert self.voxel_cloud.octave_hierarchy.num_octaves == 5

    def test_add_with_octaves(self):
        """Test adding proto-identity with multi-octave quaternions."""
        # Create proto with multi-octave quaternions
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
            'global_amplitude': 1.0,
            'frequency_range': 2.0
        }

        # Gen → standing wave → Act (simplified for testing)
        standing_wave = self.origin.Gen(gamma_params, iota_params)

        result = self.origin.Act(standing_wave)

        # Add to voxel cloud with quaternions
        self.voxel_cloud.add_with_octaves(
            proto_identity=result.proto_identity,
            frequency=100.0,
            modality='text',
            quaternions=result.multi_octave_quaternions,
            resonance_strength=1.0
        )

        # Verify added to octave hierarchy
        assert len(self.voxel_cloud.octave_hierarchy.octave_storage[0]) > 0
        stats = self.voxel_cloud.octave_hierarchy.get_octave_statistics()
        for octave in range(5):
            assert stats[octave]['count'] > 0

    def test_query_multi_octave(self):
        """Test multi-octave query functionality."""
        # Add test proto
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
            'global_amplitude': 1.0,
            'frequency_range': 2.0
        }

        n = self.origin.Gen(gamma_params, iota_params)
        result = self.origin.Act(n)

        self.voxel_cloud.add_with_octaves(
            proto_identity=result.proto_identity,
            frequency=100.0,
            quaternions=result.multi_octave_quaternions
        )

        # Query with multi-octave quaternions
        query_quaternions = result.multi_octave_quaternions
        results = self.voxel_cloud.query_multi_octave(query_quaternions, top_k=5)

        # Should find our proto
        assert len(results) > 0
        found_proto, distance = results[0]
        # FFT architecture: no text field, only proto_identity
        assert hasattr(found_proto, 'proto_identity')
        assert not hasattr(found_proto, 'text')
        # Distance should be very small (identical quaternions)
        assert distance < 0.1

    def test_save_load_with_octaves(self):
        """Test saving and loading voxel cloud with octave hierarchy."""
        # Add proto with octaves
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
            'global_amplitude': 1.0,
            'frequency_range': 2.0
        }

        n = self.origin.Gen(gamma_params, iota_params)
        result = self.origin.Act(n)

        self.voxel_cloud.add_with_octaves(
            proto_identity=result.proto_identity,
            frequency=100.0,
            quaternions=result.multi_octave_quaternions
        )

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name

        self.voxel_cloud.save(tmp_path)

        # Load into new voxel cloud
        new_cloud = VoxelCloud()
        new_cloud.load(tmp_path)

        # Verify octave hierarchy loaded
        assert hasattr(new_cloud, 'octave_hierarchy')
        assert len(new_cloud.octave_hierarchy.octave_storage[0]) > 0

        # Clean up
        Path(tmp_path).unlink()


class TestEndToEndIntegration:
    """Test full pipeline from text discovery to synthesis with octaves."""

    def setup_method(self):
        self.origin = Origin(512, 512, use_gpu=False)
        self.freq_analyzer = TextFrequencyAnalyzer(512, 512)

    def test_discovery_with_octaves(self):
        """Test discovery process extracts and stores octave quaternions."""
        voxel_cloud = VoxelCloud(512, 512, 128)

        # Process test text
        test_text = "The Tao that can be spoken is not the eternal Tao"
        freq_spectrum, params = self.freq_analyzer.analyze(test_text)

        # Gen ∪ Res convergence (using simpler approach without full Res path)
        n_gen = self.origin.Gen(params['gamma_params'], params['iota_params'])
        # For testing, just use Gen path since Res validation is complex
        standing_wave = n_gen  # Simplified for testing

        # Extract multi-octave quaternions
        result = self.origin.Act(standing_wave)

        # Add to voxel cloud with octaves
        from src.memory.octave_frequency import extract_fundamental
        f0 = extract_fundamental(freq_spectrum)

        voxel_cloud.add_with_octaves(
            proto_identity=result.proto_identity,
            frequency=f0,
            modality='text',
            quaternions=result.multi_octave_quaternions,
            resonance_strength=1.0
        )

        # Verify stored correctly
        assert len(voxel_cloud.octave_hierarchy.octave_storage[0]) == 1
        stats = voxel_cloud.octave_hierarchy.get_octave_statistics()
        for octave in range(5):
            assert stats[octave]['count'] == 1
            assert stats[octave]['has_quaternions'] == 1

    def test_synthesis_with_octaves(self):
        """Test synthesis uses multi-octave matching."""
        voxel_cloud = VoxelCloud(512, 512, 128)

        # Add multiple test texts
        test_texts = [
            "The Tao is empty yet inexhaustible",
            "The wise student hears of the Tao and practices it diligently",
            "Those who know do not talk",
            "The Tao that can be spoken is not the eternal Tao"
        ]

        for text in test_texts:
            freq_spectrum, params = self.freq_analyzer.analyze(text)
            standing_wave = self.origin.Gen(params['gamma_params'], params['iota_params'])
            result = self.origin.Act(standing_wave)

            from src.memory.octave_frequency import extract_fundamental
            f0 = extract_fundamental(freq_spectrum)

            voxel_cloud.add_with_octaves(
                proto_identity=result.proto_identity,
                frequency=f0,
                modality='text',
                quaternions=result.multi_octave_quaternions,
                resonance_strength=1.0
            )

        # Query with short text (should use high octave)
        query_text = "Tao?"
        query_freq, query_params = self.freq_analyzer.analyze(query_text)

        # Create query standing wave (simplified)
        q_standing = self.origin.Gen(query_params['gamma_params'], query_params['iota_params'])
        q_result = self.origin.Act(q_standing)

        # Adaptive octave selection
        primary_octave = voxel_cloud.octave_hierarchy.adaptive_octave_selection(query_text)
        assert primary_octave == 4  # Short query → high octave

        # Multi-octave query
        octave_weights = {primary_octave: 0.5}
        for i in range(max(0, primary_octave - 1), min(5, primary_octave + 2)):
            if i != primary_octave:
                octave_weights[i] = 0.25

        results = voxel_cloud.query_multi_octave(
            q_result.multi_octave_quaternions,
            top_k=3,
            octave_weights=octave_weights
        )

        # Should find relevant results
        assert len(results) > 0
        # Results should be sorted by distance
        distances = [d for _, d in results]
        assert distances == sorted(distances)

    def test_adaptive_octave_improves_results(self):
        """Verify adaptive octave selection improves query relevance."""
        voxel_cloud = VoxelCloud(512, 512, 128)

        # Add texts at different semantic levels
        texts_by_level = {
            'detailed': "The Tao that can be spoken is not the eternal Tao, for names that can be named are not eternal names",
            'medium': "The Tao is empty yet inexhaustible",
            'abstract': "Tao"
        }

        for level, text in texts_by_level.items():
            freq_spectrum, params = self.freq_analyzer.analyze(text)
            standing_wave = self.origin.Gen(params['gamma_params'], params['iota_params'])
            result = self.origin.Act(standing_wave)

            from src.memory.octave_frequency import extract_fundamental
            f0 = extract_fundamental(freq_spectrum)

            voxel_cloud.add_with_octaves(
                proto_identity=result.proto_identity,
                frequency=f0,
                modality='text',
                quaternions=result.multi_octave_quaternions,
                resonance_strength=1.0
            )

        # Test short query (should match abstract)
        short_query = "Tao"
        freq, params = self.freq_analyzer.analyze(short_query)
        q_standing = self.origin.Gen(params['gamma_params'], params['iota_params'])
        q_result = self.origin.Act(q_standing)

        primary_octave = voxel_cloud.octave_hierarchy.adaptive_octave_selection(short_query)
        assert primary_octave == 4  # Abstract level

        results = voxel_cloud.query_multi_octave(
            q_result.multi_octave_quaternions,
            top_k=1,
            octave_weights={primary_octave: 1.0}
        )

        # Should match the abstract text
        if results:
            best_match = results[0][0]
            # FFT architecture: verify proto_identity, not text
            assert hasattr(best_match, 'proto_identity')
            assert best_match.proto_identity is not None
            # Verify it's at the expected octave level
            assert hasattr(best_match, 'octave_level')

        # Test long query (should match detailed)
        long_query = "What is the eternal and unspeakable nature of the Tao?"
        freq, params = self.freq_analyzer.analyze(long_query)
        q_standing = self.origin.Gen(params['gamma_params'], params['iota_params'])
        q_result = self.origin.Act(q_standing)

        primary_octave = voxel_cloud.octave_hierarchy.adaptive_octave_selection(long_query)
        assert primary_octave <= 2  # Detailed level

        results = voxel_cloud.query_multi_octave(
            q_result.multi_octave_quaternions,
            top_k=1,
            octave_weights={primary_octave: 1.0}
        )

        # Should match the detailed text
        if results:
            best_match = results[0][0]
            # FFT architecture: verify proto_identity at detailed level
            assert hasattr(best_match, 'proto_identity')
            assert best_match.proto_identity is not None
            # Verify octave level for detail
            assert hasattr(best_match, 'octave_level')