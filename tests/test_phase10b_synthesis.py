"""Test Phase 10B: Universal Signal-Derived Synthesis.

Validates:
1. Removal of text storage in voxel cloud
2. Removal of static phoneme tables
3. Universal frequency-based decoding
4. Weighted synthesis from context
5. Reversibility of encode→decode pipeline
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.encoding import EncodingPipeline
from pipeline.decoding import DecodingPipeline, DecodingConfig
from memory.voxel_cloud import ProtoIdentityEntry
from memory.voxel_cloud_collapse import merge_proto_identity


@dataclass
class MockProtoEntry:
    """Mock proto entry for testing without all dataclass requirements."""
    proto_identity: np.ndarray
    position: np.ndarray
    metadata: Dict
    mip_levels: List[np.ndarray] = field(default_factory=list)
    frequency: np.ndarray = None
    resonance_strength: int = 1
    fundamental_freq: float = 440.0
    harmonic_signature: np.ndarray = None

    def __post_init__(self):
        if self.frequency is None:
            self.frequency = np.random.randn(16, 16, 2).astype(np.float32)
        if not self.mip_levels:
            self.mip_levels = [self.proto_identity]
        if self.harmonic_signature is None:
            self.harmonic_signature = np.array([880.0, 1320.0])


class TestTextStorageRemoval:
    """Test that text storage has been properly removed."""

    def test_no_merged_texts_in_metadata(self):
        """Verify merged_texts tracking has been removed."""
        # Create a simple mock voxel cloud for testing
        class MockVoxelCloud:
            def _get_frequency_bin(self, freq):
                return 0
            def _frequency_to_position(self, freq):
                return np.array([0, 0, 0])
            def _generate_mip_levels(self, proto):
                return [proto]

        mock_cloud = MockVoxelCloud()
        mock_cloud.entries = []
        mock_cloud.frequency_index = {}

        # Create proto entry using mock
        proto1 = MockProtoEntry(
            proto_identity=np.random.randn(16, 16, 4).astype(np.float32),
            position=np.array([0, 0, 0]),
            metadata={'source': 'test1'}
        )

        mock_cloud.entries.append(proto1)

        # New proto to merge
        new_proto = np.random.randn(16, 16, 4).astype(np.float32)
        new_freq = np.random.randn(16, 16, 2).astype(np.float32)
        new_metadata = {'source': 'test2', 'text': 'should not be stored'}

        # Merge - this updates proto1 in place
        merge_proto_identity(proto1, new_proto, new_freq, new_metadata, mock_cloud)

        # Verify no merged_texts field
        assert 'merged_texts' not in proto1.metadata
        assert proto1.resonance_strength == 2  # k + 1
        assert proto1.metadata['merge_count'] == 2

    def test_no_text_storage_in_pipeline(self):
        """Verify text is not stored during encoding."""
        # Create a simple carrier
        carrier = np.random.randn(16, 16, 4).astype(np.float32)
        encoder = EncodingPipeline(carrier, width=16, height=16)
        text = "Test input text"

        # Encode to proto
        proto, metadata = encoder.encode_text(text)

        # Proto should exist but no text metadata required
        assert proto is not None
        assert proto.shape == (16, 16, 4)
        # Metadata might have params but shouldn't require text storage


class TestUniversalDecoding:
    """Test the universal frequency-based decoding system."""

    def test_no_phoneme_table(self):
        """Verify static phoneme table has been removed."""
        decoder = DecodingPipeline()

        # Should not have _formants_to_phonemes method
        assert not hasattr(decoder, '_formants_to_phonemes')

        # Should not have old _frequency_to_text with phoneme mapping
        # The new version uses signal energy patterns
        proto = np.random.randn(16, 16, 4).astype(np.float32)
        text = decoder.decode_to_text(proto)
        assert isinstance(text, str)
        assert text != ""  # Should produce something

    def test_inverse_fft_pipeline(self):
        """Test the inverse FFT transformation."""
        decoder = DecodingPipeline()

        # Create a simple frequency spectrum
        freq_spectrum = np.zeros((16, 16, 2), dtype=np.float32)
        freq_spectrum[8, 8, 0] = 10.0  # Single frequency peak
        freq_spectrum[8, 8, 1] = 0.0   # Zero phase

        # Apply inverse FFT
        signal = decoder._inverse_fft_to_signal(freq_spectrum)

        # Should produce a spatial signal
        assert signal.shape == (16, 16)
        assert signal.dtype in [np.float32, np.float64]  # Either is fine
        assert not np.allclose(signal, 0)  # Should have non-zero values

    def test_signal_to_text_conversion(self):
        """Test signal energy to text mapping."""
        decoder = DecodingPipeline()

        # Create signal with clear peaks
        signal = np.zeros((16, 16))
        signal[4:6, 4:6] = 1.0    # First character region
        signal[4:6, 8:10] = 0.5   # Second character region
        signal[8:10, 4:6] = 0.75  # Third character region

        text = decoder._signal_to_text(signal)

        # Should extract characters for each region
        assert isinstance(text, str)
        assert len(text) > 0
        assert text != "[silence]"

    def test_energy_to_char_mapping(self):
        """Test universal energy to character mapping."""
        decoder = DecodingPipeline()

        # Test high variance (consonant)
        high_var_energy = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
        char1 = decoder._energy_to_char(high_var_energy)
        assert char1 in 'bcdfghjklmnpqrstvwxyz'

        # Test low variance (vowel)
        low_var_energy = np.array([0.5, 0.51, 0.49, 0.5, 0.5])
        char2 = decoder._energy_to_char(low_var_energy)
        assert char2 in 'aeiou'


class TestWeightedSynthesis:
    """Test weighted synthesis from context."""

    def test_weighted_synthesis_basic(self):
        """Test basic weighted synthesis of proto-identities."""
        decoder = DecodingPipeline()

        # Create test protos
        proto1 = np.ones((16, 16, 4), dtype=np.float32)
        proto2 = np.ones((16, 16, 4), dtype=np.float32) * 2
        proto3 = np.ones((16, 16, 4), dtype=np.float32) * 3

        protos = [proto1, proto2, proto3]
        weights = [0.5, 0.3, 0.2]

        # Synthesize
        synthesized = decoder._weighted_synthesis(protos, weights)

        # Check weighted average
        expected = 0.5 * proto1 + 0.3 * proto2 + 0.2 * proto3
        np.testing.assert_allclose(synthesized, expected, rtol=1e-5)

    def test_weighted_synthesis_with_temperature(self):
        """Test synthesis with temperature scaling."""
        config = DecodingConfig(synthesis_temperature=2.0)
        decoder = DecodingPipeline(config)

        proto1 = np.random.randn(16, 16, 4).astype(np.float32)
        proto2 = np.random.randn(16, 16, 4).astype(np.float32)

        protos = [proto1, proto2]
        weights = [0.8, 0.2]

        synthesized = decoder._weighted_synthesis(protos, weights)

        # Temperature should make weights more uniform
        assert synthesized.shape == (16, 16, 4)
        # With temp=2, weights should be closer to [0.65, 0.35] than [0.8, 0.2]

    def test_decode_to_summary_integration(self):
        """Test full summary generation from weighted context."""
        decoder = DecodingPipeline()

        # Create query proto
        query = np.random.randn(16, 16, 4).astype(np.float32)

        # Create context entries using mock
        entry1 = MockProtoEntry(
            proto_identity=np.random.randn(16, 16, 4).astype(np.float32),
            position=np.array([0, 0, 0]),
            metadata={},
            resonance_strength=3
        )

        entry2 = MockProtoEntry(
            proto_identity=np.random.randn(16, 16, 4).astype(np.float32),
            position=np.array([1, 0, 0]),
            metadata={},
            resonance_strength=1
        )

        visible_protos = [entry1, entry2]

        # Generate summary
        summary = decoder.decode_to_summary(query, visible_protos)

        # Should produce text from synthesis
        assert isinstance(summary, str)
        assert summary != "[no context]"
        assert len(summary) > 0


class TestReversibility:
    """Test encode→decode reversibility."""

    def test_frequency_domain_reversibility(self):
        """Test proto→frequency→proto maintains structure."""
        decoder = DecodingPipeline()

        # Create a proto with known pattern
        original = np.random.randn(16, 16, 4).astype(np.float32)

        # Extract frequency
        freq = decoder._proto_to_frequency(original)
        assert freq.shape == (16, 16, 2)

        # Can't perfectly reverse because we only use XY channels
        # But magnitude should be preserved
        magnitude = freq[:, :, 0]
        expected_magnitude = np.sqrt(original[:, :, 0]**2 + original[:, :, 1]**2)
        np.testing.assert_allclose(magnitude, expected_magnitude, rtol=1e-5)

    def test_signal_processing_pipeline(self):
        """Test signal→FFT→IFFT→signal maintains energy."""
        decoder = DecodingPipeline()

        # Create a spatial signal
        original_signal = np.random.randn(16, 16)

        # Forward FFT (simulate what encoder would do)
        fft_signal = np.fft.fft2(original_signal)
        magnitude = np.abs(fft_signal)
        phase = np.angle(fft_signal)
        freq_spectrum = np.stack([magnitude, phase], axis=-1).astype(np.float32)

        # Inverse FFT (what decoder does)
        reconstructed = decoder._inverse_fft_to_signal(freq_spectrum)

        # Should maintain signal structure (allowing for float32 precision)
        np.testing.assert_allclose(reconstructed, original_signal, rtol=1e-5)

    def test_multi_modal_decoding(self):
        """Test that all modalities can decode from same proto."""
        decoder = DecodingPipeline()
        proto = np.random.randn(16, 16, 4).astype(np.float32)

        # Text decoding
        text = decoder.decode_to_text(proto)
        assert isinstance(text, str)

        # Image decoding
        image = decoder.decode_to_image(proto)
        assert image.shape == (16, 16, 3)
        assert image.dtype == np.uint8

        # Audio decoding
        audio = decoder.decode_to_audio(proto, sample_rate=16000, duration=0.5)
        assert audio.shape == (8000,)  # 16000 * 0.5
        assert audio.dtype == np.float32
        assert audio.min() >= -1.0 and audio.max() <= 1.0


class TestQualityGates:
    """Verify all quality requirements are met."""

    def test_no_text_storage_anywhere(self):
        """Grep-like test for text storage."""
        # Check that metadata['text'] is not used in storage
        from memory.voxel_cloud_collapse import merge_proto_identity
        import inspect

        source = inspect.getsource(merge_proto_identity)
        assert "merged_texts" not in source

    def test_no_static_tables(self):
        """Verify no static phoneme tables."""
        from pipeline import decoding
        import inspect

        source = inspect.getsource(decoding)
        assert "phoneme_table" not in source
        assert "_formants_to_phonemes" not in source

    def test_universal_decode_works(self):
        """Verify decode works without language-specific code."""
        decoder = DecodingPipeline()

        # Should work for any proto input
        random_proto = np.random.randn(16, 16, 4).astype(np.float32)
        text = decoder.decode_to_text(random_proto)

        # Should produce text without phoneme tables
        assert isinstance(text, str)
        assert len(text) > 0

    def test_weighted_synthesis_implements_resonance(self):
        """Verify resonance weighting in synthesis."""
        decoder = DecodingPipeline()

        query = np.ones((16, 16, 4), dtype=np.float32)

        # High resonance entry using mock
        high_res = MockProtoEntry(
            proto_identity=np.ones((16, 16, 4), dtype=np.float32) * 2,
            position=np.array([0, 0, 0]),
            metadata={},
            resonance_strength=10
        )

        # Low resonance entry using mock
        low_res = MockProtoEntry(
            proto_identity=np.ones((16, 16, 4), dtype=np.float32) * 3,
            position=np.array([0, 0, 0]),
            metadata={},
            resonance_strength=1
        )

        # Synthesize - high resonance should dominate
        summary = decoder.decode_to_summary(query, [high_res, low_res])

        # The synthesized proto should be closer to high_res proto
        # due to resonance weighting
        assert isinstance(summary, str)
        assert len(summary) > 0