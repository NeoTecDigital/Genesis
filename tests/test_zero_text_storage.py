"""Validate zero raw text storage in proto-identities and metadata."""
import pytest
import numpy as np
from src.pipeline.unified_encoder import UnifiedEncoder
from src.memory.memory_hierarchy import MemoryHierarchy
from src.pipeline.fft_text_decoder import FFTTextDecoder


def test_no_text_in_octave_units():
    """Verify OctaveUnit has no text field."""
    encoder = UnifiedEncoder(MemoryHierarchy())
    result = encoder.encode("Test text for validation", destination='core')

    for unit in result.octave_units:
        assert not hasattr(unit, 'text'), "OctaveUnit should not have text attribute"
        assert not hasattr(unit, 'unit'), "OctaveUnit should not have unit attribute"
        assert unit.proto_identity is not None
        assert unit.proto_identity.shape == (512, 512, 4)
        assert unit.frequency is not None
        assert unit.frequency.shape == (512, 512, 2)


def test_no_text_in_metadata():
    """Verify metadata contains no raw text."""
    encoder = UnifiedEncoder(MemoryHierarchy())
    result = encoder.encode("The quick brown fox jumps over the lazy dog", destination='core')

    memory = encoder.memory
    forbidden = {'unit', 'text', 'content', 'raw', 'original'}

    for entry in memory.core_memory.entries:
        violations = forbidden.intersection(entry.metadata.keys())
        assert not violations, f"Metadata contains forbidden fields: {violations}"

        # Check metadata values for text leakage
        for key, value in entry.metadata.items():
            if isinstance(value, str):
                # Only allow specific metadata fields with string values
                allowed_string_fields = {'memory_type', 'source', 'destination'}
                assert key in allowed_string_fields, f"Unexpected string field in metadata: {key}={value}"


def test_fft_roundtrip_via_pipeline():
    """Verify text can be recovered via FFT decoder."""
    encoder = UnifiedEncoder(MemoryHierarchy())
    decoder = FFTTextDecoder()

    original = "The Art of War by Sun Tzu"
    result = encoder.encode(original, destination='core')

    # Verify we have units
    assert len(result.octave_units) > 0

    # Decode first unit
    decoded = decoder.decode_text(result.octave_units[0].proto_identity)

    # Partial match ok for chunked text
    assert decoded in original or original.startswith(decoded) or decoded.startswith(original[:10]), \
        f"Decoded text '{decoded}' not found in original '{original}'"


def test_no_text_in_voxel_cloud():
    """Verify VoxelCloud storage contains no raw text."""
    from src.memory.voxel_cloud import VoxelCloud

    cloud = VoxelCloud()

    # Create a proto-identity (no text)
    proto = np.random.randn(512, 512, 4).astype(np.float32)

    # Store in cloud - should only accept numerical data
    cloud.add_voxel(proto, metadata={'octave': 0, 'hash': 'abc123'})

    # Verify storage
    assert len(cloud.voxels) == 1
    stored = cloud.voxels[0]

    # Check stored data is numerical only
    assert isinstance(stored['voxel'], np.ndarray)
    assert stored['voxel'].dtype in [np.float32, np.float64]

    # Check metadata has no text fields
    forbidden = {'text', 'unit', 'content', 'raw', 'original'}
    violations = forbidden.intersection(stored['metadata'].keys())
    assert not violations, f"VoxelCloud metadata contains forbidden fields: {violations}"


def test_memory_hierarchy_no_text():
    """Verify memory hierarchy stores no raw text."""
    memory = MemoryHierarchy()
    encoder = UnifiedEncoder(memory)

    # Encode some text
    text = "To be or not to be, that is the question"
    result = encoder.encode(text, destination='core')

    # Check core memory
    for entry in memory.core_memory.entries:
        assert isinstance(entry.proto_identity, np.ndarray)
        assert entry.proto_identity.dtype in [np.float32, np.float64]

        # Verify no text in entry attributes
        assert not hasattr(entry, 'text')
        assert not hasattr(entry, 'content')
        assert not hasattr(entry, 'raw')

        # Verify metadata
        forbidden = {'text', 'unit', 'content', 'raw', 'original'}
        violations = forbidden.intersection(entry.metadata.keys())
        assert not violations

    # Check experiential memory
    for entry in memory.experiential_memory.entries:
        assert isinstance(entry.proto_identity, np.ndarray)
        assert not hasattr(entry, 'text')

        forbidden = {'text', 'unit', 'content', 'raw', 'original'}
        violations = forbidden.intersection(entry.metadata.keys())
        assert not violations


def test_proto_identity_is_pure_numerical():
    """Verify proto-identities are pure numerical arrays."""
    encoder = UnifiedEncoder(MemoryHierarchy())

    texts = [
        "Hello world",
        "Complex unicode: αβγδ 中文 emoji 🎉",
        "Numbers 123 and symbols !@#$%"
    ]

    for text in texts:
        result = encoder.encode(text, destination='core')

        for unit in result.octave_units:
            # Proto-identity must be numerical
            assert isinstance(unit.proto_identity, np.ndarray)
            assert unit.proto_identity.dtype in [np.float32, np.float64, np.complex64, np.complex128]

            # Frequency must be numerical
            assert isinstance(unit.frequency, np.ndarray)
            assert unit.frequency.dtype in [np.float32, np.float64]

            # Check for NaN/Inf
            assert np.isfinite(unit.proto_identity).all(), "Proto-identity contains NaN or Inf"
            assert np.isfinite(unit.frequency).all(), "Frequency contains NaN or Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])