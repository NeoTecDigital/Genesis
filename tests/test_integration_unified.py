"""Integration tests for the unified memory system.

Tests the integration of:
- MemoryRouter with MemoryHierarchy
- UnifiedEncoder/Decoder with production code
- Octave-aware feedback loop
- CLI commands with unified mode
"""

import numpy as np
import tempfile
import pickle
from pathlib import Path

from src.origin import Origin
from src.memory.memory_hierarchy import MemoryHierarchy
from src.memory.memory_router import MemoryRouter
from src.pipeline.unified_encoder import UnifiedEncoder
from src.pipeline.unified_decoder import UnifiedDecoder
from src.cli.commands_core import _cmd_discover_unified


def test_memory_hierarchy_with_routing():
    """Test MemoryHierarchy with routing enabled."""
    # Create hierarchy with routing
    hierarchy = MemoryHierarchy(use_routing=True)
    assert hierarchy.memory_router is not None

    # Create some test data
    protos = [np.random.randn(512, 512, 4).astype(np.float32) for _ in range(3)]
    freqs = [np.random.randn(512, 512, 2).astype(np.float32) for _ in range(3)]
    octave_units = [(4, 'a'), (0, 'test'), (-2, 'hello world')]

    # Add with routing
    counts = hierarchy.add_to_memory(
        protos, freqs, octave_units,
        context_type='foundation',
        base_metadata={'source': 'test'}
    )

    # Check that routing happened
    assert counts['core'] > 0 or counts['experiential'] > 0

    # Verify routing decisions were recorded
    assert len(hierarchy.memory_router.routing_history) == 3

    print("✓ MemoryHierarchy with routing works")


def test_unified_encoder_with_memory_hierarchy():
    """Test UnifiedEncoder integration with MemoryHierarchy."""
    # Initialize components
    origin = Origin(512, 512, use_gpu=False)
    hierarchy = MemoryHierarchy(use_routing=True)
    carrier = hierarchy.create_carrier(origin)

    encoder = UnifiedEncoder(
        memory_hierarchy=hierarchy,
        carrier=carrier
    )

    # Test text
    test_text = "The quick brown fox jumps over the lazy dog."

    # Encode with foundation context
    result = encoder.encode(
        text=test_text,
        destination='core',  # Foundation texts go to core
        octaves=[4, 0, -2],  # Characters, words, short phrases
        metadata={'source': 'test'}
    )

    # Verify results
    assert len(result.octave_units) > 0
    assert len(result.routing_decisions) > 0
    assert result.core_added >= 0
    assert result.experiential_added >= 0

    # Check octave distribution
    octaves = [u.octave for u in result.octave_units]
    assert 4 in octaves  # Characters
    assert 0 in octaves  # Words
    assert -2 in octaves  # Short phrases

    print("✓ UnifiedEncoder integration works")


def test_unified_decoder_with_memory_hierarchy():
    """Test UnifiedDecoder integration with MemoryHierarchy."""
    # Setup
    origin = Origin(512, 512, use_gpu=False)
    hierarchy = MemoryHierarchy(use_routing=True)
    carrier = hierarchy.create_carrier(origin)

    encoder = UnifiedEncoder(hierarchy, carrier)
    decoder = UnifiedDecoder(hierarchy)

    # Encode some text
    original_text = "Hello world!"
    encode_result = encoder.encode(original_text, destination='both')

    # Try to decode using one of the encoded proto-identities as query
    if encode_result.octave_units:
        query_proto = encode_result.octave_units[0].proto_identity
        decode_result = decoder.decode(
            query_proto=query_proto,
            layers='both',
            octaves=[4, 0],
            expand_octaves=True
        )

        # Verify decode results
        assert decode_result.text is not None
        assert decode_result.confidence >= 0
        assert len(decode_result.source_layers) > 0
        assert len(decode_result.octaves_used) > 0

    print("✓ UnifiedDecoder integration works")


def test_octave_aware_feedback():
    """Test octave-aware feedback loop."""
    origin = Origin(512, 512, use_gpu=False)
    hierarchy = MemoryHierarchy(use_routing=True)
    carrier = hierarchy.create_carrier(origin)

    # Add some core knowledge
    encoder = UnifiedEncoder(hierarchy, carrier)
    encoder.encode("The sun is bright.", destination='core')

    # Add experiential knowledge
    encoder.encode("The sun shines.", destination='experiential')

    # Get feedback loop
    feedback = hierarchy.feedback_loop
    assert feedback is not None

    # Compute per-octave coherence
    octave_coherences = feedback.compute_octave_coherence(
        hierarchy.experiential_memory,
        octave_range=(-4, 4)
    )

    # Should have coherence values for octaves that have data
    assert len(octave_coherences) > 0

    # Character level should have high coherence (shared letters)
    if 4 in octave_coherences:
        assert octave_coherences[4] > 0.5  # Characters should match well

    print("✓ Octave-aware feedback works")


def test_cli_unified_discover():
    """Test CLI discover command with unified mode."""
    # Create test input file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test text for unified discovery.\nAnother line of text.")
        input_path = f.name

    # Create test output path
    output_path = tempfile.mktemp(suffix='.pkl')

    # Mock args object
    class Args:
        input = input_path
        output = output_path
        modality = 'text'
        unified = True
        context_type = 'foundation'
        collapse_harmonic_tolerance = 0.05
        collapse_cosine_threshold = 0.85
        collapse_octave_tolerance = 0
        enable_collapse = True

    args = Args()
    origin = Origin(512, 512, use_gpu=False)

    # Run unified discover
    result = _cmd_discover_unified(args, origin)

    # Should succeed
    assert result == 0

    # Check output files were created
    assert Path(output_path).exists()
    assert Path(output_path.replace('.pkl', '_meta.pkl')).exists()

    # Load and verify saved data
    with open(output_path, 'rb') as f:
        saved_data = pickle.load(f)

    assert 'core_memory' in saved_data
    assert 'experiential_memory' in saved_data
    assert 'carrier' in saved_data
    assert 'encode_result' in saved_data

    # Cleanup
    Path(input_path).unlink()
    Path(output_path).unlink()
    Path(output_path.replace('.pkl', '_meta.pkl')).unlink()

    print("✓ CLI unified discover works")


def test_backward_compatibility():
    """Test that old APIs still work."""
    hierarchy = MemoryHierarchy(use_routing=False)  # Disable routing

    # Old methods should still work
    proto = np.random.randn(512, 512, 4).astype(np.float32)
    freq = np.random.randn(512, 512, 2).astype(np.float32)
    metadata = {'test': True}

    # Store using old methods
    hierarchy.store_core(proto, freq, metadata)
    hierarchy.store_experiential(proto, freq, metadata)

    # Query using old methods
    core_results = hierarchy.query_core(proto, max_results=5)
    exp_results = hierarchy.query_experiential(proto, max_results=5)

    # Should have results
    assert len(core_results) == 1
    assert len(exp_results) == 1

    print("✓ Backward compatibility maintained")


def test_migration_guide():
    """Test migration from old to new API."""
    origin = Origin(512, 512, use_gpu=False)

    # Old way (still works)
    old_hierarchy = MemoryHierarchy(use_routing=False)
    proto = origin.Res({'extraction_rate': 0.3, 'focus_sigma': 2.0,
                       'base_frequency': 440}, {'decay_rate': 0.9})
    freq = np.random.randn(512, 512, 2).astype(np.float32)
    old_hierarchy.store_core(proto, freq, {'text': 'test'})

    # New way (recommended)
    new_hierarchy = MemoryHierarchy(use_routing=True)
    carrier = new_hierarchy.create_carrier(origin)
    encoder = UnifiedEncoder(new_hierarchy, carrier)
    encoder.encode("test", destination='core')

    # Both should have data
    assert len(old_hierarchy.core_memory) > 0
    assert len(new_hierarchy.core_memory) > 0 or len(new_hierarchy.experiential_memory) > 0

    print("✓ Migration guide validated")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Unified Memory System Integration")
    print("=" * 60)

    test_memory_hierarchy_with_routing()
    test_unified_encoder_with_memory_hierarchy()
    test_unified_decoder_with_memory_hierarchy()
    test_octave_aware_feedback()
    test_cli_unified_discover()
    test_backward_compatibility()
    test_migration_guide()

    print("\n✅ All integration tests passed!")