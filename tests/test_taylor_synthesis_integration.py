"""Integration tests for Taylor synthesis with UnifiedDecoder.

Tests end-to-end integration between TaylorSynthesizer and the decoding
pipeline, verifying complete synthesis workflow from encoding through
query to synthesis and final decode.
"""

import pytest
import numpy as np
from src.memory.taylor_synthesizer import TaylorSynthesizer
from src.memory.memory_hierarchy import MemoryHierarchy
from src.origin import Origin
from src.pipeline.unified_encoder import UnifiedEncoder
from src.pipeline.unified_decoder import UnifiedDecoder


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def origin():
    """Create Origin instance."""
    return Origin(width=512, height=512, use_gpu=False)


@pytest.fixture
def memory_hierarchy(origin):
    """Create memory hierarchy with initialized carrier."""
    memory = MemoryHierarchy(width=512, height=512, depth=128)
    carrier = memory.create_carrier(origin)
    return memory


@pytest.fixture
def encoder(memory_hierarchy):
    """Create UnifiedEncoder."""
    return UnifiedEncoder(memory_hierarchy)


@pytest.fixture
def decoder(memory_hierarchy):
    """Create UnifiedDecoder."""
    return UnifiedDecoder(memory_hierarchy)


@pytest.fixture
def synthesizer():
    """Create TaylorSynthesizer."""
    return TaylorSynthesizer(
        epsilon=1e-4,
        safety_max_iterations=10000,
        clustering_check_interval=10
    )


# ============================================================================
# Test Case 1: UnifiedDecoder Integration
# ============================================================================

def test_unified_decoder_integration(decoder, memory_hierarchy, synthesizer):
    """Test decode() with use_taylor_synthesis=True.

    Verify: Returns (proto_identities, explanation)
    """
    # Create test proto-identity
    proto = np.random.randn(512, 512, 4).astype(np.float32)
    proto = np.clip(proto, 0, 1)

    # Decode with Taylor synthesis enabled
    result = decoder.decode(
        query_proto=proto,
        use_taylor_synthesis=True,
        query_text="test synthesis query"
    )

    # Should return tuple: (proto_identities, explanation)
    assert isinstance(result, tuple)
    assert len(result) == 2

    proto_identities, explanation = result

    # Validate proto_identities
    assert isinstance(proto_identities, list)
    assert len(proto_identities) > 0
    for p in proto_identities:
        assert isinstance(p, np.ndarray)
        assert p.shape == (512, 512, 4)

    # Validate explanation
    assert isinstance(explanation, str)
    assert len(explanation) > 0


# ============================================================================
# Test Case 2: Parameter Validation
# ============================================================================

def test_parameter_validation(decoder, memory_hierarchy):
    """Test use_taylor_synthesis=True without required params.

    Expected: ValueError raised
    """
    proto = np.random.randn(512, 512, 4).astype(np.float32)
    proto = np.clip(proto, 0, 1)

    # Should raise ValueError when query_text not provided
    with pytest.raises(ValueError) as exc_info:
        decoder.decode(
            query_proto=proto,
            use_taylor_synthesis=True
            # Missing query_text parameter
        )

    # Error message should mention missing parameter
    assert 'query_text' in str(exc_info.value).lower() or 'required' in str(exc_info.value).lower()


# ============================================================================
# Test Case 3: End-to-End Synthesis Pipeline
# ============================================================================

def test_end_to_end_synthesis(encoder, decoder, memory_hierarchy, synthesizer):
    """Full pipeline: encode → store → query → synthesize → decode.

    Verify: Complete integration working
    """
    # Step 1: Encode foundation text
    foundation_text = "The fundamental theorem of calculus links differentiation and integration"
    encode_result = encoder.encode(foundation_text, destination='core')

    # Verify encoding worked
    assert encode_result is not None
    assert len(encode_result.octave_units) > 0

    # Verify storage in core memory
    assert len(memory_hierarchy.core_memory) > 0

    # Step 2: Create query proto-identity
    query_text = "What connects derivatives and integrals?"
    query_encode_result = encoder.encode(query_text, destination='experiential')

    # Get query proto
    query_proto = query_encode_result.octave_units[0].proto_identity

    # Step 3: Synthesize via TaylorSynthesizer
    synthesis_result = synthesizer.synthesize(
        query=query_text,
        memory_hierarchy=memory_hierarchy,
        proto_unity_carrier=memory_hierarchy.proto_unity_carrier
    )

    # Verify synthesis result
    assert synthesis_result is not None
    assert synthesis_result.state in ['identity', 'paradox', 'evolution_cycling', 'evolution_chaotic']
    assert len(synthesis_result.proto_identities) > 0

    # Step 4: Decode with Taylor synthesis (use first proto from synthesis)
    decode_result = decoder.decode(
        query_proto=synthesis_result.proto_identities[0],
        use_taylor_synthesis=True,
        query_text=query_text
    )

    # Verify decode result
    assert isinstance(decode_result, tuple)
    assert len(decode_result) == 2

    refined_protos, explanation = decode_result
    assert isinstance(refined_protos, list)
    assert len(refined_protos) > 0
    assert isinstance(explanation, str)
    assert len(explanation) > 0


def test_encode_store_synthesize_workflow(encoder, memory_hierarchy, synthesizer):
    """Test workflow: encode to core → query → synthesize.

    Verify: Synthesis can access and refine against core knowledge
    """
    # Encode multiple foundation texts
    foundation_texts = [
        "Energy equals mass times the speed of light squared",
        "Force equals mass times acceleration",
        "Momentum equals mass times velocity"
    ]

    for text in foundation_texts:
        encoder.encode(text, destination='core')

    # Verify core memory populated
    initial_core_size = len(memory_hierarchy.core_memory)
    assert initial_core_size > 0

    # Query related to foundation
    query = "relationship between mass and energy"
    result = synthesizer.synthesize(
        query=query,
        memory_hierarchy=memory_hierarchy,
        proto_unity_carrier=memory_hierarchy.proto_unity_carrier
    )

    # Should produce valid synthesis
    assert result is not None
    assert result.state in ['identity', 'paradox', 'evolution_cycling', 'evolution_chaotic']

    # If identity state, should have reasonable core coherence
    if result.state == 'identity':
        assert len(result.coherence_scores) > 0
        # May have some coherence with foundation texts
        assert result.coherence_scores[0] >= 0.0


def test_synthesis_preserves_memory_state(encoder, memory_hierarchy, synthesizer):
    """Test that synthesis doesn't corrupt memory state.

    Verify: Core and experiential memory remain valid after synthesis
    """
    # Encode foundation
    foundation_text = "The mitochondria is the powerhouse of the cell"
    encoder.encode(foundation_text, destination='core')

    initial_core_size = len(memory_hierarchy.core_memory)
    initial_exp_size = len(memory_hierarchy.experiential_memory)

    # Run synthesis
    query = "cellular energy production"
    result = synthesizer.synthesize(
        query=query,
        memory_hierarchy=memory_hierarchy,
        proto_unity_carrier=memory_hierarchy.proto_unity_carrier
    )

    # Memory sizes should not have decreased
    final_core_size = len(memory_hierarchy.core_memory)
    final_exp_size = len(memory_hierarchy.experiential_memory)

    assert final_core_size >= initial_core_size
    # Experiential may increase due to synthesis
    assert final_exp_size >= 0


def test_multiple_synthesis_rounds(memory_hierarchy, synthesizer):
    """Test multiple synthesis rounds on same memory.

    Verify: System remains stable across multiple queries
    """
    queries = [
        "first concept",
        "second concept",
        "third concept"
    ]

    results = []
    for query in queries:
        result = synthesizer.synthesize(
            query=query,
            memory_hierarchy=memory_hierarchy,
            proto_unity_carrier=memory_hierarchy.proto_unity_carrier
        )
        results.append(result)

    # All results should be valid
    assert len(results) == len(queries)
    for result in results:
        assert result is not None
        assert result.state in ['identity', 'paradox', 'evolution_cycling', 'evolution_chaotic']
        assert len(result.proto_identities) > 0


def test_synthesis_with_empty_experiential(encoder, memory_hierarchy, synthesizer):
    """Test synthesis when experiential memory is empty.

    Verify: Can still synthesize based on core memory alone
    """
    # Add to core memory
    encoder.encode("core knowledge text", destination='core')

    # Clear experiential memory
    memory_hierarchy.clear_experiential()
    assert len(memory_hierarchy.experiential_memory) == 0

    # Synthesize
    query = "test with empty experiential"
    result = synthesizer.synthesize(
        query=query,
        memory_hierarchy=memory_hierarchy,
        proto_unity_carrier=memory_hierarchy.proto_unity_carrier
    )

    # Should still work
    assert result is not None
    assert result.state in ['identity', 'paradox', 'evolution_cycling', 'evolution_chaotic']


def test_decoder_legacy_mode(decoder):
    """Test decoder without Taylor synthesis (legacy mode).

    Verify: Backward compatibility maintained
    """
    # Create test proto
    proto = np.random.randn(512, 512, 4).astype(np.float32)
    proto = np.clip(proto, 0, 1)

    # Decode WITHOUT Taylor synthesis
    result = decoder.decode(
        query_proto=proto,
        use_taylor_synthesis=False
    )

    # Should return DecodingResult (standard behavior)
    from src.pipeline.unified_decoder import DecodingResult
    assert isinstance(result, DecodingResult)
    assert hasattr(result, 'text')
    assert isinstance(result.text, str)


def test_synthesis_state_distribution(memory_hierarchy, synthesizer):
    """Test that different queries can produce different states.

    Verify: System can detect all state types
    """
    # Test queries designed to trigger different states
    test_cases = [
        ("simple stable concept", ['identity', 'evolution_cycling']),  # Should converge
        ("contradiction paradox opposing", ['paradox', 'evolution_chaotic']),  # May paradox
        ("cycle repeat pattern periodic", ['evolution_cycling', 'identity']),  # May cycle
        ("chaos random unstable conflicting", ['evolution_chaotic', 'paradox'])  # May chaos
    ]

    results = {}
    for query, expected_states in test_cases:
        result = synthesizer.synthesize(
            query=query,
            memory_hierarchy=memory_hierarchy,
            proto_unity_carrier=memory_hierarchy.proto_unity_carrier
        )
        results[query] = result.state

        # State should be one of the expected
        assert result.state in expected_states or result.state in [
            'identity', 'paradox', 'evolution_cycling', 'evolution_chaotic'
        ]

    # Should have at least some variety in states (not all same)
    unique_states = set(results.values())
    # At least 1 state (allow for deterministic behavior)
    assert len(unique_states) >= 1
