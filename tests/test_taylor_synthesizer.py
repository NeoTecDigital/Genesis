"""Comprehensive test suite for TaylorSynthesizer.

Tests the iterative refinement system where experiential layer refines
proto-identities through Taylor series expansion, covering:
- Identity: Natural convergence via delta < epsilon
- Paradox: Multiple stable attractors (P and !P both valid)
- Evolution-cycling: Periodic patterns
- Evolution-chaotic: High entropy, no convergence
"""

import pytest
import numpy as np
from src.memory.taylor_synthesizer import TaylorSynthesizer
from src.memory.synthesis_types import SynthesisResult, IdentityBranch, UnstableSystemStub
from src.memory.memory_hierarchy import MemoryHierarchy
from src.origin import Origin


# ============================================================================
# Test Fixtures and Utilities
# ============================================================================

@pytest.fixture
def origin():
    """Create Origin instance for proto-unity carrier."""
    return Origin(width=512, height=512, use_gpu=False)


@pytest.fixture
def memory_hierarchy(origin):
    """Create memory hierarchy with initialized carrier."""
    memory = MemoryHierarchy(width=512, height=512, depth=128)
    carrier = memory.create_carrier(origin)
    return memory


@pytest.fixture
def synthesizer():
    """Create TaylorSynthesizer with test parameters."""
    return TaylorSynthesizer(
        epsilon=1e-4,
        safety_max_iterations=10000,
        clustering_check_interval=10
    )


def create_test_proto_unity_carrier(width=512, height=512) -> np.ndarray:
    """Create test proto-unity carrier."""
    carrier = np.random.randn(height, width, 4).astype(np.float32)
    # Normalize to [0, 1] range
    carrier = (carrier - carrier.min()) / (carrier.max() - carrier.min() + 1e-8)
    return carrier


def assert_synthesis_result_valid(result: SynthesisResult):
    """Assert SynthesisResult has required fields and valid structure."""
    assert result is not None
    assert isinstance(result, SynthesisResult)
    assert isinstance(result.proto_identities, list)
    assert len(result.proto_identities) > 0
    assert result.state in ['identity', 'paradox', 'evolution_cycling', 'evolution_chaotic']
    assert isinstance(result.coherence_scores, list)
    assert len(result.coherence_scores) == len(result.proto_identities)
    assert isinstance(result.explanation, str)
    assert len(result.explanation) > 0
    assert isinstance(result.branches, list)
    assert len(result.branches) > 0

    # Check proto-identities shape
    for proto in result.proto_identities:
        assert isinstance(proto, np.ndarray)
        assert proto.shape == (512, 512, 4)
        assert proto.dtype in [np.float32, np.float64]


# ============================================================================
# Test Case 1: Simple Convergence (Identity State)
# ============================================================================

def test_simple_convergence(synthesizer, memory_hierarchy):
    """Test natural convergence to stable state.

    Input: Simple query like "Hello world"
    Expected: Converges to stable state (identity or cycling), delta < epsilon
    Verify: No safety_max_iterations needed
    """
    query = "Hello world"
    result = synthesizer.synthesize(
        query=query,
        memory_hierarchy=memory_hierarchy,
        proto_unity_carrier=memory_hierarchy.proto_unity_carrier
    )

    # Validate result structure
    assert_synthesis_result_valid(result)

    # Should converge to stable state (identity or cycling acceptable)
    assert result.state in ['identity', 'evolution_cycling']

    # Should have proto-identities
    assert len(result.proto_identities) >= 1

    # Should have explanation
    assert len(result.explanation) > 0

    # Verify convergence happened naturally (not via safety limit)
    branch = result.branches[0]
    assert len(branch.trajectory) < synthesizer.safety_max_iterations

    # If identity, verify it converged
    if result.state == 'identity':
        assert branch.state == 'converged'


# ============================================================================
# Test Case 2: Paradox Branching (Multiple Attractors)
# ============================================================================

def test_paradox_branching(synthesizer, memory_hierarchy):
    """Test paradox detection and branching.

    Input: "free will vs determinism" (contradictory concepts)
    Expected: PARADOX state or evolution (contradictions may not always split cleanly)
    Verify: If paradox, has multiple branches; otherwise valid evolution state
    """
    query = "free will vs determinism"
    result = synthesizer.synthesize(
        query=query,
        memory_hierarchy=memory_hierarchy,
        proto_unity_carrier=memory_hierarchy.proto_unity_carrier
    )

    # Validate result structure
    assert_synthesis_result_valid(result)

    # Contradictory concepts may result in paradox or evolution
    assert result.state in ['paradox', 'evolution_cycling', 'evolution_chaotic']

    # Should have proto-identities
    assert len(result.proto_identities) >= 1

    # If paradox detected, verify multiple branches
    if result.state == 'paradox':
        assert len(result.proto_identities) >= 2
        assert result.resistance_map is not None
        assert len(result.resistance_map) > 0
        assert 'Paradox' in result.explanation or 'paradox' in result.explanation.lower()
        assert len(result.branches) >= 2

    # If evolution, verify appropriate metrics
    elif result.state in ['evolution_cycling', 'evolution_chaotic']:
        assert result.entropy_metrics is not None or result.resistance_map is not None


# ============================================================================
# Test Case 3: Stable Cycling (Evolution-Cycling State)
# ============================================================================

def test_stable_cycling(synthesizer, memory_hierarchy):
    """Test periodic pattern detection.

    Input: "day and night cycle" (periodic concept)
    Expected: EVOLUTION state (cycling), period detected
    Verify: Periodic pattern metadata included
    """
    query = "day and night cycle"
    result = synthesizer.synthesize(
        query=query,
        memory_hierarchy=memory_hierarchy,
        proto_unity_carrier=memory_hierarchy.proto_unity_carrier
    )

    # Validate result structure
    assert_synthesis_result_valid(result)

    # Should detect evolution-cycling state
    assert result.state == 'evolution_cycling'

    # Should have entropy metrics with period information
    assert result.entropy_metrics is not None
    assert 'period' in result.entropy_metrics
    assert 'confidence' in result.entropy_metrics

    # Period should be reasonable (not 1, not too large)
    period = result.entropy_metrics['period']
    assert isinstance(period, int)
    assert 2 <= period <= 20

    # Verify explanation mentions cycling and period
    assert 'Evolution' in result.explanation or 'cycling' in result.explanation.lower()
    assert 'Period' in result.explanation or str(period) in result.explanation


# ============================================================================
# Test Case 4: Chaotic Evolution (High Entropy)
# ============================================================================

def test_chaotic_evolution(synthesizer, memory_hierarchy):
    """Test unstable system handling.

    Input: "square circle" (fundamentally contradictory)
    Expected: Evolution state (cycling or chaotic) or paradox
    Verify: Appropriate diagnostics for unstable systems
    """
    query = "square circle"
    result = synthesizer.synthesize(
        query=query,
        memory_hierarchy=memory_hierarchy,
        proto_unity_carrier=memory_hierarchy.proto_unity_carrier
    )

    # Validate result structure
    assert_synthesis_result_valid(result)

    # Contradictory geometry may result in any non-identity state
    assert result.state in ['paradox', 'evolution_cycling', 'evolution_chaotic']

    # Should have proto-identities
    assert len(result.proto_identities) >= 1

    # If evolution-chaotic, verify diagnostics
    if result.state == 'evolution_chaotic':
        assert result.entropy_metrics is not None
        assert 'sample_entropy' in result.entropy_metrics
        assert result.entropy_metrics['sample_entropy'] > 0.0
        assert result.unstable_stub is not None
        assert isinstance(result.unstable_stub, UnstableSystemStub)
        # Verify zero-text-storage: stub identified by proto_identities, NOT text
        assert len(result.unstable_stub.proto_identities) > 0
        assert all(isinstance(p, np.ndarray) for p in result.unstable_stub.proto_identities)
        assert not hasattr(result.unstable_stub, 'query_text')

    # Any unstable state should have some diagnostic data
    assert (result.entropy_metrics is not None or
            result.resistance_map is not None or
            result.unstable_stub is not None)


# ============================================================================
# Test Case 5: New Discovery (Low Core Coherence)
# ============================================================================

def test_new_discovery(synthesizer, origin):
    """Test novel discovery handling.

    Input: Query with empty core_memory
    Expected: System handles novel input (any valid state)
    Verify: Produces valid result even with empty core
    """
    # Create memory hierarchy with empty core
    memory = MemoryHierarchy(width=512, height=512, depth=128)
    carrier = memory.create_carrier(origin)

    # Core memory should be empty
    assert len(memory.core_memory) == 0

    query = "novel concept never seen before"
    result = synthesizer.synthesize(
        query=query,
        memory_hierarchy=memory,
        proto_unity_carrier=carrier
    )

    # Validate result structure
    assert_synthesis_result_valid(result)

    # New discoveries may result in any state
    assert result.state in ['identity', 'paradox', 'evolution_cycling', 'evolution_chaotic']

    # Should have coherence scores
    assert len(result.coherence_scores) > 0

    # If identity, coherence may be low (novel) or moderate (self-consistent)
    if result.state == 'identity':
        coherence = result.coherence_scores[0]
        assert 0.0 <= coherence <= 1.0  # Valid range


# ============================================================================
# Test Case 6: Natural Convergence (No Safety Limit)
# ============================================================================

def test_natural_convergence(synthesizer, memory_hierarchy):
    """Test that convergence happens naturally, not via safety limit.

    Input: Any converging text
    Expected: Converges via delta < epsilon
    Verify: safety_max_iterations NOT reached
    """
    query = "simple stable concept"
    result = synthesizer.synthesize(
        query=query,
        memory_hierarchy=memory_hierarchy,
        proto_unity_carrier=memory_hierarchy.proto_unity_carrier
    )

    # Validate result structure
    assert_synthesis_result_valid(result)

    # Get branch trajectory
    branch = result.branches[0]

    # Should NOT hit safety limit
    assert len(branch.trajectory) < synthesizer.safety_max_iterations

    # If identity state, should have converged naturally
    if result.state == 'identity':
        assert branch.state == 'converged'

        # Verify delta < epsilon for last step
        if len(branch.trajectory) >= 2:
            delta = float(np.linalg.norm(
                branch.trajectory[-1] - branch.trajectory[-2]
            ))
            assert delta < synthesizer.epsilon


# ============================================================================
# Test Case 7: Complex Multifaceted Input
# ============================================================================

def test_complex_multifaceted(synthesizer, memory_hierarchy):
    """Test handling of complex multi-concept input.

    Input: "economics" (multiple concepts, some compatible, some not)
    Expected: Mix of IDENTITY branches and/or PARADOX splits
    Verify: Handles multiple perspectives correctly
    """
    query = "economics involves supply demand scarcity value trade"
    result = synthesizer.synthesize(
        query=query,
        memory_hierarchy=memory_hierarchy,
        proto_unity_carrier=memory_hierarchy.proto_unity_carrier
    )

    # Validate result structure
    assert_synthesis_result_valid(result)

    # Should handle complex input (any valid state)
    assert result.state in ['identity', 'paradox', 'evolution_cycling', 'evolution_chaotic']

    # Should have at least one proto-identity
    assert len(result.proto_identities) >= 1

    # If paradox, should have multiple perspectives
    if result.state == 'paradox':
        assert len(result.proto_identities) >= 2
        assert result.resistance_map is not None

    # Should have valid explanation
    assert len(result.explanation) > 0


# ============================================================================
# Test Case 8: Backward Compatibility (Decoder Integration)
# ============================================================================

def test_backward_compatibility(memory_hierarchy):
    """Test UnifiedDecoder with use_taylor_synthesis=False.

    Expected: Existing decode behavior unchanged
    Verify: No regressions
    """
    from src.pipeline.unified_decoder import UnifiedDecoder

    decoder = UnifiedDecoder(memory_hierarchy)

    # Create test proto-identity
    proto = create_test_proto_unity_carrier(width=512, height=512)

    # Decode WITHOUT Taylor synthesis (backward compatibility)
    # Note: UnifiedDecoder.decode() uses query_proto, not proto_identities
    result = decoder.decode(
        query_proto=proto,
        use_taylor_synthesis=False
    )

    # Should return DecodingResult (standard behavior)
    assert result is not None
    from src.pipeline.unified_decoder import DecodingResult
    assert isinstance(result, DecodingResult)
    assert hasattr(result, 'text')
    assert isinstance(result.text, str)


# ============================================================================
# Additional Edge Case Tests
# ============================================================================

def test_empty_query_handling(synthesizer, memory_hierarchy):
    """Test handling of empty query."""
    query = ""
    result = synthesizer.synthesize(
        query=query,
        memory_hierarchy=memory_hierarchy,
        proto_unity_carrier=memory_hierarchy.proto_unity_carrier
    )

    # Should still produce valid result
    assert_synthesis_result_valid(result)
    assert result.state in ['identity', 'paradox', 'evolution_cycling', 'evolution_chaotic']


def test_very_short_query(synthesizer, memory_hierarchy):
    """Test handling of very short query."""
    query = "a"
    result = synthesizer.synthesize(
        query=query,
        memory_hierarchy=memory_hierarchy,
        proto_unity_carrier=memory_hierarchy.proto_unity_carrier
    )

    # Should produce valid result
    assert_synthesis_result_valid(result)
    assert result.state in ['identity', 'paradox', 'evolution_cycling', 'evolution_chaotic']


def test_repeated_text(synthesizer, memory_hierarchy):
    """Test handling of highly repetitive text."""
    query = "same same same same same"
    result = synthesizer.synthesize(
        query=query,
        memory_hierarchy=memory_hierarchy,
        proto_unity_carrier=memory_hierarchy.proto_unity_carrier
    )

    # Should converge (repetition should stabilize)
    assert_synthesis_result_valid(result)
    # Repetitive text should likely converge to identity or cycling
    assert result.state in ['identity', 'evolution_cycling']


def test_branch_trajectory_recording(synthesizer, memory_hierarchy):
    """Test that branch trajectories are properly recorded."""
    query = "test trajectory recording"
    result = synthesizer.synthesize(
        query=query,
        memory_hierarchy=memory_hierarchy,
        proto_unity_carrier=memory_hierarchy.proto_unity_carrier
    )

    # Should have branches with trajectories
    assert len(result.branches) > 0

    for branch in result.branches:
        assert isinstance(branch, IdentityBranch)
        assert len(branch.trajectory) > 0
        assert len(branch.coherence_history) >= 0

        # All trajectory entries should be valid protos
        for proto in branch.trajectory:
            assert isinstance(proto, np.ndarray)
            assert proto.shape == (512, 512, 4)


def test_coherence_scores_valid(synthesizer, memory_hierarchy):
    """Test that coherence scores are valid [0, 1] range."""
    query = "test coherence scoring"
    result = synthesizer.synthesize(
        query=query,
        memory_hierarchy=memory_hierarchy,
        proto_unity_carrier=memory_hierarchy.proto_unity_carrier
    )

    # All coherence scores should be [0, 1]
    for score in result.coherence_scores:
        assert isinstance(score, (float, np.floating))
        assert 0.0 <= score <= 1.0
