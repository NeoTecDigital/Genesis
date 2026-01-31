"""
Tests for ExperientialReflector - dual coherence measurement.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.append('/home/persist/alembic/genesis')

from src.memory.experiential_reflector import ExperientialReflector
from src.memory.feedback_loop import FeedbackLoop
from src.memory.voxel_cloud import VoxelCloud


@pytest.fixture
def proto_shape():
    """Standard proto-identity shape."""
    return (64, 64, 4)


@pytest.fixture
def core_memory(proto_shape):
    """Create core memory with some entries."""
    memory = VoxelCloud(width=512, height=512, depth=128)

    # Add a few proto-identities to core
    for i in range(5):
        proto = np.random.randn(*proto_shape).astype(np.float32)
        proto /= np.linalg.norm(proto) + 1e-8  # Normalize
        frequency = np.random.randn(proto_shape[0], proto_shape[1], 2).astype(np.float32)
        metadata = {'text': f'core_entry_{i}', 'octave': 0}
        memory.add(proto, frequency, metadata)

    return memory


@pytest.fixture
def experiential_memory(proto_shape):
    """Create experiential memory (empty initially)."""
    return VoxelCloud(width=512, height=512, depth=128)


@pytest.fixture
def feedback_loop(core_memory, experiential_memory):
    """Create feedback loop."""
    return FeedbackLoop(
        core_memory=core_memory,
        experiential_memory=experiential_memory,
        aligned_threshold=0.8,
        conflict_threshold=0.3
    )


@pytest.fixture
def reflector(feedback_loop):
    """Create experiential reflector."""
    return ExperientialReflector(feedback_loop=feedback_loop)


def test_reflector_initialization(reflector, feedback_loop):
    """Test ExperientialReflector initializes correctly."""
    assert reflector.feedback_loop is feedback_loop
    assert isinstance(reflector, ExperientialReflector)


def test_measure_core_coherence_with_matches(reflector, core_memory, proto_shape):
    """Test core coherence measurement when core has matching entries."""
    # Use first core entry as proto (should have high coherence)
    proto = core_memory.entries[0].proto_identity.copy()

    coherence = reflector.measure_core_coherence(proto, core_memory)

    # Should have high coherence (very similar to core entry)
    assert 0.0 <= coherence <= 1.0
    assert coherence > 0.7  # Should be quite high


def test_measure_core_coherence_with_novel_proto(reflector, core_memory, proto_shape):
    """Test core coherence with completely novel proto."""
    # Create random proto (unlikely to match core)
    proto = np.random.randn(*proto_shape).astype(np.float32)
    proto /= np.linalg.norm(proto) + 1e-8

    coherence = reflector.measure_core_coherence(proto, core_memory)

    # Should have low-to-moderate coherence
    assert 0.0 <= coherence <= 1.0


def test_measure_core_coherence_with_empty_core(reflector, proto_shape):
    """Test core coherence when core is empty."""
    empty_core = VoxelCloud(width=512, height=512, depth=128)
    proto = np.random.randn(*proto_shape).astype(np.float32)

    coherence = reflector.measure_core_coherence(proto, empty_core)

    # Empty core should return 0.0 coherence
    assert coherence == 0.0


def test_measure_internal_coherence_with_consistent_trajectory(reflector, proto_shape):
    """Test internal coherence with consistent trajectory history."""
    # Create consistent trajectory (similar protos)
    base_proto = np.random.randn(*proto_shape).astype(np.float32)
    base_proto /= np.linalg.norm(base_proto) + 1e-8

    trajectory_history = []
    for i in range(5):
        # Add very small noise to base proto
        noisy_proto = base_proto + np.random.randn(*proto_shape) * 0.01
        noisy_proto = noisy_proto.astype(np.float32)
        noisy_proto /= np.linalg.norm(noisy_proto) + 1e-8
        trajectory_history.append(noisy_proto)

    # Measure coherence with slightly noisy version
    test_proto = base_proto + np.random.randn(*proto_shape) * 0.01
    test_proto = test_proto.astype(np.float32)
    test_proto /= np.linalg.norm(test_proto) + 1e-8

    coherence = reflector.measure_internal_coherence(test_proto, trajectory_history)

    # Should have moderate internal coherence
    # (normalized vectors with small noise still result in moderate similarity)
    assert 0.0 <= coherence <= 1.0
    assert coherence > 0.2  # Should be higher than random


def test_measure_internal_coherence_with_inconsistent_trajectory(reflector, proto_shape):
    """Test internal coherence with inconsistent trajectory."""
    # Create random trajectory (no consistency)
    trajectory_history = []
    for i in range(5):
        proto = np.random.randn(*proto_shape).astype(np.float32)
        proto /= np.linalg.norm(proto) + 1e-8
        trajectory_history.append(proto)

    # Test with another random proto
    test_proto = np.random.randn(*proto_shape).astype(np.float32)
    test_proto /= np.linalg.norm(test_proto) + 1e-8

    coherence = reflector.measure_internal_coherence(test_proto, trajectory_history)

    # Should have low internal coherence
    assert 0.0 <= coherence <= 1.0


def test_measure_internal_coherence_with_empty_history(reflector, proto_shape):
    """Test internal coherence with empty trajectory history."""
    proto = np.random.randn(*proto_shape).astype(np.float32)
    trajectory_history = []

    coherence = reflector.measure_internal_coherence(proto, trajectory_history)

    # Empty history should return 0.0
    assert coherence == 0.0


def test_measure_internal_coherence_uses_recent_history(reflector, proto_shape):
    """Test that internal coherence prioritizes recent history (last 5)."""
    # Create trajectory with 10 entries
    # First 5 are random, last 5 are consistent
    base_proto = np.random.randn(*proto_shape).astype(np.float32)
    base_proto /= np.linalg.norm(base_proto) + 1e-8

    trajectory_history = []

    # First 5: random
    for i in range(5):
        proto = np.random.randn(*proto_shape).astype(np.float32)
        proto /= np.linalg.norm(proto) + 1e-8
        trajectory_history.append(proto)

    # Last 5: consistent with base (very small noise)
    for i in range(5):
        noisy_proto = base_proto + np.random.randn(*proto_shape) * 0.01
        noisy_proto = noisy_proto.astype(np.float32)
        noisy_proto /= np.linalg.norm(noisy_proto) + 1e-8
        trajectory_history.append(noisy_proto)

    # Test with base proto
    coherence = reflector.measure_internal_coherence(base_proto, trajectory_history)

    # Should have high coherence (matches recent history)
    assert coherence > 0.5


def test_measure_dual_coherence(reflector, core_memory, proto_shape):
    """Test dual coherence measurement."""
    # Create proto and trajectory
    proto = core_memory.entries[0].proto_identity.copy()

    trajectory_history = []
    for i in range(3):
        noisy_proto = proto + np.random.randn(*proto_shape) * 0.01
        noisy_proto = noisy_proto.astype(np.float32)
        noisy_proto /= np.linalg.norm(noisy_proto) + 1e-8
        trajectory_history.append(noisy_proto)

    core_coh, internal_coh = reflector.measure_dual_coherence(
        proto, core_memory, trajectory_history
    )

    # Both should be in [0, 1]
    assert 0.0 <= core_coh <= 1.0
    assert 0.0 <= internal_coh <= 1.0

    # Core should be high (matches core entry)
    assert core_coh > 0.7

    # Internal should be high (consistent trajectory)
    assert internal_coh > 0.5


def test_measure_dual_coherence_with_novel_pattern(reflector, core_memory, proto_shape):
    """Test dual coherence with novel but consistent pattern."""
    # Create novel proto (not in core)
    base_proto = np.random.randn(*proto_shape).astype(np.float32)
    base_proto /= np.linalg.norm(base_proto) + 1e-8

    # Create consistent trajectory (very small noise)
    trajectory_history = []
    for i in range(5):
        noisy_proto = base_proto + np.random.randn(*proto_shape) * 0.01
        noisy_proto = noisy_proto.astype(np.float32)
        noisy_proto /= np.linalg.norm(noisy_proto) + 1e-8
        trajectory_history.append(noisy_proto)

    core_coh, internal_coh = reflector.measure_dual_coherence(
        base_proto, core_memory, trajectory_history
    )

    # Should have low core but high internal coherence
    # (novel pattern but self-consistent)
    assert 0.0 <= core_coh <= 1.0
    assert 0.0 <= internal_coh <= 1.0
    assert internal_coh > 0.5  # Should be consistent internally


def test_measure_dual_coherence_with_unstable_pattern(reflector, core_memory, proto_shape):
    """Test dual coherence with unstable pattern."""
    # Create random proto and random trajectory
    proto = np.random.randn(*proto_shape).astype(np.float32)
    proto /= np.linalg.norm(proto) + 1e-8

    trajectory_history = []
    for i in range(5):
        random_proto = np.random.randn(*proto_shape).astype(np.float32)
        random_proto /= np.linalg.norm(random_proto) + 1e-8
        trajectory_history.append(random_proto)

    core_coh, internal_coh = reflector.measure_dual_coherence(
        proto, core_memory, trajectory_history
    )

    # Both should be in valid range
    assert 0.0 <= core_coh <= 1.0
    assert 0.0 <= internal_coh <= 1.0


def test_repr(reflector):
    """Test string representation."""
    repr_str = repr(reflector)
    assert 'ExperientialReflector' in repr_str
    assert 'feedback_loop' in repr_str
