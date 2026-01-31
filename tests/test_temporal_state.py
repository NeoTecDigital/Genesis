"""
Tests for Phase 2: Temporal Buffer and State Classification.

Tests:
- TemporalBuffer: add, get_derivatives, predict_next
- StateClassifier: all three states (PARADOX, EVOLUTION, IDENTITY)
- VoxelCloud temporal tracking integration
- Taylor series prediction accuracy
"""

import unittest
import numpy as np
import time
import sys
from pathlib import Path

sys.path.append('/home/persist/alembic/genesis')

from src.memory.temporal_buffer import TemporalBuffer
from src.memory.state_classifier import StateClassifier, SignalState
from src.memory.voxel_cloud import VoxelCloud


class TestTemporalBuffer(unittest.TestCase):
    """Test TemporalBuffer circular buffer and derivative computation."""

    def test_init(self):
        """Test buffer initialization."""
        buffer = TemporalBuffer(max_length=50)
        self.assertTrue(len(buffer) == 0)
        self.assertTrue(buffer.max_length == 50)

    def test_add_single(self):
        """Test adding single entry."""
        buffer = TemporalBuffer(max_length=100)
        proto = np.random.randn(16, 16, 4)
        timestamp = 1.0

        buffer.add(proto, timestamp)

        self.assertTrue(len(buffer) == 1)
        self.assertTrue(buffer.buffer[0][0] == timestamp)
        np.testing.assert_array_equal(buffer.buffer[0][1], proto)

    def test_add_multiple(self):
        """Test adding multiple entries."""
        buffer = TemporalBuffer(max_length=10)

        for i in range(5):
            proto = np.ones((8, 8, 4)) * i
            buffer.add(proto, float(i))

        self.assertTrue(len(buffer) == 5)
        self.assertTrue(buffer.buffer[0][0] == 0.0)
        self.assertTrue(buffer.buffer[-1][0] == 4.0)

    def test_circular_buffer(self):
        """Test circular buffer behavior when max_length exceeded."""
        buffer = TemporalBuffer(max_length=3)

        # Add 5 entries, should keep only last 3
        for i in range(5):
            proto = np.ones((8, 8, 4)) * i
            buffer.add(proto, float(i))

        self.assertTrue(len(buffer) == 3)
        # Should have entries 2, 3, 4 (oldest removed)
        self.assertTrue(buffer.buffer[0][0] == 2.0)
        self.assertTrue(buffer.buffer[1][0] == 3.0)
        self.assertTrue(buffer.buffer[2][0] == 4.0)

    def test_first_derivative_simple(self):
        """Test first derivative with simple linear change."""
        buffer = TemporalBuffer()

        # Linear change: proto = t * [1, 1, 1, 1]
        # Derivative should be [1, 1, 1, 1]
        proto1 = np.ones((8, 8, 4)) * 0.0
        proto2 = np.ones((8, 8, 4)) * 1.0

        buffer.add(proto1, 0.0)
        buffer.add(proto2, 1.0)

        deriv = buffer.get_derivatives(order=1)
        self.assertTrue(deriv is not None)
        np.testing.assert_allclose(deriv, np.ones((8, 8, 4)), atol=1e-6)

    def test_first_derivative_zero_change(self):
        """Test first derivative with no change."""
        buffer = TemporalBuffer()

        proto = np.ones((8, 8, 4)) * 5.0
        buffer.add(proto, 0.0)
        buffer.add(proto.copy(), 1.0)

        deriv = buffer.get_derivatives(order=1)
        self.assertTrue(deriv is not None)
        np.testing.assert_allclose(deriv, np.zeros((8, 8, 4)), atol=1e-6)

    def test_first_derivative_insufficient_data(self):
        """Test first derivative with insufficient data."""
        buffer = TemporalBuffer()
        proto = np.ones((8, 8, 4))
        buffer.add(proto, 0.0)

        deriv = buffer.get_derivatives(order=1)
        self.assertTrue(deriv is None)

    def test_second_derivative_simple(self):
        """Test second derivative with constant acceleration."""
        buffer = TemporalBuffer()

        # Quadratic change: proto = t² * [1, 1, 1, 1]
        # First derivative: 2t
        # Second derivative: 2
        for t in [0.0, 1.0, 2.0]:
            proto = np.ones((8, 8, 4)) * (t ** 2)
            buffer.add(proto, t)

        deriv2 = buffer.get_derivatives(order=2)
        self.assertTrue(deriv2 is not None)
        # Second derivative should be approximately 2
        np.testing.assert_allclose(deriv2, np.ones((8, 8, 4)) * 2.0, atol=0.5)

    def test_second_derivative_insufficient_data(self):
        """Test second derivative with insufficient data."""
        buffer = TemporalBuffer()

        proto1 = np.ones((8, 8, 4))
        proto2 = np.ones((8, 8, 4)) * 2.0

        buffer.add(proto1, 0.0)
        buffer.add(proto2, 1.0)

        deriv2 = buffer.get_derivatives(order=2)
        self.assertTrue(deriv2 is None)  # Need at least 3 entries

    def test_predict_next_first_order(self):
        """Test first-order Taylor series prediction."""
        buffer = TemporalBuffer()

        # Linear change: proto = t
        proto1 = np.ones((8, 8, 4)) * 0.0
        proto2 = np.ones((8, 8, 4)) * 1.0

        buffer.add(proto1, 0.0)
        buffer.add(proto2, 1.0)

        # Predict at t=2 (delta_t=1 from last timestamp)
        predicted = buffer.predict_next(delta_t=1.0, order=1)
        self.assertTrue(predicted is not None)
        # Should predict proto ≈ 2.0
        np.testing.assert_allclose(predicted, np.ones((8, 8, 4)) * 2.0, atol=0.1)

    def test_predict_next_second_order(self):
        """Test second-order Taylor series prediction."""
        buffer = TemporalBuffer()

        # Quadratic: proto = t²
        for t in [0.0, 1.0, 2.0]:
            proto = np.ones((8, 8, 4)) * (t ** 2)
            buffer.add(proto, t)

        # Predict at t=3 (delta_t=1)
        # True value at t=3: 9.0
        # Taylor: proto(2) + deriv1*1 + 0.5*deriv2*1
        # proto(2)=4, deriv1≈4, deriv2≈2
        # Prediction: 4 + 4 + 1 = 9
        predicted = buffer.predict_next(delta_t=1.0, order=2)
        self.assertTrue(predicted is not None)
        expected = 9.0
        np.testing.assert_allclose(predicted, np.ones((8, 8, 4)) * expected, atol=1.0)

    def test_predict_next_insufficient_data(self):
        """Test prediction with insufficient data."""
        buffer = TemporalBuffer()
        proto = np.ones((8, 8, 4))
        buffer.add(proto, 0.0)

        predicted = buffer.predict_next(delta_t=1.0, order=1)
        self.assertTrue(predicted is None)

    def test_clear(self):
        """Test buffer clearing."""
        buffer = TemporalBuffer()
        for i in range(5):
            buffer.add(np.ones((8, 8, 4)) * i, float(i))

        self.assertTrue(len(buffer) == 5)
        buffer.clear()
        self.assertTrue(len(buffer) == 0)

    def test_repr(self):
        """Test string representation."""
        buffer = TemporalBuffer(max_length=50)
        repr_empty = repr(buffer)
        self.assertTrue("empty" in repr_empty)

        buffer.add(np.ones((8, 8, 4)), 0.0)
        buffer.add(np.ones((8, 8, 4)), 2.5)
        repr_filled = repr(buffer)
        self.assertTrue("2 entries" in repr_filled)
        self.assertTrue("2.50s" in repr_filled)


class TestStateClassifier(unittest.TestCase):
    """Test StateClassifier for all three states."""

    def test_init(self):
        """Test classifier initialization."""
        classifier = StateClassifier(
            evolution_threshold=0.2,
            identity_coherence=0.9,
            paradox_coherence=0.2
        )
        self.assertTrue(classifier.evolution_threshold == 0.2)
        self.assertTrue(classifier.identity_coherence == 0.9)
        self.assertTrue(classifier.paradox_coherence == 0.2)

    def test_classify_identity_stable_high_coherence(self):
        """Test IDENTITY state: stable (low derivative) + high coherence."""
        buffer = TemporalBuffer()
        classifier = StateClassifier(
            evolution_threshold=0.1,
            identity_coherence=0.85,
            paradox_coherence=0.3
        )

        # Add stable proto (no change)
        proto = np.ones((8, 8, 4)) * 5.0
        buffer.add(proto, 0.0)
        buffer.add(proto.copy(), 1.0)

        coherence = 0.95  # High coherence

        state = classifier.classify(buffer, coherence)
        self.assertTrue(state == SignalState.IDENTITY)

    def test_classify_paradox_changing_low_coherence(self):
        """Test PARADOX state: changing (high derivative) + low coherence."""
        buffer = TemporalBuffer()
        classifier = StateClassifier(
            evolution_threshold=0.1,
            identity_coherence=0.85,
            paradox_coherence=0.3
        )

        # Add rapidly changing proto
        proto1 = np.ones((8, 8, 4)) * 0.0
        proto2 = np.ones((8, 8, 4)) * 5.0  # Large change

        buffer.add(proto1, 0.0)
        buffer.add(proto2, 1.0)

        coherence = 0.2  # Low coherence (conflicting)

        state = classifier.classify(buffer, coherence)
        self.assertTrue(state == SignalState.PARADOX)

    def test_classify_evolution_moderate(self):
        """Test EVOLUTION state: moderate change + moderate coherence."""
        buffer = TemporalBuffer()
        classifier = StateClassifier(
            evolution_threshold=0.1,
            identity_coherence=0.85,
            paradox_coherence=0.3
        )

        # Moderate change
        proto1 = np.ones((8, 8, 4)) * 0.0
        proto2 = np.ones((8, 8, 4)) * 0.5  # Moderate change

        buffer.add(proto1, 0.0)
        buffer.add(proto2, 1.0)

        coherence = 0.6  # Moderate coherence

        state = classifier.classify(buffer, coherence)
        self.assertTrue(state == SignalState.EVOLUTION)

    def test_classify_evolution_changing_high_coherence(self):
        """Test EVOLUTION when changing but coherent (learning)."""
        buffer = TemporalBuffer()
        classifier = StateClassifier(
            evolution_threshold=0.1,
            identity_coherence=0.85,
            paradox_coherence=0.3
        )

        # Changing but not enough for identity
        proto1 = np.ones((8, 8, 4)) * 0.0
        proto2 = np.ones((8, 8, 4)) * 0.3

        buffer.add(proto1, 0.0)
        buffer.add(proto2, 1.0)

        coherence = 0.8  # High but not above identity threshold

        state = classifier.classify(buffer, coherence)
        self.assertTrue(state == SignalState.EVOLUTION)

    def test_classify_insufficient_data(self):
        """Test classification with insufficient data defaults to EVOLUTION."""
        buffer = TemporalBuffer()
        classifier = StateClassifier()

        # Only one entry
        buffer.add(np.ones((8, 8, 4)), 0.0)

        state = classifier.classify(buffer, coherence=0.5)
        self.assertTrue(state == SignalState.EVOLUTION)

    def test_repr(self):
        """Test string representation."""
        classifier = StateClassifier(
            evolution_threshold=0.15,
            identity_coherence=0.9,
            paradox_coherence=0.25
        )
        repr_str = repr(classifier)
        self.assertTrue("0.15" in repr_str)
        self.assertTrue("0.9" in repr_str)
        self.assertTrue("0.25" in repr_str)


class TestVoxelCloudTemporalIntegration(unittest.TestCase):
    """Test VoxelCloud temporal tracking integration."""

    def test_temporal_buffer_initialized(self):
        """Test that VoxelCloud initializes temporal buffer."""
        cloud = VoxelCloud(width=64, height=64, depth=32)
        self.assertTrue(isinstance(cloud.temporal_buffer, TemporalBuffer))
        self.assertTrue(isinstance(cloud.state_classifier, StateClassifier))

    def test_compute_coherence_empty(self):
        """Test coherence computation with empty cloud."""
        cloud = VoxelCloud(width=64, height=64, depth=32)
        proto = np.random.randn(64, 64, 4)
        coherence = cloud.compute_coherence(proto)
        self.assertTrue(coherence == 0.0)

    def test_compute_coherence_single_entry(self):
        """Test coherence computation with one entry."""
        cloud = VoxelCloud(width=64, height=64, depth=32)

        # Add first proto
        proto1 = np.random.randn(64, 64, 4)
        freq1 = np.random.randn(64, 64, 4)
        cloud.add(proto1, freq1, {'text': 'test1'})

        # Compute coherence with similar proto
        proto2 = proto1 + np.random.randn(64, 64, 4) * 0.1
        coherence = cloud.compute_coherence(proto2)

        # Should have high coherence (similar to proto1)
        self.assertTrue(0.0 <= coherence <= 1.0)
        self.assertTrue(coherence > 0.5)  # Should be similar

    def test_add_with_temporal_tracking(self):
        """Test adding proto with temporal tracking."""
        cloud = VoxelCloud(width=64, height=64, depth=32)

        proto = np.random.randn(64, 64, 4)
        freq = np.random.randn(64, 64, 4)
        metadata = {'text': 'test proto'}

        timestamp = 100.0
        cloud.add_with_temporal_tracking(proto, freq, metadata, timestamp=timestamp)

        # Check entry was added
        self.assertTrue(len(cloud.entries) == 1)

        # Check temporal buffer was updated
        self.assertTrue(len(cloud.temporal_buffer) == 1)

        # Check entry has state and coherence
        entry = cloud.entries[0]
        self.assertTrue(entry.current_state is not None)
        self.assertTrue(isinstance(entry.current_state, SignalState))
        self.assertTrue(entry.coherence_vs_core >= 0.0)
        self.assertTrue('current_state' in entry.metadata)
        self.assertTrue('timestamp' in entry.metadata)
        self.assertTrue('coherence' in entry.metadata)

    def test_temporal_tracking_multiple_adds(self):
        """Test temporal tracking with multiple adds."""
        cloud = VoxelCloud(width=64, height=64, depth=32)

        # Add multiple protos over time
        for i in range(5):
            proto = np.ones((64, 64, 4)) * i
            freq = np.random.randn(64, 64, 4)
            metadata = {'text': f'proto_{i}'}
            timestamp = float(i)

            cloud.add_with_temporal_tracking(proto, freq, metadata, timestamp=timestamp)

        # Temporal buffer should have all 5
        self.assertTrue(len(cloud.temporal_buffer) == 5)

        # Each entry should have state
        for entry in cloud.entries:
            self.assertTrue(entry.current_state is not None)
            self.assertTrue(len(entry.temporal_history) >= 1)

    def test_temporal_tracking_state_evolution(self):
        """Test that state changes from PARADOX → EVOLUTION → IDENTITY."""
        cloud = VoxelCloud(width=64, height=64, depth=32)

        # First add: should be EVOLUTION (no prior context)
        proto1 = np.random.randn(64, 64, 4)
        freq1 = np.random.randn(64, 64, 4)
        cloud.add_with_temporal_tracking(proto1, freq1, {'text': 'proto1'}, timestamp=0.0)

        state1 = cloud.entries[-1].current_state
        # First entry with no history defaults to EVOLUTION
        self.assertTrue(state1 == SignalState.EVOLUTION)

        # Second add: similar proto, stable → should trend toward IDENTITY
        proto2 = proto1 + np.random.randn(64, 64, 4) * 0.01  # Very similar
        freq2 = freq1 + np.random.randn(64, 64, 4) * 0.01
        cloud.add_with_temporal_tracking(proto2, freq2, {'text': 'proto2'}, timestamp=1.0)

        # Later entries should have higher coherence
        state2 = cloud.entries[-1].current_state
        coherence2 = cloud.entries[-1].coherence_vs_core
        self.assertTrue(coherence2 > 0.5)  # Should have moderate to high coherence

    def test_save_load_with_temporal(self):
        """Test save/load preserves temporal components."""
        import tempfile
        import os

        cloud = VoxelCloud(width=32, height=32, depth=16)

        # Add some protos with temporal tracking
        for i in range(3):
            proto = np.ones((32, 32, 4)) * i
            freq = np.random.randn(32, 32, 4)
            cloud.add_with_temporal_tracking(proto, freq, {'text': f'proto_{i}'}, timestamp=float(i))

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name

        try:
            cloud.save(temp_path)

            # Load into new cloud
            cloud2 = VoxelCloud()
            cloud2.load(temp_path)

            # Check temporal components preserved
            self.assertTrue(isinstance(cloud2.temporal_buffer, TemporalBuffer))
            self.assertTrue(isinstance(cloud2.state_classifier, StateClassifier))
            self.assertTrue(len(cloud2.temporal_buffer) == len(cloud.temporal_buffer))

            # Check entries preserved
            self.assertTrue(len(cloud2.entries) == len(cloud.entries))
            for e1, e2 in zip(cloud.entries, cloud2.entries):
                self.assertTrue(e1.current_state == e2.current_state)
                self.assertTrue(e1.coherence_vs_core == e2.coherence_vs_core)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_backward_compatibility_load(self):
        """Test loading old cloud without temporal components."""
        import tempfile
        import pickle
        import os

        # Create old-style cloud data without temporal components
        old_data = {
            'width': 32,
            'height': 32,
            'depth': 16,
            'entries': [],
            'spatial_index': {},
            'collapse_config': {},
            'synthesis_config': {}
        }

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
            pickle.dump(old_data, f)

        try:
            # Load should create default temporal components
            cloud = VoxelCloud()
            cloud.load(temp_path)

            self.assertTrue(isinstance(cloud.temporal_buffer, TemporalBuffer))
            self.assertTrue(isinstance(cloud.state_classifier, StateClassifier))

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestTaylorSeriesAccuracy(unittest.TestCase):
    """Test Taylor series prediction accuracy."""

    def test_linear_prediction_accuracy(self):
        """Test prediction accuracy for linear function."""
        buffer = TemporalBuffer()

        # Linear: f(t) = 2t
        for t in [0.0, 1.0, 2.0, 3.0]:
            proto = np.ones((8, 8, 4)) * (2.0 * t)
            buffer.add(proto, t)

        # Predict at t=4 (delta_t=1)
        predicted = buffer.predict_next(delta_t=1.0, order=2)
        expected = np.ones((8, 8, 4)) * 8.0  # f(4) = 8

        np.testing.assert_allclose(predicted, expected, atol=0.5)

    def test_quadratic_prediction_accuracy(self):
        """Test prediction accuracy for quadratic function."""
        buffer = TemporalBuffer()

        # Quadratic: f(t) = t²
        for t in [0.0, 1.0, 2.0, 3.0]:
            proto = np.ones((8, 8, 4)) * (t ** 2)
            buffer.add(proto, t)

        # Predict at t=4 (delta_t=1)
        predicted = buffer.predict_next(delta_t=1.0, order=2)
        expected = np.ones((8, 8, 4)) * 16.0  # f(4) = 16

        # Second-order Taylor should be exact for quadratic
        np.testing.assert_allclose(predicted, expected, atol=2.0)

    def test_sine_prediction_accuracy(self):
        """Test prediction accuracy for sinusoidal function (approximate)."""
        buffer = TemporalBuffer()

        # Sine: f(t) = sin(t)
        for t in np.linspace(0, 2, 10):
            proto = np.ones((8, 8, 4)) * np.sin(t)
            buffer.add(proto, t)

        # Predict at t=2.1 (delta_t=0.1)
        predicted = buffer.predict_next(delta_t=0.1, order=2)
        expected = np.ones((8, 8, 4)) * np.sin(2.1)

        # Taylor approximation should be close for small delta_t
        np.testing.assert_allclose(predicted, expected, atol=0.05)


if __name__ == '__main__':
    unittest.main([__file__, '-v'])
