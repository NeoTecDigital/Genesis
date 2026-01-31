"""
Tests for Phase 3: Feedback Loop & Reset Mechanisms.

Tests:
- FeedbackLoop: ALIGNED, LEARNING, CONFLICT scenarios
- ExperientialMemoryManager: reset, consolidation, selective reset
- MemoryHierarchy integration: full workflow with feedback and lifecycle
"""

import unittest
import numpy as np
import time

from src.memory.feedback_loop import FeedbackLoop
from src.memory.experiential_manager import ExperientialMemoryManager
from src.memory.memory_hierarchy import MemoryHierarchy
from src.memory.voxel_cloud import VoxelCloud
from src.memory.state_classifier import SignalState
from src.origin import Origin


class TestFeedbackLoop(unittest.TestCase):
    """Test FeedbackLoop self-reflection functionality."""

    def test_init(self):
        """Test feedback loop initialization."""
        core = VoxelCloud(width=64, height=64, depth=16)
        exp = VoxelCloud(width=64, height=64, depth=16)

        loop = FeedbackLoop(core, exp)

        self.assertTrue(loop.core is core)
        self.assertTrue(loop.experiential is exp)
        self.assertTrue(loop.aligned_threshold == 0.8)
        self.assertTrue(loop.conflict_threshold == 0.3)

    def test_aligned_scenario(self):
        """Test ALIGNED state: experiential matches core."""
        core = VoxelCloud(width=64, height=64, depth=16)
        exp = VoxelCloud(width=64, height=64, depth=16)

        # Add similar proto to core
        proto_core = np.random.randn(64, 64, 4).astype(np.float32)
        freq = np.random.randn(64, 64, 4).astype(np.float32)
        core.add(proto_core, freq, {'text': 'core knowledge'})

        # Create very similar experiential proto (just add small noise)
        proto_exp = proto_core + np.random.randn(64, 64, 4).astype(np.float32) * 0.01

        # Test self-reflection
        loop = FeedbackLoop(core, exp)
        coherence, state, recommendation = loop.self_reflect(
            proto_exp,
            np.array([0.0, 0.0, 0.0, 1.0])  # dummy quaternion
        )

        # Should be ALIGNED (high coherence)
        self.assertGreater(coherence, 0.8)
        self.assertTrue(state == SignalState.IDENTITY)
        self.assertTrue(recommendation == 'ALIGNED')

    def test_conflict_scenario(self):
        """Test CONFLICT state: experiential conflicts with core."""
        core = VoxelCloud(width=64, height=64, depth=16)
        exp = VoxelCloud(width=64, height=64, depth=16)

        # Add proto to core
        proto_core = np.ones((64, 64, 4), dtype=np.float32) * 5.0
        freq = np.random.randn(64, 64, 4).astype(np.float32)
        core.add(proto_core, freq, {'text': 'core knowledge'})

        # Create very different experiential proto
        proto_exp = -np.ones((64, 64, 4), dtype=np.float32) * 5.0

        # Test self-reflection
        loop = FeedbackLoop(core, exp)
        coherence, state, recommendation = loop.self_reflect(
            proto_exp,
            np.array([0.0, 0.0, 0.0, 1.0])
        )

        # Should be CONFLICT (low coherence)
        self.assertLess(coherence, 0.3)
        self.assertTrue(state == SignalState.PARADOX)
        self.assertTrue(recommendation == 'CONFLICT')

    def test_learning_scenario(self):
        """Test LEARNING state: experiential differs but not conflicting."""
        core = VoxelCloud(width=64, height=64, depth=16)
        exp = VoxelCloud(width=64, height=64, depth=16)

        # Add proto to core
        proto_core = np.random.randn(64, 64, 4).astype(np.float32)
        freq = np.random.randn(64, 64, 4).astype(np.float32)
        core.add(proto_core, freq, {'text': 'core knowledge'})

        # Create moderately different experiential proto
        proto_exp = proto_core * 0.5 + np.random.randn(64, 64, 4).astype(np.float32) * 0.5

        # Test self-reflection
        loop = FeedbackLoop(core, exp)
        coherence, state, recommendation = loop.self_reflect(
            proto_exp,
            np.array([0.0, 0.0, 0.0, 1.0])
        )

        # Should be LEARNING (moderate coherence)
        self.assertTrue(0.3 < coherence < 0.8)
        self.assertTrue(state == SignalState.EVOLUTION)
        self.assertTrue(recommendation == 'LEARNING')

    def test_no_core_knowledge(self):
        """Test feedback loop with empty core memory."""
        core = VoxelCloud(width=64, height=64, depth=16)
        exp = VoxelCloud(width=64, height=64, depth=16)

        # No entries in core
        proto_exp = np.random.randn(64, 64, 4).astype(np.float32)

        loop = FeedbackLoop(core, exp)
        coherence, state, recommendation = loop.self_reflect(
            proto_exp,
            np.array([0.0, 0.0, 0.0, 1.0])
        )

        # No core knowledge → coherence 0 → CONFLICT
        self.assertTrue(coherence == 0.0)
        self.assertTrue(state == SignalState.PARADOX)
        self.assertTrue(recommendation == 'CONFLICT')


class TestExperientialMemoryManager(unittest.TestCase):
    """Test ExperientialMemoryManager lifecycle management."""

    def test_init(self):
        """Test manager initialization."""
        carrier = np.random.randn(64, 64, 4).astype(np.float32)
        exp = VoxelCloud(width=64, height=64, depth=16)

        manager = ExperientialMemoryManager(carrier, exp)

        self.assertTrue(manager.experiential is exp)
        np.testing.assert_array_equal(manager.baseline_carrier, carrier)
        np.testing.assert_array_equal(manager.current_carrier, carrier)

    def test_reset_to_baseline(self):
        """Test reset clears experiential memory."""
        carrier = np.random.randn(64, 64, 4).astype(np.float32)
        exp = VoxelCloud(width=64, height=64, depth=16)

        # Add some entries to experiential
        for i in range(5):
            proto = np.random.randn(64, 64, 4).astype(np.float32)
            freq = np.random.randn(64, 64, 4).astype(np.float32)
            exp.add(proto, freq, {'text': f'entry_{i}'})

        self.assertGreaterEqual(len(exp), 1)  # Should have entries (or less due to collapse)

        # Reset
        manager = ExperientialMemoryManager(carrier, exp)
        manager.reset_to_baseline()

        # Should be empty
        self.assertTrue(len(exp.entries) == 0)
        self.assertTrue(len(exp.spatial_index) == 0)
        self.assertTrue(len(exp.temporal_buffer) == 0)
        np.testing.assert_array_equal(manager.current_carrier, carrier)

    def test_consolidate_identity_only(self):
        """Test consolidation only moves IDENTITY state entries."""
        carrier = np.random.randn(64, 64, 4).astype(np.float32)
        exp = VoxelCloud(width=64, height=64, depth=16)
        core = VoxelCloud(width=64, height=64, depth=16)

        # Add entries with different states
        proto1 = np.random.randn(64, 64, 4).astype(np.float32)
        freq1 = np.random.randn(64, 64, 4).astype(np.float32)
        exp.add(proto1, freq1, {'text': 'identity_entry'})
        exp.entries[-1].current_state = SignalState.IDENTITY
        exp.entries[-1].resonance_strength = 0.9

        proto2 = np.random.randn(64, 64, 4).astype(np.float32)
        freq2 = np.random.randn(64, 64, 4).astype(np.float32)
        exp.add(proto2, freq2, {'text': 'evolution_entry'})
        exp.entries[-1].current_state = SignalState.EVOLUTION
        exp.entries[-1].resonance_strength = 0.9

        proto3 = np.random.randn(64, 64, 4).astype(np.float32)
        freq3 = np.random.randn(64, 64, 4).astype(np.float32)
        exp.add(proto3, freq3, {'text': 'paradox_entry'})
        exp.entries[-1].current_state = SignalState.PARADOX
        exp.entries[-1].resonance_strength = 0.9

        # Consolidate
        manager = ExperientialMemoryManager(carrier, exp)
        consolidated = manager.consolidate_to_core(core, threshold=0.8)

        # Only IDENTITY should be consolidated
        self.assertEqual(consolidated, 1)
        self.assertGreaterEqual(len(core), 1)  # may be merged

    def test_consolidate_threshold(self):
        """Test consolidation respects resonance threshold."""
        carrier = np.random.randn(64, 64, 4).astype(np.float32)
        exp = VoxelCloud(width=64, height=64, depth=16)
        core = VoxelCloud(width=64, height=64, depth=16)

        # Add IDENTITY entries with different resonance strengths
        proto1 = np.random.randn(64, 64, 4).astype(np.float32)
        freq1 = np.random.randn(64, 64, 4).astype(np.float32)
        exp.add(proto1, freq1, {'text': 'high_resonance'})
        exp.entries[-1].current_state = SignalState.IDENTITY
        exp.entries[-1].resonance_strength = 0.9

        proto2 = np.random.randn(64, 64, 4).astype(np.float32)
        freq2 = np.random.randn(64, 64, 4).astype(np.float32)
        exp.add(proto2, freq2, {'text': 'low_resonance'})
        exp.entries[-1].current_state = SignalState.IDENTITY
        exp.entries[-1].resonance_strength = 0.5

        # Consolidate with threshold 0.8
        manager = ExperientialMemoryManager(carrier, exp)
        consolidated = manager.consolidate_to_core(core, threshold=0.8)

        # Only high resonance should be consolidated
        self.assertEqual(consolidated, 1)

    def test_selective_reset(self):
        """Test selective reset keeps IDENTITY entries."""
        carrier = np.random.randn(64, 64, 4).astype(np.float32)
        exp = VoxelCloud(width=64, height=64, depth=16)

        # Add entries with different states
        for state in [SignalState.IDENTITY, SignalState.EVOLUTION, SignalState.PARADOX]:
            proto = np.random.randn(64, 64, 4).astype(np.float32)
            freq = np.random.randn(64, 64, 4).astype(np.float32)
            exp.add(proto, freq, {'text': f'{state.name}_entry'})
            exp.entries[-1].current_state = state

        initial_count = len(exp.entries)
        self.assertGreaterEqual(initial_count, 3)

        # Selective reset - keep only IDENTITY
        manager = ExperientialMemoryManager(carrier, exp)
        removed = manager.selective_reset(keep_states={SignalState.IDENTITY})

        # Should remove EVOLUTION and PARADOX entries
        self.assertGreaterEqual(removed, 0)
        # Verify all remaining are IDENTITY
        for entry in exp.entries:
            self.assertTrue(entry.current_state == SignalState.IDENTITY)


class TestMemoryHierarchyIntegration(unittest.TestCase):
    """Test MemoryHierarchy integration with Phase 3 features."""

    def test_initialization(self):
        """Test hierarchy initialization."""
        hierarchy = MemoryHierarchy(width=64, height=64, depth=16)

        self.assertTrue(hierarchy.proto_unity_carrier is None)
        self.assertTrue(hierarchy.feedback_loop is None)
        self.assertTrue(hierarchy.experiential_manager is None)

    def test_create_carrier_initializes_phase3(self):
        """Test that create_carrier initializes feedback loop and manager."""
        hierarchy = MemoryHierarchy(width=64, height=64, depth=16)
        origin = Origin(width=64, height=64, use_gpu=False)

        carrier = hierarchy.create_carrier(origin)

        # Carrier should be initialized
        self.assertTrue(carrier is not None)
        self.assertTrue(hierarchy.proto_unity_carrier is not None)

        # Phase 3 components should be initialized
        self.assertTrue(hierarchy.feedback_loop is not None)
        self.assertTrue(hierarchy.experiential_manager is not None)

    def test_self_reflect_workflow(self):
        """Test full self-reflection workflow."""
        hierarchy = MemoryHierarchy(width=64, height=64, depth=16)
        origin = Origin(width=64, height=64, use_gpu=False)

        # Initialize carrier
        hierarchy.create_carrier(origin)

        # Add knowledge to core
        proto_core = np.random.randn(64, 64, 4).astype(np.float32)
        freq = np.random.randn(64, 64, 4).astype(np.float32)
        hierarchy.store_core(proto_core, freq, {'text': 'core knowledge'})

        # Create similar experiential thought
        proto_exp = proto_core + np.random.randn(64, 64, 4).astype(np.float32) * 0.01

        # Self-reflect
        coherence, state, recommendation = hierarchy.self_reflect(
            proto_exp,
            np.array([0.0, 0.0, 0.0, 1.0])
        )

        # Should be ALIGNED
        self.assertTrue(coherence > 0.8)
        self.assertTrue(state == SignalState.IDENTITY)
        self.assertTrue(recommendation == 'ALIGNED')

    def test_reset_workflow(self):
        """Test reset workflow."""
        hierarchy = MemoryHierarchy(width=64, height=64, depth=16)
        origin = Origin(width=64, height=64, use_gpu=False)

        # Initialize carrier
        hierarchy.create_carrier(origin)

        # Add entries to experiential
        for i in range(5):
            proto = np.random.randn(64, 64, 4).astype(np.float32)
            freq = np.random.randn(64, 64, 4).astype(np.float32)
            hierarchy.store_experiential(proto, freq, {'text': f'exp_{i}'})

        # Should have entries (or less due to collapse)
        initial_count = len(hierarchy.experiential_memory)

        # Reset
        hierarchy.reset_experiential()

        # Should be empty
        self.assertTrue(len(hierarchy.experiential_memory.entries) == 0)

    def test_consolidate_workflow(self):
        """Test consolidation workflow."""
        hierarchy = MemoryHierarchy(width=64, height=64, depth=16)
        origin = Origin(width=64, height=64, use_gpu=False)

        # Initialize carrier
        hierarchy.create_carrier(origin)

        # Add IDENTITY entries to experiential
        for i in range(3):
            proto = np.random.randn(64, 64, 4).astype(np.float32)
            freq = np.random.randn(64, 64, 4).astype(np.float32)
            hierarchy.store_experiential(proto, freq, {'text': f'identity_{i}'})
            hierarchy.experiential_memory.entries[-1].current_state = SignalState.IDENTITY
            hierarchy.experiential_memory.entries[-1].resonance_strength = 0.9

        # Add EVOLUTION entries (should not consolidate)
        for i in range(2):
            proto = np.random.randn(64, 64, 4).astype(np.float32)
            freq = np.random.randn(64, 64, 4).astype(np.float32)
            hierarchy.store_experiential(proto, freq, {'text': f'evolution_{i}'})
            hierarchy.experiential_memory.entries[-1].current_state = SignalState.EVOLUTION
            hierarchy.experiential_memory.entries[-1].resonance_strength = 0.9

        initial_core = len(hierarchy.core_memory)

        # Consolidate
        consolidated = hierarchy.consolidate(threshold=0.8)

        # Should consolidate IDENTITY entries only
        self.assertEqual(consolidated, 3)
        self.assertGreaterEqual(len(hierarchy.core_memory), initial_core + 1)  # may merge

    def test_raises_without_carrier(self):
        """Test methods raise error if carrier not initialized."""
        hierarchy = MemoryHierarchy(width=64, height=64, depth=16)

        proto = np.random.randn(64, 64, 4).astype(np.float32)
        quaternion = np.array([0.0, 0.0, 0.0, 1.0])

        # Should raise RuntimeError
        with self.assertRaises(RuntimeError):
            hierarchy.self_reflect(proto, quaternion)

        with self.assertRaises(RuntimeError):
            hierarchy.reset_experiential()

        with self.assertRaises(RuntimeError):
            hierarchy.consolidate()


if __name__ == '__main__':
    unittest.main()
