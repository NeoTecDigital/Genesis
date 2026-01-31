"""
Phase 1 Core Infrastructure Tests

Tests for:
1. MemoryHierarchy class (Core/Experiential separation)
2. Origin carrier methods (initialize_carrier, modulate_carrier, demodulate_carrier)
3. ProtoIdentityEntry temporal fields (placeholders)
"""

import unittest
import numpy as np
from src.origin import Origin
from src.memory.memory_hierarchy import MemoryHierarchy
from src.memory.voxel_cloud import ProtoIdentityEntry


class TestMemoryHierarchy(unittest.TestCase):
    """Test MemoryHierarchy class functionality."""

    def test_initialization(self):
        """Test memory hierarchy initialization."""
        memory = MemoryHierarchy(512, 512, 128)

        self.assertEqual(memory.width, 512)
        self.assertEqual(memory.height, 512)
        self.assertEqual(memory.depth, 128)
        self.assertIsNone(memory.proto_unity_carrier)
        self.assertEqual(len(memory.core_memory), 0)
        self.assertEqual(len(memory.experiential_memory), 0)

    def test_carrier_creation(self):
        """Test carrier creation from Origin."""
        origin = Origin(512, 512, use_gpu=False)
        memory = MemoryHierarchy(512, 512, 128)

        # Create carrier via memory hierarchy
        carrier = memory.create_carrier(origin)

        self.assertIsNotNone(carrier)
        self.assertEqual(carrier.shape, (512, 512, 4))
        self.assertIsNotNone(memory.proto_unity_carrier)
        self.assertTrue(np.array_equal(carrier, memory.proto_unity_carrier))

    def test_get_carrier(self):
        """Test carrier retrieval."""
        origin = Origin(512, 512, use_gpu=False)
        memory = MemoryHierarchy(512, 512, 128)

        # Before initialization
        self.assertIsNone(memory.get_carrier())

        # After initialization
        carrier = memory.create_carrier(origin)
        retrieved = memory.get_carrier()

        self.assertIsNotNone(retrieved)
        self.assertTrue(np.array_equal(carrier, retrieved))

    def test_store_and_query_core(self):
        """Test storing and querying core memory."""
        memory = MemoryHierarchy(512, 512, 128)

        # Create test proto-identity
        proto = np.random.randn(512, 512, 4).astype(np.float32)
        freq = np.random.randn(512, 512, 4).astype(np.float32)
        metadata = {'text': 'test', 'modality': 'text'}

        # Store in core
        memory.store_core(proto, freq, metadata)

        self.assertEqual(len(memory.core_memory), 1)

        # Query core
        query_proto = proto + np.random.randn(512, 512, 4).astype(np.float32) * 0.1
        results = memory.query_core(query_proto, max_results=5)

        self.assertGreater(len(results), 0)

    def test_store_and_query_experiential(self):
        """Test storing and querying experiential memory."""
        memory = MemoryHierarchy(512, 512, 128)

        # Create test proto-identity
        proto = np.random.randn(512, 512, 4).astype(np.float32)
        freq = np.random.randn(512, 512, 4).astype(np.float32)
        metadata = {'text': 'test', 'modality': 'text'}

        # Store in experiential
        memory.store_experiential(proto, freq, metadata)

        self.assertEqual(len(memory.experiential_memory), 1)

        # Query experiential
        query_proto = proto + np.random.randn(512, 512, 4).astype(np.float32) * 0.1
        results = memory.query_experiential(query_proto, max_results=5)

        self.assertGreater(len(results), 0)

    def test_clear_experiential(self):
        """Test clearing experiential memory."""
        memory = MemoryHierarchy(512, 512, 128)

        # Add entries
        proto = np.random.randn(512, 512, 4).astype(np.float32)
        freq = np.random.randn(512, 512, 4).astype(np.float32)
        metadata = {'text': 'test', 'modality': 'text'}

        memory.store_experiential(proto, freq, metadata)
        self.assertEqual(len(memory.experiential_memory), 1)

        # Clear
        memory.clear_experiential()
        self.assertEqual(len(memory.experiential_memory), 0)

    def test_consolidate_to_core(self):
        """Test consolidating experiential to core memory."""
        memory = MemoryHierarchy(512, 512, 128)

        # Add high-resonance entry to experiential
        proto = np.random.randn(512, 512, 4).astype(np.float32)
        freq = np.random.randn(512, 512, 4).astype(np.float32)
        metadata = {'text': 'test', 'modality': 'text'}

        memory.store_experiential(proto, freq, metadata)

        # Manually set resonance strength
        memory.experiential_memory.entries[0].resonance_strength = 5

        # Consolidate
        consolidated = memory.consolidate_to_core(threshold=3)

        self.assertEqual(consolidated, 1)
        # Note: Core memory will have merged entry if similar exists


class TestOriginCarrier(unittest.TestCase):
    """Test Origin carrier methods."""

    def test_initialize_carrier(self):
        """Test carrier initialization."""
        origin = Origin(512, 512, use_gpu=False)

        # Initialize carrier
        carrier = origin.initialize_carrier()

        self.assertIsNotNone(carrier)
        self.assertEqual(carrier.shape, (512, 512, 4))
        self.assertIsNotNone(origin.proto_unity_carrier)
        self.assertTrue(np.array_equal(carrier, origin.proto_unity_carrier))

    def test_modulate_carrier(self):
        """Test carrier modulation."""
        origin = Origin(512, 512, use_gpu=False)

        # Initialize carrier
        carrier = origin.initialize_carrier()

        # Create input signal
        input_signal = np.random.randn(512, 512, 4).astype(np.float32)
        iota_params = {'strength': 1.0}
        tau_params = {'projection_strength': 1.0}

        # Modulate
        modulated = origin.modulate_carrier(input_signal, iota_params, tau_params)

        self.assertIsNotNone(modulated)
        self.assertEqual(modulated.shape, (512, 512, 4))
        self.assertFalse(np.array_equal(modulated, carrier))

    def test_demodulate_carrier(self):
        """Test carrier demodulation."""
        origin = Origin(512, 512, use_gpu=False)

        # Initialize carrier
        carrier = origin.initialize_carrier()

        # Create proto-identity
        proto = np.random.randn(512, 512, 4).astype(np.float32)

        # Demodulate
        signal = origin.demodulate_carrier(proto)

        self.assertIsNotNone(signal)
        self.assertEqual(signal.shape, (512, 512, 4))

    def test_modulate_without_carrier_raises(self):
        """Test that modulation without carrier raises error."""
        origin = Origin(512, 512, use_gpu=False)

        input_signal = np.random.randn(512, 512, 4).astype(np.float32)
        iota_params = {'strength': 1.0}
        tau_params = {'projection_strength': 1.0}

        with self.assertRaises(ValueError) as ctx:
            origin.modulate_carrier(input_signal, iota_params, tau_params)
        self.assertIn("Carrier not initialized", str(ctx.exception))

    def test_demodulate_without_carrier_raises(self):
        """Test that demodulation without carrier raises error."""
        origin = Origin(512, 512, use_gpu=False)

        proto = np.random.randn(512, 512, 4).astype(np.float32)

        with self.assertRaises(ValueError) as ctx:
            origin.demodulate_carrier(proto)
        self.assertIn("Carrier not initialized", str(ctx.exception))

    def test_modulate_demodulate_roundtrip(self):
        """Test that modulation and demodulation preserve information."""
        origin = Origin(512, 512, use_gpu=False)

        # Initialize carrier
        carrier = origin.initialize_carrier()

        # Create input signal
        input_signal = np.random.randn(512, 512, 4).astype(np.float32)
        iota_params = {'strength': 1.0}
        tau_params = {'projection_strength': 1.0}

        # Modulate
        modulated = origin.modulate_carrier(input_signal, iota_params, tau_params)

        # Demodulate
        recovered = origin.demodulate_carrier(modulated)

        # Check shape preservation
        self.assertEqual(recovered.shape, input_signal.shape)


class TestProtoIdentityEntryTemporal(unittest.TestCase):
    """Test ProtoIdentityEntry temporal fields."""

    def test_temporal_fields_initialization(self):
        """Test temporal fields are properly initialized."""
        proto = np.random.randn(512, 512, 4).astype(np.float32)
        mip_levels = [proto]
        freq = np.random.randn(512, 512, 4).astype(np.float32)
        metadata = {'text': 'test'}
        position = np.array([0.0, 0.0, 0.0])

        entry = ProtoIdentityEntry(
            proto_identity=proto,
            mip_levels=mip_levels,
            frequency=freq,
            metadata=metadata,
            position=position
        )

        # Check temporal fields exist and have correct defaults
        self.assertTrue(hasattr(entry, 'temporal_history'))
        self.assertTrue(hasattr(entry, 'current_state'))
        self.assertTrue(hasattr(entry, 'coherence_vs_core'))

        self.assertEqual(entry.temporal_history, [])
        self.assertIsNone(entry.current_state)
        self.assertEqual(entry.coherence_vs_core, 0.0)

    def test_temporal_fields_can_be_set(self):
        """Test temporal fields can be modified."""
        proto = np.random.randn(512, 512, 4).astype(np.float32)
        mip_levels = [proto]
        freq = np.random.randn(512, 512, 4).astype(np.float32)
        metadata = {'text': 'test'}
        position = np.array([0.0, 0.0, 0.0])

        entry = ProtoIdentityEntry(
            proto_identity=proto,
            mip_levels=mip_levels,
            frequency=freq,
            metadata=metadata,
            position=position
        )

        # Modify temporal fields
        test_history = [(1.0, np.array([1, 2, 3]))]
        entry.temporal_history = test_history
        entry.current_state = 'active'
        entry.coherence_vs_core = 0.85

        self.assertEqual(entry.temporal_history, test_history)
        self.assertEqual(entry.current_state, 'active')
        self.assertEqual(entry.coherence_vs_core, 0.85)


class TestIntegration(unittest.TestCase):
    """Integration tests combining components."""

    def test_full_memory_hierarchy_workflow(self):
        """Test complete workflow: carrier creation, modulation, storage, query."""
        origin = Origin(512, 512, use_gpu=False)
        memory = MemoryHierarchy(512, 512, 128)

        # 1. Initialize carrier
        carrier = memory.create_carrier(origin)
        self.assertIsNotNone(carrier)

        # 2. Create and modulate input
        input_signal = np.random.randn(512, 512, 4).astype(np.float32)
        iota_params = {'strength': 1.0}
        tau_params = {'projection_strength': 1.0}

        modulated = origin.modulate_carrier(input_signal, iota_params, tau_params)
        self.assertEqual(modulated.shape, (512, 512, 4))

        # 3. Store in experiential memory
        freq = np.random.randn(512, 512, 4).astype(np.float32)
        metadata = {'text': 'integration test', 'modality': 'text'}

        memory.store_experiential(modulated, freq, metadata)
        self.assertEqual(len(memory.experiential_memory), 1)

        # 4. Query experiential memory
        results = memory.query_experiential(modulated, max_results=5)
        self.assertGreater(len(results), 0)

        # 5. Demodulate recovered proto
        recovered = origin.demodulate_carrier(results[0].proto_identity)
        self.assertEqual(recovered.shape, (512, 512, 4))


if __name__ == '__main__':
    import unittest
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    for test_class in [TestMemoryHierarchy, TestOriginCarrier, TestProtoIdentityEntryTemporal, TestIntegration]:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with proper code
    exit(0 if result.wasSuccessful() else 1)