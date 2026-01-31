"""
Integration Tests - End-to-end testing for Phase 6.

Tests full conversation pipeline integrating all phases:
- ConversationPipeline (end-to-end workflow)
- RealTimeHandler (interactive conversation)
- Full multi-turn conversations
- Multimodal inputs
- Temporal prediction
- Auto-consolidation
- Conflict recovery
"""

import pytest
import numpy as np
import time
from pathlib import Path

from src.memory.memory_hierarchy import MemoryHierarchy
from src.memory.state_classifier import SignalState
from src.memory.frequency_bands import FrequencyBand
from src.origin import Origin
from src.pipeline.conversation import ConversationPipeline
from src.pipeline.realtime_io import RealTimeHandler


@pytest.fixture
def origin():
    """Create Origin instance."""
    return Origin(width=64, height=64, use_gpu=False)


@pytest.fixture
def memory_hierarchy():
    """Create MemoryHierarchy instance."""
    return MemoryHierarchy(width=64, height=64, depth=32)


@pytest.fixture
def conversation_pipeline(memory_hierarchy, origin):
    """Create ConversationPipeline instance."""
    return ConversationPipeline(memory_hierarchy, origin)


@pytest.fixture
def realtime_handler(conversation_pipeline):
    """Create RealTimeHandler instance."""
    return RealTimeHandler(conversation_pipeline)


# ============================================================================
# ConversationPipeline Tests
# ============================================================================

class TestConversationPipeline:
    """Test ConversationPipeline end-to-end workflow."""

    def test_init(self, conversation_pipeline):
        """Test pipeline initialization."""
        assert conversation_pipeline.carrier is not None
        assert conversation_pipeline.encoder is not None
        assert conversation_pipeline.decoder is not None
        assert conversation_pipeline.projection is not None
        assert conversation_pipeline.band_clustering is not None
        assert conversation_pipeline.input_count == 0

    def test_initialize_session(self, conversation_pipeline):
        """Test session initialization."""
        metadata = conversation_pipeline.initialize_session()

        assert metadata['carrier_created'] is True
        assert metadata['memory_initialized'] is True
        assert 'core_entries' in metadata
        assert 'experiential_entries' in metadata
        assert conversation_pipeline.input_count == 0

    def test_process_input_text(self, conversation_pipeline):
        """Test full text processing workflow."""
        response = conversation_pipeline.process_input(
            "The Art of War teaches strategy",
            input_type='text'
        )

        # Check response structure
        assert 'response' in response
        assert 'coherence' in response
        assert 'state' in response
        assert 'recommendation' in response
        assert 'context_protos' in response
        assert 'metadata' in response

        # Check types
        assert isinstance(response['response'], str)
        assert 0.0 <= response['coherence'] <= 1.0
        assert response['state'] in [
            SignalState.IDENTITY,
            SignalState.EVOLUTION,
            SignalState.PARADOX
        ]
        assert response['recommendation'] in [
            'ALIGNED', 'LEARNING', 'CONFLICT'
        ]

    def test_process_input_image(self, conversation_pipeline, tmp_path):
        """Test full image processing workflow."""
        # Create dummy image
        image_path = tmp_path / "test.png"
        from PIL import Image
        img = Image.new('RGB', (64, 64), color='red')
        img.save(image_path)

        response = conversation_pipeline.process_input(
            str(image_path),
            input_type='image'
        )

        assert 'response' in response
        assert response['metadata']['input_type'] == 'image'

    def test_query_with_context_raycast(self, conversation_pipeline):
        """Test context retrieval with raycasting."""
        # Add some data to memory
        test_proto = np.random.randn(64, 64, 4).astype(np.float32)
        test_freq = np.random.randn(64, 64, 2).astype(np.float32)
        conversation_pipeline.memory.store_core(
            test_proto, test_freq, {'text': 'test data'}
        )

        # Query with raycasting
        results = conversation_pipeline.query_with_context(
            test_proto,
            use_raycast=True,
            max_results=5
        )

        assert isinstance(results, list)
        assert len(results) <= 5

    def test_query_with_context_frequency_band(self, conversation_pipeline):
        """Test context retrieval with frequency band filtering."""
        # Add data to memory
        test_proto = np.random.randn(64, 64, 4).astype(np.float32)
        test_freq = np.random.randn(64, 64, 2).astype(np.float32)
        conversation_pipeline.memory.store_core(
            test_proto, test_freq, {'text': 'test data'}
        )

        # Query with band filter
        results = conversation_pipeline.query_with_context(
            test_proto,
            use_raycast=False,
            frequency_band=FrequencyBand.MID,
            max_results=5
        )

        assert isinstance(results, list)

    def test_synthesize_response(self, conversation_pipeline):
        """Test response generation."""
        # Add context to memory
        test_proto = np.random.randn(64, 64, 4).astype(np.float32)
        test_freq = np.random.randn(64, 64, 2).astype(np.float32)
        conversation_pipeline.memory.store_core(
            test_proto, test_freq, {'text': 'The way is in training'}
        )

        # Get context
        context = conversation_pipeline.memory.query_core(test_proto, 5)

        # Synthesize response
        response = conversation_pipeline.synthesize_response(
            test_proto, context
        )

        assert isinstance(response, str)
        assert len(response) > 0

    def test_consolidate_session(self, conversation_pipeline):
        """Test session consolidation."""
        # Add experiential data
        test_proto = np.random.randn(64, 64, 4).astype(np.float32)
        test_freq = np.random.randn(64, 64, 2).astype(np.float32)
        conversation_pipeline.memory.store_experiential(
            test_proto, test_freq, {'text': 'high resonance pattern'}
        )

        # Consolidate
        count = conversation_pipeline.consolidate_session(threshold=0.7)

        assert isinstance(count, int)
        assert count >= 0

    def test_reset_session(self, conversation_pipeline):
        """Test session reset."""
        # Process some inputs
        conversation_pipeline.process_input("test input 1")
        conversation_pipeline.process_input("test input 2")

        assert conversation_pipeline.input_count > 0
        assert len(conversation_pipeline.coherence_history) > 0

        # Reset
        conversation_pipeline.reset_session()

        assert conversation_pipeline.input_count == 0
        assert len(conversation_pipeline.coherence_history) == 0


# ============================================================================
# RealTimeHandler Tests
# ============================================================================

class TestRealTimeHandler:
    """Test RealTimeHandler for interactive conversation."""

    def test_init(self, realtime_handler):
        """Test handler initialization."""
        assert realtime_handler.pipeline is not None
        assert realtime_handler.session_active is False
        assert realtime_handler.conflict_count == 0

    def test_start_session(self, realtime_handler):
        """Test session start."""
        metadata = realtime_handler.start_session()

        assert metadata['session_started'] is True
        assert 'timestamp' in metadata
        assert realtime_handler.session_active is True
        assert realtime_handler.conflict_count == 0

    def test_handle_input_aligned(self, realtime_handler):
        """Test handling input with ALIGNED state."""
        realtime_handler.start_session()

        # Add core knowledge to get ALIGNED state
        test_proto = np.random.randn(64, 64, 4).astype(np.float32)
        test_freq = np.random.randn(64, 64, 2).astype(np.float32)
        realtime_handler.pipeline.memory.store_core(
            test_proto, test_freq, {'text': 'known pattern'}
        )

        response = realtime_handler.handle_input("test input")

        assert 'response' in response
        assert 'session_state' in response
        assert 'reset_recommended' in response
        assert 'consolidated_patterns' in response

    def test_handle_input_learning(self, realtime_handler):
        """Test handling input with LEARNING state."""
        realtime_handler.start_session()

        response = realtime_handler.handle_input("new learning input")

        assert 'response' in response
        assert response['reset_recommended'] is False

    def test_handle_input_conflict(self, realtime_handler):
        """Test handling input with CONFLICT state."""
        realtime_handler.start_session()

        # Trigger multiple conflicts
        for _ in range(3):
            realtime_handler.handle_input("conflicting input")

        # Check if reset recommended
        response = realtime_handler.handle_input("another conflict")

        # Reset may be recommended after threshold
        assert 'reset_recommended' in response

    def test_stream_response(self, realtime_handler):
        """Test response streaming."""
        realtime_handler.start_session()

        response_data = realtime_handler.handle_input(
            "Generate a longer response with multiple sentences"
        )

        # Stream response
        chunks = list(realtime_handler.stream_response(response_data))

        assert len(chunks) > 0

        # Check chunk structure
        for chunk in chunks[:-1]:  # All except final
            assert 'chunk' in chunk
            assert 'chunk_index' in chunk
            assert 'total_chunks' in chunk
            assert 'coherence' in chunk
            assert 'state' in chunk

        # Check final chunk
        final_chunk = chunks[-1]
        assert final_chunk['is_final'] is True
        assert 'metadata' in final_chunk

    def test_get_session_state(self, realtime_handler):
        """Test session state retrieval."""
        realtime_handler.start_session()

        # Process some inputs
        realtime_handler.handle_input("input 1")
        realtime_handler.handle_input("input 2")

        state = realtime_handler.get_session_state()

        assert state['session_active'] is True
        assert state['inputs_processed'] == 2
        assert 'session_duration' in state
        assert 'avg_coherence' in state
        assert 'conflict_count' in state

    def test_end_session_consolidate(self, realtime_handler):
        """Test session end with consolidation."""
        realtime_handler.start_session()

        # Process inputs
        realtime_handler.handle_input("input 1")

        summary = realtime_handler.end_session(consolidate=True)

        assert summary['session_ended'] is True
        assert 'consolidated_patterns' in summary
        assert 'final_state' in summary
        assert realtime_handler.session_active is False


# ============================================================================
# End-to-End Workflow Tests
# ============================================================================

class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    def test_full_conversation_flow(self, realtime_handler):
        """Test multi-turn conversation."""
        realtime_handler.start_session()

        # Turn 1
        response1 = realtime_handler.handle_input(
            "What is strategy?"
        )
        assert 'response' in response1

        # Turn 2
        response2 = realtime_handler.handle_input(
            "Tell me more about tactics"
        )
        assert 'response' in response2

        # Turn 3
        response3 = realtime_handler.handle_input(
            "How does this relate to leadership?"
        )
        assert 'response' in response3

        # Check session state
        state = realtime_handler.get_session_state()
        assert state['inputs_processed'] == 3

    def test_multimodal_conversation(
        self,
        realtime_handler,
        tmp_path
    ):
        """Test conversation with text and image inputs."""
        realtime_handler.start_session()

        # Text input
        response1 = realtime_handler.handle_input(
            "This is a text input",
            input_type='text'
        )
        assert response1['metadata']['input_type'] == 'text'

        # Image input
        image_path = tmp_path / "test.png"
        from PIL import Image
        img = Image.new('RGB', (64, 64), color='blue')
        img.save(image_path)

        response2 = realtime_handler.handle_input(
            str(image_path),
            input_type='image'
        )
        assert response2['metadata']['input_type'] == 'image'

        # Check both stored
        state = realtime_handler.get_session_state()
        assert state['inputs_processed'] == 2

    def test_temporal_prediction_accuracy(self, conversation_pipeline):
        """Test temporal prediction over conversation."""
        # Process sequence of inputs
        inputs = [
            "First thought",
            "Second thought",
            "Third thought",
            "Fourth thought"
        ]

        predictions = []
        for text in inputs:
            response = conversation_pipeline.process_input(text)

            if response['temporal_prediction'] is not None:
                predictions.append(response['temporal_prediction'])

        # Should have predictions after first few inputs
        assert len(predictions) > 0

    def test_automatic_consolidation(self, realtime_handler):
        """Test auto-consolidation after threshold."""
        realtime_handler.start_session()

        # Add core knowledge
        test_proto = np.random.randn(64, 64, 4).astype(np.float32)
        test_freq = np.random.randn(64, 64, 2).astype(np.float32)
        realtime_handler.pipeline.memory.store_core(
            test_proto, test_freq, {'text': 'base knowledge'}
        )

        # Process multiple aligned inputs
        for i in range(6):
            response = realtime_handler.handle_input(f"aligned input {i}")

            # Check for consolidation
            if response['consolidated_patterns'] > 0:
                assert response['state'] == SignalState.IDENTITY

    def test_conflict_recovery(self, realtime_handler):
        """Test recovery from CONFLICT state."""
        realtime_handler.start_session()

        # Add core knowledge
        test_proto = np.random.randn(64, 64, 4).astype(np.float32)
        test_freq = np.random.randn(64, 64, 2).astype(np.float32)
        realtime_handler.pipeline.memory.store_core(
            test_proto, test_freq, {'text': 'established knowledge'}
        )

        # Trigger conflicts
        conflict_count = 0
        for i in range(5):
            response = realtime_handler.handle_input(
                f"conflicting statement {i}"
            )

            if response['recommendation'] == 'CONFLICT':
                conflict_count += 1

        # Check reset recommended after threshold
        if conflict_count >= realtime_handler.max_conflicts_before_reset:
            assert response['reset_recommended'] is True

        # Reset and recover
        realtime_handler.pipeline.reset_session()
        realtime_handler.conflict_count = 0  # Reset handler conflict count

        # Process aligned input
        recovery = realtime_handler.handle_input("aligned recovery input")

        # Conflict count should be reset or very low
        assert realtime_handler.conflict_count <= 1

    def test_memory_persistence(
        self,
        conversation_pipeline,
        tmp_path
    ):
        """Test memory state persistence across sessions."""
        # Add initial core knowledge
        test_proto = np.random.randn(64, 64, 4).astype(np.float32)
        test_freq = np.random.randn(64, 64, 2).astype(np.float32)
        conversation_pipeline.memory.store_core(
            test_proto, test_freq, {'text': 'persistent core knowledge'}
        )

        core_count_initial = len(conversation_pipeline.memory.core_memory)
        assert core_count_initial > 0

        # Session 1: Process inputs and consolidate
        conversation_pipeline.process_input("persistent knowledge 1")
        conversation_pipeline.process_input("persistent knowledge 2")

        exp_count_before_reset = len(
            conversation_pipeline.memory.experiential_memory
        )
        assert exp_count_before_reset > 0

        # Reset experiential
        conversation_pipeline.reset_session()

        # Experiential should be clear
        assert len(
            conversation_pipeline.memory.experiential_memory
        ) == 0

        # Core should persist
        assert len(conversation_pipeline.memory.core_memory) >= core_count_initial


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
