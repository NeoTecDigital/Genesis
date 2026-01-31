"""
Real-Time I/O Handler - Interactive conversation interface.

Provides streaming responses and session state tracking for
real-time conversational interaction with Genesis.
"""

import numpy as np
from typing import Generator, Dict, Optional
import time

from src.pipeline.conversation import ConversationPipeline
from src.memory.state_classifier import SignalState


class RealTimeHandler:
    """Real-time handler for interactive conversation."""

    def __init__(self, conversation_pipeline: ConversationPipeline):
        """Initialize real-time handler.

        Args:
            conversation_pipeline: Conversation pipeline instance
        """
        self.pipeline = conversation_pipeline
        self.session_active = False
        self.auto_consolidate_threshold = 0.8
        self.conflict_count = 0
        self.max_conflicts_before_reset = 3

    def start_session(self) -> Dict:
        """Start interactive session.

        Returns:
            Session initialization metadata
        """
        metadata = self.pipeline.initialize_session()
        self.session_active = True
        self.conflict_count = 0

        return {
            'session_started': True,
            'timestamp': time.time(),
            **metadata
        }

    def handle_input(
        self,
        user_input: str,
        input_type: str = 'text'
    ) -> Dict:
        """Handle user input and return response.

        Args:
            user_input: User input text or file path
            input_type: 'text', 'image', or 'audio'

        Returns:
            Response dictionary with streaming metadata
        """
        if not self.session_active:
            raise RuntimeError("Session not started. Call start_session() first.")

        # Process input through pipeline
        response_data = self.pipeline.process_input(user_input, input_type)

        # Track conflicts
        if response_data['recommendation'] == 'CONFLICT':
            self.conflict_count += 1
        else:
            self.conflict_count = 0  # Reset on non-conflict

        # Check if reset needed
        reset_recommended = (
            self.conflict_count >= self.max_conflicts_before_reset
        )

        # Auto-consolidate if coherence high
        consolidated = 0
        if self._should_auto_consolidate(response_data):
            consolidated = self.pipeline.consolidate_session(
                self.auto_consolidate_threshold
            )

        return {
            **response_data,
            'session_state': self.get_session_state(),
            'reset_recommended': reset_recommended,
            'consolidated_patterns': consolidated
        }

    def stream_response(self, response_data: Dict) -> Generator[Dict, None, None]:
        """Stream response chunks with state updates.

        Args:
            response_data: Response from handle_input()

        Yields:
            Response chunks with metadata
        """
        response_text = response_data['response']

        # Chunk response by sentences
        chunks = self._chunk_by_sentences(response_text)

        for i, chunk in enumerate(chunks):
            yield {
                'chunk': chunk,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'coherence': response_data['coherence'],
                'state': response_data['state'],
                'is_final': i == len(chunks) - 1
            }

        # Final yield with full metadata
        yield {
            'chunk': '',
            'chunk_index': len(chunks),
            'total_chunks': len(chunks),
            'is_final': True,
            'metadata': response_data['metadata'],
            'session_state': response_data.get('session_state', {})
        }

    def get_session_state(self) -> Dict:
        """Get current session state.

        Returns:
            Session state dictionary
        """
        coherence_history = self.pipeline.coherence_history

        return {
            'session_active': self.session_active,
            'inputs_processed': self.pipeline.input_count,
            'session_duration': (
                time.time() - self.pipeline.session_start_time
            ),
            'avg_coherence': (
                np.mean(coherence_history) if coherence_history else 0.0
            ),
            'recent_coherence': (
                coherence_history[-5:] if coherence_history else []
            ),
            'conflict_count': self.conflict_count,
            'core_memory_size': len(self.pipeline.memory.core_memory),
            'experiential_memory_size': len(
                self.pipeline.memory.experiential_memory
            )
        }

    def end_session(self, consolidate: bool = True) -> Dict:
        """End session with optional consolidation.

        Args:
            consolidate: Whether to consolidate before ending

        Returns:
            Session summary
        """
        if not self.session_active:
            return {'session_ended': False, 'reason': 'No active session'}

        # Consolidate if requested
        consolidated = 0
        if consolidate:
            consolidated = self.pipeline.consolidate_session(
                self.auto_consolidate_threshold
            )

        # Get final state
        final_state = self.get_session_state()

        # Reset session
        self.session_active = False
        self.conflict_count = 0

        return {
            'session_ended': True,
            'consolidated_patterns': consolidated,
            'final_state': final_state,
            'timestamp': time.time()
        }

    def _should_auto_consolidate(self, response_data: Dict) -> bool:
        """Check if auto-consolidation should trigger.

        Auto-consolidates when:
        - Last 5 inputs have high coherence (>0.8)
        - Current state is IDENTITY (ALIGNED)

        Args:
            response_data: Response from process_input()

        Returns:
            True if should consolidate
        """
        coherence_history = self.pipeline.coherence_history

        # Need at least 5 inputs
        if len(coherence_history) < 5:
            return False

        # Check last 5 coherences
        recent_coherences = coherence_history[-5:]
        avg_recent = np.mean(recent_coherences)

        # Check if all recent coherences are high and state is IDENTITY
        return (
            avg_recent >= self.auto_consolidate_threshold
            and response_data['state'] == SignalState.IDENTITY
        )

    def _chunk_by_sentences(self, text: str) -> list:
        """Split text into sentence chunks.

        Args:
            text: Input text

        Returns:
            List of sentence chunks
        """
        if not text:
            return []

        # Simple sentence splitting by periods
        sentences = text.split('. ')

        # Add periods back and clean
        chunks = []
        for i, sent in enumerate(sentences):
            if i < len(sentences) - 1:
                chunks.append(sent + '.')
            else:
                chunks.append(sent)

        # Filter empty chunks
        chunks = [c.strip() for c in chunks if c.strip()]

        return chunks

    def __repr__(self) -> str:
        """String representation."""
        status = "active" if self.session_active else "inactive"
        return (
            f"RealTimeHandler(\n"
            f"  session_status={status},\n"
            f"  conflict_count={self.conflict_count},\n"
            f"  inputs_processed={self.pipeline.input_count if self.session_active else 0}\n"
            f")"
        )
