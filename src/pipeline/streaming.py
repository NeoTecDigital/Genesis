"""
Streaming Synthesis - Incremental text/image processing with temporal prediction.

StreamingSynthesizer processes input streams chunk-by-chunk, tracking state transitions
and using Taylor series prediction for next proto-identity.
"""

import numpy as np
import time
from typing import Iterator, Dict, Optional, List
from pathlib import Path

from src.memory.memory_hierarchy import MemoryHierarchy
from src.memory.temporal_buffer import TemporalBuffer
from src.memory.state_classifier import StateClassifier, SignalState
from src.memory.frequency_field import TextFrequencyAnalyzer
from src.memory.fm_modulation_base import FMModulationBase


class StreamingSynthesizer:
    """Unified streaming synthesis for text/audio/video with temporal prediction."""

    def __init__(self, memory_hierarchy: MemoryHierarchy):
        """Initialize streaming synthesizer.

        Args:
            memory_hierarchy: Memory system with carrier, core, and experiential
        """
        self.memory = memory_hierarchy
        self.temporal_buffer = TemporalBuffer(max_length=100)
        self.state_classifier = StateClassifier()
        self.text_analyzer = TextFrequencyAnalyzer(
            width=memory_hierarchy.width,
            height=memory_hierarchy.height
        )
        self.modulator = FMModulationBase()

        # Ensure carrier is initialized
        if self.memory.get_carrier() is None:
            raise RuntimeError(
                "Memory hierarchy carrier not initialized. "
                "Call memory.create_carrier() first."
            )

    def process_text_stream(
        self,
        text_chunks: Iterator[str],
        output_path: Optional[str] = None
    ) -> Iterator[Dict]:
        """Process streaming text input incrementally.

        Args:
            text_chunks: Iterator yielding words/sentences
            output_path: Optional path to save results (in ./output/)

        Yields:
            {
                'chunk': current chunk,
                'proto': proto-identity,
                'state': SignalState,
                'prediction': next proto (Taylor series),
                'coherence': coherence vs core,
                'timestamp': timestamp
            }
        """
        results = []
        carrier = self.memory.get_carrier()

        for chunk in text_chunks:
            timestamp = time.time()

            # 1. Text → Frequency spectrum (dual-path)
            freq_spectrum, _ = self.text_analyzer.text_to_frequency(chunk)

            # 2. Frequency → Signal (XY channels from spectrum)
            signal = self._frequency_to_signal(freq_spectrum)

            # 3. Modulate carrier
            proto_identity = self.modulator.modulate(carrier, signal)

            # 4. Add to temporal buffer
            self.temporal_buffer.add(proto_identity, timestamp)

            # 5. Classify state
            coherence = self._compute_coherence(proto_identity)
            state = self.state_classifier.classify(
                self.temporal_buffer,
                coherence
            )

            # 6. Feedback loop (if not PARADOX)
            if state != SignalState.PARADOX and len(self.temporal_buffer) >= 2:
                self._apply_feedback(proto_identity, coherence, state)

            # 7. Taylor prediction
            prediction = self.temporal_buffer.predict_next(
                delta_t=0.1,
                order=2
            )

            result = {
                'chunk': chunk,
                'proto': proto_identity,
                'state': state,
                'prediction': prediction,
                'coherence': coherence,
                'timestamp': timestamp
            }
            results.append(result)
            yield result

        # Save results if output path provided
        if output_path and results:
            self._save_results(results, output_path)

    def synthesize_response(
        self,
        query: str,
        output_path: Optional[str] = None
    ) -> str:
        """Synthesize response to query using streaming.

        Uses viewport query to find matches, then synthesizes weighted response.

        Args:
            query: Query string
            output_path: Optional path to save response

        Returns:
            Synthesized response text
        """
        # Process query through streaming to get proto-identity
        query_chunks = self._text_to_chunks(query)
        query_proto = None
        for result in self.process_text_stream(query_chunks):
            # Use final proto as query
            if result['state'] == SignalState.IDENTITY:
                query_proto = result['proto']
                break

        if query_proto is None:
            # Fall back to last proto
            query_chunks = self._text_to_chunks(query)
            for result in self.process_text_stream(query_chunks):
                query_proto = result['proto']

        # Query both core and experiential memory
        core_matches = self.memory.query_core(query_proto, max_results=5)
        exp_matches = self.memory.query_experiential(query_proto, max_results=5)

        # Synthesize weighted blend
        response_text = self._synthesize_from_matches(
            core_matches,
            exp_matches,
            query_proto
        )

        # Save if output path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(response_text)

        return response_text

    def _frequency_to_signal(self, freq_spectrum: np.ndarray) -> np.ndarray:
        """Convert frequency spectrum to signal (XYZW format).

        Args:
            freq_spectrum: (H, W, 2) frequency spectrum [magnitude, phase]

        Returns:
            (H, W, 4) signal [X, Y, Z, W]
        """
        magnitude = freq_spectrum[:, :, 0]
        phase = freq_spectrum[:, :, 1]

        # Convert to complex representation
        X = magnitude * np.cos(phase)  # Real part
        Y = magnitude * np.sin(phase)  # Imaginary part
        Z = magnitude  # Weight/amplitude
        W = np.gradient(phase)[0]  # Phase velocity (simplified)

        return np.stack([X, Y, Z, W], axis=-1).astype(np.float32)

    def _compute_coherence(self, proto_identity: np.ndarray) -> float:
        """Compute coherence with carrier.

        Args:
            proto_identity: (H, W, 4) proto-identity

        Returns:
            Coherence score [0, 1]
        """
        carrier = self.memory.get_carrier()
        return self.modulator.detect_resonance(proto_identity, carrier)

    def _apply_feedback(
        self,
        proto_identity: np.ndarray,
        coherence: float,
        state: SignalState
    ) -> None:
        """Apply feedback loop based on state.

        Args:
            proto_identity: Current proto-identity
            coherence: Coherence score
            state: Current signal state
        """
        # Store in experiential memory
        freq_spectrum = self._extract_frequency(proto_identity)
        metadata = {
            'state': state.name,
            'coherence': coherence,
            'timestamp': time.time()
        }
        self.memory.store_experiential(proto_identity, freq_spectrum, metadata)

        # Consolidate to core if IDENTITY state with high coherence
        if state == SignalState.IDENTITY and coherence >= 0.8:
            self.memory.consolidate(threshold=0.8)

    def _extract_frequency(self, proto_identity: np.ndarray) -> np.ndarray:
        """Extract frequency spectrum from proto-identity.

        Args:
            proto_identity: (H, W, 4) proto-identity

        Returns:
            (H, W, 2) frequency spectrum
        """
        # Extract XY channels as complex representation
        X = proto_identity[:, :, 0]
        Y = proto_identity[:, :, 1]

        magnitude = np.sqrt(X**2 + Y**2)
        phase = np.arctan2(Y, X)

        return np.stack([magnitude, phase], axis=-1).astype(np.float32)

    def _text_to_chunks(self, text: str, chunk_size: int = 10) -> Iterator[str]:
        """Split text into chunks (words).

        Args:
            text: Input text
            chunk_size: Number of words per chunk

        Yields:
            Text chunks
        """
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            yield chunk

    def _synthesize_from_matches(
        self,
        core_matches: List,
        exp_matches: List,
        query_proto: np.ndarray
    ) -> str:
        """Synthesize response from memory matches.

        Args:
            core_matches: Core memory matches
            exp_matches: Experiential memory matches
            query_proto: Query proto-identity

        Returns:
            Synthesized response text
        """
        # Collect all text from matches
        core_texts = []
        exp_texts = []

        # TODO Phase 11: Derive text from frequency spectrum
        # Text is no longer stored in metadata - will be derived from signal
        # For now, skip text extraction

        # Original implementation preserved for Phase 11 adaptation:
        # for entry in core_matches:
        #     derived_text = self._derive_text_from_spectrum(entry.proto_identity)
        #     core_texts.append(derived_text)
        # for entry in exp_matches:
        #     derived_text = self._derive_text_from_spectrum(entry.proto_identity)
        #     exp_texts.append(derived_text)

        # Weight: 70% core, 30% experiential
        if core_texts or exp_texts:
            core_portion = ' '.join(core_texts[:2]) if core_texts else ''
            exp_portion = ' '.join(exp_texts[:1]) if exp_texts else ''

            if core_portion and exp_portion:
                response = f"{core_portion} {exp_portion}"
            else:
                response = core_portion or exp_portion
        else:
            response = "No matching memories found."

        return response.strip()

    def _save_results(self, results: List[Dict], output_path: str) -> None:
        """Save results to file.

        Args:
            results: List of result dictionaries
            output_path: Path to save results
        """
        import json

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = []
        for result in results:
            serializable = {
                'chunk': result['chunk'],
                'state': result['state'].name,
                'coherence': float(result['coherence']),
                'timestamp': result['timestamp']
            }
            serializable_results.append(serializable)

        with output_file.open('w') as f:
            json.dump(serializable_results, f, indent=2)
