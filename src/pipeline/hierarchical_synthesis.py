"""Hierarchical Multi-Octave Synthesis.

Synthesizes text via hierarchical assembly from frequency harmonics:
- Characters resonate into words
- Words harmonize into phrases
- Meaning emerges from multi-octave frequency patterns

Key Innovation: Resonance weighting enables intelligent assembly
- weight = similarity × log(resonance + 1)
- Higher resonance = stronger pattern frequency = more reliable
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Literal
from dataclasses import dataclass

from src.memory.memory_hierarchy import MemoryHierarchy
from src.pipeline.fft_text_decoder import FFTTextDecoder


@dataclass
class DecodingResult:
    """Result of hierarchical synthesis operation."""
    text: str
    confidence: float
    source_layers: List[str]
    octaves_used: List[int]
    metadata: Dict
    resonances: Dict[str, float]


class HierarchicalSynthesizer:
    """Synthesizes text via hierarchical multi-octave assembly."""

    def __init__(
        self,
        memory_hierarchy: MemoryHierarchy,
        fft_decoder: FFTTextDecoder,
        coherence_threshold: float = 0.75
    ):
        """Initialize with memory hierarchy and FFT decoder.

        Args:
            memory_hierarchy: MemoryHierarchy instance
            fft_decoder: FFT text decoder for proto-identity decoding
            coherence_threshold: Minimum coherence for assembly
        """
        self.memory_hierarchy = memory_hierarchy
        self.fft_decoder = fft_decoder
        self.coherence_threshold = coherence_threshold

    def synthesize(
        self,
        query_proto: np.ndarray,
        layers: Literal['core', 'experiential', 'both'] = 'both',
        max_chars: int = 200,
        octaves: List[int] = [4, 0, -2, -4]
    ) -> DecodingResult:
        """Main synthesis entry point.

        Args:
            query_proto: Query proto-identity (H×W×4)
            layers: Which memory layers to query
            max_chars: Maximum response length
            octaves: Octave levels to query

        Returns:
            DecodingResult with synthesized text
        """
        # 1. Retrieve at all octaves
        octave_results = self._retrieve_all_octaves(
            query_proto, layers, octaves
        )

        # 2. Assemble characters to words
        assembled_words = self._assemble_characters_to_words(octave_results)

        # 3. Assemble words to phrases
        assembled_phrases = self._assemble_words_to_phrases(
            assembled_words, octave_results
        )

        # 4. Synthesize final text
        final_text, confidence = self._synthesize_final_text(
            assembled_phrases, assembled_words, max_chars
        )

        return DecodingResult(
            text=final_text,
            confidence=confidence,
            source_layers=self._get_source_layers(layers),
            octaves_used=octaves,
            metadata={'method': 'hierarchical'},
            resonances=self._calculate_resonances(octave_results)
        )

    def _retrieve_all_octaves(
        self,
        query_proto: np.ndarray,
        layers: str,
        octaves: List[int]
    ) -> Dict[int, List[Tuple]]:
        """Query all octaves with resonance weighting.

        Args:
            query_proto: Query proto-identity
            layers: Which layers to query
            octaves: Octave levels

        Returns:
            Dict mapping octave → [(entry, similarity, weight), ...]
        """
        results = {}

        for octave in octaves:
            octave_entries = []

            # Query core memory if requested
            if layers in ['core', 'both']:
                core_results = self._query_layer_octave(
                    self.memory_hierarchy.core_memory,
                    query_proto,
                    octave,
                    top_k=100 if octave == 4 else 50
                )
                octave_entries.extend(core_results)

            # Query experiential memory if requested
            if layers in ['experiential', 'both']:
                exp_results = self._query_layer_octave(
                    self.memory_hierarchy.experiential_memory,
                    query_proto,
                    octave,
                    top_k=100 if octave == 4 else 50
                )
                octave_entries.extend(exp_results)

            # Weight by resonance
            weighted = []
            for entry, similarity in octave_entries:
                resonance = entry.metadata.get('resonance', 1)
                weight = similarity * np.log(resonance + 1)
                weighted.append((entry, similarity, weight))

            # Sort by weight
            results[octave] = sorted(
                weighted, key=lambda x: x[2], reverse=True
            )

        return results

    def _query_layer_octave(
        self,
        memory_layer,
        query_proto: np.ndarray,
        octave: int,
        top_k: int
    ) -> List[Tuple]:
        """Query a memory layer at specific octave.

        Args:
            memory_layer: VoxelCloud to query
            query_proto: Query proto-identity
            octave: Target octave level
            top_k: Number of results

        Returns:
            List of (entry, similarity) tuples
        """
        # Query with octave filter
        entries = memory_layer.query_by_proto_similarity(
            query_proto,
            max_results=top_k
        )

        # Filter by octave
        filtered = [
            (e, self._compute_similarity(query_proto, e.proto_identity))
            for e in entries
            if e.metadata.get('octave') == octave
        ]

        return filtered

    def _compute_similarity(
        self,
        proto1: np.ndarray,
        proto2: np.ndarray
    ) -> float:
        """Compute similarity between two proto-identities.

        Args:
            proto1: First proto-identity
            proto2: Second proto-identity

        Returns:
            Similarity score [0, 1]
        """
        from src.memory.voxel_cloud_clustering import compute_proto_similarity
        return compute_proto_similarity(proto1, proto2)

    def _assemble_characters_to_words(
        self,
        octave_results: Dict[int, List[Tuple]]
    ) -> List[Tuple[str, float, float]]:
        """Use word-level patterns to guide character grouping.

        Args:
            octave_results: Results at each octave

        Returns:
            List of (word, weight, coherence) tuples
        """
        if 4 not in octave_results or 0 not in octave_results:
            return []

        if not octave_results[4] or not octave_results[0]:
            return []

        # Decode characters
        char_results = octave_results[4][:100]
        chars = []
        for entry, sim, weight in char_results:
            decoded = self.fft_decoder.decode_text(entry.proto_identity)
            if decoded and len(decoded) > 0:
                chars.append((decoded, sim, weight))

        # Decode words
        word_results = octave_results[0][:50]
        words = []
        for entry, sim, weight in word_results:
            decoded = self.fft_decoder.decode_text(entry.proto_identity)
            if decoded and len(decoded) > 0:
                words.append((decoded, sim, weight))

        # Assembly: Match character sequences to word patterns
        assembled = []
        for word, word_sim, word_weight in words:
            coherence = self._calculate_char_word_coherence(word, chars)
            if coherence >= self.coherence_threshold:
                total_weight = word_weight * coherence
                assembled.append((word, total_weight, coherence))

        return sorted(assembled, key=lambda x: x[1], reverse=True)

    def _calculate_char_word_coherence(
        self,
        word: str,
        chars: List[Tuple[str, float, float]]
    ) -> float:
        """Score how well character results support this word.

        Args:
            word: Target word
            chars: Character results

        Returns:
            Coherence score [0, 1]
        """
        if not word or not chars:
            return 0.0

        # Count matching characters
        word_chars = list(word.lower())
        char_set = {c.lower() for c, _, _ in chars}

        matches = sum(1 for wc in word_chars if wc in char_set)
        return matches / len(word_chars) if word_chars else 0.0

    def _assemble_words_to_phrases(
        self,
        assembled_words: List[Tuple[str, float, float]],
        octave_results: Dict[int, List[Tuple]]
    ) -> List[Tuple[str, float, float]]:
        """Use phrase-level patterns to guide word ordering.

        Args:
            assembled_words: Word assembly results
            octave_results: Results at each octave

        Returns:
            List of (phrase, weight, coherence) tuples
        """
        if not assembled_words:
            return []

        if -2 not in octave_results or not octave_results[-2]:
            # No phrase level - return words as-is
            return [(w, wt, 1.0) for w, wt, _ in assembled_words]

        # Decode phrases
        phrase_results = octave_results[-2][:30]
        phrases = []
        for entry, sim, weight in phrase_results:
            decoded = self.fft_decoder.decode_text(entry.proto_identity)
            if decoded and len(decoded) > 0:
                phrases.append((decoded, sim, weight))

        # Assembly: Match word sequences to phrase patterns
        assembled = []
        for phrase, phrase_sim, phrase_weight in phrases:
            coherence = self._calculate_word_phrase_coherence(
                phrase, assembled_words
            )
            if coherence >= self.coherence_threshold:
                total_weight = phrase_weight * coherence
                assembled.append((phrase, total_weight, coherence))

        return sorted(assembled, key=lambda x: x[1], reverse=True)

    def _calculate_word_phrase_coherence(
        self,
        phrase: str,
        assembled_words: List[Tuple[str, float, float]]
    ) -> float:
        """Score how well word results support this phrase.

        Args:
            phrase: Target phrase
            assembled_words: Word assembly results

        Returns:
            Coherence score [0, 1]
        """
        if not phrase or not assembled_words:
            return 0.0

        # Split phrase into words
        phrase_words = phrase.lower().split()
        word_set = {w.lower() for w, _, _ in assembled_words}

        matches = sum(1 for pw in phrase_words if pw in word_set)
        return matches / len(phrase_words) if phrase_words else 0.0

    def _synthesize_final_text(
        self,
        assembled_phrases: List[Tuple[str, float, float]],
        assembled_words: List[Tuple[str, float, float]],
        max_chars: int
    ) -> Tuple[str, float]:
        """Combine phrases and words into final coherent text.

        Args:
            assembled_phrases: Phrase assembly results
            assembled_words: Word assembly results
            max_chars: Maximum character count

        Returns:
            (final_text, confidence)
        """
        result_parts = []
        total_confidence = 0.0
        char_count = 0

        # Prefer complete phrases first
        for phrase, weight, coherence in assembled_phrases:
            if char_count + len(phrase) <= max_chars:
                result_parts.append(phrase)
                char_count += len(phrase) + 1  # +1 for space
                total_confidence += weight

        # Fill with individual words
        for word, weight, coherence in assembled_words:
            if char_count >= max_chars:
                break
            # Avoid duplicates
            if not any(word.lower() in part.lower() for part in result_parts):
                if char_count + len(word) <= max_chars:
                    result_parts.append(word)
                    char_count += len(word) + 1
                    total_confidence += weight

        final_text = ' '.join(result_parts)
        confidence = total_confidence / len(result_parts) if result_parts else 0.0

        return final_text, min(confidence, 1.0)

    def _get_source_layers(self, layers: str) -> List[str]:
        """Get list of source layers.

        Args:
            layers: Layer specification

        Returns:
            List of layer names
        """
        if layers == 'both':
            return ['core', 'experiential']
        return [layers]

    def _calculate_resonances(
        self,
        octave_results: Dict[int, List[Tuple]]
    ) -> Dict[str, float]:
        """Calculate per-octave resonance strengths.

        Args:
            octave_results: Results at each octave

        Returns:
            Dict mapping octave → resonance strength
        """
        resonances = {}
        for octave, results in octave_results.items():
            if results:
                avg_resonance = np.mean([
                    entry.metadata.get('resonance', 1)
                    for entry, _, _ in results[:10]
                ])
                resonances[f'octave_{octave}'] = float(avg_resonance)
        return resonances
