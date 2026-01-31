"""Unified Decoder - Single API for decoding across memory layers with blending.

This component provides a unified interface for:
1. Querying across core and experiential memory layers
2. Octave-aware retrieval and reconstruction
3. Blending results from multiple layers
4. Cross-octave querying (octave N ± 1)

The decoder ensures coherent reconstruction by respecting octave
relationships and blending memories appropriately.
"""

from typing import List, Dict, Optional, Literal, Tuple, Union
import numpy as np
from dataclasses import dataclass

from src.pipeline.multi_octave_decoder import MultiOctaveDecoder
from src.pipeline.hierarchical_synthesis import HierarchicalSynthesizer
from src.memory.memory_hierarchy import MemoryHierarchy
from src.memory.taylor_synthesizer import TaylorSynthesizer
from src.memory.synthesis_types import SynthesisResult


@dataclass
class DecodingResult:
    """Result of unified decoding operation."""
    text: str
    confidence: float
    source_layers: List[str]
    octaves_used: List[int]
    metadata: Dict
    resonances: Dict[str, float]  # layer -> resonance strength


class UnifiedDecoder:
    """Unified decoder for cross-layer querying and reconstruction."""

    def __init__(self,
                 memory_hierarchy: MemoryHierarchy,
                 width: int = 512,
                 height: int = 512,
                 core_weight: float = 0.7,
                 experiential_weight: float = 0.3):
        """Initialize unified decoder.

        Args:
            memory_hierarchy: MemoryHierarchy instance
            width: Proto-identity width
            height: Proto-identity height
            core_weight: Weight for core memory results
            experiential_weight: Weight for experiential results
        """
        self.memory_hierarchy = memory_hierarchy
        self.width = width
        self.height = height
        self.core_weight = core_weight
        self.experiential_weight = experiential_weight

        # Initialize multi-octave decoder
        # Create dummy carrier for compatibility (not actually used)
        carrier = np.zeros((height, width, 4), dtype=np.float32)
        self.multi_octave_decoder = MultiOctaveDecoder(carrier)

        # Initialize hierarchical synthesizer
        self.hierarchical_synthesizer = HierarchicalSynthesizer(
            memory_hierarchy,
            self.multi_octave_decoder.fft_decoder,
            coherence_threshold=0.75
        )

        # Statistics
        self.total_queries = 0
        self.successful_decodes = 0

    def decode(self,
               query_proto: np.ndarray,
               layers: Literal['core', 'experiential', 'both'] = 'both',
               octaves: List[int] = [4, 0],
               expand_octaves: bool = True,
               blend_mode: Literal['weighted', 'max', 'average'] = 'weighted',
               metadata: Optional[Dict] = None,
               use_taylor_synthesis: bool = False,
               query_text: Optional[str] = None) -> Union[DecodingResult, Tuple[List[np.ndarray], str]]:
        """Decode proto-identity across memory layers.

        Args:
            query_proto: Query proto-identity (H×W×4)
            layers: Which memory layers to query
            octaves: Primary octave levels to query
            expand_octaves: If True, also query octaves ± 1
            blend_mode: How to blend multi-layer results
            metadata: Additional metadata for query
            use_taylor_synthesis: If True, use Taylor series synthesis (iterative refinement)
            query_text: Query text for Taylor synthesis (required if use_taylor_synthesis=True)

        Returns:
            If use_taylor_synthesis=False: DecodingResult with reconstructed text and metadata
            If use_taylor_synthesis=True: Tuple of (proto_identities, explanation)
        """
        if metadata is None:
            metadata = {}

        self.total_queries += 1

        # Taylor synthesis path (iterative refinement)
        if use_taylor_synthesis:
            if query_text is None:
                raise ValueError("query_text required for Taylor synthesis")
            if not hasattr(self.memory_hierarchy, 'proto_unity_carrier'):
                raise ValueError("memory_hierarchy must have proto_unity_carrier for Taylor synthesis")

            # Initialize Taylor synthesizer
            synthesizer = TaylorSynthesizer()

            # Run iterative synthesis
            synthesis_result = synthesizer.synthesize(
                query_text,
                self.memory_hierarchy,
                self.memory_hierarchy.proto_unity_carrier
            )

            # Return multi-identity results
            return synthesis_result.proto_identities, synthesis_result.explanation

        # Legacy decoding path (direct FFT-based retrieval)
        # Expand octave range if requested
        query_octaves = self._expand_octave_range(octaves, expand_octaves)

        # Query each requested layer
        layer_results = {}
        resonances = {}

        if layers in ['core', 'both']:
            core_results = self._query_layer(
                'core', query_proto, query_octaves
            )
            if core_results:
                layer_results['core'] = core_results
                resonances['core'] = core_results['max_resonance']

        if layers in ['experiential', 'both']:
            exp_results = self._query_layer(
                'experiential', query_proto, query_octaves
            )
            if exp_results:
                layer_results['experiential'] = exp_results
                resonances['experiential'] = exp_results['max_resonance']

        # Blend results from multiple layers
        if not layer_results:
            return DecodingResult(
                text='',
                confidence=0.0,
                source_layers=[],
                octaves_used=[],
                metadata=metadata,
                resonances={}
            )

        blended_result = self._blend_results(
            layer_results, blend_mode, resonances
        )

        # Reconstruct text using multi-octave decoder
        reconstructed_text = self._reconstruct_text(
            blended_result, query_octaves
        )

        # Calculate confidence
        confidence = self._calculate_confidence(
            blended_result, resonances
        )

        if reconstructed_text:
            self.successful_decodes += 1

        # Merge metadata properly
        result_metadata = metadata.copy()
        blended_meta = blended_result.get('metadata', {})
        if isinstance(blended_meta, dict):
            result_metadata.update(blended_meta)

        return DecodingResult(
            text=reconstructed_text,
            confidence=confidence,
            source_layers=list(layer_results.keys()),
            octaves_used=query_octaves,
            metadata=result_metadata,
            resonances=resonances
        )

    def _expand_octave_range(self,
                            octaves: List[int],
                            expand: bool) -> List[int]:
        """Expand octave range to include neighboring octaves.

        Args:
            octaves: Primary octaves
            expand: Whether to expand

        Returns:
            Expanded octave list
        """
        if not expand:
            return octaves

        expanded = set(octaves)
        for octave in octaves:
            # Add octave ± 1
            expanded.add(octave - 1)
            expanded.add(octave + 1)

        # Filter to reasonable range
        return sorted([o for o in expanded if -6 <= o <= 6])

    def _query_layer(self,
                    layer: str,
                    query_proto: np.ndarray,
                    octaves: List[int]) -> Optional[Dict]:
        """Query a specific memory layer.

        Args:
            layer: 'core' or 'experiential'
            query_proto: Query proto-identity
            octaves: Octave levels to consider

        Returns:
            Query results or None
        """
        if layer == 'core':
            memory = self.memory_hierarchy.core_memory
        else:
            memory = self.memory_hierarchy.experiential_memory

        # Query the memory layer
        entries = memory.query_by_proto_similarity(query_proto, max_results=20)

        if not entries:
            return None

        # Convert entries to results dict format
        results = {
            'protos': [e.proto_identity for e in entries],
            'frequencies': [e.frequency for e in entries],
            'resonances': [getattr(e, 'resonance_strength', 1.0) for e in entries],
            'max_resonance': max([getattr(e, 'resonance_strength', 1.0) for e in entries]) if entries else 0,
            'metadata': [e.metadata for e in entries],
            'original_entries': entries  # Store original entries for FFT decoding
        }

        # Filter by octave if metadata available
        filtered_results = self._filter_by_octave(results, octaves)

        if not filtered_results:
            return results

        return filtered_results

    def _filter_by_octave(self,
                         results: Dict,
                         octaves: List[int]) -> Optional[Dict]:
        """Filter query results by octave level.

        Args:
            results: Query results from memory
            octaves: Target octave levels

        Returns:
            Filtered results or None
        """
        # Check if results have metadata with octave info
        if 'metadata' not in results or not results['metadata']:
            return results

        filtered_indices = []
        for i, meta in enumerate(results['metadata']):
            if isinstance(meta, dict) and 'octave' in meta:
                if meta['octave'] in octaves:
                    filtered_indices.append(i)
            else:
                # Include if no octave info
                filtered_indices.append(i)

        if not filtered_indices:
            return None

        # Filter all result arrays including original_entries
        filtered = {
            'protos': [results['protos'][i] for i in filtered_indices],
            'frequencies': [results['frequencies'][i]
                          for i in filtered_indices],
            'resonances': [results['resonances'][i]
                         for i in filtered_indices],
            'max_resonance': max([results['resonances'][i]
                                 for i in filtered_indices]),
            'metadata': [results['metadata'][i] for i in filtered_indices]
        }

        # Preserve original entries if present
        if 'original_entries' in results:
            filtered['original_entries'] = [results['original_entries'][i] for i in filtered_indices]

        return filtered

    def _blend_results(self,
                       layer_results: Dict,
                       blend_mode: str,
                       resonances: Dict) -> Dict:
        """Blend results from multiple layers.

        Args:
            layer_results: Results from each layer
            blend_mode: How to blend
            resonances: Resonance strengths

        Returns:
            Blended result
        """
        if len(layer_results) == 1:
            return list(layer_results.values())[0]

        if blend_mode == 'max':
            # Return results from layer with highest resonance
            best_layer = max(resonances.keys(),
                           key=lambda k: resonances[k])
            return layer_results[best_layer]

        elif blend_mode == 'average':
            # Simple average of proto-identities
            return self._average_blend(layer_results)

        else:  # weighted
            # Weighted blend based on configured weights
            return self._weighted_blend(layer_results)

    def _weighted_blend(self, layer_results: Dict) -> Dict:
        """Perform weighted blending of layer results.

        For FFT-based decoding, we combine original entries from all layers
        instead of blending the proto-identities (which destroys text).

        Args:
            layer_results: Results from each layer

        Returns:
            Weighted blend result with original_entries preserved
        """
        weights = {
            'core': self.core_weight,
            'experiential': self.experiential_weight
        }

        # Normalize weights for available layers
        available_weight_sum = sum(weights.get(layer, 0.5)
                                  for layer in layer_results.keys())
        normalized_weights = {
            layer: weights.get(layer, 0.5) / available_weight_sum
            for layer in layer_results.keys()
        }

        # Blend proto-identities (for compatibility, though not used for FFT decoding)
        blended_proto = None
        blended_freq = None
        blended_metadata = {}
        all_original_entries = []

        for layer, results in layer_results.items():
            weight = normalized_weights[layer]

            if results['protos']:
                proto = results['protos'][0]  # Use top result
                freq = results['frequencies'][0]

                if blended_proto is None:
                    blended_proto = proto * weight
                    blended_freq = freq * weight
                else:
                    blended_proto += proto * weight
                    blended_freq += freq * weight

                # Merge metadata
                if results.get('metadata'):
                    blended_metadata[f'{layer}_metadata'] = results['metadata'][0]

            # Collect all original entries from all layers
            if 'original_entries' in results:
                all_original_entries.extend(results['original_entries'])

        return {
            'protos': [blended_proto] if blended_proto is not None else [],
            'frequencies': [blended_freq] if blended_freq is not None else [],
            'metadata': blended_metadata,
            'original_entries': all_original_entries  # Preserve for FFT decoding
        }

    def _average_blend(self, layer_results: Dict) -> Dict:
        """Simple average blending of layer results.

        Args:
            layer_results: Results from each layer

        Returns:
            Averaged result
        """
        # Equal weights for all layers
        weights = {layer: 1.0 / len(layer_results)
                  for layer in layer_results.keys()}

        # Use weighted blend with equal weights
        self.core_weight = weights.get('core', 0.5)
        self.experiential_weight = weights.get('experiential', 0.5)
        result = self._weighted_blend(layer_results)

        # Restore original weights
        self.core_weight = 0.7
        self.experiential_weight = 0.3

        return result

    def _reconstruct_text(self,
                         blended_result: Dict,
                         octaves: List[int]) -> str:
        """Reconstruct text from blended results using FFT decoder.

        CRITICAL: For FFT decoding, we need the ORIGINAL retrieved entries,
        not the blended average proto. Blending averages the frequency patterns
        which produces garbage when decoded.

        Args:
            blended_result: Blended query results
            octaves: Octave levels used

        Returns:
            Reconstructed text
        """
        # For FFT-based decoding, check if we have original entries stored
        # (blended_result should contain the raw entries before blending)
        if 'original_entries' in blended_result:
            # Use original entries directly
            text = self.multi_octave_decoder.decode_to_summary(
                np.zeros((512, 512, 4)),  # Dummy query proto
                blended_result['original_entries']
            )
            return text if text != "[no context]" and text != "[silence]" else ''

        # Fallback: if no original entries, return empty
        # (blended protos cannot be reliably decoded with FFT)
        return ''

    def _calculate_confidence(self,
                            blended_result: Dict,
                            resonances: Dict) -> float:
        """Calculate decoding confidence.

        Args:
            blended_result: Blended results
            resonances: Layer resonances

        Returns:
            Confidence score [0, 1]
        """
        if not resonances:
            return 0.0

        # Base confidence on resonance strength
        max_resonance = max(resonances.values())

        # Penalize if only one layer contributed
        layer_penalty = 1.0 if len(resonances) > 1 else 0.8

        # Calculate final confidence
        confidence = min(1.0, max_resonance * layer_penalty)

        return confidence

    def get_statistics(self) -> Dict:
        """Get decoder statistics.

        Returns:
            Dictionary with decoding statistics
        """
        return {
            'total_queries': self.total_queries,
            'successful_decodes': self.successful_decodes,
            'success_rate': (self.successful_decodes / self.total_queries
                           if self.total_queries > 0 else 0),
            'blend_weights': {
                'core': self.core_weight,
                'experiential': self.experiential_weight
            }
        }

    def reset_statistics(self):
        """Reset decoder statistics."""
        self.total_queries = 0
        self.successful_decodes = 0

    def decode_to_summary(self,
                         query_proto: np.ndarray,
                         layers: Literal['core', 'experiential', 'both'] = 'both',
                         octaves: List[int] = [4, 0]) -> str:
        """Decode proto-identity to text summary (convenience method).

        Args:
            query_proto: Query proto-identity
            layers: Which memory layers to query
            octaves: Octave levels to query

        Returns:
            Synthesized text response
        """
        result = self.decode(
            query_proto,
            layers=layers,
            octaves=octaves,
            expand_octaves=True
        )
        return result.text

    def hierarchical_synthesis(
        self,
        query_proto: np.ndarray,
        layers: Literal['core', 'experiential', 'both'] = 'both',
        max_chars: int = 200,
        coherence_threshold: float = 0.75,
        octaves: List[int] = [4, 0, -2, -4]
    ) -> DecodingResult:
        """Synthesize text using hierarchical multi-octave assembly.

        This is the CORE of Genesis - meaning emerges from frequency harmonics!
        Characters resonate into words, words harmonize into phrases.

        Args:
            query_proto: Query proto-identity (H×W×4)
            layers: Which memory layers to query
            max_chars: Maximum response length
            coherence_threshold: Minimum coherence for assembly
            octaves: Octave levels to query

        Returns:
            DecodingResult with synthesized text and metadata
        """
        # Update coherence threshold if provided
        self.hierarchical_synthesizer.coherence_threshold = coherence_threshold

        # Delegate to synthesizer
        return self.hierarchical_synthesizer.synthesize(
            query_proto,
            layers=layers,
            max_chars=max_chars,
            octaves=octaves
        )