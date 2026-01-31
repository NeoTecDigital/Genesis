"""Unified Encoder - Single API for encoding with multi-octave and memory routing.

This component provides a unified interface that combines:
1. Multi-octave encoding (characters, words, phrases)
2. Memory routing (core vs experiential)
3. Storage in the appropriate memory layer

The unified encoder simplifies the API surface and ensures consistent
metadata handling across the entire encoding pipeline.
"""

from typing import List, Dict, Optional, Literal, Tuple
from datetime import datetime
import numpy as np
from dataclasses import dataclass

from src.pipeline.multi_octave_encoder import MultiOctaveEncoder, OctaveUnit
from src.memory.memory_router import MemoryRouter, RoutingDecision
from src.memory.memory_hierarchy import MemoryHierarchy


@dataclass
class EncodingResult:
    """Result of unified encoding operation."""
    octave_units: List[OctaveUnit]
    routing_decisions: List[RoutingDecision]
    core_added: int
    experiential_added: int
    metadata: Dict


class UnifiedEncoder:
    """Unified encoder combining multi-octave encoding and memory routing."""

    # Forbidden metadata fields that would violate architecture
    FORBIDDEN_METADATA = {'unit', 'text', 'content', 'raw', 'original'}

    def __init__(self,
                 memory_hierarchy: MemoryHierarchy,
                 carrier: Optional[np.ndarray] = None,
                 width: int = 512,
                 height: int = 512):
        """Initialize unified encoder.

        Args:
            memory_hierarchy: MemoryHierarchy instance for storage
            carrier: Carrier for multi-octave encoder (optional)
            width: Proto-identity width
            height: Proto-identity height
        """
        self.memory_hierarchy = memory_hierarchy
        self.width = width
        self.height = height

        # Initialize sub-components
        if carrier is None:
            carrier = np.zeros((height, width, 4), dtype=np.float32)
        self.multi_octave_encoder = MultiOctaveEncoder(
            carrier,
            width=width,
            height=height
        )
        self.memory_router = MemoryRouter()

        # Statistics tracking
        self.total_encoded = 0
        self.total_core_stored = 0
        self.total_experiential_stored = 0

    def encode(self,
               text: str,
               destination: Literal['auto', 'core', 'experiential', 'both'] = 'auto',
               octaves: List[int] = [4, 0],
               metadata: Optional[Dict] = None) -> EncodingResult:
        """Encode text with automatic routing and storage.

        Args:
            text: Input text to encode
            destination: Target memory layer(s)
            octaves: Octave levels to encode at
            metadata: Additional metadata

        Returns:
            EncodingResult with units, routing, and statistics
        """
        if metadata is None:
            metadata = {}

        # Add destination to metadata if explicit override requested
        if destination in ['core', 'experiential', 'both']:
            metadata['destination'] = destination

        # Add timestamp if not present
        if 'timestamp' not in metadata:
            metadata['timestamp'] = datetime.now()

        # Step 1: Multi-octave encoding
        octave_units = self.multi_octave_encoder.encode_text_hierarchical(
            text, octaves=octaves
        )

        # Step 2: Route to appropriate memory layers
        context_type = self._determine_context(destination, metadata)
        routing_decisions = self.memory_router.route(
            octave_units,
            context_type=context_type,
            metadata=metadata
        )

        # Step 3: Store in memory layers
        core_added = 0
        experiential_added = 0

        for unit, decision in zip(octave_units, routing_decisions):
            stored_metadata = self._create_storage_metadata(
                unit, decision, metadata
            )

            if decision.destination in ['core', 'both']:
                self._store_in_core(unit, stored_metadata)
                core_added += 1
                self.total_core_stored += 1

            if decision.destination in ['experiential', 'both']:
                self._store_in_experiential(unit, stored_metadata)
                experiential_added += 1
                self.total_experiential_stored += 1

        self.total_encoded += len(octave_units)

        return EncodingResult(
            octave_units=octave_units,
            routing_decisions=routing_decisions,
            core_added=core_added,
            experiential_added=experiential_added,
            metadata=metadata
        )

    def _determine_context(self,
                          destination: str,
                          metadata: Dict) -> Literal['foundation', 'query', 'auto']:
        """Determine context type from destination and metadata.

        Args:
            destination: User-specified destination
            metadata: Metadata dictionary

        Returns:
            Context type for routing
        """
        if destination == 'core':
            return 'foundation'
        elif destination == 'experiential':
            return 'query'
        elif destination == 'both':
            # Check metadata for hints
            if metadata.get('is_training', False):
                return 'foundation'
            return 'auto'
        else:  # auto
            return 'auto'

    def _validate_metadata(self, metadata: Dict) -> None:
        """Validate metadata to ensure no forbidden fields.

        Args:
            metadata: Metadata dictionary to validate

        Raises:
            ValueError: If forbidden fields are found
        """
        forbidden_found = self.FORBIDDEN_METADATA.intersection(metadata.keys())
        if forbidden_found:
            raise ValueError(f"Metadata contains forbidden fields: {forbidden_found}")

    def _create_storage_metadata(self,
                                 unit: OctaveUnit,
                                 decision: RoutingDecision,
                                 base_metadata: Dict) -> Dict:
        """Create metadata for storage.

        Args:
            unit: OctaveUnit being stored
            decision: RoutingDecision for this unit
            base_metadata: Base metadata from encode call

        Returns:
            Complete metadata dictionary
        """
        metadata = {
            **base_metadata,
            'octave': unit.octave,
            'destination': decision.destination,
            'routing_reason': decision.reason,
            'encoder': 'fft',  # Changed from 'unified' to 'fft'
            'width': self.width,
            'height': self.height
        }

        # Validate metadata
        self._validate_metadata(metadata)

        return metadata

    def _store_in_core(self, unit: OctaveUnit, metadata: Dict):
        """Store unit in core memory.

        Args:
            unit: OctaveUnit to store
            metadata: Storage metadata
        """
        self.memory_hierarchy.store_core(
            proto_identity=unit.proto_identity,
            frequency=unit.frequency,
            metadata=metadata
        )

    def _store_in_experiential(self, unit: OctaveUnit, metadata: Dict):
        """Store unit in experiential memory.

        Args:
            unit: OctaveUnit to store
            metadata: Storage metadata
        """
        self.memory_hierarchy.store_experiential(
            proto_identity=unit.proto_identity,
            frequency=unit.frequency,
            metadata=metadata
        )

    def get_statistics(self) -> Dict:
        """Get encoder statistics.

        Returns:
            Dictionary with encoding statistics
        """
        return {
            'total_encoded': self.total_encoded,
            'total_core_stored': self.total_core_stored,
            'total_experiential_stored': self.total_experiential_stored,
            'routing_stats': self.memory_router.get_routing_stats(),
            'efficiency': {
                'core_ratio': (self.total_core_stored / self.total_encoded
                              if self.total_encoded > 0 else 0),
                'experiential_ratio': (self.total_experiential_stored / self.total_encoded
                                      if self.total_encoded > 0 else 0)
            }
        }

    def reset_statistics(self):
        """Reset all statistics."""
        self.total_encoded = 0
        self.total_core_stored = 0
        self.total_experiential_stored = 0
        self.memory_router.clear_history()