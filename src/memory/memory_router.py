"""Memory Router - Routes proto-identities to appropriate memory layers.

This component analyzes context and metadata to determine whether proto-identities
should be stored in core memory (long-term knowledge) or experiential memory
(short-term working memory).

Routing Logic:
    - Foundation/training texts → core memory
    - Query/inference inputs → experiential memory
    - Character/word level → both layers (configurable)
    - Phrase level → experiential only
    - User override via 'destination' parameter
"""

from typing import List, Dict, Optional, Literal, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass
class RoutingDecision:
    """Represents a routing decision for a proto-identity.

    Note: Does NOT store original text - only hash and proto-identity.
    """
    destination: Literal['core', 'experiential', 'both']
    reason: str
    metadata: Dict
    octave: int

class MemoryRouter:
    """Routes proto-identities to appropriate memory layers."""

    def __init__(self,
                 route_phrases_to_experiential: bool = True,
                 route_chars_to_both: bool = True):
        """Initialize the memory router.

        Args:
            route_phrases_to_experiential: If True, phrases go to experiential only
            route_chars_to_both: If True, chars/words go to both layers
        """
        self.route_phrases_to_experiential = route_phrases_to_experiential
        self.route_chars_to_both = route_chars_to_both
        self.routing_history: List[RoutingDecision] = []

    def route(self,
              octave_units: List,
              context_type: Optional[Literal['foundation', 'query', 'auto']] = 'auto',
              metadata: Optional[Dict] = None) -> List[RoutingDecision]:
        """Route octave units to appropriate memory layers.

        Args:
            octave_units: List of OctaveUnit objects
            context_type: Type of context (foundation/query/auto)
            metadata: Additional metadata for routing decisions

        Returns:
            List of RoutingDecision objects
        """
        if metadata is None:
            metadata = {}

        decisions = []

        # Auto-detect context if needed
        if context_type == 'auto':
            context_type = self._detect_context(metadata)

        for unit in octave_units:
            decision = self._route_unit(unit, context_type, metadata)
            decisions.append(decision)
            self.routing_history.append(decision)

        return decisions

    def _detect_context(self, metadata: Dict) -> Literal['foundation', 'query']:
        """Auto-detect context type from metadata.

        Args:
            metadata: Metadata dictionary

        Returns:
            'foundation' or 'query'
        """
        # Check for explicit markers
        if metadata.get('is_training', False):
            return 'foundation'
        if metadata.get('is_query', False):
            return 'query'
        if metadata.get('source') == 'training':
            return 'foundation'
        if metadata.get('source') == 'inference':
            return 'query'

        # Check for temporal markers (recent = query)
        if 'timestamp' in metadata:
            ts = metadata['timestamp']
            if isinstance(ts, datetime):
                age = (datetime.now() - ts).total_seconds()
                if age < 3600:  # Less than 1 hour old
                    return 'query'

        # Default to query for safety (experiential memory)
        return 'query'

    def _route_unit(self,
                    unit,
                    context_type: Literal['foundation', 'query'],
                    metadata: Dict) -> RoutingDecision:
        """Route a single octave unit.

        Args:
            unit: OctaveUnit object
            context_type: Context type
            metadata: Additional metadata

        Returns:
            RoutingDecision object
        """
        # Check for explicit destination override
        if 'destination' in metadata:
            dest = metadata['destination']
            if dest in ['core', 'experiential', 'both']:
                # Extract unit_hash from unit object
                                return RoutingDecision(
                    destination=dest,
                    reason='explicit_override',
                    metadata=metadata,
                    octave=unit.octave,
                                    )

        # Extract unit_hash from unit object (NO TEXT STORAGE)
        
        # Octave-based routing
        if unit.octave <= -2 and self.route_phrases_to_experiential:
            # Phrases go to experiential only
            return RoutingDecision(
                destination='experiential',
                reason='phrase_level_routing',
                metadata=metadata,
                octave=unit.octave,
                            )

        # Context-based routing
        if context_type == 'foundation':
            # Foundation texts go to core
            if unit.octave >= 0 and self.route_chars_to_both:
                # Character/word level can go to both
                return RoutingDecision(
                    destination='both',
                    reason='foundation_dual_storage',
                    metadata=metadata,
                    octave=unit.octave,
                                    )
            else:
                return RoutingDecision(
                    destination='core',
                    reason='foundation_text',
                    metadata=metadata,
                    octave=unit.octave,
                                    )
        else:  # query
            # Queries go to experiential
            return RoutingDecision(
                destination='experiential',
                reason='query_input',
                metadata=metadata,
                octave=unit.octave,
                            )

    def get_routing_stats(self) -> Dict:
        """Get statistics about routing decisions.

        Returns:
            Dictionary with routing statistics
        """
        if not self.routing_history:
            return {
                'total': 0,
                'core': 0,
                'experiential': 0,
                'both': 0,
                'by_octave': {},
                'by_reason': {}
            }

        stats = {
            'total': len(self.routing_history),
            'core': sum(1 for d in self.routing_history
                       if d.destination in ['core', 'both']),
            'experiential': sum(1 for d in self.routing_history
                              if d.destination in ['experiential', 'both']),
            'both': sum(1 for d in self.routing_history
                       if d.destination == 'both'),
            'by_octave': {},
            'by_reason': {}
        }

        # Count by octave
        for decision in self.routing_history:
            octave = decision.octave
            if octave not in stats['by_octave']:
                stats['by_octave'][octave] = 0
            stats['by_octave'][octave] += 1

            # Count by reason
            reason = decision.reason
            if reason not in stats['by_reason']:
                stats['by_reason'][reason] = 0
            stats['by_reason'][reason] += 1

        return stats

    def clear_history(self):
        """Clear routing history."""
        self.routing_history = []