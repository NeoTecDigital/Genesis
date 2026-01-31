"""
Memory Hierarchy - Three-layer memory system for Genesis.

Architecture:
- Proto-unity carrier (stable reference, once per session)
- Core memory (long-term learned knowledge)
- Experiential memory (short-term working memory)

The carrier is created from Origin convergence (○ → γ ∪ ε) and modulated
with inputs via FM modulation (ι/τ paths).
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Literal
from src.memory.voxel_cloud import VoxelCloud
from src.memory.feedback_loop import FeedbackLoop
from src.memory.experiential_manager import ExperientialMemoryManager
from src.memory.state_classifier import SignalState
from src.memory.memory_router import MemoryRouter, RoutingDecision
from src.memory.coherence_router import CoherenceRouter


class MemoryHierarchy:
    """
    Three-layer memory system:
    1. Proto-unity carrier (stable session reference)
    2. Core memory (long-term learned patterns)
    3. Experiential memory (short-term working buffer)
    """

    def __init__(self, width: int = 1024, height: int = 1024, depth: int = 128,
                 collapse_config: Optional[dict] = None,
                 use_routing: bool = True,
                 use_coherence_routing: bool = False):
        """
        Initialize memory hierarchy.

        Args:
            width: Spatial width for proto-identities
            height: Spatial height for proto-identities
            depth: Voxel cloud depth dimension
            collapse_config: Gravitational collapse configuration
            use_routing: Enable automatic memory routing (default: True)
            use_coherence_routing: Enable coherence-based routing (default: False)
        """
        self.width = width
        self.height = height
        self.depth = depth
        self.use_routing = use_routing

        # Proto-unity carrier
        self.proto_unity_carrier: Optional[np.ndarray] = None

        # Default collapse config
        if collapse_config is None:
            collapse_config = {
                'harmonic_tolerance': 0.05,
                'cosine_threshold': 0.85,
                'octave_tolerance': 0,
                'enable': False
            }

        # Memory layers
        self.core_memory = VoxelCloud(width, height, depth, collapse_config=collapse_config)
        self.experiential_memory = VoxelCloud(width, height, depth, collapse_config=collapse_config)

        # Phase 3 components
        self.feedback_loop: Optional[FeedbackLoop] = None
        self.experiential_manager: Optional[ExperientialMemoryManager] = None

        # Routing components
        self.memory_router = MemoryRouter(
            route_phrases_to_experiential=True,
            route_chars_to_both=True
        ) if use_routing else None

        self.coherence_router = CoherenceRouter(
            width, height
        ) if use_coherence_routing else None

    def create_carrier(self, origin) -> np.ndarray:
        """
        Create stable proto-unity carrier from Origin (once per session).
        ○ → γ ∪ ε → proto-unity (convergence)
        """
        carrier = origin.initialize_carrier()
        self.proto_unity_carrier = carrier

        # Initialize Phase 3 components
        self.feedback_loop = FeedbackLoop(self.core_memory, self.experiential_memory)
        self.experiential_manager = ExperientialMemoryManager(
            self.proto_unity_carrier,
            self.experiential_memory
        )

        return carrier

    def get_carrier(self) -> Optional[np.ndarray]:
        """Get current proto-unity carrier."""
        return self.proto_unity_carrier

    def store_core(self, proto_identity: np.ndarray, frequency: np.ndarray,
                   metadata: Dict) -> None:
        """Store proto-identity in core (long-term) memory."""
        self.core_memory.add(proto_identity, frequency, metadata)

    def store_experiential(self, proto_identity: np.ndarray,
                          frequency: np.ndarray, metadata: Dict) -> None:
        """Store proto-identity in experiential (short-term) memory."""
        self.experiential_memory.add(proto_identity, frequency, metadata)

    def query_core(self, query_proto: np.ndarray, max_results: int = 10):
        """Query core memory by proto-identity similarity."""
        return self.core_memory.query_by_proto_similarity(query_proto, max_results)

    def query_experiential(self, query_proto: np.ndarray, max_results: int = 10):
        """Query experiential memory by proto-identity similarity."""
        return self.experiential_memory.query_by_proto_similarity(query_proto, max_results)

    def clear_experiential(self) -> None:
        """Clear experiential (short-term) memory."""
        self.experiential_memory = VoxelCloud(
            self.width, self.height, self.depth
        )

    def consolidate_to_core(self, threshold: float = 0.7) -> int:
        """Consolidate high-resonance experiential to core (DEPRECATED: use consolidate())."""
        consolidated = 0
        for entry in self.experiential_memory.entries:
            if entry.resonance_strength >= threshold:
                self.core_memory.add(entry.proto_identity, entry.frequency, entry.metadata)
                consolidated += 1
        return consolidated

    def self_reflect(self, experiential_proto: np.ndarray,
                    query_quaternion: np.ndarray) -> Tuple[float, SignalState, str]:
        """Compare experiential thought against core knowledge (Phase 3)."""
        if self.feedback_loop is None:
            raise RuntimeError("Feedback loop not initialized. Call create_carrier() first.")
        return self.feedback_loop.self_reflect(experiential_proto, query_quaternion)

    def reset_experiential(self) -> None:
        """Reset experiential memory to proto-unity baseline (Phase 3)."""
        if self.experiential_manager is None:
            raise RuntimeError("Experiential manager not initialized. Call create_carrier() first.")
        self.experiential_manager.reset_to_baseline()

    def consolidate(self, threshold: float = 0.8) -> int:
        """Consolidate Identity-state patterns to core memory (Phase 3)."""
        if self.experiential_manager is None:
            raise RuntimeError("Experiential manager not initialized. Call create_carrier() first.")
        return self.experiential_manager.consolidate_to_core(self.core_memory, threshold)

    def query_with_projection(
        self,
        projection_matrix,
        frequency_band: Optional[int] = None,
        max_results: int = 10
    ):
        """Query memory with projection matrix and optional frequency band.

        Combines raycasting visibility culling with frequency band filtering.

        Args:
            projection_matrix: ProjectionMatrix instance for raycasting
            frequency_band: Optional FrequencyBand enum for filtering
            max_results: Maximum results to return

        Returns:
            List of ProtoIdentityEntry objects
        """
        # Get all entries from both core and experiential
        all_entries = list(self.core_memory.entries) + list(
            self.experiential_memory.entries
        )

        # Apply raycasting filter
        visible = []
        for entry in all_entries:
            # Extract voxel position from proto-identity
            voxel_pos = self._extract_voxel_position(entry.proto_identity)

            # Check visibility
            if projection_matrix.is_voxel_visible(voxel_pos):
                visible.append(entry)

        # Apply frequency band filter if specified
        if frequency_band is not None:
            from src.memory.frequency_bands import FrequencyBandClustering
            band_clustering = FrequencyBandClustering(num_bands=3)

            filtered = []
            for entry in visible:
                entry_band = band_clustering.assign_band(entry.frequency)
                if entry_band == frequency_band:
                    filtered.append(entry)

            visible = filtered

        # Sort by resonance and limit
        visible.sort(key=lambda x: x.resonance_strength, reverse=True)

        return visible[:max_results]

    def _extract_voxel_position(self, proto: np.ndarray) -> np.ndarray:
        """Extract 3D position from proto-identity.

        Args:
            proto: Proto-identity (H, W, 4)

        Returns:
            Position (3,) array
        """
        z_channel = proto[:, :, 2]
        h, w = z_channel.shape

        y_coords = np.arange(h).reshape(-1, 1)
        x_coords = np.arange(w).reshape(1, -1)

        total_mass = z_channel.sum()
        if total_mass > 1e-8:
            center_y = (z_channel * y_coords).sum() / total_mass
            center_x = (z_channel * x_coords).sum() / total_mass
        else:
            center_y, center_x = h / 2, w / 2

        center_z = z_channel.mean() * 100.0

        return np.array([center_x, center_y, center_z], dtype=np.float32)

    def store_with_coherence(
        self,
        proto_identity: np.ndarray,
        frequency: np.ndarray,
        metadata: Dict,
        measure_coherence: bool = True
    ) -> Dict:
        """Store proto-identity with coherence-based routing."""
        if not measure_coherence:
            self.store_experiential(proto_identity, frequency, metadata)
            return {'layer': 'experiential', 'coherence': None, 'state': None,
                    'reason': 'Coherence check disabled'}

        if self.coherence_router is None:
            self.coherence_router = CoherenceRouter(self.width, self.height)

        decision = self.coherence_router.route_by_coherence(
            proto_identity, frequency, metadata
        )

        return self._execute_storage_decision(
            proto_identity, frequency, metadata, decision
        )

    def _execute_storage_decision(self, proto, freq, meta, decision):
        """Execute storage based on coherence decision."""
        layer_map = {'core': 'proto_unity', 'experiential': 'experiential',
                     'rejected': 'rejected'}

        if decision.destination == 'core':
            self.store_core(proto, freq, meta)
        elif decision.destination == 'experiential':
            self.store_experiential(proto, freq, meta)

        return {
            'layer': layer_map.get(decision.destination, 'rejected'),
            'coherence': decision.coherence,
            'state': decision.state.name,
            'reason': decision.reason
        }

    def add_to_memory(self,
                     proto_identities: List[np.ndarray],
                     frequencies: List[np.ndarray],
                     octave_units: List,
                     context_type: Optional[Literal['foundation', 'query', 'auto']] = 'auto',
                     base_metadata: Optional[Dict] = None,
                     use_coherence_routing: bool = False) -> Dict[str, int]:
        """Add proto-identities to memory with intelligent routing."""
        if use_coherence_routing:
            return self._add_with_coherence(
                proto_identities, frequencies, octave_units, base_metadata
            )

        if not self.use_routing or self.memory_router is None:
            return self._add_legacy(proto_identities, frequencies, base_metadata)

        return self._add_with_router(
            proto_identities, frequencies, octave_units, context_type, base_metadata
        )

    def _add_with_coherence(self, protos, freqs, units, base_meta):
        """Add with coherence-based routing."""
        counts = {'core': 0, 'experiential': 0, 'rejected': 0}
        for i, (proto, freq) in enumerate(zip(protos, freqs)):
            metadata = base_meta.copy() if base_meta else {}
            if i < len(units):
                unit = units[i]
                octave = unit[0] if isinstance(unit, tuple) else getattr(unit, 'octave', None)
                metadata['octave'] = octave

            result = self.store_with_coherence(proto, freq, metadata)
            counts[result['layer']] += 1
        return counts

    def _add_legacy(self, protos, freqs, base_meta):
        """Legacy fallback - add all to experiential."""
        for proto, freq in zip(protos, freqs):
            metadata = base_meta.copy() if base_meta else {}
            self.store_experiential(proto, freq, metadata)
        return {'core': 0, 'experiential': len(protos)}

    def _add_with_router(self, protos, freqs, units, context, base_meta):
        """Add with MemoryRouter routing."""
        processed_units = self._normalize_units(units)
        decisions = self.memory_router.route(processed_units, context, base_meta)

        counts = {'core': 0, 'experiential': 0}
        for i, decision in enumerate(decisions):
            if i >= len(protos):
                break
            self._store_by_decision(protos[i], freqs[i], decision, base_meta, counts)
        return counts

    def _normalize_units(self, units):
        """Convert tuples to OctaveUnit-like objects."""
        from dataclasses import dataclass

        @dataclass
        class SimpleOctaveUnit:
            octave: int
            unit: str

        return [
            SimpleOctaveUnit(u[0], u[1]) if isinstance(u, tuple) else u
            for u in units
        ]

    def _store_by_decision(self, proto, freq, decision, base_meta, counts):
        """Store proto based on routing decision."""
        metadata = base_meta.copy() if base_meta else {}
        metadata.update(decision.metadata)
        metadata['octave'] = decision.octave
        metadata['routing_reason'] = decision.reason

        if decision.destination in ('core', 'both'):
            self.store_core(proto, freq, metadata)
            counts['core'] += 1
        if decision.destination in ('experiential', 'both'):
            self.store_experiential(proto, freq, metadata)
            counts['experiential'] += 1

    def get_session_stats(self) -> Dict:
        """Get memory session statistics.

        Returns:
            Dictionary with memory statistics
        """
        stats = {
            'core_memory': {
                'entries': len(self.core_memory),
                'avg_resonance': 0.0,
                'total_size': self.core_memory.entries[0].proto_identity.nbytes * len(self.core_memory) if len(self.core_memory) > 0 else 0
            },
            'experiential_memory': {
                'entries': len(self.experiential_memory),
                'avg_resonance': 0.0,
                'total_size': self.experiential_memory.entries[0].proto_identity.nbytes * len(self.experiential_memory) if len(self.experiential_memory) > 0 else 0
            }
        }

        # Compute average resonances
        if len(self.core_memory) > 0:
            stats['core_memory']['avg_resonance'] = np.mean([
                e.resonance_strength for e in self.core_memory.entries
            ])

        if len(self.experiential_memory) > 0:
            stats['experiential_memory']['avg_resonance'] = np.mean([
                e.resonance_strength for e in self.experiential_memory.entries
            ])

        return stats

    def auto_consolidate(
        self,
        coherence_window: int = 5,
        threshold: float = 0.8
    ) -> int:
        """Auto-consolidate based on coherence window.

        Consolidates if average coherence over window exceeds threshold.

        Args:
            coherence_window: Number of recent coherences to check
            threshold: Minimum average coherence for consolidation

        Returns:
            Number of patterns consolidated (0 if conditions not met)
        """
        if self.feedback_loop is None:
            return 0

        # Check if we have enough history
        # This would need coherence tracking in the memory hierarchy
        # For now, delegate to standard consolidate
        return self.consolidate(threshold)

    def __repr__(self) -> str:
        carrier_status = "initialized" if self.proto_unity_carrier is not None else "uninitialized"
        return (
            f"MemoryHierarchy(\n"
            f"  carrier: {carrier_status},\n"
            f"  core: {len(self.core_memory)} entries,\n"
            f"  experiential: {len(self.experiential_memory)} entries\n"
            f")"
        )
