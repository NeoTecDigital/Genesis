"""
Voxel Cloud Memory - 3D spatial structure where proto-identities live.

Proto-identities are stored in a voxel cloud with MIP levels for scale.
Viewport-based retrieval allows synthesis of new proto-identities.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import pickle
from pathlib import Path

from .voxel_helpers import (
    compute_frequency_position, box_filter_downsample, generate_mip_levels,
    compute_cosine_similarity, resize_proto, compute_resonance_weights,
    compute_distance_weights, check_frequency_match
)
from .octave_hierarchy import OctaveHierarchy, OctaveProtoIdentity
from .temporal_buffer import TemporalBuffer
from .state_classifier import StateClassifier, SignalState
from . import voxel_cloud_collapse as collapse_ops
from . import voxel_cloud_query as query_ops


@dataclass
class ProtoIdentityEntry:
    """Entry in the voxel cloud."""
    proto_identity: np.ndarray  # The actual proto-identity (H, W, 4)
    mip_levels: List[np.ndarray]  # MIP pyramid for scale
    frequency: np.ndarray  # Original frequency spectrum
    metadata: Dict  # Text, params, etc
    position: np.ndarray  # 3D position in voxel cloud
    # Frequency-based matching fields
    fundamental_freq: float = 0.0  # f₀
    octave: int = 0  # Octave level (0=base, 1=2x, -1=0.5x, etc)
    harmonic_signature: Optional[np.ndarray] = None  # 10 harmonic coefficients
    resonance_strength: int = 1  # Number of times this pattern has appeared
    # Cross-modal fields
    modality: str = 'text'  # 'text', 'image', 'audio'
    cross_modal_links: List[str] = field(default_factory=list)  # IDs of linked protos
    # Temporal tracking (Phase 2)
    temporal_history: List[Tuple[float, np.ndarray]] = field(default_factory=list)
    current_state: Optional[SignalState] = None  # SignalState enum
    coherence_vs_core: float = 0.0
    # Phase 5: Frequency band classification
    frequency_band: Optional[int] = None  # 0=LOW, 1=MID, 2=HIGH
    # WaveCube integration
    wavecube_coords: Optional[Tuple[int, int, int, float]] = None  # (X, Y, Z, W) reference


class VoxelCloud:
    """
    Voxel cloud memory structure for proto-identities.

    Proto-identities are positioned in 3D space based on their frequency characteristics.
    MIP levels provide multi-scale representation.
    """

    def __init__(self, width: int = 512, height: int = 512, depth: int = 128,
                 collapse_config: Optional[Dict] = None, synthesis_config: Optional[Dict] = None):
        self.width = width
        self.height = height
        self.depth = depth
        self.entries: List[ProtoIdentityEntry] = []
        self.spatial_index = {}  # Grid-based spatial index
        self.frequency_index: Dict[int, List[int]] = {}  # freq_bin → [entry_ids]
        self.octave_hierarchy = OctaveHierarchy(num_octaves=12)  # 12-octave hierarchy
        # Phase 2: Temporal tracking
        self.temporal_buffer = TemporalBuffer(max_length=100)
        self.state_classifier = StateClassifier()
        # Phase 5: Raycasting & frequency bands
        from .frequency_bands import FrequencyBandClustering
        self.frequency_bands = FrequencyBandClustering(num_bands=3)

        # Collapse configuration for gravitational collapse
        self.collapse_config = collapse_config or {
            'harmonic_tolerance': 0.05,
            'cosine_threshold': 0.85,
            'octave_tolerance': 0,
            'enable': True
        }

        # Synthesis configuration for resonance weighting
        self.synthesis_config = synthesis_config or {
            'use_resonance_weighting': True,
            'weight_function': 'linear',  # 'linear', 'sqrt', 'log'
            'resonance_boost': 2.0,  # multiplier for resonance importance
            'distance_decay': 0.5  # 0.0=ignore distance, 1.0=equal to resonance
        }

    def _get_frequency_bin(self, fundamental_freq: float, tolerance: float = 0.05) -> int:
        """Bin frequency for indexing. tolerance=0.05 → 5% bins

        Args:
            fundamental_freq: Frequency to bin
            tolerance: Bin size as fraction (0.05 = 5% bins)

        Returns:
            Integer bin index
        """
        if fundamental_freq <= 0:
            return 0
        return int(fundamental_freq / tolerance)

    def _frequency_to_position(self, freq_spectrum: np.ndarray) -> np.ndarray:
        """Map frequency spectrum to 3D position in voxel cloud."""
        return compute_frequency_position(freq_spectrum, self.width, self.height, self.depth)

    def _generate_mip_levels(self, proto_identity: np.ndarray, levels: int = 5) -> List[np.ndarray]:
        """Generate MIP pyramid for multi-scale representation."""
        return generate_mip_levels(proto_identity, levels)

    def _merge_proto_identity(self, existing: ProtoIdentityEntry,
                             new_proto: np.ndarray, new_freq: np.ndarray,
                             new_metadata: Dict) -> None:
        """
        Merge new proto-identity into existing one using weighted average.
        This implements gravitational collapse via constructive interference.
        """
        collapse_ops.merge_proto_identity(existing, new_proto, new_freq,
                                          new_metadata, self)

    def _find_similar_by_frequency(self, fundamental: float, harmonics: np.ndarray) -> List[ProtoIdentityEntry]:
        """Find existing proto-identities with similar frequency signatures."""
        return collapse_ops.find_similar_by_frequency(self, fundamental, harmonics)

    def _create_new_entry(self, proto_identity: np.ndarray, frequency: np.ndarray,
                         metadata: Dict, fundamental: float, harmonics: np.ndarray) -> ProtoIdentityEntry:
        """Create a new proto-identity entry."""
        position = self._frequency_to_position(frequency)
        mip_levels = self._generate_mip_levels(proto_identity)

        # Add unique ID if not present
        if 'id' not in metadata:
            metadata['id'] = f"{metadata.get('modality', 'text')}_{len(self.entries)}"

        # Assign frequency band (Phase 5)
        freq_band = self.frequency_bands.assign_band(frequency)

        # Extract WaveCube coordinates via triplanar projection
        wavecube_coords = None
        try:
            from .triplanar_projection import extract_triplanar_coordinates
            coords = extract_triplanar_coordinates(
                frequency,
                modality=metadata.get('modality', 'text'),
                octave=metadata.get('octave', 0)
            )
            wavecube_coords = (coords.x, coords.y, coords.z, coords.w)
        except Exception:
            # Silently fail - coordinates remain None
            pass

        return ProtoIdentityEntry(
            proto_identity=proto_identity.copy(),
            mip_levels=mip_levels,
            frequency=frequency.copy(),
            metadata=metadata.copy(),
            position=position,
            fundamental_freq=fundamental,
            octave=metadata.get('octave', 0),
            harmonic_signature=harmonics,
            resonance_strength=1,
            modality=metadata.get('modality', 'text'),
            cross_modal_links=[],
            frequency_band=int(freq_band),
            wavecube_coords=wavecube_coords
        )

    def compute_coherence(self, proto_identity: np.ndarray) -> float:
        """Compute coherence of proto-identity with existing entries.

        Coherence measures how well the proto-identity aligns with the existing
        memory structure. High coherence = consistent with memory. Low coherence = novel/conflicting.

        Args:
            proto_identity: Proto-identity to measure coherence for

        Returns:
            Coherence value [0, 1]
        """
        if len(self.entries) == 0:
            return 0.0  # No existing entries, coherence undefined

        # Compute similarity with all entries, take mean of top 10
        similarities = [
            compute_cosine_similarity(proto_identity, entry.proto_identity)
            for entry in self.entries
        ]

        # Take mean of top 10 similarities
        top_similarities = sorted(similarities, reverse=True)[:10]
        coherence = np.mean(top_similarities)

        return max(0.0, min(1.0, coherence))  # Clamp to [0, 1]

    def add_with_temporal_tracking(self, proto_identity: np.ndarray, frequency: np.ndarray,
                                    metadata: Dict, timestamp: Optional[float] = None) -> None:
        """Add proto with temporal state tracking.

        This is the Phase 2 version of add() that includes:
        - Temporal buffer tracking
        - State classification (PARADOX/EVOLUTION/IDENTITY)
        - Coherence computation

        Args:
            proto_identity: The proto-identity (H, W, 4)
            frequency: Frequency spectrum used to generate it
            metadata: Associated metadata (text, params, etc)
            timestamp: Optional timestamp (defaults to time.time())
        """
        import time

        if timestamp is None:
            timestamp = time.time()

        # Add to temporal buffer
        self.temporal_buffer.add(proto_identity, timestamp)

        # Compute coherence
        coherence = self.compute_coherence(proto_identity)

        # Classify state
        state = self.state_classifier.classify(self.temporal_buffer, coherence)

        # Update metadata with state and temporal info
        metadata['current_state'] = state.name
        metadata['timestamp'] = timestamp
        metadata['coherence'] = coherence

        # Call existing add() method
        self.add(proto_identity, frequency, metadata)

        # Update the last entry's current_state field (since add() just created it)
        if len(self.entries) > 0:
            self.entries[-1].current_state = state
            self.entries[-1].coherence_vs_core = coherence
            self.entries[-1].temporal_history.append((timestamp, proto_identity.copy()))

    def add(self, proto_identity: np.ndarray, frequency: np.ndarray, metadata: Dict) -> None:
        """
        Add a proto-identity to the voxel cloud with gravitational collapse.
        Similar proto-identities merge via constructive interference.

        Args:
            proto_identity: The proto-identity (H, W, 4)
            frequency: Frequency spectrum used to generate it
            metadata: Associated metadata (text, params, etc)
        """
        # Extract frequency signature
        from src.memory.octave_frequency import extract_fundamental, extract_harmonics
        fundamental = extract_fundamental(frequency)
        harmonics = extract_harmonics(frequency, fundamental)

        # Check for existing similar proto-identity (gravitational collapse)
        matches = self._find_similar_by_frequency(fundamental, harmonics)

        # If we have candidates, check proto-identity similarity
        for candidate in matches:
            if collapse_ops.check_similarity_for_collapse(
                proto_identity, candidate, self.collapse_config['cosine_threshold']
            ):
                self._merge_proto_identity(candidate, proto_identity, frequency, metadata)
                return

        # No similar entry found - create new one
        entry = self._create_new_entry(proto_identity, frequency, metadata, fundamental, harmonics)
        self.entries.append(entry)
        entry_idx = len(self.entries) - 1

        # Update spatial index
        position = entry.position
        grid_pos = tuple((position / [self.width, self.height, self.depth] * 10).astype(int))
        if grid_pos not in self.spatial_index:
            self.spatial_index[grid_pos] = []
        self.spatial_index[grid_pos].append(entry_idx)

        # Update frequency index
        freq_bin = self._get_frequency_bin(fundamental)
        if freq_bin not in self.frequency_index:
            self.frequency_index[freq_bin] = []
        self.frequency_index[freq_bin].append(entry_idx)

    def add_with_octaves(self, proto_identity: np.ndarray,
                        frequency: float = None, modality: str = 'text',
                        quaternions: Dict[int, np.ndarray] = None,
                        resonance_strength: float = 1.0) -> None:
        """Add proto-identity with multi-octave quaternions.

        Args:
            proto_identity: The proto-identity (H, W, 4)
            frequency: Base frequency (optional)
            modality: 'text', 'image', or 'audio'
            quaternions: Dict mapping octave → quaternion
            resonance_strength: Initial resonance strength
        """
        # Create OctaveProtoIdentity
        octave_proto = OctaveProtoIdentity(
            proto_identity=proto_identity,
            quaternions=quaternions if quaternions else {},
            frequency=frequency if frequency else 0.0,
            modality=modality
        )

        # Add to octave hierarchy
        self.octave_hierarchy.add_proto_identity(octave_proto)

        # Also add to traditional storage for backward compatibility
        metadata = {
            'modality': modality,
            'has_octaves': quaternions is not None
        }

        # Convert frequency float to proper spectrum for compatibility
        # Create a simple frequency spectrum from the fundamental frequency
        freq_spectrum = np.zeros((self.height, self.width, 4), dtype=np.float32)
        if frequency:
            # Place energy at a position based on frequency
            center_y = int(self.height * 0.5)
            center_x = int(self.width * min(frequency / 1000.0, 1.0))  # Map frequency to position
            freq_spectrum[center_y, center_x, 0] = frequency

        # Skip duplicate add if we don't need backward compatibility
        # self.add(proto_identity, freq_spectrum, metadata)

    def query_multi_octave(self, query_quaternions: Dict[int, np.ndarray],
                          top_k: int = 10,
                          octave_weights: Optional[Dict[int, float]] = None) -> List:
        """Query using multi-octave quaternions.

        Args:
            query_quaternions: Dict mapping octave → query quaternion
            top_k: Number of results to return
            octave_weights: Optional weights for each octave

        Returns:
            List of matching proto-identities with distances
        """
        return self.octave_hierarchy.multi_octave_query(
            query_quaternions, top_k, octave_weights
        )

    def find_by_frequency(self, query_freq: float, query_harmonics: np.ndarray,
                         octave_tolerance: int = 1, harmonic_tolerance: float = 0.3,
                         max_results: int = 10) -> List[ProtoIdentityEntry]:
        """Find proto-identities by frequency signature matching."""
        return query_ops.find_by_frequency(
            self, query_freq, query_harmonics,
            octave_tolerance, harmonic_tolerance, max_results
        )

    def query_by_proto_similarity(self, query_proto: np.ndarray, max_results: int = 10,
                                 similarity_metric: str = 'l2') -> List[ProtoIdentityEntry]:
        """Query voxel cloud by direct proto-identity similarity.

        Args:
            query_proto: Query proto-identity (H, W, 4)
            max_results: Maximum number of results to return
            similarity_metric: 'l2' (default, for FFT precision) or 'cosine'

        Returns:
            List of most similar proto-identities
        """
        return query_ops.query_by_proto_similarity(self, query_proto, max_results, similarity_metric)

    def query_viewport(self, query_freq: np.ndarray, radius: float = 50.0,
                      max_results: int = 10, use_frequency_matching: bool = False,
                      query_proto: np.ndarray = None) -> List[ProtoIdentityEntry]:
        """Query voxel cloud with a viewport centered on query frequency."""
        return query_ops.query_viewport(
            self, query_freq, radius, max_results, use_frequency_matching, query_proto
        )

    def _compute_synthesis_weights(self, visible_protos: List[ProtoIdentityEntry],
                                   query_pos: np.ndarray) -> np.ndarray:
        """Compute weights for synthesis - compatibility wrapper."""
        return compute_resonance_weights(visible_protos, query_pos, self.synthesis_config)

    def _compute_weights(self, visible_protos: List[ProtoIdentityEntry],
                        query_freq: np.ndarray, query_pos: np.ndarray) -> np.ndarray:
        """Compute blending weights for visible proto-identities."""
        # Use new resonance-based weighting if enabled
        if self.synthesis_config['use_resonance_weighting']:
            return compute_resonance_weights(visible_protos, query_pos, self.synthesis_config)

        # Original distance and frequency correlation weighting
        return compute_distance_weights(visible_protos, query_freq, query_pos)

    def synthesize(self, visible_protos: List[ProtoIdentityEntry], query_freq: np.ndarray) -> np.ndarray:
        """
        Synthesize new proto-identity from visible elements.

        This is not just retrieval - it creates a NEW proto-identity
        by blending visible elements based on their relevance to the query.

        Args:
            visible_protos: List of visible proto-identities
            query_freq: Query frequency spectrum

        Returns:
            Synthesized proto-identity (H, W, 4)
        """
        if not visible_protos:
            return np.zeros((self.height, self.width, 4), dtype=np.float32)

        # Compute weights for blending
        query_pos = self._frequency_to_position(query_freq)
        weights = self._compute_weights(visible_protos, query_freq, query_pos)

        # Weighted blend of proto-identities
        synthesized = np.zeros((self.height, self.width, 4), dtype=np.float32)

        for entry, weight in zip(visible_protos, weights):
            # Select MIP level based on weight (higher weight = finer detail)
            mip_idx = max(0, min(len(entry.mip_levels) - 1, int((1 - weight) * 4)))
            proto = entry.mip_levels[mip_idx]

            # Resize if necessary
            proto = resize_proto(proto, self.height, self.width)
            synthesized += proto * weight

        # Add slight noise for variation
        synthesized += np.random.randn(self.height, self.width, 4) * 0.01

        # Normalize
        return np.tanh(synthesized)

    def save(self, path: str) -> None:
        """Save voxel cloud to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        from src.security import safe_save_pickle

        safe_save_pickle({
            'width': self.width,
            'height': self.height,
            'depth': self.depth,
            'entries': self.entries,
            'spatial_index': self.spatial_index,
            'frequency_index': self.frequency_index,  # Phase 10A
            'collapse_config': self.collapse_config,
            'synthesis_config': self.synthesis_config,
            'octave_hierarchy': self.octave_hierarchy,
            'temporal_buffer': self.temporal_buffer,  # Phase 2
            'state_classifier': self.state_classifier,  # Phase 2
            'frequency_bands': self.frequency_bands  # Phase 5
        }, path)

    def load(self, path: str) -> None:
        """Load voxel cloud from disk securely."""
        from src.security import safe_load_pickle
        import logging

        logger = logging.getLogger(__name__)

        try:
            logger.info(f"Loading VoxelCloud from {path} with security checks")
            data = safe_load_pickle(path, backward_compatible=True)
            self.width = data['width']
            self.height = data['height']
            self.depth = data['depth']
            self.entries = data['entries']
            self.spatial_index = data['spatial_index']
            # Load frequency_index (Phase 10A, backward compatibility)
            if 'frequency_index' in data:
                self.frequency_index = data['frequency_index']
            else:
                # Rebuild frequency index from entries for backward compatibility
                self.frequency_index = {}
                for idx, entry in enumerate(self.entries):
                    freq_bin = self._get_frequency_bin(entry.fundamental_freq)
                    if freq_bin not in self.frequency_index:
                        self.frequency_index[freq_bin] = []
                    self.frequency_index[freq_bin].append(idx)
            # Load configs if present (backward compatibility)
            self.collapse_config = data.get('collapse_config', {
                'harmonic_tolerance': 0.05,
                'cosine_threshold': 0.85,
                'octave_tolerance': 0,
                'enable': True
            })
            self.synthesis_config = data.get('synthesis_config', {
                'use_resonance_weighting': True,
                'weight_function': 'linear',
                'resonance_boost': 2.0,
                'distance_decay': 0.5
            })
            # Load octave hierarchy if present (backward compatibility)
            if 'octave_hierarchy' in data:
                self.octave_hierarchy = data['octave_hierarchy']
            else:
                # Create new hierarchy for old models (upgrade to 12 octaves)
                self.octave_hierarchy = OctaveHierarchy(num_octaves=12)
            # Load temporal components (Phase 2, backward compatibility)
            if 'temporal_buffer' in data:
                self.temporal_buffer = data['temporal_buffer']
            else:
                self.temporal_buffer = TemporalBuffer(max_length=100)
            if 'state_classifier' in data:
                self.state_classifier = data['state_classifier']
            else:
                self.state_classifier = StateClassifier()
            # Load frequency bands (Phase 5, backward compatibility)
            if 'frequency_bands' in data:
                self.frequency_bands = data['frequency_bands']
            else:
                from .frequency_bands import FrequencyBandClustering
                self.frequency_bands = FrequencyBandClustering(num_bands=3)

            logger.info("Successfully loaded VoxelCloud")
        except Exception as e:
            logger.error(f"Failed to load VoxelCloud from {path}: {e}")
            raise

    def __len__(self) -> int:
        """Number of proto-identities in cloud."""
        return len(self.entries)

    def link_cross_modal_protos(self, phase_coherence_threshold: float = 0.1) -> int:
        """Link proto-identities across modalities via phase coherence."""
        return query_ops.link_cross_modal_protos(self, phase_coherence_threshold)

    def get_wavecube_reference(self, entry_idx: int) -> Optional[Tuple[int, int, int, float]]:
        """
        Get WaveCube reference coordinates for a proto-identity entry.

        Args:
            entry_idx: Index of entry in self.entries

        Returns:
            (X, Y, Z, W) quaternionic coordinates or None if not stored in WaveCube
        """
        if 0 <= entry_idx < len(self.entries):
            return self.entries[entry_idx].wavecube_coords
        return None

    def set_wavecube_reference(self, entry_idx: int, coords: Tuple[int, int, int, float]) -> None:
        """
        Set WaveCube reference coordinates for a proto-identity entry.

        Args:
            entry_idx: Index of entry in self.entries
            coords: (X, Y, Z, W) quaternionic coordinates
        """
        if 0 <= entry_idx < len(self.entries):
            self.entries[entry_idx].wavecube_coords = coords

    def find_cross_modal_links(self, entry: ProtoIdentityEntry) -> List[ProtoIdentityEntry]:
        """Find cross-modal linked proto-identities."""
        return query_ops.find_cross_modal_links(self, entry)

    def query_by_raycast(self, projection_matrix, max_results: int = 10) -> List[ProtoIdentityEntry]:
        """Query using raycast-based frustum culling (Phase 5)."""
        return query_ops.query_by_raycast(self, projection_matrix, max_results)

    def query_by_frequency_band(self, band: int, max_results: int = 10) -> List[ProtoIdentityEntry]:
        """Query by frequency band (Phase 5)."""
        return query_ops.query_by_frequency_band(self, band, max_results)

    def __repr__(self) -> str:
        total_resonance = sum(e.resonance_strength for e in self.entries)
        avg_resonance = total_resonance / len(self.entries) if self.entries else 0
        # Count modalities
        modality_counts = defaultdict(int)
        for e in self.entries:
            modality_counts[e.modality] += 1
        modality_str = ', '.join(f"{m}:{c}" for m, c in modality_counts.items())
        return f"VoxelCloud({len(self.entries)} protos [{modality_str}], avg_resonance={avg_resonance:.1f}, {self.width}x{self.height}x{self.depth})"