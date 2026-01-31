"""
Conversation Pipeline - End-to-end conversation orchestration.

Integrates all Genesis phases:
- Phase 1: MemoryHierarchy with proto-unity carrier
- Phase 2: TemporalBuffer with Taylor series prediction
- Phase 3: FeedbackLoop for self-reflection
- Phase 4: EncodingPipeline, DecodingPipeline for I/O
- Phase 5: ProjectionMatrix for raycasting, FrequencyBandClustering

Full workflow:
Input → Encode → Modulate carrier → Store experiential
→ Self-reflect → State classification → Temporal prediction
→ Query memory (raycast + frequency bands) → Decode → Response
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time

from src.memory.memory_hierarchy import MemoryHierarchy
from src.memory.state_classifier import SignalState
from src.memory.projection import ProjectionMatrix
from src.memory.frequency_bands import (
    FrequencyBandClustering, FrequencyBand
)
from src.pipeline.encoding import EncodingPipeline
from src.pipeline.decoding import DecodingPipeline
from src.origin import Origin


class ConversationPipeline:
    """End-to-end conversation pipeline orchestrating full workflow."""

    def __init__(self, memory_hierarchy: MemoryHierarchy, origin: Origin):
        """Initialize conversation pipeline.

        Args:
            memory_hierarchy: Memory system with core/experiential
            origin: Origin instance for carrier creation
        """
        self.memory = memory_hierarchy
        self.origin = origin

        # Create proto-unity carrier (once per session)
        self.carrier = self.memory.create_carrier(origin)

        # Initialize encoding/decoding pipelines
        self.encoder = EncodingPipeline(
            self.carrier,
            memory_hierarchy.width,
            memory_hierarchy.height
        )
        self.decoder = DecodingPipeline(self.carrier)

        # Initialize projection for raycasting
        self.projection = ProjectionMatrix(
            fov=60.0,
            aspect_ratio=1.0,
            near=0.1,
            far=1000.0
        )

        # Initialize frequency band clustering
        self.band_clustering = FrequencyBandClustering(num_bands=3)

        # Session state tracking
        self.session_start_time = time.time()
        self.input_count = 0
        self.coherence_history: List[float] = []

    def initialize_session(self) -> Dict:
        """Initialize conversation session.

        Returns:
            Session metadata
        """
        self.session_start_time = time.time()
        self.input_count = 0
        self.coherence_history = []

        return {
            'carrier_created': True,
            'memory_initialized': True,
            'core_entries': len(self.memory.core_memory),
            'experiential_entries': len(self.memory.experiential_memory)
        }

    def process_input(
        self,
        input_data: str,
        input_type: str = 'text'
    ) -> Dict:
        """Process input through full pipeline.

        Full workflow:
        1. Encode input to proto-identity
        2. Modulate carrier with input
        3. Store in experiential memory
        4. Self-reflect (compare vs core)
        5. State classification
        6. Temporal prediction
        7. Query memory with context
        8. Decode to response

        Args:
            input_data: Input text or file path
            input_type: 'text', 'image', or 'audio'

        Returns:
            Response dictionary with all metadata
        """
        # 1. Encode input to proto-identity
        if input_type == 'text':
            proto, metadata = self.encoder.encode_text(input_data)
        elif input_type == 'image':
            proto, metadata = self.encoder.encode_image(input_data)
        elif input_type == 'audio':
            proto, metadata = self.encoder.encode_audio(input_data)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")

        # 2. Extract frequency spectrum for storage
        freq_spectrum = self._proto_to_frequency(proto)

        # 3. Store in experiential memory
        timestamp = time.time()
        self.memory.store_experiential(proto, freq_spectrum, metadata)

        # Add to temporal buffer (on VoxelCloud, not manager)
        self.memory.experiential_memory.temporal_buffer.add(
            proto, timestamp
        )

        # 4. Self-reflect (compare vs core)
        query_quaternion = self._extract_quaternion(proto)
        coherence, state, recommendation = self.memory.self_reflect(
            proto, query_quaternion
        )

        # Track coherence history
        self.coherence_history.append(coherence)

        # 5. Temporal prediction
        temporal_prediction = (
            self.memory.experiential_memory.temporal_buffer
            .predict_next(delta_t=1.0, order=2)
        )

        # 6. Query memory with context
        context_protos = self.query_with_context(
            proto,
            use_raycast=True,
            frequency_band=None
        )

        # 7. Decode to response
        response = self.synthesize_response(proto, context_protos)

        # 8. Increment input count
        self.input_count += 1

        return {
            'response': response,
            'coherence': coherence,
            'state': state,
            'recommendation': recommendation,
            'context_protos': context_protos,
            'temporal_prediction': temporal_prediction,
            'metadata': {
                'input_type': input_type,
                'input_count': self.input_count,
                'session_duration': time.time() - self.session_start_time,
                'avg_coherence': np.mean(self.coherence_history),
                'fundamental_freq': metadata.get('fundamental_freq', 0.0)
            }
        }

    def query_with_context(
        self,
        query_proto: np.ndarray,
        use_raycast: bool = True,
        frequency_band: Optional[FrequencyBand] = None,
        max_results: int = 10
    ) -> List:
        """Query memory with raycasting and frequency band filtering.

        Args:
            query_proto: Query proto-identity
            use_raycast: Enable frustum-based raycasting
            frequency_band: Optional frequency band filter
            max_results: Maximum results to return

        Returns:
            List of ProtoIdentityEntry objects
        """
        # Query both core and experiential memory
        core_results = self.memory.query_core(query_proto, max_results)
        exp_results = self.memory.query_experiential(
            query_proto, max_results
        )

        # Combine results
        all_results = core_results + exp_results

        # Apply raycasting filter
        if use_raycast:
            all_results = self._apply_raycast_filter(all_results)

        # Apply frequency band filter
        if frequency_band is not None:
            all_results = self._apply_band_filter(
                all_results, frequency_band
            )

        # Sort by resonance strength and limit
        all_results.sort(
            key=lambda x: x.resonance_strength, reverse=True
        )

        return all_results[:max_results]

    def synthesize_response(
        self,
        query_proto: np.ndarray,
        context_protos: List
    ) -> str:
        """Generate response from query and context.

        Args:
            query_proto: Query proto-identity
            context_protos: List of context ProtoIdentityEntry

        Returns:
            Response text
        """
        if not context_protos:
            return "I don't have enough context to respond."

        # Use decoder to generate summary from context
        response = self.decoder.decode_to_summary(
            query_proto, context_protos
        )

        return response

    def consolidate_session(self, threshold: float = 0.8) -> int:
        """Consolidate Identity patterns from experiential to core.

        Args:
            threshold: Minimum coherence for consolidation

        Returns:
            Number of patterns consolidated
        """
        return self.memory.consolidate(threshold)

    def reset_session(self) -> None:
        """Reset experiential memory to baseline."""
        self.memory.reset_experiential()
        self.coherence_history = []
        self.input_count = 0

    def _proto_to_frequency(self, proto: np.ndarray) -> np.ndarray:
        """Extract frequency spectrum from proto-identity.

        Args:
            proto: Proto-identity (H, W, 4)

        Returns:
            Frequency spectrum (H, W, 2)
        """
        X = proto[:, :, 0]
        Y = proto[:, :, 1]

        magnitude = np.sqrt(X**2 + Y**2)
        phase = np.arctan2(Y, X)

        return np.stack([magnitude, phase], axis=-1).astype(np.float32)

    def _extract_quaternion(self, proto: np.ndarray) -> np.ndarray:
        """Extract quaternion from proto-identity.

        Args:
            proto: Proto-identity (H, W, 4)

        Returns:
            Quaternion (4,) unit vector
        """
        # Extract XYZW channels
        q = np.array([
            proto[:, :, 0].mean(),  # X
            proto[:, :, 1].mean(),  # Y
            proto[:, :, 2].mean(),  # Z
            proto[:, :, 3].mean()   # W
        ], dtype=np.float32)

        # Normalize to unit quaternion
        norm = np.linalg.norm(q)
        if norm > 1e-8:
            q = q / norm

        return q

    def _apply_raycast_filter(self, results: List) -> List:
        """Filter results by frustum visibility.

        Args:
            results: List of ProtoIdentityEntry

        Returns:
            Filtered list
        """
        visible = []

        for entry in results:
            # Extract voxel position from proto-identity
            voxel_pos = self._get_voxel_position(entry.proto_identity)

            # Check visibility
            if self.projection.is_voxel_visible(voxel_pos):
                visible.append(entry)

        return visible

    def _get_voxel_position(self, proto: np.ndarray) -> np.ndarray:
        """Extract 3D position from proto-identity.

        Args:
            proto: Proto-identity (H, W, 4)

        Returns:
            Position (3,) array
        """
        # Use center of mass of Z channel
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

        # Z coordinate from mean
        center_z = z_channel.mean() * 100.0  # Scale to voxel space

        return np.array([center_x, center_y, center_z], dtype=np.float32)

    def _apply_band_filter(
        self,
        results: List,
        band: FrequencyBand
    ) -> List:
        """Filter results by frequency band.

        Args:
            results: List of ProtoIdentityEntry
            band: FrequencyBand to filter by

        Returns:
            Filtered list
        """
        filtered = []

        for entry in results:
            entry_band = self.band_clustering.assign_band(entry.frequency)
            if entry_band == band:
                filtered.append(entry)

        return filtered

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ConversationPipeline(\n"
            f"  inputs_processed={self.input_count},\n"
            f"  avg_coherence={np.mean(self.coherence_history) if self.coherence_history else 0.0:.3f},\n"
            f"  core_memory={len(self.memory.core_memory)},\n"
            f"  experiential_memory={len(self.memory.experiential_memory)}\n"
            f")"
        )
