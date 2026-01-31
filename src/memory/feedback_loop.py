"""
Feedback Loop - Self-reflection: Compare experiential thinking against core knowledge.

The feedback loop measures how well current experiential thoughts align with
established core knowledge, classifying the relationship into three states:
- ALIGNED: Experiential thought matches core (coherence high)
- LEARNING: Experiential thought differs but not conflicting (moderate coherence)
- CONFLICT: Experiential thought contradicts core (low coherence)
"""

import numpy as np
from typing import Tuple, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .voxel_cloud import VoxelCloud

from .state_classifier import SignalState
from .voxel_helpers import compute_cosine_similarity


class FeedbackLoop:
    """Self-reflection: Compare experiential thinking against core knowledge.

    This implements the feedback mechanism where experiential memory (working thoughts)
    is compared against core memory (learned knowledge) to determine coherence.
    """

    def __init__(self, core_memory: 'VoxelCloud', experiential_memory: 'VoxelCloud',
                 aligned_threshold: float = 0.8,
                 conflict_threshold: float = 0.3):
        """Initialize feedback loop.

        Args:
            core_memory: Long-term learned knowledge (VoxelCloud)
            experiential_memory: Short-term working memory (VoxelCloud)
            aligned_threshold: Minimum coherence for ALIGNED state (default: 0.8)
            conflict_threshold: Maximum coherence for CONFLICT state (default: 0.3)
        """
        self.core = core_memory
        self.experiential = experiential_memory
        self.aligned_threshold = aligned_threshold
        self.conflict_threshold = conflict_threshold

    def self_reflect(self, experiential_proto: np.ndarray,
                    query_quaternion: np.ndarray) -> Tuple[float, SignalState, str]:
        """Compare current experiential thought against core knowledge.

        Process:
        1. Query core memory with same quaternion (find what core knows)
        2. Measure coherence between experiential proto and core matches
        3. Classify relationship (ALIGNED/LEARNING/CONFLICT)
        4. Return recommendation

        Args:
            experiential_proto: Current experiential proto-identity (H, W, 4)
            query_quaternion: Quaternion used to generate experiential_proto

        Returns:
            Tuple of:
            - coherence_score: [0, 1] similarity with core knowledge
            - state: SignalState (IDENTITY/EVOLUTION/PARADOX mapping to ALIGNED/LEARNING/CONFLICT)
            - recommendation: 'ALIGNED' | 'LEARNING' | 'CONFLICT'
        """
        # Query core memory with same proto-identity
        core_matches = self.core.query_by_proto_similarity(
            experiential_proto,
            max_results=10,
            similarity_metric='cosine'
        )

        # Compute coherence with core knowledge
        coherence = self._compute_coherence_with_core(
            experiential_proto, core_matches
        )

        # Classify state based on coherence
        state, recommendation = self._classify_coherence(coherence)

        return coherence, state, recommendation

    def _compute_coherence_with_core(self, experiential_proto: np.ndarray,
                                     core_matches: list,
                                     octave: int = None) -> float:
        """Compute coherence between experiential proto and core matches.

        Coherence is the similarity with the best matching core entry.
        We use the top match only, as it represents the most relevant core knowledge.
        If octave is specified, prioritize matches from the same octave.

        High coherence = aligns with core knowledge.
        Low coherence = novel/conflicting with core knowledge.

        Args:
            experiential_proto: Experiential proto-identity
            core_matches: List of matching ProtoIdentityEntry from core
            octave: Optional octave level to prioritize

        Returns:
            Coherence score [0, 1]
        """
        if not core_matches:
            # No core knowledge - coherence undefined, return 0
            return 0.0

        # If octave specified, filter for same-octave matches first
        if octave is not None:
            octave_matches = [
                m for m in core_matches
                if m.metadata.get('octave') == octave
            ]
            if octave_matches:
                core_matches = octave_matches

        # Use the top match only (most relevant core knowledge)
        # This avoids diluting coherence with irrelevant matches
        top_match = core_matches[0]
        coherence = compute_cosine_similarity(
            experiential_proto,
            top_match.proto_identity
        )

        return max(0.0, min(1.0, coherence))  # Clamp to [0, 1]

    def _classify_coherence(self, coherence: float) -> Tuple[SignalState, str]:
        """Classify coherence into state and recommendation.

        Classification:
        - coherence >= aligned_threshold → IDENTITY (ALIGNED)
        - coherence <= conflict_threshold → PARADOX (CONFLICT)
        - Otherwise → EVOLUTION (LEARNING)

        Args:
            coherence: Coherence score [0, 1]

        Returns:
            Tuple of (SignalState, recommendation_string)
        """
        if coherence >= self.aligned_threshold:
            # High coherence: experiential aligns with core
            return SignalState.IDENTITY, 'ALIGNED'
        elif coherence <= self.conflict_threshold:
            # Low coherence: experiential conflicts with core
            return SignalState.PARADOX, 'CONFLICT'
        else:
            # Moderate coherence: learning/adapting
            return SignalState.EVOLUTION, 'LEARNING'

    def compute_octave_coherence(self, experiential_memory: 'VoxelCloud',
                                 octave_range: Tuple[int, int] = (-4, 4)) -> Dict[int, float]:
        """Compute coherence scores per octave level.

        Args:
            experiential_memory: Experiential memory to analyze
            octave_range: Range of octaves to check (min, max inclusive)

        Returns:
            Dict mapping octave level to average coherence score
        """
        octave_coherences = {}

        for octave in range(octave_range[0], octave_range[1] + 1):
            # Find all experiential entries at this octave
            octave_entries = [
                e for e in experiential_memory.entries
                if e.metadata.get('octave') == octave
            ]

            if not octave_entries:
                continue

            # Compute coherence for each entry
            coherences = []
            for entry in octave_entries:
                core_matches = self.core.query_by_proto_similarity(
                    entry.proto_identity, max_results=10
                )
                coherence = self._compute_coherence_with_core(
                    entry.proto_identity, core_matches, octave
                )
                coherences.append(coherence)

            # Store average coherence for this octave
            octave_coherences[octave] = np.mean(coherences) if coherences else 0.0

        return octave_coherences

    def __repr__(self) -> str:
        """String representation."""
        return (f"FeedbackLoop("
                f"core_entries={len(self.core)}, "
                f"exp_entries={len(self.experiential)}, "
                f"aligned_threshold={self.aligned_threshold}, "
                f"conflict_threshold={self.conflict_threshold})")
