"""
ExperientialReflector - Dual coherence measurement module.

Measures coherence both:
1. Against core memory (via FeedbackLoop)
2. Within experiential layer itself (trajectory consistency)

This enables detection of internal consistency and stability in addition to
alignment with learned knowledge.
"""

import numpy as np
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .voxel_cloud import VoxelCloud

from .feedback_loop import FeedbackLoop
from .voxel_helpers import compute_cosine_similarity


class ExperientialReflector:
    """
    Dual coherence measurement for experiential memory.

    Provides two types of coherence measurements:
    1. Core coherence: How well experiential proto aligns with core memory
    2. Internal coherence: How consistent experiential proto is with its own trajectory

    This allows detection of:
    - Stable patterns (high core + high internal coherence)
    - Novel patterns (low core + high internal coherence)
    - Unstable patterns (low core + low internal coherence)
    - Conflicting patterns (various coherence combinations)
    """

    def __init__(self, feedback_loop: FeedbackLoop):
        """
        Initialize experiential reflector.

        Args:
            feedback_loop: FeedbackLoop instance for core coherence measurement
        """
        self.feedback_loop = feedback_loop

    def measure_core_coherence(
        self,
        proto: np.ndarray,
        core_memory: 'VoxelCloud'
    ) -> float:
        """
        Measure coherence against core memory.

        Uses FeedbackLoop to compare proto against core knowledge.

        Args:
            proto: Proto-identity to measure (H, W, 4)
            core_memory: Core memory VoxelCloud

        Returns:
            Coherence score [0, 1] where:
            - 1.0 = perfect alignment with core
            - 0.0 = no alignment with core
        """
        # Query core memory with proto
        core_matches = core_memory.query_by_proto_similarity(
            proto,
            max_results=10,
            similarity_metric='cosine'
        )

        # Compute coherence via FeedbackLoop's internal method
        coherence = self.feedback_loop._compute_coherence_with_core(
            proto,
            core_matches
        )

        return coherence

    def measure_internal_coherence(
        self,
        proto: np.ndarray,
        trajectory_history: List[np.ndarray]
    ) -> float:
        """
        Measure internal coherence within experiential trajectory.

        Computes average cosine similarity between proto and recent trajectory history.
        High internal coherence indicates stable/consistent experiential patterns.

        Args:
            proto: Proto-identity to measure (H, W, 4)
            trajectory_history: List of recent proto-identities in trajectory

        Returns:
            Coherence score [0, 1] where:
            - 1.0 = perfect consistency with trajectory
            - 0.0 = no consistency with trajectory
        """
        # Handle empty history
        if not trajectory_history:
            return 0.0

        # Use last N entries for coherence calculation
        # (avoids dilution from ancient history)
        n_recent = min(5, len(trajectory_history))
        recent_history = trajectory_history[-n_recent:]

        # Compute similarity with each recent entry
        similarities = []
        for historical_proto in recent_history:
            similarity = compute_cosine_similarity(proto, historical_proto)
            similarities.append(similarity)

        # Average similarity = internal coherence
        internal_coherence = np.mean(similarities) if similarities else 0.0

        # Clamp to [0, 1]
        return max(0.0, min(1.0, internal_coherence))

    def measure_dual_coherence(
        self,
        proto: np.ndarray,
        core_memory: 'VoxelCloud',
        trajectory_history: List[np.ndarray]
    ) -> Tuple[float, float]:
        """
        Measure both core and internal coherence simultaneously.

        Provides complete coherence picture:
        - Core coherence: alignment with learned knowledge
        - Internal coherence: consistency within experiential trajectory

        Args:
            proto: Proto-identity to measure (H, W, 4)
            core_memory: Core memory VoxelCloud
            trajectory_history: List of recent proto-identities in trajectory

        Returns:
            Tuple of (core_coherence, internal_coherence) where both are [0, 1]

        Example interpretations:
            (0.9, 0.9): Stable, aligned pattern
            (0.2, 0.9): Novel but consistent pattern
            (0.9, 0.2): Core-aligned but unstable
            (0.2, 0.2): Unstable, conflicting pattern
        """
        core_coherence = self.measure_core_coherence(proto, core_memory)
        internal_coherence = self.measure_internal_coherence(proto, trajectory_history)

        return core_coherence, internal_coherence

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ExperientialReflector("
            f"feedback_loop={self.feedback_loop})"
        )
