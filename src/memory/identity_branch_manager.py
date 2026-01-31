"""Identity Branch Manager for Paradox Detection and Multi-Identity Tracking.

Detects multiple stable attractors (paradox) and manages separate identity branches
(P and !P). Each branch refines independently to its own identity attractor.
"""

import numpy as np
from typing import List, Optional
from sklearn.cluster import DBSCAN, KMeans
import uuid

from src.memory.synthesis_types import IdentityBranch


class IdentityBranchManager:
    """Manages paradox detection and identity branch splitting/merging.

    When synthesis detects multiple stable attractors (paradox state), this
    manager splits the trajectory into separate branches. Each branch then
    refines independently toward its own identity.

    Key Operations:
    - detect_attractors: Find stable cluster centers in trajectory
    - detect_paradox: Check if trajectory oscillates between multiple attractors
    - split_paradox: Create separate branches for each attractor
    - merge_branches: Combine converged branches if they're similar
    """

    def __init__(
        self,
        min_clusters: int = 2,
        max_clusters: int = 5,
        eps: float = 0.3
    ):
        """Initialize branch manager.

        Args:
            min_clusters: Minimum clusters to consider paradox
            max_clusters: Maximum clusters to detect
            eps: DBSCAN epsilon parameter for density-based clustering
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.eps = eps

    def detect_attractors(
        self,
        trajectory: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Detect stable attractors (cluster centers) in refinement trajectory.

        Flattens proto-identities for clustering and identifies distinct
        stable regions in the trajectory space.

        Args:
            trajectory: List of proto-identity states (each H×W×4)

        Returns:
            List of attractor proto-identities (cluster centroids).
            Empty list if only single cluster found.
        """
        if not trajectory or len(trajectory) < self.min_clusters:
            return []

        # Flatten proto-identities for clustering
        flattened = np.array([p.flatten() for p in trajectory])

        # Try DBSCAN first for density-based clustering
        attractors = self._try_dbscan_clustering(flattened, trajectory[0].shape)
        if attractors:
            return attractors

        # Fallback to k-means if DBSCAN doesn't find multiple clusters
        return self._try_kmeans_clustering(flattened, trajectory)

    def _try_dbscan_clustering(
        self,
        flattened: np.ndarray,
        proto_shape: tuple
    ) -> List[np.ndarray]:
        """Try DBSCAN density-based clustering.

        Args:
            flattened: Flattened trajectory data
            proto_shape: Original proto-identity shape

        Returns:
            List of attractors if multiple clusters found, empty list otherwise
        """
        dbscan = DBSCAN(eps=self.eps, min_samples=2)
        labels = dbscan.fit_predict(flattened)

        # Count unique clusters (excluding noise label -1)
        unique_labels = set(labels)
        unique_labels.discard(-1)
        n_clusters = len(unique_labels)

        # If DBSCAN finds multiple clusters, use those centroids
        if n_clusters >= self.min_clusters:
            attractors = []
            for label in unique_labels:
                cluster_mask = labels == label
                cluster_points = flattened[cluster_mask]
                centroid = cluster_points.mean(axis=0)
                attractors.append(centroid.reshape(proto_shape))
            return attractors

        return []

    def _try_kmeans_clustering(
        self,
        flattened: np.ndarray,
        trajectory: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Try k-means clustering as fallback.

        Args:
            flattened: Flattened trajectory data
            trajectory: Original trajectory for shape reference

        Returns:
            List of attractors if found, empty list otherwise
        """
        # Try increasing k values to find optimal clustering
        for k in range(self.min_clusters, self.max_clusters + 1):
            if k > len(trajectory):
                break

            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(flattened)

            # Accept first valid k >= min_clusters
            if k >= self.min_clusters:
                attractors = [
                    center.reshape(trajectory[0].shape)
                    for center in kmeans.cluster_centers_
                ]
                return attractors

        return []

    def detect_paradox(
        self,
        trajectory: List[np.ndarray],
        coherence_history: List[float]
    ) -> bool:
        """Detect if trajectory exhibits paradox (oscillation between attractors).

        A paradox is detected when:
        1. Multiple stable attractors exist
        2. Trajectory oscillates between them (no clear convergence)

        Args:
            trajectory: List of proto-identity states
            coherence_history: Coherence scores over time

        Returns:
            True if paradox detected, False otherwise
        """
        # Detect attractors
        attractors = self.detect_attractors(trajectory)

        # Need at least 2 attractors for paradox
        if len(attractors) < self.min_clusters:
            return False

        # Check for oscillation: trajectory should not converge to single point
        # Look at recent trajectory variance
        if len(trajectory) < 5:
            return False

        recent_trajectory = trajectory[-5:]
        flattened = np.array([p.flatten() for p in recent_trajectory])

        # High variance indicates oscillation
        variance = np.var(flattened, axis=0).mean()

        # Check coherence isn't steadily increasing (that would be convergence)
        if len(coherence_history) >= 5:
            recent_coherence = coherence_history[-5:]
            # If coherence is increasing monotonically, not a paradox
            if all(
                recent_coherence[i] <= recent_coherence[i + 1]
                for i in range(len(recent_coherence) - 1)
            ):
                return False

        # Paradox if high variance AND multiple attractors
        return variance > 0.01  # Threshold for oscillation

    def split_paradox(
        self,
        proto: np.ndarray,
        attractors: List[np.ndarray]
    ) -> List[IdentityBranch]:
        """Split paradox into separate identity branches.

        Creates one branch per attractor, each initialized to refine
        independently toward its own identity.

        Args:
            proto: Current proto-identity (used for parent reference)
            attractors: List of attractor proto-identities

        Returns:
            List of IdentityBranch objects, one per attractor
        """
        branches = []

        for i, attractor in enumerate(attractors):
            branch = IdentityBranch(
                branch_id=f"branch_{uuid.uuid4().hex[:8]}_{i}",
                proto_identity=attractor.copy(),
                trajectory=[attractor.copy()],
                coherence_history=[],
                state='active'
            )
            branches.append(branch)

        return branches

    def merge_branches(
        self,
        branches: List[IdentityBranch]
    ) -> Optional[IdentityBranch]:
        """Merge converged branches if they're similar.

        Checks pairwise similarity between converged branches and merges
        those that have converged to the same identity.

        Args:
            branches: List of converged identity branches

        Returns:
            Merged branch if any pair has similarity > 0.95, None otherwise
        """
        if len(branches) < 2:
            return None

        # Only consider converged branches
        converged = [b for b in branches if b.state == 'converged']
        if len(converged) < 2:
            return None

        # Check pairwise similarities
        for i in range(len(converged)):
            for j in range(i + 1, len(converged)):
                branch_i = converged[i]
                branch_j = converged[j]

                # Compute similarity
                similarity = self._compute_similarity(
                    branch_i.proto_identity,
                    branch_j.proto_identity
                )

                # Merge if highly similar
                if similarity > 0.95:
                    return self._merge_two_branches(branch_i, branch_j)

        return None

    def _compute_similarity(
        self,
        proto1: np.ndarray,
        proto2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two proto-identities.

        Args:
            proto1: First proto-identity (H×W×4)
            proto2: Second proto-identity (H×W×4)

        Returns:
            Similarity score [0, 1]
        """
        flat1 = proto1.flatten()
        flat2 = proto2.flatten()

        # Cosine similarity
        dot_product = np.dot(flat1, flat2)
        norm1 = np.linalg.norm(flat1)
        norm2 = np.linalg.norm(flat2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return max(0.0, min(1.0, dot_product / (norm1 * norm2)))

    def _merge_two_branches(
        self,
        branch1: IdentityBranch,
        branch2: IdentityBranch
    ) -> IdentityBranch:
        """Merge two branches into a single branch.

        Averages proto-identities and concatenates trajectories.

        Args:
            branch1: First branch
            branch2: Second branch

        Returns:
            Merged IdentityBranch
        """
        # Average proto-identities
        merged_proto = (
            branch1.proto_identity + branch2.proto_identity
        ) / 2.0

        # Concatenate trajectories
        merged_trajectory = branch1.trajectory + branch2.trajectory

        # Concatenate coherence histories
        merged_coherence = (
            branch1.coherence_history + branch2.coherence_history
        )

        # Create merged branch
        merged_branch = IdentityBranch(
            branch_id=f"merged_{uuid.uuid4().hex[:8]}",
            proto_identity=merged_proto,
            trajectory=merged_trajectory,
            coherence_history=merged_coherence,
            state='converged'
        )

        return merged_branch
