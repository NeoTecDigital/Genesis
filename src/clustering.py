"""
Clustering for proto-unity space discovery.

Wrapper around clustering_core with convenience interface.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional

from .clustering_core import kmeans_iteration, KMeansResult


class ClusteringMethod(Enum):
    """Available clustering methods."""
    KMEANS = 'kmeans'


@dataclass
class ClusterResult:
    """Result from clustering."""
    cluster_centers: np.ndarray  # (k, feature_dim)
    cluster_labels: np.ndarray  # (n_samples,)
    cluster_sizes: np.ndarray  # (k,)
    cluster_variances: np.ndarray  # (k,)
    n_clusters: int
    inertia: float
    converged: bool


class ProtoUnityClusterer:
    """
    Clusterer for discovering proto-unity states in feature space.

    Wraps k-means with automatic cluster number selection.
    """

    def __init__(
        self,
        n_clusters: int | str = 'auto',
        method: ClusteringMethod = ClusteringMethod.KMEANS,
        max_clusters: int = 10,
        min_clusters: int = 2
    ):
        """
        Initialize clusterer.

        Args:
            n_clusters: Number of clusters or 'auto' for automatic selection
            method: Clustering method to use
            max_clusters: Maximum clusters for auto mode
            min_clusters: Minimum clusters for auto mode
        """
        self.n_clusters = n_clusters
        self.method = method
        self.max_clusters = max_clusters
        self.min_clusters = min_clusters

    def fit(self, features: np.ndarray, verbose: bool = False) -> ClusterResult:
        """
        Fit clustering model to features.

        Args:
            features: (n_samples, feature_dim) feature array
            verbose: Print progress

        Returns:
            ClusterResult with cluster centers and assignments
        """
        n_samples = features.shape[0]

        if self.n_clusters == 'auto':
            # Auto-select k based on sample size
            k = min(max(self.min_clusters, n_samples // 50), self.max_clusters)
        else:
            k = int(self.n_clusters)

        k = min(k, n_samples)  # Can't have more clusters than samples

        if verbose:
            print(f"  Fitting {self.method.value} with k={k}...")

        # Run k-means
        result: KMeansResult = kmeans_iteration(
            features,
            k=k,
            max_iters=100,
            tol=1e-4,
            init='kmeans++',
            verbose=verbose
        )

        # Compute cluster sizes and variances
        cluster_sizes = np.bincount(result.assignments, minlength=k)

        cluster_variances = np.zeros(k)
        for i in range(k):
            mask = result.assignments == i
            if mask.sum() > 0:
                cluster_points = features[mask]
                distances_sq = np.sum((cluster_points - result.centers[i]) ** 2, axis=1)
                cluster_variances[i] = np.mean(distances_sq)
            else:
                cluster_variances[i] = np.inf

        return ClusterResult(
            cluster_centers=result.centers,
            cluster_labels=result.assignments,
            cluster_sizes=cluster_sizes,
            cluster_variances=cluster_variances,
            n_clusters=k,
            inertia=result.inertia,
            converged=result.converged
        )
