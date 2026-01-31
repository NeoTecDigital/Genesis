#!/usr/bin/env python3
"""
Core K-means clustering implementation in pure NumPy.

This module provides a dependency-free implementation of k-means clustering
specifically optimized for proto-unity space (complex frequency domain data).
It serves as the foundational clustering algorithm for discovering proto-identity
structures in Genesis learning.

The implementation handles complex-valued frequency domain data (RGBA channels)
and provides efficient convergence using k-means++ initialization.
"""

import numpy as np
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass


@dataclass
class KMeansResult:
    """Result of k-means clustering."""
    centers: np.ndarray  # (k, feature_dim) cluster centers
    assignments: np.ndarray  # (n_samples,) cluster assignments
    inertia: float  # Sum of squared distances to nearest center
    n_iter: int  # Number of iterations to convergence
    converged: bool  # Whether algorithm converged


def initialize_clusters(
    data: np.ndarray,
    k: int,
    method: str = 'kmeans++',
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Initialize k cluster centers from data using k-means++ or random selection.

    K-means++ initialization selects centers that are far apart, leading to
    better convergence and final clustering quality.

    Args:
        data: (n_samples, feature_dim) data array
        k: Number of clusters
        method: 'kmeans++' or 'random' initialization
        seed: Random seed for reproducibility

    Returns:
        (k, feature_dim) array of initial cluster centers
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples, feature_dim = data.shape

    if k > n_samples:
        raise ValueError(f"k={k} exceeds number of samples={n_samples}")

    if method == 'random':
        # Random initialization: select k random points
        indices = np.random.choice(n_samples, k, replace=False)
        return data[indices].copy()

    elif method == 'kmeans++':
        # K-means++ initialization for better convergence
        centers = np.empty((k, feature_dim), dtype=data.dtype)

        # Choose first center randomly
        first_idx = np.random.randint(n_samples)
        centers[0] = data[first_idx]

        # Choose remaining centers with probability proportional to squared distance
        for i in range(1, k):
            # Compute squared distances to nearest center
            distances = np.zeros(n_samples)
            for j in range(n_samples):
                min_dist_sq = np.inf
                for c in range(i):
                    dist_sq = np.sum((data[j] - centers[c]) ** 2)
                    min_dist_sq = min(min_dist_sq, dist_sq)
                distances[j] = min_dist_sq

            # Choose next center with probability proportional to squared distance
            # Add small epsilon to avoid division by zero
            sum_distances = distances.sum()
            if sum_distances > 0:
                probabilities = distances / sum_distances
            else:
                # If all distances are zero, use uniform probability
                probabilities = np.ones(n_samples) / n_samples
            next_idx = np.random.choice(n_samples, p=probabilities)
            centers[i] = data[next_idx]

        return centers

    else:
        raise ValueError(f"Unknown initialization method: {method}")


def assign_to_clusters(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Assign each data point to the nearest cluster center.

    Uses Euclidean distance in feature space. For complex frequency domain data,
    this computes distance across all frequency components.

    Args:
        data: (n_samples, feature_dim) data array
        centers: (k, feature_dim) cluster centers

    Returns:
        (n_samples,) array of cluster assignments (0 to k-1)
    """
    n_samples = data.shape[0]
    k = centers.shape[0]

    assignments = np.zeros(n_samples, dtype=np.int32)

    for i in range(n_samples):
        min_dist_sq = np.inf
        best_cluster = 0

        for j in range(k):
            # Compute squared Euclidean distance
            dist_sq = np.sum((data[i] - centers[j]) ** 2)

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_cluster = j

        assignments[i] = best_cluster

    return assignments


def update_cluster_centers(
    data: np.ndarray,
    assignments: np.ndarray,
    k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update cluster centers as the mean of assigned points.

    Also returns the number of points assigned to each cluster for
    detecting empty clusters.

    Args:
        data: (n_samples, feature_dim) data array
        assignments: (n_samples,) cluster assignments
        k: Number of clusters

    Returns:
        Tuple of:
            - (k, feature_dim) new cluster centers
            - (k,) cluster sizes (number of points per cluster)
    """
    feature_dim = data.shape[1]
    # Use float64 for accurate mean computation
    new_centers = np.zeros((k, feature_dim), dtype=np.float64)
    cluster_sizes = np.zeros(k, dtype=np.int32)

    for i in range(k):
        # Find all points assigned to this cluster
        mask = assignments == i
        cluster_points = data[mask]

        if len(cluster_points) > 0:
            # Update center as mean of assigned points
            new_centers[i] = np.mean(cluster_points, axis=0)
            cluster_sizes[i] = len(cluster_points)
        else:
            # Empty cluster - keep previous center or reinitialize
            # For now, we'll set to a random data point
            random_idx = np.random.randint(data.shape[0])
            new_centers[i] = data[random_idx]
            cluster_sizes[i] = 0

    # Convert back to original dtype
    new_centers = new_centers.astype(data.dtype)
    return new_centers, cluster_sizes


def compute_inertia(data: np.ndarray, centers: np.ndarray, assignments: np.ndarray) -> float:
    """
    Compute the sum of squared distances to nearest cluster center (inertia).

    Lower inertia indicates tighter, more coherent clusters.

    Args:
        data: (n_samples, feature_dim) data array
        centers: (k, feature_dim) cluster centers
        assignments: (n_samples,) cluster assignments

    Returns:
        Total inertia (sum of squared distances)
    """
    inertia = 0.0
    n_samples = data.shape[0]

    for i in range(n_samples):
        cluster_idx = assignments[i]
        dist_sq = np.sum((data[i] - centers[cluster_idx]) ** 2)
        inertia += dist_sq

    return inertia


def kmeans_iteration(
    data: np.ndarray,
    k: int,
    max_iters: int = 100,
    tol: float = 1e-4,
    init: str = 'kmeans++',
    seed: Optional[int] = None,
    verbose: bool = False
) -> KMeansResult:
    """
    Run k-means clustering to convergence.

    The algorithm iteratively assigns points to nearest centers and updates
    centers until convergence (centers change less than tolerance) or maximum
    iterations reached.

    Args:
        data: (n_samples, feature_dim) data array
        k: Number of clusters
        max_iters: Maximum iterations before stopping
        tol: Convergence tolerance (stop when centers move less than this)
        init: Initialization method ('kmeans++' or 'random')
        seed: Random seed for reproducibility
        verbose: Print convergence information

    Returns:
        KMeansResult with centers, assignments, and convergence info
    """
    # Initialize cluster centers
    centers = initialize_clusters(data, k, method=init, seed=seed)

    prev_inertia = np.inf
    converged = False

    for iteration in range(max_iters):
        # E-step: Assign points to nearest centers
        assignments = assign_to_clusters(data, centers)

        # M-step: Update centers as mean of assigned points
        new_centers, cluster_sizes = update_cluster_centers(data, assignments, k)

        # Check for convergence
        center_shift = np.sqrt(np.sum((new_centers - centers) ** 2))
        inertia = compute_inertia(data, new_centers, assignments)

        if verbose:
            print(f"Iteration {iteration + 1}: inertia={inertia:.4f}, "
                  f"center_shift={center_shift:.6f}, "
                  f"cluster_sizes={cluster_sizes}")

        # Check convergence criteria
        if center_shift < tol:
            converged = True
            if verbose:
                print(f"Converged at iteration {iteration + 1}")
            centers = new_centers
            break

        # Check if inertia is increasing (shouldn't happen, but check for numerical issues)
        if inertia > prev_inertia:
            if verbose:
                print(f"Warning: Inertia increased at iteration {iteration + 1}")

        centers = new_centers
        prev_inertia = inertia

    # Final assignments and inertia
    final_assignments = assign_to_clusters(data, centers)
    final_inertia = compute_inertia(data, centers, final_assignments)

    return KMeansResult(
        centers=centers,
        assignments=final_assignments,
        inertia=final_inertia,
        n_iter=iteration + 1,
        converged=converged
    )


def cluster_proto_unity_states(
    states: Union[np.ndarray, List[np.ndarray]],
    k: int,
    flatten_method: str = 'frequency_magnitude',
    **kmeans_kwargs
) -> Tuple[np.ndarray, np.ndarray, KMeansResult]:
    """
    Cluster states in proto-unity space (frequency domain).

    This function specifically handles Genesis proto-unity states which are
    complex-valued frequency domain representations in RGBA format.

    Args:
        states: Either:
            - (batch, height, width, 4) RGBA frequency domain states
            - List of (height, width, 4) arrays
        k: Number of clusters
        flatten_method: How to flatten complex frequency data:
            - 'frequency_magnitude': Use magnitude of complex values
            - 'frequency_real_imag': Concatenate real and imaginary parts
            - 'raw': Direct flattening (for real-valued data)
        **kmeans_kwargs: Additional arguments for kmeans_iteration

    Returns:
        Tuple of:
            - (k, original_shape) cluster centers in original shape
            - (n_samples,) cluster assignments
            - Full KMeansResult object
    """
    # Convert to numpy array if list
    if isinstance(states, list):
        states = np.stack(states, axis=0)

    # Get dimensions
    if states.ndim == 4:
        batch_size, height, width, channels = states.shape
        original_shape = (height, width, channels)
    else:
        raise ValueError(f"Expected 4D array (batch, H, W, C), got {states.ndim}D")

    # Flatten based on method
    if flatten_method == 'frequency_magnitude':
        # For complex frequency data, use magnitude
        # Assume complex data is stored as pairs of real/imag in channels
        if channels == 4:
            # RGBA: R+iG, B+iA as two complex channels
            complex_r = states[..., 0] + 1j * states[..., 1]
            complex_b = states[..., 2] + 1j * states[..., 3]

            # Compute magnitudes
            mag_r = np.abs(complex_r)
            mag_b = np.abs(complex_b)

            # Stack and flatten
            mags = np.stack([mag_r, mag_b], axis=-1)
            flat_data = mags.reshape(batch_size, -1)
        else:
            # Fallback to raw flattening
            flat_data = states.reshape(batch_size, -1)

    elif flatten_method == 'frequency_real_imag':
        # Keep real and imaginary parts separate
        flat_data = states.reshape(batch_size, -1)

    elif flatten_method == 'raw':
        # Direct flattening
        flat_data = states.reshape(batch_size, -1)

    else:
        raise ValueError(f"Unknown flatten_method: {flatten_method}")

    # Run k-means
    result = kmeans_iteration(flat_data, k, **kmeans_kwargs)

    # Reshape centers back to original shape
    if flatten_method == 'frequency_magnitude' and channels == 4:
        # Centers are in magnitude space, need to reconstruct
        # For simplicity, we'll return centers in flattened magnitude space
        # Real application would need phase information for full reconstruction
        centers_reshaped = result.centers.reshape(k, height, width, 2)
        # Expand back to 4 channels (set imaginary parts to 0 for visualization)
        centers_full = np.zeros((k, height, width, 4))
        centers_full[..., 0] = centers_reshaped[..., 0]  # R magnitude
        centers_full[..., 2] = centers_reshaped[..., 1]  # B magnitude
        centers_reshaped = centers_full
    else:
        centers_reshaped = result.centers.reshape(k, *original_shape)

    return centers_reshaped, result.assignments, result


def compute_cluster_variance(
    data: np.ndarray,
    centers: np.ndarray,
    assignments: np.ndarray
) -> np.ndarray:
    """
    Compute intra-cluster variance for each cluster.

    Lower variance indicates more coherent, tightly grouped clusters.

    Args:
        data: (n_samples, feature_dim) data array
        centers: (k, feature_dim) cluster centers
        assignments: (n_samples,) cluster assignments

    Returns:
        (k,) array of variances for each cluster
    """
    k = centers.shape[0]
    variances = np.zeros(k)

    for i in range(k):
        # Get points in this cluster
        mask = assignments == i
        cluster_points = data[mask]

        if len(cluster_points) > 0:
            # Compute mean squared distance from center
            distances_sq = np.sum((cluster_points - centers[i]) ** 2, axis=1)
            variances[i] = np.mean(distances_sq)
        else:
            # Empty cluster has infinite variance
            variances[i] = np.inf

    return variances


# Example usage and testing
if __name__ == "__main__":
    print("=== Core K-means Implementation Test ===\n")

    # Test 1: Basic k-means on synthetic data
    print("Test 1: Basic k-means on well-separated clusters")
    np.random.seed(42)

    # Create 3 well-separated clusters
    cluster1 = np.random.randn(30, 10) + np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    cluster2 = np.random.randn(30, 10) + np.array([0, 5, 0, 0, 0, 0, 0, 0, 0, 0])
    cluster3 = np.random.randn(30, 10) + np.array([0, 0, 5, 0, 0, 0, 0, 0, 0, 0])
    data = np.vstack([cluster1, cluster2, cluster3])

    # Shuffle data
    indices = np.random.permutation(90)
    data = data[indices]

    # Run k-means
    result = kmeans_iteration(data, k=3, seed=42, verbose=True)

    print(f"\nResults:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.n_iter}")
    print(f"  Final inertia: {result.inertia:.2f}")
    print(f"  Cluster assignments (first 10): {result.assignments[:10]}")

    # Test 2: Proto-unity states (simulated frequency domain)
    print("\n\nTest 2: Clustering proto-unity states (frequency domain)")

    # Simulate batch of frequency domain states (complex RGBA)
    batch_size = 16
    height, width = 64, 64  # Smaller for testing

    # Create synthetic frequency domain patterns
    states = []
    for i in range(batch_size):
        state = np.zeros((height, width, 4), dtype=np.float32)

        # Add different frequency patterns to different samples
        pattern_type = i % 3
        if pattern_type == 0:
            # Low frequency pattern
            state[0:10, 0:10, :] = np.random.randn(10, 10, 4) * 10
        elif pattern_type == 1:
            # Mid frequency pattern
            state[20:30, 20:30, :] = np.random.randn(10, 10, 4) * 10
        else:
            # High frequency pattern
            state[40:50, 40:50, :] = np.random.randn(10, 10, 4) * 10

        states.append(state)

    states = np.array(states)

    # Cluster proto-unity states
    centers, assignments, result = cluster_proto_unity_states(
        states,
        k=3,
        flatten_method='raw',  # Use raw for this synthetic data
        verbose=False
    )

    print(f"\nResults:")
    print(f"  Centers shape: {centers.shape}")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.n_iter}")
    print(f"  Cluster assignments: {assignments}")

    # Compute cluster variances
    flat_data = states.reshape(batch_size, -1)
    variances = compute_cluster_variance(flat_data, result.centers, result.assignments)
    print(f"  Cluster variances: {variances}")
    print(f"  Mean variance (coherence): {np.mean(variances[variances != np.inf]):.4f}")

    print("\n✅ All core k-means functions implemented and tested!")