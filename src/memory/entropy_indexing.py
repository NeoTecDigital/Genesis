"""Entropy-based semantic clustering for VoxelCloud organization.

Computes Shannon entropy H = -Σ P(i)·log₂(P(i)) for frequency distributions
and uses entropy as a spatial coordinate for semantic neighborhood clustering.

Architecture:
    - Entropy Computation: Measure information content of frequency spectrum
    - Octave Separation: Never mix protos from different octave levels
    - Semantic Neighborhoods: Protos with similar entropy cluster together
    - 4D Indexing: (x, y, z, entropy) spatial organization

Benefits:
    - Semantic clustering: Similar information content → nearby in space
    - Faster queries: Search within entropy range first
    - Octave isolation: Character/word/phrase protos stay separated

Usage:
    entropy = compute_spectrum_entropy(frequency_spectrum)
    cluster_id = get_entropy_cluster(entropy, octave)
    # Protos with similar entropy are spatially near each other
"""

import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class EntropyMetrics:
    """Entropy analysis results for a proto-identity."""
    entropy: float  # Shannon entropy of frequency spectrum
    normalized_entropy: float  # Entropy normalized to [0, 1]
    octave: int  # Octave level
    cluster_id: int  # Entropy-based cluster ID
    information_content: float  # 1 - normalized_entropy


def compute_spectrum_entropy(frequency: np.ndarray) -> float:
    """Compute Shannon entropy of frequency spectrum.

    H = -Σ P(i)·log₂(P(i))

    Where P(i) is the probability distribution derived from magnitude spectrum.

    Args:
        frequency: Frequency spectrum (H, W, 2) [magnitude, phase]

    Returns:
        Entropy value (higher = more complex/random spectrum)
    """
    # Extract magnitude spectrum
    magnitude = frequency[:, :, 0]

    # Flatten and take absolute value to ensure positive
    flat_mag = np.abs(magnitude.flatten())

    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    flat_mag = flat_mag + epsilon

    # Normalize to probability distribution
    prob = flat_mag / flat_mag.sum()

    # Compute Shannon entropy (clip to avoid numerical issues)
    log_prob = np.log2(np.clip(prob, epsilon, 1.0))
    entropy = -np.sum(prob * log_prob)

    return float(entropy)


def normalize_entropy(entropy: float, octave: int) -> float:
    """Normalize entropy to [0, 1] based on octave-specific ranges.

    Different octave levels have different expected entropy ranges:
    - Character level (octave +4): Lower entropy (simple patterns)
    - Word level (octave 0): Medium entropy
    - Phrase level (octave -2, -4): Higher entropy (complex patterns)

    Args:
        entropy: Raw entropy value
        octave: Octave level

    Returns:
        Normalized entropy in [0, 1]
    """
    # Octave-specific entropy ranges (empirically determined)
    entropy_ranges = {
        4: (8.0, 12.0),    # Character level
        0: (10.0, 14.0),   # Word level
        -2: (12.0, 16.0),  # Short phrase level
        -4: (14.0, 18.0),  # Long phrase level
    }

    # Get range for this octave (default to word level)
    min_entropy, max_entropy = entropy_ranges.get(octave, (10.0, 14.0))

    # Normalize to [0, 1]
    normalized = (entropy - min_entropy) / (max_entropy - min_entropy)

    # Clip to [0, 1]
    return float(np.clip(normalized, 0.0, 1.0))


def get_entropy_cluster(
    entropy: float,
    octave: int,
    num_clusters: int = 10
) -> int:
    """Get entropy-based cluster ID with octave separation.

    Protos with similar entropy are assigned to same cluster.
    Each octave has its own set of clusters (no mixing).

    Args:
        entropy: Normalized entropy value [0, 1]
        octave: Octave level
        num_clusters: Number of clusters per octave

    Returns:
        Cluster ID (unique per octave)
    """
    # Quantize entropy into clusters
    cluster_within_octave = int(entropy * num_clusters)
    cluster_within_octave = min(cluster_within_octave, num_clusters - 1)

    # Create unique cluster ID: octave_offset + cluster_within_octave
    # Octave +4 → clusters 0-9
    # Octave  0 → clusters 100-109
    # Octave -2 → clusters 200-209
    # Octave -4 → clusters 300-309
    octave_offset = {
        4: 0,
        0: 100,
        -2: 200,
        -4: 300,
    }.get(octave, 100)  # Default to word level offset

    return octave_offset + cluster_within_octave


def analyze_proto_entropy(
    proto_identity: np.ndarray,
    frequency: np.ndarray,
    octave: int
) -> EntropyMetrics:
    """Comprehensive entropy analysis of proto-identity.

    Args:
        proto_identity: Proto-identity (H, W, 4)
        frequency: Frequency spectrum (H, W, 2)
        octave: Octave level

    Returns:
        EntropyMetrics with complete analysis
    """
    # Compute entropy
    raw_entropy = compute_spectrum_entropy(frequency)

    # Normalize
    norm_entropy = normalize_entropy(raw_entropy, octave)

    # Get cluster ID
    cluster_id = get_entropy_cluster(norm_entropy, octave)

    # Information content (inverse of entropy - how "structured" the pattern is)
    info_content = 1.0 - norm_entropy

    return EntropyMetrics(
        entropy=raw_entropy,
        normalized_entropy=norm_entropy,
        octave=octave,
        cluster_id=cluster_id,
        information_content=info_content
    )


def get_entropy_neighbors(
    target_entropy: float,
    octave: int,
    all_protos: List[Tuple[int, EntropyMetrics]],
    max_distance: float = 0.1,
    max_results: int = 10
) -> List[Tuple[int, EntropyMetrics, float]]:
    """Find protos with similar entropy (semantic neighbors).

    Only searches within same octave level.

    Args:
        target_entropy: Target entropy value [0, 1]
        octave: Target octave level
        all_protos: List of (proto_id, entropy_metrics) tuples
        max_distance: Maximum entropy distance for neighbors
        max_results: Maximum number of neighbors to return

    Returns:
        List of (proto_id, metrics, distance) tuples, sorted by distance
    """
    neighbors = []

    for proto_id, metrics in all_protos:
        # Only consider same octave
        if metrics.octave != octave:
            continue

        # Compute entropy distance
        distance = abs(metrics.normalized_entropy - target_entropy)

        if distance <= max_distance:
            neighbors.append((proto_id, metrics, distance))

    # Sort by distance
    neighbors.sort(key=lambda x: x[2])

    return neighbors[:max_results]


def get_octave_entropy_statistics(
    all_protos: List[Tuple[int, EntropyMetrics]]
) -> Dict[int, Dict[str, float]]:
    """Compute entropy statistics per octave level.

    Args:
        all_protos: List of (proto_id, entropy_metrics) tuples

    Returns:
        Dictionary mapping octave → statistics dict
    """
    from collections import defaultdict

    octave_entropies = defaultdict(list)

    # Group by octave
    for _, metrics in all_protos:
        octave_entropies[metrics.octave].append(metrics.normalized_entropy)

    # Compute statistics
    stats = {}
    for octave, entropies in octave_entropies.items():
        arr = np.array(entropies)
        stats[octave] = {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'median': float(np.median(arr)),
            'count': len(entropies)
        }

    return stats
