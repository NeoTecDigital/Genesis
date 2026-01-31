#!/usr/bin/env python3
"""Minimal entropy-based clustering test - FAST validation.

Tests ONLY the entropy indexing concept without slow encoding pipeline.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.memory.voxel_cloud import VoxelCloud
from src.memory.entropy_indexing import (
    compute_spectrum_entropy,
    normalize_entropy,
    get_entropy_cluster,
    analyze_proto_entropy
)

def create_mock_proto_and_frequency():
    """Create mock proto-identity and frequency spectrum."""
    proto = np.random.randn(512, 512, 4).astype(np.float32)
    freq = np.random.randn(512, 512, 2).astype(np.float32)
    return proto, freq


def test_entropy_computation():
    """Test entropy computation on synthetic data."""
    print("\n=== TEST 1: Entropy Computation ===")

    # Create different complexity patterns
    simple_freq = np.ones((512, 512, 2), dtype=np.float32) * 0.1  # Low entropy
    complex_freq = np.random.randn(512, 512, 2).astype(np.float32)  # High entropy

    simple_entropy = compute_spectrum_entropy(simple_freq)
    complex_entropy = compute_spectrum_entropy(complex_freq)

    print(f"Simple pattern entropy: {simple_entropy:.3f}")
    print(f"Complex pattern entropy: {complex_entropy:.3f}")
    print(f"✅ Complex > Simple: {complex_entropy > simple_entropy}")

    return complex_entropy > simple_entropy


def test_octave_separation():
    """Test that different octaves get different cluster IDs."""
    print("\n=== TEST 2: Octave Separation ===")

    # Same entropy but different octaves
    entropy = 0.5

    cluster_char = get_entropy_cluster(entropy, octave=4, num_clusters=10)
    cluster_word = get_entropy_cluster(entropy, octave=0, num_clusters=10)
    cluster_phrase = get_entropy_cluster(entropy, octave=-2, num_clusters=10)

    print(f"Character level (octave +4): cluster {cluster_char}")
    print(f"Word level (octave 0): cluster {cluster_word}")
    print(f"Phrase level (octave -2): cluster {cluster_phrase}")

    # All should be different despite same entropy
    clusters_unique = len(set([cluster_char, cluster_word, cluster_phrase])) == 3
    print(f"✅ All clusters unique: {clusters_unique}")

    return clusters_unique


def test_semantic_clustering():
    """Test that similar entropy values cluster together."""
    print("\n=== TEST 3: Semantic Clustering ===")

    # Similar entropies at same octave
    entropy1 = 0.45
    entropy2 = 0.47
    entropy3 = 0.90  # Very different

    cluster1 = get_entropy_cluster(entropy1, octave=0, num_clusters=10)
    cluster2 = get_entropy_cluster(entropy2, octave=0, num_clusters=10)
    cluster3 = get_entropy_cluster(entropy3, octave=0, num_clusters=10)

    print(f"Entropy 0.45 → cluster {cluster1}")
    print(f"Entropy 0.47 → cluster {cluster2}")
    print(f"Entropy 0.90 → cluster {cluster3}")

    # Similar entropies should cluster together
    similar_cluster = (cluster1 == cluster2)
    different_cluster = (cluster1 != cluster3)

    print(f"✅ Similar entropies cluster together: {similar_cluster}")
    print(f"✅ Different entropies separate: {different_cluster}")

    return similar_cluster and different_cluster


def test_full_pipeline():
    """Test full entropy analysis pipeline on synthetic protos."""
    print("\n=== TEST 4: Full Analysis Pipeline ===")

    voxel_cloud = VoxelCloud()

    # Create 10 synthetic protos at different octaves
    octaves = [4, 4, 4, 0, 0, 0, -2, -2, -2, -2]
    entropy_metrics = []

    for i, octave in enumerate(octaves):
        proto, freq = create_mock_proto_and_frequency()

        # Analyze entropy
        metrics = analyze_proto_entropy(proto, freq, octave)
        entropy_metrics.append((i, metrics))

        print(f"Proto {i} (octave {octave:+2d}): entropy={metrics.entropy:.2f}, cluster={metrics.cluster_id}")

    # Group by octave
    octave_clusters = {}
    for proto_id, metrics in entropy_metrics:
        if metrics.octave not in octave_clusters:
            octave_clusters[metrics.octave] = set()
        octave_clusters[metrics.octave].add(metrics.cluster_id)

    print(f"\nOctave +4 clusters: {octave_clusters.get(4, set())}")
    print(f"Octave  0 clusters: {octave_clusters.get(0, set())}")
    print(f"Octave -2 clusters: {octave_clusters.get(-2, set())}")

    # Verify no cluster ID overlap between octaves
    all_clusters = []
    for octave, clusters in octave_clusters.items():
        all_clusters.extend(clusters)

    no_overlap = len(all_clusters) == len(set(all_clusters))
    print(f"✅ No cluster ID overlap between octaves: {no_overlap}")

    return no_overlap


def main():
    """Run minimal entropy validation tests."""
    print("=" * 70)
    print("MINIMAL ENTROPY-BASED CLUSTERING VALIDATION")
    print("=" * 70)
    print("\nFast validation of entropy indexing concepts")
    print("(No slow encoder pipeline)")

    results = []

    results.append(("Entropy Computation", test_entropy_computation()))
    results.append(("Octave Separation", test_octave_separation()))
    results.append(("Semantic Clustering", test_semantic_clustering()))
    results.append(("Full Pipeline", test_full_pipeline()))

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✅ ALL ENTROPY INDEXING TESTS PASSED")
        print("Entropy-based organization is working correctly!")
    else:
        print(f"\n❌ {total - passed} TESTS FAILED")

    # Save results
    output_dir = Path('./output')
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'entropy_minimal_results.txt', 'w') as f:
        f.write("MINIMAL ENTROPY VALIDATION RESULTS\n")
        f.write("=" * 70 + "\n\n")
        for test_name, result in results:
            status = "PASS" if result else "FAIL"
            f.write(f"[{status}] {test_name}\n")
        f.write(f"\nTotal: {passed}/{total} passed\n")

    print(f"\nResults saved to {output_dir / 'entropy_minimal_results.txt'}")


if __name__ == '__main__':
    main()
