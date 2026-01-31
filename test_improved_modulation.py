#!/usr/bin/env python
"""Test improved FM modulation formula to verify blob problem resolution."""

import numpy as np
from scipy.spatial.distance import cosine
from src.pipeline.encoding import EncodingPipeline
from src.origin import Origin
import json


def test_improved_modulation():
    """Test if improved FM modulation fixes the blob problem."""
    print("=" * 70)
    print("TESTING IMPROVED FM MODULATION")
    print("=" * 70)

    # Initialize with GPU disabled
    origin = Origin(use_gpu=False)
    carrier = origin.initialize_carrier()

    # Foundation documents
    texts = [
        "The Tao that can be told is not the eternal Tao.",
        "The sage does not hoard. The more he helps others, the more he benefits.",
        "The highest good is like water. Water gives life to ten thousand things.",
        "The way of Heaven is to benefit others and not to injure."
    ]

    # Test with different modulation depths
    depths = [0.5, 0.7, 1.0, 1.5]
    best_result = None
    best_distance = 0

    for depth in depths:
        print(f"\n" + "=" * 70)
        print(f"Testing with modulation depth: {depth}")
        print("=" * 70)

        pipeline = EncodingPipeline(carrier=carrier, modulation_depth=depth)

        # Encode documents
        protos = []
        for i, text in enumerate(texts):
            proto, _ = pipeline.encode_text(text)
            protos.append(proto)

            # Check proto statistics
            variance = np.var(proto)
            unique_vals = len(np.unique(proto.round(decimals=3)))
            print(f"\nDoc {i}: '{text[:40]}...'")
            print(f"  Variance: {variance:.6f}")
            print(f"  Unique values: {unique_vals}")

        # Compute pairwise distances
        print("\nPairwise Distances:")
        distances = []
        for i in range(len(protos)):
            for j in range(i+1, len(protos)):
                dist = cosine(protos[i].flatten(), protos[j].flatten())
                distances.append(dist)
                print(f"  Doc {i} vs Doc {j}: {dist:.6f}")

        avg_distance = np.mean(distances)
        print(f"\nAverage distance: {avg_distance:.6f}")

        # Evaluate quality
        if avg_distance < 0.01:
            quality = "❌ BLOB PERSISTS"
        elif avg_distance < 0.05:
            quality = "⚠️  PARTIAL IMPROVEMENT"
        elif avg_distance < 0.1:
            quality = "✅ GOOD SEPARATION"
        else:
            quality = "🎯 EXCELLENT SEPARATION"

        print(f"Quality: {quality}")

        # Test clustering
        try:
            for octave_level, octave_name in [(3, "Word"), (6, "Sentence")]:
                result = pipeline.cluster_proto_identities_by_octave(
                    protos, octave_level=octave_level, n_clusters=None
                )
                unique_clusters = len(np.unique(result['cluster_labels']))
                print(f"{octave_name} level clustering: {unique_clusters} clusters, labels: {result['cluster_labels']}")
        except Exception as e:
            print(f"Clustering error: {e}")

        if avg_distance > best_distance:
            best_distance = avg_distance
            best_result = {
                'depth': depth,
                'avg_distance': avg_distance,
                'quality': quality
            }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if best_result:
        print(f"Best configuration:")
        print(f"  Modulation depth: {best_result['depth']}")
        print(f"  Average distance: {best_result['avg_distance']:.6f}")
        print(f"  Quality: {best_result['quality']}")

        if best_result['avg_distance'] > 0.05:
            print("\n✅ SUCCESS: Blob problem resolved!")
            print("   Proto-identities are now well-differentiated.")
        elif best_result['avg_distance'] > 0.01:
            print("\n⚠️  PARTIAL SUCCESS: Some improvement achieved")
            print("   Further optimization needed for full resolution.")
        else:
            print("\n❌ FAILURE: Blob problem persists")
            print("   Need more fundamental changes to the architecture.")

    return best_result


def test_phase_diversity():
    """Test if phase diversity based on content hash helps."""
    print("\n" + "=" * 70)
    print("TESTING PHASE DIVERSITY")
    print("=" * 70)

    origin = Origin(use_gpu=False)
    carrier = origin.initialize_carrier()
    pipeline = EncodingPipeline(carrier=carrier, modulation_depth=1.0)

    # Test identical texts (should still get different protos due to hash)
    identical_texts = [
        "The Tao that can be told is not the eternal Tao.",
        "The Tao that can be told is not the eternal Tao."  # Identical
    ]

    protos_identical = []
    for i, text in enumerate(identical_texts):
        proto, _ = pipeline.encode_text(text)
        protos_identical.append(proto)

    dist_identical = cosine(protos_identical[0].flatten(), protos_identical[1].flatten())
    print(f"\nIdentical texts distance: {dist_identical:.6f}")

    if dist_identical < 0.001:
        print("  ✅ Phase diversity working (identical texts get same proto)")
    else:
        print("  ⚠️  Phase diversity may be too strong")

    # Test similar but different texts
    similar_texts = [
        "The Tao that can be told is not the eternal Tao.",
        "The Dao that can be told is not the eternal Dao."  # Similar but different
    ]

    protos_similar = []
    for i, text in enumerate(similar_texts):
        proto, _ = pipeline.encode_text(text)
        protos_similar.append(proto)

    dist_similar = cosine(protos_similar[0].flatten(), protos_similar[1].flatten())
    print(f"\nSimilar texts distance: {dist_similar:.6f}")

    if dist_similar < dist_identical * 2:
        print("  ✅ Similar texts appropriately close")
    else:
        print("  ⚠️  Similar texts too far apart")


def main():
    """Run improved modulation tests."""
    # Test 1: Improved FM modulation
    result = test_improved_modulation()

    # Test 2: Phase diversity
    test_phase_diversity()

    # Save results
    if result:
        with open('/tmp/improved_modulation_results.json', 'w') as f:
            json.dump({
                'modulation_depth': float(result['depth']),
                'avg_distance': float(result['avg_distance']),
                'quality': result['quality']
            }, f, indent=2)

        print("\n" + "=" * 70)
        print("Results saved to /tmp/improved_modulation_results.json")
        print("=" * 70)


if __name__ == "__main__":
    main()