#!/usr/bin/env python
"""Test the impact of modulation depth parameter on proto-identity differentiation."""

import numpy as np
from scipy.spatial.distance import cosine
from src.pipeline.encoding import EncodingPipeline
from src.origin import Origin


def test_modulation_depth_sweep():
    """Test different modulation depths to find optimal value."""
    print("=" * 70)
    print("MODULATION DEPTH PARAMETER SWEEP")
    print("=" * 70)

    # Initialize origin and carrier (disable GPU for testing)
    origin = Origin(use_gpu=False)
    carrier = origin.initialize_carrier()

    # Foundation documents for testing
    texts = [
        "The Tao that can be told is not the eternal Tao.",
        "The sage does not hoard. The more he helps others, the more he benefits.",
        "The highest good is like water. Water gives life to ten thousand things.",
        "The way of Heaven is to benefit others and not to injure."
    ]

    # Test different modulation depths
    depths = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    results = []

    for depth in depths:
        print(f"\n" + "=" * 70)
        print(f"Testing Modulation Depth: {depth}")
        print("=" * 70)

        # Create pipeline with specific depth
        pipeline = EncodingPipeline(carrier=carrier, modulation_depth=depth)

        # Encode all documents
        protos = []
        for i, text in enumerate(texts):
            proto, _ = pipeline.encode_text(text)
            protos.append(proto)
            print(f"  Doc {i}: Proto variance = {np.var(proto):.6f}")

        # Compute pairwise distances
        distances = []
        for i in range(len(protos)):
            for j in range(i+1, len(protos)):
                dist = cosine(protos[i].flatten(), protos[j].flatten())
                distances.append(dist)

        avg_distance = np.mean(distances)
        min_distance = np.min(distances)
        max_distance = np.max(distances)

        print(f"\nDistance Statistics:")
        print(f"  Average: {avg_distance:.6f}")
        print(f"  Minimum: {min_distance:.6f}")
        print(f"  Maximum: {max_distance:.6f}")

        # Evaluate quality
        if avg_distance < 0.01:
            quality = "❌ BLOB (too similar)"
        elif avg_distance < 0.05:
            quality = "⚠️  PARTIAL (minimal separation)"
        elif avg_distance < 0.1:
            quality = "✅ GOOD (adequate separation)"
        else:
            quality = "🎯 EXCELLENT (strong separation)"

        print(f"  Quality: {quality}")

        # Test clustering at word level
        cluster_result = pipeline.cluster_proto_identities_by_octave(
            protos, octave_level=3, n_clusters=2
        )
        unique_clusters = len(np.unique(cluster_result['cluster_labels']))
        print(f"  Clustering: {unique_clusters} unique clusters from 4 docs")

        results.append({
            'depth': depth,
            'avg_distance': avg_distance,
            'min_distance': min_distance,
            'max_distance': max_distance,
            'quality': quality,
            'unique_clusters': unique_clusters
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF RESULTS")
    print("=" * 70)
    print(f"{'Depth':>6} | {'Avg Dist':>10} | {'Min Dist':>10} | {'Max Dist':>10} | {'Clusters':>8} | Quality")
    print("-" * 70)
    for r in results:
        print(f"{r['depth']:6.1f} | {r['avg_distance']:10.6f} | {r['min_distance']:10.6f} | "
              f"{r['max_distance']:10.6f} | {r['unique_clusters']:8d} | {r['quality']}")

    # Find optimal depth
    good_results = [r for r in results if r['avg_distance'] > 0.05]
    if good_results:
        optimal = max(good_results, key=lambda x: x['avg_distance'])
        print(f"\n🎯 OPTIMAL MODULATION DEPTH: {optimal['depth']}")
        print(f"   Average distance: {optimal['avg_distance']:.6f}")
    else:
        print("\n⚠️  No depth achieved good separation (>0.05 avg distance)")
        best = max(results, key=lambda x: x['avg_distance'])
        print(f"   Best attempt: depth={best['depth']} with avg distance={best['avg_distance']:.6f}")

    return results


def test_carrier_variance_impact():
    """Test impact of carrier initialization variance."""
    print("\n" + "=" * 70)
    print("CARRIER VARIANCE IMPACT TEST")
    print("=" * 70)

    texts = [
        "The Tao that can be told is not the eternal Tao.",
        "The way of Heaven is to benefit others and not to injure."
    ]

    # Test different noise scales in carrier initialization
    noise_scales = [0.01, 0.03, 0.05, 0.07, 0.1]

    for noise_scale in noise_scales:
        print(f"\nNoise Scale: {noise_scale}")

        # Create carrier with specific noise level
        origin = Origin(use_gpu=False)
        carrier = origin.initialize_carrier()

        # Add controlled noise
        noise = np.random.randn(*carrier.shape) * noise_scale
        carrier_noisy = carrier + noise

        print(f"  Carrier variance: {np.var(carrier_noisy):.6f}")

        # Test encoding with optimal depth (from previous test, assume 0.7)
        pipeline = EncodingPipeline(carrier=carrier_noisy, modulation_depth=0.7)

        protos = [pipeline.encode_text(t)[0] for t in texts]
        dist = cosine(protos[0].flatten(), protos[1].flatten())

        print(f"  Document distance: {dist:.6f}")

        if dist < 0.01:
            print(f"  ❌ Too similar")
        elif dist < 0.05:
            print(f"  ⚠️  Minimal separation")
        else:
            print(f"  ✅ Good separation")


def main():
    """Run parameter optimization tests."""
    print("\n" + "=" * 70)
    print("PARAMETER OPTIMIZATION FOR CARRIER FILTER ARCHITECTURE")
    print("=" * 70)

    # Test 1: Modulation depth sweep
    results = test_modulation_depth_sweep()

    # Test 2: Carrier variance impact
    test_carrier_variance_impact()

    # Save results
    import json
    with open('/tmp/modulation_depth_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("Results saved to /tmp/modulation_depth_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()