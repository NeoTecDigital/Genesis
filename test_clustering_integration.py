#!/usr/bin/env python
"""Test clustering integration and diagnose universal blob problem."""

import numpy as np
from scipy.spatial.distance import cosine, euclidean
from src.pipeline.encoding import EncodingPipeline
from src.origin import Origin
import json


def test_blob_problem():
    """Test if 4 foundation documents still collapse to 1 proto."""
    print("=" * 70)
    print("Testing Universal Blob Problem - Carrier Filter Architecture")
    print("=" * 70)

    # Initialize pipeline with corrected carrier filter architecture
    origin = Origin()
    carrier = origin.initialize_carrier()
    pipeline = EncodingPipeline(carrier=carrier)

    # Check carrier variance
    carrier_variance = np.var(carrier)
    print(f"\nCarrier variance: {carrier_variance:.6f}")
    print(f"Carrier shape: {carrier.shape}")
    print(f"Carrier range: [{np.min(carrier):.3f}, {np.max(carrier):.3f}]")

    # Foundation documents - should create 4 distinct proto-identities
    texts = [
        "The Tao that can be told is not the eternal Tao.",
        "The sage does not hoard. The more he helps others, the more he benefits.",
        "The highest good is like water. Water gives life to ten thousand things.",
        "The way of Heaven is to benefit others and not to injure."
    ]

    print("\n" + "=" * 70)
    print("Encoding Foundation Documents")
    print("=" * 70)

    protos = []
    for i, text in enumerate(texts):
        proto, metadata = pipeline.encode_text(text)
        protos.append(proto)
        print(f"\nDocument {i}: '{text[:50]}...'")
        print(f"  Proto shape: {proto.shape}")
        print(f"  Proto variance: {np.var(proto):.6f}")
        print(f"  Proto range: [{np.min(proto):.3f}, {np.max(proto):.3f}]")

    print("\n" + "=" * 70)
    print("Pairwise Distance Analysis")
    print("=" * 70)

    # Compute pairwise distances
    distances = []
    for i in range(len(protos)):
        for j in range(i+1, len(protos)):
            # Flatten protos for distance computation
            p1_flat = protos[i].flatten()
            p2_flat = protos[j].flatten()

            cosine_dist = cosine(p1_flat, p2_flat)
            euclidean_dist = euclidean(p1_flat, p2_flat)

            distances.append({
                'pair': (i, j),
                'cosine': cosine_dist,
                'euclidean': euclidean_dist
            })

            print(f"\nDoc {i} vs Doc {j}:")
            print(f"  Cosine distance: {cosine_dist:.6f}")
            print(f"  Euclidean distance: {euclidean_dist:.3f}")

    # Analyze results
    avg_cosine = np.mean([d['cosine'] for d in distances])
    avg_euclidean = np.mean([d['euclidean'] for d in distances])

    print("\n" + "=" * 70)
    print("Blob Problem Analysis")
    print("=" * 70)
    print(f"Average cosine distance: {avg_cosine:.6f}")
    print(f"Average Euclidean distance: {avg_euclidean:.3f}")

    if avg_cosine < 0.01:
        print("\n❌ BLOB PROBLEM DETECTED: Proto-identities are nearly identical!")
        print("   All documents collapse to a single representation.")
    elif avg_cosine < 0.05:
        print("\n⚠️  PARTIAL BLOB: Proto-identities have minimal differentiation")
        print("   Some separation exists but insufficient for retrieval.")
    else:
        print("\n✅ NO BLOB PROBLEM: Proto-identities are well-differentiated!")
        print("   Each document has a distinct representation.")

    return protos, distances


def test_clustering_at_octaves(protos):
    """Test clustering at different octave levels."""
    print("\n" + "=" * 70)
    print("Testing Octave-Level Clustering")
    print("=" * 70)

    origin = Origin()
    carrier = origin.initialize_carrier()
    pipeline = EncodingPipeline(carrier=carrier)

    # Test clustering at different octave levels
    octave_names = {
        0: "Letter",
        3: "Word",
        6: "Sentence",
        9: "Paragraph",
        11: "Document"
    }

    for octave_level, name in octave_names.items():
        print(f"\n{name} Level (Octave {octave_level}):")
        print("-" * 40)

        try:
            result = pipeline.cluster_proto_identities_by_octave(
                protos,
                octave_level=octave_level,
                n_clusters=None  # Auto-detect
            )

            print(f"  Number of clusters: {result['n_clusters']}")
            print(f"  Cluster labels: {result['cluster_labels']}")
            print(f"  Cluster sizes: {result['cluster_sizes']}")
            print(f"  Inertia: {result['inertia']:.6f}")

            # Check if clustering provides differentiation
            unique_clusters = len(np.unique(result['cluster_labels']))
            if unique_clusters == 1:
                print(f"  ⚠️  All docs in same cluster at {name} level")
            elif unique_clusters == len(protos):
                print(f"  ✅ Each doc in unique cluster at {name} level")
            else:
                print(f"  ℹ️  {unique_clusters} clusters for {len(protos)} docs")

        except Exception as e:
            print(f"  ❌ Error clustering at octave {octave_level}: {e}")


def check_fm_modulation_parameters():
    """Check current FM modulation parameters."""
    print("\n" + "=" * 70)
    print("FM Modulation Parameters Check")
    print("=" * 70)

    from src.memory.fm_modulation_base import FMModulationBase

    modulator = FMModulationBase()

    # Check attributes
    attrs = ['modulation_depth', 'frequency_range', 'phase_coupling']
    for attr in attrs:
        if hasattr(modulator, attr):
            value = getattr(modulator, attr)
            print(f"  {attr}: {value}")

    # Test modulation with sample data
    test_carrier = np.random.randn(64, 64, 4).astype(np.float32) * 0.1
    test_signal = np.random.randn(64, 64, 4).astype(np.float32) * 0.1

    modulated = modulator.modulate(test_carrier, test_signal)

    print(f"\nTest modulation:")
    print(f"  Input carrier variance: {np.var(test_carrier):.6f}")
    print(f"  Input signal variance: {np.var(test_signal):.6f}")
    print(f"  Output variance: {np.var(modulated):.6f}")
    print(f"  Output range: [{np.min(modulated):.3f}, {np.max(modulated):.3f}]")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("CLUSTERING INTEGRATION & BLOB PROBLEM DIAGNOSIS")
    print("=" * 70)

    # Test 1: Check blob problem
    protos, distances = test_blob_problem()

    # Test 2: Test octave-level clustering
    test_clustering_at_octaves(protos)

    # Test 3: Check FM modulation parameters
    check_fm_modulation_parameters()

    # Save results for analysis
    results = {
        'blob_analysis': {
            'avg_cosine': float(np.mean([d['cosine'] for d in distances])),
            'avg_euclidean': float(np.mean([d['euclidean'] for d in distances])),
            'distances': distances
        }
    }

    with open('/tmp/clustering_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("Results saved to /tmp/clustering_test_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()