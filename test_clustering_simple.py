#!/usr/bin/env python
"""Simplified test for clustering integration without GPU dependency."""

import numpy as np
from scipy.spatial.distance import cosine, euclidean
import json


def initialize_simple_carrier(width=512, height=512):
    """Initialize a simple carrier using gamma_simple and epsilon_simple."""
    # Use gamma_simple (golden ratio based) + epsilon_simple (base 2.0)
    gamma = 1.618033988749895  # Golden ratio
    epsilon = 2.0  # Base frequency

    # Create coordinate grids
    x = np.linspace(-np.pi, np.pi, width)
    y = np.linspace(-np.pi, np.pi, height)
    xx, yy = np.meshgrid(x, y)

    # Create phase-diverse carrier with frequency modulation
    carrier = np.zeros((height, width, 4), dtype=np.float32)

    # Different frequency patterns for each channel
    carrier[:, :, 0] = np.sin(gamma * xx) * np.cos(epsilon * yy)  # X: real
    carrier[:, :, 1] = np.cos(gamma * xx) * np.sin(epsilon * yy)  # Y: imag
    carrier[:, :, 2] = np.sin((gamma + epsilon) * np.sqrt(xx**2 + yy**2)) * 0.5 + 0.5  # Z: weight
    carrier[:, :, 3] = np.cos(gamma * epsilon * (xx + yy)) * 0.5  # W: phase velocity

    # Add some variance to prevent uniformity
    noise = np.random.randn(height, width, 4) * 0.01
    carrier += noise

    return carrier


def simple_fm_modulation(carrier, signal, modulation_depth=0.3):
    """Simple FM modulation without GPU."""
    # Ensure matching shapes
    if carrier.shape != signal.shape:
        signal = np.resize(signal, carrier.shape)

    # FM modulation: carrier * (1 + depth * signal)
    modulated = carrier * (1 + modulation_depth * signal)

    # Add interference patterns
    phase_shift = np.angle(signal[:, :, 0] + 1j * signal[:, :, 1])
    modulated[:, :, 0] *= np.cos(phase_shift)
    modulated[:, :, 1] *= np.sin(phase_shift)

    return modulated


def text_to_frequency_simple(text, width=512, height=512):
    """Simple text to frequency conversion."""
    # Character frequency analysis
    char_freqs = {}
    for char in text.lower():
        char_freqs[char] = char_freqs.get(char, 0) + 1

    # Create frequency spectrum
    spectrum = np.zeros((height, width, 2), dtype=np.float32)

    # Map character frequencies to spatial frequencies
    for i, (char, count) in enumerate(char_freqs.items()):
        if i >= width:
            break

        # Character code determines frequency
        freq = (ord(char) % 128) / 128.0 * np.pi

        # Count determines magnitude
        magnitude = count / len(text)

        # Create frequency pattern
        x = np.linspace(0, 2*np.pi, width)
        y = np.linspace(0, 2*np.pi, height)
        xx, yy = np.meshgrid(x, y)

        pattern = np.sin(freq * xx + i) * np.cos(freq * yy + i) * magnitude
        spectrum[:, :, 0] += pattern

        # Phase based on character position
        phase_pattern = np.angle(np.exp(1j * freq * (xx + yy)))
        spectrum[:, :, 1] += phase_pattern * magnitude

    # Normalize
    spectrum[:, :, 0] = (spectrum[:, :, 0] - np.min(spectrum[:, :, 0])) / (np.max(spectrum[:, :, 0]) - np.min(spectrum[:, :, 0]) + 1e-8)
    spectrum[:, :, 1] = np.mod(spectrum[:, :, 1], 2*np.pi)

    return spectrum


def frequency_to_signal_simple(freq_spectrum):
    """Convert frequency to signal."""
    magnitude = freq_spectrum[:, :, 0]
    phase = freq_spectrum[:, :, 1]

    # Convert to XYZW format
    X = magnitude * np.cos(phase)
    Y = magnitude * np.sin(phase)
    Z = magnitude
    W = np.gradient(phase)[0]  # Phase velocity

    return np.stack([X, Y, Z, W], axis=-1).astype(np.float32)


def encode_text_simple(text, carrier):
    """Simple text encoding without GPU."""
    # Text → Frequency
    freq_spectrum = text_to_frequency_simple(text)

    # Frequency → Signal
    signal = frequency_to_signal_simple(freq_spectrum)

    # Apply carrier filter via FM modulation
    proto = simple_fm_modulation(carrier, signal)

    return proto


def extract_octave_quaternion(proto, octave_level):
    """Extract quaternion at specific octave level."""
    # Downsample to octave resolution
    factor = 2 ** octave_level
    if factor > 1:
        h, w = proto.shape[:2]
        target_h = max(1, h // factor)
        target_w = max(1, w // factor)

        # Simple average pooling
        pooled = proto[:target_h*factor, :target_w*factor]
        pooled = pooled.reshape(target_h, factor, target_w, factor, 4)
        pooled = pooled.mean(axis=(1, 3))
    else:
        pooled = proto

    # Extract quaternion (energy-weighted average)
    weights = pooled[:, :, 2:3]  # Z channel as weight
    weighted = pooled * weights
    quaternion = np.mean(weighted, axis=(0, 1))

    # Normalize
    norm = np.linalg.norm(quaternion) + 1e-8
    quaternion /= norm

    # Reorder to [w, x, y, z]
    quaternion = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])

    return quaternion


def test_blob_problem_simple():
    """Test blob problem with simplified pipeline."""
    print("=" * 70)
    print("Testing Universal Blob Problem - Simplified Pipeline")
    print("=" * 70)

    # Initialize carrier
    carrier = initialize_simple_carrier()
    print(f"\nCarrier initialized:")
    print(f"  Shape: {carrier.shape}")
    print(f"  Variance: {np.var(carrier):.6f}")
    print(f"  Range: [{np.min(carrier):.3f}, {np.max(carrier):.3f}]")

    # Foundation documents
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
        proto = encode_text_simple(text, carrier)
        protos.append(proto)
        print(f"\nDocument {i}: '{text[:50]}...'")
        print(f"  Proto variance: {np.var(proto):.6f}")
        print(f"  Proto range: [{np.min(proto):.3f}, {np.max(proto):.3f}]")

    print("\n" + "=" * 70)
    print("Pairwise Distance Analysis")
    print("=" * 70)

    distances = []
    for i in range(len(protos)):
        for j in range(i+1, len(protos)):
            p1_flat = protos[i].flatten()
            p2_flat = protos[j].flatten()

            cos_dist = cosine(p1_flat, p2_flat)
            euc_dist = euclidean(p1_flat, p2_flat)

            distances.append({'pair': (i, j), 'cosine': cos_dist, 'euclidean': euc_dist})

            print(f"\nDoc {i} vs Doc {j}:")
            print(f"  Cosine distance: {cos_dist:.6f}")
            print(f"  Euclidean distance: {euc_dist:.3f}")

    avg_cosine = np.mean([d['cosine'] for d in distances])

    print("\n" + "=" * 70)
    print("Blob Problem Analysis")
    print("=" * 70)
    print(f"Average cosine distance: {avg_cosine:.6f}")

    if avg_cosine < 0.01:
        print("❌ BLOB PROBLEM DETECTED: Proto-identities are nearly identical!")
    elif avg_cosine < 0.05:
        print("⚠️  PARTIAL BLOB: Proto-identities have minimal differentiation")
    else:
        print("✅ NO BLOB PROBLEM: Proto-identities are well-differentiated!")

    return protos, distances


def test_octave_clustering_simple(protos):
    """Test clustering at different octaves using simple k-means."""
    print("\n" + "=" * 70)
    print("Testing Octave-Level Clustering")
    print("=" * 70)

    from sklearn.cluster import KMeans

    octave_names = {
        0: "Letter",
        3: "Word",
        6: "Sentence",
        9: "Paragraph"
    }

    for octave_level, name in octave_names.items():
        print(f"\n{name} Level (Octave {octave_level}):")
        print("-" * 40)

        # Extract quaternions at this octave
        quaternions = []
        for proto in protos:
            quat = extract_octave_quaternion(proto, octave_level)
            quaternions.append(quat)

        quaternions = np.array(quaternions)

        # Cluster with k-means
        n_clusters = min(2, len(quaternions))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(quaternions)

        print(f"  Quaternion shape: {quaternions.shape}")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Cluster labels: {labels}")

        unique_clusters = len(np.unique(labels))
        if unique_clusters == 1:
            print(f"  ⚠️  All docs in same cluster at {name} level")
        elif unique_clusters == len(protos):
            print(f"  ✅ Each doc in unique cluster at {name} level")
        else:
            print(f"  ℹ️  {unique_clusters} clusters for {len(protos)} docs")


def test_modulation_depth_impact():
    """Test how modulation depth affects differentiation."""
    print("\n" + "=" * 70)
    print("Testing Modulation Depth Impact")
    print("=" * 70)

    carrier = initialize_simple_carrier()
    texts = [
        "The Tao that can be told is not the eternal Tao.",
        "The way of Heaven is to benefit others and not to injure."
    ]

    depths = [0.1, 0.3, 0.5, 0.7, 1.0]

    for depth in depths:
        # Encode with different modulation depth
        protos = []
        for text in texts:
            freq = text_to_frequency_simple(text)
            signal = frequency_to_signal_simple(freq)
            proto = simple_fm_modulation(carrier, signal, modulation_depth=depth)
            protos.append(proto)

        # Compute distance
        cos_dist = cosine(protos[0].flatten(), protos[1].flatten())

        print(f"\nModulation depth: {depth}")
        print(f"  Cosine distance: {cos_dist:.6f}")

        if cos_dist < 0.01:
            print(f"  ❌ Too similar")
        elif cos_dist < 0.05:
            print(f"  ⚠️  Minimal separation")
        else:
            print(f"  ✅ Good separation")


def main():
    """Run simplified tests."""
    print("\n" + "=" * 70)
    print("SIMPLIFIED CLUSTERING & BLOB PROBLEM TEST")
    print("=" * 70)

    # Test blob problem
    protos, distances = test_blob_problem_simple()

    # Test octave clustering
    test_octave_clustering_simple(protos)

    # Test modulation depth impact
    test_modulation_depth_impact()

    # Save results
    results = {
        'avg_cosine': float(np.mean([d['cosine'] for d in distances])),
        'distances': [{'pair': list(d['pair']), 'cosine': float(d['cosine'])} for d in distances]
    }

    with open('/tmp/clustering_simple_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("Results saved to /tmp/clustering_simple_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()