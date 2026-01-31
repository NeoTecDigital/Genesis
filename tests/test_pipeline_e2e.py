#!/usr/bin/env python3
"""End-to-end pipeline test: Input → Memory → Output.

Tests the complete flow:
1. Text → Encode → Proto-identity
2. Proto-identity → VoxelCloud storage
3. Query → Retrieve → Decode → Text
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline.encoding import EncodingPipeline
from src.pipeline.decoding import DecodingPipeline
from src.memory.voxel_cloud import VoxelCloud
from src.memory.entropy_indexing import analyze_proto_entropy


def test_basic_encoding_decoding():
    """Test basic encode → decode cycle."""
    print("\n=== TEST 1: Basic Encoding/Decoding ===")

    # Initialize carrier (proto-unity)
    carrier = np.random.randn(512, 512, 4).astype(np.float32)
    carrier = np.tanh(carrier)  # Normalize

    encoder = EncodingPipeline(carrier)
    decoder = DecodingPipeline(carrier)

    # Test text
    original_text = "Hello World"

    # Encode
    proto, metadata = encoder.encode_text(original_text)
    print(f"Original: '{original_text}'")
    print(f"Proto shape: {proto.shape}")
    print(f"Has native_stft: {'native_stft' in metadata}")

    # Decode
    decoded_text = decoder.decode_to_text(proto, metadata)
    print(f"Decoded: '{decoded_text}'")

    # Check reconstruction
    success = (decoded_text == original_text)
    print(f"✅ Lossless reconstruction: {success}" if success else f"❌ Reconstruction failed")

    return success


def test_memory_storage_retrieval():
    """Test VoxelCloud storage and retrieval."""
    print("\n=== TEST 2: Memory Storage/Retrieval ===")

    # Initialize
    carrier = np.random.randn(512, 512, 4).astype(np.float32)
    carrier = np.tanh(carrier)

    encoder = EncodingPipeline(carrier)
    decoder = DecodingPipeline(carrier)
    voxel_cloud = VoxelCloud()

    # Store multiple texts
    texts = [
        "The quick brown fox",
        "jumps over the lazy dog",
        "Machine learning is fascinating"
    ]

    for text in texts:
        proto, metadata = encoder.encode_text(text)

        # Extract frequency spectrum for VoxelCloud
        freq_spectrum = np.zeros((512, 512, 2), dtype=np.float32)
        freq_spectrum[:, :, 0] = proto[:, :, 2]  # Use Z channel as magnitude
        freq_spectrum[:, :, 1] = np.arctan2(proto[:, :, 1], proto[:, :, 0])  # Phase

        voxel_cloud.add(proto, freq_spectrum, metadata)

    print(f"Stored {len(voxel_cloud)} proto-identities")

    # Query similar to first text
    query_text = "quick brown"
    query_proto, query_metadata = encoder.encode_text(query_text)

    # Query voxel cloud
    results = voxel_cloud.query_by_proto_similarity(query_proto, max_results=3)

    print(f"\nQuery: '{query_text}'")
    print(f"Retrieved {len(results)} results:")

    for i, entry in enumerate(results):
        # Decode retrieved proto
        retrieved_text = decoder.decode_to_text(entry.proto_identity, entry.metadata)
        print(f"  {i+1}. '{retrieved_text}'")

    success = len(results) > 0
    print(f"✅ Memory retrieval working" if success else f"❌ Memory retrieval failed")

    return success


def test_entropy_indexing():
    """Test entropy-based semantic indexing."""
    print("\n=== TEST 3: Entropy-Based Indexing ===")

    # Initialize
    carrier = np.random.randn(512, 512, 4).astype(np.float32)
    carrier = np.tanh(carrier)

    encoder = EncodingPipeline(carrier)

    # Encode texts at different octaves
    texts_and_octaves = [
        ("a", 4),  # Character level
        ("hello", 0),  # Word level
        ("the quick brown fox", -2),  # Phrase level
    ]

    entropy_results = []

    for text, octave in texts_and_octaves:
        proto, metadata = encoder.encode_text(text)

        # Extract frequency
        freq_spectrum = np.zeros((512, 512, 2), dtype=np.float32)
        freq_spectrum[:, :, 0] = proto[:, :, 2]
        freq_spectrum[:, :, 1] = np.arctan2(proto[:, :, 1], proto[:, :, 0])

        # Analyze entropy
        metrics = analyze_proto_entropy(proto, freq_spectrum, octave)
        entropy_results.append((text, octave, metrics))

        print(f"Text: '{text}' (octave {octave:+2d})")
        print(f"  Entropy: {metrics.entropy:.2f}")
        print(f"  Cluster ID: {metrics.cluster_id}")

    # Check octave separation
    cluster_ids = [m.cluster_id for _, _, m in entropy_results]
    octave_separation = len(set(cluster_ids)) == len(cluster_ids)

    print(f"\n✅ Octave separation preserved" if octave_separation else f"❌ Octave mixing detected")

    return octave_separation


def test_full_pipeline():
    """Test complete input → memory → output pipeline."""
    print("\n=== TEST 4: Full Pipeline (Input → Memory → Output) ===")

    # Initialize components
    carrier = np.random.randn(512, 512, 4).astype(np.float32)
    carrier = np.tanh(carrier)

    encoder = EncodingPipeline(carrier)
    decoder = DecodingPipeline(carrier)
    voxel_cloud = VoxelCloud()

    # INPUT: Store foundation knowledge
    foundation_texts = [
        "Water is H2O",
        "The sky is blue",
        "Python is a programming language"
    ]

    print("1. Encoding foundation knowledge...")
    for text in foundation_texts:
        proto, metadata = encoder.encode_text(text)

        # Create frequency spectrum
        freq_spectrum = np.zeros((512, 512, 2), dtype=np.float32)
        freq_spectrum[:, :, 0] = proto[:, :, 2]
        freq_spectrum[:, :, 1] = np.arctan2(proto[:, :, 1], proto[:, :, 0])

        voxel_cloud.add(proto, freq_spectrum, metadata)

    print(f"   Stored {len(voxel_cloud)} protos in memory")

    # MEMORY: Query for relevant context
    query_text = "programming"
    print(f"\n2. Querying memory with: '{query_text}'")

    query_proto, query_metadata = encoder.encode_text(query_text)
    results = voxel_cloud.query_by_proto_similarity(query_proto, max_results=3)

    print(f"   Retrieved {len(results)} relevant memories")

    # OUTPUT: Decode results
    print("\n3. Decoding results:")
    decoded_texts = []
    for i, entry in enumerate(results):
        decoded = decoder.decode_to_text(entry.proto_identity, entry.metadata)
        decoded_texts.append(decoded)
        print(f"   {i+1}. '{decoded}'")

    # Verify we got relevant result
    success = any("Python" in text for text in decoded_texts)

    print(f"\n✅ Full pipeline working - retrieved relevant context" if success
          else f"❌ Pipeline failed - no relevant context retrieved")

    return success


def main():
    """Run all pipeline tests."""
    print("=" * 70)
    print("END-TO-END PIPELINE VALIDATION")
    print("=" * 70)
    print("\nTesting: Input → Memory → Output")

    results = []

    results.append(("Basic Encoding/Decoding", test_basic_encoding_decoding()))
    results.append(("Memory Storage/Retrieval", test_memory_storage_retrieval()))
    results.append(("Entropy-Based Indexing", test_entropy_indexing()))
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
        print("\n✅ PIPELINE FULLY OPERATIONAL")
        print("Input → Memory → Output working correctly!")
    else:
        print(f"\n❌ {total - passed} TESTS FAILED")
        print("Pipeline needs fixes")

    # Save results
    output_dir = Path('./output')
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'pipeline_e2e_results.txt', 'w') as f:
        f.write("END-TO-END PIPELINE VALIDATION\n")
        f.write("=" * 70 + "\n\n")
        for test_name, result in results:
            status = "PASS" if result else "FAIL"
            f.write(f"[{status}] {test_name}\n")
        f.write(f"\nTotal: {passed}/{total} passed\n")

    print(f"\nResults saved to {output_dir / 'pipeline_e2e_results.txt'}")


if __name__ == '__main__':
    main()
