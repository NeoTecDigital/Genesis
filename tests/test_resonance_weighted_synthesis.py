#!/usr/bin/env python3
"""
Test suite for resonance-weighted synthesis in VoxelCloud.

Tests all configurable parameters and weight functions to ensure
high-resonance proto-identities receive higher weights during synthesis.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.memory.voxel_cloud import VoxelCloud, ProtoIdentityEntry
from src.memory.frequency_field import TextFrequencyAnalyzer
from src.origin import Origin


def create_test_proto_identity(text, analyzer, origin):
    """Create a proto-identity from text."""
    freq_spectrum, params = analyzer.analyze(text)
    proto_identity = origin.Gen(params['gamma_params'], params['iota_params'])
    return proto_identity, freq_spectrum, params


def test_resonance_weighting_basic():
    """Test that resonance weighting favors high-resonance protos."""
    print("Testing basic resonance weighting...")

    # Create voxel cloud with resonance weighting enabled
    synthesis_config = {
        'use_resonance_weighting': True,
        'weight_function': 'linear',
        'resonance_boost': 2.0,
        'distance_decay': 0.5
    }
    voxel_cloud = VoxelCloud(512, 512, 128, synthesis_config=synthesis_config)

    # Create test components
    analyzer = TextFrequencyAnalyzer(512, 512)
    origin = Origin(512, 512, use_gpu=False)

    # Add proto-identities with different resonance strengths
    texts = [
        "The pattern repeats",  # Will appear 5 times (high resonance)
        "Another unique pattern",  # Will appear 1 time (low resonance)
        "Different text here"  # Will appear 2 times (medium resonance)
    ]

    # Add first text 5 times (high resonance)
    for _ in range(5):
        proto, freq, params = create_test_proto_identity(texts[0], analyzer, origin)
        voxel_cloud.add(proto, freq, {'text': texts[0], 'params': params})

    # Add second text 1 time (low resonance)
    proto, freq, params = create_test_proto_identity(texts[1], analyzer, origin)
    voxel_cloud.add(proto, freq, {'text': texts[1], 'params': params})

    # Add third text 2 times (medium resonance)
    for _ in range(2):
        proto, freq, params = create_test_proto_identity(texts[2], analyzer, origin)
        voxel_cloud.add(proto, freq, {'text': texts[2], 'params': params})

    # Query with a similar pattern
    query_text = "The pattern"
    query_freq, _ = analyzer.analyze(query_text)
    query_pos = voxel_cloud._frequency_to_position(query_freq)

    # Get visible protos (should be all 3 unique ones due to collapse)
    visible_protos = voxel_cloud.query_viewport(query_freq, radius=1000.0)

    # Compute weights
    weights = voxel_cloud._compute_synthesis_weights(visible_protos, query_pos)

    # Find which proto has highest weight
    max_weight_idx = np.argmax(weights)
    highest_weighted = visible_protos[max_weight_idx]

    print(f"  Number of unique protos: {len(visible_protos)}")
    print(f"  Resonance strengths: {[p.resonance_strength for p in visible_protos]}")
    print(f"  Weights: {weights}")
    print(f"  Highest weighted proto resonance: {highest_weighted.resonance_strength}")

    # The proto with resonance_strength=5 should have highest weight
    assert highest_weighted.resonance_strength >= 3, \
        f"Expected high-resonance proto to dominate, got resonance={highest_weighted.resonance_strength}"

    print("  ✓ High-resonance proto received highest weight")


def test_weight_functions():
    """Test different weight functions (linear, sqrt, log)."""
    print("\nTesting weight functions...")

    analyzer = TextFrequencyAnalyzer(512, 512)
    origin = Origin(512, 512, use_gpu=False)

    # Test each weight function
    for weight_func in ['linear', 'sqrt', 'log']:
        print(f"\n  Testing {weight_func} weight function:")

        synthesis_config = {
            'use_resonance_weighting': True,
            'weight_function': weight_func,
            'resonance_boost': 2.0,
            'distance_decay': 0.5
        }
        voxel_cloud = VoxelCloud(512, 512, 128, synthesis_config=synthesis_config)

        # Add protos with varying resonance
        for i, count in enumerate([10, 5, 1]):
            text = f"Pattern {i}"
            for _ in range(count):
                proto, freq, params = create_test_proto_identity(text, analyzer, origin)
                voxel_cloud.add(proto, freq, {'text': text, 'params': params})

        # Query and get weights
        query_freq, _ = analyzer.analyze("Pattern query")
        query_pos = voxel_cloud._frequency_to_position(query_freq)
        visible_protos = voxel_cloud.query_viewport(query_freq, radius=1000.0)
        weights = voxel_cloud._compute_synthesis_weights(visible_protos, query_pos)

        # Sort by resonance for display
        sorted_pairs = sorted(zip(visible_protos, weights),
                            key=lambda x: x[0].resonance_strength,
                            reverse=True)

        print(f"    Resonance → Weight mapping:")
        for proto, weight in sorted_pairs:
            print(f"      Resonance {proto.resonance_strength:2d} → Weight {weight:.4f}")

        # Verify weights are monotonic with resonance
        resonances = [p.resonance_strength for p, _ in sorted_pairs]
        weights_sorted = [w for _, w in sorted_pairs]

        assert all(w1 >= w2 for w1, w2 in zip(weights_sorted[:-1], weights_sorted[1:])), \
            f"Weights not monotonic for {weight_func}"

        print(f"    ✓ {weight_func} function working correctly")


def test_resonance_boost_parameter():
    """Test the resonance_boost parameter effect."""
    print("\nTesting resonance_boost parameter...")

    analyzer = TextFrequencyAnalyzer(512, 512)
    origin = Origin(512, 512, use_gpu=False)

    # Test different boost values
    for boost in [0.0, 1.0, 5.0, 10.0]:
        print(f"\n  Testing resonance_boost={boost}:")

        synthesis_config = {
            'use_resonance_weighting': True,
            'weight_function': 'linear',
            'resonance_boost': boost,
            'distance_decay': 1.0  # Keep distance constant
        }
        voxel_cloud = VoxelCloud(512, 512, 128, synthesis_config=synthesis_config)

        # Add two protos with different resonance
        texts = ["High resonance", "Low resonance"]

        # High resonance (10 times)
        for _ in range(10):
            proto, freq, params = create_test_proto_identity(texts[0], analyzer, origin)
            voxel_cloud.add(proto, freq, {'text': texts[0], 'params': params})

        # Low resonance (1 time)
        proto, freq, params = create_test_proto_identity(texts[1], analyzer, origin)
        voxel_cloud.add(proto, freq, {'text': texts[1], 'params': params})

        # Query and compute weights
        query_freq, _ = analyzer.analyze("resonance test")
        query_pos = voxel_cloud._frequency_to_position(query_freq)
        visible_protos = voxel_cloud.query_viewport(query_freq, radius=1000.0)
        weights = voxel_cloud._compute_synthesis_weights(visible_protos, query_pos)

        # Calculate weight ratio
        high_res_idx = 0 if visible_protos[0].resonance_strength > visible_protos[1].resonance_strength else 1
        low_res_idx = 1 - high_res_idx

        weight_ratio = weights[high_res_idx] / weights[low_res_idx] if weights[low_res_idx] > 0 else float('inf')

        print(f"    High/Low weight ratio: {weight_ratio:.2f}")
        print(f"    High resonance weight: {weights[high_res_idx]:.4f}")
        print(f"    Low resonance weight: {weights[low_res_idx]:.4f}")

        # Higher boost should increase the ratio
        if boost > 0:
            assert weight_ratio > 1.0, f"High resonance should dominate with boost={boost}"

        print(f"    ✓ Boost={boost} working as expected")


def test_distance_decay_parameter():
    """Test the distance_decay parameter effect."""
    print("\nTesting distance_decay parameter...")

    analyzer = TextFrequencyAnalyzer(512, 512)
    origin = Origin(512, 512, use_gpu=False)

    # Test different decay values
    for decay in [0.0, 0.5, 1.0]:
        print(f"\n  Testing distance_decay={decay}:")

        synthesis_config = {
            'use_resonance_weighting': True,
            'weight_function': 'linear',
            'resonance_boost': 1.0,  # Keep resonance constant
            'distance_decay': decay
        }
        voxel_cloud = VoxelCloud(512, 512, 128, synthesis_config=synthesis_config)

        # Add protos at different positions with same resonance
        texts = ["Near proto", "Far proto"]

        # Add both with same resonance (2 times each)
        for text in texts:
            for _ in range(2):
                proto, freq, params = create_test_proto_identity(text, analyzer, origin)
                voxel_cloud.add(proto, freq, {'text': text, 'params': params})

        # Query close to first proto
        query_freq, _ = analyzer.analyze(texts[0])
        query_pos = voxel_cloud._frequency_to_position(query_freq)
        visible_protos = voxel_cloud.query_viewport(query_freq, radius=1000.0)

        if len(visible_protos) >= 2:
            weights = voxel_cloud._compute_synthesis_weights(visible_protos, query_pos)

            # Find near and far protos
            distances = [np.linalg.norm(p.position - query_pos) for p in visible_protos]
            near_idx = np.argmin(distances)
            far_idx = np.argmax(distances)

            print(f"    Near distance: {distances[near_idx]:.2f}")
            print(f"    Far distance: {distances[far_idx]:.2f}")
            print(f"    Near weight: {weights[near_idx]:.4f}")
            print(f"    Far weight: {weights[far_idx]:.4f}")

            # With decay=0, distance shouldn't matter much
            # With decay=1, near should have higher weight
            if decay > 0.5 and distances[far_idx] > distances[near_idx] * 2:
                assert weights[near_idx] > weights[far_idx], \
                    f"Near proto should have higher weight with decay={decay}"

            print(f"    ✓ Decay={decay} working as expected")


def test_collapse_config_disable():
    """Test that disabling collapse creates separate entries."""
    print("\nTesting collapse disable...")

    # Create voxel cloud with collapse disabled
    collapse_config = {
        'enable': False,
        'harmonic_tolerance': 0.05,
        'cosine_threshold': 0.85,
        'octave_tolerance': 0
    }
    voxel_cloud = VoxelCloud(512, 512, 128, collapse_config=collapse_config)

    analyzer = TextFrequencyAnalyzer(512, 512)
    origin = Origin(512, 512, use_gpu=False)

    # Add same text multiple times
    text = "Repeated pattern"
    for _ in range(3):
        proto, freq, params = create_test_proto_identity(text, analyzer, origin)
        voxel_cloud.add(proto, freq, {'text': text, 'params': params})

    print(f"  Entries with collapse disabled: {len(voxel_cloud)}")
    assert len(voxel_cloud) == 3, "Should have 3 separate entries with collapse disabled"

    # Now test with collapse enabled (default)
    voxel_cloud_enabled = VoxelCloud(512, 512, 128)

    for _ in range(3):
        proto, freq, params = create_test_proto_identity(text, analyzer, origin)
        voxel_cloud_enabled.add(proto, freq, {'text': text, 'params': params})

    print(f"  Entries with collapse enabled: {len(voxel_cloud_enabled)}")
    assert len(voxel_cloud_enabled) == 1, "Should have 1 merged entry with collapse enabled"
    assert voxel_cloud_enabled.entries[0].resonance_strength == 3, "Should have resonance_strength=3"

    print("  ✓ Collapse configuration working correctly")


def test_backward_compatibility():
    """Test that loading old voxel clouds works with defaults."""
    print("\nTesting backward compatibility...")

    import pickle
    import tempfile

    # Create old-style voxel cloud data
    old_data = {
        'width': 512,
        'height': 512,
        'depth': 128,
        'entries': [],
        'spatial_index': {}
    }

    # Save and reload
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        pickle.dump(old_data, f)
        temp_path = f.name

    try:
        # Load with new VoxelCloud
        voxel_cloud = VoxelCloud()
        voxel_cloud.load(temp_path)

        # Check that default configs are loaded
        assert voxel_cloud.collapse_config['enable'] == True
        assert voxel_cloud.synthesis_config['use_resonance_weighting'] == True
        assert voxel_cloud.synthesis_config['weight_function'] == 'linear'

        print("  ✓ Backward compatibility maintained")
    finally:
        os.unlink(temp_path)


def test_synthesis_comparison():
    """Compare weighted vs unweighted synthesis results."""
    print("\nComparing weighted vs unweighted synthesis...")

    analyzer = TextFrequencyAnalyzer(512, 512)
    origin = Origin(512, 512, use_gpu=False)

    # Create two voxel clouds - one with, one without resonance weighting
    config_weighted = {
        'use_resonance_weighting': True,
        'weight_function': 'linear',
        'resonance_boost': 5.0,
        'distance_decay': 0.1
    }

    config_unweighted = {
        'use_resonance_weighting': False
    }

    cloud_weighted = VoxelCloud(512, 512, 128, synthesis_config=config_weighted)
    cloud_unweighted = VoxelCloud(512, 512, 128, synthesis_config=config_unweighted)

    # Add same data to both
    texts_counts = [
        ("Common pattern", 10),
        ("Rare pattern", 1),
        ("Medium pattern", 5)
    ]

    for text, count in texts_counts:
        for _ in range(count):
            proto, freq, params = create_test_proto_identity(text, analyzer, origin)
            cloud_weighted.add(proto, freq, {'text': text, 'params': params})
            cloud_unweighted.add(proto, freq, {'text': text, 'params': params})

    # Query both
    query_freq, _ = analyzer.analyze("test query")

    visible_weighted = cloud_weighted.query_viewport(query_freq, radius=1000.0)
    visible_unweighted = cloud_unweighted.query_viewport(query_freq, radius=1000.0)

    synth_weighted = cloud_weighted.synthesize(visible_weighted, query_freq)
    synth_unweighted = cloud_unweighted.synthesize(visible_unweighted, query_freq)

    # Compare - weighted should be different from unweighted
    diff = np.mean(np.abs(synth_weighted - synth_unweighted))

    print(f"  Mean absolute difference: {diff:.6f}")
    print(f"  Weighted synthesis shape: {synth_weighted.shape}")
    print(f"  Unweighted synthesis shape: {synth_unweighted.shape}")

    # They should be different (unless by extreme chance)
    assert diff > 0, "Weighted and unweighted synthesis should differ"

    print("  ✓ Weighted synthesis produces different results")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Resonance-Weighted Synthesis Test Suite")
    print("=" * 60)

    try:
        test_resonance_weighting_basic()
        test_weight_functions()
        test_resonance_boost_parameter()
        test_distance_decay_parameter()
        test_collapse_config_disable()
        test_backward_compatibility()
        test_synthesis_comparison()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())