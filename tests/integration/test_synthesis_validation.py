#!/usr/bin/env python
"""
Comprehensive validation and fine-tuning of Text → Quaternions → Synthesis → Text pipeline.

Tests:
1. Parameter tuning for Gen/Res convergence
2. Query → Synthesis → Output quality
3. Reverse path: Quaternions → Signal → Text
"""

import sys
import os
import numpy as np
import pickle

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.origin import Origin
from src.memory.frequency_field import TextFrequencyAnalyzer
from src.memory.voxel_cloud import VoxelCloud

def test_parameter_tuning():
    """Fine-tune morphism parameters for stable convergence."""
    print("🔧 Parameter Tuning: Finding optimal ranges")
    print("=" * 60)

    origin = Origin(width=512, height=512, use_gpu=False)
    analyzer = TextFrequencyAnalyzer(width=512, height=512)

    test_texts = [
        "The Tao that can be told is not the eternal Tao",
        "What is the Tao?",
        "Emptiness and fullness arise together"
    ]

    successful_params = []

    for text in test_texts:
        print(f"\nTesting: '{text[:50]}...'")

        # Get frequency parameters
        freq_spectrum, params = analyzer.analyze(text)

        # Test Gen path
        try:
            proto_gen = origin.Gen(params['gamma_params'], params['iota_params'], input_n=None)
            gen_energy = np.linalg.norm(proto_gen)
            print(f"  ✅ Gen successful - energy: {gen_energy:.4f}")
        except Exception as e:
            print(f"  ❌ Gen failed: {e}")
            continue

        # Test Res path with relaxed validation
        # Try multiple epsilon parameter variations
        for amplitude_scale in [0.5, 0.75, 1.0, 1.25, 1.5]:
            try:
                # Modify epsilon params
                eps_params = params['epsilon_params'].copy()
                eps_params['extraction_rate'] = np.clip(
                    eps_params['extraction_rate'] * amplitude_scale,
                    0.05, 0.95
                )

                # Temporarily disable validation for testing
                proto_res = origin.Res(eps_params, params['tau_params'], input_n=None)
                res_energy = np.linalg.norm(proto_res)

                # Check convergence
                proto_unified = proto_gen + proto_res
                conv_similarity = np.dot(proto_gen.flatten(), proto_res.flatten()) / \
                                (np.linalg.norm(proto_gen) * np.linalg.norm(proto_res))

                print(f"  ✅ Res (scale={amplitude_scale:.2f}) - energy: {res_energy:.4f}, conv: {conv_similarity:.4f}")

                successful_params.append({
                    'text': text,
                    'gamma_params': params['gamma_params'],
                    'epsilon_params': eps_params,
                    'convergence': conv_similarity,
                    'gen_energy': gen_energy,
                    'res_energy': res_energy
                })

                # If good convergence, stop trying
                if abs(conv_similarity) > 0.7:
                    break

            except Exception as e:
                print(f"  ❌ Res (scale={amplitude_scale:.2f}) failed: {e}")
                continue

    # Analyze successful parameters
    if successful_params:
        print("\n📊 Successful Parameter Ranges:")
        print("-" * 60)

        base_freqs = [p['gamma_params']['base_frequency'] for p in successful_params]
        amplitudes = [p['gamma_params']['amplitude'] for p in successful_params]
        extract_rates = [p['epsilon_params']['extraction_rate'] for p in successful_params]
        convergences = [p['convergence'] for p in successful_params]

        print(f"Base frequency: {min(base_freqs):.2f} - {max(base_freqs):.2f} Hz (mean: {np.mean(base_freqs):.2f})")
        print(f"Amplitude: {min(amplitudes):.2f} - {max(amplitudes):.2f} (mean: {np.mean(amplitudes):.2f})")
        print(f"Extraction rate: {min(extract_rates):.2f} - {max(extract_rates):.2f} (mean: {np.mean(extract_rates):.2f})")
        print(f"Convergence: {min(convergences):.4f} - {max(convergences):.4f} (mean: {np.mean(convergences):.4f})")

        return successful_params
    else:
        print("\n❌ No successful parameter combinations found!")
        return []

def test_synthesis_pipeline():
    """Test complete synthesis pipeline: Query → Quaternions → Synthesis → Output"""
    print("\n🔬 Testing Synthesis Pipeline")
    print("=" * 60)

    origin = Origin(width=512, height=512, use_gpu=False)
    analyzer = TextFrequencyAnalyzer(width=512, height=512)
    cloud = VoxelCloud(width=512, height=512, depth=128)

    # Add some test proto-identities
    test_memories = [
        "The Tao that can be told is not the eternal Tao",
        "The name that can be named is not the eternal name",
        "The nameless is the beginning of heaven and earth",
        "The named is the mother of ten thousand things",
        "Ever desireless, one can see the mystery"
    ]

    print("\n📥 Adding memories to voxel cloud...")
    for text in test_memories:
        freq_spectrum, params = analyzer.analyze(text)

        # Gen path only (Res failing)
        try:
            proto = origin.Gen(params['gamma_params'], params['iota_params'], input_n=None)
            result = origin.Act(proto)

            cloud.add_with_octaves(
                proto_identity=result.proto_identity,
                frequency=params['gamma_params']['base_frequency'],
                modality='text',
                quaternions=result.multi_octave_quaternions,
                resonance_strength=1.0
            )
            print(f"  ✅ Added: {text[:60]}...")
        except Exception as e:
            print(f"  ❌ Failed to add: {e}")

    print(f"\n📊 Voxel cloud populated: {len(cloud.entries)} proto-identities")

    # Test queries
    queries = [
        "What is the Tao?",
        "Tell me about the nameless",
        "What is the beginning?"
    ]

    print("\n🔍 Testing queries:")
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 60)

        # Convert query to quaternions
        freq_spectrum, params = analyzer.analyze(query)
        query_proto = origin.Gen(params['gamma_params'], params['iota_params'], input_n=None)
        query_result = origin.Act(query_proto)

        # Adaptive octave selection
        query_octave = cloud.octave_hierarchy.adaptive_octave_selection(query)
        print(f"Adaptive octave: {query_octave} (0=detail, 4=abstract)")

        # Multi-octave query
        results = cloud.query_multi_octave(query_result.multi_octave_quaternions, top_k=3)

        print(f"\nTop {len(results)} results:")
        for i, (proto, distance) in enumerate(results, 1):
            print(f"{i}. Distance: {distance:.4f}")
            print(f"   Modality: {proto.modality}, Frequency: {proto.frequency:.2f} Hz")

        # Synthesis: Weighted combination with exponential decay
        if results:
            print("\n✨ Synthesized response:")
            # Add exponential decay weighting with decay factor
            # Scale distances to reasonable range first (distances are very small)
            distances = np.array([d for _, d in results])
            if distances.max() > 0:
                distances = distances / distances.max()  # Normalize to [0, 1]
            decay_factor = 5.0
            weights = np.exp(-distances * decay_factor)

            # Optional: minimum weight threshold
            min_weight = 0.01
            weights = np.maximum(weights, min_weight)

            # Normalize
            weights /= weights.sum()

            for (proto, distance), weight in zip(results, weights):
                print(f"  [{weight*100:.1f}%] Freq: {proto.frequency:.2f} Hz, Modality: {proto.modality}")

def test_reverse_synthesis():
    """Test reverse path: Quaternions → Signal → Text"""
    print("\n🔄 Testing Reverse Synthesis: Quaternions → Text")
    print("=" * 60)

    origin = Origin(width=512, height=512, use_gpu=False)

    # Start with a known quaternion
    test_quaternion = np.array([0.5, 0.5, 0.5, 0.5])
    test_quaternion /= np.linalg.norm(test_quaternion)  # Normalize

    print(f"Input quaternion: [{test_quaternion[0]:.4f}, {test_quaternion[1]:.4f}, "
          f"{test_quaternion[2]:.4f}, {test_quaternion[3]:.4f}]")
    print(f"Norm: {np.linalg.norm(test_quaternion):.6f}")

    # TODO: Implement reverse synthesis
    # This would require:
    # 1. Quaternion → Proto-identity (inverse Act)
    # 2. Proto-identity → Signal (inverse Gen/Res)
    # 3. Signal → Frequency spectrum (inverse morphisms)
    # 4. Frequency spectrum → Text (inverse frequency analysis)

    print("\n⚠️  Reverse synthesis not yet implemented")
    print("Required components:")
    print("  1. Act⁻¹: Quaternion → Proto-identity (reconstruct standing wave)")
    print("  2. Gen⁻¹: Proto-identity → Signal (extract parameters)")
    print("  3. Frequency⁻¹: Signal → Text (decode to characters)")

if __name__ == "__main__":
    print("=" * 60)
    print("Genesis Synthesis Validation & Fine-Tuning")
    print("=" * 60)

    # 1. Parameter tuning
    successful_params = test_parameter_tuning()

    # 2. Synthesis pipeline
    test_synthesis_pipeline()

    # 3. Reverse synthesis (future)
    test_reverse_synthesis()

    print("\n" + "=" * 60)
    print("Validation Complete")
    print("=" * 60)

    # Summary
    print("\n📋 Summary:")
    print(f"  Successful parameter combinations: {len(successful_params)}")
    print("\n💡 Next steps:")
    print("  1. Relax Res() standing wave validation")
    print("  2. Implement reverse synthesis (Quaternions → Text)")
    print("  3. Fine-tune parameter ranges based on findings")
    print("  4. Test on full Tao Te Ching dataset")
