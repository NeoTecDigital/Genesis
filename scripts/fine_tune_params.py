#!/usr/bin/env python3
"""
A/B testing script for fine-tuning wave physics parameters.

Tests different parameter combinations and evaluates Q&A quality.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import subprocess
import json
import time
import pickle
from pathlib import Path
from itertools import product
from typing import Dict, List, Tuple, Optional

from src.memory.voxel_cloud import VoxelCloud
from src.memory.frequency_field import TextFrequencyAnalyzer


def run_training(params: Dict[str, any], output_suffix: str) -> Tuple[bool, str]:
    """Run training with specific parameters."""
    output_path = f"/tmp/genesis_test_{output_suffix}.pkl"

    cmd = [
        "python", "scripts/train_foundation.py",
        "--output", output_path,
        "--harmonic-tolerance", str(params['harmonic_tolerance']),
        "--cosine-threshold", str(params['cosine_threshold']),
        "--checkpoint-interval", "10"  # More frequent checkpoints for testing
    ]

    if params['enable_collapse']:
        cmd.append("--enable-collapse")

    print(f"  Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode != 0:
            print(f"  ❌ Training failed: {result.stderr}")
            return False, output_path

        return True, output_path

    except subprocess.TimeoutExpired:
        print("  ❌ Training timed out")
        return False, output_path
    except Exception as e:
        print(f"  ❌ Training error: {e}")
        return False, output_path


def evaluate_qa_quality(model_path: str, test_queries: List[Dict]) -> Dict[str, float]:
    """Evaluate Q&A quality on test queries."""
    try:
        # Load model
        with open(model_path, 'rb') as f:
            voxel_cloud = pickle.load(f)

        freq_analyzer = TextFrequencyAnalyzer(512, 512)

        # Track metrics
        total_score = 0
        response_lengths = []
        keyword_matches = 0
        empty_responses = 0

        for query_info in test_queries:
            query = query_info['query']
            expected_keywords = query_info['keywords']

            # Get response
            query_freq, _ = freq_analyzer.analyze(query)
            visible_protos = voxel_cloud.query_viewport(query_freq, radius=50.0)

            if not visible_protos:
                empty_responses += 1
                continue

            # Extract text from top protos
            # TODO Phase 11: Derive text from frequency spectrum
            # For parameter tuning, use placeholder response
            response = "Text derivation pending"
            response_lengths.append(len(response))

            # Check for keyword matches
            if any(kw in response.lower() for kw in expected_keywords):
                keyword_matches += 1
                total_score += 1

        # Calculate metrics
        num_protos = len(voxel_cloud.entries)
        resonances = sum(1 for p in voxel_cloud.entries if p.resonance_strength > 1)

        metrics = {
            'qa_accuracy': keyword_matches / len(test_queries) if test_queries else 0,
            'avg_response_length': sum(response_lengths) / len(response_lengths) if response_lengths else 0,
            'empty_response_rate': empty_responses / len(test_queries) if test_queries else 0,
            'num_protos': num_protos,
            'resonance_rate': resonances / num_protos if num_protos > 0 else 0,
            'compression_ratio': sum(p.resonance_strength for p in voxel_cloud.entries) / num_protos if num_protos > 0 else 1
        }

        return metrics

    except Exception as e:
        print(f"  ❌ Evaluation error: {e}")
        return {
            'qa_accuracy': 0,
            'avg_response_length': 0,
            'empty_response_rate': 1,
            'num_protos': 0,
            'resonance_rate': 0,
            'compression_ratio': 1
        }


def get_test_queries() -> List[Dict]:
    """Get a subset of test queries for quick evaluation."""
    return [
        # Philosophy
        {'query': 'What is emptiness?', 'keywords': ['empty', 'void', 'nothing', 'tao']},
        {'query': 'What is dharma?', 'keywords': ['duty', 'righteousness', 'path', 'law']},
        {'query': 'What causes suffering?', 'keywords': ['desire', 'attachment', 'pain', 'loss']},
        {'query': 'What is virtue?', 'keywords': ['virtue', 'wisdom', 'justice', 'courage']},
        {'query': 'What is enlightenment?', 'keywords': ['wisdom', 'knowledge', 'truth', 'liberation']},

        # History
        {'query': 'Who was Achilles?', 'keywords': ['warrior', 'hero', 'greek', 'troy']},
        {'query': 'What did Gilgamesh seek?', 'keywords': ['immortality', 'eternal', 'life', 'death']},
        {'query': 'What was the Trojan War?', 'keywords': ['troy', 'greek', 'war', 'helen']},
        {'query': 'What are the Canterbury Tales?', 'keywords': ['pilgrim', 'tale', 'story', 'canterbury']},
        {'query': 'Who is Alice?', 'keywords': ['alice', 'wonderland', 'rabbit', 'queen']},

        # Strategy
        {'query': 'What is the art of war?', 'keywords': ['war', 'strategy', 'enemy', 'sun']},
        {'query': 'What are the laws of power?', 'keywords': ['power', 'law', 'enemy', 'friend']},
        {'query': 'How do you build habits?', 'keywords': ['habit', 'small', 'compound', 'system']},
        {'query': 'What makes a good leader?', 'keywords': ['leader', 'wisdom', 'courage', 'vision']},
        {'query': 'What is good strategy?', 'keywords': ['strategy', 'plan', 'goal', 'advantage']}
    ]


def main():
    parser = argparse.ArgumentParser(description='Fine-tune Genesis wave physics parameters')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with fewer parameter combinations')
    parser.add_argument('--output-report', type=str,
                       default='parameter_tuning_report.json',
                       help='Output report file')
    args = parser.parse_args()

    print("=" * 70)
    print("Genesis Parameter Fine-Tuning")
    print("=" * 70)

    # Parameter grid
    if args.quick:
        # Quick test with fewer combinations
        harmonic_tolerances = [0.05, 0.10]
        cosine_thresholds = [0.80, 0.90]
        enable_collapse_options = [True]
    else:
        # Full grid search
        harmonic_tolerances = [0.03, 0.05, 0.07, 0.10]
        cosine_thresholds = [0.75, 0.80, 0.85, 0.90]
        enable_collapse_options = [True, False]

    # Get test queries
    test_queries = get_test_queries()
    print(f"Using {len(test_queries)} test queries for evaluation")

    results = []
    best_score = 0
    best_config = None

    # Test each parameter combination
    total_combos = len(harmonic_tolerances) * len(cosine_thresholds) * len(enable_collapse_options)
    combo_num = 0

    for ht, ct, enable in product(harmonic_tolerances, cosine_thresholds, enable_collapse_options):
        combo_num += 1
        print(f"\n[{combo_num}/{total_combos}] Testing configuration:")
        print(f"  Harmonic tolerance: {ht}")
        print(f"  Cosine threshold: {ct}")
        print(f"  Collapse enabled: {enable}")

        params = {
            'harmonic_tolerance': ht,
            'cosine_threshold': ct,
            'enable_collapse': enable
        }

        # Create unique suffix for this test
        suffix = f"ht{ht}_ct{ct}_{'collapse' if enable else 'nocollapse'}"

        # Train with these parameters
        print("  Training model...")
        start_time = time.time()
        success, model_path = run_training(params, suffix)

        if not success:
            print("  ⚠️  Training failed, skipping evaluation")
            continue

        train_time = time.time() - start_time

        # Evaluate Q&A quality
        print("  Evaluating Q&A quality...")
        metrics = evaluate_qa_quality(model_path, test_queries)

        # Add configuration to metrics
        result = {
            **params,
            **metrics,
            'train_time': train_time,
            'model_path': model_path
        }

        results.append(result)

        # Track best configuration
        if metrics['qa_accuracy'] > best_score:
            best_score = metrics['qa_accuracy']
            best_config = result

        # Print current result
        print(f"  Results:")
        print(f"    Q&A accuracy: {metrics['qa_accuracy']:.2%}")
        print(f"    Compression ratio: {metrics['compression_ratio']:.2f}x")
        print(f"    Proto-identities: {metrics['num_protos']:,}")
        print(f"    Resonance rate: {metrics['resonance_rate']:.2%}")
        print(f"    Training time: {train_time:.1f}s")

        # Clean up test model
        try:
            Path(model_path).unlink()
        except:
            pass

    # Report results
    print("\n" + "=" * 70)
    print("Fine-Tuning Complete!")
    print("=" * 70)

    if best_config:
        print("\n🏆 Best Configuration:")
        print(f"  Harmonic tolerance: {best_config['harmonic_tolerance']}")
        print(f"  Cosine threshold: {best_config['cosine_threshold']}")
        print(f"  Collapse enabled: {best_config['enable_collapse']}")
        print(f"  Q&A accuracy: {best_config['qa_accuracy']:.2%}")
        print(f"  Compression ratio: {best_config['compression_ratio']:.2f}x")
        print(f"  Proto-identities: {best_config['num_protos']:,}")
        print(f"  Training time: {best_config['train_time']:.1f}s")

        # Additional analysis
        print("\n📊 Parameter Impact Analysis:")

        # Analyze impact of each parameter
        if len(results) > 1:
            # Group by harmonic tolerance
            ht_groups = {}
            for r in results:
                ht = r['harmonic_tolerance']
                if ht not in ht_groups:
                    ht_groups[ht] = []
                ht_groups[ht].append(r['qa_accuracy'])

            print("\n  Harmonic Tolerance Impact:")
            for ht in sorted(ht_groups.keys()):
                avg_accuracy = sum(ht_groups[ht]) / len(ht_groups[ht])
                print(f"    {ht}: {avg_accuracy:.2%} avg accuracy")

            # Group by cosine threshold
            ct_groups = {}
            for r in results:
                ct = r['cosine_threshold']
                if ct not in ct_groups:
                    ct_groups[ct] = []
                ct_groups[ct].append(r['qa_accuracy'])

            print("\n  Cosine Threshold Impact:")
            for ct in sorted(ct_groups.keys()):
                avg_accuracy = sum(ct_groups[ct]) / len(ct_groups[ct])
                print(f"    {ct}: {avg_accuracy:.2%} avg accuracy")

            # Collapse impact
            collapse_yes = [r['qa_accuracy'] for r in results if r['enable_collapse']]
            collapse_no = [r['qa_accuracy'] for r in results if not r['enable_collapse']]

            if collapse_yes and collapse_no:
                print("\n  Gravitational Collapse Impact:")
                print(f"    Enabled:  {sum(collapse_yes)/len(collapse_yes):.2%} avg accuracy")
                print(f"    Disabled: {sum(collapse_no)/len(collapse_no):.2%} avg accuracy")

    # Save detailed report
    report_path = Path(args.output_report)
    with open(report_path, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_configurations': len(results),
            'test_queries': len(test_queries),
            'best_config': best_config,
            'all_results': results
        }, f, indent=2)

    print(f"\n📄 Detailed report saved to: {report_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())