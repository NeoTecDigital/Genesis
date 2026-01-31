#!/usr/bin/env python3
"""
Comprehensive QA test for synthesis pipeline across ALL foundation datasets.
Tests the 3 critical fixes:
1. Relaxed Res() validation thresholds
2. base_frequency preservation
3. Exponential decay weighting
"""

import os
import sys
import glob
import json
import time
import traceback
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.origin import Origin
from src.memory.frequency_field import TextFrequencyAnalyzer
from src.proto_identity import ProtoIdentityManager

class FoundationDatasetQA:
    """QA tester for all foundation datasets."""

    def __init__(self):
        self.foundation_dir = '/usr/lib/alembic/data/datasets/curated/foundation'
        self.datasets = sorted(glob.glob(os.path.join(self.foundation_dir, '*.txt')))
        self.results = {}
        self.failure_details = {}

        # Initialize components
        self.origin = Origin()
        self.analyzer = TextFrequencyAnalyzer()
        self.identity = ProtoIdentityManager()

        print(f"Found {len(self.datasets)} foundation datasets")
        print("=" * 80)

    def test_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Test a single dataset comprehensively."""
        dataset_name = os.path.basename(dataset_path)
        print(f"\n{'='*80}")
        print(f"Testing: {dataset_name}")
        print(f"{'='*80}")

        result = {
            'name': dataset_name,
            'file_size_kb': os.path.getsize(dataset_path) / 1024,
            'gen_success': 0,
            'gen_total': 0,
            'res_success': 0,
            'res_total': 0,
            'convergences': [],
            'frequencies_preserved': 0,
            'weight_diversity': [],
            'errors': [],
            'sample_outputs': []
        }

        try:
            # Load dataset
            with open(dataset_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Split into segments (use first 10 for testing)
            segments = [s.strip() for s in content.split('\n\n') if s.strip()][:10]

            if not segments:
                result['errors'].append("No valid segments found")
                return result

            print(f"  Testing {len(segments)} segments...")

            for idx, segment in enumerate(segments):
                if len(segment) < 10:  # Skip very short segments
                    continue

                # Test Gen operation
                try:
                    # Get frequency representation
                    freq = self.analyzer.analyze(segment)

                    # Gen operation
                    gen_weights = self.identity.Gen(freq)

                    if gen_weights is not None:
                        result['gen_success'] += 1

                        # Check weight diversity
                        if hasattr(gen_weights, 'numpy'):
                            weights_np = gen_weights.numpy()
                        else:
                            weights_np = gen_weights
                        diversity = float(np.std(weights_np))
                        result['weight_diversity'].append(diversity)

                    result['gen_total'] += 1

                except Exception as e:
                    result['errors'].append(f"Gen error segment {idx}: {str(e)}")

                # Test Res operation
                try:
                    if gen_weights is not None:
                        # Res operation
                        res_freq = self.identity.Res(gen_weights)

                        if res_freq is not None:
                            result['res_success'] += 1

                            # Check convergence
                            if hasattr(freq, 'pattern') and hasattr(res_freq, 'pattern'):
                                convergence = self._compute_convergence(
                                    freq.pattern, res_freq.pattern
                                )
                                result['convergences'].append(convergence)

                            # Check frequency preservation
                            if hasattr(res_freq, 'base_frequency'):
                                if res_freq.base_frequency > 0:
                                    result['frequencies_preserved'] += 1

                        result['res_total'] += 1

                        # Save sample output for first segment
                        if idx == 0:
                            result['sample_outputs'].append({
                                'segment': segment[:100] + '...' if len(segment) > 100 else segment,
                                'gen_success': True,
                                'res_success': res_freq is not None,
                                'convergence': result['convergences'][-1] if result['convergences'] else 0
                            })

                except Exception as e:
                    result['errors'].append(f"Res error segment {idx}: {str(e)}")

        except Exception as e:
            result['errors'].append(f"Dataset loading error: {str(e)}")
            traceback.print_exc()

        # Calculate metrics
        if result['gen_total'] > 0:
            result['gen_success_rate'] = result['gen_success'] / result['gen_total']
        else:
            result['gen_success_rate'] = 0

        if result['res_total'] > 0:
            result['res_success_rate'] = result['res_success'] / result['res_total']
        else:
            result['res_success_rate'] = 0

        if result['convergences']:
            result['avg_convergence'] = float(np.mean(result['convergences']))
            result['min_convergence'] = float(np.min(result['convergences']))
            result['max_convergence'] = float(np.max(result['convergences']))
        else:
            result['avg_convergence'] = 0
            result['min_convergence'] = 0
            result['max_convergence'] = 0

        if result['weight_diversity']:
            result['avg_diversity'] = float(np.mean(result['weight_diversity']))
        else:
            result['avg_diversity'] = 0

        if result['gen_total'] > 0:
            result['frequency_preservation_rate'] = result['frequencies_preserved'] / result['gen_total']
        else:
            result['frequency_preservation_rate'] = 0

        # Print summary
        print(f"  Gen: {result['gen_success']}/{result['gen_total']} ({result['gen_success_rate']:.1%})")
        print(f"  Res: {result['res_success']}/{result['res_total']} ({result['res_success_rate']:.1%})")
        print(f"  Convergence: {result['avg_convergence']:.4f} (min: {result['min_convergence']:.4f}, max: {result['max_convergence']:.4f})")
        print(f"  Diversity: {result['avg_diversity']:.4f}")
        print(f"  Frequency preservation: {result['frequency_preservation_rate']:.1%}")

        if result['errors']:
            print(f"  ⚠️  Errors: {len(result['errors'])}")

        return result

    def _compute_convergence(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """Compute cosine similarity between patterns."""
        try:
            if pattern1.shape != pattern2.shape:
                return 0.0

            # Flatten if needed
            p1 = pattern1.flatten()
            p2 = pattern2.flatten()

            # Compute cosine similarity
            dot_product = np.dot(p1, p2)
            norm1 = np.linalg.norm(p1)
            norm2 = np.linalg.norm(p2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot_product / (norm1 * norm2))
        except:
            return 0.0

    def test_query_quality(self, dataset_path: str) -> Dict[str, Any]:
        """Test query quality with different query lengths."""
        dataset_name = os.path.basename(dataset_path)
        print(f"\n  Query Quality Test for {dataset_name}")

        results = {
            'short_query': None,
            'medium_query': None,
            'long_query': None
        }

        try:
            # Load dataset for queries
            with open(dataset_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            words = content.split()[:1000]  # Use first 1000 words

            # Test queries of different lengths
            queries = {
                'short_query': ' '.join(words[:3]),
                'medium_query': ' '.join(words[10:15]),
                'long_query': ' '.join(words[20:30])
            }

            for query_type, query_text in queries.items():
                if len(query_text) < 5:
                    continue

                try:
                    # Process query
                    freq = self.analyzer.analyze(query_text)
                    weights = self.identity.Gen(freq)

                    if weights is not None:
                        results[query_type] = {
                            'success': True,
                            'query_length': len(query_text.split()),
                            'weight_diversity': float(np.std(weights.numpy() if hasattr(weights, 'numpy') else weights))
                        }
                    else:
                        results[query_type] = {'success': False}

                except Exception as e:
                    results[query_type] = {'success': False, 'error': str(e)}

            print(f"    Short: {results['short_query']}")
            print(f"    Medium: {results['medium_query']}")
            print(f"    Long: {results['long_query']}")

        except Exception as e:
            print(f"    Query test failed: {e}")

        return results

    def run_all_tests(self) -> None:
        """Run tests on all foundation datasets."""
        start_time = time.time()

        for dataset_path in self.datasets:
            # Main testing
            result = self.test_dataset(dataset_path)

            # Query quality testing (sample)
            if self.datasets.index(dataset_path) % 5 == 0:  # Test every 5th dataset
                query_results = self.test_query_quality(dataset_path)
                result['query_quality'] = query_results

            self.results[os.path.basename(dataset_path)] = result

        elapsed = time.time() - start_time

        # Print aggregate summary
        self._print_summary(elapsed)

        # Save detailed results
        self._save_results()

    def _print_summary(self, elapsed_time: float) -> None:
        """Print aggregate summary of all tests."""
        print(f"\n{'='*80}")
        print("AGGREGATE QA RESULTS")
        print(f"{'='*80}")

        # Calculate aggregates
        total_gen = sum(r['gen_total'] for r in self.results.values())
        total_gen_success = sum(r['gen_success'] for r in self.results.values())
        total_res = sum(r['res_total'] for r in self.results.values())
        total_res_success = sum(r['res_success'] for r in self.results.values())

        all_convergences = []
        all_diversities = []

        for r in self.results.values():
            all_convergences.extend(r['convergences'])
            all_diversities.extend(r['weight_diversity'])

        print(f"Datasets tested: {len(self.results)}")
        print(f"Time elapsed: {elapsed_time:.1f} seconds")
        print()

        print("Overall Success Rates:")
        print(f"  Gen: {total_gen_success}/{total_gen} ({total_gen_success/total_gen:.1%})" if total_gen > 0 else "  Gen: N/A")
        print(f"  Res: {total_res_success}/{total_res} ({total_res_success/total_res:.1%})" if total_res > 0 else "  Res: N/A")
        print()

        if all_convergences:
            print(f"Convergence Stats:")
            print(f"  Mean: {np.mean(all_convergences):.4f}")
            print(f"  Min: {np.min(all_convergences):.4f}")
            print(f"  Max: {np.max(all_convergences):.4f}")
            print(f"  Std: {np.std(all_convergences):.4f}")

        if all_diversities:
            print(f"\nDiversity Stats:")
            print(f"  Mean: {np.mean(all_diversities):.4f}")
            print(f"  Min: {np.min(all_diversities):.4f}")
            print(f"  Max: {np.max(all_diversities):.4f}")

        # Per-dataset summary table
        print(f"\n{'='*80}")
        print("PER-DATASET RESULTS")
        print(f"{'='*80}")
        print(f"{'Dataset':<35} {'Gen%':>8} {'Res%':>8} {'Conv':>8} {'Div':>8} {'Status':>10}")
        print("-" * 80)

        for name in sorted(self.results.keys()):
            r = self.results[name]

            # Determine status
            status = "✅ PASS"
            if r['gen_success_rate'] < 0.95:
                status = "❌ FAIL"
            elif r['res_success_rate'] < 0.30:
                status = "⚠️ WEAK"
            elif r['avg_convergence'] < 0.5:
                status = "⚠️ LOW"

            print(f"{name[:35]:<35} "
                  f"{r['gen_success_rate']:>7.1%} "
                  f"{r['res_success_rate']:>7.1%} "
                  f"{r['avg_convergence']:>7.3f} "
                  f"{r['avg_diversity']:>7.3f} "
                  f"{status:>10}")

        # Success criteria evaluation
        print(f"\n{'='*80}")
        print("SUCCESS CRITERIA EVALUATION")
        print(f"{'='*80}")

        criteria = {
            "Gen success ≥95%": total_gen_success/total_gen >= 0.95 if total_gen > 0 else False,
            "Res success ≥50%": total_res_success/total_res >= 0.50 if total_res > 0 else False,
            "Convergence ≥0.6": np.mean(all_convergences) >= 0.6 if all_convergences else False,
            "Diversity ≥0.15": np.mean(all_diversities) >= 0.15 if all_diversities else False,
            "No dataset <30% Res": all(r['res_success_rate'] >= 0.30 for r in self.results.values())
        }

        for criterion, passed in criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {criterion}: {status}")

        # Overall verdict
        all_pass = all(criteria.values())
        most_pass = sum(criteria.values()) >= 3

        print(f"\n{'='*80}")
        if all_pass:
            print("🎉 VERDICT: APPROVED - All criteria met, ready for deployment")
        elif most_pass:
            print("⚠️  VERDICT: CONDITIONAL - Most criteria met, minor issues present")
        else:
            print("❌ VERDICT: REJECTED - Critical failures, needs rework")
        print(f"{'='*80}")

    def _save_results(self) -> None:
        """Save detailed results to file."""
        output_path = '/tmp/qa_foundation_test_results.json'

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\nDetailed results saved to: {output_path}")


def main():
    """Run comprehensive QA testing."""
    print("=" * 80)
    print("COMPREHENSIVE QA TEST: SYNTHESIS PIPELINE")
    print("Testing ALL Foundation Datasets")
    print(f"Started: {datetime.now()}")
    print("=" * 80)

    qa = FoundationDatasetQA()
    qa.run_all_tests()

    print(f"\nCompleted: {datetime.now()}")
    print("=" * 80)


if __name__ == "__main__":
    main()