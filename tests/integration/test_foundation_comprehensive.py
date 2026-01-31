#!/usr/bin/env python3
"""
Comprehensive QA test for synthesis pipeline across ALL foundation datasets.
Tests the actual discover/synthesize commands on foundation texts.
"""

import os
import sys
import glob
import json
import time
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

class ComprehensiveFoundationQA:
    """QA tester for all foundation datasets using actual CLI commands."""

    def __init__(self):
        self.foundation_dir = '/usr/lib/alembic/data/datasets/curated/foundation'
        self.datasets = sorted(glob.glob(os.path.join(self.foundation_dir, '*.txt')))
        self.results = {}
        self.test_dir = '/tmp/qa_foundation_tests'

        # Create test directory
        Path(self.test_dir).mkdir(parents=True, exist_ok=True)

        print(f"Found {len(self.datasets)} foundation datasets")
        print("=" * 80)

    def test_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Test discover and synthesize on a single dataset."""
        dataset_name = os.path.basename(dataset_path)
        file_size_kb = os.path.getsize(dataset_path) / 1024

        print(f"\n{'='*80}")
        print(f"Testing: {dataset_name} ({file_size_kb:.1f} KB)")
        print(f"{'='*80}")

        result = {
            'name': dataset_name,
            'file_size_kb': file_size_kb,
            'discover_success': False,
            'synthesize_success': False,
            'queries_tested': 0,
            'queries_successful': 0,
            'errors': [],
            'timings': {}
        }

        # Output paths
        model_path = os.path.join(self.test_dir, f"{dataset_name}.model")

        # Calculate adaptive timeout based on file size
        # At ~0.5s per segment, estimate segments from file size
        # Rough estimate: 1KB ≈ 10 segments, so 100KB ≈ 1000 segments ≈ 500s
        estimated_segments = int(file_size_kb * 10)
        timeout_seconds = max(120, min(600, estimated_segments // 2))  # 2-10 minutes
        discovery_timeout = timeout_seconds

        # Test discovery phase
        print(f"  1. Discovery phase (timeout: {discovery_timeout}s)...")
        start_time = time.time()

        discover_cmd = [
            'python', 'genesis.py', 'discover',
            '--input', dataset_path,
            '--output', model_path,
            '--dual-path',
            '--enable-collapse',
            '--max-segments', '500'  # Limit to 500 segments for QA testing
        ]

        try:
            # Show progress indicator
            print(f"     Processing segments... ", end="", flush=True)

            discover_result = subprocess.run(
                discover_cmd,
                capture_output=True,
                text=True,
                timeout=discovery_timeout,
                cwd='/home/persist/alembic/genesis'
            )

            if discover_result.returncode == 0:
                result['discover_success'] = True
                print(f"✓")
                print(f"    ✓ Discovery successful")
            else:
                print(f"✗")
                stderr_preview = discover_result.stderr[:500] if discover_result.stderr else "No error output"
                result['errors'].append(f"Discovery failed: {stderr_preview}")
                print(f"    ✗ Discovery failed: {stderr_preview[:200]}")

        except subprocess.TimeoutExpired:
            print(f"✗ (timeout)")
            result['errors'].append(f"Discovery timed out after {discovery_timeout}s")
            print(f"    ✗ Discovery timed out after {discovery_timeout}s")
        except Exception as e:
            print(f"✗ (error)")
            result['errors'].append(f"Discovery error: {str(e)}")
            print(f"    ✗ Discovery error: {e}")

        result['timings']['discovery'] = time.time() - start_time

        # Test synthesis phase with different queries
        if result['discover_success'] and Path(model_path).exists():
            print(f"  2. Synthesis phase...")

            # Load some text from the dataset for queries
            with open(dataset_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            words = content.split()[:1000]

            # Test queries of different lengths
            test_queries = [
                ' '.join(words[:3]),     # Short query
                ' '.join(words[10:15]),  # Medium query
                ' '.join(words[20:30])   # Long query
            ]

            for idx, query in enumerate(test_queries):
                if len(query) < 5:
                    continue

                result['queries_tested'] += 1
                query_type = ['short', 'medium', 'long'][idx]

                synthesize_cmd = [
                    'python', 'genesis.py', 'synthesize',
                    '--model', model_path,
                    '--query', query[:100],  # Limit query length
                    '--radius', '50.0'
                ]

                try:
                    synth_result = subprocess.run(
                        synthesize_cmd,
                        capture_output=True,
                        text=True,
                        timeout=10,
                        cwd='/home/persist/alembic/genesis'
                    )

                    if synth_result.returncode == 0:
                        result['queries_successful'] += 1
                        result['synthesize_success'] = True
                        print(f"    ✓ {query_type} query successful")
                    else:
                        print(f"    ✗ {query_type} query failed")

                except subprocess.TimeoutExpired:
                    print(f"    ✗ {query_type} query timed out")
                except Exception as e:
                    print(f"    ✗ {query_type} query error: {e}")

        # Calculate success metrics
        if result['queries_tested'] > 0:
            result['query_success_rate'] = result['queries_successful'] / result['queries_tested']
        else:
            result['query_success_rate'] = 0

        # Print summary for this dataset
        print(f"\n  Summary:")
        print(f"    Discovery: {'✓' if result['discover_success'] else '✗'}")
        print(f"    Synthesis: {result['queries_successful']}/{result['queries_tested']} queries")
        print(f"    Time: {result['timings'].get('discovery', 0):.1f}s")

        if result['errors']:
            print(f"    Errors: {len(result['errors'])}")

        return result

    def run_all_tests(self) -> None:
        """Run tests on all foundation datasets."""
        start_time = time.time()

        # Test ALL datasets (not just a subset)
        test_datasets = self.datasets

        print(f"\nTesting ALL {len(test_datasets)} datasets...")
        print("This may take 30-60 minutes depending on dataset sizes.")
        print()

        for idx, dataset_path in enumerate(test_datasets, 1):
            print(f"\n[{idx}/{len(test_datasets)}]", end=" ")
            result = self.test_dataset(dataset_path)
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
        total_datasets = len(self.results)
        discover_success = sum(1 for r in self.results.values() if r['discover_success'])
        synthesize_success = sum(1 for r in self.results.values() if r['synthesize_success'])
        total_queries = sum(r['queries_tested'] for r in self.results.values())
        successful_queries = sum(r['queries_successful'] for r in self.results.values())

        print(f"Datasets tested: {total_datasets}")
        print(f"Time elapsed: {elapsed_time:.1f} seconds")
        print()

        print("Overall Success Rates:")
        print(f"  Discovery: {discover_success}/{total_datasets} ({discover_success/total_datasets:.1%})")
        print(f"  Synthesis: {synthesize_success}/{total_datasets} ({synthesize_success/total_datasets:.1%})")
        print(f"  Queries: {successful_queries}/{total_queries} ({successful_queries/total_queries:.1%})" if total_queries > 0 else "  Queries: N/A")
        print()

        # Per-dataset summary table
        print(f"{'='*80}")
        print("PER-DATASET RESULTS")
        print(f"{'='*80}")
        print(f"{'Dataset':<35} {'Discover':>10} {'Synthesis':>10} {'Queries':>15}")
        print("-" * 80)

        for name in sorted(self.results.keys()):
            r = self.results[name]

            discover_status = '✓' if r['discover_success'] else '✗'
            synthesis_status = '✓' if r['synthesize_success'] else '✗'
            query_status = f"{r['queries_successful']}/{r['queries_tested']}"

            print(f"{name[:35]:<35} {discover_status:>10} {synthesis_status:>10} {query_status:>15}")

        # Success criteria evaluation
        print(f"\n{'='*80}")
        print("SUCCESS CRITERIA EVALUATION")
        print(f"{'='*80}")

        criteria = {
            "Discovery ≥95%": discover_success/total_datasets >= 0.95 if total_datasets > 0 else False,
            "Synthesis ≥50%": synthesize_success/total_datasets >= 0.50 if total_datasets > 0 else False,
            "Queries ≥60%": successful_queries/total_queries >= 0.60 if total_queries > 0 else False,
        }

        for criterion, passed in criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {criterion}: {status}")

        # Overall verdict
        all_pass = all(criteria.values())
        most_pass = sum(criteria.values()) >= 2

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
        output_path = '/tmp/qa_comprehensive_results.json'

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\nDetailed results saved to: {output_path}")


def main():
    """Run comprehensive QA testing."""
    print("=" * 80)
    print("COMPREHENSIVE QA TEST: SYNTHESIS PIPELINE")
    print("Testing Foundation Datasets via CLI Commands")
    print(f"Started: {datetime.now()}")
    print("=" * 80)

    qa = ComprehensiveFoundationQA()
    qa.run_all_tests()

    print(f"\nCompleted: {datetime.now()}")
    print("=" * 80)


if __name__ == "__main__":
    main()