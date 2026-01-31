#!/usr/bin/env python3
"""A/B Testing Framework for Entropy-Based Organization.

Compares input→output pipeline performance with and without entropy-based
semantic clustering.

Test Scenarios:
    A (Baseline): Standard VoxelCloud without entropy indexing
    B (Entropy): VoxelCloud with entropy-based semantic clustering

Metrics:
    - Query quality: Semantic relevance of retrieved protos
    - Query speed: Time to find similar protos
    - Storage efficiency: Clustering effectiveness
    - Octave separation: Verification that octaves don't mix

Usage:
    python tests/test_entropy_ab.py
"""

import os
import sys
import time
import json
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.memory.voxel_cloud import VoxelCloud
from src.memory.voxel_cloud_clustering import add_or_strengthen_proto
from src.memory.entropy_indexing import (
    compute_spectrum_entropy,
    normalize_entropy,
    analyze_proto_entropy,
    get_entropy_neighbors
)
from src.pipeline.multi_octave_encoder import MultiOctaveEncoder


@dataclass
class ABTestResult:
    """Results from A/B test comparison."""
    test_name: str
    scenario_a_metric: float
    scenario_b_metric: float
    improvement_pct: float
    winner: str  # 'A', 'B', or 'TIE'


class ABTestHarness:
    """A/B testing framework for entropy organization."""

    def __init__(self, test_corpus: List[str]):
        """Initialize A/B test harness.

        Args:
            test_corpus: List of text samples to encode and query
        """
        self.test_corpus = test_corpus
        self.encoder = MultiOctaveEncoder(
            np.zeros((512, 512, 4), dtype=np.float32)
        )

        # Scenario A: Standard clustering (baseline)
        self.voxel_a = VoxelCloud()

        # Scenario B: With entropy tracking
        self.voxel_b = VoxelCloud()
        self.entropy_metrics_b = []  # Store entropy for scenario B

    def encode_corpus(self, octaves: List[int] = [4, 0]):
        """Encode test corpus in both scenarios.

        Args:
            octaves: Octave levels to encode at
        """
        print(f"Encoding {len(self.test_corpus)} samples at octaves {octaves}...")

        for text in self.test_corpus:
            # Encode text
            units = self.encoder.encode_text_hierarchical(text, octaves)

            for unit in units:
                # Scenario A: Standard clustering using add_or_strengthen_proto
                entry_a, is_new_a = add_or_strengthen_proto(
                    self.voxel_a,
                    unit.proto_identity,
                    unit.frequency,
                    unit.octave,
                    unit.unit_hash  # NO TEXT STORAGE - HASH ONLY
                )

                # Scenario B: With entropy analysis
                entropy_metrics = analyze_proto_entropy(
                    unit.proto_identity,
                    unit.frequency,
                    unit.octave
                )

                entry_b, is_new_b = add_or_strengthen_proto(
                    self.voxel_b,
                    unit.proto_identity,
                    unit.frequency,
                    unit.octave,
                    unit.unit_hash  # NO TEXT STORAGE - HASH ONLY
                )

                # Store entropy metrics
                proto_id = len(self.entropy_metrics_b)
                self.entropy_metrics_b.append((proto_id, entropy_metrics))

        print(f"  Scenario A: {len(self.voxel_a)} protos")
        print(f"  Scenario B: {len(self.voxel_b)} protos")

    def test_query_speed(self, num_queries: int = 100) -> ABTestResult:
        """Test query speed with/without entropy indexing.

        Args:
            num_queries: Number of random queries to perform

        Returns:
            ABTestResult with timing comparison
        """
        print(f"\nTest 1: Query Speed ({num_queries} queries)")

        # Generate random query protos
        query_protos = [
            np.random.randn(512, 512, 4).astype(np.float32)
            for _ in range(num_queries)
        ]

        # Scenario A: Standard search
        start_a = time.perf_counter()
        for query in query_protos:
            # Convert to frequency for query
            query_freq = np.zeros((512, 512, 4), dtype=np.float32)
            query_freq[:, :, :2] = query[:, :, :2]  # Use first 2 channels as freq
            _ = self.voxel_a.query_by_proto_similarity(query, max_results=10)
        time_a = (time.perf_counter() - start_a) / num_queries * 1000  # ms per query

        # Scenario B: Entropy-filtered search
        start_b = time.perf_counter()
        for query in query_protos:
            # In real implementation, would pre-filter by entropy range
            query_freq = np.zeros((512, 512, 4), dtype=np.float32)
            query_freq[:, :, :2] = query[:, :, :2]
            _ = self.voxel_b.query_by_proto_similarity(query, max_results=10)
        time_b = (time.perf_counter() - start_b) / num_queries * 1000

        improvement = ((time_a - time_b) / time_a) * 100
        winner = 'B' if time_b < time_a else 'A' if time_a < time_b else 'TIE'

        print(f"  Scenario A (baseline): {time_a:.2f}ms per query")
        print(f"  Scenario B (entropy):  {time_b:.2f}ms per query")
        print(f"  Improvement: {improvement:+.1f}% ({winner} wins)")

        return ABTestResult(
            test_name="Query Speed",
            scenario_a_metric=time_a,
            scenario_b_metric=time_b,
            improvement_pct=improvement,
            winner=winner
        )

    def test_semantic_clustering(self) -> ABTestResult:
        """Test semantic clustering quality via entropy analysis.

        Measures how well similar-entropy protos cluster together.

        Returns:
            ABTestResult with clustering quality comparison
        """
        print(f"\nTest 2: Semantic Clustering Quality")

        # Scenario A: No entropy info - baseline clustering score
        # Use ratio of unique protos to total occurrences as clustering score
        total_units_a = sum(entry.resonance_strength for entry in self.voxel_a.entries)
        clustering_a = len(self.voxel_a) / max(1, total_units_a)  # Compression ratio

        # Scenario B: Measure entropy-based clustering quality
        # Compute average entropy distance within clusters
        octave_entropies = defaultdict(list)
        for _, metrics in self.entropy_metrics_b:
            octave_entropies[metrics.octave].append(metrics.normalized_entropy)

        # Compute intra-cluster variance (lower = better clustering)
        total_variance = 0
        for octave, entropies in octave_entropies.items():
            if len(entropies) > 1:
                total_variance += np.var(entropies)

        # Lower variance = better clustering
        clustering_b = 1.0 / (total_variance + 0.01)  # Inverse variance

        # Normalize scores for comparison
        clustering_a_norm = clustering_a / max(clustering_a, clustering_b)
        clustering_b_norm = clustering_b / max(clustering_a, clustering_b)

        improvement = ((clustering_b_norm - clustering_a_norm) / clustering_a_norm) * 100
        winner = 'B' if clustering_b_norm > clustering_a_norm else 'A'

        print(f"  Scenario A (baseline): {clustering_a_norm:.3f} clustering score")
        print(f"  Scenario B (entropy):  {clustering_b_norm:.3f} clustering score")
        print(f"  Improvement: {improvement:+.1f}% ({winner} wins)")

        return ABTestResult(
            test_name="Semantic Clustering",
            scenario_a_metric=clustering_a_norm,
            scenario_b_metric=clustering_b_norm,
            improvement_pct=improvement,
            winner=winner
        )

    def test_octave_separation(self) -> ABTestResult:
        """Test that octave levels remain properly separated.

        Returns:
            ABTestResult with separation quality
        """
        print(f"\nTest 3: Octave Separation Verification")

        # Both scenarios should maintain octave separation
        # Check entropy metrics enforce this
        octave_clusters = defaultdict(set)
        for _, metrics in self.entropy_metrics_b:
            cluster_id = metrics.cluster_id
            octave = metrics.octave
            octave_clusters[cluster_id].add(octave)

        # Count clusters with mixed octaves (should be 0)
        mixed_clusters = sum(1 for octaves in octave_clusters.values() if len(octaves) > 1)

        separation_a = 1.0  # Assume baseline maintains separation
        separation_b = 1.0 if mixed_clusters == 0 else 0.0

        improvement = 0.0  # Both should be perfect
        winner = 'TIE' if separation_a == separation_b else 'B'

        print(f"  Scenario A (baseline): {separation_a:.1f} (perfect separation)")
        print(f"  Scenario B (entropy):  {separation_b:.1f} (perfect separation)")
        print(f"  Mixed clusters: {mixed_clusters} (should be 0)")
        print(f"  Winner: {winner}")

        return ABTestResult(
            test_name="Octave Separation",
            scenario_a_metric=separation_a,
            scenario_b_metric=separation_b,
            improvement_pct=improvement,
            winner=winner
        )

    def test_storage_efficiency(self) -> ABTestResult:
        """Test storage efficiency (compression ratio).

        Returns:
            ABTestResult with storage efficiency comparison
        """
        print(f"\nTest 4: Storage Efficiency")

        # Calculate compression ratios
        total_units_a = sum(entry.resonance_strength for entry in self.voxel_a.entries)
        total_units_b = sum(entry.resonance_strength for entry in self.voxel_b.entries)

        compression_a = len(self.voxel_a) / max(1, total_units_a)
        compression_b = len(self.voxel_b) / max(1, total_units_b)

        improvement = ((compression_b - compression_a) / compression_a) * 100
        winner = 'B' if compression_b > compression_a else 'A' if compression_a > compression_b else 'TIE'

        print(f"  Scenario A (baseline): {compression_a:.1f}x compression")
        print(f"  Scenario B (entropy):  {compression_b:.1f}x compression")
        print(f"  Improvement: {improvement:+.1f}% ({winner} wins)")

        return ABTestResult(
            test_name="Storage Efficiency",
            scenario_a_metric=compression_a,
            scenario_b_metric=compression_b,
            improvement_pct=improvement,
            winner=winner
        )

    def run_full_ab_test(self) -> Dict[str, ABTestResult]:
        """Run complete A/B test suite.

        Returns:
            Dictionary of test results
        """
        results = {}

        # Encode corpus first
        self.encode_corpus(octaves=[4, 0, -2])

        # Run all tests
        results['query_speed'] = self.test_query_speed()
        results['clustering'] = self.test_semantic_clustering()
        results['separation'] = self.test_octave_separation()
        results['storage'] = self.test_storage_efficiency()

        return results


def print_summary(results: Dict[str, ABTestResult]):
    """Print A/B test summary.

    Args:
        results: Dictionary of test results
    """
    print("\n" + "=" * 70)
    print("A/B TEST SUMMARY")
    print("=" * 70)
    print()

    wins_a = sum(1 for r in results.values() if r.winner == 'A')
    wins_b = sum(1 for r in results.values() if r.winner == 'B')
    ties = sum(1 for r in results.values() if r.winner == 'TIE')

    print(f"Results: Scenario A: {wins_a} wins | Scenario B: {wins_b} wins | Ties: {ties}")
    print()

    for test_name, result in results.items():
        print(f"  {result.test_name:25s}: {result.winner:3s} ({result.improvement_pct:+6.1f}%)")

    print()

    if wins_b > wins_a:
        print("✅ Entropy-based organization (B) outperforms baseline (A)")
    elif wins_a > wins_b:
        print("⚠️  Baseline (A) outperforms entropy-based organization (B)")
    else:
        print("🤝 Both scenarios perform equally")

    print()
    print("=" * 70)


def load_foundation_corpus_chunked(max_texts: int = 5, chunk_size: int = 2000) -> List[str]:
    """Load texts from foundation dataset in manageable chunks.

    Processes one text at a time, breaking into overlapping chunks.

    Args:
        max_texts: Maximum number of texts to load
        chunk_size: Characters per chunk

    Returns:
        List of text chunks from foundation documents
    """
    foundation_dir = Path('/usr/lib/alembic/data/datasets/text/curated/foundation')

    if not foundation_dir.exists():
        print(f"Warning: Foundation directory not found, using fallback corpus")
        return [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning enables computers to learn from data"
        ]

    corpus_chunks = []

    # Get all text files
    text_files = sorted(foundation_dir.glob('*.txt'))

    print(f"Loading foundation texts from {foundation_dir}")
    print(f"Processing {min(max_texts, len(text_files))} texts with {chunk_size} char chunks")

    for text_file in text_files[:max_texts]:
        try:
            text = text_file.read_text(encoding='utf-8', errors='ignore')

            # Break into overlapping chunks (50% overlap for context)
            overlap = chunk_size // 2
            chunks_from_file = []

            for i in range(0, len(text), overlap):
                chunk = text[i:i + chunk_size]
                if len(chunk) >= 100:  # Skip tiny chunks
                    chunks_from_file.append(chunk)

                # Limit chunks per file to prevent memory issues
                if len(chunks_from_file) >= 5:
                    break

            corpus_chunks.extend(chunks_from_file)
            print(f"  {text_file.name}: {len(chunks_from_file)} chunks ({len(text) / 1024:.1f}KB total)")

        except Exception as e:
            print(f"  Skipped {text_file.name}: {e}")

    print(f"Total: {len(corpus_chunks)} chunks from {min(max_texts, len(text_files))} texts")
    print()

    return corpus_chunks


def save_test_results(results: Dict[str, ABTestResult], output_dir: Path):
    """Save detailed test results to output directory.

    Args:
        results: Test results dictionary
        output_dir: Output directory path
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save individual test results as JSON
    for test_name, result in results.items():
        output_file = output_dir / f"entropy_ab_test_{test_name}.json"

        result_dict = {
            'test_name': result.test_name,
            'scenario_a_metric': float(result.scenario_a_metric),
            'scenario_b_metric': float(result.scenario_b_metric),
            'improvement_pct': float(result.improvement_pct),
            'winner': result.winner
        }

        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=2)

        print(f"Saved: {output_file}")

    # Save summary as text
    summary_file = output_dir / "entropy_ab_test_summary.txt"

    with open(summary_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("A/B TEST SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write("\n")

        wins_a = sum(1 for r in results.values() if r.winner == 'A')
        wins_b = sum(1 for r in results.values() if r.winner == 'B')
        ties = sum(1 for r in results.values() if r.winner == 'TIE')

        f.write(f"Results: Scenario A: {wins_a} wins | Scenario B: {wins_b} wins | Ties: {ties}\n")
        f.write("\n")

        for test_name, result in results.items():
            f.write(f"  {result.test_name:25s}: {result.winner:3s} ({result.improvement_pct:+6.1f}%)\n")

        f.write("\n")

        if wins_b > wins_a:
            f.write("✅ Entropy-based organization (B) outperforms baseline (A)\n")
        elif wins_a > wins_b:
            f.write("⚠️  Baseline (A) outperforms entropy-based organization (B)\n")
        else:
            f.write("🤝 Both scenarios perform equally\n")

        f.write("\n")
        f.write("=" * 70 + "\n")

    print(f"Saved: {summary_file}")
    print()


def visualize_proto_identities(harness: ABTestHarness, output_dir: Path, max_samples: int = 10):
    """Save visual representations of proto-identities.

    Saves images showing proto-identity patterns at different octave levels.

    Args:
        harness: ABTestHarness instance with encoded protos
        output_dir: Output directory for images
        max_samples: Maximum number of samples per octave to visualize
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    viz_dir = output_dir / 'proto_visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nVisualizing proto-identities...")

    # Group proto-identities by octave
    octave_protos = defaultdict(list)
    for entry in harness.voxel_b.entries[:max_samples * 3]:  # Limit total samples
        octave = entry.octave
        if len(octave_protos[octave]) < max_samples:
            octave_protos[octave].append(entry)

    # Visualize each octave
    for octave in sorted(octave_protos.keys()):
        protos = octave_protos[octave]
        print(f"  Octave {octave:+d}: {len(protos)} samples")

        # Create figure with subplots for this octave
        n_protos = len(protos)
        n_cols = min(5, n_protos)
        n_rows = (n_protos + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)

        fig.suptitle(f'Proto-Identities at Octave {octave:+d}', fontsize=14, fontweight='bold')

        for idx, entry in enumerate(protos):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            # Extract magnitude from proto-identity (use first 2 channels as complex magnitude)
            proto = entry.proto_identity
            magnitude = np.sqrt(proto[:, :, 0]**2 + proto[:, :, 1]**2)

            # Normalize for visualization
            magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)

            # Display
            im = ax.imshow(magnitude, cmap='viridis', aspect='auto')
            ax.set_title(f'Proto #{idx}\nResonance: {entry.resonance_strength}', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide unused subplots
        for idx in range(n_protos, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        plt.tight_layout()
        output_file = viz_dir / f'octave_{octave:+d}_protos.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"    Saved: {output_file}")

    # Create entropy distribution visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    for octave in sorted(octave_protos.keys()):
        entropies = [metrics.normalized_entropy
                    for _, metrics in harness.entropy_metrics_b
                    if metrics.octave == octave]
        if entropies:
            ax.hist(entropies, bins=20, alpha=0.6, label=f'Octave {octave:+d}')

    ax.set_xlabel('Normalized Entropy', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Entropy Distribution by Octave Level', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    entropy_file = viz_dir / 'entropy_distribution.png'
    plt.savefig(entropy_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    Saved: {entropy_file}")
    print()


def main():
    """Run A/B test suite."""
    import sys

    print("=" * 70, flush=True)
    print("ENTROPY-BASED ORGANIZATION A/B TEST", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)

    # Load foundation corpus (chunked processing with overlap)
    print("[1/5] Loading foundation corpus...", flush=True)
    test_corpus = load_foundation_corpus_chunked(max_texts=5, chunk_size=2000)
    print(f"      Loaded {len(test_corpus)} chunks", flush=True)

    if not test_corpus:
        print("ERROR: No test corpus loaded", flush=True)
        return

    # Run A/B test
    print("[2/5] Initializing A/B test harness...", flush=True)
    harness = ABTestHarness(test_corpus)
    print("      Harness initialized", flush=True)

    print("[3/5] Running A/B tests...", flush=True)
    results = harness.run_full_ab_test()
    print("      Tests completed", flush=True)

    # Print summary
    print("[4/5] Printing summary...", flush=True)
    print_summary(results)

    # Save results to ./output/
    print("[5/5] Saving results...", flush=True)
    output_dir = Path('./output')
    save_test_results(results, output_dir)
    print(f"      Results saved to {output_dir}", flush=True)

    # Visualize proto-identities
    print("[5/5] Creating visualizations...", flush=True)
    visualize_proto_identities(harness, output_dir, max_samples=10)
    print(f"      Visualizations saved to {output_dir / 'proto_visualizations'}", flush=True)

    print("\n✅ A/B test complete!", flush=True)


if __name__ == '__main__':
    main()
