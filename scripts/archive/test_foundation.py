#!/usr/bin/env python3
"""
Simple test runner for Foundation Q&A tests without pytest dependency.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
from pathlib import Path
from typing import List, Dict

from src.memory.voxel_cloud import VoxelCloud
from src.memory.frequency_field import TextFrequencyAnalyzer


def synthesize(voxel_cloud: VoxelCloud, query: str, freq_analyzer: TextFrequencyAnalyzer) -> str:
    """Helper to synthesize response from voxel cloud."""
    # Convert query to frequency
    query_freq, _ = freq_analyzer.analyze(query)

    # Query viewport for relevant proto-identities
    visible_protos = voxel_cloud.query_viewport(query_freq, radius=50.0)

    if not visible_protos:
        return "No relevant patterns found"

    # Extract text from most relevant protos
    texts = []
    for entry in visible_protos[:5]:  # Top 5 most relevant
        if 'text' in entry.metadata:
            texts.append(entry.metadata['text'])

    return ' '.join(texts) if texts else "No text found"


def run_test(name: str, query: str, keywords: List[str],
             voxel_cloud: VoxelCloud, freq_analyzer: TextFrequencyAnalyzer) -> bool:
    """Run a single test."""
    print(f"  Testing: {name}")
    result = synthesize(voxel_cloud, query, freq_analyzer)

    if len(result) <= 10:
        print(f"    ❌ Response too short: {len(result)} chars")
        return False

    found_keyword = any(word in result.lower() for word in keywords)
    if not found_keyword:
        print(f"    ❌ No keywords found. Expected one of: {keywords}")
        print(f"       Response preview: {result[:100]}...")
        return False

    print(f"    ✅ Pass")
    return True


def main():
    print("=" * 70)
    print("Foundation Model Q&A Tests")
    print("=" * 70)

    # Check if model exists
    model_path = Path('/usr/lib/alembic/checkpoints/genesis/foundation_voxel_cloud.pkl')
    if not model_path.exists():
        print("❌ Foundation model not found!")
        print("   Please run: python scripts/train_foundation.py")
        return 1

    # Load model
    print("\n📂 Loading Foundation model...")
    with open(model_path, 'rb') as f:
        voxel_cloud = pickle.load(f)

    print(f"  Loaded {len(voxel_cloud.entries)} proto-identities")

    # Create frequency analyzer
    freq_analyzer = TextFrequencyAnalyzer(512, 512)

    # Define test cases
    test_cases = [
        # Philosophy tests
        ("Tao emptiness", "What is emptiness in Taoism?",
         ['empty', 'void', 'nothing', 'tao', 'way']),
        ("Dharma concept", "What is dharma?",
         ['duty', 'righteousness', 'path', 'law', 'order']),
        ("Enlightenment", "What is enlightenment?",
         ['wisdom', 'knowledge', 'truth', 'liberation', 'awakening']),

        # History tests
        ("Achilles", "Who was Achilles?",
         ['warrior', 'hero', 'greek', 'troy', 'heel']),
        ("Gilgamesh quest", "What did Gilgamesh seek?",
         ['immortality', 'eternal', 'life', 'death', 'friend']),
        ("Trojan War", "What was the Trojan War?",
         ['troy', 'greek', 'war', 'helen', 'horse']),

        # Strategy tests
        ("Art of War", "What did Sun Tzu say about deception?",
         ['war', 'deception', 'enemy', 'strategy', 'art']),
        ("Power laws", "What are the laws of power?",
         ['power', 'law', 'enemy', 'friend', 'master']),
        ("Leadership", "What makes a good leader?",
         ['leader', 'wisdom', 'courage', 'vision', 'people'])
    ]

    # Run tests
    print("\n🧪 Running Q&A tests...")
    passed = 0
    failed = 0

    for name, query, keywords in test_cases:
        if run_test(name, query, keywords, voxel_cloud, freq_analyzer):
            passed += 1
        else:
            failed += 1

    # Check compression metrics
    print("\n📊 Checking compression metrics...")
    resonances = sum(1 for p in voxel_cloud.entries if p.resonance_strength > 1)
    compression_ratio = sum(p.resonance_strength for p in voxel_cloud.entries) / len(voxel_cloud.entries)

    print(f"  Proto-identities: {len(voxel_cloud.entries):,}")
    print(f"  Resonant patterns: {resonances:,} ({100*resonances/len(voxel_cloud.entries):.1f}%)")
    print(f"  Compression ratio: {compression_ratio:.2f}x")

    # Summary
    print("\n" + "=" * 70)
    print("Test Results")
    print("=" * 70)
    print(f"  Q&A Tests: {passed} passed, {failed} failed")
    print(f"  Success rate: {100*passed/(passed+failed):.1f}%")

    if passed == len(test_cases):
        print("\n✅ All tests passed!")
        return 0
    else:
        print(f"\n⚠️  Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())