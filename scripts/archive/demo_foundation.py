#!/usr/bin/env python3
"""
Quick demo of Foundation training on a subset of documents.
Trains on just 3 documents for fast demonstration.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from pathlib import Path

from src.origin import Origin
from src.memory.frequency_field import TextFrequencyAnalyzer
from src.memory.voxel_cloud import VoxelCloud


def extract_sentences(text: str, max_sentences: int = 50) -> list:
    """Extract first N sentences for demo."""
    import re
    sentences = re.split(r'[.!?]+', text)
    clean = []
    for sent in sentences[:max_sentences]:
        sent = sent.strip()
        if len(sent) >= 10:
            sent = ' '.join(sent.split())
            clean.append(sent)
    return clean


def main():
    print("=" * 70)
    print("Genesis Foundation Demo - Quick Training")
    print("=" * 70)

    # Demo on just 3 documents
    demo_docs = [
        '/usr/lib/alembic/data/datasets/curated/foundation/tao_te_ching.txt',
        '/usr/lib/alembic/data/datasets/curated/foundation/bhagavad_gita.txt',
        '/usr/lib/alembic/data/datasets/curated/foundation/art_of_war_sun_tzu.txt'
    ]

    # Check files exist
    for doc in demo_docs:
        if not Path(doc).exists():
            print(f"❌ File not found: {doc}")
            return 1

    print(f"\n📚 Demo training on 3 Foundation documents:")
    for doc in demo_docs:
        print(f"  - {Path(doc).name}")

    # Initialize
    print("\n🔧 Initializing components...")
    freq_analyzer = TextFrequencyAnalyzer(512, 512)
    origin = Origin(512, 512, use_gpu=False)

    # Create voxel cloud with collapse enabled
    collapse_config = {
        'harmonic_tolerance': 0.05,
        'cosine_threshold': 0.85,
        'octave_tolerance': 0,
        'enable': True
    }
    synthesis_config = {
        'use_resonance_weighting': True,
        'weight_function': 'linear',
        'resonance_boost': 2.0,
        'distance_decay': 0.5
    }
    voxel_cloud = VoxelCloud(512, 512, 128, collapse_config, synthesis_config)

    # Process each document
    start_time = time.time()
    total_segments = 0

    for doc_path in demo_docs:
        doc_name = Path(doc_path).name
        print(f"\n📖 Processing: {doc_name}")

        with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Get limited sentences for demo
        sentences = extract_sentences(content, max_sentences=50)
        print(f"  Using first {len(sentences)} sentences")

        for i, sentence in enumerate(sentences):
            # Convert to frequency
            freq_spectrum, params = freq_analyzer.analyze(sentence)

            # Create proto-identity
            proto_identity = origin.Gen(
                params['gamma_params'],
                params['iota_params']
            )

            # Add to voxel cloud
            metadata = {
                'text': sentence,
                'source': doc_name,
                'modality': 'text'
            }
            voxel_cloud.add(proto_identity, freq_spectrum, metadata)
            total_segments += 1

        print(f"  ✅ Added {len(sentences)} segments")

    # Show results
    elapsed = time.time() - start_time
    num_protos = len(voxel_cloud.entries)
    compression = total_segments / num_protos if num_protos > 0 else 0
    resonances = sum(1 for p in voxel_cloud.entries if p.resonance_strength > 1)

    print("\n" + "=" * 70)
    print("Demo Training Complete!")
    print("=" * 70)
    print(f"  Total segments: {total_segments}")
    print(f"  Proto-identities: {num_protos}")
    print(f"  Compression ratio: {compression:.2f}x")
    print(f"  Resonant patterns: {resonances} ({100*resonances/num_protos:.1f}%)")
    print(f"  Training time: {elapsed:.1f}s")

    # Save demo model
    demo_path = '/tmp/genesis_demo_model.pkl'
    print(f"\n💾 Saving demo model to: {demo_path}")
    voxel_cloud.save(demo_path)

    # Test a few queries
    print("\n🧪 Testing Q&A on demo model...")
    test_queries = [
        "What is the Tao?",
        "What is dharma?",
        "What is the art of war?"
    ]

    for query in test_queries:
        print(f"\n  Query: {query}")
        query_freq, _ = freq_analyzer.analyze(query)
        visible_protos = voxel_cloud.query_viewport(query_freq, radius=50.0)

        if visible_protos:
            # Get top response
            top_entry = visible_protos[0]
            if 'text' in top_entry.metadata:
                response = top_entry.metadata['text'][:100]
                print(f"  Response: {response}...")
                print(f"  (from {top_entry.metadata.get('source', 'unknown')})")
        else:
            print("  No response found")

    print("\n✅ Demo complete!")
    print("\n📝 To train on all Foundation documents, run:")
    print("   python scripts/train_foundation.py")

    return 0


if __name__ == '__main__':
    sys.exit(main())