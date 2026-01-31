#!/usr/bin/env python3
"""
Train Genesis on all Foundation documents.

Usage:
    python scripts/train_foundation.py [options]

Options:
    --output PATH              Output voxel cloud path
    --enable-collapse          Enable gravitational collapse (default: True)
    --harmonic-tolerance FLOAT Collapse harmonic tolerance (default: 0.05)
    --cosine-threshold FLOAT   Collapse cosine threshold (default: 0.85)
    --batch-size INT          Documents to process in batch (default: 5)
    --checkpoint-interval INT  Save checkpoint every N documents
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

from src.origin import Origin
from src.memory.frequency_field import TextFrequencyAnalyzer
from src.memory.voxel_cloud import VoxelCloud


def extract_sentences(text: str, min_length: int = 10) -> List[str]:
    """Extract meaningful sentences from text."""
    import re

    # Split on sentence boundaries
    sentences = re.split(r'[.!?]+', text)

    # Clean and filter
    clean_sentences = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) >= min_length:
            # Remove excessive whitespace
            sent = ' '.join(sent.split())
            clean_sentences.append(sent)

    return clean_sentences


def process_document(
    doc_path: Path,
    freq_analyzer: TextFrequencyAnalyzer,
    origin: Origin,
    voxel_cloud: VoxelCloud,
    doc_index: int
) -> int:
    """Process a single Foundation document."""
    try:
        # Read document
        with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Extract sentences
        sentences = extract_sentences(content)
        if not sentences:
            print(f"  ⚠️  No valid sentences found in {doc_path.name}")
            return 0

        print(f"  📝 Found {len(sentences)} segments in {doc_path.name}")

        # Process each sentence
        segment_count = 0
        for i, sentence in enumerate(sentences):
            # Convert to frequency
            freq_spectrum, params_dict = freq_analyzer.analyze(sentence)

            # Apply morphisms to create proto-identity
            proto_identity = origin.Gen(
                params_dict['gamma_params'],
                params_dict['iota_params']
            )

            # Store in voxel cloud with metadata
            metadata = {
                'text': sentence,
                'source': doc_path.name,
                'doc_index': doc_index,
                'segment_index': i,
                'modality': 'text',
                'gamma_params': params_dict['gamma_params'],
                'iota_params': params_dict['iota_params']
            }
            voxel_cloud.add(proto_identity, freq_spectrum, metadata)
            segment_count += 1

            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"    Processed {i+1}/{len(sentences)} segments")

        return segment_count

    except Exception as e:
        print(f"  ❌ Error processing {doc_path.name}: {e}")
        return 0


def save_checkpoint(voxel_cloud: VoxelCloud, path: str, doc_num: int):
    """Save checkpoint of current voxel cloud."""
    checkpoint_path = path.replace('.pkl', f'_checkpoint_{doc_num}.pkl')
    voxel_cloud.save(checkpoint_path)
    print(f"  💾 Checkpoint saved: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Genesis on Foundation documents')
    parser.add_argument('--output',
                       default='/usr/lib/alembic/checkpoints/genesis/foundation_voxel_cloud.pkl',
                       help='Output voxel cloud path')
    parser.add_argument('--enable-collapse', action='store_true', default=True,
                       help='Enable gravitational collapse (default: True)')
    parser.add_argument('--harmonic-tolerance', type=float, default=0.05,
                       help='Collapse harmonic tolerance (default: 0.05)')
    parser.add_argument('--cosine-threshold', type=float, default=0.85,
                       help='Collapse cosine threshold (default: 0.85)')
    parser.add_argument('--batch-size', type=int, default=5,
                       help='Documents to process in batch (not used currently)')
    parser.add_argument('--checkpoint-interval', type=int, default=5,
                       help='Save checkpoint every N documents')
    args = parser.parse_args()

    # Find all foundation texts
    foundation_dir = Path('/usr/lib/alembic/data/datasets/curated/foundation')
    if not foundation_dir.exists():
        print(f"❌ Foundation directory not found: {foundation_dir}")
        return 1

    texts = sorted(foundation_dir.glob('*.txt'))

    # Calculate total size
    total_size = sum(t.stat().st_size for t in texts) / 1024 / 1024

    print("=" * 70)
    print("Genesis Foundation Model Training")
    print("=" * 70)
    print(f"Found {len(texts)} foundation documents")
    print(f"Total size: {total_size:.1f} MB")
    print(f"Output: {args.output}")
    print(f"Gravitational collapse: {'Enabled' if args.enable_collapse else 'Disabled'}")
    if args.enable_collapse:
        print(f"  - Harmonic tolerance: {args.harmonic_tolerance}")
        print(f"  - Cosine threshold: {args.cosine_threshold}")
    print("=" * 70)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize components
    print("\n🔧 Initializing components...")
    freq_analyzer = TextFrequencyAnalyzer(512, 512)
    origin = Origin(512, 512, use_gpu=False)  # Use CPU for stability

    # Create voxel cloud with collapse configuration
    collapse_config = {
        'harmonic_tolerance': args.harmonic_tolerance,
        'cosine_threshold': args.cosine_threshold,
        'octave_tolerance': 0,
        'enable': args.enable_collapse
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

    print("\n📚 Processing Foundation documents...")
    for i, text_file in enumerate(texts, 1):
        print(f"\n[{i}/{len(texts)}] Processing: {text_file.name}")

        # Process document
        num_segments = process_document(
            text_file, freq_analyzer, origin, voxel_cloud, i
        )
        total_segments += num_segments

        # Save checkpoint if needed
        if i % args.checkpoint_interval == 0:
            save_checkpoint(voxel_cloud, args.output, i)

    # Save final voxel cloud
    print(f"\n💾 Saving final voxel cloud to: {args.output}")
    voxel_cloud.save(args.output)

    # Calculate metrics
    elapsed = time.time() - start_time
    num_protos = len(voxel_cloud.entries)
    compression = total_segments / num_protos if num_protos > 0 else 0

    # Print summary
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print(f"  Total documents: {len(texts)}")
    print(f"  Total segments: {total_segments:,}")
    print(f"  Proto-identities: {num_protos:,}")
    if args.enable_collapse:
        print(f"  Compression ratio: {compression:.2f}x")
        # Count resonances
        resonances = sum(1 for p in voxel_cloud.entries if p.resonance_strength > 1)
        print(f"  Resonant patterns: {resonances} ({100*resonances/num_protos:.1f}%)")
    print(f"  Training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Output: {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())