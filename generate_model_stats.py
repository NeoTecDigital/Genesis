#!/usr/bin/env python3
"""Generate comprehensive model statistics from trained voxel cloud."""
import sys
import pickle
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, '/home/persist/alembic/genesis')
from src.memory.voxel_cloud import VoxelCloud

def analyze_model(model_path: str):
    """Analyze trained model and generate statistics."""
    print("=" * 70)
    print("GENESIS MODEL ANALYSIS")
    print("=" * 70)

    # Load model
    print(f"\n📂 Loading model: {model_path}")
    voxel_cloud = VoxelCloud()
    voxel_cloud.load(model_path)

    print(f"✅ Loaded successfully")
    print(f"   {voxel_cloud}")

    # Basic statistics
    print("\n" + "=" * 70)
    print("BASIC STATISTICS")
    print("=" * 70)
    print(f"  Total proto-identities: {len(voxel_cloud.entries)}")
    print(f"  Voxel cloud dimensions: {voxel_cloud.width}x{voxel_cloud.height}x{voxel_cloud.depth}")

    # Modality distribution
    modalities = [e.metadata.get('modality', 'text') for e in voxel_cloud.entries]
    modality_counts = Counter(modalities)
    print(f"\n  Modality distribution:")
    for modality, count in modality_counts.items():
        print(f"    {modality}: {count} ({count/len(voxel_cloud.entries)*100:.1f}%)")

    # Resonance statistics
    print("\n" + "=" * 70)
    print("RESONANCE ANALYSIS (Gravitational Collapse)")
    print("=" * 70)

    resonances = [e.resonance_strength for e in voxel_cloud.entries]
    unique_protos = sum(1 for r in resonances if r == 1)
    merged_protos = sum(1 for r in resonances if r > 1)
    max_resonance = max(resonances)
    avg_resonance = np.mean(resonances)

    print(f"  Unique proto-identities: {unique_protos} ({unique_protos/len(voxel_cloud.entries)*100:.1f}%)")
    print(f"  Merged proto-identities: {merged_protos} ({merged_protos/len(voxel_cloud.entries)*100:.1f}%)")
    print(f"  Maximum resonance strength: {max_resonance}")
    print(f"  Average resonance strength: {avg_resonance:.2f}")

    if merged_protos > 0:
        # Calculate compression ratio
        total_segments = sum(resonances)
        compression_ratio = total_segments / len(voxel_cloud.entries)
        print(f"\n  📊 COMPRESSION METRICS:")
        print(f"     Total input segments: {total_segments}")
        print(f"     Stored proto-identities: {len(voxel_cloud.entries)}")
        print(f"     Compression ratio: {compression_ratio:.2f}x")

    # Frequency distribution
    print("\n" + "=" * 70)
    print("FREQUENCY DISTRIBUTION")
    print("=" * 70)

    fundamentals = [e.fundamental_freq for e in voxel_cloud.entries]
    octaves = [e.octave for e in voxel_cloud.entries]

    print(f"  Fundamental frequency range: {min(fundamentals):.3f} - {max(fundamentals):.3f} Hz")
    print(f"  Mean fundamental frequency: {np.mean(fundamentals):.3f} Hz")
    print(f"  Octave distribution:")
    octave_counts = Counter(octaves)
    for octave in sorted(octave_counts.keys()):
        count = octave_counts[octave]
        print(f"    Octave {octave}: {count} ({count/len(octaves)*100:.1f}%)")

    # Proto-identity characteristics
    print("\n" + "=" * 70)
    print("PROTO-IDENTITY CHARACTERISTICS")
    print("=" * 70)

    proto_means = [e.proto_identity.mean() for e in voxel_cloud.entries]
    proto_stds = [e.proto_identity.std() for e in voxel_cloud.entries]

    print(f"  Proto-identity mean values:")
    print(f"    Range: {min(proto_means):.6f} - {max(proto_means):.6f}")
    print(f"    Average: {np.mean(proto_means):.6f}")
    print(f"  Proto-identity std values:")
    print(f"    Range: {min(proto_stds):.6f} - {max(proto_stds):.6f}")
    print(f"    Average: {np.mean(proto_stds):.6f}")

    # Spatial distribution
    print("\n" + "=" * 70)
    print("SPATIAL DISTRIBUTION")
    print("=" * 70)

    positions = np.array([e.position for e in voxel_cloud.entries])
    print(f"  X range: {positions[:,0].min():.1f} - {positions[:,0].max():.1f}")
    print(f"  Y range: {positions[:,1].min():.1f} - {positions[:,1].max():.1f}")
    print(f"  Z range: {positions[:,2].min():.1f} - {positions[:,2].max():.1f}")

    # Configuration
    print("\n" + "=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(f"  Collapse config: {voxel_cloud.collapse_config}")
    print(f"  Synthesis config: {voxel_cloud.synthesis_config}")

    # Top resonant patterns
    if merged_protos > 0:
        print("\n" + "=" * 70)
        print("TOP 10 MOST RESONANT PATTERNS")
        print("=" * 70)

        sorted_entries = sorted(voxel_cloud.entries,
                              key=lambda e: e.resonance_strength,
                              reverse=True)

        for i, entry in enumerate(sorted_entries[:10], 1):
            text = entry.metadata.get('text', 'N/A')
            resonance = entry.resonance_strength
            freq = entry.fundamental_freq
            print(f"\n  {i}. Resonance: {resonance}x | Frequency: {freq:.3f} Hz")
            print(f"     Text: {text[:80]}...")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    model_path = "/usr/lib/alembic/checkpoints/genesis/tao_voxel_cloud.pkl"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    analyze_model(model_path)
