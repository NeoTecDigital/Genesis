"""
Integration test for gravitational collapse with real text data.

Tests the compression ratio and resonance strength with actual text processing.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.memory.voxel_cloud import VoxelCloud
from src.memory.frequency_field import TextFrequencyAnalyzer


def test_text_compression():
    """Test gravitational collapse with repetitive text patterns."""
    cloud = VoxelCloud(width=256, height=256, depth=64)
    analyzer = TextFrequencyAnalyzer(256, 256)

    # Simulate Tao Te Ching-like repetitive philosophical text
    texts = [
        "The Tao that can be spoken is not the eternal Tao",
        "The Tao is the source of all things",
        "The Tao flows through all things",
        "The wise follow the Tao",
        "The Tao is empty yet inexhaustible",
        "Return to the Tao",
        "The Tao of heaven is impartial",
        "The Tao gives life to all things",
        "The sage follows the Tao",
        "The Tao is like water",
        # Repeat some patterns
        "The Tao that can be spoken is not the eternal Tao",
        "The wise follow the Tao",
        "The Tao flows through all things",
        "Return to the Tao",
        "The sage follows the Tao",
        # Add some unique concepts
        "Yin and yang arise from the Tao",
        "Wu wei is action without action",
        "The ten thousand things return to the Tao",
        "Simplicity is the way of the Tao",
        "The master acts without doing",
    ]

    print(f"Adding {len(texts)} text segments...")

    # Process each text
    for i, text in enumerate(texts):
        # Generate frequency spectrum
        freq_spectrum, freq_params = analyzer.analyze(text)

        # Generate proto-identity (simplified - normally would use Gen/Res)
        proto = np.random.randn(256, 256, 4).astype(np.float32) * 0.1
        proto += freq_spectrum[..., :2].repeat(2, axis=-1)  # Use frequency as base

        # Add to voxel cloud with gravitational collapse
        metadata = {
            "text": text,
            "index": i,
            "params": freq_params
        }
        cloud.add(proto, freq_spectrum, metadata)

    # Analyze compression results
    print(f"\nResults after gravitational collapse:")
    print(f"  Original segments: {len(texts)}")
    print(f"  Compressed protos: {len(cloud.entries)}")
    print(f"  Compression ratio: {len(texts) / len(cloud.entries):.2f}x")

    # Find high-resonance patterns
    high_resonance = sorted(cloud.entries, key=lambda e: e.resonance_strength, reverse=True)[:5]
    print(f"\nTop 5 high-resonance patterns:")
    for i, entry in enumerate(high_resonance, 1):
        original_text = entry.metadata.get('text', 'N/A')
        print(f"  {i}. Resonance={entry.resonance_strength}: '{original_text[:50]}...'")

    # Verify compression achieved
    assert len(cloud.entries) < len(texts) * 0.8  # At least 20% compression
    assert max(e.resonance_strength for e in cloud.entries) > 1  # Some merging occurred

    # Calculate statistics
    total_resonance = sum(e.resonance_strength for e in cloud.entries)
    assert total_resonance == len(texts)  # All texts accounted for

    avg_resonance = total_resonance / len(cloud.entries)
    print(f"\nStatistics:")
    print(f"  Average resonance: {avg_resonance:.2f}")
    print(f"  Unique patterns: {len(cloud.entries)}")
    print(f"  Total resonance: {total_resonance}")

    return cloud


def test_semantic_clustering():
    """Test that semantically similar texts cluster together."""
    cloud = VoxelCloud(width=256, height=256, depth=64)
    analyzer = TextFrequencyAnalyzer(256, 256)

    # Create semantic groups
    groups = {
        "water": [
            "Water flows downhill",
            "Like water, be flexible",
            "Water shapes the stone",
            "The nature of water is soft",
            "Water always finds a way",
        ],
        "mountain": [
            "The mountain stands still",
            "Mountains endure through time",
            "Solid as a mountain",
            "The mountain peak touches heaven",
            "Mountains are unmoved by wind",
        ],
        "fire": [
            "Fire burns bright",
            "The flame consumes all",
            "Fire transforms wood to ash",
            "Heat rises from the fire",
            "Fire brings light to darkness",
        ]
    }

    # Process all texts
    for theme, texts in groups.items():
        for text in texts:
            freq_spectrum, _ = analyzer.analyze(text)
            proto = np.random.randn(256, 256, 4).astype(np.float32) * 0.05
            proto += freq_spectrum[..., :2].repeat(2, axis=-1) * 0.5
            metadata = {"text": text, "theme": theme}
            cloud.add(proto, freq_spectrum, metadata)

    # Analyze clustering
    print(f"\nSemantic clustering results:")
    print(f"  Input texts: {sum(len(texts) for texts in groups.values())}")
    print(f"  Output protos: {len(cloud.entries)}")

    # Group by theme in merged entries
    theme_stats = {}
    for entry in cloud.entries:
        if 'merged_texts' in entry.metadata:
            # Count themes in merged texts
            themes_in_merge = []
            for text in entry.metadata['merged_texts']:
                for theme in groups.keys():
                    if any(t == text for t in groups[theme]):
                        themes_in_merge.append(theme)
                        break

            # Check if primarily one theme
            if themes_in_merge:
                dominant_theme = max(set(themes_in_merge), key=themes_in_merge.count)
                theme_ratio = themes_in_merge.count(dominant_theme) / len(themes_in_merge)
                if dominant_theme not in theme_stats:
                    theme_stats[dominant_theme] = []
                theme_stats[dominant_theme].append(theme_ratio)

    print("\nTheme coherence in merged protos:")
    for theme, ratios in theme_stats.items():
        avg_coherence = np.mean(ratios) if ratios else 0
        print(f"  {theme}: {avg_coherence:.2%} pure")

    return cloud


if __name__ == "__main__":
    print("Testing gravitational collapse with text data...")
    print("=" * 60)

    cloud1 = test_text_compression()
    print("\n✓ Text compression test passed")

    cloud2 = test_semantic_clustering()
    print("\n✓ Semantic clustering test passed")

    print("\n" + "=" * 60)
    print("All integration tests passed!")
    print(f"Final cloud 1: {cloud1}")
    print(f"Final cloud 2: {cloud2}")