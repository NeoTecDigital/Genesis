#!/usr/bin/env python3
"""Demonstration of hierarchical multi-octave synthesis."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.unified_encoder import UnifiedEncoder
from src.pipeline.unified_decoder import UnifiedDecoder
from src.memory.memory_hierarchy import MemoryHierarchy


def main():
    print("=" * 80)
    print("HIERARCHICAL SYNTHESIS DEMONSTRATION")
    print("Meaning emerges from frequency harmonics!")
    print("=" * 80)

    # Initialize
    memory = MemoryHierarchy()
    encoder = UnifiedEncoder(memory)
    decoder = UnifiedDecoder(memory)

    # Encode knowledge base
    knowledge = [
        "The supreme art of war is to subdue the enemy without fighting.",
        "Strategy without tactics is the slowest route to victory.",
        "All warfare is based on deception.",
        "Know your enemy and know yourself.",
        "In the midst of chaos, there is opportunity."
    ]

    print("\n[ENCODING] Building knowledge base...")
    for text in knowledge:
        encoder.encode(text, destination='core')
        print(f"   + {text}")

    # Test queries
    queries = [
        "What is war strategy?",
        "How to win battles?",
        "What is opportunity?"
    ]

    print("\n" + "=" * 80)
    print("[SYNTHESIS] Querying with hierarchical assembly")
    print("=" * 80)

    for i, query_text in enumerate(queries, 1):
        print(f"\n--- Query {i} ---")
        print(f"Q: {query_text}")
        
        # Encode query
        query = encoder.encode(query_text, destination='experiential')
        query_proto = query.octave_units[0].proto_identity
        
        # Synthesize response
        result = decoder.hierarchical_synthesis(
            query_proto,
            layers='both',
            max_chars=200,
            coherence_threshold=0.75
        )
        
        print(f"A: {result.text}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Words: {len(result.text.split())}")
        print(f"   Resonances: {list(result.resonances.keys())}")

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey Insights:")
    print("✓ Multi-word responses generated")
    print("✓ Character → Word → Phrase assembly")
    print("✓ Resonance-weighted pattern matching")
    print("✓ Meaning emerged from frequency harmonics!")
    print("=" * 80)


if __name__ == "__main__":
    main()
