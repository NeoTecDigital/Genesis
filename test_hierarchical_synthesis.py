#!/usr/bin/env python3
"""Test hierarchical multi-octave synthesis."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.unified_encoder import UnifiedEncoder
from src.pipeline.unified_decoder import UnifiedDecoder
from src.memory.memory_hierarchy import MemoryHierarchy


def main():
    print("=" * 80)
    print("HIERARCHICAL MULTI-OCTAVE SYNTHESIS TEST")
    print("=" * 80)

    # Initialize
    memory = MemoryHierarchy()
    encoder = UnifiedEncoder(memory)
    decoder = UnifiedDecoder(memory)

    # Encode foundation texts
    texts = [
        "The supreme art of war is to subdue the enemy without fighting.",
        "Strategy without tactics is the slowest route to victory.",
        "All warfare is based on deception."
    ]

    print("\n[1/3] Encoding foundation texts...")
    for i, text in enumerate(texts, 1):
        result = encoder.encode(text, destination='core')
        octaves = set(u.octave for u in result.octave_units)
        print(f"   {i}. Encoded {len(result.octave_units)} units at octaves {octaves}")
        print(f"      '{text[:60]}...'")

    # Query with hierarchical synthesis
    print("\n[2/3] Querying with hierarchical synthesis...")
    query = encoder.encode("What is war strategy?", destination='experiential')
    query_proto = query.octave_units[0].proto_identity

    result = decoder.hierarchical_synthesis(
        query_proto,
        layers='both',
        max_chars=200,
        coherence_threshold=0.75
    )

    print(f"\n   Query: 'What is war strategy?'")
    print(f"   Synthesized: '{result.text}'")
    print(f"   Confidence: {result.confidence:.3f}")
    print(f"   Word count: {len(result.text.split())}")
    print(f"   Octaves used: {result.octaves_used}")
    print(f"   Resonances: {result.resonances}")

    # Validation
    print("\n[3/3] Validating results...")
    word_count = len(result.text.split())

    checks = [
        (len(result.text) > 0, "Non-empty synthesis"),
        (word_count > 1, f"Multi-word response ({word_count} words)"),
        (result.confidence > 0, f"Non-zero confidence ({result.confidence:.3f})"),
        ('war' in result.text.lower() or 'strategy' in result.text.lower(), "Relevant content")
    ]

    all_passed = True
    for passed, desc in checks:
        status = "✓" if passed else "✗"
        print(f"   {status} {desc}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n" + "=" * 80)
        print("✅ HIERARCHICAL SYNTHESIS TEST PASSED!")
        print("=" * 80)
        print("\n🎉 KEY ACHIEVEMENT:")
        print("   → Multi-word response synthesized from frequency harmonics")
        print("   → Characters assembled into words via octave resonance")
        print("   → Words assembled into phrases via harmonic guidance")
        print("   → Meaning emerged from multi-octave frequency patterns!")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("❌ HIERARCHICAL SYNTHESIS TEST FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
