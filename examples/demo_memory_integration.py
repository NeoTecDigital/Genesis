"""Demonstration of Memory Integration Components.

This script demonstrates the prototype components working together:
1. MemoryRouter - Routes proto-identities to appropriate memory layers
2. UnifiedEncoder - Single API for encoding with routing and storage
3. UnifiedDecoder - Single API for querying and decoding across layers
"""

import numpy as np
from datetime import datetime

from src.memory.memory_hierarchy import MemoryHierarchy
from src.memory.memory_router import MemoryRouter
from src.pipeline.unified_encoder import UnifiedEncoder
from src.pipeline.unified_decoder import UnifiedDecoder


def demo_memory_integration():
    """Demonstrate the memory integration pipeline."""

    print("=" * 60)
    print("Memory Integration Demo")
    print("=" * 60)

    # Initialize memory hierarchy
    print("\n1. Initializing memory hierarchy...")
    hierarchy = MemoryHierarchy()
    print("   ✓ Created core and experiential memory layers")

    # Initialize components
    print("\n2. Initializing integration components...")
    router = MemoryRouter()
    encoder = UnifiedEncoder(hierarchy)
    decoder = UnifiedDecoder(hierarchy)
    print("   ✓ Router, encoder, and decoder ready")

    # Demonstrate routing logic
    print("\n3. Demonstrating routing logic...")
    print("   Foundation text → core memory")
    print("   Query text → experiential memory")
    print("   Phrases → experiential only")
    print("   Characters/words → both layers (configurable)")

    # Encode foundation knowledge
    print("\n4. Encoding foundation knowledge...")
    foundation_text = "The quick brown fox jumps over the lazy dog"
    result1 = encoder.encode(
        text=foundation_text,
        destination='auto',
        octaves=[4, 0],  # Character and word level
        metadata={'source': 'training', 'is_training': True}
    )
    print(f"   Encoded: {len(result1.octave_units)} units")
    print(f"   Core stored: {result1.core_added}")
    print(f"   Experiential stored: {result1.experiential_added}")

    # Encode query
    print("\n5. Encoding query/inference text...")
    query_text = "Hello world"
    result2 = encoder.encode(
        text=query_text,
        destination='auto',
        octaves=[4, 0, -2],  # Character, word, and phrase
        metadata={'source': 'inference', 'is_query': True}
    )
    print(f"   Encoded: {len(result2.octave_units)} units")
    print(f"   Core stored: {result2.core_added}")
    print(f"   Experiential stored: {result2.experiential_added}")

    # Show routing statistics
    print("\n6. Routing statistics...")
    stats = encoder.memory_router.get_routing_stats()
    print(f"   Total routed: {stats['total']}")
    print(f"   To core: {stats['core']}")
    print(f"   To experiential: {stats['experiential']}")
    print(f"   To both: {stats['both']}")
    print("   By octave:", dict(stats['by_octave']))
    print("   By reason:", dict(stats['by_reason']))

    # Demonstrate decoding
    print("\n7. Demonstrating cross-layer decoding...")
    if result2.octave_units:
        # Use first unit as query
        query_proto = result2.octave_units[0].proto_identity

        # Decode from both layers
        decode_result = decoder.decode(
            query_proto=query_proto,
            layers='both',
            octaves=[4, 0],
            blend_mode='weighted'
        )

        print(f"   Source layers: {decode_result.source_layers}")
        print(f"   Confidence: {decode_result.confidence:.3f}")
        print(f"   Resonances: {decode_result.resonances}")

    # Show encoder statistics
    print("\n8. Overall encoder statistics...")
    enc_stats = encoder.get_statistics()
    print(f"   Total encoded: {enc_stats['total_encoded']}")
    print(f"   Core storage ratio: {enc_stats['efficiency']['core_ratio']:.2%}")
    print(f"   Experiential ratio: {enc_stats['efficiency']['experiential_ratio']:.2%}")

    # Show decoder statistics
    print("\n9. Overall decoder statistics...")
    dec_stats = decoder.get_statistics()
    print(f"   Total queries: {dec_stats['total_queries']}")
    print(f"   Successful decodes: {dec_stats['successful_decodes']}")
    print(f"   Success rate: {dec_stats['success_rate']:.2%}")

    # Demonstrate octave-aware routing
    print("\n10. Octave-aware routing example...")
    phrase_result = encoder.encode(
        text="This is a longer phrase for testing",
        octaves=[-2, -4],  # Phrase levels only
        metadata={'test': 'phrase_routing'}
    )
    print(f"   Phrase units encoded: {len(phrase_result.octave_units)}")
    print(f"   Routed to experiential: {phrase_result.experiential_added}")
    print(f"   Routed to core: {phrase_result.core_added}")

    print("\n" + "=" * 60)
    print("Demo complete! All components working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    demo_memory_integration()