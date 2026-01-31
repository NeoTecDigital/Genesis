#!/usr/bin/env python3
"""Test TaylorSynthesizer integration into UnifiedDecoder."""
import numpy as np
from src.pipeline.unified_decoder import UnifiedDecoder
from src.memory.memory_hierarchy import MemoryHierarchy
from src.pipeline.unified_encoder import UnifiedEncoder

print("=" * 80)
print("Testing TaylorSynthesizer Integration into UnifiedDecoder")
print("=" * 80)

# Initialize memory hierarchy
print("\n[1/5] Initializing memory hierarchy...")
memory = MemoryHierarchy(width=512, height=512, use_routing=True)

# Initialize proto_unity_carrier properly
print("   ✓ Initializing proto_unity_carrier...")
from src.origin import Origin
origin = Origin(width=512, height=512, use_gpu=False)
carrier = memory.create_carrier(origin)
print("   ✓ Memory hierarchy initialized")

# Encode some foundation text
print("\n[2/5] Encoding foundation text to core memory...")
encoder = UnifiedEncoder(memory)
text = "The supreme art of war is to subdue the enemy without fighting."
result = encoder.encode(text, destination='core', octaves=[4, 0])
print(f"   ✓ Encoded {len(result.octave_units)} units to core memory")

# Initialize decoder
print("\n[3/5] Initializing UnifiedDecoder...")
decoder = UnifiedDecoder(memory)
print("   ✓ UnifiedDecoder ready")

# Test 1: Backward compatibility (use_taylor_synthesis=False, default)
print("\n[4/5] Testing backward compatibility (use_taylor_synthesis=False)...")
query = encoder.encode("What is war?", destination='experiential', octaves=[4, 0])
if query.octave_units:
    try:
        result = decoder.decode(
            query.octave_units[0].proto_identity,
            layers='both',
            octaves=[4, 0],
            use_taylor_synthesis=False  # Explicit default
        )
        print(f"   ✓ Legacy decode() returns DecodingResult: {type(result).__name__}")
        print(f"   ✓ Text: '{result.text}' (confidence: {result.confidence:.3f})")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        exit(1)
else:
    print("   ✗ FAILED: No query units encoded")
    exit(1)

# Test 2: Taylor synthesis (use_taylor_synthesis=True)
print("\n[5/5] Testing Taylor synthesis (use_taylor_synthesis=True)...")
try:
    proto_identities, explanation = decoder.decode(
        query.octave_units[0].proto_identity,
        layers='both',
        octaves=[4, 0],
        use_taylor_synthesis=True,
        query_text="What is war?"
    )
    print(f"   ✓ Taylor synthesis returns tuple: ({len(proto_identities)} protos, explanation)")
    print(f"   ✓ Explanation: {explanation[:100]}...")
    print(f"   ✓ Proto-identities shape: {proto_identities[0].shape}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Verify ValueError for missing query_text
print("\n[BONUS] Testing ValueError for missing query_text...")
try:
    decoder.decode(
        query.octave_units[0].proto_identity,
        use_taylor_synthesis=True
        # query_text missing!
    )
    print("   ✗ FAILED: Should raise ValueError for missing query_text")
    exit(1)
except ValueError as e:
    print(f"   ✓ Correctly raised ValueError: {e}")

print("\n" + "=" * 80)
print("TAYLOR INTEGRATION VALIDATED!")
print("- Backward compatibility maintained (use_taylor_synthesis=False)")
print("- Taylor synthesis working (use_taylor_synthesis=True)")
print("- Parameter validation working (ValueError for missing query_text)")
print("=" * 80)
