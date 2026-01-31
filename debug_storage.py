#!/usr/bin/env python3
"""Debug storage to see if protos are corrupted when stored/retrieved."""
from src.pipeline.unified_encoder import UnifiedEncoder
from src.memory.memory_hierarchy import MemoryHierarchy
from src.pipeline.fft_text_decoder import FFTTextDecoder
import numpy as np

print("=" * 80)
print("Debug Storage: Are Protos Corrupted in Memory?")
print("=" * 80)

# Initialize
memory = MemoryHierarchy(width=512, height=512, use_routing=True)
encoder = UnifiedEncoder(memory)
decoder = FFTTextDecoder()

# Test 1: Store and retrieve a single character
print("\n[Test 1] Store and retrieve single character 'T':")
text = "T"
result = encoder.encode(text, destination='core', octaves=[4])
print(f"   Stored '{text}' at octave +4")

# Get the proto we just stored
query_proto = result.octave_units[0].proto_identity
original_text = decoder.decode_text(query_proto)
print(f"   Original proto decodes to: '{original_text}'")

# Query memory to retrieve it
entries = memory.core_memory.query_by_proto_similarity(query_proto, max_results=1)
if entries:
    retrieved_proto = entries[0].proto_identity
    retrieved_text = decoder.decode_text(retrieved_proto)
    print(f"   Retrieved proto decodes to: '{retrieved_text}'")

    # Check if protos are identical
    proto_match = np.allclose(query_proto, retrieved_proto, atol=1e-6)
    print(f"   Proto arrays match: {proto_match}")
    if not proto_match:
        diff = np.abs(query_proto - retrieved_proto).max()
        print(f"   Max difference: {diff:.6e}")
else:
    print("   ✗ No entries retrieved!")

# Test 2: Store and retrieve a word
print("\n[Test 2] Store and retrieve word 'war':")
text = "war"
result = encoder.encode(text, destination='core', octaves=[0])
print(f"   Stored '{text}' at octave 0")

query_proto = result.octave_units[0].proto_identity
original_text = decoder.decode_text(query_proto)
print(f"   Original proto decodes to: '{original_text}'")

entries = memory.core_memory.query_by_proto_similarity(query_proto, max_results=1)
if entries:
    retrieved_proto = entries[0].proto_identity
    retrieved_text = decoder.decode_text(retrieved_proto)
    print(f"   Retrieved proto decodes to: '{retrieved_text}'")

    proto_match = np.allclose(query_proto, retrieved_proto, atol=1e-6)
    print(f"   Proto arrays match: {proto_match}")
    if not proto_match:
        diff = np.abs(query_proto - retrieved_proto).max()
        print(f"   Max difference: {diff:.6e}")
else:
    print("   ✗ No entries retrieved!")

# Test 3: Check if clustering is merging different protos
print("\n[Test 3] Check if clustering merges different characters:")
chars = ['a', 'b', 'c']
for char in chars:
    result = encoder.encode(char, destination='core', octaves=[4])
    query_proto = result.octave_units[0].proto_identity
    entries = memory.core_memory.query_by_proto_similarity(query_proto, max_results=1)
    if entries:
        retrieved_text = decoder.decode_text(entries[0].proto_identity)
        match = "✓" if retrieved_text == char else "✗"
        print(f"   {match} Stored '{char}' → Retrieved '{retrieved_text}'")

print("\n" + "=" * 80)
