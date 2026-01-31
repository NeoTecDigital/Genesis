#!/usr/bin/env python3
"""Debug the full synthesis flow to see where text is lost."""
from src.pipeline.unified_encoder import UnifiedEncoder
from src.pipeline.unified_decoder import UnifiedDecoder
from src.memory.memory_hierarchy import MemoryHierarchy
from src.pipeline.fft_text_decoder import FFTTextDecoder

print("=" * 80)
print("Debug Synthesis Flow")
print("=" * 80)

# Initialize
memory = MemoryHierarchy(width=512, height=512, use_routing=True)
encoder = UnifiedEncoder(memory)
decoder = UnifiedDecoder(memory)
fft_decoder = FFTTextDecoder()

# Encode foundation text
text = "The supreme art of war is to subdue the enemy without fighting."
result = encoder.encode(text, destination='core', octaves=[4, 0])
print(f"\n[1] Encoded: '{text}'")
print(f"    Units: {len(result.octave_units)}")

# Encode query
query_text = "What is the art of war?"
query = encoder.encode(query_text, destination='experiential', octaves=[4, 0])
print(f"\n[2] Query: '{query_text}'")
print(f"    Units: {len(query.octave_units)}")

# Query core memory directly to see what we get
query_proto = query.octave_units[0].proto_identity
core_entries = memory.core_memory.query_by_proto_similarity(query_proto, max_results=20)
exp_entries = memory.experiential_memory.query_by_proto_similarity(query_proto, max_results=20)

print(f"\n[3] Retrieved from memory:")
print(f"    Core: {len(core_entries)} entries")
print(f"    Experiential: {len(exp_entries)} entries")

# Decode each entry to see what text we have
print(f"\n[4] Core memory contents (top 10):")
for i, entry in enumerate(core_entries[:10]):
    decoded = fft_decoder.decode_text(entry.proto_identity)
    print(f"    [{i+1}] Octave {entry.octave:+2d}: '{decoded}'")

print(f"\n[5] Experiential memory contents (top 10):")
for i, entry in enumerate(exp_entries[:10]):
    decoded = fft_decoder.decode_text(entry.proto_identity)
    print(f"    [{i+1}] Octave {entry.octave:+2d}: '{decoded}'")

# Now call decoder.decode() to see what happens
print(f"\n[6] Calling decoder.decode()...")
result = decoder.decode(
    query_proto,
    layers='both',
    octaves=[4, 0]
)

print(f"    Result text: '{result.text}'")
print(f"    Confidence: {result.confidence:.3f}")
print(f"    Source layers: {result.source_layers}")
print(f"    Octaves used: {result.octaves_used}")

# Try decode_to_summary directly
print(f"\n[7] Calling decoder.decode_to_summary()...")
summary = decoder.decode_to_summary(
    query_proto,
    layers='both',
    octaves=[4, 0]
)
print(f"    Summary: '{summary}'")

print("\n" + "=" * 80)
