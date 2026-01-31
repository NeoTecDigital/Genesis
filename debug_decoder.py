#!/usr/bin/env python3
"""Debug decoder to see what's being retrieved and synthesized."""
from pathlib import Path
from src.pipeline.unified_encoder import UnifiedEncoder
from src.pipeline.unified_decoder import UnifiedDecoder
from src.memory.memory_hierarchy import MemoryHierarchy
from src.pipeline.fft_text_decoder import FFTTextDecoder

print("=" * 80)
print("Debug Decoder: What's Being Retrieved?")
print("=" * 80)

# Initialize
print("\n[1/3] Initializing Genesis components...")
memory = MemoryHierarchy(width=512, height=512, use_routing=True)
encoder = UnifiedEncoder(memory)
decoder = UnifiedDecoder(memory)
fft_decoder = FFTTextDecoder()
print("   ✓ Ready")

# Encode foundation text
print("\n[2/3] Encoding foundation text...")
text = "The supreme art of war is to subdue the enemy without fighting."
result = encoder.encode(text, destination='core', octaves=[4, 0])
print(f"   ✓ Encoded: '{text}'")
print(f"   ✓ Units stored: {len(result.octave_units)}")

# Check what's in core memory
print("\n[3/3] Querying and examining retrieved protos...")
query_text = "What is the art of war?"
query = encoder.encode(query_text, destination='experiential', octaves=[4, 0])
print(f"   Query: '{query_text}'")

# Query core memory directly
query_proto = query.octave_units[0].proto_identity
entries = memory.core_memory.query_by_proto_similarity(query_proto, max_results=20)
print(f"   Retrieved {len(entries)} entries from core memory")

# Decode first 10 entries to see what text they contain
print("\n   Top 10 retrieved proto-identities:")
for i, entry in enumerate(entries[:10]):
    decoded_text = fft_decoder.decode_text(entry.proto_identity)
    octave = entry.octave
    resonance = entry.resonance_strength
    print(f"      [{i+1}] Octave {octave:+2d}, Resonance {resonance:3d}: '{decoded_text}'")

# Now check experiential memory too
exp_entries = memory.experiential_memory.query_by_proto_similarity(query_proto, max_results=10)
print(f"\n   Retrieved {len(exp_entries)} entries from experiential memory")
print("   Top 5 from experiential:")
for i, entry in enumerate(exp_entries[:5]):
    decoded_text = fft_decoder.decode_text(entry.proto_identity)
    octave = entry.octave
    resonance = entry.resonance_strength
    print(f"      [{i+1}] Octave {octave:+2d}, Resonance {resonance:3d}: '{decoded_text}'")

print("\n" + "=" * 80)
