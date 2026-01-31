"""Test synthesis with meaningful text."""
from src.pipeline.unified_encoder import UnifiedEncoder
from src.pipeline.unified_decoder import UnifiedDecoder
from src.memory.memory_hierarchy import MemoryHierarchy

memory = MemoryHierarchy()
encoder = UnifiedEncoder(memory)
decoder = UnifiedDecoder(memory)

# Encode several sentences
texts = [
    "The supreme art of war is to subdue the enemy without fighting.",
    "Strategy without tactics is the slowest route to victory.",
    "Tactics without strategy is the noise before defeat."
]

for text in texts:
    encoder.encode(text, destination='core')
    print(f"Encoded: {text[:50]}...")

# Query
query = encoder.encode("What is war strategy?", destination='experiential')
synthesis = decoder.decode(
    query.octave_units[0].proto_identity,
    layers='both',
    octaves=[4, 0]
)

print(f"\nQuery: 'What is war strategy?'")
print(f"Synthesis: '{synthesis.text}'")
print(f"Length: {len(synthesis.text)} chars")
print(f"Confidence: {synthesis.confidence:.3f}")
print(f"Source layers: {synthesis.source_layers}")

# Validate
assert len(synthesis.text) > 0, "Should return text"
assert synthesis.text != '...', "Should not be just ellipsis"
print("\n✓ Real synthesis test PASSED")
