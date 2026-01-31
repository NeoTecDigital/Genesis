#!/usr/bin/env python3
"""Debug proto-identity similarity calculations."""
from src.pipeline.fft_text_encoder import FFTTextEncoder
from src.memory.voxel_helpers import compute_cosine_similarity
import numpy as np

encoder = FFTTextEncoder()

# Encode different characters
chars = ['a', 'b', 'c', 'T', 'h', 'e']
protos = {char: encoder.encode_text(char) for char in chars}

print("=" * 80)
print("Proto-Identity Similarity Matrix (FFT-encoded characters)")
print("=" * 80)
print("\nCosine similarity between characters:")
print(f"{'':>4s}", end="")
for char in chars:
    print(f"{char:>8s}", end="")
print()

for char1 in chars:
    print(f"{char1:>4s}", end="")
    for char2 in chars:
        similarity = compute_cosine_similarity(protos[char1], protos[char2])
        print(f"{similarity:>8.4f}", end="")
    print()

print("\n" + "=" * 80)
print("\nSpecific comparisons:")
for i, char1 in enumerate(chars):
    for char2 in chars[i+1:]:
        similarity = compute_cosine_similarity(protos[char1], protos[char2])
        match = "CLUSTERED" if similarity >= 0.9999 else "DISTINCT"
        print(f"   '{char1}' vs '{char2}': {similarity:.6f} → {match}")
print("=" * 80)
