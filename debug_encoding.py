#!/usr/bin/env python3
"""Debug encoding to verify FFT roundtrip works correctly."""
from src.pipeline.fft_text_encoder import FFTTextEncoder
from src.pipeline.fft_text_decoder import FFTTextDecoder

print("=" * 80)
print("Debug FFT Encoding/Decoding Roundtrip")
print("=" * 80)

encoder = FFTTextEncoder()
decoder = FFTTextDecoder()

# Test individual characters
print("\n[Test 1] Character-level encoding:")
test_chars = ['T', 'h', 'e', ' ', 'a', 'r', 't']
for char in test_chars:
    proto = encoder.encode_text(char)
    decoded = decoder.decode_text(proto)
    match = "✓" if decoded == char else "✗"
    print(f"   {match} '{char}' → '{decoded}'")

# Test words
print("\n[Test 2] Word-level encoding:")
test_words = ['The', 'art', 'of', 'war', 'is']
for word in test_words:
    proto = encoder.encode_text(word)
    decoded = decoder.decode_text(proto)
    match = "✓" if decoded == word else "✗"
    print(f"   {match} '{word}' → '{decoded}'")

# Test phrase
print("\n[Test 3] Phrase-level encoding:")
phrase = "The supreme art of war"
proto = encoder.encode_text(phrase)
decoded = decoder.decode_text(proto)
match = "✓" if decoded == phrase else "✗"
print(f"   {match} Input:   '{phrase}'")
print(f"       Output:  '{decoded}'")
print(f"       Match: {decoded == phrase}")

print("\n" + "=" * 80)
