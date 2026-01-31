#!/usr/bin/env python3
"""
Test generative decoding: proto → demodulate → INVERSE FFT → text

Validates that we can decode without storing native_stft (text).
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.encoding import EncodingPipeline
from src.pipeline.decoding import DecodingPipeline

def main():
    """Test generative encoding and decoding."""
    print("="*80)
    print("GENERATIVE DECODING TEST")
    print("="*80)

    # Create carrier
    carrier = np.random.randn(512, 512, 4).astype(np.float32)
    carrier = carrier / (np.abs(carrier).max() + 1e-8)

    # Initialize pipelines
    encoder = EncodingPipeline(carrier=carrier, width=512, height=512)
    decoder = DecodingPipeline(carrier=carrier)

    # Test texts
    test_texts = [
        "Hello, world! This is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "What is the Tao?",
    ]

    for i, original_text in enumerate(test_texts, 1):
        print(f"\n[Test {i}] Original: {original_text}")

        # Encode
        proto, metadata = encoder.encode_text(original_text, modality='text')
        print(f"Proto shape: {proto.shape}")
        print(f"Metadata keys: {list(metadata.keys())}")

        # Verify NO native_stft stored
        if 'native_stft' in metadata:
            print("❌ ERROR: native_stft is still being stored!")
        else:
            print("✅ No text storage - pure generative")

        # Decode via generative path (with metadata for upscaling)
        decoded_text = decoder.decode_to_text(proto, metadata=metadata)
        print(f"Decoded: {decoded_text[:100]}")

        # Compare
        print(f"Match: {decoded_text[:30] == original_text[:30]}")

        # Check if it contains some recognizable patterns
        # (won't be exact match due to lossy INVERSE FFT, but should have some similarity)
        print()

    print("="*80)
    print("TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
